import torch

from core.trainers.gnn_trainer import GNNTrainer
from core.utils.custom_losses import CombinedLoss, RecycleLoss, SubproblemConsistencyLoss
from core.utils.pf_losses_utils import PowerBalanceLoss
from core.utils.registry import registry


def _skip_consistency(outputs, data):
    """Stand-in for a subproblem_consistency val_loss entry on a validation
    dataset it isn't meant to run on -- see SubgraphFinetuneTrainer.
    calc_one_val_error / functional.consistency_val_indices. Returns NaN
    (not 0.0) so it reads as "not evaluated here" rather than "perfect"."""
    return torch.full((), float("nan"), device=outputs["bus"].device)


@registry.register_trainer("subgraph_finetune_trainer")
class SubgraphFinetuneTrainer(GNNTrainer):
    """
    GNNTrainer plus two things needed to run the subgraph-consistency
    finetuning augmentation (see `core.utils.create_subproblem` and
    `core.utils.custom_losses.SubproblemConsistencyLoss`):

    1. `subproblem_consistency` loss entries need a live reference to
       `self.model` to run the second (subgraph) forward pass -- not
       something plain config kwargs can provide. This trainer walks
       `train_loss`/`val_loss` (including nested inside `combined_loss`,
       arbitrarily deep) after they're built and wires it in, mirroring how
       `GNNTrainer.modify_loss` already wires `recycle_loss.source`. Combine
       it with the usual supervised/physics losses via `combined_loss`,
       e.g.:

           train_loss:
           - name: combined_loss
             loss1: combined_loss
             inp1: {loss1: canos_pf_mse, loss2: pf_constraint_violation, lamb: 0.1}
             loss2: subproblem_consistency
             inp2: {min_size: 0.1, max_size: 0.3}
             lamb: 0.05

    2. Since this augmentation is meant as a finetuning step (not training
       from scratch), the model can be warm-started from a previously
       trained run's weights via `model.pretrained_path` in the config
       (a path to a `model.pt`/state_dict file).

    3. This wires any `recycle_loss` entry anywhere in `train_loss`/
       `val_loss` -- however deeply nested inside `combined_loss`es -- whose
       `keyword` is `finetune_supervised` / `finetune_consistency` /
       `finetune_subgraph_pbl` to whichever `universal_power_balance` /
       `subproblem_consistency` instance it finds inside `train_loss[0]`'s
       own tree (see `_wire_recycle_losses`). A `recycle_loss` doing this
       can be used two ways, and both work simultaneously:

       (a) Purely for train.json logging -- a flat, top-level entry after
           train_loss[0] (never affects the backpropagated total, since only
           train_loss[0] gets .backward()).
       (b) As a genuine, independently-weighted component of the
           backpropagated total itself -- nested inside a combined_loss,
           since RecycleLoss just returns the source's live (still-
           differentiable) tensor. This is how to get 3+ independently-
           weighted components out of combined_loss (inherently a binary
           a+lamb*b combiner) without recomputing anything: e.g. universal_
           power_balance (weight fixed at 1, being loss1) + w2*subproblem_
           consistency + w3*subgraph-PBL, with subproblem_consistency's own
           `pbl_weight` left at its default 0 so the subgraph-PBL term isn't
           *also* folded into the w2-weighted consistency subtotal:

           train_loss:
           - name: combined_loss                    # outer: total = inner + w3*subgraph_pbl
             lamb: <w3>
             loss1: combined_loss                    # inner: total = PBL + w2*consistency
             inp1:
               lamb: <w2>
               loss1: universal_power_balance
               inp1: {model: CANOS}
               loss2: subproblem_consistency
               inp2: {..., pbl_weight: 0.0}           # 0.0 (the default): see note above
             loss2: recycle_loss                      # pulls subproblem_consistency.pbl_loss
             inp2:
               recycled_parameter: pbl_loss
               loss_name: "Subgraph power balance"
               keyword: finetune_subgraph_pbl
           # Logged only -- each of the three components separately, alongside
           # the combined total (train_loss[0] itself):
           - name: recycle_loss
             recycled_parameter: power_balance_mean
             loss_name: "Universal power balance subtotal"
             keyword: finetune_supervised
           - name: recycle_loss
             recycled_parameter: loss
             loss_name: "Consistency subtotal"
             keyword: finetune_consistency
           - name: recycle_loss
             recycled_parameter: pbl_loss
             loss_name: "Subgraph power balance subtotal"
             keyword: finetune_subgraph_pbl

    4. Opt-in profiling: any `combined_loss`/`subproblem_consistency` entry
       (found anywhere in train_loss/val_loss, arbitrarily nested) built
       with `profile: true` gets its timings dict reset at the start of
       every epoch and a summary printed at the end -- showing the split
       between the pre-existing objective and the added subgraph-consistency
       cost, and (for subproblem_consistency itself) a further breakdown of
       where that cost goes: adjacency/BFS/tensor-building/rebatching vs.
       the second forward pass vs. the subgraph PBL computation vs. the
       consistency loss computation. E.g.:

           train_loss:
           - name: combined_loss
             loss1: universal_power_balance
             loss2: subproblem_consistency
             inp2: {profile: true, ...}
             profile: true

    5. `functional.consistency_val_indices`: which validation dataset(s)
       (0-indexed into `dataset.datasets[1:]`, i.e. matching `val_num` in
       `calc_one_val_error`) actually evaluate any `subproblem_consistency`
       entries in `val_loss` -- e.g. `[5]` if `val_loss` includes
       `subproblem_consistency` and dataset.datasets[6] (val index 5) is the
       source case (case500) it makes sense to cut subgraphs out of, while
       datasets.datasets[2:6] are small target cases (case14/30/57/118) that
       a "consistency with a cut-down version of itself" check doesn't apply
       to. For every val dataset NOT in this list, `subproblem_consistency`
       entries are swapped out for a NaN-returning stand-in (no subgraph
       ever gets built for them) -- entries stay in place rather than being
       removed so `self.val_loss_names` (fixed-length, shared across every
       val dataset) stays aligned with what `calc_val_errors` reports for
       each one. Defaults to `[]` (skip subproblem_consistency everywhere in
       val_loss) if unset.
    """

    def customize_model_init_inputs(self, model_inputs):
        super().customize_model_init_inputs(model_inputs)
        # "pretrained_path" is only for us (see load_model below) -- the
        # model class itself doesn't accept it as a constructor kwarg.
        self._pretrained_path = model_inputs.pop("pretrained_path", None)

    def load_model(self):
        super().load_model()
        if self._pretrained_path:
            state_dict = torch.load(self._pretrained_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"\nWarm-started model weights from {self._pretrained_path}")

    def modify_loss(self):
        super().modify_loss()
        for loss in list(self.train_loss) + list(self.val_loss):
            self._wire_subproblem_loss(loss)
        self._wire_recycle_losses()
        self._profiled_losses = []
        for loss in list(self.train_loss) + list(self.val_loss):
            self._collect_profiled_losses(loss)
        # Opt-in, tied to the SAME signal as everything else (whether any
        # loss entry set profile: true) -- GNNTrainer.train_one_epoch reads
        # this attribute (via getattr, defaulting to None -- a no-op) to
        # time the first (full-grid) forward pass, which otherwise happens
        # before any train_loss entry even runs and so was invisible to
        # every other profiling hook here.
        self._forward_pass_timings = {} if self._profiled_losses else None

    def calc_one_val_error(self, val_dataloader, val_num):
        consistency_val_indices = self.config["functional"].get(
            "consistency_val_indices", []
        )
        if val_num in consistency_val_indices:
            return super().calc_one_val_error(val_dataloader, val_num)
        # Not a dataset subproblem_consistency should run on -- swap any
        # such entries out for a no-op stand-in (same position, same count,
        # see _skip_consistency) for the duration of this one dataset's
        # evaluation, then restore. Never mutates val_loss for OTHER
        # (consistency-enabled) datasets, since this only runs per-call.
        #
        # Also swaps out any recycle_loss entry reading off a
        # subproblem_consistency instance (e.g. keyword=finetune_
        # subgraph_pbl, pulling .pbl_loss -- see _wire_recycle_losses) --
        # that instance never runs its __call__ on this dataset either, so
        # its attribute would otherwise be whatever was last left over from
        # the last dataset that DID run consistency (possibly a previous
        # epoch, or simply absent/AttributeError before that's ever
        # happened once), not a value that means anything here.
        saved = {}
        for i, loss in enumerate(self.val_loss):
            if isinstance(loss, SubproblemConsistencyLoss) or (
                isinstance(loss, RecycleLoss)
                and isinstance(loss.source, SubproblemConsistencyLoss)
            ):
                saved[i] = loss
                self.val_loss[i] = _skip_consistency
        try:
            return super().calc_one_val_error(val_dataloader, val_num)
        finally:
            for i, loss in saved.items():
                self.val_loss[i] = loss

    def _wire_subproblem_loss(self, loss):
        if isinstance(loss, SubproblemConsistencyLoss):
            loss.model = self.model
        elif isinstance(loss, CombinedLoss):
            for _, _, instance in loss.components:
                self._wire_subproblem_loss(instance)

    def _collect_profiled_losses(self, loss):
        if isinstance(loss, (CombinedLoss, SubproblemConsistencyLoss)) and getattr(
            loss, "profile", False
        ):
            self._profiled_losses.append(loss)
        if isinstance(loss, CombinedLoss):
            for _, _, instance in loss.components:
                self._collect_profiled_losses(instance)

    def setup_pre_epoch(self):
        super().setup_pre_epoch()
        for loss in self._profiled_losses:
            loss.timings.clear()
        if self._forward_pass_timings is not None:
            self._forward_pass_timings.clear()

    def setup_post_epoch(self):
        super().setup_post_epoch()
        if not self._profiled_losses:
            return
        print(f"\n\U000023f1️  Profiling summary (epoch {self.epoch + 1}):")
        if self._forward_pass_timings:
            print("  [First forward pass]")
            for key, seconds in self._forward_pass_timings.items():
                print(f"    {key:<24}: {seconds:.3f}s")
        for loss in self._profiled_losses:
            label = getattr(loss, "loss_name", type(loss).__name__)
            print(f"  [{label}]")
            # Entries keyed "*#count" (see create_subproblem._bump) are
            # plain integer counters, not wall-clock seconds -- printed
            # separately so they don't get sorted/formatted as "%.3fs"
            # alongside the real timings.
            timing_items = {k: v for k, v in loss.timings.items() if not k.endswith("#count")}
            count_items = {k: v for k, v in loss.timings.items() if k.endswith("#count")}
            for key, seconds in sorted(timing_items.items(), key=lambda kv: kv[1], reverse=True):
                print(f"    {key:<24}: {seconds:.3f}s")
            for key, count in sorted(count_items.items(), key=lambda kv: kv[1], reverse=True):
                print(f"    {key[: -len('#count')]:<24}: {count} calls")

    def _wire_recycle_losses(self):
        # Finds the canonical universal_power_balance / subproblem_
        # consistency instances inside train_loss[0]'s tree, however deeply
        # nested (see docstring point 3 -- e.g. 3+ way combined_loss
        # nesting), and wires every recycle_loss found in train_loss (also
        # however deeply nested -- one might be folded into the actual
        # backpropagated total with its own weight, alongside separate ones
        # used purely for train.json logging) to whichever of the two its
        # `keyword` names. Only meaningful if train_loss[0] actually
        # contains those types; otherwise wiring train_loss is a no-op.
        #
        # val_loss is handled separately and NOT just reused from train_loss:
        # it's a flat list (never nested/combined -- every entry is computed
        # and reported independently, none of them backpropagated), so if it
        # has its own top-level subproblem_consistency/universal_power_
        # balance entries, a recycle_loss inside val_loss should read from
        # THOSE (the instances actually invoked during validation) rather
        # than train's -- pulling a subgraph subtotal off train's instance
        # into a "validation" metric would be stale/wrong, since that
        # instance only ever gets called during training steps. Falls back
        # to train's instances if val_loss doesn't have its own (e.g. a val
        # recycle_loss deliberately logging a train-side subtotal).
        if not self.train_loss:
            return
        train_pbl_source = self._find_instance(self.train_loss[0], PowerBalanceLoss)
        train_consistency_source = self._find_instance(
            self.train_loss[0], SubproblemConsistencyLoss
        )
        for loss in self.train_loss:
            self._wire_one_recycle_loss(loss, train_pbl_source, train_consistency_source)

        val_pbl_source = None
        val_consistency_source = None
        for loss in self.val_loss:
            if val_pbl_source is None and isinstance(loss, PowerBalanceLoss):
                val_pbl_source = loss
            if val_consistency_source is None and isinstance(
                loss, SubproblemConsistencyLoss
            ):
                val_consistency_source = loss
        for loss in self.val_loss:
            self._wire_one_recycle_loss(
                loss,
                val_pbl_source or train_pbl_source,
                val_consistency_source or train_consistency_source,
            )

    def _find_instance(self, loss, cls):
        if isinstance(loss, cls):
            return loss
        if isinstance(loss, CombinedLoss):
            for _, _, instance in loss.components:
                found = self._find_instance(instance, cls)
                if found is not None:
                    return found
        return None

    def _wire_one_recycle_loss(self, loss, pbl_source, consistency_source):
        if isinstance(loss, RecycleLoss):
            if loss.keyword == "finetune_supervised":
                loss.source = pbl_source
            elif loss.keyword in ("finetune_consistency", "finetune_subgraph_pbl"):
                # Both live on the same subproblem_consistency instance --
                # recycled_parameter in the config picks which attribute
                # (.loss vs .pbl_loss) each entry actually reads.
                loss.source = consistency_source
        elif isinstance(loss, CombinedLoss):
            for _, _, instance in loss.components:
                self._wire_one_recycle_loss(instance, pbl_source, consistency_source)
