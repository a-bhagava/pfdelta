import torch

from core.trainers.gnn_trainer import GNNTrainer
from core.utils.custom_losses import CombinedLoss, RecycleLoss, SubproblemConsistencyLoss
from core.utils.registry import registry


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

    3. If `train_loss[0]` is built the way the docstring above shows --
       `combined_loss(loss1=<supervised combined_loss>, loss2=subproblem_
       consistency)` -- this wires any `recycle_loss` entries elsewhere in
       `train_loss` with `keyword: finetune_supervised` /
       `keyword: finetune_consistency` / `keyword: finetune_subgraph_pbl` to
       that supervised subtotal / consistency subtotal / subgraph physics-
       residual subtotal respectively, so train.json can report all of them
       plus the combined total (train_loss[0] itself) without recomputing
       anything twice, e.g.:

           train_loss:
           - name: combined_loss          # index 0: backpropagated total
             ...
             inp2: {pbl_weight: 0.1, ...} # optional: also penalize the
                                           # subgraph's own physics residual
                                           # (see SubproblemConsistencyLoss)
           - name: recycle_loss           # logged only: supervised subtotal
             recycled_parameter: loss
             loss_name: "Supervised subtotal"
             keyword: finetune_supervised
           - name: recycle_loss           # logged only: consistency subtotal
             recycled_parameter: loss
             loss_name: "Consistency subtotal"
             keyword: finetune_consistency
           - name: recycle_loss           # logged only: subgraph PBL subtotal
             recycled_parameter: pbl_loss
             loss_name: "Subgraph PBL subtotal"
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

    def _wire_subproblem_loss(self, loss):
        if isinstance(loss, SubproblemConsistencyLoss):
            loss.model = self.model
        elif isinstance(loss, CombinedLoss):
            self._wire_subproblem_loss(loss.loss1)
            self._wire_subproblem_loss(loss.loss2)

    def _collect_profiled_losses(self, loss):
        if isinstance(loss, (CombinedLoss, SubproblemConsistencyLoss)) and getattr(
            loss, "profile", False
        ):
            self._profiled_losses.append(loss)
        if isinstance(loss, CombinedLoss):
            self._collect_profiled_losses(loss.loss1)
            self._collect_profiled_losses(loss.loss2)

    def setup_pre_epoch(self):
        super().setup_pre_epoch()
        for loss in self._profiled_losses:
            loss.timings.clear()

    def setup_post_epoch(self):
        super().setup_post_epoch()
        if not self._profiled_losses:
            return
        print(f"\n\U000023f1️  Profiling summary (epoch {self.epoch + 1}):")
        for loss in self._profiled_losses:
            label = getattr(loss, "loss_name", type(loss).__name__)
            print(f"  [{label}]")
            for key, seconds in sorted(
                loss.timings.items(), key=lambda kv: kv[1], reverse=True
            ):
                print(f"    {key:<24}: {seconds:.3f}s")

    def _wire_recycle_losses(self):
        # Only applies to the train_loss[0] == combined_loss(supervised,
        # subproblem_consistency) shape shown in the docstring above --
        # anything else, there's nothing we know how to wire, so skip quietly.
        if not self.train_loss or not isinstance(self.train_loss[0], CombinedLoss):
            return
        outer = self.train_loss[0]
        for loss in list(self.train_loss) + list(self.val_loss):
            if not isinstance(loss, RecycleLoss):
                continue
            if loss.keyword == "finetune_supervised":
                loss.source = outer.loss1
            elif loss.keyword in ("finetune_consistency", "finetune_subgraph_pbl"):
                # Both live on the same subproblem_consistency instance --
                # recycled_parameter in the config picks which attribute
                # (.loss vs .pbl_loss) each entry actually reads.
                loss.source = outer.loss2
