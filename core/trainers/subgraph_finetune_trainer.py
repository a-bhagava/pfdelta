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

    3. Before any finetuning step, it runs one validation pass over all val
       datasets with the just-warm-started model, so you have a baseline to
       compare the finetuned numbers against (recorded under val_errors key
       "-1", printed same as any other validation, saved to val.json). This
       is skipped when resuming from a checkpoint, since a baseline was
       already recorded on the original run.

    4. If `train_loss[0]` is built the way the docstring above shows --
       `combined_loss(loss1=<supervised combined_loss>, loss2=subproblem_
       consistency)` -- this wires any `recycle_loss` entries elsewhere in
       `train_loss` with `keyword: finetune_supervised` /
       `keyword: finetune_consistency` to that supervised subtotal /
       consistency subtotal respectively, so train.json can report both
       components plus the combined total (train_loss[0] itself) without
       recomputing anything twice, e.g.:

           train_loss:
           - name: combined_loss          # index 0: backpropagated total
             ...
           - name: recycle_loss           # logged only: supervised subtotal
             recycled_parameter: loss
             loss_name: "Supervised subtotal"
             keyword: finetune_supervised
           - name: recycle_loss           # logged only: consistency subtotal
             recycled_parameter: loss
             loss_name: "Consistency subtotal"
             keyword: finetune_consistency
    """

    def train(self):
        self._evaluate_pretrained_baseline()
        super().train()

    def _evaluate_pretrained_baseline(self):
        if self.checkpoint and self._checkpoint_used:
            # Resuming a finetuning run that already has its own baseline
            # entry -- don't re-evaluate (and don't touch self.epoch mid-resume).
            return
        print(
            "\n\U0001f50d Evaluating the pretrained/warm-started model on all "
            "validation sets before any finetuning step (baseline)..."
        )
        train_params = self.config["optim"]["train_params"]
        # calc_val_errors keys its entry off self.epoch when max_epoch is set
        # (see BaseTrainer.calc_val_errors) -- set both up the same way
        # BaseTrainer._train does, but with epoch=-1, so this baseline gets
        # its own distinct key instead of colliding with epoch 0's real
        # post-training validation.
        self.max_epoch = train_params.get("epochs", self.max_epoch)
        saved_epoch = self.epoch
        saved_checkpoint = self.checkpoint
        self.epoch = -1
        # calc_val_errors() also calls save_summary(), which (in epochs mode)
        # unconditionally looks up self.train_errors[self.best_point] -- give
        # it a harmless empty placeholder, since no training epoch has run
        # yet to populate a real one. And skip save_checkpoint() for this one
        # call (if checkpointing is on), since it would otherwise capture
        # this synthetic epoch=-1 into the checkpoint.
        self.train_errors["-1"] = {}
        self.checkpoint = False
        try:
            self.calc_val_errors()
        finally:
            self.checkpoint = saved_checkpoint
            self.epoch = saved_epoch
            del self.train_errors["-1"]

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

    def _wire_subproblem_loss(self, loss):
        if isinstance(loss, SubproblemConsistencyLoss):
            loss.model = self.model
        elif isinstance(loss, CombinedLoss):
            self._wire_subproblem_loss(loss.loss1)
            self._wire_subproblem_loss(loss.loss2)

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
            elif loss.keyword == "finetune_consistency":
                loss.source = outer.loss2
