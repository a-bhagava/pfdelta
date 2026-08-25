import torch

from core.trainers.gnn_trainer import GNNTrainer
from core.utils.custom_losses import CombinedLoss, SubproblemConsistencyLoss
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

    def _wire_subproblem_loss(self, loss):
        if isinstance(loss, SubproblemConsistencyLoss):
            loss.model = self.model
        elif isinstance(loss, CombinedLoss):
            self._wire_subproblem_loss(loss.loss1)
            self._wire_subproblem_loss(loss.loss2)
