import types

import torch
from torch import nn
from torch.nn.functional import mse_loss
from torch_geometric.data import HeteroData
from torch_geometric.nn import global_mean_pool

from core.utils.registry import registry
from core.utils.pf_losses_utils import PowerBalanceLoss
from core.utils.create_subproblem import _timed, build_subproblem_batch


def _per_subgraph_mse(student: torch.Tensor, teacher: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
    """
    Mean-of-per-subgraph-means squared error, matching how PowerBalanceLoss
    aggregates (see pf_losses_utils.PowerBalanceLoss.__call__): average
    within each subgraph first, THEN average across subgraphs -- so neither
    a subgraph with more rows (bigger subgraph) nor a subgraph that simply
    appears more times in the batch gets more weight than any other
    subgraph just by having more rows to average over.

    Unlike PowerBalanceLoss's own `.mean()` (safe there because every
    sample in a full-grid batch always has at least one bus), this
    explicitly restricts the final mean to subgraphs that actually
    contributed at least one row (`batch_idx.unique()`) rather than
    trusting global_mean_pool's implicit zero-fill for absent groups --
    a subgraph CAN legitimately contribute zero rows here (e.g. a
    single-bus subgraph whose only bus got promoted to local slack, so
    voltage_mask excludes it entirely -- see build_subproblem), and
    counting that as a fabricated zero-error entry would silently dilute
    the mean with a comparison that never happened.
    """
    if student.numel() == 0:
        return torch.zeros((), device=student.device)
    sq_err = (student - teacher) ** 2
    if sq_err.dim() > 1:
        sq_err = sq_err.mean(dim=-1)
    per_subgraph = global_mean_pool(sq_err, batch_idx)
    present = batch_idx.unique()
    return per_subgraph[present].mean()


def loss_loader(class_name, class_inputs, class_type):
    loss_class = registry.get_loss_class(class_name)
    assert loss_class is not None, (
        f"{class_type} {class_name} is not registered in the registry!"
    )
    if isinstance(loss_class, types.FunctionType):
        loss = loss_class
    else:
        loss = loss_class(**class_inputs)

    return loss


@registry.register_loss("GNNTorchLoss")
class GNNTorchLoss:
    """
    Wrapper for PyTorch loss functions to be used in GNN models.
    """

    def __init__(self, torch_nn_name, output_name, loss_inputs={}):
        loss_class = getattr(torch.nn, torch_nn_name, None)
        assert loss_class is not None, f"Loss {torch_nn_name} not found in torch.nn!"
        assert isinstance(loss_inputs, dict), "Loss inputs need to be a dictionary!"
        self.loss = loss_class(**loss_inputs)
        self.output_name = output_name
        self.loss_name = torch_nn_name

    def __call__(self, outputs, data):
        if "__" in self.output_name:
            key, output_name = self.output_name.split("__")
            truth = data[key].get(output_name, None)
        else:
            truth = data.get(self.output_name, None)
        assert truth is not None, f"Data does not contain {output_name}!"
        return self.loss(outputs, truth)


@registry.register_loss("recycle_loss")
class RecycleLoss:
    def __init__(self, recycled_parameter, loss_name, keyword):
        self.recycled_parameter = recycled_parameter
        self.loss_name = loss_name
        self.source = None
        self.keyword = keyword

    def __call__(self, output_dict, data):
        recycled_value = getattr(self.source, self.recycled_parameter)
        return recycled_value


@registry.register_loss("combined_loss")
class CombinedLoss:
    """
    Weighted sum of any number of loss components -- computed once each, in
    order, as sum(weight_i * value_i) -- rather than needing awkward manual
    nesting (combined_loss(loss1=combined_loss(a, b), loss2=c)) to combine
    more than two.

    Two equivalent ways to specify components:
      - `losses: [{name, weight, inputs}, ...]` -- any number of components,
        e.g.:
            losses:
            - name: universal_power_balance
              weight: 1.0
              inputs: {model: CANOS}
            - name: subproblem_consistency
              weight: 0.05
              inputs: {min_size: 10, max_size: 100}
            - name: recycle_loss   # a component can itself be a recycle_loss,
              weight: 0.1          # reading an attribute off an EARLIER
              inputs:               # component's own instance (e.g. .pbl_loss
                recycled_parameter: pbl_loss   # off the subproblem_consistency
                loss_name: "Subgraph PBL"      # above) instead of recomputing
                keyword: finetune_subgraph_pbl # anything -- components run in
                                                # order, and SubgraphFinetune
                                                # Trainer._wire_recycle_losses
                                                # wires recycle_loss.source to
                                                # a matching sibling by type
                                                # after construction.
      - `loss1`/`loss2`/`lamb`/`inp1`/`inp2` -- the original 2-component
        form, kept for backward compatibility with existing configs (e.g.
        canos_task_1_3.yaml) and with GNNTrainer.modify_loss, which reads
        `.loss1`/`.loss2` directly. Equivalent to
        `losses: [{name: loss1, weight: 1, inputs: inp1},
                   {name: loss2, weight: lamb, inputs: inp2}]`.

    `self.loss` is the combined total (same meaning as before). `self.
    components` is the list of (name, weight, instance) triples in order --
    what SubgraphFinetuneTrainer's recursive walkers use to find/wire nested
    loss instances regardless of how many components there are. `self.
    loss1`/`self.loss2`/`self.lamb` alias the first two components (present
    whenever there are at least that many) for the same backward-
    compatibility reason as the constructor form.
    """

    def __init__(
        self,
        losses=None,
        loss1=None,
        loss2=None,
        lamb=1,
        inp1={},
        inp2={},
        profile=False,
    ):
        if losses is None:
            assert loss1 is not None and loss2 is not None, (
                "combined_loss needs either `losses: [...]` or both loss1/loss2!"
            )
            losses = [
                {"name": loss1, "weight": 1, "inputs": inp1},
                {"name": loss2, "weight": lamb, "inputs": inp2},
            ]

        self.components = []
        printnames = []
        for spec in losses:
            name = spec["name"]
            weight = spec.get("weight", 1)
            inputs = spec.get("inputs", {})
            instance = self.initialize_loss(name, inputs)
            self.components.append((name, weight, instance))
            printnames.append(getattr(instance, "loss_name", name))
        self.loss_name = "+".join(printnames)

        # Backward-compat aliases -- GNNTrainer.modify_loss (and older
        # configs/code) reads these directly, assuming the 2-component shape.
        self.loss1 = self.components[0][2] if len(self.components) > 0 else None
        self.loss2 = self.components[1][2] if len(self.components) > 1 else None
        self.loss1_name = self.components[0][0] if len(self.components) > 0 else None
        self.loss2_name = self.components[1][0] if len(self.components) > 1 else None
        self.lamb = self.components[1][1] if len(self.components) > 1 else None

        # Opt-in profiling: accumulates wall-clock seconds (summed across
        # repeated calls, e.g. once per training step) spent inside each
        # named component -- see SubgraphFinetuneTrainer for reset/print
        # handling.
        self.profile = profile
        self.timings = {} if profile else None

    def initialize_loss(self, loss_name, loss_inputs):
        # This is for pytorch losses
        if getattr(nn, loss_name, None) is not None:
            loss_class = getattr(nn, loss_name)
            return loss_class(**loss_inputs)

        # This is for custom loss
        loss_class = registry.get_loss_class(loss_name)
        assert loss_class is not None, f"Loss {loss_name} not found!"

        if isinstance(loss_class, types.FunctionType):
            assert len(loss_inputs) == 0, (
                f"Custom loss {loss_name} is a function, but loss inputs were received!"
            )
            return loss_class
        else:
            return loss_class(**loss_inputs)

    def __call__(self, predictions, labels):
        total = None
        for name, weight, instance in self.components:
            with _timed(self.timings, name):
                value = instance(predictions, labels)
            total = weight * value if total is None else total + weight * value
        # Cached so a `recycle_loss` entry can report this total (e.g. for a
        # nested combined_loss) without recomputing it -- see e.g.
        # SubgraphFinetuneTrainer._wire_recycle_losses.
        self.loss = total
        return total


@registry.register_loss("subproblem_consistency")
class SubproblemConsistencyLoss:
    """
    Self-consistency train-time augmentation loss for topology transfer
    (PFDelta task 3.1). Cuts a random connected subgraph out of each sample
    in the batch, folds the full-grid model's own predicted tie-line flows
    into the boundary buses' net injection (see
    `core.utils.create_subproblem`), runs the same model again on the
    subgraph, and penalizes disagreement between the two predictions: bus
    voltages, net bus injections, and interior branch flows. Needs no
    ground truth on the target topology -- only the model's own full-grid
    prediction, which is why this is CANOS_PF-specific (relies on its
    output_dict schema and its bus/PV/PQ/slack HeteroData layout).

    Optionally (`pbl_weight > 0`, off by default) also penalizes the
    subgraph prediction's own physics residual directly -- universal_power_
    balance (KCL) computed on (sub_data, sub_outputs) alone, independent of
    the teacher. This is a genuinely different signal from the consistency
    terms above (which only ever compare student against teacher, never
    check the student against physics on its own), so it's additive rather
    than redundant. Deliberately folded in here rather than built as a
    separate top-level loss: it reuses the subgraph this loss already built
    and the second forward pass it already ran (both expensive -- see the
    profiling this class supports), so no extra subgraph construction or
    model call is needed, just one more cheap physics computation on
    tensors already in hand.

    `self.model` must be set after construction -- the owning trainer is
    responsible for that (mirroring `recycle_loss`'s `.source` injection),
    since a live `nn.Module` can't be built from plain config kwargs. See
    `core.trainers.subgraph_finetune_trainer.SubgraphFinetuneTrainer`.
    """

    def __init__(
        self,
        min_size=10,
        max_size=100,
        detach_teacher=True,
        voltage_weight=1.0,
        injection_weight=1.0,
        edge_weight=1.0,
        pbl_weight=0.0,
        seed=None,
        profile=False,
        subgraphs_per_sample=1,
    ):
        self.model = None
        self.min_size = min_size
        self.max_size = max_size
        self.detach_teacher = detach_teacher
        # How many independent subgraphs to cut per original sample, each
        # its own random draw -- see build_subproblem_batch's own docstring
        # for what this does and doesn't cost. Every loss component below
        # is pooled per-subgraph before being averaged (see
        # _per_subgraph_mse/PowerBalanceLoss's own aggregation), so with
        # this applied uniformly across the batch, a sample doesn't get
        # more total weight just because it contributed more subgraphs.
        self.subgraphs_per_sample = subgraphs_per_sample
        self.voltage_weight = voltage_weight
        self.injection_weight = injection_weight
        self.edge_weight = edge_weight
        self.pbl_weight = pbl_weight
        self.generator = (
            torch.Generator().manual_seed(seed) if seed is not None else None
        )
        # Only used here for its collect_model_predictions helper (gathers
        # per-bus net P/Q injection, handling PQ/PV/slack uniformly).
        self._pbl = PowerBalanceLoss(model="CANOS")
        self.loss_name = "Subproblem consistency"
        # Persists for the life of this loss instance (one per training run)
        # as the base-topology cache build_subproblem_batch maintains
        # across calls (see its adjacency_cache / update_base_topology_
        # cache docstrings) -- lets it track each sample's own small
        # contingency (N-1/N-2) deviation from a shared base topology
        # instead of rebuilding a whole adjacency list from scratch nearly
        # every call.
        self._adjacency_cache = {}
        # Opt-in profiling: accumulates wall-clock seconds (summed across
        # repeated calls) under "build_subproblem_batch" (itself broken down
        # further -- see that function's own stats keys, merged into this
        # same dict), "second_forward", and "loss_compute" -- see
        # SubgraphFinetuneTrainer for reset/print handling.
        self.profile = profile
        self.timings = {} if profile else None

    def __call__(self, outputs, data):
        assert self.model is not None, (
            "SubproblemConsistencyLoss.model was never set -- the owning "
            "trainer needs to inject it after building the model (see "
            "SubgraphFinetuneTrainer.modify_loss)."
        )

        sub_data, bus_map, voltage_mask, edge_map, va_offset = build_subproblem_batch(
            data,
            outputs,
            min_size=self.min_size,
            max_size=self.max_size,
            detach_teacher=self.detach_teacher,
            generator=self.generator,
            adjacency_cache=self._adjacency_cache,
            subgraphs_per_sample=self.subgraphs_per_sample,
            stats=self.timings,
        )
        with _timed(self.timings, "second_forward"):
            sub_outputs = self.model(sub_data)

        with _timed(self.timings, "subgraph_pbl"):
            # Physics residual on the subgraph's own prediction, independent
            # of the teacher -- reuses self._pbl (already built for
            # collect_model_predictions below) and its full __call__, which
            # additionally runs calculate_PBL's branch-flow KCL check and
            # caches the result as self._pbl.power_balance_mean.
            self.pbl_loss = self._pbl(sub_outputs, sub_data)

        with _timed(self.timings, "loss_compute"):
            # Per-subgraph batch index, for _per_subgraph_mse below -- every
            # component is averaged per subgraph first, then across
            # subgraphs (matching PowerBalanceLoss's own aggregation), so a
            # bigger subgraph (more rows/edges to average over, since sizes
            # are sampled independently per subgraph from [min_size,
            # max_size]) doesn't get more weight than a smaller one just by
            # contributing more terms to what used to be a flat mean.
            bus_batch = sub_data["bus"].batch

            # Bus voltages: skip any promoted local-slack rows, since their
            # (va, vm) is an echoed input on the subproblem side, not a prediction.
            teacher_bus = outputs["bus"][bus_map][voltage_mask]
            student_bus = sub_outputs["bus"][voltage_mask]
            # A promoted local slack has its angle forced to 0 (see
            # build_subproblem), shifting the student's whole angle solution
            # by a constant offset relative to the teacher's -- subtract it
            # back out of the teacher's va before comparing. Zero (a no-op)
            # for every sample that kept its original slack bus.
            teacher_va = teacher_bus[:, 0] - va_offset[voltage_mask]
            teacher_bus = torch.stack([teacher_va, teacher_bus[:, 1]], dim=-1)
            if self.detach_teacher:
                teacher_bus = teacher_bus.detach()
            self.voltage_loss = _per_subgraph_mse(student_bus, teacher_bus, bus_batch[voltage_mask])

            # Net bus injections (P, Q). A genuine prediction on either side
            # whenever that bus is PV/slack there; a harmless (~0) echoed-input
            # term when both sides see it as PQ.
            teacher_preds = self._pbl.collect_model_predictions("CANOS", data, outputs)
            student_preds = self._pbl.collect_model_predictions(
                "CANOS", sub_data, sub_outputs
            )
            _, _, Pnet_t, Qnet_t = teacher_preds["predictions"]
            _, _, Pnet_s, Qnet_s = student_preds["predictions"]
            teacher_net = torch.stack([Pnet_t[bus_map], Qnet_t[bus_map]], dim=-1)
            student_net = torch.stack([Pnet_s, Qnet_s], dim=-1)
            if self.detach_teacher:
                teacher_net = teacher_net.detach()
            self.injection_loss = _per_subgraph_mse(student_net, teacher_net, bus_batch)

            # Interior branch flows (lines kept on both sides of the cut).
            # Edges don't get their own `.batch` from Batch.from_data_list --
            # derive one from their source bus's own subgraph id (block-
            # diagonal batching means an edge never crosses subgraphs, so
            # its source bus's subgraph id is exactly its own).
            teacher_edges = outputs["edge_preds"][edge_map]
            student_edges = sub_outputs["edge_preds"]
            if self.detach_teacher:
                teacher_edges = teacher_edges.detach()
            edge_src = sub_data["bus", "branch", "bus"].edge_index[0]
            self.edge_loss = _per_subgraph_mse(student_edges, teacher_edges, bus_batch[edge_src])

            self.loss = (
                self.voltage_weight * self.voltage_loss
                + self.injection_weight * self.injection_loss
                + self.edge_weight * self.edge_loss
                + self.pbl_weight * self.pbl_loss
            )
        return self.loss


@registry.register_loss("Objective_n_Penalty")
class Objective_n_Penalty:
    """
    Combines an objective function with equality and inequality penalty functions.
    """

    def __init__(
        self,
        obj_name=None,
        ineq_name=None,
        eq_name=None,
        obj_inputs={},
        ineq_inputs={},
        eq_inputs={},
    ):
        self.obj_name = obj_name
        self.ineq_name = ineq_name
        self.eq_name = eq_name
        self.obj_fn = None
        self.ineq_fn = None
        self.eq_fn = None

        obj_active = obj_name is not None
        ineq_active = ineq_name is not None
        eq_active = eq_name is not None
        assert obj_active or ineq_active or eq_active, (
            "No objective, equality or inequality declared!!"
        )

        # Load objective function
        if self.obj_name is not None:
            self.obj_fn = loss_loader(obj_name, obj_inputs, "Objective function")

        # Load inequality functions
        if self.ineq_name is not None:
            self.ineq_fn = loss_loader(ineq_name, ineq_inputs, "Inequality functions")

        # Load equality functions
        if self.eq_name is not None:
            self.eq_fn = loss_loader(eq_name, eq_inputs, "Equality functions")

        self.create_name()

    def create_name(
        self,
    ):
        name = ""
        if self.obj_name is not None:
            obj_name = getattr(self.obj_fn, "loss_name", "Obj")
            name += obj_name
        if self.ineq_name is not None:
            ineq_name = getattr(self.ineq_fn, "loss_name", "Ineq")
            if len(name) == 0:
                name += ineq_name
            else:
                name += "+" + ineq_name
        if self.eq_name is not None:
            eq_name = getattr(self.eq_fn, "loss_name", "Eq")
            if len(name) == 0:
                name += eq_name
            else:
                name += "+" + eq_name
        self.loss_name = name

    def __call__(self, predictions, data):
        loss = 0
        if self.obj_fn is not None:
            loss += self.obj_fn(predictions, data)
        if self.ineq_fn is not None:
            loss += self.ineq_fn(predictions, data)
        if self.eq_fn is not None:
            loss += self.eq_fn(predictions, data)
        return loss.mean()


@registry.register_loss("pfn_masked_mse")
class Masked_L2_loss:
    """
    Custom loss function for the masked L2 loss.

    Args:
        output (torch.Tensor): The output of the neural network model.
        target (torch.Tensor): The target values.
        mask (torch.Tensor): The mask for the target values.

    Returns:
        torch.Tensor: The masked L2 loss.
    """

    def __init__(self, regularize=True, regcoeff=1):
        super(Masked_L2_loss, self).__init__()
        self.criterion = nn.MSELoss(reduction="mean")
        self.regularize = regularize
        self.regcoeff = regcoeff
        self.loss_name = "Masked MSE"
        if self.regularize:
            self.loss_name += ", reg."

    def __call__(self, output, data):
        if isinstance(data, HeteroData):
            target = data["bus"].y
            mask = data["bus"].x[:, 10:]
        else:
            target = data.y
            mask = data.x[:, 10:]

        masked = mask.type(torch.bool)

        # output = output * mask
        # target = target * mask
        outputl = torch.masked_select(output, masked)
        targetl = torch.masked_select(target, masked)

        loss = self.criterion(outputl, targetl)

        if self.regularize:
            masked = (1 - mask).type(torch.bool)
            output_reg = torch.masked_select(output, masked)
            target_reg = torch.masked_select(target, masked)
            loss = loss + self.regcoeff * self.criterion(output_reg, target_reg)

        return loss
