"""
Utilities for constructing power-flow "subproblems": a connected subgraph of
a full power network, with tie-lines (branches leaving the subgraph) removed
and their model-predicted flows folded into the boundary buses' net power
injection.

This underlies a self-consistency train-time augmentation aimed at PFDelta
task 3.1 (transfer to unseen grid sizes/topologies): predict on the full
grid, cut out a random connected subgraph, replace everything the subgraph
can't see with an equivalent injection at the boundary buses, run the same
model again on just the subgraph, and penalize disagreement between the two
predictions (see `core.utils.custom_losses.SubproblemConsistencyLoss`).

Why folding in the tie-line flow like this is the right thing to do: for the
*exact* AC power-flow solution, KCL at a bus n says

    Pnet(n) = L(n) + shunt_consumed(n)

where L(n) is the sum of the branch flows leaving n (in this codebase's
convention, `p_fr`/`q_fr` for a branch where n is the "from" bus, `p_to`/
`q_to` where n is the "to" bus -- see `CANOS_PF.derive_slack_output`, which
scatter-adds both unnegated onto their own endpoint). Cutting a tie line t
incident to a retained boundary bus b removes its contribution l_t(b) from
L(b); the same voltage solution stays self-consistent at b only if we also
subtract l_t(b) from Pnet(b) -- i.e. treat that removed flow as extra load
at b. This is exact when l_t(b) is the true tie-line flow (a Ward/boundary
equivalent); here we use the model's own predicted l_t(b) instead, which
turns it into a self-training/pseudo-label signal that needs no ground
truth on the target topology.
"""

import time
from contextlib import contextmanager
from typing import Optional

import torch
from torch_geometric.data import Batch, HeteroData


@contextmanager
def _timed(stats: Optional[dict], key: str):
    """No-op when `stats` is None; otherwise accumulates wall-clock seconds
    spent in the `with` block under `stats[key]` (summed across repeated
    calls, e.g. once per training step) -- used for opt-in profiling of
    where subproblem-construction time actually goes. See
    SubgraphFinetuneTrainer for how these get reset/printed per epoch."""
    if stats is None:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        stats[key] = stats.get(key, 0.0) + (time.perf_counter() - start)


def build_adjacency(edge_index: torch.Tensor, num_nodes: int) -> list:
    """Build an undirected adjacency list from a directed branch edge_index."""
    adjacency = [[] for _ in range(num_nodes)]
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for i, j in zip(src, dst):
        adjacency[i].append(j)
        adjacency[j].append(i)
    return adjacency


def sample_bus_subset(
    adjacency: list,
    seed_bus: int,
    min_size,
    max_size,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Randomly grow a connected subset of buses via BFS from `seed_bus`.

    The subset is grown one shell at a time (each bus's unvisited neighbors,
    in random order) until it reaches a target size sampled uniformly from
    [min_size, max_size]. Because it's built as a BFS tree, the returned
    subset is always connected. Assumes the grid `seed_bus` sits in is
    itself fully connected (no islanding) -- if `max_size` exceeds what's
    actually reachable from `seed_bus`, growth just stops early once the
    frontier is exhausted, returning a smaller subset than requested rather
    than failing.

    Parameters
    ----------
    adjacency : list[list[int]]
        Undirected adjacency list, e.g. from `build_adjacency`. Can be over
        a batched bus index space -- since batched samples never share
        edges, growth from a seed inside one sample never crosses into
        another.
    seed_bus : int
        Global bus index to grow the subset from.
    min_size, max_size : int
        Target subset size bounds, as absolute bus counts.
    generator : torch.Generator, optional
        RNG used for the neighbor shuffle and size sampling.

    Returns
    -------
    torch.Tensor
        Sorted LongTensor of global bus indices in the sampled subset.
    """
    lo, hi = int(min_size), int(max_size)
    lo, hi = min(lo, hi), max(lo, hi)
    target_size = torch.randint(lo, hi + 1, (1,), generator=generator).item()

    # Grow the BFS, shuffling each shell, stopping once the target size is
    # hit -- so repeated calls from the same seed still vary.
    visited = {seed_bus}
    kept = [seed_bus]
    frontier = [seed_bus]
    while frontier and len(kept) < target_size:
        next_frontier = []
        for node in frontier:
            neighbors = adjacency[node]
            perm = torch.randperm(len(neighbors), generator=generator).tolist()
            for idx in perm:
                neighbor = neighbors[idx]
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                kept.append(neighbor)
                next_frontier.append(neighbor)
                if len(kept) >= target_size:
                    break
            if len(kept) >= target_size:
                break
        frontier = next_frontier

    return torch.tensor(sorted(kept), dtype=torch.long)


def _remap_node_type(data, sub_data, node_type, link_name, keep_mask, new_index):
    """
    Filter a `node_type` node store (e.g. "gen"/"load") plus its
    `(node_type, link_name, "bus")` link edges down to whichever entries
    attach to a kept bus, remapping bus references to the new contiguous
    local indices.
    """
    link_edge_index = data[node_type, link_name, "bus"].edge_index
    node_idx, bus_idx = link_edge_index[0], link_edge_index[1]
    keep = keep_mask[bus_idx]
    kept_node_idx = node_idx[keep]
    device = kept_node_idx.device

    # Skip PyG's own batch bookkeeping keys -- "ptr" in particular is sized by
    # num_graphs + 1, not by node count, so indexing it with kept_node_idx
    # would be wrong (or out of bounds) rather than just redundant.
    skip_keys = {"num_nodes", "ptr", "batch"}
    for key, value in data[node_type].items():
        if key in skip_keys or not torch.is_tensor(value):
            continue
        sub_data[node_type][key] = value[kept_node_idx]
    # PyG can't reliably infer num_nodes for a node type with no canonical
    # ("x"-like) attribute -- e.g. "gen"'s limits/generation/slack_gen -- and
    # a sampled subgraph can easily end up with zero of them, so set it
    # explicitly to keep later batching (e.g. in build_subproblem_batch) safe.
    sub_data[node_type].num_nodes = kept_node_idx.numel()

    new_link_index = torch.stack(
        [
            torch.arange(kept_node_idx.numel(), device=device),
            new_index[bus_idx[keep]],
        ]
    )
    sub_data[node_type, link_name, "bus"].edge_index = new_link_index
    sub_data["bus", link_name, node_type].edge_index = new_link_index.flip(0)


def build_subproblem(
    data: HeteroData,
    output_dict: dict,
    bus_subset: torch.Tensor,
    add_bus_type: Optional[bool] = None,
    detach_teacher: bool = True,
) -> tuple:
    """
    Cut a subgraph out of `data` restricted to `bus_subset`, folding the
    model's own predicted tie-line flows into the retained boundary buses'
    net power injection (see module docstring) so the result is a
    self-consistent, standalone power-flow instance.

    If the case's slack (reference) bus is not in `bus_subset`, a new local
    slack is elected from within the subset -- preferring a PV bus, since a
    PV bus's reactive generation is a genuine model prediction, so comparing
    it downstream is a meaningful consistency check rather than one that
    trivially matches an echoed input. Its voltage magnitude is pinned to
    the full-grid model's own predicted vm there (no reference-frame
    ambiguity for an absolute quantity like vm), but its angle is forced to
    0 rather than the teacher's predicted va -- every real slack bus in the
    training data has va=0 as input by definition, so a nonzero value there
    would be out-of-distribution for the student's forward pass. This shifts
    the student's whole angle solution by a constant offset relative to the
    teacher's (see `va_offset` below), which the consistency loss needs to
    correct for.

    Works directly on a batched `data`/`output_dict`: `bus_subset` just
    needs to be a subset of one sample's own (batch-offset) bus indices,
    since node/edge stores outside it are automatically excluded. See
    `build_subproblem_batch` for building one subproblem per sample in a
    batch and re-batching them.

    Parameters
    ----------
    data : HeteroData
        The (optionally batched) full-grid input graph that produced
        `output_dict`.
    output_dict : dict
        CANOS_PF-style output dict from a forward pass on `data` --
        `output_dict["bus"]` (va, vm per bus) and `output_dict["edge_preds"]`
        (p_fr, q_fr, p_to, q_to per branch) are used.
    bus_subset : torch.Tensor
        Global bus indices (into `data["bus"]`) to keep.
    add_bus_type : bool, optional
        Whether to also rebuild the PV/PQ/slack sub-node-types and their
        link edges. Defaults to auto-detecting from whether `data` itself
        has a "PQ" node type.
    detach_teacher : bool
        If True, every quantity taken from `output_dict`/`data` (tie-line
        injections, the promoted slack's (va, vm)) is detached before being
        used to build the subproblem, so gradients from the subsequent
        subproblem forward pass don't flow back into the full-grid pass
        that produced them.

    Returns
    -------
    sub_data : HeteroData
        The subproblem graph, ready to feed into the same model.
    bus_map : torch.Tensor
        `sub_data["bus"]` row i corresponds to global bus index `bus_map[i]`
        in `data["bus"]`.
    voltage_mask : torch.Tensor
        Boolean mask, True for every `sub_data["bus"]` row whose predicted
        (va, vm) is a genuine model prediction rather than an echoed input
        (i.e. every row except a promoted local slack, if any).
    edge_map : torch.Tensor
        `sub_data["bus","branch","bus"]` column i corresponds to row
        `edge_map[i]` of `data["bus","branch","bus"].edge_attr` /
        `output_dict["edge_preds"]`.
    va_offset : torch.Tensor
        Same length as `bus_map`/`voltage_mask`. If a new local slack was
        promoted, every entry equals the teacher's own predicted va at that
        promoted bus (the value discarded in favor of forcing 0) -- since
        angle differences between buses are reference-independent, the
        student's whole angle solution is offset from the teacher's by
        exactly this amount, so subtract it from the teacher's va before
        comparing against the student's. All zeros (a no-op) when the
        original slack bus was kept and no promotion happened.
    """
    if add_bus_type is None:
        add_bus_type = "PQ" in data.node_types

    maybe_detach = (lambda t: t.detach()) if detach_teacher else (lambda t: t)

    device = data["bus"].x.device
    num_buses = data["bus"].num_nodes
    bus_subset = bus_subset.to(device=device, dtype=torch.long)
    n_sub = bus_subset.numel()

    keep_mask = torch.zeros(num_buses, dtype=torch.bool, device=device)
    keep_mask[bus_subset] = True
    new_index = torch.full((num_buses,), -1, dtype=torch.long, device=device)
    new_index[bus_subset] = torch.arange(n_sub, device=device)

    # ---- Branch edges: split into interior (both ends kept) and tie lines (one end kept) ----
    edge_index = data["bus", "branch", "bus"].edge_index
    edge_attr = data["bus", "branch", "bus"].edge_attr
    src, dst = edge_index[0], edge_index[1]
    src_in, dst_in = keep_mask[src], keep_mask[dst]
    interior_mask = src_in & dst_in
    tie_src_kept = src_in & ~dst_in  # tie line, retained endpoint is "from"
    tie_dst_kept = dst_in & ~src_in  # tie line, retained endpoint is "to"

    edge_preds = maybe_detach(output_dict["edge_preds"])
    bus_pred = maybe_detach(output_dict["bus"])

    # ---- Fold each cut tie-line's predicted flow into its retained endpoint's net injection ----
    # `extra_p`/`extra_q`: real/reactive power that used to leave the bus along
    # now-removed tie lines, per the full-grid model's own prediction (p_fr for
    # a line where the retained bus is "from", p_to where it's "to" -- both
    # unnegated, matching CANOS_PF.derive_slack_output's own convention).
    # Pnet_new = Pnet_old - extra_p/extra_q, i.e. pd_new = pd_old + extra_p (see
    # module docstring derivation).
    extra_p = torch.zeros(num_buses, device=device)
    extra_q = torch.zeros(num_buses, device=device)
    extra_p.scatter_add_(0, src[tie_src_kept], edge_preds[tie_src_kept, 0])
    extra_q.scatter_add_(0, src[tie_src_kept], edge_preds[tie_src_kept, 1])
    extra_p.scatter_add_(0, dst[tie_dst_kept], edge_preds[tie_dst_kept, 2])
    extra_q.scatter_add_(0, dst[tie_dst_kept], edge_preds[tie_dst_kept, 3])

    bus_type = data["bus"].bus_type[bus_subset].clone()
    bus_demand = data["bus"].bus_demand[bus_subset].clone()
    bus_gen = data["bus"].bus_gen[bus_subset].clone()
    bus_voltages = data["bus"].bus_voltages[bus_subset].clone()
    bus_shunts = data["bus"].shunt[bus_subset].clone()
    voltage_limits = data["bus"].limits[bus_subset].clone()

    bus_demand[:, 0] = bus_demand[:, 0] + extra_p[bus_subset]
    bus_demand[:, 1] = bus_demand[:, 1] + extra_q[bus_subset]

    # ---- Elect a new local slack if the case's own slack bus fell outside the subset ----
    global_slack_idx = (data["bus"].bus_type == 3).nonzero(as_tuple=True)[0]
    slack_kept = bool(keep_mask[global_slack_idx].any())
    voltage_mask = torch.ones(n_sub, dtype=torch.bool, device=device)
    # Angle reference-frame offset introduced by promoting a new local slack
    # (see below) -- 0 unless that happens. Broadcast to every retained bus
    # in this sample so build_subproblem_batch can concatenate it alongside
    # bus_map/voltage_mask; SubproblemConsistencyLoss subtracts it from the
    # teacher's angles before comparing against the student's.
    va_offset = torch.zeros(n_sub, device=device)

    new_src = new_index[src[interior_mask]]
    new_dst = new_index[dst[interior_mask]]

    if not slack_kept:
        degree = torch.zeros(n_sub, device=device)
        degree.scatter_add_(0, new_src, torch.ones_like(new_src, dtype=torch.float))
        degree.scatter_add_(0, new_dst, torch.ones_like(new_dst, dtype=torch.float))

        pv_positions = (bus_type == 2).nonzero(as_tuple=True)[0]
        if pv_positions.numel() > 0:
            promote_pos = pv_positions[degree[pv_positions].argmax()].item()
        else:
            promote_pos = int(degree.argmax().item())

        bus_type[promote_pos] = 3
        # Pin the promoted slack's voltage MAGNITUDE to the teacher's own
        # prediction there (no reference-frame ambiguity for vm -- it's an
        # absolute physical quantity). The ANGLE, though, is forced to 0
        # rather than the teacher's predicted value: every real slack bus in
        # the training data has va=0 as its input by definition (that's what
        # "reference bus" means), so the model has never seen a nonzero
        # angle fed in as a slack input -- using the teacher's arbitrary
        # value here would be out-of-distribution for the student's own
        # forward pass. This shifts the student's whole angle solution by
        # exactly `teacher_va_at_promoted_bus` relative to the teacher's
        # frame (physically: for two solutions of the same network, angle
        # DIFFERENCES between buses are reference-independent, so pinning a
        # different absolute value at the reference bus offsets every other
        # bus's angle by that same amount) -- recorded in va_offset so the
        # consistency loss can subtract it back out before comparing.
        va_offset[:] = bus_pred[bus_subset[promote_pos], 0]
        bus_voltages[promote_pos, 1] = bus_pred[bus_subset[promote_pos], 1]
        bus_voltages[promote_pos, 0] = 0.0
        voltage_mask[promote_pos] = False

    pq_mask = bus_type == 1
    pv_mask = bus_type == 2
    slack_mask = bus_type == 3

    pf_x = torch.zeros(n_sub, 2, device=device)
    pf_x[pq_mask] = bus_demand[pq_mask]
    pf_x[pv_mask, 0] = bus_gen[pv_mask, 0] - bus_demand[pv_mask, 0]
    pf_x[pv_mask, 1] = bus_voltages[pv_mask, 1]
    pf_x[slack_mask] = bus_voltages[slack_mask]

    sub_data = HeteroData()
    sub_data["bus"].x = pf_x
    sub_data["bus"].num_nodes = n_sub
    sub_data["bus"].bus_gen = bus_gen
    sub_data["bus"].bus_demand = bus_demand
    sub_data["bus"].bus_voltages = bus_voltages
    sub_data["bus"].bus_type = bus_type
    sub_data["bus"].shunt = bus_shunts
    sub_data["bus"].limits = voltage_limits

    sub_data["bus", "branch", "bus"].edge_index = torch.stack([new_src, new_dst])
    sub_data["bus", "branch", "bus"].edge_attr = edge_attr[interior_mask]
    if "edge_limits" in data["bus", "branch", "bus"]:
        sub_data["bus", "branch", "bus"].edge_limits = data[
            "bus", "branch", "bus"
        ].edge_limits[interior_mask]
    edge_map = interior_mask.nonzero(as_tuple=True)[0]

    # "gen"/"load" only exist for dataset variants that don't prune them --
    # CANOS's own dataset (PFDeltaCANOS.build_heterodata) drops both, keeping
    # only bus/PV/PQ/slack, so there's nothing to remap for the common case.
    if "gen" in data.node_types:
        _remap_node_type(data, sub_data, "gen", "gen_link", keep_mask, new_index)
    if "load" in data.node_types:
        _remap_node_type(data, sub_data, "load", "load_link", keep_mask, new_index)

    if add_bus_type:
        for node_type, mask in (("PQ", pq_mask), ("PV", pv_mask), ("slack", slack_mask)):
            positions = mask.nonzero(as_tuple=True)[0]
            sub_data[node_type].x = pf_x[positions]
            if node_type != "PQ":
                sub_data[node_type].demand = bus_demand[positions]
                sub_data[node_type].generation = bus_gen[positions]
            link_index = torch.stack(
                [torch.arange(positions.numel(), device=device), positions]
            )
            sub_data[node_type, f"{node_type}_link", "bus"].edge_index = link_index
            sub_data["bus", f"{node_type}_link", node_type].edge_index = link_index.flip(0)

    bus_map = bus_subset.clone()
    return sub_data, bus_map, voltage_mask, edge_map, va_offset


def _build_subproblem_vectorized(
    data: HeteroData,
    output_dict: dict,
    bus_subset_list: list,
    add_bus_type: Optional[bool] = None,
    detach_teacher: bool = True,
) -> tuple:
    """
    Batched equivalent of calling `build_subproblem` once per sample in
    `bus_subset_list` (in order) and concatenating the results -- does the
    whole transform (interior/tie split, tie-injection folding, slack
    promotion, node-type reconstruction) ONCE across the concatenation of
    every sample's own subset, instead of looping over samples in Python.

    This is possible with no change in behavior because almost all of
    `build_subproblem`'s math is already just elementwise/scatter ops
    against `data`'s full (already batch-wide) tensors, restricted via a
    `keep_mask`/`new_index` built from `bus_subset` -- looping per sample
    was mostly just repeating that SAME batch-wide computation from
    scratch, once per sample, plus paying for a GPU sync every single time
    regardless of whether that sample even needed it (`bool(keep_mask[
    global_slack_idx].any())`, to check whether the slack bus survived).
    Block-diagonal batching guarantees an edge's two endpoints always
    belong to the same original sample and different samples' bus_subsets
    never overlap, so building `keep_mask`/`new_index` from the UNION of
    every sample's subset is exactly equivalent to building it separately
    per sample -- nothing here needs sample boundaries EXCEPT slack
    promotion, which genuinely is per-sample-dependent (whether THIS
    sample's own slack bus survived, and which of THIS sample's own buses
    to promote if not). That's done as a single pair of segmented
    reductions (`scatter_reduce_` with `reduce="amax"`/`"amin"`, grouped by
    a `sample_id_sub` tensor) instead of a per-sample Python `argmax()
    .item()` -- reproducing `argmax()`'s own first-occurrence tie-breaking
    convention via an amin over (is-this-row-a-max ? global-row-index :
    +inf).

    Uses the exact same RNG-independent formulas as `build_subproblem`
    (bus_subset_list is assumed already sampled, e.g. by the same BFS loop
    build_subproblem_batch always used), so results are bit-for-bit
    identical to looping `build_subproblem` over the same subsets -- this
    is a performance-only rewrite, not a behavior change. See the
    `test_adjacency_cache`-style equivalence test referenced in the module
    history for how this was verified.

    Returns
    -------
    sub_data_list : list[HeteroData]
        One per sample (sliced back out of the vectorized union results),
        ready for `Batch.from_data_list` -- still delegated to PyG's own
        tested machinery for the actual ptr/batch/edge-increment batch
        bookkeeping rather than reimplementing it by hand.
    bus_map, voltage_mask, edge_map, va_offset : torch.Tensor
        Concatenated across the batch in the same sample order as
        `bus_subset_list`; see `build_subproblem`.
    """
    if add_bus_type is None:
        add_bus_type = "PQ" in data.node_types
    maybe_detach = (lambda t: t.detach()) if detach_teacher else (lambda t: t)

    device = data["bus"].x.device
    num_buses = data["bus"].num_nodes
    num_samples = len(bus_subset_list)

    sizes = [b.numel() for b in bus_subset_list]  # .numel() is metadata only, no sync
    # sample_bus_subset returns a plain CPU tensor regardless of `data`'s own
    # device (see its own implementation) -- build_subproblem always moved
    # this to `device` before use; do the same here for the concatenated
    # union, or every downstream indexing op against a GPU-resident `data`
    # fails with a device mismatch.
    bus_subset_all = torch.cat(bus_subset_list).to(device=device, dtype=torch.long)
    n_sub_total = bus_subset_all.numel()
    sizes_t = torch.tensor(sizes, device=device)
    sample_id_sub = torch.repeat_interleave(torch.arange(num_samples, device=device), sizes_t)

    keep_mask = torch.zeros(num_buses, dtype=torch.bool, device=device)
    keep_mask[bus_subset_all] = True
    new_index = torch.full((num_buses,), -1, dtype=torch.long, device=device)
    new_index[bus_subset_all] = torch.arange(n_sub_total, device=device)

    # ---- Branch edges: split into interior (both ends kept) / tie (one end kept) ----
    edge_index = data["bus", "branch", "bus"].edge_index
    edge_attr = data["bus", "branch", "bus"].edge_attr
    src, dst = edge_index[0], edge_index[1]
    src_in, dst_in = keep_mask[src], keep_mask[dst]
    interior_mask = src_in & dst_in
    tie_src_kept = src_in & ~dst_in
    tie_dst_kept = dst_in & ~src_in

    edge_preds = maybe_detach(output_dict["edge_preds"])
    bus_pred = maybe_detach(output_dict["bus"])

    # ---- Fold each cut tie-line's predicted flow into its retained endpoint's net injection ----
    extra_p = torch.zeros(num_buses, device=device)
    extra_q = torch.zeros(num_buses, device=device)
    extra_p.scatter_add_(0, src[tie_src_kept], edge_preds[tie_src_kept, 0])
    extra_q.scatter_add_(0, src[tie_src_kept], edge_preds[tie_src_kept, 1])
    extra_p.scatter_add_(0, dst[tie_dst_kept], edge_preds[tie_dst_kept, 2])
    extra_q.scatter_add_(0, dst[tie_dst_kept], edge_preds[tie_dst_kept, 3])

    bus_type = data["bus"].bus_type[bus_subset_all].clone()
    bus_demand = data["bus"].bus_demand[bus_subset_all].clone()
    bus_gen = data["bus"].bus_gen[bus_subset_all].clone()
    bus_voltages = data["bus"].bus_voltages[bus_subset_all].clone()
    bus_shunts = data["bus"].shunt[bus_subset_all].clone()
    voltage_limits = data["bus"].limits[bus_subset_all].clone()

    bus_demand[:, 0] = bus_demand[:, 0] + extra_p[bus_subset_all]
    bus_demand[:, 1] = bus_demand[:, 1] + extra_q[bus_subset_all]

    new_src = new_index[src[interior_mask]]
    new_dst = new_index[dst[interior_mask]]

    # ---- Segmented slack promotion (the one genuinely per-sample-dependent
    # step) -- always computed; the torch.where masking below makes it a
    # no-op for any sample that already kept its own slack bus, so there's
    # no need for a per-sample conditional (which would cost its own sync
    # to evaluate anyway). ----
    slack_present_row = (bus_type == 3).float()
    slack_kept_per_sample = torch.zeros(num_samples, device=device)
    slack_kept_per_sample.scatter_reduce_(
        0, sample_id_sub, slack_present_row, reduce="amax", include_self=True
    )
    slack_kept_per_sample = slack_kept_per_sample > 0.5

    degree = torch.zeros(n_sub_total, device=device)
    degree.scatter_add_(0, new_src, torch.ones_like(new_src, dtype=torch.float))
    degree.scatter_add_(0, new_dst, torch.ones_like(new_dst, dtype=torch.float))

    pv_mask_pre = bus_type == 2
    has_pv_in_sample = torch.zeros(num_samples, device=device)
    has_pv_in_sample.scatter_reduce_(
        0, sample_id_sub, pv_mask_pre.float(), reduce="amax", include_self=True
    )
    has_pv_in_sample = has_pv_in_sample > 0.5
    prefer_pv_row = has_pv_in_sample[sample_id_sub]
    neg_inf = float("-inf")
    # Candidates: PV rows only for a sample that has one (its reactive
    # generation is a genuine model prediction, worth comparing downstream
    # -- see build_subproblem's own docstring), else every row.
    candidate_degree = torch.where(
        prefer_pv_row, torch.where(pv_mask_pre, degree, neg_inf), degree
    )
    seg_max = torch.full((num_samples,), neg_inf, device=device)
    seg_max.scatter_reduce_(0, sample_id_sub, candidate_degree, reduce="amax", include_self=True)
    is_seg_max_row = candidate_degree == seg_max[sample_id_sub]
    row_idx = torch.arange(n_sub_total, device=device)
    # First-occurrence tie-break, matching plain .argmax()'s own convention:
    # among rows tied for the max, keep the smallest global row index --
    # which, since bus_subset_list is concatenated sample-major, is also
    # the smallest LOCAL row index within that sample.
    # Done in float rather than int64 -- some backends (e.g. MPS pre-macOS
    # 15) don't support amin/amax scatter_reduce on integer dtypes at all,
    # and float32 is exact for integers well past any realistic n_sub_total
    # here (batch-wide subgraph bus count), so there's no precision cost.
    tie_break = torch.where(is_seg_max_row, row_idx.float(), float(n_sub_total))
    promote_pos_global_f = torch.full((num_samples,), float(n_sub_total), device=device)
    promote_pos_global_f.scatter_reduce_(
        0, sample_id_sub, tie_break, reduce="amin", include_self=True
    )
    promote_pos_global = promote_pos_global_f.long()

    promote_this_row = torch.zeros(n_sub_total, dtype=torch.bool, device=device)
    needs_promotion_sample = ~slack_kept_per_sample
    promote_this_row[promote_pos_global[needs_promotion_sample]] = True

    bus_type = torch.where(promote_this_row, torch.full_like(bus_type, 3), bus_type)

    promoted_bus_global_idx = bus_subset_all[promote_pos_global]
    promoted_teacher_va = bus_pred[promoted_bus_global_idx, 0]
    promoted_teacher_vm = bus_pred[promoted_bus_global_idx, 1]

    voltage_mask = ~promote_this_row
    va_offset = torch.where(
        slack_kept_per_sample[sample_id_sub],
        torch.zeros(n_sub_total, device=device),
        promoted_teacher_va[sample_id_sub],
    )
    bus_voltages = bus_voltages.clone()
    bus_voltages[promote_this_row, 1] = promoted_teacher_vm[sample_id_sub][promote_this_row]
    bus_voltages[promote_this_row, 0] = 0.0

    pq_mask = bus_type == 1
    pv_mask = bus_type == 2
    slack_mask = bus_type == 3

    pf_x = torch.zeros(n_sub_total, 2, device=device)
    pf_x[pq_mask] = bus_demand[pq_mask]
    pf_x[pv_mask, 0] = bus_gen[pv_mask, 0] - bus_demand[pv_mask, 0]
    pf_x[pv_mask, 1] = bus_voltages[pv_mask, 1]
    pf_x[slack_mask] = bus_voltages[slack_mask]

    edge_index_sub = torch.stack([new_src, new_dst])
    edge_attr_sub = edge_attr[interior_mask]
    has_edge_limits = "edge_limits" in data["bus", "branch", "bus"]
    edge_limits_sub = (
        data["bus", "branch", "bus"].edge_limits[interior_mask] if has_edge_limits else None
    )
    edge_map_all = interior_mask.nonzero(as_tuple=True)[0]

    # Per-sample edge counts -- recovered the same way bus counts were
    # (edge_index_sub is already sample-major, since new_src/new_dst are),
    # needed to slice edge_index_sub/edge_attr_sub back apart below (edge
    # counts vary independently of bus counts, so `sizes` alone isn't
    # enough).
    edge_sample_id = sample_id_sub[new_src]
    edge_sizes = torch.zeros(num_samples, dtype=torch.long, device=device)
    edge_sizes.scatter_add_(0, edge_sample_id, torch.ones_like(edge_sample_id))
    edge_sizes_list = edge_sizes.tolist()

    # ---- Slice the vectorized union results back into per-sample HeteroData
    # objects. Still a Python loop, but every op in it is a cheap tensor
    # slice/construction -- none of it forces a GPU sync the way the old
    # per-sample loop's `bool(...)`/`.item()` calls did, so the GPU queue
    # never blocks waiting on the CPU between iterations. ----
    sub_data_list = []
    bus_offset = 0
    edge_offset = 0
    for k in range(num_samples):
        n_k = sizes[k]
        e_k = edge_sizes_list[k]
        bus_slice = slice(bus_offset, bus_offset + n_k)
        edge_slice = slice(edge_offset, edge_offset + e_k)

        sub_data = HeteroData()
        sub_data["bus"].x = pf_x[bus_slice]
        sub_data["bus"].num_nodes = n_k
        sub_data["bus"].bus_gen = bus_gen[bus_slice]
        sub_data["bus"].bus_demand = bus_demand[bus_slice]
        sub_data["bus"].bus_voltages = bus_voltages[bus_slice]
        sub_data["bus"].bus_type = bus_type[bus_slice]
        sub_data["bus"].shunt = bus_shunts[bus_slice]
        sub_data["bus"].limits = voltage_limits[bus_slice]

        sub_data["bus", "branch", "bus"].edge_index = edge_index_sub[:, edge_slice] - bus_offset
        sub_data["bus", "branch", "bus"].edge_attr = edge_attr_sub[edge_slice]
        if has_edge_limits:
            sub_data["bus", "branch", "bus"].edge_limits = edge_limits_sub[edge_slice]

        if "gen" in data.node_types or "load" in data.node_types:
            # _remap_node_type filters data[node_type] down to whichever
            # entries attach to a KEPT bus -- the global (union) keep_mask/
            # new_index would incorrectly pull in every OTHER sample's gen/
            # load nodes too, not just this one's, so scope both to this
            # sample's own subset instead (off the hot path: CANOS's own
            # dataset never populates gen/load -- see build_subproblem's
            # own comment above -- so this only runs for other dataset
            # variants that do).
            # Use bus_subset_all's own slice (already migrated to `device`
            # above), not bus_subset_list[k] directly -- sample_bus_subset
            # always returns a plain CPU tensor regardless of `data`'s own
            # device, so indexing a GPU tensor with it would fail exactly
            # like the bug fixed above for bus_subset_all itself.
            keep_mask_k = torch.zeros(num_buses, dtype=torch.bool, device=device)
            keep_mask_k[bus_subset_all[bus_slice]] = True
            new_index_k = torch.full((num_buses,), -1, dtype=torch.long, device=device)
            new_index_k[bus_subset_all[bus_slice]] = torch.arange(n_k, device=device)
            if "gen" in data.node_types:
                _remap_node_type(data, sub_data, "gen", "gen_link", keep_mask_k, new_index_k)
            if "load" in data.node_types:
                _remap_node_type(data, sub_data, "load", "load_link", keep_mask_k, new_index_k)

        if add_bus_type:
            local_pf_x = pf_x[bus_slice]
            local_bus_demand = bus_demand[bus_slice]
            local_bus_gen = bus_gen[bus_slice]
            local_masks = {
                "PQ": pq_mask[bus_slice],
                "PV": pv_mask[bus_slice],
                "slack": slack_mask[bus_slice],
            }
            for node_type, mask in local_masks.items():
                positions = mask.nonzero(as_tuple=True)[0]
                sub_data[node_type].x = local_pf_x[positions]
                if node_type != "PQ":
                    sub_data[node_type].demand = local_bus_demand[positions]
                    sub_data[node_type].generation = local_bus_gen[positions]
                link_index = torch.stack(
                    [torch.arange(positions.numel(), device=device), positions]
                )
                sub_data[node_type, f"{node_type}_link", "bus"].edge_index = link_index
                sub_data["bus", f"{node_type}_link", node_type].edge_index = link_index.flip(0)

        sub_data_list.append(sub_data)
        bus_offset += n_k
        edge_offset += e_k

    return sub_data_list, bus_subset_all, voltage_mask, edge_map_all, va_offset


def build_subproblem_batch(
    data,
    output_dict: dict,
    min_size=10,
    max_size=100,
    add_bus_type: Optional[bool] = None,
    detach_teacher: bool = True,
    generator: Optional[torch.Generator] = None,
    adjacency_cache: Optional[dict] = None,
    verify_every: int = 200,
    stats: Optional[dict] = None,
) -> tuple:
    """
    Apply `build_subproblem` once per sample in a batched `data`, sampling an
    independent random connected bus subset per sample, and re-batch the
    results into a single HeteroData batch ready for another forward pass.

    Returns the same four outputs as `build_subproblem`, concatenated across
    the batch in sample order -- which is also the order `Batch.from_data_
    list` lays each sample's nodes/edges out in, so `bus_map`/`voltage_mask`/
    `edge_map` can be indexed directly against a subsequent forward pass on
    the returned `sub_data`.

    Parameters
    ----------
    data : HeteroData or Batch
        Batched full-grid input that produced `output_dict`.
    output_dict : dict
        See `build_subproblem`.
    min_size, max_size : int
        Per-sample subgraph size bounds (absolute bus counts), passed to
        `sample_bus_subset`.
    add_bus_type : bool, optional
        See `build_subproblem`.
    detach_teacher : bool
        See `build_subproblem`.
    generator : torch.Generator, optional
        RNG for reproducible sampling.
    adjacency_cache : dict, optional
        Reused across calls (e.g. one dict owned by a long-lived
        SubproblemConsistencyLoss instance) to skip rebuilding the adjacency
        list when the branch topology hasn't changed since the last call --
        common when every sample is drawn from the same case with no
        contingency perturbation, so the topology is identical run-to-run
        and only the demand/generation values vary. Self-verifying, but only
        every `verify_every` calls (not every single one): on a shape match
        it's trusted without a full `torch.equal` most of the time, since on
        GPU tensors that comparison forces a device sync -- and because
        PyTorch dispatches CUDA work asynchronously, that sync doesn't just
        cost the comparison itself, it blocks on everything queued before it
        (the model's own forward pass, any preceding loss computation),
        which otherwise wouldn't actually be waited on until something later
        forces a sync anyway. Doing that every call was misattributing most
        of a training step's real GPU time to this cache check. A shape
        *mismatch* (e.g. a genuinely different case) is always caught
        immediately and triggers an unconditional rebuild + re-verify,
        regardless of `verify_every` -- only a same-shape-but-different-
        content change (topology drift with an identical bus/edge count,
        which nothing else in this pipeline currently produces) could go
        undetected for up to `verify_every` calls.
    verify_every : int, optional
        How often (in calls with a shape-matching cache) to pay for a full
        `torch.equal` re-verification instead of trusting the cache
        outright. Default 200. Irrelevant if `adjacency_cache` is None.
    stats : dict, optional
        Opt-in profiling: if given, accumulates wall-clock seconds (summed
        across repeated calls) under keys "adjacency", "sample_bus_subset",
        "build_subproblem", and "batch_from_data_list" -- see `_timed`.

    Returns
    -------
    sub_data : Batch
    bus_map, voltage_mask, edge_map, va_offset : torch.Tensor
        Concatenated across the batch; see `build_subproblem`.
    """
    bus_store = data["bus"]
    device = bus_store.x.device
    num_buses = bus_store.num_nodes
    ptr = bus_store.ptr if "ptr" in bus_store else torch.tensor([0, num_buses], device=device)

    edge_index = data["bus", "branch", "bus"].edge_index
    with _timed(stats, "adjacency"):
        cached_edge_index = (
            adjacency_cache.get("edge_index") if adjacency_cache is not None else None
        )
        if cached_edge_index is not None and cached_edge_index.shape == edge_index.shape:
            adjacency_cache["calls_since_verify"] = (
                adjacency_cache.get("calls_since_verify", 0) + 1
            )
            if adjacency_cache["calls_since_verify"] < verify_every:
                adjacency = adjacency_cache["adjacency"]
            elif torch.equal(cached_edge_index, edge_index):
                adjacency_cache["calls_since_verify"] = 0
                adjacency = adjacency_cache["adjacency"]
            else:
                adjacency = build_adjacency(edge_index, num_buses)
                adjacency_cache["edge_index"] = edge_index.detach().clone()
                adjacency_cache["adjacency"] = adjacency
                adjacency_cache["calls_since_verify"] = 0
        else:
            adjacency = build_adjacency(edge_index, num_buses)
            if adjacency_cache is not None:
                adjacency_cache["edge_index"] = edge_index.detach().clone()
                adjacency_cache["adjacency"] = adjacency
                adjacency_cache["calls_since_verify"] = 0

    # Sample a connected bus subset per graph. Still a Python loop -- BFS
    # growth is inherently sequential (see sample_bus_subset) and already
    # cheap/CPU-only (adjacency is a plain Python list of lists, and
    # `generator` is a CPU torch.Generator, so nothing here touches the
    # GPU) -- but `ptr` itself is read to host ONCE up front instead of
    # `.item()`-ing it twice per sample, which forced a GPU sync on every
    # iteration for no reason (ptr is tiny and never changes mid-loop). The
    # actual per-sample GPU-sync cost lived in the transform below, not
    # here -- see _build_subproblem_vectorized's docstring.
    with _timed(stats, "sample_bus_subset"):
        ptr_list = ptr.tolist()
        bus_subset_list = []
        for k in range(len(ptr_list) - 1):
            lo, hi = ptr_list[k], ptr_list[k + 1]
            if hi <= lo:
                continue
            seed_bus = torch.randint(lo, hi, (1,), generator=generator).item()
            bus_subset_list.append(
                sample_bus_subset(adjacency, seed_bus, min_size, max_size, generator=generator)
            )

    with _timed(stats, "build_subproblem"):
        (
            sub_data_list,
            bus_map_all,
            voltage_mask_all,
            edge_map_all,
            va_offset_all,
        ) = _build_subproblem_vectorized(
            data,
            output_dict,
            bus_subset_list,
            add_bus_type=add_bus_type,
            detach_teacher=detach_teacher,
        )

    with _timed(stats, "batch_from_data_list"):
        sub_data_batch = Batch.from_data_list(sub_data_list)

    return sub_data_batch, bus_map_all, voltage_mask_all, edge_map_all, va_offset_all
