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


def build_subproblem_batch(
    data,
    output_dict: dict,
    min_size=10,
    max_size=100,
    add_bus_type: Optional[bool] = None,
    detach_teacher: bool = True,
    generator: Optional[torch.Generator] = None,
    adjacency_cache: Optional[dict] = None,
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
        and only the demand/generation values vary. Self-verifying (checked
        via `torch.equal` against the edge_index the cached adjacency was
        built from), so it's never a correctness risk even if the topology
        does vary between calls -- it just falls back to always rebuilding.
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
        if (
            adjacency_cache is not None
            and adjacency_cache.get("edge_index") is not None
            and adjacency_cache["edge_index"].shape == edge_index.shape
            and torch.equal(adjacency_cache["edge_index"], edge_index)
        ):
            adjacency = adjacency_cache["adjacency"]
        else:
            adjacency = build_adjacency(edge_index, num_buses)
            if adjacency_cache is not None:
                adjacency_cache["edge_index"] = edge_index.detach().clone()
                adjacency_cache["adjacency"] = adjacency

    sub_data_list, bus_maps, voltage_masks, edge_maps, va_offsets = [], [], [], [], []
    for k in range(ptr.numel() - 1):
        lo, hi = int(ptr[k].item()), int(ptr[k + 1].item())
        if hi <= lo:
            continue
        seed_bus = torch.randint(lo, hi, (1,), generator=generator).item()
        with _timed(stats, "sample_bus_subset"):
            bus_subset = sample_bus_subset(
                adjacency, seed_bus, min_size, max_size, generator=generator
            )
        with _timed(stats, "build_subproblem"):
            sub_data, bus_map, voltage_mask, edge_map, va_offset = build_subproblem(
                data,
                output_dict,
                bus_subset,
                add_bus_type=add_bus_type,
                detach_teacher=detach_teacher,
            )
        sub_data_list.append(sub_data)
        bus_maps.append(bus_map)
        voltage_masks.append(voltage_mask)
        edge_maps.append(edge_map)
        va_offsets.append(va_offset)

    with _timed(stats, "batch_from_data_list"):
        sub_data_batch = Batch.from_data_list(sub_data_list)
    bus_map_all = torch.cat(bus_maps)
    voltage_mask_all = torch.cat(voltage_masks)
    edge_map_all = torch.cat(edge_maps)
    va_offset_all = torch.cat(va_offsets)

    return sub_data_batch, bus_map_all, voltage_mask_all, edge_map_all, va_offset_all
