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

import random as pyrandom
import time
from contextlib import contextmanager
from typing import Optional

import torch
from torch_geometric.data import HeteroData


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


def _bump(stats: Optional[dict], key: str) -> None:
    """No-op when `stats` is None; otherwise increments an integer counter
    at `stats[key]` -- same dict/reset lifecycle as `_timed`'s wall-clock
    entries (see SubgraphFinetuneTrainer), but a plain count rather than
    seconds. Key must end in `#count` so the trainer's printer knows to
    format it as an integer instead of a "%.3fs" duration."""
    assert key.endswith("#count"), f"_bump key must end in '#count', got {key!r}"
    if stats is None:
        return
    stats[key] = stats.get(key, 0) + 1


def build_adjacency(edge_index: torch.Tensor, num_nodes: int) -> list:
    """Build an undirected adjacency list from a directed branch edge_index."""
    adjacency = [[] for _ in range(num_nodes)]
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for i, j in zip(src, dst):
        adjacency[i].append(j)
        adjacency[j].append(i)
    return adjacency


def _canonical_edge_keys(src: torch.Tensor, dst: torch.Tensor, num_nodes_per_case: int) -> torch.Tensor:
    """Order-independent per-edge integer key (so (i,j) and (j,i) collide),
    for set operations over undirected edges."""
    lo = torch.minimum(src, dst)
    hi = torch.maximum(src, dst)
    return lo * num_nodes_per_case + hi


def update_base_topology_cache(
    edge_index: torch.Tensor,
    ptr: torch.Tensor,
    num_nodes_per_case: int,
    base_cache: dict,
) -> bool:
    """
    Maintains `base_cache` (a persistent dict, e.g. owned by a long-lived
    SubproblemConsistencyLoss instance) as the UNION of every distinct
    branch edge seen across every call -- the "believed full" topology for
    one case, in a case-LOCAL (0..num_nodes_per_case-1) index space shared
    across every sample regardless of which of that case's buses it was
    batched at.

    Exists because contingency samples (N-1/N-2 -- see the `perturbation`
    parameter on core.datasets.pfdelta_dataset.PFDeltaDataset) each carry
    only THEIR OWN active branches, so a batch's total edge count/shape
    essentially never repeats between batches -- a plain per-batch
    adjacency cache (comparing whole-batch edge_index tensors) was
    observed hitting its expensive rebuild path on ~95% of calls as a
    result. But the underlying topology any one contingency is a small
    (1-2 branch) deviation FROM is the SAME fixed case every time, and a
    single batch of samples already covers nearly all of it (a given
    branch is only offline in a small fraction of scenarios) -- so this
    converges almost immediately (usually the very first batch) and then
    rarely changes again, at which point per-batch cost drops to a few
    cheap vectorized set-membership checks instead of a full Python-loop
    rebuild over the whole batch's edges every time.

    Correctness doesn't depend on how fast this converges -- any later
    batch that reveals a not-yet-seen branch (e.g. a rare contingency
    combination the first several batches happened not to include) simply
    triggers another (still cheap, bounded by num_nodes_per_case's own
    edge count, not batch size) rebuild; efficiency does, but real
    datasets converge fast since a branch has to be simultaneously offline
    in EVERY sample of EVERY batch seen so far to still be missing.

    Populates/refreshes `base_cache["base_keys"]` (sorted unique canonical
    edge keys, LOCAL), `base_cache["base_adjacency"]` (list[list[int]],
    size num_nodes_per_case, LOCAL indices), and `base_cache["degrees"]`
    (list[int], per-node degree over that same base adjacency -- used by
    `sample_bus_subset_snowball`) in place.

    Returns
    -------
    bool
        True if the base topology actually changed this call (i.e. a
        previously-unseen branch was discovered) -- for profiling/counters,
        not required for correctness.
    """
    src, dst = edge_index[0], edge_index[1]
    # Which original sample each edge belongs to -- block-diagonal batching
    # means an edge's two endpoints always share a sample, so its source
    # bus's sample id is exactly its own.
    sample_id = torch.bucketize(src, ptr[1:], right=True)
    local_src = src - ptr[sample_id]
    local_dst = dst - ptr[sample_id]
    batch_keys = _canonical_edge_keys(local_src, local_dst, num_nodes_per_case)

    base_keys = base_cache.get("base_keys")
    if base_keys is None:
        new_keys = torch.unique(batch_keys)
        changed = True
    else:
        unseen = batch_keys[~torch.isin(batch_keys, base_keys)]
        changed = bool(unseen.numel() > 0)
        new_keys = torch.unique(torch.cat([base_keys, unseen])) if changed else base_keys

    if changed:
        base_cache["base_keys"] = new_keys
        lo_list = (new_keys // num_nodes_per_case).tolist()
        hi_list = (new_keys % num_nodes_per_case).tolist()
        adjacency = [[] for _ in range(num_nodes_per_case)]
        for i, j in zip(lo_list, hi_list):
            adjacency[i].append(j)
            adjacency[j].append(i)
        base_cache["base_adjacency"] = adjacency
        base_cache["num_nodes_per_case"] = num_nodes_per_case
        # Per-node degree over the base topology -- only used by
        # sample_bus_subset_snowball's degree-inverse weighting, but cheap
        # to keep in sync here (recomputed only when the base topology
        # itself changes, same trigger as the adjacency list above) rather
        # than recomputed by every snowball draw.
        base_cache["degrees"] = [len(neighbors) for neighbors in adjacency]
    return changed


def missing_edges_per_sample(
    edge_index: torch.Tensor,
    ptr: torch.Tensor,
    num_nodes_per_case: int,
    base_cache: dict,
) -> list:
    """
    For each sample in a batch, which of the cached base topology's edges
    (see `update_base_topology_cache`, which MUST be called first on this
    same `edge_index`/`ptr` so `base_cache` actually covers every edge this
    batch could be missing) are absent from that sample's own active
    branches -- i.e. that sample's own contingency-removed lines, as
    (local_i, local_j) pairs. Vectorized across the whole batch at once
    (no per-sample/per-edge Python loop).

    Returns
    -------
    list[list[tuple[int, int]]]
        One list per sample (in ptr order), each a short list of (local_i,
        local_j) pairs missing for that sample -- empty for a sample with
        every base branch active.
    """
    base_keys = base_cache["base_keys"]
    num_samples = ptr.numel() - 1
    num_base_edges = base_keys.numel()

    src, dst = edge_index[0], edge_index[1]
    sample_id = torch.bucketize(src, ptr[1:], right=True)
    local_src = src - ptr[sample_id]
    local_dst = dst - ptr[sample_id]
    batch_keys = _canonical_edge_keys(local_src, local_dst, num_nodes_per_case)

    # Position of each batch edge within the sorted base_keys tensor --
    # a real position (not just an insertion point) precisely because
    # update_base_topology_cache guarantees every batch_keys entry is
    # already present in base_keys by the time this runs.
    positions = torch.searchsorted(base_keys, batch_keys)
    assert torch.equal(base_keys[positions], batch_keys), (
        "missing_edges_per_sample: batch_keys not covered by base_keys -- "
        "update_base_topology_cache must run first on this same batch."
    )

    present = torch.zeros(
        num_samples, num_base_edges, dtype=torch.bool, device=edge_index.device
    )
    present[sample_id, positions] = True
    missing_mask = ~present

    base_keys_cpu = base_keys.tolist()
    result = []
    for k in range(num_samples):
        missing_positions = missing_mask[k].nonzero(as_tuple=True)[0].tolist()
        result.append(
            [
                (base_keys_cpu[p] // num_nodes_per_case, base_keys_cpu[p] % num_nodes_per_case)
                for p in missing_positions
            ]
        )
    return result


def sample_bus_subset(
    adjacency: list,
    seed_bus: int,
    min_size,
    max_size,
    generator: Optional[torch.Generator] = None,
    bus_offset: int = 0,
    excluded_neighbors: Optional[dict] = None,
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
        Undirected adjacency list. By default (bus_offset=0), indexed by
        the SAME global bus index space as `seed_bus` -- e.g. from
        `build_adjacency` over a whole batch, since batched samples never
        share edges so growth from a seed inside one sample never crosses
        into another. Can instead be a single SHARED, case-LOCAL (0..
        num_nodes_per_case-1) adjacency list reused across every sample of
        that case regardless of each one's own contingency -- pass
        `bus_offset`/`excluded_neighbors` in that mode; see
        `update_base_topology_cache`/`missing_edges_per_sample` for why.
    seed_bus : int
        Global bus index to grow the subset from.
    min_size, max_size : int
        Target subset size bounds, as absolute bus counts.
    generator : torch.Generator, optional
        RNG used for size sampling and to seed this call's own local
        (plain-Python) RNG for neighbor shuffling -- see below.
    bus_offset : int, optional
        Subtracted from a global bus index before indexing into
        `adjacency`, added back before visiting/returning -- 0 (a no-op)
        for a plain global adjacency list. Pass a sample's own `ptr[k]`
        when `adjacency` is a shared case-local list.
    excluded_neighbors : dict[int, set[int]], optional
        Maps a LOCAL node id to the set of LOCAL neighbor ids to skip when
        traversing that node -- this sample's own missing (contingency-
        offline) edges, on top of whatever's already absent from
        `adjacency` itself.

    Returns
    -------
    torch.Tensor
        Sorted LongTensor of global bus indices in the sampled subset.
    """
    lo, hi = int(min_size), int(max_size)
    lo, hi = min(lo, hi), max(lo, hi)
    target_size = torch.randint(lo, hi + 1, (1,), generator=generator).item()

    # One draw from the shared torch.Generator seeds a plain-Python RNG for
    # this call's OWN traversal -- advances `generator`'s state exactly
    # once (keeping the overall stream reproducible given a fixed top-level
    # seed), while avoiding a `torch.randperm` tensor allocation + kernel
    # dispatch for EVERY node visited during growth (up to `target_size` of
    # them, times however many subgraphs are drawn per sample) -- that
    # per-call overhead, not the shuffling itself, was the actual cost:
    # `random.shuffle` on the plain neighbor list has none of it and is
    # markedly cheaper for the small (single-digit to a few dozen)
    # neighbor lists BFS actually shuffles here.
    local_seed = torch.randint(0, 2**31 - 1, (1,), generator=generator).item()
    rng = pyrandom.Random(local_seed)

    # Grow the BFS, shuffling each shell, stopping once the target size is
    # hit -- so repeated calls from the same seed still vary.
    visited = {seed_bus}
    kept = [seed_bus]
    frontier = [seed_bus]
    while frontier and len(kept) < target_size:
        next_frontier = []
        for node in frontier:
            node_local = node - bus_offset
            neighbors_local = adjacency[node_local]
            excluded = excluded_neighbors.get(node_local) if excluded_neighbors else None
            shuffled_neighbors = neighbors_local[:]
            rng.shuffle(shuffled_neighbors)
            for neighbor_local in shuffled_neighbors:
                if excluded is not None and neighbor_local in excluded:
                    continue
                neighbor = neighbor_local + bus_offset
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


def _weighted_shuffle(items: list, weights: list, rng: pyrandom.Random) -> list:
    """
    Weighted random permutation without replacement (Efraimidis-Spirakis
    A-Res algorithm): draw `key = u ** (1/weight)` for `u ~ Uniform(0, 1)`
    per item, then sort by key descending. A higher-weight item is more
    likely to land earlier, but nothing is excluded -- every item still
    appears exactly once, same contract as `rng.shuffle`. Reduces to a
    plain uniform shuffle when every weight is equal.
    """
    keyed = [
        (rng.random() ** (1.0 / w) if w > 0 else 0.0, item)
        for item, w in zip(items, weights)
    ]
    keyed.sort(key=lambda kv: kv[0], reverse=True)
    return [item for _, item in keyed]


def sample_bus_subset_random_walk(
    adjacency: list,
    seed_bus: int,
    min_size,
    max_size,
    generator: Optional[torch.Generator] = None,
    bus_offset: int = 0,
    excluded_neighbors: Optional[dict] = None,
    restart_prob: float = 0.15,
    max_steps_factor: int = 50,
) -> torch.Tensor:
    """
    Random walk with restart (RWR): from `current` (starting at `seed_bus`),
    each step either jumps back to `seed_bus` (probability `restart_prob`)
    or moves to one random neighbor of `current`; every node visited this
    way is added to the growing subset. Stops once the target size (sampled
    uniformly from [min_size, max_size], same as `sample_bus_subset`) is
    hit, or after `max_steps_factor * target_size` steps (bounded, so a
    small/sparse region can't spin forever) -- returns a smaller subset
    than requested in that case, same "stops early rather than failing"
    contract as `sample_bus_subset`.

    The walk's own trace is always connected back to `seed_bus` (every
    node is reached by an actual edge traversal from somewhere already in
    the trace), same connectivity guarantee as BFS -- just a different
    (locally-biased, revisits allowed) exploration shape: lower
    `restart_prob` lets the walk wander further from `seed_bus` before
    reseeding, higher `restart_prob` keeps it concentrated nearby.

    Parameters mirror `sample_bus_subset` (`adjacency`/`bus_offset`/
    `excluded_neighbors`), plus:

    restart_prob : float
        Probability of jumping back to `seed_bus` at each step, instead of
        moving to a neighbor of the current node.
    max_steps_factor : int
        Step budget, as a multiple of the target size.
    """
    lo, hi = int(min_size), int(max_size)
    lo, hi = min(lo, hi), max(lo, hi)
    target_size = torch.randint(lo, hi + 1, (1,), generator=generator).item()
    local_seed = torch.randint(0, 2**31 - 1, (1,), generator=generator).item()
    rng = pyrandom.Random(local_seed)

    visited = {seed_bus}
    kept = [seed_bus]
    current = seed_bus
    max_steps = max(target_size * max_steps_factor, 100)

    for _ in range(max_steps):
        if len(kept) >= target_size:
            break
        if rng.random() < restart_prob:
            current = seed_bus
            continue
        current_local = current - bus_offset
        neighbors_local = adjacency[current_local]
        if not neighbors_local:
            current = seed_bus  # dead end (isolated node) -- reseed
            continue
        excluded = excluded_neighbors.get(current_local) if excluded_neighbors else None
        # One random neighbor, respecting exclusions. Retries (bounded by
        # this node's own degree) rather than filtering the whole list up
        # front -- cheap since exclusions are typically 0-2 entries (an
        # N-1/N-2 contingency), so this converges on the first try almost
        # always.
        neighbor_local = None
        for _ in range(len(neighbors_local)):
            candidate = rng.choice(neighbors_local)
            if excluded is not None and candidate in excluded:
                continue
            neighbor_local = candidate
            break
        if neighbor_local is None:
            current = seed_bus  # every neighbor excluded -- reseed
            continue
        current = neighbor_local + bus_offset
        if current not in visited:
            visited.add(current)
            kept.append(current)

    return torch.tensor(sorted(kept), dtype=torch.long)


def sample_bus_subset_forest_fire(
    adjacency: list,
    seed_bus: int,
    min_size,
    max_size,
    generator: Optional[torch.Generator] = None,
    bus_offset: int = 0,
    excluded_neighbors: Optional[dict] = None,
    p: float = 0.3,
) -> torch.Tensor:
    """
    Forest Fire sampling (Leskovec et al.): grows shell-by-shell like BFS,
    but at each "burning" node only a RANDOM SUBSET of its unvisited
    neighbors catches fire, rather than all of them -- the "forward-
    burning" count is drawn from a Geometric(1-p) distribution (repeatedly
    include one more candidate with probability `p`, stop with probability
    `1-p`), applied to a randomly-ordered candidate list, so which specific
    neighbors get chosen is also random. Same connectivity guarantee as
    BFS (grown via real edge traversals from `seed_bus`), but a
    structurally different, more "bursty"/tree-like shape controlled by
    `p`: near 1, behaves close to full BFS (almost everything catches
    fire); near 0, fires tend to die out quickly.

    Unlike `sample_bus_subset`/the random walk above, a fire dying out
    (every burning node's draw comes up empty) is expected, authentic
    forest-fire behavior, not just a size-budget shortfall -- for a low
    `p` this can return a subset well short of `max_size` even in a large,
    densely-connected region, which is the actual structural property this
    sampling model is meant to exhibit.

    Parameters mirror `sample_bus_subset`, plus:

    p : float
        Forward-burning probability -- higher spreads wider/faster.
    """
    lo, hi = int(min_size), int(max_size)
    lo, hi = min(lo, hi), max(lo, hi)
    target_size = torch.randint(lo, hi + 1, (1,), generator=generator).item()
    local_seed = torch.randint(0, 2**31 - 1, (1,), generator=generator).item()
    rng = pyrandom.Random(local_seed)

    visited = {seed_bus}
    kept = [seed_bus]
    frontier = [seed_bus]
    while frontier and len(kept) < target_size:
        next_frontier = []
        for node in frontier:
            if len(kept) >= target_size:
                break
            node_local = node - bus_offset
            neighbors_local = adjacency[node_local]
            excluded = excluded_neighbors.get(node_local) if excluded_neighbors else None
            candidates = [
                n for n in neighbors_local
                if (n + bus_offset) not in visited
                and (excluded is None or n not in excluded)
            ]
            if not candidates:
                continue
            rng.shuffle(candidates)
            burn_count = 0
            while burn_count < len(candidates) and rng.random() < p:
                burn_count += 1
            for neighbor_local in candidates[:burn_count]:
                neighbor = neighbor_local + bus_offset
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                kept.append(neighbor)
                next_frontier.append(neighbor)
                if len(kept) >= target_size:
                    break
        frontier = next_frontier

    return torch.tensor(sorted(kept), dtype=torch.long)


def sample_bus_subset_snowball(
    adjacency: list,
    seed_bus: int,
    min_size,
    max_size,
    generator: Optional[torch.Generator] = None,
    bus_offset: int = 0,
    excluded_neighbors: Optional[dict] = None,
    degrees: Optional[list] = None,
    degree_weight_power: float = 1.0,
) -> torch.Tensor:
    """
    Snowball sampling with degree-inverse weighting: identical BFS growth
    structure to `sample_bus_subset` (same shells, same stopping rule, same
    connectivity guarantee), but each shell's neighbor visitation order is
    a WEIGHTED random permutation (weight ∝ 1/degree(neighbor)^
    degree_weight_power -- see `_weighted_shuffle`) instead of a uniform
    one, so lower-degree ("peripheral") buses are more likely to be
    included before the target size is hit than high-degree hub buses are.
    `degree_weight_power=0` reduces exactly to plain BFS's uniform shuffle
    (every weight becomes 1); larger values bias more strongly toward
    low-degree buses.

    Parameters mirror `sample_bus_subset`, plus:

    degrees : list[int], optional
        Per-(case-local)-node degree, e.g. `[len(n) for n in adjacency]` --
        cached alongside `adjacency` itself (see `update_base_topology_
        cache`) since it depends only on the (rarely-changing) base
        topology, not on any one sample's own contingency. Required
        whenever `degree_weight_power != 0`; with it 0, weighting is
        skipped entirely and this parameter is unused.
    degree_weight_power : float
        Exponent on the inverse-degree weight.
    """
    lo, hi = int(min_size), int(max_size)
    lo, hi = min(lo, hi), max(lo, hi)
    target_size = torch.randint(lo, hi + 1, (1,), generator=generator).item()
    local_seed = torch.randint(0, 2**31 - 1, (1,), generator=generator).item()
    rng = pyrandom.Random(local_seed)

    visited = {seed_bus}
    kept = [seed_bus]
    frontier = [seed_bus]
    use_weights = degree_weight_power != 0 and degrees is not None
    while frontier and len(kept) < target_size:
        next_frontier = []
        for node in frontier:
            node_local = node - bus_offset
            neighbors_local = adjacency[node_local]
            excluded = excluded_neighbors.get(node_local) if excluded_neighbors else None
            candidates = [
                n for n in neighbors_local
                if excluded is None or n not in excluded
            ]
            if not candidates:
                continue
            if use_weights:
                weights = [1.0 / max(degrees[n], 1) ** degree_weight_power for n in candidates]
                ordered = _weighted_shuffle(candidates, weights, rng)
            else:
                ordered = candidates[:]
                rng.shuffle(ordered)
            for neighbor_local in ordered:
                neighbor = neighbor_local + bus_offset
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


# Registry of sampling strategies, keyed by the name used in a
# `sampling_strategies` config dict (see `build_subproblem_batch`). Every
# entry has the same core call signature (adjacency, seed_bus, min_size,
# max_size, generator=, bus_offset=, excluded_neighbors=), plus whatever
# strategy-specific tunable kwargs its own config entry supplies.
SAMPLING_STRATEGIES = {
    "bfs": sample_bus_subset,
    "random_walk": sample_bus_subset_random_walk,
    "forest_fire": sample_bus_subset_forest_fire,
    "snowball": sample_bus_subset_snowball,
}


def pick_sampling_strategy(sampling_strategies: dict, generator: Optional[torch.Generator] = None) -> str:
    """
    Weighted random choice of one strategy name from a `sampling_strategies`
    config dict (`{name: {"proportion": w, ...other kwargs}}`) -- weights
    need not sum to 1, they're normalized here. Uses the shared
    `torch.Generator` directly (one `torch.rand` draw), the same RNG this
    whole pipeline's other per-call choices (target size, local traversal
    seed) already come from, rather than spinning up a separate local RNG
    just for this one draw.
    """
    names = list(sampling_strategies.keys())
    weights = [float(sampling_strategies[n].get("proportion", 1.0)) for n in names]
    total = sum(weights)
    assert total > 0, "sampling_strategies proportions must sum to something positive"
    r = torch.rand(1, generator=generator).item() * total
    cum = 0.0
    for name, w in zip(names, weights):
        cum += w
        if r <= cum:
            return name
    return names[-1]  # floating-point safety net for r landing exactly on `total`


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

    Builds the final batched `sub_data` directly from the union tensors --
    no per-subgraph Python loop, no `Batch.from_data_list`. See the inline
    comment above the assembly code for why this is a safe (not just
    faster) replacement for this codebase's specific consumption pattern.

    Returns
    -------
    sub_data : HeteroData
        The full batch, ready for another forward pass.
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

    # ---- Assemble the final batched sub_data DIRECTLY from the union
    # tensors above -- no per-subgraph Python loop, no Batch.from_data_list.
    # This works with no extra bookkeeping because bus_subset_list is
    # concatenated sample-major, so every union tensor above is ALREADY
    # laid out exactly the way a properly batched object's own tensors
    # would be: `new_src`/`new_dst` (via `new_index`) are already GLOBAL
    # row indices into this same union "bus" numbering (no per-subgraph
    # offset subtraction needed, unlike the old per-sample-sliced version),
    # and a node-type mask's own `.nonzero()` over the union already comes
    # out in the same sample-major order Batch.from_data_list would
    # produce. The one thing that DOES need computing is each node type's
    # own `.ptr`/`.batch` (used by e.g. PowerBalanceLoss's global_mean_pool
    # and by SubproblemConsistencyLoss's own per-subgraph loss pooling) --
    # only ever consumed as plain tensor attributes here (see
    # SubproblemConsistencyLoss/PowerBalanceLoss/canos_utils -- none of
    # them call a Batch-specific method like .to_data_list()/.get_example()
    # /.num_graphs), so a plain HeteroData with these set by hand is
    # functionally identical to what Batch.from_data_list would have
    # returned for our purposes, without spending time reproducing its
    # full internal bookkeeping.
    ptr_sub = torch.cat([torch.zeros(1, dtype=torch.long, device=device), sizes_t.cumsum(0)])

    sub_data = HeteroData()
    sub_data["bus"].x = pf_x
    sub_data["bus"].num_nodes = n_sub_total
    sub_data["bus"].ptr = ptr_sub
    sub_data["bus"].batch = sample_id_sub
    sub_data["bus"].bus_gen = bus_gen
    sub_data["bus"].bus_demand = bus_demand
    sub_data["bus"].bus_voltages = bus_voltages
    sub_data["bus"].bus_type = bus_type
    sub_data["bus"].shunt = bus_shunts
    sub_data["bus"].limits = voltage_limits

    sub_data["bus", "branch", "bus"].edge_index = edge_index_sub
    sub_data["bus", "branch", "bus"].edge_attr = edge_attr_sub
    if has_edge_limits:
        sub_data["bus", "branch", "bus"].edge_limits = edge_limits_sub

    if "gen" in data.node_types or "load" in data.node_types:
        # _remap_node_type filters data[node_type] down to whichever
        # entries attach to a kept bus -- calling it ONCE with the UNION
        # keep_mask/new_index (rather than per-sample) is exactly
        # equivalent here (disjoint per-sample bus_subsets, same reasoning
        # as everything else above), since gen/load rows end up ordered by
        # their OWN link edges' order in `data`, which -- because block-
        # diagonal batching means a gen/load node only ever links to buses
        # within its own sample -- already groups sample-major too. Off the
        # hot path either way: CANOS's own dataset never populates gen/
        # load (see build_subproblem's own comment), so this only runs for
        # other dataset variants that do; .ptr/.batch aren't set for these
        # two types since nothing currently reads them.
        if "gen" in data.node_types:
            _remap_node_type(data, sub_data, "gen", "gen_link", keep_mask, new_index)
        if "load" in data.node_types:
            _remap_node_type(data, sub_data, "load", "load_link", keep_mask, new_index)

    if add_bus_type:
        type_masks = {"PQ": pq_mask, "PV": pv_mask, "slack": slack_mask}
        for node_type, mask in type_masks.items():
            positions = mask.nonzero(as_tuple=True)[0]  # global union "bus" row indices, sample-major
            type_sample_id = sample_id_sub[positions]
            sub_data[node_type].x = pf_x[positions]
            sub_data[node_type].num_nodes = positions.numel()
            sub_data[node_type].batch = type_sample_id
            sub_data[node_type].ptr = torch.searchsorted(
                type_sample_id, torch.arange(num_samples + 1, device=device)
            )
            if node_type != "PQ":
                sub_data[node_type].demand = bus_demand[positions]
                sub_data[node_type].generation = bus_gen[positions]
            link_index = torch.stack(
                [torch.arange(positions.numel(), device=device), positions]
            )
            sub_data[node_type, f"{node_type}_link", "bus"].edge_index = link_index
            sub_data["bus", f"{node_type}_link", node_type].edge_index = link_index.flip(0)

    return sub_data, bus_subset_all, voltage_mask, edge_map_all, va_offset


def build_subproblem_batch(
    data,
    output_dict: dict,
    min_size=10,
    max_size=100,
    add_bus_type: Optional[bool] = None,
    detach_teacher: bool = True,
    generator: Optional[torch.Generator] = None,
    adjacency_cache: Optional[dict] = None,
    subgraphs_per_sample: int = 1,
    sampling_strategies: Optional[dict] = None,
    stats: Optional[dict] = None,
) -> tuple:
    """
    Apply `build_subproblem` `subgraphs_per_sample` times per sample in a
    batched `data` (each an independently-sampled random connected bus
    subset), and re-batch every resulting subgraph -- across every sample
    -- into a single HeteroData batch ready for another forward pass.

    Returns the same four outputs as `build_subproblem`, concatenated in
    the order the subgraphs were generated (sample-major, and within a
    sample in draw order) -- which is also the order `Batch.from_data_list`
    lays each subgraph's nodes/edges out in, so `bus_map`/`voltage_mask`/
    `edge_map` can be indexed directly against a subsequent forward pass on
    the returned `sub_data`. Nothing downstream needs to know which
    subgraphs came from the same original sample -- e.g. `SubproblemConsistencyLoss`'s
    per-subgraph loss pooling treats every subgraph as its own independent
    unit regardless of provenance (see `_per_subgraph_mse`), and with a
    fixed `subgraphs_per_sample` applied uniformly to every sample, that's
    equivalent to weighting by original sample anyway (each sample
    contributes the same subgraph count, so a flat mean over all subgraphs
    already gives every sample equal total weight).

    Parameters
    ----------
    data : HeteroData or Batch
        Batched full-grid input that produced `output_dict`.
    output_dict : dict
        See `build_subproblem`.
    min_size, max_size : int
        Per-subgraph size bounds (absolute bus counts), passed to
        `sample_bus_subset`. Applied independently to every draw -- with
        `subgraphs_per_sample > 1`, a single original sample's own draws
        can land at different sizes within the same [min_size, max_size]
        range.
    add_bus_type : bool, optional
        See `build_subproblem`.
    detach_teacher : bool
        See `build_subproblem`.
    generator : torch.Generator, optional
        RNG for reproducible sampling.
    adjacency_cache : dict, optional
        Reused across calls (e.g. one dict owned by a long-lived
        SubproblemConsistencyLoss instance) as the base-topology cache for
        `update_base_topology_cache`/`missing_edges_per_sample` -- samples
        can each carry their own N-1/N-2 contingency (a small, per-sample
        deviation from a shared fixed topology; see the `perturbation`
        parameter on PFDeltaDataset), so caching a whole batch's adjacency
        wholesale doesn't help (a batch's total edge count/shape
        essentially never repeats). Instead this caches the UNION topology
        for the case ONCE (converges after the first batch or so) and
        tracks each sample's own small missing-edge patch against it,
        applied on the fly during BFS growth rather than rebuilt from
        scratch per sample. Only used when every sample in the batch has
        the same bus count (`ptr` differences all equal) -- true for any
        single case, contingency or not, since N-1/N-2 only removes
        branches, never buses. Falls back to a plain (uncached) full
        rebuild via `build_adjacency` otherwise, or if this is None.
    subgraphs_per_sample : int, optional
        How many independent subgraphs to draw per original sample in
        `data` (default 1, matching the original single-subgraph-per-
        sample behavior). Each is its own independent draw (own seed bus,
        own size sampled from [min_size, max_size], own sampling strategy
        if `sampling_strategies` is given) -- memory/compute for the
        resulting `sub_data` batch scales with the total node/edge count
        across all of them, same as increasing batch_size would.
    sampling_strategies : dict, optional
        `{strategy_name: {"proportion": w, **strategy_kwargs}}` -- e.g.
        `{"bfs": {"proportion": 0.5}, "random_walk": {"proportion": 0.5,
        "restart_prob": 0.2}}`. Each of the (subgraphs_per_sample *
        num_samples) draws independently picks one strategy at random,
        weighted by `proportion` (need not sum to 1 -- normalized; see
        `pick_sampling_strategy`), and calls it with `strategy_kwargs` on
        top of the shared min_size/max_size/generator/bus_offset/
        excluded_neighbors arguments every strategy in `SAMPLING_
        STRATEGIES` accepts. Draws are independent per call, not fixed per
        batch -- see this function's own module-level discussion of why
        that costs nothing extra (none of these strategies are vectorized
        across simultaneous draws to begin with). Defaults to None, i.e.
        always `sample_bus_subset` (plain BFS) -- the original, single-
        strategy behavior, with zero added dispatch overhead.
    stats : dict, optional
        Opt-in profiling: if given, accumulates wall-clock seconds (summed
        across repeated calls) under keys "adjacency", "sample_bus_subset",
        and "build_subproblem" (which now includes the whole batch
        assembly, not just the transform) -- see `_timed`.

    Returns
    -------
    sub_data : Batch
    bus_map, voltage_mask, edge_map, va_offset : torch.Tensor
        Concatenated across the batch; see `build_subproblem`.
    strategy_per_subgraph : list[str] or None
        Which entry of `sampling_strategies` produced each subgraph, in the
        same order as `sub_data`'s own subgraphs (i.e. index i here is
        subgraph i's own `sub_data["bus"].batch` id) -- for per-strategy
        error reporting (see `custom_losses.SubproblemConsistencyLoss`).
        None when `sampling_strategies` itself is None (nothing to break
        down by).
    """
    bus_store = data["bus"]
    device = bus_store.x.device
    num_buses = bus_store.num_nodes
    ptr = bus_store.ptr if "ptr" in bus_store else torch.tensor([0, num_buses], device=device)

    edge_index = data["bus", "branch", "bus"].edge_index
    with _timed(stats, "adjacency"):
        sample_sizes = ptr[1:] - ptr[:-1]
        num_nodes_per_case = int(sample_sizes[0].item()) if sample_sizes.numel() > 0 else num_buses
        uniform_case_size = bool(
            sample_sizes.numel() == 0 or bool((sample_sizes == num_nodes_per_case).all().item())
        )

        if adjacency_cache is not None and uniform_case_size:
            changed = update_base_topology_cache(edge_index, ptr, num_nodes_per_case, adjacency_cache)
            if changed:
                _bump(stats, "adjacency_base_updated#count")
            per_sample_missing = missing_edges_per_sample(
                edge_index, ptr, num_nodes_per_case, adjacency_cache
            )
            base_adjacency = adjacency_cache["base_adjacency"]
            use_base_mode = True
            _bump(stats, "adjacency_base_mode#count")
        else:
            # Irregular batch (different bus counts per sample -- shouldn't
            # happen for a single fixed case, but stay safe) or no cache
            # requested at all: one flat rebuild for this whole batch, no
            # caching across calls.
            adjacency = build_adjacency(edge_index, num_buses)
            use_base_mode = False
            _bump(stats, "adjacency_fallback_rebuild#count")

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
        # Only needed for strategy="snowball" -- cached alongside the base
        # adjacency when available; computed once here (not per-draw) in
        # the fallback path, where there's no persistent cache to hang it
        # off of.
        degrees = None
        if sampling_strategies is not None:
            degrees = (
                adjacency_cache.get("degrees") if use_base_mode
                else [len(n) for n in adjacency]
            )
        bus_subset_list = []
        strategy_per_subgraph = [] if sampling_strategies is not None else None
        for k in range(len(ptr_list) - 1):
            lo, hi = ptr_list[k], ptr_list[k + 1]
            if hi <= lo:
                continue
            if use_base_mode:
                # This sample's own missing-edge patch against the shared
                # base topology (see missing_edges_per_sample) -- symmetric
                # since undirected, and typically 0-2 entries (N-1/N-2).
                excluded: dict = {}
                for i, j in per_sample_missing[k]:
                    excluded.setdefault(i, set()).add(j)
                    excluded.setdefault(j, set()).add(i)
                sample_kwargs = dict(
                    adjacency=base_adjacency, bus_offset=lo, excluded_neighbors=excluded
                )
            else:
                sample_kwargs = dict(adjacency=adjacency)
            # subgraphs_per_sample independent draws per original sample --
            # own seed bus, own sampling strategy (if sampling_strategies
            # is given -- see this function's own docstring for why mixing
            # per-draw, rather than pinning one strategy per batch, doesn't
            # cost anything extra here), and own growth each time, so
            # within one sample the draws aren't just copies of each other.
            for _ in range(subgraphs_per_sample):
                seed_bus = torch.randint(lo, hi, (1,), generator=generator).item()
                if sampling_strategies is not None:
                    strategy_name = pick_sampling_strategy(sampling_strategies, generator=generator)
                    strategy_fn = SAMPLING_STRATEGIES[strategy_name]
                    strategy_kwargs = {
                        key: value
                        for key, value in sampling_strategies[strategy_name].items()
                        if key != "proportion"
                    }
                    if strategy_name == "snowball":
                        strategy_kwargs.setdefault("degrees", degrees)
                    strategy_per_subgraph.append(strategy_name)
                else:
                    strategy_fn = sample_bus_subset
                    strategy_kwargs = {}
                bus_subset_list.append(
                    strategy_fn(
                        seed_bus=seed_bus,
                        min_size=min_size,
                        max_size=max_size,
                        generator=generator,
                        **sample_kwargs,
                        **strategy_kwargs,
                    )
                )

    # No separate "batch_from_data_list" stage anymore -- _build_subproblem_
    # vectorized now assembles the final batched sub_data directly, so that
    # cost (previously its own bucket) is now folded into this one.
    with _timed(stats, "build_subproblem"):
        (
            sub_data_batch,
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

    return (
        sub_data_batch,
        bus_map_all,
        voltage_mask_all,
        edge_map_all,
        va_offset_all,
        strategy_per_subgraph,
    )
