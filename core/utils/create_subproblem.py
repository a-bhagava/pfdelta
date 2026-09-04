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
    SubgraphFinetuneTrainer for how these get reset/printed per epoch.

    Syncs (`torch.cuda.synchronize()`) both before starting AND before
    stopping the clock when CUDA is available -- CUDA ops are dispatched
    asynchronously, so a plain `time.perf_counter()` around a block of pure
    GPU work only measures how long the CPU took to QUEUE it, not how long
    the GPU actually took to run it; whichever LATER block happens to be
    the first to force a sync (e.g. a `.item()` call) then silently
    "absorbs" every earlier block's unfinished GPU work into its own
    measured time. E.g. `first_forward_pass` (pure GPU dispatch, no syncing
    op inside it) reporting a suspiciously small number while the very next
    block reports a suspiciously large one is this exact symptom -- fixed
    by draining the queue at both ends of each timed block, so each one's
    number reflects only its own work. Real (non-profiled) training is
    unaffected either way (`stats is None` short-circuits before either
    sync); this always runs (not opt-in) whenever profiling itself is on,
    trading some profiling-run wall-clock time for numbers that are
    actually trustworthy every time, not just on request."""
    if stats is None:
        yield
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
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
    cached_num_nodes_per_case = base_cache.get("num_nodes_per_case")
    if base_keys is None or cached_num_nodes_per_case != num_nodes_per_case:
        # Either a genuinely fresh cache, or `base_cache` belongs to a
        # DIFFERENT case (e.g. the same SubproblemConsistencyLoss instance,
        # and so the same adjacency_cache dict, gets reused across every
        # val dataset in one epoch's validation loop -- case14, case30,
        # ... case500 -- each with its own bus count). `base_keys`' encoding
        # (i*num_nodes_per_case+j, see _canonical_edge_keys) is only valid
        # for the num_nodes_per_case it was built under, so a size change
        # makes the OLD keys meaningless, not just stale -- merging them in
        # via torch.isin below would silently decode into bus indices out
        # of range for the new case (this is what used to crash
        # _build_subproblem_vectorized's scatter_add_ with a CUDA
        # out-of-bounds assert once consistency_val_indices started
        # covering more than one case). Must discard and rebuild, not merge.
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


#############################################################################
#   BATCHED (vectorized) sampling -- draws every subgraph in a call to
#   build_subproblem_batch AT ONCE instead of looping sample_bus_subset*
#   over samples/draws in Python. Only usable in "base mode" (see
#   build_subproblem_batch) -- every draw sharing one case-local adjacency
#   is exactly what lets D = num_samples * subgraphs_per_sample independent
#   traversals run as D-wide vectorized rounds instead of D Python-level
#   calls. The per-draw functions above (sample_bus_subset and friends)
#   stay as the correctness reference and the fallback (irregular-batch /
#   no-cache) path's implementation -- this section is purely a
#   performance-motivated alternate implementation of the SAME sampling
#   semantics, not a replacement for them.
#
#   Important: this is NOT a bit-for-bit reproduction of the sequential
#   per-draw functions' exact random draws for a given seed -- vectorizing
#   necessarily changes how the shared RNG stream gets consumed (per-round-
#   across-all-draws instead of per-node-within-one-draw-at-a-time), so
#   re-running an old seed will sample DIFFERENT (but equally valid)
#   subgraphs than before. What's preserved is the DISTRIBUTIONAL contract
#   each function's own docstring describes: connectivity, size bounds,
#   exclusion respect, and each strategy's own qualitative shape (BFS's
#   uniform shell growth, snowball's degree-inverse bias, forest fire's
#   ability to die out early, random walk's local/revisiting exploration).
#
#   Entirely CPU tensor ops throughout (never touches whatever device
#   `data`/`output_dict` live on) -- same design intent as the sequential
#   functions (adjacency is a plain Python list of lists, `generator` is a
#   CPU torch.Generator), avoiding GPU dispatch overhead for what's
#   fundamentally small, cheap-per-element work. build_subproblem_batch's
#   caller (_build_subproblem_vectorized) already moves the resulting
#   `bus_subset_list` to the right device itself, same as it always did
#   for the sequential functions' own CPU-tensor outputs.
#############################################################################

def _build_padded_adjacency(adjacency: list) -> tuple:
    """
    Dense `[N, max_degree]` neighbor table (+ a same-shape validity mask)
    from a plain adjacency list -- lets a batched traversal gather many
    nodes' neighbor rows via one tensor index instead of a per-node Python
    list lookup. Short rows are right-padded with -1 (marked invalid in
    `valid_mask`, never a real node id).

    Cheap to rebuild -- only actually called when the base topology itself
    changes (see `update_base_topology_cache`'s own convergence guarantee),
    cached alongside `base_adjacency`/`degrees` in the same `case_cache`
    dict with the same "rebuild only when the underlying adjacency object
    changed" lifecycle (see `draw_subgraphs_batched`).

    Returns
    -------
    neighbor_table, valid_mask : torch.Tensor
        Both `[len(adjacency), max_degree]`, CPU, `neighbor_table` int64
        (-1 padded), `valid_mask` bool.
    """
    num_nodes = len(adjacency)
    max_deg = max((len(n) for n in adjacency), default=0)
    max_deg = max(max_deg, 1)  # keep at least width 1 even for a totally isolated case
    neighbor_table = torch.full((num_nodes, max_deg), -1, dtype=torch.long)
    for i, neighbors in enumerate(adjacency):
        if neighbors:
            neighbor_table[i, : len(neighbors)] = torch.tensor(neighbors, dtype=torch.long)
    valid_mask = neighbor_table >= 0
    return neighbor_table, valid_mask


def _build_exclusion_keys(per_sample_missing: list, num_nodes_per_case: int) -> Optional[torch.Tensor]:
    """
    Flat, encoded set of directed `(sample_id, i, j)` edge keys to skip
    during batched traversal, built from `missing_edges_per_sample`'s
    output and symmetrized (both directions) -- the batched equivalent of
    the per-sample `excluded_neighbors` dict the sequential strategies
    build (see `build_subproblem_batch`). Encoded as a single int64
    `sample_id * N*N + i*N + j` so membership is one `torch.isin` call
    instead of a Python dict lookup per candidate.

    Returns None (not an empty tensor) when there's nothing to exclude
    anywhere in this batch -- the common case once the base topology has
    converged (most samples carry zero missing edges each batch) -- so
    callers can skip the `isin` check entirely rather than pay for a
    guaranteed-empty-result comparison every round.
    """
    triples = []
    for k, missing in enumerate(per_sample_missing):
        base = k * num_nodes_per_case * num_nodes_per_case
        for i, j in missing:
            triples.append(base + i * num_nodes_per_case + j)
            triples.append(base + j * num_nodes_per_case + i)
    if not triples:
        return None
    return torch.tensor(triples, dtype=torch.long)


def _kept_matrix_to_list(kept: torch.Tensor, bus_offset_per_draw: torch.Tensor) -> list:
    """
    `[D, N]` bool -> list of `D` sorted 1-D LongTensors of GLOBAL bus
    indices -- vectorized, no per-draw Python loop. `nonzero()` already
    returns `(draw, local)` pairs in row-major order (ascending draw, then
    ascending local index within a draw), and adding each draw's own
    constant `bus_offset` preserves that ascending order -- so splitting by
    each draw's own count directly reproduces the same "sorted, per-draw"
    convention `sample_bus_subset` and friends return.
    """
    draw_idx, local_idx = kept.nonzero(as_tuple=True)
    global_idx = local_idx + bus_offset_per_draw[draw_idx]
    counts = kept.sum(dim=1).tolist()
    return list(torch.split(global_idx, counts))


def _batched_frontier_growth(
    neighbor_table: torch.Tensor,
    valid_mask: torch.Tensor,
    seed_local: torch.Tensor,
    sample_of_draw: torch.Tensor,
    target_size: torch.Tensor,
    num_nodes_per_case: int,
    generator: Optional[torch.Generator] = None,
    degree_weight: Optional[torch.Tensor] = None,
    forest_fire_p: Optional[float] = None,
    exclusion_keys: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Batched equivalent of `sample_bus_subset` (plain BFS), `sample_bus_
    subset_snowball` (pass `degree_weight`), and `sample_bus_subset_
    forest_fire` (pass `forest_fire_p`) -- all three share the same
    "shell-by-shell" growth structure (grow the whole frontier's unvisited
    neighbors each round, stop once `target_size` is hit or the frontier
    dies out), differing only in how a round's candidates get ordered/
    filtered before the shared budget cutoff. Grows all `D = seed_local.
    numel()` independent subsets AT ONCE, one round at a time, instead of
    looping over draws in Python -- each round is fully vectorized: gather
    every active draw's frontier nodes' neighbor rows together, assign each
    candidate edge a random priority (uniform for BFS/forest-fire, or the
    same Efraimidis-Spirakis degree-inverse key `_weighted_shuffle` uses
    for snowball), resolve same-round duplicate proposals (two frontier
    nodes reaching the same unvisited candidate) by keeping the higher-
    priority one, then keep only the top `remaining_budget` proposals per
    draw.

    Parameters
    ----------
    neighbor_table, valid_mask : torch.Tensor
        `[num_nodes_per_case, max_degree]` -- see `_build_padded_adjacency`.
    seed_local, sample_of_draw, target_size : torch.Tensor
        `[D]` each -- this draw's own case-local seed bus, which of the
        batch's original samples it belongs to (for `exclusion_keys`
        lookups), and its own target subset size.
    degree_weight : torch.Tensor, optional
        `[num_nodes_per_case]`, already `1/degree**degree_weight_power` --
        gives snowball's biased candidate ordering. None (default) gives
        plain BFS/forest-fire's uniform ordering.
    forest_fire_p : float, optional
        Forward-burning probability. None (default) skips the per-source
        burn-count cap entirely (BFS/snowball: every reachable candidate
        competes for the shared budget). Given, each SOURCE node's own
        candidates are first capped to a `Geometric(1-p)`-distributed count
        (closed-form equivalent of `sample_bus_subset_forest_fire`'s
        `while burn_count < len(candidates) and rng.random() < p` loop)
        before the shared cross-source dedup/budget steps run.
    exclusion_keys : torch.Tensor, optional
        See `_build_exclusion_keys`. None if nothing to exclude anywhere in
        this batch.

    Returns
    -------
    torch.Tensor
        `[D, num_nodes_per_case]` bool -- `kept[d, n]` iff local node `n`
        is in draw `d`'s sampled subset.
    """
    device = neighbor_table.device
    D = seed_local.numel()
    N = num_nodes_per_case
    max_deg = neighbor_table.shape[1]
    idx = torch.arange(D, device=device)

    kept = torch.zeros((D, N), dtype=torch.bool, device=device)
    kept[idx, seed_local] = True
    kept_count = torch.ones(D, dtype=torch.long, device=device)
    frontier = torch.zeros((D, N), dtype=torch.bool, device=device)
    frontier[idx, seed_local] = True

    # Bounded by N -- no shell-based traversal can need more than N-1
    # rounds to reach every node of an N-node component -- a safety cap,
    # not something normally reached (draws stop naturally once every one
    # has either hit target_size or run out of frontier).
    for _ in range(N):
        active = (kept_count < target_size) & frontier.any(dim=1)
        if not bool(active.any()):
            break
        draw_idx, src_local = frontier.nonzero(as_tuple=True)
        row_active = active[draw_idx]
        draw_idx, src_local = draw_idx[row_active], src_local[row_active]
        if draw_idx.numel() == 0:
            break

        cand_rows = neighbor_table[src_local]  # [M, max_deg]
        cand_valid = valid_mask[src_local].clone()

        # Random priority per (source, candidate-slot): plain U(0,1) for
        # BFS/forest-fire, or the same "u ** (1/weight)" A-Res key
        # `_weighted_shuffle` uses for snowball's degree-inverse bias.
        u = torch.rand((cand_rows.shape[0], max_deg), generator=generator).clamp(1e-6, 1 - 1e-6)
        if degree_weight is not None:
            # Cast to u's own dtype (float32) -- degree_weight is commonly
            # float64 (built from a plain Python list of degrees), and
            # float32 ** float64 promotes to float64, which would then
            # make `priority` a different dtype than the plain-uniform
            # branch below (and than `best` further down), breaking
            # scatter_reduce_'s dtype match requirement.
            w = degree_weight[cand_rows.clamp(min=0)].clamp(min=1e-6).to(u.dtype)
            priority = torch.where(cand_valid, u.pow(1.0 / w), torch.zeros_like(u))
        else:
            priority = torch.where(cand_valid, u, torch.zeros_like(u))

        if forest_fire_p is not None:
            row_deg = cand_valid.sum(dim=1)
            p = float(forest_fire_p)
            if p <= 0.0:
                burn_count = torch.zeros_like(row_deg)
            elif p >= 1.0:
                burn_count = row_deg.clone()
            else:
                geo_u = torch.rand(cand_rows.shape[0], generator=generator).clamp(1e-6, 1 - 1e-6)
                burn_count = torch.floor(torch.log(geo_u) / float(torch.log(torch.tensor(p)))).long()
                burn_count = torch.minimum(burn_count.clamp(min=0), row_deg)
            # 0-indexed priority rank within each row (double-argsort
            # trick) -- keep only this row's own top `burn_count`.
            rank = torch.argsort(torch.argsort(-priority, dim=1), dim=1)
            cand_valid = cand_valid & (rank < burn_count.unsqueeze(1))
            priority = torch.where(cand_valid, priority, torch.zeros_like(priority))

        already_kept = kept[draw_idx.unsqueeze(1).expand(-1, max_deg), cand_rows.clamp(min=0)]
        alive = cand_valid & ~already_kept

        prop_draw = draw_idx.repeat_interleave(max_deg)
        prop_cand = cand_rows.reshape(-1)
        prop_priority = priority.reshape(-1)
        prop_alive = alive.reshape(-1)

        if exclusion_keys is not None:
            prop_sample = sample_of_draw[prop_draw]
            prop_src = src_local.repeat_interleave(max_deg)
            edge_key = prop_sample * (N * N) + prop_src * N + prop_cand.clamp(min=0)
            prop_alive = prop_alive & ~torch.isin(edge_key, exclusion_keys)

        prop_draw = prop_draw[prop_alive]
        prop_cand = prop_cand[prop_alive]
        prop_priority = prop_priority[prop_alive]
        if prop_draw.numel() == 0:
            frontier = torch.zeros((D, N), dtype=torch.bool, device=device)
            continue

        # Dedup same-round duplicate proposals for the same (draw,
        # candidate) pair (e.g. two frontier nodes both reaching the same
        # unvisited neighbor this round) -- keep only the higher-priority
        # one, via a grouped scatter-max (same style as
        # _build_subproblem_vectorized's segmented slack-promotion reduce).
        dedup_key = prop_draw * N + prop_cand
        best = torch.full((D * N,), -1.0, device=device)
        best.scatter_reduce_(0, dedup_key, prop_priority, reduce="amax", include_self=True)
        is_best = prop_priority >= best[dedup_key]
        prop_draw, prop_cand, prop_priority = prop_draw[is_best], prop_cand[is_best], prop_priority[is_best]
        dedup_key = dedup_key[is_best]
        # Exact-tie safety net (two surviving proposals landing on the
        # identical (draw, candidate) key AND identical priority) --
        # essentially never happens with continuous random priorities, but
        # drop duplicates defensively (keep the first) rather than risk
        # double-counting one candidate against the size budget below.
        dedup_key_sorted, order = torch.sort(dedup_key)
        first_of_group = torch.ones_like(dedup_key_sorted, dtype=torch.bool)
        first_of_group[1:] = dedup_key_sorted[1:] != dedup_key_sorted[:-1]
        order = order[first_of_group]
        prop_draw, prop_cand, prop_priority = prop_draw[order], prop_cand[order], prop_priority[order]

        # Global per-draw remaining-budget cutoff. Fast path first: with
        # small target sizes (this augmentation typically cuts subgraphs
        # of a handful of buses), most rounds -- especially early ones --
        # propose FEWER deduped candidates per draw than that draw's own
        # remaining budget, so no truncation is actually needed at all;
        # skip the lexsort/grouping machinery below entirely in that case
        # (just accept every surviving proposal) rather than pay its
        # overhead on every round regardless of whether it does anything.
        remaining = (target_size - kept_count).clamp(min=0)
        proposal_count = torch.zeros(D, dtype=torch.long, device=device)
        proposal_count.scatter_add_(0, prop_draw, torch.ones_like(prop_draw))
        if bool((proposal_count <= remaining).all()):
            accepted_draw = prop_draw
            accepted_cand = prop_cand
        else:
            # Single lexicographic sort key (draw ascending, priority
            # descending) via one combined float64 (safe since priority in
            # (0,1) never spills into the next draw's integer slot), then
            # each draw's own run is already priority-sorted -- "rank
            # within group" (position minus the group's own start offset)
            # directly gives who's inside the top `remaining_budget[draw]`.
            combo = prop_draw.to(torch.float64) + (1.0 - prop_priority.to(torch.float64))
            order = torch.argsort(combo)
            d_sorted = prop_draw[order]
            _, counts = torch.unique_consecutive(d_sorted, return_counts=True)
            starts = torch.cumsum(counts, 0) - counts
            group_start_per_pos = torch.repeat_interleave(starts, counts)
            rank_in_group = torch.arange(d_sorted.numel(), device=device) - group_start_per_pos
            accept = rank_in_group < remaining[d_sorted]
            accepted_draw = d_sorted[accept]
            accepted_cand = prop_cand[order][accept]

        next_frontier = torch.zeros((D, N), dtype=torch.bool, device=device)
        if accepted_draw.numel() > 0:
            next_frontier[accepted_draw, accepted_cand] = True
            kept[accepted_draw, accepted_cand] = True
            kept_count.scatter_add_(0, accepted_draw, torch.ones_like(accepted_draw))
        frontier = next_frontier

    return kept


def _batched_random_walk(
    neighbor_table: torch.Tensor,
    valid_mask: torch.Tensor,
    seed_local: torch.Tensor,
    sample_of_draw: torch.Tensor,
    target_size: torch.Tensor,
    num_nodes_per_case: int,
    generator: Optional[torch.Generator] = None,
    restart_prob: float = 0.15,
    max_steps_factor: int = 50,
    exclusion_keys: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Batched equivalent of `sample_bus_subset_random_walk` -- runs all
    `D = seed_local.numel()` independent random-with-restart walks in
    lockstep, one step at a time. A single walk's own trace is inherently
    sequential (where it goes next depends on where it just was), but the D
    walks are completely independent of each other, so each step is one
    vectorized round across all of them instead of D separate Python loops
    of up to `max_steps_factor * target_size` iterations each.

    Same contract as the sequential version: every kept node reached by a
    real edge traversal from `seed_local` (so always connected), revisits
    are allowed and don't grow the subset further, growth stops once
    `target_size` is hit or this draw's own step budget runs out (a
    smaller-than-requested result in that case, not a failure).

    Parameters mirror `_batched_frontier_growth`'s shared ones, plus
    `restart_prob`/`max_steps_factor` -- see `sample_bus_subset_random_
    walk`'s own docstring for what they control.
    """
    device = neighbor_table.device
    D = seed_local.numel()
    N = num_nodes_per_case
    max_deg = neighbor_table.shape[1]
    idx = torch.arange(D, device=device)

    kept = torch.zeros((D, N), dtype=torch.bool, device=device)
    kept[idx, seed_local] = True
    kept_count = torch.ones(D, dtype=torch.long, device=device)
    current = seed_local.clone()
    steps_left = (target_size * max_steps_factor).clamp(min=100)

    max_rounds = int(steps_left.max().item()) if steps_left.numel() > 0 else 0
    for _ in range(max_rounds):
        active = (kept_count < target_size) & (steps_left > 0)
        if not bool(active.any()):
            break
        steps_left = steps_left - active.long()

        restart = (torch.rand(D, generator=generator) < restart_prob) & active

        cand_rows = neighbor_table[current]  # [D, max_deg]
        cand_valid = valid_mask[current].clone()
        if exclusion_keys is not None:
            edge_key = (
                sample_of_draw.unsqueeze(1) * (N * N)
                + current.unsqueeze(1) * N
                + cand_rows.clamp(min=0)
            )
            cand_valid = cand_valid & ~torch.isin(edge_key, exclusion_keys)
        u = torch.where(
            cand_valid,
            torch.rand((D, max_deg), generator=generator),
            torch.full((D, max_deg), -1.0),
        )
        has_candidate = cand_valid.any(dim=1)
        choice_col = u.argmax(dim=1)
        next_local = cand_rows[idx, choice_col]

        move = active & ~restart & has_candidate
        # Dead end (isolated/all-excluded) or an explicit restart draw --
        # both reseed to seed_bus, same as the sequential version.
        reseed = active & (restart | (~has_candidate & ~restart))
        new_current = torch.where(move, next_local, current)
        new_current = torch.where(reseed, seed_local, new_current)
        current = torch.where(active, new_current, current)

        active_idx = idx[active]
        active_current = current[active]
        newly = ~kept[active_idx, active_current]
        kept[active_idx, active_current] = True
        kept_count[active_idx] += newly.long()

    return kept


def draw_subgraphs_batched(
    base_adjacency: list,
    case_cache: dict,
    ptr_list: list,
    num_nodes_per_case: int,
    min_size,
    max_size,
    subgraphs_per_sample: int,
    generator: Optional[torch.Generator],
    sampling_strategies: Optional[dict],
    per_sample_missing: list,
) -> tuple:
    """
    Batched replacement for `build_subproblem_batch`'s own per-sample,
    per-draw Python loop over `SAMPLING_STRATEGIES` -- draws ALL
    `D = (len(ptr_list) - 1) * subgraphs_per_sample` independent subgraphs
    at once instead of looping. Only usable in "base mode" (every sample
    sharing one case-local `base_adjacency`) -- see `build_subproblem_
    batch`'s own adjacency-mode selection.

    Draws mixing strategies (via `pick_sampling_strategy`'s weighted
    choice) are handled by grouping draws by their assigned strategy first
    (one vectorized call per strategy actually present this batch, over
    just that strategy's own draws), then reassembling results back into
    original sample-major, draw-order position -- so the returned lists
    line up with `build_subproblem_batch`'s existing per-draw Python-loop
    convention exactly, and `_build_subproblem_vectorized` downstream needs
    no changes at all.

    See `_batched_frontier_growth`/`_batched_random_walk` for what
    "batched" means here and this module's own note (just above
    `_build_padded_adjacency`) on the distributional- (not bit-)
    equivalence contract with `SAMPLING_STRATEGIES`'s sequential functions.

    Returns
    -------
    bus_subset_list : list[torch.Tensor]
        Length `D`, sample-major then draw-order -- each a sorted
        LongTensor of GLOBAL bus indices, same convention as
        `build_subproblem_batch`'s own per-draw loop.
    strategy_per_subgraph : list[str] or None
        Same convention as `build_subproblem_batch`'s own return value --
        None when `sampling_strategies` is None.
    """
    K = len(ptr_list) - 1
    D = K * subgraphs_per_sample
    if D == 0:
        return [], ([] if sampling_strategies is not None else None)

    N = num_nodes_per_case
    lo_per_sample = torch.tensor(ptr_list[:-1], dtype=torch.long)
    sample_of_draw = torch.arange(K, dtype=torch.long).repeat_interleave(subgraphs_per_sample)
    bus_offset_per_draw = lo_per_sample[sample_of_draw]

    lo_size, hi_size = int(min_size), int(max_size)
    lo_size, hi_size = min(lo_size, hi_size), max(lo_size, hi_size)
    target_size = torch.randint(lo_size, hi_size + 1, (D,), generator=generator)
    seed_local = torch.randint(0, N, (D,), generator=generator)

    # Cached alongside base_adjacency/degrees in the SAME case_cache dict,
    # rebuilt only when base_adjacency itself was rebuilt this call (the
    # `is` check is deliberate -- update_base_topology_cache only replaces
    # base_cache["base_adjacency"] with a NEW list object when the topology
    # actually changed, never mutates the old one in place).
    if case_cache.get("_padded_adjacency_for") is not base_adjacency:
        case_cache["_padded_adjacency"] = _build_padded_adjacency(base_adjacency)
        case_cache["_padded_adjacency_for"] = base_adjacency
    neighbor_table, valid_mask = case_cache["_padded_adjacency"]

    exclusion_keys = _build_exclusion_keys(per_sample_missing, N)

    if sampling_strategies is None:
        kept = _batched_frontier_growth(
            neighbor_table, valid_mask, seed_local, sample_of_draw, target_size,
            N, generator=generator, exclusion_keys=exclusion_keys,
        )
        return _kept_matrix_to_list(kept, bus_offset_per_draw), None

    names = list(sampling_strategies.keys())
    weights = torch.tensor(
        [float(sampling_strategies[n].get("proportion", 1.0)) for n in names], dtype=torch.float64
    )
    total = weights.sum()
    assert total > 0, "sampling_strategies proportions must sum to something positive"
    cum = torch.cumsum(weights, 0) / total
    r = torch.rand(D, generator=generator).to(torch.float64)
    strategy_idx = torch.searchsorted(cum, r, right=False).clamp(max=len(names) - 1)

    bus_subset_list = [None] * D
    strategy_per_subgraph = [None] * D
    degrees_tensor = None

    for s_idx, name in enumerate(names):
        group_positions = (strategy_idx == s_idx).nonzero(as_tuple=True)[0]
        if group_positions.numel() == 0:
            continue
        cfg = {k: v for k, v in sampling_strategies[name].items() if k != "proportion"}
        g_seed = seed_local[group_positions]
        g_sample = sample_of_draw[group_positions]
        g_target = target_size[group_positions]
        g_offset = bus_offset_per_draw[group_positions]

        if name == "random_walk":
            kept = _batched_random_walk(
                neighbor_table, valid_mask, g_seed, g_sample, g_target, N, generator=generator,
                restart_prob=float(cfg.get("restart_prob", 0.15)),
                max_steps_factor=int(cfg.get("max_steps_factor", 50)),
                exclusion_keys=exclusion_keys,
            )
        else:
            degree_weight = None
            forest_fire_p = None
            if name == "snowball":
                power = float(cfg.get("degree_weight_power", 1.0))
                if power != 0:
                    if degrees_tensor is None:
                        degrees_tensor = torch.tensor(case_cache["degrees"], dtype=torch.float64)
                    degree_weight = 1.0 / degrees_tensor.clamp(min=1).pow(power)
            elif name == "forest_fire":
                forest_fire_p = float(cfg.get("p", 0.3))
            kept = _batched_frontier_growth(
                neighbor_table, valid_mask, g_seed, g_sample, g_target, N, generator=generator,
                degree_weight=degree_weight, forest_fire_p=forest_fire_p,
                exclusion_keys=exclusion_keys,
            )

        group_subsets = _kept_matrix_to_list(kept, g_offset)
        for pos_in_group, draw_pos in enumerate(group_positions.tolist()):
            bus_subset_list[draw_pos] = group_subsets[pos_in_group]
            strategy_per_subgraph[draw_pos] = name

    return bus_subset_list, strategy_per_subgraph


#############################################################################
#   POOLED sampling -- an opt-in alternative to draw_subgraphs_batched's own
#   "grow every draw fresh, every call" behavior: a large, fixed-size pool
#   of subgraph SHAPES (case-local bus-index sets), built ONCE OFFLINE by
#   scripts/build_subgraph_pool.py and saved to disk, then just loaded
#   (once per process) and indexed into (with replacement) at train time --
#   no growth, nothing per-iteration but a lookup (plus a cheap contamination
#   check under contingency -- see below). Deliberately NOT built lazily
#   in-process (unlike base_adjacency/degrees/the padded-adjacency cache
#   above) -- every SLURM job in a sweep, and every requeue of the same
#   job, is its own fresh process, so an in-process-only cache would
#   rebuild the pool from scratch every single time; a file built once and
#   reused indefinitely across every run avoids that entirely.
#
#   ONE file per (case, min_size, max_size) -- a SEPARATE pool of pool_size
#   shapes for EACH sampling strategy inside it (keyed by name), built with
#   proportion 1.0 each (a strategy's own MIX proportion is meaningless at
#   build time -- see build_subgraph_pool_data). Which strategy a given
#   draw actually pulls from, and in what proportion, is entirely a DRAW-
#   TIME decision (draw_subgraphs_pooled's own `sampling_strategies`
#   argument, same schema/weighted-choice as draw_subgraphs_batched) -- so
#   ONE file covers every strategy mix you'd ever sweep (pure_bfs,
#   even_mix, no_bfs, ... -- see canos_task_3_1_joint_train.yaml's own
#   9-way sweep), not one file per mix. A strategy's OTHER kwargs
#   (restart_prob/p/degree_weight_power) ARE baked in at build time, since
#   they change what gets grown -- a training run's own value for these is
#   ignored when pooling (only `proportion` is read from sampling_
#   strategies at draw time; see draw_subgraphs_pooled's docstring).
#
#   Safe to use even WITH N-1/N-2 contingency (PFDeltaDataset's
#   `perturbation` "n-1"/"n-2"), as long as the pool was itself built
#   against the case's full (uncontingent) topology -- see
#   scripts/build_subgraph_pool.py, which discovers that full topology the
#   same union-across-samples way update_base_topology_cache does,
#   regardless of what perturbation the BUILD dataset itself used.
#   draw_subgraphs_pooled checks each draw's chosen shape against its own
#   sample's actual missing branches (from missing_edges_per_sample, the
#   SAME per-sample contingency info build_subgraph_batch's live-growth
#   path already computes) and redraws (same strategy, a different pool
#   entry) whenever a missing branch's both endpoints fall inside the
#   chosen shape -- conservative (may redraw a shape that would actually
#   still be connected without that edge) but never lets a genuinely-
#   affected one through. With N-1/N-2 typically removing only 1-2 branches
#   out of hundreds, and pooled subgraphs small relative to the whole case,
#   this rejection rate is negligible in practice -- see the module's own
#   test for measured numbers.
#
#   Trades some of the augmentation's own diversity for speed -- a fixed
#   pool, however large, is still finitely many distinct shapes reused
#   across every sample/epoch for the rest of the run, unlike live growth's
#   fresh draw every single call. Pick pool_size large relative to how many
#   draws you'll ever make (num_samples * subgraphs_per_sample * epochs, in
#   THIS run and every future one that reuses the same file) if that
#   tradeoff matters to you.
#############################################################################

def build_subgraph_pool_data(
    base_adjacency: list,
    num_nodes_per_case: int,
    min_size,
    max_size,
    strategies: dict,
    generator: Optional[torch.Generator],
    pool_size: int,
) -> dict:
    """
    Draws `pool_size` subgraphs for EACH strategy in `strategies`
    (`{name: {**kwargs}}` -- no "proportion", every strategy gets its own
    full `pool_size` entries; mix proportions are a draw-time-only concept,
    see `draw_subgraphs_pooled`), reusing `draw_subgraphs_batched` itself
    (a single pseudo-"sample" spanning the whole case-local index space
    `[0, num_nodes_per_case)`, `subgraphs_per_sample=pool_size`, one call
    per strategy at `proportion=1.0`) -- i.e. exactly the same growth logic
    already implemented and tested there, just asked for `pool_size` draws
    of ONE strategy at a time instead of a per-draw mix. Returns a plain
    dict ready for `torch.save` (see `scripts/build_subgraph_pool.py`, the
    actual offline entry point this is meant to be called from) or for
    direct in-process use in a test.

    Pure function -- takes no case_cache and mutates nothing; building a
    pool has nothing to do with training-time caching lifecycles.
    """
    assert pool_size > 0, "pool_size must be positive"
    assert strategies, "strategies must name at least one sampling strategy to build a pool for"
    width = int(max_size)
    strategy_pools = {}
    for name, kwargs in strategies.items():
        assert name in SAMPLING_STRATEGIES, f"Unknown sampling strategy {name!r}"
        single_strategy_cfg = {name: {"proportion": 1.0, **kwargs}}
        # Pre-populate "degrees" (normally set by update_base_topology_cache
        # as part of the adjacency-cache lifecycle build_subproblem_batch
        # drives) since draw_subgraphs_batched's own snowball path reads it
        # straight off case_cache -- there's no adjacency-cache machinery
        # running here, just this one throwaway pool-building call.
        scratch_cache: dict = {"degrees": [len(n) for n in base_adjacency]}
        pool_bus_subset_list, _ = draw_subgraphs_batched(
            base_adjacency, scratch_cache, [0, num_nodes_per_case], num_nodes_per_case,
            min_size, max_size, pool_size, generator, single_strategy_cfg, [[]],
        )
        pool_nodes = torch.nn.utils.rnn.pad_sequence(
            pool_bus_subset_list, batch_first=True, padding_value=-1
        )
        if pool_nodes.shape[1] < width:
            pad = torch.full((pool_nodes.shape[0], width - pool_nodes.shape[1]), -1, dtype=torch.long)
            pool_nodes = torch.cat([pool_nodes, pad], dim=1)
        pool_counts = torch.tensor([s.numel() for s in pool_bus_subset_list], dtype=torch.long)
        strategy_pools[name] = {
            "pool_nodes": pool_nodes,  # [pool_size, max_size] long, -1 padded, CASE-LOCAL indices
            "pool_counts": pool_counts,  # [pool_size] long
            "kwargs": kwargs,
        }

    return {
        "strategies": strategy_pools,
        "num_nodes_per_case": int(num_nodes_per_case),
        "min_size": int(min_size),
        "max_size": int(max_size),
        "pool_size": int(pool_size),
    }


def _load_subgraph_pool(case_cache: dict, pool_path: str, num_nodes_per_case: int) -> None:
    """
    Loads a pool file built by `build_subgraph_pool_data` (via
    `scripts/build_subgraph_pool.py`) into `case_cache`, once per process
    (a no-op on every call after the first for the same `pool_path` --
    checked by path, not re-read from disk every time).
    """
    if case_cache.get("_pool_path") == pool_path:
        return
    pool = torch.load(pool_path, map_location="cpu")
    assert pool["num_nodes_per_case"] == num_nodes_per_case, (
        f"Pool at {pool_path!r} was built for a {pool['num_nodes_per_case']}-bus case, "
        f"but this run's case has {num_nodes_per_case} buses -- wrong pool file for this case."
    )
    case_cache["_pool_strategies"] = pool["strategies"]  # {name: {pool_nodes, pool_counts, kwargs}}
    case_cache["_pool_path"] = pool_path


def _pool_draws_contaminated(
    chosen_nodes: torch.Tensor,
    sample_of_draw: torch.Tensor,
    per_sample_missing: list,
    num_nodes_per_case: int,
) -> torch.Tensor:
    """
    `chosen_nodes`: `[D, W]` case-local node ids (-1 padded) -- the shape
    each draw currently has selected from the pool. Returns `[D]` bool:
    True iff that draw's own sample (`sample_of_draw[d]`, indexing
    `per_sample_missing`) has a missing (offline THIS batch, per
    `missing_edges_per_sample`) branch whose BOTH endpoints appear in the
    chosen shape -- i.e. a branch this shape might have relied on for
    connectivity when it was grown against the pool's own (fully-connected)
    build-time topology, but that's actually offline for this specific
    sample. Conservative: also flags shapes where the missing branch's
    endpoints are both present but weren't actually load-bearing for
    connectivity (redundant paths exist) -- a false positive just costs an
    unnecessary redraw, never lets a genuinely-disconnected shape through.
    """
    D = chosen_nodes.shape[0]
    max_missing = max((len(m) for m in per_sample_missing), default=0)
    if max_missing == 0:
        return torch.zeros(D, dtype=torch.bool)
    K = len(per_sample_missing)
    missing_i = torch.zeros((K, max_missing), dtype=torch.long)
    missing_j = torch.zeros((K, max_missing), dtype=torch.long)
    missing_valid = torch.zeros((K, max_missing), dtype=torch.bool)
    for k, missing in enumerate(per_sample_missing):
        for m, (i, j) in enumerate(missing):
            missing_i[k, m] = i
            missing_j[k, m] = j
            missing_valid[k, m] = True
    draw_i = missing_i[sample_of_draw]  # [D, max_missing]
    draw_j = missing_j[sample_of_draw]
    draw_valid = missing_valid[sample_of_draw]

    has_i = (chosen_nodes.unsqueeze(2) == draw_i.unsqueeze(1)).any(dim=1)  # [D, max_missing]
    has_j = (chosen_nodes.unsqueeze(2) == draw_j.unsqueeze(1)).any(dim=1)  # [D, max_missing]
    return (has_i & has_j & draw_valid).any(dim=1)


def draw_subgraphs_pooled(
    case_cache: dict,
    ptr_list: list,
    num_nodes_per_case: int,
    subgraphs_per_sample: int,
    generator: Optional[torch.Generator],
    pool_path: str,
    sampling_strategies: Optional[dict] = None,
    per_sample_missing: Optional[list] = None,
    max_redraw_attempts: int = 5,
) -> tuple:
    """
    Pooled drop-in replacement for `draw_subgraphs_batched`'s own return
    convention -- loads (once per process, via `_load_subgraph_pool`) the
    pool file at `pool_path` (built offline by `scripts/build_subgraph_
    pool.py`), then for each of `D = (len(ptr_list) - 1) *
    subgraphs_per_sample` draws: picks a strategy (weighted by `sampling_
    strategies`' own `proportion`s, same convention as `draw_subgraphs_
    batched`/`pick_sampling_strategy` -- None defaults to a uniform mix
    over every strategy actually IN the pool file) and indexes a random
    entry (with replacement) out of that strategy's own sub-pool -- no
    growth, an O(1) lookup per draw. min_size/max_size and every strategy's
    non-`proportion` kwargs are NOT parameters here -- they're baked into
    the pool file; only `proportion` is read from `sampling_strategies`.

    If `per_sample_missing` is given and non-empty anywhere (see
    `missing_edges_per_sample` -- this batch's own N-1/N-2 contingency),
    each chosen shape is checked against its own sample's missing branches
    (`_pool_draws_contaminated`) and redrawn (same strategy, a different
    pool entry) up to `max_redraw_attempts` times if contaminated -- see
    this module's own POOLED sampling section for why this makes a pool
    built from a fully-connected topology still safe to use under
    contingency. Skipped entirely (zero overhead) when `per_sample_missing`
    is None/all-empty, e.g. `perturbation="n"`.
    """
    K = len(ptr_list) - 1
    D = K * subgraphs_per_sample
    if D == 0:
        return [], None

    _load_subgraph_pool(case_cache, pool_path, num_nodes_per_case)
    pool_strategies = case_cache["_pool_strategies"]

    if sampling_strategies is None:
        names = list(pool_strategies.keys())
        weights = torch.ones(len(names), dtype=torch.float64)
    else:
        names = list(sampling_strategies.keys())
        for name in names:
            assert name in pool_strategies, (
                f"sampling_strategies wants {name!r}, but pool file {pool_path!r} only has "
                f"pools for {list(pool_strategies.keys())} -- rebuild the pool with this "
                "strategy included."
            )
        weights = torch.tensor(
            [float(sampling_strategies[n].get("proportion", 1.0)) for n in names], dtype=torch.float64
        )
    total = weights.sum()
    assert total > 0, "sampling_strategies proportions must sum to something positive"
    cum = torch.cumsum(weights, 0) / total

    lo_per_sample = torch.tensor(ptr_list[:-1], dtype=torch.long)
    sample_of_draw = torch.arange(K, dtype=torch.long).repeat_interleave(subgraphs_per_sample)
    bus_offset_per_draw = lo_per_sample[sample_of_draw]

    r = torch.rand(D, generator=generator).to(torch.float64)
    strategy_idx = torch.searchsorted(cum, r, right=False).clamp(max=len(names) - 1)

    width = max(p["pool_nodes"].shape[1] for p in pool_strategies.values())
    chosen_nodes = torch.full((D, width), -1, dtype=torch.long)
    chosen_counts = torch.zeros(D, dtype=torch.long)

    def fill(positions: torch.Tensor) -> None:
        for s_idx, name in enumerate(names):
            group = positions[strategy_idx[positions] == s_idx]
            if group.numel() == 0:
                continue
            pn = pool_strategies[name]["pool_nodes"]
            pc = pool_strategies[name]["pool_counts"]
            pick = torch.randint(0, pn.shape[0], (group.numel(),), generator=generator)
            rows = pn[pick]
            if rows.shape[1] < width:
                pad = torch.full((rows.shape[0], width - rows.shape[1]), -1, dtype=torch.long)
                rows = torch.cat([rows, pad], dim=1)
            chosen_nodes[group] = rows
            chosen_counts[group] = pc[pick]

    check_contingency = bool(per_sample_missing) and any(per_sample_missing)
    pending = torch.arange(D)
    attempts = max_redraw_attempts if check_contingency else 1
    for _ in range(attempts):
        if pending.numel() == 0:
            break
        fill(pending)
        if not check_contingency:
            break
        contaminated = _pool_draws_contaminated(chosen_nodes, sample_of_draw, per_sample_missing, num_nodes_per_case)
        pending = contaminated.nonzero(as_tuple=True)[0]
    # Any still-contaminated after max_redraw_attempts (astronomically rare
    # given how small a fraction of a case a pooled subgraph covers) are
    # accepted as-is rather than looping forever.

    valid = chosen_nodes >= 0
    draw_idx, col_idx = valid.nonzero(as_tuple=True)
    global_idx = chosen_nodes[draw_idx, col_idx] + bus_offset_per_draw[draw_idx]
    counts = chosen_counts.tolist()
    bus_subset_list = list(torch.split(global_idx, counts))

    strategy_per_subgraph = [names[i] for i in strategy_idx.tolist()]
    return bus_subset_list, strategy_per_subgraph


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
    pool_path: Optional[str] = None,
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
        Internally keyed by `num_nodes_per_case` (one sub-dict per distinct
        case size) -- safe to pass the SAME dict across calls that see
        DIFFERENT cases (e.g. one SubproblemConsistencyLoss instance's own
        cache, reused across every val dataset in a validation loop), each
        case gets its own independently-converging sub-cache rather than
        corrupting a shared one (the base topology's canonical edge
        encoding is only valid for the num_nodes_per_case it was built
        under -- see update_base_topology_cache).
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
    pool_path : str, optional
        Path to a pool file built by `scripts/build_subgraph_pool.py` (see
        `draw_subgraphs_pooled`) -- when given, every draw is a random
        lookup into that precomputed pool instead of a fresh traversal.
        min_size/max_size above are ignored entirely (baked into the pool
        file); `sampling_strategies` above is STILL used -- its `proportion`
        fields decide the mix drawn from the pool file's own per-strategy
        sub-pools (any other kwargs, like restart_prob/p/degree_weight_
        power, are ignored here too -- also baked into the file). Requires
        `adjacency_cache` (uses the same per-case `case_cache` machinery,
        and needs "base mode" -- uniform bus count per sample -- same as
        the adjacency cache itself). Safe even under N-1/N-2 contingency as
        long as the pool file was itself built against the case's full
        topology -- see this module's own POOLED sampling section above.
        None (default) uses draw_subgraphs_batched (live growth) as before.

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
            # Keyed by num_nodes_per_case, not used flat -- the SAME
            # adjacency_cache dict (e.g. one SubproblemConsistencyLoss
            # instance's own, long-lived across every val dataset in a
            # validation loop) gets called against DIFFERENT cases in
            # sequence (case14, case30, ... case500), each with its own bus
            # count. update_base_topology_cache's canonical edge encoding
            # is only valid for one fixed num_nodes_per_case, so sharing a
            # single flat cache across cases used to silently corrupt into
            # out-of-range bus indices (see its own docstring/comment).
            # Bus count is a reasonable per-case key since a single case's
            # own contingency samples (N-1/N-2) only ever remove branches,
            # never buses -- see PFDeltaDataset's `perturbation` param.
            case_cache = adjacency_cache.setdefault(num_nodes_per_case, {})
            changed = update_base_topology_cache(edge_index, ptr, num_nodes_per_case, case_cache)
            if changed:
                _bump(stats, "adjacency_base_updated#count")
            per_sample_missing = missing_edges_per_sample(
                edge_index, ptr, num_nodes_per_case, case_cache
            )
            base_adjacency = case_cache["base_adjacency"]
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
        if use_base_mode and pool_path is not None:
            # Pooled: every draw is a lookup into a pool built OFFLINE
            # (scripts/build_subgraph_pool.py), loaded once per process --
            # per_sample_missing (this batch's own N-1/N-2 contingency, if
            # any -- same info the live-growth path below already needs)
            # is passed through so a chosen shape that relies on a branch
            # offline for its own sample gets redrawn -- see draw_
            # subgraphs_pooled's own docstring and this module's POOLED
            # sampling section for why that makes pooling safe even under
            # contingency, not just for perturbation="n".
            bus_subset_list, strategy_per_subgraph = draw_subgraphs_pooled(
                case_cache, ptr_list, num_nodes_per_case, subgraphs_per_sample,
                generator, pool_path, sampling_strategies, per_sample_missing,
            )
        elif use_base_mode:
            # Vectorized: draws every (sample, subgraphs_per_sample) subset
            # in this batch at once instead of looping sample_bus_subset*
            # in Python -- see draw_subgraphs_batched's own docstring for
            # why "base mode" (one shared case-local adjacency) is what
            # makes this possible, and this module's note on its
            # distributional- (not bit-) equivalence to the per-draw
            # functions below.
            bus_subset_list, strategy_per_subgraph = draw_subgraphs_batched(
                base_adjacency, case_cache, ptr_list, num_nodes_per_case,
                min_size, max_size, subgraphs_per_sample, generator,
                sampling_strategies, per_sample_missing,
            )
        else:
            # Irregular batch (different bus counts per sample) or no
            # cache requested at all -- no shared case-local adjacency to
            # batch draws against, so fall back to the original per-draw
            # Python loop (rare path; see the adjacency-mode selection
            # above).
            degrees = [len(n) for n in adjacency] if sampling_strategies is not None else None
            bus_subset_list = []
            strategy_per_subgraph = [] if sampling_strategies is not None else None
            for k in range(len(ptr_list) - 1):
                lo, hi = ptr_list[k], ptr_list[k + 1]
                if hi <= lo:
                    continue
                sample_kwargs = dict(adjacency=adjacency)
                # subgraphs_per_sample independent draws per original
                # sample -- own seed bus, own sampling strategy (if
                # sampling_strategies is given -- see this function's own
                # docstring for why mixing per-draw, rather than pinning
                # one strategy per batch, doesn't cost anything extra
                # here), and own growth each time, so within one sample
                # the draws aren't just copies of each other.
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
