"""
One-time, offline precomputation of a subgraph pool for the subgraph-
consistency augmentation (see core.utils.create_subproblem's own POOLED
sampling section, just above build_subgraph_pool_data/draw_subgraphs_
pooled). Run this ONCE per (case, min_size, max_size) you care about, then
point a training config's subproblem_consistency entries (both train_loss
and val_loss) at the saved file via `pool_path` -- every subsequent run
(and every requeue of the same job) just loads and indexes into it instead
of growing subgraphs fresh, which is where sample_bus_subset's own cost was
actually going once subgraphs_per_sample got large (see the profiling that
motivated this).

ONE file covers every strategy you list in `strategies` below AND every mix
of them a training run might later sweep (pure_bfs, even_mix, no_bfs, ...)
-- each strategy gets its own full-size sub-pool inside the file, keyed by
name; a training run's own `sampling_strategies` proportions decide the mix
drawn from those sub-pools at TRAIN time, not here. A strategy's other
kwargs (restart_prob/p/degree_weight_power) ARE fixed at build time, since
they change what actually gets grown -- list every (strategy, kwargs)
combination you'll ever want under `strategies`.

Safe to use during training even WITH N-1/N-2 contingency (`perturbation`
"n-1"/"n-2"), as long as this script itself discovers the case's FULL
(uncontingent) topology to build against -- which it does by scanning
`topology_scan_samples` samples and taking their UNION (same convergence
logic core.utils.create_subproblem.update_base_topology_cache uses at
train time), regardless of what perturbation THIS dataset happens to use.
At train time, draw_subgraphs_pooled checks each draw against its own
sample's actual missing branches and redraws if a chosen shape relies on
one that's offline -- see that function's own docstring.

Usage:
    python scripts/build_subgraph_pool.py --config core/configs/build_subgraph_pool.yaml
"""
import argparse
import os
import sys
import time

sys.path.append(os.getcwd())

import torch
import yaml
from torch_geometric.data import Batch

import core.datasets.pfdelta_variants  # noqa: F401
import core.models.canos_pf  # noqa: F401
from core.utils.create_subproblem import build_subgraph_pool_data, update_base_topology_cache
from core.utils.registry import registry


def build_dataset(dataset_name, root_dir, case_name, split, model_dataset_name, task, add_bus_type):
    dataset_class = registry.get_dataset_class(dataset_name)
    assert dataset_class is not None, f"Dataset {dataset_name!r} not found in registry!"
    return dataset_class(
        root_dir=root_dir,
        case_name=case_name,
        split=split,
        model=model_dataset_name,
        task=task,
        add_bus_type=add_bus_type,
    )


def discover_full_topology(dataset, num_scan_samples: int) -> tuple:
    """Union-across-samples topology discovery, same convergence logic as
    core.utils.create_subproblem.update_base_topology_cache uses at train
    time -- works regardless of whether `dataset` itself has N-1/N-2
    contingency (a single sample's own edge_index might be missing a
    branch; the union over enough samples isn't). Returns (base_adjacency,
    num_buses)."""
    n = min(num_scan_samples, len(dataset))
    batch = Batch.from_data_list([dataset[i] for i in range(n)])
    num_buses = dataset[0]["bus"].num_nodes
    edge_index = batch["bus", "branch", "bus"].edge_index
    ptr = batch["bus"].ptr
    base_cache: dict = {}
    update_base_topology_cache(edge_index, ptr, num_buses, base_cache)
    return base_cache["base_adjacency"], num_buses


def make_progress_printer():
    """Prints one line per chunk (see build_subgraph_pool_data's own
    chunk_size/progress_callback), flushed immediately -- SLURM's .out file
    is a redirected (non-tty) stream, so Python fully buffers stdout by
    default and these wouldn't show up live otherwise, regardless of how
    often we print. flush=True here covers this function's own prints even
    if the script is ever run without -u; submit_build_subgraph_pool.sh
    also runs python -u so every OTHER print in this file (dataset
    loading, download progress, etc.) shows up live too, not just these."""
    start = time.perf_counter()

    def callback(strategy_name, strategy_drawn, strategy_total, overall_drawn, overall_total):
        elapsed = time.perf_counter() - start
        frac = overall_drawn / overall_total
        eta = (elapsed / frac - elapsed) if frac > 0 else float("inf")
        print(
            f"  [{strategy_name}] {strategy_drawn}/{strategy_total}  "
            f"(overall {overall_drawn}/{overall_total}, {frac * 100:.1f}%)  "
            f"elapsed {elapsed:.0f}s  ETA {eta:.0f}s",
            flush=True,
        )

    return callback


def main(config_path: str) -> str:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    print(f"Loading dataset ({cfg['case_name']}, split={cfg['split']}) to discover its topology...", flush=True)
    dataset = build_dataset(
        cfg["dataset_name"], cfg["root_dir"], cfg["case_name"], cfg["split"],
        cfg["model_dataset_name"], cfg["task"], cfg["add_bus_type"],
    )
    perturbation = getattr(dataset, "perturbation", "n")
    scan_samples = cfg.get("topology_scan_samples", 200)
    print(f"perturbation={perturbation!r} -- scanning {min(scan_samples, len(dataset))} samples "
          f"for the case's full (union) topology...", flush=True)
    base_adjacency, num_buses = discover_full_topology(dataset, scan_samples)
    num_edges = sum(len(n) for n in base_adjacency)
    print(f"Case has {num_buses} buses, {num_edges} directed edges in the discovered topology.", flush=True)

    generator = torch.Generator().manual_seed(cfg.get("seed", 11))
    pool_size = int(cfg["pool_size"])
    strategies = cfg["strategies"]
    # progress_chunk_size (optional): how many subgraphs to draw per
    # printed progress line -- defaults (in build_subgraph_pool_data) to
    # pool_size // 20, i.e. ~20 lines per strategy. Smaller means more
    # frequent (chattier) updates, at a small extra per-chunk dispatch cost.
    print(f"Drawing {pool_size} subgraphs PER strategy (size {cfg['min_size']}-{cfg['max_size']}) "
          f"for: {list(strategies.keys())}...", flush=True)
    pool = build_subgraph_pool_data(
        base_adjacency, num_buses, cfg["min_size"], cfg["max_size"], strategies, generator, pool_size,
        chunk_size=cfg.get("progress_chunk_size"), progress_callback=make_progress_printer(),
    )

    output_path = cfg["output_path"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(pool, output_path)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nSaved pool to {output_path} ({file_size_mb:.1f} MB)", flush=True)

    for name, strategy_pool in pool["strategies"].items():
        sizes = strategy_pool["pool_counts"]
        print(f"  {name}: min={sizes.min().item()}, max={sizes.max().item()}, "
              f"mean={sizes.float().mean().item():.2f} (requested {cfg['min_size']}-{cfg['max_size']})")

    print(f"\nOne file, every strategy above and any mix of them -- point a training config's "
          f"pool_path (subproblem_consistency's own inputs, both train and val) at:\n  {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
