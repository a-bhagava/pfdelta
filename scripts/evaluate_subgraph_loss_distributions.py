"""
Evaluates an already-trained, FROZEN model's per-subgraph consistency/PBL
loss DISTRIBUTIONS (not just their mean, like evaluate_consistency_transfer.
py reports) on whichever validation dataset(s) you specify -- no training,
no gradient steps, no optimizer/scheduler/checkpoint machinery.

Answers: is a model equally self-consistent across every subgraph it's
handed, or do a few outlier subgraphs (e.g. a particular local topology,
or a bus that ends up promoted to local slack) drag the mean down while
most subgraphs are fine? A single reported mean can't tell you that; the
full distribution can.

Reuses SubproblemConsistencyLoss.collect_distributions (see its own
docstring in core/utils/custom_losses.py) -- toggled on directly here,
around the eval loop, rather than by a trainer: there's no "last epoch"
concept in a standalone eval script, every batch here already belongs to
the one pass you asked for, so distributions are collected the whole time
this script runs, per dataset, not conditionally.

Usage:
    python scripts/evaluate_subgraph_loss_distributions.py --config core/configs/subgraph_loss_distributions_eval.yaml
"""
import argparse
import json
import os
import statistics
import sys

sys.path.append(os.getcwd())

import torch
import yaml
from torch_geometric.loader.dataloader import DataLoader

import core.datasets.pfdelta_variants  # noqa: F401
import core.models.canos_pf  # noqa: F401
from core.utils.custom_losses import SubproblemConsistencyLoss
# Reused as-is -- these are generic to "load an already-trained model /
# resolve where to read it from / resolve where to write results", nothing
# here is specific to evaluate_consistency_transfer.py's own case x
# size-band sweep.
from scripts.evaluate_consistency_transfer import (
    build_dataset,
    build_model,
    load_weights,
    resolve_architecture,
    resolve_model_path,
    resolve_output_dir,
)

METRICS = ["angle", "magnitude", "injection", "edge", "pbl"]


@torch.no_grad()
def evaluate_one_dataset(
    model,
    dataset,
    min_size,
    max_size,
    batch_size,
    subgraphs_per_sample,
    sampling_strategies,
    seed,
    detach_teacher,
    pool_path,
    device,
) -> dict:
    """Runs SubproblemConsistencyLoss over every batch of `dataset`, with
    collect_distributions on the whole time, and returns {metric:
    [per-subgraph values]} for METRICS -- every subgraph cut from every
    batch in this ONE dataset, accumulated in one place (a fresh loss
    instance per dataset, so nothing carries over between datasets)."""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    consistency = SubproblemConsistencyLoss(
        min_size=min_size,
        max_size=max_size,
        subgraphs_per_sample=subgraphs_per_sample,
        sampling_strategies=sampling_strategies,
        seed=seed,
        detach_teacher=detach_teacher,
        pool_path=pool_path,
    )
    consistency.model = model
    consistency.collect_distributions = True
    model.eval()

    n_batches = 0
    for data in dataloader:
        data = data.to(device)
        outputs = model(data)
        consistency(outputs, data)
        n_batches += 1
    assert n_batches > 0, (
        f"empty dataloader for min_size={min_size}, max_size={max_size} -- "
        "check the dataset actually has data for this case/split."
    )
    return consistency.distributions


def summarize(values: list) -> dict:
    """min/mean/std/max (+ count) for one metric's distribution -- std is
    the SAMPLE std (n-1, statistics.stdev), None for fewer than 2 values,
    matching scripts/aggregate_run_summaries.py's own convention."""
    if not values:
        return {"count": 0, "min": None, "mean": None, "std": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else None,
        "max": max(values),
    }


def print_summary(case_name: str, distributions: dict):
    print(f"\n{case_name}:")
    for metric in METRICS:
        s = summarize(distributions[metric])
        if s["count"] == 0:
            print(f"  {metric:<10}: (no subgraphs contributed a row for this metric)")
            continue
        std_text = f"{s['std']:.6f}" if s["std"] is not None else "n/a (1 value)"
        print(
            f"  {metric:<10}: n={s['count']:<6} min={s['min']:.6f}  "
            f"mean={s['mean']:.6f}  std={std_text}  max={s['max']:.6f}"
        )


def plot_distributions(results: dict, out_path: str):
    """One figure, a grid of histograms: rows=METRICS, columns=cases (in
    `results`' own order). Each cell is one (case, metric)'s full
    per-subgraph distribution, with min/mean/std/max as directly-labeled
    reference lines rather than left for the reader to eyeball off the
    bars -- a solid line at the mean, a shaded +/-1 std band around it,
    dashed lines at min/max, and the exact numbers in a corner text box
    (still the most legible way to show 4 specific numbers on a
    histogram). Skips a cell entirely (blank, annotated) when that
    (case, metric) has no data at all."""
    import matplotlib.pyplot as plt

    case_names = list(results.keys())
    n_rows, n_cols = len(METRICS), len(case_names)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.6 * n_cols, 2.6 * n_rows), squeeze=False,
    )

    # One fixed hue for every histogram (this is a single series per cell,
    # not a categorical comparison -- color here carries no identity, so
    # it never needs to vary): the dataviz skill's default sequential hue
    # (blue) for the distribution itself, its categorical slot 2 (orange)
    # for the mean reference line layered on top -- same blue/orange
    # pairing this repo's own plot_training_objective.py already uses for
    # train/val.
    bar_color = "#2a78d6"
    mean_color = "#eb6834"

    for row, metric in enumerate(METRICS):
        for col, case_name in enumerate(case_names):
            ax = axes[row][col]
            values = results[case_name][metric]
            if not values:
                ax.axis("off")
                ax.set_title("no data", fontsize=8, color="gray")
            else:
                s = summarize(values)
                ax.hist(values, bins=30, color=bar_color, edgecolor="white", linewidth=0.3)
                ax.axvline(s["mean"], color=mean_color, linewidth=1.5, label="mean")
                if s["std"] is not None:
                    ax.axvspan(
                        s["mean"] - s["std"], s["mean"] + s["std"],
                        color=mean_color, alpha=0.15, linewidth=0,
                    )
                ax.axvline(s["min"], color="black", linewidth=1, linestyle="--", alpha=0.6)
                ax.axvline(s["max"], color="black", linewidth=1, linestyle="--", alpha=0.6)
                std_text = f"{s['std']:.4g}" if s["std"] is not None else "n/a"
                ax.text(
                    0.98, 0.95,
                    f"min {s['min']:.4g}\nmean {s['mean']:.4g}\nstd {std_text}\nmax {s['max']:.4g}\nn={s['count']}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=6.5,
                    bbox=dict(boxstyle="round", facecolor="white", edgecolor="#cccccc", alpha=0.9),
                )
                ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.2, linewidth=0.5)
            if row == 0:
                ax.set_title(case_name, fontsize=9)
            if col == 0:
                ax.set_ylabel(metric, fontsize=9)

    fig.suptitle("Per-subgraph loss distributions", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved {out_path}")


def main(config_path: str) -> dict:
    """Returns {case_name: {metric: [values]}} -- factored out of the CLI
    entry point so it can be called directly too, same as evaluate_
    consistency_transfer.py's own main()."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    architecture = resolve_architecture(cfg)
    model_path = resolve_model_path(cfg)
    output_dir = resolve_output_dir(cfg)
    os.makedirs(output_dir, exist_ok=True)

    case_names = cfg["cases"]

    print(f"Loading reference dataset ({case_names[0]}, split={cfg['split']}) for model construction...")
    reference_dataset = build_dataset(
        cfg["dataset_name"], cfg["root_dir"], case_names[0], cfg["split"],
        cfg["model_dataset_name"], cfg["task"], cfg["add_bus_type"],
    )
    print(f"Architecture: {architecture}")
    model = build_model(architecture, reference_dataset, device)
    load_weights(model, model_path)
    print(f"Loaded weights from {model_path}")

    dataset_cache = {case_names[0]: reference_dataset}

    results = {}
    for case_name in case_names:
        if case_name not in dataset_cache:
            print(f"Loading dataset for {case_name}...")
            dataset_cache[case_name] = build_dataset(
                cfg["dataset_name"], cfg["root_dir"], case_name, cfg["split"],
                cfg["model_dataset_name"], cfg["task"], cfg["add_bus_type"],
            )
        dataset = dataset_cache[case_name]
        print(f"Evaluating {case_name}...")
        distributions = evaluate_one_dataset(
            model,
            dataset,
            cfg["min_size"],
            cfg["max_size"],
            cfg.get("batch_size", 64),
            cfg.get("subgraphs_per_sample", 4),
            cfg.get("sampling_strategies"),
            cfg.get("seed"),
            cfg.get("detach_teacher", True),
            cfg.get("pool_path"),
            device,
        )
        results[case_name] = distributions
        print_summary(case_name, distributions)

    out_json = os.path.join(output_dir, "subgraph_loss_distributions.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full per-subgraph distributions to {out_json}")

    out_png = os.path.join(output_dir, "subgraph_loss_distributions.png")
    plot_distributions(results, out_png)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
