"""
Aggregates scripts/evaluate_subgraph_loss_distributions.py's own output
(subgraph_loss_distributions.json) across every run folder matching a name
pattern under one parent folder -- same {param}-placeholder discovery as
scripts/aggregate_run_summaries.py (build_name_regex reused directly, so a
pattern you've already written there works here unchanged), grouped the
same way (aggregate_over collapses e.g. seeds into one group).

What gets aggregated is one level up from that script's own summarize():
each matched run already has its OWN per-(case, metric) min/mean/std/max
(over that run's per-subgraph distribution). This script gathers those same
four numbers ACROSS every run in a group and reports the mean +/- std of
EACH of them -- e.g. "mean of means" (how consistent the typical subgraph's
loss is across seeds) alongside "mean of maxes" (how consistent the worst
outlier is across seeds), not just one pooled number. std is the SAMPLE std
(n-1, statistics.stdev), same convention as aggregate_run_summaries.py.

Which of min/mean/std/max to aggregate is configurable (`stats:` in the
config, default all four) -- e.g. `stats: [mean, max]` to skip min/std
entirely and halve the table size when you only care about the typical
case and the worst outlier.

Usage:
    python scripts/aggregate_subgraph_loss_distributions.py --config core/configs/aggregate_subgraph_loss_distributions.yaml
"""
import argparse
import glob
import json
import os
import sys

sys.path.append(os.getcwd())

import yaml

from scripts.aggregate_run_summaries import (
    _display_name,
    _parse_scalar,
    _sweep_sort_key,
    build_name_regex,
    format_cell,
    mean_std,
)
from scripts.evaluate_subgraph_loss_distributions import METRICS, summarize

ALL_STATS = ["min", "mean", "std", "max"]
DISTRIBUTIONS_FILENAME = "subgraph_loss_distributions.json"


def resolve_stats(cfg: dict) -> list:
    """Which per-run stats (min/mean/std/max) to aggregate and report --
    `stats:` in the config, defaulting to all four. Order in the config is
    preserved (so e.g. `stats: [max, mean]` puts max rows above mean rows
    in the output) rather than forcing ALL_STATS's own order."""
    stats = cfg.get("stats", ALL_STATS)
    unknown = set(stats) - set(ALL_STATS)
    assert not unknown, f"stats {sorted(unknown)} not in {ALL_STATS}"
    assert stats, "stats can't be empty -- pick at least one of " + str(ALL_STATS)
    return stats


def discover_groups(folder: str, name_pattern: str, aggregate_over: list) -> dict:
    """Same discovery as aggregate_run_summaries.discover_groups_by_name,
    just checking for DISTRIBUTIONS_FILENAME instead of summary.json --
    kept as its own small function rather than generalizing that one,
    since the two scripts' required-file contracts differ and this is the
    only place that matters. Returns {group_label: [run_folder, ...]},
    ordered by the (non-aggregated) param values."""
    regex = build_name_regex(name_pattern)
    candidates = sorted(
        d for d in glob.glob(os.path.join(folder, "*")) if os.path.isdir(d)
    )
    key_to_folders = {}
    for run_folder in candidates:
        if not os.path.exists(os.path.join(run_folder, DISTRIBUTIONS_FILENAME)):
            print(f"  (skipping {run_folder!r} -- no {DISTRIBUTIONS_FILENAME})")
            continue
        m = regex.match(os.path.basename(run_folder))
        if m is None:
            print(f"  (skipping {run_folder!r} -- doesn't match name_pattern {name_pattern!r})")
            continue
        params = {k: _parse_scalar(v) for k, v in m.groupdict().items()}
        unknown = set(aggregate_over) - set(params)
        assert not unknown, f"aggregate_over {sorted(unknown)} not in name_pattern's params {sorted(params)}"
        key = tuple((k, v) for k, v in params.items() if k not in aggregate_over)
        key_to_folders.setdefault(key, []).append(run_folder)

    groups = {}
    for key in sorted(key_to_folders, key=lambda k: tuple(_sweep_sort_key(v) for _, v in k)):
        label = ", ".join(f"{_display_name(k)}={v}" for k, v in key) if key else "all"
        groups[label] = key_to_folders[key]
    return groups


def aggregate_group(run_folders: list, stats: list = ALL_STATS) -> dict:
    """Returns {case: {metric: {stat: (agg_mean, agg_std, n_runs)}}} --
    for each (case, metric), gathers each run's OWN summarize() stat (one
    of min/mean/std/max, whichever `stats` asks for, over that run's
    per-subgraph distribution) and computes the mean+std of each ACROSS
    runs. A run missing a (case, metric) entirely (e.g. it evaluated a
    different case list, or that metric had zero subgraphs -- see
    summarize()'s own count=0 case) just doesn't contribute to that cell,
    same as summary.json's own NaN-dropping convention -- n_runs (the 3rd
    tuple element) says how many runs actually did."""
    # per_run_stats[case][metric][stat] = [run1's own stat value, run2's, ...]
    per_run_stats = {}
    for run_folder in run_folders:
        path = os.path.join(run_folder, DISTRIBUTIONS_FILENAME)
        with open(path) as f:
            distributions = json.load(f)
        for case, case_dist in distributions.items():
            by_metric = per_run_stats.setdefault(case, {})
            for metric in METRICS:
                s = summarize(case_dist.get(metric, []))
                by_stat = by_metric.setdefault(metric, {stat: [] for stat in stats})
                for stat in stats:
                    if s[stat] is not None:
                        by_stat[stat].append(s[stat])

    aggregated = {}
    for case, by_metric in per_run_stats.items():
        aggregated[case] = {}
        for metric, by_stat in by_metric.items():
            aggregated[case][metric] = {}
            for stat, values in by_stat.items():
                agg_mean, agg_std = mean_std(values)
                aggregated[case][metric][stat] = (agg_mean, agg_std, len(values))
    return aggregated


def _ordered_cases(all_results: dict) -> list:
    """Union of case names across every group, in first-seen order --
    groups share ONE eval config in the intended usage (see this script's
    own module docstring / submit_subgraph_loss_distributions_eval_batch.
    py's), so this is normally just that config's own `cases` list, but
    falling back to a union keeps this robust if they ever differ."""
    seen = []
    for aggregated in all_results.values():
        for case in aggregated:
            if case not in seen:
                seen.append(case)
    return seen


def render_markdown(all_results: dict, stats: list = ALL_STATS) -> str:
    group_labels = list(all_results.keys())
    cases = _ordered_cases(all_results)
    lines = []
    for metric in METRICS:
        lines.append(f"## {metric}\n")
        header = [""] + group_labels
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for case in cases:
            for stat in stats:
                row = [f"{case} {stat}"]
                for group_label in group_labels:
                    cell = all_results[group_label].get(case, {}).get(metric, {}).get(stat)
                    if cell is None:
                        row.append("--")
                    else:
                        agg_mean, agg_std, n_runs = cell
                        row.append(f"{format_cell(agg_mean, agg_std)} (n={n_runs})")
                lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    return "\n".join(lines)


def main(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    stats = resolve_stats(cfg)
    groups = discover_groups(cfg["folder"], cfg["name_pattern"], cfg.get("aggregate_over", []))
    assert groups, (
        f"No run folders under {cfg['folder']!r} matched name_pattern "
        f"{cfg['name_pattern']!r} (with {DISTRIBUTIONS_FILENAME} present)."
    )
    print("Groups:")
    for label, folders in groups.items():
        print(f"  {label}: {len(folders)} run(s)")
        for folder in folders:
            print(f"    {folder}")
    print()
    print(f"Aggregating stats: {stats}")
    print()

    all_results = {label: aggregate_group(folders, stats) for label, folders in groups.items()}

    output_dir = cfg.get("output_dir", cfg["folder"])
    os.makedirs(output_dir, exist_ok=True)

    out_json = os.path.join(output_dir, cfg.get("output_json_filename", "subgraph_loss_distributions_aggregate.json"))
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved full aggregated stats to {out_json}")

    out_md = os.path.join(output_dir, cfg.get("output_filename", "subgraph_loss_distributions_aggregate.md"))
    with open(out_md, "w") as f:
        f.write(render_markdown(all_results, stats))
    print(f"Saved readable summary to {out_md}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
