"""
Aggregates summary.json across a set of runs into one readable table --
e.g. comparing a few sweep values (branch_perturbation_sigma, w_sg_pbl,
...), each backed by one or more seed runs, without doing it by hand.

Three ways to say which runs go together, all driven by
core/configs/aggregate_run_summaries.yaml:

1. Auto-discovery by run name ("folder" + "name_pattern" in the yaml):
   point it at one parent folder holding just the run folders you want
   compared -- every immediate subfolder is matched against `name_pattern`,
   a template with {param} placeholders (braces required, not bare
   %param -- "_" is a valid identifier character, so without an explicit
   closing delimiter "%zero_at_epoch_sg_per_s_" would parse as ONE param
   name, not "zero_at_epoch" followed by literal "_sg_per_s_") over the
   folder's basename, e.g.
   "canos_full_pbl_zero_epoch_{zero_at_epoch}_sg_per_s_{subgraphs_per_sample}_seed_{seed}"
   -- values must not themselves contain "_". List which of those params
   to collapse across in `aggregate_over` (e.g. ["seed"]) -- every OTHER
   param extracted from the name becomes part of the column label, and
   runs with the same value for all of them get grouped/averaged together.
   No config.yaml read at all in this mode.
2. Auto-discovery by config.yaml value ("folder" + "sweep_key"): every
   immediate subfolder with a summary.json is scanned, grouped by the
   value of `sweep_key` (a dotted path into that run's own config.yaml,
   e.g. "optim.train_params.train_loss.0.w_sg_pbl" -- list indices are
   plain integers) read straight out of its config.yaml. Use this when the
   thing you're sweeping isn't in the run name, or config.yaml is the more
   reliable source.
3. Manual ("groups" in the yaml): explicit column label -> list of run
   folder paths. Use this when runs you want compared aren't conveniently
   sitting in one folder together, or don't share a name pattern/config.yaml
   field to group by.

A column's runs are averaged (mean +/- std, std only shown for 2+ runs;
std is the SAMPLE std, n-1 denominator/Bessel's correction, via
statistics.stdev, not population std) -- so a group with several seeds
behind one sweep value collapses to a single readable cell, same as doing
it by hand. NaN values (see e.g. SubgraphFinetuneTrainer's val-side NaN
stand-in for a case/metric that genuinely wasn't evaluated) are dropped
before averaging, not treated as 0 or as poisoning the whole mean -- a
cell is "NaN" only if EVERY run in that column was NaN for this row.

Usage:
    python scripts/aggregate_run_summaries.py --config core/configs/aggregate_run_summaries.yaml
"""
import argparse
import glob
import json
import math
import os
import re
import statistics
import sys

sys.path.append(os.getcwd())

import yaml


def load_summary(run_folder: str) -> dict:
    path = os.path.join(run_folder, "summary.json")
    assert os.path.exists(path), f"No summary.json in {run_folder!r}"
    with open(path) as f:
        return json.load(f)


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_nested(config: dict, dotted_path: str):
    """Walks `dotted_path` (e.g. "optim.train_params.train_loss.0.w_sg_pbl")
    into a nested config.yaml structure -- each "." segment is a dict key,
    unless it's all-digits, in which case it's a list index."""
    value = config
    for part in dotted_path.split("."):
        key = int(part) if part.isdigit() else part
        try:
            value = value[key]
        except (KeyError, IndexError, TypeError) as e:
            raise KeyError(
                f"sweep_key {dotted_path!r} -- couldn't resolve segment {part!r} "
                f"(value at that point was {value!r})"
            ) from e
    return value


def _sweep_sort_key(value):
    """Numeric sweep values sort numerically (0.03 before 0.1); anything
    else falls back to string sort, after all numeric values."""
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))


def discover_groups(folder: str, sweep_key: str) -> dict:
    """Scans `folder`'s immediate subdirectories for runs (summary.json +
    config.yaml both present; anything else is skipped, with a note) and
    groups them by the value of `sweep_key` read out of each run's own
    config.yaml. Returns {group_label: [run_folder, ...]}, ordered by the
    sweep value (numeric sort if all values are numeric)."""
    candidates = sorted(
        d for d in glob.glob(os.path.join(folder, "*")) if os.path.isdir(d)
    )
    value_to_folders = {}
    for run_folder in candidates:
        summary_path = os.path.join(run_folder, "summary.json")
        config_path = os.path.join(run_folder, "config.yaml")
        if not (os.path.exists(summary_path) and os.path.exists(config_path)):
            print(f"  (skipping {run_folder!r} -- no summary.json/config.yaml)")
            continue
        value = get_nested(load_yaml(config_path), sweep_key)
        value_to_folders.setdefault(value, []).append(run_folder)

    sweep_name = sweep_key.split(".")[-1]
    groups = {}
    for value in sorted(value_to_folders, key=_sweep_sort_key):
        groups[f"{sweep_name}={value}"] = value_to_folders[value]
    return groups


def _parse_scalar(s: str):
    """int if it parses as one, else float if it parses as one, else the
    raw string -- so e.g. "0" sorts/compares as 0, not "0" < "16" as text."""
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def build_name_regex(name_pattern: str) -> re.Pattern:
    """Turns a template like
    "canos_full_pbl_zero_epoch_{zero_at_epoch}_sg_per_s_{subgraphs_per_sample}_seed_{seed}"
    into a regex with one named group per {param}, each matching a run of
    non-"_" characters -- so this assumes param values never contain "_"
    themselves (true for the numeric seeds/epochs/etc. this is meant for).
    Braces (not bare %param) are required to unambiguously mark where a
    param name ends and literal text resumes, since "_" is itself a valid
    identifier character. Matched with re.match (anchored at the start
    only, not fullmatch) so a trailing job-index/timestamp suffix in the
    actual folder name -- e.g. "..._seed_0_0_260905_230645" -- is simply
    ignored."""
    parts = re.split(r"(\{[A-Za-z_][A-Za-z0-9_]*\})", name_pattern)
    pattern = "".join(
        f"(?P<{part[1:-1]}>[^_]+)" if part.startswith("{") else re.escape(part)
        for part in parts
    )
    return re.compile(pattern)


def discover_groups_by_name(folder: str, name_pattern: str, aggregate_over: list) -> dict:
    """Scans `folder`'s immediate subdirectories for runs (summary.json
    present; anything else skipped, with a note), extracts %params from
    each folder's basename via `name_pattern` (see build_name_regex), and
    groups by the extracted params EXCLUDING `aggregate_over` -- so runs
    that only differ in an aggregate_over param (e.g. seed) land in the
    same group and get averaged. Returns {group_label: [run_folder, ...]},
    ordered by the (non-aggregated) param values."""
    regex = build_name_regex(name_pattern)
    candidates = sorted(
        d for d in glob.glob(os.path.join(folder, "*")) if os.path.isdir(d)
    )
    key_to_folders = {}
    for run_folder in candidates:
        if not os.path.exists(os.path.join(run_folder, "summary.json")):
            print(f"  (skipping {run_folder!r} -- no summary.json)")
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
        label = ", ".join(f"{k}={v}" for k, v in key) if key else "all"
        groups[label] = key_to_folders[key]
    return groups


def extract_metric(summary: dict, path: str):
    """`path` is "train.<key>" or "val.<index>.<key>" -- <key> is taken
    VERBATIM after the first (train) or second (val) "." (no further
    splitting), so a metric name containing its own "." would still work;
    none of this repo's own metric names do, but this is safe regardless.
    <index> for "val" matches calc_one_val_error's own val_num (0-indexed
    into dataset.datasets[1:] -- for this repo's standard layout, 0=case14,
    1=case30, 2=case57, 3=case118, 4=case500)."""
    if path.startswith("train."):
        key = path[len("train."):]
        value = summary["train"].get(key)
    elif path.startswith("val."):
        rest = path[len("val."):]
        idx_str, _, key = rest.partition(".")
        value = summary["val"][int(idx_str)].get(key)
    else:
        raise ValueError(f"row path must start with 'train.' or 'val.<index>.', got {path!r}")
    return float("nan") if value is None else float(value)


def mean_std(values: list) -> tuple:
    """Drops NaN before averaging -- a cell is only "all NaN" if every run
    in that column was NaN for this row, not if even one of several seeds
    happened to be (e.g. subproblem_consistency's own NaN stand-in on a
    val dataset it didn't run on for THAT one run's config).

    Uses statistics.stdev (sample std, n-1 denominator/Bessel's correction),
    not statistics.pstdev (population std, n denominator) -- each group's
    runs are a sample of seeds, not the full population, so n-1 is the
    unbiased estimator."""
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return float("nan"), None
    if len(clean) == 1:
        return clean[0], None
    return statistics.mean(clean), statistics.stdev(clean)


def format_cell(mean: float, std) -> str:
    if math.isnan(mean):
        return "NaN"
    if std is None:
        return f"{mean:.4f}"
    return f"{mean:.4f} +/- {std:.4f}"


def resolve_groups(cfg: dict) -> dict:
    """Returns {group_label: [run_folder, ...]}, from "groups" (manual),
    "folder" + "name_pattern" (auto-discovery by run name), or "folder" +
    "sweep_key" (auto-discovery by config.yaml value) -- see the module
    docstring for all three modes."""
    if "groups" in cfg:
        return cfg["groups"]
    if "name_pattern" in cfg:
        assert "folder" in cfg, "\"name_pattern\" mode also needs \"folder\""
        groups = discover_groups_by_name(cfg["folder"], cfg["name_pattern"], cfg.get("aggregate_over", []))
    elif "sweep_key" in cfg:
        assert "folder" in cfg, "\"sweep_key\" mode also needs \"folder\""
        groups = discover_groups(cfg["folder"], cfg["sweep_key"])
    else:
        raise AssertionError(
            "config needs one of: \"groups\" (manual: label -> [run folders]), "
            "\"folder\" + \"name_pattern\" (auto-discovery by run name), or "
            "\"folder\" + \"sweep_key\" (auto-discovery by config.yaml field)"
        )
    assert groups, f"No matching run folders found under {cfg['folder']!r}"
    return groups


def build_table(cfg: dict, groups: dict) -> tuple:
    """Returns (header, rows) -- header is [""] + column labels, rows is a
    list of [row_label, cell_str, cell_str, ...], with the best (lowest,
    unless lower_is_better: false) cell in each row marked for bolding."""
    group_labels = list(groups.keys())
    header = [""] + group_labels
    lower_is_better = cfg.get("lower_is_better", True)

    # Load every run folder's summary.json once, not once per row.
    group_summaries = {
        label: [load_summary(folder) for folder in folders]
        for label, folders in groups.items()
    }

    rows = []
    bold_col = []  # index (into group_labels) of the best cell per row, or None
    for row_spec in cfg["rows"]:
        label, path = row_spec["label"], row_spec["path"]
        means = []
        cells = [label]
        for group_label in group_labels:
            values = [extract_metric(s, path) for s in group_summaries[group_label]]
            mean, std = mean_std(values)
            means.append(mean)
            cells.append(format_cell(mean, std))
        rows.append(cells)

        finite = [(i, m) for i, m in enumerate(means) if not math.isnan(m)]
        if finite:
            best_i = min(finite, key=lambda im: im[1])[0] if lower_is_better else max(finite, key=lambda im: im[1])[0]
            bold_col.append(best_i)
        else:
            bold_col.append(None)

    return header, rows, bold_col


def render_markdown(header: list, rows: list, bold_col: list) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for row, best_i in zip(rows, bold_col):
        cells = [row[0]]
        for i, cell in enumerate(row[1:]):
            cells.append(f"**{cell}**" if i == best_i else cell)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_console(header: list, rows: list) -> str:
    widths = [max(len(str(r[c])) for r in ([header] + rows)) for c in range(len(header))]
    def fmt_row(r):
        return "  ".join(str(cell).ljust(w) for cell, w in zip(r, widths))
    lines = [fmt_row(header), "  ".join("-" * w for w in widths)]
    lines += [fmt_row(r) for r in rows]
    return "\n".join(lines)


def main(config_path: str) -> str:
    cfg = load_yaml(config_path)

    groups = resolve_groups(cfg)
    print("Groups:")
    for label, folders in groups.items():
        print(f"  {label}: {len(folders)} run(s)")
        for folder in folders:
            print(f"    {folder}")
    print()

    header, rows, bold_col = build_table(cfg, groups)

    print(render_console(header, rows))

    # Default output location: the scanned folder itself in auto-discovery
    # mode (that's the whole point -- point it at a folder, get a table in
    # that folder back), otherwise output_dir must be given explicitly.
    output_dir = cfg.get("output_dir", cfg.get("folder"))
    assert output_dir is not None, "config needs \"output_dir\" (or \"folder\", in auto-discovery mode)"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, cfg.get("output_filename", "aggregated_results.md"))
    with open(output_path, "w") as f:
        f.write(render_markdown(header, rows, bold_col) + "\n")
    print(f"\nSaved to {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
