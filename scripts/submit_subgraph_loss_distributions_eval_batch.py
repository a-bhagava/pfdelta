"""
Fans scripts/evaluate_subgraph_loss_distributions.py out across every run
folder matching a name pattern, submitting each as its OWN sbatch job (so
they run in parallel, each grabbing its own GPU) instead of evaluating them
one at a time in a single job.

Discovery uses the exact same {param}-placeholder name_pattern syntax as
scripts/aggregate_run_summaries.py (see that script's own module docstring
for the full explanation of the syntax and why braces, not bare %param) --
reused directly (build_name_regex), so a pattern you've already written for
aggregation works here unchanged. Unlike aggregation, nothing here groups
by the extracted param values -- a match is a match, each gets its own job.

The eval config itself (min_size/max_size/cases/etc.) is shared, UNCHANGED,
across every discovered run -- only `source_run` varies per job (passed as
a CLI override, see evaluate_subgraph_loss_distributions.py's own --source_
run flag), via scripts/submit_subgraph_loss_distributions_eval.sh's own
optional 2nd argument. Leave model_path/output_dir unset in the config for
this mode -- both default to source_run (see resolve_model_path/resolve_
output_dir in evaluate_consistency_transfer.py), so each job's own model
weights and result files stay scoped to ITS OWN run folder; a hardcoded
model_path/output_dir would instead point every job at the same one.

Usage:
    python scripts/submit_subgraph_loss_distributions_eval_batch.py --config core/configs/subgraph_loss_distributions_eval.yaml
    python scripts/submit_subgraph_loss_distributions_eval_batch.py --config core/configs/subgraph_loss_distributions_eval.yaml --dry_run
"""
import argparse
import glob
import os
import subprocess
import sys

sys.path.append(os.getcwd())

import yaml

from scripts.aggregate_run_summaries import build_name_regex

SUBMIT_SCRIPT = "scripts/submit_subgraph_loss_distributions_eval.sh"


def discover_matching_runs(folder: str, name_pattern: str) -> list:
    """Every immediate subdirectory of `folder` whose basename matches
    `name_pattern` (see build_name_regex -- anchored at the start only, so
    a trailing job-index/timestamp suffix is ignored) AND has both
    model.pt and config.yaml (what evaluate_subgraph_loss_distributions.py
    actually needs off a source_run) -- anything else is skipped, with a
    note, same convention as aggregate_run_summaries.py's own discovery."""
    regex = build_name_regex(name_pattern)
    candidates = sorted(
        d for d in glob.glob(os.path.join(folder, "*")) if os.path.isdir(d)
    )
    matches = []
    for run_folder in candidates:
        if regex.match(os.path.basename(run_folder)) is None:
            continue
        if not (
            os.path.exists(os.path.join(run_folder, "model.pt"))
            and os.path.exists(os.path.join(run_folder, "config.yaml"))
        ):
            print(f"  (skipping {run_folder!r} -- matches the pattern but missing model.pt/config.yaml)")
            continue
        matches.append(run_folder)
    return matches


def main(config_path: str, dry_run: bool = False) -> list:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    assert "folder" in cfg and "name_pattern" in cfg, (
        "This config needs \"folder\" + \"name_pattern\" (batch discovery mode) -- "
        "for a single run, just sbatch scripts/submit_subgraph_loss_distributions_"
        "eval.sh directly instead, no need for this dispatcher."
    )

    matches = discover_matching_runs(cfg["folder"], cfg["name_pattern"])
    assert matches, (
        f"No run folders under {cfg['folder']!r} matched name_pattern "
        f"{cfg['name_pattern']!r} (with model.pt+config.yaml present)."
    )

    print(f"Found {len(matches)} matching run(s):")
    for run_folder in matches:
        print(f"  {run_folder}")
    print()

    job_ids = []
    for run_folder in matches:
        job_name = f"subgraph_dist_{os.path.basename(run_folder)}"
        command = [
            "sbatch", "--job-name", job_name,
            SUBMIT_SCRIPT, config_path, run_folder,
        ]
        if dry_run:
            print(f"[dry run] {' '.join(command)}")
            continue
        result = subprocess.run(command, capture_output=True, text=True)
        print(f"{run_folder} -> {result.stdout.strip() or result.stderr.strip()}")
        if result.returncode != 0:
            print(f"  WARNING: sbatch exited {result.returncode}: {result.stderr.strip()}")
        else:
            job_ids.append(result.stdout.strip())

    if not dry_run:
        print(f"\nSubmitted {len(job_ids)}/{len(matches)} job(s).")
    return matches


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--dry_run", action="store_true", default=False,
        help="Print the sbatch commands that would run, without submitting anything.",
    )
    args = parser.parse_args()
    main(args.config, dry_run=args.dry_run)
