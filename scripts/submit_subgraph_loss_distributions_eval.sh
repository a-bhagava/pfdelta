#!/bin/bash
# sbatch launcher for scripts/evaluate_subgraph_loss_distributions.py.
#
# Same reasoning as scripts/submit_consistency_transfer_eval.sh: this eval
# is deliberately NOT wired into the main.py/simple_batch config machinery
# (that system requires a top-level "model:" dict shape this config isn't
# built to satisfy -- see core/utils/main_utils.py::single_config), so
# it's a plain sbatch script instead.
#
# Usage (from the repo root, so the job's cwd is ~/pfdelta -- needed for
# evaluate_subgraph_loss_distributions.py's own sys.path.append(os.getcwd())):
#   sbatch scripts/submit_subgraph_loss_distributions_eval.sh
#   sbatch scripts/submit_subgraph_loss_distributions_eval.sh core/configs/some_other_eval.yaml
#   sbatch scripts/submit_subgraph_loss_distributions_eval.sh core/configs/some_other_eval.yaml runs/some_run
# The optional 2nd arg overrides the config's own source_run -- this is how
# scripts/submit_subgraph_loss_distributions_eval_batch.py fans ONE shared
# config out across many discovered runs, each its own sbatch call; you can
# also pass it by hand for a one-off run without editing the config.
#
#SBATCH --job-name=subgraph_loss_distributions_eval
#SBATCH -p mit_normal_gpu,mit_preemptable,pi_donti_gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --time=05:59:00
#SBATCH --output=subgraph_loss_distributions_eval_%j.out
# No --requeue/--signal here (unlike the training jobs' sbatch) -- this is
# a single-shot, no-gradient-steps eval with no checkpoint/resume state to
# preserve, so a preemption just means resubmitting the whole thing.

source ~/.bashrc
conda activate pfdelta2

CONFIG="${1:-core/configs/subgraph_loss_distributions_eval.yaml}"
SOURCE_RUN="${2:-}"
# -u: unbuffered stdout/stderr -- the .out file is a redirected (non-tty)
# stream, so Python fully buffers prints to it by default and nothing
# shows up until the process exits. -u forces every print (per-case
# progress, the min/mean/std/max summary) to appear live in the .out file.
if [ -n "$SOURCE_RUN" ]; then
    python -u scripts/evaluate_subgraph_loss_distributions.py --config "$CONFIG" --source_run "$SOURCE_RUN"
else
    python -u scripts/evaluate_subgraph_loss_distributions.py --config "$CONFIG"
fi
