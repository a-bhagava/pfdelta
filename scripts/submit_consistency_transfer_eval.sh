#!/bin/bash
# sbatch launcher for scripts/evaluate_consistency_transfer.py.
#
# This eval is deliberately NOT wired into the main.py/simple_batch config
# machinery -- that system hardcodes "python main.py --config <name>" and
# requires a top-level "model:" dict (see core/utils/main_utils.py::
# single_config), which consistency_transfer_eval.yaml's flat shape isn't
# built to satisfy. So this is a plain sbatch script instead, following the
# same pattern as data_generation/script_parallel.sh.
#
# Usage (from the repo root, so the job's cwd is ~/pfdelta -- needed for
# evaluate_consistency_transfer.py's own sys.path.append(os.getcwd())):
#   sbatch scripts/submit_consistency_transfer_eval.sh
#   sbatch scripts/submit_consistency_transfer_eval.sh core/configs/some_other_eval.yaml
#
#SBATCH --job-name=consistency_transfer_eval
#SBATCH -p mit_normal_gpu,mit_preemptable,pi_donti_gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --time=05:59:00
#SBATCH --output=consistency_transfer_eval_%j.out
# No --requeue/--signal here (unlike the training jobs' sbatch) -- this is
# a single-shot, no-gradient-steps eval with no checkpoint/resume state to
# preserve, so a preemption just means resubmitting the whole thing.

source ~/.bashrc
conda activate pfdelta2

CONFIG="${1:-core/configs/consistency_transfer_eval.yaml}"
python scripts/evaluate_consistency_transfer.py --config "$CONFIG"
