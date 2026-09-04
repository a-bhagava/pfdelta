#!/bin/bash
# sbatch launcher for scripts/build_subgraph_pool.py -- a one-time, CPU-only
# offline precompute (see that script's own docstring), so this deliberately
# does NOT request a GPU-provisioned partition -- the whole batched sampling
# engine (core.utils.create_subproblem.draw_subgraphs_batched and friends)
# is CPU-only tensor ops by design, a GPU would sit completely idle. No -p
# flag either, matching data_generation/script_parallel.sh's own convention
# for a CPU-only job on this cluster -- just uses the default queue.
#
# --cpus-per-task=4 is deliberate, not a placeholder: PyTorch's CPU backend
# does thread across a handful of cores for these ops, but plateaus fast --
# measured 1 thread ~32s vs 4 threads ~17s vs 11 threads ~18s for a full
# 4-strategy, 1,000,000-per-strategy pool (the size this repo's own
# build_subgraph_pool.yaml template currently asks for) -- more cores past
# 4 buys nothing here.
#
# Usage (from the repo root):
#   sbatch scripts/submit_build_subgraph_pool.sh
#   sbatch scripts/submit_build_subgraph_pool.sh core/configs/some_other_pool.yaml
#
#SBATCH --job-name=build_subgraph_pool
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=build_subgraph_pool_%j.out

source ~/.bashrc
conda activate pfdelta2

CONFIG="${1:-core/configs/build_subgraph_pool.yaml}"
python scripts/build_subgraph_pool.py --config "$CONFIG"
