#!/bin/bash

#SBATCH -p mit_normal_gpu,mit_preemptable
#SBATCH --requeue
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=canos_hyperparam_finetune_%j.out
#SBATCH --signal=USR1@90
#SBATCH --time=05:59:00

source ~/.bashrc
conda activate pfdelta2
exec python main.py --config canos_task_3_1_finetune/canos_task_3_1_finetune_2