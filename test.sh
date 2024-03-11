#!/bin/bash
#
#SBATCH --partition=debug_5min
#SBATCH --cpus-per-task=16
#SBATCH --mem=1G
#SBATCH --output=outputs/hw3_%j_stdout.txt
#SBATCH --error=outputs/hw3_%j_stderr.txt
#SBATCH --time=00:05:00
#SBATCH --job-name=hw3
#SBATCH --mail-user=brandondmichaud@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs504319/cs5043-hw3

. /home/fagg/tf_setup.sh
conda activate dnn_2024_02

python hw3_base.py -v @exp.txt @net_shallow.txt --exp_index 0 --cpus_per_task $SLURM_CPUS_PER_TASK --precache datasets_by_fold_4_objects --dataset /scratch/fagg/core50
