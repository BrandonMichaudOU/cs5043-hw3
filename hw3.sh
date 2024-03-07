#!/bin/bash
#
#SBATCH --gres=gpu:1
#SBATCH --partition=disc_dual_a100_students
#SBATCH --cpus-per-task=64
#SBATCH --mem=30G
#SBATCH --output=outputs/hw3_%j_stdout.txt
#SBATCH --error=outputs/hw3_%j_stderr.txt
#SBATCH --time=01:00:00
#SBATCH --job-name=hw3
#SBATCH --mail-user=brandondmichaud@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs504319/cs5043-hw3
#SBATCH --array=0-4

. /home/fagg/tf_setup.sh
conda activate dnn_2024_02
module load cuDNN/8.9.2.26-CUDA-12.2.0


python hw3_base.py -v @exp.txt @oscer.txt @net_shallow.txt --exp_index $SLURM_ARRAY_TASK_ID --cpus_per_task $SLURM_CPUS_PER_TASK --precache datasets_by_fold_4_objects --dataset /scratch/fagg/core50
