#!/bin/bash
#
#SBATCH --partition=normal
#SBATCH --cpus-per-task=10
#SBATCH --mem=10G
#SBATCH --output=results/hw3_%j_stdout.txt
#SBATCH --error=results/hw3_%j_stderr.txt
#SBATCH --time=01:00:00
#SBATCH --job-name=hw3
#SBATCH --mail-user=YOUR EMAIL ADDRESS
#SBATCH --mail-type=ALL
#SBATCH --chdir=YOUR DIRECTORY
#SBATCH --array=0

#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up


. /home/fagg/tf_setup.sh
conda activate dnn_2024_02
module load cuDNN/8.9.2.26-CUDA-12.2.0


python hw3_base.py -v @exp.txt @oscer.txt @net_shallow.txt --exp_index $SLURM_ARRAY_TASK_ID --cpus_per_task $SLURM_CPUS_PER_TASK --precache datasets_by_fold_4_objects --dataset /scratch/fagg/core50 
