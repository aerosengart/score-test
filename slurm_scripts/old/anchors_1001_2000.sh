#!/bin/bash
#SBATCH --partition=Leviathans
#SBATCH --job-name=small-anchor-test   # create a short name for your job
#SBATCH --nodes=1                      # node count
#SBATCH --ntasks=1                     # total number of tasks across all nodes
#SBATCH --ntasks-per-node 1
#SBATCH --time=08:00:00          # total run time limit (HH:MM:SS)
#SBATCH --array=1-1000
#SBATCH --nice=200               ## deprioritize to be nice to others

# module purge
# module load miniconda3
# conda init
# conda info --envs

source /home/arosenga/miniconda/bin/activate /home/arosenga/miniconda/envs/score-test

anchor_idx=$((SLURM_ARRAY_TASK_ID + 1000))
anchor=$(sed "${anchor_idx}q;d" $3)
python routines/splash_score_test.py -d $1 -m $2 -a $anchor -o $4