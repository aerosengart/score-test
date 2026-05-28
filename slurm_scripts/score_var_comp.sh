#!/bin/bash
#SBATCH --partition=Leviathans
#SBATCH --job-name=score_var_comp   ## create a short name for your job
#SBATCH --nodes=1                   ## node count
#SBATCH --ntasks=1                  ## total number of tasks across all nodes
#SBATCH --ntasks-per-node 1
#SBATCH --time=01:00:00          ## total run time limit (HH:MM:SS)
#SBATCH --nice=500               ## deprioritize to be nice to others

source /home/arosenga/miniconda/bin/activate /home/arosenga/miniconda/envs/score-test

## have to add because mod 1000
anchor_idx=$(($SLURM_ARRAY_TASK_ID + $5))
anchor=$(sed "${anchor_idx}q;d" $3)



python routines/run_routine.py -d $1 -m $2 -a $anchor -o $4