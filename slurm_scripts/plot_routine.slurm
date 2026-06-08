#!/bin/bash
#SBATCH --partition=Leviathans
#SBATCH --job-name=plot_routine        # create a short name for your job
#SBATCH --nodes=1                      # node count
#SBATCH --ntasks=1                     # total number of tasks across all nodes
#SBATCH --ntasks-per-node 1
#SBATCH --time=00:05:00          # total run time limit (HH:MM:SS)
#SBATCH --nice=200               ## deprioritize to be nice to others

source /home/arosenga/miniconda/bin/activate /home/arosenga/miniconda/envs/score-test

python routines/plot_routine.py --cfg $1 


