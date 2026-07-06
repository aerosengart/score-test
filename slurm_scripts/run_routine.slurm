#!/bin/bash
#SBATCH --partition=Leviathans
#SBATCH --job-name=simul_routine       # create a short name for your job
#SBATCH --nodes=1                      # node count
#SBATCH --ntasks=1                     # total number of tasks across all nodes
#SBATCH --ntasks-per-node 1
#SBATCH --time=40:00:00          # total run time limit (HH:MM:SS)
#SBATCH --nice=500               ## deprioritize to be nice to others

source /home/arosenga/miniconda/bin/activate /home/arosenga/miniconda/envs/score-test



## don't pass other arguments if none passed
if [ "$#" -ne 2 ]; then
    python routines/run_routine.py --cfg $1 --iter_id $SLURM_ARRAY_TASK_ID
else
    python routines/run_routine.py --cfg $1 --iter_id $SLURM_ARRAY_TASK_ID --flag $2
fi




