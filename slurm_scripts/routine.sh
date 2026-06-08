#!/bin/bash
#SBATCH --partition=Leviathans
#SBATCH --job-name=simul_routine       # create a short name for your job
#SBATCH --nodes=1                      # node count
#SBATCH --ntasks=1                     # total number of tasks across all nodes
#SBATCH --ntasks-per-node 1
#SBATCH --time=02:00:00          # total run time limit (HH:MM:SS)
#SBATCH --nice=200               ## deprioritize to be nice to others

source /home/arosenga/miniconda/bin/activate /home/arosenga/miniconda/envs/score-test


## don't pass other arguments if none passed
if [ "$#" -ne 5 ]; then
    python routines/run_routine.py -d $1 -m $2 -r $3 -n $4 -i $SLURM_ARRAY_TASK_ID
else
    python routines/run_routine.py -d $1 -m $2 -r $3 -n $4 -i $SLURM_ARRAY_TASK_ID -f $5
fi




