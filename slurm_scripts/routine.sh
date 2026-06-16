#!/bin/bash

## $1/$2 = number of iterations
## $2/$3 = path to routine configuration in quotations

if [[ "$#" -eq 3 ]]; then
    echo "3 args" 
    ## 3 arguments (first iter id, second iter id, path)
    routines=$(sbatch --parsable --array=$1-$2 slurm_scripts/run_routine.slurm $3)
    ## wait until job array is done
    sbatch --depend=afterok:$routines slurm_scripts/plot_routine.slurm $3
elif [[ "$#" -eq 2 ]]; then
    echo "2 args"
    ## 2 arguments (last iter id, path)
    if [[ $1 -ge 1 ]]; then
        routines=$(sbatch --parsable --array=1-$1 slurm_scripts/run_routine.slurm $2)
    else
        routines=$(sbatch --parsable --array=1 slurm_scripts/run_routine.slurm $2)
    fi
    ## wait until job array is done
    sbatch --depend=afterok:$routines slurm_scripts/plot_routine.slurm $2
else
    echo "Invalid number of arguments."
fi


