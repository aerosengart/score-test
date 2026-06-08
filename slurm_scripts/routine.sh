#!/bin/bash

## $1 = number of iterations
## $2 = path to routine configuration in quotations
if [[ $1 -gt 1 ]]; then
    routines=$(sbatch --parsable --array=1-$1 slurm_scripts/run_routine.slurm $2)
else
    routines=$(sbatch --parsable --array=1 slurm_scripts/run_routine.slurm $2)
fi
## wait until job array is done
sbatch --depend=afterok:$routines slurm_scripts/plot_routine.slurm $2
