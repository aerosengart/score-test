#!/bin/bash
#SBATCH --partition=Leviathans
#SBATCH --job-name=move-results ## create a short name for your job
#SBATCH --time=00:10:00         ## total run time limit (HH:MM:SS)
#SBATCH --nice=0                ## deprioritize to be nice to others

## $1 - scratch output directory
## $2 - final output directory
## $3 - job list
## write job output to scratch

## make sure final output directories exist
FINAL_DIR="$2/score_test"
mkdir -p "$FINAL_DIR/failure"
mkdir -p "$FINAL_DIR/success"
mkdir -p "$FINAL_DIR/unstable"

## loop over files
while IFS="" read -r p || [ -n "$p" ]
do
    SCRATCH_DIR="$1/slurm-$p"
    if [ -d "$SCRATCH_DIR" ]; then
        echo "$SCRATCH_DIR exists!"
        cp -a "$SCRATCH_DIR/score_test/failure/"* $FINAL_DIR/failure/
        cp -a "$SCRATCH_DIR/score_test/success/"* $FINAL_DIR/success/
        cp -a "$SCRATCH_DIR/score_test/unstable/"* $FINAL_DIR/unstable/
    fi
done < $3

