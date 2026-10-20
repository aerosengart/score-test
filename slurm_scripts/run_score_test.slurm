#!/bin/bash
#SBATCH --partition=Leviathans
#SBATCH --job-name=score-test   # create a short name for your job
#SBATCH --nodes=1               # node count
#SBATCH --ntasks=1              # total number of tasks across all nodes
#SBATCH --ntasks-per-node 1   
#SBATCH --cpus-per-task=2       # number of cores
#SBATCH --mem-per-cpu=30G       # memory per core
#SBATCH --time=10:00:00         # total run time limit (HH:MM:SS)
#SBATCH --nice=0              ## deprioritize to be nice to others
#SBATCH --exclude=leviathan3,leviathan6 ## have to exclude these because i do not have permission to write to /scratch on them

source /home/arosenga/miniconda/bin/activate /home/arosenga/miniconda/envs/score-test

## command that runs this file
#     sbatch --depend=afterok:$last_job_sub --array=1-$n_job_arr slurm_scripts/run_score_test.sh $d_config $m_config $anchor_list $out_dir $final_dir $i $n_anch_job $num_anchors 

## $1 - data configuration file
## $2 - model configuration file
## $3 - anchor_list.txt path
## $4 - scratch output directory
## $5 - final output directory
## $6 - starting index
## $7 - number of anchors per job
## $8 - number of anchors in anchor_list.txt

## starting index in anchor_list
start_idx=$(( (($SLURM_ARRAY_TASK_ID - 1) * $7) + $6 - 1 ))
## ending index in anchor_list
end_idx=$(( ($SLURM_ARRAY_TASK_ID * $7) + $6 - 1 )) 
## make sure we don't exceed total number of anchors in anchor_list
end_idx=$(( $end_idx < $8 ? $end_idx : $8 ))

## write job output to scratch
SCRATCH_DIR="$4/slurm-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID"
mkdir -p "$SCRATCH_DIR"

if [ $start_idx -lt $end_idx ]; then
    ## run score test
    python routines/score_test_job.py --d_config $1 --m_config $2 --anchor_list $3 --output $SCRATCH_DIR --start_idx $start_idx --end_idx $end_idx
fi

## copy output back to disk
FINAL_DIR="$5/score_test"
mkdir -p "$FINAL_DIR/failure"
mkdir -p "$FINAL_DIR/success"
mkdir -p "$FINAL_DIR/unstable"
JOB_DIR="$SCRATCH_DIR/score_test"
cp -r -a "$SCRATCH_DIR/score_test/failure/." "$FINAL_DIR/failure/"
cp -r -a "$SCRATCH_DIR/score_test/success/." "$FINAL_DIR/success/"
cp -r -a "$SCRATCH_DIR/score_test/unstable/." "$FINAL_DIR/unstable/"