#!/bin/bash
#SBATCH --partition=Leviathans
#SBATCH --job-name=score-test   # create a short name for your job
#SBATCH --nodes=1               # node count
#SBATCH --ntasks=1              # total number of tasks across all nodes
#SBATCH --ntasks-per-node 1   
#SBATCH --cpus-per-task=2       # number of cores
#SBATCH --mem-per-cpu=24G       # memory per core
#SBATCH --time=01:00:00         # total run time limit (HH:MM:SS)
#SBATCH --nice=500              ## deprioritize to be nice to others

source /home/arosenga/miniconda/bin/activate /home/arosenga/miniconda/envs/score-test

## command that runs this file
#     sbatch --depend=afterok:$last_job_sub --array=1-$n_job_arr slurm_scripts/run_score_test.sh $d_config $m_config $anchor_list $out_dir $i $n_anch_job $num_anchors

## $1 - data configuration file
## $2 - model configuration file
## $3 - anchor_list.txt path
## $4 - output directory
## $5 - starting index
## $6 - number of anchors per job
## $7 - number of anchors in anchor_list.txt

## starting index in anchor_list
start_idx=$(( (($SLURM_ARRAY_TASK_ID - 1) * $6) + $5 - 1 ))
## ending index in anchor_list
end_idx=$(( ($SLURM_ARRAY_TASK_ID * $6) + $5 - 1 )) 
## make sure we don't exceed total number of anchors in anchor_list
end_idx=$(( $end_idx < $7 ? $end_idx : $7 ))

## run score test
python routines/score_test_job.py --d_config $1 --m_config $2 --anchor_list $3 --output $4 --start_idx $start_idx --end_idx $end_idx