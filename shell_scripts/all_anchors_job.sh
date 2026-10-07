#!/bin/bash

helpFunction()
{
   echo ""
   echo "Submit slurm job arrays for running score test on SPLASH anchors."
   echo ""
   echo "Usage: $0 -a ANCHOR_LIST -d DATA_CONFIG -m MODEL_CONFIG -o OUTPUT_DIR -n ANCHORS_PER_JOB -j ANCHORS_PER_ARRAY"
   echo -e "\t-a Path to anchor list."
   echo -e "\t-d Path to data config .yaml."
   echo -e "\t-m Path to model config .yaml."
   echo -e "\t-o Output directory."
   echo -e "\t-n Number of anchors per slurm job."
   echo -e "\t-j Number of jobs per slurm array."
   exit 1 ## Exit script after printing help
}

while getopts "a:d:m:o:n:j:" opt
do
   case "$opt" in
      a ) anchor_list="$OPTARG" ;;
      d ) d_config="$OPTARG" ;;
      m ) m_config="$OPTARG" ;;
      o ) out_dir="$OPTARG" ;;
      n ) n_anch_job="$OPTARG" ;;
      j ) n_job_arr="$OPTARG" ;;
      ? ) helpFunction ;; ## Print helpFunction in case parameter is non-existent
   esac
done


## Print helpFunction in case parameters are empty
if [ -z "$anchor_list" ] || [ -z "$d_config" ] || [ -z "$m_config" ] || [ -z "$out_dir" ] || [ -z "$n_anch_job" ] || [ -z "$n_job_arr" ]
then
   echo "Some or all of the parameters are empty.";
   helpFunction
fi

## total number of anchors --- count lines in anchor_list.txt
num_anchors="$(cat $anchor_list | wc -l)"
## total anchors per array
n_anch_arr=$(( $n_anch_job*$n_job_arr ))

## submit first job array:
##    n_job_arr   - number of jobs per array
##    d_config    - path to data configuration .yaml
##    m_config    - path to model configuration .yaml
##    anchor_list - path to list of anchors (anchor_list.txt)
##    out_dir     - output directory
##    start_idx   - index of first anchor in list to run
##    n_anch_job  - number of anchors per slurm job
##    num_anchors - total number of anchors in anchor_list
last_job_sub=$(sbatch --parsable --array=1-$n_job_arr slurm_scripts/run_score_test.slurm $d_config $m_config $anchor_list $out_dir 1 $n_anch_job $num_anchors)

## index of last anchor in first job array is where we start the loop
end_idx=$(( $start_idx + $n_anch_arr + 1 ))

## loop over remaining anchors in anchor_list
for (( i=$end_idx; i<=$num_anchors; i+=$n_anch_arr )); do
   wait_job=$last_job_sub
   ## wait until previous job array is done
   last_job_sub=$(sbatch --parsable --dependency=afterany:$wait_job --array=1-$n_job_arr slurm_scripts/run_score_test.slurm $d_config $m_config $anchor_list $out_dir $i $n_anch_job $num_anchors) 
done

