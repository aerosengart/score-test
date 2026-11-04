#!/bin/bash

helpFunction()
{
   echo ""
   echo "Submit slurm job arrays for running score test on SPLASH anchors."
   echo ""
   echo "Usage: $0 -o SCRATCH_OUTPUT_DIR -f FINAL_OUTPUT_DIR -j JOB_LIST_FILE"
   echo -e "\t-o Scratch output directory."
   echo -e "\t-f Final output directory."
   echo -e "\t-j Path to job list file."
   exit 1 ## Exit script after printing help
}

while getopts "a:d:m:o:f:n:j:" opt
do
   case "$opt" in
      o ) out_dir="$OPTARG" ;;
      f ) final_dir="$OPTARG" ;;
      j ) job_list="$OPTARG" ;;
      ? ) helpFunction ;; ## Print helpFunction in case parameter is non-existent
   esac
done


## Print helpFunction in case parameters are empty
if [ -z "$out_dir" ] || [ -z "$final_dir" ] || [ -z "$job_list" ]
then
   echo "Some or all of the parameters are empty.";
   helpFunction
fi


## specify node
for (( i=1; i<=6; i+=1 )); do
   leviat="leviathan$i"
   sbatch --nodelist=$leviat slurm_scripts/move_scratch_files.sh $out_dir $final_dir $job_list
done