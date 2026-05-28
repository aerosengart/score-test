#!/bin/bash

helpFunction()
{
   echo ""
   echo "Create anchor list."
   echo ""
   echo "Usage: $0 -a ANCHOR_LIST -d DATA_CONFIG -m MODEL_CONFIG -o OUTPUT_DIR"
   echo -e "\t-a Path to anchor list."
   echo -e "\t-d Path to data config .yaml."
   echo -e "\t-m Path to model config .yaml."
   echo -e "\t-o Output directory."
   exit 1 ## Exit script after printing help
}

while getopts "a:d:m:o:" opt
do
   case "$opt" in
      a ) anchor_list="$OPTARG" ;;
      d ) d_config="$OPTARG" ;;
      m ) m_config="$OPTARG" ;;
      o ) out_dir="$OPTARG" ;;
      ? ) helpFunction ;; ## Print helpFunction in case parameter is non-existent
   esac
done


## Print helpFunction in case parameters are empty
if [ -z "$anchor_list" ] || [ -z "$d_config" ] || [ -z "$m_config" ] || [ -z "$out_dir" ] 
then
   echo "Some or all of the parameters are empty.";
   helpFunction
fi

num_anchors="$(cat $anchor_list | wc -l)"
num_anchors_mod=$(( $num_anchors % 1000 ))

for (( i=1001; i<=5000; i+=1000 )); do
   start=$(($i % 1000))
   v=$(($i + 1000 - 1))
   offset=$(($i - 1))
   j=$(( $v < $num_anchors ? $v : $num_anchors ))
   end=$(( $j - $offset ))
   sbatch --array=$start-$end slurm_scripts/anchors_filt.sh $d_config $m_config $anchor_list $out_dir $offset

   echo "sbatch --array=$start-$end slurm_scripts/anchors_filt.sh $d_config $m_config $anchor_list $out_dir $offset"
done

# for (( i=1001; i<=$num_anchors; i+=1000 )); do
#    start=$(($i % 1000))
#    v=$(($i + 1000 - 1))
#    offset=$(($i - 1))
#    j=$(( $v < $num_anchors ? $v : $num_anchors ))
#    end=$(( $j - $offset ))
#    sbatch --array=$start-$end -o /scratch/arosenga/output/output.%a.out slurm_scripts/anchors_filt.sh $d_config $m_config $anchor_list $out_dir $offset
#    echo "sbatch --array=$start-$end -o /scratch/arosenga/output/output.%a.out slurm_scripts/anchors_filt.sh $d_config $m_config $anchor_list $out_dir $offset"
# done
