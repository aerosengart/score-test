#!/bin/bash

helpFunction()
{
   echo ""
   echo "Create anchor list."
   echo ""
   echo "Usage: $0 -p PATH_BASE -o OUTPUT_DIRECTORY"
   echo -e "\t-p Path to anchor-specific S(A)TC files."
   echo -e "\t-o Output directory name."
   exit 1 ## Exit script after printing help
}

while getopts "p:o:" opt
do
   case "$opt" in
      p ) base_path="$OPTARG" ;;
      o ) out_dir="$OPTARG" ;;
      ? ) helpFunction ;; ## Print helpFunction in case parameter is non-existent
   esac
done

## Print helpFunction in case parameters are empty
if [ -z "$base_path" ] || [ -z "$out_dir" ]
then
   echo "Some or all of the parameters are empty.";
   helpFunction
fi

## Run code
## make new directory
mkdir -p "${out_dir}";

## get list of subdirectories to extract from
current=$(pwd);
cd "$base_path";
find -maxdepth 1 -iname '*.txt' -execdir basename -s '.txt' {} + > "${current}/${out_dir}/anchor_list.txt";
cd "$current";

