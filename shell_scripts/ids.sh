#!/bin/bash

helpFunction()
{
   echo ""
   echo "Fetch all BioProject BioSample IDs."
   echo ""
   echo "Usage: $0 -b BIOPROJECT_ID -o OUTPUT_DIRECTORY"
   echo -e "\t-b BioProject ID."
   echo -e "\t-o Output directory."
   exit 1 ## Exit script after printing help
}

while getopts "b:o:" opt
do
   case "$opt" in
      b ) bp_id="$OPTARG" ;;
      o ) out_dir="$OPTARG" ;;
      ? ) helpFunction ;; ## Print helpFunction in case parameter is non-existent
   esac
done

## Print helpFunction in case parameters are empty
if [ -z "$bp_id" ] || [ -z "$out_dir" ]
then
   echo "Some or all of the parameters are empty.";
   helpFunction
fi

## Run code
out_file="/ids.txt"
mkdir -p "$out_dir";
esearch -db sra -query "$bp_id" | efetch -format runinfo | grep -v "Run" | cut -d ',' -f 26,1,25 > "$out_dir$out_file"
