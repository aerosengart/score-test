#!/bin/bash

helpFunction()
{
   echo ""
   echo "Fetch SRA data by SRA IDs."
   echo ""
   echo "Usage: $0 -b PREFETCH_PATH -p PYTHON_PATH -f OPTION_FILE -o OUTPUT_DIRECTORY"
   echo -e "\t-b Path to SRA Toolkit binaries. No trailing /."
   echo -e "\t-p Path to fastq python script. No trailing /."
   echo -e "\t-f File containing SRA IDs to fetch."
   echo -e "\t-o Output directory."
   exit 1 ## Exit script after printing help
}

while getopts "b:p:f:o:" opt
do
   case "$opt" in
   	b ) pf_path="$OPTARG" ;;
      p ) py_path="$OPTARG" ;;
      f ) opt_file="$OPTARG" ;;
      o ) out_dir="$OPTARG" ;;
      ? ) helpFunction ;; ## Print helpFunction in case parameter is non-existent
   esac
done

## Print helpFunction in case parameters are empty
if [ -z "$py_path" ] || [ -z "$pf_path" ] || [ -z "$opt_file" ] || [ -z "$out_dir" ]
then
   echo "Some or all of the parameters are empty.";
   helpFunction
fi

## Run code
## fetch SRA accessions
prefetch="/prefetch"
"$pf_path$prefetch" --option-file "$opt_file" --output-directory "$out_dir/srr" 

## extract as FASTQ
fastqp="/fastq.py"
python "$py_path$fastqp" --sra_dir "$out_dir/srr" --sra_bin "$pf_path"
