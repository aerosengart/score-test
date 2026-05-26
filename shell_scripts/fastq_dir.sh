#!/bin/bash

helpFunction()
{
   echo ""
   echo "Move FASTQ files from individual accession subdirectories to same subdirectory."
   echo ""
   echo "Usage: $0 -p PATH_BASE -o OUTPUT_DIRECTORY"
   echo -e "\t-p Base path to SRA subdirectories."
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
## make new FASTQ directory
output_dir="${base_path}/../${out_dir}";
mkdir -p "$output_dir";

## make SRR directory
# mkdir -p "${base_path}/srr";

## get list of subdirectories to extract from
current=$(pwd);
cd "$base_path";
find -maxdepth 1 -name "*SRR*" -type d | sed -e 's/[\.\/]//g' > temp.txt;
cd "$current";

while read -u 9 line; 
## ignore _2 (mate pairs)
do 
   cat "${base_path}/${line}/${line}_1.fastq" >> "${output_dir}/${line}_full.fastq";
   cat "${base_path}/${line}/${line}.fastq" >> "${output_dir}/${line}_full.fastq";
   #mv "${base_path}/${line}/"*".sra" "${base_path}/srr/${line}.sra";
   echo -e "${line} \t ${line}_full.fastq";
done 9< "${base_path}/temp.txt"  > "${output_dir}/splash_input.txt";
# rm -r "${base_path}/SRR*";
rm "${base_path}/temp.txt";
