#!/bin/bash

helpFunction()
{
   echo ""
   echo "Make Bowtie2 index from reference FASTA file(s)"
   echo ""
   echo "Usage: $0 -f REFERENCE -n BASE_NAME"
   echo -e "\t-f Reference FASTA."
   echo -e "\t-n Base name for output."
   exit 1 ## Exit script after printing help
}

while getopts "f:n:" opt
do
   case "$opt" in
      f ) ref_file="$OPTARG" ;;
      n ) base_name="$OPTARG" ;;
      ? ) helpFunction ;; ## Print helpFunction in case parameter is non-existent
   esac
done

## Print helpFunction in case parameters are empty
if [ -z "$ref_file" ] || [ -z "$base_name" ]
then
   echo "Some or all of the parameters are empty.";
   helpFunction
fi

## Run code
bowtie2-build "$ref_file" "$base_name"
