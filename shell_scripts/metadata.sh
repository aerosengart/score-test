#!/bin/bash

helpFunction()
{
   echo ""
   echo "Fetch Run Metadata."
   echo ""
   echo "Usage: $0 -l BIOSAMPLE_ID_LIST -o OUTPUT_DIRECTORY"
   echo -e "\t-l List of BioSample IDs."
   echo -e "\t-o Output directory."
   exit 1 ## Exit script after printing help
}

while getopts "l:o:" opt
do
   case "$opt" in
      l ) bs_id_list="$OPTARG" ;;
      o ) out_dir="$OPTARG" ;;
      ? ) helpFunction ;; ## Print helpFunction in case parameter is non-existent
   esac
done

## Print helpFunction in case parameters are empty
if [ -z "$bs_id_list" ] || [ -z "$out_dir" ]
then
   echo "Some or all of the parameters are empty.";
   helpFunction
fi

## Run code

#### Get metadata
out_file="/metadata_temp.tsv";
while read -u 9 line_full; 
do 
   line=$( echo "$line_full" | cut -d ',' -f 3); 
   titles=$(esearch -db biosample -query "$line" | esummary | xtract -pattern DocumentSummary -element Attribute@attribute_name); 
   echo -e "biosample_id \t sample_name \t sra_id \t $titles" >> "$out_dir$out_file"; ## write header
   biosample=$(esearch -db biosample -query $line | esummary | xtract -pattern DocumentSummary -element Identifiers | sed -e 's/BioSample:\(.*\); Sample.*/\1/' | sed -r 's/[^ \t]+[ \t]//'); 
   sra=$(esearch -db biosample -query $line | esummary | xtract -pattern DocumentSummary -element Identifiers | sed 's/^.*SRA:/SRA:/' | sed -r 's/[^ \t]+[ \t]//'); 
   sample=$(esearch -db biosample -query $line | esummary | xtract -pattern DocumentSummary -element Identifiers | sed -e 's/.*name:\(.*\); SRA.*/\1/' | sed -r 's/[^ \t]+[ \t]//');
   info=$(esearch -db biosample -query $line | esummary | xtract -pattern DocumentSummary -element Attribute); 
   echo -e "$biosample \t $sample \t $sra \t $info"; echo ""; 
done 9< "$bs_id_list" >> "$out_dir$out_file"; 