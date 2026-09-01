#!/bin/bash

helpFunction()
{
   echo ""
   echo "Create anchor list."
   echo ""
   echo "Usage: $0 -p PATH_BASE -o OUTPUT_DIRECTORY"
   echo -e "\t-p Path to anchor list."
   echo -e "\t-j Path to anchor-specific json files."
   exit 1 ## Exit script after printing help
}

while getopts "p:j:" opt
do
   case "$opt" in
      p ) list_path="$OPTARG" ;;
      j ) json_path="$OPTARG" ;;
      ? ) helpFunction ;; ## Print helpFunction in case parameter is non-existent
   esac
done

## Print helpFunction in case parameters are empty
if [ -z "$list_path" ] || [ -z "$json_path" ]
then
   echo "Some or all of the parameters are empty.";
   helpFunction
fi

## change to directory of json files
current=$(pwd);
cd "$json_path";

## counter
i=0
## reach anchor file
while read p; do
    echo "${i}"
    if find -maxdepth 1 -type f -name "${p}*" -print -quit | grep -q .; then
        file_name=$(find -maxdepth 1 -type f -name "${p}*")
        new_name="$json_path/${i}_${p}.json"
        mv $file_name $new_name
    fi
    i=$(( i + 1 ))
done < $list_path

cd "$current";

