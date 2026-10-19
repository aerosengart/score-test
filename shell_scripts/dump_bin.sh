#!/bin/bash

helpFunction()
{
   echo ""
   echo "Dump bin as text."
   echo ""
   echo "Usage: $0 -s SPLASH_DIRECTORY -b BIN_PATH -o OUTPUT_DIRECTORY"
   echo -e "\t-s Path to SPLASH binaries."
   echo -e "\t-b Path to bin.satc files."
   echo -e "\t-o Output directory name."
   exit 1 ## Exit script after printing help
}

while getopts "s:b:o:" opt
do
   case "$opt" in
      s ) splash_path="$OPTARG" ;;
      b ) bin_path="$OPTARG" ;;
      o ) out_dir="$OPTARG" ;;
      ? ) helpFunction ;; ## Print helpFunction in case parameter is non-existent
   esac
done

## Print helpFunction in case parameters are empty
if [ -z "$splash_path" ] || [ -z "$bin_path" ] || [ -z "$out_dir" ]
then
   echo "Some or all of the parameters are empty.";
   helpFunction
fi

## Run code
## make new directory
mkdir -p "${out_dir}";


files=$(find "$bin_path" -maxdepth 1 -type f -name "*satc")

for file in $files; do
   bn=$(basename "$file")
   on="$out_dir/$bn"
   dump_cmd=( "$splash_path/satc_dump" "$file" "$on" )
   "${dump_cmd[@]}"
done

# for file in "$bin_path/*satc"; do
#    base_name=$(basename "$file")
#    out_name="$out_dir/$base_name"
#    echo $out_name
#    #dump_cmd = ("$splash_path" "$file" "$out_dir")
#       # ./"$splash_path"
# done


