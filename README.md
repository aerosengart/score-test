# Score Test for Variance Components in GLMMs
Code for conducting a score test of variance components in GLMMs. 

Below we describe the workflow for real data. We use the South Africa SARS-CoV-2 dataset from Viala et al. as an example. All commands were run from the current directory, so paths are relative to this location in the examples.

---

# Data Retrieval

## Retrieve all associated BioSample IDs
First, we retrieve the BioSample IDs that are associated with the BioProject. This writes `ids.txt` to the specified output directory. Each row in the file specifies a BioSample by SRR (the sequencing library for the run), SRS (the biological source material; i.e. the BioSample), and SAMN (BioSample).

Data is retrieved from the SRA at NCBI using the SRA Toolkit. The shell script just packages all of this nicely. 

#### Usage
```
bash [path to shell_scripts/ids.sh script] 
  -b [accession in quotes] 
  -o [output directory]
```

#### Example (Bal)
```
bash shell_scripts/ids.sh -b "PRJNA817806" -o "data/bal"
```

#### Example (Tryndyak)
```
bash shell_scripts/ids.sh -b "PRJNA865728" -o "data/tryndyak"
```


## Retrieve metadata for BioSamples
Next, we must retrieve the metadata associated with the BioSamples using Entrez Direct. 

This requires fetching the metadata for a single BioSample so that we can scrape the headers first. We manually create `headers.tsv` using the results returned by the first query.

We then run `python_scripts/metadata.py` from the command line to retrieve the desired metadata for all of the BioSamples. This creates `metadata.tsv` in the output directory, which contains rows corresponding to the samples in `ids.txt` and columns corresponding to the variables in `headers.tsv`.

NOTE: For European experiments, we have slightly different metadata formatting. We specify this in the Python script with the `--european` flag.

#### Usage
```
esearch -db biosample -query [sample SRS in quotes] | esummary
```
[_create headers.tsv_]
```
python [path to metadata.py] 
  --sh_dir [path to shell script directory] 
  --bsl [path to ids.txt] 
  --head [path to headers.tsv] 
  --output [output directory]
```

#### Example (Bal)
```
esearch -db biosample -query "SRS12336687" | esummary | xtract -pattern DocumentSummary -element Attribute@attribute_name
```
[_create headers.tsv_]
```
python python_scripts/metadata.py --sh_dir shell_scripts --bsl data/bal/ids.txt --head data/bal/headers.tsv --output data/bal
```

#### Example (Tryndyak)
There is actually a bunch of metadata that is not included in the NBCI SRA, so I've downloaded the spreadsheet manually from [the paper supplement](https://figshare.com/articles/dataset/Wastewater_dataset_on_the_SARS-CoV-2_sublineages_circulating_in_Central_Arkansas_USA_post-COVID-19_pandemic/28071389).



## Select BioSamples
At this stage, we filter the samples to make sure they are complete cases, etc. We use `python_scripts/filter.py`, which creates a list of the filtered sample SRR ids in the output directory as the file `input_samples.txt`. The metadata corresponding to the filtered samples is written to `metadata_filt.tsv`.

#### Usage
```
python [path to filter.py] 
  --meta [path to metadata.tsv] 
  --ids [path to ids.txt] 
  --output [output directory] 
  --min-date [min collection date in quotes] 
  --max-date [max collection date in quotes]
```

#### Example (Bal)
There are only 106 samples in this dataset, so we don't want to filter any out. I've made the time window large enough to contain all of the collection dates.
```
python python_scripts/filter_bal.py --meta data/bal/metadata.tsv --infec data/bal/infection_labels.csv --ids data/bal/ids.txt --output data/bal --min_date 2020-01-01 --max_date 2023-01-01
```

#### Example (Tryndyak)
There are 269 samples in this dataset with BioSample 'SAMN30124484' being duplicated. However, this experiment also includes quantification runs, and we only need the runs that correspond to the whole genome sequencing. We therefore use the `Sequence Read Run` column (`SRR`) in `metadata.tsv` to construct our sample list. There should be 199. 
```
python python_scripts/filter_tryndyak.py --meta data/tryndyak/metadata.tsv --ids data/tryndyak/ids.txt --output data/tryndyak
```


## Retrieve sequences
We then fetch the SRA accessions for the samples specified in `input_samples.txt` in FASTQ format. This requires specifying the path to the `bin` directory of your local installation of the SRA Toolkit as well as the path to the directory of python scripts (`fetch.sh` calls `python_scripts/fastq.py`, which itself uses the SRA Toolkit to dump the data as FASTQ). This creates a separate `/srr` in the output directory, which contains a sub-directory for each sample containing the sample .sra and .fastq file(s). 

`python_scripts/fastq.py` calls `fasterq_dump` in the SRA toolkit with the flag `--split-3`. This means that the data in the .sra file will be split if the reads are paired, which results in either 1 or 3 FASTQ files in each sample sub-directory. If there are 3 files, then that means that there was some sort of pairing going on with the reads (see [here](https://github.com/ncbi/sra-tools/wiki/HowTo:-fasterq-dump)). The file with the suffixes "_1.fastq" and "_2.fastq" contains the paired reads, and the file with just ".fastq" contains the unpaired reads. If there is only 1 file, then there was no pairing. 

We need to copy all of the FASTQ files from their subdirectories into the same directory for easy access by SPLASH, and we need a single FASTQ file for each sample. We do this with `shell_scripts/fastq_dir.sh`. This script creates `fastq/` in the output directory, which contains the _full_ FASTQ files for the samples, as well as `fastq/splash_input.txt`, which is the input text file for SPLASH. By "full", we mean that one member of each read pair as well as all of the unpaired reads for a given sample are in the file. In other words, we throw away the "_2.fastq" file if it exists.

#### Usage
```
bash [path to fetch.sh] 
  -b [path to sra/bin] 
  -p [path to directory of python scripts] 
  -f [path to input_samples.txt] 
  -o [output directory]

bash [path to fastq_dir.sh] 
  -p [base path to SRA subdirectories] 
  -o [output directory]
```

#### Example (Bal)
Saving to `/scratch` for space/speed. 
```
bash shell_scripts/fetch.sh -b ../ww-seq/sra/bin -p python_scripts -f data/bal/input_samples.txt -o /scratch/arosenga/bal
bash shell_scripts/fastq_dir.sh -p /scratch/arosenga/bal/srr -o fastq
```

#### Example (Tryndyak)
Saving to `/scratch` for space/speed. 
```
bash shell_scripts/fetch.sh -b ../ww-seq/sra/bin -p python_scripts -f data/tryndyak/input_samples.txt -o /scratch/arosenga/tryndyak
bash shell_scripts/fastq_dir.sh -p /scratch/arosenga/tryndyak/srr -o fastq
```


---

# SPLASH Analysis

## Run SPLASH
We now have all of the data needed for running SPLASH. One can run it with whatever parameters, but it must include the flags `--dump_Cjs` and `--dump_sample_anchor_target_count_bin` so that the output includes the embedding vectors and the SATC output to make the contingency tables. We also include the `--without_compactors` and `--with_pval_asymp_opt` so that we do not compute the compactors (new to SPLASH2) and do compute the asymptotic $p$-value. 

It's nice to just put the SPLASH run in a shell script for reproducability, but we will cover the contents here for completeness. The first command runs SPLASH with the desired flags. The last two commands copy the sample name/sample index mapping file and the results sub-directory to the output directory.  We run it by calling `bash [path to splash_run.sh]` from within the `fastq/` sub-directory. All of the output is written here, but we want it to be in the output directory, which is why we include the last two commands.

The results include `logs/` (logging of run), `result_Cjs/` (binarized files of $\mathbf{c}$ data), `result_satc/` (binarized files of SATC data), four .csv files of results where each row corresponds to an anchor-target pair, and `sample_name_to_id.mapping.txt` (mapping between BioSample IDs and the indexing used in SPLASH).

The shell script is finally moved to the output directory.

Note: There are many differences in the configuration, settings, and output of `SPLASH2` and `SPLASH`, and it is difficult for us to compare and contrast them exactly. We note some of the differences we notice below, and we provide our configuration in the `splash_run.sh` files for reproducibility, which mostly uses the defaults in `SPLASH2`.

- Default $k$-mer length is $31$ in `SPLASH2` and $27$ in `SPLASH`
- Default distance between anchor and target is $0$ in `SPLASH2` and inferred in `SPLASH`
- Default `anchor_mode` is unspecified in `SPLASH2` and `tile` in `SPLASH`, which also option to set the sliding window width and has a default of $5$
- Many options are offerred by `SPLASH2` to adjust $p$-value optimization and correction methods, data splitting, and what information is saved to disc. 


#### Usage
```
#!/bin/bash (shebang)

[path to splash] --dump-Cjs --dump_sample_anchor_target_count_bin \
    --without_compactors --with_pval_asymp_opt \
    --bin_path [path to splash top directory] \
    --tmp_dir [path to temporary directory for intermediate writing] \
    --logs_dir [path to logs directory for run documentation] \
    [path to input_samples.txt]

mv sample_name_to_id.mapping.txt [path to output directory]
mv result* [path to output directory]
mv [path to shell script] [path to output directory]
```

#### Example
```
#!/bin/bash

../../../splash/splash \
    --dump_Cjs \
    --dump_sample_anchor_target_count_bin \
    --without_compactors \
    --with_pval_asymp_opt \
    --max_pval_opt_for_Cjs 1 \
    --n_threads_stage_1 10 \
    --n_threads_stage_1_internal 2 \
    --n_threads_stage_2 16 \
    --bin_path ../../../splash \
    --tmp_dir ../../../analyses/viana/2026_05_29/tmp \
    --logs_dir ../../../analyses/viana/2026_05_29/logs \
    splash_input.txt

mv sample_name_to_id.mapping.txt ../../../analyses/viana/2026_05_29
mv result* ../../../analyses/viana/2026_05_29
mv ../../../analyses/viana/splash_run.sh ../../../analyses/viana/2026_05_29
```

#### Example (Bal)
First, move to the directory of FASTQ files.
`cd /scratch/arosenga/bal/fastq` or `cd /scratch/arosenga/bal/fastq2`
`bash /home/arosenga/score-test/analyses/bal/splash_run.sh`


#### Example (Tryndyak)
First, move to the directory of FASTQ files.
`cd /scratch/arosenga/tryndyak/fastq`
`bash /home/arosenga/score-test/analyses/tryndyak/splash_run.sh`





## Clean Metadata
To do the contingency table analysis, it is helpful for us to first organize our metadata (e.g. add in the SPLASH sample ids, convert covariates to factors, etc.). This is done in `clean_metadata.py`. This writes `clean_metadata.tsv` to the specified output directory and will be the reference for metadata used in subsequent analyses.

#### Usage
```
python [path to clean_metadata.py] 
  --meta [path to metadata_filt.tsv] 
  --id_key [path to sample_name_to_id.mapping.txt output by SPLASH] 
  --output [output directory]
```

#### Example (Bal)
```
python python_scripts/clean_metadata_bal.py --meta data/bal/metadata_filt.tsv --id_key analyses/bal/2026_08_24/sample_name_to_id.mapping.txt --output data/bal
```

#### Example (Jahn)
```
python python_scripts/clean_metadata_jahn.py 
  --meta data/jahn/metadata_filt.tsv 
  --id_key analyses/jahn/2026_08_19/sample_name_to_id.mapping.txt 
  --output data/jahn
```


---

# Contingency Table Analysis
## Create Anchor-Bin Key File
`anchor_bin_key.py` will go through all SATC bin files and create a key for the locations of each anchor.
The key will be saved in the specified output directory as `anchor_bin_key.csv`. If `--clean_up` is specified, the SATC dump files will be deleted. This can save space but at the expense of possibly having to recreate them later, which can be slow.

#### Usage
```
python [path to anchor_bin_key.py] 
  --sp_dir [path to splash bin files] 
  --bin_dir [path to satc sub-directory] 
  --output [output directory]
```

#### Example (Bal)
```
python python_scripts/anchor_bin_key.py --sp_dir splash --bin_dir analyses/bal/2026_08_24/result_satc --output analyses/bal/2026_08_24
```

#### Example (Jahn)
```
python python_scripts/anchor_bin_key.py 
  --sp_dir splash 
  --bin_dir analyses/jahn/2026_08_19/result_satc 
  --output analyses/jahn/2026_08_19
```


## Anchor Summary Statistics
`anchor_prop.py` will go through all of the SATC bin files and write relevant summary statistics about the anchor contingency tables to `anchor_summary.csv` and `anchor_filt_summary.csv`. The latter is a filtered version (see the Python file for specifics).

We can optionally pass the path to `clean_metadata.tsv` and specify a covariate in that file in call arguments. This will produce a third file `anchor_filt_by_{covar}.csv` which will have the summary statistics for anchors that pass the above conditions AND have at least two different unique values of the specified covariate found in the set of samples used for SPLASH so that we do not run into numerical issues down the line.

While the program iterates through the bin files above, it also separates out the data into anchor-specific S(A)TC files (without the A!). These are written to the path specified by `--scratch` (preferably some scratch directory that has fast read/write). There can be hundreds of thousands to millions of anchors, so it is good to quickly run the next step so that we only keep the files we will actually use for analysis.

#### Usage
```
python [path to anchor_prop.py] 
  --sp_dir [path to splash bin files] 
  --bin_dir [path to satc sub-directory] 
  --bin_key [path to anchor_bin_key.csv] 
  --meta [path to clean_metadata.tsv] 
  --covar [name of covariate] 
  --output [output directory] 
  --scratch [path to temporary directory for reading/writing files]
```

#### Example (Bal)
```
python python_scripts/anchor_prop_bal.py --sp_dir splash --bin_dir analyses/bal/2026_08_24/result_satc --bin_key analyses/bal/2026_08_24/anchor_bin_key.csv --meta data/bal/clean_metadata.tsv --covar primary --output analyses/bal/2026_08_24 --scratch /scratch/arosenga/bal
```

#### Example (Justen)
```
python python_scripts/anchor_prop_amman.py 
  --sp_dir splash 
  --bin_dir analyses/amman/2026_08_10/result_satc 
  --bin_key analyses/amman/2026_08_10/anchor_bin_key.csv 
  --meta data/amman/clean_metadata.tsv 
  --covar date 
  --output analyses/amman/2026_08_10 
  --scratch /scratch/arosenga/amman
```

#### Example (Jahn)
```python python_scripts/anchor_prop_jahn.py --sp_dir splash --bin_dir analyses/jahn/2026_08_19/result_satc --bin_key analyses/jahn/2026_08_19/anchor_bin_key.csv --meta data/jahn/clean_metadata.tsv --covar date --output analyses/jahn/2026_08_19 --scratch /scratch/arosenga/jahn```



## Anchor Filtering
In this step, we filter the anchor-specific S(A)TC files to just the ones present in `anchor_filt_summary.csv` and copy them to our home directory. If desired (set `--clean_up`), we delete the temporary directory created in the `--scratch` directory to save space. This also creates a list as `anchor_list.txt` in the output directory. It lists all of the anchors we want to test as separate lines.

#### Usage
```
python [path to satc_filt.py] 
  --satc_dir [path to directory of anchor-specific S(A)TC files] 
  --anchor_file [path to anchor_filt_summary.csv] 
  --output [output directory] 
  --clean_up
```

#### Example (Bal)
```
python python_scripts/satc_filt.py --satc_dir /scratch/arosenga/bal/anchor_satc --anchor_file analyses/bal/2026_08_24/anchor_filt_by_primary.csv --output analyses/bal/2026_08_24
```

This example resulted in 63,943 anchors in the filtered list.


#### Example (Jahn)
```
python python_scripts/satc_filt.py 
  --satc_dir /scratch/arosenga/jahn/anchor_satc 
  --anchor_file analyses/jahn/2026_08_19/anchor_filt_by_date.csv 
  --output analyses/jahn/2026_08_19
```

This example resulted in 22,334 anchors in the filtered list.


---

# Score Test Analysis

## Running Score Test
We then run the score test with the shell script `all_anchors_job.sh` that submits `j` slurm jobs that run `run_score_test.slurm` with different arguments. The `run_score_test.slurm` file calls `routines/splash_score_test.py`, which iteratively runs the score test on `n` number of anchors from the anchor list. The results for each anchor are output individually into sub-directories within a sub-directory that is created within the specified output directory called `score_test`.

Some anchors may run into problems, either with model fitting via `glmGamPoi` or with the score test statistic computations. If this happens, then the anchor sequence, the number of targets, the number of samples, and a short note about the error are saved as a `.csv` file to a sub-directory called `failure` within `score_test`. 

If the model fit happens to be a bit strange (i.e. very large absolute values for fixed effects estimates), then the model and test output is saved within `unstable` as a `.json`. If all goes well, then all of the output is saved in `success`. The output `.json` will have two sub-dictionaries: one corresponding to the score test results and the other corresponding to the data/model objects used. 

The score results will be under `test` and includes:

- `score_val`: score vector ($1$ or $-1$ if scaling)
- `scale_val`: Euclidean norm of the score vector (or $1$ if not scaling)
- `schur_inv`: inverse Schur complement of information matrix components corresponding to the random effects variance components
- `inf_val`: value of quadratic program (infimum search over cone)
- `one_side`: one-sided score test statistic value
- `one_side_p`: one-sided score test $p$-value (at $\alpha = 0.05$)
- `two_side`: two-sided score test statistic value
- `two_side_p`: two-sided score test $p$-value (at $\alpha = 0.05$)

The data/model information will be under `model` and includes:

- `n_samps`: number of biological samples (columns) in anchor table
- `n_targs`: number of targets (rows) in anchor table
- `q`: dimension of random effects/variance components
- `x_dims`: dimensions of `x_arr` (including dummy variables)
- `z_dims`: dimensions of `z_arr` (including dummy variables)
- `fx_effs_dims`: dimensions of `fx_effs_arr` (including dummy variable coefficients)
- `min_fx_effs`: minimum fixed effect coefficient estimate across targets (dimension should equal second dimension of `fx_effs_dims`)
- `max_fx_effs`: maximum fixed effect coefficient estimate across targets (dimension should equal second dimension of `fx_effs_dims`)
- `med_fx_effs`: median fixed effect coefficient estimate across targets (dimension should equal second dimension of `fx_effs_dims`)
- `min_disp`: minimum dispersion estimate across targets (dimension should equal $1$)
- `max_disp`: maximum dispersion estimate across targets (dimension should equal $1$)
- `med_disp`: median dispersion estimate across targets (dimension should equal $1$)

Because there are many slurm jobs, this step creates a lot of slurm output files. To delete the ones corresponding to successful runs (i.e. the empty output files), we can run the following:

`find . -type f -name '*.out' -size 0b -delete`


#### Usage
```
bash [path to all_anchors_test.sh] 
  -a  [path to anchor list .txt in quotes] 
  -d [path to data config .yaml in quotes] 
  -m [path to model config .yaml in quotes] 
  -o [output directory in quotes] 
  -n [number of anchors per slurm job] 
  -j [number of jobs per slurm array]
```

#### Example (Bal)
To test:
```
bash shell_scripts/all_anchors_job.sh -a "analyses/bal/2026_08_24/anchor_test.txt" -d "analyses/bal/2026_08_24/d_bal.yaml" -m "analyses/bal/2026_08_24/m_bal.yaml" -o "analyses/bal/2026_08_24/test/no_x" -n 5 -j 1
```

Keep `-j` low because there can be memory issues:
```
bash shell_scripts/all_anchors_job.sh -a "analyses/bal/2026_08_24/anchor_list.txt" -d "analyses/bal/2026_08_24/d_bal.yaml" -m "analyses/bal/2026_08_24/m_bal.yaml" -o "analyses/bal/2026_08_24/no_x" -n 200 -j 10
```

Problems with:
`/home/arosenga/score-test/slurm-47904_3.out`
`/home/arosenga/score-test/slurm-47904_4.out`
`/home/arosenga/score-test/slurm-47915_4.out`
`/home/arosenga/score-test/slurm-47915_4.out`
`/home/arosenga/score-test/slurm-47916_3.out`
`/home/arosenga/score-test/slurm-47916_4.out`
`/home/arosenga/score-test/slurm-47917_3.out`
`/home/arosenga/score-test/slurm-47917_4.out`
`/home/arosenga/score-test/slurm-47918_3.out`
`/home/arosenga/score-test/slurm-47918_4.out`
`/home/arosenga/score-test/slurm-47919_3.out`
`/home/arosenga/score-test/slurm-47919_4.out`
`/home/arosenga/score-test/slurm-47920_3.out`
`/home/arosenga/score-test/slurm-47920_4.out`
`/home/arosenga/score-test/slurm-47921_3.out`
`/home/arosenga/score-test/slurm-47921_4.out`
`/home/arosenga/score-test/slurm-47922_3.out`
`/home/arosenga/score-test/slurm-47922_4.out`
`/home/arosenga/score-test/slurm-47923_3.out`
`/home/arosenga/score-test/slurm-47923_4.out`
`/home/arosenga/score-test/slurm-47924_3.out`
`/home/arosenga/score-test/slurm-47924_4.out`
`/home/arosenga/score-test/slurm-47925_3.out`
`/home/arosenga/score-test/slurm-47925_4.out`
`/home/arosenga/score-test/slurm-47926_3.out`
`/home/arosenga/score-test/slurm-47926_4.out`
`/home/arosenga/score-test/slurm-47927_3.out`
`/home/arosenga/score-test/slurm-47927_4.out`
`/home/arosenga/score-test/slurm-47928_3.out`
`/home/arosenga/score-test/slurm-47928_4.out`
`/home/arosenga/score-test/slurm-47929_3.out`
`/home/arosenga/score-test/slurm-47929_4.out`
`/home/arosenga/score-test/slurm-47930_3.out`
`/home/arosenga/score-test/slurm-47930_4.out`
`/home/arosenga/score-test/slurm-47931_3.out`
`/home/arosenga/score-test/slurm-47931_4.out`

Re-ran but only third and fourth jobs of each array and added check to see if anchor is already tested (skip in this case):

```
## total number of anchors --- count lines in anchor_list.txt
num_anchors="$(cat $anchor_list | wc -l)"
## total anchors per array
n_anch_arr=$(( $n_anch_job*$n_job_arr ))

## submit first job array:
##    n_job_arr   - number of jobs per array
##    d_config    - path to data configuration .yaml
##    m_config    - path to model configuration .yaml
##    anchor_list - path to list of anchors (anchor_list.txt)
##    out_dir     - output directory
##    start_idx   - index of first anchor in list to run
##    n_anch_job  - number of anchors per slurm job
##    num_anchors - total number of anchors in anchor_list
last_job_sub=$(sbatch --parsable --array=3-4 slurm_scripts/run_score_test.slurm $d_config $m_config $anchor_list $out_dir 1 $n_anch_job $num_anchors)

## index of last anchor in first job array is where we start the loop
end_idx=$(( $start_idx + $n_anch_arr + 1 ))

## loop over remaining anchors in anchor_list
for (( i=$end_idx; i<=$num_anchors; i+=$n_anch_arr )); do
   wait_job=$last_job_sub
   ## problems with bal --- run only 3-4
   last_job_sub=$(sbatch --parsable --dependency=afterany:$wait_job --array=3-4 slurm_scripts/run_score_test.slurm $d_config $m_config $anchor_list $out_dir $i $n_anch_job $num_anchors) 
done
```



#### Example (Justen)
```
bash shell_scripts/all_anchors_job.sh 
  -a "analyses/jahn/2026_08_19/anchor_list.txt" 
  -d "analyses/jahn/2026_08_19/d_jahn.yaml" 
  -m "analyses/jahn/2026_08_19/m_jahn.yaml"
  -o "analyses/jahn/2026_08_19" 
  -n 200 
  -j 50
```


## Summarizing Results
We summarize the anchor-specific .json output with `json_summary.py`. This reads each of the .json files in the passed directory and writes the pertinent information to a `anchor_score_test.csv` in the provided output directory. This will also creates a file `anchor_score_test_signf.csv`, which only includes rows for anchors that are statistically significant at a level of $\alpha = 0.05$ based on the one-sided or two-sided score test $p$-value after Benjamini-Yekulieli correction. The columns are also extended to include the SPLASH output (from `result.after_correction.all_anchors.tsv`) so that we can look at the most frequent target and other summary statistics about the significant anchor tables. 

#### Usage
```
python [path to json_summary.py] 
  --json_dir [path to anchor .json files] 
  --splash_out_dir [path to splash output files] 
  --out_dir [output directory]
```

#### Example (Bal)
```
python python_scripts/json_summary.py --json_dir /home/arosenga/score-test/analyses/bal/2026_08_24/no_x/score_test/success --splash_out_dir /home/arosenga/score-test/analyses/bal/2026_08_24 --out_dir /home/arosenga/score-test/analyses/bal/2026_08_24/no_x/score_test
```

#### Example (Amman)
```
python python_scripts/json_summary.py 
  --json_dir /home/arosenga/score-test/analyses/amman/2026_08_10/score_test/anchor_json 
  --splash_out_dir /home/arosenga/score-test/analyses/amman/2026_08_10 
  --out_dir /home/arosenga/score-test/analyses/amman/2026_08_10/score_test
  ```

---

# Biological Analysis
This varies from dataset to dataset.

## Bal (France SARS-CoV-2 clinical samples)
Chaung et al. (2022) investigate which anchors/targets that SPLASH and OASIS deem statistically significant are "strain-defining" . This is done by aligning the $k$-mers to custom reference sequences. We emulate their process here.

### Reference Index
Following the steps explained in the supplement of their paper, we obtain archetype genomes for each of the three variants (Delta, Omicron BA.1, and Omicron BA.2) from their repository [here](https://github.com/salzman-lab/SPLASH/blob/main/cell_data/Table_S1/archetype%20genomes.tsv).

These are then put into individual `.fasta` files. We download the original reference genome, the [Wuhan-Hu-1 (NC_045512.2)](https://www.ncbi.nlm.nih.gov/nuccore/NC_045512.2?report=fasta) isolate, to create the references for constructing the Bowtie index via `bowtie-build`. We run the following from the current directory (i.e. the directory of this README). 

```
bowtie/bowtie-build -r -f data/bal/wuhan_hu.fasta,data/bal/delta.fasta,data/bal/omicron_ba1.fasta,data/bal/omicron_ba2.fasta data/bal/sc2_archs
```

This constructs four files: `sc2_archs.[1-2].ebwt` and `sc2_archs.rev.[1-2].ebwt`. These are the files needed by Bowtie to perform all of the alignment steps.

Note: We use the `-r` option because we will not be performing any paired-end alignments, so we don't need the two additional reference files that could be generated for the Bowtie index.


### Filtering to Candidate Anchors/Targets
The next step is to identify the anchors that are "strain-defining" by aligning a subset of "called" anchor $k$-mers to this index. This requires having access to the SATC tables for all of the anchors identified by SPLASH. We construct these with:

```
python python_scripts/make_satc.py --sp_dir /home/arosenga/score-test/splash --bin_dir /home/arosenga/score-test/analyses/bal/2026_08_24/result_satc --scratch /scratch/arosenga/bal/make_satc --output /scratch/arosenga/bal/satc_files
```

To construct the list of "called" anchors/targets, Chaung et al. follow the following steps (copied from their supplement and reformatted here), and we do the same:

1. We filter for anchors that are significant (with a BY corrected p value less than 0.05) and have large effect size (>0.5), yielding a list of $N$ SPLASH-called anchors. 
2. Control anchors are generated by taking the N anchors with the highest counts.
3.  For each of these anchors, we first filter out all anchors with fewer than 30 counts, only 1 unique target, or only 1 unique sample, and filter out all samples with 5 or fewer counts. 
4. Then, we discard all targets that constitute less than 5% of the remaining counts for that anchor.

Because the score test method does not compute effect sizes, we follow the same procedure but omit the effect size filtering step for these data. This procedure is implemented in `python_scripts/signf_anchors_bal.py`, which requires a configuration file:

```
d_config: '/home/arosenga/score-test/analyses/bal/2026_08_24/d_bal_filt.yaml' ## need to access scratch i guess
bin_key: '/home/arosenga/score-test/analyses/bal/2026_08_24/anchor_bin_key.csv'
satc_dump: '/home/arosenga/score-test/analyses/bal/2026_08_24/satc_dump'
splash_file: '/home/arosenga/score-test/analyses/bal/2026_08_24/result.after_correction.all_anchors.tsv'
score_file: '/home/arosenga/score-test/analyses/bal/2026_08_24/no_x/score_test/anchor_score_test_summary.csv'
alpha: 0.05
output: '/home/arosenga/score-test/analyses/bal/2026_08_24/no_x'
```

This step creates four files in the specified output directory: `splash_calls.csv`, `score_calls.csv`, `splash_calls_filt.csv`, and `score_calls_filt.csv`. The first two files contains corresponding to statistically significant anchors (according to the specified hypothesis test) that have total counts greater than or equal to $30$, high effect sizes (for SPLASH), and more than one target and sample in the table. The second two files have rows corresponding to anchor/target pairs that satisfy the other criteria discussed above.

### Alignment
All alignment-related steps are implemented in `python_scripts/align_kmers.py`. First, we need to create comma separated lists of the $k$-mers we want to align to our index. This scripts creates four `.csv` files. Each corresponds to the list of unique anchors or targets in `splash_calls_filt.csv` or `score_calls_filt.csv`. They are saved in the output directory passed as an argument (under `--output`) and named as `[splash or score]_calls_[anchs or targs].csv`. 

Note: The lists are unique within each hypothesis testing framework. There may be overlap between the anchors and/or targets called by SPLASH and our score test method. They will be aligned multiple times.

In the first step of this section, we created a reference index with the basename `sars_cov_2_archs`, which is what we want to align these four lists of $k$-mers to. As in Chaung et al., we use Bowtie with the settings `-a -v 0`. The command is called with `subprocess` within the Python script, but it is informative to cover the settings here as well. 

```
bowtie/bowtie -r -a -v 0 -x sc2_archs analyses/bal/2026_08_24/no_x/splash_calls_anchs.csv analyses/bal/2026_08_24/no_x/splash_calls_anchs_align.txt
```

- `-r`: Indicates that the comma-separated file (in `analyses/bal/2026_08_24/no_x/[calls file]`) for alignment contain raw sequences (i.e. not FASTA files).
- `-a`: Indicates that all valid alignments per sequence should be reported (subject to the constraints imposed by `-v`).
- `-v 0`: Indicates that alignments should be perfect (i.e. have at most $0$ mismatches). $0$ can instead be a non-negative integer.
- `-x sc2_archs`: Specifies the Bowtie index basename. The Bowtie documentation states that first the current directory is searched, then the index sub-directory within the directory containing the Bowtie executables, then the directory specified in the `BOWTIE_INDEXES` environment variable. Thus, we need to add `data/bal` as an environment variable with `conda env config vars set BOWTIE_INDEXES="data/bal"` first.

This creates four additional files in the output directory which correspond to the aligned sequences in each of the `[splash or score]_calls_[anchs or targs].csv` files created earlier. The files are named `[splash or score]_[anchs or targs].txt`, and each line of these files corresponds to an alignment. The contents of each file is tab-separated, and are:

- Read index in the original file
- Reference strand aligned to, `+` for forward strand, `-` for reverse
- Name (if it exists) of the read that was aligned
- Name of the reference it aligned to
- Offset (zero-indexed) into the forward reference strand where leftmost character of the alignment occurs
- Read sequence
- ASCII-encoded read qualities (all `I` because we align raw sequences)
- Number of other instances where the same sequence aligned against the same reference characters as were aligned against in the reported alignment. In other words, the number of other reference sequences the read aligned to.
- Comma-separated list of mismatch descriptors (empty in this case because `-v 0`)

To run this step, we do:

```
python python_scripts/align_kmers.py --splash_calls analyses/bal/2026_08_24/no_x/splash_calls_filt.csv --score_calls analyses/bal/2026_08_24/no_x/score_calls_filt.csv --output analyses/bal/2026_08_24/no_x/align --bowtie bowtie --index sc2_archs
```

Chaung et al. state that they consider anchors that align perfectly to at least one of the reference assemblies (Wuhan-Hu-1, Delta, Omicron BA.1, or Omicron BA.2) and also have at least one target that aligns perfectly with a reference assembly. For each of these anchors, they deem it "strain-defining" if, for any of the reference assemblies, it has at least one target that maps to it and one target that does not.

Since we ran the alignment with `-v 0` (mismatches capped at $0$), any hit returned by Bowtie is a perfect alignment. We then filter the results of the Bowtie alignment according to the steps laid out by Chaung et al. The results are saved as two different files in the output directory: `score_strain_def.tsv` and `splash_strain_def.tsv`. These files contain the strain-defining anchors/targets and 0/1-encoded information about which reference genomes they aligned to.

### Analyzing Results
Although we did our best to replicate the settings in Chaung et al., we did not reproduce their results. This is likely due to the differences in `SPLASH2` and `SPLASH` or user error. We nevertheless continue with a similar analysis to the original publication.

After [the previous step](#alignment), we have two lists: one containing the "strain-defining" anchor/target pairs called by SPLASH (`splash_strain_def.tsv`) and one containing the pairs called by our score test method (`score_strain_def.tsv`). The next step is to rank these pairs in some way. We choose to use a combination of $k$-mer abundance and BY-correct $p$-values/$p$-value bounds to do this ordering. This information (target abundance, $p$-values, etc.) is in the `splash_calls.csv` and `score_calls.csv` files that were output in [the filtering step](#filtering-to-candidate-anchorstargets) performed before alignment. 

The file `python_scripts/align_summary.py` (inner) joins the strain-defining `.tsv` files with the corresponding `.csv` files listing the anchors called as significant by SPLASH/the score test. It saves the merged dataframes to two separate files in the `align_dir` directory: `score_alignment_results.tsv` and `splash_alignment_results.tsv`. Because the $p$-values/$p$-value bounds are for each anchor table, the values in these columns will be the same across all rows corresponding to the same anchor. The same goes for any other anchor-specific information (`M`, `anch_uniqTargs`, etc.).













