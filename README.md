## Score Test for Variance Components in GLMMs
Code for conducting a score test of variance components in GLMMs. 

---

Currently, only negative binomial data and a negative binomial model are supported.

## Data Class

### Simulated Data
Simulated data are organized in the `simul_data` class. Instantiation is done via a configuration dictionary passed as a .yaml file. Here is an example:

```
outcome: 'neg_bin'
n_samps: 100
n_targs: 200
fx_slopes: 0
rd_slopes: 1
fx_effs:
  mu: 0.0
  sigma: 1.0
x:
  mu: 0.5
  sigma: 1.0
z:
  mu: 0.0
  sigma: 1.0
tau_sqr: [0.0]
disp_param: 1.0   
```

The actual data that is drawn will depend upon what `outcome` is set to. Currently, only `neg_bin` is supported (for negative binomial samples).

### Real Data
Real data is also supported, but the configuration file has a different structure because we need to specify the paths to the files containing the necessary metadata.

```
outcome: 'splash'
satc_dir: '/home/arosenga/score-test/analyses/viana/2026_05_19/anchor_satc'   ## path to anchor-specific SATC files
splash_dir: /home/arosenga/score-test/splash'                                 ## path to splash executables
metadata_file: '/home/arosenga/score-test/data/viana/clean_metadata.tsv'      ## path to clean_metadata.tsv
temp_dir: 'viana_temp'                                                        ## path to temporary directory for writing intermediate files
fx_covars: []                                                                 ## list of target-specific fixed effect covariates
rd_covars: ['date']                                                           ## list of random effect covariates
fx_slopes: 0                                                                  ## number of fixed slopes 
rd_slopes: 1                                                                  ## number of random slopes
clean_up: False                                                               ## whether to remove temporary files after instantiating
```


## Model Class
The Model class is used to organize the estimated parameters as well as the data for conducting the score test. Instantiation is also done via a configuration .yaml. An example is:

```
outcome:     'neg_bin'               ## outcome dsitribution as string; either 'neg_bin' or 'quasi_poi'
spec:        'fx_by_k_int_rd_slope'  ## model linear predictor construction; either 'fx_int_rd_int' or 'fx_by_k_int_rd_slope'
use_true:    False                   ## use true values of nuisance parameters?
scale:       True                    ## scale score to have unit norm (and scale information matrix as well)
comb:        False                   ## compute combined test of variance component(s) and fixed effect(s)?
debug:      'off'                    ## setting for recording data/model fits for debugging; one of 'off', 'error', 'full'

## distribution-specific parameters
share_gamma: True                    ## OPTIONAL: force negative binomial dispersion parameter to be shared across clusters?
```




---

Below we describe the workflow for real data. We use the South Africa SARS-CoV-2 dataset from Viala et al. as an example. All commands were run from the current directory, so paths are relative to this location in the examples.

## Data Retrieval

### Retrieve all associated BioSample IDs
First, we retrieve the BioSample IDs that are associated with the BioProject. This writes `ids.txt` to the specified output directory. Each row in the file specifies a BioSample by SRR (the sequencing library for the run), SRS (the biological source material; i.e. the BioSample), and SAMN (BioSample).

Data is retrieved from the SRA at NCBI using the SRA Toolkit. The shell script just packages all of this nicely. 

#### Usage
`bash [path to shell/ids.sh script] -b [accession in quotes] -o [output directory]`

#### Example
`bash shell/ids.sh -b "PRJNA784038" -o "data/viala"`


### Retrieve metadata for BioSamples
Next, we must retrieve the metadata associated with the BioSamples using Entrez Direct. 

This requires fetching the metadata for a single BioSample so that we can scrape the headers first. We manually create `headers.tsv` using the results returned by the first query.

We then run `python/metadata.py` from the command line to retrieve the desired metadata for all of the BioSamples. This creates `metadata.tsv` in the output directory, which contains rows corresponding to the samples in `ids.txt` and columns corresponding to the variables in `headers.tsv`.

#### Usage
`esearch -db biosample -query [sample SRS in quotes] | esummary`

[_create headers.tsv_]

`python [path to metadata.py] --sh_dir [path to shell script directory] --bsl [path to ids.txt] --head [path to headers.tsv] --output [output directory]`

#### Example
`esearch -db biosample -query "SRS11294961" | esummary`

`python python/metadata.py --sh_dir shell --bsl data/viana/ids.txt --head data/viana/headers.tsv --output data/viana`


### Select BioSamples
At this stage, we filter the samples to make sure they are complete cases, etc. We use `python/filter.py`, which creates a list of the filtered sample SRR ids in the output directory as the file `input_samples.txt`. The metadata corresponding to the filtered samples is written to `metadata_filt.tsv`.

#### Usage
`python [path to filter.py] --meta [path to metadata.tsv] --ids [path to ids.txt] --output [output directory] --min-date [min collection date in quotes] -- max-date [max collection date in quotes]`

#### Example
`python python/filter.py --meta data/viana/metadata.tsv --ids data/viana/ids.txt --output data/viana --min_date 2021-12-01 --max_date 2021-12-05`


### Retrieve sequences
We then fetch the SRA accessions for the samples specified in `input_samples.txt` in FASTQ format. This requires specifying the path to the `bin` directory of your local installation of the SRA Toolkit as well as the path to the directory of python scripts (`fetch.sh` calls `python/fastq.py`, which itself uses the SRA Toolkit to dump the data as FASTQ). This creates a separate `/srr` in the output directory, which contains a sub-directory for each sample containing the sample .sra and .fastq file(s). 

`python/fastq.py` calls `fasterq_dump` in the SRA toolkit with the flag `--split-3`. This means that the data in the .sra file will be split if the reads are paired, which results in either 1 or 3 FASTQ files in each sample sub-directory. If there are 3 files, then that means that there was some sort of pairing going on with the reads (see [here](https://github.com/ncbi/sra-tools/wiki/HowTo:-fasterq-dump)). The file with the suffixes "_1.fastq" and "_2.fastq" contains the paired reads, and the file with just ".fastq" contains the unpaired reads. If there is only 1 file, then there was no pairing. 

We need to copy all of the FASTQ files from their subdirectories into the same directory for easy access by SPLASH, and we need a single FASTQ file for each sample. We do this with `shell/fastq_dir.sh`. This script creates `fastq/` in the output directory, which contains the _full_ FASTQ files for the samples, as well as `fastq/splash_input.txt`, which is the input text file for SPLASH. By "full", we mean that one member of each read pair as well as all of the unpaired reads for a given sample are in the file. In other words, we throw away the "_2.fastq" file if it exists.


#### Usage
`bash [path to fetch.sh] -b [path to sra/bin] -p [path to directory of python scripts] -f [path to input_samples.txt] -o [output directory]`

`bash [path to fastq_dir.sh] -p [base path to SRA subdirectories] -o [output directory]`


#### Example
`bash shell/fetch.sh -b ../ww-seq/sra/bin -p python -f data/viana/input_samples.txt -o data/viana`

`bash shell/fastq_dir.sh -p data/viana/srr -o fastq`


## SPLASH Analysis

### Run SPLASH
We now have all of the data needed for running SPLASH. One can run it with whatever parameters, but it must include the flags `--dump_Cjs` and `--dump_sample_anchor_target_count_bin` so that the output includes the embedding vectors and the SATC output to make the contingency tables.

It's nice to just put the SPLASH run in a shell script for reproducability, but we will cover the contents here for completeness. The first command runs SPLASH with the desired flags. The last two commands copy the sample name/sample index mapping file and the results sub-directory to the output directory.  We run it by calling `bash [path to splash_run.sh]` from within the `fastq/` sub-directory. All of the output is written here, but we want it to be in the output directorym, which is why we include the last two commands.

The results include `logs/` (logging of run), `result_Cjs/` (binarized files of $\mathbf{c}$ data), `result_satc/` (binarized files of SATC data), four .csv files of results where each row corresponds to an anchor-target pair, and `sample_name_to_id.mapping.txt` (mapping between BioSample IDs and the indexing used in SPLASH).

The shell script is finally moved to the output directory.

#### Usage

```
#!/bin/bash (shebang)

[path to splash] --dump-Cjs --dump_sample_anchor_target_count_bin \
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
    --dump_Cjs --dump_sample_anchor_target_count_bin \
    --max_pval_opt_for_Cjs 1 \
    --n_threads_stage_1 10 \
    --n_threads_stage_1_internal 2 \
    --n_threads_stage_2 16 \
    --bin_path ../../../splash \
    --tmp_dir ../../../analyses/viala/2026_05_15/tmp \
    --logs_dir ../../../analyses/viala/2026_05_15/logs \
    splash_input.txt

mv sample_name_to_id.mapping.txt ../../../analyses/viala/2026_05_15
mv result* ../../../analyses/viala/2026_05_15
mv ../../../analyses/viala/splash_run.sh ../../../analyses/viala/2026_05_15
```


### $C_j$ Analysis
TO DO

### Filter Anchors
TO DO

### Get Anchor SATC
TO DO



## Contingency Table Analysis

### Clean Metadata
To do the contingency table analysis, it is helpful for us to first organize our metadata (e.g. add in the SPLASH sample ids, convert covariates to factors, etc.). This is done in `clean_metadata.py`. This writes `clean_metadata.tsv` to the specified output directory and will be the reference for metadata used in subsequent analyses.

#### Usage
`python [path to clean_metadata.py] --meta [path to metadata_filt.tsv] --id_key [path to sample_name_to_id.mapping.txt output by SPLASH] --output [output directory]`

#### Example
`python python/clean_metadata.py --meta data/viana/metadata_filt.tsv --id_key analyses/viana/2026_05_19/sample_name_to_id.mapping.txt --output data/viana`



### Create Anchor-Bin Key File
`anchor_bin_key.py` will go through all SATC bin files and create a key for the locations of each anchor.
The key will be saved in the specified output directory as `anchor_bin_key.csv`. If `--clean_up` is specified, the SATC dump files will be deleted. This can save space but at the expense of possibly having to recreate them later, which can be slow.

#### Usage
`python [path to anchor_bin_key.py] --sp_dir [path to splash bin files] --bin_dir [path to satc sub-directory] --output [output directory]`

#### Example
`python python/anchor_bin_key.py --sp_dir splash --bin_dir analyses/viana/2026_05_19/result_satc --output analyses/viana/2026_05_19/`


### Anchor Summary Statistics
`anchor_prop.py` will go through all of the SATC bin files and write relevant summary statistics about the anchor contingency tables to `anchor_summary.csv` and `anchor_filt_summary.csv`. The latter only contains anchors with at least 5 targets (rows) that have $\geq 65\%$ non-zero cells and at least 10 sampls (columns).

We can optionally pass the path to `clean_metadata.tsv` and specify a covariate in that file in call arguments. This will produce a third file `anchor_filt_by_{covar}.csv` which will have the summary statistics for anchors that pass the above conditions AND have at least one sample for every unique value of the specified covariate found in the set of samples used for SPLASH.

While the program iterates through the bin files above, it also separates out the data into anchor-specific SATC files (without the A!). These are written to the path specified by `--scratch` (preferably some scratch directory that has fast read/write). There can be hundreds of thousands to millions of anchors, so it is good to quickly run the next step so that we only keep the files we will actually use for analysis.

#### Usage
`python [path to anchor_prop.py] --sp_dir [path to splash bin files] --bin_dir [path to satc sub-directory] --bin_key [path to anchor_bin_key.csv] --meta [path to clean_metadata.tsv] --covar [name of covariate] --output [output directory] --scratch [path to temporary directory for reading/writing files]`

#### Example
`python python/anchor_prop.py --sp_dir splash --bin_dir analyses/viana/2026_05_19/result_satc --bin_key analyses/viana/2026_05_19/anchor_bin_key.csv --meta data/viana/clean_metadata.tsv --covar date --output analyses/viana/2026_05_19 --clean_up --scratch /scratch/arosenga`

One can optionally plot some of the summary statistics with `anchor_prop_plot.py`. 

#### Usage
`python [path to anchor_prop_plot.py] --output [output directory] --anchor_file [path to anchor_filt_summary.csv]`

#### Example
`python python/anchor_prop_plot.py --output analyses/viana/2026_05_19 --anchor_file analyses/viana/2026_05_19/anchor_filt_summary.csv`


### Anchor Filtering
In this step, we filter the anchor-specific SATC files to just the ones present in `anchor_filt_summary.csv` and copy them to our home directory. If desired (set `--clean_up`), we delete the temporary directory created in the `--scratch` directory to save space. 

#### Usage
`python [path to satc_filt.py] --satc_dir [path to directory of anchor-specific satc files] --anchor_file [path to anchor_filt_summary.csv] --output [output directory] --clean_up`

#### Example
`python python/satc_filt.py --satc_dir /scratch/arosenga/anchor_satc --anchor_file analyses/viana/2026_05_19/anchor_filt_summary.csv --output analyses/viana/2026_05_19 --clean_up`


