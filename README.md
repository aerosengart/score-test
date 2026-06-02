## Score Test for Variance Components in GLMMs
Code for conducting a score test of variance components in GLMMs. 

---

Currently, only a negative binomial model is supported.

## Data Class

### Simulated Data
Simulated data are organized in the `simul_data` class. Instantiation is done via a configuration dictionary passed as a .yaml file. Here is an example for negative binomial data.

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
disp:
  mu: 1.o
  sigma: 0.0
tau_sqr: [0.0]
disp_param: 1.0   
```

The actual data that is drawn will depend upon what `outcome` is set to (must be one of `neg_bin` or `multinom`). If outcome is set to multinomial, then an additional argument must be passed to specify the limits of the column totals.

```
col_sums:
  low: 10
  high: 100
```

### Real Data
Real data is also supported, but the configuration file has a different structure because we need to specify the paths to the files containing the necessary metadata.

```
outcome: 'splash'
satc_dir: '/home/arosenga/score-test/analyses/viana/2026_05_19/anchor_satc'   ## path to anchor-specific S(A)TC files
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
outcome:  'neg_bin'      ## outcome distribution
use_true: False          ## use true values of nuisance parameters?
scale:    True           ## scale score to have unit norm (and also scale information)?
debug:    'off'          ## debugging flag
global_disp: False       ## allow dispersion parameter to be target-specific in estimation
estim_mu:    True        ## whether to extract the estimated mean array from glmGamPoi or compute it (without accounting for size factors)
size_factors: 'col_sums' ## what to set the size factors to (one of ['col_sums'])
out_dir:  'test_plots'   ## super-directory for outputs
```


---

Below we describe the workflow for real data. We use the South Africa SARS-CoV-2 dataset from Viala et al. as an example. All commands were run from the current directory, so paths are relative to this location in the examples.

## Data Retrieval

### Retrieve all associated BioSample IDs
First, we retrieve the BioSample IDs that are associated with the BioProject. This writes `ids.txt` to the specified output directory. Each row in the file specifies a BioSample by SRR (the sequencing library for the run), SRS (the biological source material; i.e. the BioSample), and SAMN (BioSample).

Data is retrieved from the SRA at NCBI using the SRA Toolkit. The shell script just packages all of this nicely. 

#### Usage
`bash [path to shell_scripts/ids.sh script] -b [accession in quotes] -o [output directory]`

#### Example
`bash shell_scripts/ids.sh -b "PRJNA784038" -o "data/viala"`


### Retrieve metadata for BioSamples
Next, we must retrieve the metadata associated with the BioSamples using Entrez Direct. 

This requires fetching the metadata for a single BioSample so that we can scrape the headers first. We manually create `headers.tsv` using the results returned by the first query.

We then run `python_scripts/metadata.py` from the command line to retrieve the desired metadata for all of the BioSamples. This creates `metadata.tsv` in the output directory, which contains rows corresponding to the samples in `ids.txt` and columns corresponding to the variables in `headers.tsv`.

#### Usage
`esearch -db biosample -query [sample SRS in quotes] | esummary`

[_create headers.tsv_]

`python [path to metadata.py] --sh_dir [path to shell script directory] --bsl [path to ids.txt] --head [path to headers.tsv] --output [output directory]`

#### Example
`esearch -db biosample -query "SRS11294961" | esummary`

`python python_scripts/metadata.py --sh_dir shell --bsl data/viana/ids.txt --head data/viana/headers.tsv --output data/viana`


### Select BioSamples
At this stage, we filter the samples to make sure they are complete cases, etc. We use `python_scripts/filter.py`, which creates a list of the filtered sample SRR ids in the output directory as the file `input_samples.txt`. The metadata corresponding to the filtered samples is written to `metadata_filt.tsv`.

#### Usage
`python [path to filter.py] --meta [path to metadata.tsv] --ids [path to ids.txt] --output [output directory] --min-date [min collection date in quotes] -- max-date [max collection date in quotes]`

#### Example
`python python_scripts/filter.py --meta data/viana/metadata.tsv --ids data/viana/ids.txt --output data/viana --min_date 2021-12-01 --max_date 2021-12-05`


### Retrieve sequences
We then fetch the SRA accessions for the samples specified in `input_samples.txt` in FASTQ format. This requires specifying the path to the `bin` directory of your local installation of the SRA Toolkit as well as the path to the directory of python scripts (`fetch.sh` calls `python_scripts/fastq.py`, which itself uses the SRA Toolkit to dump the data as FASTQ). This creates a separate `/srr` in the output directory, which contains a sub-directory for each sample containing the sample .sra and .fastq file(s). 

`python_scripts/fastq.py` calls `fasterq_dump` in the SRA toolkit with the flag `--split-3`. This means that the data in the .sra file will be split if the reads are paired, which results in either 1 or 3 FASTQ files in each sample sub-directory. If there are 3 files, then that means that there was some sort of pairing going on with the reads (see [here](https://github.com/ncbi/sra-tools/wiki/HowTo:-fasterq-dump)). The file with the suffixes "_1.fastq" and "_2.fastq" contains the paired reads, and the file with just ".fastq" contains the unpaired reads. If there is only 1 file, then there was no pairing. 

We need to copy all of the FASTQ files from their subdirectories into the same directory for easy access by SPLASH, and we need a single FASTQ file for each sample. We do this with `shell_scripts/fastq_dir.sh`. This script creates `fastq/` in the output directory, which contains the _full_ FASTQ files for the samples, as well as `fastq/splash_input.txt`, which is the input text file for SPLASH. By "full", we mean that one member of each read pair as well as all of the unpaired reads for a given sample are in the file. In other words, we throw away the "_2.fastq" file if it exists.


#### Usage
`bash [path to fetch.sh] -b [path to sra/bin] -p [path to directory of python scripts] -f [path to input_samples.txt] -o [output directory]`

`bash [path to fastq_dir.sh] -p [base path to SRA subdirectories] -o [output directory]`


#### Example
`bash shell_scripts/fetch.sh -b ../ww-seq/sra/bin -p python -f data/viana/input_samples.txt -o data/viana`

`bash shell_scripts/fastq_dir.sh -p data/viana/srr -o fastq`


## SPLASH Analysis

### Run SPLASH
We now have all of the data needed for running SPLASH. One can run it with whatever parameters, but it must include the flags `--dump_Cjs` and `--dump_sample_anchor_target_count_bin` so that the output includes the embedding vectors and the SATC output to make the contingency tables. We also include the `--without_compactors` and `--with_pval_asymp_opt` so that we do not compute the compactors (new to SPLASH2) and do compute the asymptotic $p$-value. 

It's nice to just put the SPLASH run in a shell script for reproducability, but we will cover the contents here for completeness. The first command runs SPLASH with the desired flags. The last two commands copy the sample name/sample index mapping file and the results sub-directory to the output directory.  We run it by calling `bash [path to splash_run.sh]` from within the `fastq/` sub-directory. All of the output is written here, but we want it to be in the output directory, which is why we include the last two commands.

The results include `logs/` (logging of run), `result_Cjs/` (binarized files of $\mathbf{c}$ data), `result_satc/` (binarized files of SATC data), four .csv files of results where each row corresponds to an anchor-target pair, and `sample_name_to_id.mapping.txt` (mapping between BioSample IDs and the indexing used in SPLASH).

The shell script is finally moved to the output directory.

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


### $C_j$ Analysis
TO DO

### Filter Anchors
TO DO

### Get Anchor SATC
TO DO

### Clean Metadata
To do the contingency table analysis, it is helpful for us to first organize our metadata (e.g. add in the SPLASH sample ids, convert covariates to factors, etc.). This is done in `clean_metadata.py`. This writes `clean_metadata.tsv` to the specified output directory and will be the reference for metadata used in subsequent analyses.

#### Usage
`python [path to clean_metadata.py] --meta [path to metadata_filt.tsv] --id_key [path to sample_name_to_id.mapping.txt output by SPLASH] --output [output directory]`

#### Example
`python python_scripts/clean_metadata.py --meta data/viana/metadata_filt.tsv --id_key analyses/viana/2026_05_29/sample_name_to_id.mapping.txt --output data/viana`

---

## Contingency Table Analysis

### Create Anchor-Bin Key File
`anchor_bin_key.py` will go through all SATC bin files and create a key for the locations of each anchor.
The key will be saved in the specified output directory as `anchor_bin_key.csv`. If `--clean_up` is specified, the SATC dump files will be deleted. This can save space but at the expense of possibly having to recreate them later, which can be slow.

#### Usage
`python [path to anchor_bin_key.py] --sp_dir [path to splash bin files] --bin_dir [path to satc sub-directory] --output [output directory]`

#### Example
`python python_scripts/anchor_bin_key.py --sp_dir splash --bin_dir analyses/viana/2026_05_29/result_satc --output analyses/viana/2026_05_29/`


### Anchor Summary Statistics
`anchor_prop.py` will go through all of the SATC bin files and write relevant summary statistics about the anchor contingency tables to `anchor_summary.csv` and `anchor_filt_summary.csv`. The latter only contains anchors with at least 5 targets (rows) that have $\geq 65\%$ non-zero cells and at least 20 samples (columns).

We can optionally pass the path to `clean_metadata.tsv` and specify a covariate in that file in call arguments. This will produce a third file `anchor_filt_by_{covar}.csv` which will have the summary statistics for anchors that pass the above conditions AND have at least two different unique values of the specified covariate found in the set of samples used for SPLASH so that we do not run into numerical issues down the line.

While the program iterates through the bin files above, it also separates out the data into anchor-specific S(A)TC files (without the A!). These are written to the path specified by `--scratch` (preferably some scratch directory that has fast read/write). There can be hundreds of thousands to millions of anchors, so it is good to quickly run the next step so that we only keep the files we will actually use for analysis.

#### Usage
`python [path to anchor_prop.py] --sp_dir [path to splash bin files] --bin_dir [path to satc sub-directory] --bin_key [path to anchor_bin_key.csv] --meta [path to clean_metadata.tsv] --covar [name of covariate] --output [output directory] --scratch [path to temporary directory for reading/writing files]`

#### Example
`python python_scripts/anchor_prop.py --sp_dir splash --bin_dir analyses/viana/2026_05_29/result_satc --bin_key analyses/viana/2026_05_29/anchor_bin_key.csv --meta data/viana/clean_metadata.tsv --covar date --output analyses/viana/2026_05_29 --scratch /scratch/arosenga`

One can optionally plot some of the summary statistics with `anchor_prop_plot.py`. 

#### Usage
`python [path to anchor_prop_plot.py] --output [output directory] --anchor_file [path to anchor_filt_summary.csv]`

#### Example
`python python_scripts/anchor_prop_plot.py --output analyses/viana/2026_05_29 --anchor_file analyses/viana/2026_05_29/anchor_filt_summary.csv`


### Anchor Filtering
In this step, we filter the anchor-specific S(A)TC files to just the ones present in `anchor_filt_summary.csv` and copy them to our home directory. If desired (set `--clean_up`), we delete the temporary directory created in the `--scratch` directory to save space. 

#### Usage
`python [path to satc_filt.py] --satc_dir [path to directory of anchor-specific S(A)TC files] --anchor_file [path to anchor_filt_summary.csv] --output [output directory] --clean_up`

#### Example
`python python_scripts/satc_filt.py --satc_dir /scratch/arosenga/anchor_satc --anchor_file analyses/viana/2026_05_29/anchor_filt_by_date.csv --output analyses/viana/2026_05_29`

This example resulted in 9,282 anchors in the filtered list.


### Running Score Test
With all of the anchor-specific S(A)TC files, we can run the score test on each anchor. To do this more easily, we will run this as a SLURM job array. This will run the score test on each anchor's table _separately_ and write the results to a .json file _separately_. Later steps will condense/summarize the run outputs to get a ranked list of anchors.

First, we need a list of all of the anchors we want to test where each line is a separate anchor, which we can create one with `list_anchors.sh`. 

#### Usage
`bash [path to list_anchors.sh] -p [path to directory of anchor-specific S(A)TC files] -o [output directory]`

#### Example
`bash shell_scripts/list_anchors.sh -p "analyses/viana/2026_05_29/anchor_satc" -o "analyses/viana/2026_05_29"`


We then run the score test with a shell script that submits a job array for 1000 lines of the passed anchor list as slurm job arrays. Each array then submits a separate job that runs `routines/splash_score_test.py` for one anchor in the provided anchor list. This is done with `all_anchors_job.sh`. 

#### Usage
`bash [path to all_anchors_test.sh] -a  [path to anchor list .txt in quotes] -d [path to data config .yaml in quotes] -m [path to model config .yaml in quotes] -o [output directory in quotes]`

#### Example
`bash shell_scripts/all_anchors_job.sh -a "analyses/viana/2026_05_19/test_anchors.txt" -d "data/configs/d_viana.yaml" -m "data/configs/m_viana.yaml" -o "analyses/viana/2026_05_19"`

This will create thousands and thousands of output files, so to delete the ones corresponding to successful runs (i.e. the empty output files), we can run the following:

`find . -type f -name '*.out' -size 0b -delete`

**NOTE**: There may be a few anchor tables that run into problems. This is usually because there are too few samples, which leads to only one unique value across samples for the covariate associated with the random effect we are testing. I am not sure what is causing this because the filtering steps above should have caught this, but for now we can just ignore these anchors. 


### Analyzing Results
We analyze the anchor-specific .json output with `json_summary.py`. This reads each of the .json files in the passed directory and writes the pertinent information to a `anchor_score_test.csv` in the provided output directory.

#### Usage
`python [path to json_summary.py] --json_dir [path to anchor .json files] --out_dir [output directory]`

#### Example
`python python_scripts/json_summary.py --json_dir /home/arosenga/score-test/analyses/viana/2026_05_19/score_test/anchor_json --out_dir /home/arosenga/score-test/analyses/viana/2026_05_19/score_test`


---

## Simulations
All of the following simulations can be performed with a slurm job array or by manually calling `routines/run_routine.py`. A job array is nice for settings where we want to run many iterations (i.e. many different data-generating mechanisms). The general syntax for such a job array is:

`sbatch --array=1-[number of iterations] [path to routine.sh] [path to data config .yaml in quotes] [path to model config .yaml in quotes] [routine in quotes] [number of replicates per iteration] [OPTION 0/1 FLAG]`

This will create `[number of iterations]` different `.npz` files where each file contains the output of one iteration (which itself has the results of the `[number of replicates per iteration]` replicates drawn). The `.npz` files are saved in a sub-directory named after the routine under a sub-directory named after the configuration specified in the data config file under the output directory specified as `out_dir` in the model config file. The path will have the following form:

`[model out_dir]/d[data outcome]_m[model outcome]_fx_p[number of fixed effects]_rd_q[number of random effects]_samps[number of samples]_targs[number of targets]_[distribution specific parameters]/[routine]`

For negative binomial data, `[distribution specific parameters]` will be `dgamma[data gamma]`. For quasi-Poisson data, it will be `dphi[data phi]`. The results are saved under the name `results_iter[iteration ID].npz` where `iteration ID` is an integer that represents which iteration is being run (i.e. it is arbitrary). A `config.txt` is also created in this sub-directory that records the data config contents (since we re-use the data configuration file across many different simulations and may overwrite things accidentally...). 

To visualize the results, we use `routines/plot_routine.py`. This file combines all of the results (all iterations) into a single array, makes the appropriate plot, and saves it as a PDF in the output directory. The general usage is:

`python [path to plot_routine.py] -d [path to data config .yaml] -m [path to model config .yaml] -r [routine]`


### Empirical Distribution of Score
We can plot the empirical distribution of the score vector by simulating many datasets with the same data-generating mechanism.

#### Usage
`sbatch --array=1 [path to routine.sh] [path to data config .yaml in quotes] [path to model config .yaml in quotes] "score_var_comp" [number of replicates per iteration]`

#### Example
`sbatch --array=1 slurm_scripts/routine.sh "data/configs/d_simul.yaml" "data/configs/m_simul.yaml" "score_var_comp" 100`

We then plot the results with `routines/plot_routine.py`. This creates a histogram (or an array of histograms) that shows the empirical distribution of the score component(s) corresponding to the random effect(s) variance component(s) on the density scale.

#### Usage
`python [path to plot_routine.py] -d [path to data config .yaml] -m [path to model config .yaml] -r score_var_comp`

#### Example
This is very fast (maybe 2-3 minutes).

`python routines/plot_routine.py -d data/configs/d_simul.yaml -m data/configs/m_simul.yaml -r score_var_comp`

**NOTE**: We specify this routine as `score_var_comp` but the sub-directory will be `score_var_comp_emp_dist` since we are plotting the empirical distribution of the score component(s).


### One-Sided and Two-Sided Comparison
In basically the same way as the previous routine, we can compare QQ-plots of the one- and two-sided score test statistics over many iterations and replicates. 

#### Usage
`sbatch --array=1-[number of iterations] [path to routine.sh] [path to data config .yaml in quotes] [path to model config .yaml in quotes] "one_two_sided" [number of replicates per iteration]`

### Example
This is also pretty fast (< 10 minutes).
`sbatch --array=1-20 slurm_scripts/routine.sh "data/configs/d_simul.yaml" "data/configs/m_simul.yaml" "one_two_sided" 100`

We plot the results with `routines/plot_routine.py`. This creates two QQ-plots side-by-side. The left sub-plot shows the two-sided test statistic empirical quantiles against the $\chi^2_q$ theoretical quantiles (where $q$ is the dimension of $\boldsymbol{\tau^2}$, the random effect(s) variance(s)). The right sub-plot show the one-sided test statistic empirical quantiles against the $\bar{\chi}^2$ theoretical quantiles (the specific choice of $\bar{\chi}^2$ depends on $q$). 

#### Usage
`python [path to plot_routine.py] -d [path to data config .yaml] -m [path to model config .yaml] -r one_two_sided`

#### Example
`python routines/plot_routine.py -d data/configs/d_simul.yaml -m data/configs/m_simul.yaml -r one_two_sided`


### Power Analysis (Variance Components)
We can explore the power of the one- or two-sided score tests over different values of the random effect(s) variance component(s). We only need one replicate because we need to construct entirely new datasets for each calculation of the test statistics.

#### Usage
`sbatch --array=1-[number of iterations] [path to routine.sh] [path to data config .yaml in quotes] [path to model config .yaml in quotes] "power_by_tau_sqr" [number of replicates per iteration]`

#### Example
`sbatch --array=1-100 slurm_scripts/routine.sh "data/configs/d_simul.yaml" "data/configs/m_simul.yaml" "power_by_var_comp" 1`

We plot the results with `routines/plot_routine.py`. This creates a line plot with two lines: one for the power of the one-sided test (blue) and one for the power of the two-sided test (red). 

#### Usage
`python [path to plot_routine.py] -d [path to data config .yaml] -m [path to model config .yaml] -r power_by_var_comp`

#### Example
`python routines/plot_routine.py -d data/configs/d_simul.yaml -m data/configs/m_simul.yaml -r power_by_var_comp`



### Power Analysis (Sample Size)
We can explore the power of the one- or two-sided score tests over different values of the number of samples (for fixed $\tau^2$ and number of targets). We only need one replicate because we need to construct entirely new datasets for each calculation of the test statistics.

#### Usage
`sbatch --array=1-[number of iterations] [path to routine.sh] [path to data config .yaml in quotes] [path to model config .yaml in quotes] "power_by_n_samps" [number of replicates per iteration]`

#### Example
`sbatch --array=1-100 slurm_scripts/routine.sh "data/configs/d_simul.yaml" "data/configs/m_simul.yaml" "power_by_n_samps" 1`

We plot the results with `routines/plot_routine.py`. This creates a line plot with two lines: one for the power of the one-sided test (blue) and one for the power of the two-sided test (red). 

#### Usage
`python [path to plot_routine.py] -d [path to data config .yaml] -m [path to model config .yaml] -r power_by_n_samps`

#### Example
`python routines/plot_routine.py -d data/configs/d_simul.yaml -m data/configs/m_simul.yaml -r power_by_n_samps`


### Power Analysis (Number of Targets)
We can explore the power of the one- or two-sided score tests over different values of the number of targets (for fixed $\tau^2$ and sample size). We only need one replicate because we need to construct entirely new datasets for each calculation of the test statistics.

#### Usage
`sbatch --array=1-[number of iterations] [path to routine.sh] [path to data config .yaml in quotes] [path to model config .yaml in quotes] "power_by_n_targs" [number of replicates per iteration]`

#### Example
`sbatch --array=1-100 slurm_scripts/routine.sh "data/configs/d_simul.yaml" "data/configs/m_simul.yaml" "power_by_n_targs" 1`

We plot the results with `routines/plot_routine.py`. This creates a line plot with two lines: one for the power of the one-sided test (blue) and one for the power of the two-sided test (red). 

#### Usage
`python [path to plot_routine.py] -d [path to data config .yaml] -m [path to model config .yaml] -r power_by_n_targs`

#### Example
`python routines/plot_routine.py -d data/configs/d_simul.yaml -m data/configs/m_simul.yaml -r power_by_n_targs`



### Fixed Effects Sensitivity
We can explore the sensitivity of the one- or two-sided score test statistics to misspecification (i.e. mis-estimation) of the fixed effects coefficients.

#### Usage
`sbatch --array=1-[number of iterations] [path to routine.sh] [path to data config .yaml in quotes] [path to model config .yaml in quotes] "fx_effs_sens" [number of replicates per iteration] [0/1 flag for known dispersion parameter]`

#### Example
A bit slow (~20 minutes/iteration).
`sbatch --array=1-20 slurm_scripts/routine.sh "data/configs/d_simul.yaml" "data/configs/m_simul.yaml" "fx_effs_sens" 100 1`

`python routines/plot_routine.py -d data/configs/d_simul.yaml -m data/configs/m_simul.yaml -r fx_effs_sens`






<!-- ### Unfiltered Tests
An alternative to the above procedure is to _not_ filter the anchors and to conduct the test on _all_ anchor contingency tables. We first have to dump all of these tables. The following command does this and also creates `all_anchors_list.txt` in the output directory, which is a text file with each anchor's sequence as a separate line.

#### Usage
`python [path to make_satc.py] --sp_dir [path to splash bin files] --bin_dir [path to satc sub-directory] --scratch [path to scratch directory] --output [output directory]`

#### Example
`python python_scripts/make_satc.py --sp_dir splash --bin_dir analyses/viana/2026_05_19/result_satc --scratch /scratch/arosenga --output analyses/viana/2026_05_19`


The Leviathans each have their own `/scratch`, so we have to copy these S(A)TC files over to `/home` if we want to use a slurm job array. It is not very good to keep millions of files on `/home`, so we use a more complicated wrapper shell script to submit the array. 
 -->







