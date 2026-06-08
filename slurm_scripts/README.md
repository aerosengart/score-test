# Running Routines via SLURM
This sub-directory provides SLURM scripts for running the routines in the `routines` sub-directory on the Leviathans.

## Simulations
A job array is nice for settings where we want to run many iterations (i.e. many different data-generating mechanisms). The general syntax for such a job array is:

`sbatch --array=1-[number of iterations] [path to routine.sh] [path to routine configuration file in quotes]`

The slurm script activates the `score-test` conda environment and then submits the appropriate job according to the passed configuration file.

The job array will create `[number of iterations]` different `.npz` files (one file per job in the array) where each file contains the output of one iteration (which itself has the results of the `[number of replicates per iteration]` replicates drawn). The `.npz` files are saved in a sub-directory named after the routine under a sub-directory named after the configuration specified in the data config file under the output directory specified as `out_dir` in the model config file. The path will have the following form:

`[model out_dir]/d[data outcome]_m[model outcome]_fx_p[number of fixed effects]_rd_q[number of random effects]_samps[number of samples]_targs[number of targets]_[distribution specific parameters]/[routine]`

For negative binomial data, `[distribution specific parameters]` will be `dgamma[data gamma]`. The results are saved under the name `results_iter[iteration ID].npz` where `iteration ID` is an integer that represents which iteration is being run (i.e. it is arbitrary). A `config.txt` is also created in this sub-directory that records the data config contents (since we re-use the data configuration file across many different simulations and may overwrite things accidentally...). If the model configuration specifies logging, then additional files will be saved as `.pickle`.

The program will then submit one more job that plots the results of the simulations. This job does not execute until all simulations have completed, so it may take a while for the entire thing to finish.


### Example — Empirical Distribution of Score
We can plot the empirical distribution of the score vector by simulating many datasets with the same data-generating mechanism. This routine does not really require an array because we only need a single iteration (with a large enough number of replicates). We submit the job with:

`sbatch slurm_scripts/routine.sh 1 "routines/configs/test.yaml"`

The entire routine is run as a job array and then the results are plotted with a call to `routines/plot_routine.py`. This creates a histogram (or an array of histograms) that shows the empirical distribution of the score component(s) corresponding to the random effect(s) variance component(s) on the density scale.

Because of how the scripts are written, all of the jobs in the array need to successfully complete for the plotting to occur (i.e. if at least one iteration fails, then the plotting job will be left hanging as `DependencyNeverSatisfied` in slurm). We can manually call the plotting script so that we can still see the results of the successful jobs. 

`python routines/plot_routine.py --cfg routines/configs/poi_one_two.yaml`