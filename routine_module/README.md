# Routine Module Overview
We can run different simulation settings by creating a routine configuratin file. This file specifies the data and model configuration files to use and then dictates what sort of routine will be run and plotted. The routine script will create the output directory that is specified in the model configuration file. All results will be saved in this output directory. In addition, a sub-directory called `iter` will be created, which will contain a log file (`log.txt`) containing information about the replicate data and fitted models. The configuration files will also be copied into a sub-directory in the output directory for reproducibility. Results for each iteration will be labeled as `results_iter{IDX}.npz` and saved in the output directory. Results aggregated over iterations will be saved as `results.npz` and any plots will be saved as PDF, all in the output directory. 

The general format is as follows:

```
d_config: PATH TO DATA CONFIGURATION .YAML
m_config: PATH TO MODEL CONFIGURATION .YAML
routine:  NAME OF ROUTINE
r_args:
  ADDITIONAL ROUTINE ARGUMENTS
```

We then run the routine with the following command:

`sbatch [path to routine.sh] [number of iterations] [path to routine configuration .yaml in quotes]`

This will run the routine, save the results, and then plot the results in some appropriate way (detailed below in the specific routine sub-sections).

We might also desire to re-plot some previously saved results. We can do this with a similar command:

`bash [path to plot_routine.sh] [path to routine configuration .yaml in quotes]`

---

# Error Analysis
We can explore the type I error or type II error of the one- and two-sided score tests compared to OASIS over a grid of values for a provided parameter (power is evaluated automatically if we explore type II error). These simulations are implemented in `error_comps.py`. Currently, only varying the random effect(s) variance component(s), the number of samples, and the number of targets are implemented and have been run. Other parameters have been implemented but not tested. Below are the parameters that can be varied and the grid of values they take on:

- Number of samples (`n_samps`): $20, 50, 100, 500, 750, 1000$
- Number of targets (`n_targs`): $10, 20, 50, 100, 200, 300, 400, 500, 750, 1000$
- Negative binomial dispersion (`disp`): $1, 2, 5, 10, 20, 50, 100$
- Random effects values (`rd_effs`): $1e-4, 1e-3, 1e-2, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 4.0$
- Random effects variance components (`var`): $1e-4, 1e-3, 1e-2, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0$
- Fixed effects variance (`fx_effs_sigma`): $0.0, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0$

This routine will repeatedly draw `reps` number of new datasets according to the data/model configuration. For each dataset, it computes the OASIS (optimized), one-sided score test, and two-sided score test. If there are no fixed effects (i.e. only a fixed intercept), then OASIS (asymptotic) will also be run. It then computes an estimate of the type I (or type II) error by computing the proportion of datasets for which we reject $H_0$ (or fail to reject $H_0$). It is very important to check that the data configuration is appropriate for the error type specified. We must draw data under $H_0$ to evaluate type I error and draw data under $H_1$ to evaluate type II error. We also only need to run one iteration because we need to construct entirely new datasets for each calculation of the test statistics; however, we can run more if we desire and the results will be averaged over iterations. 

The plot that is constructed will be a line plot with the varied parameter along the $x$-axis and the error estimate (or power) on the $y$-axis. Note that the errors are only estimated at a grid of parameter values, and we interpolate between these points to create the lines. The blue line corresponds to the one-sided score test, and the red line corresponds to the two-sided score test. The green line corresponds to OASIS (optimized), and the purple line corresponds to OASIS (asymptotic). A $95\%$ confidence interval is also plotted as a shaded band around each line. These are computed via the Wilson score interval method for a binomial proportion confidence interval.

**NOTE**: We _must_ fully specify the data configuration file with all simulation parameters even thought the parameter that is varied will NOT take on the value we pass in (It can be whatever; it just needs to be provided for instantiation). Also, there currently are very few checks implemented, so the burden of making sure the routine specification is compatible with the data/model (e.g. do not choose to vary `disp` with Poisson outcomes).

## Usage and Example 
### Usage
The routine configuration file has the following format:

```
d_config: PATH TO DATA CONFIGURATION .YAML
m_config: PATH TO MODEL CONFIGURATION .YAML
routine:  'err_comp'
r_args:
  param:    PARAMETER TO VARY (one of 'var_comp', 'n_samps', 'n_targs')
  err_type: ERROR TYPE (one of 'I', 'II')
  reps:     NUMBER OF REPLICATES
  level:    SIGNIFICANCE LEVEL 
```

### Example
Below is an exmaple routine file:

```
d_config: '/home/arosenga/score-test/routine_module/examples/d_err_comp.yaml'
m_config: '/home/arosenga/score-test/routine_module/examples/m_err_comp.yaml'
routine:  'err_comp'
r_args:
  param:    'n_samps'
  err_type: 'I'
  reps:     100
  level:    0.05
```

We then run:

`sbatch slurm_scripts/routine.sh 1 "routine_module/examples/r_err_comp.yaml"`

To re-plot the results:

`bash shell_scripts/plot_routine.sh "routine_module/examples/r_err_comp.yaml"`

---

# Empirical Distribution of Score
We can plot the empirical distribution of the score vector by simulating many datasets with the same data-generating mechanism and plotting a histogram of the score vector components evaluated under $H_0$ (so be sure to generate data under $H_0$!). If the random effects have dimension greater than one, then the resulting plot will have multiple panels. Each one will correspond to coordinate of the variance component vector.

**NOTE**: We specify this routine as `score_var_comp` but the sub-directory will be `score_var_comp_emp_dist` since we are plotting the empirical distribution of the score component(s).

## Usage and Example
### Usage
The general format for the routine configuration file is:

```
d_config: PATH TO DATA CONFIGURATION .YAML
m_config: PATH TO MODEL CONFIGURATION .YAML
routine:  'score_var_comp'
r_args:
  reps:   NUMBER OF REPLICATES
```

### Example
Below is an example routine file:

```
d_config: 'routine_module/examples/d_score_var_comp.yaml'
m_config: 'routine_module/examples/m_score_var_comp.yaml'
routine:  'score_var_comp'
r_args:
  reps:   100
```

Then we run:

`sbatch slurm_scripts/routine.sh 1 "routine_module/examples/r_score_var_comp.yaml"`

To re-plot the results:

`bash shell_scripts/plot_routine.sh "routine_module/examples/r_score_var_comp.yaml"`

---

# TODO

- Test statistic comparisons: `stat_comps.py`
- Test statistic sensitivity analysis: `stat_sens.py`
