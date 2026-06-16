# Simulation-Based Routines
This sub-directory contains the functions needed to run simulation-based routines and plot the results.

We have a few helper files that aid in avoiding code duplication and organize the workflow. Dispatch dictionaries are found in `dispatch.py`, which specify which particular functions to call when a user specifies a particular routine by its shortened name in a routine configuration file. The `doc.py` file contains all functions related to documenting a routine. This includes saving intermediate results, recording the routine specification, and ensuring the appropriate output sub-directories exist.

---

## Empirical Distributions (`emp_dists.py`)
This file implements routines that involve the empirical distribution of some quantity. The additional routine arguments include:

| Name      | Type         | Description     | 
| :---      |    :---     | :---     | 
| `reps` | non-negative `int` | Number of replicates to draw. | 

The resulting plot is a histogram (or grid of histograms if $q > 1$) of the coordinate(s) of the score vector evaluated at the model parameter values (estimated or fixed at true values) corresponding to the variance component ($\boldsymbol{\tau}^2$). The shape should be approximately symmetrical about zero and should also be approximately Gaussian. 



## Test Statistic Comparisons (`stat_comps.py`)
This file implements routines that involve comparing the empirical (null) distribution of the different test statistics. 

### `score_lrt_comparison()`
This function compares the one-sided score test and the likelihood ratio test. The additional routine arguments include:

| Name      | Type         | Description     | 
| :---      |    :---     | :---     | 
| `reps` | non-negative `int` | Number of replicates to draw. | 

The resulting plot has two panels. The left panel shows the likelihood ratio test statistic empirical quantiles against $\bar{\chi}^2$ quantiles. The right panel shows the same but for the one-sided score test.

### `one_two_sided()`
This function compares the one- and two-sided score test statistics. The additional routine arguments include:

| Name      | Type         | Description     | 
| :---      |    :---     | :---     | 
| `reps` | non-negative `int` | Number of replicates to draw. | 

The resulting plot has two panels. The left panel shows the two-sided score test statistic empirical quantiles against $\bar{\chi}^2$ quantiles. The right panel shows the same but for the one-sided score test.



## Sensitivity Plots (`stat_sens.py`)
This file implements routines that evaluate the sensitivity of the score test statistic to errors in estimation of different model parameters. The additional routine arguments include:

| Name      | Type         | Description     | 
| :---      |    :---     | :---     | 
| `reps` | non-negative `int` | Number of replicates to draw. | 
| `param` | `str` | Parameter to jitter. Value must be one of 'fx_effs' (fixed effects coefficients) or 'disp' (negative binomial dispersion). |
| `other_known` | `bool` | Whether the other model parameters are fixed at their true values or not. | 

Estimation error is simulated by adding jitter (some small value) to the true values and using that as the model parameter value. The same value is added to _all_ of the fixed effects coefficients. 



## Test Error Comparisons (`error_comps.py`)
This file implements routines that compare the Type I or Type II error of `OASIS` and our score test.

For a Type I error comparison, we simulate data under $H_0$ repeatedly and compute the fraction of replicates where $H_0$ is rejected (at whatever level $\alpha$ specified in the routine configuration dictionary). For a Type II error comparison, we simlate data under $H_1$ repeatedly and compute the fraction of replicates where we fail to reject $H_0$ (at level $\alpha$). If the routine is run with multiple iterations, then the mean fraction across iterations will be plotted.

The additional routine arguments include:

| Name      | Type         | Description     | 
| :---      |    :---     | :---     | 
| `reps` | non-negative `int` | Number of replicates to draw. | 
| `level` | positive `float` | Test level. Value must be in $(0, 1)$. | 
| `err_type` | `str` | Error type. Value must be one of 'I' or 'II'. |
| `param` | `str` | Parameter to vary along x-axis. Value must be one of 'n_samps' (number of samples/columns), 'n_targs' (number of targets/rows), or 'disp' (negative binomial dispersion parameter). |

The resulting plot is a line plot where the values of `param` used are along the x-axis, and the estimated error is plotted along the y-axis. If the null model has only a fixed intercept, then the line plot will have four lines corresponding to the $p$-value bound from the original OASIS, the asymptotic $p$-value from OASIS, and the $p$-values from the one- and two-sided score tests. If the null model includes fixed slope(s), then there will only be three lines: OASIS-perp and the one- and two-sided score tests. 






