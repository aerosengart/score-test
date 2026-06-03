This module contains all of the methods and classes related to representing the datasets and resulting models in Python. The two classes are instantiated via `.yaml` configuration files. These files different slightly for each of the child classes within a given parent class. 

---

# Data
The parent class is implemented in `data.py` as `Data`. It has two child classes: one for simulated data and one for real data output by `SPLASH`. These are implemented in `simul_data.py` and `splash_data.py`, respectively. The following attributes are required for all instances of `Data` and are set prior to any sampling or data loading:

| Name      | Type         | Description     | Source | 
| :---        |    :---     | :---     | :--- |
| `config` | `dict` | Configuration dictionary. | Argument. |
| `p` | positive `int`       | Number of fixed effects (will _always_ include an intercept).  | Computed. |
| `q` | positive `int`       | Number of random effects. | Computed. | 
| `outcome` | `str` | Data distribution (one of `splash`, `neg_bin`, `poi`, or `multinom`). | Configuration file. | 
| `fx_slopes` | non-negative `int` | Number of fixed slopes. | Configuration file. |
| `rd_slopes` | non-negative `int` | Number of random slopes. | Configuration file. |


## Simulated Data
The simulated data child class is implemented in `simul_data.py`. It has three child classes: one for negative binomial data, one for multinomial data, and one for poisson data. 

Regardless of child class, simulated datasets have the following attributes:

| Name      | Type         | Description     | Source | Notation |
| :---      |    :---     | :---     | :--- | :--- |
| `n_samps` | positive `int` | Number of samples (columns). | Configuration file. | $$N_s$$ |
| `n_targs` | positive `int` | Number of targets (rows). | Configuration file. | $$N_t$$ |
| `p_star` | positive `int` | Total number of fixed effects when accounting for the fact that they are target-specific. `p_star` should equal $p$ multiplied by `n_targs`. | Computed. | $$p^*$$ |
| `tau_sqr` | `list` of non-negative `float` | Random effects variance components. The length of the list should be equal to $q$ and therefore also equal to `rd_slopes`. | Configuration file. | $$\boldsymbol{\tau}^2$$ | 
| `fx_effs_mu` | `float` | Mean for fixed effects coefficients (`fx_effs_arr`). | Configuration file. | $$\alpha_\mu$$ |
| `fx_effs_sigma` | positive `float` | Standard deviation for fixed effects coefficients (`fx_effs_arr`). | Configuration file. | $$\alpha_\sigma$$ |
| `x_mu` | `float` | Mean for fixed effects covariates (`x_arr`). | Configuration file. | $$x_{\mu}$$ |
| `x_sigma` | positive `float` | Standard deviation for fixed effects covariates (`x_arr`). | Configuration file. | $$x_\sigma$$ |
| `z_mu` | `float` | Mean for random effects covariates (`z_arr`). | Configuration file. | $$z_\mu$$ |
| `z_sigma` | positive `float` | Standard deviation for random effects covariates (`z_arr`). | Configuration file. | $$z_\sigma$$ |
| `x_arr`   | `jax.numpy.array` | Fixed effects covariate array. Because the covariate data is sample-specific, the array will have dimensions (`n_samps`, $p$) for a model with $p$ fixed effects. The first column will be all ones to account for the intercept. | Sampled. | $$\mathbf{X}$$ |
| `z_arr`   | `jax.numpy.array` | Random effects covariate array. Because the covariate data is sample-specific, the array will have dimensions (`n_samps`, $q$) for a model with $q$ random effects. | Sampled. | $$\mathbf{Z}$$ |
| `fx_effs_arr` | `jax.numpy.array`       | Fixed effects coefficient array. The fixed effects are assumed to be target-specific (put another way, they are random effects with infinite variance), so the array dimensions will be (`n_targs`, $p$) for a model with $p$ fixed effects (including intercepts). | Sampled. | $$\boldsymbol{\alpha}$$ |
| `rd_effs_arr` | `jax.numpy.array`       | Random effect coefficient array. The random effects are necessarily target-specific, so the array dimensions will be (`n_targs`, $q$) for a model with $q$ random effects. If the variance components are all zero (i.e. $H_0$ is true), then this array will be all zeros. | Sampled. | $$\boldsymbol{\beta}$$ |
| `targ_idx` | `list` of non-negative `int` | Row indices ($1$-indexed) of non-zero array counts. Length will match that of `samp_idx` and `counts`. | Computed. | — |
| `samp_idx` | `list` of non-negative `int` | Column indices ($1$-indexed) of non-zero array counts. Length will match that of `targ_idx` and `counts`. | Computed. | — |
| `counts` | `list` of positive `int` | Counts in non-zero array cells. Length will match that of `targ_idx` and `samp_idx`. | Computed. | — |
| `eta_arr` | `jax.numpy.array` | Array of linear predictors for each observation. The array will have dimensions (`n_targs`, `n_samps`). | Computed. | $$\boldsymbol{\eta}$$ |
| `mu_arr` | `jax.numpy.array` | Array of means for each observation. The array will have dimensions (`n_targs`, `n_samps`). | Computed. | $$\boldsymbol{\mu}$$ |
| `var_arr` | `jax.numpy.array` | Array of variances for each observation. The array will have dimensions (`n_targs`, `n_samps`). | Computed. | v(\boldsymbol{\mu}) |

The actual count tables are stored as three lists to account for sparsity. These lists (`targ_idx`, `samp_idx`, and `counts`) record the indices and values of non-zero cells within the count tables. The full table can be constructed with the `simul_data.Data` method `construct_y()`; however, its use is discouraged for very high-dimensional tables. 


### Negative Binomial
This class samples data according to a negative binomial generalized linear mixed model. The negative binomial setting has the following additional attributes.

| Name      | Type         | Description     | Source | Notation |
| :---      |    :---     | :---     | :--- | :--- |
| `disp_mu` | positive `float` | Mean for scale/dispersion parameter(s). | Configuration file. | $$\gamma_\mu$$ |
| `disp_sigma` | non-negative `float` | Standard deviation for scale/dispersion parameter(s). | Configuration file. | $$\gamma_\sigma$$ |
| `gamma` | `jax.numpy.array` | Array of scale/dispersion parameters. This is permitted to be target-specific if `disp_sigma` is set to be non-zero. The array will have dimensions (`n_targs`,). | Sampled. | $$\boldsymbol{\gamma}$$ ||
| `probs` | `jax.numpy.array` | Success probability for each observation. The array will have dimensions (`n_targs`, `n_samps`). | Computed. | $$\boldsymbol{\pi}$$ |
| `succs` | `jax.numpy.array` | Number of successes for each negative binomial draw. This is equal to `gamma`, so the array will have dimensions (`n_targs`,). | Computed. | $$\mathbf{s}$$ |

The link function for a negative binomial GLMM is $\log(\cdot)$, so `mu_arr` is computed as the exponential function of `eta_arr` element-wise. The variance function is $v(\mu_{i,j}) = \mu_{i,j} + \frac{1}{\gamma_i} \mu_{i,j}^2$ where the dispersion paramter(s) ($\gamma_i$ and `gamma`) may or may not be target-specific. 

Because we parametrize the distribution via the mean and variance, we have to work backwards to the probabilities and number of successes needed to sample the data. Letting $r_{i,j}$ be the number of successes (`succs`) and $\pi_{i,j}$ (`probs`) the success probability for observation $i,j$, we have the following relationship:

$$
\begin{aligned}
r_{i,j} &= \frac{\mu_{i,j}^2}{v(\mu_{i,j}) - \mu_{i,j}} = \gamma_i \
\pi_{i,j} &=  \frac{\mu_{i,j}}{v(\mu_{i,j})} = \frac{\gamma_i}{\gamma_i + \mu_{i,j}}
\end{aligned}
$$

Below is an example `.yaml` configuration file. 

```
outcome: neg_bin
n_samps: 50
n_targs: 100
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
  mu: 1.0
  sigma: 0.0
tau_sqr: [0.0]
disp_param: 1.0
```

That is, we sample i.i.d. negative binomial observations with the linear predictors given by:

$$
\begin{aligned}
\eta_{i,j} &=\boldsymbol{\alpha}_i^\top \mathbf{x}_j + \boldsymbol{\beta}_i^\top \mathbf{z}_{j} \\
\mu_{i,j} &= \exp(\eta_{i,j}) \\
\boldsymbol{\beta}_i &\overset{iid}{\sim} \mathcal{N}(\mathbf{0}_q, \text{diag}(\boldsymbol{\tau}^2))
\end{aligned}
$$

### Multinomial
This class samples data for an assumed multinomial logistic mixed model with the following additional attributes.

| Name      | Type         | Description     | Source | Notation |
| :---      |    :---     | :---     | :--- | :--- |
| `col_sums_low` | non-negative `int` | Lower bound for column sums. | Configuration file. |  — |
| `col_sums_high` | positive `int` | Upper bound for column sums. | Configuration file. |  — |
| `col_sums` | `jax.numpy.array` | Array of column sums (total counts for each sample). The array will have dimensions (`n_samps`,). | Sampled. | $$\mathbf{n}$$ |
| `probs` | `jax.numpy.array` | Class probabilities for multinomial distribution. The $(i,j)$-th entry will correspond to the probability of observing target $i$ in the reads from sample $j$. The array will have dimensions (`n_targs`, `n_samps`) because the row probability vector is _not necessarily_ the same across samples. | Computed. | $$\boldsymbol{\pi}$$ |

To account for the fact that the mean vector for each column in the count table must equal the row probability vector multiplied by the column sum, we first compute an initial mean array (`temp_mu_arr`) equal to the exponential function of `eta_arr` element-wise. This linear predictor is computed in the standard way. As in multinomial logistic regression, we compute the probability array (`probs`) as:

$$
\begin{aligned}
\pi_{0, j} &= \frac{1}{1 + \sum_{i = 2}^{N_t} m_{i,j}} \\
\pi_{i, j} &= \pi_{0,j} \times m_{i,j}
\end{aligned}
$$

where $N_t$ is the number of targets (`n_targs`), $n_j$ is the sum of the counts in column $j$, $\pi_{i,j}$ is the probability of target $i$ in sample $j$, and $m_{i,j}$ is the $(i,j)$-th entry of `temp_mu_arr`.

We then compute the mean array (`mu_arr`) and variance array (`var_arr`) according to the mean and variance formulae for the multinomial distribution. The $(i,j)$-th entry of these arrays are given, respectively, by:

$$
\begin{aligned}
\mu_{i,j} &= n_j \times \pi_{i,j} \\
v(\mu_{i,j}) &= n_j \times \pi_{i,j} \times (1 - \pi_{i,j})
\end{aligned}
$$

Below is an example `.yaml` configuration file. 

```
outcome: 'multinom'
n_samps: 100
n_targs: 100
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
col_sums:
  low: 10
  high: 100
tau_sqr: [0.0]
```


### Poisson Data
TODO

---

### Real Data
The real data child class is implemented in `splash_data.py`. Data are read in from specified files containing the counts for the table and any necessary metadata as output by `SPLASH`. The following additional attributes are set upon initialization.


| Name      | Type         | Description     | Source | Notation |
| :---      |    :---     | :---     | :--- | :--- |
| `splash_dir` | `str` | Path to directory with `SPLASH` executables. | Configuration file. | — |
| `satc_dir` | `str` | Path to directory containing anchor-specific S(A)TC files. | Configuration file. |  — |
| `metadata_file` | `str` | Path to `clean_metadata.tsv`, the cleaned metadata file output by `python_scripts/clean_metadata.py`. | Configuration file. | — |
| `temp_dir` | `str` | Path to a temporary directory used for saving intermediate results. | Configuration file. | — |
| `clean_up` | `bool` | Whether to delete temporary files upon completion. | Configuration file. | — |
| `fx_covars` | List of `str` | Names of covariates to use as fixed effects. Values must match columns in `metadata_file`. | Configuration file. |  — |
| `rd_covars` | List of `str` | Names of covariates to use as random effects. Values must match columns in `metadata_file`. | Configuration file. | — |
| `x_arr`   | `jax.numpy.array` | Fixed effects covariate array. Because the covariate data is sample-specific, the array will have dimensions (`n_samps`, $p$) for a model with $p$ fixed effects. The first column will be all ones to account for the intercept. | Loaded from file. | $$\mathbf{X}$$ |
| `z_arr`   | `jax.numpy.array` | Random effects covariate array. Because the covariate data is sample-specific, the array will have dimensions (`n_samps`, $q$) for a model with $q$ random effects. | Loaded from file. | $$\mathbf{Z}$$ |
| `tau_sqr` | `list` of non-negative `float` | Random effects variance components. The length of the list should be equal to $q$ and therefore also equal to `rd_slopes`. | Configuration file. | $$\boldsymbol{\tau}^2$$ |
| `n_samps` | positive `int` | Number of samples (columns). | Computed. | $$N_s$$ |
| `n_targs` | positive `int` | Number of targets (rows). | Computed. | $$N_t$$ |
| `targ_idx` | `list` of non-negative `int` | Row indices ($1$-indexed) of non-zero array counts. Length will match that of `samp_idx` and `counts`. | Computed. |  — |
| `samp_idx` | `list` of non-negative `int` | Column indices ($1$-indexed) of non-zero array counts. Length will match that of `targ_idx` and `counts`. | Computed. |  — |
| `counts` | `list` of positive `int` | Counts in non-zero array cells. Length will match that of `targ_idx` and `samp_idx`. | Computed. | — |
| `targ_dict` | `dict` | Dictionary specifying the corresponding index in `targ_idx` for each target sequence. | Computed. |  — |
| `samp_dict` | `dict` | Dictionary specifying the corresponding index in `samp_idx` for each sample index (from `SPLASH`). | Computed. |  — |

This class has an additional argument passed upon intialization to specify which count table to read from file. It is also saved as an attribute.

| Name      | Type         | Description     | Source | 
| :---      |    :---     | :---     | :--- | 
| `anchor` | `str` | Nucleotide sequence of anchor being tested. | Argument.

The count data are read from an anchor-specific (`anchor`) S(A)TC file that is generated by `SPLASH`. Each line in such a file specifies the number of times a given target appears in the reads of a particular sample. Because the total number of targets and samples can be quite large, the resulting count table can also be quite large. Creating the full count table (as a dense array) is discouraged. Below is an example `.yaml` configuration file.

```
outcome: 'splash'
satc_dir: '/home/arosenga/score-test/analyses/viana/2026_05_19/anchor_satc'   
splash_dir: /home/arosenga/score-test/splash'    
metadata_file: '/home/arosenga/score-test/data/viana/clean_metadata.tsv'     
temp_dir: 'viana_temp' 
fx_covars: []          
rd_covars: ['date']   
fx_slopes: 0
rd_slopes: 1
clean_up: False      
```

----

# Model
The parent class is implemented in `model.py` as `Model`. It has the following child classes: `neg_bin_model.py`. Instances of `Model` are built upon a configuration file and an instance of a `Data` child class. The configuration file dictates how the model will be fit, while the `Data` instance provides the necessary data for actually fitting said model. The following attributes are found in all instances of `Model`:

| Name      | Type         | Description     | Source |  Notation |
| :---        |    :---     | :---     | :--- | :---  | 
| `config` | `dict` | Configuration dictionary. | Argument. | — | 
| `data` | instance of `Data` | Data used for model fitting. | Argument. | — |
| `outcome` | `str`       | Outcome distribution of responses. | Configuration file. | — |
| `scale` | `bool`       | Whether to scale the score vector to have unit norm. If `True`, then the information matrix will be similarly scaled. | Configuration file. | — | 
| `use_true` | `bool`       | Whether to use the true values of the model parameters (taken from `data`). This can only be `True` for instances of `simul_data` and will result in _no estimation_ in such a case. | Configuration file. | — | 
| `debug` | `str` | Debugging settings (one of `off`, `error`, `full`). If `off`, no debugging logs will be saved to file. If `full`, all possible debugging logs (e.g. `data` instances, optimization results, etc.) will be saved to file. If `error`, debugging logs will only be saved upon error in model fitting or optimization. | Configuration file. | — | 
| `size_factors` | `str` | What adjustment to use for the `size_factors` argument in `glmGamPoi` (one of `col_sums` or `default`). If `col_sums`, then the natural logarithm of the column sums is added as a term to the model during fitting. If `default`, the default setting of `glmGamPoi` is used (a normalized version of `col_sums`). | Configuration file. | — | 
| `estim_mu` | `bool`       | Whether to use the estimated mean array output by `glmGamPoi`. If `False`, then the mean array (`mu_arr`) is computed manually as the inverse link of the linear predictor (`eta_arr`). If `True`, then the version estimated by `glmGamPoi` is used. This version includes the adjustment for variable column sums (`size_factors`) in the mean computations. | Configuration file. | — | 
| `fx_effs_arr` | `jax.numpy.array`       | Maximum likelihood estimates of fixed effects coefficients. The dimensions of the array are (`n_samps`, `p`) for a model with $p$ fixed effects. | Configuration file. | $\boldsymbol{\alpha}$ | 
| `rd_effs_arr` | `jax.numpy.array`       | Value of random effects coefficients under $H_0$ (all zeros). The dimensions of the array are (`n_samps`, `q`) for a model with $q$ random effects. | Computed. | $\boldsymbol{\beta}$ | 
| `tau_sqr` | `jax.numpy.array`       | Random effects variance components under $H_0$ (all zeros). The dimensions of the array are (`q`,) for a model with $q$ random effects. | Computed. | $\boldsymbol{\tau}^2$ | 
| `mu_arr` | `jax.numpy.array` | Estimated mean array under $H_0$. The array dimensions are (`n_targs`, `n_samps`). | Computed. | $\boldsymbol{\mu}$ |

To conduct the score test, several model-specific quantities must be computed and are implemented as child class methods. They are explained in more detail for each child class.


## Negative Binomial Model
The negative binomial model is implemented in `neg_bin_model.py`. The following attributes are distribution-specific and included for this child class:

| Name      | Type         | Description     | Source | 
| :---        |    :---     | :---     | :--- |
| `gamma` | `jax.numpy.array` | Negative binomial dispersion/scale parameter. The array dimensions are (`n_samps`,). If `global_disp` is `True`, then all values are identical. | Computed (estimated or extracted from `data`). |

The fixed effects coefficients (`fx_effs_arr`) and the dispersion parameters (`gamma`) are estimated with `glmGamPoi`, which fits a separate Gamma-Poisson (i.e. negative binomial) GLM to each target's data. The mean for target $i$ in sample $j$ is:

$$
\begin{aligned}
\mu_{i,j} &= \exp\left( \eta_{i,j} + \xi_{j} \right)
\end{aligned}
$$

where $\eta_{i,j}$ is the standard linear predictor (e.g. $\boldsymbol{\alpha}_i^\top \mathbf{x}_{j}$ for target-specific fixed effects vector, $\boldsymbol{\alpha}_i$, and sample $j$ covariate vector, $\mathbf{x}_j$), and $\xi_j$ is some adjustment made for uneven column sums (sequencing depth in genomics). Currently, $\xi_j$ is computed in one of the following ways:

$$
\begin{aligned}
\xi_j &= \log\left( \sum_{i = 1}^{N_t} y_{i,j} \right) & \left(\text{col sums}\right)\\
\xi_j &= \frac{\log\left( \sum_{i = 1}^{N_t} y_{i,j} \right)}{\exp\left(\frac{1}{N_s} \log\left( \sum_{i = 1}^{N_t} y_{i,j}  \right) \right)} & \left(\text{default}\right)
\end{aligned}
$$

where $N_t$ and $N_s$ are the number of targets (`n_targs`) and samples (`n_samps`), respectively. The score test quantities include:

| Name        | Description     | Notation | 
| :---        | :---     | :---   |
| `compute_eta()` | Computes the linear predictor under $H_0$ _without_ accounting for `size_factors`. Linear predictors are computed as $\boldsymbol{\alpha}_i^\top \mathbf{x}_j$. | $$\boldsymbol{\eta}$$ | 
| `compute_mu()` | Computes the mean array under $H_0$ as $\exp(\eta_{i,j})$ | $$\boldsymbol{\mu}$$ |
| `compute_resid()` | Computes the residuals under $H_0$ as $y_{i,j} - \mu_{i,j}$. | $$y_{i,j} - \mu_{i,j}$$ |
| `compute_g()` | Computes the link function applied to the means under $H_0$ as $\eta_{i,j}$. | $$g(\mu_{i,j})$$ |
| `compute_g_prime()` | Computes the first derivative of each $g(\mu_{i,j})$ with respect to $\mu_{i,j}$ under $H_0$ as $\frac{1}{\mu_{i,j}}$. | $$g'(\mu_{i,j})$$ |
| `compute_g_prime_2()` | Computes the second derivative of each $g(\mu_{i,j})$ with respect to $\mu_{i,j}$ under $H_0$ as $-\frac{1}{\mu^2_{i,j}}$. | $$g''(\mu_{i,j})$$ |
| `compute_v()` | Computes the variance of each observation as $\mu_{i,j} + \frac{1}{\gamma_i} \mu_{i,j}^2$. | $$v(\mu_{i,j})$$ |
| `compute_v_prime()` | Computes the first derivative of each $v(\mu_{i,j})$ with respect to $\mu_{i,j}$ under $H_0$ as $1 + \frac{2}{\gamma} \mu_{i,j}$. | $$v'(\mu_{i,j})$$ |
| `compute_v_prime_2()` | Computes the second derivative of each $v(\mu_{i,j})$ with respect to $\mu_{i,j}$ under $H_0$ as $\frac{2}{\gamma}$. | $$v''(\mu_{i,j})$$ |
