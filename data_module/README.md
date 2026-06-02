This module contains all of the methods and classes related to representing the datasets and resulting models in Python. The two classes are instantiated via `.yaml` configuration files. These files different slightly for each of the child classes within a given parent class. 

----

## Data
The parent class is implemented in `data.py` as `Data`. It has two child classes: one for simulated data and one for real data output by `SPLASH`. These are implemented in `simul_data.py` and `splash_data.py`, respectively. The following attributes are required for all instances of `Data` and are set prior to any sampling or data loading:

| Name      | Type         | Description     | Source | 
| :---        |    :---     | :---     | :--- |
| `config` | `dict` | Configuration dictionary. | Configuration file. |
| `p` | positive `int`       | Number of fixed effects (will _always_ include an intercept).  | Computed. |
| `q` | positive `int`       | Number of random effects. | Computed. | 
| `outcome` | `str` | Data distribution (one of `splash`, `neg_bin`, `poi`, or `multinom`). | Configuration file. | 
| `fx_slopes` | non-negative `int` | Number of fixed slopes. | Configuration file. |
| `rd_slopes` | non-negative `int` | Number of random slopes. | Configuration file. |


### Simulated Data
The simulated data child class is implemented in `simul_data.py`. It has three child classes: one for negative binomial data, one for multinomial data, and one for poisson data. 

Regardless of child class, simulated datasets have the following attributes:

| Name      | Type         | Description     | Source | 
| :---      |    :---     | :---     | :--- | 
| `n_samps` | positive `int` | Number of samples (columns). | Configuration file. | 
| `n_targs` | positive `int` | Number of targets (rows). | Configuration file. | 
| `p_star` | positive `int` | Total number of fixed effects when accounting for the fact that they are target-specific. `p_star` should equal $p$ multiplied by `n_targs`. | Computed. | 
| `tau_sqr` | `list` of non-negative `float` | Random effects variance components. The length of the list should be equal to $q$ and therefore also equal to `rd_slopes`. | Configuration file. | 
| `fx_effs_mu` | `float` | Mean for fixed effects coefficients (`fx_effs_arr`). | Configuration file. | 
| `fx_effs_sigma` | positive `float` | Standard deviation for fixed effects coefficients (`fx_effs_arr`). | Configuration file. | 
| `x_mu` | `float` | Mean for fixed effects covariates (`x_arr`). | Configuration file. | 
| `x_sigma` | positive `float` | Standard deviation for fixed effects covariates (`x_arr`). | Configuration file. | 
| `z_mu` | `float` | Mean for random effects covariates (`z_arr`). | Configuration file. | 
| `z_sigma` | positive `float` | Standard deviation for random effects covariates (`z_arr`). | Configuration file. | 
| `x_arr`   | `jax.numpy.array` | Fixed effects covariate array. Because the covariate data is sample-specific, the array will have dimensions (`n_samps`, $p$) for a model with $p$ fixed effects. The first column will be all ones to account for the intercept. | Sampled. |
| `z_arr`   | `jax.numpy.array` | Random effects covariate array. Because the covariate data is sample-specific, the array will have dimensions (`n_samps`, $q$) for a model with $q$ random effects. | Sampled. | 
| `fx_effs_arr` | `jax.numpy.array`       | Fixed effects coefficient array. The fixed effects are assumed to be target-specific (put another way, they are random effects with infinite variance), so the array dimensions will be (`n_targs`, $p$) for a model with $p$ fixed effects (including intercepts). | Sampled. |
| `rd_effs_arr` | `jax.numpy.array`       | Random effect coefficient array. The random effects are necessarily target-specific, so the array dimensions will be (`n_targs`, $q$) for a model with $q$ random effects. If the variance components are all zero (i.e. $H_0$ is true), then this array will be all zeros. | Sampled. | 
| `targ_idx` | `list` of non-negative `int` | Row indices of non-zero array counts. Length will match that of `samp_idx` and `counts`. | Computed. | 
| `samp_idx` | `list` of non-negative `int` | Column indices of non-zero array counts. Length will match that of `targ_idx` and `counts`. | Computed. | 
| `counts` | `list` of positive `int` | Counts in non-zero array cells. Length will match that of `targ_idx` and `samp_idx`. | Computed. | 
| `eta_arr` | `jax.numpy.array` | Array of linear predictors for each observation. The array will have dimensions (`n_targs`, `n_samps`). | Computed. | 
| `mu_arr` | `jax.numpy.array` | Array of means for each observation. The array will have dimensions (`n_targs`, `n_samps`). | Computed. | 
| `var_arr` | `jax.numpy.array` | Array of variances for each observation. The array will have dimensions (`n_targs`, `n_samps`). | Computed. | 

The actual count tables are stored as three lists to account for sparsity. These lists (`targ_idx`, `samp_idx`, and `counts`) record the indices and values of non-zero cells within the count tables. The full table can be constructed with the `simul_data.Data` method `construct_y()`; however, its use is discouraged for very high-dimensional tables. 


#### Negative Binomial
This class samples data according to a negative binomial generalized linear mixed model. The negative binomial setting has the following additional attributes.

| Name      | Type         | Description     | Source | 
| :---      |    :---     | :---     | :--- | 
| `disp_mu` | positive `float` | Mean for scale/dispersion parameter(s). | Configuration file. | 
| `disp_sigma` | non-negative `float` | Standard deviation for scale/dispersion parameter(s). | Configuration file. | 
| `gamma` | `jax.numpy.array` | Array of scale/dispersion parameters. This is permitted to be target-specific if `disp_sigma` is set to be non-zero. The array will have dimensions (`n_targs`,). | Sampled. | 
| `probs` | `jax.numpy.array` | Success probability for each observation. The array will have dimensions (`n_targs`, `n_samps`). | Computed. |
| `succs` | `jax.numpy.array` | Number of successes for each negative binomial draw. This is equal to `gamma`, so the array will have dimensions (`n_targs`,). | Computed. |

The link function for a negative binomial GLMM is $\log(\cdot)$, so `mu_arr` is computed as the exponential function of `eta_arr` element-wise. The variance function is $v(\mu_{i,j}) = \mu_{i,j} + \frac{1}{\gamma_i} \mu_{i,j}^2$ where the dispersion paramters may or may not be target-specific. 

Because we parametrize the distribution via the mean and variance, we have to work backwards to the probabilities and number of successes needed to sample the data. Letting $r_{i,j}$ be the number of successes (`succs`) and $\pi_{i,j}$ (`probs`) the success probability for observation $i,j$, we have the following relationship:

$$
\begin{aligned}
r_{i,j} &= \frac{\mu_{i,j}^2}{v(\mu_{i,j}) - \mu_{i,j}} = \gamma_i \
\pi_{i,j} &=  \frac{\mu_{i,j}}{v(\mu_{i,j})} = \frac{\gamma_i}{\gamma_i + \mu_{i,j}}
\end{aligned}
$$


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




### Real Data
The real data child class is implemented in `splash_data.py`. 






----

## Model
The parent class is implemented in `model.py` as `Model`. It has a single child class: `neg_bin_model.py`.