# Model (`model.py`)
The parent class is implemented in `model.py` as `Model`. It has the following child classes: `neg_bin_model.py`. Instances of `Model` are built upon a configuration file and an instance of a `Data` child class. The configuration file dictates how the model will be fit, while the `Data` instance provides the necessary data for actually fitting said model. The following attributes are found in all instances of `Model`:

| Name      | Type         | Description     | Source |  Notation |
| :---        |    :---     | :---     | :--- | :---  | 
| `config` | `dict` | Configuration dictionary. | Argument. | — | 
| `data` | instance of `Data` | Data used for model fitting. | Argument. | — |
| `outcome` | `str`       | Outcome distribution of responses. | Configuration file. | — |
| `scale` | `bool`       | Whether to scale the score vector to have unit norm. If `True`, then the information matrix will be similarly scaled. | Configuration file. | — | 
| `use_true` | `bool`       | Whether to use the true values of the model parameters (taken from `data`). This can only be `True` for instances of `simul_data` and will result in _no estimation_ in such a case. | Configuration file. | — | 
| `debug` | `str` | Debugging settings (one of `off`, `error`, `full`). If `off`, no debugging logs will be saved to file. If `full`, all possible debugging logs (e.g. `data` instances, optimization results, etc.) will be saved to file. If `error`, debugging logs will only be saved upon error in model fitting or optimization. | Configuration file. | — | 
| `fx_effs_arr` | `jax.numpy.array`       | Maximum likelihood estimates of fixed effects coefficients. The dimensions of the array are (`n_samps`, `p`) for a model with $p$ fixed effects. | Configuration file. | $$\boldsymbol{\alpha}$$ | 
| `rd_effs_arr` | `jax.numpy.array`       | Value of random effects coefficients under $H_0$ (all zeros). The dimensions of the array are (`n_samps`, `q`) for a model with $q$ random effects. | Computed. | $$\boldsymbol{\beta}$$ | 
| `var_comp` | `jax.numpy.array`       | Random effects variance components under $H_0$ (all zeros). The dimensions of the array are (`q`,) for a model with $q$ random effects. | Computed. | $\boldsymbol{\tau}^2$ | 
| `mu_arr` | `jax.numpy.array` | Estimated mean array under $H_0$. The array dimensions are (`n_targs`, `n_samps`). | Computed. | $$\boldsymbol{\mu}$$ |
| `offsets` | `jax.numpy.array` | Estimated offsets ($\log$ `size_factor` in `glmGamPoi`) array under $H_0$. The array dimensions are (`n_samps`,). | Computed. | $$\mathbf{o}$$ | 

To conduct the score test, several model-specific quantities must be computed and are implemented as child class methods. They are explained in more detail for each child class.


## Negative Binomial Model (`neg_bin_model.py`)
The negative binomial model is implemented in `neg_bin_model.py`. The following attributes are distribution-specific and included for this child class:

| Name      | Type         | Description     | Source | 
| :---        |    :---     | :---     | :--- |
| `disp_arr` | `jax.numpy.array` | Negative binomial dispersion/scale parameter. The array dimensions are (`n_samps`,). If `global_disp` is `True`, then all values are identical. If `global_disp` is set to some particular value, then all values are fixed at the specific value. Otherwise, `disp_arr` is estimated for each target. | Computed (estimated or extracted from `data`). |

The fixed effects coefficients (`fx_effs_arr`) and the dispersion parameters (`disp_arr`) are estimated with `glmGamPoi`, which fits a separate Gamma-Poisson (i.e. negative binomial) GLM to each target's data. The mean for target $i$ in sample $j$ is:

$$
\begin{aligned}
\mu_{i,j} &= \exp\left( \eta_{i,j} + o_{j} \right)
\end{aligned}
$$

where $\eta_{i,j}$ is the standard linear predictor (e.g. $\boldsymbol{\alpha}_i^\top \mathbf{x}_{j}$ for target-specific fixed effects vector, $\boldsymbol{\alpha}_i$, and sample $j$ covariate vector, $\mathbf{x}_j$), and $\xi_j$ is some adjustment made for uneven column sums (sequencing depth in genomics). Currently, $o_j$ (the $\log$ offset for sample $j$) is computed in one of the following ways:

$$
\begin{aligned}
o_j &= \frac{\log\left( \sum_{i = 1}^{N_t} y_{i,j} \right)}{\exp\left(\frac{1}{N_s} \sum_{j = 1}^{N_s} \log\left( \sum_{i = 1}^{N_t} y_{i,j}  \right) \right)} & \left(\text{default in glmGamPoi}\right)
\end{aligned}
$$

where $N_t$ and $N_s$ are the number of targets (`n_targs`) and samples (`n_samps`), respectively. Note that, when run on simulated data, the data-generating mechanism may or may not include this offset term in the linear predictor (see [here](#simulated-data)). (_include why this is okay?_)

The class attributes for the negative binomial model include:

| Name        | Description     | Notation | 
| :---        | :---     | :---   |
| `compute_resid()` | Computes the residuals under $H_0$ as $y_{i,j} - \mu_{i,j}$. | $$y_{i,j} - \mu_{i,j}$$ |
| `compute_g()` | Computes the link function applied to the means under $H_0$ as $\eta_{i,j}$. | $$g(\mu_{i,j})$$ |
| `compute_g_prime()` | Computes the first derivative of each $g(\mu_{i,j})$ with respect to $\mu_{i,j}$ under $H_0$ as $\frac{1}{\mu_{i,j}}$. | $$g'(\mu_{i,j})$$ |
| `compute_g_prime_2()` | Computes the second derivative of each $g(\mu_{i,j})$ with respect to $\mu_{i,j}$ under $H_0$ as $-\frac{1}{\mu^2_{i,j}}$. | $$g''(\mu_{i,j})$$ |
| `compute_v()` | Computes the variance of each observation as $\mu_{i,j} + \frac{1}{\gamma_i} \mu_{i,j}^2$. | $$v(\mu_{i,j})$$ |
| `compute_v_prime()` | Computes the first derivative of each $v(\mu_{i,j})$ with respect to $\mu_{i,j}$ under $H_0$ as $1 + \frac{2}{\gamma} \mu_{i,j}$. | $$v'(\mu_{i,j})$$ |
| `compute_v_prime_2()` | Computes the second derivative of each $v(\mu_{i,j})$ with respect to $\mu_{i,j}$ under $H_0$ as $\frac{2}{\gamma}$. | $$v''(\mu_{i,j})$$ |


---

# Model Fitting (`glmgampoi.py` and `glmmtmb.py`)
There are two ways that the model parameters can be estimated (via maximum likelihood): with `glmGamPoi` or `glmmTMB`. Both involve using `rpy2` as an interface to ther respective `R` packages. 

## `glmGamPoi`
Fitting a model with `glmGamPoi` is implemented in `glmgampoi.py`. The main function is `fit()`, which fits the desired model under the null hypothesis and returns the extracted parameter estimates. A separate Gamma-Poisson generalized linear model is fit to each target's data (with some shared of information to improve efficiency?), which results in target-specific fixed effect coefficient estimates (which coincides with our $H_0$). 

All fixed effects specified in `x` in the data configuration file will be included in the model, and a fixed intercept is _always_ included. In the case of simulated data settings, this means we assume to know the true (null) model structure (i.e. the covariates present in the model).

The models are fit _without_ shrinkage in the dispersion parameter estimation procedure. Furthermore, if `global_disp` in the model configuration is a list of float(s), then the dispersion parameter(s) will not be estimated. In the event that `global_disp` is set to `True`, a shared dispersion parameter will be estimated; otherwise, a target-specific one is estimated. Note that the variance function in `glmGamPoi` is parametrized as:

$$
v(\mu) = \mu + \gamma \mu^2
$$

Therefore, we take the reciprocal when extracting the dispersion parameter estimates from the model.

By default, `glmGamPoi` includes size factors (or offsets when on the $\log$ scale) to account for uneven sequence depth (column sums). These are estimated as the normalized (to unit geometric mean) column sums.  

**NOTE**: Only a model under $H_0$ with a negative binomial outcome can be fit with `glmGamPoi`.


## `glmmTMB`
The `glmmTMB` package allows us much more flexibility in what we can model because it can accommodate mixed effects, but it has limitations due to the way we have to organize the data. We cannot use this sub-module with real data due to memory allocation limitations in the current implementation. 


### `construct_data_frame()`
Because `glmmTMB` fits a mixed effects model, we must structure the data with dummy variables to account for the different targets. This is implemented in `construct_data_frame()`, which first converts the sparse representation of the anchor count table to a dense representation and then reshapes the count array to be of dimensions $(N_s \times N_t, 1)$. 

It then concatenates the counts, desired covariates, and a dummy variable that indicates target membership as columns in a single `pandas` dataframe. If `null_model` is `True`, then no random effects covariates (`z_arr`) are included. 


### `construct_model_formula()`
This function constructs the appropriate formula (`R` syntax) for `glmmTMB` by adding extra terms for target-specific covariates. A target-specific fixed intercept is _always_ included, and all fixed effects are also target-specific. 

Depending upon the model type (under $H_0$ or $H_1$), random effects may or may not be included. If they are included, they are included as `diag(0 + VAR | targ)` to indicate that we want only a random slope for `VAR` (i.e. without an intercept) and that we want a heterogenous but diagonal variance structure (i.e. we enforce the $q$ random slopes to be independent but to have different variances). 


### `fit_model()`
We actually fit the model with a call to `fit_model()`. Currently, only a negative binomial GLMM is implemented. This function is just a wrapper for an `rpy2` call to the `R` package `glmmTMB` with some added documentation upon error. 

If the model optimization program fails to converge and the model configuration `debug` flag is not set to `off`, then the data object and model object will both be saved to file along with details of the failed fit. If the `debug` flag is set to `full`, then the original failed fit will be diagnosed. Then the model will be re-fit with an alternative objective (`BGFS`). This alternative will then be diagnosed, and the results will be saved as well. A message will be printed in this case, but no error will be thrown.


### `extract_fx_effs()`
Due to the structure of the model data frame, extracting the fixed effect coefficient estimates is a bit tricky. We created dummy variables to account for the fact that all of the fixed effects are target-specific, so we must check the names of the estimates and record them appropriately. 

If there is a colon (`:`) in the name, then that is a particular target's slope deviation from the baseline slope, which is attributed to the first target. If the coefficient starts with `targ`, then this is a particular target's intercept deviation from the baseline intercept (`(Intercept)` in the model), which is attributed to the first target as well. 

Thus, for each fixed effect, we need to record just the baseline estimate for the first target, but _add together_ the baseline and the target-specific estimates for all other targets to get the values in the proper format.


### `extract_rd_effs()`
Extracting the random effects posterior modes is much easier. We just record the values that are returned. 

**NOTE**: In `glmmTMB`, the modes are called _conditional_ modes (I believe because we have conditioned on the observed counts), but we prefer to called them the _posterior_ modes because they are...


### `set_map_params()`
One nice aspect of `glmmTMB` is that it allows us to estimate one parameter while holding others fixed by passing values for the `map` argument in the function call. However, these parameters must be formatted for use in `glmmTMB`, which is done with `set_map_params()`.

The manipulations are similar to that in `extract_fx_effs()`. The target-specific slopes must be adjusted to be in reference to a baseline (the first target). All of the values must also be converted to the appropriate vector types in `R`.


### Wrappers
The functions `fit_model()` and `extract_coefs` provide wrappers to run the entire model fitting procedure and extract all model coefficients, respectively.

**NOTE**: In `glmmTMB`, the random effects variance components are returned as _standard deviations_. Thus, we square the values when extracting from the model fit. 


---

# References

Ahlmann-Eltze, C., & Huber, W. (04 2021). glmGamPoi: fitting Gamma-Poisson generalized linear models on single cell count data. Bioinformatics, 36(24), 5701–5702. doi:10.1093/bioinformatics/btaa1009.

Mollie E. Brooks, Kasper Kristensen, Koen J. van Benthem, Arni Magnusson, Casper W. Berg, Anders Nielsen, Hans J. Skaug, Martin Maechler and Benjamin M. Bolker (2017). glmmTMB Balances Speed and Flexibility Among Packages for Zero-inflated Generalized Linear Mixed Modeling. The R Journal, 9(2), 378-400. doi: 10.32614/RJ-2017-066.

