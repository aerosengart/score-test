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
| `size_factors` | `str` | What adjustment to use for the `size_factors` argument in `glmGamPoi` (one of `col_sums` or `default`). If `col_sums`, then the natural logarithm of the column sums is added as a term to the model during fitting. If `default`, the default setting of `glmGamPoi` is used (a normalized version of `col_sums`). | Configuration file. | — | 
| `fx_effs_arr` | `jax.numpy.array`       | Maximum likelihood estimates of fixed effects coefficients. The dimensions of the array are (`n_samps`, `p`) for a model with $p$ fixed effects. | Configuration file. | $$\boldsymbol{\alpha}$$ | 
| `rd_effs_arr` | `jax.numpy.array`       | Value of random effects coefficients under $H_0$ (all zeros). The dimensions of the array are (`n_samps`, `q`) for a model with $q$ random effects. | Computed. | $$\boldsymbol{\beta}$$ | 
| `tau_sqr` | `jax.numpy.array`       | Random effects variance components under $H_0$ (all zeros). The dimensions of the array are (`q`,) for a model with $q$ random effects. | Computed. | $\boldsymbol{\tau}^2$ | 
| `mu_arr` | `jax.numpy.array` | Estimated mean array under $H_0$. The array dimensions are (`n_targs`, `n_samps`). | Computed. | $$\boldsymbol{\mu}$$ |
| `offsets` | `jax.numpy.array` | Estimated offsets (`size_factor` in `glmGamPoi`) array under $H_0$. The array dimensions are (`n_samps`,). | Computed. | $$\mathbf{o}$$ | 

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

TODO


---

# References

Ahlmann-Eltze, C., & Huber, W. (04 2021). glmGamPoi: fitting Gamma-Poisson generalized linear models on single cell count data. Bioinformatics, 36(24), 5701–5702. doi:10.1093/bioinformatics/btaa1009

Baharav, T. Z., Tse, D., & Salzman, J. (2024). OASIS: An interpretable, finite-sample valid alternative to Pearson’s \textitX$^\textrm2$ for scientific discovery. Proceedings of the National Academy of Sciences, 121(15), e2304671121. doi:10.1073/pnas.2304671121

Chaung, K., Baharav, T. Z., Henderson, G., Zheludev, I. N., Wang, P. L., & Salzman, J. (2023). SPLASH: A statistical, reference-free genomic algorithm unifies biological discovery. Cell, 186(25), 5440-5456.e26. doi:10.1016/j.cell.2023.10.028

