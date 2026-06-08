This module contains all of the methods related to the score test computations.

---

# Intermediate Quantities (`intermed.py`)
This file 

---


# Score (`score.py`)
This file contains the methods related to computing the score test statistic and the score vector. We describe these functions in detail in later sub-sections. For easy reference, see the table below. 

| Name      | Description     | Notation |
| :---        |   :---     | :--- |
| `compute_score_var_comp()` | Compute the first derivative of the (full data) marginal log quasi-likelihood with respect to the variacnce component(s) and evaluate at the parameter values/estimates under $H_0$. | $$U_{\boldsymbol{\tau}^2}(\boldsymbol{\theta})$$ |
| `compute_test_stat()` | Compute the score test statistic for variance component(s) in a generalized linear mixed model. The test can be one- or two-sided depending upon the `one_sided` flag. | $$t$$ |


## `compute_score_var_comp()`
Arguments include:

| Name      | Type         | Description     | Default | 
| :---        |    :---     | :---     | :--- |
| `model` | `Model` instance | Instance of `Model` child class | — |
| `scale` | `bool` | Whether to scale the score vector to have unit norm. | `True` |

Returned quantities include:

| Name      | Type         | Description    | 
| :---        |    :---     | :---     |
| `wrt_tau_sqr` | `jax.numpy.array` | Vector of first-order partial derivatives of the full marginal log quasi-likelihood with respect to the coordinates of the random effects variance components evaluated at the parameter values/estimates under $H_0$. Dimensions will be (`q`,) for a model with $q$ random effects. | 
| `scale_val` | `float` | Scaling value used on the score vector. Equal to $1$ if `scale` is set to `False`. |

This function computes the first-order partial derivatives of the full data's marginal log quasi-likelihood with respect to the variance components ($\boldsymbol{\tau}^2$) and evaluates the resulting vector at the parameter values specified by `model`. The computation is given by Eq. (9) in Lin and Breslow (1997). Let $\tau^2_k$ denotes the $k$-th coordinate of $\boldsymbol{\tau}^2$, let $\hat{\boldsymbol{\theta}}$ denote all other model parameter values under $H_0$ (e.g. for the fixed effect coefficients, these are the MLEs under $H_0$, and for the random effects coefficients, these are zeros), and let a hat denote a quantity evaluated at $\boldsymbol{\theta} = \hat{\boldsymbol{\theta}}$. The component of the score corresponding to $\tau^2_k$ is given by:

$$
U_{\tau^2_k}(\hat{\boldsymbol{\theta}}) = \frac{1}{2} \sum_{i = 1}^{N_t} \left[ (\mathbf{y}_i -\hat{\boldsymbol{\mu}}_i)^\top \hat{\boldsymbol{\Delta}}_i^{-1} \hat{\boldsymbol{\Omega}}_i \mathbf{Z}_i  \left. \left[ \frac{\partial \boldsymbol{\Sigma} }{\partial \tau^2_k} \right] \right\rvert_{\boldsymbol{\tau}^2 = \mathbf{0}_q}  \mathbf{Z}_i^\top \hat{\boldsymbol{\Omega}}_i \hat{\boldsymbol{\Delta}}_i^{-1} (\mathbf{y}_i - \hat{\boldsymbol{\mu}}_i) - \text{tr} \left[ \hat{\boldsymbol{\Xi}} \mathbf{Z}_i \left. \left[ \frac{\partial \boldsymbol{\Sigma} }{\partial \tau^2_k} \boldsymbol{\Sigma} \right] \right\rvert_{\boldsymbol{\tau}^2 = \mathbf{0}_q} \boldsymbol{Z}_i^\top \right]\right]
$$


## `compute_test_stat()`
Arguments include:

| Name      | Type         | Description     | Default | 
| :---        |    :---     | :---     | :--- |
| `model` | `Model` instance | Instance of `Model` child class | — |
| `one_sided` | `bool` | Whether to compute the test statistic against a one-sided alternative. | `False` |

Returned quantitites include:

| Name      | Type         | Description    | 
| :---        |    :---     | :---     |
| `test_stat` | `float` | Test statistic value. | 


The two-sided test statistic is computed as (see Eq. (12) in Lin and Breslow (1997)):

$$
t = U^\top_{\boldsymbol{\tau}^2}(\hat{\boldsymbol{\theta}}) \tilde{I}^{-1}_{\boldsymbol{\tau}^2, \boldsymbol{\tau^2}}(\hat{\boldsymbol{\theta}}) U_{\boldsymbol{\tau}^2}(\hat{\boldsymbol{\theta}})
$$

where $U_{\boldsymbol{\tau}^2}(\hat{\boldsymbol{\theta}})$ (see `compute_score_var_comp()`) is the sub-vector of the score vector corresponding to the variance components and $\tilde{I}^{-1}_{\boldsymbol{\tau}^2, \boldsymbol{\tau^2}}(\hat{\boldsymbol{\theta}})$ is the inverse of the block of the Fisher information matrix corresponding to the variance components, both evaluated at the parameter values/estimates under $H_0$. The Fisher information block has the form:

$$
\tilde{I}^{-1}_{\boldsymbol{\tau}^2, \boldsymbol{\tau^2}}(\hat{\boldsymbol{\theta}})  = I_{\boldsymbol{\tau}^2, \boldsymbol{\tau^2}}(\hat{\boldsymbol{\theta}}) - I^\top_{\boldsymbol{\alpha}, \boldsymbol{\tau}^2} (\hat{\boldsymbol{\theta}})  I^{-1}_{\boldsymbol{\alpha}, \boldsymbol{\alpha}} (\hat{\boldsymbol{\theta}})  I_{\boldsymbol{\alpha}, \boldsymbol{\tau}^2} (\hat{\boldsymbol{\theta}})
$$

Even though there may be additional, distribution-specific parameters (e.g. $\gamma$ for negative binomial) that should be included in the score vector (and therefore also the information matrix), the computations quickly become unwieldy. In practice, the test appears to work fairly well without the adjustment, so we omit them for now.

The one-sided test statistic is given by:

TODO


----

# References


Lin, X. (1997). Variance component testing in generalised linear models with random effects. Biometrika, 84(2), 309–326. doi:10.1093/biomet/84.2.309

Silvapulle, M. J., & Silvapulle, P. (1995). A Score Test Against One-Sided Alternatives. Journal of the American Statistical Association, 90(429), 342–349. doi:10.2307/2291159
