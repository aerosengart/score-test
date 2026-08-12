# Simple Poisson (`poi_simple`)
Linear predictor under $H_0$ is only an intercept. 

## Fixed Fixed Effects (`fixed_fx_effs`)
### `dpoi_mneg_bin_fx_p1_rd_q1_samps100_targs100`

#### Settings
- Poisson outcome
- Negative binomial model
- One fixed effect (intercept) that is the same across targets
- Testing one random slope
- 100 samples
- 100 targets when applicable (directory says `targs100`, but this is only the case for routines that do not vary the target number which are denoted by *)

#### Routines
- *One-/Two-Sided Score Test Comparison (`one_two_sided`)
- Type I Error Comparison by Number of Targets (`type_I_error_by_n_targs`)
- Power Comparison by Number of Targets (`type_II_error_by_n_targs`)



## Varied Fixed Effects (`varied_fx_effs`)
### `dpoi_mneg_bin_fx_p1_rd_q1_samps100_targs100`

#### Settings
- Poisson outcome
- Negative binomial model
- One fixed effect (intercept) that is random (cluster-specific)
- Testing one random slope ("time" covariate at regular intervals and centered)
- 100 samples
- 100 targets when applicable (directory says `targs100`, but this is only the case for routines that do not vary the target number which are denoted by *)

#### Routines
- *One-/Two-Sided Score Test Comparison (`one_two_sided`)
- Type I Error Comparison by Number of Targets (`type_I_error_by_n_targs`)
- Power Comparison by Number of Targets (`type_II_error_by_n_targs`)



## Varied Fixed Effects with Parameters Known (`varied_fx_effs_true`)
### `dpoi_mneg_bin_fx_p1_rd_q1_samps100_targs100`

#### Settings
- Poisson outcome
- Negative binomial model
- One fixed effect (intercept) that is random (cluster-specific)
- Testing one random slope ("time" covariate at regular intervals and centered)
- 100 samples
- 100 targets when applicable (directory says `targs100`, but this is only the case for routines that do not vary the target number which are denoted by *)
- True parameter values are known

#### Routines
- *One-/Two-Sided Score Test Comparison (`one_two_sided`)
- Type I Error Comparison by Number of Targets (`type_I_error_by_n_targs`)
- Power Comparison by Number of Targets (`type_II_error_by_n_targs`)
