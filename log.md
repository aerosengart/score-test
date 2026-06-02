For Viana data, I have run the following:

---

## 2026 May 26

### Runs 

#### Run 1
Run score test on the first 500 anchors in the filtered anchor list. Results are in `anchors_0_500`.

`sbatch slurm_scripts/anchors_0_500.sh data/configs/d_viana.yaml data/configs/m_viana.yaml analyses/viana/2026_05_19/anchor_list.txt anchors_0_500`

Four anchors failed: 23, 30, 51, 436 (I believe this is 1-indexed). See `slurm-7067_*.out` in the output directory for more information.

#### Run 2
Ran score test on the hand-selected anchor list (`test_anchors.txt`). All but the last three anchors in that file were picked up at statistically significant by OASIS.

`sbatch slurm_scripts/anchors_test.sh data/configs/d_viana.yaml data/configs/m_viana.yaml analyses/viana/2026_05_19/test_anchors.txt anchors_test`

One anchor failed: 51. See `slurm-7567_51.out` in output directory for more information. 

---

## 2026 May 27

### Run 1
Re-run score test on the anchors that failed. 

`sbatch --array=23,30,51,436 slurm_scripts/anchors_filt.sh data/configs/d_viana.yaml data/configs/m_viana.yaml analyses/viana/2026_05_19/anchor_list.txt anchors_0_500`

Three anchors failed: 23, 30, 436. I deleted the slurm output files because they were the same errors as before. I looked at the data manually, and I think it is because there are not enough samples.

- Anchor 23  (ACATGCTTCCTGTTGTTAGGGAGGGTT): 2 samples and 51 targets
- Anchor 30  (AGAGTCGTCTTCGTCAAATTTGCAGCA): 2 samples and 40 targets
- Anchor 436 (TGAAGAGCAACCAATGGAGATTGACTA): 2 samples and 30 targets

### Run 2
Run score test on the next 500 anchors in the filtered anchor list. Results are in `anchors_501_1000`.

`sbatch --array=501-1000 slurm_scripts/anchors_filt.sh data/configs/d_viana.yaml data/configs/m_viana.yaml analyses/viana/2026_05_19/anchor_list.txt anchors_501_1000`

Two anchors failed: 813, 950. 

- Anchor 813 (TGATCATTTTCGCATTTATCGTGAAAC): 3 samples and 128 targets
- Anchor 950 (TGTGGATCCTGCTGCAAATTTGACGAA): 2 samples and 103 targets


### Run 3
Run score test on the next 1000 anchors in the filtered anchor list. Results are in `anchors_1001-2000`. We have to use a new shell script because the maximum slurm array ID is 1001.

`sbatch slurm_scripts/anchors_1001_2000.sh data/configs/d_viana.yaml data/configs/m_viana.yaml analyses/viana/2026_05_19/anchor_list.txt anchors_1001_2000`


### Other Notes
Moved `slurm_scripts/anchors_1001_2000.sh` and `slurm_scripts/anchors_0_500.sh` to `slurm_scripts/old` and deleted output (`anchors_1001_2000` and `anchors_501_1000` and `anchors_0_500`) because I figured out how to do a loop and submit all of the jobs at one time. 

Edited `slurm_scripts/anchors_filt.sh` to accept a fifth argument that defines the offset to add to the slurm array ID. This allows us to run more than 1001 jobs but still specify numbers less than 1001 in `--array`. 

### Run 4
Ran score test on the first 1000 anchors in the filtered anchor list (i.e. anchor lines 1-1000) by manually adjusting the loop in `shell_scripts/all_anchors_job.sh`. Results are in `analyses/viana/2026_05_19/score_test`. 

```
bash shell_scripts/all_anchors_job.sh -a "analyses/viana/2026_05_19/anchor_list.txt" -d "data/configs/d_viana.yaml" -m "data/configs/m_viana.yaml" -o "analyses/viana/2026_05_19"
find . -type f -name '*.out' -size 0b -delete
```

Moved slurm output files for jobs with errors to `analyses/viana/2026_05_19/score_test` manually.

---

## 2026 May 28

#### Run 1
Ran score test on the next 4000 anchors in the filtered anchor list (i.e. anchor lines 1001-5000) by manually adjusting the loop in `shell_scripts/all_anchors_job.sh`. Results are in `analyses/viana/2026_05_19/score_test`. 

```
bash shell_scripts/all_anchors_job.sh -a "analyses/viana/2026_05_19/anchor_list.txt" -d "data/configs/d_viana.yaml" -m "data/configs/m_viana.yaml" -o "analyses/viana/2026_05_19"
find . -type f -name '*.out' -size 0b -delete
```

Moved slurm output files for jobs with errors to `analyses/viana/2026_05_19/score_test` manually.

---

## 2026 May 31

#### Analyses
Ran power analyses over different choices of sample size and values of $\tau^2$. Results are in `/home/arosenga/score-test/test_routine/dneg_bin_mneg_bin_fx_p1_rd_q1_samps100_targs100_dgamma1.0`. Power appears to be consistently greater for the two-sided test. 

Ran analysis of test statistic sensitivity to fixed effects values with $\gamma$ known. Results are in `/home/arosenga/score-test/test_routine/dneg_bin_mneg_bin_fx_p1_rd_q1_samps100_targs200_dgamma1.0/test_stat_fx_effs_sens_disp_known`. Distribution of one-sided test appears to deviate quite strongly from the expected $\bar{\chi}^2$ while the two-sided test looks pretty good. 


---

## 2026 June 01

#### Updates
I edited the modules to use the estimated mean (under $H_0$) array returned by `glmGamPoi` rather than manual computation of the asqrray by first computing the linear predictor with the fixed effects MLEs. This may result in something different because there is a `size_factor` term in the implementation of `glmGamPoi` that accounts for differences in sequencing depth between samples (e.g. differences in column sums). I re-ran the one-/two-sided score test comparison to check the effect.

NOTE: This overwrote the previous results in `/home/arosenga/score-test/test_routine/dneg_bin_mneg_bin_fx_p1_rd_q1_samps100_targs100_dgamma1.0/one_two_sided`.

I added `estim_mu` and `global_disp` to the model configuration .yaml files. The first is a flag to use the estimated `Mu` from `glmGamPoi`, which is the estimated mean array under $H_0$. The latter is a flag to estimate a shared dispersion parameter ($\gamma$ for negative binomial) for all targets. 

I updated the routine documentation function to also record the model configuration .yaml. 

#### File Management
I deleted the sub-directory `/home/arosenga/score-test/analyses/viana/2026_05_19` because the results were computed using an outdated version of SPLASH.

I deleted the sub-directory `/home/arosenga/score-test/test_routine` because I have updated many of the routine functions.

#### Analyses
I ran the one-/two-sided QQ plot comparison with the following settings:

- target-specific $\gamma$, estimated $\mu$ (extracted from `glmGamPoi`)
- target-specific $\gamma$, constructed $\mu$ (by computing linear predictor, $\eta$)
- global $\gamma$, constructed $\mu$

I ran the power analysis over values of $\tau^2$ for fixed sample size and number of targets. 
