For Viana data, I have run the following:


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


### 2026 May 28

#### Run 1
Ran score test on the next 4000 anchors in the filtered anchor list (i.e. anchor lines 1001-5000) by manually adjusting the loop in `shell_scripts/all_anchors_job.sh`. Results are in `analyses/viana/2026_05_19/score_test`. 

```
bash shell_scripts/all_anchors_job.sh -a "analyses/viana/2026_05_19/anchor_list.txt" -d "data/configs/d_viana.yaml" -m "data/configs/m_viana.yaml" -o "analyses/viana/2026_05_19"
find . -type f -name '*.out' -size 0b -delete
```

Moved slurm output files for jobs with errors to `analyses/viana/2026_05_19/score_test` manually.