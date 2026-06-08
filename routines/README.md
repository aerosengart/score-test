# Running Routines via Command Line
Command line wrappers for running routines. The files in this sub-directory provide Python programs that run and plot the results of several different routines via command line arguments. 

## Simulations
To run routines that rely on simulations, we use the following syntax.

### Usage
`python [path to program file] --cfg [path to routine config file] --iter_id [iteration ID as integer]`

Here, `[path to program file]` is the path to some program in this sub-directory (one of `plot_routine.py`, `run_routine.py`, or `splash_score_test.py` currently). The `[path to routine config file]` is a `.yaml` file specifying the configuration for the routine and should be the path to a file in the `configs` sub-directory of this sub-directory. Finally, `iter_id` is a label used for saving the results of the run. It should be a non-negative integer. Results will be saved as `results_iter[iter_id].npz` or something similar (like a `.pickle` file) and will overwrite any existing results with the same name.

The configuration file should specify:

- `d_config`: absolute path to data configuration file
- `m_config`: absolute path to model configuration file
- `routine`: routine string
- `reps`: number of replicates to draw

### Example
`python routines/run_routine.py --cfg routines/configs/test.yaml --iter_id 0`

The configuration file specifies that we want to run the routine that approximates the variance component's empirical distribution.

```
d_config: '/home/arosenga/score-test/data/configs/d_test.yaml'
m_config: '/home/arosenga/score-test/data/configs/m_test.yaml'
routine: 'score_var_comp'
reps: 100
```

We can plot the results with a similar command using _the same_ configuration file and iteration ID.

`python routines/plot_routine.py --cfg routines/configs/test.yaml --iter_id 0`

## Real Data
To run routines on real data (`splash_score_test.py`), we have a slightly different syntax. 

### Usage
`python [path to program file] --cfg [path to routine config file] --anchor [anchor sequence]`

The configuration file should specify:

- `d_config`: absolute path to data configuration file
- `m_config`: absolute path to model configuration file
- `output`: absolute path to output directory (where to save results)

