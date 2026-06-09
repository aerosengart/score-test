# Simulation-Based Routines
This sub-directory contains the functions needed to run simulation-based routines and plot the results.

We have a few helper files that aid in avoiding code duplication and organize the workflow. Dispatch dictionaries are found in `dispatch.py`, which specify which particular functions to call when a user specifies a particular routine by its shortened name in a routine configuration file. The `doc.py` file contains all functions related to documenting a routine. This includes saving intermediate results, recording the routine specification, and ensuring the appropriate output sub-directories exist.

---

## Empirical Distributions (`emp_dists.py`)
This file implements routines that involve the empirical distribution of some quantity. 

