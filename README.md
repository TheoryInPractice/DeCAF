## How to reproduce paper experiments

### Requirements

- Python 3 with the following libraries
  - numpy
  - pandas
  - networkx
  - scipy
  - gurobi
  - sympy
  - matplotlib
- Gurobi Optimizer 9.1.2

### Datasets

Follow [the instructions for cricca](cricca.md) to generate input files, and store them in the following directories.

- `data/pre_preprocessing/lv`
- `data/pre_preprocessing/tf`

### Experiments

#### 1. Preprocessing

- In this directory run the following command:
  - `python3 src/exp_preprocess.py --first_seed 0 --last_seed 19`
- The directory `data/post_preprocessing` will be created.


#### 2. Kernelization

- In this directory run the following command:
  - `python3 src/exp_runkernel.py --first_seed 0 --last_seed 9 --v2`
  - Run `python3 src/exp_runkernel.py --help` for more options.
- The directory `data/post_kernels_v2` will be created.


#### 3. Decomposition

- In this directory run the following command:
  - `python3 src/exp_runbsd.py --first_seed 0 --last_seed 0 --lp-only --kernel-v2 --perf-lp-v2`
  - Run `python3 src/exp_runbsd.py --help` for more options.
- The directory `data/finaldata_kv2_pv2` will be created.
- Output files can be parsed by `src/print_pkl.py`.

