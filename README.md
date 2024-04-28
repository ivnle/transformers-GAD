# GD



## Inference
### Step 0: clone this repo and create environment
Clone the repository:
```
git clone git@github.com:jiayuww/GD.git
```
Create a new Conda environment using the provided requirements file. Replace `/path/to/your/env/gd` with the actual path where you want to store your environment:
```
conda env create -f environment.yml --prefix /path/to/your/env/gd
```

Activate the environment:
```
conda activate /path/to/your/env/gd
```
### Step 1: modify configs
Update the configurations in `run_bare.sh` and `run_gcd_build_oracle.sh` to specify: 
- `ITER`: Number of outputs to generate for each example. Set this to `50` to obtain some results before Monday, although typically `100` samples are generated.
- `CACHE_DIR`: Replace `/path/to/where/you/store/hf/models` with the actual path where you store the HuggingFace models. Ensure that there is sufficient storage space. If there are issues, set the global environment variable:
    ```
    export HF_HOME=/path/to/where/you/store/hf/models
    ```
- `GPUS`: modify this to specify the GPU to use. For example, `0` for GPU 0, `GPUS=(0 1)` for GPUs 0 and 1.
- `dtype`: Default is `float32`. Only change this if you encounter model loading issues. Changing it to `float16` should help resolve space issues.

### Step 2: run the script
Execute the scripts in the following order:
```
sh run_gcd_build_oracle.sh
sh run_bare.sh
```
Start with `run_gcd_build_oracle.sh`, then run `run_bare.sh` if time permits.
You can execute these scripts simultaneously in separate terminal windows or sessions (but start with `run_gcd_build_oracle.sh` first!). Both scripts automatically check idle GPUs and automatically start if it's free.

### Step 3: Collect the results
Collect the outputs stored in `OUTPUT_FOLDER` (`results/SLIA`) and tries in `TRIE_FOLDER` (`tries/SLIA`).
