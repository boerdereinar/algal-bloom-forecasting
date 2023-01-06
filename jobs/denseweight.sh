#!/bin/bash

#SBATCH --job-name="dense-weight"
#SBATCH --account=Education-EEMCS-Courses-CSE3000
#SBATCH --partition=compute
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --output="out/slurm-%j.out"
#SBATCH --signal=SIGUSR1@90

# Activate the environment
source /home/${USER}/.bashrc
source activate /scratch/${USER}/algal-bloom/envs/geo

# Set the python path
export PYTHONPATH=/scratch/${USER}/algal-bloom/brp-algal-bloom-forecasting:$PYTHONPATH

# Run the python module
srun python3 -m ${USER}.main \
train DenseWeight \
--root /scratch/${USER}/algal-bloom/data \
--reservoir palmar \
--window-size 0 \
--prediction-horizon 0 \
--train-test-split 0 \
--train-on-validation \
--exclude-processed \
--num-workers 8 \
--default_root_dir /scratch/${USER}/algal-bloom/checkpoints \
--accelerator cpu \
--devices 1 \
--max_epochs 1 \
--disable-logger
