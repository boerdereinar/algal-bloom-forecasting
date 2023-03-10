#!/bin/bash

#SBATCH --job-name="algal-bloom"
#SBATCH --account=Education-EEMCS-Courses-CSE3000
#SBATCH --partition=gpu
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
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
train UNet \
--root /scratch/${USER}/algal-bloom/data \
--reservoir palmar \
--window-size 5 \
--prediction-horizon 1 \
--size 256 \
--batch-size 1 \
--length 1000 \
--num-workers 8 \
--default_root_dir /scratch/${USER}/algal-bloom/checkpoints \
--accelerator gpu \
--devices 1 \
--max_epochs 10 \
--overfit
