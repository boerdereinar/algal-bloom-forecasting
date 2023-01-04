#!/bin/bash

#SBATCH --job-name="algal-bloom"
#SBATCH --account=Education-EEMCS-Courses-CSE3000
#SBATCH --partition=gpu
#SBATCH --time=1:00:00
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

# Set the checkpoint path
CHECKPOINT="<checkpoint_path>"

# Run the python module
srun python3 -m ${USER}.main \
test UNet \
--checkpoint-path $CHECKPOINT \
--save-dir /scratch/${USER}/algal-bloom/plots \
--accelerator gpu \
--devices 1
