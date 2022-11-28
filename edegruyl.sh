#!/bin/bash

#SBATCH --job-name="algal-bloom"
#SBATCH --account=Education-EEMCS-Courses-CSE3000
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --output="out/slurm-%A_%a.out"
#SBATCH --error="out/slurm-%A_%a.err"

# Activate the environment
source /home/${USER}/.bashrc
source activate /scratch/${USER}/algal-bloom/envs/geo

# Set the python path
export PYTHONPATH=/scratch/${USER}/algal-bloom/brp-algal-bloom-forecasting:$PYTHONPATH

# Run the python module
python3 -m ${USER}.main --help