#!/bin/bash

#SBATCH --job-name="algal-bloom"
#SBATCH --account=Education-EEMCS-Courses-CSE3000
#SBATCH --partition=compute
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=4G
#SBATCH --output="out/slurm-%j.out"

# Activate the environment
source /home/${USER}/.bashrc
source activate /scratch/${USER}/algal-bloom/envs/geo

# Set the python path
export PYTHONPATH=/scratch/${USER}/algal-bloom/brp-algal-bloom-forecasting:$PYTHONPATH

# Run the python module
srun python3 -m ${USER}.main \
preprocess Interpolate \
-s /scratch/${USER}/algal-bloom/data/biological \
-t /scratch/${USER}/algal-bloom/data/biological_processed \
-st LookBack \
--num-workers 3
