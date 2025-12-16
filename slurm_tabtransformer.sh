#!/bin/bash
#SBATCH --job-name=tab_transformer
#SBATCH --partition=gpu
#SBATCH --output=outputs/job_%j.out
#SBATCH --error=outputs/job_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --time=02:00:00

# Load modules
module load python/3.11
module load cuda/12.0

# Create outputs directory if it doesn't exist
mkdir -p /home/bumu60du/transformers_ovgu/outputs

# Activate virtual environment
source /home/bumu60du/transformers_ovgu/venv/bin/activate

# Run training
cd /home/bumu60du/transformers_ovgu
python3 main.py