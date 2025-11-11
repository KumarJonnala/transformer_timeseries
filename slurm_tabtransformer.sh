#!/bin/bash
#SBATCH --job-name=tab_transformer
#SBATCH --partition=gpu
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

#SBATCH --gres=shard:4,gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --time=01:00:00

srun python3 main.py