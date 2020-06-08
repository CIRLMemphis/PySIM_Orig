#!/bin/bash
#SBATCH --partition gpuq
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3000M
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=cvan@memphis.edu

# run matlab program via the run_matlab script
python -u train.py
