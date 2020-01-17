#!/bin/bash
#SBATCH --partition gpuq
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=400M
#SBATCH --time=0-0:10:00

# run matlab program via the run_matlab script
python testTF2.py
