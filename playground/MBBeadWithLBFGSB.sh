#!/bin/bash
#SBATCH --partition gpuq
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2000M
#SBATCH --time=0-1:00:00

# run matlab program via the run_matlab script
python MBBeadWithLBFGSB.py
