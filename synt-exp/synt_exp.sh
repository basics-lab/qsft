#!/bin/bash
#SBATCH --job-name=synthetic_exp
#SBATCH --account=fc_basics
#SBATCH --partition=savio2
#SBATCH --time=10:00:00
#SBATCH --output=slurm_outputs/%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=erginbas@berkeley.edu

python qspright-sample-vs-nmse.py --jobid=$SLURM_JOB_ID \
--num_subsample 2 4 6 --num_repeat 2 4 6 --b 6 7 8 \
--a 1 --noise_sd 1e-3 --n 15 --q 4 --sparsity 100 --iters 5
