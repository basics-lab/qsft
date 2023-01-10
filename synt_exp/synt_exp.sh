#!/bin/bash
#SBATCH --job-name=synt_exp
#SBATCH --account=fc_basics
#SBATCH --partition=savio2
#SBATCH --time=15:00:00
#SBATCH --output=slurm_outputs/%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=erginbas@berkeley.edu

python run-tests-complexity-vs-size.py --jobid=$SLURM_JOB_ID \
--num_subsample 1 2 --num_repeat 1 2 --b 1 2 3 4 5 6 7 \
--a 1 --snr 20 --n 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 --q 3 --sparsity 10 --iters 5

#python run-tests-nmse-vs-snr.py --jobid=$SLURM_JOB_ID \
#--num_subsample 3 --num_repeat 1 --b 7 \
#--a 1 --n 20 --q 4 --sparsity 100 250 1000 --iters 5
