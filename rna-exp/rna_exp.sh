#!/bin/bash
#SBATCH --job-name=rna_exp
#SBATCH --account=fc_basics
#SBATCH --partition=savio2
#SBATCH --time=24:00:00
#SBATCH --output=slurm_outputs/rna_exp_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=erginbas@berkeley.edu

python run-tests-complexity-vs-size.py --jobid=$SLURM_JOB_ID \
--num_subsample 2 3 4 --num_repeat 1 2 4 8  --b 3 4 5 6 7 8 9 --n 15 --iters 1