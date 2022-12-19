#!/bin/bash
#SBATCH --job-name=rna_exp
#SBATCH --account=fc_basics
#SBATCH --partition=savio2
#SBATCH --time=10:00:00
#SBATCH --output=slurm_outputs/rna_exp_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=erginbas@berkeley.edu

#python qspright-sample-vs-nmse.py --jobid=$SLURM_JOB_ID \
#--num_subsample 2 4 6 --num_repeat 6 8 10 15 20  --b 7 8 \
#--noise_sd 5e-8 --n 15 --iters 5

python qspright-sample-vs-nmse.py --jobid=13554482 \
--num_subsample 4 6 --num_repeat 6 10  --b 7 8 \
--noise_sd 8e-8 5e-8 --n 25 --iters 1