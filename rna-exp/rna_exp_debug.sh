#!/bin/bash
#SBATCH --job-name=rna_exp
#SBATCH --account=fc_basics
#SBATCH --partition=savio2
#SBATCH --time=00:10:00
#SBATCH --output=slurm_outputs/debug/rna_exp_debug_%j.out
ipython -V
which python
ipython qspright-sample-vs-estimation-accuracy.ipynb