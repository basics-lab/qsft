#!/bin/bash
#SBATCH --job-name=rna_exp
#SBATCH --account=fc_basics
#SBATCH --partition=savio2
#SBATCH --time=04:00:00
#SBATCH --output=slurm_outputs/rna_exp_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=erginbas@berkeley.edu
ipython -V
which python
ipython qspright-sample-vs-estimation-accuracy.ipynb