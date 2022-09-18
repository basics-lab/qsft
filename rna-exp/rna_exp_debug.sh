#!/bin/bash
#SBATCH --job-name=rna_exp
#SBATCH --account=fc_basics
#SBATCH --partition=savio2
#SBATCH --time=00:10:00
#SBATCH --output=slurm_outputs/rna_exp_debug_%j.out
module load python
source activate qspright
ipython -V
which python
ipython qspright-sample-vs-estimation-accuracy.ipynb
