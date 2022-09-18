#!/bin/bash
#SBATCH --job-name=rna_exp
#SBATCH --account=fc_basics
#SBATCH --partition=savio
#SBATCH --time=04:00:00
#SBATCH --output=rna_exp_%j.out
#SBATCH --error=rna_exp_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=erginbas@berkeley.edu
module load python
source activate generalized-wht
jupyter nbconvert --to script ../qspright-sample-vs-estimation-accuracy.ipynb
python ../qspright-sample-vs-estimation-accuracy.py