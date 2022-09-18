#!/bin/bash
# Job name:
#SBATCH --job-name=rna_exp
#
# Account:
#SBATCH --account=fc_basics
#
# Partition:
#SBATCH --partition=savio
#
# Wall clock limit:
#SBATCH --time=04:00:00
#
## Command(s) to run:
module load python
source activate generalized-wht
jupyter nbconvert --to script qspright-sample-vs-estimation-accuracy.ipynb
python qspright-sample-vs-estimation-accuracy.py