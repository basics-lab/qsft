#!/bin/bash
#SBATCH --job-name=rna_exp
#SBATCH --account=fc_basics
#SBATCH --partition=savio
#SBATCH --time=04:00:00
#SBATCH --output=rna_exp_debug_%j.out
#SBATCH --error=rna_exp_debug_%j.err
#SBATCH --qos=savio_debug
module load python
source activate generalized-wht
jupyter nbconvert --to script qspright-sample-vs-estimation-accuracy.ipynb
python qspright-sample-vs-estimation-accuracy.py