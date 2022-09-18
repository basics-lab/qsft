#!/bin/bash
#SBATCH --job-name=rna_exp
#SBATCH --account=fc_basics
#SBATCH --partition=savio
#SBATCH --time=00:10:00
#SBATCH --output=slurm_outputs/rna_exp_debug_%j.out
#SBATCH --qos=savio_debug
module load python
source activate generalized-wht
ipython qspright-sample-vs-estimation-accuracy.ipynb
