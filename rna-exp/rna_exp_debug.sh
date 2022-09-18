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
#SBATCH --time=00:10:00
#
# QOS:
#SBATCH --QOS=savio_debug
#
## Command(s) to run:
jupyter nbconvert --execute --to notebook qspright-sample-vs-estimation-accuracy.ipynb