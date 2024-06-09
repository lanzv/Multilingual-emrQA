#!/bin/sh
#SBATCH -J med_trans
#SBATCH -o scripts/slurm_outputs/translate_medication.out
#SBATCH -p gpu-troja
#SBATCH -G 1
#SBATCH --mem-per-gpu=50G
#SBATCH --nodelist=tdll-3gpu2
 

python translate.py --translation True --topics "medication"
