#!/bin/sh
#SBATCH -J rel_trans
#SBATCH -o scripts/slurm_outputs/translate_relations.out
#SBATCH -p gpu-troja
#SBATCH -G 1
#SBATCH --mem-per-gpu=50G
#SBATCH --nodelist=tdll-3gpu3
 

python translate.py --translation True --topics "relations"
