#!/bin/sh
#SBATCH -J le_rel_align
#SBATCH -o scripts/slurm_outputs/align_relations_levenshtein.out
#SBATCH -p gpu-troja
#SBATCH -G 1
#SBATCH --mem-per-gpu=50G
#SBATCH --nodelist=tdll-3gpu3


python measure_aligners.py --dataset_title "relations" --aligner_name "Levenshtein" --aligner_path "../models/madlad400-3b-mt" --translation_dataset "./data/translations/relations_cs.json"