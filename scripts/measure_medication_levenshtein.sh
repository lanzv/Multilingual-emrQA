#!/bin/sh
#SBATCH -J le_med_align
#SBATCH -o scripts/slurm_outputs/align_medication_levenshtein.out
#SBATCH -p gpu-troja
#SBATCH -G 1
#SBATCH --mem-per-gpu=50G
#SBATCH --nodelist=tdll-3gpu4

python measure_aligners.py --dataset_title "medication" --aligner_name "Levenshtein" --aligner_path "../models/madlad400-3b-mt" --translation_dataset "./data/translations/medication_cs.json"
