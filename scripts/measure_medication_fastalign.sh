#!/bin/sh
#SBATCH -J 1fa_med_align
#SBATCH -o scripts/slurm_outputs/align_medication_fastalign1.out
#SBATCH -p cpu-ms
#SBATCH --mem-per-cpu=50G

python measure_aligners.py --dataset_title "medication" --aligner_name "FastAlign" --aligner_path "../models/fast_align/build/czeng/czeng_medication.txt" --translation_dataset "./data/translations/medication_cs.json"
