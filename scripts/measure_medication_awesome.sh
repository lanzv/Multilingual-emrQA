#!/bin/sh
#SBATCH -J aw_med_align
#SBATCH -o scripts/slurm_outputs/align_medication_awesome.out
#SBATCH -p gpu-troja
#SBATCH -G 1
 

python measure_aligners.py --dataset_title "medication" --translation_dataset "./data/translations/medication_cs.json"