#!/bin/sh
#SBATCH -J aw_rel_align
#SBATCH -o scripts/slurm_outputs/align_relations_awesome.out
#SBATCH -p gpu-troja
#SBATCH -G 1
#SBATCH --mem-per-gpu=50G
#SBATCH --nodelist=tdll-3gpu2


python measure_aligners.py --dataset_title "relations" --translation_dataset "./data/translations/relations_cs.json"