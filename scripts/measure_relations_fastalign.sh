#!/bin/sh
#SBATCH -J fa_rel_align
#SBATCH -o scripts/slurm_outputs/align_relations_fastalign.out
#SBATCH -p cpu-ms
#SBATCH --mem-per-cpu=50G


python measure_aligners.py --dataset_title "relations" --aligner_name "FastAlign" --aligner_path "../models/fast_align/build/czeng/czeng_relations.txt" --translation_dataset "./data/translations/relations_cs.json"
