#!/bin/sh
#SBATCH -J measure_translators
#SBATCH -o scripts/slurm_outputs/measure_translators_ml7.out
#SBATCH -p gpu-ms
#SBATCH -G 1
#SBATCH --mem-per-gpu=50G
#SBATCH --nodelist=dll-3gpu4
 
for i in "NLLB_600M" "NLLB_1_3B_dis" "NLLB_1_3B" "MadLad_3B" "NLLB_3_3B" "LINDAT" "MadLad_7B" "MadLad_10B" "NLLB_54B"
do
python measure_translators.py --data_dir="../datasets/khresmoi-summary-test-set-2.0" --models_dir="../models" --model=$i
done