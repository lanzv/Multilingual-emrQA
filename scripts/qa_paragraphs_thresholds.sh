#!/bin/sh
#SBATCH -J qa
#SBATCH -o scripts/slurm_outputs/qa.out
#SBATCH -p cpu-ms









sbatch --job-name=bgmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_bg_med_0.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.00 --dataset './data/translation_aligners/Awesome/medication_bg.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.00 --dataset './data/translation_aligners/Awesome/medication_bg.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.00 --dataset './data/translation_aligners/Awesome/medication_bg.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=bgmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_bg_med_5.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.05 --dataset './data/translation_aligners/Awesome/medication_bg.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.05 --dataset './data/translation_aligners/Awesome/medication_bg.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.05 --dataset './data/translation_aligners/Awesome/medication_bg.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=bgmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_bg_med_10.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.10 --dataset './data/translation_aligners/Awesome/medication_bg.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.10 --dataset './data/translation_aligners/Awesome/medication_bg.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.10 --dataset './data/translation_aligners/Awesome/medication_bg.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=bgmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_bg_med_15.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.15 --dataset './data/translation_aligners/Awesome/medication_bg.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.15 --dataset './data/translation_aligners/Awesome/medication_bg.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.15 --dataset './data/translation_aligners/Awesome/medication_bg.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=bgmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_bg_med_20.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.20 --dataset './data/translation_aligners/Awesome/medication_bg.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.20 --dataset './data/translation_aligners/Awesome/medication_bg.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.20 --dataset './data/translation_aligners/Awesome/medication_bg.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=bgmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_bg_med_30.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.30 --dataset './data/translation_aligners/Awesome/medication_bg.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.30 --dataset './data/translation_aligners/Awesome/medication_bg.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.30 --dataset './data/translation_aligners/Awesome/medication_bg.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=bgmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_bg_med_40.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.40 --dataset './data/translation_aligners/Awesome/medication_bg.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.40 --dataset './data/translation_aligners/Awesome/medication_bg.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.40 --dataset './data/translation_aligners/Awesome/medication_bg.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=bgmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_bg_med_45.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.45 --dataset './data/translation_aligners/Awesome/medication_bg.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.45 --dataset './data/translation_aligners/Awesome/medication_bg.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.45 --dataset './data/translation_aligners/Awesome/medication_bg.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF

sbatch --job-name=bgrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_bg_rel_0.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.00 --dataset './data/translation_aligners/Awesome/relations_bg.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.00 --dataset './data/translation_aligners/Awesome/relations_bg.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.00 --dataset './data/translation_aligners/Awesome/relations_bg.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=bgrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_bg_rel_5.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.05 --dataset './data/translation_aligners/Awesome/relations_bg.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.05 --dataset './data/translation_aligners/Awesome/relations_bg.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.05 --dataset './data/translation_aligners/Awesome/relations_bg.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=bgrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_bg_rel_10.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.10 --dataset './data/translation_aligners/Awesome/relations_bg.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.10 --dataset './data/translation_aligners/Awesome/relations_bg.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.10 --dataset './data/translation_aligners/Awesome/relations_bg.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=bgrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_bg_rel_15.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.15 --dataset './data/translation_aligners/Awesome/relations_bg.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.15 --dataset './data/translation_aligners/Awesome/relations_bg.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.15 --dataset './data/translation_aligners/Awesome/relations_bg.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=bgrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_bg_rel_20.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.20 --dataset './data/translation_aligners/Awesome/relations_bg.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.20 --dataset './data/translation_aligners/Awesome/relations_bg.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.20 --dataset './data/translation_aligners/Awesome/relations_bg.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=bgrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_bg_rel_30.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.30 --dataset './data/translation_aligners/Awesome/relations_bg.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.30 --dataset './data/translation_aligners/Awesome/relations_bg.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.30 --dataset './data/translation_aligners/Awesome/relations_bg.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=bgrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_bg_rel_40.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.40 --dataset './data/translation_aligners/Awesome/relations_bg.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.40 --dataset './data/translation_aligners/Awesome/relations_bg.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.40 --dataset './data/translation_aligners/Awesome/relations_bg.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=bgrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_bg_rel_45.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.45 --dataset './data/translation_aligners/Awesome/relations_bg.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.45 --dataset './data/translation_aligners/Awesome/relations_bg.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.45 --dataset './data/translation_aligners/Awesome/relations_bg.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF




















sbatch --job-name=csmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_cs_med_0.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.00 --dataset './data/translation_aligners/Awesome/medication_cs.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.00 --dataset './data/translation_aligners/Awesome/medication_cs.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.00 --dataset './data/translation_aligners/Awesome/medication_cs.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=csmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_cs_med_5.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.05 --dataset './data/translation_aligners/Awesome/medication_cs.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.05 --dataset './data/translation_aligners/Awesome/medication_cs.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.05 --dataset './data/translation_aligners/Awesome/medication_cs.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=csmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_cs_med_10.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.10 --dataset './data/translation_aligners/Awesome/medication_cs.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.10 --dataset './data/translation_aligners/Awesome/medication_cs.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.10 --dataset './data/translation_aligners/Awesome/medication_cs.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=csmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_cs_med_15.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.15 --dataset './data/translation_aligners/Awesome/medication_cs.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.15 --dataset './data/translation_aligners/Awesome/medication_cs.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.15 --dataset './data/translation_aligners/Awesome/medication_cs.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=csmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_cs_med_20.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.20 --dataset './data/translation_aligners/Awesome/medication_cs.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.20 --dataset './data/translation_aligners/Awesome/medication_cs.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.20 --dataset './data/translation_aligners/Awesome/medication_cs.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=csmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_cs_med_30.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.30 --dataset './data/translation_aligners/Awesome/medication_cs.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.30 --dataset './data/translation_aligners/Awesome/medication_cs.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.30 --dataset './data/translation_aligners/Awesome/medication_cs.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=csmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_cs_med_37.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.37 --dataset './data/translation_aligners/Awesome/medication_cs.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.37 --dataset './data/translation_aligners/Awesome/medication_cs.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.37 --dataset './data/translation_aligners/Awesome/medication_cs.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF

sbatch --job-name=csrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_cs_rel_0.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.00 --dataset './data/translation_aligners/Awesome/relations_cs.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.00 --dataset './data/translation_aligners/Awesome/relations_cs.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.00 --dataset './data/translation_aligners/Awesome/relations_cs.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=csrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_cs_rel_5.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.05 --dataset './data/translation_aligners/Awesome/relations_cs.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.05 --dataset './data/translation_aligners/Awesome/relations_cs.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.05 --dataset './data/translation_aligners/Awesome/relations_cs.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=csrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_cs_rel_10.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.10 --dataset './data/translation_aligners/Awesome/relations_cs.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.10 --dataset './data/translation_aligners/Awesome/relations_cs.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.10 --dataset './data/translation_aligners/Awesome/relations_cs.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=csrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_cs_rel_15.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.15 --dataset './data/translation_aligners/Awesome/relations_cs.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.15 --dataset './data/translation_aligners/Awesome/relations_cs.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.15 --dataset './data/translation_aligners/Awesome/relations_cs.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=csrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_cs_rel_20.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.20 --dataset './data/translation_aligners/Awesome/relations_cs.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.20 --dataset './data/translation_aligners/Awesome/relations_cs.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.20 --dataset './data/translation_aligners/Awesome/relations_cs.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=csrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_cs_rel_30.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.30 --dataset './data/translation_aligners/Awesome/relations_cs.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.30 --dataset './data/translation_aligners/Awesome/relations_cs.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.30 --dataset './data/translation_aligners/Awesome/relations_cs.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=csrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_cs_rel_40.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.40 --dataset './data/translation_aligners/Awesome/relations_cs.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.40 --dataset './data/translation_aligners/Awesome/relations_cs.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.40 --dataset './data/translation_aligners/Awesome/relations_cs.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=csrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_cs_rel_41.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.41 --dataset './data/translation_aligners/Awesome/relations_cs.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.41 --dataset './data/translation_aligners/Awesome/relations_cs.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.41 --dataset './data/translation_aligners/Awesome/relations_cs.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF













sbatch --job-name=elmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_el_med_0.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.00 --dataset './data/translation_aligners/FastAlign/medication_el.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.00 --dataset './data/translation_aligners/FastAlign/medication_el.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.00 --dataset './data/translation_aligners/FastAlign/medication_el.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=elmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_el_med_5.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.05 --dataset './data/translation_aligners/FastAlign/medication_el.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.05 --dataset './data/translation_aligners/FastAlign/medication_el.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.05 --dataset './data/translation_aligners/FastAlign/medication_el.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=elmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_el_med_10.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.10 --dataset './data/translation_aligners/FastAlign/medication_el.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.10 --dataset './data/translation_aligners/FastAlign/medication_el.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.10 --dataset './data/translation_aligners/FastAlign/medication_el.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=elmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_el_med_15.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.15 --dataset './data/translation_aligners/FastAlign/medication_el.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.15 --dataset './data/translation_aligners/FastAlign/medication_el.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.15 --dataset './data/translation_aligners/FastAlign/medication_el.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=elmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_el_med_20.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.20 --dataset './data/translation_aligners/FastAlign/medication_el.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.20 --dataset './data/translation_aligners/FastAlign/medication_el.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.20 --dataset './data/translation_aligners/FastAlign/medication_el.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=elmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_el_med_30.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.30 --dataset './data/translation_aligners/FastAlign/medication_el.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.30 --dataset './data/translation_aligners/FastAlign/medication_el.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.30 --dataset './data/translation_aligners/FastAlign/medication_el.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=elmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_el_med_40.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.40 --dataset './data/translation_aligners/FastAlign/medication_el.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.40 --dataset './data/translation_aligners/FastAlign/medication_el.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.40 --dataset './data/translation_aligners/FastAlign/medication_el.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=elmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_el_med_50.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.50 --dataset './data/translation_aligners/FastAlign/medication_el.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.50 --dataset './data/translation_aligners/FastAlign/medication_el.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.50 --dataset './data/translation_aligners/FastAlign/medication_el.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=elmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_el_med_60.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.60 --dataset './data/translation_aligners/FastAlign/medication_el.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.60 --dataset './data/translation_aligners/FastAlign/medication_el.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.60 --dataset './data/translation_aligners/FastAlign/medication_el.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=elmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_el_med_62.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.62 --dataset './data/translation_aligners/FastAlign/medication_el.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.62 --dataset './data/translation_aligners/FastAlign/medication_el.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.62 --dataset './data/translation_aligners/FastAlign/medication_el.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF

sbatch --job-name=elrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_el_rel_0.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.00 --dataset './data/translation_aligners/FastAlign/relations_el.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.00 --dataset './data/translation_aligners/FastAlign/relations_el.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.00 --dataset './data/translation_aligners/FastAlign/relations_el.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=elrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_el_rel_5.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.05 --dataset './data/translation_aligners/FastAlign/relations_el.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.05 --dataset './data/translation_aligners/FastAlign/relations_el.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.05 --dataset './data/translation_aligners/FastAlign/relations_el.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=elrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_el_rel_10.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.10 --dataset './data/translation_aligners/FastAlign/relations_el.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.10 --dataset './data/translation_aligners/FastAlign/relations_el.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.10 --dataset './data/translation_aligners/FastAlign/relations_el.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=elrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_el_rel_15.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.15 --dataset './data/translation_aligners/FastAlign/relations_el.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.15 --dataset './data/translation_aligners/FastAlign/relations_el.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.15 --dataset './data/translation_aligners/FastAlign/relations_el.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=elrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_el_rel_20.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.20 --dataset './data/translation_aligners/FastAlign/relations_el.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.20 --dataset './data/translation_aligners/FastAlign/relations_el.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.20 --dataset './data/translation_aligners/FastAlign/relations_el.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=elrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_el_rel_30.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.30 --dataset './data/translation_aligners/FastAlign/relations_el.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.30 --dataset './data/translation_aligners/FastAlign/relations_el.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.30 --dataset './data/translation_aligners/FastAlign/relations_el.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=elrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_el_rel_40.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.40 --dataset './data/translation_aligners/FastAlign/relations_el.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.40 --dataset './data/translation_aligners/FastAlign/relations_el.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.40 --dataset './data/translation_aligners/FastAlign/relations_el.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=elrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_el_rel_44.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.44 --dataset './data/translation_aligners/FastAlign/relations_el.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.44 --dataset './data/translation_aligners/FastAlign/relations_el.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.44 --dataset './data/translation_aligners/FastAlign/relations_el.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF













sbatch --job-name=esmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_es_med_0.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.00 --dataset './data/translation_aligners/Awesome/medication_es.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.00 --dataset './data/translation_aligners/Awesome/medication_es.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.00 --dataset './data/translation_aligners/Awesome/medication_es.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=esmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_es_med_5.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.05 --dataset './data/translation_aligners/Awesome/medication_es.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.05 --dataset './data/translation_aligners/Awesome/medication_es.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.05 --dataset './data/translation_aligners/Awesome/medication_es.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=esmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_es_med_10.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.10 --dataset './data/translation_aligners/Awesome/medication_es.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.10 --dataset './data/translation_aligners/Awesome/medication_es.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.10 --dataset './data/translation_aligners/Awesome/medication_es.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=esmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_es_med_15.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.15 --dataset './data/translation_aligners/Awesome/medication_es.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.15 --dataset './data/translation_aligners/Awesome/medication_es.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.15 --dataset './data/translation_aligners/Awesome/medication_es.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=esmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_es_med_20.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.20 --dataset './data/translation_aligners/Awesome/medication_es.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.20 --dataset './data/translation_aligners/Awesome/medication_es.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.20 --dataset './data/translation_aligners/Awesome/medication_es.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=esmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_es_med_29.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.29 --dataset './data/translation_aligners/Awesome/medication_es.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.29 --dataset './data/translation_aligners/Awesome/medication_es.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.29 --dataset './data/translation_aligners/Awesome/medication_es.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF

sbatch --job-name=esrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_es_rel_0.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.00 --dataset './data/translation_aligners/Awesome/relations_es.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.00 --dataset './data/translation_aligners/Awesome/relations_es.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.00 --dataset './data/translation_aligners/Awesome/relations_es.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=esrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_es_rel_5.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.05 --dataset './data/translation_aligners/Awesome/relations_es.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.05 --dataset './data/translation_aligners/Awesome/relations_es.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.05 --dataset './data/translation_aligners/Awesome/relations_es.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=esrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_es_rel_10.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.10 --dataset './data/translation_aligners/Awesome/relations_es.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.10 --dataset './data/translation_aligners/Awesome/relations_es.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.10 --dataset './data/translation_aligners/Awesome/relations_es.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=esrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_es_rel_15.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.15 --dataset './data/translation_aligners/Awesome/relations_es.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.15 --dataset './data/translation_aligners/Awesome/relations_es.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.15 --dataset './data/translation_aligners/Awesome/relations_es.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=esrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_es_rel_20.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.20 --dataset './data/translation_aligners/Awesome/relations_es.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.20 --dataset './data/translation_aligners/Awesome/relations_es.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.20 --dataset './data/translation_aligners/Awesome/relations_es.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=esrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_es_rel_23.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.23 --dataset './data/translation_aligners/Awesome/relations_es.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.23 --dataset './data/translation_aligners/Awesome/relations_es.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.23 --dataset './data/translation_aligners/Awesome/relations_es.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF







sbatch --job-name=plmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_pl_med_0.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.00 --dataset './data/translation_aligners/Awesome/medication_pl.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.00 --dataset './data/translation_aligners/Awesome/medication_pl.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.00 --dataset './data/translation_aligners/Awesome/medication_pl.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=plmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_pl_med_5.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.05 --dataset './data/translation_aligners/Awesome/medication_pl.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.05 --dataset './data/translation_aligners/Awesome/medication_pl.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.05 --dataset './data/translation_aligners/Awesome/medication_pl.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=plmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_pl_med_10.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.10 --dataset './data/translation_aligners/Awesome/medication_pl.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.10 --dataset './data/translation_aligners/Awesome/medication_pl.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.10 --dataset './data/translation_aligners/Awesome/medication_pl.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=plmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_pl_med_15.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.15 --dataset './data/translation_aligners/Awesome/medication_pl.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.15 --dataset './data/translation_aligners/Awesome/medication_pl.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.15 --dataset './data/translation_aligners/Awesome/medication_pl.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=plmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_pl_med_20.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.20 --dataset './data/translation_aligners/Awesome/medication_pl.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.20 --dataset './data/translation_aligners/Awesome/medication_pl.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.20 --dataset './data/translation_aligners/Awesome/medication_pl.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=plmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_pl_med_30.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.30 --dataset './data/translation_aligners/Awesome/medication_pl.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.30 --dataset './data/translation_aligners/Awesome/medication_pl.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.30 --dataset './data/translation_aligners/Awesome/medication_pl.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=plmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_pl_med_40.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.40 --dataset './data/translation_aligners/Awesome/medication_pl.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.40 --dataset './data/translation_aligners/Awesome/medication_pl.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.40 --dataset './data/translation_aligners/Awesome/medication_pl.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=plmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_pl_med_45.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.45 --dataset './data/translation_aligners/Awesome/medication_pl.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.45 --dataset './data/translation_aligners/Awesome/medication_pl.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.45 --dataset './data/translation_aligners/Awesome/medication_pl.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF

sbatch --job-name=plrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_pl_rel_0.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.00 --dataset './data/translation_aligners/FastAlign/relations_pl.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.00 --dataset './data/translation_aligners/FastAlign/relations_pl.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.00 --dataset './data/translation_aligners/FastAlign/relations_pl.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=plrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_pl_rel_5.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.05 --dataset './data/translation_aligners/FastAlign/relations_pl.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.05 --dataset './data/translation_aligners/FastAlign/relations_pl.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.05 --dataset './data/translation_aligners/FastAlign/relations_pl.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=plrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_pl_rel_10.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.10 --dataset './data/translation_aligners/FastAlign/relations_pl.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.10 --dataset './data/translation_aligners/FastAlign/relations_pl.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.10 --dataset './data/translation_aligners/FastAlign/relations_pl.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=plrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_pl_rel_15.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.15 --dataset './data/translation_aligners/FastAlign/relations_pl.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.15 --dataset './data/translation_aligners/FastAlign/relations_pl.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.15 --dataset './data/translation_aligners/FastAlign/relations_pl.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=plrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_pl_rel_20.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.20 --dataset './data/translation_aligners/FastAlign/relations_pl.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.20 --dataset './data/translation_aligners/FastAlign/relations_pl.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.20 --dataset './data/translation_aligners/FastAlign/relations_pl.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=plrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_pl_rel_30.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.30 --dataset './data/translation_aligners/FastAlign/relations_pl.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.30 --dataset './data/translation_aligners/FastAlign/relations_pl.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.30 --dataset './data/translation_aligners/FastAlign/relations_pl.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=plrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_pl_rel_40.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.40 --dataset './data/translation_aligners/FastAlign/relations_pl.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.40 --dataset './data/translation_aligners/FastAlign/relations_pl.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.40 --dataset './data/translation_aligners/FastAlign/relations_pl.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=plrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_pl_rel_43.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.43 --dataset './data/translation_aligners/FastAlign/relations_pl.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.43 --dataset './data/translation_aligners/FastAlign/relations_pl.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.43 --dataset './data/translation_aligners/FastAlign/relations_pl.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF









sbatch --job-name=romqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_ro_med_0.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.00 --dataset './data/translation_aligners/Awesome/medication_ro.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.00 --dataset './data/translation_aligners/Awesome/medication_ro.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.00 --dataset './data/translation_aligners/Awesome/medication_ro.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=romqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_ro_med_5.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.05 --dataset './data/translation_aligners/Awesome/medication_ro.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.05 --dataset './data/translation_aligners/Awesome/medication_ro.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.05 --dataset './data/translation_aligners/Awesome/medication_ro.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=romqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_ro_med_10.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.10 --dataset './data/translation_aligners/Awesome/medication_ro.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.10 --dataset './data/translation_aligners/Awesome/medication_ro.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.10 --dataset './data/translation_aligners/Awesome/medication_ro.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=romqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_ro_med_15.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.15 --dataset './data/translation_aligners/Awesome/medication_ro.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.15 --dataset './data/translation_aligners/Awesome/medication_ro.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.15 --dataset './data/translation_aligners/Awesome/medication_ro.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=romqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_ro_med_20.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.20 --dataset './data/translation_aligners/Awesome/medication_ro.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.20 --dataset './data/translation_aligners/Awesome/medication_ro.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.20 --dataset './data/translation_aligners/Awesome/medication_ro.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=romqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_ro_med_30.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.30 --dataset './data/translation_aligners/Awesome/medication_ro.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.30 --dataset './data/translation_aligners/Awesome/medication_ro.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.30 --dataset './data/translation_aligners/Awesome/medication_ro.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=romqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_ro_med_38.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.38 --dataset './data/translation_aligners/Awesome/medication_ro.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.38 --dataset './data/translation_aligners/Awesome/medication_ro.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.38 --dataset './data/translation_aligners/Awesome/medication_ro.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF

sbatch --job-name=rorqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_ro_rel_0.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.00 --dataset './data/translation_aligners/Awesome/relations_ro.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.00 --dataset './data/translation_aligners/Awesome/relations_ro.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.00 --dataset './data/translation_aligners/Awesome/relations_ro.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=rorqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_ro_rel_5.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.05 --dataset './data/translation_aligners/Awesome/relations_ro.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.05 --dataset './data/translation_aligners/Awesome/relations_ro.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.05 --dataset './data/translation_aligners/Awesome/relations_ro.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=rorqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_ro_rel_10.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.10 --dataset './data/translation_aligners/Awesome/relations_ro.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.10 --dataset './data/translation_aligners/Awesome/relations_ro.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.10 --dataset './data/translation_aligners/Awesome/relations_ro.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=rorqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_ro_rel_15.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.15 --dataset './data/translation_aligners/Awesome/relations_ro.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.15 --dataset './data/translation_aligners/Awesome/relations_ro.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.15 --dataset './data/translation_aligners/Awesome/relations_ro.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=rorqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_ro_rel_20.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.20 --dataset './data/translation_aligners/Awesome/relations_ro.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.20 --dataset './data/translation_aligners/Awesome/relations_ro.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.20 --dataset './data/translation_aligners/Awesome/relations_ro.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=rorqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_ro_rel_30.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.30 --dataset './data/translation_aligners/Awesome/relations_ro.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.30 --dataset './data/translation_aligners/Awesome/relations_ro.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.30 --dataset './data/translation_aligners/Awesome/relations_ro.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=rorqa \
     --output=scripts/slurm_outputs/qa/paragraphs_thresholds/mbert_ro_rel_34.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --answers_remove_ratio 0.34 --dataset './data/translation_aligners/Awesome/relations_ro.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 2 --answers_remove_ratio 0.34 --dataset './data/translation_aligners/Awesome/relations_ro.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
python3 measure_qa.py --seed 3 --answers_remove_ratio 0.34 --dataset './data/translation_aligners/Awesome/relations_ro.json' --epochs 3 --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
