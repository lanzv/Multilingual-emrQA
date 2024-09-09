#!/bin/sh
#SBATCH -J qa
#SBATCH -o scripts/slurm_outputs/qa.out
#SBATCH -p cpu-ms

sbatch --job-name=mqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_multilingual/mbert_med_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --multilingual_train True --subset "medication" --epochs 3 --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF

sbatch --job-name=mqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_multilingual/mbert_med_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --multilingual_train True --subset "medication" --epochs 3 --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF

sbatch --job-name=mqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_multilingual/mbert_med_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --multilingual_train True --subset "medication" --epochs 3 --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF

sbatch --job-name=rqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_multilingual/mbert_rel_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --multilingual_train True --subset "relations" --epochs 3 --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF

sbatch --job-name=rqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_multilingual/mbert_rel_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --multilingual_train True --subset "relations" --epochs 3 --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF

sbatch --job-name=rqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_multilingual/mbert_rel_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --multilingual_train True --subset "relations" --epochs 3 --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF








sbatch --job-name=mqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_multilingual/distilbert_med_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --multilingual_train True --subset "medication" --epochs 3 --model_name 'mDistil' --model_path '../models/distilbert-base-multilingual-cased' --train_sample_ratio 0.2
EOF

sbatch --job-name=mqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_multilingual/distilbert_med_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --multilingual_train True --subset "medication" --epochs 3 --model_name 'mDistil' --model_path '../models/distilbert-base-multilingual-cased' --train_sample_ratio 0.2
EOF

sbatch --job-name=mqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_multilingual/distilbert_med_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --multilingual_train True --subset "medication" --epochs 3 --model_name 'mDistil' --model_path '../models/distilbert-base-multilingual-cased' --train_sample_ratio 0.2
EOF

sbatch --job-name=rqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_multilingual/distilbert_rel_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --multilingual_train True --subset "relations" --epochs 3 --model_name 'mDistil' --model_path '../models/distilbert-base-multilingual-cased' --train_sample_ratio 0.05
EOF

sbatch --job-name=rqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_multilingual/distilbert_rel_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --multilingual_train True --subset "relations" --epochs 3 --model_name 'mDistil' --model_path '../models/distilbert-base-multilingual-cased' --train_sample_ratio 0.05
EOF

sbatch --job-name=rqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_multilingual/distilbert_rel_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --multilingual_train True --subset "relations" --epochs 3 --model_name 'mDistil' --model_path '../models/distilbert-base-multilingual-cased' --train_sample_ratio 0.05
EOF





sbatch --job-name=mqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_multilingual/xlmr_med_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --multilingual_train True --subset "medication" --epochs 3 --model_name 'XLMR' --model_path '../models/xlm-roberta-base' --train_sample_ratio 0.2
EOF

sbatch --job-name=mqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_multilingual/xlmr_med_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --multilingual_train True --subset "medication" --epochs 3 --model_name 'XLMR' --model_path '../models/xlm-roberta-base' --train_sample_ratio 0.2
EOF

sbatch --job-name=mqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_multilingual/xlmr_med_3.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --multilingual_train True --subset "medication" --epochs 3 --model_name 'XLMR' --model_path '../models/xlm-roberta-base' --train_sample_ratio 0.2
EOF

sbatch --job-name=rqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_multilingual/xlmr_rel_1.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --multilingual_train True --subset "relations" --epochs 3 --model_name 'XLMR' --model_path '../models/xlm-roberta-base' --train_sample_ratio 0.05
EOF

sbatch --job-name=rqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_multilingual/xlmr_rel_2.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --multilingual_train True --subset "relations" --epochs 3 --model_name 'XLMR' --model_path '../models/xlm-roberta-base' --train_sample_ratio 0.05
EOF

sbatch --job-name=rqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_multilingual/xlmr_rel_3.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --multilingual_train True --subset "relations" --epochs 3 --model_name 'XLMR' --model_path '../models/xlm-roberta-base' --train_sample_ratio 0.05
EOF






sbatch --job-name=mqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_multilingual/xlmrlarge_med_1.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --multilingual_train True --subset "medication" --epochs 3 --model_name 'XLMRLarge' --model_path '../models/xlm-roberta-large' --train_sample_ratio 0.2
EOF

sbatch --job-name=mqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_multilingual/xlmrlarge_med_2.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --multilingual_train True --subset "medication" --epochs 3 --model_name 'XLMRLarge' --model_path '../models/xlm-roberta-large' --train_sample_ratio 0.2
EOF

sbatch --job-name=mqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_multilingual/xlmrlarge_med_3.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --multilingual_train True --subset "medication" --epochs 3 --model_name 'XLMRLarge' --model_path '../models/xlm-roberta-large' --train_sample_ratio 0.2
EOF

sbatch --job-name=rqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_multilingual/xlmrlarge_rel_1.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --multilingual_train True --subset "relations" --epochs 3 --model_name 'XLMRLarge' --model_path '../models/xlm-roberta-large' --train_sample_ratio 0.05
EOF

sbatch --job-name=rqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_multilingual/xlmrlarge_rel_2.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --multilingual_train True --subset "relations" --epochs 3 --model_name 'XLMRLarge' --model_path '../models/xlm-roberta-large' --train_sample_ratio 0.05
EOF

sbatch --job-name=rqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_multilingual/xlmrlarge_rel_3.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --multilingual_train True --subset "relations" --epochs 3 --model_name 'XLMRLarge' --model_path '../models/xlm-roberta-large' --train_sample_ratio 0.05
EOF






sbatch --job-name=mqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_multilingual/clinical_med_1.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --multilingual_train True --subset "medication" --epochs 3 --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.2
EOF

sbatch --job-name=mqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_multilingual/clinical_med_2.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --multilingual_train True --subset "medication" --epochs 3 --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.2
EOF

sbatch --job-name=mqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_multilingual/clinical_med_3.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --multilingual_train True --subset "medication" --epochs 3 --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.2
EOF

sbatch --job-name=rqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_multilingual/clinical_rel_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --multilingual_train True --subset "relations" --epochs 3 --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.05
EOF

sbatch --job-name=rqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_multilingual/clinical_rel_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --multilingual_train True --subset "relations" --epochs 3 --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.05
EOF

sbatch --job-name=rqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_multilingual/clinical_rel_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --multilingual_train True --subset "relations" --epochs 3 --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.05
EOF