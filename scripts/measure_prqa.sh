#!/bin/sh
#SBATCH -J prqa
#SBATCH -o scripts/slurm_outputs/prqa.out
#SBATCH -p cpu-ms

sbatch --job-name=bg_m_prqa \
     --output=scripts/slurm_outputs/prqa/awesome/bg_med.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --nodelist=dll-3gpu2 <<"EOF"
#!/bin/bash
python3 measure_prqa.py --dataset '../datasets/emrQA/medication_bg.json' --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF


sbatch --job-name=bg_r_prqa \
     --output=scripts/slurm_outputs/prqa/awesome/bg_rel.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --nodelist=dll-3gpu2 <<"EOF"
#!/bin/bash
python3 measure_prqa.py --dataset '../datasets/emrQA/relations_bg.json' --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF






sbatch --job-name=cs_m_prqa \
     --output=scripts/slurm_outputs/prqa/awesome/cs_med.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --nodelist=dll-3gpu3 <<"EOF"
#!/bin/bash
python3 measure_prqa.py --dataset '../datasets/emrQA/medication_cs.json' --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF


sbatch --job-name=cs_r_prqa \
     --output=scripts/slurm_outputs/prqa/awesome/cs_rel.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --nodelist=dll-3gpu3 <<"EOF"
#!/bin/bash
python3 measure_prqa.py --dataset '../datasets/emrQA/relations_cs.json' --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF





sbatch --job-name=el_m_prqa \
     --output=scripts/slurm_outputs/prqa/awesome/el_med.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --nodelist=dll-3gpu4 <<"EOF"
#!/bin/bash
python3 measure_prqa.py --dataset '../datasets/emrQA/medication_el.json' --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF

sbatch --job-name=el_r_prqa \
     --output=scripts/slurm_outputs/prqa/awesome/el_rel.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --nodelist=dll-3gpu4 <<"EOF"
#!/bin/bash
python3 measure_prqa.py --dataset '../datasets/emrQA/relations_el.json' --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF






sbatch --job-name=pl_m_prqa \
     --output=scripts/slurm_outputs/prqa/awesome/pl_med.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --nodelist=dll-3gpu5 <<"EOF"
#!/bin/bash
python3 measure_prqa.py --dataset '../datasets/emrQA/medication_pl.json' --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF


sbatch --job-name=pl_r_prqa \
     --output=scripts/slurm_outputs/prqa/awesome/pl_rel.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --nodelist=dll-3gpu5 <<"EOF"
#!/bin/bash
python3 measure_prqa.py --dataset '../datasets/emrQA/relations_pl.json' --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF






sbatch --job-name=ro_m_prqa \
     --output=scripts/slurm_outputs/prqa/awesome/ro_med.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --nodelist=dll-4gpu3 <<"EOF"
#!/bin/bash
python3 measure_prqa.py --dataset '../datasets/emrQA/medication_ro.json' --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF


sbatch --job-name=ro_r_prqa \
     --output=scripts/slurm_outputs/prqa/awesome/ro_rel.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --nodelist=dll-4gpu3 <<"EOF"
#!/bin/bash
python3 measure_prqa.py --dataset '../datasets/emrQA/relations_ro.json' --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF







sbatch --job-name=es_m_prqa \
     --output=scripts/slurm_outputs/prqa/awesome/es_med.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --nodelist=dll-3gpu1 <<"EOF"
#!/bin/bash
python3 measure_prqa.py --dataset '../datasets/emrQA/medication_es.json' --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF


sbatch --job-name=es_r_prqa \
     --output=scripts/slurm_outputs/prqa/awesome/es_rel.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --nodelist=dll-3gpu2 <<"EOF"
#!/bin/bash
python3 measure_prqa.py --dataset '../datasets/emrQA/relations_es.json' --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF






sbatch --job-name=en_m_prqa \
     --output=scripts/slurm_outputs/prqa/awesome/en_med.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --nodelist=dll-4gpu4 <<"EOF"
#!/bin/bash
python3 measure_prqa.py --dataset '../datasets/emrQA/medication_en.json' --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF

sbatch --job-name=en_r_prqa \
     --output=scripts/slurm_outputs/prqa/awesome/en_rel.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --nodelist=dll-4gpu4 <<"EOF"
#!/bin/bash
python3 measure_prqa.py --dataset '../datasets/emrQA/relations_en.json' --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
