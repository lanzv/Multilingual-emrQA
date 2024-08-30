#!/bin/sh
#SBATCH -J qa
#SBATCH -o scripts/slurm_outputs/qa.out
#SBATCH -p cpu-ms

sbatch --job-name=bgmqa \
     --output=scripts/slurm_outputs/qa/full_reports/mbert_bg_med_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --dataset './data/translation_aligners/Awesome/medication_bg.json' --epochs 3 --to_reports True --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=bgmqa \
     --output=scripts/slurm_outputs/qa/full_reports/mbert_bg_med_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 2 --dataset './data/translation_aligners/Awesome/medication_bg.json' --epochs 3 --to_reports True --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=bgmqa \
     --output=scripts/slurm_outputs/qa/full_reports/mbert_bg_med_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 3 --dataset './data/translation_aligners/Awesome/medication_bg.json' --epochs 3 --to_reports True --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF


sbatch --job-name=bgrqa \
     --output=scripts/slurm_outputs/qa/full_reports/mbert_bg_rel_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --dataset './data/translation_aligners/FastAlign/relations_bg.json' --epochs 3 --to_reports True --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=bgrqa \
     --output=scripts/slurm_outputs/qa/full_reports/mbert_bg_rel_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 2 --dataset './data/translation_aligners/FastAlign/relations_bg.json' --epochs 3 --to_reports True --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=bgrqa \
     --output=scripts/slurm_outputs/qa/full_reports/mbert_bg_rel_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 3 --dataset './data/translation_aligners/FastAlign/relations_bg.json' --epochs 3 --to_reports True --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=bgrqa \
     --output=scripts/slurm_outputs/qa/full_reports/mbert_bg_rel_4.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --dataset './data/translation_aligners/Awesome/relations_bg.json' --epochs 3 --to_reports True --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=bgrqa \
     --output=scripts/slurm_outputs/qa/full_reports/mbert_bg_rel_5.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 2 --dataset './data/translation_aligners/Awesome/relations_bg.json' --epochs 3 --to_reports True --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=bgrqa \
     --output=scripts/slurm_outputs/qa/full_reports/mbert_bg_rel_6.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 3 --dataset './data/translation_aligners/Awesome/relations_bg.json' --epochs 3 --to_reports True --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF


sbatch --job-name=csmqa \
     --output=scripts/slurm_outputs/qa/full_reports/mbert_cs_med_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --dataset './data/translation_aligners/Awesome/medication_cs.json' --epochs 3 --to_reports True --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=csmqa \
     --output=scripts/slurm_outputs/qa/full_reports/mbert_cs_med_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 2 --dataset './data/translation_aligners/Awesome/medication_cs.json' --epochs 3 --to_reports True --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=csmqa \
     --output=scripts/slurm_outputs/qa/full_reports/mbert_cs_med_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 3 --dataset './data/translation_aligners/Awesome/medication_cs.json' --epochs 3 --to_reports True --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF


sbatch --job-name=csrqa \
     --output=scripts/slurm_outputs/qa/full_reports/mbert_cs_rel_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --dataset './data/translation_aligners/FastAlign/relations_cs.json' --epochs 3 --to_reports True --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=csrqa \
     --output=scripts/slurm_outputs/qa/full_reports/mbert_cs_rel_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 2 --dataset './data/translation_aligners/FastAlign/relations_cs.json' --epochs 3 --to_reports True --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=csrqa \
     --output=scripts/slurm_outputs/qa/full_reports/mbert_cs_rel_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 3 --dataset './data/translation_aligners/FastAlign/relations_cs.json' --epochs 3 --to_reports True --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=csrqa \
     --output=scripts/slurm_outputs/qa/full_reports/mbert_cs_rel_4.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --dataset './data/translation_aligners/Awesome/relations_cs.json' --epochs 3 --to_reports True --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=csrqa \
     --output=scripts/slurm_outputs/qa/full_reports/mbert_cs_rel_5.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 2 --dataset './data/translation_aligners/Awesome/relations_cs.json' --epochs 3 --to_reports True --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=csrqa \
     --output=scripts/slurm_outputs/qa/full_reports/mbert_cs_rel_6.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 3 --dataset './data/translation_aligners/Awesome/relations_cs.json' --epochs 3 --to_reports True --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF



sbatch --job-name=elmqa \
     --output=scripts/slurm_outputs/qa/full_reports/mbert_el_med_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --dataset './data/translation_aligners/FastAlign/medication_el.json' --epochs 3 --to_reports True --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=elmqa \
     --output=scripts/slurm_outputs/qa/full_reports/mbert_el_med_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 2 --dataset './data/translation_aligners/FastAlign/medication_el.json' --epochs 3 --to_reports True --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=elmqa \
     --output=scripts/slurm_outputs/qa/full_reports/mbert_el_med_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 3 --dataset './data/translation_aligners/FastAlign/medication_el.json' --epochs 3 --to_reports True --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF


sbatch --job-name=elrqa \
     --output=scripts/slurm_outputs/qa/full_reports/mbert_el_rel_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=120G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --dataset './data/translation_aligners/FastAlign/relations_el.json' --epochs 3 --to_reports True --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=elrqa \
     --output=scripts/slurm_outputs/qa/full_reports/mbert_el_rel_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 2 --dataset './data/translation_aligners/FastAlign/relations_el.json' --epochs 3 --to_reports True --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=elrqa \
     --output=scripts/slurm_outputs/qa/full_reports/mbert_el_rel_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=120G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 3 --dataset './data/translation_aligners/FastAlign/relations_el.json' --epochs 3 --to_reports True --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF




sbatch --job-name=esmqa \
     --output=scripts/slurm_outputs/qa/full_reports/mbert_es_med_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --dataset './data/translation_aligners/Awesome/medication_es.json' --epochs 3 --to_reports True --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=esmqa \
     --output=scripts/slurm_outputs/qa/full_reports/mbert_es_med_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 2 --dataset './data/translation_aligners/Awesome/medication_es.json' --epochs 3 --to_reports True --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=esmqa \
     --output=scripts/slurm_outputs/qa/full_reports/mbert_es_med_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 3 --dataset './data/translation_aligners/Awesome/medication_es.json' --epochs 3 --to_reports True --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF


sbatch --job-name=esrqa \
     --output=scripts/slurm_outputs/qa/full_reports/mbert_es_rel_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --dataset './data/translation_aligners/Awesome/relations_es.json' --epochs 3 --to_reports True --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=esrqa \
     --output=scripts/slurm_outputs/qa/full_reports/mbert_es_rel_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 2 --dataset './data/translation_aligners/Awesome/relations_es.json' --epochs 3 --to_reports True --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=esrqa \
     --output=scripts/slurm_outputs/qa/full_reports/mbert_es_rel_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 3 --dataset './data/translation_aligners/Awesome/relations_es.json' --epochs 3 --to_reports True --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF


sbatch --job-name=plmqa \
     --output=scripts/slurm_outputs/qa/full_reports/mbert_pl_med_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --dataset './data/translation_aligners/Awesome/medication_pl.json' --epochs 3 --to_reports True --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=plmqa \
     --output=scripts/slurm_outputs/qa/full_reports/mbert_pl_med_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 2 --dataset './data/translation_aligners/Awesome/medication_pl.json' --epochs 3 --to_reports True --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=plmqa \
     --output=scripts/slurm_outputs/qa/full_reports/mbert_pl_med_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 3 --dataset './data/translation_aligners/Awesome/medication_pl.json' --epochs 3 --to_reports True --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF


sbatch --job-name=plrqa \
     --output=scripts/slurm_outputs/qa/full_reports/mbert_pl_rel_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --dataset './data/translation_aligners/FastAlign/relations_pl.json' --epochs 3 --to_reports True --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=plrqa \
     --output=scripts/slurm_outputs/qa/full_reports/mbert_pl_rel_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 2 --dataset './data/translation_aligners/FastAlign/relations_pl.json' --epochs 3 --to_reports True --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=plrqa \
     --output=scripts/slurm_outputs/qa/full_reports/mbert_pl_rel_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 3 --dataset './data/translation_aligners/FastAlign/relations_pl.json' --epochs 3 --to_reports True --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=plrqa \
     --output=scripts/slurm_outputs/qa/full_reports/mbert_pl_rel_4.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --dataset './data/translation_aligners/Awesome/relations_pl.json' --epochs 3 --to_reports True --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=plrqa \
     --output=scripts/slurm_outputs/qa/full_reports/mbert_pl_rel_5.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 2 --dataset './data/translation_aligners/Awesome/relations_pl.json' --epochs 3 --to_reports True --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=plrqa \
     --output=scripts/slurm_outputs/qa/full_reports/mbert_pl_rel_6.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 3 --dataset './data/translation_aligners/Awesome/relations_pl.json' --epochs 3 --to_reports True --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF


sbatch --job-name=romqa \
     --output=scripts/slurm_outputs/qa/full_reports/mbert_ro_med_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --dataset './data/translation_aligners/Awesome/medication_ro.json' --epochs 3 --to_reports True --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=romqa \
     --output=scripts/slurm_outputs/qa/full_reports/mbert_ro_med_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 2 --dataset './data/translation_aligners/Awesome/medication_ro.json' --epochs 3 --to_reports True --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=romqa \
     --output=scripts/slurm_outputs/qa/full_reports/mbert_ro_med_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 3 --dataset './data/translation_aligners/Awesome/medication_ro.json' --epochs 3 --to_reports True --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF


sbatch --job-name=rorqa \
     --output=scripts/slurm_outputs/qa/full_reports/mbert_ro_rel_1.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --dataset './data/translation_aligners/Awesome/relations_ro.json' --epochs 3 --to_reports True --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=rorqa \
     --output=scripts/slurm_outputs/qa/full_reports/mbert_ro_rel_2.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 2 --dataset './data/translation_aligners/Awesome/relations_ro.json' --epochs 3 --to_reports True --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=rorqa \
     --output=scripts/slurm_outputs/qa/full_reports/mbert_ro_rel_3.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 3 --dataset './data/translation_aligners/Awesome/relations_ro.json' --epochs 3 --to_reports True --paragraph_parts True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF




sbatch --job-name=enmqa \
     --output=scripts/slurm_outputs/qa/full_reports/mbert_en_med_1.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --dataset '../datasets/emrQA/medication_en.json' --epochs 3 --to_reports True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=enmqa \
     --output=scripts/slurm_outputs/qa/full_reports/mbert_en_med_2.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 2 --dataset '../datasets/emrQA/medication_en.json' --epochs 3 --to_reports True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=enmqa \
     --output=scripts/slurm_outputs/qa/full_reports/mbert_en_med_3.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 3 --dataset '../datasets/emrQA/medication_en.json' --epochs 3 --to_reports True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF


sbatch --job-name=enrqa \
     --output=scripts/slurm_outputs/qa/full_reports/mbert_en_rel_1.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 1 --dataset '../datasets/emrQA/relations_en.json' --epochs 3 --to_reports True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=enrqa \
     --output=scripts/slurm_outputs/qa/full_reports/mbert_en_rel_2.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 2 --dataset '../datasets/emrQA/relations_en.json' --epochs 3 --to_reports True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=enrqa \
     --output=scripts/slurm_outputs/qa/full_reports/mbert_en_rel_3.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa.py --seed 3 --dataset '../datasets/emrQA/relations_en.json' --epochs 3 --to_reports True --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
