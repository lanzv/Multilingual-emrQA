#!/bin/sh
#SBATCH -J qa
#SBATCH -o scripts/slurm_outputs/qa.out
#SBATCH -p cpu-ms


sbatch --job-name=bgmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mbertbg_med_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "medication" --language "BG" --epochs 3 --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF


sbatch --job-name=bgmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mbertbg_med_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "medication" --language "BG" --epochs 3 --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=bgmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mbertbg_med_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "medication" --language "BG" --epochs 3 --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF


sbatch --job-name=bgrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mbertbg_rel_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "relations" --language "BG" --epochs 3 --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=bgrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mbertbg_rel_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "relations" --language "BG" --epochs 3 --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=bgrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mbertbg_rel_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "relations" --language "BG" --epochs 3 --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF

sbatch --job-name=csmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mbertcs_med_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "medication" --language "CS" --epochs 3 --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=csmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mbertcs_med_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "medication" --language "CS" --epochs 3 --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=csmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mbertcs_med_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "medication" --language "CS" --epochs 3 --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF


sbatch --job-name=csrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mbertcs_rel_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "relations" --language "CS" --epochs 3 --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=csrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mbertcs_rel_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "relations" --language "CS" --epochs 3 --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=csrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mbertcs_rel_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "relations" --language "CS" --epochs 3 --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF



sbatch --job-name=elmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mbertel_med_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "medication" --language "EL" --epochs 3 --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=elmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mbertel_med_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "medication" --language "EL" --epochs 3 --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=elmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mbertel_med_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "medication" --language "EL" --epochs 3 --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF


sbatch --job-name=elrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mbertel_rel_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=120G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "relations" --language "EL" --epochs 3 --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=elrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mbertel_rel_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "relations" --language "EL" --epochs 3 --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=elrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mbertel_rel_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=120G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "relations" --language "EL" --epochs 3 --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF




sbatch --job-name=esmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mbertes_med_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "medication" --language "ES" --epochs 3 --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=esmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mbertes_med_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "medication" --language "ES" --epochs 3 --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=esmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mbertes_med_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "medication" --language "ES" --epochs 3 --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF


sbatch --job-name=esrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mbertes_rel_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "relations" --language "ES" --epochs 3 --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=esrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mbertes_rel_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "relations" --language "ES" --epochs 3 --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=esrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mbertes_rel_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "relations" --language "ES" --epochs 3 --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF


sbatch --job-name=plmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mbertpl_med_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "medication" --language "PL" --epochs 3 --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=plmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mbertpl_med_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "medication" --language "PL" --epochs 3 --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=plmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mbertpl_med_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "medication" --language "PL" --epochs 3 --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF


sbatch --job-name=plrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mbertpl_rel_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "relations" --language "PL" --epochs 3 --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=plrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mbertpl_rel_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "relations" --language "PL" --epochs 3 --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=plrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mbertpl_rel_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "relations" --language "PL" --epochs 3 --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF


sbatch --job-name=romqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mbertro_med_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "medication" --language "RO" --epochs 3 --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=romqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mbertro_med_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "medication" --language "RO" --epochs 3 --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=romqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mbertro_med_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "medication" --language "RO" --epochs 3 --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF


sbatch --job-name=rorqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mbertro_rel_1.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "relations" --language "RO" --epochs 3 --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=rorqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mbertro_rel_2.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "relations" --language "RO" --epochs 3 --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=rorqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mbertro_rel_3.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "relations" --language "RO" --epochs 3 --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF




sbatch --job-name=enmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mberten_full_med_1.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "medication" --language "EN_FULL" --epochs 3 --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=enmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mberten_full_med_2.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "medication" --language "EN_FULL" --epochs 3 --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=enmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mberten_full_med_3.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "medication" --language "EN_FULL" --epochs 3 --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF


sbatch --job-name=enrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mberten_full_rel_1.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "relations" --language "EN_FULL" --epochs 3 --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=enrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mberten_full_rel_2.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "relations" --language "EN_FULL" --epochs 3 --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=enrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mberten_full_rel_3.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "relations" --language "EN_FULL" --epochs 3 --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF


sbatch --job-name=enmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mberten_med_1.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "medication" --language "EN" --epochs 3 --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=enmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mberten_med_2.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "medication" --language "EN" --epochs 3 --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=enmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mberten_med_3.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "medication" --language "EN" --epochs 3 --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF


sbatch --job-name=enrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mberten_rel_1.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "relations" --language "EN" --epochs 3 --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=enrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mberten_rel_2.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "relations" --language "EN" --epochs 3 --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=enrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mberten_rel_3.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "relations" --language "EN" --epochs 3 --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF

















#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################

sbatch --job-name=bgmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mdistil_bg_med_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "medication" --language "BG" --epochs 3 --model_name 'mDistil' --model_path '../models/distilbert-base-multilingual-cased' --train_sample_ratio 0.2
EOF


sbatch --job-name=bgmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mdistil_bg_med_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "medication" --language "BG" --epochs 3 --model_name 'mDistil' --model_path '../models/distilbert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=bgmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mdistil_bg_med_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "medication" --language "BG" --epochs 3 --model_name 'mDistil' --model_path '../models/distilbert-base-multilingual-cased' --train_sample_ratio 0.2
EOF


sbatch --job-name=bgrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mdistil_bg_rel_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "relations" --language "BG" --epochs 3 --model_name 'mDistil' --model_path '../models/distilbert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=bgrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mdistil_bg_rel_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "relations" --language "BG" --epochs 3 --model_name 'mDistil' --model_path '../models/distilbert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=bgrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mdistil_bg_rel_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "relations" --language "BG" --epochs 3 --model_name 'mDistil' --model_path '../models/distilbert-base-multilingual-cased' --train_sample_ratio 0.05
EOF

sbatch --job-name=csmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mdistil_cs_med_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "medication" --language "CS" --epochs 3 --model_name 'mDistil' --model_path '../models/distilbert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=csmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mdistil_cs_med_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "medication" --language "CS" --epochs 3 --model_name 'mDistil' --model_path '../models/distilbert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=csmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mdistil_cs_med_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "medication" --language "CS" --epochs 3 --model_name 'mDistil' --model_path '../models/distilbert-base-multilingual-cased' --train_sample_ratio 0.2
EOF


sbatch --job-name=csrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mdistil_cs_rel_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "relations" --language "CS" --epochs 3 --model_name 'mDistil' --model_path '../models/distilbert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=csrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mdistil_cs_rel_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "relations" --language "CS" --epochs 3 --model_name 'mDistil' --model_path '../models/distilbert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=csrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mdistil_cs_rel_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "relations" --language "CS" --epochs 3 --model_name 'mDistil' --model_path '../models/distilbert-base-multilingual-cased' --train_sample_ratio 0.05
EOF



sbatch --job-name=elmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mdistil_el_med_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "medication" --language "EL" --epochs 3 --model_name 'mDistil' --model_path '../models/distilbert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=elmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mdistil_el_med_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "medication" --language "EL" --epochs 3 --model_name 'mDistil' --model_path '../models/distilbert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=elmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mdistil_el_med_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "medication" --language "EL" --epochs 3 --model_name 'mDistil' --model_path '../models/distilbert-base-multilingual-cased' --train_sample_ratio 0.2
EOF


sbatch --job-name=elrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mdistil_el_rel_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=120G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "relations" --language "EL" --epochs 3 --model_name 'mDistil' --model_path '../models/distilbert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=elrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mdistil_el_rel_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "relations" --language "EL" --epochs 3 --model_name 'mDistil' --model_path '../models/distilbert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=elrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mdistil_el_rel_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=120G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "relations" --language "EL" --epochs 3 --model_name 'mDistil' --model_path '../models/distilbert-base-multilingual-cased' --train_sample_ratio 0.05
EOF




sbatch --job-name=esmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mdistil_es_med_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "medication" --language "ES" --epochs 3 --model_name 'mDistil' --model_path '../models/distilbert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=esmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mdistil_es_med_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "medication" --language "ES" --epochs 3 --model_name 'mDistil' --model_path '../models/distilbert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=esmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mdistil_es_med_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "medication" --language "ES" --epochs 3 --model_name 'mDistil' --model_path '../models/distilbert-base-multilingual-cased' --train_sample_ratio 0.2
EOF


sbatch --job-name=esrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mdistil_es_rel_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "relations" --language "ES" --epochs 3 --model_name 'mDistil' --model_path '../models/distilbert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=esrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mdistil_es_rel_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "relations" --language "ES" --epochs 3 --model_name 'mDistil' --model_path '../models/distilbert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=esrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mdistil_es_rel_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "relations" --language "ES" --epochs 3 --model_name 'mDistil' --model_path '../models/distilbert-base-multilingual-cased' --train_sample_ratio 0.05
EOF


sbatch --job-name=plmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mdistil_pl_med_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "medication" --language "PL" --epochs 3 --model_name 'mDistil' --model_path '../models/distilbert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=plmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mdistil_pl_med_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "medication" --language "PL" --epochs 3 --model_name 'mDistil' --model_path '../models/distilbert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=plmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mdistil_pl_med_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "medication" --language "PL" --epochs 3 --model_name 'mDistil' --model_path '../models/distilbert-base-multilingual-cased' --train_sample_ratio 0.2
EOF


sbatch --job-name=plrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mdistil_pl_rel_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "relations" --language "PL" --epochs 3 --model_name 'mDistil' --model_path '../models/distilbert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=plrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mdistil_pl_rel_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "relations" --language "PL" --epochs 3 --model_name 'mDistil' --model_path '../models/distilbert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=plrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mdistil_pl_rel_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "relations" --language "PL" --epochs 3 --model_name 'mDistil' --model_path '../models/distilbert-base-multilingual-cased' --train_sample_ratio 0.05
EOF


sbatch --job-name=romqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mdistil_ro_med_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "medication" --language "RO" --epochs 3 --model_name 'mDistil' --model_path '../models/distilbert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=romqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mdistil_ro_med_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "medication" --language "RO" --epochs 3 --model_name 'mDistil' --model_path '../models/distilbert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=romqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mdistil_ro_med_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "medication" --language "RO" --epochs 3 --model_name 'mDistil' --model_path '../models/distilbert-base-multilingual-cased' --train_sample_ratio 0.2
EOF


sbatch --job-name=rorqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mdistil_ro_rel_1.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "relations" --language "RO" --epochs 3 --model_name 'mDistil' --model_path '../models/distilbert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=rorqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mdistil_ro_rel_2.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "relations" --language "RO" --epochs 3 --model_name 'mDistil' --model_path '../models/distilbert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=rorqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mdistil_ro_rel_3.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "relations" --language "RO" --epochs 3 --model_name 'mDistil' --model_path '../models/distilbert-base-multilingual-cased' --train_sample_ratio 0.05
EOF




sbatch --job-name=enmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mdistil_en_full_med_1.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "medication" --language "EN_FULL" --epochs 3 --model_name 'mDistil' --model_path '../models/distilbert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=enmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mdistil_en_full_med_2.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "medication" --language "EN_FULL" --epochs 3 --model_name 'mDistil' --model_path '../models/distilbert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=enmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mdistil_en_full_med_3.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "medication" --language "EN_FULL" --epochs 3 --model_name 'mDistil' --model_path '../models/distilbert-base-multilingual-cased' --train_sample_ratio 0.2
EOF


sbatch --job-name=enrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mdistil_en_full_rel_1.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "relations" --language "EN_FULL" --epochs 3 --model_name 'mDistil' --model_path '../models/distilbert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=enrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mdistil_en_full_rel_2.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "relations" --language "EN_FULL" --epochs 3 --model_name 'mDistil' --model_path '../models/distilbert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=enrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mdistil_en_full_rel_3.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "relations" --language "EN_FULL" --epochs 3 --model_name 'mDistil' --model_path '../models/distilbert-base-multilingual-cased' --train_sample_ratio 0.05
EOF


sbatch --job-name=enmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mdistil_en_med_1.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "medication" --language "EN" --epochs 3 --model_name 'mDistil' --model_path '../models/distilbert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=enmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mdistil_en_med_2.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "medication" --language "EN" --epochs 3 --model_name 'mDistil' --model_path '../models/distilbert-base-multilingual-cased' --train_sample_ratio 0.2
EOF
sbatch --job-name=enmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mdistil_en_med_3.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "medication" --language "EN" --epochs 3 --model_name 'mDistil' --model_path '../models/distilbert-base-multilingual-cased' --train_sample_ratio 0.2
EOF


sbatch --job-name=enrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mdistil_en_rel_1.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "relations" --language "EN" --epochs 3 --model_name 'mDistil' --model_path '../models/distilbert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=enrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mdistil_en_rel_2.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "relations" --language "EN" --epochs 3 --model_name 'mDistil' --model_path '../models/distilbert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
sbatch --job-name=enrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/mdistil_en_rel_3.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "relations" --language "EN" --epochs 3 --model_name 'mDistil' --model_path '../models/distilbert-base-multilingual-cased' --train_sample_ratio 0.05
EOF
















#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################

sbatch --job-name=bgmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmr_bg_med_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "medication" --language "BG" --epochs 3 --model_name 'XLMR' --model_path '../models/xlm-roberta-base' --train_sample_ratio 0.2
EOF


sbatch --job-name=bgmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmr_bg_med_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "medication" --language "BG" --epochs 3 --model_name 'XLMR' --model_path '../models/xlm-roberta-base' --train_sample_ratio 0.2
EOF
sbatch --job-name=bgmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmr_bg_med_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "medication" --language "BG" --epochs 3 --model_name 'XLMR' --model_path '../models/xlm-roberta-base' --train_sample_ratio 0.2
EOF


sbatch --job-name=bgrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmr_bg_rel_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "relations" --language "BG" --epochs 3 --model_name 'XLMR' --model_path '../models/xlm-roberta-base' --train_sample_ratio 0.05
EOF
sbatch --job-name=bgrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmr_bg_rel_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "relations" --language "BG" --epochs 3 --model_name 'XLMR' --model_path '../models/xlm-roberta-base' --train_sample_ratio 0.05
EOF
sbatch --job-name=bgrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmr_bg_rel_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "relations" --language "BG" --epochs 3 --model_name 'XLMR' --model_path '../models/xlm-roberta-base' --train_sample_ratio 0.05
EOF

sbatch --job-name=csmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmr_cs_med_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "medication" --language "CS" --epochs 3 --model_name 'XLMR' --model_path '../models/xlm-roberta-base' --train_sample_ratio 0.2
EOF
sbatch --job-name=csmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmr_cs_med_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "medication" --language "CS" --epochs 3 --model_name 'XLMR' --model_path '../models/xlm-roberta-base' --train_sample_ratio 0.2
EOF
sbatch --job-name=csmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmr_cs_med_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "medication" --language "CS" --epochs 3 --model_name 'XLMR' --model_path '../models/xlm-roberta-base' --train_sample_ratio 0.2
EOF


sbatch --job-name=csrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmr_cs_rel_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "relations" --language "CS" --epochs 3 --model_name 'XLMR' --model_path '../models/xlm-roberta-base' --train_sample_ratio 0.05
EOF
sbatch --job-name=csrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmr_cs_rel_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "relations" --language "CS" --epochs 3 --model_name 'XLMR' --model_path '../models/xlm-roberta-base' --train_sample_ratio 0.05
EOF
sbatch --job-name=csrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmr_cs_rel_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "relations" --language "CS" --epochs 3 --model_name 'XLMR' --model_path '../models/xlm-roberta-base' --train_sample_ratio 0.05
EOF



sbatch --job-name=elmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmr_el_med_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "medication" --language "EL" --epochs 3 --model_name 'XLMR' --model_path '../models/xlm-roberta-base' --train_sample_ratio 0.2
EOF
sbatch --job-name=elmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmr_el_med_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "medication" --language "EL" --epochs 3 --model_name 'XLMR' --model_path '../models/xlm-roberta-base' --train_sample_ratio 0.2
EOF
sbatch --job-name=elmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmr_el_med_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "medication" --language "EL" --epochs 3 --model_name 'XLMR' --model_path '../models/xlm-roberta-base' --train_sample_ratio 0.2
EOF


sbatch --job-name=elrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmr_el_rel_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=120G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "relations" --language "EL" --epochs 3 --model_name 'XLMR' --model_path '../models/xlm-roberta-base' --train_sample_ratio 0.05
EOF
sbatch --job-name=elrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmr_el_rel_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "relations" --language "EL" --epochs 3 --model_name 'XLMR' --model_path '../models/xlm-roberta-base' --train_sample_ratio 0.05
EOF
sbatch --job-name=elrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmr_el_rel_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=120G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "relations" --language "EL" --epochs 3 --model_name 'XLMR' --model_path '../models/xlm-roberta-base' --train_sample_ratio 0.05
EOF




sbatch --job-name=esmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmr_es_med_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "medication" --language "ES" --epochs 3 --model_name 'XLMR' --model_path '../models/xlm-roberta-base' --train_sample_ratio 0.2
EOF
sbatch --job-name=esmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmr_es_med_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "medication" --language "ES" --epochs 3 --model_name 'XLMR' --model_path '../models/xlm-roberta-base' --train_sample_ratio 0.2
EOF
sbatch --job-name=esmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmr_es_med_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "medication" --language "ES" --epochs 3 --model_name 'XLMR' --model_path '../models/xlm-roberta-base' --train_sample_ratio 0.2
EOF


sbatch --job-name=esrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmr_es_rel_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "relations" --language "ES" --epochs 3 --model_name 'XLMR' --model_path '../models/xlm-roberta-base' --train_sample_ratio 0.05
EOF
sbatch --job-name=esrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmr_es_rel_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "relations" --language "ES" --epochs 3 --model_name 'XLMR' --model_path '../models/xlm-roberta-base' --train_sample_ratio 0.05
EOF
sbatch --job-name=esrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmr_es_rel_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "relations" --language "ES" --epochs 3 --model_name 'XLMR' --model_path '../models/xlm-roberta-base' --train_sample_ratio 0.05
EOF


sbatch --job-name=plmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmr_pl_med_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "medication" --language "PL" --epochs 3 --model_name 'XLMR' --model_path '../models/xlm-roberta-base' --train_sample_ratio 0.2
EOF
sbatch --job-name=plmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmr_pl_med_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "medication" --language "PL" --epochs 3 --model_name 'XLMR' --model_path '../models/xlm-roberta-base' --train_sample_ratio 0.2
EOF
sbatch --job-name=plmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmr_pl_med_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "medication" --language "PL" --epochs 3 --model_name 'XLMR' --model_path '../models/xlm-roberta-base' --train_sample_ratio 0.2
EOF


sbatch --job-name=plrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmr_pl_rel_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "relations" --language "PL" --epochs 3 --model_name 'XLMR' --model_path '../models/xlm-roberta-base' --train_sample_ratio 0.05
EOF
sbatch --job-name=plrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmr_pl_rel_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "relations" --language "PL" --epochs 3 --model_name 'XLMR' --model_path '../models/xlm-roberta-base' --train_sample_ratio 0.05
EOF
sbatch --job-name=plrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmr_pl_rel_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "relations" --language "PL" --epochs 3 --model_name 'XLMR' --model_path '../models/xlm-roberta-base' --train_sample_ratio 0.05
EOF


sbatch --job-name=romqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmr_ro_med_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "medication" --language "RO" --epochs 3 --model_name 'XLMR' --model_path '../models/xlm-roberta-base' --train_sample_ratio 0.2
EOF
sbatch --job-name=romqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmr_ro_med_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "medication" --language "RO" --epochs 3 --model_name 'XLMR' --model_path '../models/xlm-roberta-base' --train_sample_ratio 0.2
EOF
sbatch --job-name=romqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmr_ro_med_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "medication" --language "RO" --epochs 3 --model_name 'XLMR' --model_path '../models/xlm-roberta-base' --train_sample_ratio 0.2
EOF


sbatch --job-name=rorqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmr_ro_rel_1.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "relations" --language "RO" --epochs 3 --model_name 'XLMR' --model_path '../models/xlm-roberta-base' --train_sample_ratio 0.05
EOF
sbatch --job-name=rorqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmr_ro_rel_2.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "relations" --language "RO" --epochs 3 --model_name 'XLMR' --model_path '../models/xlm-roberta-base' --train_sample_ratio 0.05
EOF
sbatch --job-name=rorqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmr_ro_rel_3.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "relations" --language "RO" --epochs 3 --model_name 'XLMR' --model_path '../models/xlm-roberta-base' --train_sample_ratio 0.05
EOF




sbatch --job-name=enmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmr_en_full_med_1.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "medication" --language "EN_FULL" --epochs 3 --model_name 'XLMR' --model_path '../models/xlm-roberta-base' --train_sample_ratio 0.2
EOF
sbatch --job-name=enmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmr_en_full_med_2.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "medication" --language "EN_FULL" --epochs 3 --model_name 'XLMR' --model_path '../models/xlm-roberta-base' --train_sample_ratio 0.2
EOF
sbatch --job-name=enmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmr_en_full_med_3.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "medication" --language "EN_FULL" --epochs 3 --model_name 'XLMR' --model_path '../models/xlm-roberta-base' --train_sample_ratio 0.2
EOF


sbatch --job-name=enrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmr_en_full_rel_1.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "relations" --language "EN_FULL" --epochs 3 --model_name 'XLMR' --model_path '../models/xlm-roberta-base' --train_sample_ratio 0.05
EOF
sbatch --job-name=enrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmr_en_full_rel_2.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "relations" --language "EN_FULL" --epochs 3 --model_name 'XLMR' --model_path '../models/xlm-roberta-base' --train_sample_ratio 0.05
EOF
sbatch --job-name=enrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmr_en_full_rel_3.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "relations" --language "EN_FULL" --epochs 3 --model_name 'XLMR' --model_path '../models/xlm-roberta-base' --train_sample_ratio 0.05
EOF


sbatch --job-name=enmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmr_en_med_1.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "medication" --language "EN" --epochs 3 --model_name 'XLMR' --model_path '../models/xlm-roberta-base' --train_sample_ratio 0.2
EOF
sbatch --job-name=enmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmr_en_med_2.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "medication" --language "EN" --epochs 3 --model_name 'XLMR' --model_path '../models/xlm-roberta-base' --train_sample_ratio 0.2
EOF
sbatch --job-name=enmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmr_en_med_3.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "medication" --language "EN" --epochs 3 --model_name 'XLMR' --model_path '../models/xlm-roberta-base' --train_sample_ratio 0.2
EOF


sbatch --job-name=enrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmr_en_rel_1.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "relations" --language "EN" --epochs 3 --model_name 'XLMR' --model_path '../models/xlm-roberta-base' --train_sample_ratio 0.05
EOF
sbatch --job-name=enrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmr_en_rel_2.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "relations" --language "EN" --epochs 3 --model_name 'XLMR' --model_path '../models/xlm-roberta-base' --train_sample_ratio 0.05
EOF
sbatch --job-name=enrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmr_en_rel_3.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "relations" --language "EN" --epochs 3 --model_name 'XLMR' --model_path '../models/xlm-roberta-base' --train_sample_ratio 0.05
EOF






#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################


sbatch --job-name=bgmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmrlarge_bg_med_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "medication" --language "BG" --epochs 3 --model_name 'XLMRLarge' --model_path '../models/xlm-roberta-large' --train_sample_ratio 0.2
EOF


sbatch --job-name=bgmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmrlarge_bg_med_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "medication" --language "BG" --epochs 3 --model_name 'XLMRLarge' --model_path '../models/xlm-roberta-large' --train_sample_ratio 0.2
EOF
sbatch --job-name=bgmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmrlarge_bg_med_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "medication" --language "BG" --epochs 3 --model_name 'XLMRLarge' --model_path '../models/xlm-roberta-large' --train_sample_ratio 0.2
EOF


sbatch --job-name=bgrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmrlarge_bg_rel_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "relations" --language "BG" --epochs 3 --model_name 'XLMRLarge' --model_path '../models/xlm-roberta-large' --train_sample_ratio 0.05
EOF
sbatch --job-name=bgrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmrlarge_bg_rel_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "relations" --language "BG" --epochs 3 --model_name 'XLMRLarge' --model_path '../models/xlm-roberta-large' --train_sample_ratio 0.05
EOF
sbatch --job-name=bgrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmrlarge_bg_rel_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "relations" --language "BG" --epochs 3 --model_name 'XLMRLarge' --model_path '../models/xlm-roberta-large' --train_sample_ratio 0.05
EOF

sbatch --job-name=csmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmrlarge_cs_med_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "medication" --language "CS" --epochs 3 --model_name 'XLMRLarge' --model_path '../models/xlm-roberta-large' --train_sample_ratio 0.2
EOF
sbatch --job-name=csmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmrlarge_cs_med_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "medication" --language "CS" --epochs 3 --model_name 'XLMRLarge' --model_path '../models/xlm-roberta-large' --train_sample_ratio 0.2
EOF
sbatch --job-name=csmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmrlarge_cs_med_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "medication" --language "CS" --epochs 3 --model_name 'XLMRLarge' --model_path '../models/xlm-roberta-large' --train_sample_ratio 0.2
EOF


sbatch --job-name=csrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmrlarge_cs_rel_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "relations" --language "CS" --epochs 3 --model_name 'XLMRLarge' --model_path '../models/xlm-roberta-large' --train_sample_ratio 0.05
EOF
sbatch --job-name=csrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmrlarge_cs_rel_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "relations" --language "CS" --epochs 3 --model_name 'XLMRLarge' --model_path '../models/xlm-roberta-large' --train_sample_ratio 0.05
EOF
sbatch --job-name=csrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmrlarge_cs_rel_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "relations" --language "CS" --epochs 3 --model_name 'XLMRLarge' --model_path '../models/xlm-roberta-large' --train_sample_ratio 0.05
EOF



sbatch --job-name=elmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmrlarge_el_med_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "medication" --language "EL" --epochs 3 --model_name 'XLMRLarge' --model_path '../models/xlm-roberta-large' --train_sample_ratio 0.2
EOF
sbatch --job-name=elmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmrlarge_el_med_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "medication" --language "EL" --epochs 3 --model_name 'XLMRLarge' --model_path '../models/xlm-roberta-large' --train_sample_ratio 0.2
EOF
sbatch --job-name=elmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmrlarge_el_med_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "medication" --language "EL" --epochs 3 --model_name 'XLMRLarge' --model_path '../models/xlm-roberta-large' --train_sample_ratio 0.2
EOF


sbatch --job-name=elrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmrlarge_el_rel_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=120G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "relations" --language "EL" --epochs 3 --model_name 'XLMRLarge' --model_path '../models/xlm-roberta-large' --train_sample_ratio 0.05
EOF
sbatch --job-name=elrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmrlarge_el_rel_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "relations" --language "EL" --epochs 3 --model_name 'XLMRLarge' --model_path '../models/xlm-roberta-large' --train_sample_ratio 0.05
EOF
sbatch --job-name=elrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmrlarge_el_rel_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=120G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "relations" --language "EL" --epochs 3 --model_name 'XLMRLarge' --model_path '../models/xlm-roberta-large' --train_sample_ratio 0.05
EOF




sbatch --job-name=esmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmrlarge_es_med_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "medication" --language "ES" --epochs 3 --model_name 'XLMRLarge' --model_path '../models/xlm-roberta-large' --train_sample_ratio 0.2
EOF
sbatch --job-name=esmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmrlarge_es_med_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "medication" --language "ES" --epochs 3 --model_name 'XLMRLarge' --model_path '../models/xlm-roberta-large' --train_sample_ratio 0.2
EOF
sbatch --job-name=esmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmrlarge_es_med_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "medication" --language "ES" --epochs 3 --model_name 'XLMRLarge' --model_path '../models/xlm-roberta-large' --train_sample_ratio 0.2
EOF


sbatch --job-name=esrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmrlarge_es_rel_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "relations" --language "ES" --epochs 3 --model_name 'XLMRLarge' --model_path '../models/xlm-roberta-large' --train_sample_ratio 0.05
EOF
sbatch --job-name=esrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmrlarge_es_rel_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "relations" --language "ES" --epochs 3 --model_name 'XLMRLarge' --model_path '../models/xlm-roberta-large' --train_sample_ratio 0.05
EOF
sbatch --job-name=esrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmrlarge_es_rel_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "relations" --language "ES" --epochs 3 --model_name 'XLMRLarge' --model_path '../models/xlm-roberta-large' --train_sample_ratio 0.05
EOF


sbatch --job-name=plmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmrlarge_pl_med_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "medication" --language "PL" --epochs 3 --model_name 'XLMRLarge' --model_path '../models/xlm-roberta-large' --train_sample_ratio 0.2
EOF
sbatch --job-name=plmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmrlarge_pl_med_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "medication" --language "PL" --epochs 3 --model_name 'XLMRLarge' --model_path '../models/xlm-roberta-large' --train_sample_ratio 0.2
EOF
sbatch --job-name=plmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmrlarge_pl_med_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "medication" --language "PL" --epochs 3 --model_name 'XLMRLarge' --model_path '../models/xlm-roberta-large' --train_sample_ratio 0.2
EOF


sbatch --job-name=plrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmrlarge_pl_rel_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "relations" --language "PL" --epochs 3 --model_name 'XLMRLarge' --model_path '../models/xlm-roberta-large' --train_sample_ratio 0.05
EOF
sbatch --job-name=plrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmrlarge_pl_rel_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "relations" --language "PL" --epochs 3 --model_name 'XLMRLarge' --model_path '../models/xlm-roberta-large' --train_sample_ratio 0.05
EOF
sbatch --job-name=plrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmrlarge_pl_rel_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "relations" --language "PL" --epochs 3 --model_name 'XLMRLarge' --model_path '../models/xlm-roberta-large' --train_sample_ratio 0.05
EOF


sbatch --job-name=romqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmrlarge_ro_med_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "medication" --language "RO" --epochs 3 --model_name 'XLMRLarge' --model_path '../models/xlm-roberta-large' --train_sample_ratio 0.2
EOF
sbatch --job-name=romqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmrlarge_ro_med_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "medication" --language "RO" --epochs 3 --model_name 'XLMRLarge' --model_path '../models/xlm-roberta-large' --train_sample_ratio 0.2
EOF
sbatch --job-name=romqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmrlarge_ro_med_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "medication" --language "RO" --epochs 3 --model_name 'XLMRLarge' --model_path '../models/xlm-roberta-large' --train_sample_ratio 0.2
EOF


sbatch --job-name=rorqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmrlarge_ro_rel_1.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "relations" --language "RO" --epochs 3 --model_name 'XLMRLarge' --model_path '../models/xlm-roberta-large' --train_sample_ratio 0.05
EOF
sbatch --job-name=rorqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmrlarge_ro_rel_2.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "relations" --language "RO" --epochs 3 --model_name 'XLMRLarge' --model_path '../models/xlm-roberta-large' --train_sample_ratio 0.05
EOF
sbatch --job-name=rorqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmrlarge_ro_rel_3.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "relations" --language "RO" --epochs 3 --model_name 'XLMRLarge' --model_path '../models/xlm-roberta-large' --train_sample_ratio 0.05
EOF




sbatch --job-name=enmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmrlarge_en_full_med_1.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "medication" --language "EN_FULL" --epochs 3 --model_name 'XLMRLarge' --model_path '../models/xlm-roberta-large' --train_sample_ratio 0.2
EOF
sbatch --job-name=enmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmrlarge_en_full_med_2.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "medication" --language "EN_FULL" --epochs 3 --model_name 'XLMRLarge' --model_path '../models/xlm-roberta-large' --train_sample_ratio 0.2
EOF
sbatch --job-name=enmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmrlarge_en_full_med_3.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "medication" --language "EN_FULL" --epochs 3 --model_name 'XLMRLarge' --model_path '../models/xlm-roberta-large' --train_sample_ratio 0.2
EOF


sbatch --job-name=enrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmrlarge_en_full_rel_1.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "relations" --language "EN_FULL" --epochs 3 --model_name 'XLMRLarge' --model_path '../models/xlm-roberta-large' --train_sample_ratio 0.05
EOF
sbatch --job-name=enrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmrlarge_en_full_rel_2.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "relations" --language "EN_FULL" --epochs 3 --model_name 'XLMRLarge' --model_path '../models/xlm-roberta-large' --train_sample_ratio 0.05
EOF
sbatch --job-name=enrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmrlarge_en_full_rel_3.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "relations" --language "EN_FULL" --epochs 3 --model_name 'XLMRLarge' --model_path '../models/xlm-roberta-large' --train_sample_ratio 0.05
EOF


sbatch --job-name=enmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmrlarge_en_med_1.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "medication" --language "EN" --epochs 3 --model_name 'XLMRLarge' --model_path '../models/xlm-roberta-large' --train_sample_ratio 0.2
EOF
sbatch --job-name=enmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmrlarge_en_med_2.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "medication" --language "EN" --epochs 3 --model_name 'XLMRLarge' --model_path '../models/xlm-roberta-large' --train_sample_ratio 0.2
EOF
sbatch --job-name=enmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmrlarge_en_med_3.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "medication" --language "EN" --epochs 3 --model_name 'XLMRLarge' --model_path '../models/xlm-roberta-large' --train_sample_ratio 0.2
EOF


sbatch --job-name=enrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmrlarge_en_rel_1.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "relations" --language "EN" --epochs 3 --model_name 'XLMRLarge' --model_path '../models/xlm-roberta-large' --train_sample_ratio 0.05
EOF
sbatch --job-name=enrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmrlarge_en_rel_2.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "relations" --language "EN" --epochs 3 --model_name 'XLMRLarge' --model_path '../models/xlm-roberta-large' --train_sample_ratio 0.05
EOF
sbatch --job-name=enrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/xlmrlarge_en_rel_3.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "relations" --language "EN" --epochs 3 --model_name 'XLMRLarge' --model_path '../models/xlm-roberta-large' --train_sample_ratio 0.05
EOF






#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################



sbatch --job-name=bgmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/clinical_bg_med_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "medication" --language "BG" --epochs 3 --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.2
EOF


sbatch --job-name=bgmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/clinical_bg_med_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "medication" --language "BG" --epochs 3 --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.2
EOF
sbatch --job-name=bgmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/clinical_bg_med_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "medication" --language "BG" --epochs 3 --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.2
EOF


sbatch --job-name=bgrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/clinical_bg_rel_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "relations" --language "BG" --epochs 3 --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.05
EOF
sbatch --job-name=bgrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/clinical_bg_rel_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "relations" --language "BG" --epochs 3 --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.05
EOF
sbatch --job-name=bgrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/clinical_bg_rel_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "relations" --language "BG" --epochs 3 --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.05
EOF

sbatch --job-name=csmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/clinical_cs_med_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "medication" --language "CS" --epochs 3 --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.2
EOF
sbatch --job-name=csmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/clinical_cs_med_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "medication" --language "CS" --epochs 3 --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.2
EOF
sbatch --job-name=csmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/clinical_cs_med_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "medication" --language "CS" --epochs 3 --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.2
EOF


sbatch --job-name=csrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/clinical_cs_rel_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "relations" --language "CS" --epochs 3 --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.05
EOF
sbatch --job-name=csrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/clinical_cs_rel_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "relations" --language "CS" --epochs 3 --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.05
EOF
sbatch --job-name=csrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/clinical_cs_rel_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "relations" --language "CS" --epochs 3 --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.05
EOF



sbatch --job-name=elmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/clinical_el_med_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "medication" --language "EL" --epochs 3 --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.2
EOF
sbatch --job-name=elmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/clinical_el_med_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "medication" --language "EL" --epochs 3 --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.2
EOF
sbatch --job-name=elmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/clinical_el_med_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "medication" --language "EL" --epochs 3 --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.2
EOF


sbatch --job-name=elrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/clinical_el_rel_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=120G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "relations" --language "EL" --epochs 3 --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.05
EOF
sbatch --job-name=elrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/clinical_el_rel_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "relations" --language "EL" --epochs 3 --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.05
EOF
sbatch --job-name=elrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/clinical_el_rel_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=120G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "relations" --language "EL" --epochs 3 --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.05
EOF




sbatch --job-name=esmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/clinical_es_med_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "medication" --language "ES" --epochs 3 --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.2
EOF
sbatch --job-name=esmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/clinical_es_med_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "medication" --language "ES" --epochs 3 --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.2
EOF
sbatch --job-name=esmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/clinical_es_med_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "medication" --language "ES" --epochs 3 --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.2
EOF


sbatch --job-name=esrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/clinical_es_rel_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "relations" --language "ES" --epochs 3 --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.05
EOF
sbatch --job-name=esrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/clinical_es_rel_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "relations" --language "ES" --epochs 3 --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.05
EOF
sbatch --job-name=esrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/clinical_es_rel_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "relations" --language "ES" --epochs 3 --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.05
EOF


sbatch --job-name=plmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/clinical_pl_med_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "medication" --language "PL" --epochs 3 --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.2
EOF
sbatch --job-name=plmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/clinical_pl_med_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "medication" --language "PL" --epochs 3 --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.2
EOF
sbatch --job-name=plmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/clinical_pl_med_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "medication" --language "PL" --epochs 3 --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.2
EOF


sbatch --job-name=plrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/clinical_pl_rel_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "relations" --language "PL" --epochs 3 --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.05
EOF
sbatch --job-name=plrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/clinical_pl_rel_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "relations" --language "PL" --epochs 3 --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.05
EOF
sbatch --job-name=plrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/clinical_pl_rel_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "relations" --language "PL" --epochs 3 --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.05
EOF


sbatch --job-name=romqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/clinical_ro_med_1.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "medication" --language "RO" --epochs 3 --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.2
EOF
sbatch --job-name=romqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/clinical_ro_med_2.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "medication" --language "RO" --epochs 3 --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.2
EOF
sbatch --job-name=romqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/clinical_ro_med_3.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "medication" --language "RO" --epochs 3 --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.2
EOF


sbatch --job-name=rorqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/clinical_ro_rel_1.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "relations" --language "RO" --epochs 3 --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.05
EOF
sbatch --job-name=rorqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/clinical_ro_rel_2.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "relations" --language "RO" --epochs 3 --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.05
EOF
sbatch --job-name=rorqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/clinical_ro_rel_3.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "relations" --language "RO" --epochs 3 --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.05
EOF




sbatch --job-name=enmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/clinical_en_full_med_1.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "medication" --language "EN_FULL" --epochs 3 --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.2
EOF
sbatch --job-name=enmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/clinical_en_full_med_2.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "medication" --language "EN_FULL" --epochs 3 --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.2
EOF
sbatch --job-name=enmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/clinical_en_full_med_3.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "medication" --language "EN_FULL" --epochs 3 --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.2
EOF


sbatch --job-name=enrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/clinical_en_full_rel_1.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "relations" --language "EN_FULL" --epochs 3 --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.05
EOF
sbatch --job-name=enrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/clinical_en_full_rel_2.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "relations" --language "EN_FULL" --epochs 3 --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.05
EOF
sbatch --job-name=enrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/clinical_en_full_rel_3.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "relations" --language "EN_FULL" --epochs 3 --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.05
EOF


sbatch --job-name=enmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/clinical_en_med_1.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "medication" --language "EN" --epochs 3 --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.2
EOF
sbatch --job-name=enmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/clinical_en_med_2.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "medication" --language "EN" --epochs 3 --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.2
EOF
sbatch --job-name=enmqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/clinical_en_med_3.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "medication" --language "EN" --epochs 3 --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.2
EOF


sbatch --job-name=enrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/clinical_en_rel_1.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 1 --subset "relations" --language "EN" --epochs 3 --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.05
EOF
sbatch --job-name=enrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/clinical_en_rel_2.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 2 --subset "relations" --language "EN" --epochs 3 --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.05
EOF
sbatch --job-name=enrqa \
     --output=scripts/slurm_outputs/qa/paragraphs_inter_monolingual/clinical_en_rel_3.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G \
     --exclude=dll-4gpu1,dll-4gpu2,dll-8gpu1,dll-8gpu2,tdll-8gpu3,tdll-8gpu4,tdll-8gpu5,tdll-8gpu6,tdll-8gpu7,dll-8gpu3,dll-8gpu4,dll-8gpu5,dll-8gpu6,dll-10gpu1,dll-10gpu2,dll-10gpu3 <<"EOF"
#!/bin/bash
python3 measure_qa_intersections.py --seed 3 --subset "relations" --language "EN" --epochs 3 --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.05
EOF
