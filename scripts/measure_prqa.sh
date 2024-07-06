#!/bin/sh
#SBATCH -J cs_prqa
#SBATCH -o scripts/slurm_outputs/cs_prqa.out
#SBATCH -p cpu-ms

sbatch --job-name=mmedlev \
     --output=scripts/slurm_outputs/prqa/medication_levenshtein_mbert.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --nodelist=dll-3gpu1 <<"EOF"
#!/bin/bash
python3 measure_prqa.py --dataset './data/translation_aligners/medication_Levenshtein.json' --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF

sbatch --job-name=mmedfa \
     --output=scripts/slurm_outputs/prqa/medication_fastalign_mbert.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --nodelist=dll-3gpu2 <<"EOF"
#!/bin/bash
python3 measure_prqa.py --dataset './data/translation_aligners/medication_FastAlign.json' --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF

sbatch --job-name=mmedaw \
     --output=scripts/slurm_outputs/prqa/medication_awesome_mbert.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --nodelist=dll-3gpu3 <<"EOF"
#!/bin/bash
python3 measure_prqa.py --dataset './data/translation_aligners/medication_Awesome.json' --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.2
EOF

sbatch --job-name=cmedlev \
     --output=scripts/slurm_outputs/prqa/medication_levenshtein_clinicalbert.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --nodelist=dll-3gpu4 <<"EOF"
#!/bin/bash
python3 measure_prqa.py --dataset './data/translation_aligners/medication_Levenshtein.json' --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.2
EOF

sbatch --job-name=cmedfa \
     --output=scripts/slurm_outputs/prqa/medication_fastalign_clinicalbert.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --nodelist=dll-3gpu5 <<"EOF"
#!/bin/bash
python3 measure_prqa.py --dataset './data/translation_aligners/medication_FastAlign.json' --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.2
EOF

sbatch --job-name=cmedaw \
     --output=scripts/slurm_outputs/prqa/medication_awesome_clinicalbert.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --nodelist=dll-3gpu1 <<"EOF"
#!/bin/bash
python3 measure_prqa.py --dataset './data/translation_aligners/medication_Awesome.json' --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.2
EOF












sbatch --job-name=mrellev \
     --output=scripts/slurm_outputs/prqa/relations_levenshtein_mbert.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --nodelist=dll-3gpu2 <<"EOF"
#!/bin/bash
python3 measure_prqa.py --dataset './data/translation_aligners/relations_Levenshtein.json' --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF

sbatch --job-name=mrelfa \
     --output=scripts/slurm_outputs/prqa/relations_fastalign_mbert.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=100G \
     --nodelist=dll-3gpu2 <<"EOF"
#!/bin/bash
python3 measure_prqa.py --dataset './data/translation_aligners/relations_FastAlign.json' --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF

sbatch --job-name=mrelaw \
     --output=scripts/slurm_outputs/prqa/relations_awesome_mbert.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --nodelist=dll-3gpu4 <<"EOF"
#!/bin/bash
python3 measure_prqa.py --dataset './data/translation_aligners/relations_Awesome.json' --model_name 'mBERT' --model_path '../models/bert-base-multilingual-cased' --train_sample_ratio 0.05
EOF

sbatch --job-name=crellev \
     --output=scripts/slurm_outputs/prqa/relations_levenshtein_clinicalbert.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --nodelist=dll-3gpu5 <<"EOF"
#!/bin/bash
python3 measure_prqa.py --dataset './data/translation_aligners/relations_Levenshtein.json' --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.05
EOF

sbatch --job-name=crelfa \
     --output=scripts/slurm_outputs/prqa/relations_fastalign_clinicalbert.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --nodelist=dll-3gpu1 <<"EOF"
#!/bin/bash
python3 measure_prqa.py --dataset './data/translation_aligners/relations_FastAlign.json' --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.05
EOF

sbatch --job-name=crelaw \
     --output=scripts/slurm_outputs/prqa/relations_awesome_clinicalbert.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=90G \
     --nodelist=dll-3gpu4 <<"EOF"
#!/bin/bash
python3 measure_prqa.py --dataset './data/translation_aligners/relations_Awesome.json' --model_name 'ClinicalBERT' --model_path '../models/Bio_ClinicalBERT' --train_sample_ratio 0.05
EOF