#!/bin/sh
#SBATCH -J resq
#SBATCH -o scripts/slurm_outputs/resq.out
#SBATCH -p cpu-ms

sbatch --job-name=bg_resq \
     --output=scripts/slurm_outputs/resq/bg_resq.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --nodelist=dll-3gpu2 <<"EOF"
#!/bin/bash
python3 measure_resq.py --model_name 'ontotext' --model_path '../models/ontotext/xlmr_medical' --medication '../datasets/emrQA/medication_bg.json' --relations '../datasets/emrQA/relations_bg.json' --resq_folder '../datasets/resq/bg' 
EOF




#sbatch --job-name=cs_resq \
#     --output=scripts/slurm_outputs/resq/cs_resq.out \
#     --partition=gpu-ms \
#     --gpus=1 \
#     --mem-per-gpu=50G \
#     --nodelist=dll-3gpu3 <<"EOF"
##!/bin/bash
#python3 measure_resq.py --model_name 'ontotext' --model_path '../models/ontotext/xlmr_medical' --medication '../datasets/emrQA/medication_cs.json' --relations '../datasets/emrQA/relations_cs.json' --resq_folder '../datasets/resq/cs' 
#EOF





sbatch --job-name=el_prqa \
     --output=scripts/slurm_outputs/resq/el_resq.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --nodelist=dll-3gpu4 <<"EOF"
#!/bin/bash
python3 measure_resq.py --model_name 'ontotext' --model_path '../models/ontotext/xlmr_medical' --medication '../datasets/emrQA/medication_el.json' --relations '../datasets/emrQA/relations_el.json' --resq_folder '../datasets/resq/el' 
EOF





sbatch --job-name=pl_prqa \
     --output=scripts/slurm_outputs/resq/pl_resq.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --nodelist=dll-4gpu3 <<"EOF"
#!/bin/bash
python3 measure_resq.py --model_name 'ontotext' --model_path '../models/ontotext/xlmr_medical' --medication '../datasets/emrQA/medication_pl.json' --relations '../datasets/emrQA/relations_pl.json' --resq_folder '../datasets/resq/pl' 
EOF


#sbatch --job-name=3mrpl_prqa \
#     --output=scripts/slurm_outputs/resq/3mrpl_resq.out \
#     --partition=gpu-ms \
#     --gpus=1 \
#     --mem-per-gpu=50G \
#     --nodelist=dll-4gpu3 <<"EOF"
##!/bin/bash
#python3 measure_resq.py --model_name 'ontotext' --model_path '../models/ontotext/xlmr_medical' --medication '../datasets/emrQA/medication_pl.json' --relations '../datasets/emrQA/relations_pl.json' --resq_folder '../datasets/resq/pl' 
#EOF




sbatch --job-name=ro_prqa \
     --output=scripts/slurm_outputs/resq/ro_resq.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --nodelist=dll-4gpu3 <<"EOF"
#!/bin/bash
python3 measure_resq.py --model_name 'ontotext' --model_path '../models/ontotext/xlmr_medical' --medication '../datasets/emrQA/medication_ro.json' --relations '../datasets/emrQA/relations_ro.json' --resq_folder '../datasets/resq/ro' 
EOF




sbatch --job-name=en_prqa \
     --output=scripts/slurm_outputs/resq/en_resq.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --nodelist=dll-4gpu4 <<"EOF"
#!/bin/bash
python3 measure_resq.py --model_name 'ontotext' --model_path '../models/ontotext/xlmr_medical' --medication '../datasets/emrQA/medication_en.json' --relations '../datasets/emrQA/relations_en.json' --resq_folder '../datasets/resq/en' 
EOF
