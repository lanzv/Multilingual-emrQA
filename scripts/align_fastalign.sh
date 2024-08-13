#!/bin/sh
#SBATCH -J emrQAtrans
#SBATCH -o scripts/slurm_outputs/translate.out
#SBATCH -p cpu-ms



sbatch --job-name=alro_med \
     --output=scripts/slurm_outputs/alignments/fastalign/ro_med.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --nodelist=dll-8gpu1 <<"EOF"
#!/bin/bash
python3 - <<END
import subprocess
from transformers import AutoModel, AutoTokenizer
import itertools
import torch
import logging
from src.utils import tokenize

fast_align_directory="../models/fast_align/build"
 !! CZENG, medication !! command = f'./force_align.py "csen/fwd_params" "csen/fwd_err" "czeng/rev_params" "czeng/rev_err" "grow-diag-final-and" < "/home/lanz/personal_work_ms/resq/CS-PL-SP-RM-emrQA/medication_parallel.txt"'
process = subprocess.run(command, cwd=fast_align_directory, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
output = process.stdout

cached_alignment = ""
for sentences, alignment in zip(content.split("\n"), output.split("\n")):
    cached_alignment += "{} ||| {}\n".format(sentences, alignment)
 !! CZENG, medication !! text_file = open("czeng_medication.txt", "w")
text_file.write(cached_alignment)
text_file.close()
END
 
python measure_aligners.py --dataset_title "medication" --aligner_name "FastAlign" --aligner_path "../models/fast_align/build/czeng/czeng_medication.txt" --translation_dataset "./data/translations/medication_cs.json"
EOF

sbatch --job-name=alro_rel \
     --output=scripts/slurm_outputs/alignments/levenshtein/ro_rel.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --nodelist=dll-8gpu2 <<"EOF"
#!/bin/bash
python3 measure_aligners.py --language "ro" --dataset_title "relations" --translation_dataset "./data/translations/relations_ro.json" --aligner_name "Levenshtein" --aligner_path "../models/madlad400-3b-mt"
EOF



sbatch --job-name=alpl_med \
     --output=scripts/slurm_outputs/alignments/levenshtein/pl_med.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --nodelist=tdll-3gpu4 <<"EOF"
#!/bin/bash
python3 measure_aligners.py --language "pl" --dataset_title "medication" --translation_dataset "./data/translations/medication_pl.json" --aligner_name "Levenshtein" --aligner_path "../models/madlad400-3b-mt"
EOF

sbatch --job-name=alpl_rel \
     --output=scripts/slurm_outputs/alignments/levenshtein/pl_rel.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --nodelist=tdll-3gpu4 <<"EOF"
#!/bin/bash
python3 measure_aligners.py --language "pl" --dataset_title "relations" --translation_dataset "./data/translations/relations_pl.json" --aligner_name "Levenshtein" --aligner_path "../models/madlad400-3b-mt"
EOF



sbatch --job-name=albg_med \
     --output=scripts/slurm_outputs/alignments/levenshtein/bg_med.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=50G \
     --nodelist=tdll-8gpu1 <<"EOF"
#!/bin/bash
python3 measure_aligners.py --language "bg" --dataset_title "medication" --translation_dataset "./data/translations/medication_bg.json" --aligner_name "Levenshtein" --aligner_path "../models/madlad400-3b-mt"
EOF

sbatch --job-name=albg_rel \
     --output=scripts/slurm_outputs/alignments/levenshtein/bg_rel.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --nodelist=dll-8gpu1 <<"EOF"
#!/bin/bash
python3 measure_aligners.py --language "bg" --dataset_title "relations" --translation_dataset "./data/translations/relations_bg.json" --aligner_name "Levenshtein" --aligner_path "../models/madlad400-3b-mt"
EOF



sbatch --job-name=alel_med \
     --output=scripts/slurm_outputs/alignments/levenshtein/el_med.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --nodelist=dll-4gpu1 <<"EOF"
#!/bin/bash
python3 measure_aligners.py --language "el" --dataset_title "medication" --translation_dataset "./data/translations/medication_el.json" --aligner_name "Levenshtein" --aligner_path "../models/madlad400-3b-mt"
EOF

sbatch --job-name=alel_rel \
     --output=scripts/slurm_outputs/alignments/levenshtein/el_rel.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --nodelist=dll-4gpu2 <<"EOF"
#!/bin/bash
python3 measure_aligners.py --language "el" --dataset_title "relations" --translation_dataset "./data/translations/relations_el.json" --aligner_name "Levenshtein" --aligner_path "../models/madlad400-3b-mt"
EOF



sbatch --job-name=alcs_med \
     --output=scripts/slurm_outputs/alignments/levenshtein/cs_med.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --nodelist=dll-8gpu1 <<"EOF"
#!/bin/bash
python3 measure_aligners.py --language "cs" --dataset_title "medication" --translation_dataset "./data/translations/medication_cs.json" --aligner_name "Levenshtein" --aligner_path "../models/madlad400-3b-mt"
EOF

sbatch --job-name=alcs_rel \
     --output=scripts/slurm_outputs/alignments/levenshtein/cs_rel.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --nodelist=dll-8gpu2 <<"EOF"
#!/bin/bash
python3 measure_aligners.py --language "cs" --dataset_title "relations" --translation_dataset "./data/translations/relations_cs.json" --aligner_name "Levenshtein" --aligner_path "../models/madlad400-3b-mt"
EOF



sbatch --job-name=ales_med \
     --output=scripts/slurm_outputs/alignments/levenshtein/es_med.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --nodelist=dll-8gpu1 <<"EOF"
#!/bin/bash
python3 measure_aligners.py --language "es" --dataset_title "medication" --translation_dataset "./data/translations/medication_es.json" --aligner_name "Levenshtein" --aligner_path "../models/madlad400-3b-mt"
EOF

sbatch --job-name=ales_rel \
     --output=scripts/slurm_outputs/alignments/levenshtein/es_rel.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --nodelist=dll-8gpu2 <<"EOF"
#!/bin/bash
python3 measure_aligners.py --language "es" --dataset_title "relations" --translation_dataset "./data/translations/relations_es.json" --aligner_name "Levenshtein" --aligner_path "../models/madlad400-3b-mt"
EOF