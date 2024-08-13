#!/bin/sh
#SBATCH -J emrQAtrans
#SBATCH -o scripts/slurm_outputs/translate.out
#SBATCH -p cpu-ms



sbatch --job-name=alro_med \
     --output=scripts/slurm_outputs/alignments/levenshtein/ro_med.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --nodelist=dll-8gpu1 <<"EOF"
#!/bin/bash
python3 measure_aligners.py --language "ro" --dataset_title "medication" --translation_dataset "./data/translations/medication_ro.json" --aligner_name "Levenshtein" --aligner_path "../models/madlad400-3b-mt"
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
     --output=scripts/slurm_outputs/alignments/levenshtein/noprompt_pl_rel.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --nodelist=dll-3gpu4 <<"EOF"
#!/bin/bash
python3 measure_aligners.py --language "pl" --dataset_title "relations" --translation_dataset "./data/translations/noprompt_relations_pl.json" --aligner_name "Levenshtein" --aligner_path "../models/madlad400-3b-mt"
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