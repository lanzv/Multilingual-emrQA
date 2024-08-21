#!/bin/sh
#SBATCH -J emrQAtrans
#SBATCH -o scripts/slurm_outputs/translate.out
#SBATCH -p cpu-troja



sbatch --job-name=alro_med \
     --output=scripts/slurm_outputs/alignments/fastalign/ro_med.out \
     --partition=gpu-troja <<"EOF"
#!/bin/bash
python3 measure_aligners.py --language "ro" --dataset_title "medication" --translation_dataset "./data/translations/medication_ro.json" --aligner_name "FastAlign" --aligner_path "../models/fast_align/build/roen/align_medication_ro_parallel.txt"
EOF

sbatch --job-name=alro_rel \
     --output=scripts/slurm_outputs/alignments/fastalign/ro_rel.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G <<"EOF"
#!/bin/bash
python3 measure_aligners.py --language "ro" --dataset_title "relations" --translation_dataset "./data/translations/relations_ro.json" --aligner_name "FastAlign" --aligner_path "../models/fast_align/build/roen/align_relations_ro_parallel.txt"
EOF



sbatch --job-name=alpl_med \
     --output=scripts/slurm_outputs/alignments/fastalign/pl_med.out \
     --partition=gpu-troja <<"EOF"
#!/bin/bash
python3 measure_aligners.py --language "pl" --dataset_title "medication" --translation_dataset "./data/translations/medication_pl.json" --aligner_name "FastAlign" --aligner_path "../models/fast_align/build/plen/align_medication_pl_parallel.txt"
EOF

sbatch --job-name=alpl_rel \
     --output=scripts/slurm_outputs/alignments/fastalign/pl_rel.out \
     --partition=gpu-troja <<"EOF"
#!/bin/bash
python3 measure_aligners.py --language "pl" --dataset_title "relations" --translation_dataset "./data/translations/relations_pl.json" --aligner_name "FastAlign" --aligner_path "../models/fast_align/build/plen/align_relations_pl_parallel.txt"
EOF



sbatch --job-name=albg_med \
     --output=scripts/slurm_outputs/alignments/fastalign/bg_med.out \
     --partition=gpu-troja <<"EOF"
#!/bin/bash
python3 measure_aligners.py --language "bg" --dataset_title "medication" --translation_dataset "./data/translations/medication_bg.json" --aligner_name "FastAlign" --aligner_path "../models/fast_align/build/bgen/align_medication_bg_parallel.txt"
EOF

sbatch --job-name=albg_rel \
     --output=scripts/slurm_outputs/alignments/fastalign/bg_rel.out \
     --partition=gpu-troja \
     --gpus=1 \
     --mem-per-gpu=90G <<"EOF"
#!/bin/bash
python3 measure_aligners.py --language "bg" --dataset_title "relations" --translation_dataset "./data/translations/relations_bg.json" --aligner_name "FastAlign" --aligner_path "../models/fast_align/build/bgen/align_relations_bg_parallel.txt"
EOF



sbatch --job-name=alel_med \
     --output=scripts/slurm_outputs/alignments/fastalign/el_med.out \
     --partition=gpu-troja <<"EOF"
#!/bin/bash
python3 measure_aligners.py --language "el" --dataset_title "medication" --translation_dataset "./data/translations/medication_el.json" --aligner_name "FastAlign" --aligner_path "../models/fast_align/build/elen/align_medication_el_parallel.txt"
EOF

sbatch --job-name=alel_rel \
     --output=scripts/slurm_outputs/alignments/fastalign/el_rel.out \
     --partition=gpu-troja <<"EOF"
#!/bin/bash
python3 measure_aligners.py --language "el" --dataset_title "relations" --translation_dataset "./data/translations/relations_el.json" --aligner_name "FastAlign" --aligner_path "../models/fast_align/build/elen/align_relations_el_parallel.txt"
EOF



sbatch --job-name=alcs_med \
     --output=scripts/slurm_outputs/alignments/fastalign/cs_med.out \
     --partition=gpu-troja <<"EOF"
#!/bin/bash
python3 measure_aligners.py --language "cs" --dataset_title "medication" --translation_dataset "./data/translations/medication_cs.json" --aligner_name "FastAlign" --aligner_path "../models/fast_align/build/csen/align_medication_cs_parallel.txt"
EOF

sbatch --job-name=alcs_rel \
     --output=scripts/slurm_outputs/alignments/fastalign/cs_rel.out \
     --partition=gpu-troja <<"EOF"
#!/bin/bash
python3 measure_aligners.py --language "cs" --dataset_title "relations" --translation_dataset "./data/translations/relations_cs.json" --aligner_name "FastAlign" --aligner_path "../models/fast_align/build/csen/align_relations_cs_parallel.txt"
EOF



sbatch --job-name=ales_med \
     --output=scripts/slurm_outputs/alignments/fastalign/es_med.out \
     --partition=gpu-troja <<"EOF"
#!/bin/bash
python3 measure_aligners.py --language "es" --dataset_title "medication" --translation_dataset "./data/translations/medication_es.json" --aligner_name "FastAlign" --aligner_path "../models/fast_align/build/esen/align_medication_es_parallel.txt"
EOF

sbatch --job-name=ales_rel \
     --output=scripts/slurm_outputs/alignments/fastalign/es_rel.out \
     --partition=gpu-troja <<"EOF"
#!/bin/bash
python3 measure_aligners.py --language "es" --dataset_title "relations" --translation_dataset "./data/translations/relations_es.json" --aligner_name "FastAlign" --aligner_path "../models/fast_align/build/esen/align_relations_es_parallel.txt"
EOF