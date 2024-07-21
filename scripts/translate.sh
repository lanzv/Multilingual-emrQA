#!/bin/sh
#SBATCH -J emrQAtrans
#SBATCH -o scripts/slurm_outputs/translate.out
#SBATCH -p cpu-ms

sbatch --job-name=pl_med \
     --output=scripts/slurm_outputs/translations/pl_med.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --nodelist=dll-3gpu1 <<"EOF"
#!/bin/bash
python3 translate.py --translation True --topics "medication" --target_language "pl" --translated_medical_info_message 'Na podstawie raportów medycznych.#Na podstawie sprawozdań lekarskich.'
EOF

sbatch --job-name=pl_rel \
     --output=scripts/slurm_outputs/translations/pl_rel.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --nodelist=dll-3gpu2 <<"EOF"
#!/bin/bash
python3 translate.py --translation True --topics "relations" --target_language "pl" --translated_medical_info_message 'Na podstawie raportów medycznych.#Na podstawie sprawozdań lekarskich.'
EOF




sbatch --job-name=bg_med \
     --output=scripts/slurm_outputs/translations/bg_med.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --nodelist=dll-3gpu2 <<"EOF"
#!/bin/bash
python3 translate.py --translation True --topics "medication" --target_language "bg" --translated_medical_info_message 'Въз основа на медицинските доклади.#Въз основа на медицински доклади.#На базата на медицински доклади.#Въз основа на медицински съобщения.'
EOF

sbatch --job-name=bg_rel \
     --output=scripts/slurm_outputs/translations/bg_rel.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --nodelist=dll-3gpu3 <<"EOF"
#!/bin/bash
python3 translate.py --translation True --topics "relations" --target_language "bg" --translated_medical_info_message 'Въз основа на медицинските доклади.#Въз основа на медицински доклади.#На базата на медицински доклади.#Въз основа на медицински съобщения.'
EOF



sbatch --job-name=el_med \
     --output=scripts/slurm_outputs/translations/el_med.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --nodelist=dll-3gpu3 <<"EOF"
#!/bin/bash
python3 translate.py --translation True --topics "medication" --target_language "el" --translated_medical_info_message 'Βασισμένο σε ιατρικές εκθέσεις.#Με βάση ιατρικές εκθέσεις.#Βάσει ιατρικών εκθέσεων.#Με βάση τις ιατρικές εκθέσεις.#Βάσει των ιατρικών εκθέσεων.#Σύμφωνα με τις ιατρικές εκθέσεις.'
EOF

sbatch --job-name=el_rel \
     --output=scripts/slurm_outputs/translations/el_rel.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --nodelist=dll-3gpu4 <<"EOF"
#!/bin/bash
python3 translate.py --translation True --topics "relations" --target_language "el" --translated_medical_info_message 'Βασισμένο σε ιατρικές εκθέσεις.#Με βάση ιατρικές εκθέσεις.#Βάσει ιατρικών εκθέσεων.#Με βάση τις ιατρικές εκθέσεις.#Βάσει των ιατρικών εκθέσεων.#Σύμφωνα με τις ιατρικές εκθέσεις.'
EOF



sbatch --job-name=ro_med \
     --output=scripts/slurm_outputs/translations/ro_med.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --nodelist=dll-3gpu4 <<"EOF"
#!/bin/bash
python3 translate.py --translation True --topics "medication" --target_language "ro" --translated_medical_info_message 'Pe baza rapoartelor medicale.'
EOF

sbatch --job-name=ro_rel \
     --output=scripts/slurm_outputs/translations/ro_rel.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --nodelist=dll-3gpu5 <<"EOF"
#!/bin/bash
python3 translate.py --translation True --topics "relations" --target_language "ro" --translated_medical_info_message 'Pe baza rapoartelor medicale.'
EOF




sbatch --job-name=cs_med \
     --output=scripts/slurm_outputs/translations/cs_med.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --nodelist=dll-3gpu4 <<"EOF"
#!/bin/bash
python3 translate.py --translation True --topics "medication" --target_language "cs" --translated_medical_info_message 'Na základě lékařských zpráv.'
EOF

sbatch --job-name=cs_rel \
     --output=scripts/slurm_outputs/translations/cs_rel.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --nodelist=dll-3gpu5 <<"EOF"
#!/bin/bash
python3 translate.py --translation True --topics "relations" --target_language "cs" --translated_medical_info_message 'Na základě lékařských zpráv.'
EOF




sbatch --job-name=es_med \
     --output=scripts/slurm_outputs/translations/es_med.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --nodelist=dll-3gpu4 <<"EOF"
#!/bin/bash
python3 translate.py --translation True --topics "medication" --target_language "es" --translated_medical_info_message 'Basado en informes médicos.#Según los informes médicos.#De acuerdo con los informes médicos.#Con base en los informes médicos.#Fundado en informes médicos.'
EOF

sbatch --job-name=es_rel \
     --output=scripts/slurm_outputs/translations/es_rel.out \
     --partition=gpu-ms \
     --gpus=1 \
     --mem-per-gpu=50G \
     --nodelist=dll-3gpu5 <<"EOF"
#!/bin/bash
python3 translate.py --translation True --topics "relations" --target_language "es" --translated_medical_info_message 'Basado en informes médicos.#Según los informes médicos.#De acuerdo con los informes médicos.#Con base en los informes médicos.#Fundado en informes médicos.'
EOF