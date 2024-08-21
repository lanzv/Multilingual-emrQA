#!/bin/sh
#SBATCH -J align_emrQA
#SBATCH -o scripts/slurm_outputs/align_emrQA.out
#SBATCH -p gpu-troja


emrqa_align() {
    local fa_dir=$1
    local parallel_emrqa=$2
    
    python3 <<CODE

fa_dir = "$fa_dir"
parallel_emrqa = "$parallel_emrqa"
fast_align_directory="../models/fast_align/build"


import subprocess
from transformers import AutoModel, AutoTokenizer
import itertools
import torch
import logging
import os
from src.utils import tokenize

command = './force_align.py "{}/fwd_params" "{}/fwd_err" "{}/rev_params" "{}/rev_err" "grow-diag-final-and" < "./{}/{}"'.format(fa_dir, fa_dir, fa_dir, fa_dir, fa_dir, parallel_emrqa)
process = subprocess.run(command, cwd=fast_align_directory, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
output = process.stdout

prallel_emrqa_path = os.path.join(fast_align_directory, fa_dir, parallel_emrqa)
with open(prallel_emrqa_path, 'r') as file:
    content = file.read()

cached_alignment = ""
for sentences, alignment in zip(content.split("\n"), output.split("\n")):
     if not len(sentences) == 0:
          cached_alignment += "{} ||| {}\n".format(sentences, alignment)

prallel_emrqa_align_path = os.path.join(fast_align_directory, fa_dir, "align_{}".format(parallel_emrqa))
text_file = open(prallel_emrqa_align_path, "w")
text_file.write(cached_alignment)
text_file.close()
CODE
}

emrqa_align "bgen" "medication_bg_parallel.txt"
emrqa_align "bgen" "relations_bg_parallel.txt"

emrqa_align "csen" "medication_cs_parallel.txt"
emrqa_align "csen" "relations_cs_parallel.txt"

emrqa_align "elen" "medication_el_parallel.txt"
emrqa_align "elen" "relations_el_parallel.txt"

emrqa_align "esen" "medication_es_parallel.txt"
emrqa_align "esen" "relations_es_parallel.txt"

emrqa_align "plen" "medication_pl_parallel.txt"
emrqa_align "plen" "relations_pl_parallel.txt"

emrqa_align "roen" "medication_ro_parallel.txt"
emrqa_align "roen" "relations_ro_parallel.txt"


# No Prompt Translations
emrqa_align "bgen" "noprompt_medication_bg_parallel.txt"
emrqa_align "bgen" "noprompt_relations_bg_parallel.txt"

emrqa_align "csen" "noprompt_medication_cs_parallel.txt"
emrqa_align "csen" "noprompt_relations_cs_parallel.txt"

emrqa_align "elen" "noprompt_medication_el_parallel.txt"
emrqa_align "elen" "noprompt_relations_el_parallel.txt"

emrqa_align "esen" "noprompt_medication_es_parallel.txt"
emrqa_align "esen" "noprompt_relations_es_parallel.txt"

emrqa_align "plen" "noprompt_medication_pl_parallel.txt"
emrqa_align "plen" "noprompt_relations_pl_parallel.txt"

emrqa_align "roen" "noprompt_medication_ro_parallel.txt"
emrqa_align "roen" "noprompt_relations_ro_parallel.txt"

