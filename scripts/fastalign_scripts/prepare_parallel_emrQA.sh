#!/bin/bash
#SBATCH -J prepare_fastalign_emrQA
#SBATCH -o scripts/slurm_outputs/prepare_fastalign_emrQA.out
#SBATCH -p cpu-ms
#SBATCH --mem=50G



run_alignment() {
    local dataset=$1
    local title=$2
    local translation=$3
    local language=$4
    
    python3 <<CODE
import argparse
import logging
from src.paragraphizer import Paragraphizer
from src.evidence_alignment.awesome import AwesomeWrapper
from src.evidence_alignment.levenshtein import LevenshteinWrapper
from src.evidence_alignment.fast_align import FastAlignWrapper
from src.evidence_alignment.evaluate import evaluate_evidence_alignment
import json
import os
import time
import random
from src.utils import split_text
import logging
import numpy as np

CODE2LANGUAGE = {
    "cs": "czech",
    "pl": "polish",
    "bg": "bulgarian",
    "ro": "romanian",
    "el": "greek", 
    "es": "spanish"
}

def align_with_fastalign(PARALLEL_CORPUS, src, tgt, target_language):
    from src.utils import tokenize
    for par_src, par_tgt in zip(src, tgt):
        src_tokens = tokenize(par_src, language="english", warnings = False)[0]
        tgt_tokens = tokenize(par_tgt, language=CODE2LANGUAGE[target_language], warnings = False)[0]
        PARALLEL_CORPUS += "{} ||| {}\n".format(' '.join(tgt_tokens), ' '.join(src_tokens))
    return PARALLEL_CORPUS
    

def split_paragraph(text, text_length_limit=750):
    if len(text) > text_length_limit:
        text1, text2 = split_text(text, False)
        text1 = split_paragraph(text1)
        text2 = split_paragraph(text2)
        return text1 + text2
    return [text]

args_dataset = "$dataset"
args_dataset_title = "$title"
args_translation_dataset = "$translation"
args_language = "$language"

PARALLEL_CORPUS = ""

# load data
with open(args_dataset, 'r') as f:
    dataset = json.load(f)
curr_data = None
# find the given sub dataset
for data in dataset["data"]:
    if data["title"] == args_dataset_title:
        curr_data = data
# filter and preprocess data
dataset = Paragraphizer.preprocess(curr_data)
dataset, train_topics = Paragraphizer.paragraphize(data = dataset, title=args_dataset_title, frequency_threshold = 0)

with open(args_translation_dataset, 'r') as f:
    translated_dataset = json.load(f)

# init scores
evidence_time = time.time()
for report_id, (src_report, tgt_report) in enumerate(zip(dataset["data"], translated_dataset["data"])):
    # translate evidences
    for src_paragraph, tgt_paragraph in zip(src_report["paragraphs"], tgt_report["paragraphs"]):
        PARALLEL_CORPUS = align_with_fastalign(PARALLEL_CORPUS, split_paragraph(src_paragraph["context"]), tgt_paragraph["context"], args_language)

text_file = open("{}_{}_parallel.txt".format(args_dataset_title, args_language), "w")
text_file.write(PARALLEL_CORPUS)
text_file.close()
CODE
}

run_alignment "./data/data.json" "medication" "./data/translations/medication_bg.json" "bg"
mv medication_bg_parallel.txt ../models/fast_align/build/bgen/medication_bg_parallel.txt
run_alignment "./data/data.json" "relations" "./data/translations/relations_bg.json" "bg"
mv relations_bg_parallel.txt ../models/fast_align/build/bgen/relations_bg_parallel.txt

run_alignment "./data/data.json" "medication" "./data/translations/medication_cs.json" "cs"
mv medication_cs_parallel.txt ../models/fast_align/build/csen/medication_cs_parallel.txt
run_alignment "./data/data.json" "relations" "./data/translations/relations_cs.json" "cs"
mv relations_cs_parallel.txt ../models/fast_align/build/csen/relations_cs_parallel.txt

run_alignment "./data/data.json" "medication" "./data/translations/medication_el.json" "el"
mv medication_el_parallel.txt ../models/fast_align/build/elen/medication_el_parallel.txt
run_alignment "./data/data.json" "relations" "./data/translations/relations_el.json" "el"
mv relations_el_parallel.txt ../models/fast_align/build/elen/relations_el_parallel.txt

run_alignment "./data/data.json" "medication" "./data/translations/medication_es.json" "es"
mv medication_es_parallel.txt ../models/fast_align/build/esen/medication_es_parallel.txt
run_alignment "./data/data.json" "relations" "./data/translations/relations_es.json" "es"
mv relations_es_parallel.txt ../models/fast_align/build/esen/relations_es_parallel.txt

run_alignment "./data/data.json" "medication" "./data/translations/medication_pl.json" "pl"
mv medication_pl_parallel.txt ../models/fast_align/build/plen/medication_pl_parallel.txt
run_alignment "./data/data.json" "relations" "./data/translations/relations_pl.json" "pl"
mv relations_pl_parallel.txt ../models/fast_align/build/plen/relations_pl_parallel.txt

run_alignment "./data/data.json" "medication" "./data/translations/medication_ro.json" "ro"
mv medication_ro_parallel.txt ../models/fast_align/build/roen/medication_ro_parallel.txt
run_alignment "./data/data.json" "relations" "./data/translations/relations_ro.json" "ro"
mv relations_ro_parallel.txt ../models/fast_align/build/roen/relations_ro_parallel.txt
