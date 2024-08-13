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

def align_with_fastalign(PARALLEL_CORPUS, src, tgt, original_evidence, original_start, target_language):
    from src.utils import tokenize
    for par_src, par_tgt in zip(src, tgt):
        src_tokens = tokenize(par_src, language="english", warnings = False)[0]
        tgt_tokens = tokenize(par_tgt, language="czech", warnings = False)[0]
        PARALLEL_CORPUS += "{} ||| {}\n".format(' '.join(tgt_tokens), ' '.join(src_tokens))
    return PARALLEL_CORPUS
    

def split_paragraph(text, text_length_limit=750):
    if len(text) > text_length_limit:
        text1, text2 = split_text(text, False)
        text1 = split_paragraph(text1)
        text2 = split_paragraph(text2)
        return text1 + text2
    return [text]



args_dataset = './data/data.json'
args_dataset_title = 'medication'
args_translation_dataset = './data/translations/medication_cs.json'
args_language = "cs"

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
        evidence_cache = {}
        for qa in tgt_paragraph["qas"]:
            new_answers = []
            for ans in qa["answers"]:
                    PARALLEL_CORPUS = align_with_fastalign(PARALLEL_CORPUS, split_paragraph(src_paragraph["context"]), tgt_paragraph["context"], ans["text"], ans["answer_start"], args_language)

text_file = open("{}_parallel.txt".format(args_dataset_title), "w")
text_file.write(PARALLEL_CORPUS)
text_file.close()
