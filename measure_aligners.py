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
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

parser = argparse.ArgumentParser()
# dataset paths
parser.add_argument('--translation_dataset', type=str, default='./data/translations/medication_cs.json')
parser.add_argument('--dataset', type=str, default='./data/data.json')
parser.add_argument('--output_dir', type=str, default='./data/translation_aligners')
parser.add_argument('--dataset_title', type=str, default='medication')
parser.add_argument('--language', type=str, default='cs')
# aligner
parser.add_argument('--aligner_name', type=str, default='Awesome')
parser.add_argument('--aligner_path', type=str, default='../models/awesome-align-with-co')
# random
parser.add_argument('--seed', type=int, help='random seed', default=55)





CODE2LANGUAGE = {
    "cs": "czech",
    "pl": "polish",
    "bg": "bulgarian",
    "ro": "romanian",
    "el": "greek"
}

MODELS = {
    "Awesome": lambda model_path: AwesomeWrapper(model_path),
    "Levenshtein": lambda model_path: LevenshteinWrapper(model_path),
    "FastAlign": lambda model_path: FastAlignWrapper(model_path)
}


ALIGN_EVIDENCE = {
    "Awesome": lambda aligner, src, tgt, original_evidence, original_start, target_language: align_with_awesome(aligner, src, tgt, original_evidence, original_start, target_language),
    "Levenshtein": lambda aligner, src, tgt, original_evidence, original_start, target_language: align_with_levenshtein(aligner, src, tgt, original_evidence, original_start, target_language),
    "FastAlign": lambda aligner, src, tgt, original_evidence, original_start, target_language: align_with_fastalign(aligner, src, tgt, original_evidence, original_start, target_language) 
}

def align_with_fastalign(aligner, src, tgt, original_evidence, original_start, target_language):
    #from src.utils import tokenize
    #for par_src, par_tgt in zip(src, tgt):
    #    src_tokens = tokenize(par_src, language="english", warnings = False)[0]
    #    tgt_tokens = tokenize(par_tgt, language="czech", warnings = False)[0]
    #    PARALLEL_CORPUS += "{} ||| {}\n".format(' '.join(tgt_tokens), ' '.join(src_tokens))
    #return "", -1, {'exact_match': 0.0, 'exact_submatch': 0.0, 'f1': 0.0, 'f1_span': 0.0, 'precision_span': 0.0, 'recall_span': 0.0, 'start_distance': 0.0, 'middle_distance': 0.0, 'end_distance': 0.0}, PARALLEL_CORPUS
    assert len(src) == len(tgt)
    new_evidence, new_start = aligner.align_evidence(src, tgt, original_evidence, original_start, reverse=True, src_language="english", tgt_language=CODE2LANGUAGE[target_language])
    prediction_evidence, prediction_start = aligner.align_evidence(tgt, src, new_evidence, new_start, reverse=False, src_language=CODE2LANGUAGE[target_language], tgt_language="english")
    scores = evaluate_evidence_alignment(prediction_evidence, prediction_start, original_evidence, original_start)
    return new_evidence, new_start, scores

def align_with_awesome(aligner, src, tgt, original_evidence, original_start, target_language):
    assert len(src) == len(tgt)
    new_evidence, new_start = aligner.align_evidence(src, tgt, original_evidence, original_start, src_language="english", tgt_language=CODE2LANGUAGE[target_language])
    prediction_evidence, prediction_start = aligner.align_evidence(tgt, src, new_evidence, new_start, src_language=CODE2LANGUAGE[target_language], tgt_language="english")
    scores = evaluate_evidence_alignment(prediction_evidence, prediction_start, original_evidence, original_start)
    return new_evidence, new_start, scores

def align_with_levenshtein(aligner, src, tgt, original_evidence, original_start, target_language):
    src = ' '.join(src)
    tgt = ' '.join(tgt)
    if len(tgt) == 0:
        logging.warning("There is an empty translation '{}' of the paragraph '{}'".format(tgt, src))
        return "", 0, evaluate_evidence_alignment("", 0, original_evidence, original_start)

    new_evidence, new_start = aligner.align_evidence(original_evidence, tgt, target_language=target_language)
    prediction_evidence, prediction_start = aligner.align_evidence(new_evidence, src, target_language="en")
    scores = evaluate_evidence_alignment(prediction_evidence, prediction_start, original_evidence, original_start)
    return new_evidence, new_start, scores

def split_paragraph(text, text_length_limit=750):
    if len(text) > text_length_limit:
        text1, text2 = split_text(text, False)
        text1 = split_paragraph(text1)
        text2 = split_paragraph(text2)
        return text1 + text2
    return [text]


def main(args):
    if not args.aligner_name in MODELS:
        logging.error("The model {} is not supported".format(args.aligner_name
        ))
        return 
    
    # load data
    with open(args.dataset, 'r') as f:
        dataset = json.load(f)
    curr_data = None
    # find the given sub dataset
    for data in dataset["data"]:
        if data["title"] == args.dataset_title:
            curr_data = data
    # filter and preprocess data
    dataset = Paragraphizer.preprocess(curr_data)
    dataset, train_topics = Paragraphizer.paragraphize(data = dataset, title=args.dataset_title, frequency_threshold = 0)

    with open(args.translation_dataset, 'r') as f:
        translated_dataset = json.load(f)

    # load aligner
    aligner = MODELS[args.aligner_name](args.aligner_path)

    output_path = os.path.join(args.output_dir, args.aligner_name, "{}_{}.json".format(args.dataset_title, args.language))

    # init scores
    f1s = []
    ems = []
    esms = []
    f1s_span = []
    ps_span = []
    rs_span = []
    str_dists = []
    mid_dists = []
    end_dists = []
    abs_str_dists = []
    abs_mid_dists = []
    abs_end_dists = []
    evidence_time = time.time()
    for report_id, (src_report, tgt_report) in enumerate(zip(dataset["data"], translated_dataset["data"])):        
        evidence_cache = {}
        # translate evidences
        for src_paragraph, tgt_paragraph in zip(src_report["paragraphs"], tgt_report["paragraphs"]):
            for qa in tgt_paragraph["qas"]:
                new_answers = []
                for ans in qa["answers"]:
                    if ans["answer_start"] in evidence_cache and ans["text"] in evidence_cache[ans["answer_start"]]:
                        translated_text, translated_answer_start, scores = evidence_cache[ans["answer_start"]][ans["text"]]
                    else:
                        # align evidence
                        translated_text, translated_answer_start, scores = ALIGN_EVIDENCE[args.aligner_name](aligner, split_paragraph(src_paragraph["context"]), tgt_paragraph["context"], ans["text"], ans["answer_start"], args.language)
                        # collect scores
                        f1s.append(scores["f1"])
                        ems.append(scores["exact_match"])
                        esms.append(scores["exact_submatch"])
                        f1s_span.append(scores["f1_span"])
                        ps_span.append(scores["precision_span"])
                        rs_span.append(scores["recall_span"])
                        str_dists.append(scores["start_distance"])
                        mid_dists.append(scores["middle_distance"])
                        end_dists.append(scores["end_distance"])
                        abs_str_dists.append(scores["absolute_start_distance"])
                        abs_mid_dists.append(scores["absolute_middle_distance"])
                        abs_end_dists.append(scores["absolute_end_distance"])
                        # add to cache
                        if not ans["answer_start"] in evidence_cache:
                            evidence_cache[ans["answer_start"]] = {}
                        evidence_cache[ans["answer_start"]][ans["text"]] = (translated_text, translated_answer_start, scores)
                    new_answers.append({"text": translated_text, "answer_start": translated_answer_start, "scores": scores})
                # rewrite answers to translated aligned ones
                qa["answers"] = new_answers
        logging.info("{}. evidences measured - em: {} f1 span: {}".format(report_id, np.mean(ems), np.mean(f1s_span)))

        # store results
        with open(output_path, 'w', encoding='utf8') as f:
            json.dump(translated_dataset, f, ensure_ascii=False)
    

    final_score = {
        "f1": np.mean(f1s),
        "exact_match": np.mean(ems),
        "exact_submatch": np.mean(esms),
        "f1_span": np.mean(f1s_span),
        "precision_span": np.mean(ps_span),
        "recall_span": np.mean(rs_span),
        "start_distance": np.mean(str_dists),
        "middle_distance": np.mean(mid_dists),
        "end_distance": np.mean(end_dists),
        "absolute_start_distance": np.mean(abs_str_dists),
        "absolute_middle_distance": np.mean(abs_mid_dists),
        "absolute_end_distance": np.mean(abs_end_dists),
        "overall_time": time.time() - evidence_time
    }
    scores = json.dumps(final_score, indent = 4) 
    print(scores)

    #text_file = open("relations_parallel.txt", "w")
    #text_file.write(PARALLEL_CORPUS)
    #text_file.close()




if __name__ == '__main__':
    args = parser.parse_args()
    random.seed(args.seed)
    main(args)