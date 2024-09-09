import json
import os
import pandas
import logging
import argparse
import re
import string
import sys
import pandas as pd
from collections import Counter
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)


directory_path = "predictions/monolingual/"
all_answers = {}
gold_answers = {}
languages = ["BG", "CS", "EL", "EN", "ES", "PL", "RO"]
seeds = set()


def return_correct_ids(dataset, predictions):
    correct_ids = set()
    for qa_id in dataset:
        if qa_id not in predictions:
            message = 'Unanswered question ' + qa_id + \
                        ' will receive score 0.'
            print(message, file=sys.stderr)
            continue
        ground_truths = dataset[qa_id]["text"]
        prediction = predictions[qa_id]
        exact_match = metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        if exact_match == 1.0:
            correct_ids.add('_'.join(qa_id.split('_')[1:]))
    return correct_ids

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))



def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)



def f1_score_set(A, B):
    # Calculate Precision and Recall
    intersection = len(A.intersection(B))
    if len(B) == 0:
        precision = 0
    else:
        precision = intersection / len(B)
    
    if len(A) == 0:
        recall = 0
    else:
        recall = intersection / len(A)
    
    # Calculate F1 Score
    if precision + recall == 0:
        return 0  # Handle division by zero
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1


logging.info("collecting data...")

# Loop through each JSON file in the directory
for filename in os.listdir(directory_path):
    if filename.endswith(".json"):
        file_path = os.path.join(directory_path, filename)
        model = filename.split("_")[0]
        lang = filename.split("_")[1]
        if lang == "EN" and filename.split("_")[2] == "FULL":
            continue
        subset = filename.split("_")[2]
        seed = filename.split("_")[3][:-5]
        seeds.add(seed)
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        if model == "gold":
            if not subset in gold_answers:
                gold_answers[subset] = {}
            if not lang in gold_answers[subset]:
                gold_answers[subset][lang] = {}
            gold_answers[subset][lang][seed] = data
        else:
            if not model in all_answers:
                all_answers[model] = {}
            if not subset in all_answers[model]:
                all_answers[model][subset] = {}
            if not lang in all_answers[model][subset]:
                all_answers[model][subset][lang] = {}
            all_answers[model][subset][lang][seed] = data
logging.info("collecting correct answers...")
# collect correct answers
correct_answers = {}
for subset in gold_answers:
    for lang in gold_answers[subset]:
        for seed in gold_answers[subset][lang]:
            for model in all_answers:
                if subset in all_answers[model] and lang in all_answers[model][subset] and seed in all_answers[model][subset][lang]:
                    if not model in correct_answers:
                        correct_answers[model] = {}
                    if not subset in correct_answers[model]:
                        correct_answers[model][subset] = {}
                    if not lang in correct_answers[model][subset]:
                        correct_answers[model][subset][lang] = {}
                    correct_answers[model][subset][lang][seed] = return_correct_ids(
                        gold_answers[subset][lang][seed], 
                        all_answers[model][subset][lang][seed])
del gold_answers
del all_answers
logging.info("computing f1 scores...")
# compute f1 similarities
medication_matrix = pd.DataFrame([[[] for _ in range(len(languages))] for _ in range(len(languages))])
medication_matrix.index = languages
medication_matrix.columns = languages
relations_matrix = pd.DataFrame([[[] for _ in range(len(languages))] for _ in range(len(languages))])
relations_matrix.index = languages
relations_matrix.columns = languages

for model in correct_answers:
    for subset in correct_answers[model]:
        for lang1 in languages:
            for lang2 in languages:
                for seed in seeds:
                    if lang1 in correct_answers[model][subset] and seed in correct_answers[model][subset][lang1]:
                        if lang2 in correct_answers[model][subset] and seed in correct_answers[model][subset][lang2]:
                            if subset == "medication":
                                medication_matrix.at[lang1, lang2].append(f1_score_set(
                                    correct_answers[model][subset][lang1][seed],
                                    correct_answers[model][subset][lang2][seed],
                                ))
                            elif subset == "relations": 
                                relations_matrix.at[lang1, lang2].append(f1_score_set(
                                    correct_answers[model][subset][lang1][seed],
                                    correct_answers[model][subset][lang2][seed],
                                ))


def compute_average(matrix):
    for row in matrix.index:
        for col in matrix.columns:
            cell = matrix.at[row, col]
            if len(cell) > 0:  # Check if the list is non-empty
                matrix.at[row, col] = sum(cell) / len(cell)  # Replace list with average
            else:
                matrix.at[row, col] = float('nan')  # Handle empty lists with NaN

# Compute the averages
compute_average(medication_matrix)
compute_average(relations_matrix)
print(medication_matrix)
print(relations_matrix)
