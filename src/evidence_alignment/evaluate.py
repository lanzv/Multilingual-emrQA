import json
import re
import string
import sys
from collections import Counter


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


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def f1_span_score(p_start, p_end, gt_start, gt_end):
    if p_start >= p_end or gt_start >= gt_end:
        return 0.0
    intersection = min(p_end, gt_end) - max(p_start, gt_start) if max(p_start, gt_start) < min(p_end, gt_end) else 0.0
    precision = 1.0 * intersection / (p_end-p_start)
    recall = 1.0 * intersection / (gt_end-gt_start)
    if precision + recall == 0.0:
        return 0.0
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def evaluate_evidence_alignment(prediction_evidence, prediction_start, ground_truth_evidence, ground_truth_start):
    exact_match = 100.0 * exact_match_score(prediction_evidence, ground_truth_evidence)
    f1 = 100.0 * f1_score(prediction_evidence, ground_truth_evidence)
    f1_span = 100.0 * f1_span_score(prediction_start, prediction_start + len(prediction_evidence), ground_truth_start, ground_truth_start + len(ground_truth_evidence))
    return {'exact_match': exact_match, 'f1': f1, 'f1_span': f1_span}