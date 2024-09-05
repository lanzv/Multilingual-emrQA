import uuid
import random
from datasets import Dataset
import json
from dataclasses import dataclass, field
from textwrap import dedent
from types import SimpleNamespace
from typing import Optional
import logging
import yaml
from datasets import DatasetDict, load_dataset


def emrqa2qa_dataset(dataset, seed=54):
    data = {
        'id': [],
        'title': [],
        'context': [],
        'question': [],
        'answers': []
    }
    random.seed(seed)
    for report in dataset["data"]:
        for paragraph_id, paragraph in enumerate(report['paragraphs']):
            for qa in paragraph['qas']:
                if len(qa["answers"]) > 0:
                    data['id'].append("{}_{}".format(qa['id'], paragraph_id))
                    data['title'].append(report['title'])
                    data['context'].append(paragraph['context'])
                    data['question'].append(qa['question'])
                    texts = [ans['text'] for ans in qa['answers']]
                    starts = [ans['answer_start'] for ans in qa['answers']]
                    data['answers'].append({'text': texts, 'answer_start': starts})

    dataset = Dataset.from_dict(data)
    dataset = dataset.shuffle(seed=seed)
    return dataset


def get_dataset_bert_format(train, dev, test, seed):
    train_dataset = emrqa2qa_dataset(train, seed)
    dev_dataset = emrqa2qa_dataset(dev, seed)
    test_dataset = emrqa2qa_dataset(test, seed)
    return train_dataset, dev_dataset, test_dataset






def paragraphs2reports(pars):
    report_data = {"data":[]}
    for reportid, report in enumerate(pars["data"]):
        report_context = ' '.join([par["context"] for par in report["paragraphs"]])
        new_report = {"context": report_context}
        offset = 0
        new_qas = []
        new_answers = {}
        new_questions = {}
        for paragraph in report["paragraphs"]:
            par_start = report_context.find(paragraph["context"], offset)
            par_end = par_start + len(paragraph["context"])
            offset = par_end
            assert report_context[par_start:par_end] == paragraph["context"]
            for qa in paragraph["qas"]:
                if len(qa["answers"]) != 0:
                    if not qa["id"] in new_answers:
                        new_answers[qa["id"]] = []
                    new_spans_answers = []
                    for ans in qa["answers"]:
                        if "scores" in ans:
                            new_spans_answers.append({"text": ans["text"], "answer_start": ans["answer_start"] + par_start, "scores": ans["scores"]})
                        else:
                            new_spans_answers.append({"text": ans["text"], "answer_start": ans["answer_start"] + par_start})
                        if not new_spans_answers[-1]["text"] == report_context[new_spans_answers[-1]["answer_start"]:new_spans_answers[-1]["answer_start"] + len(new_spans_answers[-1]["text"])]:
                            logging.info(new_spans_answers[-1]["text"])
                            logging.info(report_context[new_spans_answers[-1]["answer_start"]:new_spans_answers[-1]["answer_start"] + len(new_spans_answers[-1]["text"])])
                        assert new_spans_answers[-1]["text"] == report_context[new_spans_answers[-1]["answer_start"]:new_spans_answers[-1]["answer_start"] + len(new_spans_answers[-1]["text"])]
                    new_answers[qa["id"]] += new_spans_answers
                    if not qa["id"] in new_questions or len(new_questions[qa["id"]]) < len(qa["question"]):
                        new_questions[qa["id"]] = qa["question"]
        for qa_id in new_answers:
            new_qas.append({"id": qa_id, "question": new_questions[qa_id], "answers": new_answers[qa_id]})
        new_report["qas"] = new_qas
        report_data["data"].append({"paragraphs": [new_report], "title": report["title"]})
    return report_data



def filter_dataset(dataset, filters = {"f1_span": 80.0}):
    """
    """
    curr_data = {"data": []}
    kept = 0
    removed = 0
    kept_qas = 0
    removed_qas = 0
    for report in dataset["data"]:
        new_report = {"title": report["title"], "paragraphs": []}
        for paragraph in report["paragraphs"]:
            new_paragraph = {"qas": [], "context": paragraph["context"]}
            for qa in paragraph["qas"]:
                new_qa = {"question": qa["question"], "id": qa["id"], "answers": []}
                for ans in qa["answers"]:
                    if "scores" in ans:
                        to_remove = False
                        for metric in filters:
                            if ans["scores"][metric] < filters[metric]:
                                to_remove = True
                        if to_remove: 
                            removed += 1
                        else:
                            new_qa["answers"].append({"text": ans["text"], "answer_start": ans["answer_start"]})
                            kept += 1
                    else:
                        new_qa["answers"].append({"text": ans["text"], "answer_start": ans["answer_start"]})
                        kept += 1
                if len(new_qa["answers"]) > 0:
                    kept_qas += 1
                    new_paragraph["qas"].append(new_qa)
                else:
                    removed_qas += 1
            new_report["paragraphs"].append(new_paragraph)
        curr_data["data"].append(new_report)

    logging.info("Using the {} filtration settings: {} answers kept, {} answers removed ({} % kept)".format(filters, kept, removed, 100.0*kept/(kept+removed)))
    logging.info("Using the {} filtration settings: {} qas kept, {} qas removed ({} % kept)".format(filters, kept_qas, removed_qas, 100.0*kept_qas/(kept_qas+removed_qas)))
    return curr_data, kept_qas, removed_qas, kept, removed 