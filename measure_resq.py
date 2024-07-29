import argparse
import logging
from src.prqa.eval import Evaluate
from src.prqa.models.bert import BERTWrapperPRQA
from src.paragraphizer import Paragraphizer
from src.prqa.models.dataset import emrqa2prqa_dataset, emrqa2qa_dataset, get_dataset_bert_format
import json
import random
import os
import numpy as np
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

parser = argparse.ArgumentParser()
# dataset paths
parser.add_argument('--resq_folder', type=str, default='../datasets/resq/pl')
parser.add_argument('--medication', type=str, default='../datasets/emrQA/medication_pl.json')
parser.add_argument('--relations', type=str, default='../datasets/emrQA/relations_pl.json')
# model
parser.add_argument('--model_name', type=str, default='mBERT')
parser.add_argument('--model_path', type=str, default='../models/bert-base-multilingual-cased')
parser.add_argument('--target_average', type=float, default=254)
# paragraphizing
parser.add_argument('--medication_sample_ratio', type=float, default=0.2)
parser.add_argument('--relations_sample_ratio', type=float, default=0.05)
parser.add_argument('--epochs', type=int, default=3)

# random
parser.add_argument('--seed', type=int, help='random seed', default=2)




RESQ_DATA = [
    '../datasets/resq/bg',
    '../datasets/resq/el',
    '../datasets/resq/en',
    '../datasets/resq/pl',
    '../datasets/resq/ro'
]



MODELS = {
    "BERTbase": lambda model_path: BERTWrapperPRQA(model_path),
    "mBERT": lambda model_path: BERTWrapperPRQA(model_path),
    "ClinicalBERT": lambda model_path: BERTWrapperPRQA(model_path),
    "ontotext": lambda model_path: BERTWrapperPRQA(model_path),
}


PREPARE_DATASET = {
    "BERTbase": lambda train_pars, dev_pars, test_pars: get_dataset_bert_format(train_pars, dev_pars, test_pars),
    "mBERT": lambda train_pars, dev_pars, test_pars: get_dataset_bert_format(train_pars, dev_pars, test_pars),
    "ClinicalBERT": lambda train_pars, dev_pars, test_pars: get_dataset_bert_format(train_pars, dev_pars, test_pars),
    "ontotext": lambda train_pars, dev_pars, test_pars: get_dataset_bert_format(train_pars, dev_pars, test_pars),
}



def load_resq(folder_path):
    json_objects = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                try:
                    json_data = json.load(file)
                    json_objects.append(json_data)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from file {filename}: {e}")
    dataset = {"data": []}
    for obj in json_objects:
        assert len(obj["data"]) == 1
        dataset["data"].append(obj["data"][0])
    # fix unique ids
    for report in dataset["data"]:
        assert len(report["paragraphs"]) == 1
        for paragraph in report["paragraphs"]:
            new_qas = []
            for qa in paragraph["qas"]:
                if qa["question_id"] is None:
                    continue
                if report["hospital_id"] is not None:
                    qa["id"] = report["hospital_id"] + "_" + report["report_id"] + "_" + qa["question_id"]
                else:
                    qa["id"] = report["report_id"] + "_" + qa["question_id"]
                new_qas.append(qa)
            paragraph["qas"] = new_qas
    # rewrite complex complementary answers to alternative ones
    for report in dataset["data"]:
        assert len(report["paragraphs"]) == 1
        for paragraph in report["paragraphs"]:
            for qa in paragraph["qas"]:
                new_answers = []
                for ans in qa["answers"]:
                    if ans["answer_type"] == "complex":
                        for text, span in zip(ans["text"], ans["answer_start"]):
                            if not span == -1:
                                new_answers.append({"text": text, "answer_start": span})
                    elif ans["answer_type"] == "single" and ans["answer_start"] == -1:
                        pass
                    else:
                        new_answers.append(ans)
                qa["answers"] = new_answers
    # filter empty context
    filtered_dataset = {"data": []}
    for report in dataset["data"]:
        assert len(report["paragraphs"]) == 1
        for paragraph in report["paragraphs"]:
            if paragraph["context"] is not None:
                filtered_dataset["data"].append({
                    "title": report["report_id"],
                    "hospital_id": report["hospital_id"],
                    "report_id": report["report_id"],
                    "annotator_id": report["annotator_id"], 
                    "report_language": report["report_language"], 
                    "form_version": report["form_version"],
                    "form_language": report["form_version"],
                    "paragraphs": [paragraph]})
    dataset = filtered_dataset
    # filter empty questions
    for report in dataset["data"]:
        assert len(report["paragraphs"]) == 1
        for paragraph in report["paragraphs"]:
            new_qas = []
            for qa in paragraph["qas"]:
                if len(qa["answers"]) > 0:
                    new_qas.append(qa)
                paragraph["qas"] = new_qas
    return dataset


def sample_questions(dataset, ratio, f1_span_threshold=80.0):
    curr_data = {"data": []}
    kept = 0
    removed = 0
    for report in dataset["data"]:
        new_report = {"title": report["title"], "paragraphs": []}
        for paragraph in report["paragraphs"]:
            new_paragraph = {"qas": [], "context": " ".join(paragraph["context"])}
            for qa in paragraph["qas"]:
                new_qa = {"question": qa["question"], "id": qa["id"], "answers": []}
                for ans in qa["answers"]:
                    if "scores" not in ans or ans["scores"]["f1_span"] >= f1_span_threshold:
                        new_qa["answers"].append({"text": ans["text"], "answer_start": ans["answer_start"]})
                        kept += 1
                    else:
                        removed += 1
                if len(new_qa["answers"]) > 0:
                    new_paragraph["qas"].append(new_qa)
            new_report["paragraphs"].append(new_paragraph)
        curr_data["data"].append(new_report)

    logging.info("Using the {} % as f1 span threshold.. {} qas kept, {} qas removed ({} % kept)".format(f1_span_threshold, kept, removed, 100.0*kept/(kept+removed)))

    # adapt ratio to the number of kept and removed examples
    if ratio < 1.0:
        sample_ratio = ratio/(float(kept)/(float(kept)+float(removed)))
    else:
        sample_ratio = 1.0

    new_data = []
    total = 0
    sample = 0
    for paras in curr_data['data']:
        new_paragraphs = []
        for para in paras['paragraphs']:
            new_qas = []
            context = para['context']
            qa_num = len(para['qas'])
            total += qa_num
            sample_num = int(qa_num * sample_ratio)
            sampled_list = [i for i in range(qa_num)]
            sampled_list = random.choices(sampled_list, k=sample_num)
            for qa_id in sampled_list:
                qa = para['qas'][qa_id]
                sample += 1
                new_qas.append(qa)
            new_para = {'context': context, 'qas': new_qas}
            new_paragraphs.append(new_para)
        new_data.append({'title': paras['title'], 'paragraphs': new_paragraphs})
    new_data_json = {'data': new_data}
    return new_data_json


def split_dataset(dataset, train_ratio):
    assert train_ratio <= 1.0
    note_num = len(dataset["data"])
    train_note_num = int(train_ratio * note_num)
    note_list = random.sample(range(note_num), note_num)

    train = {'data': [], 'version': 1.0}
    for i in range(train_note_num):
        train['data'].append(dataset['data'][note_list[i]])
    dev = {'data': [], 'version': 1.0}
    for i in range(train_note_num, note_num):
        dev['data'].append(dataset['data'][note_list[i]])

    return train, dev

def merge_datasets(medication, relations, current_path, train_ratio = 0.8):
    med_train, med_dev = split_dataset(medication, train_ratio)
    rel_train, rel_dev = split_dataset(relations, train_ratio)
    train = {"data": []}
    dev = {"data": []}
    for resq_path in RESQ_DATA:
        if current_path != resq_path:
            resq_data = load_resq(resq_path)
            #train["data"] += resq_data["data"][:-10]
            dev["data"] += resq_data["data"][-10:]
        else:
            resq_data = load_resq(resq_path)
            train["data"] += resq_data["data"][:-53]
    #train["data"] += med_train["data"] + rel_train["data"]
    #dev["data"] += med_dev["data"] + rel_dev["data"]
    return train, dev#, {"data": med_train["data"] + rel_train["data"]}, {"data": med_dev["data"] + rel_dev["data"]}
    #return {"data": med_train["data"] + rel_train["data"]}, {"data": med_dev["data"] + rel_dev["data"]}


def main(args):
    if not args.model_name in MODELS:
        logging.error("The model {} is not supported".format(args.model_name))
        return 
    
    # load splited data
    resq = {"data": load_resq(args.resq_folder)["data"][-53:]}
    logging.info(len(resq["data"]))
    lengths = []
    with open(args.medication, 'r') as f:
        medication = json.load(f)
        medication = sample_questions(medication, args.medication_sample_ratio)
        for report in medication["data"]:
            for paragraph in report["paragraphs"]:
                lengths.append(len(paragraph["context"]))

    with open(args.relations, 'r') as f:
        relations = json.load(f)
        relations = sample_questions(relations, args.relations_sample_ratio)
        for report in relations["data"]:
            for paragraph in report["paragraphs"]:
                lengths.append(len(paragraph["context"]))

    logging.info("target average segment length: {}".format(np.mean(lengths)))
    test_pars, _ = Paragraphizer.paragraphize2(data = resq, title="uniform", frequency_threshold = 0, target_average=np.mean(lengths))
    train_pars, dev_pars = merge_datasets(medication, relations, args.resq_folder)

    scores = {}
    logging.info("------------- Experiment: model {}---------------".format(args.model_name))
    # prepare data
    #emrqa_train_dataset, emrqa_dev_dataset, _ = PREPARE_DATASET[args.model_name](emrqa_train_pars, emrqa_dev_pars, test_pars)
    train_dataset, dev_dataset, test_prqa_dataset = PREPARE_DATASET[args.model_name](train_pars, dev_pars, test_pars)
    logging.info("datasets are converted to Datset format")
    logging.info("{}|{}".format(len(train_dataset), len(dev_dataset)))

    # train model
    model = MODELS[args.model_name](args.model_path)
    #model.train(emrqa_train_dataset, emrqa_dev_dataset, epochs = args.epochs, disable_tqdm=True)
    model.train(train_dataset, dev_dataset, epochs = args.epochs, disable_tqdm=True)
    qa_predictions, pr_predictions, prqa_predictions = model.predict(test_prqa_dataset, disable_tqdm=True)
    # store results
    #with open("qa_predictions_{}.json".format(args.resq_folder[-2:]), 'w', encoding='utf8') as f:
    #    json.dump(qa_predictions, f, ensure_ascii=False)
    #with open("pr_predictions_{}.json".format(args.resq_folder[-2:]), 'w', encoding='utf8') as f:
    #    json.dump(pr_predictions, f, ensure_ascii=False)
    #with open("prqa_predictions_{}.json".format(args.resq_folder[-2:]), 'w', encoding='utf8') as f:
    #    json.dump(prqa_predictions, f, ensure_ascii=False)

    # evaluate
    qa_scores = Evaluate.question_answering(resq, qa_predictions)
    logging.info("QA scores: {}".format(qa_scores))
    pr_scores = Evaluate.paragraph_retrieval(test_pars, pr_predictions) # eval PR predictions on the Paragraphized dataset
    logging.info("PR scores: {}".format(pr_scores))
    prqa_scores = Evaluate.question_answering(resq, prqa_predictions)
    logging.info("PRQA scores: {}".format(prqa_scores))

    scores[args.model_name] = {}
    scores[args.model_name]["QA"] = qa_scores
    scores[args.model_name]["PR"] = pr_scores
    scores[args.model_name]["PRQA"] = prqa_scores

    scores = json.dumps(scores, indent = 4) 
    print(scores)



if __name__ == '__main__':
    args = parser.parse_args()
    random.seed(args.seed)
    main(args)