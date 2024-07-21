import argparse
import logging
from src.prqa.eval import Evaluate
from src.prqa.models.bert import BERTWrapperPRQA
from src.prqa.models.dataset import emrqa2prqa_dataset, emrqa2qa_dataset, get_dataset_bert_format
import json
import random
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
parser.add_argument('--dataset', type=str, default='../datasets/emrQA/medication_bg.json')
# model
parser.add_argument('--model_name', type=str, default='ClinicalBERT')
parser.add_argument('--model_path', type=str, default='../models/Bio_ClinicalBERT')
# paragraphizing
parser.add_argument('--train_sample_ratio', type=float, default=0.2)
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--to_reports', type=bool, default=False)

# random
parser.add_argument('--seed', type=int, help='random seed', default=2)






MODELS = {
    "BERTbase": lambda model_path: BERTWrapperPRQA(model_path),
    "mBERT": lambda model_path: BERTWrapperPRQA(model_path),
    "ClinicalBERT": lambda model_path: BERTWrapperPRQA(model_path),
}


PREPARE_DATASET = {
    "BERTbase": lambda train_pars, dev_pars, test_pars: get_dataset_bert_format(train_pars, dev_pars, test_pars),
    "mBERT": lambda train_pars, dev_pars, test_pars: get_dataset_bert_format(train_pars, dev_pars, test_pars),
    "ClinicalBERT": lambda train_pars, dev_pars, test_pars: get_dataset_bert_format(train_pars, dev_pars, test_pars),
}


def sample_dataset(data, sample_ratio):
    """
    Authors: CliniRC
    """
    new_data = []
    total = 0
    sample = 0
    for paras in data['data']:
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
    new_data_json = {'data': new_data, 'version': data['version']}
    return new_data_json


def split_dataset(dataset, train_ratio=0.7, dev_ratio=0.1, f1_span_threshold=80.0, train_sample_ratio=0.2):
    curr_data = {"data": []}
    kept = 0
    removed = 0
    for report in dataset["data"]:
        new_report = {"title": report["title"], "paragraphs": []}
        for paragraph in report["paragraphs"]:
            new_paragraph = {"qas": [], "context": paragraph["context"]}
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

    assert train_ratio + dev_ratio <= 1.0
    note_num = len(curr_data["data"])
    train_note_num, dev_note_num = int(train_ratio * note_num), int(dev_ratio * note_num)
    note_list = random.sample(range(note_num), note_num)

    train = {'data': [], 'version': 1.0}
    for i in range(train_note_num):
        train['data'].append(curr_data['data'][note_list[i]])
    dev = {'data': [], 'version': 1.0}
    for i in range(train_note_num, train_note_num + dev_note_num):
        dev['data'].append(curr_data['data'][note_list[i]])
    test = {'data': [], 'version': 1.0}
    for i in range(train_note_num + dev_note_num, note_num):
        test['data'].append(curr_data['data'][note_list[i]])
    # sample dataset
    if train_sample_ratio < 1.0:
        new_train = sample_dataset(train, train_sample_ratio/(float(kept)/(float(kept)+float(removed))))
    merged_answers_test = {'data': []}
    for report in test['data']:
        merged_qas = {}
        for paragraph in report["paragraphs"]:
            for qa in paragraph["qas"]:
                if len(qa["answers"]) != 0:
                    if not qa["id"] in merged_qas:
                        merged_qas[qa["id"]] = []
                    merged_qas[qa["id"]] += qa["answers"]
        new_qas = []
        for qa_id in merged_qas:
            new_qas.append({"id": qa_id, "answers": merged_qas[qa_id]})
        merged_answers_test['data'].append({"paragraphs": [{"qas": new_qas}]})

    return new_train, dev, test, merged_answers_test


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

def main(args):
    if not args.model_name in MODELS:
        logging.error("The model {} is not supported".format(args.model_name))
        return 
    
    # load splited data
    with open(args.dataset, 'r') as f:
        dataset = json.load(f)

    if args.to_reports:
        dataset = paragraphs2reports(dataset)

    train_pars, dev_pars, test_pars, merged_answers_test = split_dataset(dataset, train_sample_ratio=args.train_sample_ratio)

    scores = {}
    logging.info("------------- Experiment: model {}---------------".format(args.model_name))
    # prepare data

    train_dataset, dev_dataset, test_prqa_dataset = PREPARE_DATASET[args.model_name](train_pars, dev_pars, test_pars)
    logging.info("datasets are converted to Datset format")
    logging.info("{}|{}".format(len(train_dataset), len(dev_dataset)))

    # train model
    model = MODELS[args.model_name](args.model_path)
    model.train(train_dataset, dev_dataset, epochs = args.epochs, disable_tqdm=True)
    qa_predictions, pr_predictions, prqa_predictions = model.predict(test_prqa_dataset, disable_tqdm=True)
    #with open("org_qa_predictions.json", 'w', encoding='utf8') as f:
    #    json.dump(qa_predictions, f, ensure_ascii=False)
    #with open("org_pr_predictions.json", 'w', encoding='utf8') as f:
    #    json.dump(pr_predictions, f, ensure_ascii=False)
    #with open("org_prqa_predictions.json", 'w', encoding='utf8') as f:
    #    json.dump(prqa_predictions, f, ensure_ascii=False)

    # evaluate
    qa_scores = Evaluate.question_answering(merged_answers_test, qa_predictions)
    logging.info("QA scores: {}".format(qa_scores))
    pr_scores = Evaluate.paragraph_retrieval(test_pars, pr_predictions) # eval PR predictions on the Paragraphized dataset
    logging.info("PR scores: {}".format(pr_scores))
    prqa_scores = Evaluate.question_answering(merged_answers_test, prqa_predictions)
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