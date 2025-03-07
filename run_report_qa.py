import argparse
import logging
from src.qa.eval import Evaluate
from src.qa.models.bert import BERTWrapperPRQA
from src.qa.models.dataset import emrqa2qa_dataset, get_dataset_bert_format, paragraphs2reports, filter_dataset
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
parser.add_argument('--answers_remove_ratio', type=float, default=0.0)
parser.add_argument('--train_sample_ratio', type=float, default=0.2)
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--to_reports', type=bool, default=False)
parser.add_argument('--paragraph_parts', type=bool, default=False)

# random
parser.add_argument('--seed', type=int, help='random seed', default=2)






MODELS = {
    "BERTbase": lambda model_path: BERTWrapperPRQA(model_path),
    "mBERT": lambda model_path: BERTWrapperPRQA(model_path),
    "ClinicalBERT": lambda model_path: BERTWrapperPRQA(model_path),
    "XLMRLarge": lambda model_path: BERTWrapperPRQA(model_path),
    "mDistil": lambda model_path: BERTWrapperPRQA(model_path),
}


PREPARE_DATASET = {
    "BERTbase": lambda train_pars, dev_pars, test_pars, seed: get_dataset_bert_format(train_pars, dev_pars, test_pars, seed),
    "mBERT": lambda train_pars, dev_pars, test_pars, seed: get_dataset_bert_format(train_pars, dev_pars, test_pars, seed),
    "ClinicalBERT": lambda train_pars, dev_pars, test_pars, seed: get_dataset_bert_format(train_pars, dev_pars, test_pars, seed),
    "XLMRLarge": lambda train_pars, dev_pars, test_pars, seed: get_dataset_bert_format(train_pars, dev_pars, test_pars, seed),
    "mDistil": lambda train_pars, dev_pars, test_pars, seed: get_dataset_bert_format(train_pars, dev_pars, test_pars, seed),
}


def compute_f1_span_threshold(dataset, answers_remove_ratio):
    f1_span_values = []
    for report in dataset["data"]:
        for paragraph in report["paragraphs"]:
            for qa in paragraph["qas"]:
                for ans in qa["answers"]:
                    f1_span_values.append(ans["scores"]["f1_span"])
    sorted_f1_span_values = sorted(f1_span_values)
    index = int(answers_remove_ratio * (len(sorted_f1_span_values)))
    return sorted_f1_span_values[index]
    

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


def split_dataset(dataset, answers_remove_ratio, train_ratio=0.7, dev_ratio=0.1, train_sample_ratio=0.2, seed=54):
    random.seed(seed)
    curr_data = dataset
    assert train_ratio + dev_ratio <= 1.0
    note_num = len(curr_data["data"])
    train_note_num, dev_note_num = int(train_ratio * note_num), int(dev_ratio * note_num)
    note_list = random.sample(range(note_num), note_num)
    logging.info("report id list (first 10 ids): {}".format(note_list[:10]))

    train = {'data': [], 'version': 1.0}
    for i in range(train_note_num):
        train['data'].append(curr_data['data'][note_list[i]])
    dev = {'data': [], 'version': 1.0}
    for i in range(train_note_num, train_note_num + dev_note_num):
        dev['data'].append(curr_data['data'][note_list[i]])
    test = {'data': [], 'version': 1.0}
    for i in range(train_note_num + dev_note_num, note_num):
        test['data'].append(curr_data['data'][note_list[i]])

    # filter dataset by score values
    DATSET_FILTERS = {
        "f1_span": compute_f1_span_threshold(dataset, answers_remove_ratio)
    }
    train, kept, removed, kept_answers, removed_answers = filter_dataset(dataset = train, filters = DATSET_FILTERS)
    train["version"] = 1.0
    # sample dataset
    if train_sample_ratio < 1.0:
        new_train = sample_dataset(train, train_sample_ratio/(float(kept)/(float(kept)+float(removed))))

    return new_train, dev, test


def main(args):
    if not args.model_name in MODELS:
        logging.error("The model {} is not supported".format(args.model_name))
        return 
    logging.info("------------- Experiment: model {}---------------".format(args.model_name))

    # PREPARE DATASET
    # load data
    with open(args.dataset, 'r') as f:
        dataset = json.load(f)
    if args.paragraph_parts:
        for report in dataset["data"]:
            for paragraph in report["paragraphs"]:
                paragraph["context"] = ' '.join(paragraph["context"])
    # merge paragraphs to single report if set
    if args.to_reports:
        dataset = paragraphs2reports(dataset)
    # split dataset
    train_pars, dev_pars, test_pars = split_dataset(dataset, answers_remove_ratio=args.answers_remove_ratio, train_sample_ratio=args.train_sample_ratio, seed=args.seed)
    # prepare data
    train_dataset, dev_dataset, test_dataset = PREPARE_DATASET[args.model_name](train_pars, dev_pars, test_pars, args.seed)
    logging.info("datasets are converted to Datset format")
    logging.info("train/dev/test qa pairs: {}|{}|{}".format(len(train_dataset), len(dev_dataset), len(test_dataset)))
    

    # TRAIN AND EVALUATE MODEL
    # train model and predict
    model = MODELS[args.model_name](args.model_path)
    model.train(train_dataset, dev_dataset, epochs = args.epochs, disable_tqdm=True, seed=args.seed)
    qa_predictions = model.predict(test_dataset, disable_tqdm=True)
    # evaluate
    qa_scores = Evaluate.question_answering(test_dataset, qa_predictions)
    logging.info("QA scores: {}".format(qa_scores))

    scores = {}
    scores[args.model_name] = {}
    scores[args.model_name]["QA"] = qa_scores

    scores = json.dumps(scores, indent = 4) 
    print(scores)
    scores = {}
    scores[args.model_name] = {}
    scores[args.model_name][args.dataset] = {}
    scores[args.model_name][args.dataset][args.seed] = {}
    scores[args.model_name][args.dataset][args.seed][args.answers_remove_ratio] = {}
    scores[args.model_name][args.dataset][args.seed][args.answers_remove_ratio][args.to_reports] = {}
    scores[args.model_name][args.dataset][args.seed][args.answers_remove_ratio][args.to_reports]["QA"] = qa_scores

    filename = "jsons/paragraphs_thresholds/{}_{}_{}_{}_{}.json".format(args.model_name, args.dataset.replace("/", "_").replace(".", "_"), args.seed, args.answers_remove_ratio, args.to_reports)
    with open(filename, 'w') as json_file:
        json.dump(scores, json_file, indent=4)



if __name__ == '__main__':
    args = parser.parse_args()
    random.seed(args.seed)
    main(args)