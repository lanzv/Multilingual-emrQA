import argparse
import logging
from src.prqa.eval import Evaluate
from src.prqa.models.bert import BERTWrapperPRQA
from src.prqa.models.dataset import emrqa2qa_dataset, get_dataset_bert_format, paragraphs2reports, filter_dataset
import json
from datasets import concatenate_datasets
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
parser.add_argument('--subset', type=str, default='medication')
parser.add_argument('--language', type=str, default='BG')
# model
parser.add_argument('--model_name', type=str, default='ClinicalBERT')
parser.add_argument('--model_path', type=str, default='../models/Bio_ClinicalBERT')
# paragraphizing
parser.add_argument('--train_sample_ratio', type=float, default=0.2)
parser.add_argument('--train_ratio', type=float, default=0.7)
parser.add_argument('--dev_ratio', type=float, default=0.1)
parser.add_argument('--epochs', type=int, default=3)

parser.add_argument('--multilingual_train', type=bool, default=False)

# random
parser.add_argument('--seed', type=int, help='random seed', default=2)


DATASETS = {
    "medication": {
        "BG": ("./data/translation_aligners/Awesome/medication_bg.json", True, 0.15),
        "CS": ("./data/translation_aligners/Awesome/medication_cs.json", True, 0.15),
        "EL": ("./data/translation_aligners/FastAlign/medication_el.json", True, 0.15),
        "EN": ("../datasets/emrQA/medication_en.json", False, 1.0),
        "EN_FULL": ("../datasets/emrQA/medication_en.json", False, 1.0),
        "ES": ("./data/translation_aligners/Awesome/medication_es.json", True, 0.15),
        "PL": ("./data/translation_aligners/Awesome/medication_pl.json", True, 0.15),
        "RO": ("./data/translation_aligners/Awesome/medication_ro.json", True, 0.15)
    },
    "relations": {
        "BG": ("./data/translation_aligners/Awesome/relations_bg.json", True, 0.15),
        "CS": ("./data/translation_aligners/Awesome/relations_cs.json", True, 0.15),
        "EL": ("./data/translation_aligners/FastAlign/relations_el.json", True, 0.15),
        "EN": ("../datasets/emrQA/relations_en.json", False, 1.0),
        "EN_FULL": ("../datasets/emrQA/relations_en.json", False, 1.0),
        "ES": ("./data/translation_aligners/Awesome/relations_es.json", True, 0.15),
        "PL": ("./data/translation_aligners/FastAlign/relations_pl.json", True, 0.15),
        "RO": ("./data/translation_aligners/Awesome/relations_ro.json", True, 0.15)
    },
}
"""
DATASETS = {
    "medication": {
        "BG": ("./data/translation_aligners/Awesome/medication_bg.json", True, 0.0),
        "CS": ("./data/translation_aligners/Awesome/medication_cs.json", True, 0.0),
        "EL": ("./data/translation_aligners/FastAlign/medication_el.json", True, 0.0),
        "EN": ("../datasets/emrQA/medication_en.json", False, 1.0),
        "EN_FULL": ("../datasets/emrQA/medication_en.json", False, 1.0),
        "ES": ("./data/translation_aligners/Awesome/medication_es.json", True, 0.0),
        "PL": ("./data/translation_aligners/Awesome/medication_pl.json", True, 0.0),
        "RO": ("./data/translation_aligners/Awesome/medication_ro.json", True, 0.0)
    },
    "relations": {
        "BG": ("./data/translation_aligners/Awesome/relations_bg.json", True, 0.0),
        "CS": ("./data/translation_aligners/Awesome/relations_cs.json", True, 0.0),
        "EL": ("./data/translation_aligners/FastAlign/relations_el.json", True, 0.0),
        "EN": ("../datasets/emrQA/relations_en.json", False, 1.0),
        "EN_FULL": ("../datasets/emrQA/relations_en.json", False, 1.0),
        "ES": ("./data/translation_aligners/Awesome/relations_es.json", True, 0.0),
        "PL": ("./data/translation_aligners/FastAlign/relations_pl.json", True, 0.0),
        "RO": ("./data/translation_aligners/Awesome/relations_ro.json", True, 0.0)
    },
}
"""

MODELS = {
    "BERTbase": lambda model_path: BERTWrapperPRQA(model_path),
    "mBERT": lambda model_path: BERTWrapperPRQA(model_path),
    "ClinicalBERT": lambda model_path: BERTWrapperPRQA(model_path),
    "XLMRLarge": lambda model_path: BERTWrapperPRQA(model_path),
    "XLMR": lambda model_path: BERTWrapperPRQA(model_path),
    "mDistil": lambda model_path: BERTWrapperPRQA(model_path),
}


PREPARE_DATASET = {
    "BERTbase": lambda train_pars, dev_pars, test_pars, seed: get_dataset_bert_format(train_pars, dev_pars, test_pars, seed),
    "mBERT": lambda train_pars, dev_pars, test_pars, seed: get_dataset_bert_format(train_pars, dev_pars, test_pars, seed),
    "ClinicalBERT": lambda train_pars, dev_pars, test_pars, seed: get_dataset_bert_format(train_pars, dev_pars, test_pars, seed),
    "XLMRLarge": lambda train_pars, dev_pars, test_pars, seed: get_dataset_bert_format(train_pars, dev_pars, test_pars, seed),
    "XLMR": lambda train_pars, dev_pars, test_pars, seed: get_dataset_bert_format(train_pars, dev_pars, test_pars, seed),
    "mDistil": lambda train_pars, dev_pars, test_pars, seed: get_dataset_bert_format(train_pars, dev_pars, test_pars, seed),
}



def compute_f1_span_threshold(dataset, answers_remove_ratio):
    f1_span_values = []
    for report in dataset["data"]:
        for paragraph in report["paragraphs"]:
            for qa in paragraph["qas"]:
                for ans in qa["answers"]:
                    if not "scores" in ans:
                        return 0.0
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


def split_dataset(dataset, kept, removed, note_list, train_note_num, dev_note_num, note_num, train_sample_ratio=0.2):


    train = {'data': [], 'version': 1.0}
    for i in range(train_note_num):
        train['data'].append(dataset['data'][note_list[i]])
    dev = {'data': [], 'version': 1.0}
    for i in range(train_note_num, train_note_num + dev_note_num):
        dev['data'].append(dataset['data'][note_list[i]])
    test = {'data': [], 'version': 1.0}
    for i in range(train_note_num + dev_note_num, note_num):
        test['data'].append(dataset['data'][note_list[i]])
    # sample dataset
    if train_sample_ratio < 1.0:
        new_train = sample_dataset(train, train_sample_ratio/(float(kept)/(float(kept)+float(removed))))

    return new_train, dev, test

def get_set_answer_ids(subset):
    ids = set()
    for rep_id, report in enumerate(subset["data"]):
        for par_id, paragraph in enumerate(report["paragraphs"]):
            for qa_id, qa in enumerate(paragraph["qas"]):
                # rename qa id to deterministic string same for all languages
                qa["id"] = "{}_{}_{}".format(rep_id, par_id, qa_id)
                for ans_id, ans in enumerate(qa["answers"]):
                    ids.add("{}_{}_{}_{}".format(rep_id, par_id, qa_id, ans_id))
    return ids

def filter_set_by_ids(subset, id_set):
    for rep_id, report in enumerate(subset["data"]):
        for par_id, paragraph in enumerate(report["paragraphs"]):
            new_qas = []
            for qa_id, qa in enumerate(paragraph["qas"]):
                new_answers = []
                for ans_id, ans in enumerate(qa["answers"]):
                    answer_id = "{}_{}_{}_{}".format(rep_id, par_id, qa_id, ans_id)
                    if answer_id in id_set:
                        new_answers.append(ans)
                if len(new_answers) > 0:
                    qa["answers"] = new_answers
                    new_qas.append(qa)
            paragraph["qas"] = new_qas
    return subset

def extend_ids_by_language(subset, language):
    for report in subset["data"]:
        for paragraph in report["paragraphs"]:
            for qa in paragraph["qas"]:
                qa["id"] = language + "_" + qa["id"]
    return subset

def main(args):
    if not args.model_name in MODELS:
        logging.error("The model {} is not supported".format(args.model_name))
        return 
    logging.info("------------- Experiment: model {}---------------".format(args.model_name))

    # PREPARE DATASET
    split_not_prepared = True
    # load data
    trains = {}
    devs = {}
    tests = {}
    test_answer_ids_all = {}
    kept_answers, removed_answers = 0, 0
    for language in DATASETS[args.subset]:
        path, paragraph_parts, answers_remove_ratio = DATASETS[args.subset][language]
        with open(path, 'r') as f:
            dataset = json.load(f)
        if split_not_prepared:
            # prepare random train/dev/test ids split
            random.seed(args.seed)
            curr_data = dataset
            note_num = len(curr_data["data"])
            train_note_num, dev_note_num = int(args.train_ratio * note_num), int(args.dev_ratio * note_num)
            note_list = random.sample(range(note_num), note_num)
            logging.info("report id list (first 10 ids): {}".format(note_list[:10]))
            split_not_prepared = False
        if paragraph_parts:
            for report in dataset["data"]:
                for paragraph in report["paragraphs"]:
                    paragraph["context"] = ' '.join(paragraph["context"])
        # filter dataset by score values
        DATSET_FILTERS = {
            "f1_span": compute_f1_span_threshold(dataset, answers_remove_ratio)
        }
        dataset, kept, removed, kept_answers, removed_answers = filter_dataset(dataset = dataset, filters = DATSET_FILTERS)
        # split dataset
        if not language == "EN_FULL": 
            trains[language], devs[language], tests[language] = split_dataset(dataset, kept=kept, removed=removed, note_list=note_list, train_note_num=train_note_num, dev_note_num=dev_note_num, note_num=note_num, train_sample_ratio=args.train_sample_ratio)
        else:
            _, _, tests[language] = split_dataset(dataset, kept=kept, removed=removed, note_list=note_list, train_note_num=train_note_num, dev_note_num=dev_note_num, note_num=note_num, train_sample_ratio=args.train_sample_ratio)
        test_answer_ids_all[language] = get_set_answer_ids(tests[language])
        logging.info(len(test_answer_ids_all[language]))
        
    
    # get test answer ids intersection
    intersection_test_answer_ids = set.intersection(*test_answer_ids_all.values())
    logging.info("Final intersection contains {} % of original answers ({}/{})".format(100.0*round(len(intersection_test_answer_ids)/(len(test_answer_ids_all["EN"])), 2), len(intersection_test_answer_ids), len(test_answer_ids_all["EN"])))

    # filter tests by intersection answer ids
    for language in tests:
        if not language == "EN_FULL":
            tests[language] = filter_set_by_ids(tests[language], intersection_test_answer_ids)
    
    # extend all ids by language code
    for language in trains:
        extend_ids_by_language(trains[language], language)
    for language in devs:
        extend_ids_by_language(devs[language], language)
    for language in tests:
        extend_ids_by_language(tests[language], language)



    # prepare data
    if args.multilingual_train:
        train_datasets = []
        dev_datasets = []
        test_datasets = {}
        for language in trains:
            train_dataset, dev_dataset, test_dataset = PREPARE_DATASET[args.model_name](trains[language], devs[language], tests[language], args.seed)
            train_datasets.append(train_dataset)
            dev_datasets.append(dev_dataset)
            test_datasets[language] = test_dataset
        _, _, test_dataset = PREPARE_DATASET[args.model_name](trains["EN"], devs["EN"], tests["EN_FULL"], args.seed)
        test_datasets["EN_FULL"] = test_dataset
        train_dataset = concatenate_datasets(train_datasets)
        train_dataset = train_dataset.shuffle(seed=args.seed)
        dev_dataset = concatenate_datasets(dev_datasets)
        dev_dataset = dev_dataset.shuffle(seed=args.seed)
        dev_dataset = dev_dataset.train_test_split(test_size=0.14)["test"]
    else:
        if args.language == "EN_FULL": 
            train_dataset, dev_dataset, test_dataset = PREPARE_DATASET[args.model_name](trains["EN"], devs["EN"], tests["EN_FULL"], args.seed)
        else:
            train_dataset, dev_dataset, test_dataset = PREPARE_DATASET[args.model_name](trains[args.language], devs[args.language], tests[args.language], args.seed)
    logging.info("datasets are converted to Datset format")
    logging.info("train/dev/test qa pairs: {}|{}|{}".format(len(train_dataset), len(dev_dataset), len(test_dataset)))


    # TRAIN AND EVALUATE MODEL
    # train model and predict
    model = MODELS[args.model_name](args.model_path)
    model.train(train_dataset, dev_dataset, epochs = args.epochs, disable_tqdm=True, seed=args.seed)
    if not args.multilingual_train:
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
        scores[args.model_name][args.language] = {}
        scores[args.model_name][args.language][args.subset] = {}
        scores[args.model_name][args.language][args.subset][args.seed] = {}
        scores[args.model_name][args.language][args.subset][args.seed]["QA"] = qa_scores

        filename = "jsons/monolingual/{}_{}_{}_{}.json".format(args.model_name, args.language, args.subset, args.seed)
        with open(filename, 'w') as json_file:
            json.dump(scores, json_file, indent=4)
        filename = "predictions/monolingual/{}_{}_{}_{}.json".format(args.model_name, args.language, args.subset, args.seed)
        with open(filename, 'w') as json_file:
            json.dump(qa_predictions, json_file, indent=4)
        gold = {}
        for data_sample in test_dataset:
            gold[data_sample["id"]] = data_sample["answers"]
        filename = "predictions/monolingual/gold_{}_{}_{}.json".format(args.language, args.subset, args.seed)
        with open(filename, 'w') as json_file:
            json.dump(gold, json_file, indent=4)
    else:
        for language in test_datasets:
            qa_predictions = model.predict(test_datasets[language], disable_tqdm=True)
            # evaluate
            qa_scores = Evaluate.question_answering(test_datasets[language], qa_predictions)
            logging.info("QA scores: {}".format(qa_scores))

            scores = {}
            scores[args.model_name] = {}
            scores[args.model_name]["QA"] = qa_scores

            scores = json.dumps(scores, indent = 4) 
            print(scores)
            scores = {}
            scores[args.model_name] = {}
            scores[args.model_name][language] = {}
            scores[args.model_name][language][args.subset] = {}
            scores[args.model_name][language][args.subset][args.seed] = {}
            scores[args.model_name][language][args.subset][args.seed]["QA"] = qa_scores

            filename = "jsons/multilingual/{}_{}_{}_{}.json".format(args.model_name, language, args.subset, args.seed)
            with open(filename, 'w') as json_file:
                json.dump(scores, json_file, indent=4)

            filename = "predictions/multilingual/{}_{}_{}_{}.json".format(args.model_name, language, args.subset, args.seed)
            with open(filename, 'w') as json_file:
                json.dump(qa_predictions, json_file, indent=4)


            gold = {}
            for data_sample in test_datasets[language]:
                gold[data_sample["id"]] = data_sample["answers"]
            filename = "predictions/multilingual/gold_{}_{}_{}.json".format(language, args.subset, args.seed)
            with open(filename, 'w') as json_file:
                json.dump(gold, json_file, indent=4)



if __name__ == '__main__':
    args = parser.parse_args()
    random.seed(args.seed)
    main(args)