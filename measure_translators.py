from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os
import numpy as np
import sacrebleu
import fastwer
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import nltk
import logging
import argparse
import random
import time
import json
import torch
import requests
import pandas as pd
logging.getLogger().setLevel(logging.INFO)
nltk.download('punkt')
nltk.download('wordnet')

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='../datasets/khresmoi-summary-test-set-2.0')
parser.add_argument('--models_dir', type=str, default='../models')
parser.add_argument('--model', type=str, default='NLLB_600M')
parser.add_argument('--seed', type=int, help='random seed', default=55)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument("-f", type=str, default="asdasd", help="jupyter file something")

LANGUAGES = ["cs", "de", "fr", "hu", "pl", "es", "sv"]
MODEL_LIST = ["NLLB_600M", "NLLB_1_3B_dis", "NLLB_1_3B", "MadLad_3B", "NLLB_3_3B", "LINDAT", "MadLad_7B", "MadLad_10B", "NLLB_54B"]
NLLB_CODES = {
    "cs": "ces_Latn", # Czech
    "de": "deu_Latn", # German
    "fr": "fra_Latn", # French
    "hu": "hun_Latn", # Hungarian
    "pl": "pol_Latn", # Polish
    "es": "spa_Latn", # Spanish
    "sv": "swe_Latn" # Swedish
}

MODELS= {
    "LINDAT": lambda _: Lindat(None), 
    "MadLad_3B": lambda model_path: MadLadWrapper(os.path.join(model_path, "madlad400-3b-mt")),
    "MadLad_7B": lambda model_path: MadLadWrapper(os.path.join(model_path, "madlad400-7b-mt"), True),
    "MadLad_10B": lambda model_path: MadLadWrapper(os.path.join(model_path, "madlad400-10b-mt"), True),
    "NLLB_600M": lambda model_path: NLLBWrapper(os.path.join(model_path, "nllb-200-distilled-600M")),
    "NLLB_1_3B_dis": lambda model_path: NLLBWrapper(os.path.join(model_path, "nllb-200-distilled-1.3B")),
    "NLLB_1_3B": lambda model_path: NLLBWrapper(os.path.join(model_path, "nllb-200-1.3B")),
    "NLLB_3_3B": lambda model_path: NLLBWrapper(os.path.join(model_path, "nllb-200-3.3B")),
    "NLLB_54B": lambda model_path: NLLBWrapper(os.path.join(model_path, "nllb-moe-54b"), True)
    # ToDo MadLad 8B, MadLad 7B BT
}

class Lindat:
    LINDAT_LANGUAGES = {"cs", "fr", "de", "hi", "pl", "ru"} # + ukrainian
    def __init__(self, model_path):
        pass

    def translate(self, text, language: str = "cs"):
        if language in Lindat.LINDAT_LANGUAGES:
            url = 'http://lindat.mff.cuni.cz/services/translation/api/v2/models/en-{}'.format(language)
            response = requests.post(url, data = {"input_text": text})
            response.encoding='utf8'
            translated_text = response.text
            return translated_text
        else:
            return ""

    def translate_batch(self, text_list, language: str = "cs"):
        translated_list = [self.translate(sen, language) for sen in text_list]
        return translated_list

        
class MadLadWrapper:
    def __init__(self, model_path: str, efficient_load = False):
        if efficient_load:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path,
                                                           load_in_4bit=True,
                                                           torch_dtype=torch.float16,
                                                           device_map="auto")
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        logging.info("{} model and tokenizer loaded".format(model_path))

    def translate(self, text, language: str = "cs"):
        inputs = self.tokenizer("<2{}> {}".format(language, text), return_tensors="pt").input_ids.to(self.model.device)
        outputs = self.model.generate(input_ids=inputs, max_new_tokens=int(1.5*inputs.shape[1]))
        translation = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return translation

    def translate_batch(self, text_list, language: str = "cs"):
        max_length = np.max([len(text) for text in text_list])
        batch = ["<2{}> {}".format(language, text) for text in text_list]
        inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length = max_length).input_ids.to(self.model.device)
        outputs = self.model.generate(input_ids=inputs, max_new_tokens=int(1.5*max_length))
        translation = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return translation



class NLLBWrapper:
    def __init__(self, model_path: str, efficient_load = False):
        if efficient_load:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path,
                                                           load_in_4bit=True,
                                                           torch_dtype=torch.float16,
                                                           device_map="auto")
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        logging.info("{} model and tokenizer loaded".format(model_path))
        
    def translate(self, text, language: str = "cs"):
        inputs = self.tokenizer(text, return_tensors="pt").input_ids.to(self.model.device)
        outputs = self.model.generate(input_ids=inputs, forced_bos_token_id=self.tokenizer.lang_code_to_id[NLLB_CODES[language]], max_new_tokens=int(1.5*inputs.shape[1]))
        translation = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return translation

    def translate_batch(self, text_list, language: str = "cs"):
        max_length = np.max([len(text) for text in text_list])
        inputs = self.tokenizer(text_list, return_tensors="pt", padding=True, truncation=True).input_ids.to(self.model.device)
        outputs = self.model.generate(input_ids=inputs, forced_bos_token_id=self.tokenizer.lang_code_to_id[NLLB_CODES[language]], max_new_tokens=int(1.5*max_length))
        translation = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return translation


def print_md_tables(results_json="./scripts/slurm_outputs/translator_results.json"):
    with open(results_json, 'r') as f:
        results = json.load(f)
        
    average_table = {}
    for model in results:
        if not model == "LINDAT":
            average_table[model] = [round(results[model]["all"]["loading_time"], 2)]
            average_table[model].append(round(results[model]["time"], 2))
            average_table[model].append(round(results[model]["bleu"], 2))
            average_table[model].append(round(results[model]["meteor"], 4))
            average_table[model].append(round(results[model]["wer"], 2))
            average_table[model].append(round(results[model]["cer"], 2))
        else:
            time = []
            bleu = []
            meteor = []
            wer = []
            cer = []
            for lang in ["cs", "de", "fr", "pl"]:
                time.append(results[model]["all"][lang]["time"])
                bleu.append(results[model]["all"][lang]["score"]["bleu"])
                meteor.append(results[model]["all"][lang]["score"]["meteor"])
                wer.append(results[model]["all"][lang]["score"]["wer"])
                cer.append(results[model]["all"][lang]["score"]["cer"])
                
            average_table[model] = [round(results[model]["all"]["loading_time"], 2)]
            average_table[model].append(round(np.mean(time), 2))
            average_table[model].append(round(np.mean(bleu), 2))
            average_table[model].append(round(np.mean(meteor), 2))
            average_table[model].append(round(np.mean(wer), 2))
            average_table[model].append(round(np.mean(cer), 2))
            
    ids = ['loading time (s)', 'translation time (s)', 'bleu', 'meteor', 'wer', 'cer']
    df = pd.DataFrame(
        data=average_table, 
        index = ids
    )    
    print(df.to_markdown())
        

    for lang in ["cs", "de", "fr", "hu", "pl", "es", "sv"]:
        table = {}
        ids = ['translation time (s)', 'bleu', 'meteor', 'wer', 'cer']

        for model in results:
            stats = results[model]["all"][lang]
            time = stats["time"]
            scores = stats["score"]
            table[model] = []
            table[model].append(round(time, 2))
            table[model].append(round(scores["bleu"], 2))
            table[model].append(round(scores["meteor"], 4))
            table[model].append(round(scores["wer"], 2))
            table[model].append(round(scores["cer"], 2))
        df = pd.DataFrame(
            data=table, 
            index = ids
        )    
        print(df.to_markdown())

        
def load_dataset(dataset_dir: str, language: str = "cs"):
    dev_file = open(os.path.join(dataset_dir, "khresmoi-summary-dev.{}".format(language)), encoding="utf-8")
    test_file = open(os.path.join(dataset_dir, "khresmoi-summary-test.{}".format(language)), encoding="utf-8")
    dev_lines = dev_file.read().splitlines()
    test_lines = test_file.read().splitlines()
    return test_lines + dev_lines


def eval(hypotheses_sentences, references_sentences):
    bleu = sacrebleu.corpus_bleu(hypotheses_sentences, [references_sentences]).score
    meteor = np.mean([meteor_score([word_tokenize(ref)], word_tokenize(hyp)) for hyp, ref in zip(hypotheses_sentences, references_sentences)])
    wer_hyp = [' '.join(word_tokenize(sen)) for sen in hypotheses_sentences] # ToDo? Due to dots mapped to the last word of the sentence
    wer_ref = [' '.join(word_tokenize(sen)) for sen in references_sentences] # ToDo? Same as above
    wer = fastwer.score(wer_hyp, wer_ref)
    cer = fastwer.score(wer_hyp, wer_ref, char_level=True)
    return bleu, meteor, wer, cer
    
def main(args):
    batch_size = args.batch_size
    source_sentences = load_dataset(args.data_dir, "en")
    statistics = {}
    #for model_code in MODEL_LIST:
    model_code = args.model
    model_statistics = {}
    loading_time = time.time()
    model = MODELS[model_code](args.models_dir)
    model_statistics["loading_time"] = time.time() - loading_time
    logging.info("The loading time of model {} is {} s".format(model_code, model_statistics["loading_time"]))
    for language in LANGUAGES:
        model_statistics[language] = {}
        
        # Translate
        translation_time = time.time()
        translations = []
        for i in range(int(len(source_sentences)/batch_size)):
            if ((i+1)*batch_size) != len(source_sentences):
                translations += model.translate_batch(source_sentences[i*batch_size:(i+1)*batch_size], language)
            else:
                translations += model.translate_batch(source_sentences[i*batch_size:], language)
        model_statistics[language]["time"] = time.time() - translation_time
        logging.info("The {} translation using {} is successfully completed in time {} s ({} sentences)".format(language, model_code, model_statistics[language]["time"], len(translations)))
        
        # Eval
        references_sentences = load_dataset(args.data_dir, language)
        bleu, meteor, wer, cer = eval(translations, references_sentences)
        logging.info("BLEU: {}, METEOR: {}, WER: {}, CER: {}".format(bleu, meteor, wer, cer))

        
        # Save statistics
        model_statistics[language]["score"] = {}
        model_statistics[language]["score"]["bleu"] = bleu
        model_statistics[language]["score"]["meteor"] = meteor
        model_statistics[language]["score"]["wer"] = wer
        model_statistics[language]["score"]["cer"] = cer
    
        statistics[model_code] = {}
        statistics[model_code]["all"] = model_statistics
        statistics[model_code]["time"] = np.mean([model_statistics[language]["time"] for language in model_statistics if language != "loading_time"])
        statistics[model_code]["bleu"] = np.mean([model_statistics[language]["score"]["bleu"] for language in model_statistics if language != "loading_time"])
        statistics[model_code]["meteor"] = np.mean([model_statistics[language]["score"]["meteor"] for language in model_statistics if language != "loading_time"])
        statistics[model_code]["wer"] = np.mean([model_statistics[language]["score"]["wer"] for language in model_statistics if language != "loading_time"])
        statistics[model_code]["cer"] = np.mean([model_statistics[language]["score"]["cer"] for language in model_statistics if language != "loading_time"])
        logging.info("Model {} is successfully measured - TIME: {}, BLEU: {}, METEOR: {}, WER: {}, CER: {}".format(model_code,
                                             statistics[model_code]["time"], statistics[model_code]["bleu"], statistics[model_code]["meteor"],
                                             statistics[model_code]["wer"], statistics[model_code]["cer"]))

    json_stats = json.dumps(statistics, indent = 4)
    print()
    print(json_stats)
    print()
        


if __name__ == '__main__':
    args = parser.parse_args()
    random.seed(args.seed)
    main(args)
