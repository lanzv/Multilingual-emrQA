import argparse
import logging
import json
import random
import os
from src.paragraphizer import Paragraphizer
from datasets.utils.logging import disable_progress_bar
from src.translate import translate_dataset
from src.utils import load_translator_model
disable_progress_bar()
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='./data/data.json')
parser.add_argument('--output_dir', type=str, default='./data/translations')
parser.add_argument('--target_language', type=str, default='cs')
parser.add_argument('--translation_model_path', type=str, default='../models/madlad400-3b-mt')
parser.add_argument('--translation', type=bool, default=False)
parser.add_argument('--evidence_alignment', type=bool, default=False)
parser.add_argument('--topics', metavar='N', type=str, nargs='+', default=["medication", "relations"])
parser.add_argument('--seed', type=int, help='random seed', default=55)



def main(args):
    random.seed(args.seed)
    with open(args.data_path, 'r') as f:
        dataset = json.load(f)

    if args.translation:
        # load model + tokenizer
        model, tokenizer = load_translator_model(args.translation_model_path)

        for title in args.topics:
            curr_data = None
            # find the given sub dataset
            for data in dataset["data"]:
                if data["title"] == title:
                    curr_data = data

            # filter and preprocess data
            curr_data = Paragraphizer.preprocess(curr_data)
            pars, topics = Paragraphizer.paragraphize(data = curr_data, title=title, frequency_threshold = 0)

            # translate
            output_path = os.path.join(args.output_dir, "{}_{}.json".format(title, args.target_language))
            translated, time_analysis = translate_dataset(model, tokenizer, pars, target_language = args.target_language, translation_mode=True, evidence_detection_mode=False, output_file=output_path)

            # log results
            question_times = np.array(time_analysis["questions"])
            context_times = np.array(time_analysis["contexts"])
            answers_times = np.array(time_analysis["answers"])
            logging.info("Time Analysis")
            if len(question_times) > 0:
                logging.info("\tAverage Paragraph's Question Time: {} s\n\t\tThe longest paragraph qas: {} s\n\t\tThe shortest paragraph qas: {} s".format(np.average(question_times), np.max(question_times), np.min(question_times)))
            if len(answers_times) > 0:
                logging.info("\tAverage Answers Time: {} s\n\t\tThe longest answers: {} s\n\t\tThe shortest answers {} s".format(np.average(answers_times), np.max(answers_times), np.min(answers_times))) 
            if len(context_times) > 0:
                logging.info("\tAverage Report's Paragraphs Time: {} s\n\t\tThe longest report: {} s\n\t\tThe shortest report {} s".format(np.average(context_times), np.max(context_times), np.min(context_times)))
            logging.info("\tTotal {} time: {} s".format(target_file_name, (time.time() - dataset_time)))


    if args.evidence_alignment:
        # TODO
        pass        

if __name__ == '__main__':
    args = parser.parse_args()
    random.seed(args.seed)
    main(args)