import random
import logging
import os, json
import numpy as np
import time
from src.utils import load_translator_model, split_text
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)



def translate_questions(model, tokenizer, questions_to_translate, target_language: str = "cs", max_token_coef=1.5, batch_size = 20): # ToDo max_token_coef?? 1.5??
    batches = []
    for i in range(0, len(questions_to_translate), batch_size):
        if i + batch_size <= len(questions_to_translate):
            batches.append(questions_to_translate[i:i+batch_size])
    if len(questions_to_translate) % batch_size != 0:
        batches.append(questions_to_translate[-(len(questions_to_translate) % batch_size):])
    result_translations = []
    for batch in batches:
        max_length = np.max([len(text) for text in batch])
        batch = ["<2{}> {}".format(target_language, text) for text in batch]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length = max_length).input_ids.to(model.device)
        outputs = model.generate(input_ids=inputs, max_new_tokens=int(max_token_coef*max_length))
        translation = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        result_translations += translation
    assert len(result_translations) == len(questions_to_translate)
    return result_translations

def paragraph_translation_done(translation, text, translated_medical_info_message):
    done = False
    for message in translated_medical_info_message.split("#"):
        if not (len(translation[:-len(message)].strip()) < 0.8 * len(text) or not translation[-len(message):] == message):
            done = True
        if len(text) <= 40 and not (len(translation[:-len(message)].strip()) < 0.5 * len(text) or not translation[-len(message):] == message):
            done = True
    #if done == False and translation != "":
    #    logging.info("{}\n\n".format(translation))
    return done

def translate_paragraph(model, tokenizer, text, target_language = "cs", no_repeat_ngram_size=12, translated_medical_info_message="Na základě lékařských zpráv.", text_length_limit=750, disable_prompting=False):
    if len(text) > text_length_limit:
        text1, text2 = split_text(text)
        trans1 = translate_paragraph(model, tokenizer, text1, target_language, no_repeat_ngram_size, translated_medical_info_message, text_length_limit, disable_prompting)
        trans2 = translate_paragraph(model, tokenizer, text2, target_language, no_repeat_ngram_size, translated_medical_info_message, text_length_limit, disable_prompting)
        return trans1 + trans2
    
    if disable_prompting:
        inputs = tokenizer("<2{}> {}".format(target_language, text), return_tensors="pt").input_ids.to(model.device)
        outputs = model.generate(input_ids=inputs, max_new_tokens=int(1.5*inputs.shape[1]), no_repeat_ngram_size=no_repeat_ngram_size)
        translation = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return [translation]

    translation = ""
    medical_info_message = "Based on medical reports."
    spaces = "   "
    while len(spaces) <= 25 and not paragraph_translation_done(translation, text, translated_medical_info_message):
        inputs = tokenizer("<2{}> {}".format(target_language, text + spaces + medical_info_message), return_tensors="pt").input_ids.to(model.device)
        #inputs = tokenizer("<2{}> {}".format(target_language, text), return_tensors="pt").input_ids.to(model.device)
        outputs = model.generate(input_ids=inputs, max_new_tokens=int(1.5*inputs.shape[1]), no_repeat_ngram_size=no_repeat_ngram_size)
        translation = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        #return [translation]
        spaces += "     "

    spaces = "   "
    while not paragraph_translation_done(translation, text, translated_medical_info_message):
        inputs = tokenizer("<2{}> {}".format(target_language, text + spaces + medical_info_message), return_tensors="pt").input_ids.to(model.device)
        outputs = model.generate(input_ids=inputs, do_sample=True, top_k=15, top_p=0.97, max_new_tokens=int(1.5*inputs.shape[1]))
        translation = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        spaces += "     "
        if len(spaces) >= 200:
            logging.warning("The medical info message '{}' cannot be found in \n'{}'\n of the original \n'{}'\n or the length of the translated is too small. Returning translation without prompting.".format(translated_medical_info_message, translation, text))
            return translate_paragraph(model, tokenizer, text, target_language, no_repeat_ngram_size, translated_medical_info_message, text_length_limit, disable_prompting=True)

    for message in translated_medical_info_message.split("#"):
        if message == translation[-len(message):]:
            translation = translation[:-len(message)].strip()
            return [translation]

    raise Exception("There is no '{}' message in translation '{}'".format(translated_medical_info_message, translation))


def translate_dataset(model, tokenizer, dataset, target_language="cs", translation_mode = True, evidence_detection_mode = True, output_file = "./temp.json", translated_medical_info_message="Na základě lékařských zpráv.", disable_prompting=False):
    time_analysis = {"contexts": [], "questions": [], "answers": []}
    translated_dataset = {"data":[]}
    if translation_mode:
        for report_id, report in enumerate(dataset["data"]):
            # iterate over paragraphs, translate all questions
            new_report = {"paragraphs": [], "title": report["title"]}
            evidence_cache = {}
            paragraphs_to_translate = []
            for paragraph in report["paragraphs"]:
                paragraphs_to_translate.append(paragraph["context"])
                questions_to_translate = []
                new_paragraph = {"qas": []}
                for qa in paragraph["qas"]:
                    for ans in qa["answers"]:
                        ans["position_ratio"] = float(ans["answer_start"])/len(paragraph["context"])
                    new_qa = {"answers": qa["answers"], "id": qa["id"]} # Answers will be translated later
                    new_paragraph["qas"].append(new_qa)
                    questions_to_translate.append(qa["question"])
                
                # translate paragraph's questions
                question_time = time.time()
                translated_questions = translate_questions(model, tokenizer, questions_to_translate, target_language) 
                time_analysis["questions"].append(time.time() - question_time)
                for i, new_question in enumerate(translated_questions):
                    new_paragraph["qas"][i]["question"] = new_question
                new_report["paragraphs"].append(new_paragraph)

            # translate paragraphs
            context_time = time.time()
            translated_paragraphs = [translate_paragraph(model, tokenizer, par2tran, target_language, translated_medical_info_message=translated_medical_info_message, disable_prompting=disable_prompting) for par2tran in paragraphs_to_translate]
            time_analysis["contexts"].append(time.time() - context_time)
            for i, new_paragraph in enumerate(translated_paragraphs):
                new_report["paragraphs"][i]["context"] = new_paragraph
            logging.info("{}. report (pars/qs) translated".format(report_id))
            translated_dataset["data"].append(new_report)

            # store results
            with open(output_file, 'w', encoding='utf8') as f:
                json.dump(translated_dataset, f, ensure_ascii=False)
        dataset = translated_dataset
        
    if evidence_detection_mode:
        for report in dataset["data"]:
            evidence_cache = {}
            # translate evidences
            evidence_time = time.time()
            for paragraph in report["paragraphs"]:
                for qa in paragraph["qas"]:
                    new_answers = []
                    for ans in qa["answers"]:
                        original_position_ratio = ans["position_ratio"]
                        if ans["answer_start"] in evidence_cache and ans["text"] in evidence_cache[ans["answer_start"]]:
                            translated_text, translated_answer_start, scores = evidence_cache[ans["answer_start"]][ans["text"]]
                        else:
                            translated_text, translated_answer_start, scores = translate_evidence(model, tokenizer, ans["text"], paragraph["context"], target_language, original_position_ratio)
                            if not ans["answer_start"] in evidence_cache:
                                evidence_cache[ans["answer_start"]] = {}
                            evidence_cache[ans["answer_start"]][ans["text"]] = (translated_text, translated_answer_start, scores)
                        new_answers.append({"text": translated_text, "answer_start": translated_answer_start, "scores": scores})
                    qa["answers"] = new_answers
            time_analysis["answers"].append(time.time() - evidence_time)
            logging.info("evidences translated")
            # store results
            with open(output_file, 'w', encoding='utf8') as f:
                json.dump(dataset, f, ensure_ascii=False)

    return dataset, time_analysis
