
from transformers import T5ForConditionalGeneration, T5Tokenizer
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)
import re
from nltk.tokenize import word_tokenize
import stanza
from spacy_stanza import StanzaLanguage


LANGUAGE2CODE = {
    "romanian": "ro",
    "bulgarian": "bg"
}


def load_translator_model(model_path="../models/madlad400-3b-mt"):
    model = T5ForConditionalGeneration.from_pretrained(model_path, device_map="auto")
    logging.info("Model is loaded!")
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    logging.info("Tokenizer is loaded!")
    return model, tokenizer



def split_text(text, warnings=True):
    middle = len(text) // 2

    # Find all occurrences of the pattern 'abc. Def'
    pattern1 = r'[a-z]{2}\.\s+[A-Z][a-z]{1}'
    matches1 = list(re.finditer(pattern1, text))
    
    if matches1:
        # Find the occurrence closest to the middle
        closest_match = min(matches1, key=lambda x: abs(x.start() - middle))
        index = closest_match.start() + 4  # +5 to move past 'ab. '
    else:
        # If pattern1 is not found, search for the pattern 'a, b'
        pattern2 = r'[a-z],\s+[a-z]'
        matches2 = list(re.finditer(pattern2, text))
        
        if matches2:
            # Find the occurrence closest to the middle
            closest_match = min(matches2, key=lambda x: abs(x.start() - middle))
            index = closest_match.start() + 3  # +3 to move past 'a, '
            if warnings:
                logging.warning("Splitting by commas. Text '{}' can't be splitted by sentence separator.".format(text))
        else:
            # If neither pattern is found, split by the nearest space
            spaces = [m.start() for m in re.finditer(r'\s', text)]
            nearest_space = min(spaces, key=lambda x: abs(x - middle))
            index = nearest_space
            if warnings:        
                logging.warning("Splitting by spaces. Text '{}' can't be splitted by sentence separator.".format(text))


    # Split the text
    text1 = text[:index].strip()
    text2 = text[index:].strip()
    
    return text1, text2



def tokenize(paragraph, language="english", warnings = True):
    paragraph = paragraph.lower()
    if language in {"romanian", "bulgarian"}:
        snlp = stanza.Pipeline(lang=LANGUAGE2CODE[language], processors='tokenize', logging_level="ERROR")
        nlp = StanzaLanguage(snlp)
        doc = nlp(paragraph)
        tokens = []
        for token in doc:
            tokens.append(token.text)
    else:
        tokens = word_tokenize(paragraph, language=language)
    spans = []
    offset = 0
    for token in tokens:
        start = paragraph.find(token, offset)
        end = start + len(token)
        offset = end
        spans.append((start, end))

    correct_tokens = []
    correct_spans = []
    for (start, end), token in zip(spans, tokens):
        if token == paragraph[start:end]:
            correct_tokens.append(token)
            correct_spans.append((start, end))
        else:
            if warnings:
                logging.warning("Token '{}' not found in paragraph '{}' tokenized to '{}'".format(token, paragraph, tokens))
    return correct_tokens, correct_spans