
from transformers import T5ForConditionalGeneration, T5Tokenizer
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)


def load_translator_model(model_path="../models/madlad400-3b-mt"):
    model = T5ForConditionalGeneration.from_pretrained(model_path, device_map="auto")
    logging.info("Model is loaded!")
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    logging.info("Tokenizer is loaded!")
    return model, tokenizer
