import subprocess
from transformers import AutoModel, AutoTokenizer
import itertools
import torch
import logging
from src.utils import tokenize


class FastAlignWrapper:
    def __init__(self, model_path="../models/fast_align/build/czeng/czeng_medication.txt"):
        # load model
        with open(model_path, 'r') as file:
            alignments = file.read()
        self.fa_cache = {}
        for alignment in alignments.split("\n"):
            if not len(alignment) == 0:
                translated, original, alignment = alignment.split("|||")
            self.fa_cache[translated.strip() + " ||| " + original.strip()] = alignment.strip()


    def align_evidence(self, original_paragraph, translated_paragraph, original_evidence, original_start, reverse=True, src_language="english", tgt_language="czech", fast_align_directory="../models/fast_align/build"):
        # find alignments
        src_offset, tgt_offset = 0, 0
        align_words = set()
        for src_text, tgt_text in zip(original_paragraph, translated_paragraph):
            temp_align_words, src_offset, tgt_offset = self.__find_alignment(src_text, tgt_text, src_offset, tgt_offset, reverse=reverse, src_language=src_language, tgt_language=tgt_language, fast_align_directory=fast_align_directory)
            align_words = align_words | temp_align_words
        
        # merge texts to paragraphs
        original_paragraph = ' '.join(original_paragraph)
        translated_paragraph = ' '.join(translated_paragraph)
        par_src, spans_src = tokenize(original_paragraph, language=src_language)
        par_tgt, spans_tgt = tokenize(translated_paragraph, language=tgt_language)

        # find evidence
        start_src_id, end_src_id = None, None
        for token_id, (l, r) in enumerate(spans_src):
            if l <= original_start:
                start_src_id = token_id
            if end_src_id == None and original_start + len(original_evidence) <= r:
                end_src_id = token_id


        start_tgt_id = len(par_tgt)-1
        end_tgt_id = 0
        alignment_exists = False
        for i, j in align_words:
            if i >= start_src_id and i <= end_src_id:
                alignment_exists = True
                if j < start_tgt_id:
                    start_tgt_id = j
                if j > end_tgt_id:
                    end_tgt_id = j
        if not alignment_exists:
            logging.warning("returning empty evidence since there is no alignment for the evidence '{}' in the paragraph '{}' of translated '{}'".format(original_evidence, original_paragraph, translated_paragraph))
        #    logging.warning("returning '' evidence with -1 span start, there is no alignment for original evidence '{}' of original paragraph '{}' and translated paragraph '{}'".format(original_evidence, original_paragraph, translated_paragraph))
        #    return "", 0

        new_evidence = translated_paragraph[spans_tgt[start_tgt_id][0]:spans_tgt[end_tgt_id][1]]
        new_start = spans_tgt[start_tgt_id][0]
        assert new_evidence == translated_paragraph[new_start:new_start + len(new_evidence)]
        return new_evidence, new_start


    def __find_alignment(self, original_paragraph, translated_paragraph, src_offset, tgt_offset, reverse=True, src_language="english", tgt_language="czech", fast_align_directory="../models/fast_align/build"):
        par_src, par_tgt = tokenize(original_paragraph, language=src_language)[0], tokenize(translated_paragraph, language=tgt_language)[0]
        if reverse:
            input_sentences = "{} ||| {}".format(' '.join(par_tgt), ' '.join(par_src))
        else:
            input_sentences = "{} ||| {}".format(' '.join(par_src), ' '.join(par_tgt))

        assert input_sentences in self.fa_cache
        alignments = self.fa_cache[input_sentences].split()
        align_words = set()
        for alignment in alignments:
            i, j = alignment.split("-")
            if reverse:
                align_words.add((int(j) + src_offset, int(i) + tgt_offset))
            else:
                align_words.add((int(i) + src_offset, int(j) + tgt_offset))
                
        return align_words, src_offset + len(par_src), tgt_offset + len(par_tgt)
