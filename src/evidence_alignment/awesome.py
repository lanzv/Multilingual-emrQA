from transformers import AutoModel, AutoTokenizer
import itertools
import torch
import logging
from src.utils import tokenize


class AwesomeWrapper:
    def __init__(self, model_path="../models/awesome-align-with-co"):
        # load model
        self.model = AutoModel.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def align_evidence(self, original_paragraph, translated_paragraph, original_evidence, original_start, align_layer=8, threshold=1e-3, src_language="english", tgt_language="czech"):
        # find alignments
        src_offset, tgt_offset = 0, 0
        align_words = set()
        for src_text, tgt_text in zip(original_paragraph, translated_paragraph):
            temp_align_words, src_offset, tgt_offset = self.__find_alignment(src_text, tgt_text, src_offset, tgt_offset, align_layer=align_layer, threshold=threshold, src_language=src_language, tgt_language=tgt_language)
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
            if original_start + len(original_evidence) <= r:
                end_src_id = token_id


        start_tgt_id = len(par_tgt)-1
        end_tgt_id = 0
        alignment_exists = False
        for i, j in align_words:
            if i >= start_src_id and i < end_src_id:
                alignment_exists = True
                if j < start_tgt_id:
                    start_tgt_id = j
                if j > end_tgt_id:
                    end_tgt_id = j
        if not alignment_exists:
            logging.warning("returning empty paragraph as evidence since there is no alignment for the evidence '{}' in the paragraph '{}' of translated '{}'".format(original_evidence, original_paragraph, translated_paragraph))
        #    logging.warning("returning '' evidence with -1 span start, there is no alignment for original evidence '{}' of original paragraph '{}' and translated paragraph '{}'".format(original_evidence, original_paragraph, translated_paragraph))
        #    return "", 0
        
        new_evidence = translated_paragraph[spans_tgt[start_tgt_id][0]:spans_tgt[end_tgt_id][1]]
        new_start = spans_tgt[start_tgt_id][0]
        assert new_evidence == translated_paragraph[new_start:new_start + len(new_evidence)]
        return new_evidence, new_start

    
    def __find_alignment(self, original_paragraph, translated_paragraph, src_offset, tgt_offset, align_layer=8, threshold=1e-3, src_language="english", tgt_language="czech"):
        # pre-processing
        par_src, par_tgt = tokenize(original_paragraph, language=src_language)[0], tokenize(translated_paragraph, language=tgt_language)[0]
        token_src, token_tgt = [self.tokenizer.tokenize(word) for word in par_src], [self.tokenizer.tokenize(word) for word in par_tgt]
        wid_src, wid_tgt = [self.tokenizer.convert_tokens_to_ids(x) for x in token_src], [self.tokenizer.convert_tokens_to_ids(x) for x in token_tgt]
        ids_src, ids_tgt = self.tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt', model_max_length=self.tokenizer.model_max_length, truncation=True)['input_ids'], self.tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt', truncation=True, model_max_length=self.tokenizer.model_max_length)['input_ids']
        sub2word_map_src = []
        for i, word_list in enumerate(token_src):
            sub2word_map_src += [i for x in word_list]
        sub2word_map_tgt = []
        for i, word_list in enumerate(token_tgt):
            sub2word_map_tgt += [i for x in word_list]
          
        # alignment
        self.model.eval()
        with torch.no_grad():
            out_src = self.model(ids_src.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]
            out_tgt = self.model(ids_tgt.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]
          
            dot_prod = torch.matmul(out_src, out_tgt.transpose(-1, -2))
          
            softmax_srctgt = torch.nn.Softmax(dim=-1)(dot_prod)
            softmax_tgtsrc = torch.nn.Softmax(dim=-2)(dot_prod)
          
            softmax_inter = (softmax_srctgt > threshold)*(softmax_tgtsrc > threshold)
          
        align_subwords = torch.nonzero(softmax_inter, as_tuple=False)
        align_words = set()
        for i, j in align_subwords:
            align_words.add( (sub2word_map_src[i] + src_offset, sub2word_map_tgt[j] + tgt_offset) )
        
        return align_words, src_offset + len(par_src), tgt_offset + len(par_tgt)
