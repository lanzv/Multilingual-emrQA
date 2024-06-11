from transformers import AutoModel, AutoTokenizer
import itertools
import torch
import logging


class AwesomeWrapper:
    def __init__(self, model_path="../models/awesome-align-with-co"):
        # load model
        self.model = AutoModel.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def align_evidence(self, original_paragraph, translated_paragraph, original_evidence, original_start, align_layer=8, threshold=1e-3):
        # pre-processing
        par_src, par_tgt = original_paragraph.strip().split(), translated_paragraph.strip().split()
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
            align_words.add( (sub2word_map_src[i], sub2word_map_tgt[j]) )


        # find evidence
        start_src_id = len(original_paragraph[:original_start].strip().split())
        end_src_id = start_src_id + len(original_evidence.strip().split())
        if not (" ".join(par_src[start_src_id:end_src_id]) == original_evidence or " ".join(par_src[start_src_id:end_src_id])[:-1] == original_evidence):
            logging.warning("original english evidence does not correspond to the one compounded from ids\noriginal evidence: '{}'\nfound evidence: '{}'\nparagraph text: '{}'\nparagraph words: '{}'\n".format(original_paragraph, par_src, " ".join(par_src[start_src_id:end_src_id]), original_evidence))
        start_tgt_id = len(par_tgt)
        end_tgt_id = 0
        for i, j in align_words:
            if i >= start_src_id and i < end_src_id:
                if j < start_tgt_id:
                    start_tgt_id = j
                if j > end_tgt_id:
                    end_tgt_id = j

        new_evidence = ' '.join(par_tgt[start_tgt_id:end_tgt_id+1])
        new_start = 0 if len(' '.join(par_tgt[:start_tgt_id])) == 0 else len(' '.join(par_tgt[:start_tgt_id])) + 1 # + space
        if not translated_paragraph[new_start:new_start+len(new_evidence)] == new_evidence:
            logging.warning("there are extra spaces in translated paragraph '{}'\nparagraph words: '{}'\nnew evidence: '{}'\nnew start: '{}'\nevidence mapped to new_start: '{}'".format(translated_paragraph, par_tgt, new_evidence, new_start, translated_paragraph[new_start:new_start+len(new_evidence)]))
            new_start = translated_paragraph.find(new_evidence)
            logging.warning("new start: {}".format(new_start))
            if new_start == -1:
                new_start = translated_paragraph.find(par_tgt[start_tgt_id])
                new_evidence = translated_paragraph[new_start:new_start + len(new_evidence)]
                logging.warning("was not able to find the the new evidence in paragraph, finding the first word of the evidence, '{}'|{}".format(new_evidence, new_start))
        return new_evidence, new_start
