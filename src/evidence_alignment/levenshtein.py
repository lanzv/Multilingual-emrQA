
import numpy as np
from Levenshtein import ratio
from scipy.optimize import linear_sum_assignment
import heapq
import time
from src.utils import load_translator_model
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)




class LevenshteinWrapper:
    def __init__(self, model_path="../models/madlad400-3b-mt"):
        self.model, self.tokenizer = load_translator_model(model_path=model_path)

    def align_evidence(self, original_evidence, translated_paragraph, target_language="cs", original_position_ratio = 0.0, lambda_ratio=0.8): # original_position_ratio has no effect
        # translate evidence's tokens
        translated_evidence = self.__translate(original_evidence, target_language)
        translated_tokens = translated_evidence.split(' ')

        paragraph_words = translated_paragraph.split(' ')
        # precalculate edit distance table
        ratios = []
        for word in paragraph_words:
            token_ratios = []
            for token in translated_tokens:
                new_ratio = ratio(word.lower(), token.lower())
                token_ratios.append(new_ratio)
            ratios.append(token_ratios)

        # find evidence start
        scores_map = [(-1, (0, 0))]#, (-1, (0, 0)), (-1, (0, 0))] in case we would want to have more candidates for comparision with the original positions, but obsolete since it is comparing evidences +- words next to each other
        first_partity_coef = 1.5
        n = min(int(len(translated_tokens)*first_partity_coef), len(paragraph_words))
        scoring_offset = int(((first_partity_coef-1.0)/2.0)*n) # 0.2 * n
        for i in range(len(paragraph_words) - n + 1):
            bipartit_graph = []
            for j in range(n):
                word_prices = []
                for k in range(len(translated_tokens)):
                    offset_score = 1.0 if j == k + scoring_offset else 0
                    word_prices.append(lambda_ratio*ratios[i+j][k] + (1-lambda_ratio)*offset_score)
                bipartit_graph.append(word_prices)
            try:
                row_ind, column_ind = linear_sum_assignment(np.array(bipartit_graph), maximize=True)
                score = 0
                for j, k in zip(row_ind, column_ind):
                    score += bipartit_graph[j][k]
                score /= len(translated_tokens) # normalization
                indices = (i + np.min(row_ind), i + np.max(row_ind))

                new_record = True
                for j, (recorded_score, recorded_indices) in enumerate(scores_map): 
                    first_word_id, last_word_id = recorded_indices
                    if first_word_id == i + np.min(row_ind): 
                        new_record = False
                        if score > recorded_score:
                            scores_map[j] = (score, indices)
                if new_record:
                    heapq.heappushpop(scores_map, (score, indices))

            except Exception as e:
                logging.info(e)
                logging.info(bipartit_graph)

        top_scores = sorted(scores_map, reverse=True)
        top_pos_ratio_diff = 1
        best_candidate_index = -1
        candidates = []
        for i, (score, indices) in enumerate(top_scores):
            first_word_id, last_word_id = indices
            translated_evidence = ' '.join(paragraph_words[first_word_id:last_word_id+1])
            evidence_start = translated_paragraph.find(translated_evidence)
            candidates.append({"text": translated_evidence, "answer_start": evidence_start, "score": score})
            if evidence_start == -1:
                raise Exception("Evidence '{}' was not found in paragraph '{}' but should be there".format(translated_evidence, translated_paragraph))
            new_position_ratio = float(evidence_start)/float(len(translated_paragraph))
            if abs(new_position_ratio-original_position_ratio) < top_pos_ratio_diff:
                top_pos_ratio_diff = abs(new_position_ratio-original_position_ratio)
                best_candidate_index = i
        translated_evidence = candidates[best_candidate_index]["text"]
        evidence_start = candidates[best_candidate_index]["answer_start"]

        return translated_evidence, evidence_start

    def __translate(self, original_evidence, target_language, no_repeat_ngram_size=12):
        input_ids = self.tokenizer("<2{}> {}".format(target_language, original_evidence), return_tensors="pt").input_ids.to(self.model.device)
        outputs = self.model.generate(input_ids=input_ids, max_new_tokens=int(1.5*input_ids.shape[1]), no_repeat_ngram_size=no_repeat_ngram_size)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

