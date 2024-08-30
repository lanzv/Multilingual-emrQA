import evaluate as eval_lib    
#import src.prqa.squad_v1_1_evaluation_script as evaluate
from evaluate import load

class Evaluate:

    def question_answering(gold_dataset, prediction_dict):
        """
        gold_dataset ~ Dataset format
        predictions ~ {"id1": "answer text", "id2": "answer text2", ...}
        prediction for the same question from different paragraph is chosen only one,depending on the min cls
        """
        squad_metric = load("squad")
        predictions = []
        for qaid in prediction_dict:
            predictions.append({'prediction_text': prediction_dict[qaid], 'id': qaid})

        references = []
        for data_sample in gold_dataset:
            references.append({"answers": data_sample["answers"], "id": data_sample["id"]})

        results = squad_metric.compute(predictions=predictions, references=references)
        return results