# Multilingual emrQA


This repository contains the source code for the CL4Health @ NAACL2025 paper **When Multilingual Models Compete with Monolingual Domain-Specific Models in Clinical Question Answering**. 

### Overview
The paper explores the performance of general-domain multilingual models on the clinical Question Answering task using Medication and Relations subsets of emrQA. To improve model performance, we employ multilingual data augmentation, translating an English clinical QA dataset into six additional languages. Our approach involves a translation pipeline that translates report paragraphs and questions, projection of evidence (answers) into target languages, ensuring consistency in question-answer pairs across languages, and filtering out low-quality instances. We systematically evaluate several multilingual models fine-tuned in both mono- and multilingual settings.

![multilingual data augmentation workflow](./imgs/workflow.jpg)

For a detailed description of the pipeline, check the paper.


#### Key Findings

 - The translation process and subsequent QA experiments introduce unique challenges for each language.

 - Contrary to expectations, monolingual domain-specific pretraining does not always outperform general-domain multilingual pretraining on the original English test set.

 - Multilingual models exhibit strong potential for clinical support in languages lacking dedicated clinical NLP resources.


#### Performance comparison of clinical-domain monolingual and general-domain multilingual models

The following table compares BERTBase, medically pretrained BioBERT and clinically pretrained ClinicalBERT models with their multilingual counterpart, mBERT. The mBERT is evaluated in three different settings
 - w/o tgt: multilingually augmented training data exclude the original english data
 - mono: training data include only original english data
 - multi: training data include all multilingually augmented training data (english as well as all dataset translations)

| Model                |Medication EM | Medication F1| Relations EM| Relations F1|
|----------------------|--------------|--------------|-------------|-------------|
| BERTbase             | 31.0         | 72.9         | 91.1        | 96.2        |
| BioBERT              | 31.1         | 74.4         | 91.7        | 96.9        |
| ClinicalBERT         | 31.4         | 73.9         | 92.0        | 96.9        |
| mBERT (*w/o tgt*)    | 31.0         | 75.9         | 90.0        | 96.0        |
| mBERT (*mono*)       | 32.7         | 75.3         | **92.8**    | **97.3**    |
| mBERT (*multi*)      | **33.0**     | **76.7**     | 92.6        | **97.3**    |

For even more interesting observations and results, check the paper.


## How to run experiments
For more details, review the particular scripts to examine the arguments and their associated options that are explicitly hardcoded within the argument parsing dictionaries under the 'ArgumentParser' part.

### `compare_translators.py`
The script `compare_translators.py` evaluates different translation models on a medical dataset ***khresmoi***. The script loads test data, applies the specified translation model and evaluate results.

#### Arguments
- data_dir: Path to the test dataset (default: ../datasets/khresmoi-summary-test-set-2.0).
- models_dir: Directory containing downloaded translation models (default: ../models).
- model: The translation model to use. Must be one of the supported models.
- seed: Random seed for reproducibility (default: 55).
- batch_size: Number of samples processed per batch (default: 50).

#### Supported Models
The following models are supported:
- `NLLB_600M`
- `NLLB_1_3B_dis`
- `NLLB_1_3B`
- `MadLad_3B`
- `NLLB_3_3B`
- `LINDAT`
- `MadLad_7B`
- `MadLad_10B`
- `NLLB_54B`

#### Example Command
```sh
python compare_translators.py --data_dir ../datasets/khresmoi-summary-test-set-2.0 \
                              --models_dir ../models \
                              --model NLLB_600M \
                              --seed 55 \
                              --batch_size 50
```



### `run_translation.py`

The script `run_translation.py` translates paragraphs and questions from the **emrQA** English dataset into a specified target language.

#### Arguments
- data_path: Path to the input emrQA dataset (default: ./data/data.json).
- output_dir: Directory to save translated data (default: ./data/translations).
- target_language: Target language for translation (default: cs).
- disable_prompting: If True, disables translation model prompting (disable PMP) (default: False).
- translated_medical_info_message: Custom message appended to translated medical data (default: "Na základě lékařských zpráv.").
- translation_model_path: Path to the translation model (default: ../models/madlad400-3b-mt).
- translation: If True, enables translation (default: False).
- topics: List of topics to translate (default: ["medication", "relations"]).
- seed: Random seed for reproducibility (default: 55).

#### Example Command
```bash
python run_translation.py --data_path ./data/data.json \
                          --output_dir ./data/translations \
                          --target_language cs \
                          --translation_model_path ../models/madlad400-3b-mt \
                          --topics medication relations \
                          --translated_medical_info_message 'Na podstawie raportów medycznych.#Na podstawie sprawozdań lekarskich.'\
                          --seed 55
```



### `run_alignment.py`

The script `run_alignment.py` aligns answer spans from the original **emrQA** dataset to the translated paragraphs and questions. It computes alignment scores and confidence values for each aligned answer.


#### Arguments
- translation_dataset: Path to the translated dataset containing paragraphs and questions (default: ./data/translations/medication_cs.json).
- dataset: Path to the original emrQA dataset (default: ./data/data.json).
- output_dir: Directory to save aligned answer spans (default: ./data/translation_aligners).
- dataset_title: Title of the dataset being processed (default: "medication").
- language: Target language of the translated dataset (default: "cs").
- aligner_name: Name of the alignment model (default: "Awesome").
- aligner_path: Path to the alignment model (default: ../models/awesome-align-with-co).
- seed: Random seed for reproducibility (default: 55).

#### Example Command
```bash
python run_alignment.py --translation_dataset ./data/translations/medication_cs.json \
                        --dataset ./data/data.json \
                        --output_dir ./data/translation_aligners \
                        --dataset_title medication \
                        --language cs \
                        --aligner_name Awesome \
                        --aligner_path ../models/awesome-align-with-co \
                        --seed 55
```


### `run_report_qa.py`

The script `run_report_qa.py` runs a QA experiment on both full reports and paragraph-level QA tasks. It also supports the removal of low-quality instances by specifying a threshold hyperparameter, allowing you to analyze its effect on QA performance.

#### Arguments
- `dataset`: Path to the dataset translation (default: `../datasets/emrQA/medication_bg.json`).
- `model_name`: Name of the model to use for QA (default: `ClinicalBERT`).
- `model_path`: Path to the model (default: `../models/Bio_ClinicalBERT`).
- `answers_remove_ratio`: Ratio of low-quality answers to remove (default: `0.0`).
- `train_sample_ratio`: Ratio of data to be used for training (default: `0.2`).
- `epochs`: Number of training epochs (default: `3`).
- `to_reports`: If `True`, runs the QA experiment on full reports (default: `False`).
- `paragraph_parts`: If `True`, includes paragraph-level segmentation (default: `False`).
- `seed`: Random seed for reproducibility (default: `2`).

#### Example Command
```bash
python run_report_qa.py --dataset ../datasets/emrQA/medication_bg.json \
                       --model_name ClinicalBERT \
                       --model_path ../models/Bio_ClinicalBERT \
                       --answers_remove_ratio 0.1 \
                       --train_sample_ratio 0.2 \
                       --epochs 3 \
                       --to_reports True \
                       --paragraph_parts True \
                       --seed 2
```

### `run_paragraph_qa.py`

The script `run_paragraph_qa.py` is the final step of the paper, used to evaluate the performance of clinical and general-domain multilingual models on the Paragraph QA task. You can either use multilingual augmented data or fine-tune in a monolingual setting. The script evaluates both, the performance of model in an intersection test sets mode as well as in a full tests mode.

#### Arguments
- `subset`: Subset of the dataset (default: `medication`).
- `language`: Target language for the evaluation (default: `BG`).
- `model_name`: Name of the model to use for QA (default: `ClinicalBERT`).
- `model_path`: Path to the model (default: `../models/Bio_ClinicalBERT`).
- `train_sample_ratio`: Ratio of data used for training (default: `0.2`).
- `train_ratio`: Ratio of data used for training (default: `0.7`).
- `dev_ratio`: Ratio of data used for development/validation (default: `0.1`).
- `epochs`: Number of training epochs (default: `3`).
- `multilingual_train`: If `True`, uses multilingual augmented data for training (default: `False`).
- `remove_target_language_from_train`: If `True`, removes the target language data from the training set (default: `False`).
- `seed`: Random seed for reproducibility (default: `2`).

#### Example Command
```bash
python run_paragraph_qa.py --subset medication \
                           --language BG \
                           --model_name ClinicalBERT \
                           --model_path ../models/Bio_ClinicalBERT \
                           --train_sample_ratio 0.2 \
                           --train_ratio 0.7 \
                           --dev_ratio 0.1 \
                           --epochs 3 \
                           --multilingual_train False \
                           --remove_target_language_from_train False \
                           --seed 2
```





## Citation
To be completed at the time of the NAACL2025 conference - early May.
```bib
TODO, When Multilingual Models Compete with Monolingual Domain-Specific Models in Clinical Question Answering, Vojtech Lanz and Pavel Pecina, CL4Health @ NAACL2025
```
