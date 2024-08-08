# RES-Q Stats

## Preliminary QA results
 - Initial not perfect train/test split
    - Test set: last 53 res-q reports of the given language
    - Train set: the rest of reports of the given language
 - dataset/annotation mistakes/bugs 
    - duplicated question ids etc..
 - uniform segmentation -> fixing boundaries that none of answer is splited
 - complex answers ~ alternative answers (it is sufficient to find at least one)


|               | BG    | EL    | EN    | PL    | RO    |
|---------------|-------|-------|-------|-------|-------|
|# train reports| 230   | 232   | 190   | 79    | 239   |
|# test reports | 53    | 53    | 53    | 53    | 53    |

### Training using only reports of the given res-q language
| mBERT        | BG    | EL    | EN    | PL    | RO    |
|--------------|-------|-------|-------|-------|-------|
| Oracle-QA F1 | 83.11 | 86.23 | 72.62 | 67.53 | 80.85 |
| Oracle-QA EM | 75.84 | 75.49 | 59.52 | 47.91 | 68.09 |
| PR (P@1)     | 99.13 | 99.05 | 79.62 | 76.45 | 89.32 |
| PR (P@2)     | 98.50 | 97.79 | 92.72 | 87.61 | 94.38 |
| PR (P@3)     | 99.47 | 98.55 | 96.81 | 92.77 | 97.21 |
|**PR-QA F1**  | 80.61 | 83.89 | 64.33 | 58.75 | 74.90 |
|**PR-QA EM**  | 73.56 | 72.97 | 53.34 | 41.84 | 63.59 |

| Ontotext XLM | BG    | EL    | EN    | PL    | RO    |
|--------------|-------|-------|-------|-------|-------|
| Oracle-QA F1 | 85.14 | 85.99 | 71.86 | 64.85 | 75.74 |
| Oracle-QA EM | 77.90 | 74.98 | 59.96 | 45.89 | 64.25 |
| PR (P@1)     | 97.77 | 98.70 | 78.00 | 76.08 | 81.32 |
| PR (P@2)     | 97.82 | 97.55 | 91.94 | 85.40 | 85.16 |
| PR (P@3)     | 98.74 | 98.59 | 96.30 | 90.64 | 87.08 |
|**PR-QA F1**  | 81.63 | 71.92 | 62.93 | 56.17 | 67.38 |
|**PR-QA EM**  | 74.47 | 82.63 | 52.74 | 39.64 | 57.43 |


### Training using all res-q reports (all language)
| mBERT        | BG    | EL    | EN    | PL    | RO    |
|--------------|-------|-------|-------|-------|-------|
| Oracle-QA F1 | 84.85 | 86.28 | 73.05 | 69.56 | 81.46 |
| Oracle-QA EM | 77.58 | 75.85 | 60.01 | 50.54 | 68.23 |
| PR (P@1)     | 98.74 | 99.20 | 77.94 | 80.24 | 90.05 |
| PR (P@2)     | 98.50 | 98.36 | 91.99 | 88.88 | 94.57 |
| PR (P@3)     | 99.18 | 98.89 | 95.86 | 92.63 | 97.30 |
|**PR-QA F1**  | 82.12 | 83.90 | 64.34 | 61.52 | 75.72 |
|**PR-QA EM**  | 75.02 | 73.43 | 53.93 | 45.45 | 64.12 |


### Training using all res-q reports (all language) + emrQA
| mBERT        | BG    | EL    | EN    | PL    | RO    |
|--------------|-------|-------|-------|-------|-------|
| Oracle-QA F1 | 83.89 | 84.39 | 69.42 | 68.00 | 80.34 |
| Oracle-QA EM | 75.52 | 74.19 | 56.16 | 47.91 | 66.74 |
| PR (P@1)     | 98.40 | 98.51 | 78.50 | 79.16 | 90.50 |
| PR (P@2)     | 98.06 | 97.67 | 91.77 | 87.48 | 94.11 |
| PR (P@3)     | 99.18 | 98.44 | 95.91 | 92.00 | 96.99 |
|**PR-QA F1**  | 81.11 | 81.80 | 61.47 | 60.14 | 75.04 |
|**PR-QA EM**  | 73.01 | 71.67 | 50.84 | 42.96 | 63.02 |



## ? Tool bugs ?
 - empty reports (context ~ None): ***5***

After empty reports filtration:
 - empty qa['question_id']s (qa['question_id'] ~ None): ***246***
 - duplicated qa['id']s: ***2603*** (/66533 unique qa['id'])

```python
if report["hospital_id"] is not None:
    qa["id"] = report["hospital_id"] + "_" + report["report_id"] + "_" + qa["question_id"]
else:
    qa["id"] = report["report_id"] + "_" + qa["question_id"]
```

duplicated qa["question_id"]s:
 - BG
    - 'admission.admission_timestamp': 277
    - 'onset.onset_timestamp': 231
    - 'diagnosis.imaging.imaging_timestamp': 209
    - 'onset.sleep_timestamp': 8
 - EL
    - 'diagnosis.imaging.imaging_timestamp': 279
    - 'admission.admission_timestamp': 278
    - 'onset.onset_timestamp': 263
    - 'onset.sleep_timestamp': 16
    - 'discharge.destination.discharge_destination': 4
    - 'discharge.medication.antiplatelet_substances.other_antiplatelet': 1
 - EN
    - 'admission.admission_timestamp': 231
    - 'discharge.destination.discharge_destination': 14
    - 'onset.onset_timestamp': 8
    - 'diagnosis.imaging.imaging_timestamp': 1
 - PL
    - 'admission.admission_timestamp': 126
    - 'onset.onset_timestamp': 73
    - 'diagnosis.imaging.imaging_timestamp': 26
    - 'discharge.destination.discharge_destination': 19
    - 'onset.sleep_timestamp': 9
    - 'diagnosis.imaging.perfusion.perfusion_deficit': 2
 - RO
    - 'diagnosis.imaging.imaging_timestamp': 232
    - 'onset.onset_timestamp': 198
    - 'admission.admission_timestamp': 73
    - 'discharge.destination.discharge_destination': 16
    - 'onset.sleep_timestamp': 9

   



## ? Annotation mistakes ?
After empty reports filtration:
 - empty answers (answer['text'] ~ None, answer['answer_start'] ~ -1): ***16103***
 - empty asnwer text (len(answer['text']) == 0) but span is there: ***2***
 - empty hospital ids: ***266***


qa["question_id"] of empty answers (top):
 - BG
    - 'onset.inhospital_stroke': 279
    - 'post_acute_care.craniectomy': 277
    - 'post_acute_care.fever.day_1_fever_checks': 275
    - 'post_acute_care.fever.day_2_fever_checks': 271
    - 'post_acute_care.fever.day_3_fever_checks': 268
 - EL
    - 'onset.wakeup_stroke': 5
    - 'admission.medical_examination.inr_mode', 5
    - 'post_acute_care.craniectomy': 5
    - 'admission.hospitalized_in': 4
    - 'treatment.ischemic_stroke.thrombectomy_treatment': 4
 - EN
    - 'post_acute_care.patient_ventilated': 225
    - 'post_acute_care.swallow.swallowing_screening': 224
    - 'anamnesis.prestroke_mrs': 221
    - 'post_acute_care.vte.any_vte': 208
    - 'discharge.smoking_cessation_recommended': 186
 - PL
    - 'post_acute_care.craniectomy': 104
    - 'treatment.ischemic_stroke.thrombectomy_treatment': 95
    - 'post_acute_care.carotid_arteries.carotid_endarterectomy_within_2_weeks': 93
    - 'post_acute_care.patient_ventilated': 92
    - 'admission.ems_prenotification': 91
 - RO
    - 'post_acute_care.craniectomy': 272
    - 'onset.inhospital_stroke': 270
    - 'post_acute_care.patient_ventilated': 256
    - 'onset.wakeup_stroke': 251
    - 'admission.arrival_mode': 230
    

## Dataset stats (after mistakes/bugs filtration)
 - number of reports: ***1235***
 - number of questions: ***52809***
 - number of answers: ***77585***
 - answer lengths (avg): ***17.27***	(min: 1,	 max: 934)
 - question lengths (avg): ***45.29***	(min: 3,	 max: 165)

|                     | BG    | EL    | EN    | PL    | RO    |
|---------------------|-------|-------|-------|-------|-------|
| number of reports   |  283  | 285   | 243   | 132   | 292   |
| number of questions | 11663 | 14831 | 8357  | 5514  | 12444 |
| number of answers   | 17217 | 19812 | 11581 | 10763 | 18212 |
