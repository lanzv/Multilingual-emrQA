# CS-PL-SP-RM-emrQA




## Machine Translation 

### MT Model Statistics - khresmoi
The 48GB RAM node was used. Only NLLB-54B, MadLad10B and MadLad7B models are larger than the GPU free space -> load_in_4bit=True, torch_dtype=torch.float16 parameters are set while loading the model. 

We translated the english dataset into the target languages by batches of 50 sentences. That make the translation 20x faster than translating sentence by sentence.

For the evaluation, we used the ***khresmoi dataset*** (1500 medical setences translated by professionalists from the english to the cs, de, fr, hu, pl, es, sv)

The final and most accurate results are stored in the ```scripts/slurm_outputs/measure_translators.out``` file. The json output is in the file ```scripts/slurm_outputs/translator_results.json```. Rounded values are in the tables below.. 

Average table contains LINDAT averaged over only four languages (CS, DE, FR, PL) since LINDAT doesn't know SV, SE, HU. The ```scripts/slurm_outputs/translator_results.json``` file is averaged over all languages. 



|     AVERAGE          |   NLLB_600M |   NLLB_1_3B_dis |   NLLB_1_3B |   MadLad_3B |   NLLB_3_3B |   LINDAT |   MadLad_7B |   MadLad_10B |   NLLB_54B |
|:---------------------|------------:|----------------:|------------:|------------:|------------:|---------:|------------:|-------------:|-----------:|
| loading time (s)     |       6.49  |         20.02   |     20.05   |     18.04   |     40.13   |     0    |    471.67   |     601.76   |  3861.49   |
| translation time (s) |      33.6   |         54.98   |     55.81   |     98.96   |     84.64   |   838.62 |    302.84   |     326.79   |   956.99   |
| bleu                 |      31.66  |         34.19   |     33.9    |     37.68   |     35.14   |    33.74 |     37.65   |    **37.77** |    36.2    |
| meteor               |       0.567 |          0.5911 |      0.5895 |      0.6258 |      0.6022 |     0.59 |      0.6264 |    **0.6267**|     0.6102 |
| wer                  |      52.9   |         50.63   |     50.76   |     47.15   |     49.82   |    50.75 |     47.44   |    **46.98** |    48.75   |
| cer                  |      39.31  |         37.68   |     37.94   |     35.15   |     36.93   |    37.73 |     35.27   |    **34.94** |    36.14   |

|  EN -> CS (1500 sens)|   NLLB_600M |   NLLB_1_3B_dis |   NLLB_1_3B |   MadLad_3B |   NLLB_3_3B |   LINDAT |   MadLad_7B |   MadLad_10B |   NLLB_54B |
|:---------------------|------------:|----------------:|------------:|------------:|------------:|---------:|------------:|-------------:|-----------:|
| translation time (s) |      33.31  |         51.62   |     51.31   |     95.37   |     81.49   | 798.94   |    295.92   |     323.23   |    922.34  |
| bleu                 |      28.87  |         34.65   |     33.02   |     38.85   |     35.04   |  39.04   |     38.77   |    **39.28** |     38.23  |
| meteor               |       0.544 |          0.5911 |      0.5837 |      0.6367 |      0.6018 |   0.6337 |      0.6341 |    **0.6394**|      0.623 |
| wer                  |      55.41  |         50.35   |     51.62   |     45.91   |     49.97   |**45.56** |     46.15   |      45.61   |     47.28  |
| cer                  |      41.1   |         37.7    |     38.81   |     34.71   |     37.32   |  34.55   |     35.01   |    **34.38** |     35.36  |

|  EN -> DE (1500 sens)|   NLLB_600M |   NLLB_1_3B_dis |   NLLB_1_3B |   MadLad_3B |   NLLB_3_3B |   LINDAT |   MadLad_7B |   MadLad_10B |   NLLB_54B |
|:---------------------|------------:|----------------:|------------:|------------:|------------:|---------:|------------:|-------------:|-----------:|
| translation time (s) |     29.21   |          49.26  |     49.54   |      96.25  |     86.63   | 840.16   |     299.54  |     326.04   |  1024.32   |
| bleu                 |     30.08   |          31.3   |     31.4    |      34.43  |     32.59   |  30.77   |      34.47  |    **34.7**  |    33.46   |
| meteor               |      0.5732 |           0.585 |      0.5839 |       0.611 |      0.5949 |   0.5785 |    **0.613** |       0.6101 |     0.5992 |
| wer                  |     52.18   |          51.14  |     51.33   |  **49.03**  |     50.95   |  52.69   |      49.16  |    **49.03** |    50.36   |
| cer                  |     38.48   |          37.6   |     37.88   |    35.94    |     37.44   |  38.24   |      36.07  |    **35.78** |    37.19   |

|  EN -> FR (1500 sens)|   NLLB_600M |   NLLB_1_3B_dis |   NLLB_1_3B |   MadLad_3B |   NLLB_3_3B |   LINDAT |   MadLad_7B |   MadLad_10B |   NLLB_54B |
|:---------------------|------------:|----------------:|------------:|------------:|------------:|---------:|------------:|-------------:|-----------:|
| translation time (s) |      33.62  |         51.32   |     51.68   |     99.54   |     89.56   | 937.41   |    310.4    |     339.98   |   1039.16  |
| bleu                 |      46.67  |         47.65   |     48.17   |     49.21   |     47.99   |  47.28   |     48.93   |   **49.88**   |     48.3   |
| meteor               |       0.713 |          0.7188 |      0.7224 |      0.7307 |      0.7218 |   0.7144 |      0.7305 |   **0.7364** |      0.723 |
| wer                  |      41.43  |         40.67   |     39.93   |     40.33   |     40.68   |  39.65   |     41.03   |    **39.46** |     40.65  |
| cer                  |      27.82  |         27.01   |     26.94   |     26.72   |     27.17   |  27.9    |     26.87   |    **26.4**  |     26.84  |

|  EN -> HU (1500 sens)|   NLLB_600M |   NLLB_1_3B_dis |   NLLB_1_3B |   MadLad_3B |   NLLB_3_3B |   LINDAT |   MadLad_7B |   MadLad_10B |   NLLB_54B |
|:---------------------|------------:|----------------:|------------:|------------:|------------:|---------:|------------:|-------------:|-----------:|
| translation time (s) |     35.9    |         68.16   |     55.54   |    127.29   |     89.52   |        - |    313.19   |      351.09  |   949.36   |
| bleu                 |     13.04   |         15.8    |     15.29   |     19.41   |     16.96   |        - |   **20.48** |       19.94  |    18.91   |
| meteor               |      0.3577 |          0.3948 |      0.3899 |      0.4403 |      0.4114 |        - |   **0.4517**|        0.448 |     0.4317 |
| wer                  |     72.66   |         69.78   |     69.62   |     65.37   |     68.37   |        - |     64.89   |     **64.43**|    65.93   |
| cer                  |     56.87   |         55.27   |     54.9    |     52.33   |     53.62   |        - |     51.33   |     **51.29**|    51.73   |

|  EN -> PL (1500 sens)|   NLLB_600M |   NLLB_1_3B_dis |   NLLB_1_3B |   MadLad_3B |   NLLB_3_3B |   LINDAT |   MadLad_7B |   MadLad_10B |   NLLB_54B |
|:---------------------|------------:|----------------:|------------:|------------:|------------:|---------:|------------:|-------------:|-----------:|
| translation time (s) |     45.09   |           69.6  |      93.33  |    102.33   |     90.5    | 777.98   |    320.2    |     350.61   |  1016.16   |
| bleu                 |     14.97   |           17.37 |      16.94  |     20.46   |     18.41   |  17.87   |   **20.95** |      20.5    |    19.24   |
| meteor               |      0.3786 |            0.41 |       0.407 |      0.4545 |      0.4264 |   0.4163 |  **0.4598** |       0.4546 |     0.4368 |
| wer                  |     70.64   |           66.7  |      68.07  |     62.33   |     65.36   |  65.1    |   **61.8**  |      62.1    |    63.98   |
| cer                  |     55.53   |           52.33 |      53.83  |     48.11   |     50.73   |  50.24   |   **47.67** |      47.9    |    49.55   |

|  EN -> ES (1500 sens)|   NLLB_600M |   NLLB_1_3B_dis |   NLLB_1_3B |   MadLad_3B |   NLLB_3_3B |   LINDAT |   MadLad_7B |   MadLad_10B |   NLLB_54B |
|:---------------------|------------:|----------------:|------------:|------------:|------------:|---------:|------------:|-------------:|-----------:|
| translation time (s) |     27.08   |         45.81   |     45.87   |     87.71   |     80.15   |        - |    278.2    |     305.52   |   921.33   |
| bleu                 |     46.09   |         47.62   |     47.19   |   **49.05** |     48.05   |        - |     48.55   |      48.27   |    47.98   |
| meteor               |      0.7364 |          0.7462 |      0.7476 |  **0.7596** |      0.7534 |        - |      0.7555 |       0.7545 |     0.7505 |
| wer                  |     37.85   |         37.12   |     37.44   |   **35.7**  |     36.84   |        - |     36.27   |      36.48   |    36.7    |
| cer                  |     26.41   |         26.3    |     26.47   |  **25.19**  |     26.05   |        - |     25.72   |      25.69   |    26.12   |

|  EN -> SV (1500 sens)|   NLLB_600M |   NLLB_1_3B_dis |   NLLB_1_3B |   MadLad_3B |   NLLB_3_3B |   LINDAT |   MadLad_7B |   MadLad_10B |   NLLB_54B |
|:---------------------|------------:|----------------:|------------:|------------:|------------:|---------:|------------:|-------------:|-----------:|
| translation time (s) |     31      |          49.1   |      43.37  |     84.19   |     74.63   |        - |    302.41   |     291.06   |   826.25   |
| bleu                 |     41.93   |          44.95  |      45.31  |    **52.34**|     46.97   |        - |     51.42   |      51.82   |    47.26   |
| meteor               |      0.6658 |           0.692 |       0.692 |    **0.748**|      0.7059 |        - |      0.7402 |       0.7437 |     0.7071 |
| wer                  |     40.1    |          38.63  |      37.32  |    **31.4** |     36.55   |        - |     32.76   |      31.78   |    36.34   |
| cer                  |     28.93   |          27.54  |      26.77  |    **23.07**|     26.17   |        - |     24.21   |      23.14   |    26.2    |



## Evidence Alignment

When Prompting the translation (using medical message) some of paragraphs needed to be translated via DeepL / Google Translator


|  Medication                           |          |
|---------------------------------------|----------|
| overall paragraph parts               | 6 268    | 
| overall paragraphs                    | 5 081    |
| number of answers                     | 254 875  |
| number of questions                   | 232 347  |


|  Medication                           | BG         | CS       | ES       | PL       | RO         |
|---------------------------------------|------------|----------|----------|----------|------------|
| paragraph parts translated "manually" | 4 (0.06%)  | 1 (0.02%)| 0 (0.00%)| 3 (0.05%)| 3 (0.05%)  |
| paragraphs translated "manually"      | 4 (0.08%)  | 1 (0.02%)| 0 (0.00%)| 3 (0.06%)| 3 (0.06%)  |
| answers in affected paragraphs        | 789 (0.31%)| 0 (0.00%)| 0 (0.00%)| 0 (0.00%)| 471 (0.18%)|
| questions in affected paragraphs      | 704 (0.30%)| 0 (0.00%)| 0 (0.00%)| 0 (0.00%)| 407 (0.18%)|
| answers in affected parts             | 245 (0.10%)| 0 (0.00%)| 0 (0.00%)| 0 (0.00%)| 5 (0.00%)  |
| questions in affected parts           | 222 (0.10%)| 0 (0.00%)| 0 (0.00%)| 0 (0.00%)| 5 (0.00%)  |


|  Relations                            |            |
|---------------------------------------|------------|
| overall paragraph parts               | 11 054     | 
| overall paragraphs                    | 9 482      |
| number of answers                     | 1 021 514  |
| number of questions                   | 987 965    |

|  Relations                            | BG         | CS         | ES       | PL          | RO         |
|---------------------------------------|------------|------------|----------|-------------|------------|
| paragraph parts translated "manually" | 2 (0.02%)  | 6 (0.05%)  | 0 (0.00%)| 4 (0.04%)   | 1 (0.01%)  |
| paragraphs translated "manually"      | 2 (0.02%)  | 6 (0.06%)  | 0 (0.00%)| 4 (0.04%)   | 1 (0.01%)  |
| answers in affected paragraphs        | 547 (0.05%)| 445 (0.04%)| 0 (0.00%)| 1036 (0.10%)| 0 (0.00%)  |
| questions in affected paragraphs      | 547 (0.06%)| 445 (0.05%)| 0 (0.00%)| 1020 (0.10%)| 0 (0.00%)  |
| answers in affected parts             | 324 (0.03%)| 222 (0.02%)| 0 (0.00%)| 565 (0.06%) | 0 (0.00%)  |
| questions in affected parts           | 324 (0.03%)| 222 (0.02%)| 0 (0.00%)| 549 (0.06%) | 0 (0.00%)  |



### Medication

|  Awesome        | BG       | CS       | EL       | ES       | PL       | RO       |
|-----------------|----------|----------|----------|----------|----------|----------|
| f1              | 88.52    | 92.51    | 69.50    | 95.97    | 90.31    | 93.32    |
| em              | 59.03    | 66.24    | 25.54    | 73.17    | 59.26    | 67.82    |
| exact submatch  | 88.07    | 88.13    | 79.88    | 92.89    | 86.76    | 92.05    |
| f1 span         | 88.14    | 92.31    | 68.84    | 95.77    | 90.04    | 92.95    |
| precision span  | 85.10    | 90.43    | 63.03    | 94.37    | 87.49    | 90.70    |
| recall span     | 97.74    | 97.90    | 93.75    | 97.87    | 97.77    | 98.55    |
| start distance  | -10.74   | -4.72    | -29.94   | -2.22    | -8.98    | -5.21    |
| middle distance | 0.60     | 1.31     | 5.62     | 0.70     | 0.08     | 0.68     |
| end distance    | 11.94    | 7.34     | 41.17    | 3.62     | 9.15     | 6.58     |
| abs(start dst)  | 13.14    | 7.90     | 43.84    | 3.83     | 10.53    | 6.89     |
| abs(mid dst)    | 9.80     | 6.76     | 30.90    | 3.49     | 8.01     | 5.71     |
| abs(end dst)    | 13.20    | 8.50     | 43.91    | 4.29     | 10.44    | 7.37     |
| overall time    | 25308.67 | 23883.82 | 42715.21 | 22538.61 | 25391.91 | 25186.79 |

|   Levenshtein   | BG       | CS       | EL       | ES       | PL       | RO       |
|-----------------|----------|----------|----------|----------|----------|----------|
| f1              | 74.55    | 81.52    | 68.16    | 82.40    | 79.34    | 80.60    |
| em              | 12.87    | 26.46    | 10.13    | 29.22    | 20.46    | 24.89    |
| exact submatch  | 62.64    | 74.09    | 52.35    | 76.93    | 70.96    | 73.99    |
| f1 span         | 70.83    | 78.69    | 63.22    | 79.72    | 76.21    | 77.70    |
| precision span  | 65.13    | 72.56    | 58.01    | 73.27    | 69.48    | 71.36    |
| recall span     | 83.38    | 90.59    | 75.22    | 91.99    | 89.31    | 90.33    |
| start distance  | -19.47   | -15.29   | -29.11   | -16.07   | -21.20   | -17.82   |
| middle distance | -6.97    | -4.18    | -15.83   | -4.22    | -9.00    | -5.35    |
| end distance    | 5.54     | 6.94     | -2.54    | 7.62     | 3.20     | 7.13     |
| abs(start dst)  | 58.23    | 36.60    | 80.83    | 31.41    | 41.50    | 41.50    |
| abs(mid dst)    | 52.63    | 30.77    | 74.53    | 25.77    | 34.88    | 34.94    |
| abs(end dst)    | 55.28    | 32.18    | 76.58    | 27.71    | 35.89    | 36.79    |
| overall time    | 12491.69 | 12331.76 | 15930.24 | 12275.01 | 12982.80 | 12961.67 |

### Medication No Translation Prompt (straightforward translation)

|  Awesome        | BG       | CS       | EL       | ES       | PL       | RO       |
|-----------------|----------|----------|----------|----------|----------|----------|
| f1              | 81.84    | 88.71    | 69.50    | 93.06    | 88.82    | 90.03    |
| em              | 46.06    | 62.11    | 25.54    | 70.98    | 57.62    | 63.76    |
| exact submatch  | 80.55    | 84.00    | 79.88    | 90.64    | 85.06    | 89.69    |
| f1 span         | 81.30    | 88.19    | 68.84    | 92.81    | 88.46    | 89.51    |
| precision span  | 78.70    | 87.36    | 63.03    | 91.74    | 86.13    | 87.54    |
| recall span     | 93.76    | 94.74    | 93.75    | 96.96    | 96.47    | 96.50    |
| start distance  | -8.23    | -3.05    | -29.94   | -0.66    | -7.51    | -4.40    |
| middle distance | 6.86     | 4.69     | 5.62     | 3.99     | 1.86     | 3.51     |
| end distance    | 21.94    | 12.43    | 41.17    | 8.64     | 11.23    | 11.42    |
| abs(start dst)  | 24.47    | 13.73    | 43.84    | 9.89     | 12.65    | 12.15    |
| abs(mid dst)    | 19.84    | 11.69    | 30.90    | 8.92     | 10.06    | 10.42    |
| abs(end dst)    | 24.59    | 14.41    | 43.91    | 9.85     | 12.89    | 12.74    |
| overall time    | 24468.68 | 23677.85 | 26723.34 | 22788.85 | 25121.89 | 24929.89 |

|   Levenshtein   | BG       | CS       | EL       | ES       | PL       | RO       |
|-----------------|----------|----------|----------|----------|----------|----------|
| f1              | 70.71    | 79.01    | 68.16    | 80.93    | 78.33    | 78.52    |
| em              | 11.26    | 25.66    | 10.13    | 29.82    | 21.10    | 24.76    |
| exact submatch  | 59.79    | 68.51    | 52.35    | 74.54    | 69.02    | 70.78    |
| f1 span         | 65.93    | 75.02    | 63.22    | 77.65    | 74.77    | 74.78    |
| precision span  | 59.84    | 69.48    | 58.01    | 71.69    | 68.32    | 68.88    |
| recall span     | 79.12    | 85.56    | 75.22    | 88.95    | 87.42    | 86.65    |
| start distance  | -23.74   | -13.50   | -29.11   | -19.95   | -26.52   | -18.25   |
| middle distance | -9.92    | -3.10    | -15.83   | -8.46    | -14.40   | -5.96    |
| end distance    | 3.89     | 7.31     | -2.54    | 3.03     | -2.27    | 6.33     |
| abs(start dst)  | 71.14    | 46.29    | 80.83    | 42.92    | 48.99    | 50.51    |
| abs(mid dst)    | 64.83    | 41.31    | 74.53    | 37.47    | 42.48    | 44.46    |
| abs(end dst)    | 67.17    | 42.28    | 76.58    | 38.86    | 43.40    | 46.08    |
| overall time    | 12941.25 | 13249.59 | 14181.76 | 12787.90 | 13371.27 | 13692.39 |

#### CS

|                          | Levenshtein | FastAlign | Awesome     |
|--------------------------|-------------|-----------|-------------|
| f1 (%)                   | 81.5242     | 90.2794   | **92.5061** |
| exact match (%)          | 26.4555     | 52.8824   | **66.2100** |
| exact submatch (%)       | 74.0868     | 76.7266   | **88.0993** |
| f1 span (%)              | 78.6911     | 89.7825   | **92.3010** |
| precision span (%)       | 72.5578     | 89.0736   | **90.4308** |
| recall span (%)          | 90.5884     | 95.1786   | **97.8830** |
| start dist (chars)       | -15.2925    | -5.5183   | **-4.7103** |
| middle dist (chars)      | -4.1770     | 1.4585    | **1.3075**  |
| end dist (chars)         | **6.9385**  | 8.4354    | 7.3253      |
| abs(start dist) (chars)  | 36.5947     | 10.8154   | **7.9055**  |
| abs(middle dist) (chars) | 30.7720     | 9.1500    | **6.7628**  |
| abs(end dist) (chars)    | 32.1797     | 10.4930   | **8.5043**  |
| time (s)                 | 12835       | 2952      | 23221       |

### Relations

|  Awesome        | BG       | CS       | EL       | ES       | PL       | RO       |
|-----------------|----------|----------|----------|----------|----------|----------|
| f1              | 89.20    | 93.02    | 76.85    | 96.80    | 89.81    | 94.22    |
| em              | 67.68    | 71.22    | 41.74    | 79.94    | 63.45    | 76.84    |
| exact submatch  | 88.12    | 83.22    | 82.48    | 84.96    | 82.17    | 88.81    |
| f1 span         | 88.77    | 92.83    | 76.47    | 96.96    | 89.39    | 93.93    |
| precision span  | 86.94    | 92.46    | 71.97    | 97.21    | 88.23    | 93.30    |
| recall span     | 97.07    | 96.92    | 95.16    | 97.96    | 96.37    | 97.55    |
| start distance  | -11.29   | -4.34    | -22.74   | 1.76     | -7.97    | -2.27    |
| middle distance | 1.00     | 1.88     | 5.51     | 3.08     | 1.61     | 2.63     |
| end distance    | 13.28    | 8.10     | 33.77    | 4.39     | 11.20    | 7.54     |
| abs(start dst)  | 15.63    | 9.94     | 35.46    | 5.52     | 12.92    | 9.46     |
| abs(mid dst)    | 11.20    | 7.85     | 25.89    | 4.84     | 9.75     | 7.69     |
| abs(end dst)    | 13.96    | 8.69     | 35.52    | 4.73     | 12.01    | 8.04     |
| overall time    | 39761.12 | 37191.16 | 41449.54 | 35514.34 | 39882.23 | 38751.78 |

|  Levenshtein    | BG       | CS       | EL       | ES       | PL       | RO       |
|-----------------|----------|----------|----------|----------|----------|----------|
| f1              | 81.96    | 85.75    | 79.11    | 85.24    | 83.68    | 85.22    |
| em              | 29.36    | 35.62    | 26.11    | 36.59    | 30.73    | 36.52    |
| exact submatch  | 80.70    | 84.22    | 73.74    | 87.83    | 80.90    | 84.75    |
| f1 span         | 79.43    | 83.02    | 76.07    | 82.64    | 81.03    | 82.85    |
| precision span  | 72.75    | 76.65    | 69.92    | 75.73    | 74.84    | 76.58    |
| recall span     | 91.26    | 93.71    | 87.08    | 94.82    | 91.66    | 93.88    |
| start distance  | -15.62   | -13.75   | -22.45   | -12.86   | -13.57   | -14.37   |
| middle distance | -3.89    | -3.41    | -11.03   | -0.69    | -3.47    | -3.57    |
| end distance    | 7.83     | 6.93     | 0.40     | 11.48    | 6.64     | 7.22     |
| abs(start dst)  | 33.38    | 28.64    | 44.70    | 28.06    | 32.07    | 26.91    |
| abs(mid dst)    | 26.14    | 22.38    | 37.56    | 20.65    | 25.98    | 20.45    |
| abs(end dst)    | 25.93    | 21.91    | 36.80    | 20.81    | 25.44    | 20.19    |
| overall time    | 25486.86 | 24743.43 | 31139.07 | 35514.34 | 26347.79 | 25082.60 |


### Relations No Translation Prompt (straightforward translation)

|  Awesome        | BG       | CS       | EL       | ES       | PL       | RO       |
|-----------------|----------|----------|----------|----------|----------|----------|
| f1              | 86.07    | 90.70    | 76.89    | 94.82    | 87.83    | 91.76    |
| em              | 61.67    | 67.09    | 41.83    | 77.48    | 60.59    | 73.14    |
| exact submatch  | 84.69    | 80.85    | 82.50    | 83.57    | 80.82    | 86.47    |
| f1 span         | 85.65    | 90.45    | 76.51    | 94.88    | 87.37    | 91.38    |
| precision span  | 83.84    | 90.19    | 72.02    | 95.38    | 86.14    | 90.94    |
| recall span     | 94.86    | 95.12    | 95.17    | 96.21    | 95.01    | 95.64    |
| start distance  | -6.07    | 3.00     | -22.70   | 8.07     | -4.21    | 2.46     |
| middle distance | 6.81     | 9.71     | 5.50     | 9.46     | 6.05     | 8.11     |
| end distance    | 19.69    | 16.42    | 33.71    | 10.85    | 16.32    | 13.76    |
| abs(start dst)  | 22.63    | 19.43    | 35.40    | 12.89    | 19.11    | 16.53    |
| abs(mid dst)    | 18.29    | 16.87    | 25.85    | 12.02    | 15.35    | 14.38    |
| abs(end dst)    | 21.25    | 17.59    | 35.46    | 11.78    | 17.51    | 14.89    |
| overall time    | 39855.47 | 36948.81 | 62023.56 | 35670.57 | 39494.94 | 38587.98 |

|  Levenshtein    | BG       | CS       | EL       | ES       | PL       | RO       |
|-----------------|----------|----------|----------|----------|----------|----------|
| f1              | 79.81    | 83.86    | 79.11    | 83.92    | 82.14    | 83.15    |
| em              | 28.04    | 34.14    | 26.11    | 36.86    | 30.26    | 34.84    |
| exact submatch  | 78.09    | 81.62    | 73.74    | 86.08    | 79.02    | 82.73    |
| f1 span         | 76.89    | 80.96    | 76.07    | 81.05    | 79.30    | 80.38    |
| precision span  | 70.37    | 74.74    | 69.92    | 74.44    | 73.29    | 74.14    |
| recall span     | 88.46    | 91.46    | 87.08    | 92.71    | 89.64    | 91.39    |
| start distance  | -16.34   | -17.46   | -22.45   | -19.70   | -14.09   | -21.40   |
| middle distance | -4.32    | -6.93    | -11.03   | -7.59    | -4.02    | -10.09   |
| end distance    | 7.70     | 3.59     | 0.40     | 4.52     | 6.04     | 1.21     |
| abs(start dst)  | 43.42    | 37.30    | 44.70    | 37.45    | 39.03    | 39.42    |
| abs(mid dst)    | 36.14    | 31.15    | 37.56    | 30.13    | 33.13    | 32.75    |
| abs(end dst)    | 35.64    | 30.60    | 36.80    | 30.12    | 32.70    | 32.34    |
| overall time    | 26174.74 | 25563.27 | 31415.44 | 25448.92 | 26353.36 | 25813.90 |


#### CS

|                          | Levenshtein | FastAlign   | Awesome     |
|--------------------------|-------------|-------------|-------------|
| f1 (%)                   | 85.7290     | **93.2484** | 93.0164     |
| exact match (%)          | 35.5312     | 68.1078     | **71.1927** |
| exact submatch (%)       | 84.2168     | 81.8705     | **83.2066** |
| f1 span (%)              | 82.9966     | 92.5678     | **92.8223** |
| precision span (%)       | 76.6173     | **92.9844** | 92.4547     |
| recall span (%)          | 93.7152     | 95.6225     | **96.9220** |
| start dist (chars)       | -13.7699    | **-2.8955** | -4.3442     |
| middle dist (chars)      | -3.4131     | 2.0120      | **1.8843**  |
| end dist (chars)         | 6.9437      | **6.9174**  | 8.1128      |
| abs(start dist) (chars)  | 28.6880     | 10.1010     | **9.9595**  |
| abs(middle dist) (chars) | 22.4206     | 8.8024      | **7.8590**  |
| abs(end dist) (chars)    | 21.9411     | 9.1916      | **8.7056**  |
| time (s)                 | 26079       | 23344       | 37179       |




## PRQA
### Medication

|              | BG med | CS med | EL med | EN med | ES med | PL med | RO med |
|--------------|--------|--------|--------|--------|--------|--------|--------|
| kept (%)     | 79.1   | 87.2   | 46.7   | 100.0  | 94.6   | 83.5   | 88.7   |
| PR (P@1)     | 78.64  | 81.83  | 66.88  | 86.39  | 82.58  | 80.82  | 82.58  |
| PR (P@2)     | 92.69  | 95.03  | 83.47  | 97.42  | 95.61  | 94.64  | 95.38  |
| PR (P@3)     | 96.47  | 98.14  | 90.16  | 99.20  | 98.10  | 98.01  | 98.30  |
| Oracle-QA F1 | 67.50  | 70.02  | 58.17  | 73.26  | 70.52  | 69.12  | 70.91  |
| Oracle-QA EM | 21.99  | 24.00  | 17.11  | 28.83  | 25.62  | 24.52  | 25.54  |
| PR-QA F1     | 58.74  | 62.33  | 46.75  | 68.19  | 63.66  | 61.10  | 63.87  |
| PR-QA EM     | 19.39  | 21.91  | 15.02  | 27.42  | 23.40  | 22.20  | 23.45  |


### Relations

|              | BG rel | CS rel | EL rel | EN rel | ES rel | PL rel | RO rel |
|--------------|--------|--------|--------|--------|--------|--------|--------|
| kept (%)     | 79.3   | 87.9   | 57.3   | 100.0  | 95.8   | 81.2   | 89.2   |
| PR (P@1)     | 85.83  | 88.38  | 79.91  | 96.41  | 89.72  | 84.84  | 88.85  |
| PR (P@2)     | 94.67  | 95.99  | 90.22  | 99.60  | 96.70  | 94.03  | 96.09  |
| PR (P@3)     | 96.99  | 97.77  | 93.72  | 99.92  | 98.08  | 96.29  | 97.75  |
| Oracle-QA F1 | 89.40  | 91.36  | 84.19  | 91.41  | 92.14  | 88.82  | 91.48  |
| Oracle-QA EM | 73.28  | 75.01  | 57.12  | 97.09  | 76.23  | 71.05  | 76.05  |
| PR-QA F1     | 81.42  | 84.05  | 73.41  | 94.82  | 86.60  | 80.24  | 84.73  |
| PR-QA EM     | 65.91  | 68.77  | 49.28  | 88.69  | 71.14  | 63.90  | 70.04  |


### RES-Q
evaluation on 53 res-q reports of the given language
ToDo - do detailed statistics - some question IDs are proably duplicated

#### Training using only reports of the given res-q language
| mBERT        | BG    | EL    | EN    | PL    | RO    |
|--------------|-------|-------|-------|-------|-------|
| Oracle-QA F1 | 83.11 | 86.23 | 72.62 | 67.53 | 80.85 |
| Oracle-QA EM | 75.84 | 75.49 | 59.52 | 47.91 | 68.09 |
| PR (P@1)     | 99.13 | 99.05 | 79.62 | 76.45 | 89.32 |
| PR (P@2)     | 98.50 | 97.79 | 92.72 | 87.61 | 94.38 |
| PR (P@3)     | 99.47 | 98.55 | 96.81 | 92.77 | 97.21 |
| PR-QA F1     | 80.61 | 83.89 | 64.33 | 58.75 | 74.90 |
| PR-QA EM     | 73.56 | 72.97 | 53.34 | 41.84 | 63.59 |
| # reports    | 283   | 287   | 244   | 133   | 293   |

| Ontotext XLM | BG    | EL    | EN    | PL    | RO    |
|--------------|-------|-------|-------|-------|-------|
| Oracle-QA F1 | 85.14 | 85.99 | 71.86 | 64.85 | 75.74 |
| Oracle-QA EM | 77.90 | 74.98 | 59.96 | 45.89 | 64.25 |
| PR (P@1)     | 97.77 | 98.70 | 78.00 | 76.08 | 81.32 |
| PR (P@2)     | 97.82 | 97.55 | 91.94 | 85.40 | 85.16 |
| PR (P@3)     | 98.74 | 98.59 | 96.30 | 90.64 | 87.08 |
| PR-QA F1     | 81.63 | 71.92 | 62.93 | 56.17 | 67.38 |
| PR-QA EM     | 74.47 | 82.63 | 52.74 | 39.64 | 57.43 |


#### Training using all res-q reports (all language)
| mBERT        | BG    | EL    | EN    | PL    | RO    |
|--------------|-------|-------|-------|-------|-------|
| Oracle-QA F1 | 84.85 | 86.28 | 73.05 | 69.56 | 81.46 |
| Oracle-QA EM | 77.58 | 75.85 | 60.01 | 50.54 | 68.23 |
| PR (P@1)     | 98.74 | 99.20 | 77.94 | 80.24 | 90.05 |
| PR (P@2)     | 98.50 | 98.36 | 91.99 | 88.88 | 94.57 |
| PR (P@3)     | 99.18 | 98.89 | 95.86 | 92.63 | 97.30 |
| PR-QA F1     | 82.12 | 83.90 | 64.34 | 61.52 | 75.72 |
| PR-QA EM     | 75.02 | 73.43 | 53.93 | 45.45 | 64.12 |


#### Training using all res-q reports (all language) + emrQA
| mBERT        | BG    | EL    | EN    | PL    | RO    |
|--------------|-------|-------|-------|-------|-------|
| Oracle-QA F1 | 83.89 | 84.39 | 69.42 | 68.00 | 80.34 |
| Oracle-QA EM | 75.52 | 74.19 | 56.16 | 47.91 | 66.74 |
| PR (P@1)     | 98.40 | 98.51 | 78.50 | 79.16 | 90.50 |
| PR (P@2)     | 98.06 | 97.67 | 91.77 | 87.48 | 94.11 |
| PR (P@3)     | 99.18 | 98.44 | 95.91 | 92.00 | 96.99 |
| PR-QA F1     | 81.11 | 81.80 | 61.47 | 60.14 | 75.04 |
| PR-QA EM     | 73.01 | 71.67 | 50.84 | 42.96 | 63.02 |

RES-Q PL two stages training - PL-emrQA -> PL-RES-Q
EM: 38.73   F1: 54.13

RES-Q PL two stages training - EN-emrQA -> PL-RES-Q
EM: 29.90   F1: 43.95
