#!/bin/sh
#SBATCH -J engcz
#SBATCH -o scripts/slurm_outputs/engcz.out
#SBATCH -p cpu-troja
#SBATCH --mem-per-cpu=50G



python3 - <<END
from src.utils import tokenize

def process_subtitle_file(input_file1, input_file2, output_file):
    czech_sentences = []
    english_sentences = []
    with open(input_file1, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            if i%10000==0:
                print("train: {}".format(i), end="\r")
            parts = line.strip().split('\t')
            if len(parts) == 6:  # Ensure that we have all parts
                czech_sentence = parts[4].strip()
                english_sentence = parts[5].strip()
                
                czech_sentences.append(' '.join(tokenize(czech_sentence, language="czech", warnings=False)[0]))
                english_sentences.append((' '.join(tokenize(english_sentence, language="english", warnings=False)[0])))
                
    with open(input_file2, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            if i%10000==0:
                print("test: {}".format(i), end="\r")
            parts = line.strip().split('\t')
            if len(parts) == 6:  # Ensure that we have all parts
                czech_sentence = parts[4].strip()
                english_sentence = parts[5].strip()
                
                czech_sentences.append(' '.join(tokenize(czech_sentence, language="czech", warnings=False)[0]))
                english_sentences.append((' '.join(tokenize(english_sentence, language="english", warnings=False)[0])))

    with open(output_file, 'w', encoding='utf-8') as file:
        for czech, english in zip(czech_sentences, english_sentences):
            file.write(f"{english} ||| {czech}\n")

# Process the files
process_subtitle_file('../datasets/czeng20/czeng20-train', '../datasets/czeng20/czeng20-test', '../datasets/czeng20/engcz_fastalign.txt')
END
