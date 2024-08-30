import os
import json
from collections import defaultdict

# Path to the directory containing the JSON files
directory_path = "jsons/"

# Function to merge dictionaries
def merge_dicts(a, b):
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_dicts(a[key], b[key])
            else:
                a[key] = b[key]  # Overwrite if it's not a dict
        else:
            a[key] = b[key]

# Initialize an empty defaultdict for merging
merged_data = defaultdict(dict)

# Loop through each JSON file in the directory
for filename in os.listdir(directory_path):
    if filename.endswith(".json"):
        file_path = os.path.join(directory_path, filename)
        
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            merge_dicts(merged_data, data)

# Convert defaultdict to a regular dict for pretty-printing
merged_data = dict(merged_data)


# Print the merged JSON to standard output
#print(json.dumps(merged_data, indent=4))

a = {
    "./data/translation_aligners/Awesome/medication_pl.json": ("pl", "Awesome", "medication"),
    "./data/translation_aligners/Awesome/medication_cs.json": ("cs", "Awesome", "medication"),
    "./data/translation_aligners/FastAlign/medication_el.json": ("el", "FastAlign", "medication"),
    "./data/translation_aligners/FastAlign/medication_ro.json": ("ro", "FastAlign", "medication"),
    "./data/translation_aligners/FastAlign/relations_pl.json": ("pl", "FastAlign", "relations"),
    "./data/translation_aligners/FastAlign/relations_cs.json": ("cs", "FastAlign", "relations"),
    "./data/translation_aligners/Awesome/relations_ro.json": ("ro", "Awesome", "relations"),
    "../datasets/emrQA/medication_en.json": ("en", "original", "medication"),
    "./data/translation_aligners/Awesome/medication_ro.json": ("ro", "Awesome", "medication"),
    "./data/translation_aligners/Awesome/medication_es.json": ("es", "Awesome", "medication"),
    "./data/translation_aligners/Awesome/relations_bg.json": ("bg", "Awesome", "relations"),
    "../datasets/emrQA/relations_en.json": ("en", "original", "relations"),
    "./data/translation_aligners/Awesome/relations_es.json": ("es", "Awesome", "relations"),
    "./data/translation_aligners/FastAlign/relations_bg.json": ("bg", "FastAlign", "relations"),
    "./data/translation_aligners/Awesome/medication_bg.json": ("bg", "Awesome", "medication"),
    "./data/translation_aligners/Awesome/relations_pl.json": ("pl", "Awesome", "relations"),
    "./data/translation_aligners/Awesome/relations_cs.json": ("cs", "Awesome", "relations"),
    "./data/translation_aligners/FastAlign/relations_el.json": ("el", "FastAlign", "relations")
}
new_data = {"mBERT": {}}

for experiment in merged_data["mBERT"]:
    lang, align, subset = a[experiment]
    for seed in merged_data["mBERT"][experiment]:
        if not lang in new_data["mBERT"]:
            new_data["mBERT"][lang] = {}
        if not subset in new_data["mBERT"][lang]:
            new_data["mBERT"][lang][subset] = {}
        if not align in new_data["mBERT"][lang][subset]:
            new_data["mBERT"][lang][subset][align] = {}
        new_data["mBERT"][lang][subset][align][seed] = merged_data["mBERT"][experiment][seed]

print(json.dumps(new_data, indent=4))


import numpy as np
for experiment in merged_data["mBERT"]:
    ems = []
    f1s = []
    for seed in merged_data["mBERT"][experiment]:
        if not "QA" in merged_data["mBERT"][experiment][seed]:
            continue
        ems.append(merged_data["mBERT"][experiment][seed]["QA"]["exact_match"])
        f1s.append(merged_data["mBERT"][experiment][seed]["QA"]["f1"])
    print("{}: em {}, f1 {}".format(experiment, round(np.mean(ems), 2), round(np.mean(f1s), 2)))