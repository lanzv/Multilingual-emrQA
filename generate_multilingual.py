import os
import json
from collections import defaultdict

# Path to the directory containing the JSON files
directory_path = "jsons/multilingual/"



def sort_dict_alphabetically(d):
    sorted_dict = {}
    for key, value in sorted(d.items()):
        if isinstance(value, dict):
            sorted_dict[key] = sort_dict_alphabetically(value)  # Recursively sort inner dictionary
        elif isinstance(value, list):
            sorted_dict[key] = sorted(value)  # Sort list alphabetically
        else:
            sorted_dict[key] = value
    return sorted_dict

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
        language = file_path.split("_")[1]
        if language == "EN" and file_path.split("_")[2] == "FULL": 
            language = "EN_FULL"
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            for model in data:
                if "BG" in data[model]:
                    data[model][language] = data[model].pop("BG")
            merge_dicts(merged_data, data)

# Convert defaultdict to a regular dict for pretty-printing
merged_data = dict(merged_data)

new_data = {}

for model in merged_data:
    for lang in merged_data[model]:
        for subset in merged_data[model][lang]:
            for seed in merged_data[model][lang][subset]:
                for ratio in merged_data[model][lang][subset][seed]:
                    if not model in new_data:
                        new_data[model] = {}
                    if not lang in new_data[model]:
                        new_data[model][lang] = {}
                    if not subset in new_data[model][lang]:
                        new_data[model][lang][subset] = {}
                    new_data[model][lang][subset][seed] = merged_data[model][lang][subset][seed]

new_data = sort_dict_alphabetically(new_data)

print(json.dumps(new_data, indent=4))


import numpy as np
medication_table = {}
relations_table = {}
for model in new_data:
    medication_table[model] = {}
    relations_table[model] = {}
    for lang in new_data[model]:
        for subset in new_data[model][lang]:
            ems = []
            f1s = []
            if len(new_data[model][lang][subset]) != 3:
                print("{} {} {} has only {} seeds".format(model, lang, subset, len(new_data[model][lang][subset])))
            for seed in new_data[model][lang][subset]:
                if not "QA" in new_data[model][lang][subset][seed]:
                    continue
                ems.append(new_data[model][lang][subset][seed]["QA"]["exact_match"])
                f1s.append(new_data[model][lang][subset][seed]["QA"]["f1"])
            if subset == "medication":
                medication_table[model][lang] = "{} / {}".format(round(np.mean(ems), 2), round(np.mean(f1s), 2))
            if subset == "relations":
                relations_table[model][lang] = "{} / {}".format(round(np.mean(ems), 2), round(np.mean(f1s), 2))
def print_table(dict_subset, cell_length=13):
    header = "|{}|".format("EM/F1".center(cell_length))
    line = "|{}|".format("-"*cell_length)
    for model in dict_subset:
        for lang in dict_subset[model]:
            header += "{}|".format(lang.center(cell_length))
            line += "{}|".format("-"*cell_length)
        break
    table = header + "\n" + line + "\n"
    
    for model in dict_subset:
        table += "|{}|".format(model.center(cell_length))
        for lang in dict_subset[model]:
            table += "{}|".format(dict_subset[model][lang].center(cell_length))
        table += "\n"
    print(table)
print("MEDICATION")
print_table(medication_table)
print("RELATIONS")
print_table(relations_table)
