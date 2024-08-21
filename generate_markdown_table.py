import json

file_path = './scripts/results/aligner_results.json'

with open(file_path, 'r') as file:
    data = json.load(file)


# Languages available in the data
languages = ["bg", "cs", "el", "es", "pl", "ro"]
metrics = [
    "f1", "exact_match", "exact_submatch", "f1_span",
    "precision_span", "recall_span", "start_distance", 
    "middle_distance", "end_distance", "absolute_start_distance", 
    "absolute_middle_distance", "absolute_end_distance", "overall_time"
]
# Header of the Markdown table
header = "|   FastAlign          | " + " | ".join([lang.upper() for lang in languages]) + " |\n"
header += "|-------------------| " + " | ".join(["---------" for _ in languages]) + " |\n"

# Initialize table content
table_content = ""

# Populate the table content with metrics for each language
for metric in metrics:
    row = f"| {metric.replace('_', ' ')} | "
    for lang in languages:
        # Fetch the metric for each language, if available
        try:
            value = data["straightforward MadLad + splited paragraph translations"][lang]["relations"]["FastAlign"].get(metric, "")
            if isinstance(value, float):
                row += f"{value:.2f} | "
            else:
                row += f"{value} | "
        except KeyError:
            row += "         | "
    table_content += row + "\n"

# Combine header and content to form the final Markdown table
markdown_table = header + table_content

# Print the Markdown table
print(markdown_table)