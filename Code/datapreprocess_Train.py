import csv
import json

# File Path
scitail_train_tsv = "H:/Repository/scitail/SciTailV1.1/tsv_format/scitail_1.0_train.tsv"
# output file path
output_json = "scitail_alpaca_train.json"

alpaca_data = []

with open(scitail_train_tsv, "r", encoding="utf-8") as tsv_file:
    reader = csv.reader(tsv_file, delimiter="\t")
    for row in reader:
        if len(row) != 3:
            continue

        premise, hypothesis, label = row

        # Convert into Alpaca format
        alpaca_data.append({
            "instruction": "Classify the relationship between the following premise and hypothesis as either 'entails' or 'neutral'.",
            "input": f"Premise: {premise}\nHypothesis: {hypothesis}",
            "output": label
        })

# saved
with open(output_json, "w", encoding="utf-8") as json_file:
    json.dump(alpaca_data, json_file, indent=4, ensure_ascii=False)

print(f"Alpaca-format dataset has been saved to {output_json}.")

