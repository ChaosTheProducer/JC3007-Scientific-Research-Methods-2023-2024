import csv

# File Path
predictions_file = "H:/Repository/scitail/results/FineTune-Lora/lora-results.csv"
# predictions_file = "H:/Repository/scitail/results/LlamaNonFineTune/LlamaNonFineTune.csv"
ground_truth_file = "H:/Repository/scitail/SciTailV1.1/tsv_format/scitail_1.0_test.tsv"

# Mapping
def map_prediction_to_full_label(prediction):
    if prediction == "E":
        return "entails"
    elif prediction == "N":
        return "neutral"
    return "neutral"  # Default policy

# Extract correct labels.
def extract_labels_from_tsv(file_path):
    labels = []
    with open(file_path, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter="\t")
        for row in reader:
            labels.append(row[2].strip().lower())
    return labels

# Load predict results
def load_predictions(file_path):
    predictions = []
    with open(file_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            predictions.append(map_prediction_to_full_label(row["Prediction"])) 
    return predictions

# Calculate accuracy
def calculate_accuracy(predictions, ground_truth):
    correct = sum(p == g for p, g in zip(predictions, ground_truth))
    total = len(ground_truth)
    accuracy = correct / total if total > 0 else 0
    return accuracy

# Load Data
predictions = load_predictions(predictions_file)
ground_truth = extract_labels_from_tsv(ground_truth_file)

accuracy = calculate_accuracy(predictions, ground_truth)
print(f"Accuracy: {accuracy:.2%}")
