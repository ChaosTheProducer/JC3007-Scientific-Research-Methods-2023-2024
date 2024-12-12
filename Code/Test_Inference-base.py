import csv
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
import torch

# Load Llama and Tokenizer
model_path = "H:/Repository/text-generation-webui/models/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

# Add padding token for tokenizer
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 4 bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_compute_dtype=torch.float16,  
    bnb_4bit_quant_type="nf4" 
)

model = AutoModelForCausalLM.from_pretrained(model_path, 
                                             torch_dtype="auto", 
                                             device_map="auto",
                                             # quantization_config=bnb_config
                                             ) 
model.config.pad_token_id = tokenizer.pad_token_id
model.generation_config.pad_token_id = tokenizer.pad_token_id  # pad_token

scitail_test_file = "H:/Repository/scitail/ProcessedData/scitail_test.txt"
output_csv_file = "scitail_predictions_leaderboard_NonFineTune.csv"

def map_to_label(prediction):
    if "entails" in prediction.lower():
        return "E"
    elif "neutral" in prediction.lower():
        return "N"
    else:
        return "N"  

def generate_prediction(premise, hypothesis, tokenizer, model, max_length=512, temperature=0.7):
    prompt = (
        f"Based on the following premise and hypothesis, classify the relationship as 'entails' or 'neutral'. Give the answers only, think step by step.\n"
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n"
        f"Answer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", max_length=max_length, truncation=True, padding=True)
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        temperature=temperature,
        top_p=0.9,
        num_return_sequences=1,
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = result.split("Answer:")[-1].strip()
    return map_to_label(answer)

with open(scitail_test_file, "r", encoding="utf-8") as file:
    lines = file.readlines()

with open(output_csv_file, mode="w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Premise", "Hypothesis", "Prediction"])  

    for i in tqdm(range(0, len(lines), 2)): 
        premise_line = lines[i].strip()
        hypothesis_line = lines[i + 1].strip()

        if premise_line.startswith("Premise:") and hypothesis_line.startswith("Hypothesis:"):
            premise = premise_line.replace("Premise: ", "")
            hypothesis = hypothesis_line.replace("Hypothesis: ", "")
            prediction = generate_prediction(premise, hypothesis, tokenizer, model)
            writer.writerow([premise, hypothesis, prediction])  

print(f"Predictions saved to {output_csv_file}")
