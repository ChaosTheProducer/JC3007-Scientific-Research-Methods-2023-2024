import wandb
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
from peft import PrefixTuningConfig, get_peft_model
import json
from transformers import DataCollatorForSeq2Seq
import numpy as np
from transformers import BitsAndBytesConfig

wandb.init(
    project="Llama-3.1-8B-Instruct-SciTail-Prefix-Tuning",
    name="Llama-3.1-8B-SciTail-Prefix",
    config={
        "model_name": "Meta-Llama-3.1-8B-Instruct",
        "method": "Prefix Tuning",
        "num_virtual_tokens": 16,
        "learning_rate": 5e-5,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
        "num_train_epochs": 3,
    },
)

model_path = "H:/Repository/text-generation-webui/models/Meta-Llama-3.1-8B-Instruct"
data_path = "H:/Repository/scitail/ProcessedData/scitail_alpaca_train.json"  
output_dir = "H:/Repository/Llama-3.1-8B/Llama-fine-tuned/SciTail-Prefix"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_compute_dtype=torch.float16,  
    bnb_4bit_quant_type="nf4"  
)

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_path, 
                                             device_map="auto", 
                                             torch_dtype=torch.float16, 
                                             quantization_config=bnb_config
                                             )

prefix_config = PrefixTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=16,
    token_dim=model.config.hidden_size,
    prefix_projection=True,
    inference_mode=False,
)


model = get_peft_model(model, prefix_config)

with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

class SciTailDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data  
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        instruction = sample["instruction"]
        input_text = sample["input"]
        output_text = sample["output"]

        prompt = f"{instruction}\n{input_text}\nAnswer:"

        tokenized_input = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        tokenized_output = self.tokenizer(
            output_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = tokenized_input["input_ids"].squeeze() 
        labels = tokenized_output["input_ids"].squeeze()
        labels = torch.where(labels != self.tokenizer.pad_token_id, labels, torch.tensor(-100))

        attention_mask = tokenized_input["attention_mask"].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

train_dataset = SciTailDataset(data, tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

print(len(train_dataloader))  

training_args = TrainingArguments(
    output_dir=output_dir,
    run_name="SciTail-Prefix-Run",  
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=5e-5,
    num_train_epochs=3,
    logging_dir=f"{output_dir}/logs",
    save_steps=500,
    logging_steps=10,
    save_total_limit=2,
    bf16=True,
    report_to="wandb",  
)

class OptimizedDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    def __call__(self, features):
        input_ids = np.array([f["input_ids"] for f in features])
        attention_mask = np.array([f["attention_mask"] for f in features])
        labels = np.array([f["labels"] for f in features])

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }


data_collator = OptimizedDataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

# print(len(tokenized_datasets))  

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  
    tokenizer=tokenizer,
    data_collator=data_collator,  
    #logging_steps=50,
    #report_to="wandb",  
)

trainer.train()

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

wandb.finish()

print("Prefix Tuning finished and saved！")
