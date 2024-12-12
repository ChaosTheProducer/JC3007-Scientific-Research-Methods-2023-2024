# JC3007: Scientific Research Methods (2023-2024)

## Project Overview
This project investigates and compares two prominent Parameter-Efficient Fine-Tuning (PEFT) methods, **LoRA (Low-Rank Adaptation)** and **Prefix Tuning**, in adapting large language models (LLMs) to specific tasks. Using the “SciTail” dataset as a benchmark, we explore their efficiency and performance across multiple dimensions, such as inference accuracy, training time, resource utilization, and convergence rate. This research aims to provide practical insights for selecting fine-tuning strategies in resource-constrained environments.

## Repository Structure
```plaintext
├── Code/                  # Scripts for training, evaluation, and preprocessing
├── ProcessedData/         # Processed datasets
├── results/               # Example predictions are included here
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
```

## Installation
To run this project, follow the steps below:

1. Clone this repository:
   ```bash
   git clone https://github.com/<your_username>/JC3007-Scientific-Research-Methods-2023-2024.git
   cd JC3007-Scientific-Research-Methods-2023-2024
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure your environment meets the hardware requirements:
   - Python 3.10
   - NVIDIA RTX 4090 GPU (or equivalent)

4. Download the SciTail dataset and place it in the `data/` directory:
   - [SciTail Dataset](https://allenai.org/data/scitail)

## Usage

### Data Preprocessing
Prepare the Training dataset for fine-tuning by running:
```bash
Python Code/datapreprocess_Train.py
```
And for the testing data:
```bash
Python Code/datapreprocess_Test.py
```

### Training
Train Prefix Tuning models:
- **Prefix Tuning**
  ```bash
  python code/train_prefix.py
  ```
The Lora is trained via [LLamA-Factory](https://github.com/hiyouga/LLaMA-Factory) 
Please reference for my report for more training configs.

### Inference
Run the fine-tuned models on the SciTail test set:
```bash
python Code/Test_Inference-finetune.py
```
Note that you need to change the model and tokenizer path to your own.

### Examples
The example output can be seend in the results directory.

## Results

### Performance Metrics
| Method          | Accuracy (%) | Training Time (hours) | GPU Memory (GB) | Power Usage (W) |
|-----------------|--------------|-----------------------|------------------|-----------------|
| LoRA            | 85.94        | 4.5                   | 18.4 (No quantization) | 250             |
| Prefix Tuning   | 65.25        | 6.2                   | 15.7 (4-bit quantized) | 310             |

More results are shown in the report.

## Acknowledgments
This project leverages the following libraries and tools:
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Hugging Face Datasets](https://github.com/huggingface/datasets)
- [WandB](https://github.com/wandb/wandb)

The SciTail dataset was sourced from [AllenAI](https://allenai.org/data/scitail).

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact
If you have questions or feedback, feel free to contact me via GitHub or email.
