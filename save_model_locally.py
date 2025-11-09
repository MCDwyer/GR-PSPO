from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import json
import pickle 
import os

from transformers import AutoTokenizer

model_paths = ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B"]

for model_path in model_paths:
# # Save model locally
    model_name = model_path.split("/")[-1]
    if os.path.exists(f"./cached_files/{model_name}"):
        continue
    print(f"Saving {model_path} locally...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # Save to local directory
    model.save_pretrained(f"./cached_files/{model_name}")
    tokenizer.save_pretrained(f"./cached_files/{model_name}")

dataset_paths = ["gsm8k", "HuggingFaceH4/MATH-500", "EleutherAI/asdiv", "ChilleD/SVAMP",]

for dataset_path in dataset_paths:
    dataset_name = dataset_path.split("/")[-1]

    if os.path.exists(f"./cached_files/{dataset_name}"):
        continue

    try:
        dataset = load_dataset(dataset_path, "main")
    except ValueError:
        dataset = load_dataset(dataset_path)

    dataset.save_to_disk(f"./cached_files/{dataset_name}")
