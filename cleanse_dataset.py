# Purpose: Load, clean and save the dataset
import re
import os
from datasets import load_dataset, concatenate_datasets

# CONFIGURATION
DATASET_NAME = "neuralchemy/Prompt-injection-dataset"
DATASET_CONFIG = "full"
OUTPUT_DIR = "datasets/classifier"
TEST_SIZE = 1500
VAL_SIZE = 1500

# Load dataset
print("Loading dataset...")
ds = load_dataset(DATASET_NAME, DATASET_CONFIG)
print(f"Loaded → Train: {len(ds['train'])} | Val: {len(ds['validation'])} | Test: {len(ds['test'])}")

# Clean text
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[\u200b\u200c\u200d\ufeff\u00ad]', '', text)  # Remove invisible chars
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text

# Filter valid samples
def is_valid(example):
    text = example.get("text", "")
    return (len(text) >= 10 and 
            len(text) <= 2000 and 
            example.get("label") in [0, 1])

# Clean and filter
ds = ds.map(lambda x: {"text": clean_text(x["text"]), "label": x["label"]})
ds = ds.filter(is_valid)

# Combine all splits
all_data = concatenate_datasets([ds['train'], ds['validation'], ds['test']])

# Split: test(1500) + val(1500) + train(remaining)
all_data = all_data.shuffle(seed=42)
test_data = all_data.select(range(TEST_SIZE))
val_data = all_data.select(range(TEST_SIZE, TEST_SIZE + VAL_SIZE))
train_data = all_data.select(range(TEST_SIZE + VAL_SIZE, len(all_data)))

# Save
os.makedirs(OUTPUT_DIR, exist_ok=True)
train_data.to_json(f"{OUTPUT_DIR}/train.jsonl")
val_data.to_json(f"{OUTPUT_DIR}/val.jsonl")
test_data.to_json(f"{OUTPUT_DIR}/test.jsonl")

print(f"\n[DONE] Saved to {OUTPUT_DIR}/ 1. train.jsonl → {len(train_data)} samples. 2. val.jsonl   → {len(val_data)} samples 3.test.jsonl  → {len(test_data)} samples ")