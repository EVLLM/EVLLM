import os
import json
import time
import logging
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW

# PEFT imports
from peft import LoraConfig, TaskType, get_peft_model

# === Configuration ===
JSON_FILE = "/arf/scratch/egitim113/InstructionTuning/descriptive_instruction.json"
PRETRAINED_PROJECTOR = "/arf/scratch/egitim113/Uygar_cook/wheelie_20.pth"
OUTPUT_DIR = "./outputs"
BATCH_SIZE = 8
EPOCHS = 30
LR = 1e-4

# === EZSR Projector ===
class EZSRProjector(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=2048, output_dim=4096):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# === Instruction Tuning Dataset ===
class InstructionDataset(Dataset):
    def __init__(self, json_file, max_retries=3, retry_delay=5):
        with open(json_file, 'r') as f:
            raw_data = json.load(f)
        self.entries = []
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        for item in raw_data:
            path = item.get('path')
            if not os.path.exists(path):
                logging.warning(f"Missing feature file: {path}")
                continue
            self.entries.append({
                'path': path,
                'instruction': item.get('instruction', ''),
                'response': item.get('response', '')
            })
        if not self.entries:
            raise RuntimeError("No valid data entries found.")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        item = self.entries[idx]
        retries = 0
        while retries < self.max_retries:
            try:
                data = torch.load(item['path'])
                # If the file contains a dict of tensors, pick the first one
                if isinstance(data, dict):
                    tensor_values = [v for v in data.values() if isinstance(v, torch.Tensor)]
                    if not tensor_values:
                        raise ValueError(f"No tensor found in dict at {item['path']}")
                    feat = tensor_values[0]
                elif isinstance(data, torch.Tensor):
                    feat = data
                else:
                    raise ValueError(f"Unexpected data type {type(data)} in {item['path']}")
                feat = feat.to(torch.float32)
                return feat.squeeze(0), item['instruction'], item['response']
            except Exception as e:
                retries += 1
                logging.warning(f"Error loading {item['path']}: {e}. Retry {retries}/{self.retry_delay}")
                time.sleep(self.retry_delay)
        raise IOError(f"Failed to load feature file: {item['path']}")

# === Main Training Script ===
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    dataset = InstructionDataset(JSON_FILE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    for batch in dataloader:
    	print(batch)
    	        
if __name__ == '__main__':
    main()
