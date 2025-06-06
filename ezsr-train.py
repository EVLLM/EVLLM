"""
EZSR Projector Training Script with LLM Supervision
--------------------------------------------------

This script trains a lightweight projector model to align visual features with language using a frozen LLM
(e.g., Mistral-7B). It reads event-based features from a JSON file, uses a pre-trained MLP to project them into
LLM embedding space, and fine-tunes the projector via language modeling supervision.

Features:
---------
- Resilient dataset loader with retry logic for missing/corrupted .pt files
- Hugging Face authentication for model/tokenizer loading
- Auto checkpointing every N epochs
- Uses prompt + projected feature + label as the full input sequence for the LLM
- Supervised loss computed only on label tokens (prompt and features are masked)

Usage:
------
export HF_TOKEN=your_huggingface_token

python ezsr_train.py \
  --json_file /path/to/train.json \
  --pretrained_path /path/to/pretrained_projector.pth \
  --save_dir /path/to/save \
  --epochs 30 \
  --batch_size 8 \
  --prompt_text "Describe what is happening in the given event: "

Arguments:
----------
--json_file:         Path to JSON manifest containing "file_path" and "label"
--pretrained_path:   Optional checkpoint to resume from
--save_dir:          Directory to save checkpoints and final model
--hf_token:          Hugging Face access token (can also be set with HF_TOKEN env)
--epochs:            Number of training epochs (default: 30)
--save_every:        Save checkpoint every N epochs (default: 5)
--batch_size:        Mini-batch size (default: 8)
--model_name:        HF model name (default: mistralai/Mistral-7B-v0.1)
--prompt_text:       Prompt prepended to the feature sequence
"""

import os
import time
import json
import torch
import logging
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class EZSRProjector(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=2048, output_dim=4096):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class EventFeaturesDataset(Dataset):
    def __init__(self, json_file, max_retries=3, retry_delay=30):
        with open(json_file, "r") as f:
            raw_data = json.load(f)

        self.data = []
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        for item in raw_data:
            pt_path = item["file_path"].replace(".h5", ".pt")
            if not os.path.exists(pt_path):
                logging.warning(f"Missing .pt file: {pt_path}")
                continue
            item["file_path"] = pt_path
            self.data.append(item)

        if not self.data:
            logging.error("No valid .pt files found.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        attempts = 0
        while attempts < len(self.data):
            item = self.data[idx]
            for retry in range(self.max_retries):
                try:
                    feature = torch.load(item["file_path"]).to(torch.float32)
                    return feature, item["label"]
                except (OSError, IOError) as e:
                    logging.warning(f"Error reading {item['file_path']}: {e}. Retrying ({retry+1}/{self.max_retries})...")
                    time.sleep(self.retry_delay)

            logging.error(f"Skipping {item['file_path']} after {self.max_retries} retries.")
            idx = (idx + 1) % len(self.data)
            attempts += 1

        raise RuntimeError("All items failed to load in the dataset.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', type=str, required=True)
    parser.add_argument('--pretrained_path', type=str, default="/arf/scratch/egitim113/Uygar_cook/wheelie_20.pth")
    parser.add_argument('--save_dir', type=str, default="/arf/scratch/egitim113/Uygar_cook")
    parser.add_argument('--hf_token', type=str, default=os.getenv("HF_TOKEN"))
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--model_name', type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument('--prompt_text', type=str, default="Describe what is happening in the given event: ")
    args = parser.parse_args()

    basename = os.path.basename(args.json_file).replace(".json", "")
    final_save_path = os.path.join(args.save_dir, f"final-ezsr_{basename}_50.pth")
    primary_device = torch.device("cuda:0")

    # Load Dataset
    dataset = EventFeaturesDataset(args.json_file)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Load Projector
    projector = EZSRProjector().to(primary_device)
    if os.path.exists(args.pretrained_path):
        projector.load_state_dict(torch.load(args.pretrained_path, map_location=primary_device))
        logging.info(f"Loaded pretrained projector from {args.pretrained_path}")
    projector = nn.DataParallel(projector)
    projector.train()

    optimizer = optim.AdamW(projector.parameters(), lr=1e-4)

    # Load Frozen LLM
    login(token=args.hf_token)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto")
    llm.eval()
    for param in llm.parameters():
        param.requires_grad = False

    llm_device = next(llm.parameters()).device

    # Training Loop
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for batch_idx, (features, labels) in enumerate(dataloader):
            features = features.to(primary_device)

            prompt_tokens = tokenizer(args.prompt_text, return_tensors="pt", padding=True, truncation=True).to(llm_device)
            prompt_embeds = llm.get_input_embeddings()(prompt_tokens.input_ids)
            prompt_embeds = prompt_embeds.expand(features.size(0), -1, -1)

            projected_features = projector(features.squeeze(1)).unsqueeze(1).to(llm_device)
            prompt_len = prompt_tokens.input_ids.size(1)
            feature_positions = torch.full((features.size(0),), prompt_len, device=llm_device, dtype=torch.long)
            feature_pos_embeds = llm.get_input_embeddings()(feature_positions).unsqueeze(1)
            feature_embeds = projected_features + feature_pos_embeds

            gt_tokens = tokenizer(list(labels), return_tensors="pt", padding=True, truncation=True, max_length=24).to(llm_device)
            gt_embeds = llm.get_input_embeddings()(gt_tokens.input_ids)

            combined_embeds = torch.cat([prompt_embeds, feature_embeds, gt_embeds], dim=1)
            attention_mask = torch.ones(combined_embeds.shape[:-1], dtype=torch.long, device=llm_device)

            ignore_label = -100
            batch_size = features.size(0)
            prompt_and_feature = torch.full((batch_size, prompt_len + 1), ignore_label, device=llm_device, dtype=torch.long)
            labels_tensor = torch.cat([prompt_and_feature, gt_tokens.input_ids], dim=1)

            outputs = llm(inputs_embeds=combined_embeds, attention_mask=attention_mask, labels=labels_tensor)
            loss = outputs.loss
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            logging.info(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        logging.info(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.save_dir, f"ezsr_checkpoint_{basename}_epoch{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': projector.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            logging.info(f"Checkpoint saved at epoch {epoch+1}")

    torch.save(projector.module.state_dict(), final_save_path)
    logging.info(f"Final projector saved successfully at {final_save_path}")

if __name__ == "__main__":
    main()
