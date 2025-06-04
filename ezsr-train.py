import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import logging

class EZSRProjector(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=2048, output_dim=4096):
        super(EZSRProjector, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

import time

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
        start_idx = idx
        attempts = 0
        while attempts < len(self.data):
            item = self.data[idx]
            retries = 0
            while retries < self.max_retries:
                try:
                    feature = torch.load(item["file_path"]).to(torch.float32)
                    label = item["label"]
                    return feature, label
                except (OSError, IOError) as e:
                    retries += 1
                    logging.warning(
                        f"Error reading {item['file_path']}: {e}. Retrying in {self.retry_delay}s ({retries}/{self.max_retries})"
                    )
                    time.sleep(self.retry_delay)

            # Skip to next sample if max_retries failed
            logging.error(f"Skipping {item['file_path']} after {self.max_retries} retries.")
            idx = (idx + 1) % len(self.data)
            attempts += 1

        raise RuntimeError("All items failed to load in the dataset.")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', type=str, required=True)
    args = parser.parse_args()

    json_file = args.json_file
    basename = os.path.basename(json_file).replace(".json", "")
    final_save_path = f"/arf/scratch/egitim113/Uygar_cook/final-ezsr_{basename}_50.pth"
    primary_device = torch.device("cuda:0")

    # === Load Dataset ===
    dataset = EventFeaturesDataset(json_file)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # === Load Projector ===
    projector = EZSRProjector().to(primary_device)
    pretrained_path = "/arf/scratch/egitim113/Uygar_cook/wheelie_20.pth"
    if os.path.exists(pretrained_path):
        projector.load_state_dict(torch.load(pretrained_path, map_location=primary_device))
        print(f"Loaded pretrained projector from {pretrained_path}")
    projector = torch.nn.DataParallel(projector)
    projector.train()

    optimizer = optim.AdamW(projector.parameters(), lr=1e-4)

    # === Load Frozen LLM ===
    login(token="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")
    model_name = "mistralai/Mistral-7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    llm.eval()
    for param in llm.parameters():
        param.requires_grad = False

    llm_device = next(llm.parameters()).device
    prompt_text = "Describe what is happening in the given event: "
    num_epochs = 30
    save_every = 5

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (features, labels) in enumerate(dataloader):
            features = features.to(primary_device)

            prompt_tokens = tokenizer(prompt_text, return_tensors="pt", padding=True, truncation=True).to(llm_device)
            prompt_embeds = llm.get_input_embeddings()(prompt_tokens.input_ids)
            prompt_embeds = prompt_embeds.expand(features.size(0), -1, -1)

            projected_features = projector(features.squeeze(1)).unsqueeze(1).to(llm_device)
            prompt_length = prompt_tokens.input_ids.size(1)
            feature_positions = torch.full((features.size(0),), prompt_length, device=llm_device, dtype=torch.long)
            feature_pos_embeds = llm.get_input_embeddings()(feature_positions).unsqueeze(1)
            feature_embeds = projected_features + feature_pos_embeds

            gt_tokens = tokenizer(list(labels), return_tensors="pt", padding=True, truncation=True, max_length=24).to(llm_device)
            gt_embeds = llm.get_input_embeddings()(gt_tokens.input_ids)

            combined_embeds = torch.cat([prompt_embeds, feature_embeds, gt_embeds], dim=1)
            attention_mask = torch.ones(combined_embeds.shape[:-1], dtype=torch.long, device=llm_device)

            ignore_label = -100
            batch_size = features.size(0)
            prompt_and_feature = torch.full((batch_size, prompt_length + 1), ignore_label, device=llm_device, dtype=torch.long)
            labels_tensor = torch.cat([prompt_and_feature, gt_tokens.input_ids], dim=1)

            outputs = llm(inputs_embeds=combined_embeds, attention_mask=attention_mask, labels=labels_tensor)

            loss = outputs.loss
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

        if (epoch + 1) % save_every == 0:
            checkpoint_path = f"/arf/scratch/egitim113/Uygar_cook/ezsr_checkpoint_{basename}_epoch{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': projector.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}")

    torch.save(projector.module.state_dict(), final_save_path)
    print("Final projector saved successfully at", final_save_path)

if __name__ == "__main__":
    main()