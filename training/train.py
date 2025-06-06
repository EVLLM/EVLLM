"""
This script trains a multi-channel CNN + Transformer encoder model (projector)
to align event-based visual features with natural language labels using a frozen LLM (e.g., Mistral-7B).

The projector encodes 5 input channels separately using identical CNN branches, combines them
with a CLS token and positional embedding, and passes them through a Transformer encoder.
The resulting CLS token is projected into the same embedding space as the LLM and used as a feature token.

Training is done by concatenating:
  - Prompt embedding
  - Projected feature embedding
  - Ground truth label tokens
Then passing the combined sequence to the LLM with appropriate masking for loss computation.

Usage:
------
export HF_TOKEN=your_huggingface_token

python train_projector.py \
  --json_path /path/to/data.json \
  --save_dir /path/to/save/dir \
  --batch_size 4 \
  --epochs 50 \
  --prompt "Describe what is happening in the event: "
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
import argparse

# Set logging to error level to reduce verbosity
logging.set_verbosity_error()

# Disable HDF5 file locking for cluster environments
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# === Model Definitions ===
class MultiChannelEncoderTransformerCLS(nn.Module):
    """Processes 5 input channels with shared CNN branches, encodes them with Transformer, and outputs a 4096-D vector."""
    def __init__(self, output_size=(20, 30), num_layers=2):
        super().__init__()
        self.output_size = output_size
        self.input_dim = output_size[0] * output_size[1]  # 600

        # Define identical CNN branches for 5 channels
        self.branch1 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size)
        )
        self.branch2 = self.branch3 = self.branch4 = self.branch5 = self.branch1

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.input_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 6, self.input_dim))  # for 5 tokens + 1 CLS

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.projector = nn.Sequential(
            nn.Linear(self.input_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, 4096)
        )

    def forward(self, x):
        batch_size = x.size(0)
        # Each branch encodes one channel
        out1 = self.branch1(x[:, 0:1, :, :]).view(batch_size, -1)
        out2 = self.branch2(x[:, 1:2, :, :]).view(batch_size, -1)
        out3 = self.branch3(x[:, 2:3, :, :]).view(batch_size, -1)
        out4 = self.branch4(x[:, 3:4, :, :]).view(batch_size, -1)
        out5 = self.branch5(x[:, 4:5, :, :]).view(batch_size, -1)

        tokens = torch.stack([out1, out2, out3, out4, out5], dim=1)  # (B, 5, 600)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        seq = torch.cat([cls_tokens, tokens], dim=1) + self.pos_embedding

        transformer_out = self.transformer(seq)
        cls_out = transformer_out[:, 0, :]  # CLS token output
        return self.projector(cls_out)

# === Dataset Definition ===
import json
import h5py
from typing import Optional, List

class EvRepSLFeatureDataset(Dataset):
    """Loads event representation features and labels from JSON manifest, supporting .pt and .hdf5 formats."""
    def __init__(self, json_path: str, hdf5_key: Optional[str] = None, allowed_hdf5_ext: List[str] = ('.h5', '.hdf5')):
        super().__init__()
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.hdf5_key = hdf5_key
        self.allowed_hdf5_ext = allowed_hdf5_ext

    def __len__(self):
        return len(self.data)

    def _load_pt(self, file_path: str) -> torch.Tensor:
        return torch.load(file_path, map_location="cpu").float()

    def _load_hdf5(self, file_path: str) -> torch.Tensor:
        with h5py.File(file_path, "r") as f:
            array = f[self.hdf5_key][()] if self.hdf5_key else f[next(iter(f.keys()))][()]
        return torch.from_numpy(array).float()

    def __getitem__(self, idx):
        item = self.data[idx]
        file_path, label = item["file_path"], item["label"]
        ext = os.path.splitext(file_path)[1].lower()
        feature = self._load_hdf5(file_path) if ext in self.allowed_hdf5_ext else self._load_pt(file_path)
        if feature.dim() >= 4 and feature.size(0) == 1:
            feature = feature.squeeze(0)
        return feature, label

# === Training Loop ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--checkpoint_name", type=str, default="devastator.pth")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--hf_token", type=str, default=os.getenv("HF_TOKEN"))
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--prompt", type=str, default="Describe me what is happening in the given event: ")
    args = parser.parse_args()

    # Set up dataset, model, and LLM
    primary_device = torch.device("cuda:0")
    dataset = EvRepSLFeatureDataset(args.json_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print("Data Loaded Successfully")

    projector = MultiChannelEncoderTransformerCLS().to(primary_device)
    os.makedirs(args.save_dir, exist_ok=True)
    latest_checkpoint = os.path.join(args.save_dir, args.checkpoint_name)
    if os.path.exists(latest_checkpoint):
        projector.load_state_dict(torch.load(latest_checkpoint))
        print("Loaded existing projector weights from", latest_checkpoint)
    projector = torch.nn.DataParallel(projector)
    projector.train()

    from huggingface_hub import login
    login(token=args.hf_token)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto")
    llm.eval()
    for param in llm.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW(projector.parameters(), lr=1e-4)
    llm_device = next(llm.parameters()).device
    torch.autograd.set_detect_anomaly(True)

    # === Training Epochs ===
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for batch_idx, (features, labels) in enumerate(dataloader):
            features = torch.clamp(features.to(primary_device), min=-1e4, max=1e4)

            # Prompt tokenization and embedding
            prompt_tokens = tokenizer(args.prompt, return_tensors="pt", padding=True, truncation=True).to(llm_device)
            prompt_embeds = llm.get_input_embeddings()(prompt_tokens.input_ids).expand(features.size(0), -1, -1)

            # Project features into LLM space and align positionally
            projected_features = projector(features).unsqueeze(1).to(llm_device)
            if torch.isnan(projected_features).any() or torch.isinf(projected_features).any():
                print("NaN or Inf in projected features"); exit()

            prompt_length = prompt_tokens.input_ids.size(1)
            feature_positions = torch.full((features.size(0), 1), prompt_length, device=llm_device)
            feature_pos_embeds = llm.get_input_embeddings().weight[feature_positions]
            feature_embeds = projected_features + feature_pos_embeds

            # Tokenize ground truth labels
            gt_tokens = tokenizer(list(labels), return_tensors="pt", padding=True, truncation=True).to(llm_device)
            gt_embeds = llm.get_input_embeddings()(gt_tokens.input_ids)

            # Combine all inputs and create attention mask
            combined_embeds = torch.cat([prompt_embeds, feature_embeds, gt_embeds], dim=1)
            attention_mask = torch.ones(combined_embeds.shape[:-1], dtype=torch.long, device=llm_device)

            # Prepare labels: ignore prompt+feature positions, predict only labels
            ignore_label = -100
            prompt_and_feature = torch.full((features.size(0), prompt_length + 1), ignore_label, device=llm_device)
            labels_tensor = torch.cat([prompt_and_feature, gt_tokens.input_ids], dim=1)

            # Forward and backward pass
            outputs = llm(inputs_embeds=combined_embeds, attention_mask=attention_mask, labels=labels_tensor)
            loss = outputs.loss
            epoch_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(projector.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

        # Save checkpoint
        checkpoint_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save(projector.module.state_dict(), checkpoint_path)
        print(f"Projector checkpoint saved at epoch {epoch+1}")

    # Save final model
    final_checkpoint = os.path.join(args.save_dir, "final_projector.pth")
    torch.save(projector.module.state_dict(), final_checkpoint)
    print("Projector saved successfully at", final_checkpoint)

if __name__ == "__main__":
    main()
