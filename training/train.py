# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, logging

# Set logging to error level to reduce verbosity
logging.set_verbosity_error()

# --- Model Definitions ---
os.environ["HDF5_USE_FILE_LOCKING"]="FALSE"

class MultiChannelEncoderTransformerCLS(nn.Module):
    def __init__(self, output_size=(20, 30), num_layers=2):
        super().__init__()
        self.output_size = output_size
        # Each branch output is flattened: e.g., 20 * 30 = 600
        self.input_dim = output_size[0] * output_size[1]  # 600

        # Define 5 CNN branches (one per channel)
        self.branch1 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size)
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size)
        )

        # Learnable CLS token with shape: (1, 1, 600)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.input_dim))
        # Learnable positional embedding for a sequence of length 6 (1 CLS + 5 tokens)
        self.pos_embedding = nn.Parameter(torch.randn(1, 6, self.input_dim))

        # Transformer encoder with d_model set to 600
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.input_dim,
            nhead=8,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classifier: projects the CLS token output (600-dim) to 4096 dimensions
        self.projector = nn.Sequential(
            nn.Linear(self.input_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, 4096)
        )

    def forward(self, x):
        batch_size = x.size(0)

        # Process each channel individually: each branch returns (N, 600)
        out1 = self.branch1(x[:, 0:1, :, :]).view(batch_size, -1)
        out2 = self.branch2(x[:, 1:2, :, :]).view(batch_size, -1)
        out3 = self.branch3(x[:, 2:3, :, :]).view(batch_size, -1)
        out4 = self.branch4(x[:, 3:4, :, :]).view(batch_size, -1)
        out5 = self.branch5(x[:, 4:5, :, :]).view(batch_size, -1)

        # Stack tokens from CNN branches: (N, 5, 600)
        tokens = torch.stack([out1, out2, out3, out4, out5], dim=1)

        # Expand CLS token to batch size: (N, 1, 600)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # Prepend the CLS token to the token sequence: (N, 6, 600)
        seq = torch.cat([cls_tokens, tokens], dim=1)

        # Add positional embedding
        seq = seq + self.pos_embedding

        # Pass through the transformer encoder
        transformer_out = self.transformer(seq)  # (N, 6, 600)

        # Use the CLS token's output (first token) for classification
        cls_out = transformer_out[:, 0, :]  # (N, 600)
        logits = self.projector(cls_out)
        return logits

# --- Dataset Definition ---

import os
import json
import torch
from torch.utils.data import Dataset
import h5py
from typing import Optional, List

class EvRepSLFeatureDataset(Dataset):
    def __init__(self,
                 json_path: str,
                 hdf5_key: Optional[str] = None,
                 allowed_hdf5_ext: List[str] = ('.h5', '.hdf5')):
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
            if self.hdf5_key is not None:
                array = f[self.hdf5_key][()]
            else:
                # take the first dataset in the file
                first_key = next(iter(f.keys()))
                array = f[first_key][()]
        return torch.from_numpy(array).float()

    def __getitem__(self, idx):
        item = self.data[idx]
        file_path = item["file_path"]
        label = item["label"]

        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        if ext in self.allowed_hdf5_ext:
            feature = self._load_hdf5(file_path)
        else:
            # fall back to PyTorch tensor loading
            feature = self._load_pt(file_path)

        # Squeeze leading singleton batch dimension if present (1, H, W, …)
        if feature.dim() >= 4 and feature.size(0) == 1:
            feature = feature.squeeze(0)

        return feature, label

# --- Training Loop and Model Distribution ---

def main():
    # Set the primary device (GPU 0) for the projector model
    primary_device = torch.device("cuda:0")
    
    # Dataset and DataLoader setup
    dataset = EvRepSLFeatureDataset("/arf/scratch/egitim113/training/shards-evrepsl/shard_evrepsl_2.json")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    print("Data Loaded Successfully")

    # Initialize the projector model and wrap it with DataParallel for multi-GPU support
    projector = MultiChannelEncoderTransformerCLS().to(primary_device)
    # Load existing projector weights if they exist
    save_dir = "/arf/scratch/egitim113/training"
    os.makedirs(save_dir, exist_ok=True)
    latest_checkpoint = os.path.join(save_dir, "devastator.pth")
    if os.path.exists(latest_checkpoint):
        projector.load_state_dict(torch.load(latest_checkpoint))
        print("Loaded existing projector weights from", latest_checkpoint)
    projector = torch.nn.DataParallel(projector)
    projector.train()

    # Login to HuggingFace and prepare tokenizer for the LLM
    from huggingface_hub import login
    login(token="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")
    model_name = "mistralai/Mistral-7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load the LLM with device_map="auto" to distribute across available GPUs
    llm = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    llm.eval()
    for param in llm.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW(projector.parameters(), lr=1e-4)
    prompt_text = "Describe me what is happening in the given event: "

    num_epochs = 50

    # Determine the device for LLM embeddings (usually GPU0)
    llm_device = next(llm.parameters()).device
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (features, labels) in enumerate(dataloader):
            # Move features to the primary device; DataParallel will distribute automatically
            features = features.to(primary_device)
            features = torch.clamp(features, min=-1e4, max=1e4)
            prompt_tokens = tokenizer(prompt_text, return_tensors="pt", padding=True, truncation=True).to(llm_device)
            prompt_embeds = llm.get_input_embeddings()(prompt_tokens.input_ids)
            prompt_embeds = prompt_embeds.expand(features.size(0), -1, -1)


            # Obtain projected features from the projector model (using DataParallel)
            projected_features = projector(features)  # (B, 4096)
            projected_features = projected_features.unsqueeze(1)  # (B, 1, 4096)
            if torch.isnan(projected_features).any() or torch.isinf(projected_features).any():
                print("NaN or Inf in projected features")
                exit()

            # Ensure the projected features are on the same device as the LLM embeddings
            projected_features = projected_features.to(llm_device)

            prompt_length = prompt_tokens.input_ids.size(1)
            feature_positions = torch.full((features.size(0), 1), prompt_length, device=llm_device, dtype=torch.long)
            feature_pos_embeds = llm.get_input_embeddings().weight[feature_positions]
            feature_embeds = projected_features + feature_pos_embeds

            gt_tokens = tokenizer(list(labels), return_tensors="pt", padding=True, truncation=True).to(llm_device)
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
            torch.nn.utils.clip_grad_norm_(projector.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        


        checkpoint_path = os.path.join(save_dir, f"smthsmth_transformer_{epoch+1}.pth")
        torch.save(projector.module.state_dict(), checkpoint_path)
        print(f"Projector checkpoint saved at epoch {epoch+1}")

    final_checkpoint = os.path.join(save_dir, "smth_transformer.pth")
    torch.save(projector.module.state_dict(), final_checkpoint)
    print("Projector saved successfully at", final_checkpoint)

if __name__ == "__main__":
    main()
