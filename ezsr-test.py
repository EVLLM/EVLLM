import os
import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from collections import defaultdict
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
        x = F.relu(self.fc1(x))  # -> (batch, 2048)
        x = self.fc2(x)          # -> (batch, 4096)
        return x

"""
Load the saved projector
"""
device = torch.device("cuda")
projector = EZSRProjector().to(device)
projector.load_state_dict(torch.load("/arf/scratch/egitim113/Uygar_cook/wheelie_50.pth", map_location=device))
projector.eval()

"""
Load the frozen Mistral-7B and tokenizer
"""
login(token="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")
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

llm = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")
llm.to(device)
llm.eval()
# FREEZE LLM
for param in llm.parameters():
    param.requires_grad = False


"""
Iterate over a random sample of 10 files from the feature folder and print the filename with the generated text.
"""
feature_folder = "/content/drive/MyDrive/Colab_Notebooks/THU-EZSR"
all_files = [f for f in os.listdir(feature_folder) if f.endswith('.pt')]
print(f"Found {len(all_files)} feature files.")

# Select 10 random files (or all files if there are fewer than 10)
sample_files = random.sample(all_files, 10)

for filename in sample_files:
    file_path = os.path.join(feature_folder, filename)

    # Load test feature -> (batch, 5, H, W)
    test_feature = torch.load(file_path, map_location=device)
    if test_feature.dim() == 4 and test_feature.size(0) == 1:
        test_feature = test_feature.squeeze(0)  # Ensure shape -> (1, 5, H, W)

    # Build the fixed prompt and get its embeddings.
    prompt_text = "Describe what is happening in the given event: "
    prompt_tokens = tokenizer(prompt_text, return_tensors="pt").to(device)
    prompt_embeds = llm.get_input_embeddings()(prompt_tokens.input_ids)  # (1, prompt_len, model_dim)

    with torch.no_grad():
        projected_feature = projector(test_feature)  # shape: (1, model_dim)
        projected_feature = projected_feature.unsqueeze(1)  # shape: (1, 1, model_dim)

        # Retrieve the positional embedding for the feature token position
        prompt_length = prompt_tokens.input_ids.size(1)
        feature_position = torch.tensor([[prompt_length]], device=device)  # shape: (1, 1)
        feature_pos_embeds = llm.get_input_embeddings().weight[feature_position]  # shape: (1, 1, model_dim)

        # Combine projected feature with positional embedding.
        feature_embed = projected_feature + feature_pos_embeds

        # Concatenate prompt and feature token embedding.
        combined_embeds = torch.cat([prompt_embeds, feature_embed], dim=1)

        # Create attention mask for combined input.
        attention_mask = torch.ones(combined_embeds.shape[:-1], dtype=torch.long, device=device)

        generated_ids = llm.generate(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.8,
            top_p=0.95
        )

        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f"Filename: {filename} | Generated text: {generated_text}")