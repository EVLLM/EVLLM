"""
EZSR → MLLM Inference Pipeline
------------------------------

This script integrates a pretrained visual feature projector with a frozen large language model (LLM)
to generate natural language descriptions of visual events represented as tensor features.

Main Components:
----------------
1. **EZSRProjector**:
   A two-layer MLP that maps visual feature tensors (e.g., from video or event-based representations)
   into the LLM's embedding space.

2. **Frozen LLM (Mistral-7B)**:
   A causal language model used only in inference mode to generate text.
   All parameters are frozen to ensure inference efficiency and consistency.

3. **Feature Embedding Fusion**:
   - A static textual prompt is tokenized and embedded.
   - Projected visual features are inserted into the prompt stream as embeddings.
   - The insertion point is guided by positional embedding alignment.

4. **Text Generation**:
   - Combined prompt + visual embeddings are passed to the LLM.
   - Text is generated via nucleus sampling (top-p) with temperature.

5. **Random Sampling**:
   - A configurable number of .pt feature files are randomly selected from a folder.
   - Each is passed through the pipeline for caption generation.

Usage:
------
# 1. Recommended: set Hugging Face token as environment variable (safer)
export HF_TOKEN=hf_your_token_here

# 2. Run the script from terminal or shell
python ezsr_mllm_infer.py \
    --projector_path /path/to/wheelie_50.pth \
    --feature_folder /path/to/features \
    --samples 10

# Optional override (not needed if you used export above):
    --hf_token hf_your_token_here

Arguments:
----------
--projector_path:    Path to the trained EZSR projector (.pth)
--feature_folder:    Folder containing .pt feature tensors
--samples:           Number of random feature files to sample and process
--hf_token:          (Optional) Hugging Face API token (can also use HF_TOKEN env var)
--model_name:        (Optional) HF model name (default: mistralai/Mistral-7B-v0.1)
--device:            (Optional) Device to use (default: auto-detect cuda/cpu)
"""

import os
import json
import random
import torch
import argparse
import logging

from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

# === Logging Setup ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# === Projector Model ===
class EZSRProjector(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=2048, output_dim=4096):
        super(EZSRProjector, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def load_projector(path: str, device: torch.device) -> EZSRProjector:
    projector = EZSRProjector().to(device)
    projector.load_state_dict(torch.load(path, map_location=device))
    projector.eval()
    return projector


def load_llm_and_tokenizer(model_name: str, token: str, device: torch.device):
    login(token=token)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")
    llm.to(device)
    llm.eval()

    for param in llm.parameters():
        param.requires_grad = False

    return tokenizer, llm


def generate_descriptions(llm, tokenizer, projector, feature_dir: str, sample_count: int, device: torch.device):
    all_files = [f for f in os.listdir(feature_dir) if f.endswith(".pt")]
    logging.info(f"Found {len(all_files)} feature files.")

    sample_files = random.sample(all_files, min(sample_count, len(all_files)))

    for filename in sample_files:
        file_path = os.path.join(feature_dir, filename)
        test_feature = torch.load(file_path, map_location=device)
        if test_feature.dim() == 4 and test_feature.size(0) == 1:
            test_feature = test_feature.squeeze(0)

        prompt = "Describe what is happening in the given event: "
        prompt_tokens = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_embeds = llm.get_input_embeddings()(prompt_tokens.input_ids)

        with torch.no_grad():
            projected_feature = projector(test_feature).unsqueeze(1)

            prompt_len = prompt_tokens.input_ids.size(1)
            feature_pos = torch.tensor([[prompt_len]], device=device)
            pos_embed = llm.get_input_embeddings().weight[feature_pos]

            combined = torch.cat([prompt_embeds, projected_feature + pos_embed], dim=1)
            attention_mask = torch.ones(combined.shape[:-1], dtype=torch.long, device=device)

            generated_ids = llm.generate(
                inputs_embeds=combined,
                attention_mask=attention_mask,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.8,
                top_p=0.95
            )

            text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            logging.info(f"{filename} → {text}")


def main():
    parser = argparse.ArgumentParser(description="EZSR → MLLM Describer")

    parser.add_argument("--hf_token", type=str, default=os.getenv("HF_TOKEN"), help="HuggingFace token (or set HF_TOKEN env variable)")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-v0.1", help="Model checkpoint name")
    parser.add_argument("--projector_path", type=str, required=True, help="Path to the saved projector .pth file")
    parser.add_argument("--feature_folder", type=str, required=True, help="Path to the folder with .pt feature files")
    parser.add_argument("--samples", type=int, default=10, help="Number of random features to sample")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Computation device")

    args = parser.parse_args()

    if not args.hf_token:
        raise ValueError("HuggingFace token is required. Set it via --hf_token or HF_TOKEN environment variable.")

    device = torch.device(args.device)
    projector = load_projector(args.projector_path, device)
    tokenizer, llm = load_llm_and_tokenizer(args.model_name, args.hf_token, device)
    generate_descriptions(llm, tokenizer, projector, args.feature_folder, args.samples, device)


if __name__ == "__main__":
    main()
