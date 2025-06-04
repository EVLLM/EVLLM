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
JSON_FILE            = "/arf/scratch/egitim113/InstructionTuning/descriptive_instruction.json"
PRETRAINED_PROJECTOR = "/arf/scratch/egitim113/Uygar_cook/wheelie.pth"
OUTPUT_DIR           = "./outputs"
RESULTS_DIR          = os.path.join(OUTPUT_DIR, "results")
BATCH_SIZE           = 8
EPOCHS               = 10
LR                   = 1e-4

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
                'path':        path,
                'instruction': item.get('instruction', ''),
                'response':    item.get('response', '')
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
    import random

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the full dataset and shuffle
    full_dataset = InstructionDataset(JSON_FILE)
    full_entries = full_dataset.entries.copy()
    random.shuffle(full_entries)

    # Split off 5 examples for testing
    test_entries  = full_entries[:5]
    train_entries = full_entries[5:]

    # Manually create dataset objects for train/test
    train_dataset = InstructionDataset.__new__(InstructionDataset)
    train_dataset.entries      = train_entries
    train_dataset.max_retries  = full_dataset.max_retries
    train_dataset.retry_delay  = full_dataset.retry_delay

    test_dataset = InstructionDataset.__new__(InstructionDataset)
    test_dataset.entries       = test_entries
    test_dataset.max_retries   = full_dataset.max_retries
    test_dataset.retry_delay   = full_dataset.retry_delay

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize projector
    projector = EZSRProjector().to(device)
    if os.path.exists(PRETRAINED_PROJECTOR):
        projector.load_state_dict(torch.load(PRETRAINED_PROJECTOR, map_location=device))
        print(f"Loaded pretrained projector from {PRETRAINED_PROJECTOR}")
    projector = nn.DataParallel(projector)

    # HF login & tokenizer
    login(token="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")
    model_name = 'mistralai/Mistral-7B-v0.1'
    tokenizer  = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    # === LoRA Injection ===
    base_llm = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto',
        trust_remote_code=True
    )
    for param in base_llm.parameters():
        param.requires_grad = False

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )
    llm = get_peft_model(base_llm, peft_config)
    llm.print_trainable_parameters()

    optimizer = AdamW(
        list(llm.parameters()) + list(projector.parameters()),
        lr=LR
    )

    llm.train()
    projector.train()

    # === Training Loop ===
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for batch_idx, (features, instructions, responses) in enumerate(train_loader):
            bs = features.size(0)
            features = features.to(device)

            # Tokenize instructions
            instr = tokenizer(
                list(instructions),
                return_tensors='pt',
                padding=True,
                truncation=True
            ).to(llm.device)
            instr_emb = llm.get_input_embeddings()(instr.input_ids)

            # Project features and add position embedding
            proj_feats = projector(features).unsqueeze(1).to(llm.device)
            prompt_len = instr.input_ids.size(1)
            pos_ids    = torch.full((bs,), prompt_len, device=llm.device, dtype=torch.long)
            pos_emb    = llm.get_input_embeddings()(pos_ids).unsqueeze(1)
            feat_embeds = proj_feats + pos_emb

            # Tokenize responses (ground truth)
            resp = tokenizer(
                list(responses),
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=64
            ).to(llm.device)
            resp_embeds = llm.get_input_embeddings()(resp.input_ids)

            # Combine embeddings: [instruction | projected-feat | ground-truth]
            inputs_embeds  = torch.cat([instr_emb, feat_embeds, resp_embeds], dim=1)
            attention_mask = torch.ones(inputs_embeds.shape[:2], device=llm.device)

            # Prepare labels (ignore instruction + one token for feature)
            ignore_idx = -100
            prefix_len = instr_emb.size(1) + 1
            label_pad  = torch.full((bs, prefix_len),
                                     ignore_idx,
                                     device=llm.device,
                                     dtype=torch.long)
            labels     = torch.cat([label_pad, resp.input_ids], dim=1)

            outputs = llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} Batch {batch_idx} Loss {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

        # Save adapter + projector every epoch
        llm.save_pretrained(os.path.join(OUTPUT_DIR, f"lora_epoch{epoch+1}"))
        torch.save(
            projector.module.state_dict(),
            os.path.join(OUTPUT_DIR, f"projector_epoch{epoch+1}.pth")
        )

    # Final save
    llm.save_pretrained(os.path.join(OUTPUT_DIR, "lora_final"))
    torch.save(
        projector.module.state_dict(),
        os.path.join(OUTPUT_DIR, "projector_final.pth")
    )

    # === Inference on Held-out 5 Examples & Save to disk ===
    llm.eval()
    projector.eval()
    print("\n=== Inference on 5 Held-out Examples ===")

    with torch.no_grad():
        for idx, item in enumerate(test_dataset.entries):
            result = {
                "path":        item["path"],
                "instruction": item["instruction"],
                "response":    item["response"],
                "generated":   None
            }

            try:
                # Load feature
                data = torch.load(item['path'], map_location=device)
                if isinstance(data, dict):
                    feat = next(v for v in data.values() if isinstance(v, torch.Tensor))
                else:
                    feat = data
                feat = feat.to(torch.float32).unsqueeze(0).to(device)

                # Project
                feat_proj = projector(feat)

                # Build embeddings for instruction + projected feature
                instr_tok = tokenizer(
                    item["instruction"],
                    return_tensors='pt',
                    padding=True,
                    truncation=True
                ).to(llm.device)
                instr_emb = llm.get_input_embeddings()(instr_tok.input_ids)

                prompt_len = instr_tok.input_ids.size(1)
                pos_ids    = torch.tensor([prompt_len], device=llm.device)
                pos_emb    = llm.get_input_embeddings()(pos_ids).unsqueeze(1)
                feat_emb   = feat_proj.unsqueeze(1) + pos_emb

                inputs_embeds  = torch.cat([instr_emb, feat_emb], dim=1)
                attention_mask = torch.ones(inputs_embeds.shape[:2], device=llm.device)

                # Generate text
                gen_ids = llm.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=64,
                    pad_token_id=tokenizer.eos_token_id
                )
                generated_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                # Remove the prompt itself from the output if needed
                cleaned = generated_text.replace(item["instruction"], "").strip()
                result["generated"] = cleaned

                print(f"\n[{idx+1}] Instruction: {item['instruction']}")
                print(f"[Generated] {cleaned}")

            except Exception as e:
                print(f"Error on test sample {idx+1}: {e}")
                result["error"] = str(e)

            # Save this examples result as a JSON file
            out_path = os.path.join(RESULTS_DIR, f"test_{idx+1:02d}.json")
            with open(out_path, "w") as fout:
                json.dump(result, fout, indent=2)

    print(f"\nAll results written to {RESULTS_DIR}/")

if __name__ == '__main__':
    main()
