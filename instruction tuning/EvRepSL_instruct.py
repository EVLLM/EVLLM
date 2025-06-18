# -*- coding: utf-8 -*-
#!/usr/bin/env python
# ===============================================================
#  Instruction-tuning with LoRA + Multi-Channel CNN/Transformer
# ===============================================================

import os, json, time, logging, random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
import h5py
os.environ["HDF5_USE_FILE_LOCKING"]="FALSE"
# ---------- Configuration ----------------------------------------------------
PREFIX                = "/arf/scratch/egitim111/THU-Big-Dataset/Ready_EvRepSL_THU_BIG/"
JSON_FILE             = "THU-EACT-INS-2.5k.json"  # <-- Update path if needed
PRETRAINED_PROJECTOR  = "devastator.pth"
OUTPUT_DIR            = "LORA"
RESULTS_DIR           = os.path.join(OUTPUT_DIR, "results")
BATCH_SIZE, EPOCHS, LR = 2, 20, 1e-4

MODEL_NAME = "mistralai/Mistral-7B-v0.1"
HF_TOKEN   = "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"

# ---------- Projector --------------------------------------------------------
class MultiChannelEncoderTransformerCLS(nn.Module):
    def __init__(self, output_size=(20, 30), num_layers=2):
        super().__init__()
        self.input_dim = output_size[0] * output_size[1]           # 600

        def _branch():
            return nn.Sequential(
                nn.Conv2d(1, 1, 3, 2, 1), nn.ReLU(),
                nn.Conv2d(1, 1, 3, 2, 1), nn.ReLU(),
                nn.Conv2d(1, 1, 3, 2, 1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(output_size)
            )

        self.branch1 = _branch(); self.branch2 = _branch()
        self.branch3 = _branch(); self.branch4 = _branch(); self.branch5 = _branch()

        self.cls_token     = nn.Parameter(torch.randn(1, 1, self.input_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 6, self.input_dim))

        enc = nn.TransformerEncoderLayer(self.input_dim, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)

        self.projector = nn.Sequential(
            nn.Linear(self.input_dim, 2048), nn.GELU(),
            nn.Linear(2048, 4096)
        )

    def forward(self, x):                         # x (N,5,H,W)
        n = x.size(0)
        t = torch.stack([
            self.branch1(x[:, 0:1]).flatten(1),
            self.branch2(x[:, 1:2]).flatten(1),
            self.branch3(x[:, 2:3]).flatten(1),
            self.branch4(x[:, 3:4]).flatten(1),
            self.branch5(x[:, 4:5]).flatten(1)], dim=1)        # (N,5,600)

        seq = torch.cat([self.cls_token.expand(n,-1,-1), t], dim=1) + self.pos_embedding
        return self.projector(self.transformer(seq)[:, 0])     # (N,4096)

# ---------- Dataset ----------------------------------------------------------
class InstructionDataset(Dataset):
    def __init__(self, json_file, max_retries=3, retry_delay=5):
        with open(json_file) as f:
            raw = json.load(f)
        # updated for your JSON format!
        self.entries = [
            {
                "path": os.path.join(PREFIX, entry["file_path"]),
                "instruction": entry["instruction"],
                "output": entry["output"],
            }
            for entry in raw
            if os.path.exists(os.path.join(PREFIX, entry["file_path"]))
        ]
        if not self.entries:
            raise RuntimeError("No valid feature files found.")
        self.max_retries, self.retry_delay = max_retries, retry_delay

    def __len__(self): return len(self.entries)

    def __getitem__(self, idx):
        item, tries = self.entries[idx], 0
        while True:
            try:
                path, ext = item["path"], os.path.splitext(item["path"])[1].lower()
                if ext in {".pt", ".pth"}:
                    data = torch.load(path, weights_only=False, map_location="cpu")
                    if isinstance(data, dict):
                        data = next(v for v in data.values() if isinstance(v, torch.Tensor))
                elif ext in {".h5", ".hdf5"}:
                    with h5py.File(path) as h5:
                        # fallback: use the first dataset if "ev_rep_sl" doesn't exist
                        key = "ev_rep_sl" if "ev_rep_sl" in h5 else list(h5.keys())[0]
                        data = torch.from_numpy(h5[key][()])
                else:
                    raise ValueError(f"Unsupported file type: {ext}")

                if data.ndim == 4 and data.size(1) == 5: data = data.mean(dim=0)
                if data.ndim == 4 and data.size(0) == 1: data = data.squeeze(0)
                if data.ndim != 3 or data.size(0) != 5:
                    raise ValueError(f"Unexpected tensor shape {tuple(data.shape)}")

                # -------- CLAMP FEATURES HERE --------
                data = torch.clamp(data, min=-1e4, max=1e4)

                return data.float(), item["instruction"], item["output"]

            except Exception as e:
                tries += 1
                if tries >= self.max_retries: raise
                logging.warning("e")
                time.sleep(self.retry_delay)

# ---------- Main -------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True); os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset split
    full = InstructionDataset(JSON_FILE)
    test_ent, train_ent = full.entries[:5], full.entries[5:]

    train_ds = InstructionDataset.__new__(InstructionDataset)
    train_ds.entries, train_ds.max_retries, train_ds.retry_delay = \
        train_ent, full.max_retries, full.retry_delay
    test_ds  = InstructionDataset.__new__(InstructionDataset)
    test_ds.entries,  test_ds.max_retries,  test_ds.retry_delay  = \
        test_ent,  full.max_retries, full.retry_delay

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True)

    # Projector
    projector = nn.DataParallel(MultiChannelEncoderTransformerCLS().to(device))
    if os.path.exists(PRETRAINED_PROJECTOR):
        try:
            projector.load_state_dict(torch.load(PRETRAINED_PROJECTOR, map_location="cpu"), strict=False)
            print(f"[+] loaded {PRETRAINED_PROJECTOR}")
        except RuntimeError as e:
            print(f"[-] projector load failed ({e})")

    # LLM on a single GPU (fp16)
    login(token=HF_TOKEN)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    base_llm = (AutoModelForCausalLM
                .from_pretrained(MODEL_NAME,
                                 torch_dtype=torch.float16,
                                 low_cpu_mem_usage=True)
                .to(device))

    for p in base_llm.parameters(): p.requires_grad = False

    lora_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=32,
                          lora_alpha=32, lora_dropout=0.05,
                          target_modules=["q_proj", "v_proj"])
    llm = get_peft_model(base_llm, lora_cfg)
    llm.print_trainable_parameters()

    optim = AdamW(list(llm.parameters()) + list(projector.parameters()), lr=LR)

    # -------------------------- TRAIN ----------------------------------------
    llm.train(); projector.train()
    for epoch in range(EPOCHS):
        total = 0.0
        for step, (feat, instr, resp) in enumerate(train_loader):
            feat = torch.clamp(feat, min=-1e4, max=1e4)  # Clamp features (training)
            feat = feat.to(device, non_blocking=True)
            proj = projector(feat)                                 # (bs,4096)
            dev  = proj.device
            proj = proj.unsqueeze(1).to(torch.float16)

            itok = tokenizer(list(instr), return_tensors="pt",
                             padding=True, truncation=True).to(device)
            iemb = llm.get_input_embeddings()(itok.input_ids).to(dev).to(torch.float16)

            plen  = iemb.size(1)
            pos   = torch.full((feat.size(0),), plen, device=device, dtype=torch.long)
            pemb  = llm.get_input_embeddings()(pos).unsqueeze(1).to(dev).to(torch.float16)
            ftok  = proj + pemb

            rtok = tokenizer(list(resp), return_tensors="pt",
                             padding=True, truncation=True, max_length=64).to(device)
            remb = llm.get_input_embeddings()(rtok.input_ids).to(dev).to(torch.float16)

            inp   = torch.cat([iemb, ftok, remb], dim=1)
            mask  = torch.ones(inp.shape[:2], device=dev)
            pad   = torch.full((feat.size(0), plen+1), -100, device=dev, dtype=torch.long)
            labels = torch.cat([pad, rtok.input_ids.to(dev)], dim=1)

            out = llm(inputs_embeds=inp, attention_mask=mask, labels=labels)
            out.loss.backward(); optim.step(); optim.zero_grad()
            total += out.loss.item()
            if step % 10 == 0:
                print(f"Epoch {epoch+1} Step {step} Loss {out.loss.item():.4f}")

        print(f"Epoch {epoch+1} mean-loss {total/len(train_loader):.4f}")
        llm.save_pretrained(os.path.join(OUTPUT_DIR, f"lora_epoch{epoch+1}"))
        torch.save(projector.module.state_dict(),
                   os.path.join(OUTPUT_DIR, f"proj_epoch{epoch+1}.pth"))

    llm.save_pretrained(os.path.join(OUTPUT_DIR, "lora_final"))
    torch.save(projector.module.state_dict(),
               os.path.join(OUTPUT_DIR, "proj_final.pth"))

    # -------------------------- INFERENCE ------------------------------------
    llm.eval(); projector.eval()
    with torch.no_grad():
        for i in range(len(test_ds)):
            x, instr, gt = test_ds[i]
            x = torch.clamp(x, min=-1e4, max=1e4)  # Clamp features (inference)
            x = x.unsqueeze(0).to(device)
            femb = projector(x).unsqueeze(1)
            dev  = femb.device
            femb = femb.to(torch.float16)

            itok = tokenizer(instr, return_tensors="pt",
                             padding=True, truncation=True).to(device)
            iemb = llm.get_input_embeddings()(itok.input_ids).to(dev).to(torch.float16)

            pos  = torch.tensor([iemb.size(1)], device=device)
            pemb = llm.get_input_embeddings()(pos).unsqueeze(1).to(dev).to(torch.float16)
            inp  = torch.cat([iemb, femb + pemb], dim=1)
            mask = torch.ones(inp.shape[:2], device=dev, dtype=torch.long)

            gen  = llm.generate(inputs_embeds=inp, attention_mask=mask,
                                max_new_tokens=64, pad_token_id=tokenizer.eos_token_id)
            txt  = tokenizer.decode(gen[0], skip_special_tokens=True)
            pred = txt.replace(instr, "").strip()

            print(f"\n[{i+1}] {instr}\n  GT : {gt}\n  PRD: {pred}")

            json.dump({"file_path": test_ds.entries[i]["path"],
                       "instruction": instr, "output": gt,
                       "generated": pred},
                      open(os.path.join(RESULTS_DIR, f"test_{i+1:02d}.json"), "w"),
                      indent=2)

    print(f"\n Results saved in {RESULTS_DIR}")

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
