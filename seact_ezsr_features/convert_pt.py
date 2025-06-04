#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# h5_folder_to_pt_folder.py
#
# Convert every .h5 file in INPUT_DIR to a .pt file in OUTPUT_DIR.
# Edit the four variables below and run:  python convert_pt.py

import pathlib
import h5py
import torch
from tqdm import tqdm

# ---------- CONFIG ----------------------------------------------------
INPUT_DIR  = pathlib.Path("/arf/scratch/egitim113/seact_ezsr_features/ezsr_h5")   # folder with .h5 files
OUTPUT_DIR = pathlib.Path("/arf/scratch/egitim113/seact_ezsr_features/ezsr_pt")       # where .pt files will be written
KEYS       = ["event_features"]                     # dataset names inside each .h5
RECURSIVE  = False                          # True: also search subfolders
# ----------------------------------------------------------------------

def convert_one(h5_path: pathlib.Path, out_dir: pathlib.Path, keys):
    data = {}
    with h5py.File(h5_path, "r") as h5f:
        for k in keys:
            if k not in h5f:
                print(f"[WARN] {h5_path.name}: dataset '{k}' missing")
                continue
            data[k] = torch.from_numpy(h5f[k][...])

    if not data:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / h5_path.with_suffix(".pt").name
    torch.save(data, out_path)

def main():
    pattern = "**/*.h5" if RECURSIVE else "*.h5"
    h5_files = sorted(INPUT_DIR.glob(pattern))
    if not h5_files:
        print(f"No .h5 files found in {INPUT_DIR.resolve()}")
        return

    print(f"Converting {len(h5_files)} file(s) to {OUTPUT_DIR.resolve()}")
    for f in tqdm(h5_files, desc="h5 -> pt"):
        convert_one(f, OUTPUT_DIR, KEYS)
    print("Done.")

if __name__ == "__main__":
    main()
