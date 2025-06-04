import json
import os
import argparse
from math import ceil

def split_json(input_path, output_dir, num_splits):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    chunk_size = ceil(total / num_splits)

    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_splits):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total)
        shard = data[start:end]
        out_path = os.path.join(output_dir, f"shard_evrepsl_{i+1}.json")
        with open(out_path, "w", encoding="utf-8") as f_out:
            json.dump(shard, f_out, indent=2, ensure_ascii=False)
        print(f"Saved {len(shard)} entries to {out_path}")

if __name__ == "__main__":

    split_json("/arf/scratch/egitim113/training/output_evrepsl_matched.json", "shards-evrepsl", 3)