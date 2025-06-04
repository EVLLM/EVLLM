import os
import json
import re
from glob import glob


input_json_path = "/arf/scratch/egitim113/labels"
h5_folder = "/arf/scratch/egitim113/THU_Dataset/EvRepSL_smthv2"
output_json_path = "output_evrepsl_matched.json"

output_entries = []

h5_index = {}
pattern = re.compile(r"(\d+\.webm)")

for filename in os.listdir(h5_folder):
    if filename.endswith(".hdf5") and not filename.startswith("."):
        match = pattern.search(filename)
        if match:
            key = match.group(1)  # e.g., "12345.webm"
            h5_index[key] = os.path.join(h5_folder, filename)


# === Process all JSON files ===
from glob import glob

for json_file in glob(os.path.join(input_json_path, "*.json")):
    with open(json_file, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            continue

        for item in data:
            file_path = item.get("file_path", "")
            expanded_caption = item.get("expanded_caption", "")

            if not file_path or not expanded_caption:
                continue

            base_name = os.path.basename(file_path)  # e.g., "3.webm"

            if base_name in h5_index:
                output_entries.append({
                    "file_path": h5_index[base_name],
                    "label": expanded_caption
                })
            else:
                print
                continue

    print(len(output_entries))

# === SAVE OUTPUT ===
with open(output_json_path, "w", encoding="utf-8") as out_f:
    json.dump(output_entries, out_f, indent=4, ensure_ascii=False)

print(f"\n✅ Done: {len(output_entries)} matching items written to {output_json_path}")