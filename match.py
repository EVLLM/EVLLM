import os
import json
from glob import glob


input_json_path = "/arf/scratch/egitim113/labels"
h5_folder = "/arf/scratch/egitim113/EZSR_PTs"
output_json_path = "output_pts_ezsr.json"

output_entries = []

# === Traverse all JSON files in the folder ===
for json_file in glob(os.path.join(input_json_path, "*.json")):
    with open(json_file, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            
            continue

        for item in data:
            file_path = item.get("file_path", "")
            expanded_caption = item.get("expanded_caption", "")

            # Extract ID like 139015 from file_path
            basename = os.path.splitext(os.path.basename(file_path))[0]
            target_file = f"{basename}.webm.pt"
            target_path = os.path.join(h5_folder, target_file)

            if os.path.isfile(target_path):
                output_entries.append({
                    "file_path": target_path,
                    "label": expanded_caption
                })
            else:
                continue

# === SAVE OUTPUT ===
with open(output_json_path, "w", encoding="utf-8") as out_f:
    json.dump(output_entries, out_f, indent=4, ensure_ascii=False)

print(f"\n✅ Done: {len(output_entries)} matching items written to {output_json_path}")