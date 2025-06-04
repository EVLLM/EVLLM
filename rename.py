import os
import json

json_folder = "/arf/scratch/egitim113/labels"
h5_folder = "/arf/scratch/egitim113/EZSR_All"


h5_ids = set()
for filename in os.listdir(h5_folder):
    if filename.endswith(".webm.h5"):
        h5_ids.add(filename.replace(".webm.h5", ""))

# === COLLECT from JSONs ===
json_ids = set()
for json_file in os.listdir(json_folder):
    if json_file.endswith(".json"):
        with open(os.path.join(json_folder, json_file), "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                file_path = item.get("file_path", "")
                if file_path:
                    base = os.path.basename(file_path)
                    if base.endswith(".webm"):
                        json_ids.add(base.replace(".webm", ""))

# === FIND: files in EZSR_All but not in JSONs ===
unused_ids = h5_ids - json_ids

# === PRINT RESULTS ===
print(f"🎯 Total in EZSR_All: {len(h5_ids)}")
print(f"📦 Referenced in JSONs: {len(json_ids)}")
print(f"🕳️ Unused files in EZSR_All: {len(unused_ids)}")
print("\n🔎 Sample of unused files:")
for i, uid in enumerate(sorted(unused_ids)):
    print(f"{uid}")
    if i >= 9:  # print first 10 only
        break