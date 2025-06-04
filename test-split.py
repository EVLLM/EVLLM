import json, os, re

with open("/arf/scratch/egitim113/output_pts_ezsr.json", "r", encoding="utf-8") as f:
    items = json.load(f)          # list[dict]

# pull just the tail-end file names, e.g. "142816.webm.pt"
names = [os.path.basename(it["file_path"]) for it in items]

# numeric-aware ascending sort (so 2 < 10 < 100, etc.)
names_sorted = sorted(names, key=lambda s: int(re.match(r"\d+", s).group()))

# print the first 10
for n in names_sorted[:20]:
    print(n)