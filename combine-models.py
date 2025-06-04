import torch

# === Paths to trained projectors from different shards ===
model_paths = [
    "/arf/scratch/egitim113/Uygar_cook/final-ezsr_shard_1_50.pth",
    "/arf/scratch/egitim113/Uygar_cook/final-ezsr_shard_2_50.pth",
    "/arf/scratch/egitim113/Uygar_cook/final-ezsr_shard_3_50.pth"
]

# === Load all model weights ===
state_dicts = [torch.load(path, map_location="cpu") for path in model_paths]

# === Initialize averaged state_dict ===
avg_state_dict = {}

# === Parameter-wise averaging ===
for key in state_dicts[0]:
    avg_state_dict[key] = sum(sd[key] for sd in state_dicts) / len(state_dicts)

# === Save the averaged projector ===
output_path = "/arf/scratch/egitim113/Uygar_cook/wheelie_50.pth"
torch.save(avg_state_dict, output_path)
print(f"✅ Averaged projector saved at: {output_path}")