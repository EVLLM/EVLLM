import os
import shutil
import json

# === CONFIGURATION ===
json_path = "/arf/scratch/egitim113/output_pts_ezsr.json"  # Path to your JSON file
source_dir = "/arf/scratch/egitim113/EZSR_PTs"  # Source directory with .pt files
destination_dir = "/arf/scratch/egitim113/Selected_PTs"  # Destination directory to copy files into
num_files_to_copy = 10  # Number of files to copy

# === LOAD JSON FILE ===
with open(json_path, "r") as f:
    data = json.load(f)

# Extract filenames listed in the JSON
json_filenames = set(os.path.basename(entry["file_path"]) for entry in data)

# List all files in the source directory
all_files = set(os.listdir(source_dir))

# Determine files that are in the source directory but not in the JSON
unlisted_files = sorted(all_files - json_filenames)

# Create destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Select N files to copy
files_to_copy = unlisted_files[:num_files_to_copy]

# Copy files
for filename in files_to_copy:
    src = os.path.join(source_dir, filename)
    dst = os.path.join(destination_dir, filename)
    shutil.copy2(src, dst)

print(f"Copied {len(files_to_copy)} files to {destination_dir}")
