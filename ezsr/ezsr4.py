import argparse
import torch
from eva_clip import create_model_and_transforms
from dataset.dataset import load_and_preprocess_h5  # you must have this!
import os
import numpy as np
import time
import h5py

def main():
    parser = argparse.ArgumentParser(description="Extract event features from multiple .h5 folders.")
    parser.add_argument("--input_folders", type=str, nargs='+', required=True,
                        help="List of input folders containing .h5 event files")
    args = parser.parse_args()

    model_name = "EVA02-CLIP-bigE-14-plus"
    pretrained = "/arf/scratch/egitim113/pretrained/EZSR-CLIP-bigE-14-plus.pt"

    input_folders = args.input_folders
    base_output_folder = "/arf/scratch/egitim113/seact_ezsr_features"
    ## "/arf/scratch/egitim113/EZSR_smthv2"
    os.makedirs(base_output_folder, exist_ok=True)

    SENSOR_H, SENSOR_W = 1200, 800
    representation = "histogram"

    device = torch.device("cpu")
    print(f"Using device: {device}")

    model, _, preprocess = create_model_and_transforms(model_name, None, force_custom_clip=True)

    def load_checkpoint_on_cpu(model, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint)

    load_checkpoint_on_cpu(model, pretrained)
    model = model.to(device)

    total_events = 0
    start_time = time.time()

    max_retries = 3
    retry_delay = 10  # seconds

    # Collect all h5 files from all folders
    all_h5_files = []
    for folder in input_folders:
        if not os.path.isdir(folder):
            print(f"Skipping {folder}: Not a valid directory")
            continue
        folder_h5_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".h5")]
        all_h5_files.extend(folder_h5_files)

    print(f"Found {len(all_h5_files)} files to process.")

    for h5_path in all_h5_files:
        filename = os.path.basename(h5_path)
        output_folder_name = os.path.basename(os.path.normpath(os.path.dirname(h5_path)))
        final_output_folder = os.path.join(base_output_folder, output_folder_name)
        os.makedirs(final_output_folder, exist_ok=True)

        output_path = os.path.join(final_output_folder, os.path.splitext(filename)[0] + ".h5")

        if os.path.exists(output_path):
            print(f"Skipping {filename}: output already exists.")
            continue

        retries = 0
        success = False
        while not success and retries < max_retries:
            try:
                with h5py.File(h5_path, 'r') as f:
                    if 'events' not in f:
                        print(f"Skipping {filename}: Missing 'events' dataset.")
                        break
                    events = f['events'][...]
                    event_length = events.shape[0]

                if event_length == 0:
                    print(f"Skipping {filename}: No events.")
                    break

                total_events += event_length

                event = load_and_preprocess_h5(
                    h5_path, SENSOR_H, SENSOR_W, event_length, representation, None, preprocess
                ).unsqueeze(0).to(device)

                inference_start = time.time()
                with torch.no_grad():
                    event_features = model.encode_image(event)
                    event_features /= event_features.norm(dim=-1, keepdim=True)
                inference_end = time.time()

                with h5py.File(output_path, 'w') as f_out:
                    f_out.create_dataset('event_features', data=event_features.cpu().numpy())

                success = True

            except (OSError, IOError) as e:
                print(f"I/O error processing {filename}: {e}. Retrying in {retry_delay}s... ({retries+1}/{max_retries})")
                time.sleep(retry_delay)
                retries += 1
            except Exception as e:
                print(f"Unexpected error processing {filename}: {e}")
                break

        if not success:
            print(f"Failed to process {filename} after {max_retries} retries.")

    total_time = time.time() - start_time
    print(f"Processing completed. Total events: {total_events}. Total time: {total_time:.2f} sec.")

if __name__ == "__main__":
    main()
