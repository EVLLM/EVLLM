import argparse
import torch
from eva_clip import create_model_and_transforms, get_tokenizer
from dataset.dataset import load_and_preprocess
import os
import numpy as np
import time
import h5py

def main():
    parser = argparse.ArgumentParser(description="Extract event features from .npz files.")
    parser.add_argument("--input_folder", type=str, required=True, help="Input folder containing .npz event files")
    args = parser.parse_args()

    model_name = "EVA02-CLIP-bigE-14-plus"
    pretrained = "/arf/scratch/egitim113/pretrained/EZSR-CLIP-bigE-14-plus.pt"

    input_folder = args.input_folder
    base_output_folder = "/arf/scratch/egitim113/seact_ezsr_features"
    os.makedirs(base_output_folder, exist_ok=True)

    input_folder_name = os.path.basename(os.path.normpath(input_folder))
    final_output_folder = os.path.join(base_output_folder, input_folder_name)
    os.makedirs(final_output_folder, exist_ok=True)

    SENSOR_H, SENSOR_W = 346, 260
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

    npz_files = [f for f in os.listdir(input_folder) if f.endswith(".npz")]

    for npz_file in npz_files:
        event_path = os.path.join(input_folder, npz_file)
        output_path = os.path.join(final_output_folder, os.path.splitext(npz_file)[0] + ".h5")

        if os.path.exists(output_path):
            print(f"Skipping {npz_file}: output already exists.")
            continue

        retries = 0
        success = False
        while not success and retries < max_retries:
            try:
                npz_data = np.load(event_path)
                event_length = npz_data["x_pos"].shape[0]

                if event_length == 0:
                    print(f"Skipping {npz_file}: No events.")
                    break

                total_events += event_length

                event = load_and_preprocess(
                    event_path, SENSOR_H, SENSOR_W, event_length, representation, None, preprocess
                ).unsqueeze(0).to(device)

                inference_start = time.time()
                with torch.no_grad():
                    event_features = model.encode_image(event)
                    event_features /= event_features.norm(dim=-1, keepdim=True)
                inference_end = time.time()

                with h5py.File(output_path, 'w') as f:
                    f.create_dataset('event_features', data=event_features.cpu().numpy())

                success = True

            except (OSError, IOError) as e:
                import traceback
                traceback.print_exc()
                time.sleep(retry_delay)
                retries += 1

            except Exception as e:
                print(f"Error processing {npz_file}: {e}")
                import traceback
                traceback.print_exc()
                break

        if not success:
            print(f"Failed to process {npz_file} after {max_retries} retries.")

    total_time = time.time() - start_time
    print(f"Processing completed. Total events: {total_events}. Total time: {total_time:.2f} sec.")

if __name__ == "__main__":
    main()