import os
import h5py
import numpy as np
import time
import gc  # Import garbage collection module
import shutil  # Import shutil for moving files

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# Define folders
input_folder = "/arf/scratch/egitim113/seact_h5_new"
output_folder = "./npz_files"
small_files_folder = "./small_files"

# Create necessary folders if they don't exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(small_files_folder, exist_ok=True)

# Minimum file size threshold in bytes (50KB)
MIN_FILE_SIZE = 50 * 1024

# Function to convert .h5 file to .npz file with proper keys
def convert_h5_to_npz(h5_filepath, npz_filepath):
    try:
        with h5py.File(h5_filepath, 'r') as h5_file:
            if 'events' in h5_file:
                events = h5_file['events'][...]
                data_dict = {
                    'x_pos': events[:, 0],
                    'y_pos': events[:, 1],
                    'timestamp': events[:, 2],
                    'polarity': events[:, 3],
                }
                np.savez_compressed(npz_filepath, **data_dict)
            else:
                print(f"Skipping {h5_filepath}: Missing 'events' key")
    except Exception as e:
        print(f"Error processing {h5_filepath}: {e}")

# Function to process files in batches
def process_in_batches(files, batch_size=50, sleep_time=5):
    total_files = len(files)
    processed_files = 0

    for i in range(0, total_files, batch_size):
        batch = files[i:i + batch_size]
        print(f"\nProcessing batch {i // batch_size + 1} (Files {i + 1} to {min(i + batch_size, total_files)})")

        for filename in batch:
            if filename.endswith(".h5"):
                h5_filepath = os.path.join(input_folder, filename)
                file_size = os.path.getsize(h5_filepath)

                if file_size < MIN_FILE_SIZE:
                    # Move small file to the small_files_folder
                    dest_path = os.path.join(small_files_folder, filename)
                    shutil.move(h5_filepath, dest_path)
                    print(f"Moved small file: {filename} to {small_files_folder}")
                else:
                    npz_filename = os.path.splitext(filename)[0] + ".npz"
                    npz_filepath = os.path.join(output_folder, npz_filename)
                    convert_h5_to_npz(h5_filepath, npz_filepath)
                    if os.path.exists(npz_filepath):
                        processed_files += 1
                    
                    # Delete the original .h5 file
                    os.remove(h5_filepath)
                gc.collect()

        print(f"Batch {i // batch_size + 1} completed. Sleeping for {sleep_time} seconds...")
        time.sleep(sleep_time)
        gc.collect()

    print(f"\nTotal files processed: {processed_files} out of {total_files}")
    print("Batch processing completed.")

# Get the list of .h5 files in the input folder
files = os.listdir(input_folder)

# Process the files in batches
process_in_batches(files, batch_size=500, sleep_time=3)
