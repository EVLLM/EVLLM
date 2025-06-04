import os
import numpy as np
import time
import gc  # Import garbage collection module
import shutil  # Import shutil for moving files

# Define folders
input_folder = "/arf/scratch/egitim113/newdataset"
output_folder = "./npz_files"
small_files_folder = "./small_files"

# Create necessary folders if they don't exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(small_files_folder, exist_ok=True)

# Minimum file size threshold in bytes (50KB)
MIN_FILE_SIZE = 1 * 1024

# Function to convert .npy file to .npz file with proper keys
def convert_npy_to_npz(npy_filepath, npz_filepath):
    try:
        data = np.load(npy_filepath, allow_pickle=True)
        # Check if the loaded data is a 2D numpy array with at least 4 columns.
        if isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] >= 4:
            np.savez_compressed(
                npz_filepath,
                x_pos=data[:, 0],
                y_pos=data[:, 1],
                timestamp=data[:, 2],
                polarity=data[:, 3]
            )
        else:
            print(f"Skipping {npy_filepath}: Data is not in the expected format (n, >=4)")
    except Exception as e:
        print(f"Error processing {npy_filepath}: {e}")

# Function to process files in batches
def process_in_batches(files, batch_size=50, sleep_time=5):
    total_files = len(files)
    processed_files = 0

    for i in range(0, total_files, batch_size):
        batch = files[i:i + batch_size]
        print(f"\nProcessing batch {i // batch_size + 1} (Files {i + 1} to {min(i + batch_size, total_files)})")

        for filename in batch:
            if filename.endswith(".npy"):
                npy_filepath = os.path.join(input_folder, filename)
                file_size = os.path.getsize(npy_filepath)

                if file_size < MIN_FILE_SIZE:
                    # Move small file to the small_files_folder
                    dest_path = os.path.join(small_files_folder, filename)
                    shutil.move(npy_filepath, dest_path)
                    print(f"Moved small file: {filename} to {small_files_folder}")
                else:
                    npz_filename = os.path.splitext(filename)[0] + ".npz"
                    npz_filepath = os.path.join(output_folder, npz_filename)
                    convert_npy_to_npz(npy_filepath, npz_filepath)
                    if os.path.exists(npz_filepath):
                        processed_files += 1
                    
                    # Delete the original .npy file
                    os.remove(npy_filepath)
                gc.collect()

        print(f"Batch {i // batch_size + 1} completed. Sleeping for {sleep_time} seconds...")
        time.sleep(sleep_time)
        gc.collect()

    print(f"\nTotal files processed: {processed_files} out of {total_files}")
    print("Batch processing completed.")

# Get the list of .npy files in the input folder
files = os.listdir(input_folder)

# Process the files in batches
process_in_batches(files, batch_size=500, sleep_time=3)
