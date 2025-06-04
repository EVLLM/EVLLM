import os
import h5py
import numpy as np

def convert_hdf5_to_pts(input_folder, output_folder, dataset_name='ev_rep_sl'):
    os.makedirs(output_folder, exist_ok=True)
    h5_files = [f for f in os.listdir(input_folder) if f.endswith('.h5')]

    for h5_file in h5_files:
        h5_path = os.path.join(input_folder, h5_file)
        pts_filename = os.path.splitext(h5_file)[0] + '.pts'
        pts_path = os.path.join(output_folder, pts_filename)

        try:
            with h5py.File(h5_path, 'r') as f:
                if dataset_name not in f:
                    print(f"Skipping {h5_file}: '{dataset_name}' not found.")
                    continue
                data = f[dataset_name][...]

                # Flatten or reshape depending on your expected format
                # Example: flattening the 3D tensor to 2D for `.pts` (rows = features)
                data = np.squeeze(data)  # remove batch dim if exists

                if data.ndim > 2:
                    data = data.reshape(data.shape[0], -1)

                np.savetxt(pts_path, data, fmt='%.6f')
        except Exception as e:
            print(f"Failed to convert {h5_file}: {e}")

    print("Conversion completed.")

if __name__ == "__main__":
    input_folder = '/arf/scratch/egitim113/seact_ezsr_features/ezsr_h5'
    output_folder = '/arf/scratch/egitim113/seact_ezsr_features/ezsr_pt'
    convert_hdf5_to_pts(input_folder, output_folder)