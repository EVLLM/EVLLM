# -*- coding: utf-8 -*-
import numpy as np
from models import EffWNet
import torch
import os
import h5py

# assume polarites from {0, 1}
def events_to_voxel_grid(event_xs, event_ys, event_timestamps, event_polarities, resolution=(320, 240), temporal_bins=5):
    """
    Convert event-based data into a voxel grid representation.
    """
    voxel_grid = np.zeros((temporal_bins, resolution[1], resolution[0]), dtype=np.float32)
    timestamps = event_timestamps
    first_stamp = timestamps[0]
    last_stamp = timestamps[-1]
    deltaT = last_stamp - first_stamp
    if deltaT == 0:
        deltaT = 1.0  # Prevent division by zero if all timestamps are the same
    normalized_timestamps = (temporal_bins - 1) * (timestamps - first_stamp) / deltaT
    voxel_grid_flat = voxel_grid.ravel()
    tis = normalized_timestamps.astype(int)
    dts = normalized_timestamps - tis
    polarities = event_polarities.astype(int) * 2 - 1  # Convert {0, 1} to {-1, 1}
    x_coords = event_xs.astype(int)
    y_coords = event_ys.astype(int)
    vals_left = polarities * (1.0 - dts)
    vals_right = polarities * dts
    valid_indices = tis < temporal_bins
    np.add.at(voxel_grid_flat, x_coords[valid_indices] + y_coords[valid_indices] * resolution[0]
              + tis[valid_indices] * resolution[0] * resolution[1], vals_left[valid_indices])
    valid_indices = (tis + 1) < temporal_bins
    np.add.at(voxel_grid_flat, x_coords[valid_indices] + y_coords[valid_indices] * resolution[0]
              + (tis[valid_indices] + 1) * resolution[0] * resolution[1], vals_right[valid_indices])
    voxel_grid = np.reshape(voxel_grid_flat, (temporal_bins, resolution[1], resolution[0]))
    return voxel_grid

def events_to_two_channel_histogram(event_xs, event_ys, event_polarities, resolution=(320, 240)):
    """
    Convert event-based data into a two-channel histogram representation.
    """
    histogram = np.zeros((2, resolution[1], resolution[0]), dtype=np.float32)
    x_coords = event_xs.astype(int)
    y_coords = event_ys.astype(int)
    positive_events = event_polarities == 1
    negative_events = event_polarities == 0
    np.add.at(histogram[0], (y_coords[positive_events], x_coords[positive_events]), 1)
    np.add.at(histogram[1], (y_coords[negative_events], x_coords[negative_events]), 1)
    return histogram

def events_to_four_channel_representation(event_xs, event_ys, event_timestamps, event_polarities, resolution=(320, 240)):
    """
    Convert event-based data into a four-channel representation.
    """
    representation = np.zeros((4, resolution[1], resolution[0]), dtype=np.float32)
    x_coords = event_xs.astype(int)
    y_coords = event_ys.astype(int)
    positive_events = event_polarities == 1
    negative_events = event_polarities == 0
    np.add.at(representation[0], (y_coords[positive_events], x_coords[positive_events]), 1)
    np.add.at(representation[1], (y_coords[negative_events], x_coords[negative_events]), 1)
    normalized_timestamps = (event_timestamps - event_timestamps.min()) / (event_timestamps.max() - event_timestamps.min())
    np.maximum.at(representation[2], (y_coords[positive_events], x_coords[positive_events]), normalized_timestamps[positive_events])
    np.maximum.at(representation[3], (y_coords[negative_events], x_coords[negative_events]), normalized_timestamps[negative_events])
    return representation

def events_to_ev_surface(event_xs, event_ys, event_timestamps, event_polarities, resolution=(320, 240), time_window=1.0):
    """
    Convert event-based data into an EvSurface representation.
    """
    ev_surface = np.zeros((4, resolution[1], resolution[0]), dtype=np.float32)
    x_coords = event_xs.astype(int)
    y_coords = event_ys.astype(int)
    positive_events = event_polarities == 1
    negative_events = event_polarities == 0
    start_time = event_timestamps[-1] - time_window
    valid_events = event_timestamps >= start_time
    np.add.at(ev_surface[0], (y_coords[valid_events & positive_events], x_coords[valid_events & positive_events]), 1)
    np.add.at(ev_surface[1], (y_coords[valid_events & negative_events], x_coords[valid_events & negative_events]), 1)
    normalized_timestamps = (event_timestamps - start_time) / time_window
    normalized_timestamps = np.clip(normalized_timestamps, 0, 1)
    np.maximum.at(ev_surface[2], (y_coords[positive_events], x_coords[positive_events]), normalized_timestamps[positive_events])
    np.maximum.at(ev_surface[3], (y_coords[negative_events], x_coords[negative_events]), normalized_timestamps[negative_events])
    return ev_surface

def events_to_EvRep(event_xs, event_ys, event_timestamps, event_polarities, resolution=(426, 240)):
    """
    Convert event-based data into an EvRep representation using efficient matrix operations.
    """
    width, height = resolution
    E_C = np.zeros((height, width), dtype=np.int32)       # Count of events at each pixel
    E_I = np.zeros((height, width), dtype=np.int32)       # Net polarity of events at each pixel
    E_T_sum = np.zeros((height, width), dtype=np.float32)   # Sum of timestamp deltas
    E_T_sq_sum = np.zeros((height, width), dtype=np.float32)  # Sum of squared timestamp deltas
    event_polarities = np.where(event_polarities == 0, -1, event_polarities)
    np.add.at(E_C, (event_ys, event_xs), 1)
    np.add.at(E_I, (event_ys, event_xs), event_polarities)
    sort_indices = np.lexsort((event_timestamps, event_ys, event_xs))
    sorted_xs = event_xs[sort_indices]
    sorted_ys = event_ys[sort_indices]
    sorted_timestamps = event_timestamps[sort_indices]
    delta_timestamps = np.diff(sorted_timestamps, prepend=sorted_timestamps[0])
    np.add.at(E_T_sum, (sorted_ys, sorted_xs), delta_timestamps)
    np.add.at(E_T_sq_sum, (sorted_ys, sorted_xs), delta_timestamps**2)
    E_T_counts = E_C.clip(min=1)  # Avoid division by zero
    delta_mean = E_T_sum / E_T_counts
    E_T = np.sqrt(np.maximum((E_T_sq_sum / E_T_counts) - delta_mean**2, 0))
    EvRep = np.stack([E_C, E_I, E_T], axis=0)
    return EvRep

def load_RepGen(device="cuda"):
    # RepGen assumes batchified data B x 3 x H x W
    model = EffWNet(n_channels=3, out_depth=1, inc_f0=1, bilinear=True, n_lyr=4, ch1=12, 
                    c_is_const=False, c_is_scalar=False, device=device)
    model_path = "RepGen.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device=device)
    return model

def EvRep_to_EvRepSL(model, ev_rep, device="cuda"):
    ev_rep = torch.tensor(ev_rep, dtype=torch.float32).to(device=device)
    evrepsl = model(ev_rep)
    return evrepsl

MAX_RETRIES = 5          
RETRY_WAIT_S = 3   
      
def robust_save(output_path, tensor):
    """Try to write the HDF5 file, retrying on I/O errors."""
    attempts = 0
    while attempts < MAX_RETRIES:
        try:
            with h5py.File(output_path, 'w') as f:
                f.create_dataset('ev_rep_sl', data=tensor.detach().cpu().numpy())
            return True                      # success
        except OSError as e:
            # Only retry on I/O-style errors (e.g. remote FS hiccup)
            if e.errno not in (errno.EIO, errno.EAGAIN, errno.ETIMEDOUT):
                raise                       # something else is wrong – stop retries
            attempts += 1
            print(f"[I/O error] {output_path}  (attempt {attempts}/{MAX_RETRIES})")
            time.sleep(RETRY_WAIT_S)
    print(f"[WARN] Gave up on {output_path} after {MAX_RETRIES} retries.")
    return False


if __name__ == "__main__":
    folder_path   = '/arf/scratch/egitim111/THU-Big-Dataset/THU-NPZ'
    output_folder = 'Ready_EvRepSL_THU_BIG'
    os.makedirs(output_folder, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = load_RepGen(device)            # load once instead of per-file!

    for npz_file in sorted(f for f in os.listdir(folder_path) if f.endswith('.npz')):
        output_path = os.path.join(
            output_folder,
            f"{os.path.splitext(npz_file)[0]}_ev_rep_sl.hdf5"
        )
        if os.path.exists(output_path):
            print(f"[SKIP] {output_path} already exists.")
            continue

        npz_path = os.path.join(folder_path, npz_file)
        data = np.load(npz_path)

        ev_rep = events_to_EvRep(
            data['x_pos'].astype(np.uint32),
            data['y_pos'].astype(np.uint32),
            data['timestamp'].astype(np.uint32),
            data['polarity'].astype(np.uint32),
            resolution=(1280, 800)
        )
        ev_rep = np.expand_dims(ev_rep, axis=0)        # B×3×H×W
        ev_rep_sl = EvRep_to_EvRepSL(model, ev_rep, device)

        robust_save(output_path, ev_rep_sl)

    print("? All possible files processed.")
