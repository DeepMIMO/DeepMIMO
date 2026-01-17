



#%% Imports

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import deepmimo as dm

# To create sequences and videos
from thtt_ch_pred_utils import (
    get_all_sequences,
    make_sequence_video,  # noqa: F401
    process_and_save_channel,
    expand_to_uniform_sequences,
    interpolate_dataset_from_seqs
)

# To plot H for specific antennas (uses only matplotlib)
from thtt_ch_pred_plot import plot_iq_from_H

NT = 2
NR = 1

N_SAMPLES = 200_000
L = 60  # 20 for input, 40 for output
N_SUBCARRIERS = 1

SNR = 250 # [dB] NOTE: for RT, normalization must be consistent for w & w/o noise
MAX_DOOPLER = 100 # [Hz]
TIME_DELTA = 1e-3 # [s]

INTERPOLATE = True
INTERP_FACTOR = 10  # final interpolated numbers of points
                     # = number of points between samples - 2 (endpoints)
# Note: if samples are 1m apart, and we want 10cm between points, 
#       set INTERP_FACTOR = 102. 1 m / (102 - 2) = 10cm

DATA_FOLDER = f'../data/ch_pred_data_{N_SAMPLES//1000}k_{MAX_DOOPLER}hz_{L}steps'

GPU_IDX = 0
SEED = 42

RT_SCENARIO = 'asu_campus_3p5_10cm'

#%% [ANY ENV] 1. Ray tracing data generation: Load data

matrices = ['rx_pos', 'tx_pos', 'aoa_az', 'aod_az', 'aoa_el', 'aod_el', 
            'delay', 'power', 'phase', 'inter']

dataset = dm.load(RT_SCENARIO, matrices=matrices)

#%% [ANY ENV] (optional) Ray tracing data: Make video of all sequences

# make_sequence_video(dataset, folder='sweeps', ffmpeg_fps=60)

#%% [ANY ENV] 2. Ray tracing data generation: Create sequences

# Sequence length for channel prediction
PRE_INTERP_SEQ_LEN = L if not INTERPOLATE else max(L // INTERP_FACTOR + 1, 2) # min length is 2
# Note: interpolation will scale the sequence length by INTERP_FACTOR to be >= L
#       (at least 2 samples needed for interpolation)

# RT sample distance / INTERP_FACTOR = sample distance in the interpolated dataset

all_seqs = get_all_sequences(dataset, min_len=PRE_INTERP_SEQ_LEN)

# Print statistics
seq_lens = [len(seq) for seq in all_seqs]
sum_len_seqs = sum(seq_lens)
avg_len_seqs = sum_len_seqs / len(all_seqs)

print(f"Number of sequences: {len(all_seqs)}")
print(f"Average length of sequences: {avg_len_seqs:.1f}")

print(f"Number of active users: {len(dataset.get_active_idxs())}")
print(f"Total length of sequences: {sum_len_seqs}")

plt.hist(seq_lens, bins=np.arange(1, max(seq_lens) + 1))
plt.xlabel('Sequence length')
plt.ylabel('Number of sequences')
plt.title('Distribution of sequence lengths')
plt.grid()
plt.show()

dataset_ready = dataset

#%% [ANY ENV] 3. Ray tracing data generation: Create sequences for Channel Prediction

# Split all sequences in LENGTH L (output: (n_seqs, L)
all_seqs_mat_t = expand_to_uniform_sequences(all_seqs, target_len=PRE_INTERP_SEQ_LEN, stride=1)
print(f"all_seqs_mat_t.shape: {all_seqs_mat_t.shape}")

# Number of sequences to sample from original sequences
final_samples = min(N_SAMPLES, len(all_seqs_mat_t))
np.random.seed(SEED)
idxs = np.random.choice(len(all_seqs_mat_t), final_samples, replace=False)
all_seqs_mat_t2 = all_seqs_mat_t[idxs] # [:100] generate less sequences for testing
print(f"all_seqs_mat_t2.shape: {all_seqs_mat_t2.shape}")

#%% [ANY ENV] 4. Ray tracing data generation: Interpolate sequences

# When we interpolate, it's easier to sample sequences of length L from the original sequences, 
# and then create a dataset with only the interpolated results.

# If we don't interpolate, we can sample sequences of length L from the interpolated dataset.

if INTERPOLATE:
    dataset_ready = interpolate_dataset_from_seqs(
        dataset,
        all_seqs_mat_t2,
        points_per_segment=INTERP_FACTOR
    )
    print(f"dataset_ready.n_ue: {dataset_ready.n_ue}")
    tgt_shape = (all_seqs_mat_t2.shape[0], -1, NR, NT, 1)

# from thtt_ch_pred_utils import interpolate_dataset_from_seqs_old

# if INTERPOLATE:
#     dataset_ready_old = interpolate_dataset_from_seqs_old(
#         dataset,
#         all_seqs_mat_t2,
#         points_per_segment=INTERP_FACTOR
#     )
#     print(f"dataset_ready_old.n_ue: {dataset_ready_old.n_ue}")
#     tgt_shape = (all_seqs_mat_t2.shape[0], -1, NR, NT, 1)

#%% [ANY ENV] 5. Ray tracing data generation: Generate channels & Process data

# NOTE: when the product seq_len * n_seqs >> n_ue, it's better to generate channels first
#       and then take the sequences from the generated channels, because channel gen
#       is the most expensive part of the data generation process.
#       IF, instead, the product seq_len * n_seqs < n_ue, it's better to generate sequences first, 
#       trim the dataset to the necessary users, and then generate channels for the users in the sequences.
# TODO: implement this choice when selecting data...
# Currently: without interpolation, we gen channels first and select channels from sequence indices after. 
#            with interpolation, we select sequences and trim the dataset to the necessary users.
#            (with interpolation, this necessary because NEW points are in the dataset)

# Create channels
ch_params = dm.ChannelParameters()
ch_params.bs_antenna.shape = [NT, 1]
ch_params.ue_antenna.shape = [NR, 1]
ch_params.ofdm.subcarriers = N_SUBCARRIERS
ch_params.ofdm.selected_subcarriers = np.arange(N_SUBCARRIERS)
ch_params.ofdm.bandwidth = 15e3 * N_SUBCARRIERS # [Hz]
ch_params.doppler = True  # Enable doppler computation

doppler_way = 1

# Way 1 of adding doppler: same doppler to all users / paths
if doppler_way == 1:
    # Mean absolute alignment = 1/2 * MAX_DOOPLER
    dataset_ready.set_doppler(MAX_DOOPLER / 2)  # Add the same doppler to all users / paths

# Way 2 of adding doppler: different doppler per user / path assuming const. speed & direction
if doppler_way == 2:
    dataset_ready.rx_vel = np.array([10, 0, 0]) # [m/s] along x-axis

# Way 3 of adding doppler: different doppler per user / path assuming const. speed, 
#                          with direction derived from the path geometry
if doppler_way == 3:
    dataset_ready.rx_vel = np.array([10, 0, 0]) # [m/s] along x-axis
    # TODO!

# Way 4 of adding doppler: different doppler per user / path deriving speed & direction 
#                          from the path geometry
if doppler_way == 4:
    pass # not necessary here: speeds are constant
    # when we have uniform sampling & interpolation

H = dataset_ready.compute_channels(ch_params, times=np.arange(L) * TIME_DELTA)
print(f"H.shape: {H.shape}")  # (n_samples * L, NR, NT, N_SUBCARRIERS, L)

n_seqs = all_seqs_mat_t2.shape[0]
H_seq = np.zeros((n_seqs, L, NR, NT, N_SUBCARRIERS), dtype=np.complex64)

# Take sequences
for seq_idx in tqdm(range(n_seqs), desc="Taking sequences"):
    for sample_idx_in_seq in range(L):
        if INTERPOLATE:
            idx_in_h = seq_idx * L + sample_idx_in_seq
        else:
            idx_in_h = all_seqs_mat_t2[seq_idx, sample_idx_in_seq]
        H_seq[seq_idx, sample_idx_in_seq] = H[idx_in_h, ..., sample_idx_in_seq]
    # For each sequence, take the channels for the corresponding time steps
    # e.g. first sample of sequence is at time 0, last sample of sequence is at time L-1

print(f"H_seq.shape: {H_seq.shape}") # (n_samples, seq_len, n_rx_ant, n_tx_ant, subcarriers)

# Plot H - transform into (n_samples, n_rx_ant, n_tx_ant, seq_len)
H_3_plot = np.transpose(H_seq[:, :, :, :, 0], (0, 2, 3, 1))
plot_sample_idx, plot_rx_idx = plot_iq_from_H(H_3_plot)

# Unified post-processing and saving
H_norm, H_noisy_norm, h_max = process_and_save_channel(
    H_complex=H_seq,
    time_axis=1,
    data_folder=DATA_FOLDER,
    model=RT_SCENARIO + (f'_interp_{INTERP_FACTOR}' if INTERPOLATE else ''),
    snr_db=SNR
)

# Plot normalized version
plot_iq_from_H(H_3_plot / h_max, plot_sample_idx, plot_rx_idx)


# %% FUNCTIONS

import os
import subprocess

def get_consecutive_active_segments(dataset: dm.Dataset, idxs: np.ndarray,
                                    min_len: int = 1) -> list[np.ndarray]:
    """Get consecutive segments of active users.
    
    Args:
        dataset: DeepMIMO dataset
        idxs: Array of user indices to check
        
    Returns:
        List of arrays containing consecutive active user indices
    """
    active_idxs = np.where(dataset.los[idxs] != -1)[0]
    
    # Split active_idxs into arrays of consecutive indices
    splits = np.where(np.diff(active_idxs) != 1)[0] + 1
    consecutive_arrays = np.split(active_idxs, splits)
    
    # Filter out single-element arrays
    consecutive_arrays = [idxs[arr] for arr in consecutive_arrays if len(arr) > min_len]
    
    return consecutive_arrays


def get_all_sequences(dataset: dm.Dataset, min_len: int = 1) -> list[np.ndarray]:
    """
    Extract all consecutive active user index sequences from a dataset, 
    considering both rows and columns of the grid.

    For each row and each column in the dataset grid, this function finds all
    consecutive segments of active users (with length at least `min_len`) and
    returns them as a list of index arrays.

    Args:
        dataset (dm.Dataset): The dataset object, expected to provide grid_size,
            get_row_idxs, get_col_idxs, and to be compatible with
            get_consecutive_active_segments.
        min_len (int, optional): Minimum length of a segment to be included. 
            Defaults to 1.

    Returns:
        list[np.ndarray]: List of arrays, each containing indices of a consecutive
            active user segment (row-wise or column-wise).
    """
    n_cols, n_rows = dataset.grid_size
    all_seqs = []
    for k in range(n_rows):
        idxs = dataset.get_row_idxs(k)
        consecutive_arrays = get_consecutive_active_segments(dataset, idxs, min_len)
        all_seqs += consecutive_arrays

    for k in range(n_cols):
        idxs = dataset.get_col_idxs(k)
        consecutive_arrays = get_consecutive_active_segments(dataset, idxs, min_len)
        all_seqs += consecutive_arrays

    return all_seqs


def make_sequence_video(dataset, folder='sweeps', ffmpeg_fps=60):
    """
    Generate a video visualizing all row/col user sequences in the dataset.

    For each row and column, plots the consecutive active user segments and saves as PNGs.
    Then, uses ffmpeg to combine the PNGs into a video.

    Args:
        dataset: DeepMIMO dataset object.
        folder: Output folder for PNGs and video.
        ffmpeg_fps: Framerate for the output video.
    """

    os.makedirs(folder, exist_ok=True)
    n_cols, n_rows = dataset.grid_size

    for row_or_col in ['row', 'col']:
        for k in range(n_rows if row_or_col == 'row' else n_cols):
            idx_func = dataset.get_row_idxs if row_or_col == 'row' else dataset.get_col_idxs
            idxs = idx_func(k)
            consecutive_arrays = get_consecutive_active_segments(dataset, idxs)
            
            print(f"{row_or_col} {k} has {len(consecutive_arrays)} consecutive segments:")
            dataset.los.plot()
            for i, arr in enumerate(consecutive_arrays):
                print(f"Segment {i}: {len(arr)} users")
                plt.scatter(dataset.rx_pos[arr, 0], 
                            dataset.rx_pos[arr, 1], color='red', s=.5)
            
            plt.savefig(f'{folder}/asu_campus_3p5_{row_or_col}_{k:04d}.png', 
                        bbox_inches='tight', dpi=200)
            plt.close()
            # break

    # Create video from PNGs using ffmpeg
    subprocess.run([
        "ffmpeg", "-y",
        "-framerate", str(ffmpeg_fps),
        "-pattern_type", "glob",
        "-i", f"{folder}/*.png",
        "-vf", "crop=in_w:in_h-mod(in_h\\,2)",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        f"{folder}/output_{ffmpeg_fps}fps.mp4"
    ])


def expand_to_uniform_sequences(
    sequences: list[np.ndarray] | np.ndarray,
    target_len: int,
    stride: int = 1
) -> np.ndarray:
    """
    Convert a list or array of index sequences into a 2D array of fixed-length windows.

    For each input sequence, this function extracts all possible contiguous subsequences
    (windows) of length `target_len` using a sliding window with the specified `stride`.
    Sequences shorter than `target_len` are ignored.

    Args:
        sequences (list[np.ndarray] | np.ndarray): List of 1D arrays or a 2D array,
            where each element/row is a sequence of indices.
        target_len (int): Desired length of each output window.
        stride (int, optional): Step size for the sliding window. Defaults to 1.

    Returns:
        np.ndarray: 2D array of shape (n_windows, target_len), where each row is a
            window of indices from the input sequences. If no windows are found,
            returns an empty array of shape (0, target_len).
    """
    if isinstance(sequences, list):
        seq_list = [np.asarray(seq, dtype=int) for seq in sequences]
    else:
        # sequences is assumed 2D already; convert to list of 1D arrays
        seq_list = [np.asarray(sequences[i], dtype=int) for i in range(sequences.shape[0])]

    out: list[np.ndarray] = []
    for seq in seq_list:
        if len(seq) < target_len:
            continue
        for i in range(0, len(seq) - target_len + 1, stride):
            out.append(seq[i:i+target_len])
    if len(out) == 0:
        return np.empty((0, target_len), dtype=int)
    return np.stack(out, axis=0)

########### Data Generation INTERPOLATION - for ray tracing ###########


# Make a function that interpolates the path between 2 users
def interpolate_percentage(array1, array2, percents):
    """Interpolate between two points at specified percentages.
    
    Args:
        pos1: Starting position/value
        pos2: Ending position/value
        percents: Array of percentages between 0 and 1
        
    Returns:
        np.ndarray: Array of interpolated values at given percents
    """
    # Ensure percentages are between 0 and 1
    percents = np.clip(percents, 0, 1)

    # Broadcast to fit shape of interpolated array
    percents = np.reshape(percents, percents.shape + (1,) * array1.ndim)

    return array1 * (1 - percents) + array2 * percents


def interpolate_dataset_from_seqs(
    dataset: dm.Dataset | dm.MacroDataset,
    sequences: np.ndarray,
    step_meters: float | None = 0.5,
    points_per_segment: int | None = None
) -> dm.Dataset:
    """Create a new Dataset by interpolating along each sequence of indices.

    This function takes sequences of indices into a dataset and creates a new dataset by interpolating
    between consecutive points in each sequence. The interpolation can be done either:
    - Based on physical distance (step_meters): Points are placed every step_meters along each segment
    - Based on fixed count (points_per_segment): A fixed number of evenly-spaced points per segment

    Args:
        dataset: Source dataset containing the data to interpolate
        sequences: Array of shape [n_sequences, sequence_length] containing indices into dataset
        step_meters: Distance between interpolated points. Set to None to use points_per_segment.
        points_per_segment: Number of points per segment. Set to None to use step_meters.

    Returns:
        A new Dataset containing the interpolated data with shape [n_total_points, ...] where
        n_total_points depends on the interpolation parameters and sequence lengths.

    The following fields are interpolated:
        - rx_pos: Receiver positions [n_points, 3]
        - power, phase, delay: Ray parameters [n_points, n_rays] 
        - aoa_az, aod_az, aoa_el, aod_el: Angles [n_points, n_rays]
        - inter: Interaction types [n_points, n_rays] (copied from first point)
        - inter_pos: Interaction positions [n_points, n_rays, n_interactions, 3] (if present)
    """
    # Unwrap MacroDataset if necessary
    dataset = dataset.datasets[0] if isinstance(dataset, dm.MacroDataset) else dataset

    # Ensure ndarray of ints for sequences
    sequences = np.asarray(sequences, dtype=int)

    # Define arrays/fields used
    ray_fields = ['rx_pos', 'power', 'phase', 'delay', 'aoa_az', 'aod_az', 'aoa_el', 'aod_el']
    interpolation_fields = ray_fields + (['inter_pos'] if getattr(dataset, 'inter_pos', None) is not None else [])
    replication_fields = ['inter'] if getattr(dataset, 'inter', None) is not None else []

    # Prepare lists for all segments across all sequences
    start_idx_parts: list[np.ndarray] = []
    end_idx_parts: list[np.ndarray] = []
    t_parts: list[np.ndarray] = []

    # Local references for speed
    rx_pos = dataset.rx_pos

    # Build flattened segment lists and interpolation weights
    for seq_idx in tqdm(range(sequences.shape[0]), desc="Interpolating sequences"):
        seq = np.asarray(sequences[seq_idx], dtype=int)
        if seq.size < 2:
            continue
        for k in range(seq.size - 1):
            i1 = int(seq[k])
            i2 = int(seq[k + 1])

            # TODO: Endpoint handling
            # If we want to have the sequence-level endpoint, we can check when
            # if k == seq.size - 2: i3 = 1
            # and in the last 3 lines of this loop do: n_points + i3 and endpoint = bool(i3)
            # This will add the very last sample at the end of the sequence.

            # Determine number of interpolation points for this segment
            if step_meters is not None and points_per_segment is None:
                # Distance-based interpolation
                seg_dist = float(np.linalg.norm(rx_pos[i2] - rx_pos[i1]))
                n_points = max(1, int(np.ceil(seg_dist / float(step_meters))))
            else:
                # Fixed count interpolation
                n_points = 1 if points_per_segment is None else max(1, int(points_per_segment))

            # Gather indices and weights for this segment
            start_idx_parts.append(np.full(n_points, i1, dtype=int))
            end_idx_parts.append(np.full(n_points, i2, dtype=int))
            t_parts.append(np.linspace(0.0, 1.0, n_points, endpoint=False, dtype=np.float32))

    # Concatenate all segments
    if len(t_parts) > 0:
        start_idx = np.concatenate(start_idx_parts, axis=0)
        end_idx = np.concatenate(end_idx_parts, axis=0)
        t_all = np.concatenate(t_parts, axis=0)
    else:
        start_idx = np.empty((0,), dtype=int)
        end_idx = np.empty((0,), dtype=int)
        t_all = np.empty((0,), dtype=np.float32)

    # Helper to interpolate a field in one shot
    def _interpolate_field(field_array: np.ndarray) -> np.ndarray:
        if start_idx.size == 0:
            # No interpolated points
            return field_array[0:0]
        a = field_array[start_idx]
        b = field_array[end_idx]
        # reshape t for broadcasting to match field dims beyond first
        ratio = t_all.reshape((-1,) + (1,) * (a.ndim - 1)).astype(a.dtype, copy=False)
        return a * (1.0 - ratio) + b * ratio

    concatenated_data: dict[str, np.ndarray] = {}

    # Interpolate all interpolation fields at once per field
    for field in interpolation_fields:
        base = dataset[field]
        interp_vals = _interpolate_field(base)
        concatenated_data[field] = interp_vals

    # Replicate interaction fields (copy from first point of each segment)
    for field in replication_fields:
        base = dataset[field]
        if start_idx.size > 0:
            replicated = base[start_idx]
        else:
            replicated = base[0:0]
        concatenated_data[field] = replicated

    # Create new dataset with shared parameters
    new_dataset_params = {}
    for param in ['scene', 'materials', 'load_params', 'rt_params']:
        if hasattr(dataset, param):
            new_dataset_params[param] = getattr(dataset, param)

    new_dataset_params['n_ue'] = int(concatenated_data['rx_pos'].shape[0])
    new_dataset_params['parent_name'] = dataset.get('parent_name', dataset.name)
    new_dataset_params['name'] = f"{dataset.name}_interp"

    new_dataset = dm.Dataset(new_dataset_params)
    new_dataset.tx_pos = dataset.tx_pos

    # Assign all interpolated/replicated arrays
    for field in interpolation_fields + replication_fields:
        if field in concatenated_data:
            new_dataset[field] = concatenated_data[field]

    return new_dataset

