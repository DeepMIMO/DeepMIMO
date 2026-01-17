"""Scratchpad script for DeepMIMO conversion and visualization workflows."""

# %% Imports

import matplotlib.pyplot as plt
import numpy as np

import deepmimo as dm

from tqdm import tqdm

dataset = dm.load('asu_campus_3p5', max_paths=3)

idx_1 = 10
idx_2 = 11

dataset.plot_rays(idx_1, proj_3D=False)
dataset.plot_rays(idx_2, proj_3D=False)

dataset.print_rx(idx_1, path_idxs=[0])
dataset.print_rx(idx_2, path_idxs=[0])

#%% PATH INTERPOLATION 1: FOR ONE PAIR OF USERS

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
    
def interpolate_path(dataset, idx_1, idx_2, distances):
    """Interpolate all channel parameters between two users at specified distances.
    
    Args:
        dataset: DeepMIMO dataset
        idx_1: Index of first user
        idx_2: Index of second user
        distances: Array of distances from start point in meters
        
    Returns:
        dict: Dictionary containing interpolated channel parameters
    """
    # Get total distance for percentage calculation
    pos1 = dataset.rx_pos[idx_1]
    pos2 = dataset.rx_pos[idx_2]
    total_distance = np.linalg.norm(pos2 - pos1)
    
    # Convert distances to percentages (compute once, use for all)
    percentages = np.clip(distances / total_distance, 0, 1)
    
    # Interpolate all parameters
    params = {}
    params_to_interpolate = ['rx_pos', 'power', 'phase', 'delay', 
                             'aoa_az', 'aod_az', 'aoa_el', 'aod_el',
                             'inter_pos']
    
    for param in params_to_interpolate:
        if dataset[param] is None:
            params[param] = None
            continue
        val1 = dataset[param][idx_1]
        val2 = dataset[param][idx_2]
        params[param] = interpolate_percentage(val1, val2, percentages)
    
    return params

# Or get samples at specific distances
distances = [0, 0.6, 1]  # meters

params2 = interpolate_path(dataset, idx_1, idx_2, distances)
# returns just the interpolated value

#%% PATH INTERPOLATION 2: Generate all linear sequences in a scenario

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

#%% PATH INTERPOLATION 3: Create all sequences

def get_all_sequences(dataset: dm.Dataset, min_len: int = 1) -> list[np.ndarray]:
    n_cols, n_rows = dataset.grid_size
    all_seqs = []
    for k in range(n_rows):
        idxs = dataset.get_idxs('row', row_idxs=k)
        consecutive_arrays = get_consecutive_active_segments(dataset, idxs, min_len)
        all_seqs += consecutive_arrays

    for k in range(n_cols):
        idxs = dataset.get_idxs('col', col_idxs=k)
        consecutive_arrays = get_consecutive_active_segments(dataset, idxs, min_len)
        all_seqs += consecutive_arrays

    return all_seqs

all_seqs = get_all_sequences(dataset, min_len=1)

# Print statistics
sum_len_seqs = sum([len(seq) for seq in all_seqs])
avg_len_seqs = sum_len_seqs / len(all_seqs)

print(f"Number of sequences: {len(all_seqs)}")
print(f"Average length of sequences: {avg_len_seqs:.1f}")

print(f"Number of active users: {len(dataset.get_idxs('active'))}")
print(f"Total length of sequences: {sum_len_seqs}")


#%% PATH INTERPOLATION 4: Trim sequences to uniform length

def expand_to_uniform_sequences(sequences: list[np.ndarray] | np.ndarray,
                                target_len: int,
                                stride: int = 1) -> np.ndarray:
    """From a list/array of index sequences, return a 2D array of windows of length target_len.
    Sequences shorter than target_len are dropped. Uses sliding window with given stride.
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

# Expand to uniform sequences
all_seqs_mat_t = expand_to_uniform_sequences(all_seqs, target_len=10, stride=1)
print(f"all_seqs_mat_t.shape: {all_seqs_mat_t.shape}")

#%% PATH INTERPOLATION 5: Build interpolated dataset from sequences

# sample N sequences from all_trimmed_seqs_mat
N = min(100_000, len(all_seqs_mat_t))
idxs = np.random.choice(len(all_seqs_mat_t), N, replace=False)
all_seqs_mat_t2 = all_seqs_mat_t[idxs]
print(f"all_seqs_mat_t2.shape: {all_seqs_mat_t2.shape}")

def build_interpolated_dataset_from_sequences(dataset: dm.Dataset | dm.MacroDataset,
                                              sequences: np.ndarray,
                                              step_meters: float | None = 0.5,
                                              points_per_segment: int | None = None) -> dm.Dataset:
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

    assert sequences.ndim == 2, "sequences must be a 2D array [n_seq, seq_len]"
    n_sequences, seq_length = sequences.shape

    # Define arrays to interpolate
    ray_fields = ['rx_pos', 'power', 'phase', 'delay', 'aoa_az', 'aod_az', 'aoa_el', 'aod_el']
    interpolation_fields = ray_fields + (['inter_pos'] if dataset.inter_pos is not None else [])
    replication_fields = ['inter'] if dataset.inter is not None else []

    # Initialize accumulators for interpolated data
    expanded_data = {field: [] for field in interpolation_fields + replication_fields}

    for seq_idx in tqdm(range(n_sequences), desc="Interpolating sequences"):
        sequence_indices = sequences[seq_idx].astype(int)
        for segment_idx in range(seq_length - 1):
            point1_idx = int(sequence_indices[segment_idx])
            point2_idx = int(sequence_indices[segment_idx + 1])

            # Determine interpolation points for this segment
            if step_meters is not None and points_per_segment is None:
                # Distance-based interpolation
                pos1, pos2 = dataset.rx_pos[point1_idx], dataset.rx_pos[point2_idx]
                segment_distance = float(np.linalg.norm(pos2 - pos1))
                n_points = max(1, int(np.ceil(segment_distance / float(step_meters))))
            else:
                # Fixed count interpolation
                n_points = 1 if points_per_segment is None else max(1, int(points_per_segment))
            
            interp_points = np.linspace(0.0, 1.0, n_points, endpoint=False)
            if interp_points.size == 0:
                continue

            # Interpolate ray parameters
            for field in interpolation_fields:
                val1, val2 = dataset[field][point1_idx], dataset[field][point2_idx]
                expanded_data[field].append(interpolate_percentage(val1, val2, interp_points))

            # Copy interaction data from first point
            for field in replication_fields:
                data = dataset[field][point1_idx]
                expanded_data[field].append(np.tile(data[None, ...], (len(interp_points), 1)))

        # Append final endpoint of the sequence explicitly
        final_idx = sequence_indices[-1]
        for field in interpolation_fields + replication_fields:
            expanded_data[field].append(np.expand_dims(dataset[field][final_idx], 0))

    # Concatenate all interpolated data
    concatenated_data: dict[str, np.ndarray] = {}
    for field, data_list in expanded_data.items():
        if len(data_list):
            concatenated_data[field] = np.concatenate(data_list, axis=0)

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
    
    # Assign all interpolated arrays
    for field in interpolation_fields + replication_fields:
        if field in concatenated_data:
            new_dataset[field] = concatenated_data[field]

    return new_dataset

# Example: build one small batch dataset, compute channels, then you can persist or stream
example_batch_size = min(20, all_seqs_mat_t2.shape[0])
pps = 10  # fixed number of points per segment ensures uniform lengths
interp_batch_ds = build_interpolated_dataset_from_sequences(dataset, 
    all_seqs_mat_t2[:example_batch_size], 
    points_per_segment=pps)
print(f"Interpolated batch dataset users: {interp_batch_ds.n_ue}")

# Create channels
ch_params = dm.ChannelParameters()
ch_params.bs_antenna.shape = [2, 1]
ch_params.ue_antenna.shape = [1, 1]
H = interp_batch_ds.compute_channels(ch_params)
print(f"H.shape (interpolated batch): {H.shape}")

# Compute uniform output sequence length for reshape
seq_in_len = all_seqs_mat_t2.shape[1]
seq_out_len = (seq_in_len - 1) * pps + 1
H_seq = H.reshape(example_batch_size, seq_out_len, *H.shape[1:])
print(f"H_seq.shape: {H_seq.shape}") # (n_seq_batch, seq_len, n_rx_ant, n_tx_ant, subcarriers)

#%% PATH INTERPOLATION 6: Plot interpolation results

# Compare positions before/after interpolation for a few sequences
n_plot = min(3, example_batch_size)
for i in range(n_plot):
    orig_seq = all_seqs_mat_t2[i]
    start = i * seq_out_len
    end = start + seq_out_len
    interp_slice = slice(start, end)

    plt.figure(dpi=200)
    # Original positions along the sequence
    plt.plot(dataset.rx_pos[orig_seq, 0], dataset.rx_pos[orig_seq, 1], 'o', label='Original (indices)')
    # Interpolated positions
    plt.plot(interp_batch_ds.rx_pos[interp_slice, 0], interp_batch_ds.rx_pos[interp_slice, 1], '-x', label='Interpolated')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f'RX Positions: Original vs Interpolated (seq {i})')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.grid(True)
    plt.legend()
    plt.show()

#%% Compare an example variable (power of first path) before/after interpolation
var = 'inter'
for i in range(3):
    orig_seq = all_seqs_mat_t2[i]
    start = i * seq_out_len
    end = start + seq_out_len
    interp_slice = slice(start, end)

    # Compute the sample indices in the interpolated sequence that correspond to the original samples
    # These are always at positions [0, pps, 2*pps, ..., (len(orig_seq)-1)*pps]
    orig_sample_indices = np.arange(len(orig_seq)) * pps

    plt.figure(dpi=200)
    # Power for first path (path index 0)
    plt.plot(orig_sample_indices, dataset[var][orig_seq, 0], 'o-', label='Original (path 0)')
    plt.plot(np.arange(seq_out_len), interp_batch_ds[var][interp_slice, 0], 'x-', label='Interpolated (path 0)')
    plt.title(f'{var} (first path): Original vs Interpolated (seq {i})')
    plt.xlabel('Sample index along sequence')
    plt.ylabel(f'{var} [dBW]')
    plt.grid(True)
    plt.legend()
    plt.show()

#%% plot IQ from H

# Expand to uniform sequences
# all_seqs_mat_t3 = expand_to_uniform_sequences(all_seqs, target_len=95, stride=1)
# print(f"all_seqs_mat_t3.shape: {all_seqs_mat_t3.shape}")

# # sample N sequences from all_trimmed_seqs_mat
# N = min(100_000, len(all_seqs_mat_t3))
# idxs = np.random.choice(len(all_seqs_mat_t3), N, replace=False)
# all_seqs_mat_t3 = all_seqs_mat_t3[idxs]
# print(f"all_seqs_mat_t3.shape: {all_seqs_mat_t3.shape}")

# Plot H - transform to fit: (n_samples, n_rx_ant, n_tx_ant, seq_len)
H_3_plot = np.transpose(H_seq[:, :95, :, :, 0], (0, 2, 3, 1))

def plot_iq_from_H(H: np.ndarray, sample_idx: int | None = None, rx_idx: int | None = None):
    # H shape: (n_samples, n_rx, n_tx, n_time_steps), complex64
    i = np.random.randint(H.shape[0]) if sample_idx is None else sample_idx
    r = np.random.randint(H.shape[1]) if rx_idx is None else rx_idx

    plt.figure(dpi=150)
    lim = 0.0
    for t in range(H.shape[2]):
        z = H[i, r, t, :]  # complex time series
        lim = max(lim, np.max(np.abs(z)))
        plt.plot(z.real, z.imag, "*-", markersize=3, label=f"Tx antenna {t+1}")
    lim = float(lim) * 1.1
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True)
    plt.xlabel("In-Phase")
    plt.ylabel("Quadrature")
    plt.legend()
    plt.title("Channel Gain")
    plt.show()

    return i, r  # return sample and rx index

plot_sample_idx, plot_rx_idx = plot_iq_from_H(H_3_plot)


# %%
