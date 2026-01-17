"""Scratchpad script for DeepMIMO conversion and visualization workflows."""

# %% Imports

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import deepmimo as dm

from tqdm import tqdm
# from api_keys import DEEPMIMO_API_KEY

# d_10cm = dm.load('asu_campus_3p5_10cm', filter_matrices=['inter_pos'])

matrices = ['rx_pos', 'tx_pos', 'aoa_az', 'aod_az', 'aoa_el', 'aod_el', 
            'delay', 'power', 'phase', 'inter']
# dataset = dm.load('asu_campus_3p5_10cm', matrices=matrices)

dataset = dm.load('asu_campus_3p5')#, matrices=matrices)

idx_1 = 10
idx_2 = 11



#%% V4 Conversion

# Example usage
rt_folder = "./RT_SOURCES/asu_campus"

scen_name = Path(rt_folder).name
dm.convert(rt_folder, overwrite=True, scenario_name=scen_name, vis_scene=True)

dataset = dm.load('asu_campus_3p5')

# %% AODT Conversion

# aodt_scen_name = 'aerial_2025_6_18_16_43_21'  # new (1 user)
# aodt_scen_name = 'aerial_2025_6_22_16_10_16' # old (2 users)
aodt_scen_name = "aerial_2025_6_18_16_43_21_dyn"  # new (1 user, dynamic)
folder = f"aodt_scripts/{aodt_scen_name}"
# df = pd.read_parquet(str(Path(folder) / 'db_info.parquet'))

# df.head()
aodt_scen = dm.convert(folder, overwrite=True)

aodt_scen = dm.load(aodt_scen_name, max_paths=500)

#%% TESTING PATH IDS

# dataset = dm.load('asu_campus_3p5')
dataset = dm.load('asu_campus_3p5', max_paths=3)
# dataset = dataset.trim_by_path_type(['LoS', 'R'])
# dataset = dataset.trim_by_path_depth(5) # not necessary

idx_1 = 10
idx_2 = 11

# dataset.plot_rays(idx_1, proj_3D=False)
# dataset.plot_rays(idx_2, proj_3D=False)

# dataset.print_rx(idx_1, path_idxs=[0])
# dataset.print_rx(idx_2, path_idxs=[0])

#%% Assigning path IDs

def expand_interaction_codes_vectorized(inter: np.ndarray, pad_value: int = -1) -> np.ndarray:
    """
    Converts integer-encoded interaction types (e.g. 121 → [1,2,1])
    into padded array of shape (n_users, n_paths, max_n_interactions),
    handling NaN entries properly.

    Args:
        inter: np.ndarray of shape (n_users, n_paths), may contain NaN
        pad_value: int, value used for padding

    Returns:
        expanded: np.ndarray of shape (n_users, n_paths, max_n_interactions)
    """
    inter_flat = inter.flatten()

    digit_lists = []
    max_len = 0

    for valid in inter_flat:
        if np.isnan(valid):
            digit_lists.append([])  # empty for padding
        else:
            digits = [int(d) for d in str(int(valid))]
            digit_lists.append(digits)
            max_len = max(max_len, len(digits))

    # Pad all digit lists
    padded = np.full((len(digit_lists), max_len), pad_value, dtype=int)
    for i, digits in enumerate(digit_lists):
        padded[i, :len(digits)] = digits

    return padded.reshape((*inter.shape, max_len))
    
def assign_path_ids(dataset: dm.Dataset) -> np.ndarray:
    """
    Assigns unique IDs to paths based on their interaction signatures.
    Paths with the same sequence of interactions get the same ID.
    
    Args:
        dataset: DeepMIMO dataset containing path information
        
    Returns:
        path_ids: (n_users, n_paths) array of path IDs
    """
    inter_typ = expand_interaction_codes_vectorized(dataset.inter)
    
    inter_obj = dataset.inter_obj
    n_users, n_paths, _ = inter_obj.shape
    path_ids = np.zeros((n_users, n_paths), dtype=int)
    path_signature_to_id = {}
    next_id = 0

    for u in tqdm(range(n_users), desc='Assigning path IDs'):
        if dataset.los[u] == -1:
            continue

        n_paths = dataset.num_paths[u]
        for p in range(n_paths):
            n_interactions = int(dataset.num_inter[u, p])
            types = inter_typ[u, p, :n_interactions]  # (n_interactions,)
            objs = inter_obj[u, p, :n_interactions]  # (n_interactions,)

            # Combine into ordered list of (type, object)
            path_signature = tuple((int(t), int(o)) for t, o in zip(types, objs))

            # Hash or assign ID
            if path_signature not in path_signature_to_id:
                path_signature_to_id[path_signature] = next_id
                next_id += 1

            path_ids[u, p] = path_signature_to_id[path_signature]
            
    return path_ids

path_ids = assign_path_ids(dataset)

#%% Hashing user paths

def hash_user_paths(path_ids: np.ndarray, num_paths: np.ndarray) -> np.ndarray:
    """
    Creates a hash for each user based on their set of path IDs.
    Users with the same set of paths will get the same hash.
    
    Args:
        path_ids: (n_users, max_paths) array of path IDs
        num_paths: (n_users,) array with number of valid paths per user
        
    Returns:
        user_hashes: (n_users,) array of hash IDs
    """
    n_users = path_ids.shape[0]
    user_signature_to_id = {}
    user_hashes = np.zeros(n_users, dtype=int)
    next_hash_id = 0
    
    for u in tqdm(range(n_users), desc='Hashing user paths'):
        if num_paths[u] == 0:  # Skip users with no paths
            user_hashes[u] = -1
            continue
        
        # TODO: ADD 2 options:
        # 1) Sort by path ID and then tuple (likely better)
        # 2) Tuple with the order they have. (much smaller zones)

        # Get valid path IDs for this user and sort them
        valid_paths = sorted(path_ids[u, :num_paths[u]])
        path_signature = tuple(valid_paths)
        
        # Assign hash ID
        if path_signature not in user_signature_to_id:
            user_signature_to_id[path_signature] = next_hash_id
            next_hash_id += 1
            
        user_hashes[u] = user_signature_to_id[path_signature]
        
    return user_hashes

user_hashes = hash_user_paths(path_ids, dataset.num_paths)

#%% Plotting coverage map

# Generate better colors using HSV space for more distinct colors
import colorsys

def generate_distinct_colors(n):
    """Generate n visually distinct colors using HSV color space"""
    colors = []
    for i in range(n):
        # Use golden ratio to space hues evenly
        hue = (i * 0.618033988749895) % 1
        # Vary saturation and value slightly for more distinction
        sat = 0.8 + (i % 3) * 0.1  # Vary between 0.8-1.0
        val = 0.9 + (i % 2) * 0.1  # Vary between 0.9-1.0
        rgb = colorsys.hsv_to_rgb(hue, sat, val)
        colors.append(list(rgb) + [1.0])  # Add alpha=1.0
    return np.array(colors)

# Generate color map (excluding users with no paths)
valid_hashes = np.unique(user_hashes[user_hashes != -1])
n_colors = len(valid_hashes)
colors = generate_distinct_colors(n_colors)
hash_to_color = {h: colors[i] for i, h in enumerate(valid_hashes)}
hash_to_color[-1] = [1, 1, 1, 1.0]  # White for invalid users

# Create color array for all users
user_colors = np.array([hash_to_color[h] for h in user_hashes])

# Plot coverage map with colors and create proper colorbar
dataset.plot_coverage(user_colors, dpi=300)

#%% Path ID and Hash Analysis
inter_typ = expand_interaction_codes_vectorized(dataset.inter)
inter_obj = dataset.inter_obj
# Analyze path ID frequencies
path_id_counts = {}
for u in range(len(dataset.num_paths)):
    n_paths = dataset.num_paths[u]
    for p in range(n_paths):
        pid = path_ids[u, p]
        if pid not in path_id_counts:
            path_id_counts[pid] = 0
        path_id_counts[pid] += 1

# Sort by frequency
sorted_path_ids = sorted(path_id_counts.items(), key=lambda x: x[1], reverse=True)

print("\nMost common path IDs and their frequencies:")
for pid, count in sorted_path_ids[:5]:
    print(f"Path ID {pid}: {count} occurrences")

# Function to get interaction sequence for a path
def get_path_sequence(dataset, user_idx, path_idx):
    """Get the full interaction sequence for a specific path."""
    n_inter = int(dataset.num_inter[user_idx, path_idx])
    types = inter_typ[user_idx, path_idx, :n_inter]
    objs = inter_obj[user_idx, path_idx, :n_inter]
    return list(zip(types, objs))

# Find example users for each common path ID
print("\nExample interaction sequences for most common paths:")
for pid, _ in sorted_path_ids[:5]:
    # Find first user with this path ID
    for u in range(len(dataset.num_paths)):
        n_paths = dataset.num_paths[u]
        for p in range(n_paths):
            if path_ids[u, p] == pid:
                sequence = get_path_sequence(dataset, u, p)
                print(f"\nPath ID {pid}:")
                print(f"User {u}, Path {p}")
                print("Sequence:", sequence)
                break
        else:
            continue
        break

# Analyze hash 3 specifically
print("\nAnalyzing hash 3:")
hash_3_users = np.where(user_hashes == 3)[0]
print(f"Number of users with hash 3: {len(hash_3_users)}")

# Get path IDs for first few users with hash 3
print("\nPath IDs for first few users with hash 3:")
for u in hash_3_users[:5]:
    n_paths = dataset.num_paths[u]
    print(f"\nUser {u} paths:")
    for p in range(n_paths):
        pid = path_ids[u, p]
        sequence = get_path_sequence(dataset, u, p)
        print(f"Path {p} (ID {pid}): {sequence}")

# Plot spatial distribution of hash 3 users
plt.figure(figsize=(10, 6), dpi=300)
plt.scatter(dataset.rx_pos[hash_3_users, 0], dataset.rx_pos[hash_3_users, 1], 
           s=1, alpha=0.3, label='Hash 3 Users')
plt.title('Spatial Distribution of Hash 3 Users')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.grid(True)
plt.show()

#%% Create Sequences

# 1) For each hash, get the idxs of the users that have that hash.

# 2) Make a choice of 2 users that have the same hash.

# 3) Connect these 2 users. 

# 4) Sample positions along this connection

# 5) Get the closest user in the hash to each position.

# 6) Obtain a final sequence of users in the hash along the path.

# 7) Save the sequence of users in a hash dictionary.

# 8) Repeat a number of times proportional to the number of users in the hash.

# For one or two hashes, plot the sequences.

# Get indices of users for each hash value
hash_to_users = {}
for h in np.unique(user_hashes):
    if h != -1:  # Skip invalid users
        hash_to_users[h] = np.where(user_hashes == h)[0]

def sample_positions_between_users(pos1, pos2, n_samples=10):
    """Sample positions along a straight line between two users."""
    t = np.linspace(0, 1, n_samples)
    return np.array([pos1 * (1-ti) + pos2 * ti for ti in t])

def find_closest_user(position, user_positions):
    """Find the closest user to a given position."""
    distances = np.linalg.norm(user_positions - position, axis=1)
    return np.argmin(distances)

def create_user_sequence(user_idxs, dataset, n_samples=10):
    """Create a sequence of users between two randomly chosen users with the same hash."""
    if len(user_idxs) < 2:
        return None
        
    # Randomly select two different users
    u1, u2 = np.random.choice(user_idxs, size=2, replace=False)
    
    # Get their positions
    pos1 = dataset.rx_pos[u1]
    pos2 = dataset.rx_pos[u2]
    
    # Sample positions along the path
    sampled_positions = sample_positions_between_users(pos1, pos2, n_samples)
    
    # Get user positions for this hash
    hash_user_positions = dataset.rx_pos[user_idxs]
    
    # Find closest users to each sampled position
    sequence = []
    for pos in sampled_positions:
        closest_idx = find_closest_user(pos, hash_user_positions)
        sequence.append(user_idxs[closest_idx])
        
    return np.unique(sequence)

# Create sequences for each hash
sequences_per_hash = {}
n_sequences_factor = 0.2  # Create sequences for 20% of users in each hash

for hash_val, user_idxs in tqdm(hash_to_users.items(), desc='Creating sequences per hash'):
    n_users_per_hash = len(user_idxs)
    n_sequences = max(1, int(n_users_per_hash * n_sequences_factor))
    sequences = []
    
    for _ in range(n_sequences):
        seq = create_user_sequence(user_idxs, dataset)
        if seq is not None:
            sequences.append(seq)
    
    sequences_per_hash[hash_val] = sequences

#%% Plotting sequences

# Plot a few example sequences
plt.figure(figsize=(12, 8))

# # Plot all users in gray first
# plt.scatter(dataset.rx_pos[:, 0], dataset.rx_pos[:, 1], 
#            color='lightgray', alpha=0.3, label='All Users')

# Plot sequences for first two hashes with different colors
colors = ['red', 'blue', 'green', 'purple']
for i, (hash_val, sequences) in enumerate(list(sequences_per_hash.items())[1758:1762]):
    color = colors[i]
    
    # Plot users with this hash
    user_idxs = hash_to_users[hash_val]
    plt.scatter(dataset.rx_pos[user_idxs, 0], dataset.rx_pos[user_idxs, 1],
               color=color, alpha=0.5, label=f'Hash {hash_val} Users')
    
    # Plot sequences
    for seq in sequences[:2]:  # Plot first two sequences for this hash
        plt.plot(dataset.rx_pos[seq, 0], dataset.rx_pos[seq, 1],
                color=color, linestyle='--', alpha=0.8)
        plt.scatter(dataset.rx_pos[seq, 0], dataset.rx_pos[seq, 1],
                   color=color, marker='x')

plt.title('User Sequences by Hash')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.grid(True)
plt.show()


#%% Visualize user hashes first and plot statistics

# Visualize spatial distribution of first 10 hashes
plt.figure(dpi=300, figsize=(10, 6))

# Plot each hash with a different color
colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
for i, (hash_val, user_idxs) in enumerate(list(hash_to_users.items())[:10]):
    positions = dataset.rx_pos[user_idxs]
    n_users = len(user_idxs)
    
    plt.scatter(positions[:, 0], positions[:, 1], 
               s=.5, alpha=0.6, color=colors[i], 
               label=f'Hash {hash_val} (n={n_users})')

plt.title('Spatial Distribution of First 10 Hashes')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend(bbox_to_anchor=(1.0, 1.015), loc='upper left', markerscale=10)
plt.grid(True)
plt.tight_layout()
plt.show()


#%% Plot analysis of number of users per hash

plt.figure(dpi=300, figsize=(10, 6))
hash_vals = []
num_users = []
max_spreads = []
for hash_val, user_idxs in hash_to_users.items():
    hash_vals.append(hash_val)
    num_users.append(len(user_idxs))
    
    # Calculate spatial spread
    if len(user_idxs) > 0:
        positions = dataset.rx_pos[user_idxs]
        center = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - center, axis=1)
        max_spreads.append(np.max(distances))
    else:
        max_spreads.append(0)

plt.bar(hash_vals, num_users)
plt.title('Number of Users per Hash')
plt.xlabel('Hash ID')
plt.ylabel('Number of Users')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.bar(hash_vals, max_spreads)
plt.title('Max Spread of Users per Hash')
plt.xlabel('Hash ID')
plt.ylabel('Max Spread')
plt.grid(True)
plt.tight_layout()
plt.show()

#%% Plot top 5 largest hashes

# Get the top 5 hashes by number of users
hash_counts = {}
for h in user_hashes:
    if h != -1:  # Skip invalid users
        if h not in hash_counts:
            hash_counts[h] = 0
        hash_counts[h] += 1

top_5_hashes = sorted(hash_counts.items(), key=lambda x: x[1], reverse=True)[:5]

# Create scatter plot
plt.figure(figsize=(12, 8), dpi=300)

# Use distinct colors for each hash
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']  # Color-blind friendly palette

# Plot each hash
for i, (hash_val, count) in enumerate(top_5_hashes):
    # Get users with this hash
    users = np.where(user_hashes == hash_val)[0]
    
    # Get their positions
    positions = dataset.rx_pos[users]
    
    # Calculate spatial statistics
    center = np.mean(positions, axis=0)
    distances = np.linalg.norm(positions - center, axis=1)
    max_spread = np.max(distances)
    avg_spread = np.mean(distances)
    
    # Plot users
    plt.scatter(positions[:, 0], positions[:, 1], 
               s=2, alpha=0.4, color=colors[i], 
               label=f'Hash {hash_val} (n={count}, spread={avg_spread:.1f}m)')

plt.title('Spatial Distribution of Top 5 Largest Hashes')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=4)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Print detailed statistics for each hash
print("\nDetailed statistics for top 5 hashes:")
for hash_val, count in top_5_hashes:
    users = np.where(user_hashes == hash_val)[0]
    positions = dataset.rx_pos[users]
    center = np.mean(positions, axis=0)
    distances = np.linalg.norm(positions - center, axis=1)
    
    # Get path IDs for first user as example
    first_user = users[0]
    n_paths = dataset.num_paths[first_user]
    path_sequence = []
    for p in range(n_paths):
        pid = path_ids[first_user, p]
        sequence = get_path_sequence(dataset, first_user, p)
        path_sequence.append((pid, sequence))
    
    print(f"\nHash {hash_val}:")
    print(f"Number of users: {count}")
    print(f"Average distance from center: {np.mean(distances):.1f}m")
    print(f"Max distance from center: {np.max(distances):.1f}m")
    print(f"Center position: ({center[0]:.1f}, {center[1]:.1f})")
    print("Example path sequence (from first user):")
    for pid, seq in path_sequence:
        print(f"  Path ID {pid}: {seq}")

#%% What is the percentage of power in the first X paths?

def calculate_power_percentage(dataset, first_n_paths):
    """
    Calculate the percentage of total power contained in the first N paths for each user.
    
    Args:
        dataset: DeepMIMO dataset
        first_n_paths: Number of first paths to consider
        
    Returns:
        percentages: Array of power percentages for each user
    """
    # Convert powers from dBW to linear scale
    powers_linear = 10 ** (dataset.power / 10)  # Watts
    
    # Calculate total power per user (sum across all paths)
    total_power = np.nansum(powers_linear, axis=1)
    
    # Calculate power in first N paths
    power_first_n = np.nansum(powers_linear[:, :first_n_paths], axis=1)
    
    # Calculate percentage (avoiding division by zero)
    valid_users = total_power > 0
    percentages = np.zeros_like(total_power)
    percentages[valid_users] = (power_first_n[valid_users] / total_power[valid_users]) * 100
    
    return percentages

# Create coverage maps for different numbers of paths
max_paths_to_analyze = 5

for i in range(max_paths_to_analyze):
    n_paths = i + 1
    
    # Calculate power percentages
    power_percentages = calculate_power_percentage(dataset, n_paths)
    
    dataset.plot_coverage(power_percentages, dpi=300)
    plt.show()

# TODO: Measure the Channel NMSE of trimming the paths or not.
#       If the loss is small (say < -20 dB NMSE), then we can trim the paths.
#       (99.9% of the power is in the first 3 paths)

#%% PATH INTERPOLATION FOR ONE PAIR OF USERS

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

#%% Generate all linear sequences in a scenario

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
    
#%% Make video of all sequences

folder = 'sweeps'
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
            idxs_filtered = idxs[arr]
            plt.scatter(dataset.rx_pos[idxs_filtered, 0], 
                        dataset.rx_pos[idxs_filtered, 1], color='red', s=.5)
        
        plt.savefig(f'{folder}/asu_campus_3p5_{row_or_col}_{k:04d}.png', 
                    bbox_inches='tight', dpi=200)
        plt.close()
        # break

import subprocess

subprocess.run([
    "ffmpeg", "-y",
    "-framerate", "60",
    "-pattern_type", "glob",
    "-i", f"{folder}/*.png",
    "-vf", "crop=in_w:in_h-mod(in_h\\,2)",
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    f"{folder}/output_60fps.mp4"
])

#%% Create all sequences

def get_all_sequences(dataset: dm.Dataset, min_len: int = 1) -> list[np.ndarray]:
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

all_seqs = get_all_sequences(dataset, min_len=1)

# Print statistics
sum_len_seqs = sum([len(seq) for seq in all_seqs])
avg_len_seqs = sum_len_seqs / len(all_seqs)

print(f"Number of sequences: {len(all_seqs)}")
print(f"Average length of sequences: {avg_len_seqs:.1f}")

print(f"Number of active users: {len(dataset.get_active_idxs())}")
print(f"Total length of sequences: {sum_len_seqs}")


#%%

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

#%%
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

#%% plot interpolation results

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

#%%
# Compare an example variable (power of first path) before/after interpolation
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

#%%


# TODO: Take H_seq and select 100k sequences of L length. This can be done. 
# 

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



# If we have sequences 


#%%


    
