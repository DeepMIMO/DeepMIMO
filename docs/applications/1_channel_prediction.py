"""
# Channel Prediction: Path Interpolation & Doppler Effects

This comprehensive example demonstrates how to create realistic channel sequences for 
channel prediction tasks. We cover a complete progression from simple concepts to 
production-ready workflows.

## What You"ll Learn

1. **Simple Two-User Interpolation**: Understanding the basics of path interpolation
2. **Extracting All Linear Sequences**: Finding all consecutive user paths in a scenario
3. **Generating Sequence Videos**: Visualizing spatial coverage (optional)
4. **Creating Uniform-Length Sequences**: Preparing data for ML models
5. **Baseline Channels (No Interpolation)**: Computing channels from raw RT data
6. **Interpolated Channels**: Generating smoother channel sequences
7. **Interpolation + Doppler**: Adding realistic mobility effects

## Why Channel Prediction?

Channel prediction involves forecasting future wireless channel states based on past 
observations. This is crucial for:

- **Beamforming**: Proactive beam steering in mobile scenarios
- **Resource Allocation**: Optimizing spectrum usage based on predicted conditions
- **Handover Management**: Anticipating channel degradation
- **Link Adaptation**: Adjusting modulation and coding schemes preemptively

## The Progression

This example shows: **baseline → interpolation → interpolation + Doppler**, with 
visualizations comparing each stage to understand the impact of each technique.
"""

# %% [markdown]
"""
## When to Use Interpolation

**Use interpolation when:**
- Ray tracing (RT) data is sparse (large gaps between samples)
- You need smooth, continuous channel evolution
- Training data requires dense temporal sampling
- Physical user movement is continuous (not discrete jumps)

**Skip interpolation when:**
- RT data is already dense enough for your needs
- You want to preserve exact RT simulation results
- Computational cost is critical
- Studying discrete position scenarios

## When to Include Doppler

**Use Doppler when:**
- Modeling mobile users (vehicles, pedestrians, drones)
- Channel varies significantly within your prediction horizon
- Time-varying behavior is critical to your application
- Training models for real-world deployment with mobility

**Skip Doppler when:**
- Users are static or quasi-static
- Prediction horizon is very short
- Studying spatial diversity only
- Doppler effects are negligible for your carrier frequency and speeds
"""

# %% Imports

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import deepmimo as dm

# %% [markdown]
"""
## Configuration Parameters

Below we set up the main parameters for this example. These can be adjusted based on 
your specific needs:

- **RT_SCENARIO**: Ray tracing scenario to use
- **MAX_PATHS**: Limit number of multipath components
- **NT, NR**: Number of transmit and receive antennas
- **SEQ_LENGTH**: Target sequence length for channel prediction
- **INTERP_FACTOR**: How many interpolated points between RT samples
- **MAX_DOPPLER_HZ**: Maximum Doppler shift for mobility modeling
"""

# %% Configuration

# Scenario and data parameters
RT_SCENARIO = "asu_campus_3p5"
MAX_PATHS = 3  # Limit number of paths for simplicity

# Antenna configuration
NT = 2  # Number of transmit antennas
NR = 1  # Number of receive antennas
N_SUBCARRIERS = 1  # OFDM subcarriers

# Sequence parameters
SEQ_LENGTH = 60  # Target sequence length for channel prediction
N_SEQUENCES_SAMPLE = 100  # Number of sequences to generate for demonstration
INTERP_FACTOR = 10  # Points per segment for interpolation

# Doppler parameters
MAX_DOPPLER_HZ = 100  # Maximum Doppler shift [Hz]
TIME_DELTA = 1e-3  # Time between samples [s]

# Visualization
SEED = 42
np.random.seed(SEED)

print("Configuration:")
print(f"  Scenario: {RT_SCENARIO}")
print(f"  Antennas: {NT}x{NR}")
print(f"  Sequence length: {SEQ_LENGTH}")
print(f"  Interpolation factor: {INTERP_FACTOR}")
print(f"  Max Doppler: {MAX_DOPPLER_HZ} Hz")

# %% [markdown]
"""
## Load Ray Tracing Dataset

We load the DeepMIMO ray tracing dataset which contains channel parameters for multiple 
user positions. This includes:

- **rx_pos**: Receiver positions in 3D space
- **power, phase, delay**: Per-path channel parameters
- **aoa_az, aod_az, aoa_el, aod_el**: Angle of arrival/departure information
- **inter**: Interaction types (reflection, diffraction, etc.)
"""

# %% Load Dataset

matrices = ["rx_pos", "tx_pos", "aoa_az", "aod_az", "aoa_el", "aod_el",
            "delay", "power", "phase", "inter", "inter_pos"]

dataset = dm.load(RT_SCENARIO, max_paths=MAX_PATHS, matrices=matrices)

print("\nDataset loaded:")
print(f"  Total users: {dataset.n_ue}")
print(f"  Active users: {len(dataset.get_idxs('active'))}")
print(f"  Grid size: {dataset.grid_size}")
print(f"  TX position: {dataset.tx_pos}")

# %% [markdown]
"""
---
# SECTION 1: Simple Two-User Interpolation

We start with the simplest case: interpolating between just two users. This helps 
understand the interpolation concept before scaling up.

## The Interpolation Concept

Linear interpolation computes intermediate values between two known points. For wireless 
channels, we interpolate:
- **Positions**: Physical location in space
- **Channel parameters**: Power, phase, delay, angles

This creates a smooth transition as a user moves from point A to point B.
"""

# %% Section 1: Simple Two-User Interpolation

print("\n" + "="*70)
print("SECTION 1: Simple Two-User Interpolation")
print("="*70)

def interpolate_percentage(array1, array2, percents):
    """Interpolate between two arrays at specified percentages.
    
    Args:
        array1: Starting array/value
        array2: Ending array/value
        percents: Array of percentages between 0 and 1
        
    Returns:
        np.ndarray: Interpolated values at given percentages
        
    Example:
        >>> a = np.array([0, 0, 0])
        >>> b = np.array([10, 10, 10])
        >>> interpolate_percentage(a, b, np.array([0, 0.5, 1.0]))
        array([[0, 0, 0], [5, 5, 5], [10, 10, 10]])
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
        
    Example:
        >>> params = interpolate_path(dataset, 10, 11, [0, 0.5, 1.0])
        >>> params["rx_pos"].shape  # (3, 3) - 3 points, 3D positions
    """
    # Get total distance for percentage calculation
    pos1 = dataset.rx_pos[idx_1]
    pos2 = dataset.rx_pos[idx_2]
    total_distance = np.linalg.norm(pos2 - pos1)
    
    # Convert distances to percentages
    percentages = np.clip(distances / total_distance, 0, 1)
    
    # Interpolate all relevant parameters
    params = {}
    params_to_interpolate = ["rx_pos", "power", "phase", "delay", 
                             "aoa_az", "aod_az", "aoa_el", "aod_el"]
    
    for param in params_to_interpolate:
        if dataset[param] is None:
            params[param] = None
            continue
        val1 = dataset[param][idx_1]
        val2 = dataset[param][idx_2]
        params[param] = interpolate_percentage(val1, val2, percentages)
    
    return params


# Demonstrate simple two-user interpolation
idx_1 = 10
idx_2 = 11

# Visualize the two users
print(f"\nInterpolating between users {idx_1} and {idx_2}:")
dataset.plot_rays(idx_1, proj_3D=False)
dataset.plot_rays(idx_2, proj_3D=False)

# Print channel information
dataset.print_rx(idx_1, path_idxs=[0])
dataset.print_rx(idx_2, path_idxs=[0])

# Interpolate at specific distances
distances = np.array([0, 0.3, 0.6, 0.9, 1.2])  # meters
params = interpolate_path(dataset, idx_1, idx_2, distances)

print(f"\nInterpolated {len(distances)} points between users {idx_1} and {idx_2}")
print(f"Original distance: {np.linalg.norm(dataset.rx_pos[idx_2] - dataset.rx_pos[idx_1]):.2f} m")

# Visualize interpolated positions
plt.figure(figsize=(10, 6), dpi=150)
plt.plot([dataset.rx_pos[idx_1, 0], dataset.rx_pos[idx_2, 0]],
         [dataset.rx_pos[idx_1, 1], dataset.rx_pos[idx_2, 1]],
         "o-", markersize=10, label="Original users", linewidth=2)
plt.plot(params["rx_pos"][:, 0], params["rx_pos"][:, 1],
         "x--", markersize=8, label="Interpolated points", linewidth=1)
plt.gca().set_aspect("equal", adjustable="box")
plt.title("Two-User Path Interpolation: Position")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# Visualize interpolated power
plt.figure(figsize=(10, 6), dpi=150)
for path_idx in range(min(3, params["power"].shape[1])):
    # Original power
    plt.plot([0, len(distances)-1], 
             [dataset.power[idx_1, path_idx], dataset.power[idx_2, path_idx]], 
             "o-", markersize=10, label=f"Original (path {path_idx})", linewidth=2)
    # Interpolated power
    plt.plot(range(len(distances)), params["power"][:, path_idx], 
             "x--", markersize=8, label=f"Interpolated (path {path_idx})", linewidth=1)
plt.title("Two-User Path Interpolation: Power")
plt.xlabel("Sample index")
plt.ylabel("Power [dBW]")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
"""
---
# SECTION 2: Extract All Linear Sequences

Now we scale up: find all consecutive active user paths in the entire scenario. These 
sequences represent natural user trajectories along rows and columns of the grid.

## Why Extract Sequences?

In a ray tracing grid, not all positions have active users (some may be blocked by 
buildings). We want to find:
- **Consecutive active users**: Users that form a continuous path
- **Row and column sweeps**: Natural linear trajectories
- **Minimum length**: Filter out sequences too short for prediction

This gives us realistic mobility patterns for training channel prediction models.
"""

# %% Section 2: Extract All Linear Sequences

print("\n" + "="*70)
print("SECTION 2: Extract All Linear Sequences")
print("="*70)


def get_consecutive_active_segments(dataset: dm.Dataset, idxs: np.ndarray,
                                    min_len: int = 1) -> list[np.ndarray]:
    """Get consecutive segments of active users from a set of indices.
    
    Args:
        dataset: DeepMIMO dataset
        idxs: Array of user indices to check
        min_len: Minimum length of consecutive segments to keep
        
    Returns:
        List of arrays containing consecutive active user indices
        
    Example:
        >>> # For a row with active users at indices [5,6,7,10,11,20]
        >>> # Returns: [[5,6,7], [10,11], [20]] (if min_len=1)
        >>> # Returns: [[5,6,7], [10,11]] (if min_len=2)
    """
    active_idxs = np.where(dataset.los[idxs] != -1)[0]
    
    # Split active_idxs into arrays of consecutive indices
    splits = np.where(np.diff(active_idxs) != 1)[0] + 1
    consecutive_arrays = np.split(active_idxs, splits)
    
    # Filter by minimum length
    consecutive_arrays = [idxs[arr] for arr in consecutive_arrays if len(arr) > min_len]
    
    return consecutive_arrays


def get_all_sequences(dataset: dm.Dataset, min_len: int = 1) -> list[np.ndarray]:
    """Extract all consecutive active user sequences from a dataset.
    
    For each row and column in the dataset grid, finds all consecutive segments
    of active users with length at least min_len.
    
    Args:
        dataset: The dataset object with grid structure
        min_len: Minimum length of a segment to include
        
    Returns:
        List of arrays, each containing indices of a consecutive active user segment
        
    Example:
        >>> all_seqs = get_all_sequences(dataset, min_len=5)
        >>> print(f"Found {len(all_seqs)} sequences")
        >>> print(f"Average length: {np.mean([len(s) for s in all_seqs]):.1f}")
    """
    n_cols, n_rows = dataset.grid_size
    all_seqs = []
    
    # Process each row
    for k in range(n_rows):
        idxs = dataset.get_idxs("row", row_idxs=k)
        consecutive_arrays = get_consecutive_active_segments(dataset, idxs, min_len)
        all_seqs += consecutive_arrays
    
    # Process each column
    for k in range(n_cols):
        idxs = dataset.get_idxs("col", col_idxs=k)
        consecutive_arrays = get_consecutive_active_segments(dataset, idxs, min_len)
        all_seqs += consecutive_arrays
    
    return all_seqs


# Extract all sequences from the dataset
MIN_SEQ_LEN = 10  # Minimum length for a sequence to be useful
all_seqs = get_all_sequences(dataset, min_len=MIN_SEQ_LEN)

# Print statistics
seq_lens = [len(seq) for seq in all_seqs]
sum_len_seqs = sum(seq_lens)
avg_len_seqs = sum_len_seqs / len(all_seqs)

print(f"\nSequence extraction results:")
print(f"  Number of sequences: {len(all_seqs)}")
print(f"  Average length: {avg_len_seqs:.1f} users")
print(f"  Min length: {min(seq_lens)} users")
print(f"  Max length: {max(seq_lens)} users")
print(f"  Total user instances: {sum_len_seqs}")

# Visualize sequence length distribution
plt.figure(figsize=(10, 6), dpi=150)
plt.hist(seq_lens, bins=50, edgecolor="black", alpha=0.7)
plt.axvline(avg_len_seqs, color="r", linestyle="--", linewidth=2, label=f"Mean: {avg_len_seqs:.1f}")
plt.xlabel("Sequence length (number of users)")
plt.ylabel("Number of sequences")
plt.title("Distribution of Sequence Lengths")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# Visualize a few example sequences on the map
plt.figure(figsize=(12, 8), dpi=150)
dataset.los.plot()
n_plot = min(10, len(all_seqs))
for i in range(n_plot):
    seq = all_seqs[i]
    plt.plot(dataset.rx_pos[seq, 0], dataset.rx_pos[seq, 1], 
             "o-", markersize=3, linewidth=2, label=f"Seq {i} ({len(seq)} users)")
plt.title(f"Example Sequences (showing {n_plot} of {len(all_seqs)})")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
"""
---
# SECTION 3: Generate Sequence Video (Optional)

Create a video showing all sequences being traced in the scenario. This helps visualize 
the spatial coverage and understand the geometry of user trajectories.

**Note**: This is computationally expensive and requires ffmpeg. It"s commented out by 
default but available if you need to create visualizations for presentations or papers.

## What the Video Shows

- Each frame shows one row or column
- Red dots highlight consecutive active user segments
- Helps identify coverage gaps and trajectory patterns
"""

# %% Section 3: Generate Sequence Video (Optional)

print("\n" + "="*70)
print("SECTION 3: Generate Sequence Video (Optional)")
print("="*70)

import os
import subprocess


def make_sequence_video(dataset, folder="sweeps", ffmpeg_fps=60):
    """Generate a video visualizing all row/col user sequences in the dataset.
    
    For each row and column, plots the consecutive active user segments and saves as PNGs.
    Then uses ffmpeg to combine the PNGs into a video.
    
    Args:
        dataset: DeepMIMO dataset object
        folder: Output folder for PNGs and video
        ffmpeg_fps: Framerate for the output video
        
    Example:
        >>> make_sequence_video(dataset, folder="sequence_sweeps", ffmpeg_fps=30)
        >>> # Creates: sequence_sweeps/output_30fps.mp4
    """
    os.makedirs(folder, exist_ok=True)
    n_cols, n_rows = dataset.grid_size
    
    for row_or_col in ["row", "col"]:
        n_iter = n_rows if row_or_col == "row" else n_cols
        for k in tqdm(range(n_iter), desc=f"Processing {row_or_col}s"):
            idxs = dataset.get_idxs(row_or_col, **{f"{row_or_col}_idxs": k})
            consecutive_arrays = get_consecutive_active_segments(dataset, idxs)
            
            # Plot the scenario
            dataset.los.plot()
            
            # Highlight consecutive segments
            for i, arr in enumerate(consecutive_arrays):
                plt.scatter(dataset.rx_pos[arr, 0], 
                           dataset.rx_pos[arr, 1], 
                           color="red", s=2, zorder=10)
            
            plt.title(f"{row_or_col.capitalize()} {k}: {len(consecutive_arrays)} segments")
            plt.savefig(f"{folder}/{RT_SCENARIO}_{row_or_col}_{k:04d}.png", 
                       bbox_inches="tight", dpi=200)
            plt.close()
    
    # Create video from PNGs using ffmpeg
    print(f"\nCreating video with ffmpeg...")
    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-framerate", str(ffmpeg_fps),
            "-pattern_type", "glob",
            "-i", f"{folder}/*.png",
            "-vf", "crop=in_w:in_h-mod(in_h\\,2)",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            f"{folder}/output_{ffmpeg_fps}fps.mp4"
        ], check=True)
        print(f"Video created: {folder}/output_{ffmpeg_fps}fps.mp4")
    except subprocess.CalledProcessError:
        print("Failed to create video. Make sure ffmpeg is installed.")
    except FileNotFoundError:
        print("ffmpeg not found. Install ffmpeg to create videos.")


# Uncomment to generate video (takes several minutes)
# make_sequence_video(dataset, folder="sequence_sweeps", ffmpeg_fps=60)

print("\nVideo generation is optional and commented out by default.")
print("Uncomment the make_sequence_video() call above to generate a video.")

# %% [markdown]
"""
---
# SECTION 4: Create Uniform-Length Sequences

ML models need fixed-length inputs. We use a sliding window approach to convert 
variable-length sequences into uniform windows.

## Sliding Window Strategy

Given a sequence of length N and target length L:
- Extract windows: [0:L], [1:L+1], [2:L+2], ..., [N-L:N]
- Stride controls overlap (stride=1 for maximum overlap, higher for less)
- Sequences shorter than L are dropped

This maximizes the training data while maintaining temporal structure.
"""

# %% Section 4: Create Uniform-Length Sequences

print("\n" + "="*70)
print("SECTION 4: Create Uniform-Length Sequences")
print("="*70)


def expand_to_uniform_sequences(
    sequences: list[np.ndarray] | np.ndarray,
    target_len: int,
    stride: int = 1
) -> np.ndarray:
    """Convert variable-length sequences into fixed-length windows using sliding window.
    
    For each input sequence, extracts all possible contiguous subsequences (windows)
    of length target_len using a sliding window with the specified stride.
    Sequences shorter than target_len are dropped.
    
    Args:
        sequences: List of 1D arrays or 2D array where each element/row is a sequence
        target_len: Desired length of each output window
        stride: Step size for the sliding window (default: 1 for maximum overlap)
        
    Returns:
        2D array of shape (n_windows, target_len), where each row is a window
        
    Example:
        >>> seqs = [np.array([0,1,2,3,4]), np.array([10,11,12,13])]
        >>> windows = expand_to_uniform_sequences(seqs, target_len=3, stride=1)
        >>> # Returns: [[0,1,2], [1,2,3], [2,3,4], [10,11,12], [11,12,13]]
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


# For this demo, we"ll use a shorter sequence length before interpolation
# After interpolation with INTERP_FACTOR, this will expand to SEQ_LENGTH
PRE_INTERP_SEQ_LEN = max(SEQ_LENGTH // INTERP_FACTOR + 1, 2)  # min 2 for interpolation

print(f"\nCreating uniform sequences:")
print(f"  Pre-interpolation length: {PRE_INTERP_SEQ_LEN} users")
print(f"  Post-interpolation length: {(PRE_INTERP_SEQ_LEN-1) * INTERP_FACTOR + 1} points")

# Expand sequences to uniform length
all_seqs_mat = expand_to_uniform_sequences(all_seqs, target_len=PRE_INTERP_SEQ_LEN, stride=1)
print(f"  Generated {len(all_seqs_mat)} uniform-length sequences")

# Sample a subset for demonstration
final_samples = min(N_SEQUENCES_SAMPLE, len(all_seqs_mat))
np.random.seed(SEED)
sample_idxs = np.random.choice(len(all_seqs_mat), final_samples, replace=False)
all_seqs_mat_sample = all_seqs_mat[sample_idxs]

print(f"  Sampled {final_samples} sequences for demonstration")
print(f"  Shape: {all_seqs_mat_sample.shape}")

# %% [markdown]
"""
---
# SECTION 5: Baseline Channels (No Interpolation)

First, we generate channels directly from the ray tracing data without any interpolation. 
This serves as a baseline for comparison.

## Why Start with Baseline?

Comparing baseline (no interpolation) vs interpolated vs Doppler helps us understand:
- The smoothness benefit of interpolation
- The time-varying effects of Doppler
- Trade-offs between accuracy and computational cost

We"ll use the same sequences throughout to ensure fair comparison.
"""

# %% Section 5: Baseline Channels (No Interpolation)

print("\n" + "="*70)
print("SECTION 5: Baseline Channels (No Interpolation)")
print("="*70)

# Select a few sequences for detailed comparison
N_COMPARE = min(3, final_samples)
compare_seqs = all_seqs_mat_sample[:N_COMPARE]

# Setup channel parameters
ch_params = dm.ChannelParameters()
ch_params.bs_antenna.shape = [NT, 1]
ch_params.ue_antenna.shape = [NR, 1]
ch_params.ofdm.subcarriers = N_SUBCARRIERS
ch_params.ofdm.selected_subcarriers = np.arange(N_SUBCARRIERS)
ch_params.ofdm.bandwidth = 15e3 * N_SUBCARRIERS  # [Hz]
ch_params.doppler = False  # No Doppler for baseline

print(f"\nChannel parameters:")
print(f"  TX antennas: {NT}")
print(f"  RX antennas: {NR}")
print(f"  Subcarriers: {N_SUBCARRIERS}")
print(f"  Bandwidth: {ch_params.ofdm.bandwidth/1e3:.1f} kHz")

# Compute channels for baseline sequences
print(f"\nComputing baseline channels for {N_COMPARE} sequences...")
H_baseline_list = []

for seq_idx in range(N_COMPARE):
    sequence = compare_seqs[seq_idx]
    
    # Create mini-dataset with just these users
    mini_dataset = dm.Dataset({"n_ue": len(sequence)})
    mini_dataset.tx_pos = dataset.tx_pos
    
    # Copy scene information if available (needed for Doppler computation)
    for param in ["scene", "materials", "load_params", "rt_params"]:
        if hasattr(dataset, param):
            mini_dataset[param] = getattr(dataset, param)
    
    for field in ["rx_pos", "power", "phase", "delay", "aoa_az", "aod_az", "aoa_el", "aod_el"]:
        mini_dataset[field] = dataset[field][sequence]
    
    # Compute channels
    H = mini_dataset.compute_channels(ch_params)
    H_baseline_list.append(H[:, :, :, 0])  # Remove subcarrier dimension
    
H_baseline = np.stack(H_baseline_list, axis=0)
print(f"Baseline channels shape: {H_baseline.shape}")  # (N_COMPARE, seq_len, NR, NT)

# Visualize baseline channels
for seq_idx in range(N_COMPARE):
    sequence = compare_seqs[seq_idx]
    
    plt.figure(figsize=(12, 5), dpi=150)
    
    # Plot positions
    plt.subplot(1, 2, 1)
    plt.plot(dataset.rx_pos[sequence, 0], dataset.rx_pos[sequence, 1], 
             "o-", markersize=8, linewidth=2)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(f"Baseline Sequence {seq_idx}: User Positions")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.grid(True, alpha=0.3)
    
    # Plot channel magnitude over time
    plt.subplot(1, 2, 2)
    for rx_idx in range(NR):
        for tx_idx in range(NT):
            h_magnitude = np.abs(H_baseline[seq_idx, :, rx_idx, tx_idx])
            plt.plot(h_magnitude, "o-", markersize=4, 
                    label=f"RX{rx_idx+1}-TX{tx_idx+1}")
    plt.title(f"Baseline Sequence {seq_idx}: Channel Magnitude")
    plt.xlabel("User index in sequence")
    plt.ylabel("|H|")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# %% [markdown]
"""
---
# SECTION 6: Interpolated Channels

Now we generate channels with interpolation. This creates smoother channel sequences by 
adding intermediate points between RT samples.

## How Interpolation Works

For each segment (pair of consecutive RT samples):
1. Interpolate positions: Create INTERP_FACTOR points between them
2. Interpolate channel parameters: Power, phase, delay, angles
3. Stack all interpolated segments into a new dataset
4. Compute channels for all interpolated points

## Expected Benefits

- **Smoother trajectories**: Continuous channel evolution
- **Denser sampling**: Better temporal resolution for prediction
- **More training data**: INTERP_FACTOR × more samples per sequence
"""

# %% Section 6: Interpolated Channels

print("\n" + "="*70)
print("SECTION 6: Interpolated Channels")
print("="*70)


def interpolate_dataset_from_seqs(
    dataset: dm.Dataset | dm.MacroDataset,
    sequences: np.ndarray,
    step_meters: float | None = 0.5,
    points_per_segment: int | None = None
) -> dm.Dataset:
    """Create a new Dataset by interpolating along each sequence of indices.
    
    Takes sequences of indices into a dataset and creates a new dataset by interpolating
    between consecutive points in each sequence.
    
    Interpolation modes:
    - Distance-based (step_meters): Points placed every step_meters along each segment
    - Count-based (points_per_segment): Fixed number of evenly-spaced points per segment
    
    Args:
        dataset: Source dataset containing the data to interpolate
        sequences: Array of shape [n_sequences, sequence_length] containing indices
        step_meters: Distance between interpolated points (or None to use points_per_segment)
        points_per_segment: Number of points per segment (or None to use step_meters)
        
    Returns:
        New Dataset containing interpolated data with shape [n_total_points, ...]
        
    Interpolated fields:
        - rx_pos: Receiver positions [n_points, 3]
        - power, phase, delay: Ray parameters [n_points, n_rays]
        - aoa_az, aod_az, aoa_el, aod_el: Angles [n_points, n_rays]
        - inter: Interaction types [n_points, n_rays] (copied from first point)
        - inter_pos: Interaction positions (if present)
        
    Example:
        >>> # Create 10 interpolated points between each pair of users
        >>> interp_ds = interpolate_dataset_from_seqs(dataset, sequences, 
        ...                                            points_per_segment=10)
        >>> print(f"Original: {len(sequences)} x {sequences.shape[1]} users")
        >>> print(f"Interpolated: {interp_ds.n_ue} points")
    """
    # Unwrap MacroDataset if necessary
    dataset = dataset.datasets[0] if isinstance(dataset, dm.MacroDataset) else dataset
    
    # Ensure ndarray of ints for sequences
    sequences = np.asarray(sequences, dtype=int)
    
    # Define arrays/fields to process
    ray_fields = ["rx_pos", "power", "phase", "delay", "aoa_az", "aod_az", "aoa_el", "aod_el"]
    interpolation_fields = ray_fields + (["inter_pos"] if getattr(dataset, "inter_pos", None) is not None else [])
    replication_fields = ["inter"] if getattr(dataset, "inter", None) is not None else []
    
    # Prepare lists for all segments across all sequences
    start_idx_parts: list[np.ndarray] = []
    end_idx_parts: list[np.ndarray] = []
    t_parts: list[np.ndarray] = []
    
    rx_pos = dataset.rx_pos
    
    # Build flattened segment lists and interpolation weights
    for seq_idx in tqdm(range(sequences.shape[0]), desc="Preparing interpolation"):
        seq = np.asarray(sequences[seq_idx], dtype=int)
        if seq.size < 2:
            continue
        
        for k in range(seq.size - 1):
            i1 = int(seq[k])
            i2 = int(seq[k + 1])
            
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
            return field_array[0:0]
        a = field_array[start_idx]
        b = field_array[end_idx]
        # Reshape t for broadcasting
        ratio = t_all.reshape((-1,) + (1,) * (a.ndim - 1)).astype(a.dtype, copy=False)
        return a * (1.0 - ratio) + b * ratio
    
    concatenated_data: dict[str, np.ndarray] = {}
    
    # Interpolate all fields
    print("Interpolating fields...")
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
    for param in ["scene", "materials", "load_params", "rt_params"]:
        if hasattr(dataset, param):
            new_dataset_params[param] = getattr(dataset, param)
    
    new_dataset_params["n_ue"] = int(concatenated_data["rx_pos"].shape[0])
    new_dataset_params["parent_name"] = dataset.get("parent_name", dataset.name)
    new_dataset_params["name"] = f"{dataset.name}_interp"
    
    new_dataset = dm.Dataset(new_dataset_params)
    new_dataset.tx_pos = dataset.tx_pos
    
    # Assign all interpolated/replicated arrays
    for field in interpolation_fields + replication_fields:
        if field in concatenated_data:
            new_dataset[field] = concatenated_data[field]
    
    return new_dataset


# Create interpolated dataset
print(f"\nCreating interpolated dataset...")
print(f"  Interpolation factor: {INTERP_FACTOR} points per segment")
print(f"  Input: {len(compare_seqs)} sequences x {PRE_INTERP_SEQ_LEN} users")

interp_dataset = interpolate_dataset_from_seqs(
    dataset,
    compare_seqs,
    points_per_segment=INTERP_FACTOR
)

seq_out_len = (PRE_INTERP_SEQ_LEN - 1) * INTERP_FACTOR + 1
print(f"  Output: {interp_dataset.n_ue} interpolated points")
print(f"  Expected: {len(compare_seqs)} sequences x {seq_out_len} points = {len(compare_seqs) * seq_out_len}")

# Compute channels for interpolated dataset
ch_params.doppler = False  # Still no Doppler
print(f"\nComputing interpolated channels...")
H_interp_full = interp_dataset.compute_channels(ch_params)
H_interp = H_interp_full[:, :, :, 0]  # Remove subcarrier dimension
print(f"Interpolated channels shape: {H_interp.shape}")

# Adjust seq_out_len to actual interpolated length
actual_seq_out_len = interp_dataset.n_ue // N_COMPARE
print(f"Actual sequence output length: {actual_seq_out_len}")

# Reshape to separate sequences
H_interp_seq = H_interp.reshape(N_COMPARE, actual_seq_out_len, NR, NT)
print(f"Reshaped to: {H_interp_seq.shape}")  # (N_COMPARE, seq_out_len, NR, NT)

# %% [markdown]
"""
## Comparison: Baseline vs Interpolated

Now let"s visualize the differences. For each sequence we"ll compare:
1. **Positions**: Original RT points vs interpolated points
2. **Power**: Channel parameter interpolation
3. **Channel Magnitude**: Baseline (sparse) vs Interpolated (dense)

Notice how interpolation creates smooth transitions between RT samples.
"""

# %% Visualize Comparison: Baseline vs Interpolated

# Visualize comparison: baseline vs interpolated
for seq_idx in range(N_COMPARE):
    sequence = compare_seqs[seq_idx]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    
    # Original positions
    start = seq_idx * actual_seq_out_len
    end = start + actual_seq_out_len
    interp_slice = slice(start, end)
    
    # Top left: Positions
    axes[0, 0].plot(dataset.rx_pos[sequence, 0], dataset.rx_pos[sequence, 1], 
                    "o-", markersize=10, linewidth=2, label="Original (RT)")
    axes[0, 0].plot(interp_dataset.rx_pos[interp_slice, 0], 
                    interp_dataset.rx_pos[interp_slice, 1], 
                    "x-", markersize=4, linewidth=1, alpha=0.7, label="Interpolated")
    axes[0, 0].set_aspect("equal", adjustable="box")
    axes[0, 0].set_title(f"Sequence {seq_idx}: User Positions")
    axes[0, 0].set_xlabel("X [m]")
    axes[0, 0].set_ylabel("Y [m]")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Top right: Power (first path)
    orig_sample_indices = np.arange(len(sequence)) * INTERP_FACTOR
    axes[0, 1].plot(orig_sample_indices, dataset.power[sequence, 0], 
                    "o-", markersize=8, linewidth=2, label="Original (RT)")
    axes[0, 1].plot(np.arange(actual_seq_out_len), interp_dataset.power[interp_slice, 0], 
                    "x-", markersize=4, linewidth=1, alpha=0.7, label="Interpolated")
    axes[0, 1].set_title(f"Sequence {seq_idx}: Power (Path 0)")
    axes[0, 1].set_xlabel("Sample index")
    axes[0, 1].set_ylabel("Power [dBW]")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Bottom left: Baseline channel magnitude
    for rx_idx in range(NR):
        for tx_idx in range(NT):
            h_mag = np.abs(H_baseline[seq_idx, :, rx_idx, tx_idx])
            axes[1, 0].plot(orig_sample_indices, h_mag, "o-", markersize=8, 
                          linewidth=2, label=f"RX{rx_idx+1}-TX{tx_idx+1}")
    axes[1, 0].set_title(f"Baseline Channel Magnitude")
    axes[1, 0].set_xlabel("Sample index")
    axes[1, 0].set_ylabel("|H|")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Bottom right: Interpolated channel magnitude
    for rx_idx in range(NR):
        for tx_idx in range(NT):
            h_mag = np.abs(H_interp_seq[seq_idx, :, rx_idx, tx_idx])
            axes[1, 1].plot(h_mag, "x-", markersize=4, linewidth=1, 
                          alpha=0.7, label=f"RX{rx_idx+1}-TX{tx_idx+1}")
    axes[1, 1].set_title(f"Interpolated Channel Magnitude")
    axes[1, 1].set_xlabel("Sample index")
    axes[1, 1].set_ylabel("|H|")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()

# %% [markdown]
"""
---
# SECTION 7: Interpolation + Doppler

Finally, we add Doppler effects to create realistic time-varying channels. Doppler shift 
occurs when there"s relative motion between transmitter and receiver, causing the carrier 
frequency to shift.

## Doppler Shift Formula

For a mobile user with velocity **v** and angle θ relative to the wave propagation:

$$f_d = \\frac{v}{\\lambda} \\cos(\\theta) = \\frac{v \\cdot f_c}{c} \\cos(\\theta)$$

Where:
- $f_d$ = Doppler shift [Hz]
- $v$ = velocity [m/s]
- $\\lambda$ = wavelength [m]
- $f_c$ = carrier frequency [Hz]
- $c$ = speed of light [m/s]
- $\\theta$ = angle between velocity and wave direction

## Doppler Configuration Methods

We"ll demonstrate multiple approaches:

1. **Uniform Doppler** (simplest): Same Doppler for all users/paths
2. **Constant Velocity**: User has constant velocity vector (more realistic)
3. **Geometry-Derived**: Direction from path geometry, constant speed (most realistic)

We"ll compare channels with different Doppler values: 0, 50, 100, 200 Hz
"""

# %% Section 7: Interpolation + Doppler

print("\n" + "="*70)
print("SECTION 7: Interpolation + Doppler Effects")
print("="*70)


def plot_iq_constellation(H: np.ndarray, sample_idx: int | None = None, 
                         rx_idx: int | None = None, title_suffix: str = ""):
    """Plot IQ constellation diagram for channel gains.
    
    Args:
        H: Channel array of shape (n_samples, n_rx, n_tx, n_time_steps), complex
        sample_idx: Sample index to plot (random if None)
        rx_idx: RX antenna index to plot (random if None)
        title_suffix: Additional text for title
        
    Returns:
        Tuple of (sample_idx, rx_idx) used for plotting
    """
    i = np.random.randint(H.shape[0]) if sample_idx is None else sample_idx
    r = np.random.randint(H.shape[1]) if rx_idx is None else rx_idx
    
    plt.figure(figsize=(8, 8), dpi=150)
    lim = 0.0
    for t in range(H.shape[2]):
        z = H[i, r, t, :]  # complex time series
        lim = max(lim, np.max(np.abs(z)))
        plt.plot(z.real, z.imag, "o-", markersize=4, linewidth=1, 
                label=f"TX antenna {t+1}")
    
    lim = float(lim) * 1.1
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True, alpha=0.3)
    plt.xlabel("In-Phase")
    plt.ylabel("Quadrature")
    plt.legend()
    plt.title(f"IQ Constellation: Sample {i}, RX {r+1}{title_suffix}")
    plt.tight_layout()
    plt.show()
    
    return i, r


# Test different Doppler configurations
doppler_configs = [
    {"name": "No Doppler", "doppler": False, "max_doppler": 0},
    {"name": "50 Hz Uniform", "doppler": True, "max_doppler": 50},
    {"name": "100 Hz Uniform", "doppler": True, "max_doppler": 100},
    {"name": "200 Hz Uniform", "doppler": True, "max_doppler": 200},
]

H_doppler_results = {}

for config in doppler_configs:
    print(f"\n--- {config['name']} ---")
    
    # Create fresh interpolated dataset for each config
    interp_dataset_doppler = interpolate_dataset_from_seqs(
        dataset,
        compare_seqs,
        points_per_segment=INTERP_FACTOR
    )
    
    # Configure Doppler
    ch_params_doppler = dm.ChannelParameters()
    ch_params_doppler.bs_antenna.shape = [NT, 1]
    ch_params_doppler.ue_antenna.shape = [NR, 1]
    ch_params_doppler.ofdm.subcarriers = N_SUBCARRIERS
    ch_params_doppler.ofdm.selected_subcarriers = np.arange(N_SUBCARRIERS)
    ch_params_doppler.ofdm.bandwidth = 15e3 * N_SUBCARRIERS
    ch_params_doppler.doppler = config["doppler"]
    
    if config["doppler"]:
        # Way 1: Uniform Doppler (same for all users/paths)
        # Mean absolute alignment = 1/2 * MAX_DOPPLER
        interp_dataset_doppler.set_doppler(config["max_doppler"] / 2)
        print(f"  Set uniform Doppler: {config['max_doppler']/2:.1f} Hz")
    
    # Compute channels with time evolution
    times = np.arange(actual_seq_out_len) * TIME_DELTA
    print(f"  Computing channels at {len(times)} time steps...")
    print(f"  Time span: {times[-1]*1000:.1f} ms")
    
    H_full = interp_dataset_doppler.compute_channels(ch_params_doppler, times=times)
    print(f"  H shape: {H_full.shape}")  # (n_users, NR, NT, N_SUBCARRIERS, n_times)
    
    # Extract time-varying channels for each sequence
    H_seq_time = np.zeros((N_COMPARE, actual_seq_out_len, NR, NT), dtype=np.complex64)
    
    for seq_idx in range(N_COMPARE):
        for time_idx in range(actual_seq_out_len):
            user_idx = seq_idx * actual_seq_out_len + time_idx
            H_seq_time[seq_idx, time_idx] = H_full[user_idx, :, :, 0, time_idx]
    
    H_doppler_results[config["name"]] = H_seq_time
    print(f"  Final shape: {H_seq_time.shape}")  # (N_COMPARE, actual_seq_out_len, NR, NT)

# %% [markdown]
"""
## Doppler Effects: IQ Constellation Analysis

IQ (In-phase/Quadrature) constellation diagrams show the complex channel gain in the 
complex plane. As the user moves and Doppler affects the channel, the constellation 
traces a path over time.

**What to observe:**
- **No Doppler**: Smooth, static constellation points
- **With Doppler**: Points rotate around the origin over time
- **Higher Doppler**: Faster rotation, more dynamic behavior

This rotation represents the time-varying phase shift caused by user motion.
"""

# %% Visualize IQ Constellations

print("\n" + "="*70)
print("Visualizing Doppler Effects: IQ Constellations")
print("="*70)

# Compare IQ constellations for different Doppler settings
sample_idx = 0  # Use first sequence
rx_idx = 0  # Use first RX antenna

for config_name, H_result in H_doppler_results.items():
    # Transform to format expected by plot function: (n_samples, n_rx, n_tx, n_time)
    H_plot = np.transpose(H_result, (0, 2, 3, 1))
    plot_iq_constellation(H_plot, sample_idx=sample_idx, rx_idx=rx_idx, 
                         title_suffix=f" - {config_name}")

# %% [markdown]
"""
## Doppler Effects: Phase Evolution

Let"s examine how Doppler affects the channel magnitude and phase over time. The phase 
evolution is particularly telling:

- **No Doppler**: Phase changes only due to spatial variation
- **With Doppler**: Additional phase rotation proportional to Doppler shift
- **Phase rate**: $\\frac{d\\phi}{dt} = 2\\pi f_d$

Higher Doppler causes faster phase rotation, which is critical for channel prediction 
models to learn.
"""

# %% Visualize Phase Evolution

# Compare phase evolution over time
print("\nComparing phase evolution...")
for seq_idx in range(N_COMPARE):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    fig.suptitle(f"Sequence {seq_idx}: Doppler Effect Comparison", fontsize=14, fontweight="bold")
    
    for idx, (config_name, H_result) in enumerate(H_doppler_results.items()):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        # Plot magnitude and phase for first TX-RX pair
        h = H_result[seq_idx, :, 0, 0]
        magnitude = np.abs(h)
        phase = np.angle(h)
        
        # Create twin axis
        ax2 = ax.twinx()
        
        # Plot magnitude
        line1 = ax.plot(magnitude, "b-", linewidth=2, label="Magnitude")
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Magnitude |H|", color="b")
        ax.tick_params(axis="y", labelcolor="b")
        ax.grid(True, alpha=0.3)
        
        # Plot phase
        line2 = ax2.plot(phase, "r-", linewidth=2, alpha=0.7, label="Phase")
        ax2.set_ylabel("Phase [rad]", color="r")
        ax2.tick_params(axis="y", labelcolor="r")
        
        ax.set_title(config_name)
        
        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc="upper right")
    
    plt.tight_layout()
    plt.show()

# %% [markdown]
"""
## Doppler Effects: Magnitude Comparison

Finally, let"s compare channel magnitudes across all Doppler settings on one plot. This 
shows how Doppler affects the overall channel strength variability.

**Key observations:**
- Baseline patterns remain similar (spatial variation dominates)
- Doppler adds temporal variation on top of spatial
- Higher Doppler = more rapid fluctuations
"""

# %% Compare Channel Magnitudes

# Compare channel magnitudes across Doppler settings
for seq_idx in range(N_COMPARE):
    plt.figure(figsize=(12, 6), dpi=150)
    
    for config_name, H_result in H_doppler_results.items():
        h_magnitude = np.abs(H_result[seq_idx, :, 0, 0])
        plt.plot(h_magnitude, linewidth=2, marker="o", markersize=3, 
                label=config_name, alpha=0.8)
    
    plt.title(f"Sequence {seq_idx}: Channel Magnitude Comparison")
    plt.xlabel("Sample index")
    plt.ylabel("|H|")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

# %% [markdown]
"""
---
# Summary and Key Takeaways

Congratulations! You"ve completed a comprehensive journey through channel prediction data 
generation with interpolation and Doppler effects.

## What We Covered

1. ✅ **Simple Interpolation**: Learned basics with 2-user example
2. ✅ **Sequence Extraction**: Found {n_seqs} sequences from {n_users} active users
3. ✅ **Video Generation**: (Optional) tool for spatial coverage visualization
4. ✅ **Uniform Sequences**: Prepared fixed-length inputs for ML models
5. ✅ **Baseline Channels**: Computed sparse channels from raw RT data
6. ✅ **Interpolated Channels**: Generated {interp_factor}× denser, smoother sequences
7. ✅ **Doppler Effects**: Added realistic time-varying behavior (0-200 Hz)

## Key Insights

### Interpolation Benefits
- **Smoother trajectories**: Continuous channel evolution vs discrete jumps
- **More training data**: {interp_factor}× samples per sequence
- **Better prediction**: Dense sampling helps models learn temporal patterns

### Doppler Impact
- **Phase rotation**: Visible in IQ constellations
- **Time-varying channels**: Essential for mobile scenarios
- **Prediction challenge**: Models must learn both spatial and temporal dynamics

### Trade-offs
| Aspect | Baseline | Interpolation | + Doppler |
|--------|----------|---------------|-----------|
| Computation | Fastest | Moderate | Moderate |
| Data density | Sparse | Dense | Dense |
| Realism | RT-exact | Smooth | Most realistic |
| Use case | Static | Mobile (spatial) | Mobile (full) |

## Next Steps: Preparing Data for ML

### 1. Data Formatting

```python
# Split into input/output pairs (e.g., past 20 → predict next 10)
X = H_seq[:, :20, ...]  # Input: past 20 time steps
y = H_seq[:, 20:30, ...]  # Output: next 10 time steps
```

### 2. Normalization

```python
# Per-sequence normalization (recommended for diverse scenarios)
h_max = np.max(np.abs(H_seq), axis=(1,2,3), keepdims=True)
H_normalized = H_seq / h_max

# Or global normalization (consistent scale across dataset)
h_max_global = np.max(np.abs(H_seq))
H_normalized = H_seq / h_max_global
```

### 3. Train/Val/Test Split

```python
# Typical split: 70% train, 15% val, 15% test
n_train = int(0.7 * len(H_normalized))
n_val = int(0.15 * len(H_normalized))

H_train = H_normalized[:n_train]
H_val = H_normalized[n_train:n_train+n_val]
H_test = H_normalized[n_train+n_val:]
```

### 4. Model Architectures to Try

- **LSTM/GRU**: Classic sequence models, good baseline
- **Transformers**: Attention-based, captures long-range dependencies
- **CNN-LSTM**: Spatial (antenna) + temporal processing
- **Graph Neural Networks**: Exploit antenna array structure

## Additional Doppler Methods

This example used **Uniform Doppler** (Way 1). For more realism:

### Way 2: Constant Velocity Vector
```python
dataset.rx_vel = np.array([10, 0, 0])  # 10 m/s along x-axis
```

### Way 3: Geometry-Derived Direction
Compute velocity from consecutive positions:
```python
positions = dataset.rx_pos[sequence]
velocities = np.diff(positions, axis=0) / TIME_DELTA
# Assign to each user
```

### Way 4: Full Dynamic (both speed and direction)
For interpolated sequences with known positions, derive full velocity profile.

## Performance Tips for Large Datasets

- **Batch processing**: Process sequences in batches to manage memory
- **Caching**: Save interpolated datasets to disk
- **Parallelization**: Use multiprocessing for channel generation
- **GPU acceleration**: DeepMIMO supports GPU for faster computation

## Further Reading

- [DeepMIMO Tutorial: Getting Started](../../tutorials/1_getting_started.py)
- [DeepMIMO Tutorial: Channel Generation](../../tutorials/3_channel_generation.py)
- [DeepMIMO Tutorial: Doppler & Mobility](../../tutorials/5_doppler_mobility.py)

## Questions or Feedback?

- GitHub Issues: [https://github.com/DeepMIMO/DeepMIMO/issues](https://github.com/DeepMIMO/DeepMIMO/issues)
- Documentation: [https://deepmimo.github.io/DeepMIMO/](https://deepmimo.github.io/DeepMIMO/)

---

**Happy channel predicting!** 🎉
""".format(
    n_seqs=len(all_seqs),
    n_users=len(dataset.get_idxs("active")),
    interp_factor=INTERP_FACTOR
)

# %%
