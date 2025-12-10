"""Study different path-type and depth combinations and plot coverage results."""

# %%
from collections import defaultdict
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

import deepmimo as dm

save_plots = True
save_dir = "path_trim_plots"

# %%
dataset = dm.load("asu_campus_3p5")

# Calculate total number of paths for reference
n_active_ue = len(dataset.get_active_idxs())
n_active_paths = dataset.num_paths.sum()

# Get power limits from original dataset for consistency
pwr_lims = [np.nanmin(dataset.power), np.nanmax(dataset.power)]

# Define combinations of path types and depths to analyze
# Each tuple is (path_types, max_depth)
# None for path_types means all types allowed
# None for max_depth means no depth filtering
path_combinations = [
    # First few combinations without depth filtering
    (["LoS"], None),
    (["LoS", "R"], None),
    (["LoS", "R", "D"], None),
    (["LoS", "R", "S"], None),
    # Single interaction types with depth filtering
    (["R"], None),
    (["D"], None),
    (["S"], None),
    # Multiple interactions with depth filtering
    (["R", "D"], None),
    (["R", "S"], None),
    # All types with different depths
    (None, 0),
    (None, 1),
    (None, 2),
    (None, 3),
    (None, 4),
    (None, 5),
    (None, 6),
    # Joint path type and depth filtering
    (["R"], 1),
    (["R"], 2),
    (["R"], 3),
    (["R"], 4),
    (["R"], 5),  # 5 reflections in ASU
    (["D"], 1),  # only one diffraction in ASU
    (["R", "S"], 1),
    (["R", "S"], 2),
    (["R", "S"], 3),
    (["R", "S"], 4),
    (["R", "S"], 5),
    (["R", "S"], 6),  # 1 scattering event after reflections ASU
    (["R", "D"], 1),
    (["R", "D"], 2),
    (["R", "D"], 3),
    (["R", "D"], 4),
    (["R", "D"], 5),
    (["R", "D"], 6),
    (["R", "D", "S"], 6),  # all except LoS
    (["LoS", "R", "D", "S"], 6),  # 100% of path coverage
    # Additional missing combinations (just for the montage)
    (["LoS"], 0),
    (["LoS"], 1),
    (["S"], 1),
    (["LoS"], 2),
    (["D"], 2),
    (["S"], 2),
    (["LoS"], 3),
    (["D"], 3),
    (["S"], 3),
    (["LoS"], 4),
    (["D"], 4),
    (["S"], 4),
    (["LoS"], 5),
    (["D"], 5),
    (["S"], 5),
    # Additional simulations for completeness
    (["LoS", "R", "D", "S"], 1),
    (["LoS", "R", "D", "S"], 2),
    (["LoS", "R", "D", "S"], 3),
    (["LoS", "R", "D", "S"], 4),
    (["LoS", "R", "D", "S"], 5),
    (["LoS"], 6),
    (["R"], 6),
    (["D"], 6),
    (["S"], 6),
    (["LoS", "R", "D", "S"], 6),
    (["R", "D", "S"], 1),  # all except LoS
    (["R", "D", "S"], 2),
    (["R", "D", "S"], 3),
    (["R", "D", "S"], 4),
    (["R", "D", "S"], 5),
]

# Process each combination
SKIP_INITIAL_COMBINATIONS = 62

for combo_idx, (path_types, max_depth) in enumerate(path_combinations):
    if combo_idx < SKIP_INITIAL_COMBINATIONS:
        continue
    # Start with original dataset
    trimmed_dataset = dataset

    # Apply path type filtering if specified
    type_title = "All types"
    if path_types is not None:
        type_title = " + ".join(path_types)
        print(f"trimming by path type: {' + '.join(path_types)}")
        trimmed_dataset = trimmed_dataset.trim_by_path_type(path_types)

    # Apply depth filtering if specified
    depth_title = "All depths"
    if max_depth is not None:
        depth_title = f"depth≤{max_depth}"
        print(f"trimming by path depth: {max_depth}")
        trimmed_dataset = trimmed_dataset.trim_by_path_depth(max_depth)

    # Create title with trim type
    title = f"{type_title} | {depth_title}"

    # Calculate coverage percentage
    coverage = len(trimmed_dataset.get_active_idxs()) / n_active_ue
    path_coverage = trimmed_dataset.num_paths.sum() / n_active_paths

    plt_args = {
        "dpi": 200,
        "title": f"{title} ({coverage:.2%}, paths:{path_coverage:.2%})",
        "lims": pwr_lims,
    }
    # Plot power with coverage percentage in title and consistent limits
    trimmed_dataset.power.plot(**plt_args)

    if save_plots:
        path_type_str = "-".join(path_types) if path_types else "all-types"
        depth_str = str(max_depth) if max_depth is not None else "all-depths"
        exp_str = f"{combo_idx:03d}_{path_type_str}_{depth_str}"
        save_path = f"{save_dir}/pwr_exp_{exp_str}.png"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)

    plt.show()
    # break

    # FUTURE: Compute channels & save matrix for topological analysis

    # FUN FACT: trimming by depth is much faster than trimming by type.


IMG_SIZE = (1200, 800)

# Define specific columns and rows for the montage
desired_types = ["LoS", "R", "D", "S", "R+D", "R+S", "R+D+S", "LoS+R+D+S"]
desired_depths = ["1", "2", "3", "4", "5", "6"]

# Build a structured map of images
image_grid = defaultdict(dict)
MIN_FILENAME_PARTS = 5

# Parse files from extracted folder
for fname in [p.name for p in Path(save_dir).iterdir()]:
    if not fname.endswith(".png"):
        continue

    # Parse filename
    # Format: pwr_exp_[index]_[type]_[depth].png
    parts = fname.split("_")
    if len(parts) < MIN_FILENAME_PARTS:  # we expect at least 5 parts
        continue

    type_str = parts[3]  # The type is in the 4th position
    depth_str = parts[4].replace(".png", "")  # The depth is in the 5th position

    # Map type strings to our desired format
    if type_str in ["LoS", "R", "D", "S", "R-D", "R-S", "R-D-S", "LoS-R-D-S"]:
        display_type = type_str if type_str == "LoS" else type_str.replace("-", "+")
        image_grid[depth_str][display_type] = str(Path(save_dir) / fname)

# Debug print
print("\nAvailable combinations:")
for depth in sorted(image_grid.keys()):
    print(f"\nDepth {depth}:")
    for type_key in sorted(image_grid[depth].keys()):
        print(f"  {type_key}: {Path(image_grid[depth][type_key]).name}")

# Create the montage figure
nrows = len(desired_depths)
ncols = len(desired_types)
# Make figure even more compact
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.5, nrows * 1.5))

# If only one row or column, axes is 1D, convert to 2D for consistency
if nrows == 1:
    axes = np.expand_dims(axes, axis=0)
if ncols == 1:
    axes = np.expand_dims(axes, axis=1)

# Define trim coordinates
trim_coords = {
    "x": 98,
    "y": 43,
    "width": 839,
    "height": 677,
}

for i, row_key in enumerate(desired_depths):
    for j, col_key in enumerate(desired_types):
        ax = axes[i][j]
        img_path = image_grid.get(row_key, {}).get(col_key)

        if img_path and Path(img_path).exists():
            # Read and trim the image
            img = mpimg.imread(img_path)
            # Trim the image using the specified coordinates
            img_trimmed = img[
                trim_coords["y"] : trim_coords["y"] + trim_coords["height"],
                trim_coords["x"] : trim_coords["x"] + trim_coords["width"],
            ]
            ax.imshow(img_trimmed)
        else:
            ax.set_facecolor("gray")
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", color="white", fontsize=6)
        ax.axis("off")
        if i == 0:
            ax.set_title(col_key, fontsize=7, pad=0)
        if j == 0:
            # Place text to the left of the first image in each row
            ax.text(
                -0.05,
                0.5,
                f"depth ≤ {row_key}",
                fontsize=7,
                rotation=90,
                ha="right",
                va="center",
                transform=ax.transAxes,
            )

# Use negative spacing to make images overlap slightly
plt.subplots_adjust(wspace=-0.00, hspace=-0.5)
montage_path = "./path_trim_montage.png"
plt.savefig(montage_path, dpi=300, bbox_inches="tight")
plt.close()


# %% Check for missing combinations

# After building image_grid but before creating the montage, add:

print("\n=== Missing Combinations Analysis ===")
print("Format: (path_types, max_depth)")

# Map from display format to simulation format
sim_type_mapping = {
    "LoS": ["LoS"],
    "R": ["R"],
    "D": ["D"],
    "S": ["S"],
    "R+D": ["R", "D"],
    "R+S": ["R", "S"],
    "R+D+S": ["R", "D", "S"],
    "LoS-R-D-S": ["LoS", "R", "D", "S"],
}

missing_combinations = []
for depth in desired_depths:
    for type_key in desired_types:
        if depth != "all-depths" and not image_grid.get(depth, {}).get(type_key):
            # Convert display format to simulation parameters
            path_types = sim_type_mapping[type_key]
            max_depth = int(depth)
            missing_combinations.append((path_types, max_depth))

print("\nMissing combinations that need to be simulated:")
for path_types, max_depth in missing_combinations:
    print(f"(path_types={path_types}, max_depth={max_depth})")
