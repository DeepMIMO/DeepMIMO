# ruff: noqa: N999
"""# User Selection and Dataset Manipulation.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DeepMIMO/DeepMIMO/blob/main/docs/tutorials/4_dataset_manipulation.py)
&nbsp;
[![GitHub](https://img.shields.io/badge/Open_on-GitHub-181717?logo=github&style=for-the-badge)](https://github.com/DeepMIMO/DeepMIMO/blob/main/docs/tutorials/4_dataset_manipulation.py)

---

**Tutorial Overview:**
- Dataset Trimming - Trim dataset based on conditions
- Uniform Sampling - Uniform user sampling
- Linear Sampling - Linear user placement
- Rectangular Zones - Filtering in 3D bounding boxes
- Path Type/Depth Filtering - Trim by path characteristics
- Field-of-View - FoV analysis for receivers

**Related Video:** [User Sampling Video](https://youtu.be/KV0LLp0jOFc)

---
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

import deepmimo as dm

# %%
# Load dataset
scen_name = "asu_campus_3p5"
dm.download(scen_name)
dataset = dm.load(scen_name)

print(f"Original dataset size: {len(dataset.power)} users")

# %% [markdown]
# ## Dataset Trimming
#
# Trim dataset based on various conditions.

# %%
# Get active indices (users with at least one path)
active_idxs = np.where(dataset.num_paths > 0)[0]
print(f"Number of active users: {len(active_idxs)}")

# %%
# Create a subset with only active users
active_dataset = dataset.trim(idxs=active_idxs)
print(f"Active dataset size: {len(active_dataset.power)} users")

# %% [markdown]
# ## Uniform Sampling
#
# Sample users uniformly from the dataset.

# %%
# Get uniform sampling indices (every Nth user)
uniform_idxs = dataset.get_idxs(mode="uniform", steps=[10, 10])
print(f"Uniform sampling: {len(uniform_idxs)} users")

# %%
# Create uniform subset
uniform_dataset = dataset.trim(idxs=uniform_idxs)

# Visualize uniform sampling
plt.figure(figsize=(10, 6))
plt.scatter(
    dataset.rx_pos[:, 0], dataset.rx_pos[:, 1], c="lightgray", s=1, label="All users", alpha=0.5
)
plt.scatter(
    uniform_dataset.rx_pos[:, 0],
    uniform_dataset.rx_pos[:, 1],
    c="red",
    s=10,
    label="Uniform sample",
)
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Uniform User Sampling")
plt.legend()
plt.grid(visible=True)
plt.show()

# %% [markdown]
# ## Rows and Columns
#
# Select users by row/column indices.

# %%
# Get users from specific rows
row_idxs = dataset.get_idxs(mode="row", row_idxs=[0, 10, 20, 30])
print(f"Users from rows [0, 10, 20, 30]: {len(row_idxs)} users")

# %%
# Get users from specific columns
col_idxs = dataset.get_idxs(mode="col", col_idxs=[0, 50, 100])
print(f"Users from columns [0, 50, 100]: {len(col_idxs)} users")

# %% [markdown]
# ## Linear Sampling
#
# Sample users along a linear path.

# %%
# Define a linear path
start_point = dataset.rx_pos[0]
end_point = dataset.rx_pos[-1]

linear_idxs = dataset.get_idxs(mode="linear", start_pos=start_point, end_pos=end_point, n_steps=50)

print(f"Linear path sampling: {len(linear_idxs)} users")

# %%
# Visualize linear sampling
plt.figure(figsize=(10, 6))
plt.scatter(
    dataset.rx_pos[:, 0], dataset.rx_pos[:, 1], c="lightgray", s=1, alpha=0.5, label="All users"
)
linear_positions = dataset.rx_pos[linear_idxs]
plt.scatter(linear_positions[:, 0], linear_positions[:, 1], c="blue", s=20, label="Linear sample")
plt.plot(
    [start_point[0], end_point[0]], [start_point[1], end_point[1]], "r--", linewidth=2, label="Path"
)
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Linear Path Sampling")
plt.legend()
plt.grid(visible=True)
plt.show()

# %% [markdown]
# ## Rectangular Zones
#
# Filter users within 3D bounding boxes.

# %%
# Define a rectangular zone
x_min = np.min(dataset.rx_pos[:, 0])
x_max = np.mean(dataset.rx_pos[:, 0])
y_min = np.min(dataset.rx_pos[:, 1])
y_max = np.mean(dataset.rx_pos[:, 1])
z_min = 0
z_max = 10

zone_idxs = dataset.get_idxs(
    mode="limits", x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, z_min=z_min, z_max=z_max
)

print(f"Users in rectangular zone: {len(zone_idxs)}")

# %%
# Visualize zone filtering
plt.figure(figsize=(10, 6))
plt.scatter(
    dataset.rx_pos[:, 0], dataset.rx_pos[:, 1], c="lightgray", s=1, alpha=0.5, label="All users"
)
zone_positions = dataset.rx_pos[zone_idxs]
plt.scatter(zone_positions[:, 0], zone_positions[:, 1], c="green", s=10, label="Zone users")
plt.axvline(x_min, color="r", linestyle="--", linewidth=1)
plt.axvline(x_max, color="r", linestyle="--", linewidth=1)
plt.axhline(y_min, color="r", linestyle="--", linewidth=1)
plt.axhline(y_max, color="r", linestyle="--", linewidth=1)
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Rectangular Zone Filtering")
plt.legend()
plt.grid(visible=True)
plt.show()

# %% [markdown]
# ## Path Type Filtering
#
# Trim dataset by path interaction types.

# %%
# Filter users with LOS paths only
los_idxs = np.where(dataset.los == 1)[0]
print(f"Users with LOS: {len(los_idxs)}")

los_dataset = dataset.trim(idxs=los_idxs)

# %%
# Visualize LOS vs NLOS users
plt.figure(figsize=(10, 6))
nlos_idxs = np.where(dataset.los == 0)[0]

plt.scatter(
    dataset.rx_pos[nlos_idxs, 0],
    dataset.rx_pos[nlos_idxs, 1],
    c="blue",
    s=5,
    label="NLOS users",
    alpha=0.5,
)
plt.scatter(
    dataset.rx_pos[los_idxs, 0], dataset.rx_pos[los_idxs, 1], c="red", s=10, label="LOS users"
)
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("LOS vs NLOS Users")
plt.legend()
plt.grid(visible=True)
plt.show()

# %% [markdown]
# ## Path Depth Filtering
#
# Filter by number of interactions/bounces.

# %%
# Count interactions per path
if hasattr(dataset, "interactions"):
    interactions = dataset.interactions

    # Get number of interactions (digits in interaction code)
    num_interactions = np.array(
        [[len(str(int(x))) if x > 0 else 0 for x in row] for row in interactions]
    )

    # Find users with paths having exactly 1 interaction (single bounce)
    single_bounce_mask = np.any(num_interactions == 1, axis=1)
    single_bounce_idxs = np.where(single_bounce_mask)[0]

    print(f"Users with single-bounce paths: {len(single_bounce_idxs)}")

    # Visualize
    plt.figure(figsize=(10, 6))
    plt.scatter(
        dataset.rx_pos[:, 0], dataset.rx_pos[:, 1], c="lightgray", s=1, alpha=0.3, label="All users"
    )
    plt.scatter(
        dataset.rx_pos[single_bounce_idxs, 0],
        dataset.rx_pos[single_bounce_idxs, 1],
        c="orange",
        s=10,
        label="Single-bounce paths",
    )
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Users with Single-Bounce Paths")
    plt.legend()
    plt.grid(visible=True)
    plt.show()
else:
    print("Interaction data not available")

# %% [markdown]
# ## Field-of-View (FOV)
#
# Apply field-of-view filtering.

# %%
# Apply azimuth FOV
fov_az = 120  # degrees
fov_dataset = dataset.trim(bs_fov=[fov_az, 180])  # [azimuth, elevation]

print(f"Dataset after FOV filtering: {len(fov_dataset.power)} users")

# %%
# Apply both azimuth and elevation FOV
fov_az = 120  # degrees
fov_el = 60  # degrees
fov_dataset_full = dataset.trim(bs_fov=[fov_az, fov_el])

print(f"Dataset after full FOV filtering: {len(fov_dataset_full.power)} users")

# %% [markdown]
# ## Combined Filtering
#
# Combine multiple filtering criteria.

# %%
# Complex filtering: Active users, in zone, with LOS
active_idxs = np.where(dataset.num_paths > 0)[0]
zone_idxs_set = set(zone_idxs)
los_idxs_set = set(los_idxs)

# Intersection of all criteria
combined_idxs = np.array(sorted(set(active_idxs) & zone_idxs_set & los_idxs_set), dtype=int)

print(f"Users matching all criteria: {len(combined_idxs)}")

if len(combined_idxs) > 0:
    combined_dataset = dataset.trim(idxs=combined_idxs)
else:
    print("No users match all criteria, skipping combined dataset creation")

# Visualize combined filtering
if len(combined_idxs) > 0:
    plt.figure(figsize=(10, 6))
    plt.scatter(
        dataset.rx_pos[:, 0], dataset.rx_pos[:, 1], c="lightgray", s=1, alpha=0.3, label="All users"
    )
    plt.scatter(
        combined_dataset.rx_pos[:, 0],
        combined_dataset.rx_pos[:, 1],
        c="purple",
        s=15,
        label="Combined filter",
    )
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Combined Filtering: Active + Zone + LOS")
    plt.legend()
    plt.grid(visible=True)
    plt.show()
else:
    print("Skipping visualization as no users match all criteria")

# %% [markdown]
# ---
#
# ## Next Steps
#
# Continue with:
# - **Tutorial 5: Doppler and Mobility** - Add time-varying effects to your channels
# - **Tutorial 6: Beamforming** - Implement beamforming and spatial processing
