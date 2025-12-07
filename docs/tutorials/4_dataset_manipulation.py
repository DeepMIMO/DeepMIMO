"""
# User Selection and Dataset Manipulation

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
active_idxs = dataset.get_active_idxs()
print(f"Number of active users: {len(active_idxs)}")

# %%
# Create a subset with only active users
active_dataset = dataset.subset(active_idxs)
print(f"Active dataset size: {len(active_dataset.power)} users")

# %% [markdown]
# ## Uniform Sampling
#
# Sample users uniformly from the dataset.

# %%
# Get uniform sampling indices (every Nth user)
uniform_idxs = dataset.get_uniform_idxs(step=10)
print(f"Uniform sampling: {len(uniform_idxs)} users")

# %%
# Create uniform subset
uniform_dataset = dataset.subset(uniform_idxs)

# Visualize uniform sampling
plt.figure(figsize=(10, 6))
plt.scatter(dataset.user_positions[:, 0], dataset.user_positions[:, 1], 
            c='lightgray', s=1, label='All users', alpha=0.5)
plt.scatter(uniform_dataset.user_positions[:, 0], uniform_dataset.user_positions[:, 1],
            c='red', s=10, label='Uniform sample')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Uniform User Sampling')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# ## Rows and Columns
#
# Select users by row/column indices.

# %%
# Get users from specific rows
row_idxs = dataset.get_row_idxs([0, 10, 20, 30])
print(f"Users from rows [0, 10, 20, 30]: {len(row_idxs)} users")

# %%
# Get users from specific columns
col_idxs = dataset.get_col_idxs([0, 50, 100])
print(f"Users from columns [0, 50, 100]: {len(col_idxs)} users")

# %% [markdown]
# ## Linear Sampling
#
# Sample users along a linear path.

# %%
# Define a linear path
start_point = dataset.user_positions[0]
end_point = dataset.user_positions[-1]

linear_path = dm.LinearPath(start=start_point, end=end_point, num_points=50)
linear_idxs = linear_path.get_indices(dataset.user_positions)

print(f"Linear path sampling: {len(linear_idxs)} users")

# %%
# Visualize linear sampling
plt.figure(figsize=(10, 6))
plt.scatter(dataset.user_positions[:, 0], dataset.user_positions[:, 1],
            c='lightgray', s=1, alpha=0.5, label='All users')
linear_positions = dataset.user_positions[linear_idxs]
plt.scatter(linear_positions[:, 0], linear_positions[:, 1],
            c='blue', s=20, label='Linear sample')
plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]],
         'r--', linewidth=2, label='Path')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Linear Path Sampling')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# ## Rectangular Zones
#
# Filter users within 3D bounding boxes.

# %%
# Define a rectangular zone
x_limits = [np.min(dataset.user_positions[:, 0]), 
            np.mean(dataset.user_positions[:, 0])]
y_limits = [np.min(dataset.user_positions[:, 1]),
            np.mean(dataset.user_positions[:, 1])]
z_limits = [0, 10]  # Height limits

zone_idxs = dm.get_idxs_with_limits(
    dataset.user_positions,
    x_limits=x_limits,
    y_limits=y_limits,
    z_limits=z_limits
)

print(f"Users in rectangular zone: {len(zone_idxs)}")

# %%
# Visualize zone filtering
plt.figure(figsize=(10, 6))
plt.scatter(dataset.user_positions[:, 0], dataset.user_positions[:, 1],
            c='lightgray', s=1, alpha=0.5, label='All users')
zone_positions = dataset.user_positions[zone_idxs]
plt.scatter(zone_positions[:, 0], zone_positions[:, 1],
            c='green', s=10, label='Zone users')
plt.axvline(x_limits[0], color='r', linestyle='--', linewidth=1)
plt.axvline(x_limits[1], color='r', linestyle='--', linewidth=1)
plt.axhline(y_limits[0], color='r', linestyle='--', linewidth=1)
plt.axhline(y_limits[1], color='r', linestyle='--', linewidth=1)
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Rectangular Zone Filtering')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# ## Path Type Filtering
#
# Trim dataset by path interaction types.

# %%
# Filter users with LOS paths only
los_idxs = np.where(dataset.los == 1)[0]
print(f"Users with LOS: {len(los_idxs)}")

los_dataset = dataset.subset(los_idxs)

# %%
# Visualize LOS vs NLOS users
plt.figure(figsize=(10, 6))
nlos_idxs = np.where(dataset.los == 0)[0]

plt.scatter(dataset.user_positions[nlos_idxs, 0], 
            dataset.user_positions[nlos_idxs, 1],
            c='blue', s=5, label='NLOS users', alpha=0.5)
plt.scatter(dataset.user_positions[los_idxs, 0],
            dataset.user_positions[los_idxs, 1],
            c='red', s=10, label='LOS users')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('LOS vs NLOS Users')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# ## Path Depth Filtering
#
# Filter by number of interactions/bounces.

# %%
# Count interactions per path
if hasattr(dataset, 'interactions'):
    interactions = dataset.interactions
    
    # Get number of interactions (digits in interaction code)
    num_interactions = np.array([[len(str(int(x))) if x > 0 else 0 
                                   for x in row] 
                                  for row in interactions])
    
    # Find users with paths having exactly 1 interaction (single bounce)
    single_bounce_mask = np.any(num_interactions == 1, axis=1)
    single_bounce_idxs = np.where(single_bounce_mask)[0]
    
    print(f"Users with single-bounce paths: {len(single_bounce_idxs)}")
    
    # Visualize
    plt.figure(figsize=(10, 6))
    plt.scatter(dataset.user_positions[:, 0], dataset.user_positions[:, 1],
                c='lightgray', s=1, alpha=0.3, label='All users')
    plt.scatter(dataset.user_positions[single_bounce_idxs, 0],
                dataset.user_positions[single_bounce_idxs, 1],
                c='orange', s=10, label='Single-bounce paths')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Users with Single-Bounce Paths')
    plt.legend()
    plt.grid(True)
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
fov_dataset = dataset.apply_fov(fov_az=fov_az)

print(f"Dataset after FOV filtering: {len(fov_dataset.power)} users")

# %%
# Apply both azimuth and elevation FOV
fov_az = 120  # degrees
fov_el = 60  # degrees
fov_dataset_full = dataset.apply_fov(fov_az=fov_az, fov_el=fov_el)

print(f"Dataset after full FOV filtering: {len(fov_dataset_full.power)} users")

# %% [markdown]
# ## Combined Filtering
#
# Combine multiple filtering criteria.

# %%
# Complex filtering: Active users, in zone, with LOS
active_idxs = dataset.get_active_idxs()
zone_idxs_set = set(zone_idxs)
los_idxs_set = set(los_idxs)

# Intersection of all criteria
combined_idxs = list(set(active_idxs) & zone_idxs_set & los_idxs_set)
combined_idxs.sort()

print(f"Users matching all criteria: {len(combined_idxs)}")

combined_dataset = dataset.subset(combined_idxs)

# Visualize combined filtering
plt.figure(figsize=(10, 6))
plt.scatter(dataset.user_positions[:, 0], dataset.user_positions[:, 1],
            c='lightgray', s=1, alpha=0.3, label='All users')
plt.scatter(combined_dataset.user_positions[:, 0],
            combined_dataset.user_positions[:, 1],
            c='purple', s=15, label='Combined filter')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Combined Filtering: Active + Zone + LOS')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# ---
#
# ## Next Steps
#
# Continue with:
# - **Tutorial 5: Doppler and Mobility** - Add time-varying effects to your channels
# - **Tutorial 6: Beamforming** - Implement beamforming and spatial processing

