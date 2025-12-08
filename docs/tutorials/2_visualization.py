"""
# Visualization and Scene

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DeepMIMO/DeepMIMO/blob/main/docs/tutorials/2_visualization.py)
&nbsp;
[![GitHub](https://img.shields.io/badge/Open_on-GitHub-181717?logo=github&style=for-the-badge)](https://github.com/DeepMIMO/DeepMIMO/blob/main/docs/tutorials/2_visualization.py)

---

**Tutorial Overview:**
- Coverage Maps - Visualizing signal coverage
- Rays - Ray propagation visualization
- Path Plots - Visualization of different path components
- Scene & Materials - 3D scene and material visualization
- Plot Overlays - Combining different visualizations

**Related Video:** [Visualization Video](https://youtu.be/MO7h2shBhsc)

---
"""

# %%
# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import deepmimo as dm

# %%
# Load dataset
scen_name = "asu_campus_3p5"
dm.download(scen_name)
dataset = dm.load(scen_name)

# %% [markdown]
# ## Coverage Maps
#
# Visualize signal coverage across the scenario.

# %%
# Plot power coverage
dataset.power.plot()
plt.title('Power Coverage Map')
plt.show()

# %%
# Plot pathloss coverage
dataset.pathloss.plot()
plt.title('Pathloss Coverage Map')
plt.show()

# %% [markdown]
# ## Rays
#
# Visualize ray propagation paths.

# %%
# Plot rays for a user with line-of-sight
u_idx = np.where(dataset.los == 1)[0][100]
dataset.plot_rays(u_idx, proj_3D=False, dpi=100)
plt.title('Ray Propagation Paths')
plt.show()

# %% [markdown]
# ## Path Plots
#
# Visualize different path components and characteristics.

# %% [markdown]
# ### Percentage of Power

# %%
# Plot power percentage per path
if hasattr(dataset, 'power'):
    # Calculate power percentage
    linear_power = 10 ** (dataset.power / 10)
    total_power = np.sum(linear_power, axis=1, keepdims=True)
    power_pct = (linear_power / (total_power + 1e-20)) * 100
    
    plt.figure(figsize=(10, 6))
    plt.hist(power_pct[power_pct > 0].flatten(), bins=50)
    plt.xlabel('Power Percentage (%)')
    plt.ylabel('Count')
    plt.title('Distribution of Power Percentage per Path')
    plt.show()

# %% [markdown]
# ### Number of Interactions

# %%
# Visualize number of interactions per path
if hasattr(dataset, 'interactions'):
    interactions = dataset.interactions
    # Count digits in interaction codes
    num_interactions = np.array([[len(str(int(x))) if x > 0 else 0 for x in row] for row in interactions])
    
    plt.figure(figsize=(10, 6))
    plt.hist(num_interactions.flatten(), bins=range(0, 10))
    plt.xlabel('Number of Interactions')
    plt.ylabel('Count')
    plt.title('Distribution of Interaction Counts')
    plt.show()

# %% [markdown]
# ## Scene & Materials
#
# Explore the 3D scene and materials.

# %% [markdown]
# ### Scene Visualization

# %%
# Plot the 3D scene
if hasattr(dataset, 'scene'):
    dataset.scene.plot()
    plt.title('3D Scene')
    plt.show()
else:
    print("Scene data not available for this scenario")

# %% [markdown]
# ### Materials

# %%
# Display material properties
if hasattr(dataset, 'materials'):
    print("Available Materials:")
    print(dataset.materials)
else:
    print("Material data not available for this scenario")

# %% [markdown]
# ## Plot Overlays
#
# Combine different visualizations for comprehensive analysis.

# %% [markdown]
# ### 2D Scene, Coverage & Rays Overlay

# %%
# Overlay coverage map with rays
fig, ax = plt.subplots(figsize=(12, 8))

# Plot coverage as background
dataset.power.plot(ax=ax)

# Overlay rays for a selected user
los_users = np.where(dataset.los == 1)[0]
if len(los_users) > 0:
    dataset.plot_rays(los_users[50], ax=ax, proj_3D=False)

plt.title('Coverage Map with Ray Overlay')
plt.show()

# %% [markdown]
# ### 3D Scene & Rays Overlay

# %%
# 3D visualization with scene and rays
if hasattr(dataset, 'scene'):
    # Plot rays first, then overlay the scene
    ax = dataset.plot_rays(los_users[50])
    dataset.scene.plot(ax=ax)
    ax.legend().set_visible(False)
    
    plt.title('3D Scene with Rays')
    plt.show()
else:
    print("3D scene not available for this scenario")

# %% [markdown]
# ---
#
# ## Next Steps
#
# Continue with:
# - **Tutorial 3: Detailed Channel Generation** - Deep dive into channel generation parameters
# - **Tutorial 4: User Selection and Dataset Manipulation** - Learn how to filter and sample users

