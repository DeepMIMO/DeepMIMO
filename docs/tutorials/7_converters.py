"""
# Convert & Upload Ray-tracing Dataset

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DeepMIMO/DeepMIMO/blob/main/docs/tutorials/7_converters.py)
&nbsp;
[![GitHub](https://img.shields.io/badge/Open_on-GitHub-181717?logo=github&style=for-the-badge)](https://github.com/DeepMIMO/DeepMIMO/blob/main/docs/tutorials/7_converters.py)

---

**Tutorial Overview:**
- From Wireless InSite - Convert from Wireless InSite ray tracer
- From Sionna RT - Convert from Sionna RT
- From AODT - Convert from AODT format
- Upload to DeepMIMO - Upload your datasets to the DeepMIMO database

**Related Videos:**
- [Convert Video](https://youtu.be/kXY2bMWeDgg)
- [Upload Video](https://youtu.be/tNF6TN_ueU4)

---
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import deepmimo as dm

# %% [markdown]
# ## From Wireless InSite
#
# Convert ray tracing data from Wireless InSite to DeepMIMO format.

# %%
# Example: Convert from Wireless InSite
# NOTE: This requires Wireless InSite output files

# Define paths to Wireless InSite files
# insite_path = "/path/to/insite/output"
# output_path = "/path/to/deepmimo/output"

# Convert using the converter
# dm.convert(
#     source="wireless_insite",
#     input_path=insite_path,
#     output_path=output_path,
#     scenario_name="my_insite_scenario"
# )

print("Wireless InSite converter example (requires InSite files)")
print("See documentation for detailed usage")

# %% [markdown]
# ## From Sionna RT
#
# Convert from Sionna Ray Tracer.

# %%
# Example: Export from Sionna RT to DeepMIMO
# NOTE: This requires Sionna RT to be installed and configured

# In your Sionna RT script, use the exporter:
example_code = """
# In Sionna:
from deepmimo.converters import sionna_exporter

# After running your Sionna RT simulation:
paths = scene.compute_paths()

# Export to DeepMIMO format
sionna_exporter(
    paths=paths,
    output_dir="./deepmimo_output",
    scenario_name="my_sionna_scenario"
)
"""

print("Sionna RT exporter example:")
print(example_code)

# %% [markdown]
# ## From AODT
#
# Convert from AODT (Autonomous Driving) dataset format.

# %%
# Example: Convert from AODT
# NOTE: This requires AODT dataset files

# Define paths
# aodt_path = "/path/to/aodt/dataset"
# output_path = "/path/to/deepmimo/output"

# Convert using the AODT converter
# dm.convert(
#     source="aodt",
#     input_path=aodt_path,
#     output_path=output_path,
#     scenario_name="my_aodt_scenario"
# )

print("AODT converter example (requires AODT files)")
print("See documentation for detailed usage")

# %% [markdown]
# ## Upload to DeepMIMO
#
# Upload your converted scenarios to the DeepMIMO database.

# %% [markdown]
# ### Upload Scenario Files

# %%
# Example: Upload scenario to DeepMIMO database
# NOTE: This requires authentication credentials

# Upload scenario
# dm.upload(
#     scenario_path="/path/to/deepmimo/scenario",
#     scenario_name="my_custom_scenario",
#     description="Description of the scenario",
#     tags=["outdoor", "urban", "5G"],
#     api_key="your_api_key_here"
# )

print("Upload scenario example (requires API key)")
print("Visit deepmimo.net to get your API key")

# %% [markdown]
# ### Upload Additional Images

# %%
# Example: Upload scenario images/visualizations
# dm.upload_images(
#     scenario_name="my_custom_scenario",
#     image_paths=[
#         "/path/to/coverage_map.png",
#         "/path/to/3d_scene.png"
#     ],
#     api_key="your_api_key_here"
# )

print("Upload images example")

# %% [markdown]
# ### Upload Ray Tracing Source Files

# %%
# Example: Upload original ray tracing source files
# dm.upload_rt_source(
#     scenario_name="my_custom_scenario",
#     source_files=[
#         "/path/to/scene.xml",
#         "/path/to/materials.mat"
#     ],
#     ray_tracer="wireless_insite",  # or "sionna", "aodt"
#     api_key="your_api_key_here"
# )

print("Upload ray tracing source example")

# %% [markdown]
# ## Converter Workflow Example
#
# Complete workflow from ray tracing to DeepMIMO format.

# %%
# Step 1: Run ray tracing simulation
print("Step 1: Run your ray tracing simulation")
print("  - Configure scenario geometry")
print("  - Set TX/RX positions")
print("  - Run simulation")

# Step 2: Convert to DeepMIMO format
print("\nStep 2: Convert to DeepMIMO format")
print("  - Use appropriate converter (InSite, Sionna, AODT)")
print("  - Specify input/output paths")
print("  - Run conversion")

# Step 3: Validate converted data
print("\nStep 3: Validate converted data")
print("  - Load dataset with dm.load()")
print("  - Check data integrity")
print("  - Visualize results")

# Step 4: Upload to DeepMIMO (optional)
print("\nStep 4: Upload to DeepMIMO database (optional)")
print("  - Prepare scenario metadata")
print("  - Upload scenario files")
print("  - Share with community")

# %% [markdown]
# ## Validation After Conversion

# %%
# After conversion, validate your data
# converted_scen = "my_converted_scenario"
# dataset = dm.load(converted_scen)

# Check basic properties
# print(f"Number of users: {len(dataset.power)}")
# print(f"Number of paths: {dataset.power.shape[1]}")
# print(f"Data matrices available: {dir(dataset)}")

# Visualize
# dataset.power.plot()
# plt.title('Converted Scenario - Power Coverage')
# plt.show()

print("Validation example (after successful conversion)")

# %% [markdown]
# ## Supported Ray Tracers
#
# DeepMIMO supports conversion from:
#
# 1. **Wireless InSite**
#    - Full 3D ray tracing
#    - Supports all interaction types
#    - Detailed material properties
#
# 2. **Sionna RT**
#    - GPU-accelerated ray tracing
#    - Differentiable ray tracer
#    - TensorFlow integration
#
# 3. **AODT**
#    - Autonomous driving scenarios
#    - V2X communication
#    - Mobility models

# %% [markdown]
# ## Best Practices
#
# 1. **Organize your data**: Keep source files well-organized
# 2. **Document your scenario**: Include metadata and descriptions
# 3. **Validate thoroughly**: Check converted data before uploading
# 4. **Use meaningful names**: Choose descriptive scenario names
# 5. **Include visualizations**: Upload coverage maps and 3D views

# %% [markdown]
# ---
#
# ## Next Steps
#
# - **Tutorial 8: Migration Guide** - Migrating from DeepMIMO v3 to v4
# - Return to **Tutorial 1: Getting Started** for more basic operations
