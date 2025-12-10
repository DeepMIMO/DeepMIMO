# DeepMIMO Tutorials

Welcome to the DeepMIMO tutorials! These step-by-step guides will help you master the DeepMIMO dataset generator.

## Tutorial Overview

| # | Tutorial | Description | Video |
|---|----------|-------------|-------|
| 1 | [Getting Started](1_getting_started.py) | Load scenarios, generate channels, explore datasets | [▶️](https://youtu.be/LDG6IPEHY54) |
| 2 | [Visualization and Scene](2_visualization.py) | Coverage maps, ray visualization, 3D scenes | [▶️](https://youtu.be/MO7h2shBhsc) |
| 3 | [Detailed Channel Generation](3_channel_generation.py) | Channel parameters, time/freq domain, antenna rotation | [▶️](https://youtu.be/xsl6gjTEu2U) |
| 4 | [User Selection and Dataset Manipulation](4_dataset_manipulation.py) | Sampling, trimming, filtering by various criteria | [▶️](https://youtu.be/KV0LLp0jOFc) |
| 5 | [Doppler and Mobility](5_doppler_mobility.py) | Time-varying channels, mobility models, Doppler effects | [▶️](https://youtu.be/xsl6gjTEu2U) |
| 6 | [Beamforming](6_beamforming.py) | Steering vectors, beamforming visualization, MIMO | [▶️](https://youtu.be/IPVnIW2vGLE) |
| 7 | [Convert & Upload Ray-tracing Dataset](7_converters.py) | External ray tracers, format conversion, uploading | [▶️](https://youtu.be/kXY2bMWeDgg) |
| 8 | [Migration Guide](8_migration_guide.py) | Migrating from DeepMIMO v3 to v4 | [▶️](https://youtu.be/15nQWS15h3k) |

## How to Use These Tutorials

### Running Locally

1. Install DeepMIMO:
   ```bash
   pip install --pre deepmimo
   ```

2. Download a tutorial:
   ```bash
   wget https://raw.githubusercontent.com/DeepMIMO/DeepMIMO/main/docs/tutorials/1_getting_started.py
   ```

3. Run the tutorial:
   ```python
   python 1_getting_started.py
   ```

### Running in Colab

Each tutorial has a "Open in Colab" badge at the top. Click it to run the tutorial in Google Colab.

### Running with Jupyter

The tutorials are Python files with cell markers (`# %%`) that work with:
- VS Code Python extension
- JupyterLab / Jupyter Notebook (using jupytext)
- PyCharm Professional

## Tutorial Contents

### 1. Getting Started
- **Part 1**: Hello World - basic dataset loading and channel generation
- **Part 2**: Deep Dive - complex scenarios with multiple base stations  
- **Part 3**: Discovery - using `dm.info()`, aliases, and implicit computations

**Key Functions**: `dm.download()`, `dm.load()`, `dataset.compute_channels()`, `dm.info()`

### 2. Visualization and Scene
- Coverage maps for power, pathloss, and other metrics
- Ray propagation visualization
- Path plots (power percentage, interactions)
- 3D scene and materials
- Overlaying multiple visualizations

**Key Functions**: `dataset.power.plot()`, `dataset.plot_rays()`, `dataset.scene.plot()`

### 3. Detailed Channel Generation
- Configuring channel parameters
- Time-domain channel generation
- OFDM/frequency-domain channels
- Antenna array configuration and rotation
- Comparing different antenna configurations

**Key Functions**: `dm.ChannelParameters()`, `dataset.compute_channels()`

### 4. User Selection and Dataset Manipulation
- Dataset trimming and active user filtering
- Uniform and linear sampling
- Row/column selection
- Rectangular zone filtering
- Path type and depth filtering
- Field-of-view (FOV) analysis

**Key Functions**: `dataset.get_idxs()`, `dataset.trim()`

### 5. Doppler and Mobility
- Setting Doppler shifts directly
- Defining user/object velocities
- Time-varying channel generation
- Mobility patterns (linear, circular)
- Doppler spectrum analysis

**Key Functions**: `dataset.set_doppler()`, `dataset.rx_vel`, `dataset.tx_vel`, `dataset.set_timestamps()`

### 6. Beamforming
- Computing steering vectors
- Beamforming gain calculation
- Beam pattern visualization (rectangular and polar)
- Best beam selection
- Multi-user beamforming (zero-forcing)

**Key Functions**: `dm.steering_vec()`, beamforming visualization methods

### 7. Convert & Upload Ray-tracing Dataset
- Converting from Wireless InSite
- Converting from Sionna RT
- Converting from AODT
- Uploading scenarios to DeepMIMO database
- Best practices for dataset sharing

**Key Functions**: `dm.convert()`, `dm.upload()`, `dm.upload_images()`

### 8. Migration Guide
- Key differences between v3 and v4
- Updated workflows and API
- Parameter name mapping
- Migration checklist
- Common migration patterns

## Additional Resources

- **Full Manual**: [manual.ipynb](../manual.ipynb) - Comprehensive reference with all examples
- **API Documentation**: [API Reference](../api/generator.md)
- **GitHub Repository**: [DeepMIMO/DeepMIMO](https://github.com/DeepMIMO/DeepMIMO)
- **Website**: [deepmimo.net](https://deepmimo.net)

## Need Help?

- 📖 [Documentation](https://deepmimo.github.io/DeepMIMO/)
- 💬 [GitHub Discussions](https://github.com/DeepMIMO/DeepMIMO/discussions)
- 🐛 [GitHub Issues](https://github.com/DeepMIMO/DeepMIMO/issues)
- 🌐 [DeepMIMO Community](https://deepmimo.net)

## Contributing

Found an issue in a tutorial? Have an idea for a new one? Contributions are welcome!

1. Fork the repository
2. Create a new tutorial or fix an existing one
3. Submit a pull request

See [Contributing Guidelines](../resources/contributing.md) for more details.

