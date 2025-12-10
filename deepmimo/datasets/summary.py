"""Summarizes dataset characteristics.

This module is used by the Database API to send summaries to the server.
As such, the information displayed here will match the information
displayed on the DeepMIMO website.

The module is also leveraged by users to understand a dataset during development.

Usage:
    summary(scen_name, print_summary=True)

Three functions:

1. summary(scen_name, print_summary=True)
    - If print_summary is True, prints a summary of the dataset.
    - If print_summary is False, returns a string summary of the dataset.
    - Used for printing summaries to the console.
    - *Provides* the information for each DeepMIMO scenario page.

2. plot_summary(scen_name)
    - Plots several figures representing the dataset.
    - Plot 1: LOS image
    - Plot 2: 3D view of the scene (buildings, roads, trees, etc.)
    - Plot 3: 2D view of the scene with BSs and users
    - Returns None
    - *Provides* the figures for each DeepMIMO scenario page.

3. stats(scen_name)
    - (coming soon)
    - Returns a dictionary of statistics about the dataset.

"""

# Standard library imports
import time
from pathlib import Path
from typing import Any

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np

from deepmimo import consts as c

# Local imports
from deepmimo.utils import (
    get_params_path,
    load_dict_from_json,
)


def summary(scen_name: str, *, print_summary: bool = True) -> str | None:  # noqa: PLR0915
    """Print a summary of the dataset."""
    # Initialize empty string to collect output
    summary_str = ""

    # Read params.mat and provide TXRX summary, total number of tx & rx, scene size,
    # and other relevant parameters, computed/extracted from the all dicts, not just rt_params

    params_json_path = get_params_path(scen_name)

    params_dict = load_dict_from_json(params_json_path)
    rt_params = params_dict[c.RT_PARAMS_PARAM_NAME]
    scene_params = params_dict[c.SCENE_PARAM_NAME]
    material_params = params_dict[c.MATERIALS_PARAM_NAME]
    txrx_params = params_dict[c.TXRX_PARAM_NAME]

    summary_str += "\n" + "=" * 50 + "\n"
    summary_str += f"DeepMIMO {scen_name} Scenario Summary\n"
    summary_str += "=" * 50 + "\n"

    summary_str += "\n[Ray-Tracing Configuration]\n"
    summary_str += (
        f"- Ray-tracer: {rt_params[c.RT_PARAM_RAYTRACER]} "
        f"v{rt_params[c.RT_PARAM_RAYTRACER_VERSION]}\n"
    )
    summary_str += f"- Frequency: {rt_params[c.RT_PARAM_FREQUENCY] / 1e9:.1f} GHz\n"

    summary_str += "\n[Ray-tracing parameters]\n"

    # Interaction limits
    summary_str += "\nMain interaction limits\n"
    summary_str += f"- Max path depth: {rt_params[c.RT_PARAM_PATH_DEPTH]}\n"
    summary_str += f"- Max reflections: {rt_params[c.RT_PARAM_MAX_REFLECTIONS]}\n"
    summary_str += f"- Max diffractions: {rt_params[c.RT_PARAM_MAX_DIFFRACTIONS]}\n"
    summary_str += f"- Max scatterings: {rt_params[c.RT_PARAM_MAX_SCATTERING]}\n"
    summary_str += f"- Max transmissions: {rt_params[c.RT_PARAM_MAX_TRANSMISSIONS]}\n"

    # Diffuse scattering settings
    summary_str += "\nDiffuse Scattering\n"
    is_diffuse_enabled = rt_params[c.RT_PARAM_MAX_SCATTERING] > 0
    summary_str += f"- Diffuse scattering: {'Enabled' if is_diffuse_enabled else 'Disabled'}\n"
    if is_diffuse_enabled:
        summary_str += f"- Diffuse reflections: {rt_params[c.RT_PARAM_DIFFUSE_REFLECTIONS]}\n"
        summary_str += f"- Diffuse diffractions: {rt_params[c.RT_PARAM_DIFFUSE_DIFFRACTIONS]}\n"
        summary_str += f"- Diffuse transmissions: {rt_params[c.RT_PARAM_DIFFUSE_TRANSMISSIONS]}\n"
        summary_str += f"- Final interaction only: {rt_params[c.RT_PARAM_DIFFUSE_FINAL_ONLY]}\n"
        summary_str += f"- Random phases: {rt_params[c.RT_PARAM_DIFFUSE_RANDOM_PHASES]}\n"

    # Terrain settings
    summary_str += "\nTerrain\n"
    summary_str += f"- Terrain reflection: {rt_params[c.RT_PARAM_TERRAIN_REFLECTION]}\n"
    summary_str += f"- Terrain diffraction: {rt_params[c.RT_PARAM_TERRAIN_DIFFRACTION]}\n"
    summary_str += f"- Terrain scattering: {rt_params[c.RT_PARAM_TERRAIN_SCATTERING]}\n"

    # Ray casting settings
    summary_str += "\nRay Casting Settings\n"
    summary_str += f"- Number of rays: {rt_params[c.RT_PARAM_NUM_RAYS]:,}\n"
    summary_str += f"- Casting method: {rt_params[c.RT_PARAM_RAY_CASTING_METHOD]}\n"
    summary_str += f"- Casting range (az): {rt_params[c.RT_PARAM_RAY_CASTING_RANGE_AZ]:.1f}°\n"
    summary_str += f"- Casting range (el): {rt_params[c.RT_PARAM_RAY_CASTING_RANGE_EL]:.1f}°\n"
    summary_str += f"- Synthetic array: {rt_params[c.RT_PARAM_SYNTHETIC_ARRAY]}\n"

    # Scene
    summary_str += "\n[Scene]\n"
    summary_str += f"- Number of scenes: {scene_params[c.SCENE_PARAM_NUMBER_SCENES]}\n"
    summary_str += f"- Total objects: {scene_params[c.SCENE_PARAM_N_OBJECTS]:,}\n"
    summary_str += f"- Vertices: {scene_params[c.SCENE_PARAM_N_VERTICES]:,}\n"
    summary_str += f"- Faces: {scene_params[c.SCENE_PARAM_N_FACES]:,}\n"
    summary_str += f"- Triangular faces: {scene_params[c.SCENE_PARAM_N_TRIANGULAR_FACES]:,}\n"

    # Materials
    summary_str += "\n[Materials]\n"
    summary_str += f"Total materials: {len(material_params)}\n"
    for mat_props in material_params.values():
        summary_str += f"\n{mat_props[c.MATERIALS_PARAM_NAME_FIELD]}:\n"
        summary_str += f"- Permittivity: {mat_props[c.MATERIALS_PARAM_PERMITTIVITY]:.2f}\n"
        summary_str += f"- Conductivity: {mat_props[c.MATERIALS_PARAM_CONDUCTIVITY]:.2f} S/m\n"
        summary_str += f"- Scattering model: {mat_props[c.MATERIALS_PARAM_SCATTERING_MODEL]}\n"
        summary_str += (
            f"- Scattering coefficient: {mat_props[c.MATERIALS_PARAM_SCATTERING_COEF]:.2f}\n"
        )
        summary_str += (
            f"- Cross-polarization coefficient: {mat_props[c.MATERIALS_PARAM_CROSS_POL_COEF]:.2f}\n"
        )

    # TX/RX
    summary_str += "\n[TX/RX Configuration]\n"

    # Sum total number of receivers and transmitters
    n_rx = sum(
        set_info[c.TXRX_PARAM_NUM_ACTIVE_POINTS]
        for set_info in txrx_params.values()
        if set_info[c.TXRX_PARAM_IS_RX]
    )
    n_tx = sum(
        set_info[c.TXRX_PARAM_NUM_ACTIVE_POINTS]
        for set_info in txrx_params.values()
        if set_info[c.TXRX_PARAM_IS_TX]
    )
    summary_str += f"Total number of receivers: {n_rx}\n"
    summary_str += f"Total number of transmitters: {n_tx}\n"

    for set_name, set_info in txrx_params.items():
        summary_str += f"\n{set_name} ({set_info[c.TXRX_PARAM_NAME_FIELD]}):\n"
        role = []
        if set_info[c.TXRX_PARAM_IS_TX]:
            role.append("TX")
        if set_info[c.TXRX_PARAM_IS_RX]:
            role.append("RX")
        summary_str += f"- Role: {' & '.join(role)}\n"
        summary_str += f"- Total points: {set_info[c.TXRX_PARAM_NUM_POINTS]:,}\n"
        summary_str += f"- Active points: {set_info[c.TXRX_PARAM_NUM_ACTIVE_POINTS]:,}\n"
        summary_str += f"- Antennas per point: {set_info[c.TXRX_PARAM_NUM_ANT]}\n"
        summary_str += f"- Dual polarization: {set_info[c.TXRX_PARAM_DUAL_POL]}\n"

    # GPS Bounding Box
    if rt_params.get(c.RT_PARAM_GPS_BBOX, (0, 0, 0, 0)) != (0, 0, 0, 0):
        summary_str += "\n[GPS Bounding Box]\n"
        summary_str += f"- Min latitude: {rt_params[c.RT_PARAM_GPS_BBOX][0]:.2f}\n"
        summary_str += f"- Min longitude: {rt_params[c.RT_PARAM_GPS_BBOX][1]:.2f}\n"
        summary_str += f"- Max latitude: {rt_params[c.RT_PARAM_GPS_BBOX][2]:.2f}\n"
        summary_str += f"- Max longitude: {rt_params[c.RT_PARAM_GPS_BBOX][3]:.2f}\n"

    # Print summary
    if print_summary:
        print(summary_str)
        return None

    return summary_str


def _plot_3d_scene(dataset: Any, temp_dir: str, timestamp: int, *, save_imgs: bool) -> str | None:
    """Plot 3D scene visualization.

    Args:
        dataset: Dataset to plot
        temp_dir: Directory to save images
        timestamp: Timestamp for unique filename
        save_imgs: Whether to save or show the plot

    Returns:
        Path to saved image if save_imgs is True, None otherwise

    """
    try:
        scene_img_path = str(Path(temp_dir) / f"scene_{timestamp:016d}.png")
        dataset.scene.plot()
        if save_imgs:
            plt.savefig(scene_img_path, dpi=100, bbox_inches="tight")
            plt.close()
            return scene_img_path

        plt.show()
    except (OSError, RuntimeError, ValueError) as e:
        print(f"Error generating 3D scene plot: {e!s}")

    return None


def _plot_scenario_summary_2d(  # noqa: C901, PLR0912 - 2D summary requires checking multiple dataset configurations
    dataset: Any, temp_dir: str, timestamp: int, *, save_imgs: bool
) -> str | None:
    """Plot 2D scenario summary with base stations and users.

    Args:
        dataset: Dataset to plot
        temp_dir: Directory to save images
        timestamp: Timestamp for unique filename
        save_imgs: Whether to save or show the plot

    Returns:
        Path to saved image if save_imgs is True, None otherwise

    """
    try:
        img2_path = str(Path(temp_dir) / f"scenario_summary_{timestamp:016d}.png")
        txrx_sets = dataset.txrx_sets

        tx_set = next(s for s in txrx_sets if s.is_tx)
        n_bs = tx_set.num_points
        ax = dataset.scene.plot(title=False, proj_3D=False)
        bs_colors = ["red", "blue", "green", "yellow", "purple", "orange"]

        # Plot base stations
        for bs in range(n_bs):
            if bs == 0 and n_bs == 1:
                bs_dataset = dataset
                # Workaround for DynamicDataset
                if isinstance(bs_dataset.bs_pos, list):
                    bs_dataset = dataset[0]
                    print("Warning: plot_summary not supported for DynamicDatasets. ")
                    print("Plotting the summary of the first snapshot only. ")
                    print("For all snapshots, use dataset.plot_summary() instead.")
            else:
                bs_dataset = dataset[bs]
            ax.scatter(
                bs_dataset.bs_pos[0, 0],
                bs_dataset.bs_pos[0, 1],
                s=250,
                color=bs_colors[bs],
                label=f"BS {bs + 1}",
                marker="*",
            )

        # Get receiver grid dataset
        if isinstance(dataset.txrx, list):
            rx_set_id = next(s for s in txrx_sets if s.is_rx and not s.is_tx).id
            first_pair_with_rx_grid = next(
                txrx_pair_idx
                for txrx_pair_idx, txrx_dict in enumerate(dataset.txrx)
                if txrx_dict["rx_set_id"] == rx_set_id
            )
            rx_grid_dataset = dataset[first_pair_with_rx_grid]
        else:
            rx_grid_dataset = dataset

        # Select users to plot
        max_users_for_grid = 5000
        if rx_grid_dataset.has_valid_grid() and rx_grid_dataset.n_ue > max_users_for_grid:
            idxs = rx_grid_dataset.get_uniform_idxs([8, 8])
        else:
            idxs = np.arange(rx_grid_dataset.n_ue)

        ax.scatter(
            rx_grid_dataset.rx_pos[idxs, 0],
            rx_grid_dataset.rx_pos[idxs, 1],
            s=10,
            color="red",
            label="users",
            marker="o",
            alpha=0.2,
            zorder=0,
        )

        # Reorder legend handles and labels
        legend_args = {
            "ncol": 4 if n_bs == 1 else 3,
            "loc": "center",
            "bbox_to_anchor": (0.5, 1.0),
            "fontsize": 15,
        }
        three_bs = 3
        two_bs = 2
        if n_bs == three_bs:
            order = [2, 0, 3, 1, 4, 5]
        elif n_bs == two_bs:
            order = [2, 0, 3, 1, 4]
        else:
            order = [0, 1, 2, 3]
        l1 = ax.legend(**legend_args)
        l2 = ax.legend(
            [l1.legend_handles[i] for i in order],
            [l1.get_texts()[i].get_text() for i in order],
            **legend_args,
        )
        l2.set_zorder(1e9)
        for handle, text in zip(l2.legend_handles, l2.get_texts(), strict=False):
            if text.get_text() == "users":  # Match by label
                handle.set_sizes([100])  # marker area (not radius)

        if save_imgs:
            plt.savefig(img2_path, dpi=100, bbox_inches="tight")
            plt.close()
            return img2_path

        plt.show()
    except (OSError, RuntimeError, ValueError) as e:
        print(f"Error generating 2D scenario summary: {e!s}")

    return None


def plot_summary(
    scenario_name: str | None = None,
    *,
    save_imgs: bool = False,
    dataset: Any = None,  # Dataset, MacroDataset, or DynamicDataset
    plot_idx: int | list[int] | None = None,
) -> list[str]:
    """Make images for the scenario.

    Args:
        scenario_name: Scenario name
        dataset: Dataset, MacroDataset, or DynamicDataset. If provided, scenario_name is ignored.
        save_imgs: Whether to save the images to the figures directory
        plot_idx: Index or list of indices of summaries to plot. If None, all summaries are plotted.

    Returns:
        List of paths to generated images

    """
    # Import load function here to avoid circular import
    from .load import load  # noqa: PLC0415

    # Create figures directory if it doesn't exist
    temp_dir = "./figures"
    if save_imgs:
        Path(temp_dir).mkdir(parents=True, exist_ok=True)

    # Load the dataset
    if dataset is None:
        if scenario_name is None:
            msg = "Scenario name is required when dataset is not provided"
            raise ValueError(msg)
        dataset = load(scenario_name)

    # Determine which plots to generate
    if plot_idx is None:
        plot_idx = [0, 1]  # currently only 2 plots are supported
    elif isinstance(plot_idx, int):
        plot_idx = [plot_idx]

    timestamp = int(time.time() * 1000)
    img_paths = []

    # Generate requested plots
    if 0 in plot_idx:
        img_path = _plot_3d_scene(dataset, temp_dir, timestamp, save_imgs=save_imgs)
        if img_path:
            img_paths.append(img_path)

    if 1 in plot_idx:
        img_path = _plot_scenario_summary_2d(dataset, temp_dir, timestamp, save_imgs=save_imgs)
        if img_path:
            img_paths.append(img_path)

    # ISSUE: LoS is BS specific. Are we going to show the LoS for each BS?
    # Image 3: Line of Sight (LOS) - commented out for now

    return img_paths if img_paths else None
