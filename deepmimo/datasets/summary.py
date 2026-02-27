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

3. stats(scen_name, print_summary=True, bs_idx=0)
    - Prints detailed scenario statistics by default.
    - If print_summary is False, returns a string summary of the stats.
    - Supports selecting a base station index for multi-BS scenarios.

"""

# Standard library imports
import time
from pathlib import Path
from typing import Any

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath
from scipy.spatial import ConvexHull, QhullError

from deepmimo import consts as c
from deepmimo.utils import get_params_path, load_dict_from_json

_MIN_HULL_POINTS_2D = 3
_MIN_HULL_POINTS_3D = 4
_MIN_RANK_FOR_VOLUME = 3


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


def stats(  # noqa: C901, PLR0912, PLR0915
    scen_name: str, *, print_summary: bool = True, bs_idx: int = 0
) -> str | None:
    """Calculate and return a summary of the scenario statistics.

    Args:
        scen_name (str): Name of the scenario to summarize.
        print_summary (bool): Whether to print the summary (default: True).
        bs_idx (int): Base station index for statistics (default: 0, first base station).

    Returns:
        str | None: Summary string if print_summary is False, None otherwise.

    """
    # Imported lazily to avoid circular import during package initialization.
    from .dataset import DynamicDataset, MacroDataset  # noqa: PLC0415
    from .load import load  # noqa: PLC0415

    # Load the dataset for statistics calculation
    print("Calculating scenario statistics...")
    dataset = load(scen_name)

    # Handle DynamicDataset -> use first snapshot for summary statistics
    if isinstance(dataset, DynamicDataset):
        if len(dataset) == 0:
            msg = f"Dynamic scenario '{scen_name}' contains no snapshots"
            raise ValueError(msg)
        print("Dynamic scenario detected. Using snapshot 1 for statistics.")
        dataset = dataset[0]

    # Handle MacroDataset vs single Dataset
    if isinstance(dataset, MacroDataset):
        datasets = dataset.datasets
        if len(datasets) == 0:
            msg = f"Scenario '{scen_name}' contains no TX/RX dataset pairs"
            raise ValueError(msg)

        # MacroDataset may include multiple RX grids per BS.
        # Group by TX position so bs_idx maps to unique BS locations.
        bs_to_pair_indices = {}
        for pair_idx, pair_ds in enumerate(datasets):
            tx_pos = np.asarray(pair_ds.tx_pos, dtype=float)
            tx_vec = tx_pos if tx_pos.ndim == 1 else tx_pos.reshape(-1, tx_pos.shape[-1])[0]
            tx_key = tuple(np.round(tx_vec, decimals=6))
            if tx_key not in bs_to_pair_indices:
                bs_to_pair_indices[tx_key] = []
            bs_to_pair_indices[tx_key].append(pair_idx)

        bs_pair_groups = list(bs_to_pair_indices.values())
        n_bs = len(bs_pair_groups)
        selected_bs_idx = bs_idx if 0 <= bs_idx < n_bs else 0
        if selected_bs_idx != bs_idx:
            print(f"Warning: Base Station {bs_idx + 1} not found, using Base Station 1 instead")

        selected_pair_indices = bs_pair_groups[selected_bs_idx]
        selected_pair_idx = selected_pair_indices[0]
        stats_dataset = datasets[selected_pair_idx]
        print(f"Using statistics for Base Station {selected_bs_idx + 1}")
        if len(selected_pair_indices) > 1:
            print(
                f"Note: Base Station {selected_bs_idx + 1} has {len(selected_pair_indices)} "
                "TX/RX pairs; using the first pair."
            )
    else:
        # Single Dataset case (one base station)
        stats_dataset = dataset
        if bs_idx != 0:
            print(
                "Warning: This is a single-BS scenario. Base Station "
                f"{bs_idx + 1} does not exist, using the only available base station"
            )

    # Calculate path statistics
    num_paths = stats_dataset.num_paths
    los_status = stats_dataset.los
    valid_users = num_paths > 0

    def _robust_stats(values: np.ndarray) -> tuple[float, float, float, float]:
        """Return median, IQR, p10, p90 for a 1D numeric array."""
        median = float(np.median(values))
        p25 = float(np.percentile(values, 25))
        p75 = float(np.percentile(values, 75))
        p10 = float(np.percentile(values, 10))
        p90 = float(np.percentile(values, 90))
        iqr = p75 - p25
        return median, iqr, p10, p90

    # Path Statistics
    summary_str = "\nPath Statistics:\n"
    summary_str += f"- Average paths per user: {np.mean(num_paths):.1f}\n"
    summary_str += f"- Max paths per user: {np.max(num_paths)}\n"
    summary_str += f"- Min paths per user: {np.min(num_paths)}\n"

    # LOS/NLOS/No paths percentages
    los_users = np.sum(los_status == 1)
    nlos_users = np.sum(los_status == 0)
    no_path_users = np.sum(los_status == -1)
    total_users = len(los_status)

    summary_str += f"- LOS percentage: {100.0 * los_users / total_users:.1f}%\n"
    summary_str += f"- NLOS percentage: {100.0 * nlos_users / total_users:.1f}%\n"
    summary_str += f"- No paths percentage: {100.0 * no_path_users / total_users:.1f}%\n"

    # Average and maximum interactions per path
    num_interactions = stats_dataset.num_interactions
    valid_interactions = num_interactions[~np.isnan(num_interactions)]
    if len(valid_interactions) > 0:
        summary_str += f"- Average interactions per path: {np.mean(valid_interactions):.1f}\n"
        summary_str += f"- Maximum interactions: {int(np.max(valid_interactions))}\n"
    else:
        summary_str += "- Average interactions per path: 0.0\n"
        summary_str += "- Maximum interactions: 0\n"

    # Power Statistics
    summary_str += "\nPower Statistics:\n"
    pathloss = stats_dataset.pathloss
    valid_pathloss = pathloss[~np.isnan(pathloss)]
    if len(valid_pathloss) > 0:
        summary_str += f"- Average pathloss: {np.mean(valid_pathloss):.1f} dB\n"
        summary_str += f"- Min pathloss: {np.min(valid_pathloss):.1f} dB\n"
        summary_str += f"- Max pathloss: {np.max(valid_pathloss):.1f} dB\n"
        pl_median, pl_iqr, pl_p10, pl_p90 = _robust_stats(valid_pathloss)
        summary_str += f"- Median pathloss: {pl_median:.1f} dB\n"
        summary_str += f"- Pathloss IQR (p75-p25): {pl_iqr:.1f} dB\n"
        summary_str += f"- Pathloss p10/p90: {pl_p10:.1f}/{pl_p90:.1f} dB\n"
    else:
        summary_str += "- Average pathloss: N/A\n"
        summary_str += "- Min pathloss: N/A\n"
        summary_str += "- Max pathloss: N/A\n"
        summary_str += "- Median pathloss: N/A\n"
        summary_str += "- Pathloss IQR (p75-p25): N/A\n"
        summary_str += "- Pathloss p10/p90: N/A\n"

    # Delay Statistics
    summary_str += "\nDelay Statistics:\n"
    delays = stats_dataset.delay
    valid_delays = delays[~np.isnan(delays)]
    if len(valid_delays) > 0:
        summary_str += f"- Min delay: {np.min(valid_delays) * 1e9:.1f} ns\n"
        summary_str += f"- Max delay: {np.max(valid_delays) * 1e9:.1f} ns\n"
        summary_str += f"- Average delay: {np.mean(valid_delays) * 1e9:.1f} ns\n"

        # Calculate RMS delay spread per user
        rms_delays = []
        for i in range(stats_dataset.n_ue):
            user_delays = delays[i]
            user_powers = stats_dataset.power_linear[i]
            valid_mask = ~np.isnan(user_delays) & ~np.isnan(user_powers)
            if np.sum(valid_mask) > 1:
                valid_user_delays = user_delays[valid_mask]
                valid_user_powers = user_powers[valid_mask]
                mean_delay = np.sum(valid_user_delays * valid_user_powers) / np.sum(
                    valid_user_powers
                )
                rms_delay = np.sqrt(
                    np.sum((valid_user_delays - mean_delay) ** 2 * valid_user_powers)
                    / np.sum(valid_user_powers)
                )
                rms_delays.append(rms_delay)

        if len(rms_delays) > 0:
            summary_str += f"- Average RMS delay spread: {np.mean(rms_delays) * 1e9:.1f} ns\n"
            summary_str += f"- Maximum RMS delay spread: {np.max(rms_delays) * 1e9:.1f} ns\n"
        else:
            summary_str += "- Average RMS delay spread: N/A\n"
            summary_str += "- Maximum RMS delay spread: N/A\n"
    else:
        summary_str += "- Min delay: N/A\n"
        summary_str += "- Max delay: N/A\n"
        summary_str += "- Average delay: N/A\n"
        summary_str += "- Average RMS delay spread: N/A\n"
        summary_str += "- Maximum RMS delay spread: N/A\n"

    # Coverage Statistics
    summary_str += "\nCoverage Statistics:\n"
    covered_users = np.sum(valid_users)
    coverage_pct = 100.0 * covered_users / total_users
    los_coverage_pct = 100.0 * los_users / total_users
    summary_str += f"- Coverage percentage: {coverage_pct:.1f}%\n"
    summary_str += f"- LOS coverage percentage: {los_coverage_pct:.1f}%\n"

    if covered_users > 0:
        avg_paths_covered = np.mean(num_paths[valid_users])
        summary_str += f"- Average paths per covered user: {avg_paths_covered:.1f}\n"
    else:
        summary_str += "- Average paths per covered user: 0.0\n"

    # Spatial Statistics
    summary_str += "\nSpatial Statistics:\n"
    distances = stats_dataset.distance
    valid_distances = distances[~np.isnan(distances)]
    if len(valid_distances) > 0:
        summary_str += f"- Min distance to BS: {np.min(valid_distances):.1f} m\n"
        summary_str += f"- Max distance to BS: {np.max(valid_distances):.1f} m\n"
        summary_str += f"- Average distance to BS: {np.mean(valid_distances):.1f} m\n"
        dist_median, dist_iqr, dist_p10, dist_p90 = _robust_stats(valid_distances)
        summary_str += f"- Median distance to BS: {dist_median:.1f} m\n"
        summary_str += f"- Distance IQR (p75-p25): {dist_iqr:.1f} m\n"
        summary_str += f"- Distance p10/p90: {dist_p10:.1f}/{dist_p90:.1f} m\n"
    else:
        summary_str += "- Min distance to BS: N/A\n"
        summary_str += "- Max distance to BS: N/A\n"
        summary_str += "- Average distance to BS: N/A\n"
        summary_str += "- Median distance to BS: N/A\n"
        summary_str += "- Distance IQR (p75-p25): N/A\n"
        summary_str += "- Distance p10/p90: N/A\n"

    # Scene Dimensions
    summary_str += "\nScene Dimensions:\n"
    scene_bb = stats_dataset.scene.bounding_box
    summary_str += f"- Width: {scene_bb.width:.1f} m\n"
    summary_str += f"- Length: {scene_bb.length:.1f} m\n"
    summary_str += f"- Height: {scene_bb.height:.1f} m\n"
    summary_str += f"- Total area: {scene_bb.width * scene_bb.length:.1f} m²\n"
    summary_str += f"- Total volume: {scene_bb.width * scene_bb.length * scene_bb.height:.1f} m³\n"

    # Object Distribution
    summary_str += "\nObject Distribution:\n"
    scene_objects = stats_dataset.scene.objects
    object_counts = {}
    for obj in scene_objects:
        label = obj.label
        object_counts[label] = object_counts.get(label, 0) + 1

    for label, count in object_counts.items():
        summary_str += f"- {label.capitalize()}: {count}\n"

    # Building Characteristics
    buildings = [obj for obj in scene_objects if obj.label == "buildings"]
    if buildings:
        summary_str += "\nBuilding Characteristics:\n"
        building_heights = [obj.height for obj in buildings]

        summary_str += f"- Average height: {np.mean(building_heights):.1f} m\n"
        summary_str += (
            f"- Height range: {np.min(building_heights):.1f} - {np.max(building_heights):.1f} m\n"
        )
        h_median, h_iqr, h_p10, h_p90 = _robust_stats(np.asarray(building_heights, dtype=float))
        summary_str += f"- Median height: {h_median:.1f} m\n"
        summary_str += f"- Height IQR (p75-p25): {h_iqr:.1f} m\n"
        summary_str += f"- Height p10/p90: {h_p10:.1f}/{h_p90:.1f} m\n"

        # Compute volume and footprint over all buildings for exact totals
        # Use 2D convex hull area directly here for accuracy.
        def _exact_footprint_area(obj: Any) -> float:
            points_2d = np.unique(obj.vertices[:, :2], axis=0)
            if points_2d.shape[0] < _MIN_HULL_POINTS_2D:
                return 0.0
            try:
                return float(ConvexHull(points_2d).volume)
            except QhullError:
                return 0.0

        def _exact_volume(obj: Any) -> float:
            points_3d = np.unique(obj.vertices, axis=0)
            if points_3d.shape[0] < _MIN_HULL_POINTS_3D:
                return 0.0
            # Coplanar/collinear point sets are effectively zero-volume.
            if np.linalg.matrix_rank(points_3d - points_3d[0]) < _MIN_RANK_FOR_VOLUME:
                return 0.0
            try:
                return float(ConvexHull(points_3d).volume)
            except QhullError:
                return 0.0

        building_volumes = np.array([_exact_volume(obj) for obj in buildings], dtype=float)
        building_footprints = np.array(
            [_exact_footprint_area(obj) for obj in buildings],
            dtype=float,
        )
        avg_volume = np.mean(building_volumes)
        avg_footprint = np.mean(building_footprints)
        total_volume = np.sum(building_volumes)
        total_footprint = np.sum(building_footprints)

        summary_str += f"- Average volume: {avg_volume:.1f} m³\n"
        summary_str += f"- Total volume: {total_volume:.1f} m³\n"
        summary_str += f"- Average footprint: {avg_footprint:.1f} m²\n"
        summary_str += f"- Total footprint: {total_footprint:.1f} m²\n"

        # Building density over RX users (bounded to [0, 100] and directly comparable to no-path %).
        rx_xy = np.asarray(stats_dataset.rx_pos)[:, :2]
        inside_building_mask = np.zeros(rx_xy.shape[0], dtype=bool)

        for obj in buildings:
            points_2d = np.unique(obj.vertices[:, :2], axis=0)
            if points_2d.shape[0] < _MIN_HULL_POINTS_2D:
                continue
            try:
                hull = ConvexHull(points_2d)
            except QhullError:
                continue

            poly = points_2d[hull.vertices]
            x_min, y_min = np.min(poly, axis=0)
            x_max, y_max = np.max(poly, axis=0)

            # Fast bounding-box prefilter before polygon check.
            candidate_mask = (
                ~inside_building_mask
                & (rx_xy[:, 0] >= x_min)
                & (rx_xy[:, 0] <= x_max)
                & (rx_xy[:, 1] >= y_min)
                & (rx_xy[:, 1] <= y_max)
            )
            if not np.any(candidate_mask):
                continue

            in_poly = MplPath(poly, closed=True).contains_points(
                rx_xy[candidate_mask], radius=1e-9
            )
            candidate_idxs = np.where(candidate_mask)[0]
            inside_building_mask[candidate_idxs[in_poly]] = True

        building_density = 100.0 * np.sum(inside_building_mask) / total_users
        summary_str += f"- Building density (RX users): {building_density:.1f}%\n"

    # Terrain Characteristics
    terrain_objects = [obj for obj in scene_objects if obj.label == "terrain"]
    if terrain_objects:
        summary_str += "\nTerrain Characteristics:\n"
        terrain_heights = [obj.height for obj in terrain_objects]
        if terrain_heights:
            summary_str += (
                f"- Height range: {np.min(terrain_heights):.1f} - {np.max(terrain_heights):.1f} m\n"
            )
            summary_str += f"- Average height: {np.mean(terrain_heights):.1f} m\n"
            summary_str += f"- Height std dev: {np.std(terrain_heights):.1f} m\n"
            elevation_change = np.max(terrain_heights) - np.min(terrain_heights)
            summary_str += f"- Total elevation change: {elevation_change:.1f} m\n"
        else:
            summary_str += "- Height range: 0.0 - 0.0 m\n"
            summary_str += "- Average height: 0.0 m\n"
            summary_str += "- Height std dev: 0.0 m\n"
            summary_str += "- Total elevation change: 0.0 m\n"

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

    return img_paths or None
