"""DeepMIMO Visualization Module.

This module provides visualization utilities for the DeepMIMO dataset, including:
- Coverage map visualization with customizable parameters
- Path characteristics visualization
- Channel properties plotting
- Data export utilities for external visualization tools

The module uses matplotlib for generating plots and supports both 2D and 3D visualizations.
"""

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Colormap, ListedColormap
from matplotlib.figure import Figure
from tqdm import tqdm

CAT_LABELS_MAX_UNIQUE = 30
VAL_RANGE_THRESHOLD = 100


def _create_colorbar(  # noqa: PLR0913
    scatter_plot: plt.scatter,
    cov_map: np.ndarray,
    cmap: str,
    cbar_title: str = "",
    cat_labels: list[str] | None = None,
    ax: Axes | None = None,
) -> Colorbar:
    """Create a colorbar for the coverage plot, handling both continuous and categorical data.

    Args:
        scatter_plot: The scatter plot object to create colorbar for
        cov_map: The coverage map values used for coloring
        cmap: Matplotlib colormap name
        cbar_title: Title for the colorbar
        cat_labels: Optional labels for categorical values
        ax: The matplotlib axes object to attach the colorbar to

    Returns:
        matplotlib Colorbar object

    """
    fig = ax.figure if ax is not None else plt.gcf()
    valid_data = cov_map[~np.isnan(cov_map)]
    unique_vals = np.sort(np.unique(valid_data))
    n_cats = len(unique_vals)
    if cat_labels is not None and len(cat_labels) != n_cats:
        msg = (
            "Number of category labels ("
            f"{len(cat_labels)}) must match number of unique values ({n_cats})"
        )
        raise ValueError(msg)
    if n_cats < CAT_LABELS_MAX_UNIQUE or cat_labels:
        if isinstance(cmap, str):
            base_cmap = plt.colormaps[cmap]
            colors = base_cmap(np.linspace(0, 1, n_cats))
            cmap = ListedColormap(colors)
            value_to_index = {val: i for (i, val) in enumerate(unique_vals)}
            discrete_data = np.full_like(cov_map, np.nan, dtype=float)
            valid_mask = ~np.isnan(cov_map)
            discrete_data[valid_mask] = [value_to_index[val] for val in cov_map[valid_mask]]
            scatter_plot.set_array(discrete_data)
            scatter_plot.set_cmap(cmap)
        tick_locs = np.arange(n_cats)
        scatter_plot.set_clim(-0.5, n_cats - 0.5)
        cbar = fig.colorbar(
            scatter_plot,
            ax=ax,
            label=cbar_title,
            ticks=tick_locs,
            boundaries=np.arange(-0.5, n_cats + 0.5),
            values=np.arange(n_cats),
        )
        val_range = np.max(unique_vals) - np.min(unique_vals)
        str_labels = [
            str(int(val)) if val_range > VAL_RANGE_THRESHOLD else str(val)
            for val in unique_vals
        ]
        cbar.set_ticklabels(cat_labels if cat_labels else str_labels)
    else:
        cbar = fig.colorbar(scatter_plot, ax=ax, label=cbar_title)
    return cbar


def plot_coverage(  # noqa: PLR0913, C901
    rxs: np.ndarray,
    cov_map: tuple[float, ...] | list[float] | np.ndarray,
    *,
    dpi: int = 100,
    figsize: tuple = (6, 4),
    cbar_title: str = "",
    title: bool | str = False,
    scat_sz: float = 0.5,
    bs_pos: np.ndarray | None = None,
    bs_ori: np.ndarray | None = None,
    legend: bool = False,
    lims: tuple[float, float] | None = None,
    proj_3d: bool = False,
    equal_aspect: bool = False,
    tight: bool = True,
    cmap: str | list = "viridis",
    cbar_labels: list[str] | None = None,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes, Colorbar]:
    """Generate coverage map visualization for user positions.

    This function creates a customizable plot showing user positions colored by
    coverage values, with optional base station position and orientation indicators.

    Args:
        rxs (np.ndarray): User position array with shape (n_users, 3)
        cov_map (tuple[float, ...] | list[float] | np.ndarray): Coverage map values for coloring
        dpi (int): Plot resolution in dots per inch. Defaults to 300.
        figsize (Tuple[int, int]): Figure dimensions (width, height) in inches. Defaults to (6,4).
        cbar_title (str): Title for the colorbar. Defaults to ''.
        title (bool | str): Plot title. Defaults to False.
        scat_sz (float): Size of scatter markers. Defaults to 0.5.
        bs_pos (Optional[np.ndarray]): Base station position coordinates. Defaults to None.
        bs_ori (Optional[np.ndarray]): Base station orientation angles. Defaults to None.
        legend (bool): Whether to show plot legend. Defaults to False.
        lims (Optional[Tuple[float, float]]): Color scale limits (min, max). Defaults to None.
        proj_3d (bool): Whether to create 3D projection. Defaults to False.
        equal_aspect (bool): Whether to maintain equal axis scaling. Defaults to False.
        tight (bool): Whether to set tight axis limits around data points. Defaults to True.
        cmap (str | list): Matplotlib colormap name or list of colors. Defaults to 'viridis'.
        cbar_labels (Optional[list[str]]): List of labels for the colorbar. Defaults to None.
        ax (Optional[Axes]): Matplotlib Axes object. Defaults to None.
        **kwargs: Additional keyword-only options; accepts `proj_3D` alias.

    Returns:
        Tuple containing:
        - matplotlib Figure object
        - matplotlib Axes object
        - matplotlib Colorbar object

    """
    if "proj_3D" in kwargs:
        proj_3d = kwargs.pop("proj_3D")
    if kwargs:
        unexpected = ", ".join(kwargs)
        msg = f"Unexpected keyword arguments: {unexpected}"
        raise TypeError(msg)

    cmap = cmap if isinstance(cmap, (str, Colormap)) else ListedColormap(cmap)
    plt_params = {"cmap": cmap}
    if lims:
        (plt_params["vmin"], plt_params["vmax"]) = (lims[0], lims[1])
    n = 3 if proj_3d else 2
    two_d_dim = 2  # dimension count for 2D projections
    xyz_arg_names = [
        "x" if n == two_d_dim else "xs",
        "y" if n == two_d_dim else "ys",
        "zs",
    ]
    xyz = {s: rxs[:, i] for (s, i) in zip(xyz_arg_names, range(n), strict=False)}
    if not ax:
        (_, ax) = plt.subplots(
            dpi=dpi, figsize=figsize, subplot_kw={"projection": "3d"} if proj_3d else {}
        )
    cov_map = np.array(cov_map) if isinstance(cov_map, list) else cov_map
    im = ax.scatter(**xyz, c=cov_map, s=scat_sz, marker="s", **plt_params)
    cbar = _create_colorbar(im, cov_map, cmap, cbar_title, cbar_labels, ax)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    if proj_3d:
        ax.set_zlabel("z (m)")
    if bs_pos is not None:
        bs_pos = bs_pos.squeeze()
        ax.scatter(*bs_pos[:n], marker="P", c="r", label="TX")
    if bs_ori is not None and bs_pos is not None:
        r = 30
        tx_lookat = np.copy(bs_pos)
        tx_lookat[:2] += r * np.array([np.cos(bs_ori[2]), np.sin(bs_ori[2])])
        tx_lookat[2] -= r / 10 * np.sin(bs_ori[1])
        line_components = [[bs_pos[i], tx_lookat[i]] for i in range(n)]
        ax.plot(*line_components, c="k", alpha=0.5, zorder=3)
    if title:
        ax.set_title(title)
    if legend:
        ax.legend(loc="upper center", ncols=10, framealpha=0.5)
    if tight:
        s = 1
        all_points = np.vstack([rxs, bs_pos.reshape(1, -1)]) if bs_pos is not None else rxs
        (mins, maxs) = (np.min(all_points, axis=0) - s, np.max(all_points, axis=0) + s)
        ax.set_xlim([mins[0], maxs[0]])
        ax.set_ylim([mins[1], maxs[1]])
        if proj_3d:
            ax.axes.set_zlim3d([mins[2], maxs[2]])
    if equal_aspect:
        ax.set_aspect("equal")
    return (ax, cbar)


def transform_coordinates(
    coords: Any, lon_max: Any, lon_min: Any, lat_min: Any, lat_max: Any
) -> Any:
    """Transform Cartesian coordinates to geographical coordinates.

    This function converts x,y coordinates from a local Cartesian coordinate system
    to latitude/longitude coordinates using linear interpolation between provided bounds.

    Args:
        coords (np.ndarray): Array of shape (N,2) or (N,3) containing x,y coordinates
        lon_max (float): Maximum longitude value for output range
        lon_min (float): Minimum longitude value for output range
        lat_min (float): Minimum latitude value for output range
        lat_max (float): Maximum latitude value for output range

    Returns:
        Tuple[List[float], List[float]]: Lists of transformed latitudes and longitudes

    """
    lats = []
    lons = []
    (x_min, y_min) = np.min(coords, axis=0)[:2]
    (x_max, y_max) = np.max(coords, axis=0)[:2]
    for x, y in zip(coords[:, 0], coords[:, 1], strict=False):
        lons += [lon_min + (x - x_min) / (x_max - x_min) * (lon_max - lon_min)]
        lats += [lat_min + (y - y_min) / (y_max - y_min) * (lat_max - lat_min)]
    return (lats, lons)


def export_xyz_csv(  # noqa: PLR0913
    data: dict[str, Any],
    z_var: np.ndarray,
    outfile: str = "",
    *,
    google_earth: bool = False,
    lat_min: float = 33.418081,
    lat_max: float = 33.420961,
    lon_min: float = -111.932875,
    lon_max: float = -111.928567,
) -> None:
    """Export user locations and values to CSV format.

    This function generates a CSV file containing x,y,z coordinates that can be
    imported into visualization tools like Blender or Google Earth. It supports
    both Cartesian and geographical coordinate formats.

    Args:
        data (Dict[str, Any]): DeepMIMO dataset for one basestation
        z_var (np.ndarray): Values to use for z-coordinate or coloring
        outfile (str): Output CSV file path. Defaults to ''.
        google_earth (bool): Whether to convert coordinates to geographical format.
            Defaults to False.
        lat_min (float): Minimum latitude for coordinate conversion. Defaults to 33.418081.
        lat_max (float): Maximum latitude for coordinate conversion. Defaults to 33.420961.
        lon_min (float): Minimum longitude for coordinate conversion. Defaults to -111.932875.
        lon_max (float): Maximum longitude for coordinate conversion. Defaults to -111.928567.

    Returns:
        None. Writes data to CSV file.

    """
    user_idxs = np.where(data["user"]["LoS"] != -1)[0]
    locs = data["user"]["location"][user_idxs]
    if google_earth:
        (lats, lons) = transform_coordinates(
            locs, lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max
        )
    else:
        (lats, lons) = (locs[:, 0], locs[:, 1])
    data_dict = {
        "latitude" if google_earth else "x": lats if google_earth else locs[:, 0],
        "longitude" if google_earth else "y": lons if google_earth else locs[:, 1],
        "z": z_var[user_idxs],
    }
    if not outfile:
        outfile = "test.csv"
    with Path(outfile).open(mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(data_dict.keys())
        writer.writerows(zip(*data_dict.values(), strict=False))


def plot_rays(  # noqa: PLR0912, PLR0913, PLR0915, C901
    rx_loc: np.ndarray,
    tx_loc: np.ndarray,
    inter_pos: np.ndarray,
    inter: np.ndarray,
    *,
    figsize: tuple = (10, 8),
    dpi: int = 100,
    proj_3d: bool = True,
    color_by_type: bool = True,
    inter_objects: np.ndarray | None = None,
    inter_obj_labels: list[str] | None = None,
    color_rays_by_pwr: bool = False,
    powers: np.ndarray | None = None,
    show_cbar: bool = False,
    limits: tuple[float, float] | None = None,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot ray paths between transmitter and receiver with interaction points.

    For a given user, plots all ray paths connecting TX and RX through their
    respective interaction points. Each path is drawn as a sequence of segments
    connecting TX -> interaction points -> RX.

    Args:
        rx_loc (np.ndarray): Receiver location array with shape [3,]
        tx_loc (np.ndarray): Transmitter location array with shape [3,]
        inter_pos (np.ndarray): Interaction positions with shape [n_paths, max_interactions, 3]
            where n_paths is the number of rays for this user
        inter (np.ndarray): Interaction types with shape [n_paths,]
            where each path's value contains digits representing interaction types
            (e.g., 211 means type 2 for first bounce, type 1 for second and third)
        figsize (tuple, optional): Figure size in inches. Defaults to (10,8).
        dpi (int, optional): Resolution in dots per inch. Defaults to 300.
        proj_3d (bool, optional): Whether to create 3D projection. Defaults to True.
        color_by_type (bool, optional): Whether to color interaction points by their
            type. Defaults to False.
        inter_objects (Optional[np.ndarray], optional): Object ids at each
            interaction point. Defaults to None. If provided, will color the
            interaction points by the object id, and ignore the interaction type.
        inter_obj_labels (Optional[list[str]], optional): Labels for the interaction
            objects. Defaults to None. If provided, will use these labels instead of
            the object ids.
        color_rays_by_pwr (bool, optional): Whether to color rays by their power.
            Defaults to False.
        powers (Optional[np.ndarray], optional): Power values for each path.
            Required if color_rays_by_pwr is True.
        show_cbar (bool, optional): Whether to show the colorbar. Defaults to False.
        limits (Optional[Tuple[float, float]], optional): Power limits for coloring
            (min, max). If None, uses relative scaling.
        ax (Optional[Axes], optional): Matplotlib Axes object. Defaults to None.
            When provided, the figure and axes are not created.
        **kwargs: Additional keyword-only options; accepts `proj_3D` alias.

    Returns:
        Tuple containing:
        - matplotlib Figure object
        - matplotlib Axes object

    """
    if "proj_3D" in kwargs:
        proj_3d = kwargs.pop("proj_3D")
    if kwargs:
        unexpected = ", ".join(kwargs)
        msg = f"Unexpected keyword arguments: {unexpected}"
        raise TypeError(msg)

    if not ax:
        (_, ax) = plt.subplots(
            dpi=dpi, figsize=figsize, subplot_kw={"projection": "3d"} if proj_3d else {}
        )
    rx_loc = np.asarray(rx_loc)
    tx_loc = np.asarray(tx_loc)
    inter_pos = np.asarray(inter_pos)
    inter = np.asarray(inter)
    n_valid_paths = np.sum(~np.isnan(inter))

    def plot_line(start_point: Any, end_point: Any, **kwargs: Any) -> None:
        coords = [(start_point[i], end_point[i]) for i in range(3 if proj_3d else 2)]
        ax.plot(*coords, **kwargs)

    def plot_point(point: Any, **kwargs: Any) -> None:
        coords = point[:3] if proj_3d else point[:2]
        ax.scatter(*coords, **kwargs)

    interaction_colors = {0: "green", 1: "red", 2: "orange", 3: "blue", 4: "purple", -1: "gray"}
    interaction_names = {
        0: "Line-of-sight",
        1: "Reflection",
        2: "Diffraction",
        3: "Scattering",
        4: "Transmission",
        -1: "Unknown",
    }
    if inter_objects is not None:
        unique_objs = np.unique(inter_objects)
        inter_obj_colors = {obj_id: f"C{i}" for (i, obj_id) in enumerate(unique_objs)}
        if inter_obj_labels is None:
            inter_obj_labels = {obj_id: str(int(obj_id)) for obj_id in unique_objs}
    if color_rays_by_pwr:
        if powers is None:
            msg = "Powers must be provided when color_rays_by_pwr is True"
            raise ValueError(msg)
        cmap = plt.get_cmap("jet")
        if limits is not None:
            (vmin, vmax) = limits
        else:
            (vmin, vmax) = (np.nanmin(powers), np.nanmax(powers))
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        if show_cbar:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label="Power (dBm)")
    for path_idx in range(n_valid_paths):
        valid_inters = ~np.any(np.isnan(inter_pos[path_idx]), axis=1)
        path_interactions = inter_pos[path_idx][valid_inters]
        path_points = np.vstack([tx_loc, path_interactions, rx_loc])
        path_type_int = int(inter[path_idx])
        path_types = [] if path_type_int == 0 else [int(d) for d in str(path_type_int)]
        is_los = len(path_interactions) == 0
        ray_color = cmap(norm(powers[path_idx])) if color_rays_by_pwr else "g" if is_los else "r"
        ray_plt_args = {
            "color": ray_color,
            "alpha": 1 if is_los else 0.5,
            "zorder": 2 if is_los else 1,
            "linewidth": 2 if is_los else 1,
        }
        if is_los:
            plot_line(
                path_points[0],
                path_points[1],
                **ray_plt_args,
                label="LoS" if not color_rays_by_pwr else None,
            )
            continue
        for i in range(len(path_points) - 1):
            plot_line(path_points[i], path_points[i + 1], **ray_plt_args)
        if len(path_interactions) > 0:
            for i, pos in enumerate(path_interactions):
                if color_by_type and i < len(path_types) and (inter_objects is None):
                    point_color = interaction_colors.get(path_types[i], "gray")
                    point_label = interaction_names.get(path_types[i], "Unknown")
                elif inter_objects is not None:
                    point_color = inter_obj_colors.get(inter_objects[path_idx, i], "gray")
                    point_label = inter_obj_labels.get(inter_objects[path_idx, i], "unknown obj?")
                else:
                    print(f"Unclassified interaction point: path {path_idx}, inter {i}")
                    point_color = "black"
                    point_label = None
                plot_point(pos, c=point_color, marker="o", s=10, label=point_label, zorder=2)
    plot_point(tx_loc, c="white", marker="^", s=100, label="TX", zorder=3, edgecolors="black")
    plot_point(rx_loc, c="white", marker="v", s=100, label="RX", zorder=3, edgecolors="black")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    if proj_3d:
        ax.set_zlabel("z (m)")
    if color_by_type or inter_objects is not None:
        (handles, labels) = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles, strict=False))
        legend = ax.legend(by_label.values(), by_label.keys())
    else:
        legend = ax.legend()
    legend.set_bbox_to_anchor((1, 0.9))
    if not proj_3d:
        ax.set_aspect("equal")
    ax.grid()
    return ax


def plot_power_discarding(dataset: Any, trim_delay: float | None = None) -> tuple[Figure, Axes]:
    """Analyze and visualize power discarding due to path delays.

    This function analyzes what percentage of power would be discarded for each user
    if paths arriving after a certain delay are trimmed. It provides both statistical
    analysis and visualization of the power discarding distribution.

    Args:
        dataset: DeepMIMO dataset containing delays and powers
        trim_delay (Optional[float]): Delay threshold in seconds. Paths arriving after
            this delay will be considered discarded. If None, uses OFDM symbol duration
            from dataset's channel parameters. Defaults to None.
        figsize (tuple): Figure size in inches. Defaults to (12, 5).

    Returns:
        Tuple containing:
        - matplotlib Figure object
        - List of matplotlib Axes objects [stats_ax, hist_ax]

    """
    if trim_delay is None:
        if not hasattr(dataset, "channel_params"):
            msg = "Dataset has no channel parameters. Please provide trim_delay explicitly."
            raise ValueError(msg)
        trim_delay = dataset.channel_params.ofdm.subcarriers / dataset.channel_params.ofdm.bandwidth
    if np.nanmax(dataset.delay) < trim_delay:
        print(f"Maximum path delay: {np.nanmax(dataset.delay) * 1000000.0:.1f} μs")
        print(f"Trim delay: {trim_delay * 1000000.0:.1f} μs")
        print("No paths will be discarded.")
        return (None, None)
    discarded_power_ratios = []
    n_users = len(dataset.delay)
    for user_idx in tqdm(range(n_users), desc="Calculating discarded power ratios per user"):
        user_delays = dataset.delay[user_idx]
        user_powers = dataset.power_linear[user_idx]
        valid_mask = ~np.isnan(user_delays) & ~np.isnan(user_powers)
        if not np.any(valid_mask):
            discarded_power_ratios.append(0)
            continue
        valid_delays = user_delays[valid_mask]
        valid_powers = user_powers[valid_mask]
        discarded_mask = valid_delays > trim_delay
        if not np.any(discarded_mask):
            discarded_power_ratios.append(0)
            continue
        total_power = np.sum(valid_powers)
        discarded_power = np.sum(valid_powers[discarded_mask])
        discarded_power_ratios.append(discarded_power / total_power * 100)
    discarded_power_ratios = np.array(discarded_power_ratios)
    max_discard_idx = np.argmax(discarded_power_ratios)
    max_discard_ratio = discarded_power_ratios[max_discard_idx]
    mean_discard_ratio = np.mean(discarded_power_ratios)
    affected_users = np.sum(discarded_power_ratios > 0)
    non_zero_ratios = discarded_power_ratios[discarded_power_ratios > 0]
    print("\nPower Discarding Analysis")
    print("=" * 50)
    print(f"\nTrim delay: {trim_delay * 1000000.0:.1f} μs")
    print(f"Maximum delay: {np.nanmax(dataset.delay) * 1000000.0:.1f} μs\n")
    print(f"Maximum power discarded: {max_discard_ratio:.1f}%")
    print(f"Average power discarded: {mean_discard_ratio:.1f}%")
    print(f"Users with discarded paths: {affected_users}")
    (fig, ax) = plt.subplots(dpi=200, figsize=(6, 4))
    ax.hist(non_zero_ratios, bins=20)
    ax.set_title("Distribution of Discarded Power")
    ax.set_xlabel("Discarded Power (%)")
    ax.set_ylabel("Number of Users")
    ax.grid(visible=True)
    plt.tight_layout()
    return (fig, ax)
