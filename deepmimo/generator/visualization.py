"""
DeepMIMO Visualization Module.

This module provides visualization utilities for the DeepMIMO dataset, including:
- Coverage map visualization with customizable parameters
- Path characteristics visualization
- Channel properties plotting
- Data export utilities for external visualization tools

The module uses matplotlib for generating plots and supports both 2D and 3D visualizations.
"""

# Standard library imports
import csv
from typing import Optional, Tuple, Dict, Any, List

# Third-party imports
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.colors import ListedColormap


def _create_colorbar(scatter_plot: plt.scatter, cov_map: np.ndarray, cmap: str,
                     cbar_title: str = '', cat_labels: Optional[List[str]] = None,
                     ax: Optional[Axes] = None) -> Colorbar:
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
    # Get the figure from the axes
    fig = ax.figure if ax is not None else plt.gcf()
    
    # Remove NaN values for unique value calculation
    valid_data = cov_map[~np.isnan(cov_map)]
    unique_vals = np.sort(np.unique(valid_data))
    n_cats = len(unique_vals)
    
    if cat_labels is not None and len(cat_labels) != n_cats:
        raise ValueError(f"Number of category labels ({len(cat_labels)}) "
                         f"must match number of unique values ({n_cats})")
    
    if n_cats < 10 or cat_labels:  # Use discrete colorbar for small number of unique values
        # Create discrete colormap
        if isinstance(cmap, str):
            # Get base colors from the colormap
            base_cmap = plt.colormaps[cmap]
            # Create discrete colors
            colors = base_cmap(np.linspace(0, 1, n_cats))
            # Create discrete colormap
            cmap = ListedColormap(colors)
            
            # Map data to discrete indices, handling NaN values
            value_to_index = {val: i for i, val in enumerate(unique_vals)}
            discrete_data = np.full_like(cov_map, np.nan, dtype=float)
            valid_mask = ~np.isnan(cov_map)
            discrete_data[valid_mask] = [value_to_index[val] for val in cov_map[valid_mask]]
            
            # Update scatter plot with discrete data and colormap
            scatter_plot.set_array(discrete_data)
            scatter_plot.set_cmap(cmap)
        
        # Set colorbar ticks at category centers
        tick_locs = np.arange(n_cats)
        scatter_plot.set_clim(-0.5, n_cats - 0.5)
        
        # Create colorbar with centered ticks
        cbar = fig.colorbar(scatter_plot, ax=ax, label=cbar_title, ticks=tick_locs,
                            boundaries=np.arange(-0.5, n_cats + 0.5),
                            values=np.arange(n_cats))
        
        # Set tick labels
        cbar.set_ticklabels(cat_labels if cat_labels else [str(val) for val in unique_vals])
    else:  # Use continuous colorbar for many unique values
        cbar = fig.colorbar(scatter_plot, ax=ax, label=cbar_title)
    
    return cbar


def plot_coverage(rxs: np.ndarray, cov_map: tuple[float, ...] | list[float] | np.ndarray,
                  dpi: int = 100, figsize: tuple = (6,4), cbar_title: str = '',
                  title: bool | str = False, scat_sz: float = 0.5,
                  bs_pos: Optional[np.ndarray] = None, bs_ori: Optional[np.ndarray] = None,
                  legend: bool = False, lims: Optional[Tuple[float, float]] = None,
                  proj_3D: bool = False, equal_aspect: bool = False, tight: bool = True,
                  cmap: str | list = 'viridis', cbar_labels: Optional[list[str]] = None,
                  ax: Optional[Axes] = None) -> Tuple[Figure, Axes, Colorbar]:
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
        proj_3D (bool): Whether to create 3D projection. Defaults to False.
        equal_aspect (bool): Whether to maintain equal axis scaling. Defaults to False.
        tight (bool): Whether to set tight axis limits around data points. Defaults to True.
        cmap (str | list): Matplotlib colormap name or list of colors. Defaults to 'viridis'.
        cbar_labels (Optional[list[str]]): List of labels for the colorbar. Defaults to None.
        ax (Optional[Axes]): Matplotlib Axes object. Defaults to None.

    Returns:
        Tuple containing:
        - matplotlib Figure object
        - matplotlib Axes object
        - matplotlib Colorbar object
    """
    cmap = cmap if isinstance(cmap, str) else ListedColormap(cmap)
    plt_params = {'cmap': cmap}
    if lims:
        plt_params['vmin'], plt_params['vmax'] = lims[0], lims[1]
    
    n = 3 if proj_3D else 2 # n = coordinates to consider
    
    xyz_arg_names = ['x' if n==2 else 'xs', 'y' if n==2 else 'ys', 'zs']
    xyz = {s: rxs[:,i] for s,i in zip(xyz_arg_names, range(n))}
    
    if not ax:
        fig, ax = plt.subplots(dpi=dpi, figsize=figsize,
                               subplot_kw={'projection': '3d'} if proj_3D else {})
    else:
        fig = ax.figure

    cov_map = np.array(cov_map) if isinstance(cov_map, list) else cov_map

    im = ax.scatter(**xyz, c=cov_map, s=scat_sz, marker='s', **plt_params)
    
    cbar = _create_colorbar(im, cov_map, cmap, cbar_title, cbar_labels, ax)
    
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    if proj_3D:
        ax.set_zlabel('z (m)')
        
    # TX position
    if bs_pos is not None:
        bs_pos = bs_pos.squeeze()
        ax.scatter(*bs_pos[:n], marker='P', c='r', label='TX')
    
    # TX orientation
    if bs_ori is not None and bs_pos is not None:
        r = 30 # ref size of pointing direction
        tx_lookat = np.copy(bs_pos)
        tx_lookat[:2] += r * np.array([np.cos(bs_ori[2]), np.sin(bs_ori[2])]) # azimuth
        tx_lookat[2] -= r / 10 * np.sin(bs_ori[1]) # elevation
        
        line_components = [[bs_pos[i], tx_lookat[i]] for i in range(n)]
        ax.plot(*line_components, c='k', alpha=.5, zorder=3)
        
    if title:
        ax.set_title(title)
    
    if legend:
        ax.legend(loc='upper center', ncols=10, framealpha=.5)
    
    if tight:
        s = 1
        # Get plot limits including BS position if available
        all_points = np.vstack([rxs, bs_pos.reshape(1, -1)]) if bs_pos is not None else rxs
        mins, maxs = np.min(all_points, axis=0)-s, np.max(all_points, axis=0)+s
        
        ax.set_xlim([mins[0], maxs[0]])
        ax.set_ylim([mins[1], maxs[1]])
        if proj_3D:
            ax.axes.set_zlim3d([mins[2], maxs[2]])
    
    if equal_aspect: # often disrups the plot if in 3D.
        ax.set_aspect('equal')
    
    return fig, ax, cbar

def transform_coordinates(coords, lon_max, lon_min, lat_min, lat_max):
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
    x_min, y_min = np.min(coords, axis=0)[:2]
    x_max, y_max = np.max(coords, axis=0)[:2]
    for (x, y) in zip(coords[:,0], coords[:,1]):
        lons += [lon_min + ((x - x_min) / (x_max - x_min)) * (lon_max - lon_min)]
        lats += [lat_min + ((y - y_min) / (y_max - y_min)) * (lat_max - lat_min)]
    return lats, lons

def export_xyz_csv(data: Dict[str, Any], z_var: np.ndarray, outfile: str = '',
                  google_earth: bool = False, lat_min: float = 33.418081,
                  lat_max: float = 33.420961, lon_min: float = -111.932875,
                  lon_max: float = -111.928567) -> None:
    """Export user locations and values to CSV format.
    
    This function generates a CSV file containing x,y,z coordinates that can be 
    imported into visualization tools like Blender or Google Earth. It supports
    both Cartesian and geographical coordinate formats.

    Args:
        data (Dict[str, Any]): DeepMIMO dataset for one basestation
        z_var (np.ndarray): Values to use for z-coordinate or coloring
        outfile (str): Output CSV file path. Defaults to ''.
        google_earth (bool): Whether to convert coordinates to geographical format. Defaults to False.
        lat_min (float): Minimum latitude for coordinate conversion. Defaults to 33.418081.
        lat_max (float): Maximum latitude for coordinate conversion. Defaults to 33.420961.
        lon_min (float): Minimum longitude for coordinate conversion. Defaults to -111.932875.
        lon_max (float): Maximum longitude for coordinate conversion. Defaults to -111.928567.

    Returns:
        None. Writes data to CSV file.
    """
    user_idxs = np.where(data['user']['LoS'] != -1)[0]
    
    locs = data['user']['location'][user_idxs]
    
    if google_earth:
        lats, lons = transform_coordinates(locs, 
                                           lon_min=lon_min, lon_max=lon_max, 
                                           lat_min=lat_min, lat_max=lat_max)
    else:
        lats, lons = locs[:,0], locs[:,1]
    
    # Transform xy to coords and create data dict
    data_dict = {'latitude'  if google_earth else 'x': lats if google_earth else locs[:,0], 
                 'longitude' if google_earth else 'y': lons if google_earth else locs[:,1], 
                 'z': z_var[user_idxs]}
    
    if not outfile:
        outfile = 'test.csv'
    
    # Equivalent in pandas (opted out to minimize dependencies.)
    # pd.DataFrame.from_dict(data_dict).to_csv(outfile, index=False)
    
    with open(outfile, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(data_dict.keys())
        # Write the data rows
        writer.writerows(zip(*data_dict.values()))


def plot_rays(rx_loc: np.ndarray, tx_loc: np.ndarray, inter_pos: np.ndarray,
              inter: np.ndarray, figsize: tuple = (10,8), dpi: int = 100,
              proj_3D: bool = True, color_by_type: bool = True,
              inter_objects: Optional[np.ndarray] = None,
              inter_obj_labels: Optional[list[str]] = None) -> Tuple[Figure, Axes]:
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
        proj_3D (bool, optional): Whether to create 3D projection. Defaults to True.
        color_by_type (bool, optional): Whether to color interaction points by their type. Defaults to False.
        inter_objects (Optional[np.ndarray], optional): Object ids at each interaction point. Defaults to None.
            If provided, will color the interaction points by the object id, and
            ignore the interaction type.
        inter_obj_labels (Optional[list[str]], optional): Labels for the interaction objects. Defaults to None.
            If provided, will use these labels instead of the object ids.
    Returns:
        Tuple containing:
        - matplotlib Figure object
        - matplotlib Axes object
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi,
                           subplot_kw={'projection': '3d'} if proj_3D else {})
    
    # Ensure inputs are numpy arrays and have correct shape
    rx_loc = np.asarray(rx_loc)  # Shape: (3,)
    tx_loc = np.asarray(tx_loc)  # Shape: (3,)
    inter_pos = np.asarray(inter_pos)  # Shape: (n_paths, max_interactions, 3)
    inter = np.asarray(inter)  # Shape: (n_paths,)
    
    # Get number of valid paths (non-NaN interaction codes)
    n_valid_paths = np.sum(~np.isnan(inter))
    
    # Define dimension-specific plotting functions
    def plot_line(start_point, end_point, **kwargs):
        coords = [(start_point[i], end_point[i]) for i in range(3 if proj_3D else 2)]
        ax.plot(*coords, **kwargs)
        
    def plot_point(point, **kwargs):
        coords = point[:3] if proj_3D else point[:2]
        ax.scatter(*coords, **kwargs)
    
    # Define colors and names for different interaction types
    interaction_colors = {
        0: 'green',    # Line-of-sight (direct path)
        1: 'red',      # Reflection
        2: 'orange',   # Diffraction
        3: 'blue',     # Scattering
        4: 'purple',   # Transmission
        -1: 'gray'     # Unknown/Invalid
    }
    
    interaction_names = {
        0: 'Line-of-sight',
        1: 'Reflection',
        2: 'Diffraction',
        3: 'Scattering',
        4: 'Transmission',
        -1: 'Unknown'
    }
    
    if inter_objects is not None:
        unique_objs = np.unique(inter_objects)
        inter_obj_colors = {obj_id: f'C{i}' for i, obj_id in enumerate(unique_objs)}
        if inter_obj_labels is None:
            inter_obj_labels = {obj_id: str(int(obj_id)) for obj_id in unique_objs}

    # For each ray path up to number of valid paths
    for path_idx in range(n_valid_paths):
        # Get valid interaction points for this path (excluding NaN values)
        # Check if any coordinate (x,y,z) is not NaN
        valid_inters = ~np.any(np.isnan(inter_pos[path_idx]), axis=1)
        path_interactions = inter_pos[path_idx][valid_inters]
        
        # Create complete path: TX -> interactions -> RX
        path_points = np.vstack([tx_loc, path_interactions, rx_loc])
        
        # Get interaction types for this path
        # Convert to integer first to remove decimal part
        path_type_int = int(inter[path_idx])
        if path_type_int == 0:
            path_types = []
        else:
            # Convert to string and get individual digits
            path_types = [int(d) for d in str(path_type_int)]
        
        # Check if this is a LoS path
        is_los = len(path_interactions) == 0
        plt_color = 'g' if is_los else 'r'
        # Plot the ray path segments
        for i in range(len(path_points)-1):
            plot_line(path_points[i], path_points[i+1], color=plt_color, alpha=0.5, zorder=1)
        
        # Plot interaction points
        if len(path_interactions) > 0:  # If there are interaction points
            for i, pos in enumerate(path_interactions):
                if color_by_type and i < len(path_types) and inter_objects is None:
                    # Get color based on interaction type at this position
                    point_color = interaction_colors.get(path_types[i], 'gray')
                    point_label = interaction_names.get(path_types[i], 'Unknown')
                elif inter_objects is not None:
                    point_color = inter_obj_colors.get(inter_objects[path_idx, i], 'gray')
                    point_label = inter_obj_labels.get(inter_objects[path_idx, i], 'unknown obj?')
                else:
                    print(f'Unclassified interaction point: path {path_idx}, inter {i}')
                    point_color = 'black'
                    point_label = None
                
                plot_point(pos, c=point_color, marker='o', s=20, label=point_label, zorder=2)
    
    # Plot TX and RX points
    plot_point(tx_loc, c='black', marker='^', s=100, label='TX', zorder=3)
    plot_point(rx_loc, c='black', marker='v', s=100, label='RX', zorder=3)
    
    # Set axis labels
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    if proj_3D:
        ax.set_zlabel('z (m)')

    # Only show legend if color_by_type is True or if there are TX/RX points
    if color_by_type or inter_objects is not None:
        # Remove duplicate labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        legend = ax.legend(by_label.values(), by_label.keys())
    else:
        legend = ax.legend()
    
    # Put legend outside the plot
    legend.set_bbox_to_anchor((1, 0.9))
    
    # Set equal aspect ratio for better visualization
    if not proj_3D:
        ax.set_aspect('equal')
    
    ax.grid()
    return fig, ax

def plot_power_discarding(dataset, trim_delay: Optional[float] = None) -> Tuple[Figure, Axes]:
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
    # Get the trim delay - either provided or from OFDM parameters
    if trim_delay is None:
        if not hasattr(dataset, 'channel_params'):
            raise ValueError("Dataset has no channel parameters. "
                             "Please provide trim_delay explicitly.")
        trim_delay = (dataset.channel_params.ofdm.subcarriers / 
                     dataset.channel_params.ofdm.bandwidth)
    
    # check if the maximum path delay is greater than the trim delay
    if np.nanmax(dataset.delay) < trim_delay:
        print(f"Maximum path delay: {np.nanmax(dataset.delay)*1e6:.1f} μs")
        print(f"Trim delay: {trim_delay*1e6:.1f} μs")
        print("No paths will be discarded.")
        return None, None

    # Calculate discarded power ratios for each user
    discarded_power_ratios = []
    n_users = len(dataset.delay)
    for user_idx in tqdm(range(n_users), desc="Calculating discarded power ratios per user"):
        user_delays = dataset.delay[user_idx]
        user_powers = dataset.power_linear[user_idx]
        
        # Find valid (non-NaN) paths
        valid_mask = ~np.isnan(user_delays) & ~np.isnan(user_powers)
        if not np.any(valid_mask):
            discarded_power_ratios.append(0)
            continue
            
        valid_delays = user_delays[valid_mask]
        valid_powers = user_powers[valid_mask]
        
        # Find paths exceeding trim delay
        discarded_mask = valid_delays > trim_delay
        if not np.any(discarded_mask):
            discarded_power_ratios.append(0)
            continue
        
        # Calculate power ratio
        total_power = np.sum(valid_powers)
        discarded_power = np.sum(valid_powers[discarded_mask])
        discarded_power_ratios.append((discarded_power / total_power) * 100)
    
    discarded_power_ratios = np.array(discarded_power_ratios)

    # Calculate statistics
    max_discard_idx = np.argmax(discarded_power_ratios)
    max_discard_ratio = discarded_power_ratios[max_discard_idx]
    mean_discard_ratio = np.mean(discarded_power_ratios)
    affected_users = np.sum(discarded_power_ratios > 0)
    non_zero_ratios = discarded_power_ratios[discarded_power_ratios > 0]

    # Print statistics
    print("\nPower Discarding Analysis")
    print("="*50)
    print(f"\nTrim delay: {trim_delay*1e6:.1f} μs")
    print(f"Maximum delay: {np.nanmax(dataset.delay)*1e6:.1f} μs\n")
    print(f"Maximum power discarded: {max_discard_ratio:.1f}%")
    print(f"Average power discarded: {mean_discard_ratio:.1f}%")
    print(f"Users with discarded paths: {affected_users}")

    fig, ax = plt.subplots(dpi=200, figsize=(6, 4))
    ax.hist(non_zero_ratios, bins=20)
    ax.set_title("Distribution of Discarded Power")
    ax.set_xlabel("Discarded Power (%)")
    ax.set_ylabel("Number of Users")
    ax.grid(True)
    plt.tight_layout()
    return fig, ax







