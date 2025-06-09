import numpy as np
from typing import Dict, Any

from .utils.geo_utils import convert_Gps2RelativeCartesian, convert_GpsBBox2CartesianBBox

def gen_tx_pos(rt_params: Dict[str, Any]) -> np.ndarray:
    """Generate transmitter positions from GPS coordinates.
    
    Args:
        rt_params (Dict[str, Any]): Ray tracing parameters
        Required Parameters:
            - bs_lats (List[float]): Latitude coordinates of base stations
            - bs_lons (List[float]): Longitude coordinates of base stations
            - bs_heights (List[float]): Height coordinates of base stations
            - origin_lat, origin_lon (float): Origin GPS coordinates
        
    Returns:
        List[List[float]]: Transmitter positions in Cartesian coordinates
    """
    num_bs = len(rt_params['bs_lats'])
    print(f"Number of BSs: {num_bs}")
    bs_pos = []
    for bs_idx in range(num_bs):
        bs_cartesian = convert_Gps2RelativeCartesian(rt_params['bs_lats'][bs_idx], 
                                                     rt_params['bs_lons'][bs_idx],
                                                     rt_params['origin_lat'],
                                                     rt_params['origin_lon'])
        bs_pos.append([bs_cartesian[0], bs_cartesian[1], rt_params['bs_heights'][bs_idx]])
    return np.array(bs_pos)


def gen_plane_grid(min_coord1: float, max_coord1: float, 
                   min_coord2: float, max_coord2: float,
                   spacing: float, fixed_coord: float,
                   normal: str = 'z') -> np.ndarray:
    """Generate a grid of points on a plane perpendicular to a major axis.
    
    Args:
        min_coord1 (float): Minimum value for first planar coordinate
        max_coord1 (float): Maximum value for first planar coordinate
        min_coord2 (float): Minimum value for second planar coordinate
        max_coord2 (float): Maximum value for second planar coordinate
        spacing (float): Grid spacing in meters
        fixed_coord (float): Value for the coordinate along the normal axis
        normal (str): Normal axis of the plane ('x', 'y', or 'z'). Defaults to 'z'
        
    Returns:
        np.ndarray: Grid points with shape (N, 3) where N is the number of points
    """
    # Create the 2D grid
    coord1 = np.arange(min_coord1, max_coord1 + spacing, spacing)
    coord2 = np.arange(min_coord2, max_coord2 + spacing, spacing)
    grid_coord1, grid_coord2 = np.meshgrid(coord1, coord2)
    fixed = np.zeros_like(grid_coord1) + fixed_coord
    
    # Stack coordinates based on which axis is normal
    if normal.lower() == 'x':
        return np.stack([fixed.flatten(), 
                        grid_coord1.flatten(), 
                        grid_coord2.flatten()], axis=-1)
    elif normal.lower() == 'y':
        return np.stack([grid_coord1.flatten(), 
                        fixed.flatten(), 
                        grid_coord2.flatten()], axis=-1)
    else:  # 'z' is default
        return np.stack([grid_coord1.flatten(), 
                        grid_coord2.flatten(), 
                        fixed.flatten()], axis=-1)


def gen_rx_grid(rt_params: Dict[str, Any]) -> np.ndarray:
    """Generate user grid in Cartesian coordinates.
    
    Args:
        rt_params (Dict[str, Any]): Ray tracing parameters
            Required Parameters:
                - min_lat, min_lon, max_lat, max_lon (float): GPS coordinates of the area
                - origin_lat, origin_lon (float): Origin GPS coordinates
                - grid_spacing (float): Grid spacing in meters
                - ue_height (float): UE height in meters
        
    Returns:
        np.ndarray: User grid positions in Cartesian coordinates
    """
    xmin, ymin, xmax, ymax = convert_GpsBBox2CartesianBBox(
        rt_params['min_lat'], rt_params['min_lon'], 
        rt_params['max_lat'], rt_params['max_lon'], 
        rt_params['origin_lat'], rt_params['origin_lon'])
    
    user_grid = gen_plane_grid(xmin, xmax, ymin, ymax,
                               rt_params['grid_spacing'], 
                               rt_params['ue_height'])
    
    print(f"User grid shape: {user_grid.shape}")
    
    return user_grid
