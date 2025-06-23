"""
AODT Utilities Module.

This module provides utility functions used across the AODT converter modules.
"""

import numpy as np
from typing import Dict, List
from ... import consts as c

def dict_to_array(point_dict: Dict[str, float]) -> np.ndarray:
    """Convert a dictionary of coordinates to a numpy array.
    
    Args:
        point_dict: Dictionary containing coordinates with keys '1', '2', '3'
        
    Returns:
        np.ndarray: Array of shape (3,) containing [x, y, z] coordinates
    """
    return np.array([point_dict['1'], point_dict['2'], point_dict['3']], dtype=c.FP_TYPE)

def process_points(points_list: List[Dict[str, float]]) -> np.ndarray:
    """Convert a list of point dictionaries to a numpy array.
    
    Args:
        points_list: List of dictionaries, each containing coordinates with keys '1', '2', '3'
        
    Returns:
        np.ndarray: Array of shape (N, 3) containing N points
    """
    return np.array([dict_to_array(point) for point in points_list], dtype=c.FP_TYPE) 