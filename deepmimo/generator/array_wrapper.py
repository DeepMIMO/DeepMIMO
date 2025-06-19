"""
Array wrapper module for DeepMIMO.

This module provides a custom array class that wraps numpy arrays and adds plotting functionality.
"""

import numpy as np
from typing import TYPE_CHECKING, Optional, Any
from .visualization import plot_coverage

Dataset = 'Dataset' if TYPE_CHECKING else Any

class DeepMIMOArray(np.ndarray):
    """A wrapper around numpy.ndarray that adds plotting functionality.
    
    This class is used to wrap arrays in the DeepMIMO dataset that have num_rx in the first dimension.
    It adds a plot() method that uses plot_coverage to visualize the data.
    
    The plot() method handles different array shapes:
    - 1D arrays [num_rx]: Plots directly
    - 2D arrays [num_rx, num_paths]: Plots specified path index
    - 3D arrays [num_rx, num_paths, max_interactions]: Plots specified path and interaction indices
    """
    
    def __new__(cls, input_array: np.ndarray, dataset: Dataset) -> 'DeepMIMOArray':
        """Create a new DeepMIMOArray instance.
        
        This is called when creating new arrays. We need __new__ instead of __init__
        because we're inheriting from np.ndarray.
        
        Args:
            input_array: The numpy array to wrap
            dataset: The DeepMIMO dataset this array belongs to
            
        Returns:
            A new DeepMIMOArray instance
        """
        # Cast input_array to ndarray if it isn't already
        obj = np.asarray(input_array).view(cls)
        
        # Add the dataset reference
        obj.dataset = dataset
        
        return obj
    
    def __array_finalize__(self, obj: Optional[np.ndarray]) -> None:
        """Handle array creation through means other than __new__.
        
        This is called whenever a new array is created from this one, including
        through slicing, copying, etc. We need to make sure the dataset reference
        is preserved.
        
        Args:
            obj: The array being used to create this one
        """
        if obj is None: return
        
        # Preserve dataset reference through operations like slicing
        self.dataset = getattr(obj, 'dataset', None)
    
    def plot(self, path_idx: int = 0, interaction_idx: int = 0, **kwargs) -> None:
        """Plot the array using plot_coverage.
        
        Args:
            path_idx: Index of the path to plot for 2D/3D arrays. Ignored for 1D arrays.
            interaction_idx: Index of the interaction to plot for 3D arrays. Ignored for 1D/2D arrays.
            **kwargs: Additional arguments to pass to plot_coverage
        
        Raises:
            ValueError: If array has unsupported number of dimensions. Only 1D [num_rx],
                      2D [num_rx, num_paths], and 3D [num_rx, num_paths, max_interactions]
                      arrays are supported.
        """
        # Handle different array shapes
        if self.ndim == 1:
            # 1D array [num_rx] - plot directly
            data = self
        elif self.ndim == 2:
            # 2D array [num_rx, num_paths] - plot specified path
            data = self[:, path_idx]
        elif self.ndim == 3:
            # 3D array [num_rx, num_paths, max_interactions] - plot specified path and interaction
            data = self[:, path_idx, interaction_idx]
        else:
            raise ValueError(
                f"Cannot plot array with shape {self.shape}. "
                "Only the following shapes are supported:\n"
                "- 1D arrays [num_rx]\n"
                "- 2D arrays [num_rx, num_paths]\n"
                "- 3D arrays [num_rx, num_paths, max_interactions]"
            )
            
        # Use dataset's plot_coverage method directly
        self.dataset.plot_coverage(data, **kwargs) 