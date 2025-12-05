"""
MacroDataset module for DeepMIMO.

This module provides the MacroDataset class for managing collections of related 
DeepMIMO datasets that may share:
- Scene configuration
- Material properties
- Loading parameters 
- Ray-tracing parameters
"""

# Standard library imports
import inspect
from typing import List

# Local imports
from .dataset import Dataset, SHARED_PARAMS


class MacroDataset:
    """A container class that holds multiple Dataset instances and propagates operations to all children.
    
    This class acts as a simple wrapper around a list of Dataset objects. When any attribute
    or method is accessed on the MacroDataset, it automatically propagates that operation
    to all contained Dataset instances. If the MacroDataset contains only one dataset,
    it will return single value instead of a list with a single element.
    """
    
    # Methods that should only be called on the first dataset
    SINGLE_ACCESS_METHODS = [
        'info',  # Parameter info should only be shown once
    ]
    
    # Methods that should be propagated to children - automatically populated from Dataset methods
    PROPAGATE_METHODS = {
        name for name, _ in inspect.getmembers(Dataset, predicate=inspect.isfunction)
        if not name.startswith('__')  # Skip dunder methods
    }
    
    def __init__(self, datasets: list[Dataset] | None = None):
        """Initialize with optional list of Dataset instances.
        
        Args:
            datasets: List of Dataset instances. If None, creates empty list.
        """
        self.datasets = datasets if datasets is not None else []
        
    def _get_single(self, key):
        """Get a single value from the first dataset for shared parameters.
        
        Args:
            key: Key to get value for
            
        Returns:
            Single value from first dataset if key is in SHARED_PARAMS,
            otherwise returns list of values from all datasets
        """
        if not self.datasets:
            raise IndexError("MacroDataset is empty")
        return self.datasets[0][key]
        
    def __getattr__(self, name):
        """Propagate any attribute/method access to all datasets.
        
        If the attribute is a method in PROPAGATE_METHODS, call it on all children.
        If the attribute is in SHARED_PARAMS, return from first dataset.
        If there is only one dataset, return single value instead of lists.
        Otherwise, return list of results from all datasets.
        """
        # Check if it's a method we should propagate
        if name in self.PROPAGATE_METHODS:
            if name in self.SINGLE_ACCESS_METHODS:
                # For single access methods, only call on first dataset
                def single_method(*args, **kwargs):
                    return getattr(self.datasets[0], name)(*args, **kwargs)
                return single_method
            else:
                # For normal methods, propagate to all datasets
                def propagated_method(*args, **kwargs):
                    results = [getattr(dataset, name)(*args, **kwargs) for dataset in self.datasets]
                    return results[0] if len(results) == 1 else results
                return propagated_method
            
        # Handle shared parameters
        if name in SHARED_PARAMS:
            return self._get_single(name)
            
        # Default: propagate to all datasets
        results = [getattr(dataset, name) for dataset in self.datasets]
        return results[0] if len(results) == 1 else results
        
    def __getitem__(self, idx):
        """Get dataset at specified index if idx is integer, otherwise propagate to all datasets.
        
        Args:
            idx: Integer index to get specific dataset, or string key to get attribute from all datasets
            
        Returns:
            Dataset instance if idx is integer,
            single value if idx is in SHARED_PARAMS or if there is only one dataset,
            or list of results if idx is string and there are multiple datasets
        """
        if isinstance(idx, (int, slice)):
            return self.datasets[idx]
        if idx in SHARED_PARAMS:
            return self._get_single(idx)
        results = [dataset[idx] for dataset in self.datasets]
        return results[0] if len(results) == 1 else results
        
    def __setitem__(self, key, value):
        """Set item on all contained datasets.
        
        Args:
            key: Key to set
            value: Value to set
        """
        for dataset in self.datasets:
            dataset[key] = value
        
    def __len__(self):
        """Return number of contained datasets."""
        return len(self.datasets)
        
    def append(self, dataset):
        """Add a dataset to the collection.
        
        Args:
            dataset: Dataset instance to add
        """
        self.datasets.append(dataset)

