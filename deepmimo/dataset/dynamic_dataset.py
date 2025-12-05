"""
DynamicDataset module for DeepMIMO.

This module provides the DynamicDataset class for dynamic datasets that consist of 
multiple (macro)datasets across time snapshots:
- All txrx sets are the same for all time snapshots
"""

# Third-party imports
import numpy as np

# Local imports
from ..general_utils import DelegatingList
from ..txrx import get_txrx_sets
from .macro_dataset import MacroDataset


class DynamicDataset(MacroDataset):
    """A dataset that contains multiple (macro)datasets, each representing a different time snapshot."""
    
    def __init__(self, datasets: list[MacroDataset], name: str):
        """Initialize a dynamic dataset.
        
        Args:
            datasets: List of MacroDataset instances, each representing a time snapshot
            name: Base name of the scenario (without time suffix)
        """
        super().__init__(datasets)
        self.name = name
        self.names = [dataset.name for dataset in datasets]
        self.n_scenes = len(datasets)

        for dataset in datasets:
            dataset.parent_name = name
            
    def _get_single(self, key):
        """Override _get_single to handle scene differently from other shared parameters.
        
        For scene, return a DelegatingList of scenes from all datasets.
        For other shared parameters, use parent class behavior.
        """
        if key == 'scene':
            return DelegatingList([dataset.scene for dataset in self.datasets])
        return super()._get_single(key)
        
    def __getattr__(self, name):
        """Override __getattr__ to handle txrx_sets specially."""
        if name == 'txrx_sets':
            return get_txrx_sets(self.name)
        return super().__getattr__(name)
    
    def set_timestamps(self, timestamps: int | float | list[int | float] | np.ndarray) -> None:
        """Set the timestamps for the dataset.

        Args:
            timestamps(int | float | list[int | float] | np.ndarray): 
                Timestamps for each scene in the dataset. Can be:
                - Single value: Creates evenly spaced timestamps
                - List/array: Custom timestamps for each scene
        """
        self.timestamps = np.zeros(self.n_scenes)
        
        if isinstance(timestamps, (float, int)):
            self.timestamps = np.arange(0, timestamps * self.n_scenes, timestamps)
        elif isinstance(timestamps, list):
            self.timestamps = np.array(timestamps)
        
        if len(self.timestamps) != self.n_scenes:
            raise ValueError(f'Time reference must be a single value or a list of {self.n_scenes} values')
        
        if self.timestamps.ndim != 1:
            raise ValueError(f'Time reference must be single dimension.')

        self._compute_speeds()
    
    def _compute_speeds(self) -> None:
        """Compute the speeds of each scene based on the position and time differences.""" 
        # Compute position & time differences to compute speeds for each scene
        for i in range(1, self.n_scenes - 1):
            time_diff = (self.timestamps[i] - self.timestamps[i - 1])
            dataset_curr = self.datasets[i]
            dataset_prev = self.datasets[i - 1]
            rx_pos_diff = dataset_curr.rx_pos - dataset_prev.rx_pos
            tx_pos_diff = dataset_curr.tx_pos - dataset_prev.tx_pos
            obj_pos_diff = (np.vstack(dataset_curr.scene.objects.position) -
                            np.vstack(dataset_prev.scene.objects.position))
            dataset_curr.rx_vel = rx_pos_diff / time_diff
            dataset_curr.tx_vel = tx_pos_diff[0] / time_diff
            dataset_curr.scene.objects.vel = [v for v in obj_pos_diff / time_diff]

            # For the first and last pair of scenes, assume that the position and time differences 
            # are the same as for the second and second-from-last pair of scenes, respectively.
            if i == 1:
                i2 = 0
            elif i == self.n_scenes - 2:
                i2 = self.n_scenes - 1
            else:
                i2 = None

            if i2 is not None:
                dataset_2 = self.datasets[i2]
                dataset_2.rx_vel = dataset_curr.rx_vel
                dataset_2.tx_vel = dataset_curr.tx_vel
                dataset_2.scene.objects.vel = dataset_curr.scene.objects.vel
        
        return

