"""
Dataset module for DeepMIMO.

This module provides two main classes:

Dataset: For managing individual DeepMIMO datasets, including:
- Channel matrices 
- Path information (angles, powers, delays)
- Position information
- TX/RX configuration information
- Metadata

MacroDataset: For managing collections of related DeepMIMO datasets that *may* share:
- Scene configuration
- Material properties
- Loading parameters 
- Ray-tracing parameters

DynamicDataset: For dynamic datasets that consist of multiple (macro)datasets across time snapshots:
- All txrx sets are the same for all time snapshots

The Dataset class is organized into several logical sections:
1. Core Dictionary Interface - Basic dictionary-like operations and key resolution
2. Channel Computations - Channel matrices and array responses
3. Geometric Computations - Angles, rotations, and positions
4. Field of View Operations - FoV filtering and caching
5. Path and Power Computations - Path characteristics and power calculations
6. Grid and Sampling Operations - Grid info and dataset subsetting
7. Visualization - Plotting and display methods
8. Utilities and Configuration - Helper methods and class configuration
"""

# Standard library imports
import inspect
from typing import Dict, Optional, Any, List

# Third-party imports
import numpy as np
from tqdm import tqdm

# Base utilities
from ..general_utils import DotDict, spherical_to_cartesian, DelegatingList
from .. import consts as c
from ..info import info
from .visualization import plot_coverage, plot_rays
from .array_wrapper import DeepMIMOArray

# Channel generation
from .channel import _generate_MIMO_channel, ChannelParameters

# Antenna patterns and geometry
from .ant_patterns import AntennaPattern
from .geometry import (
    _rotate_angles_batch,
    _apply_FoV_batch,
    _array_response_batch,
    _ant_indices
)

# Utilities
from .generator_utils import (
    dbw2watt,
    get_uniform_idxs,
    get_grid_idxs,
    get_linear_idxs
)

# Converter utilities
from ..converters import converter_utils as cu

# Txrx set information
from ..txrx import get_txrx_sets, TxRxSet

# Summary
from ..summary import plot_summary

# Parameters that should remain consistent across datasets in a MacroDataset
SHARED_PARAMS = [
    c.SCENE_PARAM_NAME,           # Scene object
    c.MATERIALS_PARAM_NAME,       # MaterialList object
    c.LOAD_PARAMS_PARAM_NAME,     # Loading parameters
    c.RT_PARAMS_PARAM_NAME,       # Ray-tracing parameters
]


class Dataset(DotDict):
    """Class for managing DeepMIMO datasets.
    
    This class provides an interface for accessing dataset attributes including:
    - Channel matrices
    - Path information (angles, powers, delays)
    - Position information
    - TX/RX configuration information
    - Metadata
    
    Attributes can be accessed using both dot notation (dataset.channel) 
    and dictionary notation (dataset['channel']).
    
    Primary (Static) Attributes:
        power: Path powers in dBW
        phase: Path phases in degrees
        delay: Path delays in seconds (i.e. propagation time)
        aoa_az/aoa_el: Angles of arrival (azimuth/elevation)
        aod_az/aod_el: Angles of departure (azimuth/elevation)
        rx_pos: Receiver positions
        tx_pos: Transmitter position
        inter: Path interaction indicators
        inter_pos: Path interaction positions
        
    Secondary (Computed) Attributes:
        power_linear: Path powers in linear scale
        channel: MIMO channel matrices
        num_paths: Number of paths per user
        pathloss: Path loss in dB
        distances: Distances between TX and RXs
        los: Line of sight status for each receiver
        pwr_ant_gain: Powers with antenna patterns applied
        aoa_az_rot/aoa_el_rot: Rotated angles of arrival based on antenna orientation
        aod_az_rot/aod_el_rot: Rotated angles of departure based on antenna orientation
        aoa_az_rot_fov/aoa_el_rot_fov: Field of view filtered angles of arrival
        aod_az_rot_fov/aod_el_rot_fov: Field of view filtered angles of departure
        fov_mask: Field of view mask
        
    TX/RX Information:
        - tx_set_id: ID of the transmitter set
        - rx_set_id: ID of the receiver set
        - tx_idx: Index of the transmitter within its set
        - rx_idxs: List of receiver indices used
        
    Common Aliases:
        ch, pwr, rx_loc, pl, dist, n_paths, etc.
        (See aliases dictionary for complete mapping)
    """
    
    def __init__(self, data: Optional[Dict[str, Any]] = None):
        """Initialize dataset with optional data.
        
        Args:
            data: Initial dataset dictionary. If None, creates empty dataset.
        """
        super().__init__(data or {})
    
    ###########################################
    # 1. Core Interface
    ###########################################
    
    # List of arrays that should be wrapped with DeepMIMOArray
    WRAPPABLE_ARRAYS = [
        'power', 'phase', 'delay', 'aoa_az', 'aoa_el', 'aod_az', 'aod_el',
        'inter', 'los', 'channel', 'power_linear', 'pathloss', 'distance',
        'num_paths', 'inter_str', 'doppler', 'inter_obj', 'inter_int'
    ]
    
    def _wrap_array(self, key: str, value: Any) -> Any:
        """Wrap numpy arrays with DeepMIMOArray if appropriate.
        
        Args:
            key: The key/name of the array
            value: The array value to potentially wrap
            
        Returns:
            The original value or a wrapped DeepMIMOArray
        """
        if isinstance(value, np.ndarray) and key in self.WRAPPABLE_ARRAYS:
            if value.ndim == 0:
                return value
            # Only wrap arrays that have num_rx in first dimension
            if value.shape[0] == self.n_ue:
                return DeepMIMOArray(value, self, key)
        return value

    def __getitem__(self, key: str) -> Any:
        """Get an item from the dataset, computing it if necessary and wrapping if appropriate."""
        try:
            value = super().__getitem__(key)
        except KeyError:
            value, key = self._resolve_key(key)
        return self._wrap_array(key, value)
            
    def __getattr__(self, key: str) -> Any:
        """Enable dot notation access with array wrapping."""
        try:
            value = super().__getitem__(key)
        except KeyError:
            value, key = self._resolve_key(key)
        return self._wrap_array(key, value)

    def _resolve_key(self, key: str) -> Any:
        """Resolve a key through the lookup chain.
        
        Order of operations:
        1. Check if key is an alias and resolve it first
        2. Try direct access with resolved key
        3. Try computing the attribute if it's computable
        
        Args:
            key: The key to resolve
            
        Returns:
            The resolved value, and the key that was resolved
            
        Raises:
            KeyError if key cannot be resolved
        """
        # First check if it's an alias and resolve it
        resolved_key = c.DATASET_ALIASES.get(key, key)
        if resolved_key != key:
            key = resolved_key
            try:
                return super().__getitem__(key), key
            except KeyError:
                pass
            
        if key in self._computed_attributes:
            compute_method_name = self._computed_attributes[key]
            compute_method = getattr(self, compute_method_name)
            value = compute_method()
            # Cache the result, and return just the key, not the dict
            if isinstance(value, dict):
                self.update(value)
                return super().__getitem__(key), key
            else:
                self[key] = value
                return value, key
        
        raise KeyError(key)
    
    def __dir__(self):
        """Return list of valid attributes including computed ones."""
        # Include standard attributes, computed attributes, and aliases
        return list(set(
            list(super().__dir__()) + 
            list(self._computed_attributes.keys()) + 
            list(c.DATASET_ALIASES.keys())
        ))

    ###########################################
    # 2. Channel Computations
    ###########################################

    def set_channel_params(self, params: Optional[ChannelParameters] = None) -> None:
        """Set channel generation parameters.
        
        Args:
            params: Channel generation parameters. If None, uses default parameters.
        """
        if params is None:
            params = ChannelParameters()
            
        params.validate(self.n_ue)
        
        # Create a deep copy of the parameters to ensure isolation
        old_params = (super().__getitem__(c.CH_PARAMS_PARAM_NAME) 
                      if c.CH_PARAMS_PARAM_NAME in super().keys() else None)
        self.ch_params = params.deepcopy()
        
        # If rotation has changed, clear rotated angles cache
        if old_params is not None:
            old_bs_rot = old_params.bs_antenna[c.PARAMSET_ANT_ROTATION]
            old_ue_rot = old_params.ue_antenna[c.PARAMSET_ANT_ROTATION]
            new_bs_rot = params.bs_antenna[c.PARAMSET_ANT_ROTATION]
            new_ue_rot = params.ue_antenna[c.PARAMSET_ANT_ROTATION]
            if not np.array_equal(old_bs_rot, new_bs_rot) or not np.array_equal(old_ue_rot, new_ue_rot):
                self._clear_cache_rotated_angles()
        
        return params
    
    def compute_channels(self, params: Optional[ChannelParameters] = None, **kwargs) -> np.ndarray:
        """Compute MIMO channel matrices for all users.
        
        This is the main public method for computing channel matrices. It handles all the
        necessary preprocessing steps including:
        - Antenna pattern application
        - Field of view filtering
        - Array response computation
        - OFDM processing (if enabled)
        
        The computed channel will be cached and accessible as dataset.channel
        or dataset['channel'] after this call.
        
        Args:
            params: Channel generation parameters. If None, uses default parameters.
                    See ChannelParameters class for details.
            **kwargs: Additional keyword arguments to pass to ChannelParameters constructor
                    if params is None. Ignored if params is provided. 
                    If provided, overrides existing channel parameters (e.g. set_channel_params).
            
        Returns:
            numpy.ndarray: MIMO channel matrix with shape [n_users, n_rx_ant, n_tx_ant, n_subcarriers]
                          if freq_domain=True, otherwise [n_users, n_rx_ant, n_tx_ant, n_paths]
        """
        if params is None:
            if kwargs:
                params = ChannelParameters(**kwargs)
            else:
                params = self.ch_params if self.ch_params is not None else ChannelParameters()

        self.set_channel_params(params)

        # Compute array response product
        array_response_product = self._compute_array_response_product()
        
        n_paths_to_gen = params.num_paths
        
        # Whether to enable the doppler shift per path in the channel
        n_paths = np.min((n_paths_to_gen, self.delay.shape[-1]))
        default_doppler = np.zeros((self.n_ue, n_paths))
        use_doppler = self.hasattr('doppler')

        if params[c.PARAMSET_DOPPLER_EN] and not use_doppler:
            all_obj_vel = np.array([obj.vel for obj in self.scene.objects])
            # Enable doppler if any velocity component is non-zero
            use_doppler = self.tx_vel.any() or self.rx_vel.any() or all_obj_vel.any()
            if not use_doppler:
                print("No doppler in channel generation because all velocities are zero")

        dopplers = self.doppler[..., :n_paths] if use_doppler else default_doppler

        channel = _generate_MIMO_channel(
            array_response_product=array_response_product[..., :n_paths],
            powers=self._power_linear_ant_gain[..., :n_paths],
            delays=self.delay[..., :n_paths],
            phases=self.phase[..., :n_paths],
            dopplers=dopplers,
            ofdm_params=params.ofdm,
            freq_domain=params.freq_domain,
        )

        self[c.CHANNEL_PARAM_NAME] = channel  # Cache the result

        return channel
    
    ###########################################
    # 3. Geometric Computations
    ###########################################

    @property
    def tx_ori(self) -> np.ndarray:
        """Compute the orientation of the transmitter.
        
        Returns:
            Array of transmitter orientation
        """
        return self.ch_params['bs_antenna']['rotation']*np.pi/180
    
    @property
    def bs_ori(self) -> np.ndarray:
        """Alias for tx_ori - computes the orientation of the transmitter/basestation.
        
        Returns:
            Array of transmitter orientation
        """
        return self.tx_ori
    
    @property
    def rx_ori(self) -> np.ndarray:
        """Compute the orientation of the receivers.
        
        Returns:
            Array of receiver orientation
        """
        return self.ch_params['ue_antenna']['rotation']*np.pi/180

    @property
    def ue_ori(self) -> np.ndarray:
        """Alias for rx_ori - computes the orientation of the receivers/users.
        
        Returns:
            Array of receiver orientation
        """
        return self.rx_ori

    def _look_at(self, from_pos: np.ndarray, to_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Internal helper function to calculate azimuth and elevation angles for position pairs.
        
        Args:
            from_pos: Array of starting positions with shape (n, 2-3) in meters
            to_pos: Array of target positions with shape (n, 2-3) in meters
            
        Returns:
            Tuple of (azimuth_degrees, elevation_degrees) arrays with shape (n,)
        """
        # Ensure positions are arrays and handle 2D coordinates
        from_pos = np.atleast_2d(from_pos)
        to_pos = np.atleast_2d(to_pos)
        
        # Add z=0 if only 2D coordinates
        if from_pos.shape[1] == 2:
            from_pos = np.column_stack([from_pos, np.zeros(from_pos.shape[0])])
        if to_pos.shape[1] == 2:
            to_pos = np.column_stack([to_pos, np.zeros(to_pos.shape[0])])
        
        # Calculate direction vectors for all pairs at once
        direction_vectors = to_pos - from_pos
        dx, dy, dz = direction_vectors[:, 0], direction_vectors[:, 1], direction_vectors[:, 2]
        
        # Calculate azimuth (horizontal angle) for all pairs
        azimuth_rad = np.arctan2(dy, dx)
        
        # Calculate elevation (vertical angle) for all pairs
        horizontal_distance = np.sqrt(dx**2 + dy**2)
        elevation_rad = np.arctan2(dz, horizontal_distance)
        
        # Convert from radians to degrees for DeepMIMO convention
        azimuth_deg = azimuth_rad * 180.0 / np.pi
        elevation_deg = elevation_rad * 180.0 / np.pi
        
        return azimuth_deg, elevation_deg

    def bs_look_at(self, look_pos: np.ndarray | list | tuple) -> None:
        """Set the orientation of the basestation to look at a given position in 3D.

        Similar to Sionna RT's Camera.look_at() function, this method automatically
        calculates and sets the antenna rotation parameters so that the basestation
        points toward the specified target position.

        Args:
            look_pos: The position to look at (x, y, z) in meters.
                     Can be a numpy array, list, or tuple.
                     If 2D coordinates are provided, z=0 is assumed.

        Example:
            >>> # Point BS toward a specific UE
            >>> dataset.bs_look_at(dataset.rx_pos[0])
            >>> 
            >>> # Point BS toward coordinates
            >>> dataset.bs_look_at([100, 200, 10])
        """
        # Use helper function to calculate angles
        azimuth_deg, elevation_deg = self._look_at(self.tx_pos, look_pos)
        azimuth_deg, elevation_deg = azimuth_deg[0], elevation_deg[0]

        # Update the basestation antenna rotation parameters (preserve existing z_rot)
        current_rotation = np.array(self.ch_params.bs_antenna[c.PARAMSET_ANT_ROTATION])
        z_rot = current_rotation.flat[2] if current_rotation.size > 2 else 0
        self.ch_params.bs_antenna[c.PARAMSET_ANT_ROTATION] = np.array([azimuth_deg, elevation_deg, z_rot])
        
        # Clear cached rotated angles since rotation has changed
        self._clear_cache_rotated_angles()

    def ue_look_at(self, look_pos: np.ndarray | list | tuple) -> None:
        """Set the orientation of user equipment antennas to look at given position(s) in 3D.
        
        Similar to bs_look_at() function, this method automatically calculates and sets 
        the UE antenna rotation parameters so that user equipment point toward the 
        specified target position(s).
        
        Args:
            look_pos: The position(s) to look at in meters.
                     Can be:
                     - 1D array/list/tuple (x, y, z): All UEs look at the same position
                     - 2D array with shape (3,) or (2,): Same as 1D case
                     - 2D array with shape (n_users, 3): Each UE looks at different position
                     - 2D array with shape (n_users, 2): Each UE looks at different position (z=0)
                     If 2D coordinates are provided, z=0 is assumed.
        
        Example:
            >>> # All UEs look at the base station
            >>> dataset.ue_look_at(dataset.tx_pos)
            >>> 
            >>> # All UEs look at a specific coordinate
            >>> dataset.ue_look_at([100, 200, 10])
            >>> 
            >>> # Each UE looks at different positions (must match number of UEs)
            >>> look_positions = np.array([[100, 200, 10], [150, 250, 15], ...])
            >>> dataset.ue_look_at(look_positions)
        """
        # Convert look_pos to numpy array
        look_pos = np.array(look_pos)
        
        # Get user positions
        if not hasattr(self, 'rx_pos') or self.rx_pos is None:
            print("Warning: No user positions found. "
                  "Ensure positions are loaded and available in dataset.rx_pos.")
            return
        
        # Handle different input formats
        if look_pos.ndim == 1:
            # 1D array: same target for all UEs
            target_positions = np.tile(look_pos, (self.n_ue, 1))
        elif look_pos.ndim == 2:
            if look_pos.shape[0] == 1:
                # Single position for all UEs
                target_positions = np.tile(look_pos, (self.n_ue, 1))
            else:
                # Different position for each UE
                if look_pos.shape[0] != self.n_ue:
                    raise ValueError(f"Number of target positions ({look_pos.shape[0]}) must match number of users ({self.n_ue})")
                target_positions = look_pos
        else:
            raise ValueError("look_pos must be 1D or 2D array")
        
        # Calculate rotation parameters for all UEs at once
        azimuth_degrees, elevation_degrees = self._look_at(self.rx_pos, target_positions)
        
        # Update the UE antenna rotation parameters (preserve existing z_rot)
        curr_rot = np.atleast_2d(self.ch_params.ue_antenna[c.PARAMSET_ANT_ROTATION])
        z_rot_values = curr_rot[:, 2] if curr_rot.shape == (self.n_ue, 3) else np.zeros(self.n_ue)
        self.ch_params.ue_antenna[c.PARAMSET_ANT_ROTATION] = \
            np.column_stack([azimuth_degrees, elevation_degrees, z_rot_values])
        
        # Clear cached rotated angles since rotation has changed
        self._clear_cache_rotated_angles()

    def _compute_rotated_angles(self, tx_ant_params: Optional[Dict[str, Any]] = None, 
                                rx_ant_params: Optional[Dict[str, Any]] = None) -> Dict[str, np.ndarray]:
        """Compute rotated angles for all users in batch.
        
        Args:
            tx_ant_params: Dictionary containing transmitter antenna parameters. If None, uses stored params.
            rx_ant_params: Dictionary containing receiver antenna parameters. If None, uses stored params.
            
        Returns:
            Dictionary containing the rotated angles for all users
        """
        # Use stored channel parameters if none provided
        if tx_ant_params is None:
            tx_ant_params = self.ch_params.bs_antenna
        if rx_ant_params is None:
            rx_ant_params = self.ch_params.ue_antenna
            
        # Transform BS antenna rotation if needed
        bs_rotation = tx_ant_params[c.PARAMSET_ANT_ROTATION]
        if len(bs_rotation.shape) == 2 and bs_rotation.shape[0] == 3 and bs_rotation.shape[1] == 2:
            # Generate random rotations for BS
            bs_rotation = np.random.uniform(
                bs_rotation[:, 0],
                bs_rotation[:, 1],
                (3,)
            )
            self.ch_params.bs_antenna[c.PARAMSET_ANT_ROTATION] = bs_rotation
            
        # Transform UE antenna rotation if needed
        ue_rotation = rx_ant_params[c.PARAMSET_ANT_ROTATION]
        if len(ue_rotation.shape) == 1 and ue_rotation.shape[0] == 3:
            # Convert single 3D vector to array for all users
            ue_rotation = np.tile(ue_rotation, (self.n_ue, 1))
        elif len(ue_rotation.shape) == 2 and ue_rotation.shape[0] == 3 and ue_rotation.shape[1] == 2:
            # Generate random rotations for each user
            ue_rotation = np.random.uniform(
                ue_rotation[:, 0],
                ue_rotation[:, 1],
                (self.n_ue, 3)
            )
            self.ch_params.ue_antenna[c.PARAMSET_ANT_ROTATION] = ue_rotation

        # Rotate angles for all users at once
        aod_theta_rot, aod_phi_rot = _rotate_angles_batch(
            rotation=bs_rotation,
            theta=self[c.AOD_EL_PARAM_NAME],
            phi=self[c.AOD_AZ_PARAM_NAME])
        
        aoa_theta_rot, aoa_phi_rot = _rotate_angles_batch(
            rotation=ue_rotation,
            theta=self[c.AOA_EL_PARAM_NAME],
            phi=self[c.AOA_AZ_PARAM_NAME])
        
        return {
            c.AOD_EL_ROT_PARAM_NAME: aod_theta_rot,
            c.AOD_AZ_ROT_PARAM_NAME: aod_phi_rot,
            c.AOA_EL_ROT_PARAM_NAME: aoa_theta_rot,
            c.AOA_AZ_ROT_PARAM_NAME: aoa_phi_rot
        }

    def _clear_cache_rotated_angles(self) -> None:
        """Clear all cached attributes that depend on rotated angles.
        
        This includes:
        - Rotated angles
        - Field of view filtered angles (since they depend on rotated angles)
        - Line of sight status
        - Channel matrices
        - Powers with antenna gain
        """
        # Define rotated angles dependent keys
        rotated_angles_keys = {
            c.AOD_EL_ROT_PARAM_NAME, c.AOD_AZ_ROT_PARAM_NAME,
            c.AOA_EL_ROT_PARAM_NAME, c.AOA_AZ_ROT_PARAM_NAME
        }
        # Remove all rotated angles dependent keys at once
        for k in rotated_angles_keys & self.keys():
            super().__delitem__(k)
        
        # Also clear FOV cache since it depends on rotated angles
        self._clear_cache_fov()

    def _compute_single_array_response(self, ant_params: Dict, theta: np.ndarray, 
                                       phi: np.ndarray) -> np.ndarray:
        """Internal method to compute array response for a single antenna array.
        
        Args:
            ant_params: Antenna parameters dictionary
            theta: Elevation angles array
            phi: Azimuth angles array
            
        Returns:
            Array response matrix
        """
        # Use attribute access for antenna parameters
        kd = 2 * np.pi * ant_params.spacing
        ant_ind = _ant_indices(ant_params[c.PARAMSET_ANT_SHAPE])  # tuple complications..
        
        return _array_response_batch(ant_ind=ant_ind, theta=theta, phi=phi, kd=kd)

    def _compute_array_response_product(self) -> np.ndarray:
        """Internal method to compute product of TX and RX array responses.
        
        Returns:
            Array response product matrix
        """
        # Get antenna parameters from channel parameters
        tx_ant_params = self.ch_params.bs_antenna
        rx_ant_params = self.ch_params.ue_antenna
        
        # Compute individual responses
        array_response_TX = self._compute_single_array_response(
            tx_ant_params, self[c.AOD_EL_FOV_PARAM_NAME], self[c.AOD_AZ_FOV_PARAM_NAME])
            
        array_response_RX = self._compute_single_array_response(
            rx_ant_params, self[c.AOA_EL_FOV_PARAM_NAME], self[c.AOA_AZ_FOV_PARAM_NAME])
        
        # Compute product with proper broadcasting
        # [n_users, M_rx, M_tx, n_paths]
        return array_response_RX[:, :, None, :] * array_response_TX[:, None, :, :]

    ###########################################
    # 4. Field of View Operations
    ###########################################

    def apply_fov(self, bs_fov: np.ndarray = np.array([360, 180]), 
                  ue_fov: np.ndarray = np.array([360, 180])) -> None:
        """Apply field of view (FoV) filtering to the dataset.
        
        This method sets the FoV parameters and invalidates any cached FoV-dependent attributes.
        The actual filtering will be performed lazily when FoV-dependent attributes are accessed.
        
        Args:
            bs_fov: Base station FoV as [horizontal, vertical] in degrees. Defaults to [360, 180] (full sphere).
            ue_fov: User equipment FoV as [horizontal, vertical] in degrees. Defaults to [360, 180] (full sphere).
            
        Note:
            This operation affects all path-related attributes and cached computations.
            The following will be recomputed as needed when accessed:
            - FoV filtered angles
            - Number of valid paths
            - Line of sight status
            - Channel matrices
            - Powers with antenna gain
        """
        # Clear cached FoV-dependent attributes
        self._clear_cache_fov()
            
        # Store FoV parameters
        self.bs_fov = bs_fov
        self.ue_fov = ue_fov
    
    def _is_full_fov(self, fov: np.ndarray) -> bool:
        """Check if a FoV parameter represents a full sphere view.
        
        Args:
            fov: FoV parameter as [horizontal, vertical] in degrees
            
        Returns:
            bool: True if FoV represents a full sphere view
        """
        return fov[0] >= 360 and fov[1] >= 180

    def _is_fov_enabled(self) -> bool:
        """Get the current FoV status including parameters and whether they are full sphere.
        
        Returns:
            bool: True if FoV filtering is enabled
        """
        bs_fov = self.get('bs_fov')
        ue_fov = self.get('ue_fov')
        bs_full = bs_fov is not None and self._is_full_fov(bs_fov)
        ue_full = ue_fov is not None and self._is_full_fov(ue_fov)
        has_fov = (bs_fov is not None and not bs_full) or (ue_fov is not None and not ue_full)
        
        return has_fov

    def _compute_fov(self) -> Dict[str, np.ndarray]:
        """Compute field of view filtered angles for all users.
        
        This function applies field of view constraints to the rotated angles
        and stores both the filtered angles and the mask in the dataset.
        If no FoV parameters are set, assumes full FoV and returns unfiltered angles.
        
        Returns:
            Dict: Dictionary containing FoV filtered angles and mask
        """
        # Get rotated angles from dataset
        aod_theta = self[c.AOD_EL_ROT_PARAM_NAME]  # [n_users, n_paths]
        aod_phi = self[c.AOD_AZ_ROT_PARAM_NAME]    # [n_users, n_paths]
        aoa_theta = self[c.AOA_EL_ROT_PARAM_NAME]  # [n_users, n_paths]
        aoa_phi = self[c.AOA_AZ_ROT_PARAM_NAME]    # [n_users, n_paths]
        
        # Get FoV parameters and check if they are full sphere
        bs_fov = self.get('bs_fov')
        ue_fov = self.get('ue_fov')
        bs_full = bs_fov is not None and self._is_full_fov(bs_fov)
        ue_full = ue_fov is not None and self._is_full_fov(ue_fov)
        
        # If no FoV params or both are full sphere, return unfiltered angles
        if (bs_fov is None and ue_fov is None) or (bs_full and ue_full):
            return {
                c.FOV_MASK_PARAM_NAME: None,
                c.AOD_EL_FOV_PARAM_NAME: aod_theta,
                c.AOD_AZ_FOV_PARAM_NAME: aod_phi,
                c.AOA_EL_FOV_PARAM_NAME: aoa_theta,
                c.AOA_AZ_FOV_PARAM_NAME: aoa_phi
            }
        
        # Initialize mask as all True
        fov_mask = np.ones_like(aod_theta, dtype=bool)
        
        # Only apply BS FoV filtering if restricted
        if not bs_full:
            tx_mask = _apply_FoV_batch(bs_fov, aod_theta, aod_phi)
            fov_mask = np.logical_and(fov_mask, tx_mask)
            
        # Only apply UE FoV filtering if restricted
        if not ue_full:
            rx_mask = _apply_FoV_batch(ue_fov, aoa_theta, aoa_phi)
            fov_mask = np.logical_and(fov_mask, rx_mask)
        
        return {
            c.FOV_MASK_PARAM_NAME: fov_mask,
            c.AOD_EL_FOV_PARAM_NAME: np.where(fov_mask, aod_theta, np.nan),
            c.AOD_AZ_FOV_PARAM_NAME: np.where(fov_mask, aod_phi, np.nan),
            c.AOA_EL_FOV_PARAM_NAME: np.where(fov_mask, aoa_theta, np.nan),
            c.AOA_AZ_FOV_PARAM_NAME: np.where(fov_mask, aoa_phi, np.nan)
        }

    def _clear_cache_fov(self) -> None:
        """Clear all cached attributes that depend on field of view (FoV) filtering.
        
        This includes:
        - FoV filtered angles
        - FoV mask
        - Number of valid paths
        - Line of sight status
        - Channel matrices
        - Powers with antenna gain
        """
        # Define FOV-dependent keys
        fov_dependent_keys = {
            c.FOV_MASK_PARAM_NAME, c.NUM_PATHS_PARAM_NAME, c.LOS_PARAM_NAME,
            c.CHANNEL_PARAM_NAME,  c.PWR_LINEAR_ANT_GAIN_PARAM_NAME,
            c.AOD_EL_FOV_PARAM_NAME, c.AOD_AZ_FOV_PARAM_NAME,
            c.AOA_EL_FOV_PARAM_NAME, c.AOA_AZ_FOV_PARAM_NAME
        }
        # Remove all FOV-dependent keys at once
        for k in fov_dependent_keys & self.keys():
            super().__delitem__(k)

    ###########################################
    # 5. Path and Power Computations
    ###########################################

    def compute_pathloss(self, coherent: bool = True) -> np.ndarray:
        """Compute path loss in dB, assuming 0 dBm transmitted power.
        
        Args:
            coherent (bool): Whether to use coherent sum. Defaults to True
        
        Returns:
            numpy.ndarray: Path loss in dB
        """
        # Convert powers to linear scale
        powers_linear = 10 ** (self.power / 10)  # mW
        phases_rad = np.deg2rad(self.phase)
        
        # Sum complex path gains
        complex_gains = np.sqrt(powers_linear).astype(np.complex64)
        if coherent:
            complex_gains *= np.exp(1j * phases_rad)
        total_power = np.abs(np.nansum(complex_gains, axis=1))**2
        
        # Convert back to dB
        mask = total_power > 0
        pathloss = np.full_like(total_power, np.nan)
        pathloss[mask] = -10 * np.log10(total_power[mask])
        
        self[c.PATHLOSS_PARAM_NAME] = pathloss  # Cache the result
        return pathloss


    def _compute_los(self) -> np.ndarray:
        """Calculate Line of Sight status (1: LoS, 0: NLoS, -1: No paths) for each receiver.

        Uses the interaction codes defined in consts.py:
            INTERACTION_LOS = 0: Line-of-sight (direct path)
            INTERACTION_REFLECTION = 1: Reflection
            INTERACTION_DIFFRACTION = 2: Diffraction
            INTERACTION_SCATTERING = 3: Scattering
            INTERACTION_TRANSMISSION = 4: Transmission

        Returns:
            numpy.ndarray: LoS status array, shape (n_users,)
        """
        los_status = np.full(self.inter.shape[0], -1)
        
        # Get FoV mask if FoV filtering is enabled
        if self._is_fov_enabled():
            _ = self[c.AOD_AZ_ROT_PARAM_NAME]
            fov_mask = self[c.FOV_MASK_PARAM_NAME]
        else:
            fov_mask = None

        if fov_mask is not None:
            # If we have FoV filtering, only consider paths within FoV
            has_paths = np.any(fov_mask, axis=1)
            # For each user, find the first valid path within FoV
            first_valid_path = np.full(self.inter.shape[0], -1)
            for i in range(self.inter.shape[0]):
                valid_paths = np.where(fov_mask[i])[0]
                if len(valid_paths) > 0:
                    first_valid_path[i] = self.inter[i, valid_paths[0]]
        else:
            # No FoV filtering, use all paths
            has_paths = self.num_paths > 0
            first_valid_path = self.inter[:, 0]
        
        # Set NLoS status for users with paths
        los_status[has_paths] = 0
        
        # Set LoS status for users with direct path as first valid path
        los_mask = first_valid_path == c.INTERACTION_LOS
        los_status[los_mask & has_paths] = 1
        
        return los_status

    def _compute_num_paths(self) -> np.ndarray:
        """Compute number of valid paths for each user after FoV filtering."""

        if self._is_fov_enabled():
            aoa = self[c.AOA_AZ_FOV_PARAM_NAME]
        else:
            aoa = self[c.AOA_AZ_PARAM_NAME]
        
        # Count non-NaN values (NaN indicates filtered out, possibly by FoV)
        return (~np.isnan(aoa)).sum(axis=1)

    def _compute_max_paths(self) -> int:
        """Compute the maximum number of paths for any user."""
        return int(np.nanmax(self.num_paths))
    
    def _compute_max_interactions(self) -> int:
        """Compute the maximum number of interactions for any path of any user."""
        return np.nanmax(self.num_interactions).astype(int)

    def _compute_num_interactions(self) -> np.ndarray:
        """Compute number of interactions for each path of each user."""
        result = np.zeros_like(self.inter)
        result[np.isnan(self.inter)] = np.nan # no interaction
        non_zero = self.inter > 0
        result[non_zero] = np.floor(np.log10(self.inter[non_zero])) + 1
        return result

    def _compute_inter_int(self) -> np.ndarray:
        """Compute the interaction integer, with NaN values replaced by -1.
        
        Returns:
            Array of interaction integer with NaN values replaced by -1
        """
        inter_int = self.inter.copy()
        inter_int[np.isnan(inter_int)] = -1
        return inter_int.astype(int)

    def _compute_inter_str(self) -> np.ndarray:
        """Compute the interaction string.
        
        Returns:
            Array of interaction string
        """
        
        inter_raw_str = self.inter.astype(str)  # Shape: (n_users, n_paths)
        INTER_MAP = str.maketrans({'0': '', '1': 'R', '2': 'D', '3': 'S', '4': 'T'})

        # Vectorize the translation across all paths
        def translate_code(s):
            # 'nan', '221.0', '134.0', ... -> 'n', 'RRD', 'DST', ...
            return s[:-2].translate(INTER_MAP) if s != 'nan' else 'n'
        
        # Apply translation to each element in the 2D array
        return np.vectorize(translate_code)(inter_raw_str)

    def _compute_n_ue(self) -> int:
        """Return the number of UEs/receivers in the dataset."""
        return self.rx_pos.shape[0]

    def _compute_distances(self) -> np.ndarray:
        """Compute Euclidean distances between receivers and transmitter."""
        return np.linalg.norm(self.rx_pos - self.tx_pos, axis=1)

    def _compute_power_linear_ant_gain(self, tx_ant_params: Optional[Dict[str, Any]] = None,
                                       rx_ant_params: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Compute received power with antenna patterns applied.
        
        Args:
            tx_ant_params (Optional[Dict[str, Any]]): Transmitter antenna parameters. If None, uses stored params.
            rx_ant_params (Optional[Dict[str, Any]]): Receiver antenna parameters. If None, uses stored params.
            
        Returns:
            np.ndarray: Powers with antenna pattern applied, shape [n_users, n_paths]
        """
        # Use stored channel parameters if none provided
        if tx_ant_params is None:
            tx_ant_params = self.ch_params[c.PARAMSET_ANT_BS]
        if rx_ant_params is None:
            rx_ant_params = self.ch_params[c.PARAMSET_ANT_UE]
            
        # Create antenna pattern object
        antennapattern = AntennaPattern(tx_pattern=tx_ant_params[c.PARAMSET_ANT_RAD_PAT],
                                        rx_pattern=rx_ant_params[c.PARAMSET_ANT_RAD_PAT])
        
        # Get FoV filtered angles and apply antenna patterns in batch
        return antennapattern.apply_batch(power=self[c.PWR_LINEAR_PARAM_NAME],
                                          aoa_theta=self[c.AOA_EL_FOV_PARAM_NAME],
                                          aoa_phi=self[c.AOA_AZ_FOV_PARAM_NAME], 
                                          aod_theta=self[c.AOD_EL_FOV_PARAM_NAME],
                                          aod_phi=self[c.AOD_AZ_FOV_PARAM_NAME])


    def _compute_power_linear(self) -> np.ndarray:
        """Internal method to compute linear power from power in dBm"""
        return dbw2watt(self.power) 

    ###########################################
    # 6. Grid and User Sampling Operations
    ###########################################

    def _compute_grid_info(self) -> Dict[str, np.ndarray]:
        """Internal method to compute grid size and spacing information from receiver positions.
        
        Returns:
            Dict containing:
                grid_size: Array with [x_size, y_size] - number of points in each dimension
                grid_spacing: Array with [x_spacing, y_spacing] - spacing between points in meters
        """
        x_positions = np.unique(self.rx_pos[:, 0])
        y_positions = np.unique(self.rx_pos[:, 1])
        
        grid_size = np.array([len(x_positions), len(y_positions)])
        grid_spacing = np.array([
            np.mean(np.diff(x_positions)) if len(x_positions) > 1 else 0,
            np.mean(np.diff(y_positions)) if len(y_positions) > 1 else 0
        ])
        
        return {
            'grid_size': grid_size,
            'grid_spacing': grid_spacing
        }

    def has_valid_grid(self) -> bool:
        """Check if the dataset has a valid grid structure.
        
        A valid grid means that:
        1. The total number of points in the grid matches the number of receivers
        2. The receivers are arranged in a regular grid pattern
        
        Returns:
            bool: True if dataset has valid grid structure, False otherwise
        """
        # Check if total grid points match number of receivers
        grid_points = np.prod(self.grid_size)
        
        return grid_points == self.n_ue

    def get_active_idxs(self) -> np.ndarray:
        """Return indices of active users.
        
        Returns:
            Array of indices of active users
        """
        return np.where(self.num_paths > 0)[0]

    def get_linear_idxs(self, start_pos: np.ndarray, end_pos: np.ndarray, n_steps: int, filter_repeated: bool = True) -> np.ndarray:
        """Return indices of users along a linear path between two points.
        
        Args:
            start_pos: Starting position coordinates (2D or 3D) [x, y] or [x, y, z]
            end_pos: Ending position coordinates (2D or 3D) [x, y] or [x, y, z]
            n_steps: Number of steps along the path 
            filter_repeated: Whether to filter repeated positions (default: True)
        
        Returns:
            Array of indices of users along the linear path
        """ 
        return get_linear_idxs(self.rx_pos, start_pos, end_pos, n_steps, filter_repeated)

    def get_uniform_idxs(self, steps: List[int]) -> np.ndarray:
        """Return indices of users at uniform intervals.
        
        Args:
            steps: List of sampling steps for each dimension [x_step, y_step]
            
        Returns:
            Array of indices for uniformly sampled users
            
        Raises:
            ValueError: If dataset does not have a valid grid structure
        """
        return get_uniform_idxs(self.n_ue, self.grid_size, steps)
    
    def get_row_idxs(self, row_idxs: int | list[int] | np.ndarray) -> np.ndarray:
        """Return indices of users in the specified rows, assuming a grid structure.
        
        Args:
            row_idxs: Array of row indices to include in the new dataset

        Returns:
            Array of indices of receivers in the specified rows
        """
        return get_grid_idxs(self.grid_size, 'row', row_idxs)
        
    def get_col_idxs(self, col_idxs: int | list[int] | np.ndarray) -> np.ndarray:
        """Return indices of users in the specified columns, assuming a grid structure.
        
        Args:
            col_idxs: Array of column indices to include in the new dataset

        Returns:
            Array of indices of receivers in the specified columns
        """
        return get_grid_idxs(self.grid_size, 'col', col_idxs)

    
    ###########################################
    # 7. Subsetting and Trimming
    ###########################################

    def subset(self, idxs: np.ndarray) -> 'Dataset':
        """Create a new dataset containing only the selected indices.
        
        Args:
            idxs: Array of indices to include in the new dataset
            
        Returns:
            Dataset: A new dataset containing only the selected indices
        """
        # Create a new dataset with initial data
        initial_data = {}
        
        # Copy shared parameters that should remain consistent across datasets
        for param in SHARED_PARAMS:
            if self.hasattr(param):
                initial_data[param] = getattr(self, param)
            
        # Directly set n_ue
        initial_data['n_ue'] = len(idxs)
        
        # Create new dataset with initial data
        new_dataset = Dataset(initial_data)
        
        # Copy all attributes
        for attr, value in self.to_dict().items():
            # skip private and already handled attributes
            if not attr.startswith('_') and attr not in SHARED_PARAMS + ['n_ue']:
                if isinstance(value, np.ndarray) and len(value.shape) == 0:
                    print(f'{attr} is a scalar')
                if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == self.n_ue:
                    # Copy and index arrays with UE dimension
                    setattr(new_dataset, attr, value[idxs])
                else:
                    # Copy other attributes as is
                    setattr(new_dataset, attr, value)
                
        return new_dataset

    def _clear_all_caches(self) -> None:
        """Clear all caches."""
        self._clear_cache_core()
        self._clear_cache_rotated_angles()
        self._clear_cache_fov()
        self._clear_cache_doppler()

    def _clear_cache_core(self) -> None:
        """Clear all cached attributes that don't have dedicated clearing functions.
        
        This includes:
        - Line of sight status
        - Number of paths
        - Number of interactions
        - Channel matrices
        - Powers with antenna gain
        - Inter-object related attributes
        - Other computed attributes
        """
        # Define cache keys for attributes without dedicated clearing functions
        cache_keys = {
            # Core computed attributes
            c.NUM_PATHS_PARAM_NAME,
            c.LOS_PARAM_NAME,
            c.NUM_INTERACTIONS_PARAM_NAME,
            c.MAX_INTERACTIONS_PARAM_NAME,
            c.INTER_STR_PARAM_NAME,
            c.INTER_INT_PARAM_NAME,
            c.CHANNEL_PARAM_NAME,
            c.PWR_LINEAR_ANT_GAIN_PARAM_NAME,
            c.INTER_OBJECTS_PARAM_NAME
        }
        
        # Remove all cache keys at once
        for k in cache_keys & self.keys():
            super().__delitem__(k)

    def _trim_by_path(self, path_mask: np.ndarray) -> 'Dataset':
        """Helper function to trim paths based on a boolean mask.
        
        Args:
            path_mask: Boolean array of shape [n_users, n_paths] indicating which paths to keep.
            
        Returns:
            A new Dataset with trimmed paths.
        """
        # Create a new dataset with the same structure
        aux_dataset = self.deepcopy()
        
        # List of fundamental arrays that need to be trimmed
        path_arrays = [
            c.POWER_PARAM_NAME,
            c.PHASE_PARAM_NAME,
            c.DELAY_PARAM_NAME,
            c.AOA_AZ_PARAM_NAME,
            c.AOA_EL_PARAM_NAME,
            c.AOD_AZ_PARAM_NAME,
            c.AOD_EL_PARAM_NAME,
            c.INTERACTIONS_PARAM_NAME,
            c.INTERACTIONS_POS_PARAM_NAME,
        ]
        
        # Set invalid paths to NaN
        for array_name in path_arrays:
            aux_dataset[array_name][~path_mask] = np.nan
        
        # Create new order for each user: valid paths first, then invalid paths
        new_order = np.argsort(~path_mask, axis=1)  # False (valid) comes before True (invalid)
        
        # Reorder all path arrays
        for array_name in path_arrays:
            if array_name == c.INTERACTIONS_POS_PARAM_NAME:
                # Handle 4D array (n_users, n_paths, n_interactions, 3)
                aux_dataset[array_name] = np.take_along_axis(aux_dataset[array_name], new_order[:, :, None, None], axis=1)
            else:
                # Handle 2D arrays (n_users, n_paths)
                aux_dataset[array_name] = np.take_along_axis(aux_dataset[array_name], new_order, axis=1)
        
        # Compress arrays to remove unused paths
        data_dict = {k: v for k, v in aux_dataset.items() if isinstance(v, np.ndarray) and k in path_arrays}
        compressed_data = cu.compress_path_data(data_dict)
        
        # Update dataset with compressed arrays
        for key, value in compressed_data.items():
            aux_dataset[key] = value
        
        # Clear all caches since we modified fundamental data
        aux_dataset._clear_all_caches()
        
        return aux_dataset

    def _trim_by_index(self, idxs: np.ndarray) -> 'Dataset':
        """Rename previous subset function.
        
        Args:
            idxs: The indices to trim the dataset by.
        """
        return self.subset(idxs)
    
    def _trim_by_fov(self, fov: float) -> 'Dataset':
        """Trim the dataset by FoV.
        
        This function performs the same as applying the FoV filter to the dataset. 
        It is possible to apply it based on a BS and UE rotation, either already
        defined in the channel parameters or based on input arguments.

        BENEFIT: makes all the FoV logic unnecessary. 

        NOTE: before making this function, decide what to do with the rotated angles.
        Should we always have a .apply_fov(ue_fov, bs_fov) or .apply_rot(ue_rot, bs_rot) 
        methods that apply in-place or return new datasets?
        ANS: Yes. aoa_az should always be with respect to the BS/UE orientation.

        Args:
            fov: The FoV to trim the dataset by.
        """
        return 0

    def trim_by_path_depth(self, path_depth: int) -> 'Dataset':
        """Trim the dataset to keep only paths with at most the specified number of interactions.
        
        Args:
            path_depth: Maximum number of interactions allowed in a path.
            
        Returns:
            A new Dataset with paths trimmed to the specified depth.
        """
        # Create mask for paths with valid depth
        path_mask = np.zeros_like(self.inter, dtype=bool)
        
        # Get number of interactions for each path
        n_interactions = self._compute_num_interactions()
        
        # Keep paths with valid number of interactions
        path_mask = n_interactions <= path_depth
        
        return self._trim_by_path(path_mask)

    def trim_by_path_type(self, allowed_types: List[str]) -> 'Dataset':
        """Trim the dataset to keep only paths with allowed interaction types.
        
        Args:
            allowed_types: List of allowed interaction types. Can be any combination of:
                'LoS': Line of sight
                'R': Reflection
                'D': Diffraction
                'S': Scattering
                'T': Transmission
                
        Returns:
            A new Dataset with paths trimmed to only include allowed interaction types.
        """
        # Map string types to interaction codes
        type_to_code = {
            'LoS': c.INTERACTION_LOS,
            'R': c.INTERACTION_REFLECTION,
            'D': c.INTERACTION_DIFFRACTION,
            'S': c.INTERACTION_SCATTERING,
            'T': c.INTERACTION_TRANSMISSION,
        }
        
        # Convert allowed types to codes
        allowed_codes = [type_to_code[t] for t in allowed_types]
        
        # Create mask for paths with allowed interaction types
        path_mask = np.zeros_like(self.inter, dtype=bool)
        
        # For each path, check if all its interactions are allowed
        for user_idx in range(self.n_ue):
            for path_idx in range(self.inter.shape[1]):
                # Skip if no interaction
                if np.isnan(self.inter[user_idx, path_idx]):
                    continue
                    
                # Get interaction code as string and check each digit
                inter_str = str(int(self.inter[user_idx, path_idx]))
                is_valid = all(int(digit) in allowed_codes for digit in inter_str)
                path_mask[user_idx, path_idx] = is_valid
        
        return self._trim_by_path(path_mask)

    ###########################################
    # 8. Visualization
    ###########################################

    def plot_coverage(self, cov_map, **kwargs):
        """Plot the coverage of the dataset.
        
        Args:
            cov_map: The coverage map to plot.
            **kwargs: Additional keyword arguments to pass to the plot_coverage function.
        """
        return plot_coverage(self.rx_pos, cov_map, bs_pos=self.tx_pos.T, bs_ori=self.tx_ori, **kwargs)
    
    def plot_rays(self, idx: int, color_strat: str = 'none', **kwargs):
        """Plot the rays of the dataset.
        
        Args:
            idx: Index of the user to plot rays for
            color_strat: Strategy for coloring rays by power. Can be:
                - 'none': Don't color by power (default)
                - 'relative': Color by power relative to min/max of this user's paths
                - 'absolute': Color by power using absolute limits from all users
            **kwargs: Additional keyword arguments to pass to the plot_rays function.
        """
        if kwargs.get('color_by_inter_obj', False):
            inter_objs = self.inter_objects[idx]
            inter_obj_labels = {obj_id: obj.name for obj_id, obj in enumerate(self.scene.objects)}
        else:
            inter_objs = None
            inter_obj_labels = None
        kwargs.pop('color_by_inter_obj', None)

        default_kwargs = {
            'proj_3D': True,
            'color_by_type': True,
            'inter_objects': inter_objs,
            'inter_obj_labels': inter_obj_labels,
        }
        
        # Handle power-based coloring
        if color_strat != 'none':
            default_kwargs['color_rays_by_pwr'] = True
            default_kwargs['powers'] = self.power[idx]
            
            if color_strat == 'absolute':
                default_kwargs['limits'] = (np.nanmin(self.power), np.nanmax(self.power))
            
            if 'show_cbar' not in kwargs.keys():
                kwargs['show_cbar'] = True

        default_kwargs.update(kwargs)
        return plot_rays(self.rx_pos[idx], self.tx_pos[0], self.inter_pos[idx],
                         self.inter[idx], **default_kwargs)
    
    def plot_summary(self, **kwargs):
        """Plot the summary of the dataset."""
        return plot_summary(dataset=self, **kwargs)
    
    ###########################################
    # 9. Doppler Computations
    ###########################################
    
    @property
    def rx_vel(self) -> np.ndarray:
        """Get the velocities of the users.

        Returns:
            np.ndarray: The velocities of the users in cartesian coordinates. [n_ue, 3] [m/s]
        """
        # check if this exists, and initialize to zeros if not
        if not self.hasattr('_rx_vel'):
            self._rx_vel = np.zeros((self.n_ue, 3))
        return self._rx_vel

    @rx_vel.setter
    def rx_vel(self, velocities: np.ndarray | list | tuple) -> None:
        """Set the velocities of the users.
        
        Args:
            velocities: The velocities of the users in cartesian coordinates. [m/s]
        
        Returns:
            The velocities of the users in spherical coordinates.
        """
        self._clear_cache_doppler()
        
        if type(velocities) == list or type(velocities) == tuple:
            velocities = np.array(velocities)
            
        if velocities.ndim == 1:
            # [3,] -> [n_ue, 3]
            self._rx_vel = np.repeat(velocities[None, :], self.n_ue, axis=0)
        else:
            if velocities.shape[1] != 3:
                raise ValueError('Velocities must be in cartesian coordinates (n_ue, 3)')
            if velocities.shape[0] != self.n_ue:
                raise ValueError('Number of users must match number of velocities (n_ue, 3)')
            
            self._rx_vel = velocities
            
        return

    def print_rx(self, idx: int, path_idxs: np.ndarray | list[int] | None = None) -> None:
        """Print detailed information about a specific user.
        
        Args:
            idx: Index of the user to print information for
            path_idxs: Optional array of path indices to print. If None, prints all paths.
            
        Raises:
            IndexError: If idx is out of range or if any path index is out of range
        """
        if idx < 0 or idx >= self.n_ue:
            raise IndexError(f"User index {idx} is out of range [0, {self.n_ue})")
        
        # If path_idxs is None, use all valid paths
        if path_idxs is None:
            path_idxs = np.arange(self.num_paths[idx])
        else:
            path_idxs = np.array(path_idxs)
            # Validate path indices
            if np.any((path_idxs < 0) | (path_idxs >= self.num_paths[idx])):
                raise IndexError(f"Path indices must be in range [0, {self.num_paths[idx]})")

        print("\nUser Information:")
        print(f"Position: {self.rx_pos[idx]}")
        print(f"Velocity: {self.rx_vel[idx]}")

        print("\nPath Information:")
        print(f"Number of paths selected: {len(path_idxs)} (total: {self.num_paths[idx]})")
        print(f"Powers (dBm): {self.power[idx][path_idxs]}")
        print(f"Phases (deg): {self.phase[idx][path_idxs]}")
        print(f"Delays (us): {self.delay[idx][path_idxs]*1e6}")

        print("\nAngles:")
        print(f"Azimuth of Departure (deg): {self.aod_phi[idx][path_idxs]}")
        print(f"Elevation of Departure (deg): {self.aod_theta[idx][path_idxs]}")
        print(f"Azimuth of Arrival (deg): {self.aoa_phi[idx][path_idxs]}")
        print(f"Elevation of Arrival (deg): {self.aoa_theta[idx][path_idxs]}")

        print("\nInteraction Information:")
        print(f"Interaction types: {self.inter[idx][path_idxs]}")
        print(f"Number of interactions: {self.num_interactions[idx][path_idxs]}")
        print("Interaction positions:")
        for p_idx, path_idx in enumerate(path_idxs):
            n_inter = int(self.num_interactions[idx][path_idx])
            if np.isnan(n_inter):
                print(f"  Path {path_idx}: No interactions")
                continue
            print(f"  Path {path_idx} ({n_inter} interactions):")
            for i in range(n_inter):
                print(f"    {p_idx+1}: {self.inter_pos[idx][path_idx][p_idx]}")

        if self.hasattr('inter_objects'):
            print("\nInteraction objects:")
            for p_idx, path_idx in enumerate(path_idxs):
                n_inter = int(self.num_interactions[idx][path_idx])
                if np.isnan(n_inter):
                    print(f"  Path {path_idx}: No interactions")
                    continue
                print(f"  Path {path_idx} ({n_inter} interactions):")
                for i in range(n_inter):
                    print(f"    {p_idx+1}: {self.inter_objects[idx][path_idx][p_idx]}")

    @property
    def tx_vel(self) -> np.ndarray:
        """Get the velocities of the base stations.

        Returns:
            np.ndarray: The velocities of the base stations in cartesian coordinates. [3,] [m/s]
        """
        if not self.hasattr('_tx_vel'):
            self._tx_vel = np.zeros(3)
        return self._tx_vel

    @tx_vel.setter
    def tx_vel(self, velocities: np.ndarray | list | tuple) -> np.ndarray:
        """Set the velocities of the base stations.
        
        Args:
            velocities: The velocities of the base stations in cartesian coordinates. [m/s]
        
        Returns:
            The velocities of the base stations in cartesian coordinates. [3,] [m/s]
        """
        self._clear_cache_doppler()

        if type(velocities) == list or type(velocities) == tuple:
            velocities = np.array(velocities)

        if velocities.ndim != 1:
            raise ValueError('Tx velocity must be in a single cartesian coordinate (3,)')
        
        self._tx_vel = velocities
        return
    
    def set_doppler(self, doppler: float | list[float] | np.ndarray) -> None:
        """Set the doppler frequency shifts.
        
        Args:
            doppler: The doppler frequency shifts. [n_ue, max_paths] [Hz]
                There are 3 options for the shape of the doppler array:
                1. 1 value for all paths and users. [1,] [Hz]
                2. a value for each user. [n_ue,] [Hz]
                3. a value for each user and each path. [n_ue, max_paths] [Hz]
        """
        doppler = np.array([doppler]) if type(doppler) in [float, int] else np.array(doppler)

        if doppler.ndim == 1 and doppler.shape[0] == 1:
            doppler = np.ones((self.n_ue, self.max_paths)) * doppler[0]
        elif doppler.ndim == 1 and doppler.shape[0] == self.n_ue:
            doppler = np.repeat(doppler[None, :], self.max_paths, axis=1).reshape((self.n_ue, self.max_paths))
        elif doppler.ndim == 2 and doppler.shape[0] == self.n_ue and doppler.shape[1] == self.max_paths:
            pass
        else:
            raise ValueError(f'Invalid doppler shape: {doppler.shape}')
        
        self.doppler = doppler
        return

    def set_obj_vel(self, obj_idx: int | list[int], 
                    vel: list[float] | list[list[float]] | np.ndarray) -> None:
        """Update the velocity of an object.
        
        Args:
            obj_idx: The index of the object to update.
            vel: The velocity of the object in 3D cartesian coordinates. [m/s]

        Returns:
            None
        """
        if type(vel) == list or type(vel) == tuple:
            vel = np.array(vel)
        if vel.ndim == 1:
            vel = np.repeat(vel[None, :], len(obj_idx), axis=0)
        if vel.shape[0] != len(obj_idx):
            raise ValueError('Number of velocities must match number of objects')
        
        if type(obj_idx) == int:
            obj_idx = [obj_idx]
        for idx, obj_id in enumerate(obj_idx):
            self.scene.objects[obj_id].vel = vel[idx]

        self._clear_cache_doppler()
        return

    def _clear_cache_doppler(self) -> None:
        """Clear all cached attributes that depend on doppler computation."""
        try:
            super().__delitem__(c.DOPPLER_PARAM_NAME)
        except KeyError:
            pass  # Doppler cache is already cleared
            
    
    def _compute_doppler(self) -> np.ndarray:
        """Compute the doppler frequency shifts.
        
        Returns:
            np.ndarray: The doppler frequency shifts. [n_ue, max_paths] [Hz]

        NOTE: this Doppler computation is matching the Sionna Doppler computation.
              See Sionna.rt.Paths.doppler in: https://nvlabs.github.io/sionna/rt/api/paths.html
        """
        self.doppler_enabled = True
        doppler = np.zeros((self.n_ue, self.max_paths))
        if not self.doppler_enabled:
            return doppler
        
        wavelength = c.SPEED_OF_LIGHT / self.rt_params.frequency # [m]
        
        # Compute outgoing wave directions for all users and paths, at rx, tx, and interactions
        ones = np.ones((self.n_ue, self.max_paths, 1))
        tx_coord_cat = np.concatenate((ones, 
                                       np.deg2rad(self.aod_el)[..., None],
                                       np.deg2rad(self.aod_az)[..., None]), axis=-1)
        rx_coord_cat = -np.concatenate((ones,
                                        np.deg2rad(self.aoa_el)[..., None],
                                        np.deg2rad(self.aoa_az)[..., None]), axis=-1)

        k_tx = spherical_to_cartesian(tx_coord_cat) # [n_ue, max_paths, 3]
        k_rx = spherical_to_cartesian(rx_coord_cat) # [n_ue, max_paths, 3]

        k_i = self._compute_inter_angles() # [n_ue, max_paths, max_interactions, 3]

        inter_objects = self._compute_inter_objects()
        for ue_i in tqdm(range(self.n_ue), desc='Computing doppler per UE'):
            n_paths = self.num_paths[ue_i]
            for path_i in range(n_paths):
                if np.isnan(self.inter[ue_i, path_i]):
                    continue
                n_inter = self.num_interactions[ue_i, path_i]

                # Compute doppler for this path (using spherical coordinates)
                tx_doppler = np.dot(k_tx[ue_i, path_i], self.tx_vel) / wavelength
                rx_doppler = np.dot(k_rx[ue_i, path_i], self.rx_vel[ue_i]) / wavelength

                path_dopplers = [0]

                for i in range(int(n_inter)):  # i = interaction index(0, 1, ..., n_inter-1)
                    # Get object index
                    inter_obj_idx = inter_objects[ue_i, path_i, i]
                    if np.isnan(inter_obj_idx):
                        continue
                    
                    # Get object velocity
                    v_i = self.scene.objects[int(inter_obj_idx)].vel # [m/s]

                    # Get outgoing angle difference (between consecutive interactions)
                    # (comes from the taylor expansion of the doppler shift)
                    ki_diff = k_i[ue_i, path_i, i+1] - k_i[ue_i, path_i, i]

                    path_dopplers += [np.dot(v_i, ki_diff) / wavelength]
                
                # Compute doppler frequency shift
                doppler[ue_i, path_i] = tx_doppler - rx_doppler + np.sum(path_dopplers)

        return doppler
    
    def _compute_inter_angles(self) -> np.ndarray:
        """Compute the outgoing angles for all users and paths.
        
        For each path, computes N-1 angles where N is the number of interactions.
        Each angle represents the direction of propagation between consecutive interactions.
        The angles are returned in radians as [azimuth, elevation].
        
        Returns:
            np.ndarray: Array of shape [n_users, n_paths, max_interactions+1, 3] containing
                        the unit vectors between interactions (x, y, z)
        """
        inter_angles = np.zeros((self.n_ue, self.max_paths, self.max_inter+1, 3))

        # Use the interaction positions to compute angles between each interaction
        for ue_i in tqdm(range(self.n_ue), desc='Computing interaction angles per UE'):
            for path_i in range(self.max_paths):
                n_inter = self.num_interactions[ue_i, path_i]
                
                # Skip if no interactions
                if np.isnan(n_inter) or n_inter == 0:
                    continue
                
                # For each pair of consecutive interactions, compute the angle
                for i in range(-1, int(n_inter)):
                    # Get positions of current and next interaction
                    if i == -1:
                        pos1 = self.tx_pos
                    else:
                        pos1 = self.inter_pos[ue_i, path_i, i]
                    
                    if i == n_inter - 1:
                        pos2 = self.rx_pos[ue_i]
                    else:
                        pos2 = self.inter_pos[ue_i, path_i, i + 1]
                    
                    # Compute vector between interactions
                    vec = pos2 - pos1

                    # Store unit vector
                    inter_angles[ue_i, path_i, i+1] = vec / np.linalg.norm(vec)
        
        return inter_angles
    
    def _compute_inter_objects(self) -> np.ndarray:
        """Compute the objects that interact with each path of each user.
        
        For each path, computes N-1 objects where N is the number of interactions.
        Each object represents the object that the path interacts with.
        The objects are returned as the object index.

        Returns:
            np.ndarray: The objects that interact with each path of each user. [n_ue, max_paths, max_interactions]
        """
        inter_obj_ids = np.zeros((self.n_ue, self.max_paths, self.max_inter)) * np.nan

        # Ensure there is only one terrain object
        terrain_objs = [obj for obj in self.scene.objects if obj.label == 'terrain']
        if len(terrain_objs) > 1:
            raise ValueError('There should be only one terrain object')
        terrain_obj = terrain_objs[0]
        terrain_z_coord = terrain_obj.bounding_box.z_max # [m]
        
        non_terrain_objs = [obj for obj in self.scene.objects if obj.label != 'terrain']
        
        obj_centers = np.array([obj.bounding_box.center for obj in non_terrain_objs])
        obj_ids = np.array([obj.object_id for obj in non_terrain_objs])
        
        # Use the interaction positions to compute angles between each interaction
        for ue_i in tqdm(range(self.n_ue), desc='Computing interaction objects per UE'):
            for path_i in range(self.max_paths):
                n_inter = self.num_interactions[ue_i, path_i]
                
                # Skip if no interactions
                if np.isnan(n_inter) or n_inter == 0:
                    continue
                
                # For each pair of consecutive interactions, compute the object
                for i in range(int(n_inter)):
                    # Get positions of current and next interaction
                    i_pos = self.inter_pos[ue_i, path_i, i]  # Current interaction position
                    
                    # Check if the interaction is with the terrain
                    if np.isclose(i_pos[2], terrain_z_coord, rtol=0, atol=1e-3): # 1cm tolerance
                        inter_obj_ids[ue_i, path_i, i] = terrain_obj.object_id
                        continue
                    
                    # Get the distance between the interaction and the object
                    dist = np.linalg.norm(obj_centers - i_pos, axis=1)

                    # Get the object index
                    obj_idx = np.argmin(dist)
                    inter_obj_ids[ue_i, path_i, i] = obj_ids[obj_idx]
                    
        return inter_obj_ids

    ###########################################
    # 10. Utilities and Computation Methods
    ###########################################

    def _get_txrx_sets(self) -> list[TxRxSet]:
        """Get the txrx sets for the dataset.

        Returns:
            list[TxRxSet]: The txrx sets for the dataset.
        """
        return get_txrx_sets(self.get('parent_name', self.name))
    
    # Dictionary mapping attribute names to their computation methods
    # (in order of computation)
    _computed_attributes = {
        c.N_UE_PARAM_NAME: '_compute_n_ue',
        c.NUM_PATHS_PARAM_NAME: '_compute_num_paths',
        c.MAX_PATHS_PARAM_NAME: '_compute_max_paths',
        c.NUM_INTERACTIONS_PARAM_NAME: '_compute_num_interactions',
        c.MAX_INTERACTIONS_PARAM_NAME: '_compute_max_interactions',
        c.DIST_PARAM_NAME: '_compute_distances',
        c.PATHLOSS_PARAM_NAME: 'compute_pathloss',
        c.CHANNEL_PARAM_NAME: 'compute_channels',
        c.LOS_PARAM_NAME: '_compute_los',
        c.CH_PARAMS_PARAM_NAME: 'set_channel_params',
        c.DOPPLER_PARAM_NAME: '_compute_doppler',
        c.INTER_OBJECTS_PARAM_NAME: '_compute_inter_objects',
        
        # Power linear
        c.PWR_LINEAR_PARAM_NAME: '_compute_power_linear',
        
        # Rotated angles
        c.AOA_AZ_ROT_PARAM_NAME: '_compute_rotated_angles',
        c.AOA_EL_ROT_PARAM_NAME: '_compute_rotated_angles', 
        c.AOD_AZ_ROT_PARAM_NAME: '_compute_rotated_angles',
        c.AOD_EL_ROT_PARAM_NAME: '_compute_rotated_angles',
        'array_response_product': '_compute_array_response_product',
        
        # Field of view
        c.FOV_MASK_PARAM_NAME: '_compute_fov',
        c.AOA_AZ_FOV_PARAM_NAME: '_compute_fov',
        c.AOA_EL_FOV_PARAM_NAME: '_compute_fov',
        c.AOD_AZ_FOV_PARAM_NAME: '_compute_fov',
        c.AOD_EL_FOV_PARAM_NAME: '_compute_fov',
        
        # Power with antenna gain
        c.PWR_LINEAR_ANT_GAIN_PARAM_NAME: '_compute_power_linear_ant_gain',
        
        # Grid information
        'grid_size': '_compute_grid_info',
        'grid_spacing': '_compute_grid_info',

        # Interactions
        c.INTER_STR_PARAM_NAME: '_compute_inter_str',
        c.INTER_INT_PARAM_NAME: '_compute_inter_int',

        # Txrx set information
        'txrx_sets': '_get_txrx_sets',
    }

    def info(self, param_name: str | None = None) -> None:
        """Display help information about DeepMIMO dataset parameters.
        
        Args:
            param_name: Name of the parameter to get info about.
                       If None or 'all', displays information for all parameters.
                       If the parameter name is an alias, shows info for the resolved parameter.
        """
        # If it's an alias, resolve it first
        if param_name in c.DATASET_ALIASES:
            resolved_name = c.DATASET_ALIASES[param_name]
            print(f"'{param_name}' is an alias for '{resolved_name}'")
            param_name = resolved_name
            
        info(param_name)


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
    