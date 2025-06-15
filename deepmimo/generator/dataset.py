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
from ..general_utils import DotDict
from .. import consts as c
from ..info import info
from .visualization import plot_coverage, plot_rays

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
)

# Converter utilities
from ..converter import converter_utils as cu

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
        power: Path powers in dBm
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
    
    ###########################################
    # 1. Core Interface
    ###########################################
    
    def __init__(self, data: Optional[Dict[str, Any]] = None):
        """Initialize dataset with optional data.
        
        Args:
            data: Initial dataset dictionary. If None, creates empty dataset.
        """
        super().__init__(data or {})

    def __getattr__(self, key: str) -> Any:
        """Enable dot notation access."""
        try:
            return super().__getitem__(key)
        except KeyError:
            return self._resolve_key(key)

    def __getitem__(self, key: str) -> Any:
        """Get an item from the dataset, computing it if necessary."""
        try:
            return super().__getitem__(key)
        except KeyError:
            return self._resolve_key(key)

    def _resolve_key(self, key: str) -> Any:
        """Resolve a key through the lookup chain.
        
        Order of operations:
        1. Check if key is an alias and resolve it first
        2. Try direct access with resolved key
        3. Try computing the attribute if it's computable
        
        Args:
            key: The key to resolve
            
        Returns:
            The resolved value
            
        Raises:
            KeyError if key cannot be resolved
        """
        # First check if it's an alias and resolve it
        resolved_key = c.DATASET_ALIASES.get(key, key)
        if resolved_key != key:
            key = resolved_key
            try:
                return super().__getitem__(key)
            except KeyError:
                pass
            
        if key in self._computed_attributes:
            compute_method_name = self._computed_attributes[key]
            compute_method = getattr(self, compute_method_name)
            value = compute_method()
            # Cache the result
            if isinstance(value, dict):
                self.update(value)
                return super().__getitem__(key)
            else:
                self[key] = value
                return value
        
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
    
    def compute_channels(self, params: Optional[ChannelParameters] = None) -> np.ndarray:
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
            
        Returns:
            numpy.ndarray: MIMO channel matrix with shape [n_users, n_rx_ant, n_tx_ant, n_subcarriers]
                          if freq_domain=True, otherwise [n_users, n_rx_ant, n_tx_ant, n_paths]
        """
        if params is None:
            params = ChannelParameters() if self.ch_params is None else self.ch_params

        self.set_channel_params(params)

        np.random.seed(1001)
        
        # Compute array response product
        array_response_product = self._compute_array_response_product()
        
        n_paths_to_gen = params.num_paths
        
        # Whether to enable the doppler shift per path in the channel
        if params[c.PARAMSET_DOPPLER_EN]:
            dopplers = self.doppler[..., :n_paths_to_gen]
        else:
            dopplers = np.zeros((self.n_ue, n_paths_to_gen))

        channel = _generate_MIMO_channel(
            array_response_product=array_response_product[..., :n_paths_to_gen],
            powers=self._power_linear_ant_gain[..., :n_paths_to_gen],
            delays=self.delay[..., :n_paths_to_gen],
            phases=self.phase[..., :n_paths_to_gen],
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
            
        # Rotate angles for all users at once
        aod_theta_rot, aod_phi_rot = _rotate_angles_batch(
            rotation=tx_ant_params[c.PARAMSET_ANT_ROTATION],
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
    
    def get_row_idxs(self, row_idxs: list[int] | np.ndarray) -> np.ndarray:
        """Return indices of users in the specified rows, assuming a grid structure.
        
        Args:
            row_idxs: Array of row indices to include in the new dataset

        Returns:
            Array of indices of receivers in the specified rows
        """
        # For each row, get all indices in that row
        indices = []
        for row in row_idxs:
            # Each row contains grid_size[0] elements (number of columns)
            row_start = row * self.grid_size[0]
            row_indices = np.arange(row_start, row_start + self.grid_size[0])
            indices.extend(row_indices)
        return np.array(indices)
        
    def get_col_idxs(self, col_idxs: list[int] | np.ndarray) -> np.ndarray:
        """Return indices of users in the specified columns, assuming a grid structure.
        
        Args:
            col_idxs: Array of column indices to include in the new dataset

        Returns:
            Array of indices of receivers in the specified columns
        """
        # For each column, get all indices in that column
        indices = []
        for col in col_idxs:
            # Each column contains grid_size[1] elements
            col_indices = col + np.arange(self.grid_size[1]) * self.grid_size[0]
            indices.extend(col_indices)
        return np.array(indices)

    
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
            if hasattr(self, param):
                initial_data[param] = getattr(self, param)
            
        # Directly set n_ue
        initial_data['n_ue'] = len(idxs)
        
        # Create new dataset with initial data
        new_dataset = Dataset(initial_data)
        
        # Copy all attributes
        for attr, value in self.to_dict().items():
            # skip private and already handled attributes
            if not attr.startswith('_') and attr not in SHARED_PARAMS + ['n_ue']:
                if isinstance(value, np.ndarray) and value.shape[0] == self.n_ue:
                    # Copy and index arrays with UE dimension
                    setattr(new_dataset, attr, value[idxs])
                else:
                    # Copy other attributes as is
                    setattr(new_dataset, attr, value)
                
        return new_dataset

    def _trim_by_path(self, path_mask: np.ndarray) -> 'Dataset':
        """Helper function to trim paths based on a boolean mask.
        
        Args:
            path_mask: Boolean array of shape [n_users, n_paths] indicating which paths to keep.
            
        Returns:
            A new Dataset with trimmed paths.
        """
        # Create a new dataset with the same structure
        new_dataset = self.deepcopy()
        
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
            new_dataset[array_name][~path_mask] = np.nan
        
        # Create new order for each user: valid paths first, then invalid paths
        new_order = np.argsort(~path_mask, axis=1)  # False (valid) comes before True (invalid)
        
        # Reorder all path arrays
        for array_name in path_arrays:
            if array_name == c.INTERACTIONS_POS_PARAM_NAME:
                # Handle 4D array (n_users, n_paths, n_interactions, 3)
                new_dataset[array_name] = np.take_along_axis(new_dataset[array_name], new_order[:, :, None, None], axis=1)
            else:
                # Handle 2D arrays (n_users, n_paths)
                new_dataset[array_name] = np.take_along_axis(new_dataset[array_name], new_order, axis=1)
        
        # Compress arrays to remove unused paths
        data_dict = {k: v for k, v in new_dataset.items() if isinstance(v, np.ndarray)}
        compressed_data = cu.compress_path_data(data_dict)
        
        # Update dataset with compressed arrays
        for key, value in compressed_data.items():
            new_dataset[key] = value
        
        # Reset computed attributes
        new_dataset._computed_attributes = {}  # Clear computed attributes
        
        return new_dataset

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
    
    def plot_rays(self, idx: int, **kwargs):
        """Plot the rays of the dataset.
        
        Args:
            **kwargs: Additional keyword arguments to pass to the plot_rays function.
        """
        default_kwargs = {
            'proj_3D': True,
            'color_by_type': True,
        }
        default_kwargs.update(kwargs)
        return plot_rays(self.rx_pos[idx], self.tx_pos[0], self.inter_pos[idx],
                         self.inter[idx], **default_kwargs)
    
    def plot_summary(self, **kwargs):
        """Plot the summary of the dataset."""
        return plot_summary(dataset=self, **kwargs)
    
    ###########################################
    # 9. Doppler Computations
    ###########################################
    
    def _cartesian_to_spherical(self, cartesian_coords: np.ndarray) -> np.ndarray:
        """Convert Cartesian coordinates to spherical coordinates.
        
        Args:
            cartesian_coords: Array of shape [n_points, 3] containing Cartesian coordinates (x, y, z)
            
        Returns:
            Array of shape [n_points, 2] containing spherical coordinates (azimuth, elevation) in radians
        """
        spherical_coords = np.zeros((cartesian_coords.shape[0], 2))
        spherical_coords[:, 0] = np.arctan2(cartesian_coords[:, 1], cartesian_coords[:, 0])  # azimuth
        spherical_coords[:, 1] = np.arctan2(cartesian_coords[:, 2], 
                                          np.sqrt(cartesian_coords[:, 0]**2 + cartesian_coords[:, 1]**2))  # elevation
        return spherical_coords

    def set_rx_vel(self, velocities: np.ndarray) -> np.ndarray:
        """Set the velocities of the users.
        
        Args:
            velocities: The velocities of the users in cartesian coordinates. [m/s]
        
        Returns:
            The velocities of the users in spherical coordinates.
        """
        self._clear_cache_doppler()
        
        if velocities.ndim == 1:
            # [3,] -> [n_ue, 3]
            self.rx_vel = np.repeat(velocities[None, :], self.n_ue, axis=0)
        else:
            if velocities.shape[1] != 3:
                raise ValueError('Velocities must be in cartesian coordinates (n_ue, 3)')
            if velocities.shape[0] != self.n_ue:
                raise ValueError('Number of users must match number of velocities (n_ue, 3)')
            
            self.rx_vel = velocities
        
        # Convert to spherical coordinates
        self.rx_vel_s = self._cartesian_to_spherical(self.rx_vel)
        return self.rx_vel_s
        # self.rx_vel = np.zeros((self.n_ue, 2))

    def set_tx_vel(self, velocities: np.ndarray) -> np.ndarray:
        """Set the velocities of the base stations."""
        self._clear_cache_doppler()
        if velocities.ndim != 1:
            raise ValueError('Tx velocity must be in a single cartesian coordinate (2,)')
        
        self.tx_vel = velocities
        self.tx_vel_s = self._cartesian_to_spherical(self.tx_vel[None, :])
        self.tx_vel_s = self.tx_vel_s[0]
        return self.tx_vel_s

    def _clear_cache_doppler(self) -> None:
        """Clear all cached attributes that depend on doppler computation."""
        try:
            super().__delitem__(c.DOPPLER_PARAM_NAME)
        except KeyError:
            print('Doppler cache is already cleared')
    
    def _compute_doppler(self) -> np.ndarray:
        """Compute the doppler frequency shifts."""
        self.doppler_enabled = True
        max_paths = np.nanmax(self.num_paths)
        doppler = np.zeros((self.n_ue, max_paths)) + 2 
        if not self.doppler_enabled:
            return doppler
        
        freq = 1e9
        wavelength = 3e8 / freq
        
        # Compute k_tx and k_rx for all users and paths
        k_tx = np.zeros((self.n_ue, max_paths, 2))
        k_rx = np.zeros((self.n_ue, max_paths, 2))
        
        # Compute wave vectors for all users and paths
        k_tx[:, :, 0] = np.deg2rad(self.aod_az)   # azimuth of departure (tx angle)
        k_tx[:, :, 1] = np.deg2rad(self.aod_el)   # elevation of departure (tx angle)
        k_rx[:, :, 0] = -np.deg2rad(self.aoa_az)  # azimuth of arrival (rx angle)
        k_rx[:, :, 1] = -np.deg2rad(self.aoa_el)  # elevation of arrival (rx angle)
        
        k_i = self._compute_interaction_angles() # [n_ue, max_paths, max_interactions, 2]

        # Initialize velocity arrays
        v_tx = np.zeros((1, 3))
        v_rx = np.zeros((self.n_ue, 3))
        
        for ue_i in tqdm(range(self.n_ue), desc='Computing doppler per UE'):
            n_paths = self.num_paths[ue_i]
            for path_i in range(n_paths):
                if np.isnan(self.inter[ue_i, path_i]):
                    continue
                n_inter = self.num_interactions[ue_i, path_i]

                # Compute doppler for this path (using spherical coordinates)
                tx_doppler = np.dot(k_tx[ue_i, path_i], self.tx_vel_s) / wavelength
                rx_doppler = np.dot(k_rx[ue_i, path_i], self.rx_vel_s[ue_i]) / wavelength

                path_dopplers = []

                for i in range(1, int(n_inter)):  # i = interaction index(0, 1, ..., n_inter-1)
                    # Get object index
                    
                    # Get velocity of the object / Rx / Tx
                    
                    # Get outgoing angle
                    v_i = np.array([0, 0]) # TODO: get the velocity of the object

                    ki_diff = k_i[ue_i, path_i, i] - k_i[ue_i, path_i, i-1]
                    path_dopplers += [np.dot(v_i, ki_diff) / wavelength]
                
                # Compute doppler frequency shift
                doppler[ue_i, path_i] = tx_doppler - rx_doppler + np.sum(path_dopplers)

        return doppler
    
    def _compute_interaction_angles(self) -> np.ndarray:
        """Compute the outgoing angles for all users and paths.
        
        For each path, computes N-1 angles where N is the number of interactions.
        Each angle represents the direction of propagation between consecutive interactions.
        The angles are returned in radians as [azimuth, elevation].
        
        Returns:
            np.ndarray: Array of shape [n_users, n_paths, max_interactions, 2] containing
                       the azimuth and elevation angles between interactions in radians.
        """
        max_interactions = np.nanmax(self.num_interactions).astype(int)
        max_paths = np.nanmax(self.num_paths).astype(int)
        interaction_angles = np.zeros((self.n_ue, max_paths, max_interactions, 2))

        # Use the interaction positions to compute angles between each interaction
        for ue_i in tqdm(range(self.n_ue), desc='Computing interaction angles per UE'):
            for path_i in range(max_paths):
                n_inter = self.num_interactions[ue_i, path_i]
                
                # Skip if no interactions
                if np.isnan(n_inter) or n_inter <= 1:
                    continue
                    
                # For each pair of consecutive interactions, compute the angle
                for i in range(int(n_inter) - 1):
                    # Get positions of current and next interaction
                    pos1 = self.inter_pos[ue_i, path_i, i]  # Current interaction
                    pos2 = self.inter_pos[ue_i, path_i, i + 1]  # Next interaction
                    
                    # Compute vector between interactions
                    vec = pos2 - pos1
                    
                    # Compute azimuth (in x-y plane)
                    azimuth = np.arctan2(vec[1], vec[0])
                    
                    # Compute elevation (angle from x-y plane)
                    # First get horizontal distance
                    horizontal_dist = np.sqrt(vec[0]**2 + vec[1]**2)
                    elevation = np.arctan2(vec[2], horizontal_dist)
                    
                    # Store angles in radians
                    interaction_angles[ue_i, path_i, i] = [azimuth, elevation]
        
        # TODO: Use vectorized version
        # # Get all interaction positions
        # # Shape: [n_users, n_paths, max_interactions, 3]
        # inter_pos = self.inter_pos

        # # Create mask for valid interactions
        # # Shape: [n_users, n_paths, max_interactions]
        # valid_mask = np.arange(max_interactions)[None, None, :] < self.num_interactions[..., None]

        # # Get vectors between consecutive interactions
        # # Shape: [n_users, n_paths, max_interactions-1, 3]
        # vec = inter_pos[..., 1:, :] - inter_pos[..., :-1, :]

        # # Compute horizontal distance for elevation calculation
        # # Shape: [n_users, n_paths, max_interactions-1]
        # horizontal_dist = np.sqrt(vec[..., 0]**2 + vec[..., 1]**2)

        # # Compute azimuth (in x-y plane)
        # # Shape: [n_users, n_paths, max_interactions-1]
        # azimuth = np.arctan2(vec[..., 1], vec[..., 0])

        # # Compute elevation (angle from x-y plane)
        # # Shape: [n_users, n_paths, max_interactions-1]
        # elevation = np.arctan2(vec[..., 2], horizontal_dist)

        # # Stack angles and store in output array
        # # Shape: [n_users, n_paths, max_interactions-1, 2]
        # angles = np.stack([azimuth, elevation], axis=-1)

        # # Store angles in output array, only for valid interactions
        # valid_mask_prev = valid_mask[..., :-1]  # Shift mask to match angles
        # interaction_angles[..., :-1, :] = np.where(valid_mask_prev[..., None], angles, 0)

        return interaction_angles
    

    ###########################################
    # 10. Utilities and Computation Methods
    ###########################################

    def _get_txrx_sets(self) -> list[TxRxSet]:
        """Get the txrx sets for the dataset."""
        return get_txrx_sets(self.get('parent_name', self.name))
    
    # Dictionary mapping attribute names to their computation methods
    # (in order of computation)
    _computed_attributes = {
        c.N_UE_PARAM_NAME: '_compute_n_ue',
        c.NUM_PATHS_PARAM_NAME: '_compute_num_paths',
        c.NUM_INTERACTIONS_PARAM_NAME: '_compute_num_interactions',
        c.DIST_PARAM_NAME: '_compute_distances',
        c.PATHLOSS_PARAM_NAME: 'compute_pathloss',
        c.CHANNEL_PARAM_NAME: 'compute_channels',
        c.LOS_PARAM_NAME: '_compute_los',
        c.CH_PARAMS_PARAM_NAME: 'set_channel_params',
        c.DOPPLER_PARAM_NAME: '_compute_doppler',
        
        # Power linear
        c.PWR_LINEAR_PARAM_NAME: '_compute_power_linear',
        
        # Rotated angles
        c.AOA_AZ_ROT_PARAM_NAME: '_compute_rotated_angles',
        c.AOA_EL_ROT_PARAM_NAME: '_compute_rotated_angles', 
        c.AOD_AZ_ROT_PARAM_NAME: '_compute_rotated_angles',
        c.AOD_EL_ROT_PARAM_NAME: '_compute_rotated_angles',
        'array_response_product': '_compute_array_response_product',
        
        # Field of view
        'fov': '_compute_fov',
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
    
    SINGLE_ACCESS_METHODS = MacroDataset.SINGLE_ACCESS_METHODS + ['txrx_sets']

    def __init__(self, datasets: list[MacroDataset], name: str):
        """Initialize a dynamic dataset.
        
        Args:
            datasets: List of MacroDataset instances, each representing a time snapshot
            name: Base name of the scenario (without time suffix)
        """
        super().__init__(datasets)
        self.name = name
        self.names = [dataset.name for dataset in datasets]

        for dataset in datasets:
            dataset.parent_name = name
        
    def __getattr__(self, name):
        """Override __getattr__ to handle txrx_sets specially."""
        if name == 'txrx_sets':
            return get_txrx_sets(self.name)
        return super().__getattr__(name)
