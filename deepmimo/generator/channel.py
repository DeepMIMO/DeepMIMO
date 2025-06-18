"""
Channel module for DeepMIMO.

This module provides functionality for MIMO channel generation, including:
- Channel parameter management through the ChannelParameters class
- OFDM path generation and verification 
- Channel matrix computation

The main function is generate_MIMO_channel() which generates MIMO channel matrices
based on path information from ray-tracing and antenna configurations.
"""

import numpy as np
from tqdm import tqdm
from typing import Dict, Optional
from copy import deepcopy
from .. import consts as c
from ..general_utils import DotDict, compare_two_dicts

class ChannelParameters(DotDict):
    """Class for managing channel generation parameters.
    
    This class provides an interface for setting and accessing various parameters
    needed for MIMO channel generation, including:
    - BS/UE antenna array configurations
    - OFDM parameters
    - Channel domain settings (time/frequency)
    
    The parameters can be accessed directly using dot notation (e.g. params.bs_antenna.shape)
    or using dictionary notation (e.g. params['bs_antenna']['shape']).
    """
    # Default channel generation parameters
    DEFAULT_PARAMS = {
        # BS Antenna Parameters
        c.PARAMSET_ANT_BS: {
            c.PARAMSET_ANT_SHAPE: np.array([8, 1]), # Antenna dimensions in X - Y - Z
            c.PARAMSET_ANT_SPACING: 0.5,
            c.PARAMSET_ANT_ROTATION: np.array([0, 0, 0]), # Rotation around X - Y - Z axes
            c.PARAMSET_ANT_RAD_PAT: c.PARAMSET_ANT_RAD_PAT_VALS[0] # 'isotropic'
        },
        
        # UE Antenna Parameters
        c.PARAMSET_ANT_UE: {
            c.PARAMSET_ANT_SHAPE: np.array([1, 1]), # Antenna dimensions in X - Y - Z
            c.PARAMSET_ANT_SPACING: 0.5,
            c.PARAMSET_ANT_ROTATION: np.array([0, 0, 0]), # Rotation around X - Y - Z axes
            c.PARAMSET_ANT_RAD_PAT: c.PARAMSET_ANT_RAD_PAT_VALS[0] # 'isotropic'
        },
        
        c.PARAMSET_DOPPLER_EN: 0,
        c.PARAMSET_NUM_PATHS: c.MAX_PATHS, 
        
        c.PARAMSET_FD_CH: 1, # OFDM channel if 1, Time domain if 0
        
        # OFDM Parameters
        c.PARAMSET_OFDM: {
            c.PARAMSET_OFDM_SC_NUM: 512, # Number of total subcarriers
            c.PARAMSET_OFDM_SC_SAMP: np.arange(1), # Select subcarriers to generate
            c.PARAMSET_OFDM_BANDWIDTH: 10e6, # Hz
            c.PARAMSET_OFDM_LPF: 0 # Receive Low Pass / ADC Filter
        }
    }

    def __init__(self, data: Optional[Dict] = None):
        """Initialize channel generation parameters.
        
        Args:
            data: Optional dictionary containing channel parameters to override defaults
        """
        # Initialize with deep copy of defaults
        super().__init__(deepcopy(self.DEFAULT_PARAMS))
        
        # Update with provided data if any
        if data is not None:
            self.update(data)

    def validate(self, n_ues: int) -> 'ChannelParameters':
        """Validate channel generation parameters.
        
        This method checks that channel generation parameters are valid and
        consistent with the dataset configuration.
        
        Args:
            n_ues (int): Number of UEs to validate against
            
        Returns:
            ChannelParameters: Self for method chaining

        Raises:
            ValueError: If parameters are invalid or inconsistent
        """
        # Notify the user if some keyword is not used (likely set incorrectly)
        additional_keys = compare_two_dicts(self, ChannelParameters())
        if len(additional_keys):
            print('The following parameters seem unnecessary:')
            print(additional_keys)
        
        # BS Antenna Rotation
        if c.PARAMSET_ANT_ROTATION in self[c.PARAMSET_ANT_BS].keys():
            rotation_shape = self[c.PARAMSET_ANT_BS][c.PARAMSET_ANT_ROTATION].shape
            assert (len(rotation_shape) == 1 and rotation_shape[0] == 3), \
                    'The BS antenna rotation must be a 3D vector'
        else:
            self[c.PARAMSET_ANT_BS][c.PARAMSET_ANT_ROTATION] = None

        # UE Antenna Rotation
        if (c.PARAMSET_ANT_ROTATION in self[c.PARAMSET_ANT_UE].keys() and \
            self[c.PARAMSET_ANT_UE][c.PARAMSET_ANT_ROTATION] is not None):
            rotation_shape = self[c.PARAMSET_ANT_UE][c.PARAMSET_ANT_ROTATION].shape
            cond_1 = len(rotation_shape) == 1 and rotation_shape[0] == 3
            cond_2 = len(rotation_shape) == 2 and rotation_shape[0] == 3 and rotation_shape[1] == 2
            cond_3 = (rotation_shape[0] == n_ues)
        
            assert_str = ('The UE antenna rotation must either be a 3D vector for ' +
                         'constant values or 3 x 2 matrix for random values')
            assert cond_1 or cond_2 or cond_3, assert_str
        else:
            self[c.PARAMSET_ANT_UE][c.PARAMSET_ANT_ROTATION] = np.array([0, 0, 0])
        
        # BS Antenna Radiation Pattern
        if (c.PARAMSET_ANT_RAD_PAT in self[c.PARAMSET_ANT_BS].keys() and \
            self[c.PARAMSET_ANT_BS][c.PARAMSET_ANT_ROTATION] is not None):
            assert_str = ("The BS antenna radiation pattern must have " + 
                         f"one of the following values: {str(c.PARAMSET_ANT_RAD_PAT_VALS)}")
            assert self[c.PARAMSET_ANT_BS][c.PARAMSET_ANT_RAD_PAT] in c.PARAMSET_ANT_RAD_PAT_VALS, assert_str
        else:
            self[c.PARAMSET_ANT_BS][c.PARAMSET_ANT_RAD_PAT] = c.PARAMSET_ANT_RAD_PAT_VALS[0]
            
        # UE Antenna Radiation Pattern
        if c.PARAMSET_ANT_RAD_PAT in self[c.PARAMSET_ANT_UE].keys() and \
            self[c.PARAMSET_ANT_UE][c.PARAMSET_ANT_ROTATION] is not None:
            assert_str = ("The UE antenna radiation pattern must have one of the " + 
                         f"following values: {str(c.PARAMSET_ANT_RAD_PAT_VALS)}")
            assert self[c.PARAMSET_ANT_UE][c.PARAMSET_ANT_RAD_PAT] in c.PARAMSET_ANT_RAD_PAT_VALS, assert_str
        else:
            self[c.PARAMSET_ANT_UE][c.PARAMSET_ANT_RAD_PAT] = c.PARAMSET_ANT_RAD_PAT_VALS[0]
                                             
        return self

class OFDM_PathGenerator:
    """Class for generating OFDM paths with specified parameters.
    
    This class handles the generation of OFDM paths including optional
    low-pass filtering.
    
    Attributes:
        OFDM_params (dict): OFDM parameters
        subcarriers (array): Selected subcarrier indices
        total_subcarriers (int): Total number of subcarriers
        delay_d (array): Delay domain array
        delay_to_OFDM (array): Delay to OFDM transform matrix
    """
    
    def __init__(self, params: Dict, subcarriers: np.ndarray):
        """Initialize OFDM path generator.
        
        Args:
            params (dict): OFDM parameters
            subcarriers (array): Selected subcarrier indices
        """
        self.OFDM_params = params
        self.subcarriers = subcarriers  # selected
        self.total_subcarriers = self.OFDM_params[c.PARAMSET_OFDM_SC_NUM]
        
        self.delay_d = np.arange(self.OFDM_params[c.PARAMSET_OFDM_SC_NUM])
        self.delay_to_OFDM = np.exp(-1j * 2 * np.pi / self.total_subcarriers * 
                                    np.outer(self.delay_d, self.subcarriers))
    
    def generate(self, pwr: np.ndarray, toa: np.ndarray, phs: np.ndarray, Ts: float, dopplers: np.ndarray) -> np.ndarray:
        """Generate OFDM paths.
        
        Args:
            pwr (array): Path powers
            toa (array): Times of arrival
            phs (array): Path phases
            Ts (float): Sampling period
            
        Returns:
            array: Generated OFDM paths
        """
        # Add a new dimension to the end of the array to allow for broadcasting
        power = pwr[..., None]
        delay_n = toa[..., None] / Ts
        phase = phs[..., None]
        doppler_n = dopplers[..., None]
    
        # Ignore paths over CP
        paths_over_FFT = (delay_n >= self.OFDM_params[c.PARAMSET_OFDM_SC_NUM])
        power[paths_over_FFT] = 0
        delay_n[paths_over_FFT] = self.OFDM_params[c.PARAMSET_OFDM_SC_NUM]
        doppler_n[paths_over_FFT] = 0
        
        # Reshape path_const to be compatible with broadcasting
        path_const = np.sqrt(power / self.total_subcarriers) * np.exp(1j * (np.deg2rad(phase) + 2 * np.pi * doppler_n))
        if self.OFDM_params[c.PARAMSET_OFDM_LPF]: # Low-pass filter (LPF) convolution
            path_const = path_const * np.sinc(self.delay_d - delay_n) @ self.delay_to_OFDM
        else: # Path construction without LPF
            path_const = path_const * np.exp(-1j * (2 * np.pi / self.total_subcarriers) * 
                                             np.outer(delay_n.ravel(), self.subcarriers))
        return path_const

def _generate_MIMO_channel(array_response_product: np.ndarray,
                           powers: np.ndarray,
                           delays: np.ndarray,
                           phases: np.ndarray,
                           dopplers: np.ndarray,
                           ofdm_params: Dict,
                           freq_domain: bool = True) -> np.ndarray:
    """Generate MIMO channel matrices.
    
    This function generates MIMO channel matrices based on path information and
    pre-computed array responses. It supports both time and frequency domain
    channel generation.
    
    Args:
        array_response_product: Product of TX and RX array responses [n_users, M_rx, M_tx, n_paths]
        powers: Linear path powers [W] with antenna gains applied [n_users, n_paths]
        toas: Times of arrival [n_users, n_paths]
        phases: Path phases [n_users, n_paths]
        dopplers: Doppler frequency shifts [Hz] for each user and path.
        ofdm_params: OFDM parameters
        freq_domain: Whether to generate frequency domain channel. Defaults to True.

    Returns:
        numpy.ndarray: MIMO channel matrices with shape (n_users, n_rx_ant, n_tx_ant, n_paths/subcarriers)
    """
    Ts = 1 / ofdm_params[c.PARAMSET_OFDM_BANDWIDTH]
    subcarriers = ofdm_params[c.PARAMSET_OFDM_SC_SAMP]
    path_gen = OFDM_PathGenerator(ofdm_params, subcarriers)

    # Check if any paths exceed OFDM symbol duration
    if freq_domain:
        ofdm_symbol_duration = ofdm_params[c.PARAMSET_OFDM_SC_NUM] * Ts
        subcarrier_spacing = ofdm_params[c.PARAMSET_OFDM_BANDWIDTH] / ofdm_params[c.PARAMSET_OFDM_SC_NUM]  # Hz
        max_delay = np.nanmax(delays)
        
        if max_delay > ofdm_symbol_duration:
            print("\nWarning: Some path delays exceed OFDM symbol duration")
            print("-" * 50)
            print(f"OFDM Configuration:")
            print(f"- Number of subcarriers (N): {ofdm_params[c.PARAMSET_OFDM_SC_NUM]}")
            print(f"- Bandwidth (B): {ofdm_params[c.PARAMSET_OFDM_BANDWIDTH]/1e6:.1f} MHz")
            print(f"- Subcarrier spacing (Δf = B/N): {subcarrier_spacing/1e3:.1f} kHz")
            print(f"- Symbol duration (T = 1/Δf = N/B): {ofdm_symbol_duration*1e6:.1f} μs")
            print(f"\nPath Information:")
            print(f"- Maximum path delay: {max_delay*1e6:.1f} μs")
            print(f"- Excess delay: {(max_delay - ofdm_symbol_duration)*1e6:.1f} μs")
            print("\nPaths arriving after the symbol duration will be clipped.")
            print("To avoid clipping, either:")
            print("1. Increase the number of subcarriers (N)")
            print("2. Decrease the bandwidth (B)")
            print(f"3. Switch to time-domain channel generation (set ch_params['{c.PARAMSET_FD_CH}'] = 0)")
            # print(f"4. (not recommended) Turn off OFDM path trimming (set ch_params['{c.PARAMSET_OFDM_PATH_TRIM}'] = False)")
            print("-" * 50)

    n_ues = powers.shape[0]
    max_paths = powers.shape[1]
    M_rx, M_tx = array_response_product.shape[1:3]
    
    last_ch_dim = len(subcarriers) if freq_domain else max_paths
    channel = np.zeros((n_ues, M_rx, M_tx, last_ch_dim), dtype=np.csingle)
    
    # Pre-compute NaN masks for all users using powers
    nan_masks = ~np.isnan(powers)  # [n_users, n_paths]
    valid_path_counts = np.sum(nan_masks, axis=1)  # [n_users]

    # Generate channels for each user
    for i in tqdm(range(n_ues), desc='Generating channels'):
        # Get valid paths for this user
        non_nan_mask = nan_masks[i]
        n_paths = valid_path_counts[i]
        
        # Skip users with no valid paths
        if n_paths == 0:
            continue
            
        # Get pre-computed array product for this user (with NaN handling)
        array_product = array_response_product[i][..., non_nan_mask]  # [M_rx, M_tx, n_valid_paths]
        
        # Get pre-computed values for this user
        power = powers[i, non_nan_mask]
        delays_user = delays[i, non_nan_mask]
        phases_user = phases[i, non_nan_mask]
        dopplers_user = dopplers[i, non_nan_mask]
        
        if freq_domain: # OFDM
            path_gains = path_gen.generate(pwr=power, toa=delays_user, phs=phases_user, Ts=Ts, dopplers=dopplers_user).T
            channel[i] = np.nansum(array_product[..., None, :] * path_gains[None, None, :, :], axis=-1)
        else: # TD channel
            path_gains = np.sqrt(power) * np.exp(1j * (np.deg2rad(phases_user) + 2 * np.pi * dopplers_user))
            channel[i, ..., :n_paths] = array_product * path_gains[None, None, :]

    return channel