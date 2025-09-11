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
from typing import Dict, Optional, Any
from copy import deepcopy
from .. import consts as c
from ..general_utils import DotDict, compare_two_dicts, deep_dict_merge

def _convert_lists_to_arrays(obj: Any) -> Any:
    """Recursively convert lists to numpy arrays in nested dictionaries.
    
    This function traverses through nested dictionaries and converts any list
    values to numpy arrays. This allows users to provide parameters as lists
    instead of requiring explicit np.array() calls.
    
    Args:
        obj: Object to process (dict, list, or other type)
        
    Returns:
        Object with lists converted to numpy arrays
    """
    if isinstance(obj, DotDict):
        obj.update({key: _convert_lists_to_arrays(value) for key, value in obj.items()})
        return obj
    elif isinstance(obj, dict):
        return {key: _convert_lists_to_arrays(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return np.array(obj)
    else:
        return obj

def _validate_ant_rot(rotation: np.ndarray, n_ues: Optional[int] = None) -> np.ndarray:
    """Validate antenna rotation parameters.
    
    Args:
        rotation: Rotation angles array (3D vector or 3 x 2 matrix)
        n_ues: Number of UEs (only needed for UE rotation validation)
        
    Returns:
        Validated rotation array
        
    Raises:
        AssertionError: If rotation format is invalid
    """
    if rotation is None:
        return np.array([0, 0, 0])
        
    rotation_shape = rotation.shape
    cond_1 = len(rotation_shape) == 1 and rotation_shape[0] == 3  # Fixed 3D vector
    cond_2 = len(rotation_shape) == 2 and rotation_shape[0] == 3 and rotation_shape[1] == 2  # Random ranges
    cond_3 = n_ues is not None and rotation_shape[0] == n_ues  # Per-user rotations
    
    assert_str = ('The antenna rotation must either be a 3D vector for ' +
                  'constant values or 3 x 2 matrix for random values')
    if n_ues is not None:
        assert_str += ' or an n_ues x 3 matrix for per-user values'
    
    assert cond_1 or cond_2 or (n_ues is not None and cond_3), assert_str
    
    return rotation

def _validate_ant_rad_pat(pattern: Optional[str] = None) -> str:
    """Validate antenna radiation pattern.
    
    Args:
        pattern: Radiation pattern string
        
    Returns:
        Validated pattern string
        
    Raises:
        AssertionError: If pattern is invalid
    """
    if pattern is None:
        return c.PARAMSET_ANT_RAD_PAT_VALS[0]
        
    assert_str = ("The antenna radiation pattern must have one of the " + 
                  f"following values: {str(c.PARAMSET_ANT_RAD_PAT_VALS)}")
    assert pattern in c.PARAMSET_ANT_RAD_PAT_VALS, assert_str
    
    return pattern

class ChannelParameters(DotDict):
    """Class for managing channel generation parameters.
    
    This class provides an interface for setting and accessing various parameters
    needed for MIMO channel generation, including:
    - BS/UE antenna array configurations
    - OFDM parameters
    - Channel domain settings (time/frequency)
    
    The parameters can be accessed directly using dot notation (e.g. params.bs_antenna.shape)
    or using dictionary notation (e.g. params['bs_antenna']['shape']).
    
    Examples:
        # Initialize with default parameters
        params = ChannelParameters()
        
        # Initialize with specific parameters
        params = ChannelParameters(doppler=True, freq_domain=True)
        
        # Initialize with nested parameters (lists are automatically converted to numpy arrays during validation)
        params = ChannelParameters(bs_antenna={'shape': [4, 4]})  # Other bs_antenna fields preserved
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

    def __init__(self, data: Optional[Dict] = None, **kwargs):
        """Initialize channel generation parameters.
        
        Args:
            data: Optional dictionary containing channel parameters to override defaults
            **kwargs: Additional parameters to override defaults. 
                These will be merged with data if provided.
                For nested parameters, provide them as dictionaries (e.g. bs_antenna={'shape': [4,4]})
                Only specified fields will be overridden, other fields will keep their default values.
                Lists will be automatically converted to numpy arrays during validation.
        """
        # Initialize with deep copy of defaults
        super().__init__(deepcopy(self.DEFAULT_PARAMS))
        
        # Update with provided data if any
        if data is not None:
            self.update(deep_dict_merge(self, data))
            
        # Update with kwargs if any provided
        if kwargs:
            self.update(deep_dict_merge(self, kwargs))

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
        # Convert lists to arrays before validation
        self = _convert_lists_to_arrays(self)
        
        # Notify the user if some keyword is not used (likely set incorrectly)
        additional_keys = compare_two_dicts(self, ChannelParameters())
        if len(additional_keys):
            print('The following parameters seem unnecessary:')
            print(additional_keys)
        
        # BS Antenna Rotation
        if c.PARAMSET_ANT_ROTATION in self[c.PARAMSET_ANT_BS].keys():
            self[c.PARAMSET_ANT_BS][c.PARAMSET_ANT_ROTATION] = _validate_ant_rot(
                self[c.PARAMSET_ANT_BS][c.PARAMSET_ANT_ROTATION])
        else:
            self[c.PARAMSET_ANT_BS][c.PARAMSET_ANT_ROTATION] = np.array([0, 0, 0])

        # UE Antenna Rotation
        if c.PARAMSET_ANT_ROTATION in self[c.PARAMSET_ANT_UE].keys():
            self[c.PARAMSET_ANT_UE][c.PARAMSET_ANT_ROTATION] = _validate_ant_rot(
                self[c.PARAMSET_ANT_UE][c.PARAMSET_ANT_ROTATION], n_ues)
        else:
            self[c.PARAMSET_ANT_UE][c.PARAMSET_ANT_ROTATION] = np.array([0, 0, 0])
        
        # BS Antenna Radiation Pattern
        if c.PARAMSET_ANT_RAD_PAT in self[c.PARAMSET_ANT_BS].keys():
            self[c.PARAMSET_ANT_BS][c.PARAMSET_ANT_RAD_PAT] = _validate_ant_rad_pat(
                self[c.PARAMSET_ANT_BS][c.PARAMSET_ANT_RAD_PAT])
        else:
            self[c.PARAMSET_ANT_BS][c.PARAMSET_ANT_RAD_PAT] = c.PARAMSET_ANT_RAD_PAT_VALS[0]
            
        # UE Antenna Radiation Pattern
        if c.PARAMSET_ANT_RAD_PAT in self[c.PARAMSET_ANT_UE].keys():
            self[c.PARAMSET_ANT_UE][c.PARAMSET_ANT_RAD_PAT] = _validate_ant_rad_pat(
                self[c.PARAMSET_ANT_UE][c.PARAMSET_ANT_RAD_PAT])
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
        self.subcarriers = subcarriers  # selected (shape [K_sel])
        self.total_subcarriers = self.OFDM_params[c.PARAMSET_OFDM_SC_NUM]
        
        self.delay_d = np.arange(self.OFDM_params[c.PARAMSET_OFDM_SC_NUM])
        # [N, K_sel]
        self.delay_to_OFDM = np.exp(
            -1j * 2 * np.pi / self.total_subcarriers
            * np.outer(self.delay_d, self.subcarriers)
        )
    
    def generate(self, pwr: np.ndarray, toa: np.ndarray, phs: np.ndarray, Ts: float, dopplers: np.ndarray) -> np.ndarray:
        """Generate OFDM paths (ORIGINAL, no time axis and incorrect Doppler usage).
        
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
    
        # Ignore paths over CP/FFT
        paths_over_FFT = (delay_n >= self.OFDM_params[c.PARAMSET_OFDM_SC_NUM])
        power[paths_over_FFT] = 0
        delay_n[paths_over_FFT] = self.OFDM_params[c.PARAMSET_OFDM_SC_NUM]
        doppler_n[paths_over_FFT] = 0
        
        # Reshape path_const to be compatible with broadcasting
        path_const = np.sqrt(power / self.total_subcarriers) * np.exp(
            1j * (np.deg2rad(phase) + 2 * np.pi * doppler_n)
        )
        if self.OFDM_params[c.PARAMSET_OFDM_LPF]:  # Low-pass filter (LPF) convolution
            path_const = path_const * (np.sinc(self.delay_d - delay_n) @ self.delay_to_OFDM)
        else:  # Path construction without LPF
            path_const = path_const * np.exp(
                -1j * (2 * np.pi / self.total_subcarriers)
                * np.outer(delay_n.ravel(), self.subcarriers)
            )
        return path_const

    # -------------------- V2 (vectorized time-aware) --------------------

    def generate_v2(
        self,
        pwr: np.ndarray,
        toa: np.ndarray,
        phs: np.ndarray,
        Ts: float,
        dopplers: np.ndarray,
        times: float | np.ndarray,
    ) -> np.ndarray:
        """
        Generate per-path, per-subcarrier, per-time path gains with correct Doppler progression.

        Inputs:
            pwr       : [P]       linear powers per path
            toa       : [P]       times of arrival (seconds)
            phs       : [P]       initial phases (degrees)
            Ts        : scalar    sampling period (seconds)
            dopplers  : [P]       Doppler frequencies (Hz)
            times     : scalar or [N_t] times (seconds) at which to sample

        Returns:
            h_pkn     : [P, K_sel, N_t] complex64 path gains
                        (if times is scalar, N_t = 1; caller may squeeze)
        """
        # Ensure 1-D numpy arrays
        pwr = np.asarray(pwr)
        toa = np.asarray(toa)
        phs = np.asarray(phs)
        dopplers = np.asarray(dopplers)

        times = np.atleast_1d(times).astype(float)    # [N_t]
        Nt = times.shape[0]
        K = len(self.subcarriers)
        P = pwr.shape[0]

        # Base dimensions
        power   = pwr[:, None]                         # [P, 1]
        delay_n = (toa / Ts)[:, None]                  # [P, 1] (sample units)
        phase0  = np.deg2rad(phs)[:, None]             # [P, 1] (radians)
        fD      = dopplers[:, None]                    # [P, 1] (Hz)

        # Ignore paths over FFT (clip to zero)
        over = delay_n >= self.OFDM_params[c.PARAMSET_OFDM_SC_NUM]
        if np.any(over):
            power[over]   = 0.0
            delay_n[over] = self.OFDM_params[c.PARAMSET_OFDM_SC_NUM]
            fD[over]      = 0.0

        # Doppler-induced phase over time: [P, N_t]
        thetaD = 2 * np.pi * fD * times[None, :]       # [P, N_t]

        # Per-path complex amplitude vs time (before frequency shaping): [P, N_t]
        a_pt = np.sqrt(power / self.total_subcarriers) * np.exp(1j * (phase0 + thetaD))  # [P, N_t]

        if self.OFDM_params[c.PARAMSET_OFDM_LPF]:
            # LPF delay shaping: lpf [P, N] then project to selected subcarriers [N, K] -> [P, K]
            lpf = np.sinc(self.delay_d[None, :] - delay_n)                                 # [P, N]
            h_pk = lpf @ self.delay_to_OFDM                                                # [P, K]
            # Broadcast time: [P, K, N_t]
            h_pkn = (a_pt[:, None, :]) * (h_pk[:, :, None])
        else:
            # Geometric per-subcarrier phase from delay: exp(-j 2π/N * delay_n * k)
            # delay_n: [P,1], subcarriers: [K] -> [P,K]
            delay_phase = np.exp(
                -1j * (2 * np.pi / self.total_subcarriers) * (delay_n @ self.subcarriers[None, :])
            )                                                                               # [P, K]
            # Broadcast time: [P, K, N_t]
            h_pkn = (a_pt[:, None, :]) * (delay_phase[:, :, None])

        return h_pkn.astype(np.complex64, copy=False)


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
        
        if freq_domain:  # OFDM
            path_gains = path_gen.generate(
                pwr=power, toa=delays_user, phs=phases_user, Ts=Ts, dopplers=dopplers_user
            ).T  # [K_sel, P]
            channel[i] = np.nansum(array_product[..., None, :] * path_gains[None, None, :, :], axis=-1)
        else:  # TD channel
            path_gains = np.sqrt(power) * np.exp(1j * (np.deg2rad(phases_user) + 2 * np.pi * dopplers_user))
            channel[i, :, :, :n_paths] = array_product * path_gains[None, None, :]

    return channel


# -------------------- V2 (vectorized time-aware) --------------------

def _generate_MIMO_channel_v2(
    array_response_product: np.ndarray,
    powers: np.ndarray,
    delays: np.ndarray,
    phases: np.ndarray,
    dopplers: np.ndarray,
    ofdm_params: Dict,
    times: float | np.ndarray = 0.0,
    freq_domain: bool = True,
    squeeze_time: bool = True,
) -> np.ndarray:
    """
    Generate MIMO channel matrices with a vectorized time dimension (correct Doppler progression).

    Inputs:
        array_response_product : [n_users, M_rx, M_tx, P_max]
        powers                 : [n_users, P_max]       (linear, antenna gains applied)
        delays                 : [n_users, P_max]       (seconds)
        phases                 : [n_users, P_max]       (degrees)
        dopplers               : [n_users, P_max]       (Hz)
        ofdm_params            : dict
        times                  : scalar or [N_t] in seconds
        freq_domain            : bool; if True -> per-subcarrier CFR, else time-domain per-path
        squeeze_time           : if True and N_t == 1, drop the time dim for backward compatibility

    Returns:
        If freq_domain:
            [n_users, M_rx, M_tx, K_sel, N_t]  (or [n_users, M_rx, M_tx, K_sel] if squeezed)
        Else (time-domain per-path gains):
            [n_users, M_rx, M_tx, P_max, N_t]  (or [n_users, M_rx, M_tx, P_max] if squeezed)
    """
    # Time handling
    times = np.atleast_1d(times).astype(float)  # [N_t]
    Nt = times.shape[0]

    Ts = 1.0 / ofdm_params[c.PARAMSET_OFDM_BANDWIDTH]
    subcarriers = ofdm_params[c.PARAMSET_OFDM_SC_SAMP]
    K = len(subcarriers)
    path_gen = OFDM_PathGenerator(ofdm_params, subcarriers)

    # Delay sanity for OFDM mode
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
            print("-" * 50)

    n_ues = powers.shape[0]
    P_max = powers.shape[1]
    M_rx, M_tx = array_response_product.shape[1:3]

    last_ch_dim = (K if freq_domain else P_max)

    # Allocate output
    if Nt == 1 and squeeze_time:
        channel = np.zeros((n_ues, M_rx, M_tx, last_ch_dim), dtype=np.csingle)
    else:
        channel = np.zeros((n_ues, M_rx, M_tx, last_ch_dim, Nt), dtype=np.csingle)

    # Masks per user (valid paths)
    nan_masks = ~np.isnan(powers)  # [n_users, P_max]
    valid_path_counts = np.sum(nan_masks, axis=1)  # [n_users]

    for i in tqdm(range(n_ues), desc='Generating channels (v2)'):
        non_nan_mask = nan_masks[i]
        n_paths = valid_path_counts[i]
        if n_paths == 0:
            continue

        # Slice per-user arrays
        array_product = array_response_product[i][..., non_nan_mask]  # [M_rx, M_tx, P]
        power_u   = powers[i, non_nan_mask]
        delays_u  = delays[i, non_nan_mask]
        phases_u  = phases[i, non_nan_mask]
        fD_u      = dopplers[i, non_nan_mask]

        if freq_domain:
            # path_gains: [P, K, N_t]
            path_gains = path_gen.generate_v2(
                pwr=power_u, toa=delays_u, phs=phases_u, Ts=Ts, dopplers=fD_u, times=times
            )  # complex64
            # Combine with array responses: [M_rx, M_tx, P] x [P, K, N_t] -> [M_rx, M_tx, K, N_t]
            Hi_fk_t = np.einsum('rtp,pkn->rtkn', array_product, path_gains, optimize=True).astype(np.complex64, copy=False)
            channel[i] = Hi_fk_t[..., 0] if Nt == 1 else Hi_fk_t
        else:
            # Time-domain per-path gains:
            # a_pt = sqrt(P) * exp(j(phi + 2π fD t))  -> [P, N_t]
            phase0 = np.deg2rad(phases_u)[:, None]                # [P,1]
            a_pt = np.sqrt(power_u)[:, None] * np.exp(1j * (phase0 + 2 * np.pi * fD_u[:, None] * times[None, :]))  # [P,N_t]
            # Combine: [M_rx,M_tx,P] x [P,N_t] -> [M_rx,M_tx,P,N_t]
            Hi_pt = np.einsum('rtp,pn->r t p n', array_product, a_pt, optimize=True).astype(np.complex64, copy=False)
            # Pad zeros for invalid paths to preserve last_ch_dim
            shape = (M_rx, M_tx, P_max) if Nt == 1 else (M_rx, M_tx, P_max, Nt)
            out = np.zeros(shape, dtype=np.complex64)
            out[..., :n_paths] = Hi_pt[..., 0] if Nt == 1 else Hi_pt
            channel[i] = out

    return channel