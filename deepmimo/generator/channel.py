"""Channel module for DeepMIMO.

This module provides functionality for MIMO channel generation, including:
- Channel parameter management through the ChannelParameters class
- OFDM path generation and verification
- Channel matrix computation

The main function is generate_mimo_channel() which generates MIMO channel matrices
based on path information from ray-tracing and antenna configurations.
"""

from copy import deepcopy
from typing import Any, ClassVar

import numpy as np
from tqdm import tqdm

from deepmimo import consts as c
from deepmimo.utils import DotDict, compare_two_dicts, deep_dict_merge

ROT_DIM = 3
RANGE_DIM = 2


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
    if isinstance(obj, dict):
        return {key: _convert_lists_to_arrays(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return np.array(obj)
    return obj


def _validate_ant_rot(rotation: np.ndarray, n_ues: int | None = None) -> np.ndarray:
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
    cond_1 = len(rotation_shape) == 1 and rotation_shape[0] == ROT_DIM  # Fixed 3D vector
    cond_2 = (
        len(rotation_shape) == RANGE_DIM
        and rotation_shape[0] == ROT_DIM
        and rotation_shape[1] == RANGE_DIM
    )  # Random ranges
    cond_3 = n_ues is not None and rotation_shape[0] == n_ues  # Per-user rotations

    assert_str = (
        "The antenna rotation must either be a 3D vector for "
        "constant values or 3 x 2 matrix for random values"
    )
    if n_ues is not None:
        assert_str += " or an n_ues x 3 matrix for per-user values"

    if not (cond_1 or cond_2 or (n_ues is not None and cond_3)):
        raise ValueError(assert_str)

    return rotation


def _validate_ant_rad_pat(pattern: str | None = None) -> str:
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

    assert_str = (
        "The antenna radiation pattern must have one of the "
        f"following values: {c.PARAMSET_ANT_RAD_PAT_VALS!s}"
    )
    if pattern not in c.PARAMSET_ANT_RAD_PAT_VALS:
        raise ValueError(assert_str)

    return pattern


class ChannelParameters(DotDict):
    """Class for managing channel generation parameters.

    This class provides an interface for setting and accessing various parameters
    needed for MIMO channel generation, including:
    - BS/UE antenna array configurations
    - OFDM parameters
    - Channel domain settings (time/frequency)

    The parameters can be accessed directly using dot notation (e.g. `params.bs_antenna.shape`)
    or using dictionary notation (e.g. `params['bs_antenna']['shape']`).

    Examples:
        **Default parameters**

        ```python
        params = ChannelParameters()
        ```

        **Specific parameters**

        ```python
        params = ChannelParameters(doppler=True, freq_domain=True)
        ```

        **Nested parameters** (lists are converted to numpy arrays during validation)

        ```python
        params = ChannelParameters(bs_antenna={"shape": [4, 4]})
        ```

    """

    # Default channel generation parameters
    DEFAULT_PARAMS: ClassVar[dict[str, Any]] = {
        # BS Antenna Parameters
        c.PARAMSET_ANT_BS: {
            c.PARAMSET_ANT_SHAPE: np.array([8, 1]),  # Antenna dimensions in X - Y - Z
            c.PARAMSET_ANT_SPACING: 0.5,
            c.PARAMSET_ANT_ROTATION: np.array([0, 0, 0]),  # Rotation around X - Y - Z axes
            c.PARAMSET_ANT_RAD_PAT: c.PARAMSET_ANT_RAD_PAT_VALS[0],  # 'isotropic'
        },
        # UE Antenna Parameters
        c.PARAMSET_ANT_UE: {
            c.PARAMSET_ANT_SHAPE: np.array([1, 1]),  # Antenna dimensions in X - Y - Z
            c.PARAMSET_ANT_SPACING: 0.5,
            c.PARAMSET_ANT_ROTATION: np.array([0, 0, 0]),  # Rotation around X - Y - Z axes
            c.PARAMSET_ANT_RAD_PAT: c.PARAMSET_ANT_RAD_PAT_VALS[0],  # 'isotropic'
        },
        c.PARAMSET_DOPPLER_EN: 0,
        c.PARAMSET_NUM_PATHS: c.MAX_PATHS,
        c.PARAMSET_FD_CH: 1,  # OFDM channel if 1, Time domain if 0
        # OFDM Parameters
        c.PARAMSET_OFDM: {
            c.PARAMSET_OFDM_SC_NUM: 512,  # Number of total subcarriers
            c.PARAMSET_OFDM_SC_SAMP: np.arange(1),  # Select subcarriers to generate
            c.PARAMSET_OFDM_BANDWIDTH: 10e6,  # Hz
            c.PARAMSET_OFDM_LPF: 0,  # Receive Low Pass / ADC Filter
        },
    }

    def __init__(self, data: dict | None = None, **kwargs: Any) -> None:
        """Initialize channel generation parameters.

        Args:
            data: Optional dictionary containing channel parameters to override defaults
            **kwargs: Additional parameters to override defaults.
                These will be merged with data if provided.
                For nested parameters, provide as dicts (e.g. bs_antenna={'shape': [4,4]}).
                Only specified fields are overridden; other fields keep default values.
                Lists are converted to numpy arrays during validation.

        """
        # Initialize with deep copy of defaults
        super().__init__(deepcopy(self.DEFAULT_PARAMS))

        # Update with provided data if any
        if data is not None:
            self.update(deep_dict_merge(self, data))

        # Update with kwargs if any provided
        if kwargs:
            self.update(deep_dict_merge(self, kwargs))

    def _validate_antenna_params(self, params: DotDict, n_ues: int | None = None) -> None:
        """Validate and set antenna rotation and radiation pattern for BS or UE.

        Args:
            params: Antenna parameters dict (bs_antenna or ue_antenna)
            n_ues: Number of UEs (only needed for UE rotation validation)

        """
        # Validate rotation
        if c.PARAMSET_ANT_ROTATION in params:
            params[c.PARAMSET_ANT_ROTATION] = _validate_ant_rot(
                params[c.PARAMSET_ANT_ROTATION], n_ues
            )
        else:
            params[c.PARAMSET_ANT_ROTATION] = np.array([0, 0, 0])

        # Validate radiation pattern
        if c.PARAMSET_ANT_RAD_PAT in params:
            params[c.PARAMSET_ANT_RAD_PAT] = _validate_ant_rad_pat(params[c.PARAMSET_ANT_RAD_PAT])
        else:
            params[c.PARAMSET_ANT_RAD_PAT] = c.PARAMSET_ANT_RAD_PAT_VALS[0]

    def _validate_ofdm_subcarriers(self, ofdm_params: dict) -> None:
        """Validate OFDM subcarrier selection parameters.

        Args:
            ofdm_params: OFDM parameters dictionary

        Raises:
            ValueError: If subcarrier parameters are invalid

        """
        if c.PARAMSET_OFDM_SC_SAMP not in ofdm_params or c.PARAMSET_OFDM_SC_NUM not in ofdm_params:
            return

        sc_sel = np.asarray(ofdm_params[c.PARAMSET_OFDM_SC_SAMP])

        if sc_sel.ndim > 1:
            msg = f"'{c.PARAMSET_OFDM_SC_SAMP}' must be a 1-D array"
            raise ValueError(msg)

        if sc_sel.size == 0:
            msg = f"'{c.PARAMSET_OFDM_SC_SAMP}' must be a non-empty array"
            raise ValueError(msg)

        if not np.issubdtype(sc_sel.dtype, np.integer):
            try:
                sc_sel = sc_sel.astype(int, copy=False)
                ofdm_params[c.PARAMSET_OFDM_SC_SAMP] = sc_sel
            except Exception as err:
                msg = (
                    f"'{c.PARAMSET_OFDM_SC_SAMP}' must contain integer indices "
                    "(invalid values provided)"
                )
                raise ValueError(msg) from err

        n_sc = ofdm_params[c.PARAMSET_OFDM_SC_NUM]
        if np.any(sc_sel < 0) or np.any(sc_sel >= n_sc):
            error_msg = f"'{c.PARAMSET_OFDM_SC_SAMP}' values must be within [0, {n_sc - 1}]\n"
            error_msg += f"Got max value: {np.max(sc_sel)}.\n"
            error_msg += f"Adjust ch_params.{c.PARAMSET_OFDM}.{c.PARAMSET_OFDM_SC_SAMP} or "
            error_msg += f"ch_params.{c.PARAMSET_OFDM}.{c.PARAMSET_OFDM_SC_NUM}."
            raise ValueError(error_msg)

    def validate(self, n_ues: int) -> "ChannelParameters":
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
        self_converted = _convert_lists_to_arrays(self)

        # Notify the user if some keyword is not used (likely set incorrectly)
        additional_keys = compare_two_dicts(self_converted, ChannelParameters())
        if len(additional_keys):
            print("The following parameters seem unnecessary:")
            print(additional_keys)

        # Validate BS antenna parameters
        self._validate_antenna_params(self_converted[c.PARAMSET_ANT_BS])

        # Validate UE antenna parameters
        self._validate_antenna_params(self_converted[c.PARAMSET_ANT_UE], n_ues)

        # Validate OFDM parameters
        if c.PARAMSET_OFDM in self_converted:
            self._validate_ofdm_subcarriers(self_converted[c.PARAMSET_OFDM])

        return self_converted


class OFDMPathGenerator:
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

    def __init__(self, params: dict, subcarriers: np.ndarray) -> None:
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
            -1j * 2 * np.pi / self.total_subcarriers * np.outer(self.delay_d, self.subcarriers),
        )

    def generate(  # noqa: PLR0913
        self,
        pwr: np.ndarray,
        toa: np.ndarray,
        phs: np.ndarray,
        ts: float,
        dopplers: np.ndarray,
        times: float | np.ndarray,
    ) -> np.ndarray:
        """Generate per-path, per-subcarrier, per-time path gains with correct Doppler progression.

        Computes: h[p,k,n] = √(P/N) · exp(j(φ₀ + 2πfD·t)) · exp(-j2πτk/N)
        where N=total subcarriers, k=subcarrier index, τ=delay (in samples), fD=Doppler

        Inputs:
            pwr       : [P]       linear powers per path
            toa       : [P]       times of arrival (seconds)
            phs       : [P]       initial phases (degrees)
            ts        : scalar    sampling period (seconds)
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

        times = np.atleast_1d(times).astype(float)  # [N_t]

        # Base dimensions
        power = pwr[:, None]  # [P, 1]
        delay_n = (toa / ts)[:, None]  # [P, 1] (sample units)
        phase0 = np.deg2rad(phs)[:, None]  # [P, 1] (radians)
        fd = dopplers[:, None]  # [P, 1] (Hz)

        # Ignore paths over FFT (clip to zero)
        over = delay_n >= self.OFDM_params[c.PARAMSET_OFDM_SC_NUM]
        if np.any(over):
            power[over] = 0.0
            delay_n[over] = self.OFDM_params[c.PARAMSET_OFDM_SC_NUM]
            fd[over] = 0.0

        # Doppler-induced phase over time: [P, N_t]
        theta_d = 2 * np.pi * fd * times[None, :]  # [P, N_t]

        # Per-path complex amplitude vs time (before frequency shaping): [P, N_t]
        a_pt = np.sqrt(power / self.total_subcarriers) * np.exp(
            1j * (phase0 + theta_d),
        )  # [P, N_t]

        if self.OFDM_params[c.PARAMSET_OFDM_LPF]:
            # LPF delay shaping: lpf [P, N] then project to selected subcarriers [N, K] -> [P, K]
            lpf = np.sinc(self.delay_d[None, :] - delay_n)  # [P, N]
            h_pk = lpf @ self.delay_to_OFDM  # [P, K]
            # Broadcast time: [P, K, N_t]
            h_pkn = (a_pt[:, None, :]) * (h_pk[:, :, None])
        else:
            # Geometric per-subcarrier phase from delay: exp(-j 2π/N * delay_n * k)
            # delay_n: [P,1], subcarriers: [K] -> [P,K]
            delay_phase = np.exp(
                -1j * (2 * np.pi / self.total_subcarriers) * (delay_n @ self.subcarriers[None, :]),
            )  # [P, K]
            # Broadcast time: [P, K, N_t]
            h_pkn = (a_pt[:, None, :]) * (delay_phase[:, :, None])

        return h_pkn.astype(np.complex64, copy=False)


def _check_ofdm_compatibility(ofdm_params: dict, delays: np.ndarray) -> None:
    """Check if path delays are compatible with OFDM symbol duration."""
    ts = 1.0 / ofdm_params[c.PARAMSET_OFDM_BANDWIDTH]
    ofdm_symbol_duration = ofdm_params[c.PARAMSET_OFDM_SC_NUM] * ts
    subcarrier_spacing = (
        ofdm_params[c.PARAMSET_OFDM_BANDWIDTH] / ofdm_params[c.PARAMSET_OFDM_SC_NUM]
    )  # Hz
    max_delay = np.nanmax(delays)

    if max_delay <= ofdm_symbol_duration:
        return

    print("\nWarning: Some path delays exceed OFDM symbol duration")
    print("-" * 50)
    print("OFDM Configuration:")
    print(f"- Number of subcarriers (N): {ofdm_params[c.PARAMSET_OFDM_SC_NUM]}")
    print(f"- Bandwidth (B): {ofdm_params[c.PARAMSET_OFDM_BANDWIDTH] / 1e6:.1f} MHz")
    print(f"- Subcarrier spacing (Δf = B/N): {subcarrier_spacing / 1e3:.1f} kHz")
    print(f"- Symbol duration (T = 1/Δf = N/B): {ofdm_symbol_duration * 1e6:.1f} μs")
    print("\nPath Information:")
    print(f"- Maximum path delay: {max_delay * 1e6:.1f} μs")
    print(f"- Excess delay: {(max_delay - ofdm_symbol_duration) * 1e6:.1f} μs")
    print("\nPaths arriving after the symbol duration will be clipped.")
    print("To avoid clipping, either:")
    print("1. Increase the number of subcarriers (N)")
    print("2. Decrease the bandwidth (B)")
    print(
        f"3. Switch to time-domain channel generation (set ch_params['{c.PARAMSET_FD_CH}'] = 0)",
    )
    print("-" * 50)


def _compute_single_freq_channel(
    array_product: np.ndarray,
    path_gains: np.ndarray,
    *,
    squeeze_time: bool,
) -> np.ndarray:
    """Compute frequency-domain channel for a single link.

    Args:
        array_product: [M_rx, M_tx, P] antenna array responses for valid paths
        path_gains: [P, K, N_t] per-path gains for each subcarrier and time
        squeeze_time: If True and single time sample, squeeze time dimension

    Returns:
        [M_rx, M_tx, K] if squeezed, else [M_rx, M_tx, K, N_t]

    """
    # Combine with array responses: [M_rx, M_tx, P] x [P, K, N_t] -> [M_rx, M_tx, K, N_t]
    channel = np.einsum("rtp,pkn->rtkn", array_product, path_gains, optimize=True).astype(
        np.complex64,
        copy=False,
    )
    # Squeeze time dimension if single snapshot
    if squeeze_time and channel.shape[-1] == 1:
        return channel[..., 0]
    return channel


def _compute_single_time_channel(
    array_product: np.ndarray,
    path_gains: np.ndarray,
    p_max: int,
    *,
    squeeze_time: bool,
) -> np.ndarray:
    """Compute time-domain channel for a single link.

    Args:
        array_product: [M_rx, M_tx, P] antenna array responses for valid paths
        path_gains: [P, N_t] per-path complex gains over time
        p_max: Maximum number of paths (for zero-padding)
        squeeze_time: If True and single time sample, squeeze time dimension

    Returns:
        [M_rx, M_tx, P_max] if squeezed, else [M_rx, M_tx, P_max, N_t]

    """
    m_rx, m_tx, n_paths = array_product.shape
    n_times = path_gains.shape[1]

    # Combine: [M_rx, M_tx, P] x [P, N_t] -> [M_rx, M_tx, P, N_t]
    channel = np.einsum("rtp,pn->rtpn", array_product, path_gains, optimize=True).astype(
        np.complex64,
        copy=False,
    )

    # Pad zeros for invalid paths to preserve p_max dimension
    if squeeze_time and n_times == 1:
        # Output: [M_rx, M_tx, P_max]
        out = np.zeros((m_rx, m_tx, p_max), dtype=np.complex64)
        out[..., :n_paths] = channel[..., 0]
    else:
        # Output: [M_rx, M_tx, P_max, N_t]
        out = np.zeros((m_rx, m_tx, p_max, n_times), dtype=np.complex64)
        out[..., :n_paths, :] = channel

    return out


def _generate_mimo_channel(  # noqa: PLR0913 - explicit path arrays preferred for clarity
    array_response_product: np.ndarray,
    power: np.ndarray,
    delay: np.ndarray,
    phase: np.ndarray,
    doppler: np.ndarray,
    ofdm_params: dict,
    *,
    times: float | np.ndarray = 0.0,
    freq_domain: bool = True,
    squeeze_time: bool = True,
) -> np.ndarray:
    """Generate MIMO channel matrices in frequency or time domain.

    Args:
        array_response_product: [n_users, M_rx, M_tx, P_max] antenna array responses
        power: [n_users, P_max] path powers (linear scale)
        delay: [n_users, P_max] path delays (seconds)
        phase: [n_users, P_max] path phases (degrees)
        doppler: [n_users, P_max] Doppler frequencies (Hz)
        ofdm_params: OFDM parameters dictionary
        times: Time samples (scalar or array, in seconds)
        freq_domain: If True, generate frequency-domain (OFDM) channel
        squeeze_time: If True and single time sample, squeeze time dimension

    Returns:
        Channel matrix with shape depending on domain and time:
        - Freq domain, single time: [n_users, M_rx, M_tx, K_subcarriers]
        - Freq domain, multi time: [n_users, M_rx, M_tx, K_subcarriers, N_t]
        - Time domain, single time: [n_users, M_rx, M_tx, P_max]
        - Time domain, multi time: [n_users, M_rx, M_tx, P_max, N_t]

    """
    # Time handling
    times_arr = np.atleast_1d(times).astype(float)  # [N_t]
    n_times = times_arr.shape[0]

    ts = 1.0 / ofdm_params[c.PARAMSET_OFDM_BANDWIDTH]
    subcarriers = ofdm_params[c.PARAMSET_OFDM_SC_SAMP]
    k_subcarriers = len(subcarriers)
    path_gen = OFDMPathGenerator(ofdm_params, subcarriers)

    # Delay sanity for OFDM mode
    if freq_domain:
        _check_ofdm_compatibility(ofdm_params, delay)

    n_ues = power.shape[0]
    p_max = power.shape[1]
    m_rx, m_tx = array_response_product.shape[1:3]

    last_ch_dim = k_subcarriers if freq_domain else p_max

    # Allocate output
    if n_times == 1 and squeeze_time:
        channel = np.zeros((n_ues, m_rx, m_tx, last_ch_dim), dtype=np.csingle)
    else:
        channel = np.zeros((n_ues, m_rx, m_tx, last_ch_dim, n_times), dtype=np.csingle)

    # Masks per user (valid paths)
    nan_masks = ~np.isnan(power)  # [n_users, P_max]
    valid_path_counts = np.sum(nan_masks, axis=1)  # [n_users]

    for i in tqdm(range(n_ues), desc="Generating channels"):
        non_nan_mask = nan_masks[i]
        n_paths = valid_path_counts[i]
        if n_paths == 0:
            continue

        # Slice per-user arrays
        array_product = array_response_product[i][..., non_nan_mask]  # [M_rx, M_tx, P]

        # Per-user path data
        user_power = power[i, non_nan_mask]
        user_delay = delay[i, non_nan_mask]
        user_phase = phase[i, non_nan_mask]
        user_doppler = doppler[i, non_nan_mask]

        if freq_domain:
            # Generate OFDM path gains: h[p,k,n] = √(P/N) · exp(j(φ₀ + 2πfD·t)) · exp(-j2πτk/N)
            # where N=total subcarriers, k=subcarrier index, τ=delay, fD=Doppler
            # Delays create per-subcarrier phase shifts (frequency-dependent)
            path_gains = path_gen.generate(
                pwr=user_power,
                toa=user_delay,
                phs=user_phase,
                ts=ts,
                dopplers=user_doppler,
                times=times_arr,
            )
            channel[i] = _compute_single_freq_channel(
                array_product, path_gains, squeeze_time=squeeze_time
            )
        else:
            # Generate time-domain path gains: a[p,n] = √P · exp(j(φ₀ + 2πfD·t))
            # Delays are represented structurally: each path index p has delay[p]
            phase0 = np.deg2rad(user_phase)[:, None]  # [P, 1]
            path_gains = np.sqrt(user_power)[:, None] * np.exp(
                1j * (phase0 + 2 * np.pi * user_doppler[:, None] * times_arr[None, :]),
            )  # [P, N_t]
            channel[i] = _compute_single_time_channel(
                array_product, path_gains, p_max, squeeze_time=squeeze_time
            )

    return channel
