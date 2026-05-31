"""Channel module for DeepMIMO.

This module provides functionality for MIMO channel generation, including:
- Channel parameter management through the ChannelParameters class
- OFDM path generation and verification
- Channel matrix computation

The main function is generate_mimo_channel() which generates MIMO channel matrices
based on path information from ray-tracing and antenna configurations.
"""

from __future__ import annotations

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

    def validate(self, n_ues: int) -> ChannelParameters:
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
        # float32 view of the selected subcarriers: keeps the batched delay phase (the dominant
        # frequency-domain intermediate) in complex64 instead of upcasting to complex128.
        self.subcarriers_f32 = np.asarray(subcarriers, dtype=np.float32)
        # Arithmetic-spacing of the selected subcarriers (sc[j] = start + step*j). When uniform
        # (the common arange / evenly-subsampled case) the per-subcarrier delay phase factorizes,
        # letting the batched frequency path replace the K transcendental exp() per path with a
        # cheap cumulative product (see _uniform_delay_response). Subcarriers are integer indices,
        # so exact equality of the successive differences detects spacing without false positives;
        # any non-uniform set falls back to the direct exp.
        sc_arr = np.asarray(subcarriers)
        sc_diffs = np.diff(sc_arr)
        self.subcarriers_uniform = bool(sc_diffs.size == 0 or np.all(sc_diffs == sc_diffs[0]))
        self.subcarriers_step = float(sc_diffs[0]) if sc_diffs.size else 1.0
        self.subcarriers_start = float(sc_arr[0]) if sc_arr.size else 0.0
        # The [N, K_sel] delay->OFDM transform is only used by the LPF branch. Build it lazily so
        # it is never materialized (an [N, K_sel] complex128 exp) when LPF is off, the common case.
        self._delay_to_OFDM: np.ndarray | None = None

    @property
    def delay_to_OFDM(self) -> np.ndarray:  # noqa: N802 - established attribute name (OFDM acronym)
        """[N, K_sel] delay-to-OFDM transform, built on first use (LPF branch only)."""
        if self._delay_to_OFDM is None:
            self._delay_to_OFDM = np.exp(
                -1j * 2 * np.pi / self.total_subcarriers * np.outer(self.delay_d, self.subcarriers),
            )
        return self._delay_to_OFDM

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

    def generate_batch(  # noqa: PLR0913
        self,
        pwr: np.ndarray,
        toa: np.ndarray,
        phs: np.ndarray,
        ts: float,
        dopplers: np.ndarray,
        times: float | np.ndarray,
    ) -> np.ndarray:
        """Vectorized OFDM path gains for a batch (chunk) of links.

        Batched counterpart of :meth:`generate`: identical math applied to ``B`` links at
        once. Padded/invalid paths MUST already be zeroed by the caller (power, delay,
        phase and Doppler set to 0) so that they yield exactly zero path gain.

        Inputs:
            pwr       : [B, P]    linear powers per path (0 on padded paths)
            toa       : [B, P]    times of arrival (seconds; 0 on padded paths)
            phs       : [B, P]    initial phases (degrees; 0 on padded paths)
            ts        : scalar    sampling period (seconds)
            dopplers  : [B, P]    Doppler frequencies (Hz; 0 on padded paths)
            times     : scalar or [N_t] times (seconds) at which to sample

        Returns:
            h_pbkn    : [B, P, K_sel, N_t] complex64 path gains.

        """
        # h[b, p, k, n] = amplitude[b, p, n] * delay_response[b, p, k]
        amp, delay_resp = _freq_path_factors(self, pwr, toa, phs, dopplers, ts, times)
        h_pbkn = amp[:, :, None, :] * delay_resp[:, :, :, None]  # [B, P, K, N_t]
        return h_pbkn.astype(np.complex64, copy=False)


# Subcarrier-recurrence block length: exact exp() anchors are recomputed every _SC_RECURRENCE_BLOCK
# subcarriers so the complex64 cumulative product's drift stays far below the channel's rtol=1e-4
# (drift ~ block * 2**-24 ~ 4e-6, independent of the subcarrier count).
_SC_RECURRENCE_BLOCK = 64


def _uniform_delay_response(path_gen: OFDMPathGenerator, delay_n: np.ndarray) -> np.ndarray:
    """Build the [B, P, K] delay response for arithmetic (evenly spaced) subcarriers.

    For ``sc[j] = start + step*j`` the per-subcarrier phase factorizes as
    ``exp(-j 2*pi/N * delay_n * sc[j]) = base0 * q**j``, with the step
    ``q = exp(-j 2*pi/N * delay_n * step)``. The K transcendental ``exp()`` evaluations per path
    therefore collapse to a cumulative product of cheap complex multiplies. To stay in complex64
    without letting the cumulative product's magnitude drift away from the unit circle, exact
    ``exp()`` anchors are recomputed every ``_SC_RECURRENCE_BLOCK`` subcarriers and combined
    with a single in-block ramp via one broadcast multiply. This matches the direct ``exp()`` to
    well within the channel's rtol=1e-4 (it is in fact closer to the exact value).

    Args:
        path_gen: OFDM path generator (supplies the subcarrier start/step and total count).
        delay_n: [B, P] path delays in sample units (already clipped for over-FFT paths).

    Returns:
        [B, P, K] complex64 delay response.

    """
    b, p = delay_n.shape
    k_sel = path_gen.subcarriers_f32.shape[0]
    start = path_gen.subcarriers_start
    step = path_gen.subcarriers_step
    block = min(_SC_RECURRENCE_BLOCK, k_sel)
    n_blk = -(-k_sel // block)  # ceil(k_sel / block)

    # Per-path angular increment per unit subcarrier index, kept in float64 for accurate anchors.
    w = (-2.0 * np.pi / path_gen.total_subcarriers) * delay_n  # [B, P]

    # In-block ramp q**l, l in [0, block): a complex64 cumprod whose drift is bounded by `block`.
    q = np.exp(1j * (w * step)).astype(np.complex64)  # [B, P]
    ramp = np.empty((b, p, block), dtype=np.complex64)
    ramp[:, :, 0] = 1.0
    ramp[:, :, 1:] = q[:, :, None]
    np.cumprod(ramp, axis=2, out=ramp)  # [B, P, block]

    # Exact per-block anchors exp(-j 2*pi/N * delay_n * (start + step*block*m)), built in float64.
    m = np.arange(n_blk, dtype=np.float64)
    anchors = np.exp(1j * (w[:, :, None] * (start + step * block * m)[None, None, :])).astype(
        np.complex64,
    )  # [B, P, n_blk]

    # anchor[m] * ramp[l] tiles the full response; reshape the (block, in-block) grid and trim to K.
    resp = (anchors[:, :, :, None] * ramp[:, :, None, :]).reshape(b, p, n_blk * block)
    return resp[:, :, :k_sel]


def _freq_path_factors(  # noqa: PLR0913
    path_gen: OFDMPathGenerator,
    power: np.ndarray,
    delay: np.ndarray,
    phase: np.ndarray,
    doppler: np.ndarray,
    ts: float,
    times: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Factor the OFDM path gains of a chunk into amplitude and delay-response parts.

    The OFDM path gain factorizes as ``h[b, p, k, n] = amp[b, p, n] * delay_resp[b, p, k]``:

    - ``amp``        : [B, P, N_t] ``sqrt(P/N) * exp(j(phi0 + 2*pi*fD*t))`` (power/phase/Doppler)
    - ``delay_resp`` : [B, P, K]   ``exp(-j 2*pi*tau_n*k/N)`` (or the sinc-LPF projection)

    Keeping the two factors separate lets a single-time caller fold the per-path amplitude into
    the smaller matmul operand, avoiding the [B, P, K, N_t] gains intermediate. The heavy
    [B, P, K] delay response is built directly in float32/complex64 so the frequency-domain path
    never upcasts to complex128. Padded/over-FFT paths are clipped to zero amplitude, matching
    :meth:`OFDMPathGenerator.generate`.

    Returns:
        (amp [B, P, N_t] complex64, delay_resp [B, P, K] complex64)

    """
    n_sc = path_gen.total_subcarriers
    times_arr = np.atleast_1d(times).astype(np.float64)  # [N_t]

    # Per-path scalars are kept in float64 (cheap [B, P] arrays) for full phase accuracy.
    power = np.asarray(power, dtype=np.float64)  # [B, P]
    delay_n = np.asarray(delay, dtype=np.float64) / ts  # [B, P] (sample units)
    phase0 = np.deg2rad(np.asarray(phase, dtype=np.float64))  # [B, P]
    fd = np.asarray(doppler, dtype=np.float64)  # [B, P]

    # Ignore paths over the FFT window (clip to zero), matching generate().
    over = delay_n >= n_sc
    if np.any(over):
        power = np.where(over, 0.0, power)
        delay_n = np.where(over, float(n_sc), delay_n)
        fd = np.where(over, 0.0, fd)

    # Amplitude [B, P, N_t]: build the (small) phase argument in float64, cast to float32 at the
    # exp so the complex result is complex64.
    theta = phase0[:, :, None] + (2.0 * np.pi) * fd[:, :, None] * times_arr[None, None, :]
    amp = np.sqrt(power / n_sc)[:, :, None].astype(np.float32) * np.exp(
        1j * theta.astype(np.float32),
    )  # complex64

    if path_gen.OFDM_params[c.PARAMSET_OFDM_LPF]:
        # LPF delay shaping: lpf [B, P, N] projected onto the selected subcarriers -> [B, P, K].
        lpf = np.sinc(path_gen.delay_d[None, None, :] - delay_n[:, :, None])  # [B, P, N]
        delay_resp = (lpf @ path_gen.delay_to_OFDM).astype(np.complex64)  # [B, P, K]
    elif path_gen.subcarriers_uniform:
        # Geometric per-subcarrier phase from delay: exp(-j 2*pi/N * delay_n * k). For arithmetic
        # subcarriers this factorizes, so a cumulative product replaces the K transcendental exp()
        # per path (the dominant frequency-domain cost) while staying in complex64.
        delay_resp = _uniform_delay_response(path_gen, delay_n)
    else:
        # Non-uniform subcarriers: build the large [B, P, K] phase argument in float32 (keeps the
        # exp in complex64) and exponentiate directly.
        ang = (
            np.float32(-2.0 * np.pi / n_sc)
            * delay_n[:, :, None].astype(np.float32)
            * path_gen.subcarriers_f32[None, None, :]
        )  # [B, P, K] float32
        delay_resp = np.exp(1j * ang)  # complex64

    return amp, delay_resp


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


def _compute_freq_channel_batch(  # noqa: PLR0913
    array_product: np.ndarray,
    path_gen: OFDMPathGenerator,
    power: np.ndarray,
    delay: np.ndarray,
    phase: np.ndarray,
    doppler: np.ndarray,
    ts: float,
    times: np.ndarray,
    *,
    squeeze_time: bool,
) -> np.ndarray:
    """Combine array responses with OFDM path gains for a batch (chunk) of links.

    The channel is the path-summed product of the antenna array response and the OFDM path
    gains, ``H[b, r, t, k, n] = sum_p arp[b, r, t, p] * amp[b, p, n] * delay_resp[b, p, k]``.
    The path sum is a BLAS matmul over the path axis: reshaping ``arp`` to [B, M_rx*M_tx, P]
    and contracting with the [B, P, K(*N_t)] path gains gives [B, M_rx*M_tx, K(*N_t)], which is
    reshaped back. Padded paths carry zero amplitude, so summing over all ``P`` paths equals the
    sum over valid paths only.

    For a single time sample (the common case) the per-path amplitude is folded into the smaller
    of the two matmul operands, so the [B, P, K] gains intermediate is never materialized.

    Args:
        array_product: [B, M_rx, M_tx, P] antenna array responses (finite; padded paths may hold
            arbitrary finite values, they are nulled by the zero amplitudes).
        path_gen: OFDM path generator describing the subcarriers / OFDM parameters.
        power: [B, P] linear path powers (0 on padded paths).
        delay: [B, P] path delays in seconds (0 on padded paths).
        phase: [B, P] path phases in degrees (0 on padded paths).
        doppler: [B, P] Doppler frequencies in Hz (0 on padded paths).
        ts: sampling period (seconds).
        times: [N_t] time samples in seconds.
        squeeze_time: If True and single time sample, squeeze the time dimension.

    Returns:
        [B, M_rx, M_tx, K] if squeezed, else [B, M_rx, M_tx, K, N_t].

    """
    b, m_rx, m_tx, n_paths = array_product.shape
    m_ant = m_rx * m_tx
    n_times = np.atleast_1d(times).shape[0]
    a2 = array_product.reshape(b, m_ant, n_paths)

    if n_times == 1:
        amp, delay_resp = _freq_path_factors(path_gen, power, delay, phase, doppler, ts, times)
        k_sel = delay_resp.shape[2]
        amp1 = amp[:, :, 0]  # [B, P]
        # Fold the per-path amplitude into the smaller operand, then contract paths with one
        # matmul: [B, M_ant, P] @ [B, P, K] -> [B, M_ant, K]. No [B, P, K] gains array is built.
        if m_ant <= k_sel:
            channel = np.matmul(a2 * amp1[:, None, :], delay_resp)
        else:
            channel = np.matmul(a2, delay_resp * amp1[:, :, None])
        channel = channel.reshape(b, m_rx, m_tx, k_sel).astype(np.complex64, copy=False)
        return channel if squeeze_time else channel[..., None]

    # Multiple time samples: build the full [B, P, K, N_t] gains and contract the path axis.
    gains = path_gen.generate_batch(
        pwr=power,
        toa=delay,
        phs=phase,
        ts=ts,
        dopplers=doppler,
        times=times,
    )
    k_sel = gains.shape[2]
    g2 = gains.reshape(b, n_paths, k_sel * n_times)
    channel = np.matmul(a2, g2).reshape(b, m_rx, m_tx, k_sel, n_times)
    return channel.astype(np.complex64, copy=False)


def _compute_time_channel_batch(  # noqa: PLR0913
    array_product: np.ndarray,
    valid_mask: np.ndarray,
    power: np.ndarray,
    phase: np.ndarray,
    doppler: np.ndarray,
    times: np.ndarray,
    *,
    squeeze_time: bool,
) -> np.ndarray:
    """Compute the time-domain channel for a batch (chunk) of links.

    Time-domain path gains are ``a[p, n] = sqrt(P) * exp(j(phi0 + 2*pi*fD*t))`` and the
    delay is represented structurally by the path index. Valid paths are kept compacted at
    the front of the path axis (matching the per-user reference); padded paths become
    zeros. Real datasets are front-loaded, so compaction is skipped unless padding is
    actually interspersed.

    Args:
        array_product: [B, M_rx, M_tx, P] antenna array responses.
        valid_mask: [B, P] boolean mask of valid (non-padded) paths.
        power: [B, P] linear path powers (0 on padded paths).
        phase: [B, P] path phases in degrees (0 on padded paths).
        doppler: [B, P] Doppler frequencies in Hz (0 on padded paths).
        times: [N_t] time samples in seconds.
        squeeze_time: If True and single time sample, squeeze the time dimension.

    Returns:
        [B, M_rx, M_tx, P] if squeezed, else [B, M_rx, M_tx, P, N_t].

    """
    n_times = times.shape[0]

    # Per-path complex gains over time at original path positions ([B, P, N_t]). Padded paths are
    # zero because their power was zeroed by the caller. The phase argument is built in float64 (a
    # small [B, P, N_t] array, no antenna axis) then cast to float32 at the exp so the gains - and
    # therefore the dominant [B, M_rx, M_tx, P, N_t] outer product below - stay complex64. Building
    # them in complex128 (the previous behavior) doubled this path's memory traffic and made the
    # batched outer product lose to the per-UE loop at large antenna counts (M >= 32/side).
    theta = np.deg2rad(phase)[:, :, None] + (2 * np.pi) * doppler[:, :, None] * times[None, None, :]
    gains = np.sqrt(power).astype(np.float32)[:, :, None] * np.exp(1j * theta.astype(np.float32))

    array_resp = array_product
    # Only reorder when padding is interspersed (False followed by True in the mask).
    needs_compaction = not bool(np.all(valid_mask[:, :-1] >= valid_mask[:, 1:]))
    if needs_compaction:
        order = np.argsort(~valid_mask, axis=1, kind="stable")  # valid first, in order
        gains = np.take_along_axis(gains, order[:, :, None], axis=1)
        array_resp = np.take_along_axis(array_product, order[:, None, None, :], axis=3)

    # Outer product keeping the path axis: [B, M_rx, M_tx, P, N_t] (complex64).
    channel = (array_resp[:, :, :, :, None] * gains[:, None, None, :, :]).astype(
        np.complex64,
        copy=False,
    )

    if squeeze_time and n_times == 1:
        return channel[..., 0]
    return channel


# Target peak working-memory budget (bytes) for a single batched UE chunk. This bounds the
# temporaries created per chunk; the full output array is always allocated in addition.
_CHUNK_MEMORY_BUDGET_BYTES = 512 * 1024 * 1024  # 512 MiB

# When users are processed in valid-path-count order (frequency domain), aim for this many
# chunks: enough that each chunk spans a narrow count range and trims to a small per-chunk path
# count, while keeping per-chunk Python/BLAS-dispatch overhead negligible.
_SORTED_TARGET_CHUNKS = 16

# Minimum spread (max - min) in per-user valid-path counts before count-sorting + per-chunk
# trimming is worthwhile; below this the gather/scatter cost would not be recovered.
_MIN_COUNT_SPREAD_FOR_SORT = 2


def _estimate_chunk_size(  # noqa: PLR0913
    *,
    freq_domain: bool,
    lpf: bool,
    m_rx: int,
    m_tx: int,
    p_max: int,
    k_sel: int,
    n_times: int,
    total_subcarriers: int,
    materialize_product: bool = False,
    budget_bytes: int = _CHUNK_MEMORY_BUDGET_BYTES,
) -> int:
    """Estimate a UE-chunk size that keeps peak working memory near ``budget_bytes``.

    Accounts for the dominant per-chunk temporaries of the batched computation. The
    frequency-domain path is kept entirely in complex64 (float32 phase arguments), so the
    budget reflects 8-byte (not 16-byte) intermediates and therefore does not over-shrink
    chunks for large K or M. When ``materialize_product`` is True the per-chunk [M_rx, M_tx, P]
    outer product of the separate RX/TX responses (complex64) is also budgeted for. Always
    returns at least 1.
    """
    bytes_c64 = 8  # complex64
    bytes_f32 = 4  # float32 (phase argument of the delay response)
    m_ant = m_rx * m_tx
    if freq_domain:
        per_user = (
            p_max * k_sel * (bytes_c64 + bytes_f32)  # delay response (c64) + phase argument (f32)
            + p_max * n_times * bytes_c64  # amplitude (c64)
            + m_ant * k_sel * n_times * bytes_c64  # matmul result (c64)
        )
        if n_times > 1:
            per_user += p_max * k_sel * n_times * bytes_c64  # full [B, P, K, N_t] gains (c64)
        if lpf:
            per_user += p_max * total_subcarriers * bytes_c64  # sinc kernel + LPF projection
    else:
        per_user = (
            m_ant * p_max * n_times * (2 * bytes_c64)  # outer product (c128 build + c64 result)
            + m_ant * p_max * bytes_c64  # gathered array response
        )
    if materialize_product:
        per_user += m_ant * p_max * bytes_c64  # per-chunk RX x TX product (complex64)
    per_user = max(per_user, 1)
    return max(1, int(budget_bytes // per_user))


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


def _resolve_response_dims(
    array_response_product: np.ndarray | None,
    array_response_rx: np.ndarray | None,
    array_response_tx: np.ndarray | None,
) -> tuple[bool, int, int]:
    """Validate the array-response inputs and resolve (use_separate, M_rx, M_tx).

    Exactly one form must be provided: the full ``array_response_product`` or both the
    separate ``array_response_rx`` and ``array_response_tx`` responses.
    """
    if array_response_product is not None:
        m_rx, m_tx = array_response_product.shape[1:3]
        return False, m_rx, m_tx
    if array_response_rx is None or array_response_tx is None:
        msg = (
            "Provide either array_response_product, or both array_response_rx and "
            "array_response_tx."
        )
        raise ValueError(msg)
    return True, array_response_rx.shape[1], array_response_tx.shape[1]


def _chunk_array_product(
    array_response_product: np.ndarray | None,
    array_response_rx: np.ndarray | None,
    array_response_tx: np.ndarray | None,
    idx: slice | np.ndarray,
) -> np.ndarray:
    """Return the [B, M_rx, M_tx, P] array-response product for a single UE chunk.

    ``idx`` selects the chunk's users: a contiguous ``slice`` (returns a view, the default) or
    an index array (a gather, used when users are processed in valid-path-count order). When
    responses are carried separately, the M_rx x M_tx product is formed for this chunk only
    (via ``np.einsum``), so the full product is never materialized for all users at once:
    ``arp[b, r, t, p] = rx[b, r, p] * tx[b, t, p]``.
    """
    if array_response_product is not None:
        return array_response_product[idx]
    return np.einsum(
        "brp,btp->brtp",
        array_response_rx[idx],
        array_response_tx[idx],
        optimize=True,
    )


def _trim_chunk_paths(  # noqa: PLR0913
    arp_b: np.ndarray,
    mask_b: np.ndarray,
    power_b: np.ndarray,
    delay_b: np.ndarray,
    phase_b: np.ndarray,
    doppler_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Trim a chunk's path axis to its valid extent, cutting padded-path FLOPs.

    Valid paths are compacted to the front (only when padding is interspersed; real and sorted
    front-loaded chunks skip this), then every array is sliced to the chunk's maximum valid path
    count. Because padded paths carry zero power (hence zero amplitude), dropping the trailing
    all-padding columns leaves the path-summed channel unchanged. When chunks are ordered by
    valid-path count this removes most of the padded columns that otherwise dominate large-K /
    large-M OFDM contractions.

    Returns the (possibly) trimmed ``(arp_b, power_b, delay_b, phase_b, doppler_b)``.
    """
    # Compact only when padding is interspersed (a False followed by a True in the mask).
    if not bool(np.all(mask_b[:, :-1] >= mask_b[:, 1:])):
        order = np.argsort(~mask_b, axis=1, kind="stable")  # valid paths first, order preserved
        mask_b = np.take_along_axis(mask_b, order, axis=1)
        power_b = np.take_along_axis(power_b, order, axis=1)
        delay_b = np.take_along_axis(delay_b, order, axis=1)
        phase_b = np.take_along_axis(phase_b, order, axis=1)
        doppler_b = np.take_along_axis(doppler_b, order, axis=1)
        arp_b = np.take_along_axis(arp_b, order[:, None, None, :], axis=3)

    c_max = max(1, int(mask_b.sum(axis=1).max()))
    if c_max < mask_b.shape[1]:
        arp_b = arp_b[:, :, :, :c_max]
        power_b = power_b[:, :c_max]
        delay_b = delay_b[:, :c_max]
        phase_b = phase_b[:, :c_max]
        doppler_b = doppler_b[:, :c_max]
    return arp_b, power_b, delay_b, phase_b, doppler_b


def _generate_mimo_channel(  # noqa: PLR0913 - explicit path arrays preferred for clarity
    array_response_product: np.ndarray | None = None,
    *,
    power: np.ndarray,
    delay: np.ndarray,
    phase: np.ndarray,
    doppler: np.ndarray,
    ofdm_params: dict,
    array_response_rx: np.ndarray | None = None,
    array_response_tx: np.ndarray | None = None,
    times: float | np.ndarray = 0.0,
    freq_domain: bool = True,
    squeeze_time: bool = True,
    chunk_size: int | None = None,
) -> np.ndarray:
    """Generate MIMO channel matrices in frequency or time domain.

    Users (UEs) are processed in batches (chunks) and the whole computation is vectorized
    across the chunk, which amortizes Python-loop overhead while keeping peak memory
    bounded. Results are identical (within floating tolerance) to a per-UE computation.

    The antenna array response can be supplied in either of two equivalent ways:

    - ``array_response_product``: the full pre-formed [n_users, M_rx, M_tx, P_max] outer
      product of the RX and TX responses (legacy interface).
    - ``array_response_rx`` + ``array_response_tx``: the RX [n_users, M_rx, P_max] and TX
      [n_users, M_tx, P_max] responses carried separately. The M_rx x M_tx product is then
      formed one chunk at a time (via ``np.einsum``), so the full product is never held for
      all users simultaneously. This is the memory-efficient path used by the dataset.

    Exactly one of the two forms must be provided.

    Args:
        array_response_product: [n_users, M_rx, M_tx, P_max] antenna array responses, or
            None when ``array_response_rx``/``array_response_tx`` are given instead.
        power: [n_users, P_max] path powers (linear scale)
        delay: [n_users, P_max] path delays (seconds)
        phase: [n_users, P_max] path phases (degrees)
        doppler: [n_users, P_max] Doppler frequencies (Hz)
        ofdm_params: OFDM parameters dictionary
        array_response_rx: [n_users, M_rx, P_max] RX array responses (separate form).
        array_response_tx: [n_users, M_tx, P_max] TX array responses (separate form).
        times: Time samples (scalar or array, in seconds)
        freq_domain: If True, generate frequency-domain (OFDM) channel
        squeeze_time: If True and single time sample, squeeze time dimension
        chunk_size: Number of users processed per batch. If None (default), a size is
            derived from a peak working-memory budget. Output is independent of this value.

    Returns:
        Channel matrix with shape depending on domain and time:
        - Freq domain, single time: [n_users, M_rx, M_tx, K_subcarriers]
        - Freq domain, multi time: [n_users, M_rx, M_tx, K_subcarriers, N_t]
        - Time domain, single time: [n_users, M_rx, M_tx, P_max]
        - Time domain, multi time: [n_users, M_rx, M_tx, P_max, N_t]

    """
    use_separate, m_rx, m_tx = _resolve_response_dims(
        array_response_product, array_response_rx, array_response_tx
    )

    # Time handling
    times_arr = np.atleast_1d(times).astype(float)  # [N_t]
    n_times = times_arr.shape[0]

    ts = 1.0 / ofdm_params[c.PARAMSET_OFDM_BANDWIDTH]
    subcarriers = ofdm_params[c.PARAMSET_OFDM_SC_SAMP]
    k_subcarriers = len(subcarriers)

    # The OFDM path generator is only used by the frequency-domain path; building it eagerly
    # for the time domain would waste an [N, K_sel]-scaling construction (time-domain timing
    # would otherwise grow with the subcarrier count for no reason).
    path_gen = OFDMPathGenerator(ofdm_params, subcarriers) if freq_domain else None

    # Delay sanity for OFDM mode
    if freq_domain:
        _check_ofdm_compatibility(ofdm_params, delay)

    n_ues = power.shape[0]
    p_max = power.shape[1]

    last_ch_dim = k_subcarriers if freq_domain else p_max

    # Allocate output
    if n_times == 1 and squeeze_time:
        channel = np.zeros((n_ues, m_rx, m_tx, last_ch_dim), dtype=np.csingle)
    else:
        channel = np.zeros((n_ues, m_rx, m_tx, last_ch_dim, n_times), dtype=np.csingle)

    # Valid-path mask (NaN padding marks invalid/padded paths)
    nan_masks = ~np.isnan(power)  # [n_users, P_max]
    valid_counts = nan_masks.sum(axis=1)  # [n_users]

    # Frequency domain only: order users by valid-path count so each chunk can be trimmed to its
    # (small) local maximum, removing the padded-path FLOPs that dominate large-K / large-M OFDM
    # contractions. Only worth it when counts vary enough to recover the gather/scatter cost; the
    # time-domain path keeps its delay structure on the path axis and is left in natural order.
    count_spread = int(valid_counts.max() - valid_counts.min()) if n_ues else 0
    trim_paths = freq_domain and n_ues > 1 and count_spread >= _MIN_COUNT_SPREAD_FOR_SORT
    user_order = np.argsort(valid_counts, kind="stable") if trim_paths else None

    # Resolve the UE-chunk size (memory-budgeted when not provided)
    if chunk_size is None:
        chunk_size = _estimate_chunk_size(
            freq_domain=freq_domain,
            lpf=bool(ofdm_params[c.PARAMSET_OFDM_LPF]),
            m_rx=m_rx,
            m_tx=m_tx,
            p_max=p_max,
            k_sel=k_subcarriers,
            n_times=n_times,
            total_subcarriers=ofdm_params[c.PARAMSET_OFDM_SC_NUM],
            materialize_product=use_separate,
        )
        if trim_paths:
            # Prefer several count-sorted chunks so the per-chunk trim is effective (a single
            # chunk would trim only to the global maximum count, i.e. not at all).
            chunk_size = min(chunk_size, max(16, -(-n_ues // _SORTED_TARGET_CHUNKS)))
    chunk_size = max(1, int(chunk_size))

    n_chunks = (n_ues + chunk_size - 1) // chunk_size
    for start in tqdm(range(0, n_ues, chunk_size), total=n_chunks, desc="Generating channels"):
        end = min(start + chunk_size, n_ues)
        # idx selects the chunk's users: a contiguous slice (view) by default, or a count-sorted
        # index array (gather) when trimming. channel[idx] = ... writes back to original rows.
        idx = user_order[start:end] if trim_paths else slice(start, end)
        mask_b = nan_masks[idx]  # [B, P_max]
        if not mask_b.any():
            continue  # all-padding chunk -> leave zeros (matches 0-path behavior)

        # [B, M_rx, M_tx, P_max]; product formed per-chunk when responses are separate.
        arp_b = _chunk_array_product(
            array_response_product, array_response_rx, array_response_tx, idx
        )
        # Zero padded paths so they contribute exactly zero (avoids NaN propagation).
        power_b = np.where(mask_b, power[idx], 0.0)
        phase_b = np.where(mask_b, phase[idx], 0.0)
        doppler_b = np.where(mask_b, doppler[idx], 0.0)

        if freq_domain:
            # OFDM channel H[b,r,t,k,n] = Σ_p arp·√(P/N)·exp(j(φ₀+2πfD·t))·exp(-j2πτk/N);
            # delays create per-subcarrier (frequency-dependent) phase shifts.
            delay_b = np.where(mask_b, delay[idx], 0.0)
            if trim_paths:
                arp_b, power_b, delay_b, phase_b, doppler_b = _trim_chunk_paths(
                    arp_b, mask_b, power_b, delay_b, phase_b, doppler_b
                )
            channel[idx] = _compute_freq_channel_batch(
                arp_b,
                path_gen,
                power_b,
                delay_b,
                phase_b,
                doppler_b,
                ts,
                times_arr,
                squeeze_time=squeeze_time,
            )
        else:
            # Time-domain gains a[b,p,n] = √P·exp(j(φ₀+2πfD·t)); delay is the path index.
            channel[idx] = _compute_time_channel_batch(
                arp_b,
                mask_b,
                power_b,
                phase_b,
                doppler_b,
                times_arr,
                squeeze_time=squeeze_time,
            )

    return channel
