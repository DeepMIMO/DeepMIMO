"""Channel generation tests for DeepMIMO."""

from copy import deepcopy

import numpy as np
import pytest

from deepmimo import consts as c
from deepmimo.generator.channel import (
    ChannelParameters,
    OFDMPathGenerator,
    _check_ofdm_compatibility,
    _compute_single_freq_channel,
    _compute_single_time_channel,
    _convert_lists_to_arrays,
    _generate_mimo_channel,
    _validate_ant_rad_pat,
    _validate_ant_rot,
)


def test_channel_parameters_defaults() -> None:
    """Test default channel parameters."""
    params = ChannelParameters()
    assert c.PARAMSET_ANT_BS in params
    assert c.PARAMSET_ANT_UE in params
    assert params[c.PARAMSET_ANT_BS][c.PARAMSET_ANT_SPACING] == 0.5


def test_channel_parameters_validation() -> None:
    """Test parameter validation."""
    params = ChannelParameters()
    # Test valid params
    params.validate(n_ues=10)

    # Test invalid OFDM subcarriers
    params[c.PARAMSET_OFDM][c.PARAMSET_OFDM_SC_SAMP] = np.array([-1])
    with pytest.raises(ValueError, match="values must be within"):
        params.validate(n_ues=10)

    # Reset and test another invalid case
    params = ChannelParameters()
    params[c.PARAMSET_OFDM][c.PARAMSET_OFDM_SC_SAMP] = "invalid"  # Not an array/list of ints
    with pytest.raises(ValueError, match="invalid"):
        params.validate(n_ues=10)


def test_convert_lists_to_arrays() -> None:
    """Test helper for converting lists to arrays."""
    data = {"a": [1, 2, 3], "b": {"c": [4, 5]}}
    converted = _convert_lists_to_arrays(data)
    assert isinstance(converted["a"], np.ndarray)
    assert isinstance(converted["b"]["c"], np.ndarray)


def test_generate_mimo_channel_time_domain() -> None:
    """Test channel generation in time domain."""
    # Setup mock inputs
    n_users = 2
    n_rx_ant = 1
    n_tx_ant = 2
    n_paths = 3

    # [n_users, M_rx, M_tx, P_max]
    array_response = np.ones((n_users, n_rx_ant, n_tx_ant, n_paths), dtype=complex)
    powers = np.ones((n_users, n_paths))
    delays = np.zeros((n_users, n_paths))
    phases = np.zeros((n_users, n_paths))
    dopplers = np.zeros((n_users, n_paths))

    ofdm_params = ChannelParameters.DEFAULT_PARAMS[c.PARAMSET_OFDM]

    # Generate channel (Time Domain)
    channel = _generate_mimo_channel(
        array_response_product=array_response,
        power=powers,
        delay=delays,
        phase=phases,
        doppler=dopplers,
        ofdm_params=ofdm_params,
        times=0.0,
        freq_domain=False,
    )

    # Expected shape: [n_users, M_rx, M_tx, P_max] (squeezed time)
    assert channel.shape == (n_users, n_rx_ant, n_tx_ant, n_paths)

    # Check values: should be close to sqrt(1) * 1 * 1 = 1 (since power=1, phase=0)
    # Actually power is linear here.
    assert np.allclose(np.abs(channel), 1.0)


def test_generate_mimo_channel_freq_domain() -> None:
    """Test channel generation in frequency domain (OFDM)."""
    n_users = 1
    n_rx_ant = 1
    n_tx_ant = 1
    n_paths = 1

    array_response = np.ones((n_users, n_rx_ant, n_tx_ant, n_paths), dtype=complex)
    powers = np.ones((n_users, n_paths))  # Linear power 1.0
    delays = np.zeros((n_users, n_paths))
    phases = np.zeros((n_users, n_paths))
    dopplers = np.zeros((n_users, n_paths))

    ofdm_params = ChannelParameters.DEFAULT_PARAMS[c.PARAMSET_OFDM]
    # Set known subcarriers
    ofdm_params[c.PARAMSET_OFDM_SC_SAMP] = np.array([0, 1])  # 2 subcarriers

    channel = _generate_mimo_channel(
        array_response_product=array_response,
        power=powers,
        delay=delays,
        phase=phases,
        doppler=dopplers,
        ofdm_params=ofdm_params,
        times=0.0,
        freq_domain=True,
    )

    # Expected shape: [n_users, M_rx, M_tx, K_sel]
    assert channel.shape == (n_users, n_rx_ant, n_tx_ant, 2)

    # Check values. Power splits over total subcarriers in OFDM usually?
    # In generate_mimo_channel:
    # a_pt = sqrt(power / total_subcarriers) ...
    # So magnitude should be related to total_subcarriers.
    total_sc = ofdm_params[c.PARAMSET_OFDM_SC_NUM]
    expected_mag = np.sqrt(1.0 / total_sc)

    # Allow some tolerance for floating point
    assert np.allclose(np.abs(channel), expected_mag, rtol=1e-5)


def test_doppler_progression() -> None:
    """Test that channel evolves with time due to Doppler."""
    n_users = 1
    n_rx_ant = 1
    n_tx_ant = 1
    n_paths = 1

    array_response = np.ones((n_users, n_rx_ant, n_tx_ant, n_paths), dtype=complex)
    powers = np.ones((n_users, n_paths))
    delays = np.zeros((n_users, n_paths))
    phases = np.zeros((n_users, n_paths))
    dopplers = np.ones((n_users, n_paths)) * 100.0  # 100 Hz Doppler

    ofdm_params = ChannelParameters.DEFAULT_PARAMS[c.PARAMSET_OFDM]
    times = np.array([0.0, 0.0025])  # 0 and 1/4 cycle (period = 1/100 = 0.01s)

    channel = _generate_mimo_channel(
        array_response_product=array_response,
        power=powers,
        delay=delays,
        phase=phases,
        doppler=dopplers,
        ofdm_params=ofdm_params,
        times=times,
        freq_domain=False,
        squeeze_time=False,
    )

    # channel shape: [1, 1, 1, 1, 2]
    ch_t0 = channel[0, 0, 0, 0, 0]
    ch_t1 = channel[0, 0, 0, 0, 1]

    # t=0 -> phase 0
    # t=0.0025 -> phase 2*pi*100*0.0025 = pi/2 -> j

    # a_pt = sqrt(P) * exp(j(phi + 2pi fD t))
    assert np.isclose(ch_t0, 1.0 + 0j)
    assert np.isclose(ch_t1, 0.0 + 1j, atol=1e-5)


def test_validate_ant_rot_none_returns_zero_vector() -> None:
    """None rotation should return a zero vector."""
    result = _validate_ant_rot(None)
    np.testing.assert_array_equal(result, [0, 0, 0])


def test_validate_ant_rot_1d_vector() -> None:
    """A 3-element 1D vector should pass unchanged."""
    rot = np.array([10.0, 20.0, 30.0])
    result = _validate_ant_rot(rot)
    np.testing.assert_array_equal(result, rot)


def test_validate_ant_rot_3x2_range_matrix() -> None:
    """A 3x2 matrix (random ranges) should pass unchanged."""
    rot = np.array([[0.0, 10.0], [0.0, 5.0], [-5.0, 5.0]])
    result = _validate_ant_rot(rot)
    np.testing.assert_array_equal(result, rot)


def test_validate_ant_rot_per_user() -> None:
    """An (n_ues, 3) matrix should pass when n_ues is provided."""
    rot = np.zeros((4, 3))
    result = _validate_ant_rot(rot, n_ues=4)
    np.testing.assert_array_equal(result, rot)


def test_validate_ant_rot_invalid_raises() -> None:
    """Invalid shapes should raise ValueError."""
    with pytest.raises(ValueError, match="antenna rotation"):
        _validate_ant_rot(np.array([1.0, 2.0]))  # 2-element vector


def test_validate_ant_rad_pat_none_returns_default() -> None:
    """None pattern should return the first valid pattern (isotropic)."""
    result = _validate_ant_rad_pat(None)
    assert result == c.PARAMSET_ANT_RAD_PAT_VALS[0]


def test_validate_ant_rad_pat_valid() -> None:
    """All defined valid patterns should pass through unchanged."""
    for pat in c.PARAMSET_ANT_RAD_PAT_VALS:
        assert _validate_ant_rad_pat(pat) == pat


def test_validate_ant_rad_pat_invalid_raises() -> None:
    """Unknown pattern strings should raise ValueError."""
    with pytest.raises(ValueError, match="antenna radiation pattern"):
        _validate_ant_rad_pat("omni_magic")


# ---------------------------------------------------------------------------
# Additional coverage tests for channel.py
# ---------------------------------------------------------------------------

# ── ChannelParameters.__init__ with data dict (line 198) ───────────────────


def test_channel_parameters_init_with_data_dict() -> None:
    """Passing a data dict should deep-merge with defaults."""
    params = ChannelParameters({"bs_antenna": {"shape": [4, 4]}})
    bs = params[c.PARAMSET_ANT_BS]
    # shape should be overridden to [4, 4]
    assert list(bs[c.PARAMSET_ANT_SHAPE]) == [4, 4]
    # spacing should remain at default value
    assert bs[c.PARAMSET_ANT_SPACING] == 0.5


def test_channel_parameters_init_with_kwargs() -> None:
    """Passing kwargs should deep-merge with defaults."""
    params = ChannelParameters(bs_antenna={"shape": [2, 2]})
    bs = params[c.PARAMSET_ANT_BS]
    assert list(bs[c.PARAMSET_ANT_SHAPE]) == [2, 2]
    # Other defaults should survive
    assert bs[c.PARAMSET_ANT_SPACING] == 0.5


def test_channel_parameters_init_data_and_kwargs() -> None:
    """Data dict and kwargs should both be applied."""
    params = ChannelParameters({"doppler": 1}, ue_antenna={"shape": [2, 1]})
    assert params[c.PARAMSET_DOPPLER_EN] == 1
    assert list(params[c.PARAMSET_ANT_UE][c.PARAMSET_ANT_SHAPE]) == [2, 1]


# ── _validate_antenna_params without rotation key (line 214) ───────────────


def test_validate_antenna_params_missing_rotation_sets_default() -> None:
    """Missing rotation key should default to [0, 0, 0]."""
    cp = ChannelParameters()
    bs_params = deepcopy(dict(cp[c.PARAMSET_ANT_BS]))
    del bs_params[c.PARAMSET_ANT_ROTATION]

    cp._validate_antenna_params(bs_params)  # noqa: SLF001

    np.testing.assert_array_equal(bs_params[c.PARAMSET_ANT_ROTATION], [0, 0, 0])


def test_validate_antenna_params_missing_rad_pat_sets_default() -> None:
    """Missing radiation_pattern key should default to first valid pattern."""
    cp = ChannelParameters()
    bs_params = deepcopy(dict(cp[c.PARAMSET_ANT_BS]))
    del bs_params[c.PARAMSET_ANT_RAD_PAT]

    cp._validate_antenna_params(bs_params)  # noqa: SLF001

    assert bs_params[c.PARAMSET_ANT_RAD_PAT] == c.PARAMSET_ANT_RAD_PAT_VALS[0]


# ── _validate_ofdm_subcarriers edge cases (lines 233-248) ──────────────────


def test_validate_ofdm_subcarriers_2d_array_raises() -> None:
    """A 2-D array for selected_subcarriers should raise ValueError."""
    cp = ChannelParameters()
    bad_ofdm = {
        c.PARAMSET_OFDM_SC_SAMP: np.array([[0, 1], [2, 3]]),
        c.PARAMSET_OFDM_SC_NUM: 512,
    }
    with pytest.raises(ValueError, match="1-D array"):
        cp._validate_ofdm_subcarriers(bad_ofdm)  # noqa: SLF001


def test_validate_ofdm_subcarriers_empty_array_raises() -> None:
    """An empty array for selected_subcarriers should raise ValueError."""
    cp = ChannelParameters()
    bad_ofdm = {
        c.PARAMSET_OFDM_SC_SAMP: np.array([]),
        c.PARAMSET_OFDM_SC_NUM: 512,
    }
    with pytest.raises(ValueError, match="non-empty"):
        cp._validate_ofdm_subcarriers(bad_ofdm)  # noqa: SLF001


def test_validate_ofdm_subcarriers_float_converted_to_int() -> None:
    """Float-valued selected_subcarriers should be silently cast to int."""
    cp = ChannelParameters()
    ofdm = {
        c.PARAMSET_OFDM_SC_SAMP: np.array([0.0, 1.0, 2.0]),
        c.PARAMSET_OFDM_SC_NUM: 512,
    }
    cp._validate_ofdm_subcarriers(ofdm)  # noqa: SLF001

    assert np.issubdtype(ofdm[c.PARAMSET_OFDM_SC_SAMP].dtype, np.integer)
    np.testing.assert_array_equal(ofdm[c.PARAMSET_OFDM_SC_SAMP], [0, 1, 2])


# ── validate() with extra keys prints warning (lines 286-287) ──────────────


def test_validate_extra_keys_prints_warning(capsys) -> None:
    """Unknown top-level keys should trigger a printed warning."""
    cp = ChannelParameters()
    cp["totally_unknown_param"] = "some_value"

    cp.validate(10)

    captured = capsys.readouterr()
    assert "unnecessary" in captured.out
    assert "totally_unknown_param" in captured.out


# ── _check_ofdm_compatibility when delay exceeds OFDM duration (lines 421-438)


def test_check_ofdm_compatibility_excess_delay_prints_warning(capsys) -> None:
    """Delays exceeding the OFDM symbol duration should print a warning."""
    ofdm_params = deepcopy(ChannelParameters.DEFAULT_PARAMS[c.PARAMSET_OFDM])
    n_sc = ofdm_params[c.PARAMSET_OFDM_SC_NUM]  # 512
    bandwidth = ofdm_params[c.PARAMSET_OFDM_BANDWIDTH]  # 10 MHz
    ts = 1.0 / bandwidth
    symbol_duration = n_sc * ts  # ~51.2 µs

    # Delay more than one symbol duration
    delays = np.array([[symbol_duration * 2.0]])

    _check_ofdm_compatibility(ofdm_params, delays)

    captured = capsys.readouterr()
    assert "Warning" in captured.out
    assert "symbol duration" in captured.out


def test_check_ofdm_compatibility_no_warning_when_within_budget(capsys) -> None:
    """Delays within the OFDM symbol duration should produce no output."""
    ofdm_params = deepcopy(ChannelParameters.DEFAULT_PARAMS[c.PARAMSET_OFDM])
    n_sc = ofdm_params[c.PARAMSET_OFDM_SC_NUM]
    bandwidth = ofdm_params[c.PARAMSET_OFDM_BANDWIDTH]
    ts = 1.0 / bandwidth
    symbol_duration = n_sc * ts

    delays = np.array([[symbol_duration * 0.5]])  # well within budget

    _check_ofdm_compatibility(ofdm_params, delays)

    captured = capsys.readouterr()
    assert captured.out == ""


# ── OFDMPathGenerator.generate - 'over' condition zeros power (lines 379-381)


def test_ofdm_path_generator_over_clips_to_zero() -> None:
    """Paths with delay_n >= n_subcarriers should have zeroed-out power."""
    ofdm_params = deepcopy(ChannelParameters.DEFAULT_PARAMS[c.PARAMSET_OFDM])
    n_sc = ofdm_params[c.PARAMSET_OFDM_SC_NUM]  # 512
    bandwidth = ofdm_params[c.PARAMSET_OFDM_BANDWIDTH]
    ts = 1.0 / bandwidth
    symbol_duration = n_sc * ts

    subcarriers = np.array([0, 1, 2])
    gen = OFDMPathGenerator(ofdm_params, subcarriers)

    # Two paths: first exceeds OFDM window, second is normal
    pwr = np.array([1.0, 0.5])
    toa = np.array([symbol_duration + 1e-9, 0.0])  # path 0: over; path 1: ok
    phs = np.array([0.0, 0.0])
    dopplers = np.array([0.0, 0.0])

    h = gen.generate(pwr, toa, phs, ts, dopplers, times=0.0)

    # First path (over) should be all zeros
    np.testing.assert_allclose(np.abs(h[0]), 0.0, atol=1e-10)
    # Second path should be non-zero
    assert np.any(np.abs(h[1]) > 0)


# ── OFDMPathGenerator.generate with LPF=True (lines 393-396) ───────────────


def test_ofdm_path_generator_lpf_true_returns_correct_shape() -> None:
    """LPF=True path should return the same shape as the non-LPF path."""
    ofdm_params_no_lpf = deepcopy(ChannelParameters.DEFAULT_PARAMS[c.PARAMSET_OFDM])
    ofdm_params_lpf = deepcopy(ChannelParameters.DEFAULT_PARAMS[c.PARAMSET_OFDM])
    ofdm_params_lpf[c.PARAMSET_OFDM_LPF] = True

    subcarriers = np.array([0, 1, 2])

    gen_no_lpf = OFDMPathGenerator(ofdm_params_no_lpf, subcarriers)
    gen_lpf = OFDMPathGenerator(ofdm_params_lpf, subcarriers)

    ts = 1.0 / ofdm_params_no_lpf[c.PARAMSET_OFDM_BANDWIDTH]
    pwr = np.array([1.0])
    toa = np.array([0.0])
    phs = np.array([0.0])
    dopplers = np.array([0.0])

    h_no_lpf = gen_no_lpf.generate(pwr, toa, phs, ts, dopplers, times=0.0)
    h_lpf = gen_lpf.generate(pwr, toa, phs, ts, dopplers, times=0.0)

    assert h_no_lpf.shape == h_lpf.shape


def test_ofdm_path_generator_lpf_true_nonzero_output() -> None:
    """LPF branch should produce non-zero gains for a valid path."""
    ofdm_params = deepcopy(ChannelParameters.DEFAULT_PARAMS[c.PARAMSET_OFDM])
    ofdm_params[c.PARAMSET_OFDM_LPF] = True
    subcarriers = np.array([0, 1, 2])
    gen = OFDMPathGenerator(ofdm_params, subcarriers)
    ts = 1.0 / ofdm_params[c.PARAMSET_OFDM_BANDWIDTH]

    h = gen.generate(
        pwr=np.array([1.0]),
        toa=np.array([0.0]),
        phs=np.array([0.0]),
        ts=ts,
        dopplers=np.array([0.0]),
        times=0.0,
    )
    assert np.any(np.abs(h) > 0)


# ── _compute_single_freq_channel (lines 441-466) ────────────────────────────


def test_compute_single_freq_channel_squeeze_time() -> None:
    """squeeze_time=True with N_t=1 should drop the time dimension."""
    array_product = np.ones((2, 3, 4), dtype=complex)  # [M_rx, M_tx, P]
    path_gains = np.ones((4, 5, 1), dtype=complex)  # [P, K, N_t=1]

    result = _compute_single_freq_channel(array_product, path_gains, squeeze_time=True)

    assert result.shape == (2, 3, 5)
    assert result.dtype == np.complex64


def test_compute_single_freq_channel_no_squeeze() -> None:
    """squeeze_time=False should keep the time dimension even with N_t=1."""
    array_product = np.ones((2, 3, 4), dtype=complex)
    path_gains = np.ones((4, 5, 1), dtype=complex)

    result = _compute_single_freq_channel(array_product, path_gains, squeeze_time=False)

    assert result.shape == (2, 3, 5, 1)


def test_compute_single_freq_channel_multi_time() -> None:
    """Multiple time samples should produce output with the time dimension."""
    n_t = 4
    array_product = np.ones((2, 3, 4), dtype=complex)
    path_gains = np.ones((4, 5, n_t), dtype=complex)

    result = _compute_single_freq_channel(array_product, path_gains, squeeze_time=True)

    # N_t > 1: squeeze_time has no effect, time dim preserved
    assert result.shape == (2, 3, 5, n_t)


# ── _compute_single_time_channel (line 576 via _generate_mimo_channel) ──────


def test_compute_single_time_channel_squeeze() -> None:
    """squeeze_time=True with N_t=1 should return [M_rx, M_tx, P_max]."""
    array_product = np.ones((2, 3, 4), dtype=complex)  # [M_rx, M_tx, P]
    path_gains = np.ones((4, 1), dtype=complex)  # [P, N_t=1]
    p_max = 6

    result = _compute_single_time_channel(array_product, path_gains, p_max, squeeze_time=True)

    assert result.shape == (2, 3, p_max)
    assert result.dtype == np.complex64
    # First 4 paths filled, remaining 2 zero-padded
    assert np.all(result[..., :4] != 0)
    np.testing.assert_allclose(np.abs(result[..., 4:]), 0.0)


def test_compute_single_time_channel_no_squeeze() -> None:
    """squeeze_time=False should return [M_rx, M_tx, P_max, N_t]."""
    n_t = 3
    array_product = np.ones((2, 3, 4), dtype=complex)
    path_gains = np.ones((4, n_t), dtype=complex)
    p_max = 5

    result = _compute_single_time_channel(array_product, path_gains, p_max, squeeze_time=False)

    assert result.shape == (2, 3, p_max, n_t)
    assert result.dtype == np.complex64


# ── _generate_mimo_channel: zero-path user skipped (line 576) ────────────────


def test_generate_mimo_channel_all_nan_user_skipped() -> None:
    """User with all-NaN powers should be skipped (channel row stays zero)."""
    ofdm_params = deepcopy(ChannelParameters.DEFAULT_PARAMS[c.PARAMSET_OFDM])

    n_users, m_rx, m_tx, p_max = 2, 1, 1, 2
    array_response = np.ones((n_users, m_rx, m_tx, p_max), dtype=complex)

    power = np.array([[np.nan, np.nan], [1.0, 1.0]])
    delay = np.zeros((n_users, p_max))
    phase = np.zeros((n_users, p_max))
    doppler = np.zeros((n_users, p_max))

    channel = _generate_mimo_channel(
        array_response_product=array_response,
        power=power,
        delay=delay,
        phase=phase,
        doppler=doppler,
        ofdm_params=ofdm_params,
        times=0.0,
        freq_domain=False,
    )

    # User 0 (all-NaN paths) must remain zero; user 1 must be non-zero.
    np.testing.assert_allclose(np.abs(channel[0]), 0.0)
    assert np.any(np.abs(channel[1]) > 0)


# ── _validate_ofdm_subcarriers: missing keys early-return (line 233) ─────────


def test_validate_ofdm_subcarriers_missing_keys_returns_early() -> None:
    """Params without SC_SAMP/SC_NUM keys should return without raising."""
    cp = ChannelParameters()
    cp._validate_ofdm_subcarriers({})  # noqa: SLF001


# ---------------------------------------------------------------------------
# Regression: the batched generator must match an independent per-UE reference
# ---------------------------------------------------------------------------


def _make_ofdm_params(n_sc=64, selected=(0, 5, 17, 31), bandwidth=10e6, lpf=0):
    """Build a fresh OFDM params dict (never mutate the shared class defaults)."""
    return {
        c.PARAMSET_OFDM_SC_NUM: n_sc,
        c.PARAMSET_OFDM_SC_SAMP: np.array(selected),
        c.PARAMSET_OFDM_BANDWIDTH: bandwidth,
        c.PARAMSET_OFDM_LPF: lpf,
    }


def _make_synthetic(n_users, n_rx, n_tx, p_max, seed=0):
    """Synthetic ragged multi-user paths with NaN padding.

    Guarantees at least one user with 0 valid paths, one user with the maximum number
    of paths, and one user with interspersed (non-front-loaded) NaN padding. The array
    response is finite everywhere (including padded paths) so the test also verifies that
    padded paths contribute exactly zero.
    """
    rng = np.random.default_rng(seed)
    power = np.full((n_users, p_max), np.nan)
    delay = np.full((n_users, p_max), np.nan)
    phase = np.full((n_users, p_max), np.nan)
    doppler = np.full((n_users, p_max), np.nan)
    array_response = (
        rng.standard_normal((n_users, n_rx, n_tx, p_max))
        + 1j * rng.standard_normal((n_users, n_rx, n_tx, p_max))
    )

    counts = rng.integers(0, p_max + 1, size=n_users)
    counts[0] = 0  # user with no valid paths
    counts[1] = p_max  # user with the maximum number of paths
    for i in range(n_users):
        n = counts[i]
        if n == 0:
            continue
        if i == 2 and 0 < n < p_max:  # interspersed padding (exercises compaction)
            idx = np.sort(rng.choice(p_max, size=n, replace=False))
        else:  # front-loaded layout (as produced by the real converters)
            idx = np.arange(n)
        power[i, idx] = rng.uniform(0.05, 1.0, n)
        delay[i, idx] = rng.uniform(0.0, 4.0e-6, n)
        phase[i, idx] = rng.uniform(0.0, 360.0, n)
        doppler[i, idx] = rng.uniform(-300.0, 300.0, n)
    return array_response, power, delay, phase, doppler


def _reference_mimo_channel(  # noqa: PLR0913, PLR0915
    array_response_product, power, delay, phase, doppler, ofdm_params, times,
    *, freq_domain, squeeze_time,
):
    """Independent, explicit per-UE reference implementation of the channel."""
    times_arr = np.atleast_1d(times).astype(float)
    n_times = times_arr.shape[0]
    ts = 1.0 / ofdm_params[c.PARAMSET_OFDM_BANDWIDTH]
    subc = np.asarray(ofdm_params[c.PARAMSET_OFDM_SC_SAMP])
    n_sc = ofdm_params[c.PARAMSET_OFDM_SC_NUM]
    lpf_on = bool(ofdm_params[c.PARAMSET_OFDM_LPF])
    k_sel = len(subc)
    n_users, m_rx, m_tx, p_max = array_response_product.shape
    last = k_sel if freq_domain else p_max

    if n_times == 1 and squeeze_time:
        out = np.zeros((n_users, m_rx, m_tx, last), dtype=np.csingle)
    else:
        out = np.zeros((n_users, m_rx, m_tx, last, n_times), dtype=np.csingle)

    delay_d = np.arange(n_sc)
    delay_to_ofdm = np.exp(-1j * 2 * np.pi / n_sc * np.outer(delay_d, subc))

    for i in range(n_users):
        mask = ~np.isnan(power[i])
        if not mask.any():
            continue
        ap = array_response_product[i][..., mask]  # [m_rx, m_tx, P]
        pw = power[i, mask].astype(float)
        ph = phase[i, mask].astype(float)
        dp = doppler[i, mask].astype(float)
        if freq_domain:
            delay_n = (delay[i, mask].astype(float) / ts)
            pwr = pw.copy()
            fd = dp.copy()
            over = delay_n >= n_sc
            pwr[over] = 0.0
            delay_n = delay_n.copy()
            delay_n[over] = n_sc
            fd[over] = 0.0
            a_pt = np.sqrt(pwr / n_sc)[:, None] * np.exp(
                1j * (np.deg2rad(ph)[:, None] + 2 * np.pi * fd[:, None] * times_arr[None, :])
            )  # [P, N_t]
            if lpf_on:
                lpf = np.sinc(delay_d[None, :] - delay_n[:, None])  # [P, N]
                h_pk = lpf @ delay_to_ofdm  # [P, K]
            else:
                h_pk = np.exp(
                    -1j * (2 * np.pi / n_sc) * (delay_n[:, None] * subc[None, :])
                )  # [P, K]
            gains = (a_pt[:, None, :] * h_pk[:, :, None]).astype(np.complex64)  # [P, K, N_t]
            ch = np.einsum("rtp,pkn->rtkn", ap, gains).astype(np.complex64)
            out[i] = ch[..., 0] if (squeeze_time and n_times == 1) else ch
        else:
            gains = np.sqrt(pw)[:, None] * np.exp(
                1j * (np.deg2rad(ph)[:, None] + 2 * np.pi * dp[:, None] * times_arr[None, :])
            )  # [P, N_t]
            ch = np.einsum("rtp,pn->rtpn", ap, gains).astype(np.complex64)  # [m_rx, m_tx, P, N_t]
            n_paths = ap.shape[2]
            if squeeze_time and n_times == 1:
                tmp = np.zeros((m_rx, m_tx, p_max), dtype=np.complex64)
                tmp[..., :n_paths] = ch[..., 0]
            else:
                tmp = np.zeros((m_rx, m_tx, p_max, n_times), dtype=np.complex64)
                tmp[..., :n_paths, :] = ch
            out[i] = tmp
    return out


@pytest.mark.parametrize("freq_domain", [True, False])
@pytest.mark.parametrize("lpf", [0, 1])
@pytest.mark.parametrize("times", [0.0, np.array([0.0, 5e-5, 1e-4])])
@pytest.mark.parametrize("chunk_size", [None, 1, 4, 1000])
def test_batched_channel_matches_reference(freq_domain, lpf, times, chunk_size):
    """Batched generator must equal the per-UE reference (freq + time, all options)."""
    if not freq_domain and lpf:
        pytest.skip("LPF only applies to the frequency-domain path")

    n_users, n_rx, n_tx, p_max = 23, 2, 3, 6
    array_response, power, delay, phase, doppler = _make_synthetic(n_users, n_rx, n_tx, p_max)
    ofdm_params = _make_ofdm_params(lpf=lpf)
    squeeze_time = np.ndim(times) == 0

    reference = _reference_mimo_channel(
        array_response, power, delay, phase, doppler, ofdm_params, times,
        freq_domain=freq_domain, squeeze_time=squeeze_time,
    )
    result = _generate_mimo_channel(
        array_response_product=array_response,
        power=power,
        delay=delay,
        phase=phase,
        doppler=doppler,
        ofdm_params=ofdm_params,
        times=times,
        freq_domain=freq_domain,
        squeeze_time=squeeze_time,
        chunk_size=chunk_size,
    )

    assert result.shape == reference.shape
    assert result.dtype == np.csingle
    assert np.allclose(result, reference, rtol=1e-4, atol=1e-6)


def test_batched_channel_zero_path_users_are_zero():
    """Users with zero valid paths must produce all-zero channels (freq and time)."""
    n_users, n_rx, n_tx, p_max = 4, 2, 2, 5
    array_response, power, delay, phase, doppler = _make_synthetic(n_users, n_rx, n_tx, p_max)
    ofdm_params = _make_ofdm_params()

    for freq_domain in (True, False):
        result = _generate_mimo_channel(
            array_response_product=array_response,
            power=power,
            delay=delay,
            phase=phase,
            doppler=doppler,
            ofdm_params=ofdm_params,
            times=0.0,
            freq_domain=freq_domain,
        )
        # User 0 is constructed to have zero valid paths.
        assert np.all(result[0] == 0)
        # A user with valid paths should be non-zero.
        assert np.any(result[1] != 0)


# ---------------------------------------------------------------------------
# Uniform-subcarrier delay-phase recurrence (the fast path in _freq_path_factors)
# ---------------------------------------------------------------------------


def test_ofdm_path_generator_detects_uniform_subcarriers() -> None:
    """Arithmetic-spaced subcarriers enable the recurrence; non-uniform sets fall back."""
    params = _make_ofdm_params(n_sc=512)

    contiguous = OFDMPathGenerator(params, np.arange(10))
    assert contiguous.subcarriers_uniform
    assert contiguous.subcarriers_step == 1.0
    assert contiguous.subcarriers_start == 0.0

    strided = OFDMPathGenerator(params, np.arange(4, 40, 3))
    assert strided.subcarriers_uniform
    assert strided.subcarriers_step == 3.0
    assert strided.subcarriers_start == 4.0

    non_uniform = OFDMPathGenerator(params, np.array([0, 5, 17, 31]))
    assert not non_uniform.subcarriers_uniform

    # A single selected subcarrier is degenerate-uniform (no spacing to violate).
    single = OFDMPathGenerator(params, np.array([7]))
    assert single.subcarriers_uniform


@pytest.mark.parametrize(
    ("n_sc", "selected"),
    [
        (300, np.arange(200)),        # contiguous; crosses several 64-wide recurrence blocks
        (512, np.arange(0, 256, 2)),  # evenly strided (step 2); also crosses block boundaries
        (128, np.arange(64)),         # exactly one recurrence block wide
        (96, np.arange(65)),          # one element past a block boundary (re-anchor edge)
    ],
)
def test_batched_freq_channel_uniform_subcarriers_match_reference(n_sc, selected) -> None:
    """Uniform subcarriers take the recurrence path; must match the direct-exp reference."""
    ofdm_params = _make_ofdm_params(n_sc=n_sc, selected=selected)
    # Guard: these sets must actually exercise the recurrence (not the fallback).
    assert OFDMPathGenerator(ofdm_params, np.asarray(selected)).subcarriers_uniform

    n_users, n_rx, n_tx, p_max = 17, 2, 2, 6
    array_response, power, delay, phase, doppler = _make_synthetic(
        n_users, n_rx, n_tx, p_max, seed=3
    )

    reference = _reference_mimo_channel(
        array_response, power, delay, phase, doppler, ofdm_params, 0.0,
        freq_domain=True, squeeze_time=True,
    )
    result = _generate_mimo_channel(
        array_response_product=array_response,
        power=power,
        delay=delay,
        phase=phase,
        doppler=doppler,
        ofdm_params=ofdm_params,
        times=0.0,
        freq_domain=True,
        squeeze_time=True,
    )

    assert result.shape == reference.shape
    assert result.dtype == np.csingle
    assert np.allclose(result, reference, rtol=1e-4, atol=1e-6)
