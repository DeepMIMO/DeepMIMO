import pytest
import numpy as np
from deepmimo.generator.channel import (
    ChannelParameters,
    _generate_MIMO_channel,
    _convert_lists_to_arrays,
)
from deepmimo import consts as c


def test_channel_parameters_defaults():
    """Test default channel parameters"""
    params = ChannelParameters()
    assert c.PARAMSET_ANT_BS in params
    assert c.PARAMSET_ANT_UE in params
    assert params[c.PARAMSET_ANT_BS][c.PARAMSET_ANT_SPACING] == 0.5


def test_channel_parameters_validation():
    """Test parameter validation"""
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
    with pytest.raises(ValueError):
        params.validate(n_ues=10)


def test_convert_lists_to_arrays():
    """Test helper for converting lists to arrays"""
    data = {"a": [1, 2, 3], "b": {"c": [4, 5]}}
    converted = _convert_lists_to_arrays(data)
    assert isinstance(converted["a"], np.ndarray)
    assert isinstance(converted["b"]["c"], np.ndarray)


def test_generate_mimo_channel_time_domain():
    """Test channel generation in time domain"""
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
    channel = _generate_MIMO_channel(
        array_response, powers, delays, phases, dopplers, ofdm_params, times=0.0, freq_domain=False
    )

    # Expected shape: [n_users, M_rx, M_tx, P_max] (squeezed time)
    assert channel.shape == (n_users, n_rx_ant, n_tx_ant, n_paths)

    # Check values: should be close to sqrt(1) * 1 * 1 = 1 (since power=1, phase=0)
    # Actually power is linear here.
    assert np.allclose(np.abs(channel), 1.0)


def test_generate_mimo_channel_freq_domain():
    """Test channel generation in frequency domain (OFDM)"""
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

    channel = _generate_MIMO_channel(
        array_response, powers, delays, phases, dopplers, ofdm_params, times=0.0, freq_domain=True
    )

    # Expected shape: [n_users, M_rx, M_tx, K_sel]
    assert channel.shape == (n_users, n_rx_ant, n_tx_ant, 2)

    # Check values. Power splits over total subcarriers in OFDM usually?
    # In generate_MIMO_channel:
    # a_pt = sqrt(power / total_subcarriers) ...
    # So magnitude should be related to total_subcarriers.
    total_sc = ofdm_params[c.PARAMSET_OFDM_SC_NUM]
    expected_mag = np.sqrt(1.0 / total_sc)

    # Allow some tolerance for floating point
    assert np.allclose(np.abs(channel), expected_mag, rtol=1e-5)


def test_doppler_progression():
    """Test that channel evolves with time due to Doppler"""
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

    channel = _generate_MIMO_channel(
        array_response,
        powers,
        delays,
        phases,
        dopplers,
        ofdm_params,
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
