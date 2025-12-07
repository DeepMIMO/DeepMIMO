"""Tests for DeepMIMO Channel Module."""

import pytest
import numpy as np
from deepmimo.generator.channel import ChannelParameters, OFDM_PathGenerator
from deepmimo import consts as c

def test_channel_parameters():
    cp = ChannelParameters()
    assert c.PARAMSET_ANT_ROTATION in cp.bs_antenna
    assert c.PARAMSET_ANT_SPACING in cp.bs_antenna
    # Default values
    assert cp.bs_antenna[c.PARAMSET_ANT_SPACING] == 0.5

def test_ofdm_path_generator():
    # Test parameters
    params = {
        c.PARAMSET_OFDM_SC_NUM: 64,
        c.PARAMSET_OFDM_BANDWIDTH: 10e6,
        c.PARAMSET_OFDM_SC_SAMP: np.array([0, 1, 2]), # Select 3 subcarriers
        c.PARAMSET_OFDM_LPF: 0
    }
    subcarriers = params[c.PARAMSET_OFDM_SC_SAMP]
    
    gen = OFDM_PathGenerator(params, subcarriers)
    
    # Mock path data: 2 paths
    pwr = np.array([1.0, 0.5])
    toa = np.array([0.0, 1e-7])
    phs = np.array([0.0, 90.0])
    Ts = 1.0 / params[c.PARAMSET_OFDM_BANDWIDTH]
    dopplers = np.array([0.0, 100.0])
    times = np.array([0.0]) # 1 timestamp
    
    h = gen.generate(pwr, toa, phs, Ts, dopplers, times)
    
    # Expected shape: [P, K, Nt] -> [2, 3, 1]
    assert h.shape == (2, 3, 1)
    
    # Check zero power paths (clipped)
    # If we set a delay very large
    toa_long = np.array([100.0]) # >> symbol duration
    h_long = gen.generate(np.array([1.0]), toa_long, np.array([0.0]), Ts, np.array([0.0]), times)
    assert np.allclose(h_long, 0)


