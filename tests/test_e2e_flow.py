"""End-to-end flow tests for DeepMIMO."""

import numpy as np

from deepmimo import Dataset
from deepmimo import consts as c
from deepmimo.generator.channel import ChannelParameters
from deepmimo.datasets.dataset import DynamicDataset


def test_full_generation_flow() -> None:
    """Test a complete dataset generation and channel computation flow."""
    # 1. Initialize Dataset
    ds = Dataset()
    ds.n_ue = 2
    ds.rx_pos = np.array([[10, 0, 1.5], [20, 0, 1.5]])
    ds.tx_pos = np.array([0, 0, 10])

    # Mock Scene
    class MockScene:
        def __init__(self) -> None:
            self.objects = []

    ds.scene = MockScene()
    ds.tx_vel = np.zeros(3)  # Needed if scene objects empty or for doppler check

    # 2. Setup mock channel data (normally loaded from files)
    # 2 paths per user
    ds.num_paths = np.array([2, 2])
    ds.power = np.ones((2, 2)) * -80  # -80 dBm
    ds.phase = np.zeros((2, 2))
    ds.delay = np.array([[1e-7, 2e-7], [1.5e-7, 2.5e-7]])
    ds.aod_el = np.zeros((2, 2))
    ds.aod_az = np.zeros((2, 2))
    ds.aoa_el = np.zeros((2, 2))
    ds.aoa_az = np.zeros((2, 2))
    ds.inter = np.zeros((2, 2))  # LoS

    # 3. Configure Channel Parameters
    params = ChannelParameters()
    params[c.PARAMSET_ANT_BS][c.PARAMSET_ANT_SHAPE] = np.array([4, 1, 1])  # 4 antennas
    params[c.PARAMSET_ANT_UE][c.PARAMSET_ANT_SHAPE] = np.array([1, 1, 1])  # 1 antenna
    params[c.PARAMSET_OFDM][c.PARAMSET_OFDM_SC_NUM] = 64
    params[c.PARAMSET_OFDM][c.PARAMSET_OFDM_SC_SAMP] = np.arange(10)  # first 10 SCs
    params[c.PARAMSET_OFDM][c.PARAMSET_OFDM_BANDWIDTH] = 10e6

    # 4. Compute Channels
    channel = ds.compute_channels(params)

    # Verify channel shape: [n_users, M_rx, M_tx, K_sel]
    assert channel.shape == (2, 1, 4, 10)

    # 5. Verify Path Loss Computation
    pl = ds.compute_pathloss()
    assert pl.shape == (2,)
    assert not np.isnan(pl).any()


def test_dynamic_dataset_flow() -> None:
    """Test dynamic dataset flow."""
    # Create two datasets representing time steps
    ds1 = Dataset({"name": "ds1"})
    ds1.n_ue = 1
    ds1.rx_pos = np.array([[10, 0, 0]])
    ds1.tx_pos = np.array([[0, 0, 10]])

    ds2 = Dataset({"name": "ds2"})
    ds2.n_ue = 1
    ds2.rx_pos = np.array([[11, 0, 0]])  # Moved 1m
    ds2.tx_pos = np.array([[0, 0, 10]])

    # Mock scene objects for speed computation
    class MockObjectList:
        def __init__(self) -> None:
            class Obj:
                position = np.array([0, 0, 0])
                vel = np.array([0, 0, 0])

            self.objs = [Obj()]

        @property
        def position(self):
            return [o.position for o in self.objs]

        @property
        def vel(self):
            return [o.vel for o in self.objs]

        @vel.setter
        def vel(self, v) -> None:
            for i, o in enumerate(self.objs):
                o.vel = v[i]

    ds1.scene = type("Scene", (), {"objects": MockObjectList()})()
    ds2.scene = type("Scene", (), {"objects": MockObjectList()})()

    dyn = DynamicDataset([ds1, ds2], name="dyn_test")

    # Set timestamps
    dyn.set_timestamps(1.0)  # 1s interval

    # Check speeds
    assert np.allclose(ds1.rx_vel, [[1, 0, 0]])  # 1m / 1s = 1m/s
