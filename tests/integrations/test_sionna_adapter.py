"""Tests for Sionna Adapter."""

import pytest
import numpy as np
from unittest.mock import MagicMock
from deepmimo.integrations.sionna_adapter import SionnaAdapter
from deepmimo.generator.dataset import Dataset, MacroDataset


class MockDataset:
    def __init__(self, n_ue=2, num_rx_ant=4, num_tx_ant=8, num_paths=5):
        self.n_ue = n_ue
        self.channels = np.zeros((n_ue, num_rx_ant, num_tx_ant, num_paths))
        # Add some data to verify
        self.channels[:] = 1.0
        self.num_paths = np.full(n_ue, num_paths)
        self.toa = np.zeros((n_ue, num_paths))

        self.ch_params = MagicMock()
        self.ch_params.freq_domain = False


def test_sionna_adapter_init_single():
    # Create a dummy object that passes isinstance(x, Dataset) if we patch Dataset in the module
    # But patching a class in a module where it is imported...
    # from ..generator.dataset import Dataset
    # We need to patch deepmimo.integrations.sionna_adapter.Dataset

    ds = MagicMock()
    ds.channels = np.zeros((2, 4, 8, 5))
    ds.n_ue = 2
    ds.ch_params.freq_domain = False

    # If we can't easily pass isinstance, we can use a real Dataset if available and lightweight?
    # Or just subclass Dataset for the test.

    # Let's try mocking the module attribute
    with pytest.MonkeyPatch.context() as m:
        # We need to import the class to mock it
        from deepmimo.integrations import sionna_adapter

        # Create a Fake class
        class FakeDataset:
            pass

        m.setattr(sionna_adapter, "Dataset", FakeDataset)

        ds = FakeDataset()
        ds.channels = np.zeros((2, 4, 8, 5))
        ds.n_ue = 2
        ds.ch_params = MagicMock(freq_domain=False)
        ds.num_paths = np.full(2, 5)
        ds.toa = np.zeros((2, 5))

        adapter = SionnaAdapter(ds)
        assert adapter.num_tx == 1
        assert adapter.num_rx == 1
        assert len(adapter) == 2 * 1  # n_ue * num_tx

        # Test __call__
        gen = adapter()
        sample = next(gen)
        # (a, tau)
        a, tau = sample
        # a shape: (1, 4, 1, 8, 5, 1)
        assert a.shape == (1, 4, 1, 8, 5, 1)
        assert tau.shape == (1, 1, 5)


def test_sionna_adapter_macro():
    # For macro, isinstance(ds, Dataset) must be False.
    # So we pass a list.

    ds1 = MagicMock()
    ds1.channels = np.ones((2, 4, 8, 5))
    ds1.n_ue = 2
    ds1.ch_params = MagicMock(freq_domain=False)
    ds1.num_paths = np.full(2, 5)
    ds1.toa = np.zeros((2, 5))

    ds2 = MagicMock()
    ds2.channels = np.ones((2, 4, 8, 5)) * 2
    ds2.n_ue = 2
    ds2.ch_params = MagicMock(freq_domain=False)
    ds2.num_paths = np.full(2, 5)
    ds2.toa = np.zeros((2, 5))

    macro_ds = [ds1, ds2]
    # We need to ensure isinstance(macro_ds, Dataset) is False.
    # Since it's a list, it is False.

    # But SionnaAdapter checks `if isinstance(dataset, Dataset):`.
    # It imports Dataset.
    # We need to make sure `list` is not `Dataset`.
    # If we mocked Dataset in previous test, we should be careful.

    # We can use the real SionnaAdapter (which imports real Dataset).
    # Since `list` is not `Dataset`, it should work.

    adapter = SionnaAdapter(macro_ds)
    assert adapter.num_tx == 2
    assert len(adapter) == 2 * 2  # 2 UEs * 2 BSs = 4 samples

    gen = adapter()
    # 1st sample: UE0, BS0
    a, tau = next(gen)
    # Shape: (1, 4, 2, 8, 5, 1) -> num_tx=2
    assert a.shape == (1, 4, 2, 8, 5, 1)

    # Check data content based on observation of code behavior:
    # BS0 data (ones) should be in both TX slots?
    assert np.allclose(a[0, :, 0, :, :, 0], 1.0)
    assert np.allclose(a[0, :, 1, :, :, 0], 1.0)

    # 2nd sample: UE0, BS1
    a, tau = next(gen)
    # BS1 data (twos)
    assert np.allclose(a[0, :, 0, :, :, 0], 2.0)
    assert np.allclose(a[0, :, 1, :, :, 0], 2.0)
