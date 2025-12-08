"""Tests for Wireless Insite P2M parser."""

from pathlib import Path

import numpy as np
import pytest

from deepmimo import consts as c
from deepmimo.converters.wireless_insite import p2m_parser


@pytest.fixture
def sample_paths_file(tmp_path):
    """Create a sample .paths.p2m file."""
    file_path = tmp_path / "test.paths.p2m"
    with Path(file_path).open("w") as f:
        # 21 header lines
        f.writelines(f"# Header {i + 1}\n" for i in range(21))

        # Line 22: n_rxs = 2
        f.write("2\n")

        # RX 0: 1 path
        f.write("0 1\n")  # N
        f.write("# RX Header 1\n")  # N+1

        # Path 1
        # id total_inter power phase delay aoa_el aoa_az aod_el aod_az
        # 1 interaction
        f.write("1 1 -80.5 45.0 1.2e-6 30.0 60.0 120.0 -10.0\n")  # N+2
        f.write("Tx-R-Rx\n")  # N+3
        f.write("0 0 10\n")  # N+4 (TX Pos - skipped by paths_parser, read by extract_tx_pos)
        # Interaction (Reflection)
        f.write("50 50 5\n")  # N+5
        # RX Pos (Missing in previous version)
        f.write("100 0 1.5\n")  # N+6

        # RX 1: 0 paths
        f.write("1 0\n")  # N+7

    return str(file_path)


@pytest.fixture
def sample_pl_file(tmp_path):
    """Create a sample .pl.p2m file."""
    file_path = tmp_path / "test.pl.p2m"
    with Path(file_path).open("w") as f:
        f.write("# Header\n")
        # idx X Y Z dist loss
        f.write("0 10.0 20.0 30.0 100.0 80.0\n")
        f.write("1 15.0 25.0 35.0 110.0 85.0\n")
    return str(file_path)


def test_paths_parser(sample_paths_file) -> None:
    """Test parsing of paths file."""
    data = p2m_parser.paths_parser(sample_paths_file)

    # Check dimensions
    # n_rxs = 2 (from file)
    assert data[c.POWER_PARAM_NAME].shape[0] == 2

    # Check RX 0 (has path)
    assert data[c.POWER_PARAM_NAME][0, 0] == np.float32(-80.5)
    assert data[c.DELAY_PARAM_NAME][0, 0] == np.float32(1.2e-6)
    assert data[c.INTERACTIONS_PARAM_NAME][0, 0] == np.float32(1.0)  # 'R' -> 1

    # Check Interaction Position
    # Should correspond to "50 50 5"
    assert np.allclose(data[c.INTERACTIONS_POS_PARAM_NAME][0, 0, 0], [50, 50, 5])


def test_extract_tx_pos(sample_paths_file) -> None:
    """Test extracting TX position."""
    tx_pos = p2m_parser.extract_tx_pos(sample_paths_file)
    assert np.allclose(tx_pos, [0, 0, 10])


def test_read_pl_p2m_file(sample_pl_file) -> None:
    """Test reading PL file."""
    xyz, dist, pl = p2m_parser.read_pl_p2m_file(sample_pl_file)

    assert xyz.shape == (2, 3)
    assert dist.shape == (2, 1)
    assert pl.shape == (2, 1)

    assert np.allclose(xyz[0], [10.0, 20.0, 30.0])
    assert dist[0] == 100.0
    assert pl[0] == 80.0
