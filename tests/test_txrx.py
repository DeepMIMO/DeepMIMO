"""Tests for DeepMIMO TXRX module."""

from unittest.mock import patch

from deepmimo import consts as c
from deepmimo.txrx import (
    TxRxPair,
    TxRxSet,
    get_txrx_pairs,
    get_txrx_sets,
    print_available_txrx_pair_ids,
)


def test_txrx_set_initialization() -> None:
    """Test TxRxSet initialization."""
    ts = TxRxSet(name="BS1", id=1, is_tx=True, num_points=10)
    assert ts.name == "BS1"
    assert ts.is_tx is True
    assert ts.is_rx is False
    assert ts.num_points == 10

    d = ts.to_dict()
    assert d["name"] == "BS1"
    assert d["id"] == 1


def test_txrx_pair() -> None:
    """Test TxRxPair."""
    tx = TxRxSet(name="TX", id=0, is_tx=True)
    rx = TxRxSet(name="RX", id=1, is_rx=True)
    pair = TxRxPair(tx=tx, rx=rx, tx_idx=5)

    assert pair.tx == tx
    assert pair.rx == rx
    assert pair.tx_idx == 5
    assert pair.get_ids() == (0, 1)
    assert "TX" in str(pair)
    assert "RX" in str(pair)


@patch("deepmimo.txrx.load_dict_from_json")
@patch("deepmimo.txrx.get_params_path")
def test_get_txrx_sets(mock_get_path, mock_load_json) -> None:
    """Test getting TXRX sets from params."""
    mock_params = {
        c.TXRX_PARAM_NAME: {
            "txrx_set_0": {"name": "BS", "id": 0, "is_tx": True},
            "txrx_set_1": {"name": "UE", "id": 1, "is_rx": True},
            "other_key": "ignore",
        }
    }
    mock_load_json.return_value = mock_params

    sets = get_txrx_sets("test_scen")
    assert len(sets) == 2
    assert sets[0].name == "BS"
    assert sets[1].name == "UE"


def test_get_txrx_pairs() -> None:
    """Test pairing logic."""
    tx1 = TxRxSet(name="TX1", id=0, is_tx=True, num_points=2)
    tx2 = TxRxSet(name="TX2", id=1, is_tx=True, num_points=1)
    rx1 = TxRxSet(name="RX1", id=2, is_rx=True)

    # TX1 has 2 points -> 2 pairs with RX1
    # TX2 has 1 point -> 1 pair with RX1
    # Total 3 pairs

    pairs = get_txrx_pairs([tx1, tx2, rx1])
    assert len(pairs) == 3

    # Check pairs content
    # Pair 0: TX1[0] -> RX1
    assert pairs[0].tx == tx1
    assert pairs[0].tx_idx == 0
    assert pairs[0].rx == rx1
    # Pair 1: TX1[1] -> RX1
    assert pairs[1].tx == tx1
    assert pairs[1].tx_idx == 1
    assert pairs[1].rx == rx1
    # Pair 2: TX2[0] -> RX1
    assert pairs[2].tx == tx2
    assert pairs[2].tx_idx == 0
    assert pairs[2].rx == rx1


@patch("deepmimo.txrx.get_txrx_sets")
def test_print_pairs(mock_get_sets, capsys) -> None:
    """Test printing pairs."""
    tx = TxRxSet(name="TX", id=0, is_tx=True, num_points=1)
    rx = TxRxSet(name="RX", id=1, is_rx=True)
    mock_get_sets.return_value = [tx, rx]

    print_available_txrx_pair_ids("test")
    captured = capsys.readouterr()

    assert "TX/RX Pair IDs" in captured.out
    assert "0" in captured.out  # tx id
    assert "1" in captured.out  # rx id


def test_txrx_set_repr_tx_only() -> None:
    """Test __repr__ for TX-only set."""
    ts = TxRxSet(name="BS1", id=1, is_tx=True, is_rx=False, num_points=10)
    repr_str = repr(ts)
    assert "TXSet" in repr_str
    assert "BS1" in repr_str
    assert "points=10" in repr_str


def test_txrx_set_repr_rx_only() -> None:
    """Test __repr__ for RX-only set."""
    ts = TxRxSet(name="UE1", id=2, is_tx=False, is_rx=True, num_points=5)
    repr_str = repr(ts)
    assert "RXSet" in repr_str
    assert "UE1" in repr_str
    assert "points=5" in repr_str


def test_txrx_set_repr_both() -> None:
    """Test __repr__ for set that is both TX and RX."""
    ts = TxRxSet(name="Relay", id=3, is_tx=True, is_rx=True, num_points=1)
    repr_str = repr(ts)
    assert "TXRXSet" in repr_str
    assert "Relay" in repr_str


def test_txrx_set_repr_neither() -> None:
    """Test __repr__ for set that is neither TX nor RX (edge case)."""
    ts = TxRxSet(name="Unknown", id=4, is_tx=False, is_rx=False, num_points=0)
    repr_str = repr(ts)
    assert "UnknownSet" in repr_str
    assert "Unknown" in repr_str
