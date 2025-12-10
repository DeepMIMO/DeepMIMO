"""TX/RX set handling for DeepMIMO.

This module provides the base TxRxSet class used by all ray tracing converters
to represent transmitter and receiver configurations.
"""

from dataclasses import asdict, dataclass, field

from deepmimo import consts as c
from deepmimo.utils import get_params_path, load_dict_from_json


@dataclass
class TxRxSet:
    """Base TX/RX set class for DeepMIMO.

    This class represents a set of transmitters or receivers in a ray tracing simulation.
    It is used by all ray tracing converters (Wireless Insite, Sionna, etc.) to store
    TX/RX configuration information during conversion.

    Example:
        Wireless Insite IDs = [3, 7, 8]
        DeepMIMO (after conversion) TX/RX Sets: [0, 1, 2]
        DeepMIMO (after generation): only individual tx and rx indices

    """

    name: str = ""
    id_orig: int = 0  # Original ray tracer txrx ID (mainly for Wireless Insite)
    id: int = 0  # TxRxSet index for saving after conversion and generation
    is_tx: bool = False
    is_rx: bool = False

    num_points: int = 0  # all points
    num_active_points: int = 0  # number of points with at least one path

    num_ant: int = 1  # number of antenna elements
    dual_pol: bool = False  # if antenna supports dual polarization

    # For forward compatibility (only single antenna supported for now)
    ant_rel_positions: list = field(
        default_factory=lambda: [[0, 0, 0]],
    )  # relative to the center of the antenna
    array_orientation: list = field(default_factory=lambda: [0, 0, 0])  # [azimuth, elevation, roll]

    def to_dict(self) -> dict:
        """Convert TxRxSet to a dictionary.

        Returns:
            Dictionary containing all attributes of the TxRxSet.

        """
        return asdict(self)

    def __repr__(self) -> str:
        """Return string representation of TxRxSet."""
        role = "TX" if self.is_tx else ""
        role += "RX" if self.is_rx else ""
        role = role if role else "Unknown"
        return f"{role}Set(name='{self.name}', id={self.id}, points={self.num_points})"


@dataclass
class TxRxPair:
    """Represents a pair of transmitter and receiver in a ray tracing simulation.

    This class is used to store the configuration of a transmitter and receiver pair,
    including their positions, antenna properties, and other relevant information.

    """

    tx: TxRxSet = field(default_factory=TxRxSet)
    rx: TxRxSet = field(default_factory=TxRxSet)
    tx_idx: int = 0

    def __repr__(self) -> str:
        """Return string representation of TxRxPair."""
        return f"TxRxPair(tx={self.tx.name}[{self.tx_idx}], rx={self.rx.name})"

    def get_ids(self) -> tuple[int, int]:
        """Get the IDs of the transmitter and receiver."""
        return self.tx.id, self.rx.id


def get_txrx_sets(scenario_name: str) -> list[TxRxSet]:
    """Get all available transmitter-receiver sets for a given scenario.

    This function reads the scenario parameters file and creates a list of TxRxSet objects
    based on the available transmitter and receiver sets.

    Args:
        scenario_name: Name of the DeepMIMO scenario

    Returns:
        List of TxRxSet objects representing all available transmitter-receiver sets

    Example:
        >>> txrx_sets = dm.get_txrx_sets('O1_60')
        >>> for txrx_set in txrx_sets:
        ...     print(txrx_set)
        TxRxSet(name='tx_array', idx=1, points=2)
        TxRxSet(name='rx_array', idx=2, points=1)

    """
    # Load parameters file
    params_file = get_params_path(scenario_name)
    params = load_dict_from_json(params_file)

    # Extract TxRxSets from parameters
    txrx_sets = []
    for key, val in params[c.TXRX_PARAM_NAME].items():
        if key.startswith("txrx_set_"):
            txrx_sets.append(TxRxSet(**val))

    return txrx_sets


def get_txrx_pairs(txrx_sets: list[TxRxSet]) -> list[TxRxPair]:
    """Create all possible transmitter-receiver pairs from a list of TxRxSet objects.

    This function pairs all transmitter sets with all receiver sets, decoupling
    individual transmitters from sets that have multiple transmitters.

    Args:
        txrx_sets: List of TxRxSet objects to create pairs from

    Returns:
        List of TxRxPair objects representing all valid transmitter-receiver combinations

    Example:
        If we have:
        - TxRxSet1 with 2 TXs
        - TxRxSet2 with 1 TX
        - TxRxSet3 with only RXs

        The function will create pairs:
        - TX1 from Set1 with all RXs from Set3
        - TX2 from Set1 with all RXs from Set3
        - TX from Set2 with all RXs from Set3

    """
    tx_sets = [s for s in txrx_sets if s.is_tx]
    rx_sets = [s for s in txrx_sets if s.is_rx]

    pairs = []

    # For each TX set
    for tx_set in tx_sets:
        pairs.extend(
            TxRxPair(tx=tx_set, rx=rx_set, tx_idx=tx_idx)
            for tx_idx in range(tx_set.num_points)
            for rx_set in rx_sets
        )

    return pairs


def print_available_txrx_pair_ids(scenario_name: str) -> None:
    """Print all available transmitter-receiver pair IDs for a given scenario.

    This function reads the scenario parameters file and prints the IDs of all
    available transmitter-receiver pairs.

    Args:
        scenario_name: Name of the DeepMIMO scenario

    """
    txrx_sets = get_txrx_sets(scenario_name)
    pairs = get_txrx_pairs(txrx_sets)

    print("\nTX/RX Pair IDs")
    print("-" * 25)
    print(f"{'Pair':^6} | {'TX ID':^6} | {'RX ID':^6}")
    print("-" * 25)
    for idx, pair in enumerate(pairs):
        tx_id, rx_id = pair.get_ids()
        print(f"{idx:^6} | {tx_id:^6} | {rx_id:^6}")
    print("-" * 25)

