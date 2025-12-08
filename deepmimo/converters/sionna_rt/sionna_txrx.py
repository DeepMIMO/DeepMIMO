"""Sionna Ray Tracing TX/RX Module.

This module handles loading and converting transmitter and receiver data from Sionna's format to DeepMIMO's format.
"""

from deepmimo.txrx import TxRxSet


def read_txrx(rt_params_dict: dict) -> dict:
    """Read and convert TX/RX data from Sionna format.

    Args:
        setup_dict: Dictionary containing Sionna setup parameters

    Returns:
        Dict containing TX/RX configuration in DeepMIMO format

    """
    raw_params = rt_params_dict["raw_params"]
    txrx_dict = {}
    # Create TX and RX objects in a loop
    for i in range(2):
        is_tx = i == 0  # First iteration is TX, second is RX
        obj = TxRxSet()
        obj.is_tx = is_tx
        obj.is_rx = not is_tx

        obj.name = "tx_array" if is_tx else "rx_array"
        obj.id_orig = i
        obj.id = i  # 0-indexed

        # Set antenna properties
        obj.num_ant = 1 if rt_params_dict["synthetic_array"] else raw_params[obj.name + "_num_ant"]
        obj.ant_rel_positions = raw_params[obj.name + "_ant_pos"]
        obj.dual_pol = raw_params[obj.name + "_num_ant"] != raw_params[obj.name + "_size"]
        # num_ant refers to single polarized elements.
        # size refers to the number of elements in the array, which can be dual polarized.
        # if dual_pol is True, then num_ant = 2 * size.

        txrx_dict[f"txrx_set_{i}"] = obj.to_dict()

    return txrx_dict
