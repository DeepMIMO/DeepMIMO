"""TX/RX metadata extraction for MATLAB RT JSON exports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from deepmimo.core.txrx import TxRxSet

from ._metadata_common import RX_SET_ID, TX_SET_ID, validate_export

if TYPE_CHECKING:
    from .schema import MatlabRTExport


def build_txrx_metadata(export: MatlabRTExport) -> dict[str, dict[str, Any]]:
    """Build DeepMIMO TX/RX set metadata for one MATLAB TX set and RX set."""
    validate_export(export)
    tx_set = TxRxSet(
        name="matlab_tx",
        id_orig=TX_SET_ID,
        id=TX_SET_ID,
        is_tx=True,
        is_rx=False,
        num_points=export.num_tx,
        num_active_points=export.num_tx,
        num_ant=1,
        dual_pol=False,
    )
    rx_set = TxRxSet(
        name="matlab_rx",
        id_orig=RX_SET_ID,
        id=RX_SET_ID,
        is_tx=False,
        is_rx=True,
        num_points=export.num_rx,
        num_active_points=export.num_rx,
        num_ant=1,
        dual_pol=False,
    )

    return {
        f"txrx_set_{TX_SET_ID}": tx_set.to_dict(),
        f"txrx_set_{RX_SET_ID}": rx_set.to_dict(),
    }
