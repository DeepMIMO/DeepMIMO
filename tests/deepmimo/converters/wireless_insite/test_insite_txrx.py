"""Tests for Wireless Insite TXRX."""

import numpy as np
from unittest.mock import MagicMock
from deepmimo.converters.wireless_insite import insite_txrx
from deepmimo.txrx import TxRxSet

def test_insite_txrx_set_grid():
    data = {
        "ConformToTerrain": {"remcom_rxapi_Boolean": True},
        "OutputID": {"remcom_rxapi_Integer": 1},
        "ShortDescription": {"remcom_rxapi_String": "Grid1"},
        "UseAPGAcceleration": {"remcom_rxapi_Boolean": False},
        "ControlPoints": {
            "remcom_rxapi_ProjectedPointList": {
                "ProjectedPoint": [{
                    "remcom_rxapi_CartesianPoint": {
                        "X": {"remcom_rxapi_Double": 0},
                        "Y": {"remcom_rxapi_Double": 0},
                        "Z": {"remcom_rxapi_Double": 0}
                    }
                }]
            }
        },
        "LengthX": {"remcom_rxapi_Double": 10},
        "LengthY": {"remcom_rxapi_Double": 10},
        "Spacing": {"remcom_rxapi_Double": 5}
    }
    
    txrx = insite_txrx.InSiteTxRxSet.from_dict(data, "grid")
    assert txrx.set_type == "grid"
    points = txrx.generate_points()
    # 0 to 10 step 5 -> 0, 5, 10 (3 points). 3x3=9 points.
    assert len(points) == 9

def test_convert_sets_to_deepmimo():
    # Mock InSiteTxRxSet
    iset = MagicMock(spec=insite_txrx.InSiteTxRxSet)
    iset.data = {} # Explicitly add data attribute as dict
    iset.generate_points.return_value = np.zeros((1, 3))
    iset.transmitter = {"antenna": "A", "rotations": "R"}
    iset.receiver = {"antenna": "A", "rotations": "R"} # Same antennas -> 1 set
    iset.copy.return_value = iset
    
    dm_set = TxRxSet(id=0, name="S", id_orig=0, is_tx=True, is_rx=True, num_points=1, num_active_points=1, num_ant=1, dual_pol=False, array_orientation=[0,0,0])
    iset.to_deepmimo_txrxset.return_value = dm_set
    
    sets, locs = insite_txrx.convert_sets_to_deepmimo([iset])
    assert len(sets) == 1
    assert 0 in locs
    
    # Different antennas -> split
    iset.receiver = {"antenna": "B", "rotations": "R"}
    iset2 = MagicMock()
    iset.copy.return_value = iset2
    iset2.to_deepmimo_txrxset.return_value = dm_set # Mock return
    
    sets, locs = insite_txrx.convert_sets_to_deepmimo([iset])
    assert len(sets) == 2 # 1 TX-only, 1 RX-only (modified iset)

