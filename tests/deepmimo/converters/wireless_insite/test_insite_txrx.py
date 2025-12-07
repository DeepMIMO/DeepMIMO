"""Tests for Wireless Insite TXRX."""

import numpy as np
from unittest.mock import MagicMock, patch, mock_open
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

@patch("deepmimo.converters.wireless_insite.insite_txrx.parse_insite_xml")
@patch("deepmimo.converters.wireless_insite.insite_txrx.Path")
def test_read_txrx(mock_path, mock_parse_xml):
    """Test reading TXRX sets from XML."""
    # Mock glob to return a file
    mock_path.return_value.glob.return_value = ["test.xml"]
    
    # Mock parse_insite_xml return
    mock_parse_xml.return_value = {
        "remcom_rxapi_Job": {
            "Scene": {
                "remcom_rxapi_Scene": {
                    "TxRxSetList": {
                        "remcom_rxapi_TxRxSetList": {
                            "TxRxSet": [
                                # TX Set (Point)
                                {
                                    "remcom_rxapi_TxSet": {
                                        "OutputID": {"remcom_rxapi_Integer": 1},
                                        "ShortDescription": {"remcom_rxapi_String": "TxSet1"},
                                        "ConformToTerrain": {"remcom_rxapi_Boolean": False},
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
                                        "TransmitterList": {
                                            "remcom_rxapi_TransmitterList": {
                                                "Transmitter": [{
                                                    "Point": {
                                                        "remcom_rxapi_CartesianPoint": {
                                                            "X": {"remcom_rxapi_Double": 0},
                                                            "Y": {"remcom_rxapi_Double": 0},
                                                            "Z": {"remcom_rxapi_Double": 0}
                                                        }
                                                    },
                                                    "AntennaID": {"remcom_rxapi_Integer": 1},
                                                    "Orientation": {"remcom_rxapi_Rotation": {"P1": {"remcom_rxapi_Double": 0}}}
                                                }]
                                            }
                                        }
                                    }
                                },
                                # RX Set (Grid)
                                {
                                    "remcom_rxapi_GridSet": { # Using GridSet for grid
                                        "OutputID": {"remcom_rxapi_Integer": 2},
                                        "ShortDescription": {"remcom_rxapi_String": "RxGrid1"},
                                        "ConformToTerrain": {"remcom_rxapi_Boolean": False},
                                        "UseAPGAcceleration": {"remcom_rxapi_Boolean": False},
                                        "LengthX": {"remcom_rxapi_Double": 10},
                                        "LengthY": {"remcom_rxapi_Double": 10},
                                        "Spacing": {"remcom_rxapi_Double": 10},
                                        "ControlPoints": {
                                            "remcom_rxapi_ProjectedPointList": {
                                                "ProjectedPoint": [{
                                                    "remcom_rxapi_CartesianPoint": {"X": {"remcom_rxapi_Double": 0}, "Y": {"remcom_rxapi_Double": 0}, "Z": {"remcom_rxapi_Double": 0}}
                                                }]
                                            }
                                        },
                                        "Receiver": {
                                            "remcom_rxapi_Receiver": {
                                                "Antenna": { "remcom_rxapi_Isotropic": {
                                                    "Polarization": {"remcom_rxapi_PolarizationEnum": "Vertical"},
                                                    "PowerThreshold": {"remcom_rxapi_Double": -250}
                                                }},
                                                "AntennaRotations": { "remcom_rxapi_Rotations": {
                                                    "Bearing": {"remcom_rxapi_Double": 0},
                                                    "Pitch": {"remcom_rxapi_Double": 0},
                                                    "Roll": {"remcom_rxapi_Double": 0}
                                                }}
                                            }
                                        }
                                    }
                                }
                            ]
                        }
                    }
                }
            },
            "Waveforms": {
                "remcom_rxapi_WaveformList": {
                    "Waveform": [{"remcom_rxapi_Sinusoid": {"Frequency": {"remcom_rxapi_Double": 28e9}}}]
                }
            }
        }
    }
    
    tx_set_data = mock_parse_xml.return_value["remcom_rxapi_Job"]["Scene"]["remcom_rxapi_Scene"]["TxRxSetList"]["remcom_rxapi_TxRxSetList"]["TxRxSet"][0]["remcom_rxapi_TxSet"]
    tx_set_data["Transmitter"] = {
        "remcom_rxapi_Transmitter": {
            "Antenna": { "remcom_rxapi_Isotropic": {
                "Polarization": {"remcom_rxapi_PolarizationEnum": "Vertical"},
                "PowerThreshold": {"remcom_rxapi_Double": -250}
            }},
            "AntennaRotations": { "remcom_rxapi_Rotations": {
                "Bearing": {"remcom_rxapi_Double": 0},
                "Pitch": {"remcom_rxapi_Double": 0},
                "Roll": {"remcom_rxapi_Double": 0}
            }}
        }
    }

    # Mock plotting function to avoid TclError
    with patch("deepmimo.converters.wireless_insite.insite_txrx.plot_txrx_sets") as mock_plot:
        sets = insite_txrx.read_txrx("rt_folder", plot=True)
        assert mock_plot.called
    
    # Should have 2 sets (1 TX, 1 RX)
    assert len(sets) == 2
    assert sets["txrx_set_0"]["is_tx"]
    assert sets["txrx_set_1"]["is_rx"]
