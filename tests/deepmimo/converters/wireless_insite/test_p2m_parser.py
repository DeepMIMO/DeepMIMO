import pytest
import numpy as np
import os
from deepmimo.converters.wireless_insite import p2m_parser

def test_read_pl_p2m_file(tmp_path):
    """Test reading path loss p2m file."""
    # Use implicit concatenation for lines to avoid indentation issues
    file_content = (
        "# <Transmitter Set: Tx: 1 BS - Point 1>\n"
        "# <Receiver Set: Rx: 2 ue_grid>\n"
        "# <X(m)> <Y(m)> <Z(m)> <Distance(m)> <PathLoss(dB)>"
    )
    file_content += "".join(
        [f"\n{i} -90 -60 1.5 96.0 108.0" for i in range(1, 11)]
    )

    test_file = tmp_path / "test.pl.p2m"
    with open(test_file, "w") as fp:
        fp.write(file_content)

    xyz, dist, pl = p2m_parser.read_pl_p2m_file(str(test_file))
    
    assert xyz.shape == (10, 3)
    assert dist.shape == (10, 1)
    assert pl.shape == (10, 1)
    assert xyz[0, 0] == -90.0
    assert dist[0, 0] == 96.0
    assert pl[0, 0] == 108.0

