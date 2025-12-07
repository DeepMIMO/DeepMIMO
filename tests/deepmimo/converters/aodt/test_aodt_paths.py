"""Tests for AODT Paths."""

import numpy as np
from deepmimo.converters.aodt import aodt_paths
from deepmimo import consts as c

def test_transform_interaction_types():
    # LoS: only emission and reception
    types_los = np.array(["emission", "reception"], dtype=object)
    assert aodt_paths._transform_interaction_types(types_los) == c.INTERACTION_LOS
    
    # Reflection
    types_ref = np.array(["emission", "reflection", "reception"], dtype=object)
    assert aodt_paths._transform_interaction_types(types_ref) == 1.0
    
    # Ref + Diff
    types_mixed = np.array(["emission", "reflection", "diffraction", "reception"], dtype=object)
    assert aodt_paths._transform_interaction_types(types_mixed) == 12.0

