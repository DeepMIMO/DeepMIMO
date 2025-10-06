"""
Test module for Sionna Ray Tracing converter functionality.

This module contains tests for the Sionna RT converter, particularly focusing on
the interaction type conversion between Sionna and DeepMIMO formats.
"""
#%%
import numpy as np
from deepmimo.converters.sionna_rt.sionna_converter import get_sionna_interaction_types
from deepmimo import consts as c


def test_get_sionna_interaction_types():
    """Test conversion of Sionna interaction types to DeepMIMO codes."""
    
    print("\n=== Testing Sionna to DeepMIMO Interaction Type Mapping ===")
    print("\nSionna interaction codes:")
    print("0 = Line of Sight (LoS)")
    print("1 = Reflection(s)")
    print("2 = Single Diffraction")
    print("3 = Scattering (possibly with reflections)")
    print("4 = RIS (not supported)")
    
    # Test data dimensions
    n_users = 3
    max_paths = 4
    max_interactions = 5
    
    # Create test types array (N_USERS x MAX_PATHS)
    types = np.array([
        # User 1: LoS, 1 reflection, 2 reflections, diffraction
        [0, 1, 1, 2],
        # User 2: scattering, 1 refl + scattering, RIS (4), NaN
        [3, 3, 4, np.nan],
        # User 3: 3 reflections, LoS, 2 refl + scattering, NaN
        [1, 0, 3, np.nan]
    ], dtype=np.float32)
    
    print("\nInput Sionna types matrix:")
    print(types)
    print("\nDetailed explanation of test cases:")
    
    print("\nUser 1 paths:")
    print("Path 1: Sionna 0 (LoS)              -> DeepMIMO 0")
    print("Path 2: Sionna 1 (1 reflection)     -> DeepMIMO 1")
    print("Path 3: Sionna 1 (2 reflections)    -> DeepMIMO 11")
    print("Path 4: Sionna 2 (diffraction)      -> DeepMIMO 2")
    
    print("\nUser 2 paths:")
    print("Path 1: Sionna 3 (scattering)       -> DeepMIMO 3")
    print("Path 2: Sionna 3 (refl + scat)      -> DeepMIMO 13")
    print("Path 3: Sionna 4 (RIS)             -> Error/NaN")
    print("Path 4: NaN (no path)              -> DeepMIMO 0")
    
    print("\nUser 3 paths:")
    print("Path 1: Sionna 1 (3 reflections)    -> DeepMIMO 111")
    print("Path 2: Sionna 0 (LoS)              -> DeepMIMO 0")
    print("Path 3: Sionna 3 (2 refl + scat)    -> DeepMIMO 113")
    print("Path 4: NaN (no path)              -> DeepMIMO 0")
    
    # Create a copy without RIS for main test
    types_no_ris = types.copy()
    types_no_ris[1, 2] = np.nan  # Replace RIS with NaN
    
    print("\nInput types after removing RIS:")
    print(types_no_ris)
    
    # Create test interaction positions (N_USERS x MAX_PATHS x MAX_INTERACTIONS x 3)
    inter_pos = np.zeros((n_users, max_paths, max_interactions, 3)) * np.nan
    
    print("\nSetting up test paths:")
    # User 1
    print("\nUser 1:")
    print("- Path 1: LoS (no interactions)")
    print("- Path 2: Single reflection")
    inter_pos[0, 1, 0] = [1, 1, 1]
    print("- Path 3: Two reflections")
    inter_pos[0, 2, 0:2] = [[1, 1, 1], [2, 2, 2]]
    print("- Path 4: Single diffraction")
    inter_pos[0, 3, 0] = [3, 3, 3]
    
    # User 2
    print("\nUser 2:")
    print("- Path 1: Single scattering")
    inter_pos[1, 0, 0] = [4, 4, 4]
    print("- Path 2: One reflection + scattering")
    inter_pos[1, 1, 0:2] = [[5, 5, 5], [6, 6, 6]]
    print("- Path 3: RIS (will be skipped)")
    inter_pos[1, 2, 0] = [7, 7, 7]
    print("- Path 4: NaN (no path)")
    
    # User 3
    print("\nUser 3:")
    print("- Path 1: Three reflections")
    inter_pos[2, 0, 0:3] = [[8, 8, 8], [9, 9, 9], [10, 10, 10]]
    print("- Path 2: LoS (no interactions)")
    print("- Path 3: Two reflections + scattering")
    inter_pos[2, 2, 0:3] = [[11, 11, 11], [12, 12, 12], [13, 13, 13]]
    print("- Path 4: NaN (no path)")
    
    # Expected output
    expected = np.zeros((n_users, max_paths), dtype=np.float32)
    # User 1
    expected[0, 0] = c.INTERACTION_LOS  # LoS
    expected[0, 1] = 1  # Single reflection
    expected[0, 2] = 11  # Two reflections
    expected[0, 3] = c.INTERACTION_DIFFRACTION  # Single diffraction
    
    # User 2
    expected[1, 0] = 3  # Single scattering
    expected[1, 1] = 13  # One reflection + scattering
    expected[1, 2] = 0  # RIS (not supported)
    expected[1, 3] = 0  # NaN path
    
    # User 3
    expected[2, 0] = 111  # Three reflections
    expected[2, 1] = c.INTERACTION_LOS  # LoS
    expected[2, 2] = 113  # Two reflections + scattering
    expected[2, 3] = 0  # NaN path
    
    print("\nExpected DeepMIMO interaction codes:")
    print(expected)
    
    # Test main functionality (without RIS)
    print("\nTesting main functionality (without RIS)...")
    result = get_sionna_interaction_types(types_no_ris, inter_pos)
    print("\nActual DeepMIMO interaction codes:")
    print(result)
    
    print("\nComparing results...")
    np.testing.assert_array_almost_equal(result, expected)
    print("Main test passed successfully! Results match expected values.")
    
    # Test RIS error separately
    print("\nTesting RIS error handling...")
    try:
        get_sionna_interaction_types(types, inter_pos)
        print("WARNING: RIS error test failed - expected NotImplementedError")
    except NotImplementedError as e:
        print(f"RIS error test passed successfully! Error: {str(e)}")


def test_edge_cases():
    """Test edge cases for interaction type conversion."""
    
    print("\n=== Testing Edge Cases ===")
    
    # Test empty arrays
    print("\nTesting empty arrays:")
    types = np.zeros((0, 5), dtype=np.float32)
    inter_pos = np.zeros((0, 5, 3, 3), dtype=np.float32)
    print("Input shape:", types.shape)
    result = get_sionna_interaction_types(types, inter_pos)
    print("Output shape:", result.shape)
    assert result.shape == (0, 5), "Empty array test failed"
    print("Empty array test passed successfully!")
    
    # Test all NaN
    print("\nTesting all NaN values:")
    types = np.full((2, 3), np.nan, dtype=np.float32)
    inter_pos = np.full((2, 3, 4, 3), np.nan, dtype=np.float32)
    print("Input:\n", types)
    result = get_sionna_interaction_types(types, inter_pos)
    print("Output:\n", result)
    assert np.all(result == 0), "All NaN test failed"
    print("All NaN test passed successfully!")
    
    # Test all zeros
    print("\nTesting all zeros:")
    types = np.zeros((2, 3), dtype=np.float32)
    inter_pos = np.zeros((2, 3, 4, 3), dtype=np.float32)
    print("Input:\n", types)
    result = get_sionna_interaction_types(types, inter_pos)
    print("Output:\n", result)
    assert np.all(result == 0), "All zeros test failed"
    print("All zeros test passed successfully!")


if __name__ == '__main__':
    print("\nRunning main interaction type tests...")
    test_get_sionna_interaction_types()
    
    print("\nRunning edge case tests...")
    test_edge_cases()
    
    print("\nAll tests completed successfully!") 