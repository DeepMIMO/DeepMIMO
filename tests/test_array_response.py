"""Tests for array response functions in geometry module."""
#%%
import numpy as np
import time
from deepmimo.generator.python.geometry import array_response, array_response_batch, ant_indices


def test_array_response_batch_matches_single():
    """Test that batched array response matches single user version."""
    print("\nTesting batch vs single user array response matching...")
    
    # Test parameters
    panel_size = (4, 2)  # 8 antennas
    kd = 2 * np.pi * 0.5  # typical value
    n_users = 10
    n_paths = 5
    
    # Generate random angles (some NaN)
    rng = np.random.default_rng(42)
    theta = rng.uniform(0, np.pi, size=(n_users, n_paths))
    phi = rng.uniform(0, 2*np.pi, size=(n_users, n_paths))
    
    # Add some NaN values randomly
    nan_mask = rng.random(size=(n_users, n_paths)) < 0.2
    theta[nan_mask] = np.nan
    phi[nan_mask] = np.nan
    
    # Get antenna indices
    ant_ind = ant_indices(panel_size)
    
    # Calculate responses using batch function
    batch_responses = array_response_batch(ant_ind, theta, phi, kd)
    # Reshape batch responses to match single response shape
    batch_responses = np.moveaxis(batch_responses, 1, 2)  # [batch_size, n_paths, N]
    
    # Calculate responses one by one
    n_ant = len(ant_ind)
    single_responses = np.zeros((n_users, n_paths, n_ant), dtype=np.complex128)
    for i in range(n_users):
        for j in range(n_paths):
            if not np.isnan(theta[i,j]):
                single_responses[i,j,:] = array_response(ant_ind, theta[i,j], phi[i,j], kd).ravel()
    
    # Compare results
    try:
        np.testing.assert_allclose(batch_responses, single_responses, rtol=1e-10)
        print("✓ Batch and single user responses match exactly")
    except AssertionError as e:
        print("✗ Batch and single user responses differ:")
        print(e)


def test_array_response_batch_performance(n_users: int):
    """Compare performance of batch vs single user processing.
    
    Args:
        n_users: Number of users to test with
    """
    # Test parameters - larger arrays for meaningful timing
    panel_size = (8, 4)  # 32 antennas
    kd = 2 * np.pi * 0.5
    n_paths = 50
    
    # Generate random angles (some NaN)
    rng = np.random.default_rng(42)
    theta = rng.uniform(0, np.pi, size=(n_users, n_paths))
    phi = rng.uniform(0, 2*np.pi, size=(n_users, n_paths))
    
    # Add some NaN values randomly
    nan_mask = rng.random(size=(n_users, n_paths)) < 0.2
    theta[nan_mask] = np.nan
    phi[nan_mask] = np.nan
    
    # Get antenna indices
    ant_ind = ant_indices(panel_size)
    n_ant = len(ant_ind)
    
    # Time batch version
    t0 = time.perf_counter()
    batch_responses = array_response_batch(ant_ind, theta, phi, kd)
    batch_responses = np.moveaxis(batch_responses, 1, 2)  # [batch_size, n_paths, N]
    batch_time = time.perf_counter() - t0
    
    # Time single version
    t0 = time.perf_counter()
    single_responses = np.zeros((n_users, n_paths, n_ant), dtype=np.complex128)
    for i in range(n_users):
        for j in range(n_paths):
            if not np.isnan(theta[i,j]):
                single_responses[i,j,:] = array_response(ant_ind, theta[i,j], phi[i,j], kd).ravel()
    single_time = time.perf_counter() - t0
    
    # Verify results match
    try:
        np.testing.assert_allclose(batch_responses, single_responses, rtol=1e-10)
        match_str = "✓"
    except AssertionError:
        match_str = "✗"
    
    return batch_time, single_time, match_str


def test_array_response_batch_edge_cases():
    """Test edge cases for batched array response."""
    print("\nTesting edge cases...")
    
    panel_size = (2, 2)
    kd = 2 * np.pi * 0.5
    ant_ind = ant_indices(panel_size)
    n_ant = len(ant_ind)
    
    # Test single user, single path
    print("Testing single user, single path...")
    theta = np.array([[np.pi/4]])
    phi = np.array([[np.pi/3]])
    result = array_response_batch(ant_ind, theta, phi, kd)
    try:
        assert result.shape == (1, n_ant, 1)
        print("✓ Single user/path shape correct")
    except AssertionError:
        print(f"✗ Wrong shape: got {result.shape}, expected (1, {n_ant}, 1)")
    
    # Test all NaN angles
    print("\nTesting all NaN angles...")
    theta = np.full((2, 3), np.nan)
    phi = np.full((2, 3), np.nan)
    result = array_response_batch(ant_ind, theta, phi, kd)
    try:
        assert result.shape == (2, n_ant, 3)
        assert np.all(result == 0)
        print("✓ All-NaN case handled correctly")
    except AssertionError:
        print("✗ All-NaN case failed")
    
    # Test partial NaN angles
    print("\nTesting partial NaN angles...")
    theta = np.array([[np.pi/4, np.nan, np.pi/3],
                     [np.nan, np.pi/6, np.pi/2]])
    phi = np.array([[np.pi/3, np.nan, np.pi/4],
                   [np.nan, np.pi/2, np.pi/6]])
    result = array_response_batch(ant_ind, theta, phi, kd)
    
    # Calculate expected result using single-user version for comparison
    expected = np.zeros((2, n_ant, 3), dtype=np.complex128)
    for i in range(2):
        for j in range(3):
            if not np.isnan(theta[i,j]):
                expected[i,:,j] = array_response(ant_ind, theta[i,j], phi[i,j], kd).ravel()
    
    try:
        assert result.shape == (2, n_ant, 3)
        assert np.all(np.isnan(theta) == (result == 0).all(axis=1))  # Zero where NaN
        np.testing.assert_allclose(result, expected, rtol=1e-10)
        print("✓ Partial-NaN case handled correctly")
    except AssertionError as e:
        print("✗ Partial-NaN case failed:")
        print(e)
    
    # Test no NaN angles
    print("\nTesting no NaN angles...")
    theta = np.full((2, 3), np.pi/4)
    phi = np.full((2, 3), np.pi/3)
    result = array_response_batch(ant_ind, theta, phi, kd)
    try:
        assert result.shape == (2, n_ant, 3)
        assert not np.any(np.isnan(result))
        print("✓ No-NaN case handled correctly")
    except AssertionError:
        print("✗ No-NaN case failed")


def run_scaling_tests():
    """Run performance tests with different numbers of users."""
    print("\nRunning scaling tests...")
    print("-" * 80)
    header = f"{'Users':>10} | {'Single (s)':>12} | {'Batch (s)':>12} | {'Speedup':>8} | {'µs/user':>10} | {'Match':>5}"
    print(header)
    print("-" * 80)
    
    # Test with different numbers of users
    n_users_list = [100, 1000, 10000, 100000]
    results = []
    
    for n_users in n_users_list:
        batch_time, single_time, match_str = test_array_response_batch_performance(n_users)
        speedup = single_time/batch_time
        us_per_user = (batch_time/n_users)*1e6
        
        results.append((n_users, single_time, batch_time, speedup))
        print(f"{n_users:10,d} | {single_time:12.3f} | {batch_time:12.3f} | {speedup:8.1f}x | {us_per_user:10.1f} | {match_str:>5}")
    
    print("-" * 80)


if __name__ == '__main__':
    print("Running array response tests...")
    test_array_response_batch_matches_single()
    test_array_response_batch_edge_cases()
    run_scaling_tests()
    print("\nAll tests completed.") 
