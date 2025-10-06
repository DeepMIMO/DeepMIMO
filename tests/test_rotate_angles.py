"""
Test case for rotate_angles and rotate_angles_batch
"""

#%% rotate angles test case
import numpy as np
from deepmimo.generator.geometry import rotate_angles, rotate_angles_batch
import time

def run_correctness_tests():
    """Run tests to verify correctness of both rotation functions."""
    print("=== Detailed Single-Angle Tests ===")
    # Test with a single user and path first for debugging
    theta_single = np.array([45.0])  # Single elevation angle
    phi_single = np.array([90.0])    # Single azimuth angle
    rotation_zero = np.array([0.0, 0.0, 0.0])

    # Test original function
    theta_orig, phi_orig = rotate_angles(
        rotation=rotation_zero,
        theta=theta_single[0],
        phi=phi_single[0])

    # Test batch function
    theta_batch, phi_batch = rotate_angles_batch(
        rotation=rotation_zero,
        theta=theta_single,
        phi=phi_single)

    print("Single angle test with zero rotation (both in radians):")
    print(f"Original: theta={theta_orig:.10f}, phi={phi_orig:.10f}")
    print(f"Batch:    theta={theta_batch[0]:.10f}, phi={phi_batch[0]:.10f}")

    # Now test with non-zero rotation
    rotation_test = np.array([30.0, 45.0, 60.0])

    # Test original function
    theta_orig_rot, phi_orig_rot = rotate_angles(
        rotation=rotation_test,
        theta=theta_single[0],
        phi=phi_single[0])

    # Test batch function
    theta_batch_rot, phi_batch_rot = rotate_angles_batch(
        rotation=rotation_test,
        theta=theta_single,
        phi=phi_single)

    print("\nSingle angle test with rotation [30°, 45°, 60°] (both in radians):")
    print(f"Original: theta={theta_orig_rot:.10f}, phi={phi_orig_rot:.10f}")
    print(f"Batch:    theta={theta_batch_rot[0]:.10f}, phi={phi_batch_rot[0]:.10f}")

    # Print intermediate values for debugging
    print("\nDebugging intermediate values in original function:")
    rotation_rad = np.deg2rad(rotation_test)
    print(f"sin_alpha = sin(phi - gamma) = sin({np.deg2rad(phi_single[0])} - {rotation_rad[2]}) = {np.sin(np.deg2rad(phi_single[0]) - rotation_rad[2])}")
    print(f"sin_beta = sin(beta) = sin({rotation_rad[1]}) = {np.sin(rotation_rad[1])}")
    print(f"sin_gamma = sin(alpha) = sin({rotation_rad[0]}) = {np.sin(rotation_rad[0])}")

    print("\nDebugging intermediate values in batch function:")
    rotation_batch = np.deg2rad(rotation_test[None, :])
    phi_rad = np.deg2rad(phi_single)
    print(f"sin_alpha = sin(phi - gamma) = sin({phi_rad[0]} - {rotation_batch[0, 2]}) = {np.sin(phi_rad[0] - rotation_batch[0, 2])}")
    print(f"sin_beta = sin(beta) = sin({rotation_batch[0, 1]}) = {np.sin(rotation_batch[0, 1])}")
    print(f"sin_gamma = sin(alpha) = sin({rotation_batch[0, 0]}) = {np.sin(rotation_batch[0, 0])}")

    print("\n=== Batch Tests ===")
    # Test setup for batch comparisons
    n_users = 100
    n_paths = 5
    np.random.seed(42)  # For reproducibility
    
    # Test 1: Same rotation for all users
    print("\nTest 1: Same rotation for all users")
    theta = np.random.uniform(0, 180, (n_users, n_paths))
    phi = np.random.uniform(0, 360, (n_users, n_paths))
    rotation_single = np.array([30, 45, 60])
    
    # Method 1: Using original rotate_angles function
    theta_rot_1 = np.zeros_like(theta)
    phi_rot_1 = np.zeros_like(phi)
    for i in range(n_users):
        theta_rot_1[i], phi_rot_1[i] = rotate_angles(
            rotation=rotation_single,
            theta=theta[i],
            phi=phi[i])
    
    # Method 2: Using batch function
    theta_rot_2, phi_rot_2 = rotate_angles_batch(
        rotation=rotation_single,
        theta=theta,
        phi=phi)
    
    # Compare results
    theta_diff = np.abs(theta_rot_1 - theta_rot_2)
    phi_diff = np.abs(phi_rot_1 - phi_rot_2)
    print(f"Maximum difference in theta: {theta_diff.max():.10f} radians")
    print(f"Maximum difference in phi: {phi_diff.max():.10f} radians")
    print(f"Mean difference in theta: {theta_diff.mean():.10f} radians")
    print(f"Mean difference in phi: {phi_diff.mean():.10f} radians")
    
    # Test 2: Different rotation per user
    print("\nTest 2: Different rotation per user")
    rotation_per_user = np.random.uniform(-90, 90, (n_users, 3))
    
    # Method 1: Using original rotate_angles function
    theta_rot_3 = np.zeros_like(theta)
    phi_rot_3 = np.zeros_like(phi)
    for i in range(n_users):
        theta_rot_3[i], phi_rot_3[i] = rotate_angles(
            rotation=rotation_per_user[i],
            theta=theta[i],
            phi=phi[i])
    
    # Method 2: Using batch function
    theta_rot_4, phi_rot_4 = rotate_angles_batch(
        rotation=rotation_per_user,
        theta=theta,
        phi=phi)
    
    # Compare results
    theta_diff_2 = np.abs(theta_rot_3 - theta_rot_4)
    phi_diff_2 = np.abs(phi_rot_3 - phi_rot_4)
    print(f"Maximum difference in theta: {theta_diff_2.max():.10f} radians")
    print(f"Maximum difference in phi: {phi_diff_2.max():.10f} radians")
    print(f"Mean difference in theta: {theta_diff_2.mean():.10f} radians")
    print(f"Mean difference in phi: {phi_diff_2.mean():.10f} radians")

def run_performance_tests(n_users=1000, n_paths=5, n_trials=10):
    """Run performance benchmarks comparing both rotation functions.
    
    Args:
        n_users: Number of users to test with
        n_paths: Number of paths per user
        n_trials: Number of trials to average over
    """
    print(f"\nPerformance test with {n_users} users, {n_paths} paths, {n_trials} trials:")
    
    # Generate test data
    theta = np.random.uniform(0, 180, (n_users, n_paths))
    phi = np.random.uniform(0, 360, (n_users, n_paths))
    rotation_single = np.array([30, 45, 60])
    rotation_batch = np.random.uniform(-90, 90, (n_users, 3))
    
    # Test 1: Single rotation for all users
    print("\nTest 1: Single rotation for all users")
    
    # Time original function
    times_orig = []
    for _ in range(n_trials):
        start = time.perf_counter()
        for i in range(n_users):
            theta_rot, phi_rot = rotate_angles(
                rotation=rotation_single,
                theta=theta[i],
                phi=phi[i])
        times_orig.append(time.perf_counter() - start)
    
    # Time batch function
    times_batch = []
    for _ in range(n_trials):
        start = time.perf_counter()
        theta_rot, phi_rot = rotate_angles_batch(
            rotation=rotation_single,
            theta=theta,
            phi=phi)
        times_batch.append(time.perf_counter() - start)
    
    print(f"Original function: {np.mean(times_orig):.6f}s ± {np.std(times_orig):.6f}s")
    print(f"Batch function:    {np.mean(times_batch):.6f}s ± {np.std(times_batch):.6f}s")
    print(f"Speedup:          {np.mean(times_orig)/np.mean(times_batch):.2f}x")
    
    # Test 2: Different rotation per user
    print("\nTest 2: Different rotation per user")
    
    # Time original function
    times_orig = []
    for _ in range(n_trials):
        start = time.perf_counter()
        for i in range(n_users):
            theta_rot, phi_rot = rotate_angles(
                rotation=rotation_batch[i],
                theta=theta[i],
                phi=phi[i])
        times_orig.append(time.perf_counter() - start)
    
    # Time batch function
    times_batch = []
    for _ in range(n_trials):
        start = time.perf_counter()
        theta_rot, phi_rot = rotate_angles_batch(
            rotation=rotation_batch,
            theta=theta,
            phi=phi)
        times_batch.append(time.perf_counter() - start)
    
    print(f"Original function: {np.mean(times_orig):.6f}s ± {np.std(times_orig):.6f}s")
    print(f"Batch function:    {np.mean(times_batch):.6f}s ± {np.std(times_batch):.6f}s")
    print(f"Speedup:          {np.mean(times_orig)/np.mean(times_batch):.2f}x")

if __name__ == "__main__":
    # Run correctness tests
    print("Running correctness tests...")
    run_correctness_tests()
    
    # Run performance tests with different sizes
    print("\nRunning performance tests...")
    
    # Small batch
    run_performance_tests(n_users=100, n_paths=5, n_trials=10)
    
    # Medium batch
    run_performance_tests(n_users=1000, n_paths=5, n_trials=10)
    
    # Large batch
    run_performance_tests(n_users=10000, n_paths=5, n_trials=10)

    # Very large batch
    run_performance_tests(n_users=100000, n_paths=5, n_trials=10)
