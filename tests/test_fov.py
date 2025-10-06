"""
Test case for apply_FoV and apply_FoV_batch functions.
"""

#%%
import numpy as np
from deepmimo.generator.python.geometry import apply_FoV, apply_FoV_batch
import time

def run_correctness_tests():
    """Run tests to verify correctness of both FoV functions."""
    print("=== Detailed Single-Angle Tests ===")
    # Test with a single user and path first for debugging
    theta_single = np.array([np.pi/4])  # Single elevation angle in radians (45°)
    phi_single = np.array([np.pi/2])    # Single azimuth angle in radians (90°)
    fov = np.array([360.0, 180.0])      # Full FoV in degrees

    print("\nInput angles:")
    print(f"theta = {np.rad2deg(theta_single[0])}° ({theta_single[0]:.4f} rad)")
    print(f"phi = {np.rad2deg(phi_single[0])}° ({phi_single[0]:.4f} rad)")
    print(f"FoV = {fov}°")

    # Test original function
    print("\nOriginal function processing:")
    theta_mod = np.mod(theta_single[0], 2*np.pi)
    phi_mod = np.mod(phi_single[0], 2*np.pi)
    fov_rad = np.deg2rad(fov)
    print(f"After mod 2π: theta = {theta_mod:.4f} rad, phi = {phi_mod:.4f} rad")
    print(f"FoV in radians: {fov_rad}")
    
    mask_orig = apply_FoV(
        fov=fov,
        theta=theta_single[0],
        phi=phi_single[0])

    # Test batch function
    print("\nBatch function processing:")
    print(f"Input angles: theta = {np.rad2deg(theta_single[0])}° ({theta_single[0]:.4f} rad)")
    print(f"             phi = {np.rad2deg(phi_single[0])}° ({phi_single[0]:.4f} rad)")
    
    mask_batch = apply_FoV_batch(
        fov=fov,
        theta=theta_single,
        phi=phi_single)

    print("\nResults for full FoV:")
    print(f"Original: mask={mask_orig}")
    print(f"Batch:    mask={mask_batch[0]}")

    # Now test with limited FoV
    fov_limited = np.array([180.0, 90.0])  # Limited FoV in degrees
    print(f"\nTesting with limited FoV = {fov_limited}°")

    # Test original function
    print("\nOriginal function processing:")
    fov_rad = np.deg2rad(fov_limited)
    print(f"FoV in radians: {fov_rad}")
    
    mask_orig_limited = apply_FoV(
        fov=fov_limited,
        theta=theta_single[0],
        phi=phi_single[0])

    # Test batch function
    mask_batch_limited = apply_FoV_batch(
        fov=fov_limited,
        theta=theta_single,
        phi=phi_single)

    print("\nResults for limited FoV:")
    print(f"Original: mask={mask_orig_limited}")
    print(f"Batch:    mask={mask_batch_limited[0]}")

    print("\n=== Batch Tests ===")
    # Test setup for batch comparisons
    n_users = 100
    n_paths = 5
    np.random.seed(42)  # For reproducibility
    
    # Test 1: Full FoV
    print("\nTest 1: Full FoV")
    # Generate angles in radians
    theta = np.random.uniform(0, np.pi, (n_users, n_paths))   # radians
    phi = np.random.uniform(0, 2*np.pi, (n_users, n_paths))   # radians
    fov_full = np.array([360.0, 180.0])  # degrees
    
    # Method 1: Using original apply_FoV function
    mask_1 = np.zeros((n_users, n_paths), dtype=bool)
    for i in range(n_users):
        for j in range(n_paths):
            mask_1[i,j] = apply_FoV(
                fov=fov_full,
                theta=theta[i,j],
                phi=phi[i,j])
    
    # Method 2: Using batch function
    mask_2 = apply_FoV_batch(
        fov=fov_full,
        theta=theta,
        phi=phi)
    
    # Compare results
    mask_diff = np.abs(mask_1.astype(int) - mask_2.astype(int))
    print(f"Maximum difference in masks: {mask_diff.max()}")
    print(f"Total differences: {np.sum(mask_diff)}")
    
    if np.sum(mask_diff) > 0:
        # Show some examples where they differ
        diff_indices = np.where(mask_diff > 0)
        diff_list = list(zip(*diff_indices))  # Convert zip object to list
        print("\nExample differences:")
        for idx in diff_list[:5]:  # Show first 5 differences
            i, j = idx
            print(f"\nUser {i}, Path {j}:")
            print(f"Angles: theta = {np.rad2deg(theta[i,j]):.2f}° ({theta[i,j]:.4f} rad)")
            print(f"        phi = {np.rad2deg(phi[i,j]):.2f}° ({phi[i,j]:.4f} rad)")
            print(f"Original mask: {mask_1[i,j]}")
            print(f"Batch mask: {mask_2[i,j]}")
    
    # Test 2: Limited FoV
    print("\nTest 2: Limited FoV")
    fov_limited = np.array([180.0, 90.0])  # degrees
    
    # Method 1: Using original apply_FoV function
    mask_3 = np.zeros((n_users, n_paths), dtype=bool)
    for i in range(n_users):
        for j in range(n_paths):
            mask_3[i,j] = apply_FoV(
                fov=fov_limited,
                theta=theta[i,j],
                phi=phi[i,j])
    
    # Method 2: Using batch function
    mask_4 = apply_FoV_batch(
        fov=fov_limited,
        theta=theta,
        phi=phi)
    
    # Compare results
    mask_diff_2 = np.abs(mask_3.astype(int) - mask_4.astype(int))
    print(f"Maximum difference in masks: {mask_diff_2.max()}")
    print(f"Total differences: {np.sum(mask_diff_2)}")
    
    if np.sum(mask_diff_2) > 0:
        # Show some examples where they differ
        diff_indices = np.where(mask_diff_2 > 0)
        diff_list = list(zip(*diff_indices))  # Convert zip object to list
        print("\nExample differences:")
        for idx in diff_list[:5]:  # Show first 5 differences
            i, j = idx
            print(f"\nUser {i}, Path {j}:")
            print(f"Angles: theta = {np.rad2deg(theta[i,j]):.2f}° ({theta[i,j]:.4f} rad)")
            print(f"        phi = {np.rad2deg(phi[i,j]):.2f}° ({phi[i,j]:.4f} rad)")
            print(f"Original mask: {mask_3[i,j]}")
            print(f"Batch mask: {mask_4[i,j]}")

def run_performance_tests(n_users=1000, n_paths=5, n_trials=10):
    """Run performance benchmarks comparing both FoV functions.
    
    Args:
        n_users: Number of users to test with
        n_paths: Number of paths per user
        n_trials: Number of trials to average over
    """
    print(f"\nPerformance test with {n_users} users, {n_paths} paths, {n_trials} trials:")
    
    # Generate test data in radians
    theta = np.random.uniform(0, np.pi, (n_users, n_paths))    # radians
    phi = np.random.uniform(0, 2*np.pi, (n_users, n_paths))    # radians
    fov = np.array([180.0, 90.0])  # degrees
    
    # Test 1: Original function
    print("\nTest 1: Original function")
    times_orig = []
    for _ in range(n_trials):
        start = time.perf_counter()
        mask = np.zeros((n_users, n_paths), dtype=bool)
        for i in range(n_users):
            for j in range(n_paths):
                mask[i,j] = apply_FoV(
                    fov=fov,
                    theta=theta[i,j],
                    phi=phi[i,j])
        times_orig.append(time.perf_counter() - start)
    
    # Test 2: Batch function
    print("\nTest 2: Batch function")
    times_batch = []
    for _ in range(n_trials):
        start = time.perf_counter()
        mask = apply_FoV_batch(
            fov=fov,
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
    
    # # Run performance tests with different sizes
    # print("\nRunning performance tests...")
    
    # # Small batch
    run_performance_tests(n_users=100, n_paths=5, n_trials=10)
    
    # Medium batch
    run_performance_tests(n_users=1000, n_paths=5, n_trials=10)
    
    # Large batch
    run_performance_tests(n_users=10000, n_paths=5, n_trials=10)
    
    # Very large batch
    run_performance_tests(n_users=100000, n_paths=5, n_trials=10) 