#%%

import numpy as np
import time
from deepmimo.generator.python.ant_patterns import AntennaPattern

class TestAntennaPatterns:
    def __init__(self, verbose=False):
        self.passed = 0
        self.failed = 0
        self.verbose = verbose
        
    def assert_array_almost_equal(self, a, b, decimal=7):
        try:
            np.testing.assert_array_almost_equal(a, b, decimal=decimal)
            return True
        except AssertionError:
            return False
            
    def assert_greater(self, a, b):
        return a > b
    
    def run_test(self, test_func):
        test_name = test_func.__name__
        try:
            test_func()
            print(f"✓ {test_name} passed")
            self.passed += 1
        except AssertionError as e:
            print(f"✗ {test_name} failed: {str(e)}")
            self.failed += 1
            
    def test_single_isotropic(self):
        """Test single application with isotropic pattern."""
        pattern = AntennaPattern(tx_pattern='isotropic', rx_pattern='isotropic')
        power = np.array([1.0, 2.0, 3.0])
        angles = np.array([30.0, 45.0, 60.0])
        
        result = pattern.apply(power=power, 
                             aoa_theta=angles, 
                             aoa_phi=angles,
                             aod_theta=angles, 
                             aod_phi=angles)
        
        assert self.assert_array_almost_equal(result, power), "Single isotropic pattern failed"

    def test_batch_isotropic(self):
        """Test batch application with isotropic pattern."""
        pattern = AntennaPattern(tx_pattern='isotropic', rx_pattern='isotropic')
        power = np.array([[1.0, 2.0], [3.0, 4.0]])
        angles = np.array([[30.0, 45.0], [60.0, 90.0]])
        
        result = pattern.apply_batch(power=power,
                                   aoa_theta=angles,
                                   aoa_phi=angles,
                                   aod_theta=angles,
                                   aod_phi=angles)
        
        assert self.assert_array_almost_equal(result, power), "Batch isotropic pattern failed"

    def test_single_dipole(self):
        """Test single application with half-wave dipole pattern.
        
        Tests the half-wave dipole pattern characteristics:
        1. Maximum gain at 90° (broadside)
        2. Nulls at 0° and 180° (endfire)
        3. At 45°, gain should be approximately 0.08 relative to maximum
        """
        pattern = AntennaPattern(tx_pattern='halfwave-dipole', rx_pattern='halfwave-dipole')
        
        # Test at different angles
        test_cases = [
            # power, theta, expected relative gain
            (1.0, 90.0, 1.0),     # Maximum gain at 90 degrees (broadside)
            (1.0, 0.0, 0.0),      # Null at 0 degrees (endfire)
            (1.0, 180.0, 0.0),    # Null at 180 degrees (endfire)
            (1.0, 45.0, 0.08)     # Gain at 45 degrees (~0.08 of maximum)
        ]
        
        for power_val, theta_val, expected_rel_gain in test_cases:
            power = np.array([power_val])
            theta = np.array([np.deg2rad(theta_val)])  # Convert to radians
            phi = np.array([0.0])
            
            result = pattern.apply(power=power,
                                 aoa_theta=theta,
                                 aoa_phi=phi,
                                 aod_theta=theta,
                                 aod_phi=phi)
            
            if expected_rel_gain == 0:
                assert abs(result[0]) < 1e-10, f"Expected zero gain at {theta_val} degrees"
            else:
                # Normalize result to maximum gain for comparison
                max_result = pattern.apply(power=np.array([1.0]),
                                         aoa_theta=np.array([np.pi/2]),
                                         aoa_phi=np.array([0.0]),
                                         aod_theta=np.array([np.pi/2]),
                                         aod_phi=np.array([0.0]))
                relative_gain = result[0] / max_result[0]
                if self.verbose:
                    print(f"\nDebug - Angle: {theta_val}°")
                    print(f"Absolute gain: {result[0]:.6f}")
                    print(f"Max gain: {max_result[0]:.6f}")
                    print(f"Relative gain: {relative_gain:.6f}")
                    print(f"Expected relative gain: {expected_rel_gain}")
                
                assert abs(relative_gain - expected_rel_gain) < 0.01, \
                    f"Unexpected gain at {theta_val} degrees. Got {relative_gain:.4f}, expected {expected_rel_gain}"

    def test_batch_dipole(self):
        """Test batch application with half-wave dipole pattern."""
        pattern = AntennaPattern(tx_pattern='halfwave-dipole', rx_pattern='halfwave-dipole')
        
        # Test multiple angles simultaneously
        power = np.ones((4, 1))  # Same power for all test cases
        theta = np.deg2rad(np.array([[90.0], [0.0], [180.0], [45.0]]))  # Different angles
        phi = np.zeros_like(theta)
        
        result = pattern.apply_batch(power=power,
                                   aoa_theta=theta,
                                   aoa_phi=phi,
                                   aod_theta=theta,
                                   aod_phi=phi)
        
        # Normalize results
        max_val = result[0,0]  # Value at 90 degrees
        normalized = result / max_val
        
        # Check expected pattern
        assert abs(normalized[0,0] - 1.0) < 0.01, "Unexpected gain at 90 degrees"
        assert abs(normalized[1,0]) < 1e-10, "Expected zero gain at 0 degrees"
        assert abs(normalized[2,0]) < 1e-10, "Expected zero gain at 180 degrees"
        assert abs(normalized[3,0] - 0.08) < 0.01, "Unexpected gain at 45 degrees"

    def test_1d_to_2d_conversion(self):
        """Test that 1D inputs are correctly handled in batch processing."""
        pattern = AntennaPattern(tx_pattern='isotropic', rx_pattern='isotropic')
        power = np.array([1.0, 2.0])
        angles = np.array([30.0, 45.0])
        
        result = pattern.apply_batch(power=power,
                                   aoa_theta=angles,
                                   aoa_phi=angles,
                                   aod_theta=angles,
                                   aod_phi=angles)
        
        assert result.shape == (1, 2), "1D to 2D shape conversion failed"
        assert self.assert_array_almost_equal(result[0], power), "1D to 2D value conversion failed"

    def test_performance(self):
        """Test performance of batch vs single processing."""
        pattern = AntennaPattern(tx_pattern='halfwave-dipole', rx_pattern='halfwave-dipole')
        n_samples = 10000
        
        if self.verbose:
            print(f"\nTesting performance with {n_samples} users...")
        
        # Generate test data
        power = np.random.rand(n_samples)
        angles = np.random.rand(n_samples) * 180  # Random angles between 0 and 180
        
        # Time single processing
        start_time = time.time()
        _ = pattern.apply(power=power,
                         aoa_theta=angles,
                         aoa_phi=angles,
                         aod_theta=angles,
                         aod_phi=angles)
        single_time = time.time() - start_time
        
        # Time batch processing
        start_time = time.time()
        _ = pattern.apply_batch(power=power,
                              aoa_theta=angles,
                              aoa_phi=angles,
                              aod_theta=angles,
                              aod_phi=angles)
        batch_time = time.time() - start_time
        
        print(f"\nPerformance Test Results ({n_samples} users):")
        print(f"Single processing time: {single_time:.4f} seconds")
        print(f"Batch processing time: {batch_time:.4f} seconds")
        print(f"Speedup factor: {single_time/batch_time:.2f}x")
        
        assert batch_time <= single_time, "Batch processing should be faster than single processing"

def run_tests(verbose=False):
    """Run all tests and print summary.
    
    Args:
        verbose (bool): If True, print detailed debug information.
    """
    test_suite = TestAntennaPatterns(verbose=verbose)
    
    # Run functional tests
    print("\nRunning functional tests...")
    test_suite.run_test(test_suite.test_single_isotropic)
    test_suite.run_test(test_suite.test_batch_isotropic)
    test_suite.run_test(test_suite.test_single_dipole)
    test_suite.run_test(test_suite.test_batch_dipole)
    test_suite.run_test(test_suite.test_1d_to_2d_conversion)
    
    # Run performance test
    print("\nRunning performance test...")
    test_suite.run_test(test_suite.test_performance)
    
    # Print summary
    print(f"\nTest Summary:")
    print(f"Passed: {test_suite.passed}")
    print(f"Failed: {test_suite.failed}")
    print(f"Total: {test_suite.passed + test_suite.failed}")
    
    return test_suite.failed == 0

if __name__ == '__main__':
    success = run_tests(verbose=False)  # Set to True for detailed debug output

