"""Created on Wed Jan  8 14:26:47 2025

@author: joao

test for compute_number_paths(.)
"""

import numpy as np


def comp_next_pwr_10(arr):
    # def compute_number_paths(arr):
    # Handle zero separately
    result = np.zeros_like(arr, dtype=int)

    # For non-zero values, calculate order
    non_zero = arr > 0
    result[non_zero] = np.floor(np.log10(arr[non_zero])).astype(int) + 1

    return result


# Example array
arr = np.array([0, 1, 9, 10, 99, 100, 999, 1000])
# Custom Orders: [0 1 1 2 2 3 3 4]

# Compute the custom orders
custom_orders = comp_next_pwr_10(arr)
print("Custom Orders:", custom_orders)
