"""Dictionary utility functions for DeepMIMO.

This module provides utility functions for working with dictionaries,
including deep merging and comparison operations.
"""

from copy import deepcopy
from typing import Any


def deep_dict_merge(dict1: dict[str, Any], dict2: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries, preserving values from dict1 for keys not in dict2.

    This function recursively merges two dictionaries, keeping values from dict1
    for keys that are not present in dict2. For keys present in both dictionaries,
    if both values are dictionaries, they are recursively merged. Otherwise, the
    value from dict2 is used.

    Args:
        dict1: Base dictionary to merge into
        dict2: Dictionary with values to override

    Returns:
        Merged dictionary

    Example:
        >>> dict1 = {'a': 1, 'b': {'c': 2, 'd': 3}}
        >>> dict2 = {'b': {'c': 4}}
        >>> deep_dict_merge(dict1, dict2)
        {'a': 1, 'b': {'c': 4, 'd': 3}}

    """
    if hasattr(dict1, "to_dict"):
        dict1 = dict1.to_dict()
    if hasattr(dict2, "to_dict"):
        dict2 = dict2.to_dict()
    result = deepcopy(dict1)
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_dict_merge(result[key], value)
        else:
            result[key] = value
    return result


def compare_two_dicts(dict1: dict[str, Any], dict2: dict[str, Any]) -> bool:
    """Compare two dictionaries for equality.

    This function performs a deep comparison of two dictionaries, handling
    nested dictionaries.

    Args:
        dict1 (dict): First dictionary to compare
        dict2 (dict): Second dictionary to compare

    Returns:
        set: Set of keys in dict1 that are not in dict2

    """
    additional_keys = dict1.keys() - dict2.keys()
    for key, item in dict1.items():
        if isinstance(item, dict) and key in dict2:
            additional_keys = additional_keys | compare_two_dicts(item, dict2[key])
    return additional_keys
