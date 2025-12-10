"""Data structure utilities for DeepMIMO.

This module provides custom data structures used throughout DeepMIMO,
including dot-notation dictionaries, delegating lists, and verbose printing utilities.
"""

from collections.abc import Mapping
from pprint import pformat
from typing import Any, TypeVar

import numpy as np

K = TypeVar("K", bound=str)
V = TypeVar("V")


class DotDict(Mapping[K, V]):
    """A dictionary subclass that supports dot notation access to nested dictionaries.

    This class allows accessing dictionary items using both dictionary notation (d['key'])
    and dot notation (d.key). It automatically converts nested dictionaries to DotDict
    instances to maintain dot notation access at all levels.

    Example:
        >>> d = DotDict({'a': 1, 'b': {'c': 2}})
        >>> d.a
        1
        >>> d.b.c
        2
        >>> d['b']['c']
        2
        >>> list(d.keys())
        ['a', 'b']

    """

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        """Initialize DotDict with a dictionary.

        Args:
            data: Dictionary to convert to DotDict

        """
        self._data = {}
        if data:
            for key, value in data.items():
                if isinstance(value, dict):
                    self._data[key] = DotDict(value)
                else:
                    self._data[key] = value

    def __getattr__(self, key: str) -> Any:
        """Enable dot notation access to dictionary items."""
        try:
            return self._data[key]
        except KeyError as err:
            raise AttributeError(key) from err

    def __setattr__(self, key: str, value: Any) -> None:
        """Enable dot notation assignment with property support.

        This method first checks if the attribute is a property with a setter.
        If it is, it uses the property setter. Otherwise, it falls back to
        storing the value in the internal dictionary.
        """
        if key == "_data":
            super().__setattr__(key, value)
            return
        attr = getattr(type(self), key, None)
        if isinstance(attr, property) and attr.fset is not None:
            attr.fset(self, value)
        else:
            if isinstance(value, dict) and (not isinstance(value, DotDict)):
                value = DotDict(value)
            self._data[key] = value

    def __getitem__(self, key: str) -> Any:
        """Enable dictionary-style access."""
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Enable dictionary-style assignment."""
        if isinstance(value, dict) and (not isinstance(value, DotDict)):
            value = DotDict(value)
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        """Enable dictionary-style deletion."""
        del self._data[key]

    def update(self, other: dict[str, Any]) -> None:
        """Update the dictionary with elements from another dictionary."""
        processed = {
            k: DotDict(v) if isinstance(v, dict) and (not isinstance(v, DotDict)) else v
            for (k, v) in other.items()
        }
        self._data.update(processed)

    def __len__(self) -> int:
        """Return the length of the underlying data dictionary."""
        return len(self._data)

    def __iter__(self) -> Any:
        """Return an iterator over the data dictionary keys."""
        return iter(self._data)

    def __dir__(self) -> Any:
        """Return list of valid attributes."""
        return list(set(list(super().__dir__()) + list(self._data.keys())))

    def keys(self) -> Any:
        """Return dictionary keys."""
        return self._data.keys()

    def values(self) -> Any:
        """Return dictionary values."""
        return self._data.values()

    def items(self) -> Any:
        """Return dictionary items as (key, value) pairs."""
        return self._data.items()

    def get(self, key: str, default: Any = None) -> Any:
        """Get value for key, returning default if key doesn't exist."""
        return self._data.get(key, default)

    def hasattr(self, key: str) -> bool:
        """Safely check if a key exists in the dictionary.

        This method provides a safe way to check for attribute existence
        without raising KeyError, similar to Python's built-in hasattr().

        Args:
            key: The key to check for

        Returns:
            bool: True if the key exists, False otherwise

        """
        return key in self._data

    def to_dict(self) -> dict:
        """Convert DotDict back to a regular dictionary.

        Returns:
            dict: Regular dictionary representation

        """
        result = {}
        for key, value in self._data.items():
            if isinstance(value, DotDict):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def deepcopy(self) -> "DotDict":
        """Create a deep copy of the DotDict instance.

        This method creates a completely independent copy of the DotDict,
        including nested dictionaries and numpy arrays. This ensures that
        modifications to the copy won't affect the original.

        Returns:
            DotDict: A deep copy of this instance

        """
        result = {}
        for key, value in self._data.items():
            if isinstance(value, DotDict):
                result[key] = value.deepcopy()
            elif isinstance(value, dict):
                result[key] = DotDict(value).deepcopy()
            elif isinstance(value, np.ndarray):
                result[key] = value.copy()
            else:
                result[key] = value
        return type(self)(result)

    def __repr__(self) -> str:
        """Return string representation of dictionary."""
        return pformat(self._data)


class DelegatingList(list):
    """A list subclass that delegates method calls to each item in the list.

    When a method is called on this class, it will be called on each item in the list
    and the results will be returned as a list.
    """

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to each item in the list.

        If the attribute is a method, it will be called on each item and results returned as a list.
        If the attribute is a property, a list of property values will be returned.
        If the attribute is a list-like object, it will be wrapped in a DelegatingList.
        """
        if not self:
            msg = f"Empty list has no attribute '{name}'"
            raise AttributeError(msg)
        first_attr = getattr(self[0], name)
        if callable(first_attr):

            def method(*args: Any, **kwargs: Any) -> "DelegatingList":
                results = [getattr(item, name)(*args, **kwargs) for item in self]
                return DelegatingList(results)

            return method
        results = [getattr(item, name) for item in self]
        return DelegatingList(results)

    def __setattr__(self, name: str, value: Any) -> None:
        """Delegate attribute assignment to each item in the list.

        If value is a list/iterable, each item in the list gets the corresponding value.
        Otherwise, all items get the same value.
        """
        if name in self.__dict__:
            super().__setattr__(name, value)
            return
        if not self:
            msg = f"Empty list has no attribute '{name}'"
            raise AttributeError(msg)
        if (
            hasattr(value, "__iter__")
            and (not isinstance(value, (str, bytes)))
            and (len(value) == len(self))
        ):
            for item, val in __builtins__["zip"](self, value):
                setattr(item, name, val)
        else:
            for item in self:
                setattr(item, name, value)


class PrintIfVerbose:
    """A callable class that conditionally prints messages based on verbosity setting.

    The only purpose of this class is to avoid repeating "if verbose:" all the time.

    Usage:
        vprint = PrintIfVerbose(verbose);
        vprint(message)

    Args:
        verbose (bool): Flag to control whether messages should be printed.

    """

    def __init__(self, *, verbose: bool) -> None:
        """Store verbosity flag.

        Args:
            verbose: Print messages when True.

        """
        self.verbose = verbose

    def __call__(self, message: str) -> None:
        """Print the message if verbose mode is enabled.

        Args:
            message (str): The message to potentially print.

        """
        if self.verbose:
            print(message)

