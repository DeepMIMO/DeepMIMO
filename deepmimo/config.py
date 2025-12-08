"""DeepMIMO Configuration Module.

This module provides a singleton configuration class for DeepMIMO that allows
setting and retrieving global configuration values. It can be used to configure
various aspects of the DeepMIMO framework, such as ray tracing parameters,
computation settings, and other global variables.

Usage:
    # Set a configuration value
    deepmimo.config.set('ray_tracer_version', '3.0.0')

    # Get a configuration value
    version = deepmimo.config.get('ray_tracer_version')

    # Print all current configurations
    deepmimo.config.print_config()

    # Reset to defaults
    deepmimo.config.reset()

    # Alternative function-like interface
    deepmimo.config('ray_tracer_version')  # Get value
    deepmimo.config('ray_tracer_version', '3.0.0')  # Set value
    deepmimo.config(use_gpu=True)  # Set using keyword
    deepmimo.config()  # Print all configs
"""

# Import constants for ray tracer versions
from typing import Any

from .consts import (
    RAYTRACER_VERSION_AODT,
    RAYTRACER_VERSION_SIONNA,
    RAYTRACER_VERSION_WIRELESS_INSITE,
    RT_SOURCES_FOLDER,
    SCENARIOS_FOLDER,
)


class DeepMIMOConfig:
    """Singleton configuration class for DeepMIMO.

    This class implements a singleton pattern to ensure there's only one
    configuration instance throughout the application.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """Initialize the configuration with default values."""
        self._config = {
            # Ray tracer versions
            "wireless_insite_version": RAYTRACER_VERSION_WIRELESS_INSITE,
            "sionna_version": RAYTRACER_VERSION_SIONNA,
            "aodt_version": RAYTRACER_VERSION_AODT,
            # Use GPU
            "use_gpu": False,
            # GPU device ID
            "gpu_device_id": 0,
            # Folder containing scenarios (extracted and ZIPs)
            "scenarios_folder": SCENARIOS_FOLDER,
            # Folder containing ray tracing source files
            "rt_sources_folder": RT_SOURCES_FOLDER,
        }

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value.

        Args:
            key (str): The configuration key to set.
            value (Any): The value to set for the configuration key.

        """
        if key in self._config:
            self._config[key] = value
        else:
            print(f"Warning: Configuration key '{key}' does not exist. Adding as new key.")
            self._config[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.

        Args:
            key (str): The configuration key to get.
            default (Any): The default value to return if the key doesn't exist.

        Returns:
            Any: The configuration value for the given key, or the default value if the key doesn't exist.

        """
        return self._config.get(key, default)

    def reset(self) -> None:
        """Reset all configuration values to their defaults."""
        self._initialize()

    def get_config_str(self) -> str:
        """Return a string representation of the configuration."""
        result = "\nDeepMIMO Configuration:\n"
        result += "-" * 50 + "\n"
        for key, value in self._config.items():
            result += f"{key}: {value}\n"
        result += "-" * 50
        return result

    def print_config(self) -> None:
        """Print all current configuration values."""
        print(self.get_config_str())

    def get_all(self) -> dict[str, Any]:
        """Get all configuration values.

        Returns:
            dict: A dictionary containing all configuration values.

        """
        return self._config.copy()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Function-like interface for the configuration.

        If no arguments are provided, print all current configuration values.
        If only the key is provided as a positional argument, get the configuration value for that key.
        If both key and value are provided as positional arguments, set the configuration value for that key.
        If keyword arguments are provided, set the configuration values for those keys.

        Args:
            *args: Positional arguments. If one argument is provided, it's treated as a key to get.
                  If two arguments are provided, they're treated as a key-value pair to set.
            **kwargs: Keyword arguments. Each keyword-value pair sets a configuration value.

        Returns:
            If getting a configuration value, returns the value for the given key.
            If setting configuration values, returns None.
            If printing all configuration values, returns None.

        """
        # If no arguments are provided, print all configuration values
        if not args and not kwargs:
            self.print_config()
            return None

        # If only keyword arguments are provided, set configuration values
        if not args and kwargs:
            for key, value in kwargs.items():
                self.set(key, value)
            return None

        # If one positional argument is provided, get the configuration value
        if len(args) == 1 and not kwargs:
            return self.get(args[0])

        # If two positional arguments are provided, set the configuration value
        if len(args) == 2 and not kwargs:
            self.set(args[0], args[1])
            return None

        # If both positional arguments and keyword arguments are provided, raise an error
        if args and kwargs:
            msg = "Cannot mix positional arguments and keyword arguments"
            raise ValueError(msg)
        return None

    def __repr__(self) -> str:
        """Return a string representation of the configuration."""
        return self.get_config_str()


# Create a singleton instance
config = DeepMIMOConfig()

# Export the config instance
__all__ = ["config"]
