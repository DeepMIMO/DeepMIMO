"""Ray Tracing Parameters Module.

This module provides the base class for ray tracing parameters used across different
ray tracing engines (Wireless Insite, Sionna, etc.). It defines common parameters
and functionality while allowing engine-specific extensions.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class RayTracingParameters:
    """Base class for ray tracing parameters.

    This class defines common parameters across different ray tracing engines.
    Each specific engine (Wireless Insite, Sionna, etc.) should extend this class
    with its own parameters and methods.

    Note: All parameters are required to allow child classes to add their own required
    parameters. Default values can be set in __post_init__.
    """

    # Ray Tracing Engine info
    raytracer_name: str  # Name of ray tracing engine (from constants)
    raytracer_version: str  # Version of ray tracing engine

    # Frequency (determines material properties)
    frequency: float  # Center frequency in Hz

    # Ray tracing interaction settings
    max_path_depth: int  # Maximum number of interactions (ideally, R + D + S + T)
    max_reflections: int  # Maximum number of reflections (R)
    max_diffractions: int  # Maximum number of diffractions (D)
    max_scattering: int  # Maximum number of diffuse scattering events (S)
    max_transmissions: int  # Maximum number of transmissions (T)

    # Details on diffraction, scattering, and transmission
    diffuse_reflections: int = 0  # Number of reflections allowed in paths with diffuse scattering
    diffuse_diffractions: int = 0  # Number of diffractions allowed in paths with diffuse scattering
    diffuse_transmissions: int = (
        0  # Number of transmissions allowed in paths with diffuse scattering
    )
    diffuse_final_interaction_only: bool = (
        False  # Whether to only consider diffuse scattering at final interaction
    )
    diffuse_random_phases: bool = False  # Whether to randomize phases of diffuse scattering

    terrain_reflection: bool = False  # Whether to allow reflections on terrain
    terrain_diffraction: bool = False  # Whether to allow diffractions on terrain
    terrain_scattering: bool = False  # Whether to allow scattering on terrain

    # Ray casting settings
    num_rays: int = 1000000  # Number of rays to launch (per antenna)
    ray_casting_method: str = "uniform"  # 'uniform' (e.g. a fibonacci sphere) or '...'
    synthetic_array: bool = True  # Whether to use a synthetic array

    # Ray casting range (when casting method is uniform, centered at antenna boresight)
    ray_casting_range_az: float = 360.0  # casting range in azimuth (degrees)
    ray_casting_range_el: float = 180.0  # casting range in elevation (degrees)

    # GPS Bounding Box
    gps_bbox: tuple[float, float, float, float] = (
        0,
        0,
        0,
        0,
    )  # (min_lat, min_lon, max_lat, max_lon)
    # This box corresponds to the area fetched from the real world and (usually) to the user grid

    # Raw parameters storage
    raw_params: dict = field(default_factory=dict)  # Store original parameters from engine

    def to_dict(self) -> dict:
        """Convert parameters to dictionary format.

        Returns:
            Dictionary containing all parameters

        """
        return asdict(self)

    @classmethod
    def from_dict(cls, params_dict: dict, raw_params: dict | None = None) -> "RayTracingParameters":
        """Create RayTracingParameters from a dictionary.

        Args:
            params_dict: Dictionary containing parameter values
            raw_params: Optional dictionary containing original engine parameters

        Returns:
            RayTracingParameters object

        """
        # Store raw parameters if provided
        if raw_params is not None:
            params_dict["raw_params"] = raw_params
        return cls(**params_dict)

    @classmethod
    def read_parameters(cls, load_folder: str | Path) -> "RayTracingParameters":
        """Read parameters from a folder.

        This is an abstract method that should be implemented by each engine-specific
        subclass to read parameters in the appropriate format.

        Args:
            load_folder: Path to folder containing parameter files

        Returns:
            RayTracingParameters object

        Raises:
            NotImplementedError: If not implemented by subclass

        """
        raise NotImplementedError("Must be implemented by subclass")
