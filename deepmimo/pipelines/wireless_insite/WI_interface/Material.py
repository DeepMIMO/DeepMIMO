"""Material module for electromagnetic simulation.

This module provides a dataclass for representing material properties in electromagnetic simulations
and a method to parse material properties from a file.
"""

from dataclasses import dataclass, fields
from typing import Self, TypeVar

import numpy as np

T = TypeVar("T", bound="Material")


@dataclass
class Material:
    """Class representing material properties for electromagnetic simulation.

    This class stores various electromagnetic properties of materials used in simulations,
    such as conductivity, permittivity, and scattering parameters.

    Attributes:
        fields_diffusively_scattered (float): Fraction of fields diffusively scattered
        cross_polarized_power (float): Cross-polarized power ratio
        directive_alpha (int): Alpha parameter for directive scattering
        directive_beta (int): Beta parameter for directive scattering
        directive_lambda (float): Lambda parameter for directive scattering
        conductivity (float): Electrical conductivity in S/m
        permittivity (float): Relative permittivity
        roughness (float): Surface roughness parameter
        thickness (float): Material thickness in meters

    """

    fields_diffusively_scattered: float = 0.0
    cross_polarized_power: float = 0.0
    directive_alpha: int = 4.0
    directive_beta: int = 4.0
    directive_lambda: float = 0.5
    conductivity: float = 0.0
    permittivity: float = 1.0
    roughness: float = 0.0
    thickness: float = 0.0

    @classmethod
    def from_file(cls, file_path: str) -> Self | None:
        """Parse material properties from a file.

        Reads a material file and extracts the properties defined in the Material class.
        The file should contain lines with property names followed by their values.

        Args:
            file_path (str): Path to the material file

        Returns:
            Optional[Material]: A Material instance with properties from the file,
                               or None if the file is not found

        """
        try:
            with open(file_path) as f:
                file_lines = f.readlines()
        except FileNotFoundError:
            return None

        # Get field names and types from the Material dataclass
        material_attributes: dict[str, type[np.float64 | np.int64]] = {}
        for field in fields(cls):
            field_type = np.float64 if field.type is float else np.int64
            material_attributes[field.name] = field_type

        # Initialize a dictionary to store the parsed values
        parsed_values: dict[str, float | int] = {}

        # Parse the file
        for line in file_lines:
            for keyword, dtype in material_attributes.items():
                if line.startswith(keyword):
                    # Extract the value and convert to the appropriate type
                    value = dtype(line.split(" ")[-1].strip())
                    parsed_values[keyword] = value
                    break

        # Create the Material object by unpacking the dictionary
        return cls(**parsed_values)
