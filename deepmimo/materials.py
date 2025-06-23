"""
Core material representation module.

This module provides the base class for representing materials and their properties,
including electromagnetic and scattering characteristics.
"""

from dataclasses import dataclass, asdict, astuple
from typing import Dict, ClassVar, List, Set

@dataclass
class Material:
    """Base material class for DeepMIMO.
    
    This class represents materials in a standardized way across different ray tracers.
    It includes electromagnetic, scattering, and physical properties.

    Attributes:
        id (int): Material identifier
        name (str): Material name/label
        permittivity (float): Relative permittivity
        conductivity (float): Conductivity in S/m
        
        # Scattering properties
        scattering_model (str): Type of scattering model
        scattering_coefficient (float): Scattering coefficient (0-1)
        cross_polarization_coefficient (float): Cross-polarization ratio
        
        # Directive scattering parameters
        alpha_r (float): Real part of scattering exponent
        alpha_i (float): Imaginary part of scattering exponent
        lambda_param (float): Forward/backward ratio
        
        # Physical properties
        roughness (float): Surface roughness in meters
        thickness (float): Material thickness in meters
        
        # Optional attenuation properties for foliage
        vertical_attenuation (float): Vertical attenuation in dB/m
        horizontal_attenuation (float): Horizontal attenuation in dB/m
        
        # Optional ITU-R P.2040 parameters for frequency-dependent permittivity
        # ε = a + b*f^c + j*(d*f^c), where f is frequency in GHz
        itu_a (float): Constant term in real part
        itu_b (float): Coefficient of frequency-dependent term in real part
        itu_c (float): Frequency exponent
        itu_d (float): Coefficient of frequency-dependent imaginary part

        Notes:
            - Scattering modeling based on https://ieeexplore.ieee.org/document/4052607
            (common approach to backscattering in ray tracing software)
            - ITU-R P.2040 parameters are optional
    """
    # Scattering model types
    SCATTERING_NONE = 'none'
    SCATTERING_LAMBERTIAN = 'lambertian'
    SCATTERING_DIRECTIVE = 'directive'
    
    # Required properties
    id: int
    name: str
    permittivity: float
    conductivity: float
    
    # Scattering properties
    scattering_model: str = SCATTERING_NONE
    scattering_coefficient: float = 0.0
    cross_polarization_coefficient: float = 0.0
    
    # Directive scattering parameters
    alpha_r: float = 4.0
    alpha_i: float = 4.0
    lambda_param: float = 0.5
    
    # Physical properties
    roughness: float = 0.0
    thickness: float = 0.0
    
    # Optional attenuation properties for foliage
    vertical_attenuation: float = 0.0
    horizontal_attenuation: float = 0.0
    
    # Optional ITU-R P.2040 parameters
    itu_a: float = None
    itu_b: float = None
    itu_c: float = None
    itu_d: float = None

class MaterialList:
    """Container for managing a collection of materials."""
    
    def __init__(self):
        """Initialize an empty material list."""
        self._materials: List[Material] = []
    
    def __getitem__(self, idx: int | List[int]) -> 'Material | MaterialList':
        """Get material(s) by index or indices.
        
        Args:
            idx: Single index or list of indices
            
        Returns:
            Single Material if idx is int, or MaterialList if idx is list
        """
        if isinstance(idx, int):
            return self._materials[idx]
        else:
            # Create new MaterialList with selected materials
            materials = MaterialList()
            materials.add_materials([self._materials[i] for i in idx])
            return materials
    
    def __len__(self) -> int:
        """Get number of materials."""
        return len(self._materials)
    
    def __iter__(self):
        """Iterate over materials."""
        return iter(self._materials)
        
    def __repr__(self) -> str:
        """Get string representation of the material list.
        
        Returns:
            String containing list of materials
        """
        return str(self._materials)
        
    def add_materials(self, materials: List[Material]) -> None:
        """Add materials to the collection.
        
        Args:
            materials: List of Material objects to add
        """
        # Add to main list and filter duplicates
        self._materials.extend(materials)
        self._filter_duplicates()
        
        # Assign IDs after filtering
        for i, mat in enumerate(self._materials):
            mat.id = i
    
    def _filter_duplicates(self) -> None:
        """Remove duplicate materials based on their properties."""
        unique_materials = []
        seen: Set[tuple] = set()
        
        for mat in self._materials:
            # Create hashable key from properties (excluding id)
            mat_key = astuple(mat)[1:]  # Skip the id field
            
            if mat_key not in seen:
                seen.add(mat_key)
                unique_materials.append(mat)
        
        self._materials = unique_materials

    def to_dict(self) -> Dict:
        """Get dictionary representation of all materials.
        
        Returns:
            Dict mapping material IDs to their properties. Note that when saved
            to .mat format, numeric keys will be converted to strings (e.g., '0', '1', etc.)
        """
        return {f'material_{mat.id}': asdict(mat) for mat in self._materials}

    @classmethod
    def from_dict(cls, materials_dict: Dict) -> 'MaterialList':
        """Create MaterialList from dictionary representation.
        
        Args:
            materials_dict: Dictionary mapping material IDs to their properties
            
        Returns:
            MaterialList containing the materials from the dictionary
        """
        materials_list = cls()
        materials = []
        
        for _, mat_data in materials_dict.items():
            # Convert string numeric values to float
            for key, value in mat_data.items():
                if isinstance(value, str) and any(c in value for c in 'e+-0123456789.'):
                    try:
                        mat_data[key] = float(value)
                    except ValueError:
                        pass  # Keep as string if conversion fails
            materials.append(Material(**mat_data))
        
        materials_list.add_materials(materials)
        return materials_list
