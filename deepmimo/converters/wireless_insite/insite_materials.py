"""
Wireless Insite Material Representation.

This module provides classes for representing materials and their properties as defined
in Wireless Insite ray tracing software.

This module provides:
- Base material class with electromagnetic properties
- Foliage material class with attenuation properties
- Conversion utilities to standard DeepMIMO material format

The module serves as the interface between Wireless Insite's material definitions
and DeepMIMO's standardized material representation.
"""
# Standard library imports
import os
from typing import List, Dict
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint

# Local imports
from .setup_parser import parse_file  # For parsing Wireless InSite setup-like files
from ...materials import Material, MaterialList  # Base material classes


@dataclass
class InsiteMaterial:
    """Wireless Insite material representation.
    
    This class represents materials as defined in Wireless Insite, including their
    electromagnetic and scattering properties.

    Attributes:
        id (int): Material identifier. Defaults to -1.
        name (str): Material name. Defaults to ''.
        conductivity (float): Conductivity in S/m. Defaults to 0.0.
        permittivity (float): Relative permittivity. Defaults to 0.0.
        roughness (float): Surface roughness in meters. Defaults to 0.0.
        thickness (float): Material thickness in meters. Defaults to 0.0.
        diffuse_scattering_model (str): Scattering model type. Defaults to ''.
        fields_diffusively_scattered (float): Fraction of scattered power. Defaults to 0.0.
        cross_polarized_power (float): Cross-polarization ratio. Defaults to 0.0.
        directive_alpha (float): Forward scattering width. Defaults to 4.0.
        directive_beta (float): Backward scattering width. Defaults to 4.0.
        directive_lambda (float): Forward/backward ratio. Defaults to 0.5.

    Notes:
        Scattering model parameters based on [1] and extended with cross-polarization terms.
        At present, according to the Wireless InSite 3.3.0 Reference Manual, section 10.5,
        all materials are nonmagnetic, and their permeability is that of free space (µ0 = 4π x 10e-7 H/m). 
    
    Sources:
        [1] A Diffuse Scattering Model for Urban Propagation Prediction - Vittorio Degli-Esposti 2001
            https://ieeexplore.ieee.org/document/4052607
        
    """
    
    id: int = -1
    name: str = ''
    conductivity: float = 0.0
    permittivity: float = 0.0
    roughness: float = 0.0
    thickness: float = 0.0
    
    # Scattering properties
    diffuse_scattering_model: str = ''
    fields_diffusively_scattered: float = 0.0
    cross_polarized_power: float = 0.0
    directive_alpha: float = 4.0
    directive_beta: float = 4.0
    directive_lambda: float = 0.5

    def to_material(self) -> Material:
        """Convert InsiteMaterial to standard DeepMIMO Material.
        
        Returns:
            Material: Standardized material representation
            
        Notes:
            Maps Wireless Insite scattering models to standard DeepMIMO models:
            - '' -> none
            - 'lambertian' -> lambertian
            - 'directive'/'directive_with_backscatter' -> directive
        """
        # Map scattering model names
        model_mapping = {
            '': Material.SCATTERING_NONE,
            'lambertian': Material.SCATTERING_LAMBERTIAN,
            'directive': Material.SCATTERING_DIRECTIVE,
            'directive_with_backscatter': Material.SCATTERING_DIRECTIVE
        }
        
        return Material(
            id=self.id,
            name=self.name,
            permittivity=self.permittivity,
            conductivity=self.conductivity,
            scattering_model=model_mapping.get(self.diffuse_scattering_model, Material.SCATTERING_NONE),
            scattering_coefficient=self.fields_diffusively_scattered,
            cross_polarization_coefficient=self.cross_polarized_power,
            alpha_r=self.directive_alpha,
            alpha_i=self.directive_beta,
            lambda_param=self.directive_lambda,
            roughness=self.roughness,
            thickness=self.thickness
        )


@dataclass
class InsiteFoliage:
    """Wireless Insite foliage material representation.
    
    This class represents vegetation/foliage materials in Wireless Insite, including
    their attenuation and electromagnetic properties.

    Attributes:
        id (int): Material identifier. Defaults to -1.
        name (str): Material name. Defaults to ''.
        thickness (float): Material thickness in meters. Defaults to 0.0.
        density (float): Material density in kg/m³. Defaults to 0.0.
        vertical_attenuation (float): Vertical attenuation in dB/m. Defaults to 0.0.
        horizontal_attenuation (float): Horizontal attenuation in dB/m. Defaults to 0.0.
        permittivity_vr (float): Vertical relative permittivity. Defaults to 0.0.
        permittivity_hr (float): Horizontal relative permittivity. Defaults to 0.0.

    Notes:
        Implementation based on Wireless InSite 3.3.0 Reference Manual, section 10.5
    """
    
    id: int = -1
    name: str = ''
    thickness: float = 0.0
    density: float = 0.0
    vertical_attenuation: float = 0.0
    horizontal_attenuation: float = 0.0
    permittivity_vr: float = 0.0
    permittivity_hr: float = 0.0

    def to_material(self) -> Material:
        """Convert InsiteFoliage to standard DeepMIMO Material.
        
        Returns:
            Material: Standardized material representation with foliage properties
        """
        return Material(
            id=self.id,
            name=self.name,
            permittivity=self.permittivity_vr,
            thickness=self.thickness,
            scattering_model=Material.SCATTERING_NONE,
            vertical_attenuation=self.vertical_attenuation,
            horizontal_attenuation=self.horizontal_attenuation,
        )

def parse_materials_from_file(file: Path) -> List[Material]:
    """Parse materials from a single Wireless Insite file.
    
    Args:
        file: Path to file to read
        
    Returns:
        List of Material objects
    """
    document = parse_file(file)
    materials = []
    
    for prim in document.keys():
        mat_entries = document[prim].values['Material']
        mat_entries = [mat_entries] if not isinstance(mat_entries, list) else mat_entries
        
        for mat in mat_entries:
            if 'diffuse_scattering_model' not in mat.values and mat.values.get('thickness', False):
                # Foliage!
                insite_mat = InsiteFoliage(
                    name=mat.name,
                    thickness=float(mat.values['thickness']),
                    density=float(mat.values['density']),
                    vertical_attenuation=float(mat.values['VerticalAttenuation']),
                    horizontal_attenuation=float(mat.values['HorizontalAttenuation']),
                    permittivity_vr=float(mat.values['permittivity_vr']),
                    permittivity_hr=float(mat.values['permittivity_hr']),
                )
            else:
                # Create InsiteMaterial object
                insite_mat = InsiteMaterial(
                    name=mat.name,
                    diffuse_scattering_model=mat.values.get('diffuse_scattering_model', ''),
                    fields_diffusively_scattered=float(mat.values.get('fields_diffusively_scattered', 0.0)),
                    cross_polarized_power=float(mat.values.get('cross_polarized_power', 0.0)),
                    directive_alpha=float(mat.values.get('directive_alpha', 4.0)),
                    directive_beta=float(mat.values.get('directive_beta', 4.0)),
                    directive_lambda=float(mat.values.get('directive_lambda', 0.5)),
                    conductivity=float(mat.values['DielectricLayer'].values['conductivity']),
                    permittivity=float(mat.values['DielectricLayer'].values['permittivity']),
                    roughness=float(mat.values['DielectricLayer'].values['roughness']),
                    thickness=float(mat.values['DielectricLayer'].values['thickness'])
                )
            
            # Convert to base Material
            materials.append(insite_mat.to_material())
    
    return materials

def read_materials(sim_folder: str, verbose: bool = False) -> Dict:
    """Read materials from a Wireless Insite simulation folder.
    
    Args:
        sim_folder: Path to simulation folder containing material files (.city, .ter, .veg)
        verbose: Whether to print debug information
        
    Returns:
        Dict containing materials and their properties
    """
    sim_folder = Path(sim_folder)
    if not sim_folder.exists():
        raise ValueError(f"Simulation folder does not exist: {sim_folder}")
    
    # Initialize material list
    material_list = MaterialList()
    
    # Find all material files
    material_files = []
    for ext in ['.city', '.ter', '.veg', '.flp', '.obj']:
        material_files.extend(sim_folder.glob(f"*{ext}"))
    
    if not material_files:
        raise ValueError(f"No material files found in {sim_folder}")
    
    # Parse materials from each file
    for file in material_files:
        print(f"Parsing materials from {file}")
        materials = parse_materials_from_file(file)
        material_list.add_materials(materials)
            
    if verbose:
        print('\nMaterial list:')
        pprint(material_list.to_dict())
        
    return material_list.to_dict()


if __name__ == "__main__":
    # Test directory with material files
    test_dir = r"./P2Ms/simple_street_canyon_test/"
    
    # Get all files in test directory
    files = []
    for root, _, filenames in os.walk(test_dir):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    
    print(f"\nTesting materials extraction from: {test_dir}")
    print("-" * 50)
    
    # Basic test
    materials_dict = read_materials(test_dir, verbose=True)
    print(f"\nTotal materials found: {len(materials_dict)}")
