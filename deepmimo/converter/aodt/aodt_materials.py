"""
AODT Materials Module.

This module handles reading and processing material properties from materials.parquet,
following ITU-R P.2040 standard for building materials and structures.
"""

import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, Tuple

from ...materials import Material, MaterialList
from .. import converter_utils as cu

@dataclass
class AODTMaterial:
    """AODT material representation following ITU-R P.2040 standard.
    
    This class represents materials as defined in AODT, including their
    electromagnetic, scattering, and frequency-dependent properties.

    Attributes:
        id (int): Material identifier
        label (str): Material name/label
        
        # ITU-R P.2040 parameters for frequency-dependent permittivity
        # ε = a + b*f^c + j*(d*f^c), where f is frequency in GHz
        itu_r_p2040_a (float): Constant term in real part
        itu_r_p2040_b (float): Coefficient of frequency-dependent term in real part
        itu_r_p2040_c (float): Frequency exponent
        itu_r_p2040_d (float): Coefficient of frequency-dependent imaginary part
        
        # Scattering properties based on Degli-Esposti model
        scattering_coeff (float): Scattering coefficient (0-1)
        scattering_xpd (float): Cross-polarization discrimination ratio
        rms_roughness (float): RMS surface roughness in meters
        
        # Directive scattering parameters
        exponent_alpha_r (float): Real part of scattering exponent
        exponent_alpha_i (float): Imaginary part of scattering exponent
        lambda_r (float): Real part of wavelength factor
        
        # Physical properties
        thickness_m (float): Material thickness in meters

    References:
        [1] ITU-R P.2040-3: "Effects of building materials and structures on radio wave 
            propagation above about 100 MHz"
        [2] V. Degli-Esposti et al.: "Measurement and modelling of scattering 
            from buildings"
        [3] E. M. Vitucci et al.: "A Reciprocal Heuristic Model for Diffuse 
            Scattering From Walls and Surfaces"
    """
    # Fields match exactly with materials.parquet columns
    id: int
    label: str
    
    # ITU-R P.2040 parameters
    itu_r_p2040_a: float
    itu_r_p2040_b: float
    itu_r_p2040_c: float
    itu_r_p2040_d: float
    
    # Scattering properties
    scattering_coeff: float
    scattering_xpd: float
    rms_roughness: float
    
    # Directive scattering parameters
    exponent_alpha_r: float
    exponent_alpha_i: float
    lambda_r: float
    
    # Physical properties
    thickness_m: float

    def to_material(self, freq_ghz: float = 1.0) -> Material:
        """Convert AODTMaterial to standard DeepMIMO Material.
        
        Args:
            freq_ghz: Frequency in GHz for calculating permittivity. Defaults to 1.0 GHz.
            
        Returns:
            Material: Standardized material representation
            
        Notes:
            - Permittivity is calculated using ITU-R P.2040 formula at the specified frequency
            - Conductivity is derived from the imaginary part of permittivity
            - AODT uses directive scattering by default
            - Both calculated permittivity/conductivity and ITU parameters are stored
        """
        # Calculate complex permittivity at specified frequency
        # ε = a + b*f^c + j*(d*f^c)
        eps_real = self.itu_r_p2040_a + self.itu_r_p2040_b * (freq_ghz**self.itu_r_p2040_c)
        eps_imag = self.itu_r_p2040_d * (freq_ghz**self.itu_r_p2040_c)
        
        # Convert imaginary permittivity to conductivity
        # σ = ω*ε0*ε" = 2π*f*ε0*(d*f^c)
        # ε0 = 8.854e-12 F/m
        eps0 = 8.854e-12
        conductivity = 2 * np.pi * (freq_ghz * 1e9) * eps0 * eps_imag
        
        return Material(
            id=self.id,
            name=self.label,
            # Store both calculated values and ITU parameters
            permittivity=eps_real,
            conductivity=conductivity,
            # Store original ITU-R P.2040 parameters
            itu_a=self.itu_r_p2040_a,
            itu_b=self.itu_r_p2040_b,
            itu_c=self.itu_r_p2040_c,
            itu_d=self.itu_r_p2040_d,
            # Scattering properties
            scattering_model=Material.SCATTERING_DIRECTIVE,  # AODT uses directive scattering
            scattering_coefficient=self.scattering_coeff,
            cross_polarization_coefficient=self.scattering_xpd,
            alpha_r=self.exponent_alpha_r,
            alpha_i=self.exponent_alpha_i,
            lambda_param=self.lambda_r,
            roughness=self.rms_roughness,
            thickness=self.thickness_m
        )

def read_materials(rt_folder: str, save_folder: str = None) -> Tuple[Dict, Dict[str, int]]:
    """Read material properties from materials.parquet.

    Args:
        rt_folder: Path to folder containing materials.parquet
        save_folder: Optional path to save converted materials. If None, materials won't be saved.

    Returns:
        Tuple containing:
            - Dict containing materials and their properties
            - Dict mapping material labels to their indices

    Raises:
        FileNotFoundError: If materials.parquet is not found
        ValueError: If required parameters are missing
    """
    materials_file = os.path.join(rt_folder, 'materials.parquet')
    if not os.path.exists(materials_file):
        raise FileNotFoundError(f"materials.parquet not found in {rt_folder}")

    # Read materials data
    df = pd.read_parquet(materials_file)
    if len(df) == 0:
        raise ValueError("materials.parquet is empty")

    # Initialize material list and indices mapping
    material_list = MaterialList()
    material_indices = {}

    # Process materials
    materials = []
    for i, mat in df.iterrows():
        # Create AODT material directly from dataframe row
        mat_dict = mat.to_dict()
        mat_dict['id'] = i  # Add ID to the dictionary
        aodt_material = AODTMaterial(**mat_dict)
        
        # Convert to DeepMIMO material
        material = aodt_material.to_material()
        materials.append(material)
        material_indices[mat['label']] = i

    # Add all materials to the list
    material_list.add_materials(materials)

    # Save material indices only if save_folder is provided
    if save_folder is not None:
        cu.save_mat(material_indices, 'materials', save_folder)

    return material_list.to_dict()