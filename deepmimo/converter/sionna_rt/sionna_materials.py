"""
Sionna Ray Tracing Materials Module.

This module handles loading and converting material data from Sionna's format to DeepMIMO's format.
"""

import os
from typing import Dict, Tuple

from ...materials import Material, MaterialList
from ...general_utils import load_pickle
from ..converter_utils import save_mat

def read_materials(load_folder: str, save_folder: str) -> Tuple[Dict, Dict[str, int]]:
    """Read materials from a Sionna RT simulation folder.
    
    Args:
        load_folder: Path to simulation folder containing material files
        save_folder: Path to save converted materials
        
    Returns:
        Tuple of (Dict containing materials and their categorization,
                 Dict mapping object names to material indices)
    """
    # Load Sionna materials
    material_properties = load_pickle(os.path.join(load_folder, 'sionna_materials.pkl'))
    material_indices = load_pickle(os.path.join(load_folder, 'sionna_material_indices.pkl'))

    # Initialize material list
    material_list = MaterialList()
    
    # Attribute matching for scattering models
    scat_model = {
        'LambertianPattern': Material.SCATTERING_LAMBERTIAN,
        'DirectivePattern': Material.SCATTERING_DIRECTIVE,
        'BackscatteringPattern': Material.SCATTERING_DIRECTIVE  # directive = backscattering
    }

    # Convert each Sionna material to DeepMIMO Material
    materials = []
    for i, mat_property in enumerate(material_properties):
        # Get scattering model type and handle case where scattering is disabled
        scattering_model = scat_model[mat_property['scattering_pattern']]
        scat_coeff = mat_property['scattering_coefficient']
        scattering_model = Material.SCATTERING_NONE if not scat_coeff else scattering_model
        
        # Create Material object
        def safe_float(val, default=0.0):
            return float(val) if val is not None else default
        material = Material(
            id=i,
            name=f'material_{i}',  # Default name if not provided
            permittivity=float(mat_property['relative_permittivity']),
            conductivity=float(mat_property['conductivity']),
            scattering_model=scattering_model,
            scattering_coefficient=float(scat_coeff),
            cross_polarization_coefficient=float(mat_property['xpd_coefficient']),
            alpha_r=safe_float(mat_property['alpha_r']),
            alpha_i=safe_float(mat_property['alpha_i']),
            lambda_param=safe_float(mat_property['lambda_'])
        )
        materials.append(material)
    
    # Add all materials to buildings category by default
    # This can be modified if Sionna provides material categorization
    material_list.add_materials(materials)
    
    # Save materials indices to matrix file
    save_mat(material_indices, 'materials', save_folder)
    
    return material_list.to_dict(), material_indices 