"""
AODT Materials Module.

This module handles reading and processing material properties from materials.parquet,
including ITU-R P.2040 parameters and scattering properties.
"""

import os
import pandas as pd
from typing import Dict, Any

def read_materials(rt_folder: str) -> Dict[str, Any]:
    """Read material properties from materials.parquet.

    Args:
        rt_folder (str): Path to folder containing materials.parquet.

    Returns:
        Dict[str, Any]: Dictionary containing material properties including:
            - label: Material name/label
            - itu_params: ITU-R P.2040 parameters (a, b, c, d)
            - scattering: Scattering properties
                - xpd: Cross-polarization discrimination
                - roughness: RMS surface roughness
                - coefficient: Scattering coefficient
            - permittivity: Complex permittivity parameters
                - alpha_r: Real part of exponent
                - alpha_i: Imaginary part of exponent
                - lambda_r: Real part of wavelength factor
            - thickness: Material thickness in meters

    Raises:
        FileNotFoundError: If materials.parquet is not found.
        ValueError: If required parameters are missing.
    """
    materials_file = os.path.join(rt_folder, 'materials.parquet')
    if not os.path.exists(materials_file):
        raise FileNotFoundError(f"materials.parquet not found in {rt_folder}")

    # Read materials data
    df = pd.read_parquet(materials_file)
    if len(df) == 0:
        raise ValueError("materials.parquet is empty")

    # Process materials
    materials_dict = {}
    for _, material in df.iterrows():
        mat_dict = {
            'label': material['label'],
            'itu_params': {
                'a': float(material['itu_r_p2040_a']),
                'b': float(material['itu_r_p2040_b']),
                'c': float(material['itu_r_p2040_c']),
                'd': float(material['itu_r_p2040_d'])
            },
            'scattering': {
                'xpd': float(material['scattering_xpd']),
                'roughness': float(material['rms_roughness']),
                'coefficient': float(material['scattering_coeff'])
            },
            'permittivity': {
                'alpha_r': float(material['exponent_alpha_r']),
                'alpha_i': float(material['exponent_alpha_i']),
                'lambda_r': float(material['lambda_r'])
            },
            'thickness': float(material['thickness_m'])
        }
        materials_dict[material['label']] = mat_dict

    return materials_dict 