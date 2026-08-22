"""Wireless Insite Material Representation.

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepmimo.core.materials import Material, MaterialList  # Base material classes

# Local imports
from .setup_parser import parse_file  # For parsing Wireless InSite setup-like files


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
        Wireless InSite 3.3.0 Reference Manual (section 10.5) states all materials are
        nonmagnetic, with permeability of free space (µ0 = 4π x 10e-7 H/m).

    Sources:
        [1] A Diffuse Scattering Model for Urban Propagation Prediction
            - Vittorio Degli-Esposti 2001
            https://ieeexplore.ieee.org/document/4052607

    """

    id: int = -1
    name: str = ""
    conductivity: float = 0.0
    permittivity: float = 0.0
    roughness: float = 0.0
    thickness: float = 0.0

    # Scattering properties
    diffuse_scattering_model: str = ""
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
            "": Material.SCATTERING_NONE,
            "lambertian": Material.SCATTERING_LAMBERTIAN,
            "directive": Material.SCATTERING_DIRECTIVE,
            "directive_with_backscatter": Material.SCATTERING_DIRECTIVE,
        }

        return Material(
            id=self.id,
            name=self.name,
            permittivity=self.permittivity,
            conductivity=self.conductivity,
            scattering_model=model_mapping.get(
                self.diffuse_scattering_model,
                Material.SCATTERING_NONE,
            ),
            scattering_coefficient=self.fields_diffusively_scattered,
            cross_polarization_coefficient=self.cross_polarized_power,
            alpha_r=self.directive_alpha,
            alpha_i=self.directive_beta,
            lambda_param=self.directive_lambda,
            roughness=self.roughness,
            thickness=self.thickness,
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
    name: str = ""
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
            conductivity=0.0,
        )


# Extensions of files that declare materials. Wireless InSite writes standalone
# geometry as ".object"; ".obj" is kept for compatibility with renamed exports.
MATERIAL_FILE_EXTS = (".city", ".ter", ".veg", ".flp", ".obj", ".object")


def _outer_layer(mat: Any) -> Any:
    """Return the dielectric layer used to characterise a material.

    Layered materials (e.g. "ITU Layered drywall") declare several
    ``begin_<DielectricLayer>`` blocks. They are parsed into a list; the last
    entry is used, matching the value this parser has always reported for
    single-layer materials.

    Args:
        mat: Parsed material node.

    Returns:
        The dielectric layer node to read properties from.

    """
    layers = mat.all_values.get("DielectricLayer")
    return layers[-1] if layers else mat.values["DielectricLayer"]


# Conductivity used for perfect electric conductors. Wireless InSite marks these
# with a bare "PEC" label and declares no dielectric layer. A large finite value is
# used instead of infinity so the exported material stays valid JSON, and is
# numerically equivalent for reflection (|gamma| ~ 1 either way).
PEC_CONDUCTIVITY = 1e7
PEC_PERMITTIVITY = 1.0


def _build_material(mat: Any) -> Material:
    """Build a Material from a parsed Wireless Insite material node.

    Args:
        mat: Parsed material node.

    Returns:
        The corresponding Material.

    """
    if "PEC" in mat.labels:
        return InsiteMaterial(
            name=mat.name,
            diffuse_scattering_model=mat.values.get("diffuse_scattering_model", ""),
            fields_diffusively_scattered=float(
                mat.values.get("fields_diffusively_scattered", 0.0),
            ),
            cross_polarized_power=float(mat.values.get("cross_polarized_power", 0.0)),
            directive_alpha=float(mat.values.get("directive_alpha", 4.0)),
            directive_beta=float(mat.values.get("directive_beta", 4.0)),
            directive_lambda=float(mat.values.get("directive_lambda", 0.5)),
            conductivity=PEC_CONDUCTIVITY,
            permittivity=PEC_PERMITTIVITY,
            roughness=float(mat.values.get("roughness", 0.0)),
            thickness=float(mat.values.get("thickness", 0.0)),
        ).to_material()

    if "diffuse_scattering_model" not in mat.values and mat.values.get("thickness", False):
        # Foliage!
        insite_mat = InsiteFoliage(
            name=mat.name,
            thickness=float(mat.values["thickness"]),
            density=float(mat.values["density"]),
            vertical_attenuation=float(mat.values["VerticalAttenuation"]),
            horizontal_attenuation=float(mat.values["HorizontalAttenuation"]),
            permittivity_vr=float(mat.values["permittivity_vr"]),
            permittivity_hr=float(mat.values["permittivity_hr"]),
        )
    else:
        insite_mat = InsiteMaterial(
            name=mat.name,
            diffuse_scattering_model=mat.values.get("diffuse_scattering_model", ""),
            fields_diffusively_scattered=float(
                mat.values.get("fields_diffusively_scattered", 0.0),
            ),
            cross_polarized_power=float(mat.values.get("cross_polarized_power", 0.0)),
            directive_alpha=float(mat.values.get("directive_alpha", 4.0)),
            directive_beta=float(mat.values.get("directive_beta", 4.0)),
            directive_lambda=float(mat.values.get("directive_lambda", 0.5)),
            conductivity=float(_outer_layer(mat).values["conductivity"]),
            permittivity=float(_outer_layer(mat).values["permittivity"]),
            roughness=float(_outer_layer(mat).values["roughness"]),
            thickness=float(_outer_layer(mat).values["thickness"]),
        )
    return insite_mat.to_material()


def parse_materials_with_indices(file: Path) -> list[tuple[int, Material]]:
    """Parse materials from a file, keeping the index each one declares.

    Wireless InSite faces reference materials by the index written inside the
    material block (``Material 3``). Those indices are neither contiguous nor
    equal to the order of declaration, so they must be read explicitly rather
    than inferred from position.

    Args:
        file: Path to file to read.

    Returns:
        List of (declared index, material) pairs, in order of declaration.

    """
    document = parse_file(file)
    indexed = []

    for prim in document:
        # all_values keeps every <Material> block; values would keep only the last.
        mat_entries = document[prim].all_values.get("Material", [])
        for mat in mat_entries:
            # Fall back to declaration order for files that omit the index.
            declared = mat.values.get("Material")
            declared_idx = int(declared) if isinstance(declared, int | str) else len(indexed)
            indexed.append((declared_idx, _build_material(mat)))

    return indexed


def parse_materials_from_file(file: Path) -> list[Material]:
    """Parse materials from a single Wireless Insite file.

    Args:
        file: Path to file to read

    Returns:
        List of Material objects

    """
    return [material for _, material in parse_materials_with_indices(file)]


def read_materials(sim_folder: str, *, verbose: bool = False) -> dict:
    """Read materials from a Wireless Insite simulation folder.

    Args:
        sim_folder: Path to simulation folder containing material files (.city, .ter, .veg)
        verbose: Whether to print debug information

    Returns:
        Dict containing materials and their properties

    """
    sim_folder = Path(sim_folder)
    if not sim_folder.exists():
        msg = f"Simulation folder does not exist: {sim_folder}"
        raise ValueError(msg)

    # Initialize material list
    material_list = MaterialList()

    # Find all material files
    material_files = []
    for ext in MATERIAL_FILE_EXTS:
        material_files.extend(sim_folder.glob(f"*{ext}"))

    if not material_files:
        msg = f"No material files found in {sim_folder}"
        raise ValueError(msg)

    # Parse materials from each file
    for file in material_files:
        print(f"Parsing materials from {file}")
        materials = parse_materials_from_file(file)
        material_list.add_materials(materials)

    if verbose:
        print("\nMaterial list:")

    return material_list.to_dict()


if __name__ == "__main__":
    # Test directory with material files
    test_dir = r"./P2Ms/simple_street_canyon_test/"

    # Get all files in test directory
    files = [
        str(Path(root) / filename)
        for root, _, filenames in os.walk(test_dir)
        for filename in filenames
    ]

    print(f"\nTesting materials extraction from: {test_dir}")
    print("-" * 50)

    # Basic test
    materials_dict = read_materials(test_dir, verbose=True)
    print(f"\nTotal materials found: {len(materials_dict)}")
