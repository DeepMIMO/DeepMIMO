"""Sionna Utils Module.

This module contains utility functions for Sionna.
"""

import sionna
import sionna.rt as sionna_rt
from sionna.rt import (
    BackscatteringPattern,
    PlanarArray,
    RadioMaterial,
    Scene,
    load_scene,
)

# --- Material parameters for itu_concrete ---
# Values calibrated for 3.5 GHz urban environments (ITU-R P.2040 Table 3).
# Scattering pattern must be assigned as an object, not a string, when
# modifying an existing RadioMaterial (string assignment is only valid in
# the RadioMaterial constructor).
_CONCRETE_SCATTERING_COEFF = 0.4
_CONCRETE_XPD_COEFF = 0.4
_CONCRETE_ALPHA_R = 4
_CONCRETE_ALPHA_I = 4
_CONCRETE_LAMBDA = 0.75

# --- Asphalt material parameters ---
# ITU-R P.2040 does not include asphalt; these values come from measurement
# campaigns at similar frequencies and match common simulation practice.
_ASPHALT_PERMITTIVITY = 5.72
_ASPHALT_CONDUCTIVITY = 5e-4
_ASPHALT_SCATTERING_COEFF = 0.4
_ASPHALT_XPD_COEFF = 0.4


def get_sionna_version() -> str | None:
    """Try to get Sionna or Sionna RT version string, or return None if not found."""
    try:
        if hasattr(sionna, "__version__"):
            return sionna.__version__
        if hasattr(sionna_rt, "__version__"):
            return sionna_rt.__version__
    except (ImportError, AttributeError):
        pass
    return None


def set_materials(scene: Scene) -> Scene:
    """Set radio material properties for a custom Sionna scene.

    Applies calibrated ITU-based parameters to concrete surfaces and creates
    an asphalt material for road/path objects.  Only modifies objects with
    known material names; unknown materials are left unchanged.
    """
    for obj in scene.objects.values():
        print(f"Setting material for {obj.name}")
        mat_name = scene.objects[obj.name].radio_material.name
        print(f"Material name: {mat_name}")

        if mat_name == "itu_concrete":
            mat = scene.objects[obj.name].radio_material
            mat.scattering_coefficient = _CONCRETE_SCATTERING_COEFF
            mat.xpd_coefficient = _CONCRETE_XPD_COEFF
            # Must assign a BackscatteringPattern object here — assigning a
            # string to an existing RadioMaterial raises a TypeError in 2.0.
            mat.scattering_pattern = BackscatteringPattern(
                alpha_r=_CONCRETE_ALPHA_R,
                alpha_i=_CONCRETE_ALPHA_I,
                lambda_=_CONCRETE_LAMBDA,
            )
        elif mat_name in ["itu_wet_ground", "itu_brick"]:
            # Default ITU parameters are acceptable for these materials
            continue
        else:
            print(f"Unknown material: {mat_name}")

    # String form of scattering_pattern is accepted by the RadioMaterial
    # constructor (but not by assignment on an existing material, see above).
    asphalt_material = RadioMaterial(
        name="asphalt",
        relative_permittivity=_ASPHALT_PERMITTIVITY,
        conductivity=_ASPHALT_CONDUCTIVITY,
        scattering_coefficient=_ASPHALT_SCATTERING_COEFF,
        xpd_coefficient=_ASPHALT_XPD_COEFF,
        scattering_pattern="backscattering",
    )
    scene.add(asphalt_material)

    for obj in scene.objects:
        if "road" in obj or "path" in obj:
            scene.objects[obj].radio_material = asphalt_material
            print(f"Set asphalt material for {obj}")

    return scene


def create_base_scene(scene_path: str, center_frequency: float) -> Scene:
    """Create a base Sionna scene with a single isotropic antenna at TX and RX.

    Args:
        scene_path: Path to the .xml scene file, or a Sionna built-in scene
            path string (e.g. ``sionna.rt.scene.munich``).
        center_frequency: Carrier frequency in Hz.

    Returns:
        Configured Sionna Scene object.

    """
    args: dict = {"merge_shapes": False}  # keep objects separate for material assignment
    if scene_path:
        args["filename"] = scene_path

    scene = load_scene(**args)

    scene.frequency = center_frequency

    # Default to a single isotropic element; callers can override after creation
    single_element = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    scene.tx_array = single_element
    scene.rx_array = single_element
    return scene
