# %%
import xml.etree.ElementTree as ET
from pprint import pprint
from typing import Any

# InSite XML type constants
I_INT = "remcom_rxapi_Integer"
I_DOUBLE = "remcom_rxapi_Double"
I_BOOL = "remcom_rxapi_Boolean"
I_STRING = "remcom_rxapi_String"
I_POLARIZATION = "remcom_rxapi_PolarizationEnum"


def xml_to_dict(element: ET.Element) -> dict[str, Any] | str | None:
    """Convert XML to a dictionary structure"""
    result: dict[str, Any] = {}

    # Add attributes if any
    if element.attrib:
        if "Value" in element.attrib:
            # Handle special case of Value attribute
            value = element.attrib["Value"]
            try:
                if "." in value:
                    return float(value)
                return int(value)
            except ValueError:
                if value.lower() == "true":
                    return True
                if value.lower() == "false":
                    return False
                return value
        else:
            result.update(element.attrib)

    # Add children
    for child in element:
        child_data = xml_to_dict(child)
        tag = child.tag.replace("remcom::rxapi::", "remcom_rxapi_")

        if tag in result:
            # If the tag already exists, convert it to a list if it isn't already
            if not isinstance(result[tag], list):
                result[tag] = [result[tag]]
            result[tag].append(child_data)
        else:
            result[tag] = child_data

    # If the element has no children and no attributes, return None
    if not result and not element.attrib:
        return None

    return result


def parse_insite_xml(xml_file: str) -> dict[str, Any]:
    """Parse InSite XML file into a dictionary"""
    # Read and clean the XML content
    with open(xml_file, encoding="utf-8") as f:
        content = f.read()

    # Remove DOCTYPE and replace :: with _
    content = content.replace("<!DOCTYPE InSite>", "")
    content = content.replace("::", "_")

    # Parse XML and convert to dict
    root = ET.fromstring(content)
    return xml_to_dict(root)


def _get_ray_tracing_params(xml_file: str) -> dict[str, Any]:
    """Extract ray tracing parameters from InSite XML file.

    Args:
        xml_file: Path to InSite XML file

    Returns:
        Dictionary of ray tracing parameters

    """
    raise NotImplementedError("Not implemented yet")
    # The .setup seems to work better, but the .xml is more complete.
    # So we should use the .setup file to get the parameters, and the .xml
    # file to get the txrx data. For now, we'll keep it like that (changes in beta.)

    from deepmimo.config import config
    from deepmimo.consts import RAYTRACER_NAME_WIRELESS_INSITE
    from deepmimo.rt_params import RayTracingParameters

    # Parse XML and get data
    data = parse_insite_xml(xml_file)

    # Get waveform data for frequency
    waveform = data["remcom_rxapi_Job"]["Scene"]["remcom_rxapi_Scene"]["WaveformList"][
        "remcom_rxapi_WaveformList"
    ]["Waveform"]["remcom_rxapi_Sinusoid"]

    # Get model data
    model = data["remcom_rxapi_Job"]["Model"]["remcom_rxapi_X3D"]

    # Get APG acceleration data
    apg_accel = model["APGAccelerationParameters"]["remcom_rxapi_APGAccelerationParameters"]

    # Get diffuse scattering data
    diffuse_scat = model["DiffuseScattering"]["remcom_rxapi_DiffuseScattering"]

    # Extract parameters
    params = {
        # Ray Tracing Engine info
        "raytracer_name": RAYTRACER_NAME_WIRELESS_INSITE,
        "raytracer_version": config.get("wireless_insite_version"),
        # Frequency from waveform
        "frequency": waveform["CarrierFrequency"][I_DOUBLE],
        # Ray tracing interaction settings
        "max_path_depth": apg_accel["PathDepth"][I_INT],
        "max_reflections": model["MaxReflections"][I_INT],
        "max_diffractions": model["Diffractions"][I_INT],
        "max_scattering": 1 if diffuse_scat["Enabled"][I_BOOL] else 0,
        "max_transmissions": model.get("MaxTransmissions", {I_INT: 0})[I_INT],
        # Diffuse scattering settings
        "diffuse_reflections": diffuse_scat["DiffuseReflections"][I_INT],
        "diffuse_diffractions": diffuse_scat["DiffuseDiffractions"][I_INT],
        "diffuse_transmissions": diffuse_scat["DiffuseTransmissions"][I_INT],
        "diffuse_final_interaction_only": diffuse_scat["FinalInteractionOnly"][I_BOOL],
        "diffuse_random_phases": False,  # Not supported in InSite
        # Terrain interaction settings
        "terrain_reflection": model.get("TerrainReflections", {I_BOOL: True})[I_BOOL],
        "terrain_diffraction": model.get("TerrainDiffractions", {I_BOOL: False})[I_BOOL],
        "terrain_scattering": model.get("TerrainScattering", {I_BOOL: False})[I_BOOL],
        # Ray casting settings
        "num_rays": 360 // model["RaySpacing"][I_DOUBLE] * 180,  # Calculate from ray spacing
        "ray_casting_method": "uniform",  # InSite uses uniform ray casting
        "synthetic_array": True,  # Currently only synthetic arrays are supported
        # Store raw parameters for reference
        "raw_params": {
            "waveform": waveform,
            "model": model,
            "apg_acceleration": apg_accel,
            "diffuse_scattering": diffuse_scat,
        },
    }

    return RayTracingParameters.from_dict(params)


if __name__ == "__main__":
    xml_file = r"F:\deepmimo_loop_ready\o1b_28\O1_28B.RT_O1_28B.xml"

    # Parse XML and get TxRxSets
    data = parse_insite_xml(xml_file)
    pprint(data)

    # Get ray tracing parameters
    # rt_params = _get_ray_tracing_params(xml_file)
    # print("\nRay Tracing Parameters:")
    # from pprint import pprint
    # pprint(rt_params)
