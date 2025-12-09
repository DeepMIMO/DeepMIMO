"""Utilities for parsing Wireless InSite XML outputs."""

# %%
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

# InSite XML type constants
I_INT = "remcom_rxapi_Integer"
I_DOUBLE = "remcom_rxapi_Double"
I_BOOL = "remcom_rxapi_Boolean"
I_STRING = "remcom_rxapi_String"
I_POLARIZATION = "remcom_rxapi_PolarizationEnum"


def _parse_value_attribute(value: str) -> bool | int | float | str:
    """Parse the 'Value' attribute string into a specific type."""
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        lower_val = value.lower()
        if lower_val == "true":
            return True
        if lower_val == "false":
            return False
        return value


def xml_to_dict(
    element: ET.Element,
) -> dict[str, Any] | str | int | float | bool | None:
    """Convert XML to a dictionary structure."""
    # Handle special case of Value attribute
    if "Value" in element.attrib:
        return _parse_value_attribute(element.attrib["Value"])

    result: dict[str, Any] = {}

    # Add attributes if any
    if element.attrib:
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

    # If the element has no children and no attributes (result is empty), check for text
    if not result:
        if element.text and element.text.strip():
            return element.text.strip()
        return None

    return result


def parse_insite_xml(xml_file: str) -> dict[str, Any]:
    """Parse InSite XML file into a dictionary."""
    # Read and clean the XML content
    with Path(xml_file).open(encoding="utf-8") as f:
        content = f.read()

    # Remove DOCTYPE and replace :: with _
    content = content.replace("<!DOCTYPE InSite>", "")
    content = content.replace("::", "_")

    # Parse XML and convert to dict
    root = ET.fromstring(content)  # noqa: S314 (Wireless InSite output is trusted input)
    # Include root element in result
    return {root.tag: xml_to_dict(root)}


if __name__ == "__main__":
    xml_file = r"F:\deepmimo_loop_ready\o1b_28\O1_28B.RT_O1_28B.xml"

    # Parse XML and get TxRxSets
    data = parse_insite_xml(xml_file)

    # Get ray tracing parameters
    # rt_params = _get_ray_tracing_params(xml_file)
    # print("\nRay Tracing Parameters:")
    # from pprint import pprint
    # pprint(rt_params)
