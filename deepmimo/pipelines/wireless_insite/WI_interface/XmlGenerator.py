"""XmlGenerator module for XML file generation.

This module provides functionality to generate XML files for electromagnetic simulations,
including study area, ray tracing parameters, and features.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

from lxml import etree

from .Material import Material
from .SetupEditor import SetupEditor
from .TxRxEditor import TxRxEditor

# XML parser
XML_PARSER = etree.XMLParser(recover=True)

# Format string for floating-point values
FLOAT_STR = "%.17g"


class XmlGenerator:
    """Class for generating XML files for electromagnetic simulation.

    This class provides methods to generate XML files for electromagnetic simulations,
    including study area, ray tracing parameters, and features.

    Attributes:
        version (int): Version of the XML format
        scenario_path (str): Path to the scenario directory
        scenario (SetupEditor): SetupEditor instance for the scenario
        name (str): Name of the scenario
        txrx (TxRxEditor): TxRxEditor instance for the scenario
        terrain (Material): Material properties for the terrain
        city (Material): Material properties for the city
        road (Material): Material properties for the road
        xml (etree._ElementTree): XML tree for the study area
        root (etree._Element): Root element of the XML tree
        scene_root (etree._Element): Scene element of the XML tree
        antenna_template_xml (etree._ElementTree): Template for antenna XML
        geometry_city_template_xml (etree._ElementTree): Template for city geometry XML
        geometry_terrain_template_xml (etree._ElementTree): Template for terrain geometry XML
        txrx_point_template_xml (etree._ElementTree): Template for point transmitter/receiver XML
        txrx_grid_template_xml (etree._ElementTree): Template for grid transmitter/receiver XML

    """

    def __init__(
        self,
        scenario_path: str,
        setup: SetupEditor,
        txrx: TxRxEditor,
        version: int = 3,
    ) -> None:
        """Initialize the XmlGenerator with a scenario path and setup instance.

        Args:
            scenario_path (str): Path to the scenario directory
            setup (SetupEditor): SetupEditor instance
            version (int, optional): Version of the XML format. Defaults to 3.

        """
        self.version = version
        self.scenario_path = scenario_path
        self.setup = setup
        self.name = self.setup.name
        self.txrx = txrx

        # Extract material properties from terrain / city / road files
        self.terrain = Material.from_file(self.setup.get_terrain_path())
        self.city = Material.from_file(self.setup.get_city_path())
        # self.road = Material.from_file(self.setup.get_road_path())

        script_dir = str(Path(str(Path(__file__).resolve()).parent))
        self.xml_template_folder = str(Path(script_dir) / "..", "resources", "xml")
        study_area_template = str(Path(self.xml_template_folder) / "template.study_area.xml")
        if self.version >= 4:
            study_area_template = study_area_template.replace(".xml", ".v4.xml")
        self.xml = etree.parse(study_area_template, XML_PARSER)

        self.root = self.xml.getroot()
        self.update_name()

        self.scene_root = self.root.find(".//Scene")[0]
        self.load_templates()

    def load_templates(self) -> None:
        """Load XML templates for antennas, geometries, and transmitters/receivers."""
        antenna_path = str(Path(self.xml_template_folder) / "Antenna.xml")
        geometry_city_path = str(Path(self.xml_template_folder) / "GeometryCity.xml")
        geometry_terrain_path = str(Path(self.xml_template_folder) / "GeometryTerrain.xml")
        tx_point_path = str(Path(self.xml_template_folder) / "TxRxPoint.xml")
        tx_grid_path = str(Path(self.xml_template_folder) / "TxRxGrid.xml")

        if self.version >= 4:
            paths_to_replace = [
                tx_point_path,
                tx_grid_path,
                geometry_city_path,
                geometry_terrain_path,
            ]
            for path in paths_to_replace:
                path = path.replace(".xml", ".v4.xml")

        self.antenna_template_xml = etree.parse(antenna_path, XML_PARSER)
        self.geometry_city_template_xml = etree.parse(geometry_city_path, XML_PARSER)
        self.geometry_terrain_template_xml = etree.parse(geometry_terrain_path, XML_PARSER)
        self.txrx_point_template_xml = etree.parse(tx_point_path, XML_PARSER)
        self.txrx_grid_template_xml = etree.parse(tx_grid_path, XML_PARSER)

    def update_name(self) -> None:
        """Update the name in the XML file."""
        tmp = self.root.find(".//OutputPrefix")
        tmp[0].attrib["Value"] = self.name

        tmp = self.root.find(".//PathResultsDatabase")
        tmp[0][0][0][0][0].attrib["Value"] = (
            tmp[0][0][0][0][0].attrib["Value"].replace("template", self.name)
        )

    def set_carrier_freq(self) -> None:
        """Set the carrier frequency in the XML file."""
        tmp = self.root.findall(".//CarrierFrequency")
        for t in tmp:
            t[0].attrib["Value"] = FLOAT_STR % (self.setup.carrier_frequency)

    def set_bandwidth(self) -> None:
        """Set the bandwidth in the XML file."""
        tmp = self.root.findall(".//Bandwidth")
        for t in tmp:
            t[0].attrib["Value"] = FLOAT_STR % (
                self.setup.bandwidth / 1e6
            )  # bandwidth is in MHz unit in the xml file

    def set_study_area(self) -> None:
        """Set the study area parameters in the XML file."""
        tmp = self.root.findall(".//StudyArea")[0]

        MaxZ = tmp.findall(".//MaxZ")[0]
        MaxZ[0].attrib["Value"] = FLOAT_STR % (self.setup.study_area.zmax)

        MinZ = tmp.findall(".//MinZ")[0]
        MinZ[0].attrib["Value"] = FLOAT_STR % (self.setup.study_area.zmin)

        X = tmp.findall(".//X")[0]
        X[0].attrib["Value"] = " ".join(
            [f"{i:.4g}" for i in self.setup.study_area.all_vertex[:, 0]],
        )

        Y = tmp.findall(".//Y")[0]
        Y[0].attrib["Value"] = " ".join(
            [f"{i:.4g}" for i in self.setup.study_area.all_vertex[:, 1]],
        )

    def set_origin(self) -> None:
        """Set the origin parameters in the XML file."""
        tmp = self.root.findall(".//Origin")[0]

        Latitude = tmp.findall(".//Latitude")[0]
        Latitude[0].attrib["Value"] = FLOAT_STR % (self.setup.origin_lat)

        Longitude = tmp.findall(".//Longitude")[0]
        Longitude[0].attrib["Value"] = FLOAT_STR % (self.setup.origin_lon)

    def set_ray_tracing_param(self) -> None:
        """Set the ray tracing parameters in the XML file."""
        tmp = self.root.findall(".//Model")[0]

        x = tmp.findall(".//MaximumPathsPerReceiverPoint")[0]
        x[0].attrib["Value"] = "%d" % (self.setup.ray_tracing_param.max_paths)

        x = tmp.findall(".//RaySpacing")[0]
        x[0].attrib["Value"] = FLOAT_STR % (self.setup.ray_tracing_param.ray_spacing)

        x = tmp.findall(".//Reflections")[0]
        x[0].attrib["Value"] = "%d" % (self.setup.ray_tracing_param.max_reflections)

        x = tmp.findall(".//Transmissions")[0]
        x[0].attrib["Value"] = "%d" % (self.setup.ray_tracing_param.max_transmissions)

        x = tmp.findall(".//Diffractions")[0]
        x[0].attrib["Value"] = "%d" % (self.setup.ray_tracing_param.max_diffractions)

        x = tmp.findall(".//DiffuseScatteringEnabled")[0]
        x[0].attrib["Value"] = str(self.setup.ray_tracing_param.ds_enable).lower()

        x = tmp.findall(".//DiffuseScatteringReflections")[0]
        x[0].attrib["Value"] = "%d" % (self.setup.ray_tracing_param.ds_max_reflections)

        x = tmp.findall(".//DiffuseScatteringTransmissions")[0]
        x[0].attrib["Value"] = "%d" % (self.setup.ray_tracing_param.ds_max_transmissions)

        x = tmp.findall(".//DiffuseScatteringDiffractions")[0]
        x[0].attrib["Value"] = "%d" % (self.setup.ray_tracing_param.ds_max_diffractions)

        x = tmp.findall(".//DiffuseScatteringFinalInteractionOnly")[0]
        x[0].attrib["Value"] = str(self.setup.ray_tracing_param.ds_final_interaction_only).lower()

    def set_antenna(self) -> None:
        """Set the antenna parameters in the XML file."""
        antenna_parent = self.scene_root.findall("AntennaList")[0][0]
        new_antenna = etree.fromstring(etree.tostring(self.antenna_template_xml), XML_PARSER)
        antenna_parent.append(new_antenna)  # insert b before a

    def set_txrx(self) -> None:
        """Set the transmitter/receiver parameters in the XML file.

        This method handles both single points and arrays of points for the 'points' type,
        and grid configurations for the 'grid' type.
        """
        txrx_parent = self.scene_root.findall("TxRxSetList")[0][0]
        for txrx in self.txrx.txrx[::-1]:
            if txrx.txrx_type == "points":
                if txrx.txrx_pos.ndim == 1:
                    # Single point case
                    new_txrx = etree.fromstring(
                        etree.tostring(self.txrx_point_template_xml),
                        XML_PARSER,
                    )
                    x = new_txrx.findall(".//X")[0]
                    x[0].attrib["Value"] = FLOAT_STR % txrx.txrx_pos[0]
                    y = new_txrx.findall(".//Y")[0]
                    y[0].attrib["Value"] = FLOAT_STR % txrx.txrx_pos[1]
                    z = new_txrx.findall(".//Z")[0]
                    z[0].attrib["Value"] = FLOAT_STR % txrx.txrx_pos[2]
                    txrx_parent.append(new_txrx)
                else:
                    # Multiple points case - create multiple ControlPoints groups
                    new_txrx = etree.fromstring(
                        etree.tostring(self.txrx_point_template_xml),
                        XML_PARSER,
                    )
                    control_points_parent = new_txrx.findall(".//ControlPoints")[0]

                    control_points_parent.clear()
                    point_list = etree.SubElement(
                        control_points_parent,
                        "remcom__rxapi__ProjectedPointList",
                    )

                    # Add each point as a ProjectedPointList/ProjectedPoint
                    for point in txrx.txrx_pos:
                        projected_point = etree.SubElement(point_list, "ProjectedPoint")
                        cartesian_point = etree.SubElement(
                            projected_point,
                            "remcom__rxapi__CartesianPoint",
                        )

                        x = etree.SubElement(cartesian_point, "X")
                        x_value = etree.SubElement(x, "remcom__rxapi__Double")
                        x_value.attrib["Value"] = FLOAT_STR % point[0]

                        y = etree.SubElement(cartesian_point, "Y")
                        y_value = etree.SubElement(y, "remcom__rxapi__Double")
                        y_value.attrib["Value"] = FLOAT_STR % point[1]

                        z = etree.SubElement(cartesian_point, "Z")
                        z_value = etree.SubElement(z, "remcom__rxapi__Double")
                        z_value.attrib["Value"] = FLOAT_STR % point[2]

                    txrx_parent.append(new_txrx)

            elif txrx.txrx_type == "grid":
                new_txrx = etree.fromstring(etree.tostring(self.txrx_grid_template_xml), XML_PARSER)
                x = new_txrx.findall(".//X")[0]
                x[0].attrib["Value"] = FLOAT_STR % txrx.txrx_pos[0]
                y = new_txrx.findall(".//Y")[0]
                y[0].attrib["Value"] = FLOAT_STR % txrx.txrx_pos[1]
                z = new_txrx.findall(".//Z")[0]
                z[0].attrib["Value"] = FLOAT_STR % txrx.txrx_pos[2]

                # Set grid parameters
                if txrx.grid_side is not None:
                    side1 = new_txrx.findall(".//LengthX")[0]
                    side1[0].attrib["Value"] = FLOAT_STR % txrx.grid_side[0]
                    side2 = new_txrx.findall(".//LengthY")[0]
                    side2[0].attrib["Value"] = FLOAT_STR % txrx.grid_side[1]
                if txrx.grid_spacing is not None:
                    spacing = new_txrx.findall(".//Spacing")[0]
                    spacing[0].attrib["Value"] = FLOAT_STR % txrx.grid_spacing
                txrx_parent.append(new_txrx)

            else:
                raise ValueError("Unsupported TxRx type: " + txrx.txrx_type)

            conform_terrain = new_txrx.findall(".//ConformToTerrain")[0]
            conform_terrain[0].attrib["Value"] = "true" if txrx.conform_to_terrain else "false"

            OutputID = new_txrx.findall(".//OutputID")[0]
            OutputID[0].attrib["Value"] = "%d" % txrx.txrx_id
            ShortDescription = new_txrx.findall(".//ShortDescription")[0]
            ShortDescription[0].attrib["Value"] = txrx.txrx_name

            receiver_parent = new_txrx.findall(".//Receiver")[0]
            antenna = receiver_parent.findall(".//Antenna")[0]
            new_antenna = etree.fromstring(etree.tostring(self.antenna_template_xml), XML_PARSER)
            receiver_parent[0].insert(
                receiver_parent[0].index(antenna),
                new_antenna,
            )  # insert b before a
            receiver_parent[0].remove(antenna)

            transmitter_parent = new_txrx.findall(".//Transmitter")[0]
            antenna = transmitter_parent.findall(".//Antenna")[0]
            new_antenna = etree.fromstring(etree.tostring(self.antenna_template_xml), XML_PARSER)
            transmitter_parent[0].insert(
                transmitter_parent[0].index(antenna),
                new_antenna,
            )  # insert b before a
            transmitter_parent[0].remove(antenna)

            if not txrx.is_transmitter:
                new_txrx[0].remove(transmitter_parent)

            if not txrx.is_receiver:
                new_txrx[0].remove(receiver_parent)

            txrx_parent.append(new_txrx)

    def set_geometry(self) -> None:
        """Set the geometry parameters in the XML file."""
        geometry_parent = self.scene_root.findall("GeometryList")[0][0]

        # Map feature types to their corresponding materials and templates
        feature_config = {
            "terrain": {
                "material": self.terrain,
                "template": self.geometry_terrain_template_xml,
                "path_replace": "./pathto.ter",
            },
            "city": {
                "material": self.city,
                "template": self.geometry_city_template_xml,
                "path_replace": "./pathto.city",
            },
            # 'road': {
            #     'material': self.road,
            #     'template': self.geometry_city_template_xml,
            #     'path_replace': './pathto.city'
            # }
        }

        for feature in self.setup.features:
            if feature.type not in feature_config:
                raise ValueError("Unsupported Geometry type: " + feature.type)

            config = feature_config[feature.type]
            new_geometry = etree.fromstring(etree.tostring(config["template"]), XML_PARSER)

            # Set material properties
            self._set_material_properties(new_geometry, config["material"])

            # Replace path placeholder
            new_geometry = etree.tostring(new_geometry, encoding="unicode")
            new_geometry = new_geometry.replace(config["path_replace"], feature.path)

            # Convert back to XML and append
            new_geometry = bytes(new_geometry, "utf-8")
            new_geometry = etree.fromstring(new_geometry, XML_PARSER)
            geometry_parent.append(new_geometry)

    def _set_material_properties(self, geometry: etree._Element, material: Material) -> None:
        """Set material properties in the geometry XML.

        Args:
            geometry (etree._Element): Geometry element
            material (Material): Material properties

        """
        properties = [
            ("Conductivity", material.conductivity, FLOAT_STR),
            ("Permittivity", material.permittivity, FLOAT_STR),
            ("Roughness", material.roughness, FLOAT_STR),
            ("Thickness", material.thickness, FLOAT_STR),
            ("Alpha", material.directive_alpha, "%d"),
            ("Beta", material.directive_beta, "%d"),
            ("CrossPolFraction", material.cross_polarized_power, FLOAT_STR),
            ("Lambda", material.directive_lambda, FLOAT_STR),
            ("ScatteringFactor", material.fields_diffusively_scattered, FLOAT_STR),
        ]

        for prop_name, value, format_str in properties:
            x = geometry.findall(f".//{prop_name}")[0]
            x[0].attrib["Value"] = format_str % value

    def update(self) -> None:
        """Update all parameters in the XML file."""
        self.set_antenna()
        self.set_txrx()
        self.set_geometry()
        self.set_study_area()

        # self.set_origin() # NOTE: the lat/lon location of the origin is not used in the ray tracing
        # It is passed to the setup editor, BUT when we set it in the XML, something breaks.
        # This needs to be investigated if we want to use solely the XML to convert scenarios.

        self.set_carrier_freq()
        self.set_bandwidth()

        self.set_ray_tracing_param()

    def save(self, save_path: str) -> None:
        """Save the XML file.

        Args:
            save_path (str): Path to save the XML file

        """
        ET.indent(self.root, space="  ", level=0)
        t = str(etree.tostring(self.root, pretty_print=True, encoding="unicode"))
        t = "<!DOCTYPE InSite>\n" + t
        t = t.replace("remcom__rxapi__", "remcom::rxapi::")
        # clean the output file before writing
        Path(save_path).open("w+").close()
        with Path(save_path).open("w") as f:
            f.write(t)


if __name__ == "__main__":
    xml = XmlGenerator("scenario_test/", "gwc.setup")
    xml.update()
    xml.save("scenario_test/gwc.study_area.xml")
