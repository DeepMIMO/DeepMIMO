"""Convert PLY files to city files for electromagnetic simulation.

This module provides functionality to convert PLY (Polygon File Format) files to
city files used in electromagnetic simulations, including material properties.
"""

from pathlib import Path
from typing import Any, TextIO

from plyfile import PlyData  # type: ignore[import]


def convert_ply2city(
    ply_path: str,
    material_path: str,
    save_path: str,
    object_name: str | None = None,
) -> tuple[int, int]:
    """Convert a PLY file to a city file with material properties.

    Args:
        ply_path (str): Path to the PLY file
        material_path (str): Path to the material file
        save_path (str): Path to save the city file
        object_name (Optional[str], optional): Name of the object.
                                             If None, derived from ply_path. Defaults to None.

    Returns:
        Tuple[int, int]: Number of vertices and faces in the city file

    """
    if not object_name:
        object_name = ply_path.split(".")[0].split("/")[-1]

    ply_data = PlyData.read(ply_path)
    with Path(material_path).open() as f:
        material_sec = f.readlines()

    with Path(save_path).open("w") as f:
        f.write("Format type:keyword version: 1.1.0\n")
        f.write("begin_<city> " + object_name + "\n")
        write_reference_sec(f)
        write_material_sec(f, material_sec)
        write_face_sec(f, ply_data)
        f.write("end_<city>\n")
    return (len(ply_data["vertex"]), len(ply_data["face"]))


def write_reference_sec(f: Any) -> None:
    """Write the reference section to the city file.

    Args:
        f: File object to write to

    """
    script_dir = str(Path(str(Path(__file__).resolve()).parent))
    ref_path = str(Path(script_dir) / "resources", "reference_section.txt")
    with Path(ref_path).open() as f1:
        reference_sec = f1.readlines()
    return f.writelines(reference_sec)


def write_material_sec(f: Any, material_sec: list[str]) -> None:
    """Write the material section to the city file.

    Args:
        f: File object to write to
        material_sec (List[str]): Material section lines

    """
    # Replace 'none' by 'directive_with_backscatter' in 'diffuse_scattering_model'
    idx_to_replace = next(
        line_idx
        for line_idx, line in enumerate(material_sec)
        if line == "diffuse_scattering_model none\n"
    )
    material_sec[idx_to_replace] = "diffuse_scattering_model directive_with_backscatter\n"

    return f.writelines(material_sec)


def write_face_sec(f: TextIO, ply_data: PlyData) -> None:
    """Write the face section to the city file.

    Args:
        f: File object to write to
        ply_data (PlyData): PLY data containing vertices and faces

    """
    f.write("begin_<structure_group> \n")
    f.write("begin_<structure> \n")
    f.write("begin_<sub_structure> \n")

    for face in ply_data["face"]:
        vertex_idx = face[0]
        num_vertex = vertex_idx.size
        f.write("begin_<face> \n")
        f.write("Material 1\n")
        f.write(f"nVertices {num_vertex}\n")
        for v in vertex_idx:
            x = ply_data["vertex"][v][0]
            y = ply_data["vertex"][v][1]
            z = ply_data["vertex"][v][2]
            f.write(f"{x:.10f} ")
            f.write(f"{y:.10f} ")
            f.write(f"{z:.10f}\n")
        f.write("end_<face>\n")

    f.write("end_<sub_structure>\n")
    f.write("end_<structure>\n")
    f.write("end_<structure_group>\n")


def convert_to_city_file(
    ply_root: str,
    city_root: str,
    feature_name: str,
    material_path: str,
) -> str | None:
    """Convert a PLY file to a city feature file.

    Args:
        ply_root (str): Root directory containing PLY files
        city_root (str): Root directory to save city files
        feature_name (str): Name of the feature
        material_path (str): Path to the material file

    Returns:
        Optional[str]: Name of the city file if conversion successful, None otherwise

    """
    ply_path = str(Path(ply_root) / f"{feature_name}.ply")
    save_path = str(Path(city_root) / f"{feature_name}.city")

    if Path(ply_path).exists():
        num_vertex, num_faces = convert_ply2city(ply_path, material_path, save_path)
        print(f"Converted {num_vertex} vertices and {num_faces} faces for {feature_name}")
        return f"{feature_name}.city"
    print(f"Warning: {ply_path} not found. Skipping {feature_name} conversion.")
    return None


if __name__ == "__main__":
    ply_path = "scenario/city_models/scenario_0/gwc_building.ply"
    material_path = "resources/material/ITU Concrete 2.4 GHz.mtl"
    save_path = "scenario/city_models/scenario_0/gwc_building.city"

    (num_vertex, num_faces) = convert_ply2city(ply_path, material_path, save_path)

    print(f"Converted {num_vertex} vertexes and {num_faces} faces")

    ply_path = "scenario/city_models/scenario_0/gwc_road.ply"
    material_path = "resources/material/Asphalt_1GHz.mtl"
    save_path = "scenario/city_models/scenario_0/gwc_road.city"

    (num_vertex, num_faces) = convert_ply2city(ply_path, material_path, save_path)

    print(f"Converted {num_vertex} vertexes and {num_faces} faces")
