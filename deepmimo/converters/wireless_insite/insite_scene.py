"""Parser for Wireless InSite physical object files.

This module provides functionality to parse physical object files (.city, .ter, .veg)
from Wireless InSite into DeepMIMO's physical object representation.
"""

import re
from dataclasses import astuple
from pathlib import Path

import numpy as np
from tqdm import tqdm

from deepmimo.core.materials import MaterialList
from deepmimo.core.scene import (
    CAT_BUILDINGS,
    CAT_FLOORPLANS,
    CAT_OBJECTS,
    CAT_TERRAIN,
    CAT_VEGETATION,
    Face,
    PhysicalElement,
    Scene,
    get_object_faces,
)

from .insite_materials import parse_materials_with_indices

# Map file extensions to their corresponding labels
OBJECT_LABELS: dict[str, str] = {
    ".city": CAT_BUILDINGS,
    ".ter": CAT_TERRAIN,
    ".veg": CAT_VEGETATION,
    ".flp": CAT_FLOORPLANS,
    ".obj": CAT_OBJECTS,
    ".object": CAT_OBJECTS,
}

# Blocks and fields used by the geometry-preserving parser.
RE_FACE = re.compile(r"begin_<face>(.*?)end_<face>", re.DOTALL)
RE_FACE_MATERIAL = re.compile(r"^\s*Material\s+(\d+)\s*$", re.MULTILINE)
RE_FACE_VERTEX = re.compile(
    r"^\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$",
    re.MULTILINE,
)


def build_material_maps(files: list[str]) -> dict[str, dict[int, int]]:
    """Map each file's declared material indices onto the scenario material list.

    Wireless InSite numbers materials per file, and those numbers are neither
    contiguous nor ordered by declaration, so the mapping has to be built from
    the indices the files actually declare. The scenario-wide list is assembled
    in the same order as :func:`~deepmimo.converters.wireless_insite.
    insite_materials.read_materials` so the indices agree with ``params.json``.

    Args:
        files: Paths of the geometry files, in the order materials are read.

    Returns:
        Mapping of file path to {index declared in file: scenario-wide index}.

    """
    material_list = MaterialList()
    material_maps: dict[str, dict[int, int]] = {}
    for file in files:
        indexed = parse_materials_with_indices(Path(file))
        material_list.add_materials([material for _, material in indexed])
        global_keys = [astuple(mat)[1:] for mat in material_list._materials]  # noqa: SLF001
        material_maps[file] = {
            declared_idx: global_keys.index(astuple(material)[1:])
            for declared_idx, material in indexed
            if astuple(material)[1:] in global_keys
        }
    return material_maps


def read_scene(folder_path: str | Path, *, lossless: bool = False) -> Scene:
    """Create a Scene from a folder containing Wireless InSite files.

    This function searches the given folder for .city, .ter, and .veg files
    and creates a Scene containing all the objects defined in those files.

    Args:
        folder_path: Path to folder containing Wireless InSite files
        lossless: If True, keep the exact geometry declared in the files instead
            of simplifying each object to its convex hull. Required for indoor
            scenes, where the hull collapses a room and its contents into a box.

    Returns:
        Scene containing all objects from the files

    Raises:
        ValueError: If folder doesn't exist or no valid files found

    """
    folder = Path(folder_path)
    if not folder.exists():
        msg = f"Folder does not exist: {folder}"
        raise ValueError(msg)

    scene = Scene()
    next_object_id = 0  # Track the next available object ID

    # Find all files with matching extensions
    found_files = {ext: [] for ext in OBJECT_LABELS}
    for file in folder.glob("*"):
        suffix = file.suffix.lower()
        if suffix in OBJECT_LABELS:
            found_files[suffix].append(str(file))

    # Check if any valid files were found
    if not any(files for files in found_files.values()):
        msg = f"No valid files (.city, .ter, .veg) found in {folder}"
        raise ValueError(msg)

    ordered_files = [file for type_files in found_files.values() for file in type_files]
    material_maps = build_material_maps(ordered_files) if lossless else {}

    # Parse each type of file and add to scene
    for type_files in found_files.values():
        if not type_files:
            continue

        # Parse all files of this type
        for file in type_files:
            parser = PhysicalObjectParser(
                file,
                starting_id=next_object_id,
                lossless=lossless,
                material_map=material_maps.get(file),
            )
            objects = parser.parse()
            next_object_id += len(objects)  # Update next available ID
            scene.add_objects(objects)

    return scene


class PhysicalObjectParser:
    """Parser for Wireless InSite physical object files (.city, .ter, .veg)."""

    def __init__(
        self,
        file_path: str,
        starting_id: int = 0,
        *,
        lossless: bool = False,
        material_map: dict[int, int] | None = None,
    ) -> None:
        """Initialize parser with file path.

        Args:
            file_path: Path to the physical object file (.city, .ter, .veg)
            starting_id: Starting ID for objects in this file (default: 0)
            lossless: If True, keep each face exactly as declared in the file
                instead of replacing the object with its convex hull.
            material_map: Maps the material indices declared in this file to
                indices in the scenario-wide material list.

        """
        self.lossless = lossless
        self.material_map = material_map or {}
        self.file_path = Path(file_path)
        if self.file_path.suffix not in OBJECT_LABELS:
            msg = f"Unsupported file type: {self.file_path.suffix}"
            raise ValueError(msg)

        self.label = OBJECT_LABELS[self.file_path.suffix]
        self.name = self.file_path.stem  # Get filename without extension
        self.starting_id = starting_id

    def parse(self) -> list[PhysicalElement]:
        """Parse the file and return a list of physical objects.

        Returns:
            List of PhysicalElement objects with appropriate labels

        """
        # Read file content
        with Path(self.file_path).open() as f:
            content = f.read()

        file_base = Path(self.file_path).name

        if self.lossless:
            return self._parse_lossless(content)

        # Extract objects using extract_objects
        object_vertices = extract_objects(content)

        # Convert each set of vertices into a PhysicalElement object
        n_obj = len(object_vertices)
        objects = []
        for i, vertices in tqdm(
            enumerate(object_vertices),
            total=n_obj,
            desc=f"Processing objs in {file_base}",
        ):
            vertices_array = np.array(vertices)

            name = f"{self.name}_{i}"

            # Generate faces using the convex-hull approach
            object_faces = get_object_faces(vertices_array)
            if object_faces is None:
                print(f"Failed to generate faces for object {name}")
                continue

            # Convert faces to Face objects
            faces = [Face(vertices=face) for face in object_faces]

            # Create PhysicalElement object with appropriate label and global ID
            obj = PhysicalElement(
                faces=faces,
                name=name,
                object_id=self.starting_id + i,
                label=self.label,
            )
            objects.append(obj)

        return objects

    def _parse_lossless(self, content: str) -> list[PhysicalElement]:
        """Parse the file keeping every face exactly as declared.

        Args:
            content: Raw file content.

        Returns:
            List of PhysicalElement objects with their original geometry.

        """
        objects = []
        for i, face_tuples in enumerate(extract_object_faces(content)):
            faces = [
                Face(
                    vertices=np.array(vertices),
                    material_idx=self.material_map.get(material_idx, 0),
                )
                for material_idx, vertices in face_tuples
            ]
            objects.append(
                PhysicalElement(
                    faces=faces,
                    name=f"{self.name}_{i}",
                    object_id=self.starting_id + i,
                    label=self.label,
                ),
            )
        return objects


def parse_face_block(block: str) -> tuple[int, list[tuple[float, float, float]]] | None:
    """Parse a single ``begin_<face>`` block.

    Args:
        block: Text between the face block delimiters.

    Returns:
        Tuple of (material index declared on the face, vertex list), or None if
        the block holds fewer than three vertices.

    """
    min_vertices_for_face = 3
    vertices = [(float(x), float(y), float(z)) for x, y, z in RE_FACE_VERTEX.findall(block)]
    if len(vertices) < min_vertices_for_face:
        return None
    material_match = RE_FACE_MATERIAL.search(block)
    material_idx = int(material_match.group(1)) if material_match else 0
    return material_idx, vertices


def extract_object_faces(content: str) -> list[list[tuple[int, list]]]:
    """Group a file's faces into objects, preserving their exact geometry.

    Faces are grouped by shared vertices, the same connectivity rule
    :func:`extract_objects` uses, so an object contains the same faces in either
    representation. Unlike :func:`extract_objects` - which reduces each object to
    a bag of vertices for the convex-hull path - this keeps the declared faces and
    their material indices, which is what indoor scenes need: a hull turns a room
    into a solid block and discards its vertical walls entirely.

    Args:
        content: Raw file content from a Wireless InSite object file.

    Returns:
        List of objects, each a list of (material index, vertices) face tuples.

    """
    faces = []
    vertex_to_faces: dict[tuple[float, float, float], set[int]] = {}
    for block in RE_FACE.findall(content):
        parsed = parse_face_block(block)
        if parsed is None:
            continue
        idx = len(faces)
        faces.append(parsed)
        for vertex in parsed[1]:
            vertex_to_faces.setdefault(vertex, set()).add(idx)

    objects = []
    processed: set[int] = set()
    for start in range(len(faces)):
        if start in processed:
            continue
        group, stack = [], [start]
        while stack:
            current = stack.pop()
            if current in processed:
                continue
            processed.add(current)
            group.append(current)
            for vertex in faces[current][1]:
                stack.extend(vertex_to_faces[vertex] - processed)
        objects.append([faces[i] for i in sorted(group)])
    return objects


def extract_objects(content: str) -> list[list[tuple[float, float, float]]]:
    """Extract physical objects from Wireless InSite file content.

    This function parses the file content to extract and group vertices that form
    complete physical objects (buildings, terrain, etc). It uses face connectivity
    to determine which vertices belong to the same object.

    Args:
        content (str): Raw file content from Wireless InSite object file

    Returns:
        list of list of tuple: List of objects, where each object is a list of
            (x,y,z) vertex coordinate tuples

    """
    # Split content into faces
    face_pattern = r"begin_<face>(.*?)end_<face>"
    faces = re.findall(face_pattern, content, re.DOTALL)

    # Pattern to match coordinates in face definitions
    vertex_pattern = r"-?\d+\.\d+\s+-?\d+\.\d+\s+-?\d+\.\d+"

    # Pre-process all vertices for all faces
    face_vertices = []
    vertex_to_faces = {}  # Map vertices to the faces they belong to

    for i, face in enumerate(faces):
        # Extract and convert vertices once
        vertices = []
        for v in re.findall(vertex_pattern, face):
            x, y, z = map(float, v.split())
            vertex = (x, y, z)
            vertices.append(vertex)
            # Build reverse mapping of vertex -> faces
            if vertex not in vertex_to_faces:
                vertex_to_faces[vertex] = {i}
            else:
                vertex_to_faces[vertex].add(i)
        face_vertices.append(vertices)

    # Group faces that share vertices to form objects
    objects = []
    processed_faces = set()

    for i in range(len(faces)):
        if i in processed_faces:
            continue

        # Start a new object with this face
        object_vertices = set()
        face_stack = [i]

        while face_stack:
            current_face_idx = face_stack.pop()
            if current_face_idx in processed_faces:
                continue

            current_vertices = face_vertices[current_face_idx]
            processed_faces.add(current_face_idx)

            # Add vertices to object
            object_vertices.update(current_vertices)

            # Find connected faces using vertex_to_faces mapping
            connected_faces = set()
            for vertex in current_vertices:
                connected_faces.update(vertex_to_faces[vertex])

            # Add unprocessed connected faces to stack
            face_stack.extend(f for f in connected_faces if f not in processed_faces)

        if object_vertices:
            objects.append(list(object_vertices))

    return objects


if __name__ == "__main__":
    # Test parsing and matrix export
    test_dir = r"./P2Ms/simple_street_canyon_test/"

    # Create scene from test directory
    scene = read_scene(test_dir)

    # Visualize
    scene.plot()
