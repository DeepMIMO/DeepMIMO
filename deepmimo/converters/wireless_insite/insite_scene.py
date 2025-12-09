"""Parser for Wireless InSite physical object files.

This module provides functionality to parse physical object files (.city, .ter, .veg)
from Wireless InSite into DeepMIMO's physical object representation.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from deepmimo.scene import (
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

# Map file extensions to their corresponding labels
OBJECT_LABELS: dict[str, str] = {
    ".city": CAT_BUILDINGS,
    ".ter": CAT_TERRAIN,
    ".veg": CAT_VEGETATION,
    ".flp": CAT_FLOORPLANS,
    ".obj": CAT_OBJECTS,
}
MIN_TRI_POINTS = 3


def read_scene(folder_path: str | Path) -> Scene:
    """Create a Scene from a folder containing Wireless InSite files.

    This function searches the given folder for .city, .ter, and .veg files
    and creates a Scene containing all the objects defined in those files.

    Args:
        folder_path: Path to folder containing Wireless InSite files

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

    # Parse each type of file and add to scene
    for type_files in found_files.values():
        if not type_files:
            continue

        # Parse all files of this type
        for file in type_files:
            parser = PhysicalObjectParser(file, starting_id=next_object_id)
            objects = parser.parse()
            next_object_id += len(objects)  # Update next available ID
            scene.add_objects(objects)

    return scene


def visualize_road_object(
    name: str,
    vertices: np.ndarray,
    faces: list[list[tuple[float, float, float]]],
) -> None:
    """Visualize a road object and its generated faces."""
    # Save vertices for testing
    save_path = f"road_vertices_{name.replace(' ', '_')}.npy"
    np.save(save_path, vertices)
    print(f"Saved vertices to {save_path}")

    fig = plt.figure(figsize=(15, 5))

    # Plot original vertices
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c="b", marker="o")
    ax1.set_title(f"Original Vertices\n{name}")

    # Plot generated faces
    if faces:
        ax2 = fig.add_subplot(122, projection="3d")
        print(f"\nAnalyzing faces for {name}:")
        for i, face in enumerate(faces):
            face_array = np.array(face)
            print(f"Face {i}:")
            print(f"  Vertices: {len(face)}")
            print(f"  Unique XY points: {len(np.unique(face_array[:, :2], axis=0))}")
            print(f"  Z range: {face_array[:, 2].min():.3f} to {face_array[:, 2].max():.3f}")

            # Plot face as a line connecting vertices
            face_array_closed = np.vstack([face_array, face_array[0]])  # Close the loop
            ax2.plot(
                face_array_closed[:, 0],
                face_array_closed[:, 1],
                face_array_closed[:, 2],
                "-o",
                alpha=0.5,
            )

            # Fill face if it has at least 3 unique points
            unique_points = np.unique(face_array[:, :2], axis=0)
            if len(unique_points) >= MIN_TRI_POINTS:
                try:
                    from matplotlib.tri import Triangulation  # noqa: PLC0415

                    tri = Triangulation(face_array[:, 0], face_array[:, 1])
                    ax2.plot_trisurf(
                        face_array[:, 0],
                        face_array[:, 1],
                        face_array[:, 2],
                        triangles=tri.triangles,
                        alpha=0.2,
                    )
                except (ImportError, ValueError, RuntimeError) as err:
                    print(f"  Warning: Could not triangulate face: {err!s}")
            else:
                print("  Warning: Face has fewer than 3 unique points in XY plane")

        ax2.set_title("Generated Faces")

    plt.tight_layout()
    plt.savefig(f"road_debug_{name.replace(' ', '_')}.png")
    plt.close()


class PhysicalObjectParser:
    """Parser for Wireless InSite physical object files (.city, .ter, .veg)."""

    def __init__(self, file_path: str, starting_id: int = 0) -> None:
        """Initialize parser with file path.

        Args:
            file_path: Path to the physical object file (.city, .ter, .veg)
            starting_id: Starting ID for objects in this file (default: 0)

        """
        self.file_path = Path(file_path)
        if self.file_path.suffix not in OBJECT_LABELS:
            msg = f"Unsupported file type: {self.file_path.suffix}"
            raise ValueError(msg)

        self.label = OBJECT_LABELS[self.file_path.suffix]
        self.name = self.file_path.stem  # Get filename without extension
        self.starting_id = starting_id

    def parse(self, *, force_fast_mode: bool = True) -> list[PhysicalElement]:
        """Parse the file and return a list of physical objects.

        Returns:
            List of PhysicalElement objects with appropriate labels

        """
        # Read file content
        with Path(self.file_path).open() as f:
            content = f.read()

        file_base = Path(self.file_path).name

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

            # Use detailed mode for roads to preserve their geometry
            use_fast_mode = "road" not in self.name.lower()
            self.name = f"{self.name}_{i}"

            # Generate faces
            object_faces = get_object_faces(vertices_array, fast=use_fast_mode or force_fast_mode)
            if object_faces is None:
                print(f"Failed to generate faces for object {self.name}")
                continue

            # Convert faces to Face objects
            faces = [Face(vertices=face) for face in object_faces]

            # Create PhysicalElement object with appropriate label and global ID
            obj = PhysicalElement(
                faces=faces,
                name=self.name,
                object_id=self.starting_id + i,
                label=self.label,
            )
            objects.append(obj)

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
