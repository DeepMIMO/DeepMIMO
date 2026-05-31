"""Physical world representation module.

Provides geometry classes (BoundingBox, Face, PhysicalElement, PhysicalElementGroup, Scene)
and helpers for generating object faces via convex-hull simplification. Scenes can be exported
either in the default convex-hull ("hull") representation or in a lossless triangular-mesh
("mesh") representation that preserves the original geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull

from deepmimo.consts import (
    MAT_FMT,
    PARAMS_FILENAME,
    SCENE_MESH_FACES_FILENAME,
    SCENE_MESH_MATERIALS_FILENAME,
    SCENE_PARAM_N_FACES,
    SCENE_PARAM_N_OBJECTS,
    SCENE_PARAM_N_TRIANGULAR_FACES,
    SCENE_PARAM_N_VERTICES,
    SCENE_PARAM_NAME,
    SCENE_PARAM_NUMBER_SCENES,
    SCENE_PARAM_REPRESENTATION,
    SCENE_REPRESENTATION_HULL,
    SCENE_REPRESENTATION_MESH,
)
from deepmimo.utils import (
    DelegatingList,
    load_dict_from_json,
    load_mat,
    save_dict_as_json,
    save_mat,
)

if TYPE_CHECKING:
    from deepmimo.core.materials import MaterialList

CAT_BUILDINGS: str = "buildings"
CAT_TERRAIN: str = "terrain"
CAT_VEGETATION: str = "vegetation"
CAT_FLOORPLANS: str = "floorplans"
CAT_OBJECTS: str = "objects"
ELEMENT_CATEGORIES = [CAT_BUILDINGS, CAT_TERRAIN, CAT_VEGETATION, CAT_FLOORPLANS, CAT_OBJECTS]


@dataclass
class BoundingBox:
    """Represents a 3D bounding box with min/max coordinates."""

    bounds: np.ndarray

    def __init__(  # noqa: PLR0913
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        z_min: float,
        z_max: float,
    ) -> None:
        """Initialize bounding box with min/max coordinates."""
        self.bounds = np.array([[x_min, y_min, z_min], [x_max, y_max, z_max]])

    @property
    def x_min(self) -> float:
        """Get minimum x coordinate."""
        return self.bounds[0, 0]

    @property
    def x_max(self) -> float:
        """Get maximum x coordinate."""
        return self.bounds[1, 0]

    @property
    def y_min(self) -> float:
        """Get minimum y coordinate."""
        return self.bounds[0, 1]

    @property
    def y_max(self) -> float:
        """Get maximum y coordinate."""
        return self.bounds[1, 1]

    @property
    def z_min(self) -> float:
        """Get minimum z coordinate."""
        return self.bounds[0, 2]

    @property
    def z_max(self) -> float:
        """Get maximum z coordinate."""
        return self.bounds[1, 2]

    @property
    def width(self) -> float:
        """Get the width (X dimension) of the bounding box."""
        return self.x_max - self.x_min

    @property
    def length(self) -> float:
        """Get the length (Y dimension) of the bounding box."""
        return self.y_max - self.y_min

    @property
    def height(self) -> float:
        """Get the height (Z dimension) of the bounding box."""
        return self.z_max - self.z_min

    @property
    def center(self) -> np.ndarray:
        """Get the center of the bounding box."""
        return np.array(
            [
                (self.x_max + self.x_min) / 2,
                (self.y_max + self.y_min) / 2,
                (self.z_max + self.z_min) / 2,
            ],
        )


class Face:
    """Represents a single face (surface) of a physical object.

    This class implements a dual representation for faces:
    1. Primary representation: Convex hull faces (stored in vertices)
    - More efficient for storage
    - Better for most geometric operations
    - Suitable for ray tracing and wireless simulations

    2. Secondary representation: Triangular faces (generated on demand)
    - Available through triangular_faces property
    - Better for detailed visualization
    - Preserves exact geometry when needed
    - Generated using fan triangulation

    This dual representation allows the system to be efficient while maintaining
    the ability to represent detailed geometry when required.
    """

    def __init__(
        self,
        vertices: list[tuple[float, float, float]] | np.ndarray,
        material_idx: int | np.integer = 0,
    ) -> None:
        """Initialize a face from its vertices.

        Args:
            vertices: List of (x, y, z) coordinates or numpy array of shape (N, 3)
                defining the face vertices in counter-clockwise order
            material_idx: Index of the material for this face (default: 0)

        """
        self.vertices = np.asarray(vertices, dtype=np.float32)
        self.material_idx = int(material_idx)
        self._normal: np.ndarray | None = None
        self._area: float | None = None
        self._centroid: np.ndarray | None = None
        self._triangular_faces: list[np.ndarray] | None = None

    @property
    def normal(self) -> np.ndarray:
        """Get the normal vector of the face."""
        if self._normal is None:
            v1 = self.vertices[1] - self.vertices[0]
            v2 = self.vertices[2] - self.vertices[0]
            normal = np.cross(v1, v2)
            self._normal = normal / np.linalg.norm(normal)
        return self._normal

    @property
    def triangular_faces(self) -> list[np.ndarray]:
        """Get the triangular faces that make up this face."""
        if self._triangular_faces is None:
            tri_vertex_count = 3
            if len(self.vertices) == tri_vertex_count:
                self._triangular_faces = [self.vertices]
            else:
                triangles = []
                for i in range(1, len(self.vertices) - 1):
                    triangle = np.array([self.vertices[0], self.vertices[i], self.vertices[i + 1]])
                    triangles.append(triangle)
                self._triangular_faces = triangles
        return self._triangular_faces

    @property
    def num_triangular_faces(self) -> int:
        """Get the number of triangular faces."""
        return len(self.triangular_faces)

    @property
    def area(self) -> float:
        """Get the area of the face."""
        if self._area is None:
            n = self.normal
            proj_axis = np.argmax(np.abs(n))
            other_axes = [i for i in range(3) if i != proj_axis]
            points = self.vertices[:, other_axes]
            x = points[:, 0]
            y = points[:, 1]
            x_next = np.roll(x, -1)
            y_next = np.roll(y, -1)
            self._area = 0.5 * np.abs(np.sum(x * y_next - x_next * y))
        return self._area

    @property
    def centroid(self) -> np.ndarray:
        """Get the centroid of the face."""
        if self._centroid is None:
            self._centroid = np.mean(self.vertices, axis=0)
        return self._centroid


class PhysicalElement:
    """Base class for physical objects in the wireless environment."""

    DEFAULT_LABELS: ClassVar[set[str]] = {
        CAT_BUILDINGS,
        CAT_TERRAIN,
        CAT_VEGETATION,
        CAT_FLOORPLANS,
        CAT_OBJECTS,
    }

    def __init__(
        self,
        faces: list[Face],
        object_id: int = -1,
        label: str = CAT_OBJECTS,
        color: str = "",
        name: str = "",
    ) -> None:
        """Initialize a physical object from its faces.

        Args:
            faces: List of Face objects defining the object
            object_id: Unique identifier for the object (default: -1)
            label: Label identifying the type of object (default: 'objects')
            color: Color for visualization (default: '', which means use default color)
            name: Optional name for the object (default: '')

        """
        self._faces = faces
        self.object_id = object_id
        self.label = label if label in self.DEFAULT_LABELS else CAT_OBJECTS
        self.color = color
        self.name = name
        self._vel: np.ndarray = np.zeros(3)
        all_vertices = np.vstack([face.vertices for face in faces])
        self.vertices = all_vertices
        self.bounding_box: BoundingBox
        self._footprint_area: float | None = None
        self._position: np.ndarray | None = None
        self._hull: ConvexHull | None = None
        self._hull_volume: float | None = None
        self._hull_surface_area: float | None = None
        self._materials: set[int] | None = None
        self._compute_bounding_box()

    def _compute_bounding_box(self) -> None:
        """Compute the object's bounding box."""
        mins = np.min(self.vertices, axis=0)
        maxs = np.max(self.vertices, axis=0)
        self.bounding_box = BoundingBox(
            x_min=mins[0],
            x_max=maxs[0],
            y_min=mins[1],
            y_max=maxs[1],
            z_min=mins[2],
            z_max=maxs[2],
        )

    @property
    def height(self) -> float:
        """Get the height of the object."""
        return self.bounding_box.height

    @property
    def faces(self) -> list[Face]:
        """Get the faces of the object."""
        return self._faces

    @property
    def hull(self) -> ConvexHull:
        """Get the convex hull of the object."""
        if self._hull is None:
            self._hull = ConvexHull(self.vertices)
        return self._hull

    @property
    def hull_volume(self) -> float:
        """Get the volume of the object using its convex hull."""
        if self._hull_volume is None:
            self._hull_volume = self.hull.volume
        return self._hull_volume

    @property
    def hull_surface_area(self) -> float:
        """Get the surface area of the object using its convex hull."""
        if self._hull_surface_area is None:
            self._hull_surface_area = self.hull.area
        return self._hull_surface_area

    @property
    def footprint_area(self) -> float:
        """Get the area of the object's footprint using 2D convex hull."""
        if self._footprint_area is None:
            points_2d = self.vertices[:, :2]
            self._footprint_area = ConvexHull(points_2d).area
        return self._footprint_area

    @property
    def volume(self) -> float:
        """Get the volume of the object using its convex hull."""
        return self.hull_volume

    def to_dict(self, vertex_map: dict[tuple[float, ...], int]) -> dict:
        """Convert physical object to dictionary format.

        Args:
            vertex_map: Dictionary mapping vertex tuples to their global indices

        Returns:
            Dict containing object metadata with face vertex and material indices

        """
        obj_metadata = {
            "name": self.name,
            "label": self.label,
            "id": self.object_id,
            "face_vertex_idxs": [],
            "face_material_idxs": [],
        }
        for face in self.faces:
            face_vertex_indices = []
            for tri_vertices in face.triangular_faces:
                for vertex in tri_vertices:
                    vertex_tuple = tuple(vertex)
                    if vertex_tuple not in vertex_map:
                        vertex_map[vertex_tuple] = len(vertex_map)
                    if vertex_map[vertex_tuple] not in face_vertex_indices:
                        face_vertex_indices.append(vertex_map[vertex_tuple])
            obj_metadata["face_vertex_idxs"].append(face_vertex_indices)
            obj_metadata["face_material_idxs"].append(face.material_idx)
        return obj_metadata

    @classmethod
    def from_dict(cls: PhysicalElement, data: dict, vertices: np.ndarray) -> PhysicalElement:
        """Create physical object from dictionary format.

        Args:
            data: Dictionary containing object data
            vertices: Array of vertex coordinates (shape: N_vertices x 3)

        Returns:
            PhysicalElement: Created object

        """
        faces = [
            Face(vertices=vertices[vertex_idxs], material_idx=material_idx)
            for (vertex_idxs, material_idx) in zip(
                data["face_vertex_idxs"],
                data["face_material_idxs"],
                strict=False,
            )
        ]
        return cls(faces=faces, name=data["name"], object_id=data["id"], label=data["label"])

    def to_mesh_arrays(
        self,
        vertex_map: dict[tuple[float, ...], int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Serialize the object as a lossless triangular mesh.

        Unlike :meth:`to_dict` (which stores simplified convex-hull faces), this
        preserves the object's exact geometry by emitting one entry per triangle.
        Vertices are deduplicated through the shared ``vertex_map``, mirroring how
        :meth:`Scene.export_data` builds its global vertex pool.

        Args:
            vertex_map: Dictionary mapping vertex tuples to their global indices.
                Updated in place as new vertices are encountered.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - tri_vertex_idxs: int array of shape (N_triangles, 3) with the
                  global vertex indices of each triangle.
                - tri_material_idxs: int array of shape (N_triangles,) with the
                  material index of each triangle.

        """
        tri_vertex_idxs: list[list[int]] = []
        tri_material_idxs: list[int] = []
        for face in self.faces:
            for triangle in face.triangular_faces:
                triangle_indices = []
                for vertex in triangle:
                    vertex_tuple = tuple(vertex)
                    if vertex_tuple not in vertex_map:
                        vertex_map[vertex_tuple] = len(vertex_map)
                    triangle_indices.append(vertex_map[vertex_tuple])
                tri_vertex_idxs.append(triangle_indices)
                tri_material_idxs.append(face.material_idx)
        return (
            np.array(tri_vertex_idxs, dtype=np.int32).reshape(-1, 3),
            np.array(tri_material_idxs, dtype=np.int32),
        )

    @classmethod
    def from_mesh_arrays(
        cls: PhysicalElement,
        data: dict,
        tri_vertex_idxs: np.ndarray,
        tri_material_idxs: np.ndarray,
        vertices: np.ndarray,
    ) -> PhysicalElement:
        """Create a physical object from a lossless triangular mesh.

        Each triangle becomes its own :class:`Face`, preserving the exact geometry
        that was exported with ``Scene.export_data(..., lossless=True)``.

        Args:
            data: Dictionary containing object metadata (name, label, id).
            tri_vertex_idxs: int array (N_triangles, 3) of global vertex indices.
            tri_material_idxs: int array (N_triangles,) of material indices.
            vertices: Array of vertex coordinates (shape: N_vertices x 3).

        Returns:
            PhysicalElement: Created object with one Face per triangle.

        """
        faces = [
            Face(vertices=vertices[triangle_idxs], material_idx=int(material_idx))
            for (triangle_idxs, material_idx) in zip(
                tri_vertex_idxs,
                tri_material_idxs,
                strict=False,
            )
        ]
        return cls(faces=faces, name=data["name"], object_id=data["id"], label=data["label"])

    @property
    def position(self) -> np.ndarray:
        """Get the center of mass (position) of the object."""
        if self._position is None:
            bb = self.bounding_box
            self._position = np.array(
                [
                    (bb.x_max + bb.x_min) / 2,
                    (bb.y_max + bb.y_min) / 2,
                    (bb.z_max + bb.z_min) / 2,
                ],
            )
        return self._position

    def plot(
        self,
        ax: plt.Axes | None = None,
        mode: Literal["faces", "tri_faces"] = "faces",
        alpha: float = 0.8,
        color: str | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot the object using the specified visualization mode.

        Args:
            ax: Matplotlib 3D axes to plot on (if None, creates new figure)
            mode: Visualization mode - either 'faces' or 'tri_faces' (default: 'faces')
            alpha: Transparency for visualization (default: 0.8)
            color: Color for visualization (default: None, uses object's color)

        """
        ax = ax or plt.subplots(1, 1, subplot_kw={"projection": "3d"})[1]
        if mode == "faces":
            vertices_list = [face.vertices for face in self.faces]
        elif mode == "tri_faces":
            vertices_list = [tri for face in self.faces for tri in face.triangular_faces]
        for vertices in vertices_list:
            poly3d = Poly3DCollection([vertices], alpha=alpha)
            plot_color = self.color or color
            poly3d.set_facecolor(plot_color)
            poly3d.set_edgecolor("black")
            ax.add_collection3d(poly3d)
        return (ax.get_figure(), ax)

    @property
    def materials(self) -> set[int]:
        """Get set of material indices used by this object."""
        if self._materials is None:
            self._materials = list({face.material_idx for face in self._faces})
        return self._materials

    @property
    def vel(self) -> np.ndarray:
        """Get the speed vector of the object in Cartesian coordinates [m/s]."""
        return self._vel

    @vel.setter
    def vel(self, value: np.ndarray | list | tuple) -> None:
        """Set the velocity vector of the object.

        Args:
            value: Either a float (magnitude only) or a 3D vector [m/s]

        """
        if isinstance(value, (list, tuple)):
            value = np.array(value)
        if value.shape != (3,):
            msg = "Velocity must be a 3D vector (x, y, z) in meters per second"
            raise ValueError(msg)
        self._vel = value

    def __repr__(self) -> str:
        """Return a concise string representation of the physical element.

        Returns:
            str: String representation showing key element information

        """
        bb = self.bounding_box
        dims = f"{bb.width:.0f} x {bb.length:.0f} x {bb.height:.0f} m"
        return (
            "PhysicalElement("
            f"name='{self.name}', id={self.object_id}, label='{self.label}', "
            f"faces={len(self._faces)}, dims={dims})"
        )


class PhysicalElementGroup:
    """Represents a group of physical objects that can be queried and manipulated together."""

    def __init__(self, objects: list[PhysicalElement]) -> None:
        """Initialize a group of physical objects."""
        self._objects = objects
        self._bounding_box: BoundingBox | None = None

    def __len__(self) -> int:
        """Get number of objects in group."""
        return len(self._objects)

    def __iter__(self) -> Any:
        """Iterate over objects in group."""
        return iter(self._objects)

    def __getitem__(self, idx: int) -> PhysicalElement:
        """Get object by index."""
        return self._objects[idx]

    def __repr__(self) -> str:
        """Return a concise string representation of the physical element group."""
        obj_list = "\n".join(f"  {obj}" for obj in self._objects)
        return f"PhysicalElementGroup(objects={len(self._objects)})\nObjects:\n{obj_list}"

    def get_materials(self) -> list[int]:
        """Get list of material indices used by objects in this group."""
        return list(set().union(*(obj.materials for obj in self._objects)))

    def get_objects(
        self,
        label: str | None = None,
        material: int | None = None,
    ) -> PhysicalElementGroup:
        """Get objects filtered by label and/or material.

        Args:
            label: Optional label to filter objects by
            material: Optional material index to filter objects by

        Returns:
            PhysicalElementGroup containing filtered objects

        """
        objects = self._objects
        if label:
            objects = [obj for obj in objects if obj.label == label]
        if material:
            objects = [obj for obj in objects if material in obj.materials]
        return PhysicalElementGroup(objects)

    @property
    def bounding_box(self) -> BoundingBox:
        """Get the bounding box containing all objects."""
        if self._bounding_box is None:
            if not self._objects:
                msg = "Group is empty"
                raise ValueError(msg)
            boxes = [obj.bounding_box.bounds for obj in self._objects]
            boxes = np.array(boxes)
            global_min = np.min(boxes[:, 0], axis=0)
            global_max = np.max(boxes[:, 1], axis=0)
            self._bounding_box = BoundingBox(
                x_min=global_min[0],
                x_max=global_max[0],
                y_min=global_min[1],
                y_max=global_max[1],
                z_min=global_min[2],
                z_max=global_max[2],
            )
        return self._bounding_box


class Scene:
    """Represents a physical scene with various objects affecting wireless propagation."""

    DEFAULT_VISUALIZATION_SETTINGS: ClassVar[dict[str, dict[str, Any]]] = {
        CAT_TERRAIN: {"z_order": 1, "alpha": 0.1, "color": "black"},
        CAT_VEGETATION: {"z_order": 2, "alpha": 0.8, "color": "green"},
        CAT_BUILDINGS: {"z_order": 3, "alpha": 0.6, "color": None},
        CAT_FLOORPLANS: {"z_order": 4, "alpha": 0.8, "color": "blue"},
        CAT_OBJECTS: {"z_order": 5, "alpha": 0.8, "color": "red"},
    }

    def __init__(self) -> None:
        """Initialize an empty scene."""
        self.objects = DelegatingList()
        self.visualization_settings = self.DEFAULT_VISUALIZATION_SETTINGS.copy()
        self.face_indices = []
        self._current_index = 0
        # Geometry representation of the loaded scene. Defaults to the convex-hull
        # ("hull") simplification; set to "mesh" only when reconstructed from the
        # lossless triangular-mesh files (see ``_from_data_mesh``).
        self.representation: str = SCENE_REPRESENTATION_HULL
        self._objects_by_category: dict[str, list[PhysicalElement]] = {
            cat: [] for cat in ELEMENT_CATEGORIES
        }
        self._objects_by_material: dict[int, list[PhysicalElement]] = {}
        self._materials: MaterialList | None = None

    @property
    def bounding_box(self) -> BoundingBox:
        """Get the bounding box containing all objects."""
        return self.get_objects().bounding_box

    def set_visualization_settings(self, label: str, settings: dict) -> None:
        """Set visualization settings for a specific label."""
        self.visualization_settings[label] = settings

    def add_object(self, obj: PhysicalElement) -> None:
        """Add a physical object to the scene.

        Args:
            obj: PhysicalElement to add

        """
        if obj.object_id == -1:
            obj.object_id = len(self.objects)
        obj_indices = []
        for face in obj.faces:
            face_indices = self._add_face(face)
            obj_indices.append(face_indices)
        for material_idx in obj.materials:
            if material_idx not in self._objects_by_material:
                self._objects_by_material[material_idx] = []
            self._objects_by_material[material_idx].append(obj)
        category = obj.label if obj.label in ELEMENT_CATEGORIES else CAT_OBJECTS
        if category not in self._objects_by_category:
            self._objects_by_category[category] = []
        self._objects_by_category[category].append(obj)
        self.face_indices.append(obj_indices)
        self.objects.append(obj)
        self._bounding_box = None

    def add_objects(self, objects: list[PhysicalElement]) -> None:
        """Add multiple physical objects to the scene.

        Args:
            objects: List of PhysicalElement objects to add

        """
        for obj in objects:
            self.add_object(obj)

    def _add_face(self, face: Face) -> list[int]:
        """Add a face and return indices of its triangular faces.

        Args:
            face: Face to add

        Returns:
            List of indices for the face's triangular faces

        """
        n_triangles = face.num_triangular_faces
        triangle_indices = list(range(self._current_index, self._current_index + n_triangles))
        self._current_index += n_triangles
        return triangle_indices

    def get_objects(
        self,
        label: str | None = None,
        material: int | None = None,
    ) -> PhysicalElementGroup:
        """Get objects filtered by label and/or material.

        Args:
            label: Optional label to filter objects by
            material: Optional material index to filter objects by

        Returns:
            PhysicalElementGroup containing filtered objects

        """
        if label:
            objects = self._objects_by_category.get(label, [])
        elif material:
            objects = self._objects_by_material.get(material, [])
        else:
            objects = self.objects
        group = PhysicalElementGroup(objects)
        return group.get_objects(material=material) if material else group

    def export_data(self, base_folder: str, *, lossless: bool = False) -> dict:
        """Export scene data to files and return metadata dictionary.

        Two on-disk representations are supported:

        - Hull (default, ``lossless=False``): the legacy convex-hull representation.
          Writes ``vertices.npz`` (the global deduplicated vertex pool) and
          ``objects.json`` (per-object convex-hull faces). Compact and unchanged from
          previous versions.
        - Mesh (``lossless=True``): a lossless triangular mesh that preserves the
          exact geometry. Writes the same ``vertices.npz`` pool plus ``faces.npz``
          (one ``(N_tri, 3)`` vertex-index array per object) and ``materials.npz``
          (one ``(N_tri,)`` material-index array per object). Object metadata is
          stored in ``objects.json``.

        Args:
            base_folder: Base folder to store the scene files.
            lossless: If True, export the lossless triangular mesh instead of the
                convex-hull simplification. Defaults to False.

        Returns:
            Dict containing metadata needed to reload the scene (including the
            ``representation`` flag).

        """
        Path(base_folder).mkdir(parents=True, exist_ok=True)
        if lossless:
            return self._export_data_mesh(base_folder)

        vertex_map = {}
        objects_metadata = []
        for obj in self.objects:
            obj_metadata = obj.to_dict(vertex_map)
            objects_metadata.append(obj_metadata)
        all_vertices = [None] * len(vertex_map)
        for vertex, idx in vertex_map.items():
            all_vertices[idx] = vertex
        vertices = np.array(all_vertices)
        save_mat(vertices, "vertices", f"{base_folder}/vertices.mat")
        save_dict_as_json(f"{base_folder}/objects.json", objects_metadata)
        return {
            SCENE_PARAM_NUMBER_SCENES: 1,
            SCENE_PARAM_N_OBJECTS: len(self.objects),
            SCENE_PARAM_N_VERTICES: len(vertices),
            SCENE_PARAM_N_FACES: sum(len(obj.faces) for obj in self.objects),
            SCENE_PARAM_N_TRIANGULAR_FACES: sum(
                len(obj_face_idxs) for obj_face_idxs in self.face_indices
            ),
            SCENE_PARAM_REPRESENTATION: SCENE_REPRESENTATION_HULL,
        }

    def _export_data_mesh(self, base_folder: str) -> dict:
        """Export the scene as a lossless triangular mesh.

        See :meth:`export_data` for the on-disk layout and rationale.

        Args:
            base_folder: Base folder to store the scene files.

        Returns:
            Dict containing metadata needed to reload the scene.

        """
        vertex_map: dict[tuple[float, ...], int] = {}
        objects_metadata = []
        faces_arrays: dict[str, np.ndarray] = {}
        materials_arrays: dict[str, np.ndarray] = {}
        n_triangular_faces = 0
        for obj_idx, obj in enumerate(self.objects):
            mesh_key = str(obj_idx)
            tri_vertex_idxs, tri_material_idxs = obj.to_mesh_arrays(vertex_map)
            faces_arrays[mesh_key] = tri_vertex_idxs
            materials_arrays[mesh_key] = tri_material_idxs
            n_triangular_faces += len(tri_vertex_idxs)
            objects_metadata.append(
                {
                    "name": obj.name,
                    "label": obj.label,
                    "id": obj.object_id,
                    "mesh_key": mesh_key,
                },
            )
        all_vertices = [None] * len(vertex_map)
        for vertex, idx in vertex_map.items():
            all_vertices[idx] = vertex
        vertices = np.array(all_vertices) if all_vertices else np.zeros((0, 3), dtype=np.float32)
        save_mat(vertices, "vertices", f"{base_folder}/vertices.mat")
        np.savez_compressed(f"{base_folder}/{SCENE_MESH_FACES_FILENAME}", **faces_arrays)
        np.savez_compressed(f"{base_folder}/{SCENE_MESH_MATERIALS_FILENAME}", **materials_arrays)
        save_dict_as_json(f"{base_folder}/objects.json", objects_metadata)
        return {
            SCENE_PARAM_NUMBER_SCENES: 1,
            SCENE_PARAM_N_OBJECTS: len(self.objects),
            SCENE_PARAM_N_VERTICES: len(vertices),
            SCENE_PARAM_N_FACES: sum(len(obj.faces) for obj in self.objects),
            SCENE_PARAM_N_TRIANGULAR_FACES: n_triangular_faces,
            SCENE_PARAM_REPRESENTATION: SCENE_REPRESENTATION_MESH,
        }

    @classmethod
    def from_data(cls: Any, base_folder: str) -> Scene:
        """Create scene from data files.

        Detects the scene representation (convex-hull ``"hull"`` or lossless
        ``"mesh"``) and reconstructs the objects accordingly. Scenarios without an
        explicit representation flag (legacy datasets) are loaded as ``"hull"``.

        Args:
            base_folder: Base folder containing the scene files.

        """
        if cls._is_mesh_representation(base_folder):
            return cls._from_data_mesh(base_folder)

        scene = cls()
        try:
            vertices = load_mat(f"{base_folder}/vertices.{MAT_FMT}", "vertices")
            objects_metadata = load_dict_from_json(f"{base_folder}/objects.json")
        except FileNotFoundError:
            print(
                "FileNotFoundError: "
                f"{base_folder}/vertices.mat or {base_folder}/objects.json not found",
            )
            vertices = np.array([])
            objects_metadata = []
        except Exception as e:
            msg = f"Error loading scene from {base_folder}: {e}"
            raise RuntimeError(msg) from e
        for object_data in objects_metadata:
            obj = PhysicalElement.from_dict(object_data, vertices)
            scene.add_object(obj)
        return scene

    @staticmethod
    def _is_mesh_representation(base_folder: str) -> bool:
        """Detect whether a scenario folder uses the lossless mesh representation.

        Detection is primarily based on the presence of the mesh files
        (``faces.npz`` and ``materials.npz``). As a fallback, a local ``params.json``
        scene block flag is consulted. Missing signals are treated as the legacy
        hull representation, keeping backward compatibility with old scenarios.

        Args:
            base_folder: Base folder containing the scene files.

        Returns:
            bool: True if the folder uses the mesh representation.

        """
        base_path = Path(base_folder)
        faces_path = base_path / SCENE_MESH_FACES_FILENAME
        materials_path = base_path / SCENE_MESH_MATERIALS_FILENAME
        if faces_path.exists() and materials_path.exists():
            return True
        params_path = base_path / f"{PARAMS_FILENAME}.json"
        if params_path.exists():
            try:
                params = load_dict_from_json(str(params_path))
                scene_block = params.get(SCENE_PARAM_NAME, {})
                return scene_block.get(SCENE_PARAM_REPRESENTATION) == SCENE_REPRESENTATION_MESH
            except Exception:  # noqa: BLE001 - never let detection break loading
                return False
        return False

    @classmethod
    def _from_data_mesh(cls: Any, base_folder: str) -> Scene:
        """Reconstruct a scene from the lossless triangular-mesh files.

        Args:
            base_folder: Base folder containing the scene files.

        """
        scene = cls()
        scene.representation = SCENE_REPRESENTATION_MESH
        faces_path = f"{base_folder}/{SCENE_MESH_FACES_FILENAME}"
        materials_path = f"{base_folder}/{SCENE_MESH_MATERIALS_FILENAME}"
        try:
            vertices = load_mat(f"{base_folder}/vertices.{MAT_FMT}", "vertices")
            objects_metadata = load_dict_from_json(f"{base_folder}/objects.json")
            with np.load(faces_path) as faces_npz:
                faces_arrays = {key: faces_npz[key] for key in faces_npz.files}
            with np.load(materials_path) as materials_npz:
                materials_arrays = {key: materials_npz[key] for key in materials_npz.files}
        except FileNotFoundError:
            print(f"FileNotFoundError: mesh scene files not found in {base_folder}")
            return scene
        except Exception as e:
            msg = f"Error loading mesh scene from {base_folder}: {e}"
            raise RuntimeError(msg) from e
        for object_data in objects_metadata:
            mesh_key = object_data["mesh_key"]
            tri_vertex_idxs = faces_arrays[mesh_key]
            tri_material_idxs = materials_arrays[mesh_key]
            if len(tri_vertex_idxs) == 0:
                continue
            obj = PhysicalElement.from_mesh_arrays(
                object_data,
                tri_vertex_idxs,
                tri_material_idxs,
                vertices,
            )
            scene.add_object(obj)
        return scene

    def _plot_objects_3d(self, ax: plt.Axes, label_groups: dict, mode: str) -> None:
        """Plot objects in 3D mode.

        Args:
            ax: Matplotlib 3D axes to plot on
            label_groups: Dictionary mapping labels to lists of objects
            mode: Visualization mode ('faces' or 'tri_faces')

        """
        default_vis_settings = {"z_order": 3, "alpha": 0.8, "color": None}
        for label, objects in label_groups.items():
            if label == CAT_TERRAIN:
                continue  # terrain (ground planes) causes depth-sort artifacts in 3D
            vis_settings = self.visualization_settings.get(label, default_vis_settings)
            n_objects = len(objects)
            colors = (
                plt.cm.rainbow(np.linspace(0, 1, n_objects))
                if vis_settings["color"] is None
                else [vis_settings["color"]] * n_objects
            )
            for obj_idx, obj in enumerate(objects):
                color = obj.color or colors[obj_idx]
                obj.plot(ax, mode=mode, alpha=vis_settings["alpha"], color=color)

    def _plot_objects_2d(self, ax: plt.Axes, label_groups: dict) -> None:
        """Plot objects in 2D mode (top-down view).

        Args:
            ax: Matplotlib 2D axes to plot on
            label_groups: Dictionary mapping labels to lists of objects

        """
        default_vis_settings = {"z_order": 3, "alpha": 0.8, "color": None}
        for label, objects in label_groups.items():
            vis_settings = self.visualization_settings.get(label, default_vis_settings)
            n_objects = len(objects)
            colors = (
                plt.cm.rainbow(np.linspace(0, 1, n_objects))
                if vis_settings["color"] is None
                else [vis_settings["color"]] * n_objects
            )
            for obj_idx, obj in enumerate(objects):
                color = obj.color or colors[obj_idx]
                vertices_2d = obj.vertices[:, :2]
                hull = ConvexHull(vertices_2d)
                hull_vertices = vertices_2d[hull.vertices]
                ax.fill(
                    hull_vertices[:, 0],
                    hull_vertices[:, 1],
                    alpha=vis_settings["alpha"],
                    color=color,
                    label=label if obj_idx == 0 else "",
                )

    def _configure_plot_axes(
        self,
        ax: plt.Axes,
        *,
        proj_3d: bool,
        title: bool,
        label_groups: dict,
        legend: bool,
    ) -> None:
        """Configure axes labels, limits, and legend.

        Args:
            ax: Matplotlib axes to configure
            proj_3d: Whether this is a 3D plot
            title: Whether to show title
            label_groups: Dictionary of label groups (for legend check)
            legend: Whether to show legend

        """
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        if proj_3d:
            ax.set_zlabel("Z (m)")
        if title:
            ax.set_title(self._get_title_with_counts())
        if proj_3d:
            ax.view_init(elev=40, azim=-45)
            self._set_axes_lims_to_scale(ax)
        else:
            ax.set_aspect("equal")
            ax.grid(visible=True, alpha=0.3)
        if len(label_groups) > 1 and legend:
            ax.legend()

    def plot(  # noqa: PLR0913 - scientific visualization API needs customization options
        self,
        *,
        title: bool = True,
        mode: Literal["faces", "tri_faces"] = "faces",
        ax: plt.Axes | None = None,
        proj_3d: bool = True,
        figsize: tuple = (10, 10),
        dpi: int = 100,
        legend: bool = False,
        **kwargs: Any,
    ) -> plt.Axes:
        """Create a visualization of the scene.

        The scene can be visualized in either 2D (top-down view) or 3D mode:

        3D Mode (proj_3d=True):
            Two representation options:
            1. 'faces' (default) - Uses the primary convex hull representation
            - More efficient for visualization
            - Cleaner look for simple geometric shapes
            - Suitable for most visualization needs

            2. 'tri_faces' - Uses the secondary triangular representation
            - Shows detailed geometry
            - Better for debugging geometric issues
            - More accurate representation of complex shapes

        2D Mode (proj_3d=False):
            Creates a top-down view showing object footprints:
            - Projects all objects onto x-y plane
            - Uses convex hulls for efficient visualization
            - Better for understanding spatial layout
            - More efficient for large scenes

        Args:
            title: Whether to display the title (default: True)
            mode: Visualization mode for 3D - either 'faces' or 'tri_faces' (default: 'faces')
            ax: Matplotlib axes to plot on (if None, creates new figure)
            proj_3d: Whether to create 3D projection (default: True)
            **kwargs: Additional keyword-only options; accepts `proj_3D` alias.
            figsize: Figure dimensions (width, height) in inches (default: (10, 10))
            dpi: Plot resolution in dots per inch (default: 100)
            legend: Whether to show legend for objects/materials (default: False)

        Returns:
            matplotlib Axes object

        """
        if "proj_3D" in kwargs:
            proj_3d = kwargs.pop("proj_3D")

        if len(self.objects) == 0:
            print("No objects in scene - skipping plot")
            return ax

        # Create axes if not provided
        if ax is None:
            (_, ax) = plt.subplots(
                figsize=figsize,
                dpi=dpi,
                subplot_kw={"projection": "3d" if proj_3d else None},
            )

        # Group objects by label
        label_groups = {}
        for obj in self.objects:
            if obj.label not in label_groups:
                label_groups[obj.label] = []
            label_groups[obj.label].append(obj)

        # Plot objects based on mode
        if proj_3d:
            self._plot_objects_3d(ax, label_groups, mode)
        else:
            self._plot_objects_2d(ax, label_groups)

        # Configure axes, labels, and legend
        self._configure_plot_axes(
            ax, proj_3d=proj_3d, title=title, label_groups=label_groups, legend=legend
        )

        return ax

    def _set_axes_lims_to_scale(self, ax: plt.Axes, zoom: float = 1.3) -> None:
        """Set axis limits based on scene bounding box with equal scaling.

        Args:
            ax: Matplotlib 3D axes to set limits on
            zoom: Zoom factor (>1 zooms out, <1 zooms in)

        """
        bb = self.bounding_box
        center_x = (bb.x_max + bb.x_min) / 2
        center_y = (bb.y_max + bb.y_min) / 2
        center_z = (bb.z_max + bb.z_min) / 2
        max_range = max(bb.width, bb.length, bb.height) / 2 / zoom
        ax.set_xlim3d([center_x - max_range, center_x + max_range])
        ax.set_ylim3d([center_y - max_range, center_y + max_range])
        ax.set_zlim3d([center_z - max_range, center_z + max_range])
        ax.set_box_aspect([1, 1, 1])

    def _get_title_with_counts(self) -> str:
        """Generate a title string with object counts for each label.

        Returns:
            Title string with object counts

        """
        label_counts = {}
        for obj in self.objects:
            label_counts[obj.label] = label_counts.get(obj.label, 0) + 1
        counts = []
        for label, count in label_counts.items():
            label_name = label.capitalize()
            if count == 1 and label_name.endswith("s"):
                label_name = label_name[:-1]
            counts.append(f"{label_name}: {count}")
        return ", ".join(counts)

    def count_objects_by_label(self) -> dict[str, int]:
        """Count the number of objects for each label in the scene.

        Returns:
            dict[str, int]: Dictionary mapping labels to their counts

        """
        label_counts = {}
        for obj in self.objects:
            label = obj.label
            label_counts[label] = label_counts.get(label, 0) + 1
        return label_counts

    def __repr__(self) -> str:
        """Return a concise string representation of the scene.

        Returns:
            str: String representation showing key scene information

        """
        label_counts = self.count_objects_by_label()
        bb = self.bounding_box
        dims = f"{bb.width:.1f} x {bb.length:.1f} x {bb.height:.1f} m"
        counts = [f"{label}: {count}" for (label, count) in label_counts.items()]
        counts_str = ", ".join(counts)
        return f"Scene({len(self.objects)} objects [{counts_str}], dims = {dims})"


def _get_faces_convex_hull(vertices: np.ndarray) -> list[list[tuple[float, float, float]]]:
    """Generate faces using convex hull approach (fast but simplified).

    Args:
        vertices: Array of vertex coordinates (shape: N x 3)

    Returns:
        List of faces, where each face is a list of (x,y,z) vertex coordinates

    """
    points_2d = vertices[:, :2]
    heights = vertices[:, 2]
    object_height = np.max(heights) - np.min(heights)
    base_height = np.min(heights)
    try:
        hull = ConvexHull(points_2d)
        base_shape = points_2d[hull.vertices]
    except Exception:
        rank_threshold = 2
        if np.linalg.matrix_rank(points_2d - points_2d[0]) < rank_threshold:
            print("Convex hull failed - collinear vertices")
            return None
        raise
    bottom_face = [(x, y, base_height) for (x, y) in base_shape]
    top_face = [(x, y, base_height + object_height) for (x, y) in base_shape]
    side_faces = []
    for i in range(len(base_shape)):
        j = (i + 1) % len(base_shape)
        side = [bottom_face[i], bottom_face[j], top_face[j], top_face[i]]
        side_faces.append(side)
    return [bottom_face, top_face, *side_faces]


def get_object_faces(
    vertices: list[tuple[float, float, float]],
) -> list[list[tuple[float, float, float]]]:
    """Generate faces for a physical object from its vertices.

    Uses a convex-hull approach to create a simplified geometric shape with top,
    bottom and side faces. This is efficient but loses geometric detail. To retain
    the exact triangular geometry, export the scene with the lossless mesh
    representation (see ``Scene.export_data(..., lossless=True)``).

    Args:
        vertices: List of (x,y,z) vertex coordinates for the object

    Returns:
        List of faces, where each face is a list of (x,y,z) vertex coordinates

    """
    min_vertices_for_face = 3
    vertices = np.array(vertices)
    if len(vertices) < min_vertices_for_face:
        return None
    return _get_faces_convex_hull(vertices)
