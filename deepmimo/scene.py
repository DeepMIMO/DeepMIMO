"""Physical world representation module.

Provides geometry classes (BoundingBox, Face, PhysicalElement, PhysicalElementGroup, Scene)
and helper routines for road/mesh handling (2D face generation, endpoint detection, trimming,
path compression, angle deviation, intersection checks, and TSP path ordering).
"""

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull

from .consts import MAT_FMT, SCENE_PARAM_NUMBER_SCENES
from .general_utils import (
    DelegatingList,
    load_dict_from_json,
    load_mat,
    save_dict_as_json,
    save_mat,
)

if TYPE_CHECKING:
    from .materials import MaterialList

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
        self: Any,
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
    def x_min(self: Any) -> float:
        """Get minimum x coordinate."""
        return self.bounds[0, 0]

    @property
    def x_max(self: Any) -> float:
        """Get maximum x coordinate."""
        return self.bounds[1, 0]

    @property
    def y_min(self: Any) -> float:
        """Get minimum y coordinate."""
        return self.bounds[0, 1]

    @property
    def y_max(self: Any) -> float:
        """Get maximum y coordinate."""
        return self.bounds[1, 1]

    @property
    def z_min(self: Any) -> float:
        """Get minimum z coordinate."""
        return self.bounds[0, 2]

    @property
    def z_max(self: Any) -> float:
        """Get maximum z coordinate."""
        return self.bounds[1, 2]

    @property
    def width(self: Any) -> float:
        """Get the width (X dimension) of the bounding box."""
        return self.x_max - self.x_min

    @property
    def length(self: Any) -> float:
        """Get the length (Y dimension) of the bounding box."""
        return self.y_max - self.y_min

    @property
    def height(self: Any) -> float:
        """Get the height (Z dimension) of the bounding box."""
        return self.z_max - self.z_min

    @property
    def center(self: Any) -> np.ndarray:
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
        self: Any,
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
    def normal(self: Any) -> np.ndarray:
        """Get the normal vector of the face."""
        if self._normal is None:
            v1 = self.vertices[1] - self.vertices[0]
            v2 = self.vertices[2] - self.vertices[0]
            normal = np.cross(v1, v2)
            self._normal = normal / np.linalg.norm(normal)
        return self._normal

    @property
    def triangular_faces(self: Any) -> list[np.ndarray]:
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
    def num_triangular_faces(self: Any) -> int:
        """Get the number of triangular faces."""
        return len(self.triangular_faces)

    @property
    def area(self: Any) -> float:
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
    def centroid(self: Any) -> np.ndarray:
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
        self: Any,
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

    def _compute_bounding_box(self: Any) -> None:
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
    def height(self: Any) -> float:
        """Get the height of the object."""
        return self.bounding_box.height

    @property
    def faces(self: Any) -> list[Face]:
        """Get the faces of the object."""
        return self._faces

    @property
    def hull(self: Any) -> ConvexHull:
        """Get the convex hull of the object."""
        if self._hull is None:
            self._hull = ConvexHull(self.vertices)
        return self._hull

    @property
    def hull_volume(self: Any) -> float:
        """Get the volume of the object using its convex hull."""
        if self._hull_volume is None:
            self._hull_volume = self.hull.volume
        return self._hull_volume

    @property
    def hull_surface_area(self: Any) -> float:
        """Get the surface area of the object using its convex hull."""
        if self._hull_surface_area is None:
            self._hull_surface_area = self.hull.area
        return self._hull_surface_area

    @property
    def footprint_area(self: Any) -> float:
        """Get the area of the object's footprint using 2D convex hull."""
        if self._footprint_area is None:
            points_2d = self.vertices[:, :2]
            self._footprint_area = ConvexHull(points_2d).area
        return self._footprint_area

    @property
    def volume(self: Any) -> float:
        """Get the volume of the object using its convex hull."""
        return self.hull_volume

    def to_dict(self: Any, vertex_map: dict[tuple[float, ...], int]) -> dict:
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
    def from_dict(cls: Any, data: dict, vertices: np.ndarray) -> "PhysicalElement":
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

    @property
    def position(self: Any) -> np.ndarray:
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
        self: Any,
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
    def materials(self: Any) -> set[int]:
        """Get set of material indices used by this object."""
        if self._materials is None:
            self._materials = list({face.material_idx for face in self._faces})
        return self._materials

    @property
    def vel(self: Any) -> np.ndarray:
        """Get the speed vector of the object in Cartesian coordinates [m/s]."""
        return self._vel

    @vel.setter
    def vel(self: Any, value: np.ndarray | list | tuple) -> None:
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

    def __repr__(self: Any) -> str:
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

    def __init__(self: Any, objects: list[PhysicalElement]) -> None:
        """Initialize a group of physical objects."""
        self._objects = objects
        self._bounding_box: BoundingBox | None = None

    def __len__(self: Any) -> int:
        """Get number of objects in group."""
        return len(self._objects)

    def __iter__(self: Any) -> Any:
        """Iterate over objects in group."""
        return iter(self._objects)

    def __getitem__(self: Any, idx: int) -> PhysicalElement:
        """Get object by index."""
        return self._objects[idx]

    def __repr__(self: Any) -> str:
        """Return a concise string representation of the physical element group."""
        obj_list = "\n".join(f"  {obj}" for obj in self._objects)
        return f"PhysicalElementGroup(objects={len(self._objects)})\nObjects:\n{obj_list}"

    def get_materials(self: Any) -> list[int]:
        """Get list of material indices used by objects in this group."""
        return list(set().union(*(obj.materials for obj in self._objects)))

    def get_objects(
        self: Any,
        label: str | None = None,
        material: int | None = None,
    ) -> "PhysicalElementGroup":
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
    def bounding_box(self: Any) -> BoundingBox:
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

    def __init__(self: Any) -> None:
        """Initialize an empty scene."""
        self.objects = DelegatingList()
        self.visualization_settings = self.DEFAULT_VISUALIZATION_SETTINGS.copy()
        self.face_indices = []
        self._current_index = 0
        self._objects_by_category: dict[str, list[PhysicalElement]] = {
            cat: [] for cat in ELEMENT_CATEGORIES
        }
        self._objects_by_material: dict[int, list[PhysicalElement]] = {}
        self._materials: MaterialList | None = None

    @property
    def bounding_box(self: Any) -> BoundingBox:
        """Get the bounding box containing all objects."""
        return self.get_objects().bounding_box

    def set_visualization_settings(self: Any, label: str, settings: dict) -> None:
        """Set visualization settings for a specific label."""
        self.visualization_settings[label] = settings

    def add_object(self: Any, obj: PhysicalElement) -> None:
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

    def add_objects(self: Any, objects: list[PhysicalElement]) -> None:
        """Add multiple physical objects to the scene.

        Args:
            objects: List of PhysicalElement objects to add

        """
        for obj in objects:
            self.add_object(obj)

    def _add_face(self: Any, face: Face) -> list[int]:
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
        self: Any,
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

    def export_data(self: Any, base_folder: str) -> dict:
        """Export scene data to files and return metadata dictionary.

        Creates matrix files for vertices, faces and materials in the base folder.
        Returns a dictionary containing metadata needed to reload the scene.

        Args:
            base_folder: Base folder to store matrix files

        Returns:
            Dict containing metadata needed to reload the scene

        """
        Path(base_folder).mkdir(parents=True, exist_ok=True)
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
            "n_objects": len(self.objects),
            "n_vertices": len(vertices),
            "n_faces": sum(len(obj.faces) for obj in self.objects),
            "n_triangular_faces": sum(len(obj_face_idxs) for obj_face_idxs in self.face_indices),
        }

    @classmethod
    def from_data(cls: Any, base_folder: str) -> "Scene":
        """Create scene from metadata dictionary and data files.

        Args:
            base_folder: Base folder containing matrix files

        """
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

    def plot(  # noqa: PLR0912, PLR0913, C901
        self: Any,
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
        if ax is None:
            (_, ax) = plt.subplots(
                figsize=figsize,
                dpi=dpi,
                subplot_kw={"projection": "3d" if proj_3d else None},
            )
        label_groups = {}
        for obj in self.objects:
            if obj.label not in label_groups:
                label_groups[obj.label] = []
            label_groups[obj.label].append(obj)
        default_vis_settings = {"z_order": 3, "alpha": 0.8, "color": None}
        for label, objects in label_groups.items():
            vis_settings = self.visualization_settings.get(label, default_vis_settings)
            n_objects = len(objects)
            if vis_settings["color"] is None:
                colors = plt.cm.rainbow(np.linspace(0, 1, n_objects))
            else:
                colors = [vis_settings["color"]] * n_objects
            for obj_idx, obj in enumerate(objects):
                color = obj.color or colors[obj_idx]
                if proj_3d:
                    obj.plot(ax, mode=mode, alpha=vis_settings["alpha"], color=color)
                else:
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
        return ax

    def _set_axes_lims_to_scale(self: Any, ax: Any, zoom: float = 1.3) -> None:
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

    def _get_title_with_counts(self: Any) -> str:
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

    def count_objects_by_label(self: Any) -> dict[str, int]:
        """Count the number of objects for each label in the scene.

        Returns:
            Dict[str, int]: Dictionary mapping labels to their counts

        """
        label_counts = {}
        for obj in self.objects:
            label = obj.label
            label_counts[label] = label_counts.get(label, 0) + 1
        return label_counts

    def __repr__(self: Any) -> str:
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


def _calculate_angle_deviation(p1: Any, p2: Any, p3: Any) -> Any:
    """Calculate the deviation from a straight line at point p2.

    Returns angle in degrees, where:
    - 0° means the path p1->p2->p3 forms a straight line
    - 180° means the path doubles back on itself.
    """
    if np.allclose(p1, p2) or np.allclose(p2, p3):
        return 180.0
    v1 = p2 - p1
    v2 = p3 - p2
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    return np.degrees(np.arccos(dot_product))


def _ccw(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    """Check if points are in counter-clockwise order."""
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> bool:
    """Check if two line segments intersect."""
    return _ccw(p1, q1, q2) != _ccw(p2, q1, q2) and _ccw(p1, p2, q1) != _ccw(p1, p2, q2)


def _tsp_held_karp_no_intersections(points: np.ndarray) -> tuple[float, list[int]]:  # noqa: PLR0912, C901
    """Held-Karp TSP with angle penalty + intersection check.

    Returns:
        Tuple[float, List[int]]: Minimum cost and path

    """
    n = len(points)
    cost_cache: dict[tuple[int, int], tuple[float, list[int]]] = {}
    for k in range(1, n):
        dist = np.linalg.norm(points[0] - points[k])
        cost_cache[1 << k, k] = (dist, [0, k])
    for subset_size in range(2, n):
        for subset in itertools.combinations(range(1, n), subset_size):
            bits = sum(1 << x for x in subset)
            for k in subset:
                prev_bits = bits & ~(1 << k)
                res = []
                for m in subset:
                    if m == k:
                        continue
                    (prev_cost, prev_path) = cost_cache.get((prev_bits, m), (float("inf"), []))
                    if not prev_path:
                        continue
                    new_seg = (points[m], points[k])
                    intersects = False
                    for i in range(len(prev_path) - 2):
                        (a, b) = (prev_path[i], prev_path[i + 1])
                        if _segments_intersect(points[a], points[b], new_seg[0], new_seg[1]):
                            intersects = True
                            break
                    if intersects:
                        continue
                    if len(prev_path) > 1:
                        angle_cost = _calculate_angle_deviation(
                            points[prev_path[-2]],
                            points[m],
                            points[k],
                        )
                    else:
                        angle_cost = 0
                    cost = prev_cost + np.linalg.norm(points[m] - points[k]) + angle_cost
                    res.append((cost, [*prev_path, k]))
                if res:
                    cost_cache[bits, k] = min(res)
    bits = (1 << n) - 2
    res = []
    for k in range(1, n):
        if (bits, k) not in cost_cache:
            continue
        (cost, path) = cost_cache[bits, k]
        new_seg = (points[k], points[0])
        intersects = False
        for i in range(len(path) - 2):
            (a, b) = (path[i], path[i + 1])
            if _segments_intersect(points[a], points[b], new_seg[0], new_seg[1]):
                intersects = True
                break
        if intersects:
            continue
        angle_cost = _calculate_angle_deviation(points[path[-2]], points[k], points[0])
        final_cost = cost + np.linalg.norm(points[k] - points[0]) + angle_cost
        res.append((final_cost, [*path, 0]))
    return min(res) if res else (float("inf"), [])


def _detect_endpoints(
    points_2d: np.ndarray,
    min_distance: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect endpoints of a road by finding pairs of points that are furthest apart.

    Points closer than min_distance are considered duplicates and only one is kept.

    Args:
        points_2d: Array of 2D points (N x 2)
        min_distance: Minimum distance between points to consider them distinct

    Returns:
        List of endpoint indices alternating between the two detected pairs.

    """
    kept_indices = []
    used_points = set()
    for i in range(len(points_2d)):
        if i in used_points:
            continue
        distances = np.linalg.norm(points_2d - points_2d[i], axis=1)
        close_points = np.where(distances < min_distance)[0]
        used_points.update(close_points)
        kept_indices.append(i)
    filtered_points = points_2d[kept_indices]
    distances = np.linalg.norm(filtered_points[:, np.newaxis] - filtered_points, axis=2)
    (i1, j1) = np.unravel_index(np.argmax(distances), distances.shape)
    distances_masked = distances.copy()
    distances_masked[i1, :] = -np.inf
    distances_masked[:, i1] = -np.inf
    distances_masked[j1, :] = -np.inf
    distances_masked[:, j1] = -np.inf
    (i2, j2) = np.unravel_index(np.argmax(distances_masked), distances_masked.shape)
    return [kept_indices[i] for i in [i1, i2, j1, j2]]


def _signed_distance_to_curve(
    point: np.ndarray,
    curve_fit: np.poly1d,
    x_range: tuple[float, float],
) -> tuple[float, np.ndarray]:
    """Calculate signed perpendicular distance from point to curve.

    Positive distance means point is on one side, negative on the other.

    Args:
        point: Point to calculate distance to
        curve_fit: Polynomial fit to the curve
        x_range: Range of x-values for the curve

    Returns:
        Tuple[float, np.ndarray]: Signed distance and closest point on curve

    """
    curve_x = np.linspace(x_range[0], x_range[1], 1000)
    curve_y = curve_fit(curve_x)
    curve_points = np.column_stack((curve_x, curve_y))
    distances = np.linalg.norm(curve_points - point, axis=1)
    closest_idx = np.argmin(distances)
    closest_point = curve_points[closest_idx]
    if closest_idx < len(curve_x) - 1:
        tangent = curve_points[closest_idx + 1] - curve_points[closest_idx]
    else:
        tangent = curve_points[closest_idx] - curve_points[closest_idx - 1]
    tangent = tangent / np.linalg.norm(tangent)
    normal = np.array([-tangent[1], tangent[0]])
    vec_to_point = point - closest_point
    signed_dist = np.dot(vec_to_point, normal)
    return (signed_dist, closest_point)


def _trim_points_protected(
    points: np.ndarray,
    protected_indices: list[int],
    max_points: int = 14,
) -> list[int]:
    """Trims points while preserving protected indices and maintaining road shape.

    Uses reference points along the curve to select closest points above and below.
    Assumes endpoints are included in protected_indices.

    Args:
        points: Array of point coordinates (N x 2)
        protected_indices: List of indices that should not be removed
        max_points: Maximum number of points to keep
        debug: Whether to show debug plots

    Returns:
        List of indices of the kept points

    """
    protected_indices = set(protected_indices)
    if max_points < len(protected_indices):
        msg = "max_points must be >= number of protected points"
        raise ValueError(msg)
    if len(points) < len(protected_indices):
        msg = "len(points) must be >= max_points"
        raise ValueError(msg)
    x = points[:, 0]
    y = points[:, 1]
    z = np.polyfit(x, y, 3)
    curve_fit = np.poly1d(z)
    x_range = (x.min(), x.max())
    distances_and_closest = [
        _signed_distance_to_curve(points[i], curve_fit, x_range) for i in range(len(points))
    ]
    distances = np.array([d for (d, _) in distances_and_closest])
    ref_positions = [0.25, 0.5, 0.75]
    x_refs = x_range[0] + (x_range[1] - x_range[0]) * np.array(ref_positions)
    ref_points = np.column_stack((x_refs, curve_fit(x_refs)))
    kept_indices = set(protected_indices)
    for ref_point in ref_points:
        dists_to_ref = np.linalg.norm(points - ref_point, axis=1)
        above_curve = distances > 0
        below_curve = distances < 0
        above_indices = [
            i for i in range(len(points)) if above_curve[i] and i not in protected_indices
        ]
        below_indices = [
            i for i in range(len(points)) if below_curve[i] and i not in protected_indices
        ]
        above_indices = sorted(above_indices, key=lambda i: dists_to_ref[i])
        below_indices = sorted(below_indices, key=lambda i: dists_to_ref[i])
        for idx in above_indices:
            if idx not in kept_indices:
                kept_indices.add(idx)
                break
        for idx in below_indices:
            if idx not in kept_indices:
                kept_indices.add(idx)
                break
    return sorted(kept_indices)


def _compress_path(points: np.ndarray, path: list[int], angle_threshold: float = 1.0) -> list[int]:
    """Compress a path by removing points that are nearly collinear with their neighbors.

    Args:
        points: Array of point coordinates (N x 2)
        path: List of indices forming the path
        angle_threshold: Minimum angle deviation (in degrees) to keep a point

    Returns:
        List of indices forming the compressed path

    """
    min_path_len = 3
    if len(path) <= min_path_len:
        return path
    compressed = [path[0]]
    for i in range(1, len(path) - 1):
        prev_idx = compressed[-1]
        curr_idx = path[i]
        next_idx = path[i + 1]
        angle = _calculate_angle_deviation(points[prev_idx], points[curr_idx], points[next_idx])
        if angle > angle_threshold:
            compressed.append(curr_idx)
    compressed.append(path[-1])
    return compressed


def _get_2d_face(
    vertices: np.ndarray,
    z_tolerance: float = 0.1,
    max_points: int = 10,
    *,
    compress: bool = True,
    angle_threshold: float = 1.0,
) -> list[tuple[float, float, float]]:
    """Generate a 2D face from a set of vertices.

    Args:
        vertices: Array of vertex coordinates (shape: N x 3)
        z_tolerance: Tolerance for z-coordinate variation - targetted for roads
        max_points: Maximum number of points to consider
        compress: Whether to compress the final path
        angle_threshold: Angle threshold for collinearity

    Returns:
        List of (x,y,z) vertex coordinates for the face

    """
    if not np.allclose(vertices[:, 2], vertices[0, 2], atol=z_tolerance):
        msg = "Vertices are not 2D"
        raise ValueError(msg)
    endpoints = _detect_endpoints(vertices[:, :2])
    kept_indices = _trim_points_protected(
        vertices[:, :2],
        protected_indices=endpoints,
        max_points=max_points,
    )
    points_filtered = vertices[kept_indices]
    (_, best_path) = _tsp_held_karp_no_intersections(points_filtered[:, :2])
    if compress:
        compressed_path = _compress_path(
            points_filtered,
            best_path,
            angle_threshold=angle_threshold,
        )
        final_points = points_filtered[compressed_path[:-1]]
    else:
        final_points = points_filtered[best_path[:-1]]
    return [final_points]


def get_object_faces(
    vertices: list[tuple[float, float, float]],
    *,
    fast: bool = True,
) -> list[list[tuple[float, float, float]]]:
    """Generate faces for a physical object from its vertices.

    This function supports two modes:
    1. Fast mode (default):
       - Uses convex hull to create a simplified geometric shape
       - Creates top, bottom and side faces
       - More efficient but loses geometric detail

    2. Detailed mode:
       - Detects coplanar sets of vertices to form faces
       - Preserves original geometry
       - Slower but more accurate

    Args:
        vertices: List of (x,y,z) vertex coordinates for the object
        fast: Whether to use fast convex-hull mode or detailed coplanar detection

    Returns:
        List of faces, where each face is a list of (x,y,z) vertex coordinates

    """
    min_vertices_for_face = 3
    vertices = np.array(vertices)
    if len(vertices) < min_vertices_for_face:
        return None
    return _get_faces_convex_hull(vertices) if fast else _get_2d_face(vertices)


if __name__ == "__main__":
    points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    path = [0, 1, 2, 3]
    compressed = _compress_path(points, path)
    print(compressed)

    def plot_points(points: Any, path: Any = None, title: Any = "") -> None:
        """Visualize a set of points and optionally connect them."""
        plt.figure(figsize=(8, 6))
        plt.scatter(points[:, 0], points[:, 1], color="blue")
        for i, (x, y) in enumerate(points):
            plt.text(x + 1, y + 1, str(i), fontsize=9)
        if path:
            for i in range(len(path) - 1):
                (p1, p2) = (points[path[i]], points[path[i + 1]])
                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], "r-")
        plt.title(title)
        plt.axis("equal")
        plt.grid(visible=True)
        plt.show()
