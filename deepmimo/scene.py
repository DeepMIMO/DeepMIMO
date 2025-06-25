"""
Physical world representation module.

This module provides core classes for representing physical objects in a wireless environment,
including buildings, terrain, vegetation, and other structures that affect wireless propagation.

Module Organization:
1. Constants - Categories and labels for physical elements
2. Core Classes - Main classes for scene representation:
   - BoundingBox: 3D bounding box representation
   - Face: Surface representation with dual face approach
   - PhysicalElement: Base class for physical objects
   - PhysicalElementGroup: Group operations on physical elements
   - Scene: Complete physical environment representation
3. Object handling: Get faces of objects as lists of vertices
    - get_object_faces: Generate faces for physical objects
    - Road object handling:
        - _get_2d_face: Get 2D face of road objects (calls all functions below)
        - _detect_endpoints: Detect endpoints (terminations) of road objects
        - _trim_points_protected: Trim points of road objects
        - _compress_path: Compress path of road objects
        - _calculate_angle_deviation: Calculate angle deviation 
          (used in _compress_path and _tsp_held_karp_no_intersections)
        - _ccw: Check if points are in counter-clockwise order (used in _segments_intersect)
        - _segments_intersect: Check if two line segments intersect (used in _tsp_held_karp_no_intersections)
        - _tsp_held_karp_no_intersections: Held-Karp TSP with angle penalty + intersection check
          (used in _get_2d_face)
        - _signed_distance_to_curve: Calculate signed distance from point to curve
          (used in _trim_points_protected to trim along the road)

The `Scene` class acts as a container for multiple `PhysicalElement` objects, 
each representing a distinct object in the environment. Each `PhysicalElement` is 
composed of `Face` objects, which define the surfaces of the element and are associated 
with materials. The `BoundingBox` class provides spatial boundaries for these elements. 
Together, these components allow for the representation and manipulation of complex environments, 
with functionalities for plotting and material management integrated into the scene.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import itertools
from scipy.spatial import ConvexHull
from scipy.io import savemat, loadmat
from typing import List, Dict, Tuple, Literal, Optional, Set
from dataclasses import dataclass
from pathlib import Path
from .materials import MaterialList
from .consts import SCENE_PARAM_NUMBER_SCENES
from .general_utils import (
    load_dict_from_json,
    save_dict_as_json,
    cartesian_to_spherical,
    DelegatingList
)

#------------------------------------------------------------------------------
# Constants
#------------------------------------------------------------------------------

# Physical element categories
CAT_BUILDINGS: str = 'buildings'      # Building structures
CAT_TERRAIN: str = 'terrain'          # Ground/terrain surfaces
CAT_VEGETATION: str = 'vegetation'    # Vegetation/foliage
CAT_FLOORPLANS: str = 'floorplans'    # Indoor floorplans
CAT_OBJECTS: str = 'objects'          # Other scene objects

# All valid categories (used for search - can be extended by users)
ELEMENT_CATEGORIES = [
    CAT_BUILDINGS,
    CAT_TERRAIN,
    CAT_VEGETATION,
    CAT_FLOORPLANS,
    CAT_OBJECTS
]

#------------------------------------------------------------------------------
# Core Classes
#------------------------------------------------------------------------------

@dataclass
class BoundingBox:
    """Represents a 3D bounding box with min/max coordinates."""
    # Store bounds as numpy array for min/max of x,y,z
    bounds: np.ndarray  # shape: (2, 3) for min/max of x,y,z
    
    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float, z_min: float, z_max: float):
        """Initialize bounding box with min/max coordinates."""
        self.bounds = np.array([
            [x_min, y_min, z_min],  # mins
            [x_max, y_max, z_max]   # maxs
        ])
    
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
        return np.array([
            (self.x_max + self.x_min) / 2,
            (self.y_max + self.y_min) / 2, 
            (self.z_max + self.z_min) / 2])

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
    
    def __init__(self, vertices: List[Tuple[float, float, float]] | np.ndarray, 
                 material_idx: int | np.integer = 0):
        """Initialize a face from its vertices.
        
        Args:
            vertices: List of (x, y, z) coordinates or numpy array of shape (N, 3)
                defining the face vertices in counter-clockwise order
            material_idx: Index of the material for this face (default: 0)
        """
        self.vertices = np.asarray(vertices, dtype=np.float32)
        self.material_idx = int(material_idx)  # Convert to Python int
        self._normal: np.ndarray | None = None
        self._area: float | None = None
        self._centroid: np.ndarray | None = None
        self._triangular_faces: List[np.ndarray] | None = None
        
    @property
    def normal(self) -> np.ndarray:
        """Get the normal vector of the face."""
        if self._normal is None:
            # Calculate normal using cross product of two edges
            v1 = self.vertices[1] - self.vertices[0]
            v2 = self.vertices[2] - self.vertices[0]
            normal = np.cross(v1, v2)
            self._normal = normal / np.linalg.norm(normal)
        return self._normal
    
    @property
    def triangular_faces(self) -> List[np.ndarray]:
        """Get the triangular faces that make up this face."""
        if self._triangular_faces is None:
            # If face is already a triangle, return it as is
            if len(self.vertices) == 3:
                self._triangular_faces = [self.vertices]
            else:
                # Triangulate the face using fan triangulation
                # This assumes the face is convex and planar
                triangles = []
                for i in range(1, len(self.vertices) - 1):
                    triangle = np.array([
                        self.vertices[0],
                        self.vertices[i],
                        self.vertices[i + 1]
                    ])
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
            # Project vertices onto the plane defined by the normal
            n = self.normal
            # Find the coordinate axis most aligned with the normal
            proj_axis = np.argmax(np.abs(n))
            # Get the other two axes for projection
            other_axes = [i for i in range(3) if i != proj_axis]
            
            # Project points onto the selected plane
            points = self.vertices[:, other_axes]
            
            # Calculate area using shoelace formula
            x = points[:, 0]
            y = points[:, 1]
            # Roll arrays for vectorized computation
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
    
    # Default labels that can be used (users may define their own - only used for search)
    DEFAULT_LABELS = {CAT_BUILDINGS, CAT_TERRAIN, CAT_VEGETATION, CAT_FLOORPLANS, CAT_OBJECTS}
    
    def __init__(self, faces: List[Face], object_id: int = -1, 
                 label: str = CAT_OBJECTS, color: str = '', name: str = ''):
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
        self._vel: np.ndarray = np.zeros(3)  # Speed vector in Cartesian coordinates [m/s]
        
        # Extract all vertices from faces for bounding box computation
        all_vertices = np.vstack([face.vertices for face in faces])
        self.vertices = all_vertices
        self.bounding_box: BoundingBox
        self._footprint_area: float | None = None
        self._position: np.ndarray | None = None
        self._hull: ConvexHull | None = None
        self._hull_volume: float | None = None
        self._hull_surface_area: float | None = None
        
        # Cache material indices
        self._materials: Optional[Set[int]] = None
        
        # Compute bounding box immediately
        self._compute_bounding_box()
    
    def _compute_bounding_box(self) -> None:
        """Compute the object's bounding box."""
        mins = np.min(self.vertices, axis=0)
        maxs = np.max(self.vertices, axis=0)
        self.bounding_box = BoundingBox(
            x_min=mins[0], x_max=maxs[0],
            y_min=mins[1], y_max=maxs[1],
            z_min=mins[2], z_max=maxs[2]
        )
    
    @property
    def height(self) -> float:
        """Get the height of the object."""
        return self.bounding_box.height

    @property
    def faces(self) -> List[Face]:
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
            # Project all vertices to 2D and compute convex hull
            points_2d = self.vertices[:, :2]
            self._footprint_area = ConvexHull(points_2d).area
        return self._footprint_area
    
    @property
    def volume(self) -> float:
        """Get the volume of the object using its convex hull."""
        return self.hull_volume
    
    def to_dict(self, vertex_map: Dict[Tuple[float, ...], int]) -> Dict:
        """Convert physical object to dictionary format.
        
        Args:
            vertex_map: Dictionary mapping vertex tuples to their global indices
            
        Returns:
            Dict containing object metadata with face vertex and material indices
        """
        obj_metadata = {
            'name': self.name,
            'label': self.label,
            'id': self.object_id,
            'face_vertex_idxs': [],
            'face_material_idxs': []
        }
        
        # Process each face
        for face in self.faces:
            # Get vertex indices for this face
            face_vertex_indices = []
            for tri_vertices in face.triangular_faces:
                for vertex in tri_vertices:
                    vertex_tuple = tuple(vertex)
                    if vertex_tuple not in vertex_map:
                        vertex_map[vertex_tuple] = len(vertex_map)
                    if vertex_map[vertex_tuple] not in face_vertex_indices:
                        face_vertex_indices.append(vertex_map[vertex_tuple])
            
            # Store vertex indices and material index
            obj_metadata['face_vertex_idxs'].append(face_vertex_indices)
            obj_metadata['face_material_idxs'].append(face.material_idx)
        
        return obj_metadata
    
    @classmethod
    def from_dict(cls, data: Dict, vertices: np.ndarray) -> 'PhysicalElement':
        """Create physical object from dictionary format.
        
        Args:
            data: Dictionary containing object data
            vertices: Array of vertex coordinates (shape: N_vertices x 3)
            
        Returns:
            PhysicalElement: Created object
        """
        faces = [Face(vertices=vertices[vertex_idxs], material_idx=material_idx)
                 for vertex_idxs, material_idx in zip(data['face_vertex_idxs'], data['face_material_idxs'])]
        return cls(faces=faces, name=data['name'], object_id=data['id'], label=data['label'])

    @property
    def position(self) -> np.ndarray:
        """Get the center of mass (position) of the object."""
        if self._position is None:
            bb = self.bounding_box
            # Calculate center as midpoint of bounding box
            self._position = np.array([
                (bb.x_max + bb.x_min) / 2,
                (bb.y_max + bb.y_min) / 2, 
                (bb.z_max + bb.z_min) / 2
            ])
        return self._position

    def plot(self, ax: Optional[plt.Axes] = None, mode: Literal['faces', 'tri_faces'] = 'faces',
            alpha: float = 0.8, color: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
        """Plot the object using the specified visualization mode.
        
        Args:
            ax: Matplotlib 3D axes to plot on (if None, creates new figure)
            mode: Visualization mode - either 'faces' or 'tri_faces' (default: 'faces')
            alpha: Transparency for visualization (default: 0.8)
            color: Color for visualization (default: None, uses object's color)
        """
        ax = ax or plt.subplots(1, 1, subplot_kw={'projection': '3d'})[1]
        
        # Get vertices based on mode
        if mode == 'faces':
            vertices_list = [face.vertices for face in self.faces]
        elif mode == 'tri_faces':
            vertices_list = [tri for face in self.faces for tri in face.triangular_faces]
        
        # Plot all vertices
        for vertices in vertices_list:
            poly3d = Poly3DCollection([vertices], alpha=alpha)
            plot_color = self.color or color
            poly3d.set_facecolor(plot_color)
            poly3d.set_edgecolor('black')
            ax.add_collection3d(poly3d)
            
        return ax.get_figure(), ax

    @property
    def materials(self) -> Set[int]:
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
        # Ensure value is a 3D vector
        if type(value) == list or type(value) == tuple:
            value = np.array(value)
        if value.shape != (3,):
            raise ValueError("Velocity must be a 3D vector (x, y, z) in meters per second")
        self._vel = value
        
    def __repr__(self) -> str:
        """Return a concise string representation of the physical element.
        
        Returns:
            str: String representation showing key element information
        """
        bb = self.bounding_box
        dims = f"{bb.width:.0f} x {bb.length:.0f} x {bb.height:.0f} m"
        return (f"PhysicalElement(name='{self.name}', id={self.object_id}, "
                f"label='{self.label}', faces={len(self._faces)}, dims={dims})")

class PhysicalElementGroup:
    """Represents a group of physical objects that can be queried and manipulated together."""
    
    def __init__(self, objects: List[PhysicalElement]):
        """Initialize a group of physical objects."""
        self._objects = objects
        self._bounding_box: Optional[BoundingBox] = None
        
    def __len__(self) -> int:
        """Get number of objects in group."""
        return len(self._objects)
        
    def __iter__(self):
        """Iterate over objects in group."""
        return iter(self._objects)
        
    def __getitem__(self, idx: int) -> PhysicalElement:
        """Get object by index."""
        return self._objects[idx]
    
    def __repr__(self) -> str:
        """Return a concise string representation of the physical element group."""
        obj_list = "\n".join(f"  {obj}" for obj in self._objects)
        return (f"PhysicalElementGroup(objects={len(self._objects)})\n"
                f"Objects:\n{obj_list}")

    def get_materials(self) -> List[int]:
        """Get list of material indices used by objects in this group."""
        return list(set().union(*(obj.materials for obj in self._objects)))
    
    def get_objects(self, label: Optional[str] = None, material: Optional[int] = None) -> 'PhysicalElementGroup':
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
                raise ValueError("Group is empty")
            
            # Collect all object bounding boxes
            boxes = [obj.bounding_box.bounds for obj in self._objects]
            boxes = np.array(boxes)  # Shape: (N, 2, 3)
            
            # Compute global bounds
            global_min = np.min(boxes[:, 0], axis=0)  # Min of mins
            global_max = np.max(boxes[:, 1], axis=0)  # Max of maxs
            
            self._bounding_box = BoundingBox(
                x_min=global_min[0], x_max=global_max[0],
                y_min=global_min[1], y_max=global_max[1],
                z_min=global_min[2], z_max=global_max[2]
            )
        return self._bounding_box

class Scene:
    """Represents a physical scene with various objects affecting wireless propagation."""
    
    # Default visualization settings for different labels
    DEFAULT_VISUALIZATION_SETTINGS = {
        CAT_TERRAIN: {'z_order': 1, 'alpha': 0.1, 'color': 'grey'},
        CAT_VEGETATION: {'z_order': 2, 'alpha': 0.8, 'color': 'green'},
        CAT_BUILDINGS: {'z_order': 3, 'alpha': 0.8, 'color': None},  # use random color
        CAT_FLOORPLANS: {'z_order': 4, 'alpha': 0.8, 'color': 'blue'},
        CAT_OBJECTS: {'z_order': 5, 'alpha': 0.8, 'color': 'blue'}
    }
    
    def __init__(self):
        """Initialize an empty scene."""
        self.objects = DelegatingList()
        self.visualization_settings = self.DEFAULT_VISUALIZATION_SETTINGS.copy()
        
        # Matrix storage tracking
        self.face_indices = []  # List[List[List[int]]] for [object][face][triangle_indices]
        self._current_index = 0
        
        # Initialize tracking dictionaries
        self._objects_by_category: Dict[str, List[PhysicalElement]] = {
            cat: [] for cat in ELEMENT_CATEGORIES
        }
        self._objects_by_material: Dict[int, List[PhysicalElement]] = {}
        self._materials: Optional[MaterialList] = None
        
    @property
    def bounding_box(self) -> BoundingBox:
        """Get the bounding box containing all objects."""
        return self.get_objects().bounding_box
    
    def set_visualization_settings(self, label: str, settings: Dict) -> None:
        """Set visualization settings for a specific label."""
        self.visualization_settings[label] = settings

    def add_object(self, obj: PhysicalElement) -> None:
        """Add a physical object to the scene.
        
        Args:
            obj: PhysicalElement to add
        """
        if obj.object_id == -1:
            obj.object_id = len(self.objects)
        
        # Add faces to scene and track indices
        obj_indices = []
        for face in obj.faces:
            face_indices = self._add_face(face)
            obj_indices.append(face_indices)
        
        # Track object by materials
        for material_idx in obj.materials:
            if material_idx not in self._objects_by_material:
                self._objects_by_material[material_idx] = []
            self._objects_by_material[material_idx].append(obj)
        
        # Track object by category
        category = obj.label if obj.label in ELEMENT_CATEGORIES else CAT_OBJECTS
        if category not in self._objects_by_category:
            self._objects_by_category[category] = []
        self._objects_by_category[category].append(obj)
        
        self.face_indices.append(obj_indices)
        self.objects.append(obj)
        self._bounding_box = None  # Reset cached bounding box
    
    def add_objects(self, objects: List[PhysicalElement]) -> None:
        """Add multiple physical objects to the scene.
        
        Args:
            objects: List of PhysicalElement objects to add
        """
        for obj in objects:
            self.add_object(obj)
    
    def _add_face(self, face: Face) -> List[int]:
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
    
    def get_objects(self, label: Optional[str] = None, material: Optional[int] = None) -> PhysicalElementGroup:
        """Get objects filtered by label and/or material.
        
        Args:
            label: Optional label to filter objects by
            material: Optional material index to filter objects by
            
        Returns:
            PhysicalElementGroup containing filtered objects
        """
        # Get initial objects based on first filter
        if label:
            objects = self._objects_by_category.get(label, [])
        elif material:
            objects = self._objects_by_material.get(material, [])
        else:
            objects = self.objects
            
        # Create group and apply material filter if needed
        group = PhysicalElementGroup(objects)
        
        return group.get_objects(material=material) if material else group
    
    def export_data(self, base_folder: str) -> Dict:
        """Export scene data to files and return metadata dictionary.
        
        Creates matrix files for vertices, faces and materials in the base folder.
        Returns a dictionary containing metadata needed to reload the scene.
        
        Args:
            base_folder: Base folder to store matrix files
            
        Returns:
            Dict containing metadata needed to reload the scene
        """
        # Create base folder if it doesn't exist
        Path(base_folder).mkdir(parents=True, exist_ok=True)
        
        # Initialize vertex mapping
        vertex_map = {}  # Maps (x,y,z) tuple to vertex index
        
        # Convert objects to metadata format
        objects_metadata = []
        for obj in self.objects:
            # to_dict will update vertex_map as needed
            obj_metadata = obj.to_dict(vertex_map)
            objects_metadata.append(obj_metadata)
        
        # Convert vertices to array
        all_vertices = [None] * len(vertex_map)
        for vertex, idx in vertex_map.items():
            all_vertices[idx] = vertex
        vertices = np.array(all_vertices)  # Shape: (N_vertices, 3)
        
        # Save matrices
        savemat(f"{base_folder}/vertices.mat", {'vertices': vertices})
        save_dict_as_json(f"{base_folder}/objects.json", objects_metadata)
        
        return {
            SCENE_PARAM_NUMBER_SCENES: 1,
            'n_objects': len(self.objects),
            'n_vertices': len(vertices),
            'n_faces': sum(len(obj.faces) for obj in self.objects),
            'n_triangular_faces': sum(len(obj_face_idxs) for obj_face_idxs in self.face_indices)
        }
    
    @classmethod
    def from_data(cls, base_folder: str) -> 'Scene':
        """Create scene from metadata dictionary and data files.
        
        Args:
            base_folder: Base folder containing matrix files
        """
        scene = cls()
        try:
            vertices = loadmat(f"{base_folder}/vertices.mat")['vertices']
            objects_metadata = load_dict_from_json(f"{base_folder}/objects.json")
        except FileNotFoundError:
            print(f"FileNotFoundError: {base_folder}/vertices.mat or {base_folder}/objects.json not found")
            vertices = np.array([])
            objects_metadata = []
        except Exception as e:
            raise Exception(f"Error loading scene from {base_folder}: {e}")

        # Create objects using metadata
        for object_data in objects_metadata:
            obj = PhysicalElement.from_dict(object_data, vertices)
            scene.add_object(obj)
        
        return scene
    
    def plot(self, title: bool = True, mode: Literal['faces', 'tri_faces'] = 'faces', 
             ax: Optional[plt.Axes] = None, proj_2d: bool = False, figsize: tuple = (10, 10), 
             dpi: int = 100, legend: bool = False) -> plt.Axes:
        """Create a visualization of the scene.
        
        The scene can be visualized in either 2D (top-down view) or 3D mode:
        
        3D Mode (proj_2d=False):
            Two representation options:
            1. 'faces' (default) - Uses the primary convex hull representation
            - More efficient for visualization
            - Cleaner look for simple geometric shapes
            - Suitable for most visualization needs
               
            2. 'tri_faces' - Uses the secondary triangular representation
            - Shows detailed geometry
            - Better for debugging geometric issues
            - More accurate representation of complex shapes
               
        2D Mode (proj_2d=True):
            Creates a top-down view showing object footprints:
            - Projects all objects onto x-y plane
            - Uses convex hulls for efficient visualization
            - Better for understanding spatial layout
            - More efficient for large scenes
        
        Args:
            title: Whether to display the title (default: True)
            save: Whether to save the plot to a file (default: False)
            filename: Name of the file to save the plot to (if save is True)
            mode: Visualization mode for 3D - either 'faces' or 'tri_faces' (default: 'faces')
            ax: Matplotlib axes to plot on (if None, creates new figure)
            proj_2d: Whether to create a 2D projection (top-down view) (default: False)
            figsize: Figure dimensions (width, height) in inches (default: (10, 10))
            dpi: Plot resolution in dots per inch (default: 100)
            
        Returns:
            matplotlib Axes object
        """
        if ax is None:
            _, ax = plt.subplots(figsize=figsize, dpi=dpi,
                               subplot_kw={'projection': None if proj_2d else '3d'})
        
        # Group objects by label
        label_groups = {}
        for obj in self.objects:
            if obj.label not in label_groups:
                label_groups[obj.label] = []
            label_groups[obj.label].append(obj)
        
        # Plot each label group
        default_vis_settings = {'z_order': 3, 'alpha': 0.8, 'color': None}
        for label, objects in label_groups.items():
            # Get visualization settings for this label
            vis_settings = self.visualization_settings.get(label, default_vis_settings)
            
            # Use rainbow colormap for objects without fixed color
            n_objects = len(objects)
            if vis_settings['color'] is None:
                colors = plt.cm.rainbow(np.linspace(0, 1, n_objects))
            else:
                colors = [vis_settings['color']] * n_objects
            
            for obj_idx, obj in enumerate(objects):
                # Determine color (same for faces and hull)
                color = obj.color or colors[obj_idx]
                
                if proj_2d:
                    # Project vertices to 2D (x-y plane)
                    vertices_2d = obj.vertices[:, :2]
                    
                    # Create convex hull of 2D points
                    hull = ConvexHull(vertices_2d)
                    hull_vertices = vertices_2d[hull.vertices]
                    
                    # Plot the hull as a filled polygon
                    ax.fill(hull_vertices[:, 0], hull_vertices[:, 1],
                            alpha=vis_settings['alpha'], color=color,
                            label=label if obj_idx == 0 else "")
                else:
                    # Plot object with specified 3D mode
                    obj.plot(ax, mode=mode, alpha=vis_settings['alpha'], color=color)
        
        # Set axis labels
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        if not proj_2d:
            ax.set_zlabel('Z (m)')
        
        if title:
            ax.set_title(self._get_title_with_counts())
        
        if proj_2d:
            # Set equal aspect ratio for proper scaling in 2D
            ax.set_aspect('equal')
            # Add grid for 2D view
            ax.grid(True, alpha=0.3)
        else:
            # Set the view angle for better 3D perspective
            ax.view_init(elev=40, azim=-45)
            # Set 3D axes limits to scale
            self._set_axes_lims_to_scale(ax)
        
        # Add legend if there are multiple labels
        if len(label_groups) > 1 and legend:
            ax.legend()
        
        return ax

    def _set_axes_lims_to_scale(self, ax, zoom: float = 1.3):
        """Set axis limits based on scene bounding box with equal scaling.
        
        Args:
            ax: Matplotlib 3D axes to set limits on
            zoom: Zoom factor (>1 zooms out, <1 zooms in)
        """
        bb = self.bounding_box
        
        # Find center point
        center_x = (bb.x_max + bb.x_min) / 2
        center_y = (bb.y_max + bb.y_min) / 2
        center_z = (bb.z_max + bb.z_min) / 2
        
        # Use the largest dimension to ensure equal scaling
        max_range = max(bb.width, bb.length, bb.height) / 2 / zoom
        
        # Set limits equidistant from center
        ax.set_xlim3d([center_x - max_range, center_x + max_range])
        ax.set_ylim3d([center_y - max_range, center_y + max_range])
        ax.set_zlim3d([center_z - max_range, center_z + max_range])
        
        # Ensure equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
    
    def _get_title_with_counts(self) -> str:
        """Generate a title string with object counts for each label.
        
        Returns:
            Title string with object counts
        """
        # Count objects by label
        label_counts = {}
        for obj in self.objects:
            label_counts[obj.label] = label_counts.get(obj.label, 0) + 1
        
        # Format counts
        counts = []
        for label, count in label_counts.items():
            label_name = label.capitalize()
            if count == 1 and label_name.endswith('s'):
                label_name = label_name[:-1]
            counts.append(f"{label_name}: {count}")
        
        return ", ".join(counts)

    def count_objects_by_label(self) -> Dict[str, int]:
        """Count the number of objects for each label in the scene.
        
        Returns:
            Dict[str, int]: Dictionary mapping labels to their counts
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
        # Get object counts by label
        label_counts = self.count_objects_by_label()
        
        # Get scene dimensions
        bb = self.bounding_box
        dims = f"{bb.width:.1f} x {bb.length:.1f} x {bb.height:.1f} m"
        
        # Format object counts
        counts = [f"{label}: {count}" for label, count in label_counts.items()]
        counts_str = ", ".join(counts)
        
        return f"Scene({len(self.objects)} objects [{counts_str}], dims = {dims})"

#------------------------------------------------------------------------------
# Utilities
#------------------------------------------------------------------------------

def _get_faces_convex_hull(vertices: np.ndarray) -> List[List[Tuple[float, float, float]]]:
    """Generate faces using convex hull approach (fast but simplified).
    
    Args:
        vertices: Array of vertex coordinates (shape: N x 3)
        
    Returns:
        List of faces, where each face is a list of (x,y,z) vertex coordinates
    """
    # Extract base points (x,y coordinates)
    points_2d = vertices[:, :2]
    
    # Get object height
    heights = vertices[:, 2]
    object_height = np.max(heights) - np.min(heights)
    base_height = np.min(heights)
    
    # Create convex hull for base shape
    try:
        hull = ConvexHull(points_2d)
        base_shape = points_2d[hull.vertices]
    except Exception as e:
        if np.linalg.matrix_rank(points_2d - points_2d[0]) < 2:
            print('Convex hull failed - collinear vertices')
            return None
        else:
            raise e
    
    # Create top and bottom faces
    bottom_face = [(x, y, base_height) for x, y in base_shape]
    top_face = [(x, y, base_height + object_height) for x, y in base_shape]
    
    # Create side faces
    side_faces = []
    for i in range(len(base_shape)):
        j = (i + 1) % len(base_shape)
        side = [
            bottom_face[i],
            bottom_face[j],
            top_face[j],
            top_face[i]
        ]
        side_faces.append(side)
    
    return [bottom_face, top_face] + side_faces

def _calculate_angle_deviation(p1, p2, p3):
    """Calculate the deviation from a straight line at point p2.
    Returns angle in degrees, where:
    - 0° means the path p1->p2->p3 forms a straight line
    - 180° means the path doubles back on itself
    """
    if np.allclose(p1, p2) or np.allclose(p2, p3):
        return 180.0
    v1 = p2 - p1  # Vector from p1 to p2
    v2 = p3 - p2  # Vector from p2 to p3
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)

    return np.degrees(np.arccos(dot_product))

def _ccw(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    """Check if points are in counter-clockwise order."""
    return (c[1]-a[1]) * (b[0]-a[0]) > (b[1]-a[1]) * (c[0]-a[0])

def _segments_intersect(p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> bool:
    """Check if two line segments intersect."""
    return _ccw(p1, q1, q2) != _ccw(p2, q1, q2) and _ccw(p1, p2, q1) != _ccw(p1, p2, q2)

def _tsp_held_karp_no_intersections(points: np.ndarray) -> Tuple[float, List[int]]:
    """Held-Karp TSP with angle penalty + intersection check.
    Returns:
        Tuple[float, List[int]]: Minimum cost and path
    """
    n = len(points)
    C = {}
    
    for k in range(1, n):
        dist = np.linalg.norm(points[0] - points[k])
        C[(1 << k, k)] = (dist, [0, k])

    for subset_size in range(2, n):
        for subset in itertools.combinations(range(1, n), subset_size):
            bits = sum(1 << x for x in subset)
            for k in subset:
                prev_bits = bits & ~(1 << k)
                res = []
                for m in subset:
                    if m == k:
                        continue
                    prev_cost, prev_path = C.get((prev_bits, m), (float('inf'), []))
                    if not prev_path:
                        continue
                    # Check for intersections
                    new_seg = (points[m], points[k])
                    intersects = False
                    for i in range(len(prev_path) - 2):
                        a, b = prev_path[i], prev_path[i + 1]
                        if _segments_intersect(points[a], points[b], new_seg[0], new_seg[1]):
                            intersects = True
                            break
                    if intersects:
                        continue
                    angle_cost = _calculate_angle_deviation(points[prev_path[-2]], points[m], points[k]) if len(prev_path) > 1 else 0
                    cost = prev_cost + np.linalg.norm(points[m] - points[k]) + angle_cost
                    res.append((cost, prev_path + [k]))
                if res:
                    C[(bits, k)] = min(res)

    bits = (1 << n) - 2
    res = []
    for k in range(1, n):
        if (bits, k) not in C:
            continue
        cost, path = C[(bits, k)]
        new_seg = (points[k], points[0])
        intersects = False
        for i in range(len(path) - 2):
            a, b = path[i], path[i + 1]
            if _segments_intersect(points[a], points[b], new_seg[0], new_seg[1]):
                intersects = True
                break
        if intersects:
            continue
        angle_cost = _calculate_angle_deviation(points[path[-2]], points[k], points[0])
        final_cost = cost + np.linalg.norm(points[k] - points[0]) + angle_cost
        res.append((final_cost, path + [0]))

    return min(res) if res else (float('inf'), [])

def _detect_endpoints(points_2d: np.ndarray, min_distance: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Detect the endpoints of a road by finding pairs of points that are furthest from each other.
    Points that are closer than min_distance to each other are considered duplicates and only one is kept.
    
    Args:
        points_2d: Array of 2D points (N x 2)
        min_distance: Minimum distance between points to consider them distinct
        
    Returns:
        List of indices for the endpoints, alternating between pairs
        (first point of pair 1, first point of pair 2, second point of pair 1, second point of pair 2)
    """
    # First, filter out points that are too close together
    kept_indices = []
    used_points = set()
    
    for i in range(len(points_2d)):
        if i in used_points:
            continue
            
        # Find all points close to this one
        distances = np.linalg.norm(points_2d - points_2d[i], axis=1)
        close_points = np.where(distances < min_distance)[0]
        
        # Mark all close points as used and keep only the current one
        used_points.update(close_points)
        kept_indices.append(i)
    
    # Use only the filtered points for endpoint detection
    filtered_points = points_2d[kept_indices]
    
    # Calculate pairwise distances between filtered points
    distances = np.linalg.norm(filtered_points[:, np.newaxis] - filtered_points, axis=2)
    
    # Find the first pair of points (maximally distant)
    i1, j1 = np.unravel_index(np.argmax(distances), distances.shape)
    
    # Mask out the first pair to find second pair
    distances_masked = distances.copy()
    distances_masked[i1, :] = -np.inf
    distances_masked[:, i1] = -np.inf
    distances_masked[j1, :] = -np.inf
    distances_masked[:, j1] = -np.inf
    
    # Find the second pair of points
    i2, j2 = np.unravel_index(np.argmax(distances_masked), distances_masked.shape)
    
    # Map back to original indices
    original_indices = [kept_indices[i] for i in [i1, i2, j1, j2]]
    
    # Return indices in alternating order
    return original_indices

def _signed_distance_to_curve(point: np.ndarray, curve_fit: np.poly1d, 
                              x_range: Tuple[float, float]) -> Tuple[float, np.ndarray]:
    """Calculate signed perpendicular distance from point to curve.
    Positive distance means point is on one side, negative on the other.
    
    Args:
        point: Point to calculate distance to
        curve_fit: Polynomial fit to the curve
        x_range: Range of x-values for the curve
        
    Returns:
        Tuple[float, np.ndarray]: Signed distance and closest point on curve
    """
    # Generate points along the curve
    curve_x = np.linspace(x_range[0], x_range[1], 1000)
    curve_y = curve_fit(curve_x)
    curve_points = np.column_stack((curve_x, curve_y))
    
    # Find closest point on curve
    distances = np.linalg.norm(curve_points - point, axis=1)
    closest_idx = np.argmin(distances)
    closest_point = curve_points[closest_idx]
    
    # Get tangent vector at closest point
    if closest_idx < len(curve_x) - 1:
        tangent = curve_points[closest_idx + 1] - curve_points[closest_idx]
    else:
        tangent = curve_points[closest_idx] - curve_points[closest_idx - 1]
    tangent = tangent / np.linalg.norm(tangent)
    
    # Get normal vector (rotate tangent 90 degrees counterclockwise)
    normal = np.array([-tangent[1], tangent[0]])
    
    # Calculate signed distance
    vec_to_point = point - closest_point
    signed_dist = np.dot(vec_to_point, normal)
    
    return signed_dist, closest_point

def _trim_points_protected(points: np.ndarray, protected_indices: List[int], 
                           max_points: int = 14) -> List[int]:
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
    
    assert max_points >= len(protected_indices), "max_points must be >= number of protected points"
    assert len(points) >= len(protected_indices), "len(points) must be >= max_points"
    
    # Fit initial curve through all points
    x = points[:, 0]
    y = points[:, 1]
    z = np.polyfit(x, y, 3)
    curve_fit = np.poly1d(z)
    x_range = (x.min(), x.max())
    
    # Calculate signed distances for all points
    distances_and_closest = [_signed_distance_to_curve(points[i], curve_fit, x_range) 
                             for i in range(len(points))]
    distances = np.array([d for d, _ in distances_and_closest])
    
    # Generate reference points at 1/4, 2/4 and 3/4 along the curve
    ref_positions = [0.25, 0.5, 0.75]  # 1/4, 2/4, 3/4
    x_refs = x_range[0] + (x_range[1] - x_range[0]) * np.array(ref_positions)
    ref_points = np.column_stack((x_refs, curve_fit(x_refs)))
    
    # Start with protected points
    kept_indices = set(protected_indices)
    
    for ref_point in ref_points:
        # Calculate distances to this reference point
        dists_to_ref = np.linalg.norm(points - ref_point, axis=1)
        
        # Split points into above and below curve
        above_curve = distances > 0
        below_curve = distances < 0
        
        # Find closest non-protected points above and below
        above_indices = [i for i in range(len(points)) 
                        if above_curve[i] and i not in protected_indices]
        below_indices = [i for i in range(len(points)) 
                        if below_curve[i] and i not in protected_indices]
        
        # Sort by distance to reference point
        above_indices = sorted(above_indices, key=lambda i: dists_to_ref[i])
        below_indices = sorted(below_indices, key=lambda i: dists_to_ref[i])
        
        # Take exactly one point from above and one from below (that aren't already kept)
        for idx in above_indices:
            if idx not in kept_indices:
                kept_indices.add(idx)
                break
        
        for idx in below_indices:
            if idx not in kept_indices:
                kept_indices.add(idx)
                break
    
    return sorted(list(kept_indices))

def _compress_path(points: np.ndarray, path: List[int], angle_threshold: float = 1.0) -> List[int]:
    """Compress a path by removing points that are nearly collinear with their neighbors.
    
    Args:
        points: Array of point coordinates (N x 2)
        path: List of indices forming the path
        angle_threshold: Minimum angle deviation (in degrees) to keep a point
        
    Returns:
        List of indices forming the compressed path
    """
    if len(path) <= 3:  # Can't compress paths with 3 or fewer points
        return path
        
    # We'll build the compressed path starting with the first point
    compressed = [path[0]]
    
    # Iterate through interior points (skip first and last)
    for i in range(1, len(path)-1):
        # Get the previous, current, and next points
        prev_idx = compressed[-1]  # Last point in compressed path
        curr_idx = path[i]        # Current point we're considering
        next_idx = path[i+1]      # Next point in original path
        
        # Calculate angle at current point
        angle = _calculate_angle_deviation(
            points[prev_idx],
            points[curr_idx],
            points[next_idx]
        )
        
        # If angle is significant (> threshold), keep the point
        if angle > angle_threshold:
            compressed.append(curr_idx)
    
    # Always add the last point to close the loop
    compressed.append(path[-1])
    
    return compressed

def _get_2d_face(vertices: np.ndarray, z_tolerance: float = 0.1, max_points: int = 10,
                 compress: bool = True, angle_threshold: float = 1.0) -> List[Tuple[float, float, float]]:
    """Generate a 2D face from a set of vertices.
    
    Args:
        vertices: Array of vertex coordinates (shape: N x 3)
        z_tolerance: Tolerance for z-coordinate variation - targetted for roads
        max_points: Maximum number of points to consider
        angle_threshold: Angle threshold for collinearity
        
    Returns:
        List of (x,y,z) vertex coordinates for the face
    """
    # Ensure vertices are 2D (simple test - if z-coordinates are within tolerance)
    if not np.allclose(vertices[:, 2], vertices[0, 2], atol=z_tolerance):
        raise ValueError("Vertices are not 2D")
    
    # Detect endpoints  
    endpoints = _detect_endpoints(vertices[:, :2])
    
    # Filter points and convert to 2D (by discarding z-coordinate)
    kept_indices = _trim_points_protected(vertices[:, :2], 
                                         protected_indices=endpoints, 
                                         max_points=max_points)
    points_filtered = vertices[kept_indices]

    _, best_path = _tsp_held_karp_no_intersections(points_filtered[:, :2])

    if compress:
        compressed_path = _compress_path(points_filtered, best_path, 
                                        angle_threshold=angle_threshold)
        final_points = points_filtered[compressed_path[:-1]]
    else:
        final_points = points_filtered[best_path[:-1]]
    
    return [final_points]  # Return as list of faces (single face)

def get_object_faces(vertices: List[Tuple[float, float, float]], fast: bool = True) -> List[List[Tuple[float, float, float]]]:
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
        fast: Whether to use fast mode (default: True)
        
    Returns:
        List of faces, where each face is a list of (x,y,z) vertex coordinates
    """
    vertices = np.array(vertices)
    if len(vertices) < 3:
        return None
    
    if fast:
        faces = _get_faces_convex_hull(vertices)
    else:
        faces = _get_2d_face(vertices)
    
    return faces

if __name__ == "__main__":
    # Test the functions
    points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    path = [0, 1, 2, 3]
    compressed = _compress_path(points, path)
    print(compressed)

    # Plot helper
    def plot_points(points, path=None, title=""):
        plt.figure(figsize=(8, 6))
        plt.scatter(points[:, 0], points[:, 1], color='blue')
        for i, (x, y) in enumerate(points):
            plt.text(x + 1, y + 1, str(i), fontsize=9)
        if path:
            for i in range(len(path) - 1):
                p1, p2 = points[path[i]], points[path[i+1]]
                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-')
        plt.title(title)
        plt.axis('equal')
        plt.grid(True)
        plt.show()
