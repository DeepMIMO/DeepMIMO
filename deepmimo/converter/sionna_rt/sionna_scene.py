"""
Sionna Ray Tracing Scene Module.

This module handles loading and converting scene data from Sionna's format to DeepMIMO's format.
"""

import os
from typing import List

from ...general_utils import load_pickle

from ...scene import (
    PhysicalElement, 
    Face, 
    Scene,
    CAT_BUILDINGS,
    CAT_TERRAIN,
    get_object_faces
)

def read_scene(load_folder: str, material_indices: List[int]) -> Scene:
    """Load scene data from Sionna format.
    
    This function converts Sionna's triangular mesh representation into DeepMIMO's
    scene format. While we receive the scene as triangular faces, we store it using
    convex hull faces for efficiency. The Face class in DeepMIMO can handle both
    representations:
    1. Convex hull faces (more efficient for storage and most operations)
    2. Triangular faces (available when needed for detailed visualization)
    
    Args:
        load_folder: Path to folder containing Sionna scene files
        material_indices: List of material indices, one per object
        
    Returns:
        Scene: Loaded scene with all objects
    """
    # Load raw data - already in correct format
    vertices = load_pickle(os.path.join(load_folder, 'sionna_vertices.pkl')) # (N_VERTICES, 3) 
    objects = load_pickle(os.path.join(load_folder, 'sionna_objects.pkl')) # Dict with vertex index ranges
    
    # Create scene
    scene = Scene()

    terrain_keywords = ['plane', 'floor', 'terrain', 'roads', 'paths']
    
    # Process each object
    for id_counter, (name, vertex_range) in enumerate(objects.items()):
        try:
            # Get vertex range for this object
            start_idx, end_idx = vertex_range
            
            # if 'roads' in name or 'paths' in name or 'terrain' in name:
            #     print(f'Skipping paths/terrain/road {name}')
            #     continue

            # Attribute the correct label to the object
            is_floor = any(word in name.lower() for word in terrain_keywords)
            obj_label = CAT_TERRAIN if is_floor else CAT_BUILDINGS
            
            # Get material index for this object
            material_idx = material_indices[id_counter]
            
            # Get vertices for this object
            object_vertices = []
            for i in range(start_idx, end_idx):
                vertex = vertices[i]
                vertex_tuple = (float(vertex[0]), float(vertex[1]), float(vertex[2]))
                object_vertices.append(vertex_tuple)
            
            # Generate faces using convex hull approach
            use_fast_mode = 'road' not in name.lower()
            generated_faces = get_object_faces(object_vertices, fast=use_fast_mode)
            
            # Create Face objects with material indices
            object_faces = []
            for face_vertices in generated_faces:
                face = Face(
                    vertices=face_vertices,
                    material_idx=material_idx
                )
                object_faces.append(face)
            
            # Create object
            obj = PhysicalElement(
                faces=object_faces,
                object_id=id_counter,
                label=obj_label,
                name=name
            )
            scene.add_object(obj)
            
        except Exception as e:
            print(f"Error processing object {name}: {str(e)}")
            raise

    return scene 