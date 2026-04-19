"""Sionna Ray Tracing Scene Module.

This module handles loading and converting scene data from Sionna's format to DeepMIMO's format.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from deepmimo.core.scene import (
    CAT_BUILDINGS,
    CAT_TERRAIN,
    Face,
    PhysicalElement,
    Scene,
    get_object_faces,
)
from deepmimo.utils import load_pickle


def _split_into_components(
    obj_vertices: np.ndarray,
    obj_faces: np.ndarray,
) -> list[np.ndarray]:
    """Split a mesh into connected vertex groups via scipy graph connectivity.

    Args:
        obj_vertices: (N, 3) vertex array for this object.
        obj_faces: (F, 3) face indices local to obj_vertices. Empty → single component.

    Returns:
        List of vertex arrays, one per connected component.

    """
    n_verts = len(obj_vertices)
    if n_verts == 0:
        return []
    if len(obj_faces) == 0:
        return [obj_vertices]

    rows = np.concatenate([obj_faces[:, 0], obj_faces[:, 1], obj_faces[:, 2]])
    cols = np.concatenate([obj_faces[:, 1], obj_faces[:, 2], obj_faces[:, 0]])
    graph = coo_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, cols)),
        shape=(n_verts, n_verts),
    )
    _, labels = connected_components(graph, directed=False)

    return [obj_vertices[labels == lbl] for lbl in np.unique(labels)]


def read_scene(load_folder: str, material_indices: list[int]) -> Scene:
    """Load scene data from Sionna format.

    Converts Sionna's triangular mesh representation into DeepMIMO's scene format.
    When ``sionna_faces.pkl`` is present, merged city meshes are decomposed into
    individual connected components before the per-building convex hull is computed —
    preventing a single export object that spans the whole city from collapsing into
    one giant bounding box.

    Args:
        load_folder: Path to folder containing Sionna scene files.
        material_indices: List of material indices, one per object.

    Returns:
        Scene: Loaded scene with all objects.

    """
    vertices = load_pickle(str(Path(load_folder) / "sionna_vertices.pkl"))
    objects = load_pickle(str(Path(load_folder) / "sionna_objects.pkl"))

    try:
        all_faces = load_pickle(str(Path(load_folder) / "sionna_faces.pkl"))
    except FileNotFoundError:
        all_faces = np.zeros((0, 3), dtype=np.int64)

    # Precompute per-object face groups in one O(F) pass.
    # The exporter writes faces per-object in vertex-offset order, so face[:,0]
    # falls within [start_idx, end_idx) for the object that owns it.
    obj_items = list(objects.items())
    if len(all_faces) > 0:
        end_indices = np.array([vr[1] for _, vr in obj_items])
        face_obj_ids = np.searchsorted(end_indices, all_faces[:, 0], side="right")
        obj_faces_map = [all_faces[face_obj_ids == i] for i in range(len(obj_items))]
    else:
        obj_faces_map = [np.zeros((0, 3), dtype=np.int64)] * len(obj_items)

    scene = Scene()
    terrain_keywords = ["plane", "floor", "terrain", "roads", "paths", "ground"]
    obj_counter = 0

    for mat_idx_pos, (name, vertex_range) in enumerate(obj_items):
        try:
            start_idx, end_idx = vertex_range
            is_floor = any(word in name.lower() for word in terrain_keywords)
            obj_label = CAT_TERRAIN if is_floor else CAT_BUILDINGS
            material_idx = material_indices[mat_idx_pos]

            obj_vertices = vertices[start_idx:end_idx]
            obj_faces = obj_faces_map[mat_idx_pos] - start_idx

            components = _split_into_components(obj_vertices, obj_faces)
            n_components = len(components)

            for comp_idx, comp_verts in enumerate(components):
                comp_name = name if n_components == 1 else f"{name}_{comp_idx}"
                use_fast = "road" not in comp_name.lower()
                generated_faces = get_object_faces(comp_verts, fast=use_fast)
                if generated_faces is None:
                    continue

                object_faces = [
                    Face(vertices=fv, material_idx=material_idx) for fv in generated_faces
                ]
                obj = PhysicalElement(
                    faces=object_faces,
                    object_id=obj_counter,
                    label=obj_label,
                    name=comp_name,
                )
                scene.add_object(obj)
                obj_counter += 1

        except Exception as e:
            print(f"Error processing object {name}: {e!s}")
            raise

    return scene
