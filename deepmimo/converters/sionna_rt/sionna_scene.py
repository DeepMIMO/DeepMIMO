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

# (vertices, material_idx, label, name, use_fast)
_Component = tuple[np.ndarray, int, str, str, bool]


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


def _aabb_overlap_ratio(
    mn_i: np.ndarray,
    mx_i: np.ndarray,
    mn_j: np.ndarray,
    mx_j: np.ndarray,
) -> float:
    """Return intersection volume / min(vol_i, vol_j), or 0 if no overlap."""
    vol_i = float(np.prod(np.maximum(mx_i - mn_i, 0.0)))
    vol_j = float(np.prod(np.maximum(mx_j - mn_j, 0.0)))
    int_vol = float(np.prod(np.maximum(np.minimum(mx_i, mx_j) - np.maximum(mn_i, mn_j), 0.0)))
    min_vol = min(vol_i, vol_j)
    return int_vol / min_vol if (int_vol > 0.0 and min_vol > 0.0) else 0.0


def _uf_find(parent: list[int], x: int) -> int:
    """Union-find root lookup with path compression (modifies parent in place)."""
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _build_clusters(
    all_components: list[_Component],
    *,
    deduplicate: bool,
    overlap_threshold: float,
) -> list[list[int]]:
    """Return index clusters for ``all_components``, merging if ``deduplicate``."""
    if not deduplicate:
        return [[i] for i in range(len(all_components))]
    building_idx = [i for i, c in enumerate(all_components) if c[2] == CAT_BUILDINGS]
    other_idx = [i for i, c in enumerate(all_components) if c[2] != CAT_BUILDINGS]
    local_clusters = _cluster_by_aabb([all_components[i] for i in building_idx], overlap_threshold)
    clusters: list[list[int]] = [[building_idx[k] for k in cl] for cl in local_clusters]
    clusters.extend([[i] for i in other_idx])
    return clusters


def _cluster_by_aabb(
    components: list[_Component],
    overlap_threshold: float = 0.8,
) -> list[list[int]]:
    """Group component indices whose AABBs substantially overlap.

    Two components are considered the same physical object when the smaller
    component's AABB volume is at least ``overlap_threshold`` contained within
    the larger component's AABB.  This is the typical relationship between a
    building's dominant material mesh (e.g. concrete walls) and its embedded
    detail meshes (e.g. glass windows).

    Only components sharing the same scene label are considered for merging.

    Args:
        components: Sequence of ``(vertices, material_idx, label, name, use_fast)``
            tuples.
        overlap_threshold: Fraction of the smaller AABB volume that must lie
            inside the larger AABB to trigger a merge.  Default 0.8.

    Returns:
        List of index lists, one per cluster.

    """
    n = len(components)
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    aabb_mins: list[np.ndarray] = []
    aabb_maxs: list[np.ndarray] = []
    for verts, *_ in components:
        aabb_mins.append(verts.min(axis=0))
        aabb_maxs.append(verts.max(axis=0))

    parent = list(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if components[j][2] != components[i][2]:
                continue
            ratio = _aabb_overlap_ratio(aabb_mins[i], aabb_maxs[i], aabb_mins[j], aabb_maxs[j])
            if ratio >= overlap_threshold:
                parent[_uf_find(parent, i)] = _uf_find(parent, j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(_uf_find(parent, i), []).append(i)
    return list(clusters.values())


def read_scene(
    load_folder: str,
    material_indices: list[int],
    *,
    deduplicate: bool = True,
    overlap_threshold: float = 0.8,
) -> Scene:
    """Load scene data from Sionna format.

    Converts Sionna's triangular mesh representation into DeepMIMO's scene format.
    When ``sionna_faces.pkl`` is present, merged city meshes are decomposed into
    individual connected components before the per-building convex hull is computed —
    preventing a single export object that spans the whole city from collapsing into
    one giant bounding box.

    When Sionna loads a scene with the default ``merge_shapes=True``, all mesh
    objects sharing a material are fused into one mesh.  A building with both
    concrete walls and glass windows therefore produces two separate connected
    components — one per material — whose convex hulls overlap in the DeepMIMO
    scene view.  ``deduplicate=True`` (the default) clusters those overlapping
    building components into a single DeepMIMO object using AABB containment,
    so each physical building appears exactly once.

    Args:
        load_folder: Path to folder containing Sionna scene files.
        material_indices: List of material indices, one per object.
        deduplicate: Merge building components whose bounding boxes substantially
            overlap.  Resolves the visual duplication that arises from Sionna's
            default ``merge_shapes=True`` scene loading.  Set to ``False`` to
            keep the original one-object-per-component behaviour (useful for
            debugging or for scenes where the heuristic groups objects
            incorrectly).
        overlap_threshold: Fraction of the smaller component's AABB volume that
            must lie within the larger component's AABB to be considered the
            same physical building.  Only used when ``deduplicate=True``.
            Default 0.8.

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

    terrain_keywords = ["plane", "floor", "terrain", "roads", "paths", "ground"]

    # --- Phase 1: split every material mesh into connected components ---
    all_components: list[_Component] = []
    for mat_idx_pos, (name, vertex_range) in enumerate(obj_items):
        try:
            start_idx, end_idx = vertex_range
            is_floor = any(word in name.lower() for word in terrain_keywords)
            obj_label = CAT_TERRAIN if is_floor else CAT_BUILDINGS
            material_idx = material_indices[mat_idx_pos]

            obj_vertices = vertices[start_idx:end_idx]
            obj_faces = obj_faces_map[mat_idx_pos] - start_idx

            components = _split_into_components(obj_vertices, obj_faces)
            n_comps = len(components)

            for comp_idx, comp_verts in enumerate(components):
                comp_name = name if n_comps == 1 else f"{name}_{comp_idx}"
                use_fast = "road" not in comp_name.lower()
                all_components.append((comp_verts, material_idx, obj_label, comp_name, use_fast))

        except Exception as e:
            print(f"Error processing object {name}: {e!s}")
            raise

    # --- Phase 2: optionally cluster overlapping building components ---
    clusters = _build_clusters(
        all_components, deduplicate=deduplicate, overlap_threshold=overlap_threshold
    )

    # --- Phase 3: one DeepMIMO object per cluster ---
    scene = Scene()
    obj_counter = 0
    for cluster in clusters:
        merged_verts = np.vstack([all_components[k][0] for k in cluster])
        _, material_idx, obj_label, comp_name, _ = all_components[cluster[0]]
        use_fast = "road" not in comp_name.lower()

        generated_faces = get_object_faces(merged_verts, fast=use_fast)
        if generated_faces is None:
            continue

        object_faces = [Face(vertices=fv, material_idx=material_idx) for fv in generated_faces]
        obj = PhysicalElement(
            faces=object_faces,
            object_id=obj_counter,
            label=obj_label,
            name=comp_name,
        )
        scene.add_object(obj)
        obj_counter += 1

    return scene
