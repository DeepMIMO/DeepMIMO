"""Sionna Ray Tracing Exporter.

This module provides functionality to export Sionna ray tracing data.
Sionna does not provide sufficient built-in tools for saving ray tracing results to
disk, so this module handles exporting Paths and Scene objects into serialisable
dictionary formats. This allows ray tracing results to be saved and reused without
re-running computationally expensive simulations.

Requires Sionna RT 2.0+. Import explicitly — DeepMIMO does not require sionna:

    from deepmimo.exporters import sionna_exporter
    sionna_exporter(scene, path_list, my_compute_path_params, save_folder)

"""

import os
from pathlib import Path
from typing import Any

import numpy as np

from deepmimo.utils import save_pickle

try:
    import sionna.rt

    from deepmimo.pipelines.sionna_rt.sionna_utils import get_sionna_version

    Paths = sionna.rt.Paths
    Scene = sionna.rt.Scene
except ImportError:
    msg = (
        "Sionna ray tracing functionality requires additional dependencies. "
        "Please install them using: pip install 'deepmimo[sionna]'"
    )
    raise ImportError(msg) from None

COORD_DIM = 3


def _get_scene_objects(scene: Scene) -> dict[str, Any]:
    """Return scene objects dictionary, supporting legacy private attribute."""
    return getattr(scene, "scene_objects", getattr(scene, "_scene_objects", {}))


def _paths_to_dict(paths: Paths) -> dict[str, Any]:
    """Export paths to a filtered dictionary with only selected keys."""
    members_names = dir(paths)
    members_objects = [getattr(paths, attr) for attr in members_names]
    return {
        attr_name: attr_obj
        for (attr_obj, attr_name) in zip(members_objects, members_names, strict=False)
        if not callable(attr_obj)
        and not isinstance(attr_obj, Scene)
        and not attr_name.startswith("__")
        and not attr_name.startswith("_")
    }


def export_paths(path_list: Paths | list[Paths]) -> list[dict[str, Any]]:
    """Export Sionna paths to filtered dictionaries with the relevant fields.

    Args:
        path_list (Paths | list[Paths]): Paths object (single TX) or list of Paths (multi-TX).

    Returns:
        list[dict[str, Any]]: Paths converted to dictionaries with only the keys we need.

    Notes:
        Shared fields: tau, phi_r, phi_t, theta_r, theta_t, sources, targets, vertices,
        rx_array, tx_array.

        ``a`` is exported as a complex array from the (real, imag) tensor pair.
        ``interactions`` holds per-depth interaction bitflags (Sionna 2.0).
        targets/sources are transposed to (N, 3) on export.

    """
    relevant_keys = [
        "sources",
        "targets",
        "tau",
        "phi_r",
        "phi_t",
        "theta_r",
        "theta_t",
        "vertices",
        "interactions",
    ]

    path_list = [path_list] if not isinstance(path_list, list) else path_list
    paths_dict_list = []
    for path_obj in path_list:
        path_dict = _paths_to_dict(path_obj)
        dict_filtered = {key: path_dict[key].numpy() for key in relevant_keys}

        # a is a (real, imag) tensor pair in Sionna 1.x/2.0
        dict_filtered["a"] = path_dict["a"][0].numpy() + 1j * path_dict["a"][1].numpy()

        # Transpose targets and sources to (N, 3)
        for key in ["targets", "sources"]:
            dict_filtered[key] = path_dict[key].numpy().T

        paths_dict_list += [dict_filtered]
    return paths_dict_list


def export_scene_materials(scene: Scene) -> tuple[list[dict[str, Any]], list[int]]:
    """Export materials in a Sionna Scene.

    Args:
        scene (Scene): Sionna Scene object.

    Returns:
        tuple[list[dict[str, Any]], list[int]]: Material property dictionaries and the material
        index per object.

    """
    scene_objects = _get_scene_objects(scene)
    obj_materials = []
    for obj in scene_objects.values():
        obj_materials += [obj.radio_material]
    unique_materials = set(obj_materials)
    unique_mat_names = [mat.name for mat in unique_materials]
    n_objs = len(scene_objects)
    obj_mat_indices = np.zeros(n_objs, dtype=int)
    for obj_idx, obj_mat in enumerate(obj_materials):
        obj_mat_indices[obj_idx] = unique_mat_names.index(obj_mat.name)
    materials_dict_list = []
    for material in unique_materials:
        alpha_r = getattr(material.scattering_pattern, "alpha_r", None)
        alpha_i = getattr(material.scattering_pattern, "alpha_i", None)
        lambda_ = (
            material.scattering_pattern.lambda_.numpy()
            if hasattr(material.scattering_pattern, "lambda_")
            else None
        )
        materials_dict = {
            "name": material.name,
            "conductivity": material.conductivity.numpy(),
            "relative_permeability": 1.0,  # removed in Sionna RT >=1.0
            "relative_permittivity": material.relative_permittivity.numpy(),
            "scattering_coefficient": material.scattering_coefficient.numpy(),
            "scattering_pattern": type(material.scattering_pattern).__name__,
            "alpha_r": alpha_r,
            "alpha_i": alpha_i,
            "lambda_": lambda_,
            "xpd_coefficient": material.xpd_coefficient.numpy(),
        }
        materials_dict_list += [materials_dict]
    return materials_dict_list, obj_mat_indices


def _scene_to_dict(scene: Scene) -> dict[str, Any]:
    """Export a Sionna Scene to a dictionary."""
    members_names = dir(scene)
    bug_attrs = ["paths_solver"]
    members_objects = [getattr(scene, attr) for attr in members_names if attr not in bug_attrs]
    return {
        attr_name[1:]: attr_obj
        for (attr_obj, attr_name) in zip(members_objects, members_names, strict=False)
        if not callable(attr_obj)
        and not isinstance(attr_obj, sionna.rt.Scene)
        and not attr_name.startswith("__")
    }


def export_scene_rt_params(scene: Scene, **compute_paths_kwargs: Any) -> dict[str, Any]:
    """Extract ray-tracing parameters from a Scene and ``PathSolver`` call arguments.

    Args:
        scene (Scene): Sionna Scene object.
        **compute_paths_kwargs: Keyword arguments passed to ``PathSolver.__call__``.

    Returns:
        dict[str, Any]: Consolidated ray-tracing parameters.

    """
    scene_dict = _scene_to_dict(scene)

    rx_array = scene_dict["rx_array"]
    tx_array = scene_dict["tx_array"]
    wavelength = scene.wavelength
    rx_array_ant_pos = rx_array.positions(wavelength).numpy()
    tx_array_ant_pos = tx_array.positions(wavelength).numpy()

    synthetic_array = scene_dict.get(
        "synthetic_array",
        compute_paths_kwargs.get("synthetic_array", False),
    )

    rt_params_dict = {
        "bandwidth": scene_dict["bandwidth"].numpy(),
        "frequency": scene_dict["frequency"].numpy(),
        "rx_array_size": rx_array.array_size,
        "rx_array_num_ant": rx_array.num_ant,
        "rx_array_ant_pos": rx_array_ant_pos,
        "tx_array_size": tx_array.array_size,
        "tx_array_num_ant": tx_array.num_ant,
        "tx_array_ant_pos": tx_array_ant_pos,
        "synthetic_array": synthetic_array,
        "raytracer_version": get_sionna_version(),
    }

    default_compute_paths_params = {
        "max_depth": 3,
        "max_num_paths_per_src": 1000000,
        "samples_per_src": 1000000,
        "synthetic_array": True,
        "los": True,
        "specular_reflection": True,
        "diffuse_reflection": False,
        "refraction": True,
        "seed": 42,
    }
    default_compute_paths_params.update(compute_paths_kwargs)
    raw_params = {**rt_params_dict, **default_compute_paths_params}

    # Aliases used by the converter's sionna_rt_params reader
    aliases = {
        "num_samples": raw_params["samples_per_src"],
        "reflection": bool(raw_params["specular_reflection"]),
        "diffraction": False,
        "scattering": bool(raw_params["diffuse_reflection"]),
    }

    return {**raw_params, **aliases}


def export_scene_buildings(scene: Scene) -> tuple[np.ndarray, dict]:
    """Export building geometry from a Sionna Scene.

    Args:
        scene (Scene): Sionna Scene object.

    Returns:
        tuple[np.ndarray, dict]: ``vertice_matrix`` (n_vertices x 3) and ``obj_index_map``
        mapping object name to (start_idx, end_idx).

    """
    all_vertices = []
    obj_index_map = {}

    vertex_offset = 0

    scene_objects = _get_scene_objects(scene)
    for obj_name, obj in scene_objects.items():
        shape = obj._mi_shape  # noqa: SLF001
        n_v = shape.vertex_count()
        obj_vertices = np.array(shape.vertex_position(np.arange(n_v))).T

        if obj_vertices.size == 0:
            continue
        if obj_vertices.ndim == 1:
            obj_vertices = obj_vertices.reshape(1, -1)
        if obj_vertices.shape[1] > COORD_DIM:
            obj_vertices = obj_vertices[:, :COORD_DIM]
        if obj_vertices.shape[1] < COORD_DIM:
            pad_width = COORD_DIM - obj_vertices.shape[1]
            obj_vertices = np.pad(obj_vertices, ((0, 0), (0, pad_width)), "constant")

        all_vertices.append(obj_vertices)
        obj_index_map[obj_name] = (vertex_offset, vertex_offset + obj_vertices.shape[0])
        vertex_offset += obj_vertices.shape[0]

    vertice_matrix = np.zeros((0, 3)) if len(all_vertices) == 0 else np.vstack(all_vertices)

    return vertice_matrix, obj_index_map


def sionna_exporter(
    scene: Scene,
    path_list: list[Paths] | Paths,
    my_compute_path_params: dict,
    save_folder: str,
) -> None:
    """Export a complete Sionna simulation to a format that can be converted by DeepMIMO.

    This function exports all necessary data from a Sionna ray tracing simulation to files
    that can be converted into the DeepMIMO format. The exported data includes:
    - Ray paths and their properties
    - Scene materials and their properties
    - Ray tracing parameters used in the simulation
    - Scene geometry (vertices and objects)

    Args:
        scene (Scene): Sionna Scene containing the simulation environment.
        path_list (list[Paths] | Paths): Ray paths from Sionna's ray tracer (single or multi TX).
        my_compute_path_params (dict): Parameters passed to ``PathSolver.__call__`` (Sionna does
            not persist them internally).
        save_folder (str): Directory to write the exported files.

    """
    paths_dict_list = path_list if isinstance(path_list[0], dict) else export_paths(path_list)
    materials_dict_list, material_indices = export_scene_materials(scene)
    rt_params = export_scene_rt_params(scene, **my_compute_path_params)
    vertice_matrix, obj_index_map = export_scene_buildings(scene)

    os.makedirs(save_folder, exist_ok=True)  # noqa: PTH103

    save_vars_dict = {
        "sionna_paths.pkl": paths_dict_list,
        "sionna_materials.pkl": materials_dict_list,
        "sionna_material_indices.pkl": material_indices,
        "sionna_rt_params.pkl": rt_params,
        "sionna_vertices.pkl": vertice_matrix,
        "sionna_objects.pkl": obj_index_map,
    }

    for filename, variable in save_vars_dict.items():
        save_pickle(variable, str(Path(save_folder) / filename))


__all__ = ["sionna_exporter"]
