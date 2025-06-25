"""Sionna Ray Tracing Exporter.

This module provides functionality to export Sionna ray tracing data. 
This is necessary because Sionna (as of v0.19.1) does not provide sufficient built-in
tools for saving ray tracing results to disk.

The module handles exporting Paths and Scene objects from Sionna's ray tracer
into dictionary formats that can be serialized. This allows ray tracing
results to be saved and reused without re-running computationally expensive
ray tracing simulations.

This has been tested with Sionna v0.19.1 and may work with earlier versions.

DeepMIMO does not require sionna to be installed.
To keep it this way AND use this module, you need to import it explicitly:

# Import the module:
from deepmimo.exporters import sionna_exporter

sionna_exporter(scene, path_list, my_compute_path_params, save_folder)

"""

import os
import numpy as np
from typing import Tuple, List, Dict, Any

from ..general_utils import save_pickle

try:
    import sionna.rt
    from ...pipelines.sionna_rt.sionna_utils import get_sionna_version, is_sionna_v1
    Paths = sionna.rt.Paths
    Scene = sionna.rt.Scene
except ImportError:
    raise ImportError(
        "Sionna ray tracing functionality requires additional dependencies. "
        "Please install them using: pip install 'deepmimo[sionna1]' or 'deepmimo[sionna019]'"
    )

def _paths_to_dict(paths: Paths) -> List[dict]:
    """Exports paths to a filtered dictionary with only selected keys """
    members_names = dir(paths)
    members_objects = [getattr(paths, attr) for attr in members_names]
    data = {attr_name : attr_obj for (attr_obj, attr_name)
            in zip(members_objects,members_names)
            if not callable(attr_obj) and
                not isinstance(attr_obj, Scene) and
                not attr_name.startswith("__") and
                not attr_name.startswith("_")}
    return data

def export_paths(path_list):
    """Export paths to a filtered dictionary with only selected keys.
    
    Note:
    - in both versions:
        - 'tau' is a float array
        - 'phi_r' and 'phi_t' are float arrays
        - 'theta_r' and 'theta_t' are float arrays
        - 'sources' and 'targets' are lists of positions
        - 'vertices' is a list of vertices
        - 'rx_array' and 'tx_array' are arrays of positions
    - Sionna 0.x:
        - 'a' is a complex array
        - 'types' is are the types of each interaction
    - Sionna 1.x:
        - 'a' is a tuple of real and imaginary parts
        - 'interactions' is are the types of each interaction
        
    Args:
        paths (Paths): Sionna Paths object

    Returns:
        List[dict]: List of dictionaries with only selected keys
    """

    sionna_v1 = is_sionna_v1()
    relevant_keys = ['sources', 'targets', 'tau', 'phi_r', 'phi_t', 
                     'theta_r', 'theta_t', 'vertices']
    relevant_keys += ['interactions'] if sionna_v1 else ['types']

    path_list = [path_list] if type(path_list) != list else path_list
    paths_dict_list = []
    for path_obj in path_list:
        path_dict = _paths_to_dict(path_obj)
        dict_filtered = {key: path_dict[key].numpy() for key in relevant_keys}

        # Process a (complex) array independently
        if sionna_v1:
            dict_filtered['a'] = path_dict['a'][0].numpy() + 1j * path_dict['a'][1].numpy()
        else:
            dict_filtered['a'] = path_dict['a'].numpy()

        # Transpose targets and sources for Sionna 1.x
        if sionna_v1:
            for key in ['targets', 'sources']:
                dict_filtered[key] = path_dict[key].numpy().T

        paths_dict_list += [dict_filtered]
    return paths_dict_list

def export_scene_materials(scene: Scene) -> Tuple[List[Dict[str, Any]], List[int]]:
    """ Export the materials in a Sionna Scene to a list of dictionaries.
    
    Args:
        scene (Scene): Sionna Scene object

    Returns:
        Tuple[List[Dict[str, Any]], List[int]]: List of dictionaries with material properties and indices
    """
    obj_materials = []
    for _, obj in scene._scene_objects.items():
        obj_materials += [obj.radio_material]
    unique_materials = set(obj_materials)
    unique_mat_names = [mat.name for mat in unique_materials]
    n_objs = len(scene._scene_objects)
    obj_mat_indices = np.zeros(n_objs, dtype=int)
    for obj_idx, obj_mat in enumerate(obj_materials):
        obj_mat_indices[obj_idx] = unique_mat_names.index(obj_mat.name)
    # Do some light processing to add dictionaries to a list in a pickable format
    materials_dict_list = []
    for material in unique_materials:
        # Use 1.0 if relative_permeability is missing (Sionna RT >=1.0)
        if hasattr(material, 'relative_permeability'):
            rel_perm = material.relative_permeability.numpy()
        else:
            rel_perm = 1.0
        # Safely access scattering_pattern attributes; not all patterns have alpha_r, alpha_i, lambda_
        alpha_r = getattr(material.scattering_pattern, 'alpha_r', None)  # LambertianPattern etc. may not have this
        alpha_i = getattr(material.scattering_pattern, 'alpha_i', None)
        lambda_ = material.scattering_pattern.lambda_.numpy() if hasattr(material.scattering_pattern, 'lambda_') else None
        materials_dict = {
            'name': material.name,
            'conductivity': material.conductivity.numpy(),
            'relative_permeability': rel_perm,
            'relative_permittivity': material.relative_permittivity.numpy(),
            'scattering_coefficient': material.scattering_coefficient.numpy(),
            'scattering_pattern': type(material.scattering_pattern).__name__,
            'alpha_r': alpha_r,  # May be None if not present
            'alpha_i': alpha_i,  # May be None if not present
            'lambda_': lambda_,  # May be None if not present
            'xpd_coefficient': material.xpd_coefficient.numpy(),   
        }
        materials_dict_list += [materials_dict]
    return materials_dict_list, obj_mat_indices

def _scene_to_dict(scene: Scene) -> Dict[str, Any]: 
    """ Export a Sionna Scene to a dictionary, like to Paths.to_dict() """
    members_names = dir(scene)
    bug_attrs =  ['paths_solver']
    members_objects = [getattr(scene, attr) for attr in members_names
                       if attr not in bug_attrs]
    data = {attr_name[1:] : attr_obj for (attr_obj, attr_name)
            in zip(members_objects, members_names)
            if not callable(attr_obj) and
               not isinstance(attr_obj, sionna.rt.Scene) and
               not attr_name.startswith("__")}
    return data

def export_scene_rt_params(scene: Scene, **compute_paths_kwargs) -> Dict[str, Any]:
    """ Extract parameters from Scene (and from compute_paths arguments)"""
    scene_dict = _scene_to_dict(scene)
    
    # Safely get antenna positions for rx_array and tx_array
    rx_array = scene_dict['rx_array']
    tx_array = scene_dict['tx_array']
    sionna_v1 = is_sionna_v1()
    if sionna_v1:
        wavelength = scene.wavelength
        rx_array_ant_pos = rx_array.positions(wavelength).numpy()  
        tx_array_ant_pos = tx_array.positions(wavelength).numpy()
    else:
        rx_array_ant_pos = rx_array.positions
        tx_array_ant_pos = tx_array.positions
    
    # Safely get synthetic_array option (from scene_dict or compute_paths_kwargs)
    synthetic_array = scene_dict.get('synthetic_array', compute_paths_kwargs.get('synthetic_array', False))
    
    rt_params_dict = dict(
        bandwidth=scene_dict['bandwidth'].numpy(),
        frequency=scene_dict['frequency'].numpy(),

        rx_array_size=rx_array.array_size,  # dual-pol if diff than num_ant
        rx_array_num_ant=rx_array.num_ant,
        rx_array_ant_pos=rx_array_ant_pos,  # relative to ref.
        
        tx_array_size=tx_array.array_size, 
        tx_array_num_ant=tx_array.num_ant,
        tx_array_ant_pos=tx_array_ant_pos,
        
        synthetic_array=synthetic_array,  # record the option used
        
        # custom
        raytracer_version=get_sionna_version(),
    )

    if sionna_v1:
        default_compute_paths_params = dict( # Sionna 1.x default values
            max_depth = 3,
            max_num_paths_per_src = 1000000,
            samples_per_src = 1000000,
            synthetic_array = True,
            los = True,
            specular_reflection = True,
            diffuse_reflection = False,
            refraction = True,
            seed = 42
        )
    else:
        default_compute_paths_params = dict( # Sionna 0.x default values
            max_depth=3, 
            method='fibonacci',
            num_samples=1000000,
            los=True,
            reflection=True,
            diffraction=False,
            scattering=False,
            scat_keep_prob=0.001,
            edge_diffraction=False,
            scat_random_phases=True
        )

    default_compute_paths_params.update(compute_paths_kwargs)
    raw_params = {**rt_params_dict, **default_compute_paths_params}

    # Mapping from Sionna 1.0.2 to common (0.19 / DeepMIMO) parameters
    newer_params_mapping = {
        'num_samples': raw_params['samples_per_src'],
        'reflection': bool(raw_params['specular_reflection']),
        'diffraction': False, #bool(raw_params['diffraction']),
        'scattering': bool(raw_params['diffuse_reflection']),
    } if sionna_v1 else {}

    return {**raw_params, **newer_params_mapping}

def export_scene_buildings(scene: Scene) -> Tuple[np.ndarray, Dict]:
    """ Export the vertices and faces of buildings in a Sionna Scene.
    Output:
        vertice_matrix: n_vertices_in_scene x 3 (xyz coordinates)
        obj_index_map: Dict with object name as key and (start_idx, end_idx) as value
    """
    all_vertices = []
    obj_index_map = {}  # Stores the name and starting index of each object
    
    vertex_offset = 0
    
    sionna_v1 = is_sionna_v1()
    for obj_name, obj in scene._scene_objects.items():
    
        # Get vertices
        n_v = obj._mi_shape.vertex_count()
        obj_vertices = np.array(obj._mi_shape.vertex_position(np.arange(n_v)))
        if sionna_v1:
            obj_vertices = obj_vertices.T

        # Robust shape check: skip empty
        if obj_vertices.size == 0:
            continue
        # Ensure 2D
        if obj_vertices.ndim == 1:
            obj_vertices = obj_vertices.reshape(1, -1)
        # If more than 3 columns, take only first 3 (x, y, z)
        if obj_vertices.shape[1] > 3:
            obj_vertices = obj_vertices[:, :3]
        # If less than 3 columns, pad with zeros
        if obj_vertices.shape[1] < 3:
            pad_width = 3 - obj_vertices.shape[1]
            obj_vertices = np.pad(obj_vertices, ((0,0),(0,pad_width)), 'constant')
        # Now shape is (N, 3)
        
        # Append vertices to global list
        all_vertices.append(obj_vertices)
    
        # Store object index range
        obj_index_map[obj_name] = (vertex_offset, vertex_offset + obj_vertices.shape[0])
        
        # Update vertex offset
        vertex_offset += obj_vertices.shape[0]
    
    # Convert lists to numpy arrays
    if len(all_vertices) == 0:
        vertice_matrix = np.zeros((0,3))
    else:
        vertice_matrix = np.vstack(all_vertices)

    return vertice_matrix, obj_index_map

def sionna_exporter(scene: Scene, path_list: List[Paths] | Paths, 
                    my_compute_path_params: Dict, save_folder: str):
    """ Export a complete Sionna simulation to a format that can be converted by DeepMIMO.
    
    This function exports all necessary data from a Sionna ray tracing simulation to files
    that can be converted into the DeepMIMO format. The exported data includes:
    - Ray paths and their properties
    - Scene materials and their properties 
    - Ray tracing parameters used in the simulation
    - Scene geometry (vertices and objects)

    Args:
        scene (Scene): The Sionna Scene object containing the simulation environment
        path_list (List[Paths] | Paths): Ray paths computed by Sionna's ray tracer, either
            for a single transmitter (Paths) or multiple transmitters (List[Paths])
        my_compute_path_params (Dict): Dictionary containing the parameters used in
            Sionna's compute_paths() function. This is needed since Sionna does not
            save these parameters internally.
        save_folder (str): Directory path where the exported files will be saved

    Note:
        - This function has been tested with Sionna v0.19.1 and v1.0.2.
        - In Sionna 1.x, the paths are exported during RT, so no need to export them here
    """
    paths_dict_list = path_list if type(path_list[0]) == dict else export_paths(path_list)
    materials_dict_list, material_indices = export_scene_materials(scene)
    rt_params = export_scene_rt_params(scene, **my_compute_path_params)
    vertice_matrix, obj_index_map = export_scene_buildings(scene)
    
    os.makedirs(save_folder, exist_ok=True)
    
    save_vars_dict = {
        # filename: variable_to_save
        'sionna_paths.pkl': paths_dict_list,
        'sionna_materials.pkl': materials_dict_list,
        'sionna_material_indices.pkl': material_indices,
        'sionna_rt_params.pkl': rt_params,
        'sionna_vertices.pkl': vertice_matrix,
        'sionna_objects.pkl': obj_index_map,
    }
    
    for filename, variable in save_vars_dict.items():
        save_pickle(variable, os.path.join(save_folder, filename))

    return

# Explicitly declare what should be imported when using 'from .sionna_exporter import *'
__all__ = ['sionna_exporter']