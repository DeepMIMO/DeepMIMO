"""Sionna Ray Tracing Function."""


# Standard library imports
import os
from typing import Any

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow (excessive) logging

# Third-party imports
import numpy as np
from tqdm import tqdm

# Local imports
from .sionna_utils import create_base_scene, set_materials
from ...converter.sionna_rt import sionna_exporter
from ...config import config

# Version check constant
IS_LEGACY_VERSION = config.get('sionna_version').startswith('0.1')

# Conditional TensorFlow import based on Sionna version
if IS_LEGACY_VERSION:
    if not config.get('sionna_version').startswith('0.19'):
        raise Warning("Pipeline untested for versions <0.19 and >1.0.2")
    try:
        import tensorflow as tf  # type: ignore
        tf.random.set_seed(1)
        gpus = tf.config.list_physical_devices('GPU')
        print("TensorFlow sees GPUs:", gpus)
    except ImportError:
        print("TensorFlow not found. Please install TensorFlow to use Sionna ray tracing.")
        tf = None

try:
    if IS_LEGACY_VERSION:
        from sionna.rt import Transmitter, Receiver
    else: # version 1.x
        from sionna.rt import Transmitter, Receiver, PathSolver
        import drjit as dr
except ImportError:
    raise ImportError("Sionna not found. Please install Sionna to use ray tracing functionality.")

class _DataLoader:
    """DataLoader class for Sionna RT that returns user indices for raytracing."""
    def __init__(self, data, batch_size):
        self.data = np.array(data)
        self.batch_size = batch_size
        self.num_samples = len(data)
        self.indices = np.arange(self.num_samples)

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= self.num_samples:
            raise StopIteration
        start_idx = self.current_idx
        end_idx = min(self.current_idx + self.batch_size, self.num_samples)
        batch_indices = self.indices[start_idx:end_idx]
        self.current_idx = end_idx
        return self.data[batch_indices]

def _compute_paths(scene, p_solver, compute_paths_rt_params):
    """Helper function to compute paths based on Sionna version."""
    if IS_LEGACY_VERSION:
        paths = scene.compute_paths(**compute_paths_rt_params)
    else:  # version 1.x
        paths = p_solver(scene=scene, **compute_paths_rt_params)
    
    paths.normalize_delays = False
    return paths if IS_LEGACY_VERSION else export_paths_to_cpu(paths)

def raytrace_sionna(osm_folder: str, tx_pos: np.ndarray, rx_pos: np.ndarray, **rt_params: Any) -> str:
    """Run ray tracing for the scene."""
    # Create scene
    scene_name = (f"sionna_{rt_params['carrier_freq']/1e9:.1f}GHz_"
                    f"{rt_params['max_reflections']}R_{rt_params['max_diffractions']}D_"
                    f"{1 if rt_params['ds_enable'] else 0}S")

    scene_folder = os.path.join(osm_folder, scene_name)
    
    if rt_params['use_builtin_scene']:
        xml_path = getattr(sionna.rt.scene, rt_params['builtin_scene_path'])
    else:
        xml_path = os.path.join(osm_folder, "scene.xml")  # Created by Blender OSM Export!
    
    print(f"XML scene path: {xml_path}")
    scene = create_base_scene(xml_path, rt_params['carrier_freq'])
    if not rt_params['use_builtin_scene']:
        scene = set_materials(scene)
    
    # Map general parameters to Sionna RT parameters
    if IS_LEGACY_VERSION:
        compute_paths_rt_params = {
            "los": rt_params['los'],
            "max_depth": rt_params['max_reflections'],
            "diffraction": bool(rt_params['max_diffractions']),
            "scattering": rt_params['ds_enable'],
            "num_samples": rt_params['n_samples_per_src']
        }
        scene.synthetic_array = rt_params['synthetic_array']
    else: # version 1.x
        compute_paths_rt_params = {
            "los": rt_params['los'],
            "synthetic_array": rt_params['synthetic_array'], 
            "samples_per_src": rt_params['n_samples_per_src'],
            "max_num_paths_per_src": rt_params['max_paths_per_src'],
            "max_depth": rt_params['max_reflections'],
            # "diffraction": bool(rt_params['max_diffractions']),
            "specular_reflection": bool(rt_params['max_reflections']),
            "diffuse_reflection": rt_params['ds_enable'],
            "refraction": rt_params['refraction']
        }

    # Add BSs
    num_bs = len(tx_pos)
    for b in range(num_bs): 
        pwr_dbm = tf.Variable(0, dtype=tf.float32) if IS_LEGACY_VERSION else 0
        tx = Transmitter(position=tx_pos[b], name=f"BS_{b}", power_dbm=pwr_dbm)
        scene.add(tx)
        print(f"Added BS_{b} at position {tx_pos[b]}")

    indices = np.arange(rx_pos.shape[0])

    data_loader = _DataLoader(indices, rt_params['batch_size'])
    path_list = []

    p_solver = None if IS_LEGACY_VERSION else PathSolver()
    
    if rt_params['bs2bs']:
        # Ray-tracing BS-BS paths
        print("Ray-tracing BS-BS paths")
        for b in range(num_bs):
            scene.add(Receiver(name=f"rx_{b}", position=tx_pos[b]))

        paths = _compute_paths(scene, p_solver, compute_paths_rt_params)
        path_list.append(paths)

        for b in range(num_bs):
            scene.remove(f"rx_{b}")

    # Ray-tracing BS-UE paths
    for batch in tqdm(data_loader, desc="Ray-tracing BS-UE paths", unit='batch'):
        for i in batch:
            scene.add(Receiver(name=f"rx_{i}", position=rx_pos[i]))
        
        paths = _compute_paths(scene, p_solver, compute_paths_rt_params)
        path_list.append(paths)
        
        for i in batch:
            scene.remove(f"rx_{i}")
    
    # Save Sionna outputs
    print("Saving Sionna outputs")
    sionna_rt_folder = os.path.join(scene_folder, "sionna_export/")
    if IS_LEGACY_VERSION:
        sionna_exporter.export_to_deepmimo(scene, path_list, rt_params, sionna_rt_folder)
    else:
        sionna_exporter.export_to_deepmimo_v2(scene, path_list, rt_params, sionna_rt_folder)

    return sionna_rt_folder

import sionna.rt

Paths = sionna.rt.Paths
Scene = sionna.rt.Scene

def to_dict(paths: Paths) -> dict:
    """Exports paths to a filtered dictionary with only selected keys """
    members_names = dir(paths)
    members_objects = [getattr(paths, attr) for attr in members_names]
    data = {attr_name : attr_obj for (attr_obj, attr_name)
            in zip(members_objects, members_names)
            if not callable(attr_obj) and
                not isinstance(attr_obj, Scene) and
                not attr_name.startswith("__") and
                not attr_name.startswith("_")}
    return data


def export_paths_to_cpu(paths_obj: Paths) -> dict:
    """Exports paths to a filtered dictionary with only selected keys.
    
    This function is (CURRENTLY) only applied to Sionna 1.x. 
    (Sionna 0.x currently takes care of the conversion in the exporter - Need organizing!)

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
    """
    relevant_keys = ['a', 'tau', 'phi_r', 'phi_t', 'theta_r', 'theta_t',
                     'sources', 'targets', 'vertices']
    relevant_keys += ['types'] if IS_LEGACY_VERSION else ['interactions']

    path_dict = to_dict(paths_obj)
    path_dict_keys = path_dict.keys()
    
    dict_filtered = {}
    for key in relevant_keys:
        if key in path_dict_keys:
            if key == 'a' and not IS_LEGACY_VERSION:
                # Compose complex array from real and imaginary parts
                dict_filtered[key] = path_dict[key][0].numpy() + 1j * path_dict[key][1].numpy()
            else:
                dict_filtered[key] = path_dict[key].numpy()
    
    return dict_filtered