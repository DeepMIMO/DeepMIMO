"""Sionna Ray Tracing Function.

This module contains the raytracing function for Sionna.

It is a wrapper around the Sionna RT API, and it is used to raytrace the scene.

Pipeline untested for versions <0.19 and >1.0.2.

"""


# Standard library imports
import os
from typing import Any, Optional, Callable

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow (excessive) logging

# Third-party imports
import numpy as np
import mitsuba as mi
from tqdm import tqdm

# Local imports
from .sionna_utils import create_base_scene, set_materials, is_sionna_v1, get_sionna_version
from ...converter.sionna_rt import sionna_exporter

# Version check constant
IS_LEGACY_VERSION = not is_sionna_v1()

# Conditional TensorFlow import based on Sionna version
if IS_LEGACY_VERSION:
    if not get_sionna_version().startswith('0.19'):
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
    import sionna.rt
except ImportError:
    raise ImportError("Sionna not found. Install Sionna to use ray tracing functionality.")

from sionna.rt import Transmitter, Receiver

if not IS_LEGACY_VERSION:
    from sionna.rt import PathSolver
    import drjit as dr
else:
    PathSolver = None

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

def _compute_paths(scene: sionna.rt.Scene, p_solver: Optional[PathSolver], compute_paths_rt_params: dict, 
                   cpu_offload: bool = True, path_inspection_func: Optional[Callable] = None):
    """Helper function to compute paths based on Sionna version.

    Args:
        scene: The scene to compute paths for.
        p_solver: The path solver to use.
        compute_paths_rt_params: The parameters to pass to the path solver.
        cpu_offload: Whether to offload the paths to the CPU.
        path_inspection_func: A function to inspect the paths after computation.

    Returns:
        The paths object in a format that can be saved to pickle. 
    """

    if IS_LEGACY_VERSION:
        paths = scene.compute_paths(**compute_paths_rt_params)
    else:  # version 1.x
        paths = p_solver(scene=scene, **compute_paths_rt_params)
    
    paths.normalize_delays = False

    if path_inspection_func is not None:
        path_inspection_func(paths)

    # Export paths to CPU if requested
    if cpu_offload and not IS_LEGACY_VERSION:
        paths = sionna_exporter.export_paths(paths)[0]
    
    return paths

def raytrace_sionna(base_folder: str, tx_pos: np.ndarray, rx_pos: np.ndarray, **rt_params: Any) -> str:
    """Run ray tracing for the scene."""
    
    if rt_params['create_scene_folder']:
        # Create scene
        scene_name = (f"sionna_{rt_params['carrier_freq']/1e9:.1f}GHz_"
                    f"{rt_params['max_reflections']}R_{rt_params['max_diffractions']}D_"
                    f"{1 if rt_params['ds_enable'] else 0}S")

        scene_folder = os.path.join(base_folder, scene_name)
    else:
        scene_folder = base_folder
    
    if rt_params['use_builtin_scene']:
        xml_path = getattr(sionna.rt.scene, rt_params['builtin_scene_path'])
    else:
        xml_path = os.path.join(base_folder, "scene.xml")  # Created by Blender OSM Export!
    
    print(f"XML scene path: {xml_path}")
    scene = create_base_scene(xml_path, rt_params['carrier_freq'])
    if not rt_params['use_builtin_scene']:
        scene = set_materials(scene)
    
    if rt_params['scene_edit_func'] is not None:
        rt_params['scene_edit_func'](scene)

    # Change objects in the scene
    if rt_params['obj_idx'] is not None:
        for i, obj_idx in enumerate(rt_params['obj_idx']):
            obj = scene.objects[obj_idx]
            if rt_params['obj_pos'] is not None:
                obj.position = mi.Vector3f(rt_params['obj_pos'][i])
            if rt_params['obj_ori'] is not None:
                obj.orientation = mi.Vector3f(rt_params['obj_ori'][i])
            if rt_params['obj_vel'] is not None:
                obj.velocity = mi.Vector3f(rt_params['obj_vel'][i])
    
    # Map general parameters to Sionna RT parameters
    if IS_LEGACY_VERSION:
        compute_paths_rt_params = {
            "los": rt_params['los'],
            "max_depth": rt_params['max_reflections'],
            "diffraction": bool(rt_params['max_diffractions']),
            "scattering": rt_params['ds_enable'],
            "num_samples": rt_params['n_samples_per_src'],
            "scat_random_phases": rt_params['scat_random_phases'],
            "edge_diffraction": rt_params['edge_diffraction'],
            'ris': False,
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

    # Helper function to get None or index of array if not None
    none_or_index = lambda x, i: None if x is None else x[i]

    # Add BSs
    num_bs = len(tx_pos)
    for b in range(num_bs): 
        pwr_dbm = tf.Variable(0, dtype=tf.float32) if IS_LEGACY_VERSION else 0
        vel_dict = {} if IS_LEGACY_VERSION else {'velocity': none_or_index(rt_params['tx_vel'], b)}
        tx = Transmitter(name=f"BS_{b}", position=tx_pos[b], 
                         orientation=none_or_index(rt_params['tx_ori'], b), 
                         power_dbm=pwr_dbm, **vel_dict)
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
            vel_dict = {} if IS_LEGACY_VERSION else {'velocity': none_or_index(rt_params['tx_vel'], b)}
            scene.add(Receiver(name=f"rx_{b}", position=tx_pos[b], 
                               orientation=none_or_index(rt_params['tx_ori'], b), 
                               **vel_dict))

        paths = _compute_paths(scene, p_solver, compute_paths_rt_params, 
                               cpu_offload=rt_params['cpu_offload'], 
                               path_inspection_func=rt_params['path_inspection_func'])
        path_list.append(paths)

        for b in range(num_bs):
            scene.remove(f"rx_{b}")
        

    # Ray-tracing BS-UE paths
    for batch in tqdm(data_loader, desc="Ray-tracing BS-UE paths", unit='batch'):
        for i in batch:
            vel_dict = {} if IS_LEGACY_VERSION else {'velocity': none_or_index(rt_params['rx_vel'], i)}
            scene.add(Receiver(name=f"rx_{i}", position=rx_pos[i], 
                               orientation=none_or_index(rt_params['rx_ori'], i),
                               **vel_dict))
        
        paths = _compute_paths(scene, p_solver, compute_paths_rt_params, 
                               cpu_offload=rt_params['cpu_offload'], 
                               path_inspection_func=rt_params['path_inspection_func'])
        path_list.append(paths)
        
        for i in batch:
            scene.remove(f"rx_{i}")
    
    # Save Sionna outputs
    print("Saving Sionna outputs")

    sionna_exporter.export_to_deepmimo(scene, path_list, rt_params, scene_folder)
    return scene_folder