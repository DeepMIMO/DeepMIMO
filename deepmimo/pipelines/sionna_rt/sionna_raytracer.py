"""Sionna Ray Tracing Function.

This module contains the raytracing function for Sionna.

It is a wrapper around the Sionna RT API, and it is used to raytrace the scene.

Pipeline untested for versions <0.19 and >1.0.2.

"""
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mitsuba as mi
import numpy as np
import sionna.rt as sionna_rt
from sionna.rt import Receiver, Transmitter
from tqdm import tqdm

from deepmimo.exporters.sionna_exporter import export_paths, sionna_exporter

from .sionna_utils import create_base_scene, get_sionna_version, is_sionna_v1, set_materials

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

IS_LEGACY_VERSION = not is_sionna_v1()
if IS_LEGACY_VERSION:
    if not get_sionna_version().startswith("0.19"):
        msg = "Pipeline untested for versions <0.19 and >1.0.2"
        raise Warning(msg)
    try:
        import tensorflow as tf

        tf.random.set_seed(1)
        gpus = tf.config.list_physical_devices("GPU")
        print("TensorFlow sees GPUs:", gpus)
    except ImportError:
        print("TensorFlow not found. Please install TensorFlow to use Sionna ray tracing.")
        tf = None
    PathSolver = None
else:
    from sionna.rt import PathSolver

class _DataLoader:
    """DataLoader class for Sionna RT that returns user indices for raytracing."""

    def __init__(self: Any, data: Any, batch_size: Any) -> None:
        self.data = np.array(data)
        self.batch_size = batch_size
        self.num_samples = len(data)
        self.indices = np.arange(self.num_samples)

    def __len__(self: Any) -> int:
        return int(np.ceil(self.num_samples / self.batch_size))

    def __iter__(self: Any) -> Any:
        self.current_idx = 0
        return self

    def __next__(self: Any) -> Any:
        if self.current_idx >= self.num_samples:
            raise StopIteration
        start_idx = self.current_idx
        end_idx = min(self.current_idx + self.batch_size, self.num_samples)
        batch_indices = self.indices[start_idx:end_idx]
        self.current_idx = end_idx
        return self.data[batch_indices]

def _compute_paths(
    scene: sionna_rt.Scene,
    p_solver: PathSolver | None,
    compute_paths_rt_params: dict,
    *,
    cpu_offload: bool = True,
    path_inspection_func: Callable | None = None,
) -> Any:
    """Compute paths based on the active Sionna version.

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
    else:
        paths = p_solver(scene=scene, **compute_paths_rt_params)
    paths.normalize_delays = False
    if path_inspection_func is not None:
        path_inspection_func(paths)
    if cpu_offload and (not IS_LEGACY_VERSION):
        paths = export_paths(paths)[0]
    return paths

def raytrace_sionna(  # noqa: PLR0915, PLR0912, C901
    base_folder: str,
    tx_pos: np.ndarray,
    rx_pos: np.ndarray,
    **rt_params: Any,
) -> str:
    """Run ray tracing for the scene."""
    if rt_params["create_scene_folder"]:
        carrier_ghz = rt_params["carrier_freq"] / 1e9
        scattering_flag = 1 if rt_params["ds_enable"] else 0
        scene_name = (
            f"sionna_{carrier_ghz:.1f}GHz_{rt_params['max_reflections']}R_"
            f"{rt_params['max_diffractions']}D_{scattering_flag}S"
        )
        scene_folder = str(Path(base_folder) / scene_name)
    else:
        scene_folder = base_folder
    if rt_params["use_builtin_scene"]:
        xml_path = getattr(sionna_rt.scene, rt_params["builtin_scene_path"], None)
    else:
        xml_path = str(Path(base_folder) / "scene.xml")
    print(f"XML scene path: {xml_path}")
    scene = create_base_scene(xml_path, rt_params["carrier_freq"])
    if not rt_params["use_builtin_scene"]:
        scene = set_materials(scene)
    if rt_params["scene_edit_func"] is not None:
        rt_params["scene_edit_func"](scene)
    if rt_params["obj_idx"] is not None:
        for (i, obj_idx) in enumerate(rt_params["obj_idx"]):
            obj = scene.objects[obj_idx]
            if rt_params["obj_pos"] is not None:
                obj.position = mi.Vector3f(rt_params["obj_pos"][i])
            if rt_params["obj_ori"] is not None:
                obj.orientation = mi.Vector3f(rt_params["obj_ori"][i])
            if rt_params["obj_vel"] is not None:
                obj.velocity = mi.Vector3f(rt_params["obj_vel"][i])
    if IS_LEGACY_VERSION:
        compute_paths_rt_params = {
            "los": rt_params["los"],
            "max_depth": rt_params["max_reflections"],
            "diffraction": bool(rt_params["max_diffractions"]),
            "scattering": rt_params["ds_enable"],
            "num_samples": rt_params["n_samples_per_src"],
            "scat_random_phases": rt_params["scat_random_phases"],
            "edge_diffraction": rt_params["edge_diffraction"],
            "ris": False,
        }
        scene.synthetic_array = rt_params["synthetic_array"]
    else:
        compute_paths_rt_params = {
            "los": rt_params["los"],
            "synthetic_array": rt_params["synthetic_array"],
            "samples_per_src": rt_params["n_samples_per_src"],
            "max_num_paths_per_src": rt_params["max_paths_per_src"],
            "max_depth": rt_params["max_reflections"],
            "specular_reflection": bool(rt_params["max_reflections"]),
            "diffuse_reflection": rt_params["ds_enable"],
            "refraction": rt_params["refraction"],
        }

    def none_or_index(x: Any, i: Any) -> Any:
        return None if x is None else x[i]
    num_bs = len(tx_pos)
    for b in range(num_bs):
        pwr_dbm = tf.Variable(0, dtype=tf.float32) if IS_LEGACY_VERSION else 0
        vel_dict = (
            {}
            if IS_LEGACY_VERSION
            else {"velocity": none_or_index(rt_params["tx_vel"], b)}
        )
        tx = Transmitter(
            name=f"BS_{b}",
            position=tx_pos[b],
            orientation=none_or_index(rt_params["tx_ori"], b),
            power_dbm=pwr_dbm,
            **vel_dict,
        )
        scene.add(tx)
        print(f"Added BS_{b} at position {tx_pos[b]}")
    indices = np.arange(rx_pos.shape[0])
    data_loader = _DataLoader(indices, rt_params["batch_size"])
    path_list = []
    p_solver = None if IS_LEGACY_VERSION else PathSolver()
    if rt_params["bs2bs"]:
        print("Ray-tracing BS-BS paths")
        for b in range(num_bs):
            vel_dict = (
                {}
                if IS_LEGACY_VERSION
                else {"velocity": none_or_index(rt_params["tx_vel"], b)}
            )
            scene.add(
                Receiver(
                    name=f"rx_{b}",
                    position=tx_pos[b],
                    orientation=none_or_index(rt_params["tx_ori"], b),
                    **vel_dict,
                ),
            )
        paths = _compute_paths(
            scene,
            p_solver,
            compute_paths_rt_params,
            cpu_offload=rt_params["cpu_offload"],
            path_inspection_func=rt_params["path_inspection_func"],
        )
        path_list.append(paths)
        for b in range(num_bs):
            scene.remove(f"rx_{b}")
    for batch in tqdm(data_loader, desc="Ray-tracing BS-UE paths", unit="batch"):
        for i in batch:
            vel_dict = (
                {}
                if IS_LEGACY_VERSION
                else {"velocity": none_or_index(rt_params["rx_vel"], i)}
            )
            scene.add(
                Receiver(
                    name=f"rx_{i}",
                    position=rx_pos[i],
                    orientation=none_or_index(rt_params["rx_ori"], i),
                    **vel_dict,
                ),
            )
        paths = _compute_paths(
            scene,
            p_solver,
            compute_paths_rt_params,
            cpu_offload=rt_params["cpu_offload"],
            path_inspection_func=rt_params["path_inspection_func"],
        )
        path_list.append(paths)
        for i in batch:
            scene.remove(f"rx_{i}")
    print("Saving Sionna outputs")
    sionna_exporter(scene, path_list, rt_params, scene_folder)
    return scene_folder
