"""Sionna Ray Tracing Function.

This module contains the raytracing function for Sionna.

It is a wrapper around the Sionna RT API, and it is used to raytrace the scene.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import mitsuba as mi
import numpy as np
import sionna.rt as sionna_rt
from sionna.rt import PathSolver, Receiver, Transmitter
from tqdm import tqdm

from deepmimo.exporters.sionna_exporter import export_paths, sionna_exporter

from .sionna_utils import create_base_scene, set_materials


class _DataLoader:
    """DataLoader class for Sionna RT that returns user indices for raytracing."""

    def __init__(self, data: Any, batch_size: Any) -> None:
        self.data = np.array(data)
        self.batch_size = batch_size
        self.num_samples = len(data)
        self.indices = np.arange(self.num_samples)

    def __len__(self) -> int:
        return int(np.ceil(self.num_samples / self.batch_size))

    def __iter__(self) -> Any:
        self.current_idx = 0
        return self

    def __next__(self) -> Any:
        if self.current_idx >= self.num_samples:
            raise StopIteration
        start_idx = self.current_idx
        end_idx = min(self.current_idx + self.batch_size, self.num_samples)
        batch_indices = self.indices[start_idx:end_idx]
        self.current_idx = end_idx
        return self.data[batch_indices]


def _compute_paths(
    scene: sionna_rt.Scene,
    p_solver: PathSolver,
    compute_paths_rt_params: dict,
    *,
    cpu_offload: bool = True,
    path_inspection_func: Callable | None = None,
) -> Any:
    """Compute paths using the Sionna 2.0 PathSolver API.

    Args:
        scene: The scene to compute paths for.
        p_solver: The PathSolver instance.
        compute_paths_rt_params: Parameters forwarded to PathSolver.__call__.
        cpu_offload: Whether to convert paths to a serialisable dict immediately.
        path_inspection_func: Optional hook called on the raw Paths object.

    Returns:
        Paths object or serialised dict, depending on cpu_offload.

    """
    paths = p_solver(scene=scene, **compute_paths_rt_params)
    if path_inspection_func is not None:
        path_inspection_func(paths)
    if cpu_offload:
        paths = export_paths(paths)[0]
    return paths


def raytrace_sionna(  # noqa: PLR0912, C901
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
        for i, obj_idx in enumerate(rt_params["obj_idx"]):
            obj = scene.objects[obj_idx]
            if rt_params["obj_pos"] is not None:
                obj.position = mi.Vector3f(rt_params["obj_pos"][i])
            if rt_params["obj_ori"] is not None:
                obj.orientation = mi.Vector3f(rt_params["obj_ori"][i])
            if rt_params["obj_vel"] is not None:
                obj.velocity = mi.Vector3f(rt_params["obj_vel"][i])

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
        tx = Transmitter(
            name=f"BS_{b}",
            position=tx_pos[b],
            orientation=none_or_index(rt_params["tx_ori"], b),
            power_dbm=0,
            velocity=none_or_index(rt_params["tx_vel"], b),
        )
        scene.add(tx)
        print(f"Added BS_{b} at position {tx_pos[b]}")
    indices = np.arange(rx_pos.shape[0])
    data_loader = _DataLoader(indices, rt_params["batch_size"])
    path_list = []
    p_solver = PathSolver()
    if rt_params["bs2bs"]:
        print("Ray-tracing BS-BS paths")
        for b in range(num_bs):
            scene.add(
                Receiver(
                    name=f"rx_{b}",
                    position=tx_pos[b],
                    orientation=none_or_index(rt_params["tx_ori"], b),
                    velocity=none_or_index(rt_params["tx_vel"], b),
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
            scene.add(
                Receiver(
                    name=f"rx_{i}",
                    position=rx_pos[i],
                    orientation=none_or_index(rt_params["rx_ori"], i),
                    velocity=none_or_index(rt_params["rx_vel"], i),
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
