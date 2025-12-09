"""Web Export utilities for DeepMIMO web visualizer.

This module provides functionality to export DeepMIMO datasets to binary format
for the DeepMIMO web visualizer. It handles multi-TX/RX set datasets and creates
binary files with proper naming conventions.
"""

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np


def export_dataset_to_binary(
    dataset: Any,
    dataset_name: str,
    output_dir: str = "./datasets",
) -> None:
    """Export DeepMIMO dataset to binary format for web visualizer.

    This function handles both single datasets and MacroDatasets with multiple TX/RX sets.
    It creates binary files with proper naming convention and metadata for the web visualizer.

    Args:
        dataset: DeepMIMO Dataset or MacroDataset object
        dataset_name: Name of the scenario
        output_dir: Output directory for binary files (default: "./datasets")

    """
    base_dir = Path(output_dir) / dataset_name
    base_dir.mkdir(parents=True, exist_ok=True)

    tx_rx_sets_info = []

    # Check if this is a MacroDataset (has multiple datasets)
    if hasattr(dataset, "datasets"):
        print(f"Processing MacroDataset with {len(dataset.datasets)} TX/RX sets")
        tx_rx_sets_info = process_macro_dataset(dataset, base_dir)
    else:
        # Handle single Dataset
        print("Processing single Dataset")
        set_info = process_single_dataset_to_binary(dataset, base_dir, 1, 1)
        tx_rx_sets_info.append(set_info)

    # Save metadata
    metadata = {
        "numTxRxSets": len(tx_rx_sets_info),
        "txRxSets": tx_rx_sets_info,
    }

    with Path(base_dir / "metadata.bin").open("w") as f:
        json.dump(metadata, f)

    print(f"Export completed for {dataset_name} with {len(tx_rx_sets_info)} TX/RX sets")


def _process_macro_dataset(dataset: Any, base_dir: Path) -> list:
    """Process MacroDataset using TX/RX set information from datasets."""
    tx_rx_sets_info = []

    print("Extracting TX/RX set IDs from datasets...")

    for i, single_dataset in enumerate(dataset.datasets):
        # Get TX/RX set IDs from the dataset's txrx attribute
        tx_set_id = single_dataset["txrx"]["tx_set_id"]
        rx_set_id = single_dataset["txrx"]["rx_set_id"]

        print(
            "Dataset "
            f"{i}: TX set {tx_set_id}, RX set {rx_set_id} "
            f"(rx_count={len(single_dataset.rx_pos)})",
        )

        set_info = process_single_dataset_to_binary(single_dataset, base_dir, tx_set_id, rx_set_id)
        tx_rx_sets_info.append(set_info)

    return tx_rx_sets_info


def _save_binary_array(arr: np.ndarray, file_path: str | Path) -> None:
    """Save numpy array as simple binary format for web visualizer.

    Format: [dtype_code(4 bytes)][shape_dims(4 bytes)][shape_values(4 bytes each)][flattened_data]

    Args:
        arr: Numpy array to save
        file_path: Path where to save the binary file

    Raises:
        Exception: If array cannot be saved

    """
    try:
        with Path(file_path).open("wb") as f:
            # Write dtype code (padded to 4 bytes for alignment)
            dtype_code = {
                "float32": b"f\x00\x00\x00",
                "float64": b"d\x00\x00\x00",
                "int32": b"i\x00\x00\x00",
            }.get(str(arr.dtype), b"f\x00\x00\x00")
            f.write(dtype_code)

            # Write shape
            f.write(struct.pack("<I", len(arr.shape)))
            f.write(struct.pack(f"<{len(arr.shape)}I", *arr.shape))

            # Write flattened data
            arr_data = arr.astype(dtype_code[0:1].decode()).tobytes()
            f.write(arr_data)
    except Exception as e:
        print(f"Error saving array to {file_path}: {e}")
        raise


def _trim_interactions(
    inter: Any,
    inter_pos: Any,
    max_paths: int,
    min_inter_dims: int,
) -> tuple[Any, Any]:
    """Trim interaction arrays to a maximum number of paths."""
    trimmed_inter = inter
    trimmed_inter_pos = inter_pos

    if inter is not None and hasattr(inter, "ndim") and inter.ndim >= min_inter_dims:
        trimmed_inter = inter[:, :max_paths]

    if inter_pos is not None and hasattr(inter_pos, "ndim") and inter_pos.ndim >= min_inter_dims:
        trimmed_inter_pos = inter_pos[:, :max_paths]

    return trimmed_inter, trimmed_inter_pos


def _persist_properties(
    properties: dict[str, Any],
    base_dir: Path,
    tx_set_id: int,
    rx_set_id: int,
) -> None:
    """Save non-empty dataset properties to disk."""
    for name, data in properties.items():
        if data is None or not hasattr(data, "dtype"):
            continue

        try:
            file_path = base_dir / f"{name}_tx_{tx_set_id}_rx_{rx_set_id}.bin"
            save_binary_array(data, file_path)
            print(f"Saved {name} for RX set {rx_set_id}, TX set {tx_set_id} to {file_path}")
        except (OSError, ValueError) as e:
            print(f"Error saving {name} for RX set {rx_set_id}, TX set {tx_set_id}: {e}")


def _process_single_dataset_to_binary(
    dataset: Any,
    base_dir: Path,
    tx_set_id: int,
    rx_set_id: int,
) -> dict:
    """Process a single dataset to binary format for web visualizer.

    Args:
        dataset: DeepMIMO dataset object
        base_dir: Base directory for output files
        tx_set_id: TX set identifier for file naming
        rx_set_id: RX set identifier for file naming

    Returns:
        dict: TX/RX set information

    """
    max_paths = 5  # Limit number of paths stored
    min_inter_dims = 2

    # Validation: Check if dataset has basic required data
    if not hasattr(dataset, "rx_pos") or len(dataset.rx_pos) == 0:
        print(f"Warning: Dataset TX={tx_set_id}, RX={rx_set_id} has no rx_pos data, skipping")
        return {
            "tx_set": tx_set_id,
            "rx_set": rx_set_id,
            "totalUsers": 0,
            "usersPerRow": 0,
            "numRows": 0,
        }

    # Extract basic information
    total_users = len(dataset.rx_pos)
    unique_x = np.unique(dataset.rx_pos[:, 0])
    unique_y = np.unique(dataset.rx_pos[:, 1])
    users_per_row = len(unique_x)
    num_rows = len(unique_y)

    # Create TX/RX set info
    set_info = {
        "rx_set": rx_set_id,
        "tx_set": tx_set_id,
        "totalUsers": total_users,
        "usersPerRow": users_per_row,
        "numRows": num_rows,
        "bounds": {
            "x": [float(np.min(dataset.rx_pos[:, 0])), float(np.max(dataset.rx_pos[:, 0]))],
            "y": [float(np.min(dataset.rx_pos[:, 1])), float(np.max(dataset.rx_pos[:, 1]))],
            "z": [float(np.min(dataset.rx_pos[:, 2])), float(np.max(dataset.rx_pos[:, 2]))],
        },
    }

    print(f"Processing TX/RX set {tx_set_id}/{rx_set_id} with {total_users} users")

    # Process TX position
    tx_pos = dataset.tx_pos
    if isinstance(tx_pos, np.ndarray):
        if tx_pos.ndim == 1:
            tx_pos = tx_pos.reshape(1, -1)
        print(f"Processed tx_pos to shape {tx_pos.shape}")

    # Process interaction data with path limits
    inter = dataset.inter if hasattr(dataset, "inter") else None
    inter_pos = dataset.inter_pos if hasattr(dataset, "inter_pos") else None

    inter, inter_pos = _trim_interactions(inter, inter_pos, max_paths, min_inter_dims)

    # Collect all properties to save
    properties = {
        "rx_pos": dataset.rx_pos,
        "tx_pos": tx_pos,
        "power": dataset.power if hasattr(dataset, "power") else None,
        "inter": inter,
        "inter_pos": inter_pos,
        "aoa_az": dataset.aoa_az if hasattr(dataset, "aoa_az") else None,
        "aoa_el": dataset.aoa_el if hasattr(dataset, "aoa_el") else None,
        "aod_az": dataset.aod_az if hasattr(dataset, "aod_az") else None,
        "aod_el": dataset.aod_el if hasattr(dataset, "aod_el") else None,
        "toa": dataset.toa if hasattr(dataset, "toa") else None,
        "phase": dataset.phase if hasattr(dataset, "phase") else None,
    }

    # Save each property as a binary file with TX/RX set identifier
    _persist_properties(properties, base_dir, tx_set_id, rx_set_id)

    # Process scene data if available
    if hasattr(dataset, "scene") and dataset.scene is not None:
        try:
            process_scene_to_binary(dataset.scene, base_dir)
        except (OSError, RuntimeError) as e:
            print(f"Error processing scene data: {e}")

    return set_info


def _process_scene_to_binary(scene: Any, base_dir: Path) -> None:  # noqa: C901, PLR0912
    """Process scene data to binary format.

    Args:
        scene: Scene object from dataset
        base_dir: Base directory for output files

    """
    print(f"Processing scene data: {type(scene)}")

    # Get all objects in the scene
    buildings_obj = scene.get_objects("buildings")
    terrain_obj = scene.get_objects("terrain")
    vegetation_obj = scene.get_objects("vegetation")

    # Process buildings
    if buildings_obj:
        buildings = []
        max_faces = 0
        max_vertices = 0

        # First pass to find maximum dimensions
        for building in buildings_obj:
            max_faces = max(max_faces, len(building.faces))
            for face in building.faces:
                max_vertices = max(max_vertices, len(face.vertices))

        # Second pass to create padded arrays
        for building in buildings_obj:
            building_data = np.zeros((max_faces, max_vertices, 3), dtype=np.float32)
            for i, face in enumerate(building.faces):
                for j, vertex in enumerate(face.vertices):
                    building_data[i, j] = vertex
            buildings.append(building_data)

        buildings_array = np.array(buildings, dtype=np.float32)
        file_path = base_dir / "buildings.bin"
        save_binary_array(buildings_array, file_path)
        print(f"Saved buildings to {file_path}")

    # Process terrain objects
    if terrain_obj:
        terrain = []
        max_faces = 0
        max_vertices = 0

        # First pass to find maximum dimensions
        for terr in terrain_obj:
            max_faces = max(max_faces, len(terr.faces))
            for face in terr.faces:
                max_vertices = max(max_vertices, len(face.vertices))

        # Second pass to create padded arrays
        for terr in terrain_obj:
            terrain_data = np.zeros((max_faces, max_vertices, 3), dtype=np.float32)
            for i, face in enumerate(terr.faces):
                for j, vertex in enumerate(face.vertices):
                    terrain_data[i, j] = vertex
            terrain.append(terrain_data)

        terrain_array = np.array(terrain, dtype=np.float32)
        file_path = base_dir / "terrain.bin"
        save_binary_array(terrain_array, file_path)
        print(f"Saved terrain to {file_path}")

    # Process vegetation objects
    if vegetation_obj:
        vegetation = []
        max_faces = 0
        max_vertices = 0

        # First pass to find maximum dimensions
        for veg in vegetation_obj:
            max_faces = max(max_faces, len(veg.faces))
            for face in veg.faces:
                max_vertices = max(max_vertices, len(face.vertices))

        # Second pass to create padded arrays
        for veg in vegetation_obj:
            vegetation_data = np.zeros((max_faces, max_vertices, 3), dtype=np.float32)
            for i, face in enumerate(veg.faces):
                for j, vertex in enumerate(face.vertices):
                    vegetation_data[i, j] = vertex
            vegetation.append(vegetation_data)

        vegetation_array = np.array(vegetation, dtype=np.float32)
        file_path = base_dir / "vegetation.bin"
        save_binary_array(vegetation_array, file_path)
        print(f"Saved vegetation to {file_path}")

# Public aliases for external callers and tests.
save_binary_array = _save_binary_array
process_single_dataset_to_binary = _process_single_dataset_to_binary
process_macro_dataset = _process_macro_dataset
process_scene_to_binary = _process_scene_to_binary
