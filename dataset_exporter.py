"""
DeepMIMO Dataset Exporter for Web Visualizer
Exports datasets to binary format compatible with the web visualizer
"""

import numpy as np
import json
import struct
from pathlib import Path


def save_binary_array(arr, file_path):
    """Save numpy array as simple binary format:
    [dtype_code(4 bytes)][shape_dims(4 bytes)][shape_values(4 bytes each)][flattened_data]
    """
    try:
        with open(file_path, 'wb') as f:
            # Write dtype code (padded to 4 bytes for alignment)
            dtype_code = {
                'float32': b'f\x00\x00\x00',
                'float64': b'd\x00\x00\x00',
                'int32': b'i\x00\x00\x00'
            }.get(str(arr.dtype), b'f\x00\x00\x00')
            f.write(dtype_code)
            
            # Write shape
            f.write(struct.pack('<I', len(arr.shape)))
            f.write(struct.pack(f'<{len(arr.shape)}I', *arr.shape))
            
            # Write flattened data
            arr_data = arr.astype(dtype_code[0:1].decode()).tobytes()
            f.write(arr_data)
    except Exception as e:
        print(f"Error saving array to {file_path}: {e}")
        raise


def export_dataset_to_binary(dataset, dataset_name, output_dir=None, max_paths=5):
    """
    Export a DeepMIMO dataset to binary format for web visualizer.
    
    This function can be called as dataset.to_binary() when integrated into the library.
    
    Args:
        dataset: DeepMIMO dataset object (or macro dataset)
        dataset_name: Name of the dataset/scenario
        output_dir: Output directory path (default: './datasets' in current directory)
        max_paths: Maximum number of multipath components to save
    """
    
    # Set default output directory to current directory
    if output_dir is None:
        output_dir = './datasets'
    
    # Convert to Path object
    base_dir = Path(output_dir) / dataset_name
    base_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Exporting dataset '{dataset_name}' to {base_dir}")
    
    # Handle macro datasets (multiple TX/RX sets) vs single datasets
    if hasattr(dataset, '__len__') and len(dataset) > 1:
        # Macro dataset with multiple TX/RX combinations
        print(f"Found macro dataset with {len(dataset)} TX/RX combinations")
        _export_macro_dataset(dataset, base_dir, max_paths)
    else:
        # Single dataset (could still have multiple TX/RX sets if loaded with rx_set='all')
        if hasattr(dataset, '__len__') and len(dataset) == 1:
            # Extract single dataset from macro dataset
            single_dataset = dataset[0]
        else:
            # Already a single dataset
            single_dataset = dataset
            
        print(f"Found single dataset")
        _export_single_dataset(single_dataset, base_dir, max_paths)


def _export_macro_dataset(macro_dataset, base_dir, max_paths):
    """Export macro dataset with multiple TX/RX combinations"""
    
    # Collect metadata for all TX/RX sets
    tx_rx_sets_info = []
    
    for i, dataset in enumerate(macro_dataset):
        print(f"Processing TX/RX combination {i + 1} of {len(macro_dataset)}")
        
        # Get metadata for this TX/RX set
        set_info = _get_dataset_metadata(dataset, i + 1, i + 1)  # Assuming 1:1 mapping
        tx_rx_sets_info.append(set_info)
        
        # Export data for this TX/RX set
        _export_dataset_data(dataset, base_dir, i + 1, i + 1, max_paths)
    
    # Save combined metadata
    metadata = {
        'numTxRxSets': len(macro_dataset),
        'txRxSets': tx_rx_sets_info
    }
    
    with open(base_dir / 'metadata.bin', 'w') as f:
        json.dump(metadata, f)
    
    # Export scene data (shared across all TX/RX sets)
    # Use the first dataset's scene since it should be the same for all
    _export_scene_data(macro_dataset[0], base_dir)


def _export_single_dataset(dataset, base_dir, max_paths):
    """Export single dataset (may still have multiple internal TX/RX sets)"""
    
    # Check if dataset has multiple TX/RX sets internally
    if isinstance(getattr(dataset, 'rx_pos', None), list):
        num_sets = len(dataset.rx_pos)
        print(f"Found {num_sets} internal TX/RX sets")
        
        # Process each internal TX/RX set
        tx_rx_sets_info = []
        for set_idx in range(num_sets):
            set_info = _get_internal_set_metadata(dataset, set_idx)
            tx_rx_sets_info.append(set_info)
            _export_internal_set_data(dataset, base_dir, set_idx, max_paths)
            
        metadata = {
            'numTxRxSets': num_sets,
            'txRxSets': tx_rx_sets_info
        }
    else:
        # Single TX/RX set
        print("Found single TX/RX set")
        set_info = _get_dataset_metadata(dataset, 1, 1)
        _export_dataset_data(dataset, base_dir, 1, 1, max_paths)
        
        metadata = {
            'numTxRxSets': 1,
            'txRxSets': [set_info]
        }
    
    with open(base_dir / 'metadata.bin', 'w') as f:
        json.dump(metadata, f)
    
    # Export scene data
    _export_scene_data(dataset, base_dir)


def _get_dataset_metadata(dataset, rx_set_num, tx_set_num):
    """Get metadata for a single dataset"""
    rx_pos = dataset.rx_pos
    total_users = len(rx_pos)
    unique_x = np.unique(rx_pos[:, 0])
    unique_y = np.unique(rx_pos[:, 1])
    users_per_row = len(unique_x)
    num_rows = len(unique_y)
    
    return {
        'rx_set': rx_set_num,
        'tx_set': tx_set_num,
        'totalUsers': total_users,
        'usersPerRow': users_per_row,
        'numRows': num_rows,
        'bounds': {
            'x': [float(np.min(rx_pos[:,0])), float(np.max(rx_pos[:,0]))],
            'y': [float(np.min(rx_pos[:,1])), float(np.max(rx_pos[:,1]))],
            'z': [float(np.min(rx_pos[:,2])), float(np.max(rx_pos[:,2]))]
        }
    }


def _get_internal_set_metadata(dataset, set_idx):
    """Get metadata for internal TX/RX set within a dataset"""
    rx_pos = dataset.rx_pos[set_idx]
    total_users = len(rx_pos)
    unique_x = np.unique(rx_pos[:, 0])
    unique_y = np.unique(rx_pos[:, 1])
    users_per_row = len(unique_x)
    num_rows = len(unique_y)
    
    return {
        'rx_set': set_idx + 1,
        'tx_set': set_idx + 1,  # Assuming 1:1 mapping
        'totalUsers': total_users,
        'usersPerRow': users_per_row,
        'numRows': num_rows,
        'bounds': {
            'x': [float(np.min(rx_pos[:,0])), float(np.max(rx_pos[:,0]))],
            'y': [float(np.min(rx_pos[:,1])), float(np.max(rx_pos[:,1]))],
            'z': [float(np.min(rx_pos[:,2])), float(np.max(rx_pos[:,2]))]
        }
    }


def _export_dataset_data(dataset, base_dir, rx_set_num, tx_set_num, max_paths):
    """Export data for a single dataset"""
    
    # Get data
    rx_pos = dataset.rx_pos
    tx_pos = getattr(dataset, 'tx_pos', None)
    power = getattr(dataset, 'power', None)
    aoa_az = getattr(dataset, 'aoa_az', None)
    aoa_el = getattr(dataset, 'aoa_el', None)
    aod_az = getattr(dataset, 'aod_az', None)
    aod_el = getattr(dataset, 'aod_el', None)
    toa = getattr(dataset, 'toa', None)
    phase = getattr(dataset, 'phase', None)
    inter = getattr(dataset, 'inter', None)
    inter_pos = getattr(dataset, 'inter_pos', None)
    
    # Process tx_pos
    if tx_pos is not None and isinstance(tx_pos, np.ndarray):
        if tx_pos.ndim == 1:
            tx_pos = tx_pos.reshape(1, -1)
    
    # Process multipath data with path limits
    if inter is not None and hasattr(inter, 'ndim') and inter.ndim >= 2:
        inter = inter[:, :max_paths]
    
    if inter_pos is not None and hasattr(inter_pos, 'ndim') and inter_pos.ndim >= 2:
        inter_pos = inter_pos[:, :max_paths]
    
    # Save all properties
    properties = {
        'rx_pos': rx_pos,
        'tx_pos': tx_pos,
        'power': power,
        'inter': inter,
        'inter_pos': inter_pos,
        'aoa_az': aoa_az,
        'aoa_el': aoa_el,
        'aod_az': aod_az,
        'aod_el': aod_el,
        'toa': toa,
        'phase': phase,
    }
    
    for name, data in properties.items():
        if data is not None:
            try:
                file_path = base_dir / f'{name}_rx_{rx_set_num}_tx_{tx_set_num}.bin'
                save_binary_array(data, file_path)
                print(f"Saved {name} for RX set {rx_set_num}, TX set {tx_set_num}")
            except Exception as e:
                print(f"Error saving {name} for RX set {rx_set_num}, TX set {tx_set_num}: {e}")


def _export_internal_set_data(dataset, base_dir, set_idx, max_paths):
    """Export data for internal TX/RX set within a dataset"""
    
    rx_set_num = set_idx + 1
    tx_set_num = set_idx + 1  # Assuming 1:1 mapping
    
    def get_set_data(attr_name, set_index):
        if hasattr(dataset, attr_name):
            attr_value = getattr(dataset, attr_name)
            if isinstance(attr_value, list):
                if len(attr_value) > set_index:
                    return attr_value[set_index]
                else:
                    print(f"Warning: {attr_name} list doesn't have data for set {set_index}")
                    return None
            else:
                # If it's not a list, use the same data for all sets
                return attr_value
        return None
    
    # Get data for this specific TX/RX set
    rx_pos = get_set_data('rx_pos', set_idx)
    tx_pos = get_set_data('tx_pos', set_idx)
    power = get_set_data('power', set_idx)
    aoa_az = get_set_data('aoa_az', set_idx)
    aoa_el = get_set_data('aoa_el', set_idx)
    aod_az = get_set_data('aod_az', set_idx)
    aod_el = get_set_data('aod_el', set_idx)
    toa = get_set_data('toa', set_idx)
    phase = get_set_data('phase', set_idx)
    inter = get_set_data('inter', set_idx)
    inter_pos = get_set_data('inter_pos', set_idx)
    
    # Process tx_pos
    if tx_pos is not None:
        if isinstance(tx_pos, np.ndarray):
            if tx_pos.ndim == 1:
                tx_pos = tx_pos.reshape(1, -1)
        elif isinstance(tx_pos, list):
            try:
                tx_pos = np.array([tx_pos])
            except:
                tx_pos = None
    
    # Process multipath data with path limits
    if inter is not None and hasattr(inter, 'ndim') and inter.ndim >= 2:
        inter = inter[:, :max_paths]
    
    if inter_pos is not None and hasattr(inter_pos, 'ndim') and inter_pos.ndim >= 2:
        inter_pos = inter_pos[:, :max_paths]
    
    # Save all properties
    properties = {
        'rx_pos': rx_pos,
        'tx_pos': tx_pos,
        'power': power,
        'inter': inter,
        'inter_pos': inter_pos,
        'aoa_az': aoa_az,
        'aoa_el': aoa_el,
        'aod_az': aod_az,
        'aod_el': aod_el,
        'toa': toa,
        'phase': phase,
    }
    
    for name, data in properties.items():
        if data is not None:
            try:
                file_path = base_dir / f'{name}_rx_{rx_set_num}_tx_{tx_set_num}.bin'
                save_binary_array(data, file_path)
                print(f"Saved {name} for RX set {rx_set_num}, TX set {tx_set_num}")
            except Exception as e:
                print(f"Error saving {name} for RX set {rx_set_num}, TX set {tx_set_num}: {e}")


def _export_scene_data(dataset, base_dir):
    """Export scene data (buildings, terrain, vegetation)"""
    
    scene = getattr(dataset, 'scene', None)
    if scene is None:
        print("No scene data found")
        return
        
    print(f"Processing scene data")
    try:
        # Get all objects in the scene
        buildings_obj = scene.get_objects('buildings')
        terrain_obj = scene.get_objects('terrain')
        vegetation_obj = scene.get_objects('vegetation')
        
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
            file_path = base_dir / 'buildings.bin'
            save_binary_array(buildings_array, file_path)
            print(f"Saved buildings")

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
            file_path = base_dir / 'terrain.bin'
            save_binary_array(terrain_array, file_path)
            print(f"Saved terrain")

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
            file_path = base_dir / 'vegetation.bin'
            save_binary_array(vegetation_array, file_path)
            print(f"Saved vegetation")
            
    except Exception as e:
        print(f"Error processing scene data: {e}")