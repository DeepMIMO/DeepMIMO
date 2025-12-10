"""AODT Transmitter/Receiver Configuration Module.

This module handles reading and processing:
1. Radio Unit (RU) configurations from rus.parquet
2. User Equipment (UE) configurations from ues.parquet
3. Antenna panel configurations from panels.parquet
4. Antenna patterns from patterns.parquet
"""

from pathlib import Path
from typing import Any

import numpy as np

from deepmimo.core.txrx import TxRxSet

from . import aodt_utils as au
from .safe_import import pd

# AODT antenna pattern types and their descriptions
PATTERN_TYPES = {
    0: "isotropic",  # Radiates equally in all directions
    1: "infinitesimal",  # Theoretical point source
    2: "halfwave dipole",  # Standard λ/2 dipole antenna
    3: "rectangular microstrip",  # Patch antenna
    # ≥100: Custom pattern
}

PATTERN_TYPE_HALF_WAVE = 2
CUSTOM_PATTERN_THRESHOLD = 100


def read_panels(rt_folder: str) -> dict[str, Any]:
    """Read antenna panel configurations.

    Args:
        rt_folder (str): Path to folder containing panels.parquet.

    Returns:
        dict[str, Any]: Dictionary mapping panel IDs to configurations.

    """
    panels_file = str(Path(rt_folder) / "panels.parquet")
    if not Path(panels_file).exists():
        return {}

    df = pd.read_parquet(panels_file)
    if len(df) == 0:
        return {}

    panels = {}
    for _, panel in df.iterrows():
        panel_dict = {
            "name": panel["panel_name"],
            "antenna_names": panel["antenna_names"],
            "pattern_indices": panel["antenna_pattern_indices"],
            "frequencies": np.array(panel["frequencies"]),
            "thetas": np.array(panel["thetas"]),
            "phis": np.array(panel["phis"]),
            "reference_freq": float(panel["reference_freq"]),
            "dual_polarized": bool(panel["dual_polarized"]),
            "array_config": {
                "num_horz": int(panel["num_loc_antenna_horz"]),
                "num_vert": int(panel["num_loc_antenna_vert"]),
                "spacing_horz": float(panel["antenna_spacing_horz"]),
                "spacing_vert": float(panel["antenna_spacing_vert"]),
                "roll_angle_first": float(panel["antenna_roll_angle_first_polz"]),
                "roll_angle_second": float(panel["antenna_roll_angle_second_polz"]),
            },
        }
        panels[panel["panel_id"]] = panel_dict
    return panels


def convert_to_deepmimo_txrxset(tx_rx_data: dict[str, Any], *, is_tx: bool, id_: int) -> TxRxSet:
    """Convert AODT TX/RX data to DeepMIMO TxRxSet format.

    Args:
        tx_rx_data (dict[str, Any]): Dictionary containing TX/RX configuration
        is_tx (bool): Whether this is a transmitter (True) or receiver (False)
        id_ (int): Unique identifier for the DeepMIMO set

    Returns:
        TxRxSet: DeepMIMO format TX/RX set

    """
    # Get panel configuration
    panel = tx_rx_data["panel"]
    array_config = panel["array_config"]

    # Calculate total number of antenna elements
    num_ant = array_config["num_horz"] * array_config["num_vert"]
    if panel["dual_polarized"]:
        num_ant *= 2

    # Create antenna positions array
    ant_positions = []
    for i in range(array_config["num_horz"]):
        for j in range(array_config["num_vert"]):
            x = i * array_config["spacing_horz"]
            y = j * array_config["spacing_vert"]
            ant_positions.append([x, y, 0])
            if panel["dual_polarized"]:
                ant_positions.append([x, y, 0])  # Same position, different polarization

    # Get number of points (default to 1 for transmitters)
    num_points = tx_rx_data.get("num_points", 1)

    # Create TxRxSet
    return TxRxSet(
        name=f"{'tx' if is_tx else 'rx'}_{tx_rx_data['id']}",
        id_orig=tx_rx_data["id"],
        id=id_,
        is_tx=is_tx,
        is_rx=not is_tx,
        num_points=num_points,  # Use actual number of points
        num_active_points=num_points,  # Initially all points are active
        num_ant=num_ant,
        dual_pol=panel["dual_polarized"],
        ant_rel_positions=ant_positions,
        array_orientation=[
            float(tx_rx_data["mech_azimuth"]),
            float(tx_rx_data["mech_tilt"]),
            float(panel["array_config"]["roll_angle_first"]),
        ],
    )


def validate_isotropic_patterns(rt_folder: str, panels: dict[str, Any]) -> None:
    """Validate that all antenna patterns are isotropic (type 0).

    This function performs two checks:
    1. Each panel must have exactly one pattern ID
    2. Each unique pattern ID must correspond to an isotropic pattern

    Pattern types in AODT:
        0: Isotropic - Radiates equally in all directions
        1: Infinitesimal - Theoretical point source
        2: Halfwave Dipole - Standard λ/2 dipole antenna
        3: Rectangular Microstrip - Patch antenna
        ≥100: Custom pattern

    Args:
        rt_folder (str): Path to folder containing patterns.parquet
        panels (dict[str, Any]): Dictionary of panel configurations

    Raises:
        FileNotFoundError: If patterns.parquet is not found
        ValueError: If:
            - patterns.parquet is empty
            - Any panel has multiple pattern IDs
            - Any pattern ID is not found
            - Any pattern is not isotropic (type != 0)

    """
    # Read patterns file
    patterns_file = str(Path(rt_folder) / "patterns.parquet")
    if not Path(patterns_file).exists():
        msg = f"patterns.parquet not found in {rt_folder}"
        raise FileNotFoundError(msg)

    patterns_df = pd.read_parquet(patterns_file)
    if len(patterns_df) == 0:
        msg = "patterns.parquet is empty"
        raise ValueError(msg)

    # Check that each panel has exactly one pattern ID
    for panel_id, panel in panels.items():
        unique_pattern_ids = np.unique(panel["pattern_indices"])
        if len(unique_pattern_ids) != 1:
            msg = f"Panel {panel_id} has {len(unique_pattern_ids)} patterns. Expected exactly 1."
            raise ValueError(
                msg,
            )

    # Get all unique pattern IDs across all panels
    unique_pattern_ids = {
        panel["pattern_indices"][0]  # Safe to use [0] after above check
        for panel in panels.values()
    }

    # Check that each pattern ID corresponds to an isotropic pattern
    for pattern_id in unique_pattern_ids:
        pattern = patterns_df[patterns_df["pattern_id"] == pattern_id]
        if len(pattern) == 0:
            msg = f"Pattern ID {pattern_id} not found in patterns.parquet"
            raise ValueError(msg)

        pattern_type = pattern.iloc[0]["pattern_type"]
        if pattern_type == PATTERN_TYPE_HALF_WAVE:
            print(
                f"WARNING: Pattern ID {pattern_id} uses halfwave dipole antenna (type=2)."
                "Ray tracing results may be inaccurate.",
            )
        elif pattern_type != 0:
            pattern_desc = PATTERN_TYPES.get(
                pattern_type,
                "custom" if pattern_type >= CUSTOM_PATTERN_THRESHOLD else "unknown",
            )
            msg = (
                f"Pattern ID {pattern_id} uses {pattern_desc} antenna (type={pattern_type}). "
                "Only isotropic antennas (type=0) are supported."
            )
            raise ValueError(
                msg,
            )


def read_transmitters(rt_folder: str, panels: dict[str, Any]) -> list[dict[str, Any]]:
    """Read and process Radio Units (RUs) from parquet file.

    Args:
        rt_folder (str): Path to folder containing rus.parquet
        panels (dict[str, Any]): Dictionary of panel configurations

    Returns:
        list[dict[str, Any]]: List of processed transmitter configurations

    Raises:
        FileNotFoundError: If rus.parquet is not found
        ValueError: If rus.parquet is empty

    """
    # Read RUs file
    rus_file = str(Path(rt_folder) / "rus.parquet")
    if not Path(rus_file).exists():
        msg = f"rus.parquet not found in {rt_folder}"
        raise FileNotFoundError(msg)

    rus_df = pd.read_parquet(rus_file)
    if len(rus_df) == 0:
        msg = "rus.parquet is empty"
        raise ValueError(msg)

    # Process RUs
    transmitters = []
    for _, ru in rus_df.iterrows():
        tx = {
            "id": int(ru["ID"]),
            "position": np.array(ru["position"]),
            "power": float(ru["radiated_power"]),
            "mech_tilt": float(ru["mech_tilt"]),
            "mech_azimuth": float(ru["mech_azimuth"]),
            "scs": int(ru["subcarrier_spacing"]),
            "fft_size": int(ru["fft_size"]),
            "panel": next(panels[i] for i in ru["panel"]),  # use only first panel
            "du_id": int(ru["du_id"]) if not pd.isna(ru["du_id"]) else None,
            "du_manual_assign": bool(ru["du_manual_assign"]),
        }
        transmitters.append(tx)
    return transmitters


def read_receivers(rt_folder: str, panels: dict[str, Any]) -> list[dict[str, Any]]:
    """Read and process User Equipment (UEs) from parquet file.

    Args:
        rt_folder (str): Path to folder containing ues.parquet
        panels (dict[str, Any]): Dictionary of panel configurations

    Returns:
        list[dict[str, Any]]: List of processed receiver configurations

    Raises:
        FileNotFoundError: If ues.parquet is not found
        ValueError: If ues.parquet is empty

    """
    # Read UEs file
    ues_file = str(Path(rt_folder) / "ues.parquet")
    if not Path(ues_file).exists():
        msg = f"ues.parquet not found in {rt_folder}"
        raise FileNotFoundError(msg)

    ues_df = pd.read_parquet(ues_file)
    if len(ues_df) == 0:
        msg = "ues.parquet is empty"
        raise ValueError(msg)

    # Process UEs
    receivers = []
    for _, ue in ues_df.iterrows():
        # Get initial orientation (azimuth) from route if available
        # route_orientations = ue['route_orientations'][time_idx]

        # # Convert orientations to array using aodt_utils
        # orientations_array = au.process_points(route_orientations)
        # # Take first orientation's azimuth
        # initial_azimuth = float(orientations_array[0, 0])
        initial_azimuth = 0.0  # Temporary placeholder until orientations clarified.
        # The orientations are not clear...
        # raise issue if ue has multiple panels
        if len(ue["panel"]) > 1:
            msg = f"UE {ue['ID']} has multiple panels: {ue['panel']}"
            raise ValueError(msg)

        rx = {
            "id": int(ue["ID"]),
            "is_manual": bool(ue["is_manual"]),
            "is_manual_mobility": bool(ue["is_manual_mobility"]),
            "power": float(ue["radiated_power"]),
            "height": float(ue["height"]),
            "mech_tilt": float(ue["mech_tilt"]),
            "mech_azimuth": initial_azimuth,  # Using initial orientation from route
            "panel": next(panels[i] for i in ue["panel"]),
            "indoor": bool(ue["is_indoor_mobility"]),
            "bler_target": float(ue["bler_target"]),
            "mobility": {
                "batch_indices": np.array(ue["batch_indices"]),
                "waypoints": {
                    "ids": np.array(ue["waypoint_ids"]),
                    "points": np.array(ue["waypoint_points"]),
                    "stops": np.array(ue["waypoint_stops"]),
                    "speeds": np.array(ue["waypoint_speeds"]),
                },
                "trajectory": {
                    "ids": np.array(ue["trajectory_ids"]),
                    "points": np.array(ue["trajectory_points"]),
                    "stops": np.array(ue["trajectory_stops"]),
                    "speeds": np.array(ue["trajectory_speeds"]),
                },
                "route": {
                    "positions": np.array(ue["route_positions"]),
                    "orientations": np.array(ue["route_orientations"]),
                    "speeds": np.array(ue["route_speeds"]),
                    "times": np.array(ue["route_times"]),
                },
            },
        }
        receivers.append(rx)
    return receivers


def is_dynamic(rt_params: dict[str, Any]) -> bool:
    """Check if the scenario is dynamic.

    Args:
        rt_params (dict[str, Any]): Ray tracing parameters dictionary.

    Returns:
        bool: True if the scenario is dynamic, False otherwise.

    """
    # check batches
    # Example raw params: slots_per_batch/symbols_per_slot/duration/interval
    raw_params = rt_params.get("raw_params", {})
    if raw_params.get("slots_per_batch", 0) > 0:  # use slots per batch timing
        return raw_params.get("num_batches", 1) * raw_params.get("slots_per_batch", 1) > 1
    # use duration and interval timing
    return raw_params.get("duration", 0) // raw_params.get("interval", 0) > 1


def read_txrx(rt_folder: str, rt_params: dict[str, Any]) -> dict[str, Any]:
    """Read transmitter and receiver configurations.

    Args:
        rt_folder (str): Path to folder containing configuration files.
        rt_params (dict[str, Any]): Ray tracing parameters dictionary.

    Returns:
        dict[str, Any]: Dictionary containing TX/RX configurations in DeepMIMO format.

    Raises:
        FileNotFoundError: If required files are not found.
        ValueError: If required parameters are missing.

    """
    # Read antenna configurations
    panels = read_panels(rt_folder)

    # Validate that all patterns are isotropic
    validate_isotropic_patterns(rt_folder, panels)

    # Update frequency in rt_params from first panel
    first_panel_id = next(iter(panels))
    rt_params["frequency"] = panels[first_panel_id]["reference_freq"]

    # Read and process transmitters and receivers
    transmitters = read_transmitters(rt_folder, panels)
    receivers = read_receivers(rt_folder, panels)

    # Convert to DeepMIMO format
    txrx_dict = {}

    # Convert transmitters
    for i, tx in enumerate(transmitters):
        txrx_set = convert_to_deepmimo_txrxset(tx, is_tx=True, id_=i)
        txrx_dict[f"txrx_set_{i}"] = txrx_set.to_dict()

    # Convert all receivers into a single set
    rx_start_id = len(transmitters)

    # Use first receiver's data for configuration
    first_rx = receivers[0]

    # Get initial positions from route data (time_idx = 0)
    time_idx = int(Path(rt_folder).name.split("_")[-1]) if is_dynamic(rt_params) else 0

    rx_positions = [
        au.process_points(rx["mobility"]["route"]["positions"][0])[time_idx] for rx in receivers
    ]

    rx_data = {
        "id": first_rx["id"],  # Use first RX's ID
        "panel": first_rx["panel"],
        "mech_tilt": first_rx["mech_tilt"],
        "mech_azimuth": first_rx["mech_azimuth"],
        "positions": rx_positions,  # Store all positions from route data
        "num_points": len(receivers),  # Total number of receivers
    }

    txrx_set = convert_to_deepmimo_txrxset(rx_data, is_tx=False, id_=rx_start_id)
    txrx_dict[f"txrx_set_{rx_start_id}"] = txrx_set.to_dict()

    return txrx_dict
