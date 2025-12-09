"""Wireless Insite P2M File Parser Module.

This module provides low-level functionality for parsing Wireless Insite .p2m files.
It handles the detailed parsing of path data files, extracting information such as:

- Path angles (arrival/departure)
- Path delays, powers and phases
- Interaction coordinates and types
- Transmitter/receiver positions

The module supports both single and multi-path scenarios, with various interaction
types (reflection, diffraction, scattering, etc). It serves as the foundation for
higher-level path data processing in insite_paths.py.

Main Functions:
    paths_parser(): Parse complete path information from .paths.p2m files
    extract_tx_pos(): Extract transmitter positions from path files
    read_pl_p2m_file(): Read position and path loss data from .pl.p2m files

Note:
    This module is typically not used directly, but rather through the higher-level
    functions in insite_paths.py.

"""

import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

from deepmimo import consts as c
from deepmimo.converters import converter_utils as cu

# Configuration constants for parsing p2m files
LINE_START = 22  # Skip info lines and n_rxs line.

# Interaction Type Map for Wireless Insite
INTERACTIONS_MAP = {
    "R": c.INTERACTION_REFLECTION,  # Reflection
    "D": c.INTERACTION_DIFFRACTION,  # Diffraction
    "DS": c.INTERACTION_SCATTERING,  # Diffuse Scattering
    "T": c.INTERACTION_TRANSMISSION,  # Transmission
    "F": c.INTERACTION_TRANSMISSION,  # Transmission through Leafs/Trees/Vegetation
    "X": c.INTERACTION_TRANSMISSION,  # Transmission through Leafs/Trees/Vegetation
}


def paths_parser(file: str) -> dict[str, np.ndarray]:
    """Parse a Wireless Insite paths.p2m file to extract path information.

    This function reads and processes a .p2m file to extract detailed path information
    for each receiver, including angles, delays, powers, and interaction points.

    Args:
        file (str): Path to the .p2m file to parse

    Returns:
        Dict[str, np.ndarray]: Dictionary containing path information with keys:
            - aoa_az/el: Angles of arrival (azimuth/elevation) (degrees)
            - aod_az/el: Angles of departure (azimuth/elevation) (degrees)
            - toa: Time of arrival (s)
            - power: Received power (dBW)
            - phase: Path phase (degrees)
            - inter: Interaction codes
            - inter_pos: Interaction point positions

    Note:
        1- The function limits the maximum number of paths per receiver to MAX_PATHS(25)
        and the maximum interactions per path to MAX_INTER_PER_PATH(10).
        2- Wireless Insite assumes 0 dBm transmitted power and measures the received power
        in dBm. In DeepMIMO, we assume 0 dBW transmitted power and measure the received
        power in dBW. The relative power values are the same in both cases, so the
        conversion is straightforward.

    """
    # Read file
    print(f"Reading p2m paths file: {Path(file).name}...")
    with Path(file).open() as file_handle:
        lines = file_handle.readlines()

    n_rxs = int(lines[LINE_START - 1])

    # Make data dictionary (to save for each tx set pair)
    data = {
        c.AOA_AZ_PARAM_NAME: np.zeros((n_rxs, c.MAX_PATHS), dtype=np.float32) * np.nan,
        c.AOA_EL_PARAM_NAME: np.zeros((n_rxs, c.MAX_PATHS), dtype=np.float32) * np.nan,
        c.AOD_AZ_PARAM_NAME: np.zeros((n_rxs, c.MAX_PATHS), dtype=np.float32) * np.nan,
        c.AOD_EL_PARAM_NAME: np.zeros((n_rxs, c.MAX_PATHS), dtype=np.float32) * np.nan,
        c.DELAY_PARAM_NAME: np.zeros((n_rxs, c.MAX_PATHS), dtype=np.float32) * np.nan,
        c.POWER_PARAM_NAME: np.zeros((n_rxs, c.MAX_PATHS), dtype=np.float32) * np.nan,
        c.PHASE_PARAM_NAME: np.zeros((n_rxs, c.MAX_PATHS), dtype=np.float32) * np.nan,
        c.INTERACTIONS_PARAM_NAME: np.zeros((n_rxs, c.MAX_PATHS), dtype=np.float32) * np.nan,
        c.INTERACTIONS_POS_PARAM_NAME: np.zeros(
            (n_rxs, c.MAX_PATHS, c.MAX_INTER_PER_PATH, 3),
            dtype=np.float32,
        )
        * np.nan,
    }

    line_idx = LINE_START
    for rx_i in tqdm(range(n_rxs), desc="Processing paths of each RX"):
        line = lines[line_idx]

        # The start of each "path info" is the rx idx and *number of paths*
        rx_n_paths = int(line.split()[1])

        if rx_n_paths == 0:
            line_idx += 1
            continue

        n_paths_to_read = min(rx_n_paths, c.MAX_PATHS)

        line_idx += 2

        for path_idx in range(n_paths_to_read):
            # Line 1 (Example: '1 2 -133.1 31.9 1.7e-06 84.0 40.3 90.9 -13.4\n')
            line = lines[line_idx]
            _i1, i2, i3, i4, i5, i6, i7, i8, i9 = tuple(line.split())
            # (no need) i1 = <path number>
            # (no need) i2 = <total interactions for path> (not including Tx and Rx)
            data[c.POWER_PARAM_NAME][rx_i, path_idx] = np.float32(i3)  # i3 = <received power(dBm)>
            data[c.PHASE_PARAM_NAME][rx_i, path_idx] = np.float32(i4)  # i4 = <phase(deg)>
            data[c.DELAY_PARAM_NAME][rx_i, path_idx] = np.float32(i5)  # i5 = <time of arrival(sec)>
            data[c.AOA_EL_PARAM_NAME][rx_i, path_idx] = np.float32(i6)  # i6 = <arrival theta(deg)>
            data[c.AOA_AZ_PARAM_NAME][rx_i, path_idx] = np.float32(i7)  # i7 = <arrival phi(deg)>
            data[c.AOD_EL_PARAM_NAME][rx_i, path_idx] = np.float32(
                i8,
            )  # i8 = <departure theta(deg)>
            data[c.AOD_AZ_PARAM_NAME][rx_i, path_idx] = np.float32(i9)  # i9 = <departure phi(deg)>

            # Line 2 (Example: "Tx-D-R-Rx")
            line = lines[line_idx + 1]
            inter_strs = line.split("-")[1:-1]  # Example: ['D', 'R']
            # Map to interactions integers ['2', '1'] and join -> '21'
            inter_total_s = "".join([str(INTERACTIONS_MAP[i_str]) for i_str in inter_strs])
            data[c.INTERACTIONS_PARAM_NAME][rx_i, path_idx] = (
                np.float32(inter_total_s) if inter_total_s else 0
            )

            # Line 3-end (Example: "166 104 22")
            n_iteractions = int(i2)
            for inter_idx in range(n_iteractions):
                line = lines[line_idx + 3 + inter_idx]  # skip tx and rx in 1st & last lines
                xyz = [np.float32(i) for i in line.split()]
                data[c.INTERACTIONS_POS_PARAM_NAME][rx_i, path_idx, inter_idx] = xyz

            line_idx += 4 + n_iteractions  # add number of description lines each path has

    # Remove extra paths and bounces
    return cu.compress_path_data(data)


def extract_tx_pos(filename: str) -> np.ndarray:
    """Extract transmitter position from a paths.p2m file.

    This function reads through a .p2m file to find and extract the transmitter
    position coordinates from the first valid path information.

    Args:
        filename (str): Path to the .p2m file to read

    Returns:
        np.ndarray: Array containing transmitter [x, y, z] coordinates of shape (3,)

    Note:
        The function skips receivers with no paths until it finds one with
        valid path information containing the transmitter position.

    """
    # Read file
    print("Reading paths file looking for tx position... ", end="")

    with Path(filename).open() as file:
        for _ in range(LINE_START - 1):  # Skip the first 20 lines
            next(file)

        n_rxs = int(file.readline())

        for _ in range(n_rxs):
            # The start of each "path info" is the rx idx and *number of paths*
            rx_n_paths = int(file.readline().split()[1])

            if rx_n_paths == 0:
                continue

            # Found user with paths!
            for _ in range(3):  # Skip 3 lines with other info
                next(file)

            # Read position
            tx_pos_line = file.readline()
            tx_pos = np.array([float(i) for i in tx_pos_line.split()], dtype=np.float32)
            print("Tx pos found!")
            break

    if "tx_pos" not in locals():
        print(
            "Not found tx_pos. This is likely because there are no paths. \n"
            "Using hack of searching for tx pos in pl file...",
        )

        # Step 1 - swapping tx and rx indices in the filename
        basename = Path(filename).name
        str_list = list(basename)

        # Objective: 'O1_28.paths.t001_21.r014.p2m' -> 'O1_28.paths.t001_14.r021.p2m'

        new_str = str_list[:-11]  # copy first part, e.g. 'O1_28.paths.t001_'
        new_str += str_list[-6:-4]  # put rx number in tx position
        new_str += str_list[-9:-6]  # add middle ('.r0')
        new_str += str_list[-11:-9]  # put tx number in rx position
        new_str += str_list[-4:]  # add end ('.p2m')

        new_basename = "".join(new_str)
        new_filename = str(Path(str(Path(filename).parent)) / new_basename)

        # Step 2 - parse pl file to get tx pos
        try:
            pl_filename = new_filename.replace(".paths", ".pl")
            xyz_matrix, _, _ = read_pl_p2m_file(pl_filename)
            tx_pos = xyz_matrix[0]
            if tx_pos is not None and len(tx_pos) == 0:
                print(f"TX position not found in pl file {pl_filename}")
                tx_pos = None
        except (OSError, ValueError, IndexError) as err:
            print(f"\nWarning: Could not extract TX position from {new_filename}: {err}")
            tx_pos = None

    return tx_pos


def read_pl_p2m_file(filename: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read position and path loss data from a .p2m file.

    Args:
        filename: Path to the .p2m file to read

    Returns:
        Tuple containing:
        - xyz: Array of positions with shape (n_points, 3)
        - dist: Array of distances with shape (n_points, 1)
        - path_loss: Array of path losses with shape (n_points, 1)

    """
    if not filename.endswith(".p2m"):
        msg = "Expected a .p2m file"
        raise ValueError(msg)
    if ".pl." not in filename:
        msg = "Expected pathloss .p2m file containing '.pl.'"
        raise ValueError(msg)

    # Initialize empty lists for matrices
    xyz_list = []
    dist_list = []
    path_loss_list = []

    # Define (regex) patterns to match numbers (optionally signed floats)
    re_data = r"-?\d+\.?\d*"

    with Path(filename).open() as fp:
        lines = fp.readlines()

    for line in lines:
        if line[0] != "#":
            data = re.findall(re_data, line)
            xyz_list.append([float(data[1]), float(data[2]), float(data[3])])  # XYZ (m)
            dist_list.append([float(data[4])])  # distance (m)
            path_loss_list.append([float(data[5])])  # path loss (dB)

    # Convert lists to numpy arrays
    xyz_matrix = np.array(xyz_list, dtype=np.float32)
    dist_matrix = np.array(dist_list, dtype=np.float32)
    path_loss_matrix = np.array(path_loss_list, dtype=np.float32)

    return xyz_matrix, dist_matrix, path_loss_matrix


if __name__ == "__main__":
    file = "./P2Ms/ASU_campus_just_p2m/study_area_asu5/asu_campus.paths.t001_01.r004.p2m"
    file = (
        "./P2Ms/simple_street_canyon/study_rays=0.25_res=2m_3ghz/"
        "simple_street_canyon_test.paths.t001_01.r002.p2m"
    )
    data = paths_parser(file)
    print(data)
