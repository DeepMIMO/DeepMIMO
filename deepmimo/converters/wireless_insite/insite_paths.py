"""Path data handling for Wireless Insite conversion.

This module provides high-level functionality for processing path data from Wireless Insite
path files (.paths.p2m). It serves as a bridge between the low-level p2m file parsing
and the high-level scenario conversion by:

1. Managing TX/RX point information and mapping
2. Coordinating path data extraction for all TX/RX pairs
3. Saving path data in DeepMIMO format

Dependencies:
- p2m_parser.py: Low-level parsing of .p2m files
- converter_utils.py: Utility functions for data conversion and saving

Main Functions:
    read_paths(): Process and save path data for all TX/RX pairs
    update_txrx_points(): Update TX/RX point information with path data
"""

import os
from pathlib import Path

import numpy as np

from ... import consts as c
from ...general_utils import get_mat_filename, save_mat
from .p2m_parser import extract_tx_pos, paths_parser, read_pl_p2m_file


def update_txrx_points(
    txrx_dict: dict,
    rx_set_id: int,
    rx_pos: np.ndarray,
    path_loss: np.ndarray,
) -> None:
    """Update TxRx set information with point counts and inactive indices.

    Args:
        txrx_dict: Dictionary containing TxRx set information
        rx_set_id: Index of the receiver set to update
        rx_pos: Array of receiver positions
        path_loss: Array of path loss values

    """
    # Update number of points
    n_points = rx_pos.shape[0]
    if txrx_dict[f"txrx_set_{rx_set_id}"]["num_points"] != n_points:
        print(
            f"Warning: Number of points in {rx_set_id} does not match number of points in "
            f"{txrx_dict[f'txrx_set_{rx_set_id}']['id_orig']}",
        )
        txrx_dict[f"txrx_set_{rx_set_id}"]["num_points"] = n_points

    # Find inactive points (those with path loss of 250 dB)
    inactive_idxs = np.where(path_loss == 250.0)[0]
    txrx_dict[f"txrx_set_{rx_set_id}"]["num_active_points"] = n_points - len(inactive_idxs)


def read_paths(rt_folder: str, output_folder: str, txrx_dict: dict) -> None:
    """Create path data from a folder containing Wireless Insite files.

    This function:
    1. Uses provided TX/RX set configurations
    2. Finds all path files for each TX/RX pair
    3. Parses and saves path data for each pair
    4. Extracts and saves position information
    5. Updates TX/RX point information

    Args:
        rt_folder: Path to folder containing .setup, .txrx, and material files
        txrx_dict: Dictionary containing TX/RX set information from read_txrx
        output_folder: Path to folder where .mat files will be saved

    Raises:
        ValueError: If folder doesn't exist or required files not found

    """
    p2m_folder = next(p for p in Path(rt_folder).iterdir() if p.is_dir())
    if not p2m_folder.exists():
        raise ValueError(f"Folder does not exist: {p2m_folder}")

    # Get TX/RX sets from dictionary in a deterministic order
    tx_sets = [txrx_dict[key] for key in sorted(txrx_dict.keys()) if txrx_dict[key]["is_tx"]]
    rx_sets = [txrx_dict[key] for key in sorted(txrx_dict.keys()) if txrx_dict[key]["is_rx"]]

    # Find any p2m file to extract project name (e.g. project_name.paths.t001_01.r001.p2m)
    proj_name = list(p2m_folder.glob("*.p2m"))[0].name.split(".")[0]

    # Calculate total number of TX/RX pairs (for progress tracking)
    n_tot_txs = sum(tx_set["num_points"] for tx_set in tx_sets)
    total_pairs = n_tot_txs * len(rx_sets)
    print(
        f"Found {n_tot_txs} TXs (across {len(tx_sets)} TX sets) and {len(rx_sets)} RXs "
        f"= {total_pairs} TX-RX set pairs",
    )
    processed_pairs = 0

    # Dictionary to store TX positions as we find them
    tx_positions = {}  # key: (tx_set_id, tx_idx), value: position

    for tx_set in tx_sets:
        # Discover number of TX points by checking file existence
        for tx_idx in range(tx_set["num_points"]):
            # Process this TX point with all RX sets
            tx_key = (tx_set["id"], tx_idx)

            for rx_set in rx_sets:
                processed_pairs += 1
                print(
                    f"\rProcessing TX/RX pairs: {processed_pairs}/{total_pairs} "
                    f"({(processed_pairs / total_pairs) * 100:.1f}%)",
                )

                base_filename = f"{proj_name}.paths.t{tx_idx + 1:03}_{tx_set['id_orig']:02}.r{rx_set['id_orig']:03}.p2m"
                paths_p2m_file = p2m_folder / base_filename

                if not paths_p2m_file.exists():
                    raise FileNotFoundError(f"\n P2M path file not found: {paths_p2m_file}")

                # Parse path data
                data = paths_parser(str(paths_p2m_file))

                # Try to extract TX position if we don't have it yet
                if tx_key not in tx_positions:
                    try:
                        tx_pos = extract_tx_pos(str(paths_p2m_file))
                        if tx_pos is not None:
                            tx_positions[tx_key] = tx_pos
                    except Exception as e:
                        print(
                            f"\nWarning: Could not extract TX position from {paths_p2m_file}: {e}",
                        )
                        continue

                # Use stored TX position
                data[c.TX_POS_PARAM_NAME] = tx_positions[tx_key]

                # Extract RX positions and path loss from .pl.p2m file
                pl_p2m_file = str(paths_p2m_file).replace(".paths.", ".pl.")
                data[c.RX_POS_PARAM_NAME], _, path_loss = read_pl_p2m_file(pl_p2m_file)

                # Update TX/RX point information using the set's id
                update_txrx_points(txrx_dict, rx_set["id"], data[c.RX_POS_PARAM_NAME], path_loss)

                # Save each data key using the set's id
                for key in data.keys():
                    mat_file = get_mat_filename(key, tx_set["id"], tx_idx, rx_set["id"])
                    save_mat(data[key], key, os.path.join(output_folder, mat_file))

    # Remove TX sets that have no paths with any receivers
    for tx_set in tx_sets:
        tx_set_has_paths = False
        for tx_idx in range(tx_set["num_points"]):
            if (tx_set["id"], tx_idx) in tx_positions:
                tx_set_has_paths = True
                break
        if not tx_set_has_paths:
            print(
                f"\nWarning: TX set {tx_set['id']} has no paths with any receivers - removing from txrx_dict",
            )
            del txrx_dict[f"txrx_set_{tx_set['id']}"]

    print("\nPath processing completed!")


if __name__ == "__main__":
    # Test directory with path files
    test_dir = r"./P2Ms/simple_street_canyon_test/"
    p2m_folder = r"./P2Ms/simple_street_canyon_test/p2m"
    output_folder = r"./P2Ms/simple_street_canyon_test/mat_files"

    print(f"\nTesting path data extraction from: {test_dir}")
    print("-" * 50)

    # First get TX/RX information
    from .insite_txrx import read_txrx

    txrx_dict = read_txrx(test_dir, p2m_folder, output_folder)

    # Create path data from test directory
    read_paths(p2m_folder, txrx_dict, output_folder)
