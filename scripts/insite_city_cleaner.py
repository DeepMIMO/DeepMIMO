# %%
"""Utilities for cleaning InSite-generated city folders."""

import shutil
from pathlib import Path

import pandas as pd


def safe_delete(path: str | Path, *, safe_mode: bool = True) -> None:
    """Delete a file or directory unless running in safe mode."""
    print(f"{'[SAFE MODE] ' if safe_mode else ''}Would delete: {path}")
    if not safe_mode:
        if Path(path).is_dir():
            shutil.rmtree(path)
        else:
            Path(path).unlink()


def clean_city_folders(
    csv_path: str | Path,
    base_folder: str | Path,
    *,
    safe_mode: bool = True,
) -> None:
    """Clean generated city folders using CSV mapping and optional safe mode."""
    print(f"{'[SAFE MODE] ' if safe_mode else ''}Starting folder processing...")

    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Iterate through each row in the CSV
    for _index, row in df.iterrows():
        city_name = row["name"].lower()  # Get the first column value and convert to lowercase
        new_folder_name = f"{city_name}_28"

        # Construct bbox string from coordinates
        bbox_str = (
            f"{row['bbox_minlat']}_{row['bbox_minlon']}_{row['bbox_maxlat']}_{row['bbox_maxlon']}"
        )
        bbox_str = bbox_str.replace(".", "-")
        bbox_pattern = str(Path(base_folder) / f"bbox_*{bbox_str}*")
        matching_folders = list(Path().glob(bbox_pattern))

        if matching_folders:
            bbox_folder = matching_folders[0]  # Take the first matching folder
            prefix = "[SAFE MODE] " if safe_mode else ""
            print(f"{prefix}Found matching folder for coordinates {bbox_str}")

            # First, handle the osm folder and other root level files
            print(f"\n{'[SAFE MODE] ' if safe_mode else ''}Cleaning root directory...")
            for item in [p.name for p in Path(bbox_folder).iterdir()]:
                print(f"item = {item}")
                if item.startswith("insite_"):
                    continue
                safe_delete(str(Path(bbox_folder) / item), safe_mode)

            # Rename the folder
            new_path = str(Path(base_folder) / new_folder_name)
            print(f"{'[SAFE MODE] ' if safe_mode else ''}Would rename: {bbox_folder} -> {new_path}")

            if not safe_mode:
                Path(bbox_folder).rename(new_path)
                process_folder_contents(new_path, safe_mode)
            else:
                process_folder_contents(bbox_folder, safe_mode)
        else:
            prefix = "[SAFE MODE] " if safe_mode else ""
            print(f"{prefix}WARNING: No matching folder found for coordinates {bbox_str}")
            continue


def process_folder_contents(folder_path: str | Path, *, safe_mode: bool = True) -> None:
    """Inspect folder for InSite artifacts and optionally remove them."""
    # Find the insite folder
    insite_folder = None
    for item in [p.name for p in Path(folder_path).iterdir()]:
        if item.startswith("insite"):
            insite_folder = str(Path(folder_path) / item)
            break

    if insite_folder:
        print(f"{'[SAFE MODE] ' if safe_mode else ''}Found insite folder: {insite_folder}")
        # Move all contents from insite folder up one level
        for item in [p.name for p in Path(insite_folder).iterdir()]:
            src = str(Path(insite_folder) / item)
            dst = str(Path(folder_path) / item)
            print(f"{'[SAFE MODE] ' if safe_mode else ''}Would move: {src} -> {dst}")

            if not safe_mode:
                shutil.move(src, dst)

        if not safe_mode:
            Path(insite_folder).rmdir()
        prefix = "[SAFE MODE] " if safe_mode else ""
        print(f"{prefix}Would remove empty insite folder: {insite_folder}")

    # Delete specific folders and files
    items_to_delete = ["intermediate_files", "study_area_mat", "parameters.txt"]
    for item in items_to_delete:
        item_path = str(Path(folder_path) / item)
        if Path(item_path).exists():
            safe_delete(item_path, safe_mode)


# %%
if __name__ == "__main__":
    # Replace these paths with your actual paths
    csv_path = r"F:\deepmimo_loop_ready\base.csv"  # Path to your CSV file
    base_folder = r"F:\city_1m_3r_diff+scat_28"
    # Current directory or specify the path where bbox folders are

    # Run in safe mode by default (True). Set to False to actually perform the operations
    safe_mode = False

    clean_city_folders(csv_path, base_folder, safe_mode)

    if safe_mode:
        print("\n[SAFE MODE] This was a dry run. No files were actually modified.")
        print("To perform the actual operations, set safe_mode = False")
