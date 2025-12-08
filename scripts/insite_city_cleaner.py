# %%
import glob
import os
import shutil

import pandas as pd


def safe_delete(path, safe_mode=True) -> None:
    """Helper function to delete files or directories with safe mode support."""
    print(f"{'[SAFE MODE] ' if safe_mode else ''}Would delete: {path}")
    if not safe_mode:
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)


def clean_city_folders(csv_path, base_folder, safe_mode=True) -> None:
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
        bbox_pattern = os.path.join(base_folder, f"bbox_*{bbox_str}*")
        matching_folders = glob.glob(bbox_pattern)

        if matching_folders:
            bbox_folder = matching_folders[0]  # Take the first matching folder
            print(
                f"{'[SAFE MODE] ' if safe_mode else ''}Found matching folder for coordinates {bbox_str}",
            )

            # First, handle the osm folder and other root level files
            print(f"\n{'[SAFE MODE] ' if safe_mode else ''}Cleaning root directory...")
            for item in os.listdir(bbox_folder):
                print(f"item = {item}")
                if item.startswith("insite_"):
                    continue
                safe_delete(os.path.join(bbox_folder, item), safe_mode)

            # Rename the folder
            new_path = os.path.join(base_folder, new_folder_name)
            print(f"{'[SAFE MODE] ' if safe_mode else ''}Would rename: {bbox_folder} -> {new_path}")

            if not safe_mode:
                os.rename(bbox_folder, new_path)
                process_folder_contents(new_path, safe_mode)
            else:
                process_folder_contents(bbox_folder, safe_mode)
        else:
            print(
                f"{'[SAFE MODE] ' if safe_mode else ''}WARNING: No matching folder found for coordinates {bbox_str}",
            )
            continue


def process_folder_contents(folder_path, safe_mode=True) -> None:
    # Find the insite folder
    insite_folder = None
    for item in os.listdir(folder_path):
        if item.startswith("insite"):
            insite_folder = os.path.join(folder_path, item)
            break

    if insite_folder:
        print(f"{'[SAFE MODE] ' if safe_mode else ''}Found insite folder: {insite_folder}")
        # Move all contents from insite folder up one level
        for item in os.listdir(insite_folder):
            src = os.path.join(insite_folder, item)
            dst = os.path.join(folder_path, item)
            print(f"{'[SAFE MODE] ' if safe_mode else ''}Would move: {src} -> {dst}")

            if not safe_mode:
                shutil.move(src, dst)

        if not safe_mode:
            os.rmdir(insite_folder)
        print(
            f"{'[SAFE MODE] ' if safe_mode else ''}Would remove empty insite folder: {insite_folder}",
        )

    # Delete specific folders and files
    items_to_delete = ["intermediate_files", "study_area_mat", "parameters.txt"]
    for item in items_to_delete:
        item_path = os.path.join(folder_path, item)
        if os.path.exists(item_path):
            safe_delete(item_path, safe_mode)


# %%
if __name__ == "__main__":
    # Replace these paths with your actual paths
    csv_path = r"F:\deepmimo_loop_ready\base.csv"  # Path to your CSV file
    base_folder = r"F:\city_1m_3r_diff+scat_28"  # Current directory or specify the path where bbox folders are

    # Run in safe mode by default (True). Set to False to actually perform the operations
    safe_mode = False

    clean_city_folders(csv_path, base_folder, safe_mode)

    if safe_mode:
        print("\n[SAFE MODE] This was a dry run. No files were actually modified.")
        print("To perform the actual operations, set safe_mode = False")
