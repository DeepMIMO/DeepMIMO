import os
import re
import shutil
from pathlib import Path


def find_files_to_delete(
    base_path,
    safe_mode=True,
    delete_extra_deepmimo=True,
    delete_extra_p2m=True,
    delete_extra_objs=True,
) -> None:
    """Loop through subfolders and find files/folders to potentially delete.

    Args:
        base_path (str): Base directory to start search
        safe_mode (bool): If True, only print what would be deleted

    """
    # Ensure base path exists
    if not Path(base_path).exists():
        print(f"Base path {base_path} does not exist!")
        return

    exts = [".city", ".ter", ".veg", ".flp", ".object"]
    xml_line_endings = [ext + '"/>' for ext in exts]
    # Get immediate subfolders
    subfolders = [f.path for f in os.scandir(base_path) if f.is_dir()]

    for subfolder in subfolders:
        # print(f"\nProcessing subfolder: {subfolder}")

        # Find all subfolders within this folder
        sub_subfolders = [f.path for f in os.scandir(subfolder) if f.is_dir()]

        if not sub_subfolders:
            print(f"No subfolders found in {subfolder}")
            continue

        # Handle _deepmimo folders
        if len(sub_subfolders) > 1:
            print(f"Warning: Multiple subfolders found in {subfolder}, using first one")
            if delete_extra_deepmimo:
                dm_folder = next(
                    sub_subfolder
                    for sub_subfolder in sub_subfolders
                    if sub_subfolder.endswith(c.DEEPMIMO_CONVERSION_SUFFIX)
                )
                if safe_mode:
                    print(f"Would delete folder: {dm_folder}")
                else:
                    print(f"Deleting folder: {dm_folder}")
                    shutil.rmtree(dm_folder)

        if delete_extra_objs:
            # read xml file and extract files in <Geometry> tags
            files_in_folder = os.scandir(subfolder)
            xml_files = [f for f in files_in_folder if f.name.endswith(".xml")]
            # print if there more than one xml file
            if len(xml_files) > 1:
                print(f"Warning: Multiple xml files found in {subfolder}, using first one")

            xml_file = xml_files[0].path  # Get the actual file path
            print(f"Processing xml file: {xml_file}")

            # Create fresh iterator for files
            files_in_folder = os.scandir(subfolder)

            # [.DIAG]
            # Delete all files that end in .diag but basename is not = xml basename
            for file in files_in_folder:
                if file.name.endswith(".diag") and file.name[:-5] != xml_files[0].name[:-4]:
                    if safe_mode:
                        print(f"Would delete file: {file.path}")
                    else:
                        print(f"Deleting file: {file.path}")
                        Path(file.path).unlink()
                if file.name.endswith(".txrx") or file.name.endswith(".setup"):
                    print(file.name)
            continue
            # [.CITY, .TER, .VEG, .FLP, .OBJECT]
            # Delete all files in the folder that are not referenced in the xml file
            xml_referenced_files = []

            try:
                with Path(xml_file).open() as file:
                    for line in file:
                        line = line.strip()
                        if any(line.endswith(end) for end in xml_line_endings):
                            # Extract value between quotes after Value=
                            if match := re.search(r'Value="./([^"]*)"', line):
                                xml_referenced_files.append(match.group(1))

                # Now check all files in the folder
                for file in os.scandir(subfolder):
                    if file.is_file():
                        if any(file.name.endswith(ext) for ext in exts):
                            if file.name not in xml_referenced_files:
                                if safe_mode:
                                    print(f"Would delete file: {file.path}")
                                else:
                                    print(f"Deleting file: {file.path}")
                                    Path(file.path).unlink()

            except Exception as e:
                print(f"Error processing files: {e}")

        if delete_extra_p2m:
            target_folder = [f.path for f in os.scandir(subfolder) if f.is_dir()][-1]
            print(f"Checking files in: {target_folder}")

            # Find files to delete (those NOT containing .paths. or .pl.)
            for file in os.scandir(target_folder):
                if file.is_file():
                    if ".paths." not in file.name and ".pl." not in file.name:
                        if safe_mode:
                            print(f"Would delete file: {file.path}")
                        else:
                            print(f"Deleting file: {file.path}")
                            Path(file.path).unlink()
        break


if __name__ == "__main__":
    # Run the script with safe mode on
    base_path = r"F:\city_1m_3r_diff+scat_28"
    find_files_to_delete(
        base_path,
        safe_mode=True,
        delete_extra_deepmimo=False,
        delete_extra_p2m=True,
        delete_extra_objs=False,
    )
