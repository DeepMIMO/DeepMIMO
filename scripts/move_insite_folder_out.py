# %%
"""Utility to move InSite folders out of scenario directories."""

import shutil
from pathlib import Path


def move_insite_folder_out(folder: str | Path, *, safe_mode: bool = True) -> None:
    """Move contents of an 'insite' subfolder up one level."""
    insite_path = str(Path(folder) / "insite")

    if Path(insite_path).is_dir():
        for item in [p.name for p in Path(insite_path).iterdir()]:
            src = str(Path(insite_path) / item)
            dst = str(Path(folder) / item)
            print(f"Moving {src} to {dst}")
            if not safe_mode:
                shutil.move(src, dst)
        print(f"Removing {insite_path}")
        if not safe_mode:
            Path(insite_path).rmdir()


# %%
move_insite_folder_out("P2Ms/city_21_taito_city_3p5")

# %%

loop_folder = "P2Ms/new_scenarios_backup"

for folder in [p.name for p in Path(loop_folder).iterdir()]:
    move_insite_folder_out(str(Path(loop_folder) / folder), safe_mode=False)

# %%
