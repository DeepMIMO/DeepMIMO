# %%
import os
import shutil


def move_insite_folder_out(folder, safe_mode=True):
    insite_path = os.path.join(folder, "insite")

    if os.path.isdir(insite_path):
        for item in os.listdir(insite_path):
            src = os.path.join(insite_path, item)
            dst = os.path.join(folder, item)
            print(f"Moving {src} to {dst}")
            if not safe_mode:
                shutil.move(src, dst)
        print(f"Removing {insite_path}")
        if not safe_mode:
            os.rmdir(insite_path)


# %%
move_insite_folder_out("P2Ms/city_21_taito_city_3p5")

# %%

loop_folder = "P2Ms/new_scenarios_backup"

for folder in os.listdir(loop_folder):
    move_insite_folder_out(os.path.join(loop_folder, folder), safe_mode=False)

# %%
