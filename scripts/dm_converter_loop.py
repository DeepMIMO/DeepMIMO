# %% Imports

import json
import os
import time
from pathlib import Path

import deepmimo as dm

# Main execution
EXECUTION_MODE = "collect_errors"  # 'collect_errors' | 'retry_errors'
ERROR_LOG_FILE = "conversion_errors.json"

# base_path = "F:/deepmimo_loop_ready"
base_path = r"M:\AutoRayTracingSionna\all_runs_sionna\run_04-07-2025_18H13M23S"
subfolders = [f.path for f in os.scandir(base_path) if f.is_dir()]

# Load previous errors if in retry mode
error_scenarios = []
if EXECUTION_MODE == "retry_errors" and Path(ERROR_LOG_FILE).exists():
    with Path(ERROR_LOG_FILE).open() as f:
        error_scenarios_to_retry = {item[0] for item in json.load(f)}
    subfolders = [f for f in subfolders if Path(f).name in error_scenarios_to_retry]

# %%

timing_results = {}

# For conversion
# for subfolder in subfolders:
#     scen_name = Path(subfolder).name
#     print(f'running: {subfolder}')

# For zipping
for scen_name in dm.get_available_scenarios():
    if not scen_name.startswith("city_") or not scen_name.endswith("_s"):
        continue

    start = time.time()
    try:
        # scen_name = dm.convert(subfolder, overwrite=True, scenario_name=scen_name,
        #                        vis_scene=True, print_params=False)
        # print(f"Conversion successful: {scen_name}")
        stop = time.time()

        # dm.upload(scen_name, key=DEEPMIMO_API_KEY)
        # print(f"Upload successful: {scen_name}")

        # dm.upload_images(scen_name, key=DEEPMIMO_API_KEY, img_paths=[img_path])
        # break

        # print(f"Upload successful: {scen_name}")
    except Exception as e:
        stop = start
        if EXECUTION_MODE == "retry_errors":
            raise  # Re-raise the exception in retry mode
        error_scenarios.append((scen_name, str(e)))
        if EXECUTION_MODE == "collect_errors" and error_scenarios:
            with Path(ERROR_LOG_FILE).open("w") as f:
                json.dump(error_scenarios, f, indent=2)

    timing_results[scen_name] = stop - start

# %%
if timing_results:
    print("\nPer-scenario timing:")
    print("-" * 50)
    print(f"{'Scenario':<30} | {'Time (s)':<10}")
    print("-" * 50)
    for scen, duration in sorted(timing_results.items(), key=lambda x: x[1], reverse=True):
        print(f"{scen:<30} | {duration:>10.2f}")

if error_scenarios:
    print(f"\nErrors ({len(error_scenarios)}):")
    for name, err in error_scenarios:
        print(f"{name:<30} | {err[:47]}")
    print(f"\nError scenarios saved to {ERROR_LOG_FILE}")

# Cleanup in retry mode if all successful
if EXECUTION_MODE == "retry_errors" and not error_scenarios and Path(ERROR_LOG_FILE).exists():
    Path(ERROR_LOG_FILE).unlink()
    print("All retries successful - error log removed")
