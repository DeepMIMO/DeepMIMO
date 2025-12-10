"""Utility to rename txrx keys to txrx_sets in params.json files."""

# %%
from pathlib import Path

import deepmimo as dm

scenarios = dm.get_available_scenarios()

for scen_name in scenarios:
    if scen_name == "asu_campus":
        continue
    print(f"Processing: {scen_name}")

    params_json_path = dm.get_params_path(scen_name)

    # replace all occurrences of 'txrx' by 'txrx_sets'
    with Path(params_json_path).open() as file:
        text = file.read()

    text = text.replace('"txrx"', '"txrx_sets"')

    # print(text)

    # write back to file
    with Path(params_json_path).open("w") as file:
        file.write(text)
