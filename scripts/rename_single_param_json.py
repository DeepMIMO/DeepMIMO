# %%
import deepmimo as dm

scenarios = dm.get_available_scenarios()

for scen_name in scenarios:
    if scen_name == "asu_campus":
        continue
    print(f"Processing: {scen_name}")

    params_json_path = dm.get_params_path(scen_name)

    # replace all occurrences of 'txrx' by 'txrx_sets'
    with open(params_json_path) as file:
        text = file.read()

    text = text.replace('"txrx"', '"txrx_sets"')

    # print(text)

    # write back to file
    with open(params_json_path, "w") as file:
        file.write(text)
