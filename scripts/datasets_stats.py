import deepmimo as dm

scen_list = dm.get_available_scenarios()

scen_name_list = []
n_ue_list = []
n_active_list = []
for scen in scen_list:
    if "city_" in scen:
        city_idx = int(scen.split("_")[1])
        if city_idx > 20:
            print(scen)
            dataset = dm.load(scen)[0]
            n_ue_list.append(dataset.n_ue)
            n_active_list.append(len(dataset.get_active_idxs()))
            scen_name_list.append(scen)

# Print formatted table
print("\nScenario Statistics:")
print("-" * 65)
print(f"{'Scenario':<32} | {'Total':<8} | {'Active':<8} | {'Ratio':<8}")
print("-" * 65)
for scen, n_ue, n_active in zip(scen_name_list, n_ue_list, n_active_list, strict=False):
    ratio = n_active / n_ue if n_ue > 0 else 0
    print(f"{scen:<32} | {n_ue:>8} | {n_active:>8} | {ratio:>8.0%}")
print("-" * 65)

# Print averages
avg_total = sum(n_ue_list) / len(n_ue_list)
avg_active = sum(n_active_list) / len(n_active_list)
avg_ratio = avg_active / avg_total
print(f"{'AVERAGE':<32} | {avg_total:>8.0f} | {avg_active:>8.0f} | {avg_ratio:>8.0%}")
print("-" * 65)
