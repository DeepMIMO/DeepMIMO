# %% Antenna indices positions

import matplotlib.pyplot as plt

import deepmimo as dm

ants = dm.generator.geometry._ant_indices([6, 4])
plt.scatter(ants[:, 1], ants[:, 2], zorder=3)

# Add labels for each antenna element
for i, (y, z) in enumerate(ants[:, 1:]):
    plt.annotate(f"Ant {i}", (y, z), xytext=(5, 0), textcoords="offset points")

plt.xlabel("y axis")
plt.ylabel("z axis")
plt.title("Antenna Array (6 x 4) Position to Index Mapping")
plt.grid()
plt.show()
