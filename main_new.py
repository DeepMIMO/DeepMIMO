import numpy as np
import os
import shutil
import deepmimo as dm
import matplotlib.pyplot as plt
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, PathSolver
import sionna.rt.scene as scene_lib

# Parameters
carrier_freq = 3.5e9  # Hz
max_depth = 5
num_users = 5
random_seed = 12345

# Temporary output folders for RT results
rt_base = "/tmp/sionna_rt_synth_compare"
rt_folder_off = os.path.join(rt_base, "non_synthetic")
rt_folder_on  = os.path.join(rt_base, "synthetic")

# Clean up previous results
for folder in [rt_folder_off, rt_folder_on]:
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

# Sionna scene setup (common)
scene = load_scene(scene_lib.munich)
scene.frequency = carrier_freq
scene.seed = random_seed

# Set up transmitter (4x4 array)
tx = Transmitter(name="tx", position=[8.5, 21, 27], display_radius=2)
tx.array = PlanarArray(num_rows=4, num_cols=4, vertical_spacing=0.5, horizontal_spacing=0.5, pattern="iso", polarization="V")
scene.add(tx)

# Set up receivers (users in a line)
user_positions = np.array([[45 + i, 90, 1.5] for i in range(num_users)])
for idx, pos in enumerate(user_positions):
    rx = Receiver(name=f"rx_{idx}", position=pos, display_radius=2)
    scene.add(rx)
    scene.receivers[f"rx_{idx}"].rx_array = PlanarArray(num_rows=4, num_cols=4, vertical_spacing=0.5, horizontal_spacing=0.5, pattern="iso", polarization="V")

scene.rx_array = PlanarArray(num_rows=4, num_cols=4, vertical_spacing=0.5, horizontal_spacing=0.5, pattern="iso", polarization="V")

# Path computation parameters (common)
base_path_params = dict(
    max_depth=max_depth,
    los=True,
    specular_reflection=True,
    diffuse_reflection=False,
    refraction=True,
    seed=41
)

# PathSolver instance
p_solver = PathSolver()

# Run Sionna RT simulation for both synthetic_array off and on
rt_folders = {False: rt_folder_off, True: rt_folder_on}
datasets = {}

def is_sionna_v1():
    try:
        import sionna
        if hasattr(sionna, '__version__'):
            version_str = sionna.__version__
        elif hasattr(sionna.rt, '__version__'):
            version_str = sionna.rt.__version__
        else:
            print("Warning: Could not determine Sionna version, assuming v1.x+.")
            return True
        return int(version_str.split('.')[0]) >= 1
    except Exception as e:
        print(f"Warning: Sionna version check failed ({e}), assuming v1.x+.")
        return True

for synth_flag in [False, True]:
    print(f"\n===== Sionna RT simulation: synthetic_array={synth_flag} =====")
    path_params = base_path_params.copy()
    path_params['synthetic_array'] = synth_flag

    for idx, pos in enumerate(user_positions):
        scene.receivers[f"rx_{idx}"].position = pos

    # Ensure transmitter array is set before each PathSolver call
    tx.array = PlanarArray(num_rows=4, num_cols=4, vertical_spacing=0.5, horizontal_spacing=0.5, pattern="iso", polarization="V")
    scene.tx_array = tx.array

    paths = p_solver(
        scene=scene,
        max_depth=path_params['max_depth'],
        los=path_params['los'],
        specular_reflection=path_params['specular_reflection'],
        diffuse_reflection=path_params['diffuse_reflection'],
        refraction=path_params['refraction'],
        synthetic_array=path_params['synthetic_array'],
        seed=path_params['seed']
    )

    from deepmimo.converter.sionna_rt import sionna_exporter
    sionna_exporter.export_to_deepmimo(scene, [paths], path_params, rt_folders[synth_flag])

    scen_name = dm.convert(rt_folders[synth_flag], overwrite=True)
    datasets[synth_flag] = dm.load(scen_name)[0]

H_off = datasets[False]['channel']
H_on  = datasets[True]['channel']

if H_off is not None and H_on is not None:
    print("H_off shape:", H_off.shape)
    print("H_on shape:", H_on.shape)
    print("Frobenius norm (off):", np.linalg.norm(H_off))
    print("Frobenius norm (on):", np.linalg.norm(H_on))

    H_off_plot = np.abs(H_off[:, 0, :, 0])
    H_on_plot  = np.abs(H_on[:, 0, :, 0])

    plt.figure()
    plt.imshow(H_off_plot, aspect='auto', interpolation='none')
    plt.colorbar(label='|H_off|')
    plt.title('H magnitude map (synthetic_array=off)')
    plt.xlabel('TX Antenna Index')
    plt.ylabel('User Index')
    plt.tight_layout()
    plt.savefig('H_map_deepmimo_off.png')
    plt.close()

    plt.figure()
    plt.imshow(H_on_plot, aspect='auto', interpolation='none')
    plt.colorbar(label='|H_on|')
    plt.title('H magnitude map (synthetic_array=on)')
    plt.xlabel('TX Antenna Index')
    plt.ylabel('User Index')
    plt.tight_layout()
    plt.savefig('H_map_deepmimo_on.png')
    plt.close()

    H_diff = H_off_plot - H_on_plot
    plt.figure()
    plt.imshow(np.abs(H_diff), aspect='auto', interpolation='none')
    plt.colorbar(label='|H_off - H_on|')
    plt.title('H Difference Map (DeepMIMO)')
    plt.xlabel('TX Antenna Index')
    plt.ylabel('User Index')
    plt.tight_layout()
    plt.savefig('H_diff_map_deepmimo.png')
    plt.close()
else:
    print("No H in one or both datasets. Check if the scenarios were converted and loaded correctly.")