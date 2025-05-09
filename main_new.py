import numpy as np
from tqdm import tqdm
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, PathSolver
from sionna.rt.utils import subcarrier_frequencies
from deepmimo.converter.sionna_rt import sionna_exporter
import deepmimo as dm
import sionna.rt
import scipy.io as sio
import os
import shutil
import matplotlib.pyplot as plt

# Parameters
carrier_freq = 3.5e9  # Hz
max_depth = 5
num_users = 5  # Number of users
user_batch_size = 1  # Number of users processed at once
random_seed = 12345

# Set up the scene
scene = load_scene(sionna.rt.scene.munich)
scene.frequency = carrier_freq
scene.seed = random_seed

# Set up transmitter (4x4 array)
tx = Transmitter(name="tx", position=[8.5, 21, 27], display_radius=2)
tx.array = PlanarArray(
    num_rows=4,
    num_cols=4,
    vertical_spacing=0.5,
    horizontal_spacing=0.5,
    pattern="iso",
    polarization="V"
)
scene.add(tx)
scene.tx_array = tx.array  # Assign to both for compatibility

# Generate user grid (users in a line)
user_positions = np.array([[45 + i, 90, 1.5] for i in range(num_users)])

# Add receivers to the scene
for idx, pos in enumerate(user_positions):
    rx = Receiver(name=f"rx_{idx}", position=pos, display_radius=2)
    scene.add(rx)
    scene.receivers[f"rx_{idx}"].rx_array = PlanarArray(
        num_rows=4,
        num_cols=4,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V"
    )

# Explicitly set scene.rx_array after adding all receivers
scene.rx_array = PlanarArray(
    num_rows=4,
    num_cols=4,
    vertical_spacing=0.5,
    horizontal_spacing=0.5,
    pattern="iso",
    polarization="V"
)

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

H_dict = {}
for synth_flag, folder in zip([False, True], ["sionna_test_scen_non_synth", "sionna_test_scen_synth"]):
    print(f"\n===== synthetic_array={synth_flag} case start =====")
    path_params = base_path_params.copy()
    path_params['synthetic_array'] = synth_flag

    # Place all receivers at the correct positions
    for idx, pos in enumerate(user_positions):
        scene.receivers[f"rx_{idx}"].position = pos

    # Process all users at once with the path solver
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

    # Compute the final channel matrix (H) (narrowband, single frequency)
    a = paths['a'] if isinstance(paths, dict) else paths.a
    if isinstance(a, tuple):
        a = a[0]
    print(f"a.shape: {a.shape}")
    tau = paths['tau'] if isinstance(paths, dict) else paths.tau
    if isinstance(tau, tuple):
        tau = tau[0]
    print(f"synthetic_array={synth_flag} a.shape: {a.shape}, tau.shape: {tau.shape}")
    num_users_ = a.shape[0]
    num_tx = a.shape[1]
    num_tx_ant = a.shape[3]
    num_paths = a.shape[4]

    if num_tx != 1:
        print(f"Warning: Number of transmitters (num_tx) is {num_tx}, which is different from expected. Only the first TX will be used.")

    H = np.zeros((num_users_, num_tx_ant), dtype=np.complex64)
    fc = carrier_freq
    for u in range(num_users_):
        for ta in range(num_tx_ant):
            amps_np = np.array(a[u, 0, 0, ta, :])  # Use t=0 only
            if tau.shape == a.shape:
                delays_np = np.array(tau[u, 0, 0, ta, :])
            else:
                delays_np = np.array(tau[u, 0, :])
            valid = np.abs(amps_np) > 0
            amps_np = amps_np[valid]
            delays_np = delays_np[valid]
            if len(amps_np) > 0:
                H[u, ta] = np.sum(amps_np * np.exp(-1j * 2 * np.pi * fc * delays_np))
    print(f"H.shape: {H.shape}, num_users: {num_users_}, num_tx_ant: {num_tx_ant}")
    H_dict[synth_flag] = H
    print(f"synthetic_array={synth_flag} H Frobenius norm: {np.linalg.norm(H)}")
    # Visualize H magnitude map
    plt.figure()
    plt.imshow(np.abs(H), aspect='auto', interpolation='none')
    plt.colorbar(label='|H|')
    plt.title(f'H magnitude map (synthetic_array={synth_flag})')
    plt.xlabel('TX Antenna Index')
    plt.ylabel('User Index')
    plt.tight_layout()
    plt.savefig(f'H_map_synth_{synth_flag}.png')
    plt.close()

# Compare and visualize the difference of H between the two options
if (False in H_dict) and (True in H_dict):
    H_diff = H_dict[False] - H_dict[True]
    print(f"H(False) - H(True) Frobenius norm: {np.linalg.norm(H_diff)}")
    H_diff_abs = np.abs(H_diff)
    H_diff_map = H_diff_abs.reshape(num_users, -1)
    plt.figure()
    plt.imshow(H_diff_map, aspect='auto', interpolation='none')
    plt.colorbar(label='|H(False) - H(True)|')
    plt.title('H Difference Map')
    plt.xlabel('TX Antenna Index')
    plt.ylabel('User Index')
    plt.tight_layout()
    plt.savefig('H_diff_map.png')
    plt.close()

print("Number of transmitters:", len(scene.transmitters))
for tx_name in scene.transmitters:
    tx_obj = scene.transmitters[tx_name]
    print(tx_obj.name, tx_obj.position)
