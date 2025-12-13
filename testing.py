"""Scratchpad script for DeepMIMO conversion and visualization workflows."""

#%%

import numpy as np
import math

# ============================================================
# Example for DeepMIMO angle convention
#
# Goal:
# - Download and load one DeepMIMO scenario
# - Select 10 LOS UEs
# - Build covariance from frequency-domain channel samples (subcarriers)
# - Run 2D MUSIC twice:
#   (A) using DeepMIMO steering_vec() as provided
#   (B) using a "polar-theta" steering vector consistent with DeepMIMO internal array response
#
# If dataset aoa_el behaves like POLAR theta (measured from +z),
# then (A) will be wrong because steering_vec() assumes input theta is ELEVATION
# and converts theta_polar = pi/2 - theta_elev internally.
# ============================================================

# -------------------------
# 0) Import DeepMIMO steering_vec
# -------------------------

from deepmimo.generator.geometry import steering_vec
DM_STEER = steering_vec


# -------------------------
# 1) DeepMIMO-like antenna indexing for a (H, V) URA panel
#    Matches deepmimo.generator.geometry._ant_indices
# -------------------------
def dm_ant_indices(panel_size_hv):
    H, V = panel_size_hv
    gamma_x = np.tile(np.arange(1), H * V)  # all zeros for x in this panel model
    gamma_y = np.tile(np.repeat(np.arange(H), 1), V)
    gamma_z = np.repeat(np.arange(V), H)
    return np.vstack([gamma_x, gamma_y, gamma_z]).T  # (M, 3)


# -------------------------
# 2) Polar-angle array response phase (matches DeepMIMO _array_response_phase)
#    theta is POLAR angle from +z axis
# -------------------------
def array_response_phase_polar(theta_polar_rad, phi_rad, kd):
    gamma_x = 1j * kd * np.sin(theta_polar_rad) * np.cos(phi_rad)
    gamma_y = 1j * kd * np.sin(theta_polar_rad) * np.sin(phi_rad)
    gamma_z = 1j * kd * np.cos(theta_polar_rad)
    return np.vstack([gamma_x, gamma_y, gamma_z]).T


# -------------------------
# 3) "Correct" steering vector if theta input is already polar
# -------------------------
def steering_vec_polar(panel_size_hv, phi_deg, theta_polar_deg, spacing=0.5):
    ant_ind = dm_ant_indices(panel_size_hv)
    kd = 2 * np.pi * spacing

    phi = np.deg2rad(phi_deg)
    theta_polar = np.deg2rad(theta_polar_deg)

    gamma = array_response_phase_polar(theta_polar, phi, kd)
    a = np.exp(ant_ind @ gamma.T).reshape(-1)
    return a / (np.linalg.norm(a) + 1e-12)


# -------------------------
# 4) MUSIC
# -------------------------
def music_2d(R, panel_size_hv, spacing, d_signal, az_grid_deg, el_grid_deg, steering_func):
    M = panel_size_hv[0] * panel_size_hv[1]
    assert R.shape == (M, M)

    w, V = np.linalg.eigh(R)
    idx = np.argsort(w)[::-1]
    V = V[:, idx]
    En = V[:, d_signal:]
    EnH = En.conj().T

    P = np.zeros((len(el_grid_deg), len(az_grid_deg)), dtype=float)
    for i_el, el in enumerate(el_grid_deg):
        for i_az, az in enumerate(az_grid_deg):
            a = steering_func(panel_size_hv, float(az), float(el), float(spacing))
            denom = np.linalg.norm(EnH @ a) ** 2
            P[i_el, i_az] = 1.0 / (denom + 1e-12)

    imax = np.unravel_index(np.argmax(P), P.shape)
    el_hat = float(el_grid_deg[imax[0]])
    az_hat = float(az_grid_deg[imax[1]])
    return az_hat, el_hat, P


def wrap_to_180_deg(x):
    # wrap difference to [-180, 180)
    x = (x + 180.0) % 360.0 - 180.0
    return x


def circ_abs_deg(a, b):
    # absolute circular difference in degrees for azimuth
    d = abs(wrap_to_180_deg(a - b))
    return d


# -------------------------
# 5) Main: download scenario and test 10 LOS samples
# -------------------------
def run_deepmimo_music_angle_demo(
    scen_name="city_1_losangeles_3p5",
    tx_set_id=1,
    rx_set_id=0,
    ue_shape_hv=(8, 8),
    d_lambda=0.5,
    snr_db=25.0,
    B=20e6,
    N_fft=1024,
    N_pilots=128,
    n_test=10,
    make_heatmap=True,
):
    import deepmimo as dm

    # ---- Download & load (DeepMIMO will cache internally; we do not export/save custom files)
    dm.download(scen_name)
    dataset = dm.load(scen_name, tx_sets=[tx_set_id], rx_sets=[rx_set_id])

    # ---- Configure channel parameters
    ch_params = dm.ChannelParameters()

    # BS: single antenna
    ch_params.bs_antenna.shape = np.array([1, 1])
    ch_params.bs_antenna.spacing = 0.5
    ch_params.bs_antenna.rotation = np.array([0, 0, 0])
    ch_params.bs_antenna.radiation_pattern = "isotropic"

    # UE: URA panel
    ch_params.ue_antenna.shape = np.array(list(ue_shape_hv))
    ch_params.ue_antenna.spacing = d_lambda
    ch_params.ue_antenna.rotation = np.array([0, 0, 0])
    ch_params.ue_antenna.radiation_pattern = "isotropic"

    # Frequency-domain channel on selected subcarriers
    ch_params.freq_domain = True
    ch_params.ofdm.bandwidth = B
    ch_params.ofdm.subcarriers = N_fft
    start = (N_fft - N_pilots) // 2
    ch_params.ofdm.selected_subcarriers = np.arange(start, start + N_pilots)
    ch_params.ofdm.rx_filter = 0

    # Use strongest 1 path in channel generation (still keep original path metadata in dataset)
    ch_params.num_paths = 1

    dataset.set_channel_params(ch_params)
    dataset.compute_channels(ch_params)

    H_all = np.asarray(dataset.channel)   # (N, M_rx, M_tx, Ksel)
    aoa_az = np.asarray(dataset.aoa_az)   # (N, Pmax)
    aoa_el = np.asarray(dataset.aoa_el)   # (N, Pmax)
    power = np.asarray(dataset.power)     # (N, Pmax)

    # LOS flag is computed from inter[:,0] == 0 in this library
    try:
        los = np.asarray(dataset.los)     # (N,)
    except Exception:
        los = None

    N = H_all.shape[0]
    Hdim, Vdim = ue_shape_hv
    M = Hdim * Vdim
    assert H_all.shape[1] == M and H_all.shape[2] == 1

    # ---- Build candidate index list: prefer LOS
    cand = []
    for i in range(N):
        if los is not None and int(los[i]) != 1:
            continue
        # pick the strongest valid path index from metadata (not from generated channel)
        pwr = power[i, :]
        valid = np.isfinite(pwr)
        if not np.any(valid):
            continue
        idx = int(np.nanargmax(pwr))
        if not (np.isfinite(aoa_az[i, idx]) and np.isfinite(aoa_el[i, idx])):
            continue
        cand.append((i, idx))
        if len(cand) >= n_test:
            break

    if len(cand) < n_test:
        print(f"[Warning] Only found {len(cand)} LOS samples (requested {n_test}).")

    # ---- MUSIC grids
    az_grid = np.linspace(0, 360, 361)
    el_grid = np.linspace(0, 90, 91)

    # ---- Define steering wrappers
    def steering_case_A(panel_size_hv, phi_deg, theta_deg, spacing):
        # Case A: use DeepMIMO steering_vec() directly.
        # It assumes input theta is ELEVATION (0=xy-plane, 90=+z) and converts internally.
        return DM_STEER(array=panel_size_hv, phi=phi_deg, theta=theta_deg, spacing=spacing)

    def steering_case_B(panel_size_hv, phi_deg, theta_deg, spacing):
        # Case B: corrected polar-theta steering vector (no elevation->polar conversion).
        return steering_vec_polar(panel_size_hv, phi_deg, theta_deg, spacing)

    rng = np.random.RandomState(2025)

    errs_A = []
    errs_B = []

    # ---- Evaluate 10 samples
    for t, (ue_i, p_i) in enumerate(cand):
        az_true = float(aoa_az[ue_i, p_i])
        el_true = float(aoa_el[ue_i, p_i])

        # Observation across subcarriers (treated as snapshots)
        Y = H_all[ue_i, :, 0, :]  # (M, Ksel)

        # Add noise at target SNR
        P_sig = np.mean(np.abs(Y) ** 2)
        snr_lin = 10 ** (snr_db / 10.0)
        noise_var = P_sig / snr_lin
        sigma = math.sqrt(noise_var / 2.0)
        noise = sigma * (rng.randn(*Y.shape) + 1j * rng.randn(*Y.shape)).astype(np.complex64)
        Y_noisy = Y + noise

        # Covariance
        R = (Y_noisy @ Y_noisy.conj().T) / Y_noisy.shape[1]

        # MUSIC Case A
        az_hat_A, el_hat_A, P_A = music_2d(
            R=R,
            panel_size_hv=ue_shape_hv,
            spacing=d_lambda,
            d_signal=1,
            az_grid_deg=az_grid,
            el_grid_deg=el_grid,
            steering_func=steering_case_A,
        )

        # MUSIC Case B
        az_hat_B, el_hat_B, P_B = music_2d(
            R=R,
            panel_size_hv=ue_shape_hv,
            spacing=d_lambda,
            d_signal=1,
            az_grid_deg=az_grid,
            el_grid_deg=el_grid,
            steering_func=steering_case_B,
        )

        # Errors
        da_A = circ_abs_deg(az_hat_A, az_true)
        de_A = abs(el_hat_A - el_true)
        da_B = circ_abs_deg(az_hat_B, az_true)
        de_B = abs(el_hat_B - el_true)

        errs_A.append((da_A, de_A))
        errs_B.append((da_B, de_B))

        print(
            f"[{t:02d}] UE={ue_i} | TRUE az={az_true:7.2f}, el={el_true:7.2f} | "
            f"A: az={az_hat_A:7.2f}, el={el_hat_A:7.2f}, err=({da_A:5.2f},{de_A:5.2f}) | "
            f"B: az={az_hat_B:7.2f}, el={el_hat_B:7.2f}, err=({da_B:5.2f},{de_B:5.2f})"
        )

        # Optional: heatmaps for the first sample only
        if make_heatmap and t == 0:
            import matplotlib.pyplot as plt

            plt.figure()
            plt.title("Case A: DeepMIMO steering_vec() (potential mismatch)")
            plt.imshow(
                10 * np.log10(P_A + 1e-12),
                origin="lower",
                aspect="auto",
                extent=[az_grid[0], az_grid[-1], el_grid[0], el_grid[-1]],
            )
            plt.xlabel("Azimuth (deg)")
            plt.ylabel("El (deg)")
            plt.colorbar()
            plt.scatter([az_true], [el_true], marker="x")
            plt.scatter([az_hat_A], [el_hat_A], marker="o")
            plt.tight_layout()

            plt.figure()
            plt.title("Case B: Polar-theta steering vector (consistent with internal array response)")
            plt.imshow(
                10 * np.log10(P_B + 1e-12),
                origin="lower",
                aspect="auto",
                extent=[az_grid[0], az_grid[-1], el_grid[0], el_grid[-1]],
            )
            plt.xlabel("Azimuth (deg)")
            plt.ylabel("El (deg)")
            plt.colorbar()
            plt.scatter([az_true], [el_true], marker="x")
            plt.scatter([az_hat_B], [el_hat_B], marker="o")
            plt.tight_layout()

            plt.show()

    # ---- Summary
    if len(errs_A) > 0:
        mean_da_A = float(np.mean([e[0] for e in errs_A]))
        mean_de_A = float(np.mean([e[1] for e in errs_A]))
        mean_da_B = float(np.mean([e[0] for e in errs_B]))
        mean_de_B = float(np.mean([e[1] for e in errs_B]))

        print("\n=== Summary over tested samples ===")
        print(f"Case A mean errors: az={mean_da_A:.3f} deg, el={mean_de_A:.3f} deg")
        print(f"Case B mean errors: az={mean_da_B:.3f} deg, el={mean_de_B:.3f} deg")


if __name__ == "__main__":
    run_deepmimo_music_angle_demo(
        scen_name="city_1_losangeles_3p5",
        tx_set_id=1,
        rx_set_id=0,
        ue_shape_hv=(8, 8),
        d_lambda=0.5,
        snr_db=25.0,
        B=20e6,
        N_fft=1024,
        N_pilots=128,
        n_test=10,
        make_heatmap=True,   # set False if you only want text output
    )

#%%
dataset = dm.load('city_1_losangeles_3p5', tx_sets=[1], rx_sets=[0])

#%%
import deepmimo as dm
import matplotlib.pyplot as plt
params = dm.ChannelParameters()

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5), tight_layout=True)

# Define 3 different rotations to show
rotations = [
    np.array([0, 0, 0]),  # Facing +x
    np.array([0, 0, 180]),  # Facing -x
    np.array([0, 0, -135]),
]  # Facing 45º between -x and -y

titles = [
    "Orientation along +x (0°)",
    "Orientation along -x (180°)",
    "Orientation at 45º between -x and -y (-135°)",
]

# Plot each azimuth rotation
for i, (rot, title) in enumerate(zip(rotations, titles, strict=False)):
    # Update channel parameters with new rotation
    params.bs_antenna.rotation = rot
    dataset.set_channel_params(params)  # safest way to set params

    # Create coverage plot in current subplot
    dm.plot_coverage(
        dataset.rx_pos,
        dataset.los,
        bs_pos=dataset.tx_pos.T,
        bs_ori=dataset.tx_ori,
        ax=axes[i],
        title=title,
        cbar_title="LoS status",
    )

# %% [markdown]
# #### Elevation

# %%
params = dm.ChannelParameters()

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5), subplot_kw={"projection": "3d"}, tight_layout=True)

# Define 3 different rotations to show
rotations = [
    np.array([0, 0, -180]),  # Facing -x
    np.array([0, 30, -180]),  # Facing 30º below -x in XZ plane
    np.array([0, 60, -180]),
]  # Facing 60º below -x in XZ plane

titles = [
    "Orientation along -x (180°)",
    "Orientation at 30º between -x and -z",
    "Orientation at 60º between -x and -z",
]

# Plot each azimuth rotation
for i, (rot, title) in enumerate(zip(rotations, titles, strict=False)):
    # Update channel parameters with new rotation
    params.bs_antenna.rotation = rot
    dataset.set_channel_params(params)

    # Create coverage plot in current subplot
    dataset.plot_coverage(
        dataset.los,
        proj_3D=True,
        ax=axes[i],
        title=title,
        cbar_title="LoS status",
    )
    axes[i].view_init(elev=5, azim=-90)  # Set view to xz plane to see tilt
    axes[i].set_yticks([])  # Remove y-axis ticks to unclutter the plot

# %% [markdown]
# ## Advanced Operations

# %% [markdown]
# ### Field-of-View

# %% [markdown]
# #### Azimuth

# %%
# First plot with no FoV filtering (full coverage)
dataset.plot_coverage(dataset.los)

# %%
params = dm.ChannelParameters()
params["bs_antenna"]["rotation"] = np.array([0, 0, -135])
dataset.set_channel_params(params)

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5), tight_layout=True)

# Define 3 FoV
fovs = [
    np.array([180, 180]),  # Facing -x
    np.array([90, 180]),  # Facing 30º below -x in XZ plane
    np.array([60, 180]),
]  # Facing 60º below -x in XZ plane

titles = [f"FoV = {fov[0]} x {fov[1]}°" for fov in fovs]

# Plot each FoV setting
for i, (fov, title) in enumerate(zip(fovs, titles, strict=False)):
    print(f"Iteration {i}: Setting FoV to {fov}")
    # Create a temporary dataset with FoV applied
    dataset_fov = dataset.trim_by_fov(bs_fov=fov)
    dataset_fov.plot_coverage(dataset_fov.los, ax=axes[i], title=title, cbar_title="LoS status")

# Note: trim_by_fov returns a new dataset with paths outside FoV removed
# The original dataset remains unchanged

# %%

# %% [markdown]
# #### Elevation

# %%
params = dm.ChannelParameters()
params["bs_antenna"]["rotation"] = np.array([0, 30, -135])
dataset.set_channel_params(params)

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5), tight_layout=True)

# Define 3 FoV
fovs = [
    np.array([360, 90]),  # Facing -x
    np.array([360, 45]),  # Facing 30º below -x in XZ plane
    np.array([360, 30]),
]  # Facing 60º below -x in XZ plane

titles = [f"FoV = {fov[0]} x {fov[1]}°" for fov in fovs]

# Plot each FoV setting
for i, (fov, title) in enumerate(zip(fovs, titles, strict=False)):
    print(f"Iteration {i}: Setting FoV to {fov}")
    # Create a temporary dataset with FoV applied
    dataset_fov = dataset.trim_by_fov(bs_fov=fov)
    dataset_fov.plot_coverage(dataset_fov.los, ax=axes[i], title=title, cbar_title="LoS status")

# Note: trim_by_fov returns a new dataset with paths outside FoV removed
# To see path information affected by fov, index arrays with: dataset.los != -1
