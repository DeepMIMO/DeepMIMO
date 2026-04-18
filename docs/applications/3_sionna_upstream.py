"""DeepMIMO → Sionna: Link-Level Simulation."""
# %% [markdown]
# # DeepMIMO → Sionna: Link-Level Simulation.
#
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DeepMIMO/DeepMIMO/blob/main/docs/applications/3_sionna_upstream.py)
# &nbsp;
# [![GitHub](https://img.shields.io/badge/Open_on-GitHub-181717?logo=github&style=for-the-badge)](https://github.com/DeepMIMO/DeepMIMO/blob/main/docs/applications/3_sionna_upstream.py)
#
# ---
#
# **What this notebook covers:**
# 1. Load a DeepMIMO scenario and select active users
# 2. Compute time-domain per-path channel coefficients
# 3. Use `SionnaAdapter` to export `(a, tau)` in Sionna's expected format
# 4. Build the OFDM channel matrix `H[k]` from path gains and delays
# 5. Compute per-user spectral efficiency and plot a CDF
#
# **Why this workflow?**
# DeepMIMO captures full multipath geometry from ray tracing.
# `SionnaAdapter` bridges that data to Sionna's `(a, tau)` format so you can
# plug realistic channels into any Sionna-compatible link-level pipeline without
# re-running the ray tracer.
#
# **Requirements:**
# ```bash
# pip install deepmimo        # core (no Sionna needed for this notebook)
# pip install 'deepmimo[sionna]'  # if you also want sionna.rt utilities
# ```
#
# ---

# %%
# %pip install deepmimo  # uncomment if not installed

# %% [markdown]
# ## Imports

# %%
import numpy as np

import deepmimo as dm
from deepmimo.integrations.sionna_adapter import SionnaAdapter

# %% [markdown]
# ## Configuration

# %%
SCENARIO   = "asu_campus_3p5"  # ASU campus, 3.5 GHz
N_PATHS    = 5                  # per-path channels (sets channel resolution)
N_UE       = 500                # number of UEs to simulate

# OFDM system parameters
SUBCARRIERS   = 64              # number of subcarriers
SUBCAR_SPACING = 15e3           # 15 kHz (4G/5G NR SCS)
SNR_DB_RANGE  = np.arange(-10, 31, 5)  # SNR sweep [dB]

# %% [markdown]
# ## Load Scenario

# %%
dm.download(SCENARIO)
dataset_full = dm.load(SCENARIO)
print(dataset_full)

# %% [markdown]
# ## Select Active Users
#
# Not all grid positions have line-of-sight or reflected paths. We keep only
# users with at least one active multipath component.

# %%
active_idxs = np.where(np.array(dataset_full.num_paths) > 0)[0]
print(f"Active users: {len(active_idxs)} / {dataset_full.n_ue}")

# Sample N_UE users spread across the active set
rng = np.random.default_rng(42)
sample_idxs = rng.choice(active_idxs, size=min(N_UE, len(active_idxs)), replace=False)
sample_idxs.sort()

dataset = dataset_full.trim(idxs=sample_idxs)
print(f"Using {dataset.n_ue} users for simulation")

# %% [markdown]
# ## Compute Time-Domain Per-Path Channels
#
# `freq_domain=False` keeps the full multipath representation: the last
# dimension of `channels` is the path index, each entry being the complex
# channel gain for that path.

# %%
ch_params = dm.ChannelParameters()
ch_params.freq_domain = False
ch_params.num_paths   = N_PATHS

dataset.compute_channels(ch_params)

channels = np.array(dataset.channels)  # (N_UE, n_rx_ant, n_tx_ant, N_PATHS)
toas     = np.array(dataset.toa)       # (N_UE, max_paths)
n_paths_per_ue = np.array(dataset.num_paths)

print(f"Channel array shape: {channels.shape}")
print(f"  (n_ue={channels.shape[0]}, n_rx_ant={channels.shape[1]}, "
      f"n_tx_ant={channels.shape[2]}, n_paths={channels.shape[3]})")

# %% [markdown]
# ## SionnaAdapter: Export (a, tau)
#
# `SionnaAdapter` converts the DeepMIMO dataset into Sionna's standard
# per-path format:
#
# - **`a`** — complex channel coefficients:
#   shape `(num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps)`
# - **`tau`** — path delays [s]:
#   shape `(num_rx, num_tx, num_paths)`
#
# This is the exact input format used by Sionna's OFDM channel model.

# %%
adapter = SionnaAdapter(dataset)
print(f"Adapter channel shape: {adapter.ch_shape}")
print(f"Adapter delay shape:   {adapter.t_shape}")
print(f"Total samples:         {len(adapter)}")

# Collect all (a, tau) pairs from the generator
all_a   = []
all_tau = []
for a_sample, tau_sample in adapter():
    all_a.append(a_sample.copy())
    all_tau.append(tau_sample.copy())

all_a   = np.concatenate(all_a,   axis=0)  # (N_UE, 1, 1, n_tx_ant, N_PATHS, 1)
all_tau = np.concatenate(all_tau, axis=0)  # (N_UE, 1, N_PATHS)

print(f"\nCollected a shape:   {all_a.shape}")
print(f"Collected tau shape: {all_tau.shape}")

# %% [markdown]
# ## Build OFDM Channel Matrix
#
# Given per-path gains `a_l` and delays `tau_l`, the channel at subcarrier
# frequency `f_k` is:
#
# $$H[k] = \sum_{l=0}^{L-1} a_l \cdot e^{-j 2\pi f_k \tau_l}$$
#
# We compute this for all users and all subcarriers to obtain the full
# OFDM channel tensor `H` of shape `(N_UE, n_rx_ant, n_tx_ant, N_subcarriers)`.

# %%
# Subcarrier frequencies centred around 0 (baseband)
subcar_freqs = (np.arange(SUBCARRIERS) - SUBCARRIERS // 2) * SUBCAR_SPACING  # (K,)

# a: (N_UE, n_rx_ant, n_tx_ant, N_PATHS, 1)  →  broadcast over time step
# tau: (N_UE, 1, N_PATHS)  →  delays per UE/path
a_ue   = all_a[:, 0, 0, :, :, 0]   # (N_UE, n_tx_ant, N_PATHS)
tau_ue = all_tau[:, 0, :]           # (N_UE, N_PATHS)

# Phase shift: exp(-j*2*pi*f_k*tau_l) → (N_UE, N_PATHS, K)
phase = np.exp(-1j * 2 * np.pi * tau_ue[:, :, np.newaxis] * subcar_freqs[np.newaxis, np.newaxis, :])

# Sum over paths: H[u, ant, k] = sum_l a[u, ant, l] * phase[u, l, k]
# a_ue: (N_UE, n_tx_ant, N_PATHS) → (N_UE, n_tx_ant, N_PATHS, 1)
H = np.einsum("ual,ulk->uak", a_ue, phase)  # (N_UE, n_tx_ant, K)

print(f"OFDM channel matrix H shape: {H.shape}")
print("  (n_ue, n_tx_ant, n_subcarriers)")

# %% [markdown]
# ## Spectral Efficiency (Capacity)
#
# For each user and SNR level, compute the achievable rate assuming
# independent OFDM subcarriers with matched-filter combining at the receiver:
#
# $$C = \frac{1}{K} \sum_{k=1}^{K} \log_2 \left(1 + \text{SNR} \cdot \|h_k\|^2\right)$$

# %%
# Per-subcarrier channel power (summed over TX antennas): (N_UE, K)
H_power = np.sum(np.abs(H) ** 2, axis=1)  # (N_UE, K)

# Capacity vs SNR sweep
capacities = np.zeros((len(SNR_DB_RANGE), dataset.n_ue))
for snr_idx, snr_db in enumerate(SNR_DB_RANGE):
    snr_linear = 10 ** (snr_db / 10)
    # Average over subcarriers
    capacities[snr_idx] = np.mean(np.log2(1 + snr_linear * H_power), axis=1)

mean_cap = capacities.mean(axis=1)
print("\nMean spectral efficiency vs SNR:")
print(f"  {'SNR (dB)':>10}  {'Capacity (bit/s/Hz)':>20}")
for snr_db, cap in zip(SNR_DB_RANGE, mean_cap, strict=False):
    print(f"  {snr_db:>10.0f}  {cap:>20.2f}")

# %% [markdown]
# ## CDF of Spectral Efficiency
#
# Plot the CDF at a fixed SNR to show the per-user distribution across the
# scenario.

# %%
import matplotlib.pyplot as plt  # noqa: E402

PLOT_SNR_DB = 20
snr_idx = np.searchsorted(SNR_DB_RANGE, PLOT_SNR_DB)
se_at_snr = capacities[snr_idx]

sorted_se = np.sort(se_at_snr)
cdf = np.arange(1, len(sorted_se) + 1) / len(sorted_se)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(sorted_se, cdf, linewidth=2)
axes[0].set_xlabel("Spectral Efficiency [bit/s/Hz]")
axes[0].set_ylabel("CDF")
axes[0].set_title(f"CDF of SE at SNR = {PLOT_SNR_DB} dB\n({SCENARIO}, {dataset.n_ue} UEs)")
axes[0].grid(visible=True, alpha=0.3)
axes[0].set_xlim(left=0)

axes[1].plot(SNR_DB_RANGE, mean_cap, marker="o", linewidth=2)
axes[1].set_xlabel("SNR [dB]")
axes[1].set_ylabel("Mean SE [bit/s/Hz]")
axes[1].set_title(f"Mean Spectral Efficiency vs SNR\n({SCENARIO})")
axes[1].grid(visible=True, alpha=0.3)

plt.tight_layout()
plt.savefig("sionna_upstream_se.png", dpi=100, bbox_inches="tight")
plt.show()

print(f"\nMedian SE at {PLOT_SNR_DB} dB SNR: {np.median(se_at_snr):.2f} bit/s/Hz")
print(f"Mean   SE at {PLOT_SNR_DB} dB SNR: {np.mean(se_at_snr):.2f} bit/s/Hz")

# %% [markdown]
# ## Summary
#
# The **DeepMIMO → Sionna** upstream pipeline:
#
# | Step | Tool | Output |
# |------|------|--------|
# | 1. Load scenario | `dm.load` | `Dataset` with paths |
# | 2. Per-path channels | `compute_channels(freq_domain=False)` | `a` array |
# | 3. Adapt to Sionna format | `SionnaAdapter` | `(a, tau)` per UE |
# | 4. OFDM channel | manual einsum | `H[u, ant, k]` |
# | 5. Capacity | Shannon formula | SE CDF / SNR curve |
#
# The `(a, tau)` output from `SionnaAdapter` is directly compatible with
# Sionna's OFDM channel models when the full `sionna` package is installed,
# enabling end-to-end differentiable link-level simulation on DeepMIMO data.
