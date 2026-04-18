"""DeepMIMO → Sionna: 5G NR PUSCH BLER Simulation."""
# %% [markdown]
# # DeepMIMO → Sionna: 5G NR PUSCH BLER Simulation
#
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DeepMIMO/DeepMIMO/blob/main/docs/applications/3_sionna_upstream.py)
# &nbsp;
# [![GitHub](https://img.shields.io/badge/Open_on-GitHub-181717?logo=github&style=for-the-badge)](https://github.com/DeepMIMO/DeepMIMO/blob/main/docs/applications/3_sionna_upstream.py)
#
# ---
#
# **What this notebook covers:**
# 1. Load a DeepMIMO ray-tracing scenario and extract per-path channel data
# 2. Wire realistic DeepMIMO channels into Sionna's `CIRDataset`
# 3. Run a complete 5G NR PUSCH link-level simulation with Sionna's PHY stack
# 4. Plot a BLER curve and compare with a Rayleigh fading baseline
#
# **Why this workflow?**
# DeepMIMO provides realistic multipath channels from ray-tracing.
# Sionna's `CIRDataset` accepts any generator that yields `(a, tau)` tuples
# and feeds them through a full 5G NR receiver chain — no manual OFDM math
# required.  The resulting BLER curve reflects the *fading structure* of the
# real environment, not just a simplified statistical model.
#
# **Requirements:**
# ```bash
# pip install deepmimo
# pip install 'deepmimo[sionna]'   # sionna-rt
# pip install sionna-no-rt          # Sionna PHY layer (PUSCH, LDPC, OFDM, …)
# ```
#
# ---

# %%
# %pip install deepmimo sionna-no-rt  # uncomment if not installed

# %% [markdown]
# ## Imports

# %%
from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import sionna.phy
import torch
from sionna.phy.channel import CIRDataset, OFDMChannel, RayleighBlockFading
from sionna.phy.nr import PUSCHConfig, PUSCHReceiver, PUSCHTransmitter
from sionna.phy.utils import compute_ber, ebnodb2no, sim_ber

import deepmimo as dm

if TYPE_CHECKING:
    from collections.abc import Generator

sionna.phy.config.seed = 42                    # reproducible Monte Carlo

# %% [markdown]
# ## Configuration

# %%
SCENARIO   = "asu_campus_3p5"   # ASU campus, 3.5 GHz
N_PATHS    = 5                   # max multipath components to use
N_BS_ANT   = 8                   # BS receive antennas (1-RX SIMO uplink)

# Simulation parameters — increase for smoother, more accurate curves
BATCH_SIZE         = 64           # UEs per Monte Carlo step
MAX_MC_ITER        = 100          # max batches per SNR point
NUM_TARGET_ERRORS  = 100          # stop early once this many block errors collected

# Eb/N0 sweep — values sent to sim_ber
EBNO_DB_RANGE = np.arange(-5, 26, 2.5)

# %% [markdown]
# ## Load DeepMIMO Scenario
#
# We load per-path complex coefficients (`a`) and delays (`tau`) for each UE.
# Path loss is baked into the raw coefficients; we normalise it out below
# because link-level simulation controls SNR via the noise variance only.

# %%
dm.download(SCENARIO)
dataset = dm.load(SCENARIO)

ch_params = dm.ChannelParameters()
ch_params.freq_domain = False     # time-domain: gives per-path (a, tau)
ch_params.num_paths   = N_PATHS

dataset.compute_channels(ch_params)

# Keep only UEs that have at least one active path
active_idxs = np.where(np.array(dataset.num_paths) > 0)[0]
dataset     = dataset.trim(idxs=active_idxs)
print(f"Active UEs: {dataset.n_ue:,}")

# Extract arrays once so the generator is fast
all_channels = np.array(dataset.channels)  # [n_ue, 1, N_BS_ANT, N_PATHS]
all_toas     = np.array(dataset.toa)        # [n_ue, max_paths]
all_num_paths = np.array(dataset.num_paths) # [n_ue]

print(f"Channel array: {all_channels.shape}  (n_ue, n_ue_ant, n_bs_ant, n_paths)")

# %% [markdown]
# ## Build a 5G NR PUSCH Transmitter / Receiver
#
# We use Sionna's default `PUSCHConfig` (15 kHz SCS, QPSK, rate-1/2 LDPC,
# one 5G NR slot).  The transmitter produces frequency-domain resource grids;
# the receiver runs LS channel estimation, LMMSE equalisation, and LDPC
# decoding.

# %%
pusch_config = PUSCHConfig()          # 5G NR defaults: 1 UE, 1 TX antenna
pusch_tx = PUSCHTransmitter(pusch_config, output_domain="freq")
pusch_rx = PUSCHReceiver(pusch_tx,    input_domain="freq")

resource_grid = pusch_tx.resource_grid
print(f"Subcarrier spacing : {resource_grid.subcarrier_spacing/1e3:.0f} kHz")
print(f"OFDM symbols       : {resource_grid.num_ofdm_symbols}")
print(f"Subcarriers        : {resource_grid.fft_size}")
print(f"Bandwidth          : {resource_grid.bandwidth/1e6:.2f} MHz")
# _tb_size is not part of the public API yet — derive it from one forward pass
_, _b_probe = pusch_tx(1)
print(f"Transport block    : {_b_probe.shape[-1]} bits")

# %% [markdown]
# ## Create a DeepMIMO CIR Generator
#
# Sionna's `CIRDataset` wraps any Python generator that yields `(a, tau)`:
#
# - **`a`** — complex path coefficients, shape
#   `[num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]`
# - **`tau`** — path delays in seconds, shape `[num_rx, num_tx, num_paths]`
#
# Our uplink setup: UE is the single-antenna transmitter (TX), BS is the
# 8-antenna receiver (RX).  DeepMIMO stores the *downlink* channel (BS → UE),
# so we apply channel reciprocity: H_UL = H_DL^H (conjugate transpose).
#
# We also **normalise** each channel realisation to unit total power so that
# the noise variance `no` fully controls SNR — the standard convention for
# link-level BLER simulation.

# %%
def deepmimo_ul_cir_gen(rng_seed: int = 0) -> Generator[tuple[torch.Tensor, torch.Tensor]]:
    """Infinite generator: yields uplink CIR tuples from the DeepMIMO dataset.

    Each call draws a random UE, converts its downlink channel to uplink via
    reciprocity, normalises it to unit power, and packs it into the shape
    expected by CIRDataset.
    """
    n_ue = dataset.n_ue
    rng  = np.random.default_rng(rng_seed)
    # Shuffle once; cycle through the dataset infinitely so sim_ber never
    # runs out of UEs regardless of batch size or iteration count.
    order = rng.permutation(n_ue).tolist()
    cursor = 0

    while True:
        i = order[cursor % n_ue]
        cursor += 1

        h_dl  = all_channels[i]                       # [1, N_BS_ANT, N_PATHS]
        # Reciprocity: h_ul[rx_bs, tx_ue] = conj(h_dl[rx_ue, tx_bs])
        h_ul  = np.conj(h_dl.transpose(1, 0, 2))      # [N_BS_ANT, 1, N_PATHS]
        n_act = min(int(all_num_paths[i]), N_PATHS)

        # Normalise to unit power so SNR = 1/no
        pwr = np.sum(np.abs(h_ul[:, :, :n_act]) ** 2)
        if pwr > 0:
            h_ul = h_ul / np.sqrt(pwr)

        # Pack into CIRDataset shape
        a   = np.zeros((1, N_BS_ANT, 1, 1, N_PATHS, 1), dtype=np.csingle)
        tau = np.zeros((1, 1, N_PATHS),                   dtype=np.single)
        a[0, :, 0, :, :n_act, 0] = h_ul[:, :, :n_act]
        tau[0, 0, :n_act]         = all_toas[i, :n_act]

        yield (torch.from_numpy(a).to(torch.complex64),
               torch.from_numpy(tau).to(torch.float32))


# Wrap the generator in a CIRDataset so it behaves like any Sionna channel model
dm_cir_dataset = CIRDataset(
    cir_generator = deepmimo_ul_cir_gen,
    batch_size    = BATCH_SIZE,
    num_rx        = 1,
    num_rx_ant    = N_BS_ANT,
    num_tx        = 1,
    num_tx_ant    = 1,          # single-antenna UE
    num_paths     = N_PATHS,
    num_time_steps= 1,
)

dm_channel = OFDMChannel(dm_cir_dataset, resource_grid)

# Also build a Rayleigh block-fading baseline for comparison
rayleigh_model   = RayleighBlockFading(num_rx=1, num_rx_ant=N_BS_ANT,
                                        num_tx=1, num_tx_ant=1)
rayleigh_channel = OFDMChannel(rayleigh_model, resource_grid)

# %% [markdown]
# ## Define the Monte Carlo Function
#
# `sim_ber` expects a callable `mc_fun(batch_size, ebno_db) -> (b, b_hat)`.
# We build one factory that wires the chosen channel into the PUSCH pipeline.

# %%
def make_mc_fun(channel: torch.nn.Module) -> object:
    """Return a Monte Carlo step function that uses the given channel block."""
    def mc_fun(batch_size: int, ebno_db: float) -> tuple[torch.Tensor, torch.Tensor]:
        # Convert Eb/N0 [dB] to noise variance per complex symbol
        no = ebnodb2no(ebno_db,
                       pusch_config.tb.num_bits_per_symbol,
                       pusch_config.tb.target_coderate,
                       resource_grid)
        x, b    = pusch_tx(batch_size)    # transmit resource grid + info bits
        y       = channel(x, no)          # apply fading channel + AWGN noise
        b_hat   = pusch_rx(y, no)         # LS estimation + LMMSE + LDPC decode
        return b, b_hat
    return mc_fun

# Quick sanity check at 10 dB before running the full sweep.
# Note: CIRDataset has a fixed batch size, so we must use BATCH_SIZE here.
no_test = ebnodb2no(10.0, pusch_config.tb.num_bits_per_symbol,
                    pusch_config.tb.target_coderate, resource_grid)
x_t, b_t = pusch_tx(BATCH_SIZE)
y_t       = dm_channel(x_t, no_test)
b_hat_t   = pusch_rx(y_t, no_test)
print(f"Sanity check at 10 dB Eb/N0: BER = {compute_ber(b_t, b_hat_t).item():.4f}")

# %% [markdown]
# ## Run BLER Simulation
#
# `sim_ber` collects block errors at each SNR point until `NUM_TARGET_ERRORS`
# are seen or `MAX_MC_ITER` batches are exhausted, then moves to the next
# SNR point.  Early stopping kicks in once BLER drops to zero.

# %%
print("Simulating DeepMIMO channel BLER ...")
ber_dm, bler_dm = sim_ber(
    mc_fun                  = make_mc_fun(dm_channel),
    ebno_dbs                = torch.tensor(EBNO_DB_RANGE, dtype=torch.float32),
    batch_size              = BATCH_SIZE,
    max_mc_iter             = MAX_MC_ITER,
    num_target_block_errors = NUM_TARGET_ERRORS,
    early_stop              = True,
    verbose                 = True,
)

print("\nSimulating Rayleigh fading BLER ...")
ber_rl, bler_rl = sim_ber(
    mc_fun                  = make_mc_fun(rayleigh_channel),
    ebno_dbs                = torch.tensor(EBNO_DB_RANGE, dtype=torch.float32),
    batch_size              = BATCH_SIZE,
    max_mc_iter             = MAX_MC_ITER,
    num_target_block_errors = NUM_TARGET_ERRORS,
    early_stop              = True,
    verbose                 = True,
)

# %% [markdown]
# ## Plot BLER and BER Curves

# %%
ber_dm  = ber_dm.cpu().numpy()
bler_dm = bler_dm.cpu().numpy()
ber_rl  = ber_rl.cpu().numpy()
bler_rl = bler_rl.cpu().numpy()

# Replace exact zeros with a floor so log scale plots look clean
FLOOR = 1e-4
bler_dm = np.where(bler_dm == 0, FLOOR, bler_dm)
bler_rl = np.where(bler_rl == 0, FLOOR, bler_rl)
ber_dm  = np.where(ber_dm  == 0, FLOOR, ber_dm)
ber_rl  = np.where(ber_rl  == 0, FLOOR, ber_rl)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# --- BLER ---
ax = axes[0]
ax.semilogy(EBNO_DB_RANGE, bler_dm, "b-o",  label="DeepMIMO (ray-tracing)")
ax.semilogy(EBNO_DB_RANGE, bler_rl, "r--s", label="Rayleigh fading")
ax.axhline(0.1, color="gray", linestyle=":", linewidth=0.8, label="10% BLER")
ax.set_xlabel("Eb/N0 [dB]")
ax.set_ylabel("BLER")
ax.set_title(f"5G NR PUSCH BLER\n({SCENARIO}, {N_BS_ANT}-ant BS, {N_PATHS} paths)")
ax.legend()
ax.grid(visible=True, which="both", alpha=0.3)
ax.set_ylim([FLOOR / 2, 1.5])

# --- BER ---
ax = axes[1]
ax.semilogy(EBNO_DB_RANGE, ber_dm, "b-o",  label="DeepMIMO (ray-tracing)")
ax.semilogy(EBNO_DB_RANGE, ber_rl, "r--s", label="Rayleigh fading")
ax.set_xlabel("Eb/N0 [dB]")
ax.set_ylabel("BER")
ax.set_title(f"5G NR PUSCH BER\n({SCENARIO}, {N_BS_ANT}-ant BS, {N_PATHS} paths)")
ax.legend()
ax.grid(visible=True, which="both", alpha=0.3)
ax.set_ylim([FLOOR / 2, 1.0])

plt.tight_layout()
plt.savefig("sionna_pusch_bler.png", dpi=100, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Summary
#
# | Step | Tool | Role |
# |------|------|------|
# | 1. Load dataset | `dm.load` | Ray-traced paths: coefficients + delays |
# | 2. Build CIR generator | `deepmimo_ul_cir_gen` | UL reciprocity + normalisation |
# | 3. Wrap as channel model | `CIRDataset` + `OFDMChannel` | Sionna-compatible channel block |
# | 4. 5G NR transmitter | `PUSCHTransmitter` | Coded + modulated OFDM slot |
# | 5. 5G NR receiver | `PUSCHReceiver` | LS + LMMSE + LDPC |
# | 6. Simulate | `sim_ber` | BLER / BER vs Eb/N0 |
#
# **Key design choices:**
# - **Channel normalisation**: path loss is removed; SNR is set entirely by
#   `no`.  This is standard for link-level simulation.
# - **Reciprocity**: we have the DL channel (BS → UE); UL = conjugate
#   transpose of DL is a valid approximation for TDD and slowly-varying
#   channels.
# - **Rayleigh baseline**: unit-variance i.i.d. Rayleigh fading, same SIMO
#   antenna count.  Any gap between the two curves reflects the richer
#   scattering structure captured by ray-tracing.
