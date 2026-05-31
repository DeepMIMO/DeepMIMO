"""Benchmark MIMO channel generation: original per-UE loop vs vectorized batching.

Profiles the wall-clock time of ``_generate_mimo_channel`` (the vectorized
implementation on this branch) against the original per-UE Python loop (kept here
verbatim as the baseline), sweeping:

  * number of users,
  * number of (selected) OFDM subcarriers,
  * number of antennas per side (M_rx = M_tx),

in both the frequency and time domains. The result is a 6-subplot figure
(rows = domain, columns = swept dimension) with a log-scaled time axis, plus a
printed speedup table.

Both implementations are fed the SAME ``array_response_product`` so the comparison
isolates the loop-vs-vectorized algorithmic change.

Usage:
    python benchmarks/benchmark_channel_speed.py            # full sweep
    python benchmarks/benchmark_channel_speed.py --quick    # smaller/faster
    python benchmarks/benchmark_channel_speed.py --repeats 5 --out /tmp/bench.png
"""

from __future__ import annotations

import argparse
import time
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from deepmimo import consts as c  # noqa: E402
import deepmimo.generator.channel as _channel_mod  # noqa: E402
from deepmimo.generator.channel import (  # noqa: E402
    OFDMPathGenerator,
    _check_ofdm_compatibility,
    _compute_single_freq_channel,
    _compute_single_time_channel,
    _generate_mimo_channel,
)

# Silence the per-chunk progress bar so it neither pollutes output nor skews timing.
_channel_mod.tqdm = lambda iterable=None, **_kw: iterable if iterable is not None else []


# ----------------------------------------------------------------------------------
# Baseline: the original per-UE Python loop (verbatim from `main`, progress bar removed).
# ----------------------------------------------------------------------------------
def _generate_mimo_channel_baseline(
    array_response_product,
    power,
    delay,
    phase,
    doppler,
    ofdm_params,
    *,
    times=0.0,
    freq_domain=True,
    squeeze_time=True,
):
    """Original implementation: loop over users, compute each link independently."""
    times_arr = np.atleast_1d(times).astype(float)
    n_times = times_arr.shape[0]

    ts = 1.0 / ofdm_params[c.PARAMSET_OFDM_BANDWIDTH]
    subcarriers = ofdm_params[c.PARAMSET_OFDM_SC_SAMP]
    k_subcarriers = len(subcarriers)
    path_gen = OFDMPathGenerator(ofdm_params, subcarriers)

    if freq_domain:
        _check_ofdm_compatibility(ofdm_params, delay)

    n_ues = power.shape[0]
    p_max = power.shape[1]
    m_rx, m_tx = array_response_product.shape[1:3]
    last_ch_dim = k_subcarriers if freq_domain else p_max

    if n_times == 1 and squeeze_time:
        channel = np.zeros((n_ues, m_rx, m_tx, last_ch_dim), dtype=np.csingle)
    else:
        channel = np.zeros((n_ues, m_rx, m_tx, last_ch_dim, n_times), dtype=np.csingle)

    nan_masks = ~np.isnan(power)
    valid_path_counts = np.sum(nan_masks, axis=1)

    for i in range(n_ues):
        non_nan_mask = nan_masks[i]
        n_paths = valid_path_counts[i]
        if n_paths == 0:
            continue

        array_product = array_response_product[i][..., non_nan_mask]
        user_power = power[i, non_nan_mask]
        user_delay = delay[i, non_nan_mask]
        user_phase = phase[i, non_nan_mask]
        user_doppler = doppler[i, non_nan_mask]

        if freq_domain:
            path_gains = path_gen.generate(
                pwr=user_power,
                toa=user_delay,
                phs=user_phase,
                ts=ts,
                dopplers=user_doppler,
                times=times_arr,
            )
            channel[i] = _compute_single_freq_channel(
                array_product, path_gains, squeeze_time=squeeze_time
            )
        else:
            phase0 = np.deg2rad(user_phase)[:, None]
            path_gains = np.sqrt(user_power)[:, None] * np.exp(
                1j * (phase0 + 2 * np.pi * user_doppler[:, None] * times_arr[None, :]),
            )
            channel[i] = _compute_single_time_channel(
                array_product, path_gains, p_max, squeeze_time=squeeze_time
            )

    return channel


# ----------------------------------------------------------------------------------
# Synthetic inputs + timing
# ----------------------------------------------------------------------------------
def make_inputs(n_users, m_rx, m_tx, p_max, n_sc_total, n_sc_sel, *, seed=0):
    """Build ragged (NaN-padded) synthetic path data and a full array-response product."""
    rng = np.random.default_rng(seed)
    power = rng.uniform(0.1, 1.0, (n_users, p_max))
    delay = rng.uniform(0.0, 1e-7, (n_users, p_max))  # well within any OFDM symbol
    phase = rng.uniform(0.0, 360.0, (n_users, p_max))
    doppler = rng.uniform(-100.0, 100.0, (n_users, p_max))

    # Ragged: each user keeps between 1 and p_max valid paths; pad the rest with NaN.
    n_valid = rng.integers(1, p_max + 1, n_users)
    invalid = np.arange(p_max)[None, :] >= n_valid[:, None]
    for arr in (power, delay, phase, doppler):
        arr[invalid] = np.nan

    arp = (
        rng.standard_normal((n_users, m_rx, m_tx, p_max))
        + 1j * rng.standard_normal((n_users, m_rx, m_tx, p_max))
    ).astype(np.complex64)

    selected = np.unique(np.linspace(0, n_sc_total - 1, n_sc_sel).astype(int))
    ofdm = {
        c.PARAMSET_OFDM_SC_NUM: n_sc_total,
        c.PARAMSET_OFDM_SC_SAMP: selected,
        c.PARAMSET_OFDM_BANDWIDTH: 10e6,
        c.PARAMSET_OFDM_LPF: 0,
    }
    return arp, power, delay, phase, doppler, ofdm


def bench(fn, *, repeats):
    """Return the best (min) wall-clock time over ``repeats`` runs, after one warmup."""
    fn()  # warmup (allocations, BLAS thread spin-up, etc.)
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def verify():
    """Sanity check: the vectorized result must match the baseline."""
    arp, power, delay, phase, doppler, ofdm = make_inputs(64, 4, 4, 12, 256, 64, seed=1)
    for fd in (True, False):
        base = _generate_mimo_channel_baseline(
            arp, power, delay, phase, doppler, ofdm, freq_domain=fd
        )
        opt = _generate_mimo_channel(
            arp,
            power=power,
            delay=delay,
            phase=phase,
            doppler=doppler,
            ofdm_params=ofdm,
            freq_domain=fd,
        )
        if not np.allclose(base, opt, rtol=1e-4, atol=1e-6):
            msg = f"Baseline and vectorized disagree (freq_domain={fd}) - benchmark invalid"
            raise AssertionError(msg)
    print("Correctness check passed: vectorized == baseline (rtol=1e-4).")


def run_sweep(name, values, base_cfg, *, repeats):
    """Time baseline vs vectorized across ``values`` for the dimension ``name``."""
    res = {"x": list(values), "base_freq": [], "opt_freq": [], "base_time": [], "opt_time": []}
    for v in values:
        cfg = dict(base_cfg)
        if name == "users":
            cfg["n_users"] = v
        elif name == "subcarriers":
            cfg["n_sc_sel"] = v
            cfg["n_sc_total"] = max(cfg["n_sc_total"], v)
        elif name == "antennas":
            cfg["m_rx"] = cfg["m_tx"] = v

        arp, power, delay, phase, doppler, ofdm = make_inputs(
            cfg["n_users"], cfg["m_rx"], cfg["m_tx"], cfg["p_max"], cfg["n_sc_total"], cfg["n_sc_sel"]
        )
        for fd, bkey, okey in ((True, "base_freq", "opt_freq"), (False, "base_time", "opt_time")):
            base_call = partial(
                _generate_mimo_channel_baseline, arp, power, delay, phase, doppler, ofdm,
                freq_domain=fd,
            )
            opt_call = partial(
                _generate_mimo_channel, arp,
                power=power, delay=delay, phase=phase, doppler=doppler, ofdm_params=ofdm,
                freq_domain=fd,
            )
            res[bkey].append(bench(base_call, repeats=repeats))
            res[okey].append(bench(opt_call, repeats=repeats))
        print(f"  {name}={v}: done")
    return res


# ----------------------------------------------------------------------------------
# Plotting / reporting
# ----------------------------------------------------------------------------------
def plot(results, out_path):
    sweeps = [
        ("users", "Number of users", results["users"]),
        ("subcarriers", "Number of subcarriers", results["subcarriers"]),
        ("antennas", "Antennas per side (M_rx = M_tx)", results["antennas"]),
    ]
    domains = [
        ("Frequency domain", "base_freq", "opt_freq"),
        ("Time domain", "base_time", "opt_time"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for row, (dlabel, bk, ok) in enumerate(domains):
        for col, (skey, xlabel, res) in enumerate(sweeps):
            ax = axes[row, col]
            x = res["x"]
            base = np.asarray(res[bk])
            opt = np.maximum(np.asarray(res[ok]), 1e-9)
            ax.plot(x, base, "o-", color="tab:red", label="per-UE loop (baseline)")
            ax.plot(x, opt, "s-", color="tab:green", label="vectorized")
            ax.set_yscale("log")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("time per call (s)")
            ax.set_title(f"{dlabel} vs {skey}")
            ax.grid(True, which="both", alpha=0.3)
            speedup_max = base[-1] / opt[-1]
            ax.legend(title=f"{speedup_max:.0f}x faster at largest size", fontsize=8)

    fig.suptitle(
        "MIMO channel generation: per-UE loop vs vectorized batching (log-scale time)",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"\nSaved figure to {out_path}")


def print_summary(results):
    print("\n=== Speedup (baseline time / vectorized time) ===")
    header = f"{'sweep':<13}{'domain':<7}{'x_min':>7}{'spd_min':>9}{'x_max':>8}{'spd_max':>9}"
    print(header)
    print("-" * len(header))
    for skey in ("users", "subcarriers", "antennas"):
        res = results[skey]
        for dlabel, bk, ok in (("freq", "base_freq", "opt_freq"), ("time", "base_time", "opt_time")):
            base = np.asarray(res[bk])
            opt = np.maximum(np.asarray(res[ok]), 1e-9)
            spd = base / opt
            print(
                f"{skey:<13}{dlabel:<7}{res['x'][0]:>7}{spd[0]:>8.1f}x"
                f"{res['x'][-1]:>8}{spd[-1]:>8.1f}x"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="smaller sizes / fewer repeats")
    parser.add_argument("--repeats", type=int, default=None, help="timed repeats per point")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "channel_speed_benchmark.png",
        help="output figure path",
    )
    args = parser.parse_args()

    repeats = args.repeats if args.repeats is not None else (2 if args.quick else 3)

    if args.quick:
        users = [25, 50, 100, 250, 500, 1000]
        subcarriers = [16, 32, 64, 128, 256]
        antennas = [1, 2, 4, 8, 16]
    else:
        users = [25, 50, 100, 250, 500, 1000, 2000, 4000]
        subcarriers = [16, 32, 64, 128, 256, 512, 1024]
        antennas = [1, 2, 4, 8, 16, 32]

    verify()

    t_start = time.perf_counter()
    results = {}
    print("\nUsers sweep (M=4, P=15, sel_sc=64):")
    results["users"] = run_sweep(
        "users", users,
        {"n_users": 500, "m_rx": 4, "m_tx": 4, "p_max": 15, "n_sc_total": 256, "n_sc_sel": 64},
        repeats=repeats,
    )
    print("Subcarriers sweep (users=500, M=4, P=15):")
    results["subcarriers"] = run_sweep(
        "subcarriers", subcarriers,
        {"n_users": 500, "m_rx": 4, "m_tx": 4, "p_max": 15, "n_sc_total": 1024, "n_sc_sel": 64},
        repeats=repeats,
    )
    print("Antennas sweep (users=500, P=15, sel_sc=64):")
    results["antennas"] = run_sweep(
        "antennas", antennas,
        {"n_users": 500, "m_rx": 4, "m_tx": 4, "p_max": 15, "n_sc_total": 256, "n_sc_sel": 64},
        repeats=repeats,
    )
    print(f"\nTotal benchmark time: {time.perf_counter() - t_start:.1f}s")

    print_summary(results)
    plot(results, args.out)


if __name__ == "__main__":
    main()
