"""Benchmark MIMO channel generation: original per-UE loop vs vectorized batching.

Profiles the wall-clock time of ``_generate_mimo_channel`` (the vectorized
implementation on this branch) against the original per-UE Python loop (kept here
verbatim as the baseline), sweeping:

  * number of users        (up to 200k),
  * number of (selected) OFDM subcarriers,
  * number of antennas per side (M_rx = M_tx, up to 256),

in both the frequency and time domains. The result is a 6-subplot figure
(rows = domain, columns = swept dimension) with log-log time axes and, on a
secondary axis, the speedup ratio (baseline / vectorized) with a 1x break-even
reference line. A speedup table is also printed.

Both implementations are fed the SAME ``array_response_product`` so the comparison
isolates the loop-vs-vectorized algorithmic change.

Large configurations are skipped automatically when their estimated peak memory
exceeds the budget (``--mem-budget-gb``), and timing is adaptive (slow points are
sampled once, fast points repeated), so the full sweep stays bounded (~2-3 min).

Usage:
    python scripts/benchmark_channel_speed.py                 # full sweep
    python scripts/benchmark_channel_speed.py --quick         # smaller/faster
    python scripts/benchmark_channel_speed.py --mem-budget-gb 6 --out /tmp/b.png
"""

from __future__ import annotations

import argparse
import gc
import os
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

_BYTES_PER_C64 = 8
_MEM_SAFETY = 1.7  # headroom for transient intermediates beyond product + output


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
# Synthetic inputs + memory accounting + timing
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


def peak_bytes(cfg, *, freq_domain):
    """Estimate peak memory (bytes) for one (config, domain): product + output + headroom."""
    n, mr, mt, p = cfg["n_users"], cfg["m_rx"], cfg["m_tx"], cfg["p_max"]
    k = cfg["n_sc_sel"] if freq_domain else p
    product = n * mr * mt * p * _BYTES_PER_C64
    output = n * mr * mt * k * _BYTES_PER_C64
    return int((product + output) * _MEM_SAFETY)


def total_ram_bytes():
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return 8 * 1024**3


def bench(fn, *, max_repeats, time_budget=2.5):
    """Adaptive timing: slow calls sampled once, fast calls repeated (best of N)."""
    t0 = time.perf_counter()
    fn()
    first = time.perf_counter() - t0
    if first > time_budget:  # already slow; one (warm-ish) sample is enough
        return first
    reps = max(1, min(max_repeats, int(time_budget / max(first, 1e-4))))
    best = first  # the first run doubles as warmup for fast configs
    for _ in range(reps):
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


def run_sweep(name, values, base_cfg, *, max_repeats, budget_bytes):
    """Time baseline vs vectorized across ``values`` for the dimension ``name``."""
    res = {"x": list(values), "base_freq": [], "opt_freq": [], "base_time": [], "opt_time": []}
    domains = ((True, "base_freq", "opt_freq"), (False, "base_time", "opt_time"))
    for v in values:
        cfg = dict(base_cfg)
        if name == "users":
            cfg["n_users"] = v
        elif name == "subcarriers":
            cfg["n_sc_sel"] = v
            cfg["n_sc_total"] = max(cfg["n_sc_total"], v)
        elif name == "antennas":
            cfg["m_rx"] = cfg["m_tx"] = v

        peaks = {fd: peak_bytes(cfg, freq_domain=fd) for fd, _, _ in domains}
        if min(peaks.values()) > budget_bytes:  # cannot even build the cheaper domain
            for key in ("base_freq", "opt_freq", "base_time", "opt_time"):
                res[key].append(np.nan)
            print(f"  {name}={v}: SKIPPED ({min(peaks.values()) / 1e9:.1f} GB > budget)")
            continue

        arp, power, delay, phase, doppler, ofdm = make_inputs(
            cfg["n_users"], cfg["m_rx"], cfg["m_tx"], cfg["p_max"], cfg["n_sc_total"], cfg["n_sc_sel"]
        )
        for fd, bkey, okey in domains:
            if peaks[fd] > budget_bytes:
                res[bkey].append(np.nan)
                res[okey].append(np.nan)
                continue
            base_call = partial(
                _generate_mimo_channel_baseline, arp, power, delay, phase, doppler, ofdm,
                freq_domain=fd,
            )
            opt_call = partial(
                _generate_mimo_channel, arp,
                power=power, delay=delay, phase=phase, doppler=doppler, ofdm_params=ofdm,
                freq_domain=fd,
            )
            res[bkey].append(bench(base_call, max_repeats=max_repeats))
            res[okey].append(bench(opt_call, max_repeats=max_repeats))
        del arp, power, delay, phase, doppler
        gc.collect()
        print(f"  {name}={v}: done")
    return res


# ----------------------------------------------------------------------------------
# Plotting / reporting
# ----------------------------------------------------------------------------------
def plot(results, out_path):
    sweeps = [
        ("users", "Number of users", results["users"]),
        ("subcarriers", "Number of (selected) subcarriers", results["subcarriers"]),
        ("antennas", "Antennas per side (M_rx = M_tx)", results["antennas"]),
    ]
    domains = [
        ("Frequency domain", "base_freq", "opt_freq"),
        ("Time domain", "base_time", "opt_time"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    for row, (dlabel, bk, ok) in enumerate(domains):
        for col, (skey, xlabel, res) in enumerate(sweeps):
            ax = axes[row, col]
            x = np.asarray(res["x"], dtype=float)
            base = np.asarray(res[bk], dtype=float)
            opt = np.asarray(res[ok], dtype=float)
            l1, = ax.plot(x, base, "o-", color="tab:red", label="per-UE loop (baseline)")
            l2, = ax.plot(x, opt, "s-", color="tab:green", label="vectorized")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("time per call (s)")
            ax.set_title(f"{dlabel} vs {skey}")
            ax.grid(True, which="both", alpha=0.3)

            ax2 = ax.twinx()
            with np.errstate(divide="ignore", invalid="ignore"):
                speedup = base / opt
            l3, = ax2.plot(
                x, speedup, "^:", color="tab:blue", alpha=0.85,
                label="speedup (baseline / vectorized)",
            )
            ax2.axhline(1.0, color="gray", ls="--", lw=1.0)
            ax2.set_ylim(bottom=0)  # linear speedup axis; keep 1x break-even in frame
            ax2.set_ylabel("speedup (x)", color="tab:blue")
            ax2.tick_params(axis="y", labelcolor="tab:blue")
            peak = np.nanmax(speedup) if np.isfinite(speedup).any() else float("nan")
            ax.legend(handles=[l1, l2, l3], fontsize=7, loc="best", title=f"peak {peak:.0f}x")

    fig.suptitle(
        "MIMO channel generation: per-UE loop vs vectorized "
        "(log-log time; blue = speedup on linear axis, dashed = 1x break-even)",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"\nSaved figure to {out_path}")


def print_summary(results):
    print("\n=== Speedup (baseline / vectorized);  NaN x-points were memory-skipped ===")
    header = f"{'sweep':<13}{'domain':<6}{'min_x':>9}{'@min':>8}{'max_x':>9}{'@max':>8}{'peak':>8}"
    print(header)
    print("-" * len(header))
    for skey in ("users", "subcarriers", "antennas"):
        res = results[skey]
        x = np.asarray(res["x"], dtype=float)
        for dom, bk, ok in (("freq", "base_freq", "opt_freq"), ("time", "base_time", "opt_time")):
            base = np.asarray(res[bk], dtype=float)
            opt = np.asarray(res[ok], dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                spd = base / opt
            valid = np.isfinite(spd)
            if not valid.any():
                print(f"{skey:<13}{dom:<6}{'(all skipped)':>34}")
                continue
            xs, ss = x[valid], spd[valid]
            print(
                f"{skey:<13}{dom:<6}{int(xs[0]):>9}{ss[0]:>7.1f}x"
                f"{int(xs[-1]):>9}{ss[-1]:>7.1f}x{np.nanmax(ss):>7.1f}x"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="smaller sizes / fewer repeats")
    parser.add_argument("--repeats", type=int, default=None, help="max timed repeats per point")
    parser.add_argument(
        "--mem-budget-gb",
        type=float,
        default=round(0.55 * total_ram_bytes() / 1e9, 1),
        help="skip configs whose estimated peak memory exceeds this many GB",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "channel_speed_benchmark.png",
        help="output figure path",
    )
    args = parser.parse_args()

    repeats = args.repeats if args.repeats is not None else (2 if args.quick else 4)
    budget_bytes = int(args.mem_budget_gb * 1e9)
    print(f"Memory budget: {args.mem_budget_gb} GB  |  max repeats: {repeats}")

    if args.quick:
        users = [25, 100, 500, 2000, 8000]
        subcarriers = [16, 64, 256, 1024]
        antennas = [1, 4, 16, 64]
    else:
        users = [25, 50, 100, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 200000]
        subcarriers = [16, 32, 64, 128, 256, 512, 1024]
        antennas = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    verify()

    t_start = time.perf_counter()
    results = {}
    print("\nUsers sweep (M=4, P=15, sel_sc=64):")
    results["users"] = run_sweep(
        "users", users,
        {"n_users": 500, "m_rx": 4, "m_tx": 4, "p_max": 15, "n_sc_total": 256, "n_sc_sel": 64},
        max_repeats=repeats, budget_bytes=budget_bytes,
    )
    print("Subcarriers sweep (users=500, M=4, P=15):")
    results["subcarriers"] = run_sweep(
        "subcarriers", subcarriers,
        {"n_users": 500, "m_rx": 4, "m_tx": 4, "p_max": 15, "n_sc_total": 1024, "n_sc_sel": 64},
        max_repeats=repeats, budget_bytes=budget_bytes,
    )
    print("Antennas sweep (users=128, P=15, sel_sc=16):")
    results["antennas"] = run_sweep(
        "antennas", antennas,
        {"n_users": 128, "m_rx": 4, "m_tx": 4, "p_max": 15, "n_sc_total": 16, "n_sc_sel": 16},
        max_repeats=repeats, budget_bytes=budget_bytes,
    )
    print(f"\nTotal benchmark time: {time.perf_counter() - t_start:.1f}s")

    print_summary(results)
    plot(results, args.out)


if __name__ == "__main__":
    main()
