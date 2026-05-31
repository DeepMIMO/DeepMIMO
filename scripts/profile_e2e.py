"""End-to-end profiling: scenario load + MIMO channel generation.

Profiles ``deepmimo.load()`` and ``Dataset.compute_channels()`` on a real
on-disk scenario with cProfile (sorted by self time), to surface the dominant
cost now that channel generation and the interaction/Doppler maths are
vectorized.

Usage:
    .venv/bin/python scripts/profile_e2e.py [scenario_name]
"""

from __future__ import annotations

import cProfile
import gc
import io
import pstats
import sys
import time

import numpy as np

import deepmimo as dm
from deepmimo import consts as c
from deepmimo.datasets.dataset import MacroDataset
from deepmimo.generator.channel import ChannelParameters


def profile(label, fn, *, top=18):
    """Run ``fn`` under cProfile, print wall-time + the top self-time rows."""
    pr = cProfile.Profile()
    gc.collect()
    t0 = time.perf_counter()
    pr.enable()
    out = fn()
    pr.disable()
    dt = time.perf_counter() - t0
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(top)
    body = "\n".join(
        line for line in s.getvalue().splitlines()
        if line.strip() and "function calls" not in line and "Ordered by" not in line
    )
    print(f"\n{'=' * 90}\n# {label}   ->   {dt:.3f} s\n{'=' * 90}")
    print(body)
    return out, dt


def leaf(ds):
    """Drill a MacroDataset/DynamicDataset down to a single leaf Dataset."""
    obj = ds
    while isinstance(obj, MacroDataset):  # DynamicDataset subclasses MacroDataset
        obj = obj[0]
    return obj


def make_params(bs_shape, n_sc_sel, *, freq=True):
    p = ChannelParameters()
    p.bs_antenna[c.PARAMSET_ANT_SHAPE] = np.array(bs_shape)
    p[c.PARAMSET_FD_CH] = 1 if freq else 0
    p.ofdm[c.PARAMSET_OFDM_SC_SAMP] = np.arange(n_sc_sel)
    return p


def main():
    scen = sys.argv[1] if len(sys.argv) > 1 else "asu_campus_3p5"

    ds, _ = profile(f"load('{scen}')", lambda: dm.load(scen))
    d = leaf(ds)
    try:
        n_ue = int(d.n_ue)
    except Exception:
        n_ue = int(getattr(d, "power", np.empty((0,))).shape[0])
    print(
        f"\nleaf dataset = {type(d).__name__}   n_ue = {n_ue}   "
        f"delay.shape = {tuple(d.delay.shape)}"
    )

    configs = [
        ("compute_channels  freq  BS 8x1   512 sc  (cold cache)", [8, 1], 512, True),
        ("compute_channels  freq  BS 8x1   512 sc  (warm cache)", [8, 1], 512, True),
        ("compute_channels  freq  BS 8x8   64 sc   (cold cache)", [8, 8], 64, True),
        ("compute_channels  time  BS 8x1            (cold cache)", [8, 1], 1, False),
    ]
    for label, bs_shape, n_sc, freq in configs:
        p = make_params(bs_shape, n_sc, freq=freq)
        profile(label, lambda p=p: d.compute_channels(p))
        d[c.CHANNEL_PARAM_NAME] = None  # drop the cached channel to free memory
        gc.collect()


if __name__ == "__main__":
    main()
