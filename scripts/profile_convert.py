"""Profile ray-tracer -> DeepMIMO conversion with cProfile (sorted by self time).

Usage:
    .venv/bin/python scripts/profile_convert.py [rt_folder] [scenario_name]
"""

from __future__ import annotations

import cProfile
import io
import pstats
import sys
import time

import matplotlib

matplotlib.use("Agg")  # headless; avoid any GUI from scene.plot()

import deepmimo as dm


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/insite_asu/asu_campus"
    name = sys.argv[2] if len(sys.argv) > 2 else "asu_campus_insite_prof"

    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    dm.convert(path, scenario_name=name, overwrite=True, vis_scene=False, print_params=False)
    pr.disable()
    dt = time.perf_counter() - t0

    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(30)
    body = "\n".join(
        line for line in s.getvalue().splitlines()
        if line.strip() and "function calls" not in line and "Ordered by" not in line
    )
    print(f"\n{'=' * 90}\n# convert('{path}')   ->   {dt:.3f} s\n{'=' * 90}")
    print(body)


if __name__ == "__main__":
    main()
