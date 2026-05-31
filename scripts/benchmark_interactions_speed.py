"""Benchmark interaction/doppler computations: original loops vs vectorized NumPy.

Profiles the wall-clock time of the vectorized implementations on this branch
(``Dataset._compute_inter_angles``, ``Dataset._compute_inter_objects`` and
``Dataset._compute_doppler``) against the original triple-nested Python loops
(kept here verbatim as the baseline), sweeping:

  * number of users     (up to ~50k),
  * max interaction depth (``max_inter``).

Both implementations run on the SAME lightweight synthetic stand-in object that
exposes exactly the attributes the three Dataset methods read, so the comparison
isolates the loop-vs-vectorized algorithmic change. The vectorized side reuses
the real Dataset methods (bound to the stand-in) so we benchmark the shipped code.

The result is a 2x3 figure (rows = swept dimension, columns = the three
functions) with a log-y time axis and, on a secondary axis, the speedup ratio
(baseline / vectorized) on a LINEAR axis with a 1x break-even reference line.
A speedup table is also printed. Timing is adaptive (slow points are sampled
once, fast points repeated), so the full sweep stays bounded.

Usage:
    python scripts/benchmark_interactions_speed.py            # full sweep
    python scripts/benchmark_interactions_speed.py --quick    # smaller/faster
    python scripts/benchmark_interactions_speed.py --out /tmp/b.png
"""

from __future__ import annotations

import argparse
import gc
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from deepmimo import consts as c  # noqa: E402
from deepmimo.datasets.dataset import Dataset  # noqa: E402
from deepmimo.utils import spherical_to_cartesian  # noqa: E402


# ----------------------------------------------------------------------------------
# Lightweight scene + stand-in (object_id == position in `objects`).
# ----------------------------------------------------------------------------------
class _BBox:
    def __init__(self, center, z_max):
        self.center = np.asarray(center, dtype=float)
        self.z_max = float(z_max)


class _Obj:
    def __init__(self, object_id, label, center, z_max, vel):
        self.object_id = object_id
        self.label = label
        self.bounding_box = _BBox(center, z_max)
        self.vel = np.asarray(vel, dtype=float)


class _Scene:
    def __init__(self, objects):
        self.objects = objects


class _StandIn:
    """Holds the attributes the three Dataset methods read; reuses their code.

    Binding the real (unbound) Dataset methods as class attributes turns them into
    methods of the stand-in, so ``_compute_doppler`` correctly calls the vectorized
    ``_compute_inter_angles`` / ``_compute_inter_objects`` on ``self``.
    """

    _compute_inter_angles = Dataset._compute_inter_angles
    _compute_inter_objects = Dataset._compute_inter_objects
    _compute_doppler = Dataset._compute_doppler

    def __init__(self, **attrs):
        self.__dict__.update(attrs)


def _num_interactions_from_inter(inter: np.ndarray) -> np.ndarray:
    """Replicate Dataset._compute_num_interactions (digit-length encoding)."""
    result = np.zeros_like(inter)
    result[np.isnan(inter)] = np.nan
    non_zero = inter > 0
    result[non_zero] = np.floor(np.log10(inter[non_zero])) + 1
    return result


def make_inputs(n_ue, n_paths, n_depth, n_objects, *, seed=0):
    """Build a synthetic stand-in with ragged, NaN-padded path/interaction data."""
    rng = np.random.default_rng(seed)
    terrain_z = 0.0

    objects = [_Obj(0, "terrain", [0.0, 0.0, terrain_z / 2], terrain_z, rng.uniform(-3, 3, 3))]
    for oid in range(1, n_objects + 1):
        center = rng.uniform(-200, 200, 3)
        objects.append(
            _Obj(oid, "buildings", center, center[2] + rng.uniform(1, 20), rng.uniform(-10, 10, 3))
        )
    scene = _Scene(objects)

    aoa_az = np.full((n_ue, n_paths), np.nan)
    aoa_el = np.full((n_ue, n_paths), np.nan)
    aod_az = np.full((n_ue, n_paths), np.nan)
    aod_el = np.full((n_ue, n_paths), np.nan)
    inter = np.full((n_ue, n_paths), np.nan)
    inter_pos = np.full((n_ue, n_paths, n_depth, 3), np.nan)

    num_paths_per_ue = rng.integers(1, n_paths + 1, n_ue)
    num_paths_per_ue[0] = n_paths  # guarantee max_paths == n_paths
    for u in range(n_ue):
        npaths = int(num_paths_per_ue[u])
        for p in range(npaths):
            aoa_az[u, p] = rng.uniform(-180, 180)
            aoa_el[u, p] = rng.uniform(1, 179)
            aod_az[u, p] = rng.uniform(-180, 180)
            aod_el[u, p] = rng.uniform(1, 179)
            roll = rng.random()
            if roll < 0.15:
                inter[u, p] = np.nan
                continue
            if roll < 0.30:
                inter[u, p] = 0
                continue
            k = int(rng.integers(1, n_depth + 1))
            if u == 0 and p == 0:
                k = n_depth  # guarantee max_inter == n_depth
            digits = rng.integers(1, 5, k)
            inter[u, p] = int("".join(str(d) for d in digits))
            pts = rng.uniform(-150, 150, (k, 3))
            snap = rng.random(k) < 0.4
            pts[snap, 2] = terrain_z + rng.uniform(-0.0008, 0.0008, int(snap.sum()))
            inter_pos[u, p, :k] = pts

    num_interactions = _num_interactions_from_inter(inter)
    num_paths = (~np.isnan(aoa_az)).sum(axis=1)
    max_paths = int(np.nanmax(num_paths))
    max_inter = int(np.nanmax(num_interactions))

    return _StandIn(
        n_ue=n_ue,
        max_paths=max_paths,
        max_inter=max_inter,
        num_paths=num_paths,
        num_interactions=num_interactions,
        inter=inter,
        inter_pos=inter_pos,
        tx_pos=np.array([rng.uniform(-5, 5), rng.uniform(-5, 5), 12.0]),
        rx_pos=rng.uniform(-100, 100, (n_ue, 3)),
        tx_vel=rng.uniform(-20, 20, 3),
        rx_vel=rng.uniform(-20, 20, (n_ue, 3)),
        aoa_az=aoa_az,
        aoa_el=aoa_el,
        aod_az=aod_az,
        aod_el=aod_el,
        scene=scene,
        rt_params=SimpleNamespace(frequency=28e9),
        doppler_enabled=True,
    )


# ----------------------------------------------------------------------------------
# Baseline: verbatim copies of the original triple-nested loop implementations.
# ----------------------------------------------------------------------------------
def baseline_inter_angles(d):
    inter_angles = np.zeros((d.n_ue, d.max_paths, d.max_inter + 1, 3))
    for ue_i in range(d.n_ue):
        for path_i in range(d.max_paths):
            n_inter = d.num_interactions[ue_i, path_i]
            if np.isnan(n_inter) or n_inter == 0:
                continue
            for i in range(-1, int(n_inter)):
                pos1 = d.tx_pos if i == -1 else d.inter_pos[ue_i, path_i, i]
                pos2 = d.rx_pos[ue_i] if i == n_inter - 1 else d.inter_pos[ue_i, path_i, i + 1]
                vec = pos2 - pos1
                inter_angles[ue_i, path_i, i + 1] = vec / np.linalg.norm(vec)
    return inter_angles


def baseline_inter_objects(d):
    inter_obj_ids = np.zeros((d.n_ue, d.max_paths, d.max_inter)) * np.nan
    terrain_obj = next(obj for obj in d.scene.objects if obj.label == "terrain")
    terrain_z_coord = terrain_obj.bounding_box.z_max
    non_terrain_objs = [obj for obj in d.scene.objects if obj.label != "terrain"]
    obj_centers = np.array([obj.bounding_box.center for obj in non_terrain_objs])
    obj_ids = np.array([obj.object_id for obj in non_terrain_objs])
    for ue_i in range(d.n_ue):
        for path_i in range(d.max_paths):
            n_inter = d.num_interactions[ue_i, path_i]
            if np.isnan(n_inter) or n_inter == 0:
                continue
            for i in range(int(n_inter)):
                i_pos = d.inter_pos[ue_i, path_i, i]
                if np.isclose(i_pos[2], terrain_z_coord, rtol=0, atol=0.001):
                    inter_obj_ids[ue_i, path_i, i] = terrain_obj.object_id
                    continue
                dist = np.linalg.norm(obj_centers - i_pos, axis=1)
                inter_obj_ids[ue_i, path_i, i] = obj_ids[np.argmin(dist)]
    return inter_obj_ids


def baseline_doppler(d):
    doppler = np.zeros((d.n_ue, d.max_paths))
    wavelength = c.SPEED_OF_LIGHT / d.rt_params.frequency
    ones = np.ones((d.n_ue, d.max_paths, 1))
    tx_coord_cat = np.concatenate(
        (ones, np.deg2rad(d.aod_el)[..., None], np.deg2rad(d.aod_az)[..., None]), axis=-1
    )
    rx_coord_cat = -np.concatenate(
        (ones, np.deg2rad(d.aoa_el)[..., None], np.deg2rad(d.aoa_az)[..., None]), axis=-1
    )
    k_tx = spherical_to_cartesian(tx_coord_cat)
    k_rx = spherical_to_cartesian(rx_coord_cat)
    k_i = baseline_inter_angles(d)
    inter_objects = baseline_inter_objects(d)
    for ue_i in range(d.n_ue):
        n_paths = d.num_paths[ue_i]
        for path_i in range(n_paths):
            if np.isnan(d.inter[ue_i, path_i]):
                continue
            n_inter = d.num_interactions[ue_i, path_i]
            tx_doppler = np.dot(k_tx[ue_i, path_i], d.tx_vel) / wavelength
            rx_doppler = np.dot(k_rx[ue_i, path_i], d.rx_vel[ue_i]) / wavelength
            path_dopplers = [0]
            for i in range(int(n_inter)):
                inter_obj_idx = inter_objects[ue_i, path_i, i]
                if np.isnan(inter_obj_idx):
                    continue
                v_i = d.scene.objects[int(inter_obj_idx)].vel
                ki_diff = k_i[ue_i, path_i, i + 1] - k_i[ue_i, path_i, i]
                path_dopplers += [np.dot(v_i, ki_diff) / wavelength]
            doppler[ue_i, path_i] = tx_doppler - rx_doppler + np.sum(path_dopplers)
    return doppler


# ----------------------------------------------------------------------------------
# Function registry: (label, baseline_fn, vectorized_fn).
# ----------------------------------------------------------------------------------
FUNCS = [
    ("inter_angles", baseline_inter_angles, lambda d: d._compute_inter_angles()),  # noqa: SLF001
    ("inter_objects", baseline_inter_objects, lambda d: d._compute_inter_objects()),  # noqa: SLF001
    ("doppler", baseline_doppler, lambda d: d._compute_doppler()),  # noqa: SLF001
]


# ----------------------------------------------------------------------------------
# Timing / verification
# ----------------------------------------------------------------------------------
def bench(fn, *, max_repeats, time_budget=2.0):
    """Adaptive timing: slow calls sampled once, fast calls repeated (best of N)."""
    t0 = time.perf_counter()
    fn()
    first = time.perf_counter() - t0
    if first > time_budget:
        return first
    reps = max(1, min(max_repeats, int(time_budget / max(first, 1e-4))))
    best = first
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def verify():
    """Sanity check: the vectorized result must match the baseline on every function."""
    for seed in (0, 1, 2):
        d = make_inputs(n_ue=48, n_paths=6, n_depth=4, n_objects=6, seed=seed)
        base_obj, opt_obj = baseline_inter_objects(d), d._compute_inter_objects()  # noqa: SLF001
        if not np.array_equal(base_obj, opt_obj, equal_nan=True):
            msg = "inter_objects: vectorized != baseline - benchmark invalid"
            raise AssertionError(msg)
        for name, base_fn, opt_fn in (FUNCS[0], FUNCS[2]):
            base, opt = base_fn(d), opt_fn(d)
            if not np.allclose(base, opt, rtol=1e-6, atol=1e-9, equal_nan=True):
                msg = f"{name}: vectorized != baseline - benchmark invalid"
                raise AssertionError(msg)
    print("Correctness check passed: vectorized == baseline (rtol=1e-6, exact object ids).")


def run_sweep(name, values, base_cfg, *, max_repeats):
    """Time baseline vs vectorized across ``values`` for the dimension ``name``."""
    res = {"x": list(values)}
    for label, _, _ in FUNCS:
        res[f"base_{label}"] = []
        res[f"opt_{label}"] = []

    for v in values:
        cfg = dict(base_cfg)
        if name == "users":
            cfg["n_ue"] = int(v)
        elif name == "max_inter":
            cfg["n_depth"] = int(v)
        d = make_inputs(**cfg)

        for label, base_fn, opt_fn in FUNCS:
            res[f"base_{label}"].append(bench(partial(base_fn, d), max_repeats=max_repeats))
            res[f"opt_{label}"].append(bench(partial(opt_fn, d), max_repeats=max_repeats))
        del d
        gc.collect()
        print(f"  {name}={v}: done")
    return res


# ----------------------------------------------------------------------------------
# Plotting / reporting
# ----------------------------------------------------------------------------------
def plot(results, out_path):
    sweeps = [
        ("users", "Number of users", "log", results["users"]),
        ("max_inter", "Max interaction depth (max_inter)", "linear", results["max_inter"]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    for row, (skey, xlabel, xscale, res) in enumerate(sweeps):
        x = np.asarray(res["x"], dtype=float)
        for col, (label, _, _) in enumerate(FUNCS):
            ax = axes[row, col]
            base = np.asarray(res[f"base_{label}"], dtype=float)
            opt = np.asarray(res[f"opt_{label}"], dtype=float)
            l1, = ax.plot(x, base, "o-", color="tab:red", label="loop (baseline)")
            l2, = ax.plot(x, opt, "s-", color="tab:green", label="vectorized")
            ax.set_xscale(xscale)
            ax.set_yscale("log")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("time per call (s)")
            ax.set_title(f"{label} vs {skey}")
            ax.grid(True, which="both", alpha=0.3)

            ax2 = ax.twinx()
            with np.errstate(divide="ignore", invalid="ignore"):
                speedup = base / opt
            l3, = ax2.plot(x, speedup, "^:", color="tab:blue", alpha=0.85, label="speedup")
            ax2.axhline(1.0, color="gray", ls="--", lw=1.0)
            ax2.set_ylim(bottom=0)  # linear speedup axis; keep 1x break-even in frame
            ax2.set_ylabel("speedup (x)", color="tab:blue")
            ax2.tick_params(axis="y", labelcolor="tab:blue")
            peak = np.nanmax(speedup) if np.isfinite(speedup).any() else float("nan")
            ax.legend(handles=[l1, l2, l3], fontsize=7, loc="best", title=f"peak {peak:.0f}x")

    fig.suptitle(
        "Interaction/doppler computations: loop vs vectorized "
        "(log-y time; blue = speedup on linear axis, dashed = 1x break-even)",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"\nSaved figure to {out_path}")


def print_summary(results):
    print("\n=== Speedup (baseline / vectorized) ===")
    header = f"{'sweep':<11}{'function':<15}{'min_x':>9}{'@min':>9}{'max_x':>9}{'@max':>9}{'peak':>9}"
    print(header)
    print("-" * len(header))
    for skey in ("users", "max_inter"):
        res = results[skey]
        x = np.asarray(res["x"], dtype=float)
        for label, _, _ in FUNCS:
            base = np.asarray(res[f"base_{label}"], dtype=float)
            opt = np.asarray(res[f"opt_{label}"], dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                spd = base / opt
            print(
                f"{skey:<11}{label:<15}{int(x[0]):>9}{spd[0]:>8.1f}x"
                f"{int(x[-1]):>9}{spd[-1]:>8.1f}x{np.nanmax(spd):>8.1f}x"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="smaller sizes / fewer repeats")
    parser.add_argument("--repeats", type=int, default=None, help="max timed repeats per point")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "interactions_speed_benchmark.png",
        help="output figure path",
    )
    args = parser.parse_args()

    repeats = args.repeats if args.repeats is not None else (2 if args.quick else 5)
    print(f"Max repeats: {repeats}")

    if args.quick:
        users = [50, 250, 1000, 4000]
        depths = [1, 2, 4, 6]
        users_cfg = {"n_ue": 1000, "n_paths": 6, "n_depth": 4, "n_objects": 20}
        depth_cfg = {"n_ue": 2000, "n_paths": 6, "n_depth": 4, "n_objects": 20}
    else:
        users = [50, 250, 1000, 5000, 20000, 50000]
        depths = [1, 2, 3, 4, 6, 8]
        users_cfg = {"n_ue": 1000, "n_paths": 6, "n_depth": 4, "n_objects": 20}
        depth_cfg = {"n_ue": 5000, "n_paths": 6, "n_depth": 4, "n_objects": 20}

    verify()

    t_start = time.perf_counter()
    results = {}
    print(f"\nUsers sweep (n_paths={users_cfg['n_paths']}, n_depth={users_cfg['n_depth']}, "
          f"n_objects={users_cfg['n_objects']}):")
    results["users"] = run_sweep("users", users, users_cfg, max_repeats=repeats)
    print(f"Max-interaction sweep (n_ue={depth_cfg['n_ue']}, n_paths={depth_cfg['n_paths']}, "
          f"n_objects={depth_cfg['n_objects']}):")
    results["max_inter"] = run_sweep("max_inter", depths, depth_cfg, max_repeats=repeats)
    print(f"\nTotal benchmark time: {time.perf_counter() - t_start:.1f}s")

    print_summary(results)
    plot(results, args.out)


if __name__ == "__main__":
    main()
