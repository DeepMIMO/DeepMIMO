"""Benchmark the Wireless InSite ``paths_parser``: original vs optimized.

Times the optimized :func:`deepmimo.converters.wireless_insite.p2m_parser.paths_parser`
on this branch against the original implementation (kept here verbatim as the
baseline) on the real ~126 MB ASU campus paths file, and prints the speedup.

The baseline built every field with ``np.float32("<str>")`` (a slow per-scalar
numpy construction) inside a pure-Python per-(rx, path, interaction) loop. The
optimized version parses with Python ``float`` and commits each receiver's path
fields with vectorized assignments, which is byte-for-byte identical (float64 ->
float32 rounding matches ``np.float32`` of the string) but much faster.

Usage:
    python scripts/benchmark_p2m_parse.py
    python scripts/benchmark_p2m_parse.py --file /path/to/some.paths.p2m --repeats 5

If the input file is absent the script prints a notice and exits 0 (so it can be
run in environments where the large fixture has not been unpacked).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import time
from pathlib import Path

import numpy as np

from deepmimo import consts as c
from deepmimo.converters import converter_utils as cu
from deepmimo.converters.wireless_insite import p2m_parser

_DEFAULT_FILE = "/tmp/insite_asu/asu_campus/study_area_asu5/asu_campus.paths.t001_01.r004.p2m"

# Frozen verbatim constants from the original module.
_LINE_START = 22
_INTERACTIONS_MAP = {
    "R": c.INTERACTION_REFLECTION,
    "D": c.INTERACTION_DIFFRACTION,
    "DS": c.INTERACTION_SCATTERING,
    "T": c.INTERACTION_TRANSMISSION,
    "F": c.INTERACTION_TRANSMISSION,
    "X": c.INTERACTION_TRANSMISSION,
}


def _baseline_paths_parser(file: str) -> dict[str, np.ndarray]:
    """Verbatim copy of the ORIGINAL ``paths_parser`` (progress bar / print removed)."""
    with Path(file).open() as file_handle:
        lines = file_handle.readlines()

    n_rxs = int(lines[_LINE_START - 1])

    data = {
        c.AOA_AZ_PARAM_NAME: np.zeros((n_rxs, c.MAX_PATHS), dtype=np.float32) * np.nan,
        c.AOA_EL_PARAM_NAME: np.zeros((n_rxs, c.MAX_PATHS), dtype=np.float32) * np.nan,
        c.AOD_AZ_PARAM_NAME: np.zeros((n_rxs, c.MAX_PATHS), dtype=np.float32) * np.nan,
        c.AOD_EL_PARAM_NAME: np.zeros((n_rxs, c.MAX_PATHS), dtype=np.float32) * np.nan,
        c.DELAY_PARAM_NAME: np.zeros((n_rxs, c.MAX_PATHS), dtype=np.float32) * np.nan,
        c.POWER_PARAM_NAME: np.zeros((n_rxs, c.MAX_PATHS), dtype=np.float32) * np.nan,
        c.PHASE_PARAM_NAME: np.zeros((n_rxs, c.MAX_PATHS), dtype=np.float32) * np.nan,
        c.INTERACTIONS_PARAM_NAME: np.zeros((n_rxs, c.MAX_PATHS), dtype=np.float32) * np.nan,
        c.INTERACTIONS_POS_PARAM_NAME: np.zeros(
            (n_rxs, c.MAX_PATHS, c.MAX_INTER_PER_PATH, 3),
            dtype=np.float32,
        )
        * np.nan,
    }

    line_idx = _LINE_START
    for rx_i in range(n_rxs):
        line = lines[line_idx]
        rx_n_paths = int(line.split()[1])

        if rx_n_paths == 0:
            line_idx += 1
            continue

        n_paths_to_read = min(rx_n_paths, c.MAX_PATHS)
        line_idx += 2

        for path_idx in range(n_paths_to_read):
            line = lines[line_idx]
            _i1, i2, i3, i4, i5, i6, i7, i8, i9 = tuple(line.split())
            data[c.POWER_PARAM_NAME][rx_i, path_idx] = np.float32(i3)
            data[c.PHASE_PARAM_NAME][rx_i, path_idx] = np.float32(i4)
            data[c.DELAY_PARAM_NAME][rx_i, path_idx] = np.float32(i5)
            data[c.AOA_EL_PARAM_NAME][rx_i, path_idx] = np.float32(i6)
            data[c.AOA_AZ_PARAM_NAME][rx_i, path_idx] = np.float32(i7)
            data[c.AOD_EL_PARAM_NAME][rx_i, path_idx] = np.float32(i8)
            data[c.AOD_AZ_PARAM_NAME][rx_i, path_idx] = np.float32(i9)

            line = lines[line_idx + 1]
            inter_strs = line.split("-")[1:-1]
            inter_total_s = "".join([str(_INTERACTIONS_MAP[i_str]) for i_str in inter_strs])
            data[c.INTERACTIONS_PARAM_NAME][rx_i, path_idx] = (
                np.float32(inter_total_s) if inter_total_s else 0
            )

            n_iteractions = int(i2)
            for inter_idx in range(n_iteractions):
                line = lines[line_idx + 3 + inter_idx]
                xyz = [np.float32(i) for i in line.split()]
                data[c.INTERACTIONS_POS_PARAM_NAME][rx_i, path_idx, inter_idx] = xyz

            line_idx += 4 + n_iteractions

    return cu.compress_path_data(data)


def _silence_tqdm() -> None:
    """Replace the parser's tqdm with an identity passthrough (no progress-bar skew)."""
    p2m_parser.tqdm = lambda iterable=None, **_kw: iterable if iterable is not None else []


def _time_call(fn, file: str, repeats: int) -> list[float]:
    """Time ``fn(file)`` ``repeats`` times (stdout suppressed), returning per-run seconds."""
    timings = []
    for _ in range(repeats):
        with contextlib.redirect_stdout(io.StringIO()):
            t0 = time.perf_counter()
            fn(file)
            timings.append(time.perf_counter() - t0)
    return timings


def _outputs_identical(file: str) -> bool:
    """Return True iff optimized and baseline produce byte-for-byte identical output."""
    with contextlib.redirect_stdout(io.StringIO()):
        opt = p2m_parser.paths_parser(file)
        base = _baseline_paths_parser(file)
    if opt.keys() != base.keys():
        return False
    return all(
        opt[k].dtype == base[k].dtype
        and opt[k].shape == base[k].shape
        and np.array_equal(opt[k], base[k], equal_nan=True)
        for k in base
    )


def main() -> None:
    """Run the baseline-vs-optimized timing comparison and print a summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", default=_DEFAULT_FILE, help="path to a .paths.p2m file")
    parser.add_argument("--repeats", type=int, default=3, help="timed repeats per parser")
    args = parser.parse_args()

    file = args.file
    if not Path(file).exists():
        print(f"[skip] Paths file not found: {file}")
        print("       Unpack it first, e.g.:")
        print("       rm -rf /tmp/insite_asu && unzip -q asu_campus_p2m.zip -d /tmp/insite_asu")
        return

    size_mb = Path(file).stat().st_size / 1e6
    print(f"File: {file}  ({size_mb:.1f} MB)")
    print(f"Repeats: {args.repeats}\n")

    _silence_tqdm()

    # Warm the OS page cache so the first timed read is not penalized.
    with contextlib.redirect_stdout(io.StringIO()):
        _baseline_paths_parser(file)

    print("Checking byte-for-byte equivalence... ", end="", flush=True)
    identical = _outputs_identical(file)
    print("IDENTICAL" if identical else "DIFFERENT (!)")
    if not identical:
        msg = "Optimized output differs from baseline; benchmark numbers are meaningless."
        raise SystemExit(msg)

    base_times = _time_call(_baseline_paths_parser, file, args.repeats)
    opt_times = _time_call(p2m_parser.paths_parser, file, args.repeats)

    base_best, opt_best = min(base_times), min(opt_times)
    base_mean = sum(base_times) / len(base_times)
    opt_mean = sum(opt_times) / len(opt_times)

    print("\n--- Results (seconds) ---")
    print(f"baseline (original) : best {base_best:7.3f}  mean {base_mean:7.3f}  runs {_fmt(base_times)}")
    print(f"optimized           : best {opt_best:7.3f}  mean {opt_mean:7.3f}  runs {_fmt(opt_times)}")
    print("\n--- Speedup (baseline / optimized) ---")
    print(f"best-of-{args.repeats}: {base_best / opt_best:.2f}x")
    print(f"mean    : {base_mean / opt_mean:.2f}x")


def _fmt(times: list[float]) -> str:
    """Format a list of timings compactly."""
    return "[" + ", ".join(f"{t:.3f}" for t in times) + "]"


if __name__ == "__main__":
    main()
