"""Byte-for-byte regression tests for the optimized ``paths_parser``.

This module freezes a *verbatim* copy of the original (pre-optimization)
``paths_parser`` implementation as :func:`_baseline_paths_parser` and asserts
that the current (optimized)
:func:`deepmimo.converters.wireless_insite.p2m_parser.paths_parser` produces
byte-for-byte identical output: identical dict keys, array dtypes, shapes,
values and NaN placement.

Coverage of the synthetic fixtures:
    - a receiver with 0 paths
    - receivers with multiple paths
    - paths with multiple interactions
    - every interaction type in ``INTERACTIONS_MAP`` (R, D, DS, T, F, X)
    - a line-of-sight path (``Tx-Rx`` -> 0 interactions -> code 0)
    - a receiver with more than ``MAX_PATHS`` paths (clipping)
    - a path with more than ``MAX_INTER_PER_PATH`` interactions (the original
      raises ``IndexError``; the optimized parser must raise identically)

If the real 126 MB ASU campus paths file is present under ``/tmp`` it is also
checked, guarded to skip when absent.
"""

from pathlib import Path

import numpy as np
import pytest

from deepmimo import consts as c
from deepmimo.converters import converter_utils as cu
from deepmimo.converters.wireless_insite import p2m_parser

# --- Frozen verbatim constants from the original module (file-format semantics) ---
_BASELINE_LINE_START = 22
_BASELINE_INTERACTIONS_MAP = {
    "R": c.INTERACTION_REFLECTION,
    "D": c.INTERACTION_DIFFRACTION,
    "DS": c.INTERACTION_SCATTERING,
    "T": c.INTERACTION_TRANSMISSION,
    "F": c.INTERACTION_TRANSMISSION,
    "X": c.INTERACTION_TRANSMISSION,
}


def _baseline_paths_parser(file: str) -> dict[str, np.ndarray]:
    """Verbatim copy of the ORIGINAL ``paths_parser`` (pre-optimization).

    The only changes versus the shipped original are cosmetic and do not affect
    the returned data: the ``tqdm`` progress wrapper and the ``print`` call were
    removed so the test suite stays quiet. The parsing / conversion logic
    (``np.float32`` per-field construction, line indexing, clipping and the final
    ``compress_path_data`` call) is preserved exactly so this acts as a frozen
    correctness oracle.
    """
    with Path(file).open() as file_handle:
        lines = file_handle.readlines()

    n_rxs = int(lines[_BASELINE_LINE_START - 1])

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

    line_idx = _BASELINE_LINE_START
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
            inter_total_s = "".join(
                [str(_BASELINE_INTERACTIONS_MAP[i_str]) for i_str in inter_strs],
            )
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


# --------------------------------------------------------------------------- #
# Synthetic .p2m fixture construction helpers
# --------------------------------------------------------------------------- #

_HEADER_LINES = 21  # comment lines preceding the <n_rxs> line (LINE_START - 1)

# A path's 7 numeric header fields, in file order:
# (power, phase, delay, aoa_el, aoa_az, aod_el, aod_az)
Fields = tuple[float, float, float, float, float, float, float]


def _path_block(pnum: int, inter_str: str, xyzs: list[list[float]], fields: Fields) -> str:
    """Build one path block: header line, interaction string, tx pos, xyz*, rx pos."""
    n_inter = len(xyzs)
    header = f"{pnum} {n_inter} " + " ".join(str(v) for v in fields)
    rows = [header, inter_str, "0.0 0.0 10.0"]  # last entry = Tx pos (skipped)
    rows += [" ".join(str(v) for v in xyz) for xyz in xyzs]
    rows.append("100.0 0.0 1.5")  # Rx pos (skipped)
    return "\n".join(rows) + "\n"


def _rx_block(rx_idx: int, paths: list[str]) -> str:
    """Build a receiver block: '<rx_idx> <n_paths>' [+ summary line + path blocks]."""
    if not paths:
        return f"{rx_idx} 0\n"
    block = f"{rx_idx} {len(paths)}\n"
    block += "-100.0 1.0e-06 1.0e-07\n"  # rx point summary line (skipped)
    return block + "".join(paths)


def _write_p2m(path: Path, rx_blocks: list[str]) -> str:
    """Assemble a full synthetic .p2m file from receiver blocks."""
    with path.open("w") as f:
        f.writelines(f"# Header comment {i + 1}\n" for i in range(_HEADER_LINES))
        f.write(f"{len(rx_blocks)}\n")
        for block in rx_blocks:
            f.write(block)
    return str(path)


def _comprehensive_rx_blocks() -> list[str]:
    """Receiver blocks covering 0-paths, LOS, every interaction type and combos."""
    los = _path_block(
        1, "Tx-Rx", [], (-70.1, 12.5, 3.33e-07, 80.0, 10.0, 95.0, -20.0),
    )
    refl = _path_block(
        2, "Tx-R-Rx", [[10.5, -20.25, 3.125]], (-85.5, -44.0, 1.2e-06, 30.0, 60.0, 120.0, -10.0),
    )
    diff = _path_block(
        3, "Tx-D-Rx", [[1.1, 2.2, 3.3]], (-90.0, 170.0, 1.5e-06, 33.0, -80.0, 90.0, -145.0),
    )
    scat = _path_block(
        4, "Tx-DS-Rx", [[-5.5, 6.6, 7.7]], (-100.25, 0.0, 1.9e-06, 44.0, 44.0, 91.0, -137.0),
    )
    trans_t = _path_block(
        5, "Tx-T-Rx", [[8.0, 9.0, 1.0]], (-95.0, -5.0, 2.0e-06, 50.0, 12.0, 88.0, 4.0),
    )
    trans_f = _path_block(
        6, "Tx-F-Rx", [[2.5, 3.5, 4.5]], (-96.5, 8.0, 2.1e-06, 51.0, 13.0, 87.0, 5.0),
    )
    trans_x = _path_block(
        7, "Tx-X-Rx", [[6.25, 7.25, 8.25]], (-97.5, 9.0, 2.2e-06, 52.0, 14.0, 86.0, 6.0),
    )
    # Multi-interaction path mixing several interaction types (R, DS, D, T, F, X).
    multi = _path_block(
        8,
        "Tx-R-DS-D-T-F-X-Rx",
        [
            [-114.203, -153.07, 15.6214],
            [-161.121, -113.029, 9.14275],
            [-216.551, -160.169, 1.5],
            [12.0, 13.0, 14.0],
            [15.0, 16.0, 17.0],
            [18.0, 19.0, 20.0],
        ],
        (-133.172, 31.9552, 1.72008e-06, 84.004, 40.3789, 90.961, -137.465),
    )

    return [
        _rx_block(1, []),  # rx with 0 paths
        _rx_block(2, [los, refl, diff]),  # multiple paths incl. LOS
        _rx_block(3, [scat, trans_t, trans_f, trans_x]),  # remaining single types
        _rx_block(4, [multi]),  # multi-interaction path
        _rx_block(5, []),  # another 0-path rx (trailing)
    ]


def _gt_max_paths_rx(n_paths: int = 30) -> str:
    """Build a single receiver block with more than ``MAX_PATHS`` paths (clipping)."""
    paths = [
        _path_block(
            k + 1,
            "Tx-R-Rx",
            [[float(k), float(k) + 1.0, float(k) + 2.0]],
            (-80.0 - k, float(k), 1.0e-06 + k * 1e-9, 30.0 + k, 60.0 - k, 120.0, -10.0),
        )
        for k in range(n_paths)
    ]
    return _rx_block(1, paths)


# --------------------------------------------------------------------------- #
# Comparison helper
# --------------------------------------------------------------------------- #


def _assert_identical(opt: dict[str, np.ndarray], base: dict[str, np.ndarray]) -> None:
    """Assert two parser outputs are byte-for-byte identical."""
    assert opt.keys() == base.keys()
    for key, b in base.items():
        a = opt[key]
        assert a.dtype == b.dtype, f"{key}: dtype {a.dtype} != {b.dtype}"
        assert a.shape == b.shape, f"{key}: shape {a.shape} != {b.shape}"
        # Values + NaN placement.
        assert np.array_equal(a, b, equal_nan=True), f"{key}: values differ"
        # Bit-exact check on finite entries (also distinguishes -0.0 from +0.0).
        finite = ~np.isnan(a)
        assert np.array_equal(
            a[finite].view(np.uint8), b[finite].view(np.uint8),
        ), f"{key}: finite bit pattern differs"


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_comprehensive_fixture_byte_identical(tmp_path) -> None:
    """All interaction types, LOS, 0-path rxs and multi-interaction paths."""
    file = _write_p2m(tmp_path / "comprehensive.paths.p2m", _comprehensive_rx_blocks())
    opt = p2m_parser.paths_parser(file)
    base = _baseline_paths_parser(file)
    _assert_identical(opt, base)

    # Sanity: scattering code (DS -> 3) is present.
    assert np.float32(3.0) in opt[c.INTERACTIONS_PARAM_NAME]


def test_single_path_single_interaction_byte_identical(tmp_path) -> None:
    """Minimal file: one rx, one path, one reflection."""
    refl = _path_block(
        1, "Tx-R-Rx", [[50.0, 50.0, 5.0]], (-80.5, 45.0, 1.2e-06, 30.0, 60.0, 120.0, -10.0),
    )
    file = _write_p2m(
        tmp_path / "single.paths.p2m", [_rx_block(1, [refl]), _rx_block(2, [])],
    )
    opt = p2m_parser.paths_parser(file)
    base = _baseline_paths_parser(file)
    _assert_identical(opt, base)


def test_gt_max_paths_clipping_byte_identical(tmp_path) -> None:
    """A trailing receiver with >MAX_PATHS paths is clipped to MAX_PATHS identically."""
    file = _write_p2m(tmp_path / "many.paths.p2m", [_gt_max_paths_rx(30)])
    opt = p2m_parser.paths_parser(file)
    base = _baseline_paths_parser(file)
    _assert_identical(opt, base)
    assert opt[c.POWER_PARAM_NAME].shape[1] <= c.MAX_PATHS


def test_gt_max_paths_followed_by_rx_matches_baseline(tmp_path) -> None:
    """>MAX_PATHS clipping leaves the cursor mid-receiver, mis-aligning the next rx.

    This is a latent property of the original line-stride logic: after a receiver
    with more than ``MAX_PATHS`` paths, the unread path lines shift every later
    receiver. The optimized parser must reproduce that behavior bit-for-bit, which
    here means raising the identical ``ValueError`` from the 9-field header unpack.
    """
    file = _write_p2m(
        tmp_path / "many_then_rx.paths.p2m",
        [_gt_max_paths_rx(30), _rx_block(2, [])],
    )
    with pytest.raises(ValueError, match="not enough values to unpack") as base_exc:
        _baseline_paths_parser(file)
    with pytest.raises(ValueError, match="not enough values to unpack") as opt_exc:
        p2m_parser.paths_parser(file)
    assert str(opt_exc.value) == str(base_exc.value)


def test_gt_max_inter_per_path_matches_baseline(tmp_path) -> None:
    """>MAX_INTER_PER_PATH interactions: optimized must match baseline behavior.

    The original parser indexes ``inter_pos[..., inter_idx]`` directly, so more
    than ``MAX_INTER_PER_PATH`` interactions raises ``IndexError``. The optimized
    parser must reproduce that exact failure mode (same type and message).
    """
    n_inter = c.MAX_INTER_PER_PATH + 2
    inter_str = "Tx-" + "-".join(["R"] * n_inter) + "-Rx"
    xyzs = [[float(i), float(i) + 1.0, float(i) + 2.0] for i in range(n_inter)]
    path = _path_block(
        1, inter_str, xyzs, (-80.0, 45.0, 1.0e-06, 30.0, 60.0, 120.0, -10.0),
    )
    file = _write_p2m(tmp_path / "biginter.paths.p2m", [_rx_block(1, [path])])

    with pytest.raises(IndexError, match="out of bounds") as base_exc:
        _baseline_paths_parser(file)
    with pytest.raises(IndexError, match="out of bounds") as opt_exc:
        p2m_parser.paths_parser(file)
    assert str(opt_exc.value) == str(base_exc.value)


_REAL_FILE = Path(
    "/tmp/insite_asu/asu_campus/study_area_asu5/asu_campus.paths.t001_01.r004.p2m",  # noqa: S108
)


@pytest.mark.skipif(not _REAL_FILE.exists(), reason="real ASU campus paths file absent")
def test_real_file_byte_identical() -> None:
    """The optimized parser matches the baseline on the real 126 MB paths file."""
    opt = p2m_parser.paths_parser(str(_REAL_FILE))
    base = _baseline_paths_parser(str(_REAL_FILE))
    _assert_identical(opt, base)
