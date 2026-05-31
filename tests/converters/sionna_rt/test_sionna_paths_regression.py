"""Byte-for-byte regression tests for the vectorized Sionna path conversion.

``transform_interaction_types`` and ``_process_paths_batch`` were rewritten from
per-receiver / per-path Python loops into vectorized NumPy. This module freezes
*verbatim* copies of the original (pre-optimization) implementations as
:func:`_baseline_transform_interaction_types` and
:func:`_baseline_process_paths_batch` and asserts the current vectorized
implementations produce byte-for-byte identical output.

Synthetic fixtures exercise the edge cases the optimization must preserve:
    - receivers with 0 active paths (inactive count + all-NaN rows),
    - LoS paths (all-zero interaction slots -> code 0),
    - every Sionna 2.0 interaction type and multi-bounce sequences,
    - NONE(0) padding *between* bounces (e.g. ``[1, 0, 1]`` -> code 11),
    - interaction positions that are exactly 0.0 with a non-zero type (must NOT
      be nulled) vs. NONE slots (must be nulled),
    - more than ``MAX_PATHS`` active paths (clip-then-sort-by-power),
    - targets absent from the global ``rx_pos`` grid (skipped, not counted),
    - single- and multi-antenna array layouts.

The optimized sort is deterministic (stable); the original used the default
(quicksort) ``argsort``. For *distinct* per-receiver magnitudes both orderings
are identical, so all fixtures use distinct magnitudes to keep the comparison
byte-exact. (Exact magnitude ties are physically degenerate and their ordering
is arbitrary in the original too.)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from deepmimo import consts as c
from deepmimo.converters.sionna_rt import sionna_paths
from deepmimo.converters.sionna_rt.sionna_paths import (
    _SIONNA_TO_DEEPMIMO,
    _build_rx_pos_index,
)
from deepmimo.utils import load_pickle

# --- Frozen verbatim constants from the original module (algorithm semantics) ---
_BASELINE_SIONNA_NONE = 0
_BASELINE_MULTI_ANT_NDIM = 3


# --------------------------------------------------------------------------- #
# Verbatim copies of the ORIGINAL loop implementations (frozen oracle).
# --------------------------------------------------------------------------- #


def _baseline_transform_interaction_types(types: np.ndarray) -> np.ndarray:
    """Verbatim copy of the ORIGINAL per-path ``transform_interaction_types``."""
    n_paths = types.shape[0]
    result = np.zeros(n_paths, dtype=np.float32)

    for i in range(n_paths):
        path = types[i]

        if np.all(path == 0):
            result[i] = c.INTERACTION_LOS
            continue

        non_zero_indices = np.where(path != 0)[0]
        valid_raw = path[: non_zero_indices[-1] + 1]

        remapped = [_SIONNA_TO_DEEPMIMO.get(int(x), int(x)) for x in valid_raw if x != 0]
        result[i] = float("".join(str(v) for v in remapped))

    return result


def _baseline_process_paths_batch(  # noqa: PLR0913, PLR0915
    paths_dict: dict,
    data: dict,
    t: int,
    targets: np.ndarray,
    rx_pos: np.ndarray,
    tx_ant_idx: int = 0,
    rx_ant_idx: int = 0,
) -> int:
    """Verbatim copy of the ORIGINAL per-receiver ``_process_paths_batch``."""
    inactive_count = 0

    a = paths_dict["a"]
    tau = paths_dict["tau"]
    phi_r = paths_dict["phi_r"]
    phi_t = paths_dict["phi_t"]
    theta_r = paths_dict["theta_r"]
    theta_t = paths_dict["theta_t"]
    vertices = paths_dict["vertices"]
    types = paths_dict["interactions"]

    tx_idx = t

    if theta_r.ndim > _BASELINE_MULTI_ANT_NDIM:
        a = a[:, rx_ant_idx, tx_idx, tx_ant_idx, :]
        tau = tau[:, rx_ant_idx, tx_idx, tx_ant_idx, :]
        phi_r = phi_r[:, rx_ant_idx, tx_idx, tx_ant_idx, :]
        phi_t = phi_t[:, rx_ant_idx, tx_idx, tx_ant_idx, :]
        theta_r = theta_r[:, rx_ant_idx, tx_idx, tx_ant_idx, :]
        theta_t = theta_t[:, rx_ant_idx, tx_idx, tx_ant_idx, :]
        types = types[:, :, rx_ant_idx, :, tx_ant_idx]
        vertices = vertices[:, :, rx_ant_idx, tx_idx, tx_ant_idx, :]
    else:
        a = a[:, 0, tx_idx, 0, :]
        tau = tau[:, tx_idx, :]
        phi_r = phi_r[:, tx_idx, :]
        phi_t = phi_t[:, tx_idx, :]
        theta_r = theta_r[:, tx_idx, :]
        theta_t = theta_t[:, tx_idx, :]
        vertices = vertices[:, :, tx_idx, ...]

    n_rx = a.shape[0]
    for rel_rx_idx in range(n_rx):
        abs_idx_arr = np.where(np.all(rx_pos == targets[rel_rx_idx], axis=1))[0]
        if len(abs_idx_arr) == 0:
            continue
        abs_idx = abs_idx_arr[0]

        amp = a[rel_rx_idx]

        non_zero_path_idxs = np.where(amp != 0)[0][: c.MAX_PATHS]
        n_paths = len(non_zero_path_idxs)
        if n_paths == 0:
            inactive_count += 1
            continue

        sorted_path_idxs = np.argsort(np.abs(amp[non_zero_path_idxs]))[::-1]
        path_idxs = non_zero_path_idxs[sorted_path_idxs]

        data[c.POWER_PARAM_NAME][abs_idx, :n_paths] = 20 * np.log10(np.abs(amp[path_idxs]))
        data[c.PHASE_PARAM_NAME][abs_idx, :n_paths] = np.angle(amp[path_idxs], deg=True)

        data[c.AOA_AZ_PARAM_NAME][abs_idx, :n_paths] = np.rad2deg(phi_r[rel_rx_idx, path_idxs])
        data[c.AOD_AZ_PARAM_NAME][abs_idx, :n_paths] = np.rad2deg(phi_t[rel_rx_idx, path_idxs])
        data[c.AOA_EL_PARAM_NAME][abs_idx, :n_paths] = np.rad2deg(theta_r[rel_rx_idx, path_idxs])
        data[c.AOD_EL_PARAM_NAME][abs_idx, :n_paths] = np.rad2deg(theta_t[rel_rx_idx, path_idxs])

        data[c.DELAY_PARAM_NAME][abs_idx, :n_paths] = tau[rel_rx_idx, path_idxs]

        path_types = types[:, rel_rx_idx, tx_idx, path_idxs].swapaxes(0, 1)

        inter_pos_rx = vertices[:, rel_rx_idx, path_idxs, :].swapaxes(0, 1)
        n_interactions = inter_pos_rx.shape[1]
        inter_pos_rx[path_types == _BASELINE_SIONNA_NONE] = np.nan
        data[c.INTERACTIONS_POS_PARAM_NAME][abs_idx, :n_paths, :n_interactions, :] = inter_pos_rx

        data[c.INTERACTIONS_PARAM_NAME][abs_idx, :n_paths] = _baseline_transform_interaction_types(
            path_types
        )

    return inactive_count


# --------------------------------------------------------------------------- #
# Synthetic Sionna 2.0 batch construction.
# --------------------------------------------------------------------------- #


def _distinct_amplitudes(rng: np.random.Generator, n_active: int) -> np.ndarray:
    """Return ``n_active`` complex amplitudes with strictly distinct magnitudes."""
    # Distinct magnitudes (sorted noise then perturbed) -> deterministic argsort.
    mags = np.sort(rng.uniform(0.5, 50.0, n_active))
    mags += np.arange(n_active) * 1e-3  # guarantee strict monotonic uniqueness
    phases = rng.uniform(-np.pi, np.pi, n_active)
    return (mags * np.exp(1j * phases)).astype(np.complex128)


def _random_interaction_row(rng: np.random.Generator, max_depth: int) -> np.ndarray:
    """One per-path interaction sequence using Sionna enum values {0,1,2,4,8}."""
    sionna_vals = np.array([1, 2, 4, 8])
    roll = rng.random()
    if roll < 0.25:
        return np.zeros(max_depth, dtype=np.float64)  # LoS
    k = int(rng.integers(1, max_depth + 1))
    row = np.zeros(max_depth, dtype=np.float64)
    row[:k] = rng.choice(sionna_vals, size=k)
    return row


def _make_single_ant_batch(  # noqa: PLR0913
    rng: np.random.Generator,
    *,
    n_rx: int,
    n_tx: int,
    max_paths: int,
    max_depth: int,
    max_active: int | None = None,
) -> dict:
    """Build a single-antenna Sionna 2.0 ``paths_dict`` with diverse receivers."""
    if max_active is None:
        max_active = max_paths
    a = np.zeros((n_rx, 1, n_tx, 1, max_paths), dtype=np.complex128)
    tau = np.full((n_rx, n_tx, max_paths), np.nan)
    phi_r = np.full((n_rx, n_tx, max_paths), np.nan)
    phi_t = np.full((n_rx, n_tx, max_paths), np.nan)
    theta_r = np.full((n_rx, n_tx, max_paths), np.nan)
    theta_t = np.full((n_rx, n_tx, max_paths), np.nan)
    interactions = np.zeros((max_depth, n_rx, n_tx, max_paths), dtype=np.float64)
    vertices = np.full((max_depth, n_rx, n_tx, max_paths, 3), np.nan)

    for rx in range(n_rx):
        for tx in range(n_tx):
            n_active = int(rng.integers(0, max_active + 1))
            if n_active == 0:
                continue
            # Active paths live at arbitrary (sorted) Sionna slots.
            slots = np.sort(rng.choice(max_paths, size=n_active, replace=False))
            a[rx, 0, tx, 0, slots] = _distinct_amplitudes(rng, n_active)
            tau[rx, tx, slots] = rng.uniform(1e-8, 1e-6, n_active)
            phi_r[rx, tx, slots] = rng.uniform(-np.pi, np.pi, n_active)
            phi_t[rx, tx, slots] = rng.uniform(-np.pi, np.pi, n_active)
            theta_r[rx, tx, slots] = rng.uniform(0, np.pi, n_active)
            theta_t[rx, tx, slots] = rng.uniform(0, np.pi, n_active)
            for s in slots:
                row = _random_interaction_row(rng, max_depth)
                interactions[:, rx, tx, s] = row
                for d in range(max_depth):
                    if row[d] != 0:
                        vertices[d, rx, tx, s] = rng.uniform(-100, 100, 3)

    return {
        "a": a,
        "tau": tau,
        "phi_r": phi_r,
        "phi_t": phi_t,
        "theta_r": theta_r,
        "theta_t": theta_t,
        "interactions": interactions,
        "vertices": vertices,
    }


def _make_multi_ant_batch(  # noqa: PLR0913
    rng: np.random.Generator,
    *,
    n_rx: int,
    n_rx_ant: int,
    n_tx: int,
    n_tx_ant: int,
    max_paths: int,
    max_depth: int,
) -> dict:
    """Build a multi-antenna Sionna 2.0 ``paths_dict`` (antenna dims inserted)."""
    shape = (n_rx, n_rx_ant, n_tx, n_tx_ant, max_paths)
    a = np.zeros(shape, dtype=np.complex128)
    tau = np.full(shape, np.nan)
    phi_r = np.full(shape, np.nan)
    phi_t = np.full(shape, np.nan)
    theta_r = np.full(shape, np.nan)
    theta_t = np.full(shape, np.nan)
    interactions = np.zeros((max_depth, n_rx, n_rx_ant, n_tx, n_tx_ant, max_paths))
    vertices = np.full((max_depth, n_rx, n_rx_ant, n_tx, n_tx_ant, max_paths, 3), np.nan)

    for rx in range(n_rx):
        for ra in range(n_rx_ant):
            for tx in range(n_tx):
                for ta in range(n_tx_ant):
                    n_active = int(rng.integers(0, max_paths + 1))
                    if n_active == 0:
                        continue
                    slots = np.sort(rng.choice(max_paths, size=n_active, replace=False))
                    a[rx, ra, tx, ta, slots] = _distinct_amplitudes(rng, n_active)
                    tau[rx, ra, tx, ta, slots] = rng.uniform(1e-8, 1e-6, n_active)
                    phi_r[rx, ra, tx, ta, slots] = rng.uniform(-np.pi, np.pi, n_active)
                    phi_t[rx, ra, tx, ta, slots] = rng.uniform(-np.pi, np.pi, n_active)
                    theta_r[rx, ra, tx, ta, slots] = rng.uniform(0, np.pi, n_active)
                    theta_t[rx, ra, tx, ta, slots] = rng.uniform(0, np.pi, n_active)
                    for s in slots:
                        row = _random_interaction_row(rng, max_depth)
                        interactions[:, rx, ra, tx, ta, s] = row
                        for d in range(max_depth):
                            if row[d] != 0:
                                vertices[d, rx, ra, tx, ta, s] = rng.uniform(-100, 100, 3)

    return {
        "a": a,
        "tau": tau,
        "phi_r": phi_r,
        "phi_t": phi_t,
        "theta_r": theta_r,
        "theta_t": theta_t,
        "interactions": interactions,
        "vertices": vertices,
    }


def _run(process_fn, paths_dict, *, targets, rx_pos, use_index, **kwargs) -> tuple[dict, int]:
    """Preallocate a data dict, run ``process_fn`` once and return (data, inactive)."""
    data = sionna_paths._preallocate_data(len(rx_pos))  # noqa: SLF001
    data[c.RX_POS_PARAM_NAME] = rx_pos
    mapper = _build_rx_pos_index(rx_pos) if use_index else rx_pos
    inactive = process_fn(paths_dict, data, kwargs.pop("t", 0), targets, mapper, **kwargs)
    return data, inactive


def _assert_data_identical(opt: dict, base: dict) -> None:
    """Assert two converted data dicts are byte-for-byte identical."""
    path_keys = [
        c.POWER_PARAM_NAME,
        c.PHASE_PARAM_NAME,
        c.AOA_AZ_PARAM_NAME,
        c.AOA_EL_PARAM_NAME,
        c.AOD_AZ_PARAM_NAME,
        c.AOD_EL_PARAM_NAME,
        c.DELAY_PARAM_NAME,
        c.INTERACTIONS_PARAM_NAME,
        c.INTERACTIONS_POS_PARAM_NAME,
    ]
    for key in path_keys:
        a = opt[key]
        b = base[key]
        assert a.dtype == b.dtype, f"{key}: dtype {a.dtype} != {b.dtype}"
        assert a.shape == b.shape, f"{key}: shape {a.shape} != {b.shape}"
        assert np.array_equal(a, b, equal_nan=True), f"{key}: values differ"
        finite = ~np.isnan(a)
        assert np.array_equal(a[finite].view(np.uint8), b[finite].view(np.uint8)), (
            f"{key}: finite bit pattern differs"
        )


# --------------------------------------------------------------------------- #
# Tests: transform_interaction_types
# --------------------------------------------------------------------------- #

_HANDCRAFTED_TYPES = np.array(
    [
        [0, 0, 0, 0, 0],  # LoS -> 0
        [1, 0, 0, 0, 0],  # specular -> 1
        [1, 1, 0, 0, 0],  # 2x specular -> 11
        [1, 1, 1, 0, 0],  # 3x specular -> 111
        [2, 0, 0, 0, 0],  # diffuse -> 3
        [4, 0, 0, 0, 0],  # refraction -> 4
        [8, 0, 0, 0, 0],  # diffraction -> 2
        [1, 2, 0, 0, 0],  # refl + scatter -> 13
        [1, 8, 0, 0, 0],  # refl + diffraction -> 12
        [8, 1, 0, 0, 0],  # diffraction + refl -> 21
        [4, 1, 2, 8, 0],  # mixed -> 4132
        [1, 0, 1, 0, 0],  # padding zero between bounces -> 11
        [1, 2, 4, 8, 1],  # full-depth mixed -> 13421
    ],
    dtype=np.float64,
)


def test_transform_interaction_types_handcrafted_matches_loop() -> None:
    """Hand-crafted interaction patterns match the original loop exactly."""
    opt = sionna_paths.transform_interaction_types(_HANDCRAFTED_TYPES)
    base = _baseline_transform_interaction_types(_HANDCRAFTED_TYPES)
    assert opt.dtype == base.dtype == np.float32
    np.testing.assert_array_equal(opt, base)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 7])
@pytest.mark.parametrize("max_depth", [1, 3, 5])
def test_transform_interaction_types_random_matches_loop(seed: int, max_depth: int) -> None:
    """Randomized interaction sequences match the original loop exactly."""
    rng = np.random.default_rng(seed)
    rows = [_random_interaction_row(rng, max_depth) for _ in range(200)]
    types = np.stack(rows)
    opt = sionna_paths.transform_interaction_types(types)
    base = _baseline_transform_interaction_types(types)
    np.testing.assert_array_equal(opt, base)


def test_transform_interaction_types_empty() -> None:
    """Zero-path input returns an empty float32 array (matches the loop)."""
    types = np.zeros((0, 4), dtype=np.float64)
    opt = sionna_paths.transform_interaction_types(types)
    base = _baseline_transform_interaction_types(types)
    assert opt.shape == base.shape == (0,)
    assert opt.dtype == np.float32


# --------------------------------------------------------------------------- #
# Tests: _process_paths_batch (single antenna)
# --------------------------------------------------------------------------- #

# (seed, n_rx, n_tx, max_paths, max_depth)
_SINGLE_CASES = [
    (0, 30, 1, 12, 3),
    (1, 50, 1, 8, 5),
    (2, 16, 1, 20, 4),
    (3, 64, 1, 10, 2),
]


@pytest.mark.parametrize("case", _SINGLE_CASES)
def test_process_batch_single_antenna_byte_identical(case) -> None:
    """Single-antenna batch conversion is byte-identical to the original loop."""
    seed, n_rx, n_tx, max_paths, max_depth = case
    rng = np.random.default_rng(seed)
    paths_dict = _make_single_ant_batch(
        rng, n_rx=n_rx, n_tx=n_tx, max_paths=max_paths, max_depth=max_depth
    )
    # Global rx grid: batch targets placed at non-trivial global indices.
    targets = rng.uniform(-200, 200, (n_rx, 3))
    extra = rng.uniform(-200, 200, (7, 3))
    rx_pos = np.vstack([extra[:3], targets, extra[3:]])

    opt_data, opt_inactive = _run(
        sionna_paths._process_paths_batch,  # noqa: SLF001
        paths_dict,
        targets=targets,
        rx_pos=rx_pos,
        use_index=True,
    )
    base_data, base_inactive = _run(
        _baseline_process_paths_batch, paths_dict, targets=targets, rx_pos=rx_pos, use_index=False
    )
    assert opt_inactive == base_inactive
    _assert_data_identical(opt_data, base_data)


@pytest.mark.parametrize("n_tx", [2, 3])
def test_process_batch_multi_tx_columns(n_tx: int) -> None:
    """Each TX column is converted identically (single antenna, multi-TX dict)."""
    rng = np.random.default_rng(100 + n_tx)
    n_rx, max_paths, max_depth = 24, 10, 4
    paths_dict = _make_single_ant_batch(
        rng, n_rx=n_rx, n_tx=n_tx, max_paths=max_paths, max_depth=max_depth
    )
    targets = rng.uniform(-200, 200, (n_rx, 3))
    rx_pos = targets.copy()
    for tx in range(n_tx):
        opt_data, opt_inactive = _run(
            sionna_paths._process_paths_batch,  # noqa: SLF001
            paths_dict,
            targets=targets,
            rx_pos=rx_pos,
            use_index=True,
            t=tx,
        )
        base_data, base_inactive = _run(
            _baseline_process_paths_batch,
            paths_dict,
            targets=targets,
            rx_pos=rx_pos,
            use_index=False,
            t=tx,
        )
        assert opt_inactive == base_inactive
        _assert_data_identical(opt_data, base_data)


def test_process_batch_more_than_max_paths() -> None:
    """>MAX_PATHS active paths: clip-to-first-MAX_PATHS then sort-by-power identically."""
    rng = np.random.default_rng(11)
    max_paths = c.MAX_PATHS + 15
    paths_dict = _make_single_ant_batch(
        rng, n_rx=12, n_tx=1, max_paths=max_paths, max_depth=3, max_active=max_paths
    )
    targets = rng.uniform(-200, 200, (12, 3))
    rx_pos = targets.copy()
    opt_data, opt_inactive = _run(
        sionna_paths._process_paths_batch,  # noqa: SLF001
        paths_dict,
        targets=targets,
        rx_pos=rx_pos,
        use_index=True,
    )
    base_data, base_inactive = _run(
        _baseline_process_paths_batch, paths_dict, targets=targets, rx_pos=rx_pos, use_index=False
    )
    assert opt_inactive == base_inactive
    assert opt_data[c.POWER_PARAM_NAME].shape[1] <= c.MAX_PATHS
    _assert_data_identical(opt_data, base_data)


def test_process_batch_all_inactive() -> None:
    """A batch of receivers with zero paths: all-NaN rows + full inactive count."""
    rng = np.random.default_rng(5)
    n_rx = 10
    paths_dict = _make_single_ant_batch(
        rng, n_rx=n_rx, n_tx=1, max_paths=6, max_depth=3, max_active=0
    )
    targets = rng.uniform(-200, 200, (n_rx, 3))
    rx_pos = targets.copy()
    opt_data, opt_inactive = _run(
        sionna_paths._process_paths_batch,  # noqa: SLF001
        paths_dict,
        targets=targets,
        rx_pos=rx_pos,
        use_index=True,
    )
    base_data, base_inactive = _run(
        _baseline_process_paths_batch, paths_dict, targets=targets, rx_pos=rx_pos, use_index=False
    )
    assert opt_inactive == base_inactive == n_rx
    _assert_data_identical(opt_data, base_data)


def test_process_batch_target_not_in_rx_pos() -> None:
    """Targets absent from the global grid are skipped (not counted), like the loop."""
    rng = np.random.default_rng(9)
    n_rx = 20
    paths_dict = _make_single_ant_batch(rng, n_rx=n_rx, n_tx=1, max_paths=8, max_depth=3)
    targets = rng.uniform(-200, 200, (n_rx, 3))
    # Only half the targets exist in the global grid; the rest must be skipped.
    rx_pos = targets[::2].copy()
    opt_data, opt_inactive = _run(
        sionna_paths._process_paths_batch,  # noqa: SLF001
        paths_dict,
        targets=targets,
        rx_pos=rx_pos,
        use_index=True,
    )
    base_data, base_inactive = _run(
        _baseline_process_paths_batch, paths_dict, targets=targets, rx_pos=rx_pos, use_index=False
    )
    assert opt_inactive == base_inactive
    _assert_data_identical(opt_data, base_data)


def test_process_batch_zero_position_not_nulled() -> None:
    """Interaction positions of exactly 0.0 with a non-zero type are preserved.

    NONE(0) depth slots must be NaN; a real bounce that happens to sit at the
    origin (coordinate 0.0) must NOT be nulled. This pins the type-based (not
    value-based) masking used by both implementations.
    """
    n_rx, n_tx, max_paths, max_depth = 1, 1, 2, 3
    a = np.zeros((n_rx, 1, n_tx, 1, max_paths), dtype=np.complex128)
    a[0, 0, 0, 0, 0] = 2.0 + 1.0j  # one active path
    tau = np.full((n_rx, n_tx, max_paths), np.nan)
    tau[0, 0, 0] = 5e-7
    angles = {k: np.full((n_rx, n_tx, max_paths), np.nan) for k in range(4)}
    for arr in angles.values():
        arr[0, 0, 0] = 0.5
    interactions = np.zeros((max_depth, n_rx, n_tx, max_paths))
    # Path 0: reflection at depth 0 (position exactly origin), NONE at depth 1,2.
    interactions[0, 0, 0, 0] = sionna_paths.SIONNA_INTERACTION_SPECULAR
    vertices = np.full((max_depth, n_rx, n_tx, max_paths, 3), np.nan)
    vertices[0, 0, 0, 0] = [0.0, 0.0, 0.0]  # real bounce sitting at the origin

    paths_dict = {
        "a": a,
        "tau": tau,
        "phi_r": angles[0],
        "phi_t": angles[1],
        "theta_r": angles[2],
        "theta_t": angles[3],
        "interactions": interactions,
        "vertices": vertices,
    }
    targets = np.array([[1.0, 2.0, 3.0]])
    rx_pos = targets.copy()
    opt_data, _ = _run(
        sionna_paths._process_paths_batch,  # noqa: SLF001
        paths_dict,
        targets=targets,
        rx_pos=rx_pos,
        use_index=True,
    )
    base_data, _ = _run(
        _baseline_process_paths_batch, paths_dict, targets=targets, rx_pos=rx_pos, use_index=False
    )
    _assert_data_identical(opt_data, base_data)
    # The origin bounce is retained (not NaN); depth>=1 slots are NaN padding.
    assert np.array_equal(opt_data[c.INTERACTIONS_POS_PARAM_NAME][0, 0, 0], [0.0, 0.0, 0.0])
    assert np.all(np.isnan(opt_data[c.INTERACTIONS_POS_PARAM_NAME][0, 0, 1]))


# --------------------------------------------------------------------------- #
# Tests: _process_paths_batch (multi antenna)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("n_rx_ant", "n_tx_ant"), [(2, 1), (1, 2), (2, 2)])
def test_process_batch_multi_antenna_byte_identical(n_rx_ant: int, n_tx_ant: int) -> None:
    """Multi-antenna slicing + conversion is byte-identical across all elements."""
    rng = np.random.default_rng(20 + n_rx_ant * 10 + n_tx_ant)
    n_rx, n_tx, max_paths, max_depth = 18, 1, 8, 4
    paths_dict = _make_multi_ant_batch(
        rng,
        n_rx=n_rx,
        n_rx_ant=n_rx_ant,
        n_tx=n_tx,
        n_tx_ant=n_tx_ant,
        max_paths=max_paths,
        max_depth=max_depth,
    )
    targets = rng.uniform(-200, 200, (n_rx, 3))
    rx_pos = targets.copy()
    for ra in range(n_rx_ant):
        for ta in range(n_tx_ant):
            opt_data, opt_inactive = _run(
                sionna_paths._process_paths_batch,  # noqa: SLF001
                paths_dict,
                targets=targets,
                rx_pos=rx_pos,
                use_index=True,
                tx_ant_idx=ta,
                rx_ant_idx=ra,
            )
            base_data, base_inactive = _run(
                _baseline_process_paths_batch,
                paths_dict,
                targets=targets,
                rx_pos=rx_pos,
                use_index=False,
                tx_ant_idx=ta,
                rx_ant_idx=ra,
            )
            assert opt_inactive == base_inactive
            _assert_data_identical(opt_data, base_data)


# --------------------------------------------------------------------------- #
# Optional end-to-end check against a real Sionna 2.0 export under /tmp.
# --------------------------------------------------------------------------- #

_REAL_EXPORT = Path("/tmp/dm_bench/city_ny_2p0/sionna_paths.pkl")  # noqa: S108


@pytest.mark.skipif(not _REAL_EXPORT.exists(), reason="real Sionna 2.0 export absent")
def test_real_export_first_batch_byte_identical() -> None:
    """Optimized vs baseline ``_process_paths_batch`` on a real export's first batch."""
    path_dict_list = load_pickle(str(_REAL_EXPORT))
    paths_dict = path_dict_list[0]
    targets = paths_dict["targets"]
    rx_pos = targets.copy()
    opt_data, opt_inactive = _run(
        sionna_paths._process_paths_batch,  # noqa: SLF001
        paths_dict,
        targets=targets,
        rx_pos=rx_pos,
        use_index=True,
    )
    base_data, base_inactive = _run(
        _baseline_process_paths_batch, paths_dict, targets=targets, rx_pos=rx_pos, use_index=False
    )
    assert opt_inactive == base_inactive
    _assert_data_identical(opt_data, base_data)
