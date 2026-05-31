"""Equivalence tests for the vectorized interaction/doppler computations.

`Dataset._compute_inter_angles`, `Dataset._compute_inter_objects` and
`Dataset._compute_doppler` were rewritten from triple-nested Python loops into
vectorized NumPy. These tests pin the vectorized output against verbatim copies
of the original loop implementations (kept below as the baseline) on synthetic
data that exercises every edge case:

* NaN / zero interaction counts (LoS and padded paths),
* terrain z-snapping (atol=0.001) vs. nearest-object-center assignment,
* per-path interaction counts up to ``max_inter``,
* paths beyond ``num_paths`` that still carry interaction codes,
* the object-velocity gather (including a terrain object with non-zero velocity).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from deepmimo import consts as c
from deepmimo.datasets.dataset import Dataset, _nearest_center_idx
from deepmimo.utils import spherical_to_cartesian

# ---------------------------------------------------------------------------
# Lightweight scene stand-ins (object_id == position in `objects`, matching the
# real Scene contract that both the loop and vectorized code rely on).
# ---------------------------------------------------------------------------


class _BBox:
    def __init__(self, center, z_max) -> None:
        self.center = np.asarray(center, dtype=float)
        self.z_max = float(z_max)


class _Obj:
    def __init__(self, object_id, label, center, z_max, vel) -> None:
        self.object_id = object_id
        self.label = label
        self.bounding_box = _BBox(center, z_max)
        self.vel = np.asarray(vel, dtype=float)


class _Scene:
    def __init__(self, objects) -> None:
        self.objects = objects


def _make_scene(rng: np.random.Generator, n_objects: int, terrain_z: float) -> _Scene:
    """Terrain object at index 0; ``n_objects`` non-terrain objects after it."""
    objects = [
        _Obj(
            object_id=0,
            label="terrain",
            center=[rng.uniform(-50, 50), rng.uniform(-50, 50), terrain_z / 2],
            z_max=terrain_z,
            vel=rng.uniform(-3, 3, 3),  # non-zero terrain velocity exercises the gather
        )
    ]
    for oid in range(1, n_objects + 1):
        center = rng.uniform(-100, 100, 3)
        objects.append(
            _Obj(oid, "buildings", center, center[2] + rng.uniform(1, 20), rng.uniform(-10, 10, 3))
        )
    return _Scene(objects)


def _assign_path(rng, *, within: bool, n_depth: int, terrain_z: float, full_depth: bool):
    """Return ``(inter_code, positions | None)`` for one path, mirroring real data."""
    roll = rng.random()
    if within and roll < 0.15:
        return np.nan, None  # valid path but NaN code -> doppler skips it
    if within and roll < 0.30:
        return 0.0, None  # LoS: zero interactions
    if (not within) and roll < 0.5:
        return np.nan, None  # padded paths are usually empty

    k = n_depth if full_depth else int(rng.integers(1, n_depth + 1))
    digits = rng.integers(1, 5, k)  # interaction types 1..4 (no leading zeros)
    pts = rng.uniform(-100, 100, (k, 3))
    snap = rng.random(k) < 0.4  # snap some points onto terrain z (within atol=0.001)
    pts[snap, 2] = terrain_z + rng.uniform(-0.0008, 0.0008, int(snap.sum()))
    return float(int("".join(str(d) for d in digits))), pts


def _make_dataset(  # noqa: PLR0913
    *,
    seed: int,
    n_ue: int,
    n_paths: int,
    n_depth: int,
    n_objects: int,
    terrain_z: float = 0.0,
    force_full: bool = True,
) -> Dataset:
    """Build a synthetic Dataset exercising the interaction/doppler edge cases."""
    rng = np.random.default_rng(seed)
    scene = _make_scene(rng, n_objects, terrain_z)

    aoa_az = np.full((n_ue, n_paths), np.nan)
    aoa_el = np.full((n_ue, n_paths), np.nan)
    aod_az = np.full((n_ue, n_paths), np.nan)
    aod_el = np.full((n_ue, n_paths), np.nan)
    inter = np.full((n_ue, n_paths), np.nan)
    inter_pos = np.full((n_ue, n_paths, n_depth, 3), np.nan)

    for u in range(n_ue):
        num_paths = n_paths if (force_full and u == 0) else int(rng.integers(0, n_paths + 1))
        for p in range(n_paths):
            within = p < num_paths
            if within:
                aoa_az[u, p] = rng.uniform(-180, 180)
                aoa_el[u, p] = rng.uniform(1, 179)
                aod_az[u, p] = rng.uniform(-180, 180)
                aod_el[u, p] = rng.uniform(1, 179)
            code, pts = _assign_path(
                rng,
                within=within,
                n_depth=n_depth,
                terrain_z=terrain_z,
                full_depth=(force_full and u == 0 and p == 0),
            )
            inter[u, p] = code
            if pts is not None:
                inter_pos[u, p, : pts.shape[0]] = pts

    ds = Dataset(
        {
            "n_ue": n_ue,
            "rx_pos": rng.uniform(-100, 100, (n_ue, 3)),
            "tx_pos": np.array([rng.uniform(-5, 5), rng.uniform(-5, 5), 12.0]),
            "aoa_az": aoa_az,
            "aoa_el": aoa_el,
            "aod_az": aod_az,
            "aod_el": aod_el,
            "inter": inter,
            "inter_pos": inter_pos,
        }
    )
    ds.scene = scene
    ds.rt_params = SimpleNamespace(frequency=28e9)
    ds.tx_vel = rng.uniform(-20, 20, 3)
    ds.rx_vel = rng.uniform(-20, 20, (n_ue, 3))
    return ds


# ---------------------------------------------------------------------------
# Baseline: verbatim copies of the original triple-nested loop implementations.
# ---------------------------------------------------------------------------


def _baseline_inter_angles(d: Dataset) -> np.ndarray:
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


def _baseline_inter_objects(d: Dataset) -> np.ndarray:
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


def _baseline_doppler(d: Dataset) -> np.ndarray:
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
    k_i = _baseline_inter_angles(d)
    inter_objects = _baseline_inter_objects(d)
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# (seed, n_ue, n_paths, n_depth, n_objects, force_full)
_CASES = [
    (0, 40, 6, 4, 8, True),  # common case: max_paths == n_paths, max_inter == n_depth
    (1, 24, 5, 3, 5, False),  # ragged: max_paths < n_paths and/or max_inter < n_depth
    (2, 12, 4, 4, 1, True),  # single non-terrain object (degenerate nearest search)
    (3, 30, 7, 5, 12, False),  # larger depth / object count
]


@pytest.mark.parametrize("case", _CASES)
def test_inter_angles_matches_loop(case) -> None:
    """Vectorized interaction angles must match the original loop."""
    seed, n_ue, n_paths, n_depth, n_objects, full = case
    ds = _make_dataset(
        seed=seed,
        n_ue=n_ue,
        n_paths=n_paths,
        n_depth=n_depth,
        n_objects=n_objects,
        force_full=full,
    )
    vec = ds._compute_inter_angles()  # noqa: SLF001
    base = _baseline_inter_angles(ds)
    assert vec.shape == base.shape
    np.testing.assert_array_equal(np.isnan(vec), np.isnan(base))  # identical NaN placement
    np.testing.assert_allclose(vec, base, rtol=1e-6, atol=1e-9, equal_nan=True)


@pytest.mark.parametrize("case", _CASES)
def test_inter_objects_matches_loop(case) -> None:
    """Vectorized object-id assignment must match the original loop exactly."""
    seed, n_ue, n_paths, n_depth, n_objects, full = case
    ds = _make_dataset(
        seed=seed,
        n_ue=n_ue,
        n_paths=n_paths,
        n_depth=n_depth,
        n_objects=n_objects,
        force_full=full,
    )
    vec = ds._compute_inter_objects()  # noqa: SLF001
    base = _baseline_inter_objects(ds)
    assert vec.shape == base.shape
    np.testing.assert_array_equal(np.isnan(vec), np.isnan(base))
    np.testing.assert_array_equal(vec, base)  # exact (integer object ids / NaN padding)


@pytest.mark.parametrize("case", _CASES)
def test_doppler_matches_loop(case) -> None:
    """Vectorized doppler must match the original loop within tolerance."""
    seed, n_ue, n_paths, n_depth, n_objects, full = case
    ds = _make_dataset(
        seed=seed,
        n_ue=n_ue,
        n_paths=n_paths,
        n_depth=n_depth,
        n_objects=n_objects,
        force_full=full,
    )
    vec = ds._compute_doppler()  # noqa: SLF001
    base = _baseline_doppler(ds)
    assert vec.shape == base.shape
    np.testing.assert_array_equal(np.isnan(vec), np.isnan(base))
    np.testing.assert_allclose(vec, base, rtol=1e-6, atol=1e-9, equal_nan=True)


def test_chunked_nearest_matches_unchunked() -> None:
    """The chunked nearest-center search must equal a brute-force argmin."""
    rng = np.random.default_rng(7)
    points = rng.uniform(-100, 100, (500, 3))
    centers = rng.uniform(-100, 100, (37, 3))
    brute = np.array([np.argmin(np.linalg.norm(centers - p, axis=1)) for p in points])
    chunked = _nearest_center_idx(points, centers, max_bytes=4096)  # force many chunks
    np.testing.assert_array_equal(chunked, brute)


def test_doppler_sets_enabled_flag() -> None:
    """``_compute_doppler`` must (re)enable the ``doppler_enabled`` flag, as before."""
    ds = _make_dataset(seed=5, n_ue=6, n_paths=4, n_depth=3, n_objects=4)
    ds.doppler_enabled = False
    result = ds._compute_doppler()  # noqa: SLF001
    assert ds.doppler_enabled is True
    assert result.shape == (ds.n_ue, ds.max_paths)
