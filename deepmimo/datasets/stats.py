"""Statistics helpers and public entrypoint for DeepMIMO datasets."""

from typing import Any

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from deepmimo import consts as c

__all__ = ["stats"]

_MIN_POLY_POINTS = 3
_MIN_HULL_POINTS_3D = 4
_MIN_RANK_FOR_VOLUME = 3
_STATS_REQUIRED_MATRICES = (
    c.AOA_AZ_PARAM_NAME,
    c.DELAY_PARAM_NAME,
    c.INTERACTIONS_PARAM_NAME,
    c.PHASE_PARAM_NAME,
    c.POWER_PARAM_NAME,
    c.RX_POS_PARAM_NAME,
    c.TX_POS_PARAM_NAME,
)
_POWER_STATS_KEYS = {
    "avg": "avg_pathloss",
    "min": "min_pathloss",
    "max": "max_pathloss",
    "median": "median_pathloss",
    "iqr": "pathloss_iqr",
    "p10": "pathloss_p10",
    "p90": "pathloss_p90",
}
_SPATIAL_STATS_KEYS = {
    "avg": "avg_distance_bs",
    "min": "min_distance_bs",
    "max": "max_distance_bs",
    "median": "median_distance_bs",
    "iqr": "distance_iqr",
    "p10": "distance_p10",
    "p90": "distance_p90",
}
_SECTION_SPECS = (
    (
        "path",
        "Path Statistics",
        (
            ("raw", "Average paths per user", "{avg_paths_per_user:.1f}"),
            ("raw", "Max paths per user", "{max_paths_per_user}"),
            ("raw", "Min paths per user", "{min_paths_per_user}"),
            ("raw", "LOS percentage", "{los_percentage:.1f}%"),
            ("raw", "NLOS percentage", "{nlos_percentage:.1f}%"),
            ("raw", "No paths percentage", "{no_paths_percentage:.1f}%"),
            ("raw", "Average interactions per path", "{avg_interactions_per_path:.1f}"),
            ("raw", "Maximum interactions", "{max_interactions}"),
        ),
    ),
    (
        "power",
        "Power Statistics",
        (
            ("optional", "Average pathloss", "avg_pathloss", " dB", 1),
            ("optional", "Min pathloss", "min_pathloss", " dB", 1),
            ("optional", "Max pathloss", "max_pathloss", " dB", 1),
            ("optional", "Median pathloss", "median_pathloss", " dB", 1),
            ("optional", "Pathloss IQR (p75-p25)", "pathloss_iqr", " dB", 1),
            ("pair", "Pathloss p10/p90", "pathloss_p10", "pathloss_p90", " dB", 1),
        ),
    ),
    (
        "delay",
        "Delay Statistics",
        (
            ("optional", "Min delay", "min_delay_ns", " ns", 1),
            ("optional", "Max delay", "max_delay_ns", " ns", 1),
            ("optional", "Average delay", "avg_delay_ns", " ns", 1),
            ("optional", "Average RMS delay spread", "avg_rms_delay_ns", " ns", 1),
            ("optional", "Maximum RMS delay spread", "max_rms_delay_ns", " ns", 1),
        ),
    ),
    (
        "coverage",
        "Coverage Statistics",
        (
            ("raw", "Coverage percentage", "{coverage_percentage:.1f}%"),
            ("raw", "LOS coverage percentage", "{los_coverage_percentage:.1f}%"),
            ("raw", "Average paths per covered user", "{avg_paths_per_covered_user:.1f}"),
        ),
    ),
    (
        "spatial",
        "Spatial Statistics",
        (
            ("optional", "Min distance to BS", "min_distance_bs", " m", 1),
            ("optional", "Max distance to BS", "max_distance_bs", " m", 1),
            ("optional", "Average distance to BS", "avg_distance_bs", " m", 1),
            ("optional", "Median distance to BS", "median_distance_bs", " m", 1),
            ("optional", "Distance IQR (p75-p25)", "distance_iqr", " m", 1),
            ("pair", "Distance p10/p90", "distance_p10", "distance_p90", " m", 1),
        ),
    ),
    (
        "scene",
        "Scene Dimensions",
        (
            ("raw", "Width", "{width:.1f} m"),
            ("raw", "Length", "{length:.1f} m"),
            ("raw", "Height", "{height:.1f} m"),
            ("raw", "Total area", "{total_area:.1f} m²"),
            ("raw", "Total volume", "{total_volume:.1f} m³"),
        ),
    ),
)
_BUILDING_SPECS = (
    ("raw", "Average height", "{avg_height:.1f} m"),
    ("raw", "Height range", "{min_height:.1f} - {max_height:.1f} m"),
    ("raw", "Median height", "{median_height:.1f} m"),
    ("raw", "Height IQR (p75-p25)", "{height_iqr:.1f} m"),
    ("raw", "Height p10/p90", "{height_p10:.1f}/{height_p90:.1f} m"),
    ("raw", "Average volume", "{avg_volume:.1f} m³"),
    ("raw", "Total volume", "{total_volume:.1f} m³"),
    ("raw", "Average footprint", "{avg_footprint:.1f} m²"),
    ("raw", "Total footprint", "{total_footprint:.1f} m²"),
    ("raw", "Building density", "{building_density:.1f}%"),
)
_TERRAIN_SPECS = (
    ("raw", "Height range", "{min_height:.1f} - {max_height:.1f} m"),
    ("raw", "Average height", "{avg_height:.1f} m"),
    ("raw", "Height std dev", "{height_std:.1f} m"),
    ("raw", "Total elevation change", "{total_elevation_change:.1f} m"),
)


def _robust_stats(values: np.ndarray) -> tuple[float, float, float, float]:
    """Return median, IQR, p10, p90 for a 1D numeric array."""
    median = float(np.median(values))
    p25 = float(np.percentile(values, 25))
    p75 = float(np.percentile(values, 75))
    p10 = float(np.percentile(values, 10))
    p90 = float(np.percentile(values, 90))
    return median, p75 - p25, p10, p90


def _as_1d(values: Any, *, dtype: Any = float) -> np.ndarray:
    """Convert input values to a flattened 1D NumPy array."""
    return np.asarray(values, dtype=dtype).reshape(-1)


def _valid_values(values: Any, *, dtype: Any = float) -> np.ndarray:
    """Return non-NaN values as a flat numeric array."""
    array = _as_1d(values, dtype=dtype)
    return array[~np.isnan(array)]


def _empty_descriptive_stats(keys: dict[str, str]) -> dict[str, float | None]:
    """Return an empty descriptive-stats dictionary using the provided output keys."""
    return dict.fromkeys(keys.values(), None)


def _compute_descriptive_stats(
    values: Any,
    *,
    keys: dict[str, str],
    scale: float = 1.0,
) -> dict[str, float | None]:
    """Compute avg/min/max/median/IQR/p10/p90 for a numeric array."""
    valid_values = _valid_values(values)
    stats = _empty_descriptive_stats(keys)
    if valid_values.size == 0:
        return stats

    median, iqr, p10, p90 = _robust_stats(valid_values)
    summary = {
        "avg": float(np.mean(valid_values) * scale),
        "min": float(np.min(valid_values) * scale),
        "max": float(np.max(valid_values) * scale),
        "median": median * scale,
        "iqr": iqr * scale,
        "p10": p10 * scale,
        "p90": p90 * scale,
    }
    return {output_key: summary[input_key] for input_key, output_key in keys.items()}


def _empty_path_stats_and_context() -> tuple[dict[str, Any], dict[str, Any]]:
    """Return default path stats/context for empty inputs."""
    return (
        {
            "avg_paths_per_user": 0.0,
            "max_paths_per_user": 0,
            "min_paths_per_user": 0,
            "los_percentage": 0.0,
            "nlos_percentage": 0.0,
            "no_paths_percentage": 0.0,
            "avg_interactions_per_path": 0.0,
            "max_interactions": 0,
        },
        {
            "num_paths": np.array([], dtype=float),
            "valid_users": np.array([], dtype=bool),
            "los_users": 0,
            "total_users": 0,
        },
    )


def _compute_path_stats(dataset: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute path statistics and return context for dependent sections."""
    num_paths = _as_1d(dataset.num_paths)
    los_status = _as_1d(dataset.los, dtype=int)
    if num_paths.size == 0 or los_status.size == 0:
        return _empty_path_stats_and_context()
    if num_paths.size != los_status.size:
        msg = "num_paths and los arrays have inconsistent lengths"
        raise ValueError(msg)

    los_users = int(np.sum(los_status == 1))
    nlos_users = int(np.sum(los_status == 0))
    no_path_users = int(np.sum(los_status == -1))
    total_users = len(los_status)
    valid_users = num_paths > 0

    valid_interactions = _valid_values(dataset.num_interactions)
    avg_inter_per_path = float(np.mean(valid_interactions)) if valid_interactions.size else 0.0
    max_interactions = int(np.max(valid_interactions)) if valid_interactions.size else 0
    return (
        {
            "avg_paths_per_user": float(np.mean(num_paths)),
            "max_paths_per_user": int(np.max(num_paths)),
            "min_paths_per_user": int(np.min(num_paths)),
            "los_percentage": 100.0 * los_users / total_users,
            "nlos_percentage": 100.0 * nlos_users / total_users,
            "no_paths_percentage": 100.0 * no_path_users / total_users,
            "avg_interactions_per_path": avg_inter_per_path,
            "max_interactions": max_interactions,
        },
        {
            "num_paths": num_paths,
            "valid_users": valid_users,
            "los_users": los_users,
            "total_users": total_users,
        },
    )


def _compute_power_stats(dataset: Any) -> dict[str, float | None]:
    """Compute pathloss statistics."""
    return _compute_descriptive_stats(dataset.pathloss, keys=_POWER_STATS_KEYS)


def _compute_rms_delays(delays: np.ndarray, power_linear: np.ndarray, n_ue: int) -> list[float]:
    """Compute per-user RMS delay spread values."""
    delays_2d = np.asarray(delays, dtype=float).reshape(n_ue, -1)
    powers_2d = np.asarray(power_linear, dtype=float).reshape(n_ue, -1)
    valid_mask = ~np.isnan(delays_2d) & ~np.isnan(powers_2d)
    valid_count = np.sum(valid_mask, axis=1)

    valid_powers = np.where(valid_mask, powers_2d, 0.0)
    sum_powers = np.sum(valid_powers, axis=1)
    valid_users = (valid_count > 1) & (sum_powers > 0.0)
    if not np.any(valid_users):
        return []

    weighted_delays = np.where(valid_mask, delays_2d * powers_2d, 0.0)
    mean_delays = np.divide(
        np.sum(weighted_delays, axis=1),
        sum_powers,
        out=np.zeros_like(sum_powers),
        where=sum_powers > 0.0,
    )
    squared_diff = np.where(valid_mask, (delays_2d - mean_delays[:, None]) ** 2, 0.0)
    weighted_squared_diff = squared_diff * valid_powers
    rms_delays = np.sqrt(
        np.divide(
            np.sum(weighted_squared_diff, axis=1),
            sum_powers,
            out=np.zeros_like(sum_powers),
            where=sum_powers > 0.0,
        )
    )
    return rms_delays[valid_users].astype(float).tolist()


def _compute_delay_stats(dataset: Any) -> dict[str, float | None]:
    """Compute delay and RMS delay spread statistics."""
    valid_delays = _valid_values(dataset.delay)
    stats: dict[str, float | None] = {
        "min_delay_ns": None,
        "max_delay_ns": None,
        "avg_delay_ns": None,
        "avg_rms_delay_ns": None,
        "max_rms_delay_ns": None,
    }
    if valid_delays.size == 0:
        return stats

    rms_delays = _compute_rms_delays(dataset.delay, dataset.power_linear, int(dataset.n_ue))
    stats.update(
        {
            "min_delay_ns": float(np.min(valid_delays) * 1e9),
            "max_delay_ns": float(np.max(valid_delays) * 1e9),
            "avg_delay_ns": float(np.mean(valid_delays) * 1e9),
            "avg_rms_delay_ns": float(np.mean(rms_delays) * 1e9) if rms_delays else None,
            "max_rms_delay_ns": float(np.max(rms_delays) * 1e9) if rms_delays else None,
        }
    )
    return stats


def _compute_coverage_stats(
    num_paths: np.ndarray,
    valid_users: np.ndarray,
    los_users: int,
    total_users: int,
) -> dict[str, float]:
    """Compute coverage statistics."""
    if total_users == 0:
        return {
            "coverage_percentage": 0.0,
            "los_coverage_percentage": 0.0,
            "avg_paths_per_covered_user": 0.0,
        }
    covered_users = int(np.sum(valid_users))
    avg_paths_covered = float(np.mean(num_paths[valid_users])) if covered_users else 0.0
    return {
        "coverage_percentage": 100.0 * covered_users / total_users,
        "los_coverage_percentage": 100.0 * los_users / total_users,
        "avg_paths_per_covered_user": avg_paths_covered,
    }


def _compute_spatial_stats(dataset: Any) -> dict[str, float | None]:
    """Compute distance-to-base-station statistics."""
    return _compute_descriptive_stats(dataset.distance, keys=_SPATIAL_STATS_KEYS)


def _compute_channel_stats(stats_dataset: Any) -> dict[str, Any]:
    """Compute path/power/delay/coverage/spatial metrics for one TX/RX dataset."""
    path_stats, context = _compute_path_stats(stats_dataset)
    return {
        "path": path_stats,
        "power": _compute_power_stats(stats_dataset),
        "delay": _compute_delay_stats(stats_dataset),
        "coverage": _compute_coverage_stats(
            context["num_paths"],
            context["valid_users"],
            context["los_users"],
            context["total_users"],
        ),
        "spatial": _compute_spatial_stats(stats_dataset),
    }


def _footprint_polygon_and_area(obj: Any) -> tuple[np.ndarray, float]:
    """Return a 2D convex-hull footprint polygon and area for a scene object."""
    points_2d = np.unique(np.asarray(obj.vertices, dtype=float)[:, :2], axis=0)
    if points_2d.shape[0] < _MIN_POLY_POINTS:
        return np.empty((0, 2), dtype=float), 0.0
    try:
        hull_2d = ConvexHull(points_2d)
    except QhullError:
        return np.empty((0, 2), dtype=float), 0.0
    return points_2d[hull_2d.vertices], float(hull_2d.volume)


def _building_volume(obj: Any) -> float:
    """Return convex-hull volume for a scene object, handling degenerate geometry safely."""
    points_3d = np.unique(np.asarray(obj.vertices, dtype=float), axis=0)
    if points_3d.shape[0] < _MIN_HULL_POINTS_3D:
        return 0.0
    if np.linalg.matrix_rank(points_3d - points_3d[0]) < _MIN_RANK_FOR_VOLUME:
        return 0.0
    try:
        return float(ConvexHull(points_3d).volume)
    except QhullError:
        return 0.0


def _compute_scene_dimensions(dataset: Any) -> dict[str, float]:
    """Compute scene bounding box dimensions."""
    scene_bb = dataset.scene.bounding_box
    return {
        "width": float(scene_bb.width),
        "length": float(scene_bb.length),
        "height": float(scene_bb.height),
        "total_area": float(scene_bb.width * scene_bb.length),
        "total_volume": float(scene_bb.width * scene_bb.length * scene_bb.height),
    }


def _compute_object_distribution(scene_objects: list[Any]) -> dict[str, int]:
    """Compute counts of scene object labels."""
    object_counts: dict[str, int] = {}
    for obj in scene_objects:
        object_counts[obj.label] = object_counts.get(obj.label, 0) + 1
    return object_counts


def _compute_building_stats(
    buildings: list[Any],
    scene_total_area: float,
) -> dict[str, float] | None:
    """Compute building-specific characteristics."""
    if not buildings:
        return None

    building_heights = np.asarray([obj.height for obj in buildings], dtype=float)
    h_median, h_iqr, h_p10, h_p90 = _robust_stats(building_heights)
    building_volumes = np.asarray([_building_volume(obj) for obj in buildings], dtype=float)
    building_footprints = np.asarray(
        [_footprint_polygon_and_area(obj)[1] for obj in buildings],
        dtype=float,
    )
    total_footprint = float(np.sum(building_footprints))
    building_density = 100.0 * total_footprint / scene_total_area if scene_total_area > 0 else 0.0
    return {
        "avg_height": float(np.mean(building_heights)),
        "min_height": float(np.min(building_heights)),
        "max_height": float(np.max(building_heights)),
        "median_height": h_median,
        "height_iqr": h_iqr,
        "height_p10": h_p10,
        "height_p90": h_p90,
        "avg_volume": float(np.mean(building_volumes)),
        "total_volume": float(np.sum(building_volumes)),
        "avg_footprint": float(np.mean(building_footprints)),
        "total_footprint": total_footprint,
        "building_density": float(building_density),
    }


def _compute_terrain_stats(terrain_objects: list[Any]) -> dict[str, float] | None:
    """Compute terrain-specific characteristics."""
    if not terrain_objects:
        return None

    terrain_heights = np.asarray([obj.height for obj in terrain_objects], dtype=float)
    if terrain_heights.size == 0:
        terrain_heights = np.array([0.0], dtype=float)
    t_min = float(np.min(terrain_heights))
    t_max = float(np.max(terrain_heights))
    return {
        "min_height": t_min,
        "max_height": t_max,
        "avg_height": float(np.mean(terrain_heights)),
        "height_std": float(np.std(terrain_heights)),
        "total_elevation_change": t_max - t_min,
    }


def _compute_scene_stats(scene_dataset: Any) -> dict[str, Any]:
    """Compute scene/object-dependent statistics for one scene."""
    scene_objects = scene_dataset.scene.objects
    scene_stats = _compute_scene_dimensions(scene_dataset)
    return {
        "scene": scene_stats,
        "objects": _compute_object_distribution(scene_objects),
        "buildings": _compute_building_stats(
            [obj for obj in scene_objects if obj.label == "buildings"],
            scene_stats["total_area"],
        ),
        "terrain": _compute_terrain_stats([obj for obj in scene_objects if obj.label == "terrain"]),
    }


def _format_optional_value(value: float | None, *, unit: str = "", precision: int = 1) -> str:
    """Format an optional numeric value; return N/A when missing."""
    if value is None:
        return "N/A"
    return f"{value:.{precision}f}{unit}"


def _format_optional_pair(
    first: float | None,
    second: float | None,
    *,
    unit: str = "",
    precision: int = 1,
) -> str:
    """Format an optional pair of numeric values as '<first>/<second><unit>'."""
    if first is None or second is None:
        return "N/A"
    return f"{first:.{precision}f}/{second:.{precision}f}{unit}"


def _format_line(label: str, value: str) -> str:
    """Format one bullet line inside a stats section."""
    return f"- {label}: {value}\n"


def _format_optional_line(
    label: str,
    value: float | None,
    *,
    unit: str = "",
    precision: int = 1,
) -> str:
    """Format one optional numeric bullet line."""
    return _format_line(label, _format_optional_value(value, unit=unit, precision=precision))


def _format_pair_line(
    label: str,
    first: float | None,
    second: float | None,
    *,
    unit: str = "",
    precision: int = 1,
) -> str:
    """Format one optional pair bullet line."""
    return _format_line(label, _format_optional_pair(first, second, unit=unit, precision=precision))


def _format_section(title: str, lines: list[str]) -> str:
    """Wrap a list of formatted lines with a section title."""
    return f"\n{title}:\n" + "".join(lines)


def _format_section_lines(
    stats_dict: dict[str, Any],
    line_specs: tuple[tuple[Any, ...], ...],
) -> list[str]:
    """Format one section from raw/optional/pair line specifications."""
    lines: list[str] = []
    for spec in line_specs:
        kind = spec[0]
        if kind == "raw":
            (_, label, template) = spec
            lines.append(_format_line(label, template.format(**stats_dict)))
        elif kind == "optional":
            (_, label, key, unit, precision) = spec
            lines.append(
                _format_optional_line(label, stats_dict[key], unit=unit, precision=precision)
            )
        else:
            (_, label, first_key, second_key, unit, precision) = spec
            lines.append(
                _format_pair_line(
                    label,
                    stats_dict[first_key],
                    stats_dict[second_key],
                    unit=unit,
                    precision=precision,
                )
            )
    return lines


def _format_object_distribution(object_counts: dict[str, int]) -> str:
    """Format object distribution block."""
    lines = [_format_line(label.capitalize(), str(count)) for label, count in object_counts.items()]
    return _format_section("Object Distribution", lines)


def _format_stats(stats_dict: dict[str, Any]) -> str:
    """Format computed statistics into a printable summary string."""
    sections = [
        _format_section(title, _format_section_lines(stats_dict[key], line_specs))
        for (key, title, line_specs) in _SECTION_SPECS
    ]
    sections.append(_format_object_distribution(stats_dict["objects"]))
    if stats_dict["buildings"] is not None:
        sections.append(
            _format_section(
                "Building Characteristics",
                _format_section_lines(stats_dict["buildings"], _BUILDING_SPECS),
            )
        )
    if stats_dict["terrain"] is not None:
        sections.append(
            _format_section(
                "Terrain Characteristics",
                _format_section_lines(stats_dict["terrain"], _TERRAIN_SPECS),
            )
        )
    return "".join(sections)


def _stats_pair_header(txrx: dict[str, Any]) -> str:
    """Return section header for one TX/RX pair."""
    tx_set = int(txrx.get("tx_set_id", -1))
    tx_idx = int(txrx.get("tx_idx", -1))
    rx_set = int(txrx.get("rx_set_id", -1))
    return f"\n[TXset {tx_set} (tx_idx {tx_idx}) | RXset {rx_set}]\n"


def _resolve_stats_datasets(
    dataset: Any,
    scen_name: str,
) -> tuple[list[Any], list[str]]:
    """Resolve loaded data to the list of TX/RX datasets used for statistics."""
    from .dataset import DynamicDataset, MacroDataset  # noqa: PLC0415

    messages: list[str] = []
    if isinstance(dataset, DynamicDataset):
        if len(dataset) == 0:
            msg = f"Dynamic scenario '{scen_name}' contains no snapshots"
            raise ValueError(msg)
        messages.append("Dynamic scenario detected. Using snapshot 1 for statistics.")
        dataset = dataset[0]

    datasets = dataset.datasets if isinstance(dataset, MacroDataset) else [dataset]
    if len(datasets) == 0:
        msg = f"Scenario '{scen_name}' contains no TX/RX dataset pairs"
        raise ValueError(msg)

    if isinstance(dataset, MacroDataset):
        messages.append(f"Computing separate statistics for {len(datasets)} TX/RX pair(s).")
    return list(datasets), messages


def _selector_matches_id(
    selector: dict[int, list[int] | str] | list[int] | str | None,
    set_id: int | None,
) -> bool:
    """Return True when a TX/RX set selector allows the given set ID."""
    if selector is None or isinstance(selector, str):
        return True
    if set_id is None:
        return False
    if isinstance(selector, dict):
        return set_id in selector
    return set_id in selector


def _filter_stats_datasets(
    stats_datasets: list[Any],
    *,
    scen_name: str,
    tx_sets: dict[int, list[int] | str] | list[int] | str | None,
    rx_sets: dict[int, list[int] | str] | list[int] | str | None,
) -> list[Any]:
    """Filter resolved datasets against explicit selectors and raise on empty matches."""
    if tx_sets is None and (rx_sets is None or isinstance(rx_sets, str)):
        return stats_datasets

    filtered_datasets = []
    for stats_dataset in stats_datasets:
        txrx = stats_dataset.get("txrx", {})
        tx_set_id = txrx.get("tx_set_id")
        rx_set_id = txrx.get("rx_set_id")
        if _selector_matches_id(tx_sets, tx_set_id) and _selector_matches_id(rx_sets, rx_set_id):
            filtered_datasets.append(stats_dataset)

    if not filtered_datasets:
        msg = (
            f"Scenario '{scen_name}' has no datasets matching "
            f"tx_sets={tx_sets}, rx_sets={rx_sets}"
        )
        raise ValueError(msg)
    return filtered_datasets


def _compute_stats(
    stats_dataset: Any,
    *,
    cached_scene_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute all statistics for one TX/RX dataset."""
    combined_stats = _compute_channel_stats(stats_dataset)
    if cached_scene_stats is None:
        cached_scene_stats = _compute_scene_stats(stats_dataset)
    combined_stats.update(cached_scene_stats)
    return combined_stats


def _collect_stats(stats_datasets: list[Any]) -> list[str]:
    """Compute and format statistics for each selected TX/RX pair."""
    scene_stats_cache: dict[int, dict[str, Any]] = {}
    section_strs: list[str] = []
    for stats_dataset in stats_datasets:
        header = _stats_pair_header(stats_dataset.get("txrx", {}))
        scene_cache_key = id(stats_dataset.scene)
        if scene_cache_key not in scene_stats_cache:
            scene_stats_cache[scene_cache_key] = _compute_scene_stats(stats_dataset)
        computed_stats = _compute_stats(
            stats_dataset,
            cached_scene_stats=scene_stats_cache[scene_cache_key],
        )
        section_strs.append(header + _format_stats(computed_stats))
    return section_strs


def stats(
    scen_name: str,
    *,
    tx_sets: dict[int, list[int] | str] | list[int] | str | None = None,
    rx_sets: dict[int, list[int] | str] | list[int] | str | None = None,
    print_summary: bool = True,
) -> str | None:
    """Calculate and return scenario statistics for selected TX/RX pairs."""
    from .load import load  # noqa: PLC0415

    if isinstance(tx_sets, (list, dict)) and len(tx_sets) == 0:
        msg = "tx_sets must be non-empty when provided, e.g. tx_sets=[4]"
        raise ValueError(msg)
    if isinstance(rx_sets, (list, dict)) and len(rx_sets) == 0:
        msg = "rx_sets must be non-empty when provided"
        raise ValueError(msg)

    if print_summary:
        print("Calculating scenario statistics...")

    load_kwargs: dict[str, Any] = {"matrices": list(_STATS_REQUIRED_MATRICES)}
    if tx_sets is not None:
        load_kwargs["tx_sets"] = tx_sets
    if rx_sets is not None:
        load_kwargs["rx_sets"] = rx_sets
    try:
        dataset = load(scen_name, **load_kwargs)
    except ValueError as err:
        msg = (
            f"Failed to load scenario '{scen_name}' for "
            f"(tx_sets={tx_sets}, rx_sets={rx_sets}). {err!s}"
        )
        raise ValueError(msg) from err

    stats_datasets, messages = _resolve_stats_datasets(dataset, scen_name)
    stats_datasets = _filter_stats_datasets(
        stats_datasets,
        scen_name=scen_name,
        tx_sets=tx_sets,
        rx_sets=rx_sets,
    )
    for message in messages:
        if print_summary:
            print(message)

    summary_str = "\n".join(_collect_stats(stats_datasets))

    if print_summary:
        print(summary_str)
        return None
    return summary_str
