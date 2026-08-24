"""Interactive 3D dashboard for inspecting DeepMIMO scenarios.

Opens any scenario in the scenarios folder, whatever produced it - Wireless
InSite, Sionna RT or Infinigen - and lets you fly around its geometry the way
you would in Blender: orbit, pan, dolly, framed views, and a section plane for
looking inside a closed interior.

Geometry is packed into one float32 buffer with a manifest of per-material
ranges and drawn by a small hand-written WebGL renderer, so a 50k-triangle
interior loads as a couple of megabytes and needs no browser toolchain, no CDN
and no network beyond the forwarded port. Coverage maps are still rendered
server-side, where matplotlib is already available.

Usage:
    uv run scripts/scenario_dashboard/server.py [--port 8000] [--scenarios DIR]
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402

import deepmimo as dm  # noqa: E402

from jobs import JobManager, list_scenes  # noqa: E402
# Height of the default horizontal section, roughly a standing receiver.
DEFAULT_CUT_HEIGHT = 1.3

# Last packed scenario, so orbiting does not repack the geometry each frame.
_GEOMETRY_CACHE: dict[str, Any] = {}

# Room types Infinigen can be restricted to.
#: What each building preset does, shown under the picker. Infinigen ships one
#: constraint script and it describes a dwelling, so these vary the building's
#: shape rather than its room vocabulary - see scripts/pipelines/INFINIGEN.md.
SCENE_TYPE_HELP = {
    "home": "Infinigen's dwelling — about 3 m storeys",
    "tall_space": "5.5 m storeys: hall-like volumes and longer reverberation",
    "compact": "near-square footprint",
    "elongated": "long and narrow — corridor-dominated propagation",
}

ROOM_TYPES = (
    "", "LivingRoom", "Kitchen", "Bedroom", "Bathroom", "DiningRoom",
    "Utility", "Garage", "Balcony", "Closet", "Hallway", "Staircase",
)

# Colour per ITU material, so a section is readable at a glance.
MATERIAL_COLOURS = {
    "concrete": "#8d949c",
    "brick": "#a9553f",
    "plasterboard": "#c9b78f",
    "wood": "#b5793a",
    "glass": "#3fa9c9",
    "metal": "#5f7183",
    "ceiling_board": "#d7d7d7",
    "floorboard": "#c99b6b",
}


def _png(fig: plt.Figure) -> bytes:
    """Render a figure to PNG bytes and close it.

    Args:
        fig: Figure to render.

    Returns:
        PNG image bytes.

    """
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def _section_segments(faces: list[np.ndarray], height: float) -> list[np.ndarray]:
    """Intersect triangles with a horizontal plane.

    Args:
        faces: Face vertex arrays.
        height: Plane height.

    Returns:
        List of (2, 2) segment endpoint arrays in the xy plane.

    """
    segments = []
    for vertices in faces:
        if vertices[:, 2].min() > height or vertices[:, 2].max() < height:
            continue
        crossings = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            if a >= len(vertices) or b >= len(vertices):
                continue
            za, zb = vertices[a, 2], vertices[b, 2]
            if za == zb or (za - height) * (zb - height) > 0:
                continue
            t = (height - za) / (zb - za)
            crossings.append(vertices[a, :2] + t * (vertices[b, :2] - vertices[a, :2]))
        if len(crossings) >= 2:
            segments.append(np.array(crossings[:2]))
    return segments


def _load(name: str) -> list:
    """Load a scenario as a list of TX/RX datasets.

    Args:
        name: Scenario name.

    Returns:
        List of datasets, one per TX/RX pair.

    """
    data = dm.load(name)
    return list(data) if isinstance(data, dm.MacroDataset) else [data]


#: Per-receiver maps the coverage panel can draw. ``better`` says which end of the
#: scale is the good one, which is what picks the best server for a receiver.
COVERAGE_METRICS: dict[str, dict[str, Any]] = {
    "power": {"tab": "power", "title": "received power",
              "unit": "RX power [dBm]", "better": "high"},
    "pathloss": {"tab": "loss", "title": "path loss",
                 "unit": "pathloss [dB]", "better": "low"},
    "los": {"tab": "LOS", "title": "line of sight", "unit": "", "better": "high"},
    "delay": {"tab": "delay", "title": "delay spread",
              "unit": "RMS delay spread [ns]", "better": "low"},
    "paths": {"tab": "paths", "title": "path count",
              "unit": "paths per receiver", "better": "high"},
}

#: Line-of-sight states, in plot order. The two hues are the categorical slots
#: validated against this panel's surface; "no path" is deliberately not a hue,
#: because it is missing data rather than a third category.
LOS_STATES = ((1, "line of sight", "#3987e5"), (0, "obstructed", "#d95926"))
LOS_NONE = "#3a4149"


def _receiver_metric(dataset: Any, metric: str) -> np.ndarray:
    """Compute one per-receiver quantity from a dataset.

    Args:
        dataset: A single TX-RX pair dataset.
        metric: Key from :data:`COVERAGE_METRICS`.

    Returns:
        One value per receiver, NaN where the receiver is unserved.

    """
    if metric == "pathloss":
        return np.asarray(dataset.pathloss, dtype=float)

    # Per-path powers are in dBm, so they have to be summed in the linear domain.
    power_db = np.asarray(dataset.power, dtype=float)
    linear = np.where(np.isfinite(power_db), 10 ** (power_db / 10.0), 0.0)
    total = linear.sum(axis=1)
    served = total > 0

    if metric == "power":
        return np.where(served, 10 * np.log10(np.where(served, total, 1.0)), np.nan)
    if metric == "paths":
        return np.where(served, np.asarray(dataset.num_paths, dtype=float), np.nan)
    if metric == "los":
        los = np.asarray(dataset.los, dtype=float)
        return np.where(los < 0, np.nan, los)
    if metric == "delay":
        # Power-weighted RMS delay spread, the second central moment of the
        # power delay profile.
        toa = np.where(np.isfinite(dataset.toa), np.asarray(dataset.toa, dtype=float), 0.0)
        mean = np.divide(
            (linear * toa).sum(axis=1), total,
            out=np.full(total.shape, np.nan), where=served,
        )
        variance = np.divide(
            (linear * (toa - mean[:, None]) ** 2).sum(axis=1), total,
            out=np.full(total.shape, np.nan), where=served,
        )
        return np.sqrt(variance) * 1e9
    msg = f"unknown coverage metric {metric!r}"
    raise ValueError(msg)


def _best_server(pairs: list, metric: str) -> np.ndarray:
    """Pick, per receiver, which transmitter serves it best.

    The choice is always made on received power, not on the metric being drawn:
    the delay spread of whichever transmitter happens to have the shortest one
    is not a meaningful map.

    Args:
        pairs: One dataset per transmitter.
        metric: Metric being drawn, used only for its direction.

    Returns:
        Index of the serving transmitter per receiver, -1 where none serves.

    """
    powers = np.vstack([_receiver_metric(p, "power") for p in pairs])
    served = np.isfinite(powers).any(axis=0)
    filled = np.where(np.isfinite(powers), powers, -np.inf)
    return np.where(served, filled.argmax(axis=0), -1)


def _as_grid(positions: np.ndarray, values: np.ndarray) -> tuple | None:
    """Rasterise receiver values onto their own grid, if they lie on one.

    A regular receiver grid drawn as a field reads as coverage; drawn as dots it
    reads as dots. Scenarios whose receivers are not gridded fall back to a
    scatter.

    Args:
        positions: Receiver positions, (n, 3).
        values: One value per receiver.

    Returns:
        ``(image, extent)`` for imshow, or None if the receivers are not gridded.

    """
    xs, ys = np.unique(positions[:, 0].round(3)), np.unique(positions[:, 1].round(3))
    if len(xs) < 2 or len(ys) < 2 or len(xs) * len(ys) > 4 * len(values):
        return None
    image = np.full((len(ys), len(xs)), np.nan)
    col = np.searchsorted(xs, positions[:, 0].round(3))
    row = np.searchsorted(ys, positions[:, 1].round(3))
    image[row, col] = values
    dx, dy = (xs[1] - xs[0]) / 2, (ys[1] - ys[0]) / 2
    return image, (xs[0] - dx, xs[-1] + dx, ys[0] - dy, ys[-1] + dy)


def _style_axes(fig: Any, ax: Any) -> None:
    """Apply the panel's dark surface to a figure.

    Args:
        fig: Matplotlib figure.
        ax: Matplotlib axes.

    """
    fig.patch.set_facecolor("#171d24")
    ax.set_facecolor("#12161b")
    for spine in ax.spines.values():
        spine.set_color("#39424c")
    ax.tick_params(colors="#95a0ad", labelsize=8)
    ax.xaxis.label.set_color("#95a0ad")
    ax.yaxis.label.set_color("#95a0ad")
    ax.title.set_color("#dfe4ea")


def render_coverage(name: str, tx: int, metric: str = "pathloss") -> bytes:
    """Render one coverage map for a transmitter, or for the best server.

    Geometry is not drawn here: the 3D view handles it interactively, so this
    stays a map of the receiver grid.

    Args:
        name: Scenario name.
        tx: Transmitter index, or -1 for the best server across transmitters.
        metric: Key from :data:`COVERAGE_METRICS`.

    Returns:
        PNG image bytes.

    """
    if metric not in COVERAGE_METRICS:
        metric = "pathloss"
    spec = COVERAGE_METRICS[metric]
    pairs = _load(name)
    positions = np.asarray(pairs[0].rx_pos, dtype=float)

    if tx < 0 and len(pairs) > 1:
        per_tx = np.vstack([_receiver_metric(p, metric) for p in pairs])
        serving = _best_server(pairs, metric)
        values = np.where(
            serving >= 0, per_tx[np.clip(serving, 0, None), np.arange(per_tx.shape[1])], np.nan,
        )
        label = "best server"
    else:
        values = _receiver_metric(pairs[min(max(tx, 0), len(pairs) - 1)], metric)
        label = f"TX {max(tx, 0)}"

    served = np.isfinite(values)
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    gridded = _as_grid(positions, values)

    if metric == "los":
        colours = ListedColormap([hue for _, _, hue in reversed(LOS_STATES)])
        norm = BoundaryNorm([-0.5, 0.5, 1.5], colours.N)
        if gridded:
            image, extent = gridded
            masked = np.ma.masked_invalid(image)
            colours.set_bad(LOS_NONE)
            ax.imshow(masked, extent=extent, origin="lower", cmap=colours, norm=norm,
                      interpolation="nearest")
        else:
            ax.scatter(positions[~served, 0], positions[~served, 1], s=4, c=LOS_NONE)
            ax.scatter(positions[served, 0], positions[served, 1], c=values[served],
                       s=4, cmap=colours, norm=norm)
        # Identity is never colour alone: the states are named in the legend.
        handles = [Patch(facecolor=hue, label=text) for _, text, hue in LOS_STATES]
        handles.append(Patch(facecolor=LOS_NONE, label="no path"))
        legend = ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.85)
        legend.get_frame().set_facecolor("#171d24")
        legend.get_frame().set_edgecolor("#39424c")
        for text in legend.get_texts():
            text.set_color("#dfe4ea")
        summary = f"{100 * np.nanmean(values == 1):.0f}% line of sight"
    else:
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad(LOS_NONE)
        # A handful of deep-shadow receivers can span half the scale on their
        # own, flattening every room into one colour. The ramp covers the middle
        # 98% and the title still reports the true extremes.
        limits = (
            np.nanpercentile(values, [1, 99]).tolist() if served.any() else [0.0, 1.0]
        )
        if limits[0] == limits[1]:
            limits = [limits[0] - 0.5, limits[1] + 0.5]
        if gridded:
            image, extent = gridded
            drawn = ax.imshow(np.ma.masked_invalid(image), extent=extent, origin="lower",
                              cmap=cmap, interpolation="nearest",
                              vmin=limits[0], vmax=limits[1])
        else:
            ax.scatter(positions[~served, 0], positions[~served, 1], s=4, c=LOS_NONE)
            drawn = ax.scatter(positions[served, 0], positions[served, 1],
                               c=values[served], s=4, cmap=cmap,
                               vmin=limits[0], vmax=limits[1])
        bar = fig.colorbar(drawn, ax=ax, label=spec["unit"])
        bar.ax.yaxis.label.set_color("#95a0ad")
        bar.ax.tick_params(colors="#95a0ad", labelsize=8)
        bar.outline.set_edgecolor("#39424c")
        if served.any():
            lo, mid, hi = (np.nanpercentile(values, q) for q in (0, 50, 100))
            summary = f"median {mid:.1f} · range {lo:.1f} to {hi:.1f}"
        else:
            summary = "no served receivers"

    for pair in pairs:
        tx_pos = np.asarray(pair.tx_pos).reshape(-1, 3)
        ax.scatter(tx_pos[:, 0], tx_pos[:, 1], marker="*", s=150, c="#f5f2e8",
                   edgecolor="#12161b", linewidth=0.8, zorder=5)

    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(
        f"{spec['title']} · {label} · {100 * served.mean():.0f}% served\n{summary}",
        fontsize=9,
    )
    _style_axes(fig, ax)
    fig.tight_layout()
    return _png(fig)


def scenario_summary(name: str) -> dict[str, Any]:
    """Summarise a scenario for the info panel.

    Args:
        name: Scenario name.

    Returns:
        Dict of summary fields.

    """
    pairs = _load(name)
    dataset = pairs[0]
    scene = dataset.scene
    vertices = np.concatenate([f.vertices for o in scene.objects for f in o.faces])
    total = served = 0
    low, high = np.inf, -np.inf
    for pair in pairs:
        values = np.asarray(pair.pathloss).ravel()
        total += values.size
        finite = values[np.isfinite(values)]
        served += finite.size
        if finite.size:
            low, high = min(low, finite.min()), max(high, finite.max())
    return {
        "name": name,
        "representation": scene.representation,
        "objects": len(scene.objects),
        "faces": sum(len(o.faces) for o in scene.objects),
        "materials": len(dataset.materials),
        "tx": len(pairs),
        "rx": int(np.asarray(dataset.rx_pos).shape[0]),
        "served_pct": round(100 * served / max(total, 1), 1),
        "pathloss_min": None if not np.isfinite(low) else round(float(low), 1),
        "pathloss_max": None if not np.isfinite(high) else round(float(high), 1),
        "extent_m": [round(float(np.ptp(vertices[:, i])), 1) for i in range(3)],
    }


# Colour per ITU material. Naming a material from its constants alone does not
# work: ITU-R P.2040 puts concrete at eps=5.24 and glass at 6.27, far too close
# to separate reliably, so object names are consulted first.
ITU_COLOURS: dict[str, list[float]] = {
    "metal": [0.42, 0.49, 0.56],
    "glass": [0.25, 0.66, 0.79],
    "concrete": [0.55, 0.58, 0.61],
    "brick": [0.66, 0.35, 0.26],
    "plasterboard": [0.79, 0.72, 0.56],
    "ceiling_board": [0.84, 0.84, 0.84],
    "floorboard": [0.79, 0.61, 0.42],
    "wood": [0.71, 0.47, 0.23],
}
ITU_TOKENS = tuple(ITU_COLOURS)


def _material_style(
    material: Any,
    index: int,
    object_names: set[str],
) -> tuple[str, str, list[float]]:
    """Name and colour a material, preferring evidence from the objects using it.

    Exporters encode the material in the object name - ``window_glass``,
    ``floor_concrete`` - which survives conversion, so that is the most reliable
    signal. Where it is absent the electromagnetic constants are used instead,
    which separates metal cleanly by conductivity but cannot tell ITU concrete
    from ITU glass, whose permittivities are 5.24 and 6.27.

    Args:
        material: Material record from the scenario.
        index: Material index.
        object_names: Names of the objects that reference this material.

    Returns:
        Tuple of (display name, material class, RGB in 0-1).

    """
    for token in ITU_TOKENS:
        if any(token in name.lower() for name in object_names):
            return f"{token} ({index})", token, ITU_COLOURS[token]

    get = material.get if isinstance(material, dict) else lambda k, d=None: getattr(material, k, d)
    conductivity = float(get("conductivity", 0.0) or 0.0)
    permittivity = float(get("permittivity", 1.0) or 1.0)
    metal_conductivity = 1e3
    dielectric_high = 8.0
    if conductivity > metal_conductivity:
        return f"metal ({index})", "metal", ITU_COLOURS["metal"]
    if permittivity >= dielectric_high:
        return f"concrete ({index})", "concrete", ITU_COLOURS["concrete"]
    return f"material {index}", "other", [0.62, 0.58, 0.52]


def scene_geometry(name: str) -> tuple[bytes, dict[str, Any]]:
    """Pack a scenario's triangles into one float32 buffer plus a manifest.

    Triangles are grouped by material rather than by object: a converted
    interior can carry tens of thousands of objects but only a few dozen
    materials, and the viewer draws one batch per group.

    Args:
        name: Scenario name.

    Returns:
        Tuple of (positions buffer, manifest dict).

    """
    pairs = _load(name)
    dataset = pairs[0]
    materials = list(dataset.materials.values()) if hasattr(dataset.materials, "values") else list(
        dataset.materials,
    )

    by_material: dict[int, list[np.ndarray]] = {}
    names_by_material: dict[int, set[str]] = {}
    for obj in dataset.scene.objects:
        for face in obj.faces:
            material_idx = int(face.material_idx)
            vertices = np.asarray(face.vertices, dtype=np.float32)
            bucket = by_material.setdefault(material_idx, [])
            names_by_material.setdefault(material_idx, set()).add(obj.name)
            for i in range(1, len(vertices) - 1):
                bucket.append(vertices[[0, i, i + 1]])

    chunks, groups, start = [], [], 0
    for material_idx in sorted(by_material):
        triangles = by_material[material_idx]
        if not triangles:
            continue
        block = np.concatenate(triangles).astype(np.float32)
        chunks.append(block)
        record = materials[material_idx] if material_idx < len(materials) else {}
        label, material, colour = _material_style(
            record, material_idx, names_by_material.get(material_idx, set()),
        )
        groups.append(
            {
                "name": label,
                "material": material,
                "start": start,
                "count": len(block),
                "color": colour,
                "triangles": len(block) // 3,
            },
        )
        start += len(block)

    positions = np.concatenate(chunks) if chunks else np.zeros((0, 3), dtype=np.float32)
    lo = positions.min(axis=0).tolist() if len(positions) else [0.0, 0.0, 0.0]
    hi = positions.max(axis=0).tolist() if len(positions) else [1.0, 1.0, 1.0]

    markers = {"tx": []}
    for pair in pairs:
        markers["tx"].extend(np.asarray(pair.tx_pos).reshape(-1, 3).tolist())
    rx = np.asarray(dataset.rx_pos)
    markers["rx_z"] = float(np.median(rx[:, 2])) if rx.size else None

    manifest = {
        "groups": groups,
        "bbox": [lo, hi],
        "markers": markers,
        "vertices": int(len(positions)),
    }
    return positions.tobytes(), manifest


def scene_rays(name: str, rx_index: int, tx_index: int) -> dict[str, Any]:
    """Build the propagation paths reaching one receiver.

    Each path is the polyline transmitter -> interaction points -> receiver.
    ``inter_pos`` is padded with NaN out to the maximum interaction count, so
    the unused slots are trimmed per path.

    Args:
        name: Scenario name.
        rx_index: Receiver index.
        tx_index: Transmitter index.

    Returns:
        Dict with the receiver position and one entry per path.

    """
    pairs = _load(name)
    dataset = pairs[min(tx_index, len(pairs) - 1)]
    positions = np.asarray(dataset.rx_pos)
    rx_index = int(np.clip(rx_index, 0, len(positions) - 1))

    rx = positions[rx_index]
    tx = np.asarray(dataset.tx_pos).reshape(-1, 3)[0]
    inter = np.asarray(dataset.inter_pos)[rx_index]
    power = np.asarray(dataset.power)[rx_index]
    delay = np.asarray(dataset.delay)[rx_index]
    # One letter per bounce - R reflection, D diffraction, S scattering,
    # T transmission - so a path can be coloured by what happened along it.
    kinds = np.asarray(dataset.inter_str)[rx_index]

    paths = []
    for path_index in range(inter.shape[0]):
        if not np.isfinite(delay[path_index]):
            continue
        bounces = inter[path_index]
        bounces = bounces[np.isfinite(bounces).all(axis=1)]
        points = [tx.tolist(), *bounces.tolist(), rx.tolist()]
        paths.append(
            {
                "points": points,
                "power_db": float(power[path_index]),
                "delay_ns": float(delay[path_index]) * 1e9,
                "bounces": int(len(bounces)),
                "interactions": str(kinds[path_index] or ""),
            },
        )
    paths.sort(key=lambda p: -p["power_db"])
    return {
        "rx_index": rx_index,
        "rx": rx.tolist(),
        "tx": tx.tolist(),
        "paths": paths,
    }


def nearest_receiver(name: str, point: tuple[float, float, float]) -> int:
    """Find the receiver closest to a point, in the horizontal plane.

    Args:
        name: Scenario name.
        point: (x, y, z) picked in the 3D view.

    Returns:
        Index of the nearest receiver.

    """
    positions = np.asarray(_load(name)[0].rx_pos)
    return int(np.argmin(np.sum((positions[:, :2] - np.array(point[:2])) ** 2, axis=1)))


PAGE = r"""<!doctype html>
<meta charset="utf-8"><title>DeepMIMO scenario studio</title>
<style>
 :root{--bg:#0f1318;--panel:#161c23;--line:#252d36;--ink:#dfe4ea;--dim:#8c96a2;--accent:#3d7fd6}
 html,body{height:100%;margin:0}
 body{font:13px/1.45 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:var(--bg);
      color:var(--ink);display:flex;flex-direction:column}
 header{padding:8px 14px;background:#0b0e12;border-bottom:1px solid var(--line);flex:none;
        display:flex;align-items:center;gap:14px}
 h1{font-size:14px;margin:0;font-weight:600;letter-spacing:.01em}
 main{flex:1;display:flex;min-height:0}
 aside{width:320px;flex:none;display:flex;flex-direction:column;
       border-right:1px solid var(--line);background:#12171d}
 aside>.sec,aside>.err{flex:none}
 #panels{overflow:auto;flex:1}
 #actions{flex:none;padding:10px 12px;border-top:1px solid var(--line);background:#0e1319}
 .sec{border-bottom:1px solid var(--line)}
 .sec>h2{font-size:11px;margin:0;padding:9px 12px;color:var(--dim);text-transform:uppercase;
         letter-spacing:.08em;cursor:pointer;display:flex;align-items:center;gap:6px;user-select:none}
 .sec>h2:hover{color:var(--ink)}
 .sec .body{padding:0 12px 12px}
 .sec.closed .body{display:none}
 .g{display:grid;grid-template-columns:1fr 1fr;gap:8px}
 label{display:block;font-size:11px;color:var(--dim);margin:7px 0 2px}
 input,select{width:100%;box-sizing:border-box;background:#0b0f14;color:var(--ink);
      border:1px solid var(--line);border-radius:5px;padding:5px 7px;font-size:12px}
 input[type=range]{padding:0}
 input[type=checkbox]{width:auto;margin-right:6px}
 .chk{display:flex;align-items:center;font-size:12px;color:var(--ink);margin-top:9px}
 button{background:var(--accent);color:#fff;border:0;border-radius:6px;padding:9px;font-weight:600;
        cursor:pointer;font-size:12px;width:100%;margin-top:11px}
 button:hover{filter:brightness(1.12)}
 button:disabled{background:#2a323c;color:#6b7682;cursor:not-allowed}
 button.ghost{background:#232c38;color:#c7d0da;font-weight:500;padding:6px;margin-top:6px}
 button.ghost.on{background:#b8842a;color:#fff}
 #runwhy{color:#e0a33c}
 #stage{flex:1;position:relative;min-width:0}
 canvas{width:100%;height:100%;display:block}
 .float{position:absolute;background:rgba(11,14,18,.9);border:1px solid var(--line);
        border-radius:7px;padding:8px 10px;font-size:11px}
 #hud{right:10px;top:10px;color:var(--dim);text-align:right}
 #legend{left:10px;bottom:10px;max-height:42%;overflow:auto}
 #legend div{display:flex;align-items:center;gap:6px;cursor:pointer;padding:1px 0}
 #legend i{width:10px;height:10px;border-radius:2px;flex:none}
 #legend .off{opacity:.32;text-decoration:line-through}
 #covbox{position:absolute;right:10px;bottom:10px;width:34%;min-width:200px;max-width:80%;
      border:1px solid var(--line);border-radius:7px;overflow:hidden;
      background:#171d24;resize:horizontal;direction:rtl}
 #cov{display:block;width:100%;height:auto;direction:ltr}
 #prog{position:absolute;left:50%;transform:translateX(-50%);top:10px;width:min(560px,60%);
       display:none}
 .bar{height:6px;background:#222a33;border-radius:3px;overflow:hidden;margin:7px 0 5px}
 .bar>i{display:block;height:100%;background:var(--accent);width:0;transition:width .35s ease}
 .prow{display:flex;justify-content:space-between;gap:10px;color:var(--dim)}
 .prow b{color:var(--ink);font-weight:600}
 #plog{margin-top:6px;max-height:120px;overflow:auto;font:10.5px ui-monospace,monospace;
       color:#7f8994;white-space:pre-wrap}
 table{width:100%;border-collapse:collapse;font-size:11.5px}
 td{padding:2px 0;color:var(--dim)} td:last-child{text-align:right;color:var(--ink)}
 .err{color:#ff9090;white-space:pre-wrap;font:11px ui-monospace,monospace;padding:0 12px 10px}
 .hint{color:#69737e;font-size:10.5px;margin-top:5px}
 .tabs{display:flex;flex-wrap:wrap;gap:4px;margin:2px 0 4px}
 .tabs button{flex:1 1 28%;padding:4px 6px;font-size:10.5px;border-radius:5px;
   background:#1b222b;color:#95a0ad;border:1px solid #2b333d;cursor:pointer}
 .tabs button:hover{color:#dfe4ea;border-color:#3d4855}
 .tabs button.on{background:#2b6cb8;color:#fff;border-color:#2b6cb8}
 .sec.skip .body{opacity:.35;pointer-events:none}
 .sec.skip h2::after{content:' — skipped';color:#69737e;font-weight:400}
</style>
<header>
  <h1>DeepMIMO scenario studio</h1>
  <span id="genstate" class="hint"></span>
</header>
<main>
  <aside>
    <div id="panels">
    <div class="sec" id="s-scene"><h2>▾ Scenario</h2><div class="body">
      <select id="scenario" onchange="loadScene()"></select>
      <table id="summary"></table>
    </div></div>

    <div class="sec" id="s-src"><h2>▾ Source</h2><div class="body">
      <label>Geometry</label>
      <select id="p_source" onchange="sourcePicked()">
        <option value="new">Generate a new scene</option>
      </select>
      <label>…or a scene folder path</label>
      <input id="p_scenepath" placeholder="/path/to/scene (overrides the list)"
             oninput="sourcePicked()">
      <div id="matmodeRow" style="display:none">
        <label>Materials</label>
        <select id="p_matmode" onchange="sourceChanged()">
          <option value="auto">ITU names where known — re-derived at the new frequency</option>
          <option value="exact">Exactly as stored — valid at the original frequency</option>
        </select>
      </div>
      <div class="chk" id="reexportRow" style="display:none">
        <input type="checkbox" id="p_reexport" onchange="sourcePicked()">
        <span>Re-export geometry (re-apply budgets)</span></div>
      <div id="srchint" class="hint">A scene is built once and can be traced any number
        of times — reuse one to sweep frequency, reflections or TX/RX placement.</div>
    </div></div>

    <div class="sec" id="s-gen"><h2>▾ Generate scene</h2><div class="body">
      <label>Name</label>
      <input id="p_name" value="studio_scene" oninput="NAME_PICKED = true">
      <div class="g">
        <div><label>Seed</label><input id="p_seed" type="number" value="1"></div>
        <div><label>Building</label><select id="p_scene" onchange="sceneTypeChanged()"></select></div>
      </div>
      <div><label>Storey height [m]</label>
        <input id="p_wallh" type="number" step="0.1" placeholder="preset"></div>
      <div class="hint" id="scenehint"></div>
      <div class="g">
        <div><label>Rooms to furnish</label>
          <input id="p_rooms" type="number" min="0" value="1"
                 oninput="roomsChanged()"></div>
        <div><label>Restrict to type</label><select id="p_room"></select></div>
      </div>
      <div class="hint" id="roomshint"></div>
      <label>Furniture density <span id="l_furn" class="hint"></span></label>
      <input id="p_furniture" type="range" min="1" max="12" step="0.5" value="6"
             oninput="document.getElementById('l_furn').textContent='×'+this.value">
      <div class="chk"><input type="checkbox" id="p_fast" checked>
        <span>Fast solve (minutes, not hours)</span></div>
      <div class="hint">Full solve raises small-object detail but ran ~9 h for one
        apartment on this machine.</div>
    </div></div>

    <div class="sec closed" id="s-exp"><h2>▸ Geometry budget</h2><div class="body">
      <div class="g">
        <div><label>Architecture</label><input id="p_arch" type="number" value="2500"></div>
        <div><label>Furniture</label><input id="p_furn_b" type="number" value="1500"></div>
        <div><label>Ornament</label><input id="p_orn" type="number" value="120"></div>
        <div><label>Min size [m]</label><input id="p_min" type="number" step="0.01" value="0.10"></div>
      </div>
      <div class="chk"><input type="checkbox" id="p_doors" checked>
        <span>Open doorways</span></div>
      <div class="hint">Triangles kept per object. Ornament keeps its volume, not
        its detail; objects below the min size are dropped.</div>
    </div></div>

    <div class="sec" id="s-rt"><h2>▾ Ray tracing</h2><div class="body">
      <div class="g">
        <div><label>Frequency [GHz]</label><input id="p_freq" type="number" step="0.1" value="3.5"></div>
        <div><label>Max reflections</label><input id="p_refl" type="number" value="4"></div>
        <div><label>RX spacing [m]</label><input id="p_space" type="number" step="0.05" value="0.3"
             oninput="previewGrid()"></div>
        <div><label>RX height [m]</label><input id="p_rxh" type="number" step="0.1" value="1.2"
             oninput="previewGrid()"></div>
        <div><label>Transmitters</label><input id="p_ntx" type="number" value="2"></div>
        <div><label>Rays / source</label><input id="p_samp" type="number" step="50000" value="250000"></div>
        <div><label>TX height [m]</label><input id="p_txh" type="number" step="0.1" value="2.6"></div>
      </div>
      <div class="chk"><input type="checkbox" id="p_diff" checked>
        <span>Diffraction</span></div>
      <div class="hint">Without diffraction a cluttered interior reports large
        false outages: a warehouse went 87% → 53% served.</div>

      <label>Transmitter positions <span class="hint">x,y,z ; x,y,z</span></label>
      <input id="p_txpos" placeholder="auto — below the ceiling">
      <button id="pickTx" class="ghost" onclick="toggleTxPick()">Pick in 3D</button>
      <label>Receiver footprint <span class="hint">xmin,ymin,xmax,ymax</span></label>
      <input id="p_rxb" placeholder="auto — the whole footprint" oninput="previewGrid()">
      <div class="g">
        <button id="drawRx" class="ghost" onclick="toggleRxDraw()">Draw footprint</button>
        <button class="ghost" onclick="resetFootprint()">Whole footprint</button>
      </div>
      <div class="chk"><input type="checkbox" id="showGrid" checked onchange="previewGrid()">
        <span>Preview grid in 3D</span></div>
      <div class="hint" id="gridinfo"></div>


    </div></div>

    <div class="sec" id="s-view"><h2>▾ View</h2><div class="body">
      <label>Look</label>
      <select id="look" onchange="view.setTheme(this.value)">
        <option value="studio">Studio — white, lit from above</option>
        <option value="dark">Dark</option>
      </select>
      <div class="chk"><input type="checkbox" id="shadeOn" checked
             onchange="view.setShading(this.checked)">
        <span>Shadows and ambient occlusion</span></div>
      <div class="chk"><input type="checkbox" id="groundOn" checked
             onchange="view.setGround(this.checked)">
        <span>Ground plane</span></div>
      <div class="chk"><input type="checkbox" id="clipOn" checked onchange="applyClip()">
        <span>Cut at height</span></div>
      <input type="range" id="clipZ" min="0" max="10" step="0.05" value="1.3" oninput="applyClip()">
      <div class="hint" id="clipLabel">1.30 m</div>
      <div class="chk"><input type="checkbox" id="invY" onchange="view.setInvertPitch(this.checked)">
        <span>Invert vertical orbit</span></div>
      <label>Coverage map</label>
      <div id="covtabs" class="tabs"></div>
      <label>Coverage transmitter</label>
      <select id="tx" onchange="loadCoverage()"><option value="-1">best server</option></select>
      <div class="chk"><input type="checkbox" id="rayMode" checked>
        <span>Click a point to trace its rays</span></div>
      <label>Ray colour</label>
      <select id="rayColour" onchange="setRayColour(this.value)">
        <option value="power">received power</option>
        <option value="interaction">interaction type</option>
      </select>
      <div id="raylegend" class="hint"></div>
      <label>Ray thickness <span id="l_rayw" class="hint">4.5 px</span></label>
      <input id="rayW" type="range" min="1" max="12" step="0.5" value="4.5"
             oninput="view.setRayWidth(this.value);
                      document.getElementById('l_rayw').textContent=(+this.value).toFixed(1)+' px'">
      <div id="rayinfo" class="hint"></div>
      <div class="hint">drag orbit · shift+drag pan · wheel zoom · F frame · 1/3/7 views</div>
    </div></div>
    </div>
    <div class="err" id="err"></div>
    <div id="resume" style="display:none;padding:0 12px 10px">
      <button id="resumeBtn" onclick="convertTraced()">Convert traced paths</button>
    </div>
    <div id="actions">
      <button id="run" onclick="startRun()">Generate → trace → convert</button>
      <div id="runwhy" class="hint"></div>
    </div>
  </aside>

  <div id="stage">
    <canvas id="gl"></canvas>
    <div class="float" id="hud">—</div>
    <div class="float" id="legend"></div>
    <div class="float" id="prog">
      <div class="prow"><b id="pstage">—</b><span id="peta"></span></div>
      <div class="bar"><i id="pbar"></i></div>
      <div class="prow"><span id="pdetail"></span><span id="pelapsed"></span></div>
      <div id="plog"></div>
    </div>
    <div id="covbox" title="drag the lower-left corner to resize"><img id="cov"></div>
  </div>
</main>
<script type="module">
import { Viewer, INTERACTIONS } from './viewer.js';
const view = new Viewer(document.getElementById('gl'));
window.view = view;
let MANIFEST = null, POLL = null;
const $ = id => document.getElementById(id);

document.querySelectorAll('.sec>h2').forEach(h => h.onclick = () => {
  const sec = h.parentElement;
  sec.classList.toggle('closed');
  h.textContent = (sec.classList.contains('closed') ? '▸ ' : '▾ ') + h.textContent.slice(2);
});

window.applyClip = () => {
  const z = parseFloat($('clipZ').value);
  $('clipLabel').textContent = z.toFixed(2) + ' m';
  view.setClip($('clipOn').checked, z);
};

function fmt(sec) {
  if (sec === null || sec === undefined) return '';
  if (sec < 90) return Math.round(sec) + 's';
  const m = Math.floor(sec / 60);
  return m < 90 ? m + 'm' : Math.floor(m / 60) + 'h ' + (m % 60) + 'm';
}

function drawLegend() {
  const box = $('legend'); box.innerHTML = '';
  // Sionna needs one material instance per group, so a 41-object scene carries
  // 41 materials of a dozen classes. The legend is about classes; listing every
  // instance said "concrete" fourteen times and meant nothing.
  const classes = new Map();
  for (const g of MANIFEST.groups) {
    const key = g.material || g.name;
    const entry = classes.get(key) || {color: g.color, triangles: 0, names: []};
    entry.triangles += g.triangles;
    entry.names.push(g.name);
    classes.set(key, entry);
  }
  const hidden = new Set();
  for (const [key, entry] of [...classes].sort((a, b) => b[1].triangles - a[1].triangles)) {
    const c = entry.color.map(v => Math.round(v * 255)).join(',');
    const count = entry.names.length > 1 ? ` <span style="color:#69737e">×${entry.names.length}</span>` : '';
    const row = document.createElement('div');
    row.title = entry.names.join(', ');
    row.innerHTML = `<i style="background:rgb(${c})"></i><span>${key}${count}</span>` +
                    `<span style="margin-left:auto;color:#69737e">${entry.triangles.toLocaleString()}</span>`;
    row.onclick = () => {
      row.classList.toggle('off');
      if (row.classList.contains('off')) entry.names.forEach(n => hidden.add(n));
      else entry.names.forEach(n => hidden.delete(n));
      view.setHidden([...hidden]);
    };
    box.appendChild(row);
  }
}

window.loadScene = async () => {
  const name = $('scenario').value;
  if (!name) return;
  $('err').textContent = ''; $('hud').textContent = 'loading…';
  try {
    MANIFEST = await (await fetch('api/geometry?name=' + encodeURIComponent(name))).json();
    const buf = await (await fetch('api/geometry.bin?name=' + encodeURIComponent(name))).arrayBuffer();
    view.setScene(new Float32Array(buf), MANIFEST.groups, MANIFEST.bbox, MANIFEST.markers);
    $('clipZ').max = Math.max(3, MANIFEST.bbox[1][2].toFixed(2)); applyClip();
    drawLegend();
    const s = await (await fetch('api/summary?name=' + encodeURIComponent(name))).json();
    $('summary').innerHTML = Object.entries(s)
      .map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join('');
    $('hud').textContent =
      `${(MANIFEST.vertices / 3).toLocaleString()} triangles · ` +
      `${new Set(MANIFEST.groups.map(g => g.material || g.name)).size} materials ` +
      `in ${MANIFEST.groups.length} instances`;
    view.pickZ = MANIFEST.markers && MANIFEST.markers.rx_z != null
      ? MANIFEST.markers.rx_z : 1.2;
    RAY_PATHS = [];
    view.setRays([]);
    drawRayLegend();
    $('rayinfo').textContent = '';
    previewGrid();
    followViewer();
    const tx = $('tx');
    tx.innerHTML = '<option value="-1">best server</option>';
    for (let i = 0; i < s.tx; i++) tx.add(new Option('TX ' + i, i));
    loadCoverage();
  } catch (e) { $('err').textContent = e.message || String(e); }
};

async function showRays(query) {
  const r = await fetch('api/rays?' + query.toString());
  if (!r.ok) { $('err').textContent = await r.text(); return; }
  const data = await r.json();
  RAY_PATHS = data.paths || [];
  view.setRays(RAY_PATHS, RAY_COLOUR);
  drawRayLegend();
  const best = data.paths[0];
  $('rayinfo').innerHTML = data.paths.length
    ? `RX ${data.rx_index} at ${data.rx.map(v => v.toFixed(1)).join(', ')} — ` +
      `${data.paths.length} paths, strongest ${best.power_db.toFixed(1)} dBm ` +
      `at ${best.delay_ns.toFixed(1)} ns over ${best.bounces} bounces`
    : `RX ${data.rx_index}: no paths`;
}

window.showRays = showRays;

let rxDrawing = false, rxCorner = null;

function sceneFootprint() {
  if (!MANIFEST) return null;
  const [lo, hi] = MANIFEST.bbox;
  return [lo[0] + 0.3, lo[1] + 0.3, hi[0] - 0.3, hi[1] - 0.3];
}

function currentBounds() {
  const raw = $('p_rxb').value.trim();
  if (raw) {
    const v = raw.split(',').map(Number);
    if (v.length === 4 && v.every(Number.isFinite)) return v;
  }
  return sceneFootprint();
}

window.resetFootprint = () => { $('p_rxb').value = ''; previewGrid(); };

window.previewGrid = () => {
  if (!MANIFEST) return;
  const b = currentBounds();
  const spacing = Math.max(+$('p_space').value || 0.3, 0.05);
  const z = +$('p_rxh').value || 1.2;
  if (!b || !$('showGrid').checked) { view.setMarkers([], []); $('gridinfo').textContent = ''; return; }
  const pts = [], cols = [];
  // Cap the preview so a fine grid over a large scene stays interactive; the
  // count reported is the real one that will be traced.
  const nx = Math.floor((b[2] - b[0]) / spacing) + 1;
  const ny = Math.floor((b[3] - b[1]) / spacing) + 1;
  const total = Math.max(nx, 0) * Math.max(ny, 0);
  const step = Math.max(1, Math.ceil(Math.sqrt(total / 20000)));
  for (let i = 0; i < nx; i += step) {
    for (let j = 0; j < ny; j += step) {
      pts.push(b[0] + i * spacing, b[1] + j * spacing, z);
      cols.push(0.30, 0.72, 0.95);
    }
  }
  for (const t of (MANIFEST.markers?.tx || [])) { pts.push(...t); cols.push(1, 0.25, 0.2); }
  for (const chunk of ($('p_txpos').value.split(';'))) {
    const v = chunk.split(',').map(Number);
    if (v.length === 3 && v.every(Number.isFinite)) { pts.push(...v); cols.push(1, 0.85, 0.2); }
  }
  view.setMarkers(pts, cols, 4.0);
  $('gridinfo').textContent =
    `${total.toLocaleString()} receivers at ${spacing} m, z = ${z} m` +
    (step > 1 ? ` (preview shows every ${step}${step === 2 ? 'nd' : 'th'})` : '');
};

window.toggleRxDraw = () => {
  rxDrawing = !rxDrawing; rxCorner = null;
  $('drawRx').classList.toggle('on', rxDrawing);
  $('drawRx').textContent = rxDrawing ? 'Click two corners…' : 'Draw footprint';
};

let txPicking = false;
window.toggleTxPick = () => {
  txPicking = !txPicking;
  $('pickTx').classList.toggle('on', txPicking);
  $('pickTx').textContent = txPicking ? 'Picking — click in the scene' : 'Pick in 3D';
};

view.onPick(hit => {
  if (rxDrawing) {
    if (!rxCorner) {
      rxCorner = hit;
      $('drawRx').textContent = 'Click the opposite corner';
    } else {
      const b = [
        Math.min(rxCorner[0], hit[0]), Math.min(rxCorner[1], hit[1]),
        Math.max(rxCorner[0], hit[0]), Math.max(rxCorner[1], hit[1]),
      ].map(v => v.toFixed(2));
      $('p_rxb').value = b.join(',');
      toggleRxDraw();
      previewGrid();
    }
    return;
  }
  if (txPicking) {
    // Transmitters hang below the ceiling; the pick plane is at receiver height.
    const z = (+$('p_txh').value || (MANIFEST ? MANIFEST.bbox[1][2] - 0.3 : 2.6)).toFixed(2);
    const entry = `${hit[0].toFixed(2)},${hit[1].toFixed(2)},${z}`;
    const field = $('p_txpos');
    field.value = field.value.trim() ? field.value.trim() + '; ' + entry : entry;
    previewGrid();
    return;
  }
  if (!$('rayMode').checked) return;
  showRays(new URLSearchParams({
    name: $('scenario').value, x: hit[0], y: hit[1],
    tx: Math.max(0, +$('tx').value),
  }));
});

let COV_METRIC = 'power';

window.setCoverageMetric = (key) => {
  COV_METRIC = key;
  [...$('covtabs').children].forEach(b => b.classList.toggle('on', b.dataset.key === key));
  loadCoverage();
};

window.loadCoverage = () => {
  // The metric tabs are built before the scenario list arrives, so the first
  // call can land with nothing selected.
  if (!$('scenario').value) return;
  $('cov').src = `api/coverage?name=${encodeURIComponent($('scenario').value)}` +
                 `&tx=${$('tx').value}&metric=${COV_METRIC}&_=${Date.now()}`;
};

window.startRun = async () => {
  const body = {
    name: $('p_name').value,
    scene_type: $('p_scene').value, wall_height: +$('p_wallh').value || 0,
    source: $('p_scenepath').value.trim() || $('p_source').value,
    reexport: $('p_reexport').checked,
    material_mode: $('p_matmode').value,
    seed: +$('p_seed').value, room_type: $('p_room').value,
    tx_pos: $('p_txpos').value.trim(), rx_bounds: $('p_rxb').value.trim(),
    max_rooms: Math.max(0, +$('p_rooms').value || 0),
    furniture: +$('p_furniture').value, fast: $('p_fast').checked,
    architecture_budget: +$('p_arch').value, furniture_budget: +$('p_furn_b').value,
    ornament_budget: +$('p_orn').value, min_size: +$('p_min').value,
    open_doors: $('p_doors').checked,
    frequency: +$('p_freq').value * 1e9, max_reflections: +$('p_refl').value,
    spacing: +$('p_space').value, rx_height: +$('p_rxh').value,
    n_tx: +$('p_ntx').value, samples: +$('p_samp').value, diffraction: $('p_diff').checked,
    tx_height: +$('p_txh').value,
  };
  RUNNING = true; $('run').disabled = true; $('err').textContent = '';
  try {
    const r = await fetch('api/run', {method: 'POST', body: JSON.stringify(body)});
    if (!r.ok) throw new Error(await r.text());
    $('prog').style.display = 'block';
    poll();
  } catch (e) { $('err').textContent = e.message; RUNNING = false; sourceChanged(); }
};

async function poll() {
  clearInterval(POLL);
  POLL = setInterval(async () => {
    const j = await (await fetch('api/job')).json();
    if (j.status === 'none') { clearInterval(POLL); return; }
    $('pstage').textContent = j.stage;
    $('pbar').style.width = (j.overall * 100).toFixed(1) + '%';
    $('pdetail').textContent = j.detail || '';
    $('pelapsed').textContent = fmt(j.elapsed_seconds) + ' elapsed';
    $('peta').textContent = j.eta_seconds ? fmt(j.eta_seconds) + ' left' : '';
    $('plog').textContent = (j.log || []).slice(-8).join('\n');
    if (j.status !== 'running') {
      clearInterval(POLL); RUNNING = false; sourceChanged();
      if (j.status === 'failed') $('err').textContent = j.error || 'job failed';
      RESUMABLE = j.resumable || null;
      $('resume').style.display = RESUMABLE ? 'block' : 'none';
      await refreshScenes();
      if (j.scenario) { await refreshScenarios(); $('scenario').value = j.scenario; loadScene(); }
      followViewer();
    }
  }, 1200);
}

let SCENES = [], SCENE_TYPES = {}, RESUMABLE = null;
let RAY_PATHS = [], RAY_COLOUR = 'power';

window.setRayColour = (mode) => {
  RAY_COLOUR = mode;
  view.setRays(RAY_PATHS, mode);
  drawRayLegend();
};

function drawRayLegend() {
  const box = $('raylegend');
  if (RAY_COLOUR !== 'interaction' || !RAY_PATHS.length) {
    box.innerHTML = RAY_PATHS.length
      ? 'brightest paths carry the most power' : '';
    return;
  }
  // The hues sit below 3:1 against the scene, so the names carry the identity.
  box.innerHTML = (view.rayKinds || []).map(k => {
    const e = INTERACTIONS[k] || INTERACTIONS.S;
    const rgb = e.color.map(v => Math.round(v * 255)).join(',');
    return `<span style="display:inline-flex;align-items:center;gap:4px;margin-right:8px">` +
           `<i style="width:9px;height:9px;border-radius:2px;background:rgb(${rgb})"></i>` +
           `${e.label}</span>`;
  }).join('');
}

window.convertTraced = async () => {
  if (!RESUMABLE) return;
  $('resumeBtn').disabled = true; $('err').textContent = '';
  try {
    const body = {name: $('p_name').value, convert_only: RESUMABLE};
    const r = await fetch('api/run', {method: 'POST', body: JSON.stringify(body)});
    if (!r.ok) throw new Error(await r.text());
    $('prog').style.display = 'block'; RUNNING = true; poll();
  } catch (e) { $('err').textContent = e.message; }
  $('resumeBtn').disabled = false;
};

window.roomsChanged = () => {
  const n = Math.max(0, +$('p_rooms').value || 0);
  $('roomshint').textContent = n === 0
    ? 'all of them — hours on this machine'
    : n === 1
      ? 'one room gets furniture; the rest are laid out but empty (~3 min)'
      : `${n} rooms get furniture. Cost climbs steeply, not linearly: ` +
        'one room ran ~3 min, four ran into hours';
};

window.sceneTypeChanged = () => {
  $('scenehint').textContent = SCENE_TYPES[$('p_scene').value] || '';
};
// Generating is the expensive stage, so it is never what a run does by default:
// the source follows the scenario on screen until the user chooses otherwise.
let SOURCE_PICKED = false, NAME_PICKED = false, HAS_JOBS = true, RUNNING = false;
let CAN_GENERATE = true, GENERATOR_REASON = '', TRACER_REASON = '';

window.sourcePicked = () => { SOURCE_PICKED = true; sourceChanged(); };

function followViewer() {
  if (SOURCE_PICKED) return;
  const viewing = $('scenario').value;
  const options = [...$('p_source').querySelectorAll('option')].map(o => o.value);
  const built = viewing && [viewing + '__scene', viewing].find(
    name => options.includes(name) && SCENES.find(s => s.name === name)?.has_geometry);
  const rebuild = viewing && options.includes('scenario:' + viewing) && 'scenario:' + viewing;
  $('p_source').value = built || rebuild || 'new';
  sourceChanged();
}

const sceneLabel = s => (s.label || s.name) + (s.has_geometry ? '' : ' (needs export)');

function fillSources(sel, scenes, scenarios) {
  const keep = sel.value;
  sel.innerHTML = '';
  sel.add(new Option('Generate a new scene', 'new'));
  if (scenes.length) {
    const built = document.createElement('optgroup');
    built.label = 'Scene already built';
    scenes.forEach(s => built.appendChild(new Option(sceneLabel(s), s.name)));
    sel.appendChild(built);
  }
  if (scenarios.length) {
    const converted = document.createElement('optgroup');
    converted.label = 'DeepMIMO scenario — rebuild and trace again';
    scenarios.forEach(n => converted.appendChild(new Option(n, 'scenario:' + n)));
    sel.appendChild(converted);
  }
  sel.value = [...sel.querySelectorAll('option')].some(o => o.value === keep) ? keep : 'new';
}

window.sourceChanged = () => {
  const typed = $('p_scenepath').value.trim();
  const value = typed || $('p_source').value, reuse = value !== 'new';
  const fromScenario = value.startsWith('scenario:');
  const scene = SCENES.find(s => s.name === value);
  $('p_source').disabled = !!typed;
  // "__scene" is the folder a rebuild writes into, not part of the scenario name.
  const shortName = (fromScenario ? value.slice(9)
                  : scene ? (scene.label || scene.name)
                  : value.replace(/[\/]+$/, '').split('/').pop()).replace(/__scene$/, '');

  $('reexportRow').style.display = reuse && !fromScenario ? 'flex' : 'none';
  $('matmodeRow').style.display = fromScenario ? 'block' : 'none';
  if (reuse && scene && !scene.has_geometry) $('p_reexport').checked = true;
  if (reuse && scene && !scene.has_blend) $('p_reexport').checked = false;
  $('p_reexport').disabled = !!(scene && !scene.has_blend);

  const rebuilding = fromScenario || (reuse && $('p_reexport').checked);
  $('s-gen').classList.toggle('skip', reuse);
  $('s-exp').classList.toggle('skip', reuse && !(rebuilding && !fromScenario));
  $('run').textContent = !reuse ? 'Generate → trace → convert'
                       : fromScenario ? 'Rebuild → trace → convert'
                       : rebuilding ? 'Export → trace → convert' : 'Trace → convert';
  const blocked = !HAS_JOBS ? TRACER_REASON : (!reuse && !CAN_GENERATE ? GENERATOR_REASON : '');
  $('run').disabled = !!blocked || RUNNING;
  if (blocked) {
    $('runwhy').textContent = blocked;
  } else {
    $('runwhy').textContent = !reuse
      ? 'This will build a new scene from scratch — minutes to hours. To trace ' +
        'geometry you already have, pick it under Source.'
      : `No generation: this traces ${shortName} with the ray tracing parameters above.`;
  }
  $('srchint').textContent = !reuse
    ? 'A scene is built once and can be traced any number of times.'
    : fromScenario
      ? `Rebuilding the geometry and materials of ${shortName} into a Mitsuba scene, ` +
        'then tracing it with the parameters below. The original scenario is left alone.'
      : (rebuilding ? `Re-exporting ${shortName} from Blender, then tracing it. `
                    : `Tracing the geometry already in ${shortName}. `) +
        'The scenario is written under the name above, so one scene can hold ' +
        'several traces.';
  if (!NAME_PICKED) $('p_name').value = reuse ? shortName + '_rt' : 'studio_scene';
};

async function refreshScenes() {
  SCENES = (await (await fetch('api/scenes')).json()).scenes || [];
  const sc = await (await fetch('api/scenarios')).json();
  fillSources($('p_source'), SCENES, sc.scenarios || []);
  if (SOURCE_PICKED) sourceChanged(); else followViewer();
}

async function refreshScenarios() {
  const sc = await (await fetch('api/scenarios')).json();
  const ss = $('scenario'); const keep = ss.value;
  ss.innerHTML = ''; sc.scenarios.forEach(s => ss.add(new Option(s, s)));
  if (sc.scenarios.includes(keep)) ss.value = keep;
  return sc.scenarios;
}

(async () => {
  const opt = await (await fetch('api/options')).json();
  opt.room_types.forEach(r => $('p_room').add(new Option(r || 'any room type', r)));
  const tabs = $('covtabs');
  Object.entries(opt.coverage_metrics || {}).forEach(([key, title]) => {
    const b = document.createElement('button');
    b.textContent = title; b.dataset.key = key;
    b.onclick = () => setCoverageMetric(key);
    tabs.appendChild(b);
  });
  setCoverageMetric(COV_METRIC);
  SCENE_TYPES = opt.scene_types || {};
  Object.keys(SCENE_TYPES).forEach(k => $('p_scene').add(new Option(k, k)));
  sceneTypeChanged();
  roomsChanged();
  SCENES = opt.scenes || [];
  $('l_furn').textContent = '×' + $('p_furniture').value;
  HAS_JOBS = !!opt.has_jobs;
  CAN_GENERATE = !!opt.can_generate;
  TRACER_REASON = opt.tracer_reason || 'ray tracing unavailable';
  GENERATOR_REASON = opt.generator_reason || 'generation unavailable';
  $('genstate').textContent = !HAS_JOBS ? TRACER_REASON
                            : !CAN_GENERATE ? GENERATOR_REASON : '';
  const names = await refreshScenarios();
  fillSources($('p_source'), SCENES, names);
  followViewer();
  const q = new URLSearchParams(location.search);
  if (q.get('scenario') && names.includes(q.get('scenario'))) $('scenario').value = q.get('scenario');
  if (q.get('cut')) $('clipZ').value = q.get('cut');
  if (names.length) await loadScene();
  if (q.get('rx')) {
    showRays(new URLSearchParams({name: $('scenario').value, rx: q.get('rx'), tx: 0}));
  }
  const j = await (await fetch('api/job')).json();
  if (j.status === 'running') {
    $('prog').style.display = 'block'; RUNNING = true; $('run').disabled = true; poll();
  }
})();
</script>
"""


class Handler(BaseHTTPRequestHandler):
    """Serves the dashboard page and its PNG/JSON endpoints."""

    scenarios_dir: Path = Path()
    scene_roots: tuple[Path, ...] = ()
    generator_path: str | None = None
    generator_reason: str | None = None
    tracer_path: str | None = None
    tracer_reason: str | None = None
    jobs: JobManager | None = None
    generator_path: str | None = None

    def log_message(self, *args: Any) -> None:  # noqa: D102 - quiet by design
        return

    def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
        """Write one response.

        Args:
            body: Response body.
            content_type: MIME type.
            status: HTTP status code.

        """
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        """Route one request."""
        url = urlparse(self.path)
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            self._route(url.path, query)
        except Exception:  # noqa: BLE001 - surface any failure in the browser
            self._send(traceback.format_exc().encode(), "text/plain", status=500)

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        """Start a pipeline run."""
        if urlparse(self.path).path != "/api/run":
            self._send(b"not found", "text/plain", status=404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            params = json.loads(self.rfile.read(length) or b"{}")
            if self.jobs is None:
                msg = "job runner disabled: pass --gen-python and --rt-python"
                raise RuntimeError(msg)  # noqa: TRY301
            job = self.jobs.submit(params)
            self._send(json.dumps(job.snapshot()).encode(), "application/json")
        except Exception:  # noqa: BLE001 - surface any failure in the browser
            self._send(traceback.format_exc().encode(), "text/plain", status=500)

    def _scenes(self) -> list[dict[str, Any]]:
        """List generated scenes that can be traced again without regenerating.

        Returns:
            Reusable scenes, newest first.

        """
        if self.jobs is None:
            return []
        return list_scenes(self.jobs.work_root, *self.scene_roots)

    def _route(self, path: str, query: dict[str, str]) -> None:
        """Dispatch a request to its endpoint.

        Args:
            path: Request path.
            query: Parsed query parameters.

        """
        if path in ("/", "/index.html"):
            self._send(PAGE.encode(), "text/html; charset=utf-8")
        elif path == "/api/scenarios":
            names = sorted(
                p.name for p in self.scenarios_dir.iterdir() if (p / "params.json").exists()
            ) if self.scenarios_dir.is_dir() else []
            self._send(json.dumps({"scenarios": names}).encode(), "application/json")
        elif path == "/api/options":
            self._send(
                json.dumps(
                    {
                        "room_types": list(ROOM_TYPES),
                        "scene_types": SCENE_TYPE_HELP,
                        "coverage_metrics": {
                            key: spec["tab"] for key, spec in COVERAGE_METRICS.items()
                        },
                        "has_jobs": self.jobs is not None,
                        "can_generate": bool(self.generator_path),
                        "generator": self.generator_path,
                        "generator_reason": self.generator_reason,
                        "tracer": self.tracer_path,
                        "tracer_reason": self.tracer_reason,
                        "scenes": self._scenes(),
                    },
                ).encode(),
                "application/json",
            )
        elif path == "/api/scenes":
            self._send(json.dumps({"scenes": self._scenes()}).encode(), "application/json")
        elif path == "/api/job":
            job = self.jobs.get(query["id"]) if query.get("id") else self.jobs.latest()
            payload = job.snapshot() if job else {"status": "none"}
            self._send(json.dumps(payload).encode(), "application/json")
        elif path == "/api/rays":
            rx = query.get("rx")
            index = (
                int(rx)
                if rx is not None
                else nearest_receiver(
                    query["name"],
                    (float(query["x"]), float(query["y"]), float(query.get("z", 0))),
                )
            )
            self._send(
                json.dumps(
                    scene_rays(query["name"], index, int(query.get("tx", "0"))),
                ).encode(),
                "application/json",
            )
        elif path == "/api/coverage":
            self._send(
                render_coverage(
                    query["name"], int(query.get("tx", "-1")),
                    query.get("metric", "pathloss"),
                ),
                "image/png",
            )
        elif path == "/api/geometry":
            name = query["name"]
            if _GEOMETRY_CACHE.get("name") != name:
                buffer, manifest = scene_geometry(name)
                _GEOMETRY_CACHE.update(name=name, buffer=buffer, manifest=manifest)
            self._send(json.dumps(_GEOMETRY_CACHE["manifest"]).encode(), "application/json")
        elif path == "/api/geometry.bin":
            name = query["name"]
            if _GEOMETRY_CACHE.get("name") != name:
                buffer, manifest = scene_geometry(name)
                _GEOMETRY_CACHE.update(name=name, buffer=buffer, manifest=manifest)
            self._send(_GEOMETRY_CACHE["buffer"], "application/octet-stream")
        elif path in ("/viewer.js", "/mat4.js"):
            asset = Path(__file__).parent / "static" / path.lstrip("/")
            self._send(asset.read_bytes(), "text/javascript")
        elif path == "/api/summary":
            self._send(json.dumps(scenario_summary(query["name"])).encode(), "application/json")
        else:
            self._send(b"not found", "text/plain", status=404)


#: Modules each stage needs, and the environments they usually live in.
GENERATOR_MODULES = ("bpy", "infinigen")
GENERATOR_VENVS = (".venv-infinigen", "venv-infinigen", ".venv-infinigen3")
TRACER_MODULES = ("mitsuba", "sionna")
TRACER_VENVS = (".venv-rt", "venv-rt", ".venv-sionna")


def _has_modules(interpreter: str, modules: tuple[str, ...]) -> bool:
    """Ask an interpreter whether it can import a set of modules.

    ``find_spec`` is used rather than a real import: importing Mitsuba or bpy
    costs seconds, and all that matters here is whether they are installed.

    Args:
        interpreter: Path to a Python interpreter.
        modules: Module names to look for.

    Returns:
        True if every module is importable.

    """
    names = ",".join(f"'{module}'" for module in modules)
    probe = (
        "import importlib.util as u, sys; "
        f"sys.exit(0 if all(u.find_spec(m) for m in ({names},)) else 1)"
    )
    try:
        return subprocess.run(  # noqa: S603 - the interpreter path comes from a flag
            [interpreter, "-c", probe], check=False, capture_output=True, timeout=60,
        ).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _find_python(venvs: tuple[str, ...], modules: tuple[str, ...]) -> str | None:
    """Look beside the repository for an interpreter that has the given modules.

    Args:
        venvs: Virtual environment folder names to try, in order.
        modules: Module names the interpreter must provide.

    Returns:
        Path to a usable interpreter, or None.

    """
    here = Path(__file__).resolve()
    for base in (here.parents[2], here.parents[3]):
        for name in venvs:
            candidate = base / name / "bin" / "python"
            if candidate.exists() and _has_modules(str(candidate), modules):
                return str(candidate)
    return None


def _resolve_interpreter(
    given: str | None,
    venvs: tuple[str, ...],
    modules: tuple[str, ...],
    label: str,
) -> tuple[str | None, str | None]:
    """Pick an interpreter for one stage and say why it is unusable.

    Checking at startup matters more than it sounds: an interpreter that cannot
    import what it needs otherwise fails deep inside a job, minutes after the
    button was pressed.

    Args:
        given: Interpreter passed on the command line, if any.
        venvs: Environment names to search when nothing was passed.
        modules: Modules the interpreter must provide.
        label: Stage name, for the message.

    Returns:
        The interpreter to use and a reason it cannot be used.

    """
    missing = " and ".join(modules)
    if given:
        if not Path(given).exists():
            return None, f"{label} disabled: {given} does not exist"
        if not _has_modules(given, modules):
            return None, f"{label} disabled: {given} cannot import {missing}"
        return given, None

    found = _find_python(venvs, modules)
    if found:
        return found, None
    return None, (
        f"{label} disabled: no interpreter with {missing} found. "
        f"Pass one explicitly, or create a {venvs[0]} beside the repository"
    )


def main() -> None:
    """Start the dashboard server."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--scenarios",
        default=None,
        help="Scenarios folder (defaults to DeepMIMO's, relative to the working directory).",
    )
    parser.add_argument(
        "--gen-python",
        default=None,
        help=(
            "Interpreter with Infinigen and bpy, for generating and exporting scenes. "
            "Auto-detected from a sibling .venv-infinigen if not given."
        ),
    )
    parser.add_argument(
        "--rt-python",
        default=None,
        help=(
            "Interpreter with Sionna and DeepMIMO, for ray tracing. Auto-detected "
            "from a sibling .venv-rt, or from this one if it has Sionna."
        ),
    )
    parser.add_argument(
        "--work",
        default=str(Path.home() / ".deepmimo" / "scenario_studio"),
        help=(
            "Folder for generated scenes. Defaults outside the repository: a "
            "furnished scene is hundreds of megabytes and does not belong in git."
        ),
    )
    parser.add_argument(
        "--scene-root",
        action="append",
        default=[],
        metavar="DIR",
        help=(
            "Extra folder of already-built scenes to offer for tracing. "
            "Repeatable; scenes in it are listed by full path."
        ),
    )
    args = parser.parse_args()

    Handler.scenarios_dir = Path(args.scenarios or dm.get_scenarios_dir())
    Handler.scene_roots = tuple(Path(root).resolve() for root in args.scene_root)

    # Tracing and generating need different environments, and tracing an
    # existing scene needs no generator at all - so they are resolved apart.
    rt_python = args.rt_python
    if not rt_python and _has_modules(sys.executable, TRACER_MODULES):
        rt_python = sys.executable
    rt_python, rt_reason = _resolve_interpreter(
        rt_python, TRACER_VENVS, TRACER_MODULES, "ray tracing",
    )
    gen_python, gen_reason = _resolve_interpreter(
        args.gen_python, GENERATOR_VENVS, GENERATOR_MODULES, "generation",
    )

    Handler.generator_path = gen_python
    Handler.generator_reason = gen_reason
    Handler.tracer_path = rt_python
    Handler.tracer_reason = rt_reason
    if rt_python:
        runner = Path(__file__).resolve().parents[1] / "pipelines" / "infinigen_pipeline_runner.py"
        Handler.jobs = JobManager(
            gen_python=gen_python or "",
            rt_python=rt_python,
            runner=runner,
            work_root=Path(args.work).resolve(),
        )
    print(f"ray tracing: {rt_python or rt_reason}")
    print(f"generation:  {gen_python or gen_reason}")
    print(f"scenarios: {Handler.scenarios_dir}")
    print(f"dashboard: http://127.0.0.1:{args.port}")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
