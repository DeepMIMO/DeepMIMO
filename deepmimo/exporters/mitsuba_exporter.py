"""Export a DeepMIMO scenario back to a Mitsuba scene for re-tracing.

A converted scenario keeps its geometry and its material constants, so it can be
turned back into a Mitsuba scene and ray traced again - at a different carrier
frequency, with different transmitters, or with a different solver setting -
without the original Wireless InSite, Sionna or Blender source.

Two material modes are supported. ``"auto"`` writes an ``itu-radio-material``
whenever the scenario's material name matches an ITU class Sionna knows, so the
constants are re-derived at the new frequency; anything else keeps the exact
permittivity and conductivity stored in the scenario. ``"exact"`` always keeps
the stored constants, which are only valid at the frequency they were measured
for.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable

MITSUBA_VERSION = "2.1.0"

#: Material classes Sionna's ``itu-radio-material`` plugin can evaluate.
ITU_TYPES = frozenset(
    {
        "concrete",
        "brick",
        "plasterboard",
        "wood",
        "glass",
        "ceiling_board",
        "chipboard",
        "plywood",
        "marble",
        "floorboard",
        "metal",
        "very_dry_ground",
        "medium_dry_ground",
        "wet_ground",
    },
)

#: Trailing frequency annotations the converters add, e.g. "ITU Concrete 3.5 GHz".
_FREQUENCY_SUFFIX = re.compile(r"[\s_]*[\d.]+\s*[gmk]?hz\s*$", re.IGNORECASE)

#: Coordinates are welded at this resolution, well below any wavelength of interest.
WELD_DECIMALS = 5

MIN_POLYGON_VERTICES = 3


def normalise_material_name(name: str) -> str:
    """Reduce a scenario material name to a bare material word.

    Args:
        name: Material name as stored in the scenario, e.g. ``"ITU Concrete 3.5 GHz"``.

    Returns:
        A lowercase identifier such as ``"concrete"``.

    """
    text = _FREQUENCY_SUFFIX.sub("", str(name or "")).strip()
    text = re.sub(r"^itu[\s_-]*", "", text, flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _triangulate(polygon: np.ndarray) -> list[tuple[int, int, int]]:
    """Split a planar polygon into triangles.

    A fan is only correct for polygons that are star-shaped from their first
    vertex, so concave faces are ear clipped instead. Ray tracing is sensitive to
    the difference: a fan over a concave outline adds surface that is not there.

    Args:
        polygon: ``(n, 3)`` vertices in order around the face.

    Returns:
        Index triples into ``polygon``.

    """
    count = len(polygon)
    if count < MIN_POLYGON_VERTICES:
        return []
    if count == MIN_POLYGON_VERTICES:
        return [(0, 1, 2)]

    flat = _flatten(polygon)
    if flat is None:
        return [(0, i, i + 1) for i in range(1, count - 1)]
    return _ear_clip(flat) or [(0, i, i + 1) for i in range(1, count - 1)]


def _flatten(polygon: np.ndarray) -> np.ndarray | None:
    """Drop the axis most aligned with the face normal to get 2-D coordinates.

    Args:
        polygon: ``(n, 3)`` vertices.

    Returns:
        ``(n, 2)`` coordinates, or None if the face is degenerate.

    """
    points = np.asarray(polygon, dtype=np.float64)
    normal = np.zeros(3)
    for i in range(len(points)):
        current, following = points[i], points[(i + 1) % len(points)]
        normal += np.cross(current, following)
    if not np.isfinite(normal).all() or np.linalg.norm(normal) < 1e-12:  # noqa: PLR2004
        return None
    drop = int(np.argmax(np.abs(normal)))
    keep = [axis for axis in range(3) if axis != drop]
    return points[:, keep]


def _signed_area(points: np.ndarray) -> float:
    """Compute the signed area of a 2-D polygon.

    Args:
        points: ``(n, 2)`` coordinates.

    Returns:
        Positive for counter-clockwise winding.

    """
    following = np.roll(points, -1, axis=0)
    return 0.5 * float(np.sum(points[:, 0] * following[:, 1] - following[:, 0] * points[:, 1]))


def _inside(triangle: np.ndarray, point: np.ndarray) -> bool:
    """Test whether a point lies inside a triangle.

    Args:
        triangle: ``(3, 2)`` triangle corners.
        point: ``(2,)`` point.

    Returns:
        True if the point is inside or on the edge.

    """
    a, b, c = triangle
    d1 = (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])
    d2 = (c[0] - b[0]) * (point[1] - b[1]) - (c[1] - b[1]) * (point[0] - b[0])
    d3 = (a[0] - c[0]) * (point[1] - c[1]) - (a[1] - c[1]) * (point[0] - c[0])
    return not ((d1 < 0 or d2 < 0 or d3 < 0) and (d1 > 0 or d2 > 0 or d3 > 0))


def _ear_clip(points: np.ndarray) -> list[tuple[int, int, int]]:
    """Ear clip a simple 2-D polygon.

    Args:
        points: ``(n, 2)`` coordinates in order around the polygon.

    Returns:
        Index triples, or an empty list if the polygon could not be clipped.

    """
    remaining = list(range(len(points)))
    if _signed_area(points) < 0:
        remaining.reverse()

    triangles: list[tuple[int, int, int]] = []
    guard = 0
    limit = len(points) ** 2
    while len(remaining) > MIN_POLYGON_VERTICES and guard < limit:
        guard += 1
        for position in range(len(remaining)):
            i = remaining[position - 1]
            j = remaining[position]
            k = remaining[(position + 1) % len(remaining)]
            triangle = points[[i, j, k]]
            if _signed_area(triangle) <= 0:
                continue
            others = [idx for idx in remaining if idx not in (i, j, k)]
            if any(_inside(triangle, points[idx]) for idx in others):
                continue
            triangles.append((i, j, k))
            remaining.remove(j)
            break
        else:
            return []
    if len(remaining) == MIN_POLYGON_VERTICES:
        triangles.append(tuple(remaining))  # type: ignore[arg-type]
    return triangles


def _safe(name: str, fallback: str = "object") -> str:
    """Turn any label into a Mitsuba-safe identifier.

    Args:
        name: Raw label.
        fallback: Value to use when nothing survives.

    Returns:
        A lowercase identifier.

    """
    cleaned = re.sub(r"[^a-z0-9]+", "_", str(name or "").lower()).strip("_")
    return cleaned or fallback


def write_ply(vertices: np.ndarray, triangles: np.ndarray, path: Path) -> None:
    """Write a triangulated mesh as an ASCII PLY.

    Args:
        vertices: ``(n, 3)`` coordinates.
        triangles: ``(m, 3)`` vertex indices.
        path: Destination path.

    """
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
        f"element face {len(triangles)}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    body = [f"{x:.5f} {y:.5f} {z:.5f}" for x, y, z in vertices]
    body += [f"3 {a} {b} {c}" for a, b, c in triangles]
    path.write_text("\n".join(header + body) + "\n", encoding="utf-8")


def _weld(triangles: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Merge duplicate vertices across a group's triangles.

    Converters emit each triangle with its own corners; leaving them unmerged
    multiplies the vertex count and makes the re-converted scenario look like
    thousands of separate objects.

    Args:
        triangles: List of ``(3, 3)`` triangles.

    Returns:
        Unique vertices and the triangles as index triples.

    """
    corners = np.concatenate(triangles).astype(np.float64)
    keys = np.round(corners, WELD_DECIMALS)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    vertices = corners[np.sort(first)]
    order = np.argsort(np.argsort(first))
    faces = order[inverse].reshape(-1, 3)
    return vertices, faces


def _material_entry(record: Any, index: int, mode: str) -> dict[str, Any]:
    """Describe how one scenario material should be written.

    Args:
        record: Material record from the scenario.
        index: Its index in the material list.
        mode: ``"auto"`` or ``"exact"``.

    Returns:
        A dict with the BSDF type, id and properties.

    """
    raw = getattr(record, "name", None) or f"material_{index}"
    word = normalise_material_name(raw)
    if mode == "auto" and word in ITU_TYPES:
        return {"bsdf": "itu-radio-material", "itu": word, "source": raw}
    return {
        "bsdf": "radio-material",
        "permittivity": float(getattr(record, "permittivity", 1.0) or 1.0),
        "conductivity": float(getattr(record, "conductivity", 0.0) or 0.0),
        "source": raw,
        "word": word or f"material_{index}",
    }


def _bsdf_lines(bsdf_id: str, entry: dict[str, Any]) -> list[str]:
    """Render one BSDF element.

    Args:
        bsdf_id: Identifier to give the BSDF.
        entry: Material description from :func:`_material_entry`.

    Returns:
        XML lines.

    """
    if entry["bsdf"] == "itu-radio-material":
        return [
            f'    <bsdf type="itu-radio-material" id="{bsdf_id}">',
            f'        <string name="type" value="{entry["itu"]}"/>',
            "    </bsdf>",
        ]
    return [
        f'    <bsdf type="radio-material" id="{bsdf_id}">',
        f'        <float name="relative_permittivity" value="{entry["permittivity"]:.6g}"/>',
        f'        <float name="conductivity" value="{entry["conductivity"]:.6g}"/>',
        "    </bsdf>",
    ]


def export_scene(  # noqa: PLR0913 - one knob per grouping and material rule
    objects: Iterable[Any],
    materials: list[Any],
    folder: str | Path,
    *,
    group_by: str = "object",
    material_mode: str = "auto",
    max_shapes: int = 600,
) -> dict[str, Any]:
    """Write a DeepMIMO scene as a Mitsuba scene folder.

    Sionna merges shapes that share one material *instance*, so every shape is
    given its own BSDF: that is what keeps object identity through a re-trace.
    Scenes converted from a hull representation can carry a shape per wall, so
    grouping falls back to one shape per material once ``max_shapes`` is passed.

    Args:
        objects: Scene objects, each with ``name`` and ``faces``.
        materials: Material records indexed by ``face.material_idx``.
        folder: Destination folder; ``scene.xml`` and ``meshes/`` are written into it.
        group_by: ``"object"`` keeps one shape per object and material,
            ``"material"`` merges everything sharing a material.
        material_mode: ``"auto"`` or ``"exact"``; see the module docstring.
        max_shapes: Shape count above which ``"object"`` grouping is abandoned.

    Returns:
        A report with the shape count, triangle count and per-material treatment.

    Raises:
        ValueError: If the scene holds no usable geometry.

    """
    scene_folder = Path(folder)
    mesh_folder = scene_folder / "meshes"
    mesh_folder.mkdir(parents=True, exist_ok=True)

    # Objects are kept apart by index, not by name: a converter may give many
    # objects the same name (every building called "buildings", say), and
    # merging those would throw away the identity this grouping exists for.
    per_object: dict[tuple[int, int], list[np.ndarray]] = {}
    labels: dict[int, str] = {}
    for obj_index, obj in enumerate(objects):
        labels[obj_index] = _safe(getattr(obj, "name", "") or f"object_{obj_index}")
        for face in obj.faces:
            polygon = np.asarray(face.vertices, dtype=np.float64)
            triangles = [polygon[list(triple)] for triple in _triangulate(polygon)]
            if triangles:
                per_object.setdefault((obj_index, int(face.material_idx)), []).extend(triangles)

    if not per_object:
        msg = "the scene holds no faces to export"
        raise ValueError(msg)

    regrouped = group_by != "object" or len(per_object) > max_shapes
    buckets: dict[tuple[int, int], list[np.ndarray]] = {}
    for (obj_index, material_idx), triangles in per_object.items():
        key = (-1, material_idx) if regrouped else (obj_index, material_idx)
        buckets.setdefault(key, []).extend(triangles)

    entries = {
        idx: _material_entry(
            materials[idx] if idx < len(materials) else None, idx, material_mode,
        )
        for idx in {material_idx for _, material_idx in buckets}
    }

    used: dict[str, int] = {}
    shapes: list[dict[str, Any]] = []
    triangle_total = 0
    for order, (key, triangles) in enumerate(sorted(buckets.items())):
        obj_index, material_idx = key
        label = labels.get(obj_index, "scene")
        entry = entries[material_idx]
        word = entry.get("itu") or entry.get("word") or "material"
        stem = word if regrouped else f"{label}_{word}"
        used[stem] = used.get(stem, 0) + 1
        if used[stem] > 1:
            stem = f"{stem}_{used[stem] - 1}"

        vertices, faces = _weld(triangles)
        write_ply(vertices, faces, mesh_folder / f"{stem}.ply")
        triangle_total += len(faces)
        # A material id may not collide with an ITU class name unless it is
        # meant as one, so every id is numbered and prefixed.
        bsdf_id = (
            f"mat-itu_{entry['itu']}_{order}"
            if entry["bsdf"] == "itu-radio-material"
            else f"mat-dm_{word}_{order}"
        )
        shapes.append({"stem": stem, "bsdf_id": bsdf_id, "entry": entry})

    lines = [f'<scene version="{MITSUBA_VERSION}">', "", "    <!-- Materials -->"]
    for shape in shapes:
        lines += _bsdf_lines(shape["bsdf_id"], shape["entry"])
    lines += ["", "    <!-- Shapes -->"]
    for shape in shapes:
        lines += [
            f'    <shape type="ply" id="mesh-{shape["stem"]}">',
            f'        <string name="filename" value="meshes/{shape["stem"]}.ply"/>',
            '        <boolean name="face_normals" value="true"/>',
            f'        <ref id="{shape["bsdf_id"]}" name="bsdf"/>',
            "    </shape>",
        ]
    lines += ["", "</scene>", ""]
    (scene_folder / "scene.xml").write_text("\n".join(lines), encoding="utf-8")

    treatment = {
        entry["source"]: (
            f"itu:{entry['itu']}"
            if entry["bsdf"] == "itu-radio-material"
            else f"exact:eps={entry['permittivity']:.4g},sigma={entry['conductivity']:.4g}"
        )
        for entry in entries.values()
    }
    return {
        "scene_xml": str(scene_folder / "scene.xml"),
        "shapes": len(shapes),
        "triangles": triangle_total,
        "grouped_by": "material" if regrouped else "object",
        "objects": len(labels),
        "materials": treatment,
    }


def export_scenario(
    scenario: str,
    folder: str | Path,
    *,
    group_by: str = "object",
    material_mode: str = "auto",
    max_shapes: int = 600,
) -> dict[str, Any]:
    """Export a converted DeepMIMO scenario as a Mitsuba scene folder.

    Args:
        scenario: Scenario name, as passed to :func:`deepmimo.load`.
        folder: Destination folder.
        group_by: ``"object"`` or ``"material"``.
        material_mode: ``"auto"`` or ``"exact"``.
        max_shapes: Shape count above which object grouping is abandoned.

    Returns:
        The export report, with the scenario's carrier frequency added so a
        re-trace can warn when the frequency changes.

    """
    import deepmimo as dm  # noqa: PLC0415 - keeps the exporter importable without a scenario

    dataset = dm.load(scenario)
    first = dataset[0] if hasattr(dataset, "__getitem__") and not hasattr(
        dataset, "scene",
    ) else dataset
    materials = first.materials
    material_list = list(materials.values()) if hasattr(materials, "values") else list(materials)

    report = export_scene(
        first.scene.objects,
        material_list,
        folder,
        group_by=group_by,
        material_mode=material_mode,
        max_shapes=max_shapes,
    )
    # A Dataset raises KeyError for a missing field rather than AttributeError,
    # so getattr's default does not apply and the lookup has to be guarded.
    try:
        rt_params = dict(first.rt_params)
    except (KeyError, AttributeError, TypeError):
        rt_params = {}

    report["scenario"] = scenario
    report["frequency"] = float(rt_params.get("frequency", 0.0) or 0.0)
    report["original_rt"] = {
        key: rt_params[key]
        for key in ("max_reflections", "max_diffractions", "max_transmissions", "num_rays")
        if key in rt_params
    }
    (Path(folder) / "source.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
