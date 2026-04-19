"""OSM to Sionna/Mitsuba scene generator — no Blender required.

Downloads building footprints from OpenStreetMap via the Overpass API,
extrudes them into 3D meshes, and writes a Mitsuba 2.1 XML scene file
that Sionna RT can load directly with ``sionna.rt.load_scene()``.

Typical usage::

    from deepmimo.pipelines.osm_to_mitsuba import generate_scene

    scene_folder = generate_scene(
        minlat=40.7485, minlon=-73.9865,
        maxlat=40.7545, maxlon=-73.9800,
        scene_folder="my_scene",
    )

The returned ``scene_folder`` contains:

- ``scene.xml``         — Mitsuba scene description
- ``meshes/ground.ply`` — flat ground plane
- ``meshes/building_N.ply`` — one PLY mesh per OSM building

Requirements: ``requests``, ``numpy``, and ``utm`` (all in deepmimo base deps).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import requests

from deepmimo.pipelines.utils.geo_utils import xy_from_latlong

OVERPASS_TIMEOUT = 90  # seconds for the Overpass query itself
_OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]
_OVERPASS_HEADERS = {
    "User-Agent": "DeepMIMO/4 (https://github.com/DeepMIMO/DeepMIMO; deepmimo@nvidia.com)",
    "Accept": "application/json",
}
DEFAULT_BUILDING_HEIGHT = 10.0  # meters when OSM tag absent
FLOOR_HEIGHT_PER_LEVEL = 3.0  # meters per floor for buildings:levels tag
GROUND_PADDING = 30.0  # extra meters around the bbox for the ground plane
MITSUBA_VERSION = "2.1.0"

# ITU material type strings (must match Sionna's itu-radio-material plugin)
BUILDING_MATERIAL = "concrete"
GROUND_MATERIAL = "concrete"


# ---------------------------------------------------------------------------
# OSM querying
# ---------------------------------------------------------------------------


def query_osm_buildings(
    minlat: float,
    minlon: float,
    maxlat: float,
    maxlon: float,
) -> list[dict[str, Any]]:
    """Download building footprints from OpenStreetMap for a GPS bounding box.

    Uses the Overpass API.  Each returned dict has:
    - ``coords``: list of (lon, lat) tuples forming the closed polygon
    - ``height``: estimated building height in metres

    Args:
        minlat: South boundary latitude.
        minlon: West boundary longitude.
        maxlat: North boundary latitude.
        maxlon: East boundary longitude.

    Returns:
        List of building dicts, each with ``coords`` and ``height`` keys.

    """
    query = f"""
[out:json][timeout:{OVERPASS_TIMEOUT}][maxsize:1073741824];
(
  way["building"]({minlat},{minlon},{maxlat},{maxlon});
);
out body;
>;
out skel qt;
"""
    response = None
    for mirror in _OVERPASS_MIRRORS:
        try:
            response = requests.get(
                mirror,
                params={"data": query},
                headers=_OVERPASS_HEADERS,
                timeout=OVERPASS_TIMEOUT + 10,
            )
            if response.status_code == requests.codes.ok:
                break
            print(f"Overpass mirror {mirror} returned {response.status_code}, trying next…")
        except requests.exceptions.RequestException as exc:
            print(f"Overpass mirror {mirror} failed: {exc}, trying next…")
    response.raise_for_status()
    data = response.json()

    # Build a fast node-id → (lon, lat) lookup
    nodes: dict[int, tuple[float, float]] = {}
    for elem in data["elements"]:
        if elem["type"] == "node":
            nodes[elem["id"]] = (elem["lon"], elem["lat"])

    buildings = []
    for elem in data["elements"]:
        if elem["type"] != "way":
            continue
        tags = elem.get("tags", {})
        if "building" not in tags:
            continue

        coords = [nodes[nid] for nid in elem.get("nodes", []) if nid in nodes]
        min_polygon_vertices = 3
        if len(coords) < min_polygon_vertices:
            continue

        height = _parse_height(tags)
        buildings.append({"coords": coords, "height": height})

    return buildings


def _parse_height(tags: dict[str, str]) -> float:
    """Extract building height from OSM tags, falling back to a default."""
    for key in ("height", "building:height"):
        if key in tags:
            try:
                return float(tags[key].split()[0])  # strip " m" suffix if present
            except (ValueError, IndexError):
                pass
    if "building:levels" in tags:
        try:
            return float(tags["building:levels"]) * FLOOR_HEIGHT_PER_LEVEL
        except ValueError:
            pass
    return DEFAULT_BUILDING_HEIGHT


# ---------------------------------------------------------------------------
# Mesh generation
# ---------------------------------------------------------------------------


def _extrude_footprint(
    footprint_xy: list[tuple[float, float]],
    height: float,
) -> tuple[np.ndarray, list[list[int]]] | tuple[None, None]:
    """Extrude a 2D polygon footprint into a closed 3D building mesh.

    Args:
        footprint_xy: Ordered (x, y) pairs in metres.  May include a
            duplicate closing vertex, which is removed automatically.
        height: Building height in metres.

    Returns:
        ``(vertices, triangles)`` where *vertices* is (N, 3) float32 and
        *triangles* is a list of [i, j, k] index triples.  Returns
        ``(None, None)`` if the polygon has fewer than 3 distinct vertices.

    """
    pts = list(footprint_xy)
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]
    n = len(pts)
    if n < 3:  # noqa: PLR2004
        return None, None

    # Vertex layout: bottom ring 0..n-1, top ring n..2n-1
    verts: list[list[float]] = [[x, y, 0.0] for x, y in pts] + [
        [x, y, float(height)] for x, y in pts
    ]

    # Side walls: N quads → 2N triangles
    sides = [
        tri for i in range(n) for j in [(i + 1) % n] for tri in ([i, j, j + n], [i, j + n, i + n])
    ]

    # Roof: fan from first top vertex (vertex index n)
    roof = [[n, n + i, n + i + 1] for i in range(1, n - 1)]

    return np.array(verts, dtype=np.float32), sides + roof


def _ground_plane(
    cx: float, cy: float, half_w: float, half_h: float
) -> tuple[np.ndarray, list[list[int]]]:
    """Return vertices and triangles for a flat rectangular ground plane."""
    verts = np.array(
        [
            [cx - half_w, cy - half_h, 0.0],
            [cx + half_w, cy - half_h, 0.0],
            [cx + half_w, cy + half_h, 0.0],
            [cx - half_w, cy + half_h, 0.0],
        ],
        dtype=np.float32,
    )
    tris = [[0, 1, 2], [0, 2, 3]]
    return verts, tris


# ---------------------------------------------------------------------------
# PLY and XML writers
# ---------------------------------------------------------------------------


def _write_ply(vertices: np.ndarray, triangles: list[list[int]], path: Path) -> None:
    """Write an ASCII PLY file with the given triangulated mesh."""
    n_v = len(vertices)
    n_f = len(triangles)
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {n_v}",
        "property float x",
        "property float y",
        "property float z",
        f"element face {n_f}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    lines.extend(f"{v[0]:.4f} {v[1]:.4f} {v[2]:.4f}" for v in vertices)
    lines.extend(f"3 {t[0]} {t[1]} {t[2]}" for t in triangles)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_mitsuba_xml(
    scene_folder: Path,
    mesh_entries: list[dict[str, str]],
) -> Path:
    """Write a Mitsuba scene.xml referencing the given PLY mesh files."""
    mat_ids = sorted({e["material"] for e in mesh_entries})

    lines = [f'<scene version="{MITSUBA_VERSION}">', "", "    <!-- Materials -->"]
    for mat_id in mat_ids:
        lines += [
            f'    <bsdf type="itu-radio-material" id="{mat_id}">',
            f'        <string name="type" value="{mat_id}"/>',
            "    </bsdf>",
        ]

    lines += ["", "    <!-- Shapes -->"]
    for entry in mesh_entries:
        lines += [
            f'    <shape type="ply" id="{entry["id"]}">',
            f'        <string name="filename" value="{entry["filename"]}"/>',
            '        <boolean name="face_normals" value="true"/>',
            f'        <ref id="{entry["material"]}" name="bsdf"/>',
            "    </shape>",
        ]

    lines += ["", "</scene>", ""]
    xml_path = scene_folder / "scene.xml"
    xml_path.write_text("\n".join(lines), encoding="utf-8")
    return xml_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_scene(  # noqa: PLR0913
    minlat: float,
    minlon: float,
    maxlat: float,
    maxlon: float,
    scene_folder: str,
    *,
    ground_padding: float = GROUND_PADDING,
    verbose: bool = True,
) -> str:
    """Generate a Mitsuba scene from OpenStreetMap building data.

    Downloads building footprints for the given GPS bounding box, extrudes
    them into 3D meshes, creates a ground plane, and writes a Mitsuba XML
    scene file that Sionna RT can load directly.

    Args:
        minlat: South boundary latitude.
        minlon: West boundary longitude.
        maxlat: North boundary latitude.
        maxlon: East boundary longitude.
        scene_folder: Output directory.  Created if it does not exist.
        ground_padding: Ground plane extends this many metres beyond the
            bounding box on each side (default 30 m).
        verbose: Print progress messages.

    Returns:
        Path to the generated *scene folder* (not the XML file), matching
        the convention of ``raytrace_sionna`` and the Blender-based pipeline.

    """
    folder = Path(scene_folder)
    folder.mkdir(parents=True, exist_ok=True)

    # If the scene was already generated, skip the Overpass download.
    if (folder / "scene.xml").exists() and (folder / "osm_gps_origin.txt").exists():
        if verbose:
            print(f"Scene already exists at {folder}, skipping generation.")
        return str(folder)

    (folder / "meshes").mkdir(exist_ok=True)

    # Local origin at centre of bounding box (UTM metres)
    origin_lat = (minlat + maxlat) / 2
    origin_lon = (minlon + maxlon) / 2
    ox, oy = xy_from_latlong(origin_lat, origin_lon)

    # Save origin so downstream code (get_origin_coords) can read it
    (folder / "osm_gps_origin.txt").write_text(f"{origin_lat}\n{origin_lon}\n", encoding="utf-8")

    # Ground-plane half-extents
    corners = [(minlat, minlon), (maxlat, maxlon)]
    xs = [xy_from_latlong(lat, lon)[0] - ox for lat, lon in corners]
    ys = [xy_from_latlong(lat, lon)[1] - oy for lat, lon in corners]
    half_w = (max(xs) - min(xs)) / 2 + ground_padding
    half_h = (max(ys) - min(ys)) / 2 + ground_padding

    # --- Ground plane ---
    gv, gt = _ground_plane(0.0, 0.0, half_w, half_h)
    _write_ply(gv, gt, folder / "meshes" / "ground.ply")
    mesh_entries: list[dict[str, str]] = [
        {"id": "mesh-ground", "filename": "meshes/ground.ply", "material": GROUND_MATERIAL},
    ]

    # --- Buildings ---
    if verbose:
        print(
            f"Querying OpenStreetMap for buildings in "
            f"[{minlat:.4f},{minlon:.4f}]-[{maxlat:.4f},{maxlon:.4f}]..."
        )
    buildings = query_osm_buildings(minlat, minlon, maxlat, maxlon)
    if verbose:
        print(f"Found {len(buildings)} buildings")

    n_written = 0
    for idx, bldg in enumerate(buildings):
        footprint_xy = [
            (xy_from_latlong(lat, lon)[0] - ox, xy_from_latlong(lat, lon)[1] - oy)
            for lon, lat in bldg["coords"]
        ]
        verts, tris = _extrude_footprint(footprint_xy, bldg["height"])
        if verts is None:
            continue

        ply_rel = f"meshes/building_{idx}.ply"
        _write_ply(verts, tris, folder / ply_rel)
        mesh_entries.append(
            {
                "id": f"mesh-building_{idx}",
                "filename": ply_rel,
                "material": BUILDING_MATERIAL,
            }
        )
        n_written += 1

    if verbose:
        print(f"Wrote {n_written} building meshes  ({len(buildings) - n_written} skipped)")

    xml_path = _write_mitsuba_xml(folder, mesh_entries)
    if verbose:
        print(f"Scene written to {xml_path}")

    return str(folder)
