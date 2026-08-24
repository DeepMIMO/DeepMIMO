"""Convert a furnished Infinigen indoor scene into a Mitsuba scene for ray tracing.

Infinigen generates photorealistic interiors - real room graphs, doors, windows
and a full fit-out - procedurally varied and freely licensed. It budgets its
geometry for what a camera sees, so a furnished apartment runs to millions of
polygons, most of them in foliage and shelf ornaments.

At a 3.5 GHz carrier the wavelength is ~8.5 cm, which changes what that geometry
is worth. A leaf is invisible; the pot it grows from is a solid object that
blocks. So nothing here is thrown away for being decorative - every object is
kept, and each is decimated to a triangle budget chosen for what it does to a
wave. Architecture keeps the most detail because its placement defines the
environment, furniture rather less, and ornament is reduced to the volume it
occupies. Only objects smaller than the wavelength itself are dropped, because a
wave cannot resolve them at all.

Materials come from Infinigen's shader names, which describe appearance and so
proxy composition well, mapped onto ITU-R P.2040 materials.

``bpy`` is imported lazily inside :func:`export_blend`, so the classification and
material rules stay importable and testable without Blender.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# Mitsuba scene format version, matching the other pipelines.
MITSUBA_VERSION = "2.1.0"

# ITU material names, as Sionna's itu-radio-material plugin expects them.
MAT_CONCRETE = "concrete"
MAT_BRICK = "brick"
MAT_PLASTERBOARD = "plasterboard"
MAT_WOOD = "wood"
MAT_GLASS = "glass"
MAT_METAL = "metal"

# Objects whose largest dimension is below this are dropped: a 3.5 GHz wave has
# a ~8.5 cm wavelength and cannot resolve them.
MIN_OBJECT_SIZE_M = 0.10

# Door leaves, as opposed to the openings they hang in. Infinigen closes every
# door, which walls each room off from the next: a shut 0.18 m panel is a
# serious obstruction at 3.5 GHz, and a scenario of sealed rooms says more about
# the doors than the building. Leaving the openings clear is the usual choice
# for indoor propagation studies, so the leaves are dropped by default.
# Infinigen has many door factories - PanelDoor, GlassPanelDoor, LiteDoor,
# LouverDoor and more - so any factory naming a door is treated as a leaf, and
# the parts of the opening that should stay are excluded by name instead.
DOOR_OPENING_PATTERNS: tuple[str, ...] = ("doorframe", "doorway", "doorsill", "doorstep")

# How Infinigen's shader names map onto ITU materials, matched in order.
SHADER_MATERIAL_RULES: tuple[tuple[str, str], ...] = (
    ("glass", MAT_GLASS),
    ("mirror", MAT_METAL),
    ("metal", MAT_METAL),
    ("steel", MAT_METAL),
    ("aluminum", MAT_METAL),
    ("hardware", MAT_METAL),
    ("plaster", MAT_PLASTERBOARD),
    ("drywall", MAT_PLASTERBOARD),
    ("brick", MAT_BRICK),
    ("concrete", MAT_CONCRETE),
    ("marble", MAT_CONCRETE),
    ("ceramic", MAT_CONCRETE),
    ("tile", MAT_CONCRETE),
    ("stone", MAT_CONCRETE),
    ("wood", MAT_WOOD),
    ("shelves", MAT_WOOD),
    ("plastic", MAT_WOOD),
    ("fabric", MAT_WOOD),
    ("lampshade", MAT_WOOD),
)
DEFAULT_MATERIAL = MAT_WOOD

# What an object does to a wave, which sets how much of its detail is worth
# keeping.
CLASS_ARCHITECTURE = "architecture"
CLASS_FURNITURE = "furniture"
CLASS_ORNAMENT = "ornament"

# Matched in order against the lower-cased factory name, so compound names come
# before the general tokens they contain: a CeilingLightFactory is a light
# fitting rather than ceiling, and a bookcase is furniture rather than books.
CLASSIFICATION_RULES: tuple[tuple[str, str], ...] = (
    ("ceilinglight", CLASS_ORNAMENT),
    ("bookcase", CLASS_FURNITURE),
    ("bookshelf", CLASS_FURNITURE),
    ("bookcolumn", CLASS_FURNITURE),
    ("doorknob", CLASS_ORNAMENT),
    ("doorhandle", CLASS_ORNAMENT),
    ("windowsill", CLASS_ARCHITECTURE),
    ("trinket", CLASS_ORNAMENT),
    ("bookstack", CLASS_ORNAMENT),
    ("tabletop", CLASS_ORNAMENT),
    ("shelfdecor", CLASS_ORNAMENT),
    # Architecture: the shell and its openings.
    ("wall", CLASS_ARCHITECTURE),
    ("floor", CLASS_ARCHITECTURE),
    ("ceiling", CLASS_ARCHITECTURE),
    ("door", CLASS_ARCHITECTURE),
    ("window", CLASS_ARCHITECTURE),
    ("skirting", CLASS_ARCHITECTURE),
    ("stair", CLASS_ARCHITECTURE),
    ("room", CLASS_ARCHITECTURE),
    ("pillar", CLASS_ARCHITECTURE),
    ("column", CLASS_ARCHITECTURE),
    ("balcony", CLASS_ARCHITECTURE),
    # Furniture: large enough to block or reflect at head height.
    ("cabinet", CLASS_FURNITURE),
    ("wardrobe", CLASS_FURNITURE),
    ("dresser", CLASS_FURNITURE),
    ("table", CLASS_FURNITURE),
    ("desk", CLASS_FURNITURE),
    ("chair", CLASS_FURNITURE),
    ("stool", CLASS_FURNITURE),
    ("sofa", CLASS_FURNITURE),
    ("couch", CLASS_FURNITURE),
    ("bed", CLASS_FURNITURE),
    ("shelf", CLASS_FURNITURE),
    ("shelves", CLASS_FURNITURE),
    ("counter", CLASS_FURNITURE),
    ("bathtub", CLASS_FURNITURE),
    ("toilet", CLASS_FURNITURE),
    ("sink", CLASS_FURNITURE),
    ("oven", CLASS_FURNITURE),
    ("stove", CLASS_FURNITURE),
    ("fridge", CLASS_FURNITURE),
    ("refrigerator", CLASS_FURNITURE),
    ("dishwasher", CLASS_FURNITURE),
    ("television", CLASS_FURNITURE),
    ("monitor", CLASS_FURNITURE),
    ("rug", CLASS_FURNITURE),
    ("curtain", CLASS_FURNITURE),
    # Ornament: kept as a blocking volume, stripped of detail.
    ("plant", CLASS_ORNAMENT),
    ("flower", CLASS_ORNAMENT),
    ("foliage", CLASS_ORNAMENT),
    ("leaf", CLASS_ORNAMENT),
    ("vase", CLASS_ORNAMENT),
    ("bottle", CLASS_ORNAMENT),
    ("book", CLASS_ORNAMENT),
    ("fruit", CLASS_ORNAMENT),
    ("food", CLASS_ORNAMENT),
    ("cup", CLASS_ORNAMENT),
    ("bowl", CLASS_ORNAMENT),
    ("plate", CLASS_ORNAMENT),
    ("utensil", CLASS_ORNAMENT),
    ("towel", CLASS_ORNAMENT),
    ("pillow", CLASS_ORNAMENT),
    ("blanket", CLASS_ORNAMENT),
    ("art", CLASS_ORNAMENT),
    ("decor", CLASS_ORNAMENT),
    ("lamp", CLASS_ORNAMENT),
    ("light", CLASS_ORNAMENT),
    ("mollusk", CLASS_ORNAMENT),
    ("nature", CLASS_ORNAMENT),
)

# Triangles kept per object, by class. Even architecture is budgeted: an
# Infinigen window frame carries tens of thousands of triangles of moulding
# profile that a 8.5 cm wave cannot resolve, while the aperture it frames
# matters a great deal. Ornament keeps just enough to hold its volume.
DEFAULT_BUDGETS: dict[str, int] = {
    CLASS_ARCHITECTURE: 2500,
    CLASS_FURNITURE: 1500,
    CLASS_ORNAMENT: 120,
}


def map_shader_to_material(shader_name: str | None) -> str:
    """Map an Infinigen shader name onto an ITU material.

    Args:
        shader_name: Shader name such as ``"shader_window_glass"``, or None.

    Returns:
        ITU material name.

    """
    if not shader_name:
        return DEFAULT_MATERIAL
    lowered = shader_name.lower()
    for token, material in SHADER_MATERIAL_RULES:
        if token in lowered:
            return material
    return DEFAULT_MATERIAL


def factory_name(object_name: str) -> str:
    """Strip Infinigen's instance decoration from an object name.

    ``"PanelDoorFactory(2243540).spawn_asset(35)"`` becomes ``"PanelDoorFactory"``
    and ``"living-room_0/0.001"`` becomes ``"living-room_0/0"``.

    Args:
        object_name: Raw Blender object name.

    Returns:
        The bare factory or room name.

    """
    return re.sub(r"[(.].*$", "", object_name)


def is_door_leaf(object_name: str) -> bool:
    """Check whether an object is a door panel rather than its opening.

    Infinigen names the leaf after its factory (``PanelDoorFactory``) and also
    emits a plain ``door`` box per opening; the surrounding frame and reveal
    belong to the room mesh and are unaffected.

    Args:
        object_name: Raw Blender object name.

    Returns:
        True if the object is a door leaf.

    """
    name = factory_name(object_name).lower()
    if any(token in name for token in DOOR_OPENING_PATTERNS):
        return False
    return "door" in name


def classify_object(object_name: str) -> str:
    """Classify an object by what it does to a wave.

    Args:
        object_name: Raw Blender object name.

    Returns:
        One of :data:`CLASS_ARCHITECTURE`, :data:`CLASS_FURNITURE`,
        :data:`CLASS_ORNAMENT`.

    """
    name = factory_name(object_name).lower()
    for token, object_class in CLASSIFICATION_RULES:
        if token in name:
            return object_class
    return CLASS_FURNITURE


def group_name(object_name: str, material: str) -> str:
    """Build the scene-object name for a mesh group.

    Args:
        object_name: Raw Blender object name.
        material: ITU material name.

    Returns:
        A group name safe to use as a Mitsuba shape id.

    """
    base = factory_name(object_name).lower()
    base = re.sub(r"factory$", "", base)
    base = re.sub(r"[^a-z0-9]+", "_", base).strip("_") or "object"
    return f"{base}_{material}"


def write_ply(
    vertices: list[tuple[float, float, float]],
    triangles: list[tuple[int, int, int]],
    path: Path,
) -> None:
    """Write a triangulated mesh as an ASCII PLY.

    Args:
        vertices: Vertex coordinates.
        triangles: Vertex-index triples.
        path: Destination path.

    """
    lines = [
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
    lines.extend(f"{x:.4f} {y:.4f} {z:.4f}" for x, y, z in vertices)
    lines.extend(f"3 {a} {b} {c}" for a, b, c in triangles)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_mitsuba_xml(scene_folder: Path, groups: dict[str, str]) -> Path:
    """Write a Mitsuba scene.xml referencing one PLY per group.

    Two Sionna behaviours drive the naming. It strips the ``mesh-`` and ``mat-``
    prefixes to name scene objects and materials, and rejects a material whose
    name collides with an object - so materials take a numbered ITU namespace
    rather than reusing the group name. And shapes that share one material
    *instance* are merged into a single unnamed object, so every group gets its
    own BSDF even when two groups use the same ITU material.

    Args:
        scene_folder: Folder holding ``meshes/``.
        groups: Mapping of group name to ITU material name.

    Returns:
        Path to the written XML file.

    """
    material_ids = {
        group: f"itu_{material}_{index}"
        for index, (group, material) in enumerate(sorted(groups.items()))
    }
    lines = [f'<scene version="{MITSUBA_VERSION}">', "", "    <!-- Materials -->"]
    for group, material in sorted(groups.items()):
        lines += [
            f'    <bsdf type="itu-radio-material" id="mat-{material_ids[group]}">',
            f'        <string name="type" value="{material}"/>',
            "    </bsdf>",
        ]
    lines += ["", "    <!-- Shapes -->"]
    for group in sorted(groups):
        lines += [
            f'    <shape type="ply" id="mesh-{group}">',
            f'        <string name="filename" value="meshes/{group}.ply"/>',
            '        <boolean name="face_normals" value="true"/>',
            f'        <ref id="mat-{material_ids[group]}" name="bsdf"/>',
            "    </shape>",
        ]
    lines += ["", "</scene>", ""]
    xml_path = scene_folder / "scene.xml"
    xml_path.write_text("\n".join(lines), encoding="utf-8")
    return xml_path


def export_blend(  # noqa: C901, PLR0913, PLR0915 - one knob per export stage
    blend_path: str | Path,
    scene_folder: str | Path,
    *,
    budgets: dict[str, int] | None = None,
    min_size: float = MIN_OBJECT_SIZE_M,
    open_doors: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Export a furnished Infinigen ``.blend`` as a Mitsuba scene.

    Requires ``bpy``, imported here so the rules above stay importable without
    Blender.

    Args:
        blend_path: Path to the Infinigen scene file.
        scene_folder: Output directory for ``scene.xml`` and ``meshes/``.
        budgets: Triangles kept per object, keyed by class. Defaults to
            :data:`DEFAULT_BUDGETS`.
        min_size: Objects whose largest dimension is below this are dropped, in
            metres.
        open_doors: Drop door leaves so every doorway is an opening, letting
            rays pass between rooms. The frames and reveals are unaffected.
        verbose: Print a summary.

    Returns:
        Dict describing what was exported, also written to ``export.json``.

    """
    import bpy  # noqa: PLC0415 - heavy optional dependency, only needed here

    bpy.ops.wm.open_mainfile(filepath=str(blend_path))
    budgets = budgets or DEFAULT_BUDGETS

    folder = Path(scene_folder)
    mesh_dir = folder / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    buckets: dict[str, tuple[str, list, list, dict]] = {}
    stats = {
        "kept": 0,
        "doors_opened": 0,
        "dropped_subwavelength": 0,
        "decimated": 0,
        "polys_in": 0,
        "polys_out": 0,
        "by_class": {},
    }

    for obj in list(bpy.data.objects):
        if obj.type != "MESH" or not obj.visible_get():
            continue
        n_polys = len(obj.data.polygons)
        stats["polys_in"] += n_polys
        if n_polys == 0:
            continue

        if max(obj.dimensions) < min_size:
            stats["dropped_subwavelength"] += n_polys
            continue

        if open_doors and is_door_leaf(obj.name):
            stats["doors_opened"] += 1
            continue

        obj_class = classify_object(obj.name)
        budget = budgets.get(obj_class)
        if budget is not None and n_polys > budget:
            modifier = obj.modifiers.new(name="dm_decimate", type="DECIMATE")
            modifier.ratio = budget / n_polys
            stats["decimated"] += 1

        depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        mesh.calc_loop_triangles()
        matrix = evaluated.matrix_world
        shaders = [slot.material.name if slot.material else None for slot in obj.material_slots]

        for tri in mesh.loop_triangles:
            shader = shaders[tri.material_index] if tri.material_index < len(shaders) else None
            material = map_shader_to_material(shader)
            group = group_name(obj.name, material)
            bucket = buckets.setdefault(group, (material, [], [], {}))
            corners = []
            for vertex_index in tri.vertices:
                point = matrix @ mesh.vertices[vertex_index].co
                key = (round(point.x, 4), round(point.y, 4), round(point.z, 4))
                index = bucket[3].get(key)
                if index is None:
                    index = len(bucket[1])
                    bucket[3][key] = index
                    bucket[1].append((point.x, point.y, point.z))
                corners.append(index)
            bucket[2].append(tuple(corners))
            stats["polys_out"] += 1

        evaluated.to_mesh_clear()
        stats["kept"] += 1
        stats["by_class"][obj_class] = stats["by_class"].get(obj_class, 0) + 1

    groups = {group: material for group, (material, _, tris, _) in buckets.items() if tris}
    for group, (_, vertices, triangles, _) in buckets.items():
        if triangles:
            write_ply(vertices, triangles, mesh_dir / f"{group}.ply")
    write_mitsuba_xml(folder, groups)

    summary = {"source": str(blend_path), "groups": groups, **stats}
    (folder / "export.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if verbose:
        print(
            f"exported {stats['polys_out']:,} triangles in {len(groups)} groups "
            f"from {stats['polys_in']:,} ({stats['kept']} objects, "
            f"{stats['decimated']} decimated, {stats['doors_opened']} doors opened)",
        )
    return summary
