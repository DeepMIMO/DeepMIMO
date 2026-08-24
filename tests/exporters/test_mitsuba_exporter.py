"""Tests for turning a DeepMIMO scene back into a Mitsuba scene.

The exporter exists so an already-converted scenario can be ray traced again
without its original source files, so the tests focus on what a re-trace depends
on: the geometry surviving unchanged, and each material reaching Sionna in a form
it will read.
"""

import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np
import pytest

from deepmimo.exporters import mitsuba_exporter as mx


def _scene_root(folder):
    """Parse a scene the exporter just wrote.

    Args:
        folder: Folder holding ``scene.xml``.

    Returns:
        The root element.

    """
    return ET.parse(folder / "scene.xml").getroot()  # noqa: S314 - our own output


def _face(vertices, material_idx=0):
    """Build a stand-in for a scene face.

    Args:
        vertices: Polygon corners.
        material_idx: Index into the material list.

    Returns:
        An object with the attributes the exporter reads.

    """
    return SimpleNamespace(vertices=np.asarray(vertices, dtype=float), material_idx=material_idx)


def _material(name, permittivity=5.24, conductivity=0.12):
    """Build a stand-in for a scenario material.

    Args:
        name: Material name.
        permittivity: Relative permittivity.
        conductivity: Conductivity in S/m.

    Returns:
        An object with the attributes the exporter reads.

    """
    return SimpleNamespace(name=name, permittivity=permittivity, conductivity=conductivity)


def _triangle_area(polygon, triangles):
    """Sum the area of a triangulation.

    Args:
        polygon: ``(n, 3)`` polygon vertices.
        triangles: Index triples.

    Returns:
        Total area.

    """
    total = 0.0
    for a, b, c in triangles:
        p, q, r = polygon[a], polygon[b], polygon[c]
        total += 0.5 * float(np.linalg.norm(np.cross(q - p, r - p)))
    return total


U_SHAPE = np.array(
    [[0, 0, 0], [3, 0, 0], [3, 3, 0], [2, 3, 0], [2, 1, 0], [1, 1, 0], [1, 3, 0], [0, 3, 0]],
    dtype=float,
)


def test_concave_face_keeps_its_area() -> None:
    """A fan over a U-shape invents surface a ray tracer would then reflect off."""
    triangles = mx._triangulate(U_SHAPE)  # noqa: SLF001 - the triangulation is the unit under test
    fan = [(0, i, i + 1) for i in range(1, len(U_SHAPE) - 1)]

    assert _triangle_area(U_SHAPE, triangles) == pytest.approx(7.0)
    assert _triangle_area(U_SHAPE, fan) == pytest.approx(11.0)


def test_vertical_face_is_not_flattened_away() -> None:
    """Projection must drop the axis along the normal, not a fixed one."""
    wall = np.array([[5, 0, 0], [5, 4, 0], [5, 4, 3], [5, 0, 3]], dtype=float)

    triangles = mx._triangulate(wall)  # noqa: SLF001 - see above

    assert _triangle_area(wall, triangles) == pytest.approx(12.0)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ITU Concrete 3.5 GHz", "concrete"),
        ("itu_ceiling_board", "ceiling_board"),
        ("ITU Wet earth 3.5 GHz", "wet_earth"),
        ("material_7", "material_7"),
        (None, ""),
    ],
)
def test_material_names_lose_their_frequency_annotation(raw, expected) -> None:
    """Converters stamp the frequency into the name; the material word is what matters."""
    assert mx.normalise_material_name(raw) == expected


def test_known_itu_materials_are_written_as_itu(tmp_path) -> None:
    """An ITU material must reach Sionna as a plugin it can re-derive."""
    objects = [SimpleNamespace(name="building", faces=[_face([[0, 0, 0], [1, 0, 0], [1, 1, 0]])])]

    mx.export_scene(objects, [_material("ITU Concrete 3.5 GHz")], tmp_path)

    bsdf = _scene_root(tmp_path).find("bsdf")
    assert bsdf.attrib["type"] == "itu-radio-material"
    assert bsdf.find("string[@name='type']").attrib["value"] == "concrete"
    assert bsdf.attrib["id"].startswith("mat-itu_")


def test_unknown_materials_keep_their_measured_constants(tmp_path) -> None:
    """Anything Sionna has no ITU class for must round-trip its own numbers."""
    objects = [SimpleNamespace(name="wall", faces=[_face([[0, 0, 0], [1, 0, 0], [1, 1, 0]])])]

    mx.export_scene(objects, [_material("material_3", 18.18, 0.7645)], tmp_path)

    bsdf = _scene_root(tmp_path).find("bsdf")
    assert bsdf.attrib["type"] == "radio-material"
    permittivity = bsdf.find("float[@name='relative_permittivity']").attrib["value"]
    conductivity = bsdf.find("float[@name='conductivity']").attrib["value"]
    assert float(permittivity) == pytest.approx(18.18)
    assert float(conductivity) == pytest.approx(0.7645)
    # Sionna rejects a plain BSDF whose name is an ITU class, so ids stay namespaced.
    assert not bsdf.attrib["id"].removeprefix("mat-").startswith("itu")


def test_exact_mode_never_substitutes_itu_constants(tmp_path) -> None:
    """The stored constants are the scenario's own measurement; exact mode keeps them."""
    objects = [SimpleNamespace(name="wall", faces=[_face([[0, 0, 0], [1, 0, 0], [1, 1, 0]])])]

    mx.export_scene(
        objects, [_material("ITU Concrete 3.5 GHz", 5.31, 0.0899)], tmp_path,
        material_mode="exact",
    )

    bsdf = _scene_root(tmp_path).find("bsdf")
    assert bsdf.attrib["type"] == "radio-material"
    permittivity = bsdf.find("float[@name='relative_permittivity']").attrib["value"]
    assert float(permittivity) == pytest.approx(5.31)


def test_objects_sharing_a_name_stay_separate(tmp_path) -> None:
    """Converters may call every building "buildings"; merging them loses the scene."""
    objects = [
        SimpleNamespace(name="buildings", faces=[_face([[0, 0, 0], [1, 0, 0], [1, 1, 0]])]),
        SimpleNamespace(name="buildings", faces=[_face([[5, 5, 0], [6, 5, 0], [6, 6, 0]])]),
    ]

    report = mx.export_scene(objects, [_material("ITU Concrete 3.5 GHz")], tmp_path)

    assert report["shapes"] == 2
    assert report["grouped_by"] == "object"
    assert len(list((tmp_path / "meshes").glob("*.ply"))) == 2


def test_every_shape_gets_its_own_material_instance(tmp_path) -> None:
    """Sionna merges shapes that share a material instance, erasing object identity."""
    objects = [
        SimpleNamespace(name="a", faces=[_face([[0, 0, 0], [1, 0, 0], [1, 1, 0]])]),
        SimpleNamespace(name="b", faces=[_face([[5, 5, 0], [6, 5, 0], [6, 6, 0]])]),
    ]

    mx.export_scene(objects, [_material("ITU Concrete 3.5 GHz")], tmp_path)

    root = _scene_root(tmp_path)
    refs = [shape.find("ref").attrib["id"] for shape in root.findall("shape")]
    assert len(set(refs)) == len(refs)


def test_a_crowded_scene_falls_back_to_material_grouping(tmp_path) -> None:
    """A hull conversion can hold a shape per wall; that is too many for one scene."""
    objects = [
        SimpleNamespace(name=f"wall_{i}", faces=[_face([[i, 0, 0], [i + 1, 0, 0], [i + 1, 1, 0]])])
        for i in range(8)
    ]

    report = mx.export_scene(
        objects, [_material("ITU Concrete 3.5 GHz")], tmp_path, max_shapes=4,
    )

    assert report["grouped_by"] == "material"
    assert report["shapes"] == 1
    assert report["triangles"] == 8


def test_duplicate_corners_are_welded(tmp_path) -> None:
    """Converters emit each triangle with its own corners; unwelded, one object looks like many."""
    shared = [[0, 0, 0], [1, 0, 0], [1, 1, 0]]
    objects = [
        SimpleNamespace(
            name="strip",
            faces=[_face(shared), _face([[1, 1, 0], [1, 0, 0], [2, 1, 0]])],
        ),
    ]

    mx.export_scene(objects, [_material("ITU Concrete 3.5 GHz")], tmp_path)

    ply = (tmp_path / "meshes").glob("*.ply")
    header = next(ply).read_text().splitlines()
    vertex_line = next(line for line in header if line.startswith("element vertex"))
    assert int(vertex_line.split()[-1]) == 4


def test_an_empty_scene_is_rejected(tmp_path) -> None:
    """Writing a scene with no geometry would fail later, inside the ray tracer."""
    with pytest.raises(ValueError, match="no faces"):
        mx.export_scene([SimpleNamespace(name="empty", faces=[])], [_material("x")], tmp_path)
