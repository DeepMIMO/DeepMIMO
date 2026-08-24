"""Tests for the Infinigen to Mitsuba scene export rules.

Only the classification and material rules are covered; the extraction itself
needs Blender, which is not a test dependency.
"""

import pytest

from deepmimo.pipelines import infinigen_to_mitsuba as itm


@pytest.mark.parametrize(
    ("shader", "expected"),
    [
        ("shader_window_glass", itm.MAT_GLASS),
        ("shader_brushed_black_metal", itm.MAT_METAL),
        ("shader_mirror", itm.MAT_METAL),
        ("shader_plaster", itm.MAT_PLASTERBOARD),
        ("shader_hardwood_floor", itm.MAT_WOOD),
        ("shader_marble_shader_rectangle_tile_tile", itm.MAT_CONCRETE),
        ("shader_unheard_of", itm.DEFAULT_MATERIAL),
        (None, itm.DEFAULT_MATERIAL),
    ],
)
def test_shader_maps_to_itu_material(shader, expected) -> None:
    """Infinigen names shaders after appearance, which proxies composition."""
    assert itm.map_shader_to_material(shader) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        # Compound names must not match the general token they contain.
        ("CeilingLightFactory(1).spawn_asset(2)", itm.CLASS_ORNAMENT),
        ("SimpleBookcaseFactory(3)", itm.CLASS_FURNITURE),
        ("BookStackFactory(4)", itm.CLASS_ORNAMENT),
        ("NatureShelfTrinketsFactory(5)", itm.CLASS_ORNAMENT),
        # Straightforward cases.
        ("WindowFactory(6)", itm.CLASS_ARCHITECTURE),
        ("living-room_0/0", itm.CLASS_ARCHITECTURE),
        ("skirtingboard_support", itm.CLASS_ARCHITECTURE),
        ("BathtubFactory(7)", itm.CLASS_FURNITURE),
        ("LargePlantContainerFactory(8)", itm.CLASS_ORNAMENT),
        ("SomethingUnknownFactory(9)", itm.CLASS_FURNITURE),
    ],
)
def test_object_classification(name, expected) -> None:
    """Objects are classified by what they do to a wave."""
    assert itm.classify_object(name) == expected


def test_ornament_is_kept_not_dropped() -> None:
    """Ornament keeps a budget: a pot plant blocks even though a leaf does not."""
    assert itm.DEFAULT_BUDGETS[itm.CLASS_ORNAMENT] > 0
    assert (
        itm.DEFAULT_BUDGETS[itm.CLASS_ORNAMENT]
        < itm.DEFAULT_BUDGETS[itm.CLASS_FURNITURE]
        < itm.DEFAULT_BUDGETS[itm.CLASS_ARCHITECTURE]
    )


def test_factory_name_strips_instance_decoration() -> None:
    """Infinigen decorates names with the factory id and spawn index."""
    assert itm.factory_name("PanelDoorFactory(2243540).spawn_asset(35)") == "PanelDoorFactory"
    assert itm.factory_name("living-room_0/0.001") == "living-room_0/0"


def test_group_name_is_a_safe_shape_id() -> None:
    """Group names become Mitsuba shape ids, so they must be plain."""
    name = itm.group_name("PanelDoorFactory(12).spawn_asset(3)", itm.MAT_WOOD)

    assert name == "paneldoor_wood"
    assert name.replace("_", "").isalnum()


def test_materials_get_a_namespace_distinct_from_objects(tmp_path) -> None:
    """Sionna rejects a material whose name collides with a scene object.

    It strips the ``mesh-``/``mat-`` prefixes to derive both names, so a shape
    id and a material id that differ only by prefix would collide.
    """
    groups = {"window_metal": itm.MAT_METAL, "window_wood": itm.MAT_WOOD}

    xml = itm.write_mitsuba_xml(tmp_path, groups).read_text()

    for group in groups:
        assert f'id="mesh-{group}"' in xml
        assert f'id="mat-{group}"' not in xml
    # One BSDF instance per group, or Sionna merges the shapes into one object.
    assert xml.count('<bsdf type="itu-radio-material"') == len(groups)


def test_write_ply_roundtrips(tmp_path) -> None:
    """The PLY writer emits a header the loaders agree on."""
    path = tmp_path / "m.ply"

    itm.write_ply([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)], [(0, 1, 2)], path)
    lines = path.read_text().splitlines()

    assert lines[0] == "ply"
    assert "element vertex 3" in lines
    assert "element face 1" in lines
    assert lines[-1] == "3 0 1 2"


@pytest.mark.parametrize(
    "name",
    [
        "PanelDoorFactory(3912436).spawn_asset(0)",
        "GlassPanelDoorFactory(430087).spawn_asset(3)",
        "door",
        "door.001",
    ],
)
def test_door_leaves_are_recognised(name) -> None:
    """Door leaves are found so the openings can be left clear.

    Infinigen shuts every door, and a shut 0.18 m panel is a serious
    obstruction at 3.5 GHz - a scenario of sealed rooms would say more about
    the doors than the building.
    """
    assert itm.is_door_leaf(name)


@pytest.mark.parametrize(
    "name",
    ["WindowFactory(6)", "doorframe", "living-room_0/0", "SingleCabinetFactory(2)"],
)
def test_non_leaves_are_left_alone(name) -> None:
    """Frames, windows and furniture are not door leaves."""
    assert not itm.is_door_leaf(name)


@pytest.mark.parametrize(
    "name",
    [
        "LiteDoorFactory(8841965).spawn_asset(2)",
        "LouverDoorFactory(5646270).spawn_asset(3)",
        "PanelDoorFactory(1).spawn_asset(0)",
        "GlassPanelDoorFactory(2).spawn_asset(1)",
    ],
)
def test_every_door_factory_variant_is_a_leaf(name) -> None:
    """Infinigen ships many door factories, not just the panel ones."""
    assert itm.is_door_leaf(name)
