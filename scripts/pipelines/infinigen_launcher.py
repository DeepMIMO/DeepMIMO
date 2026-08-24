"""Run Infinigen's indoor generator with a room mix gin can express.

Infinigen decides which kinds of room a building contains from
``RoomConstants.room_type``, a set of :class:`infinigen.core.tags.Semantics`
members. Gin can only write literals, not enum members, so a plain ``-p``
override cannot reach it — which is why every scene comes out a dwelling.

This launcher registers one configurable that builds that set, then hands over
to ``infinigen_examples.generate_indoors`` untouched. Every other argument
passes straight through.

    python infinigen_launcher.py --seed 1 --task coarse -p \
        'RoomConstants.room_type=@deepmimo_room_types()' \
        'deepmimo_room_types.add=["office"]'

Room kinds can only be *added*. Infinigen ships one constraint script, for
dwellings, and it demands that every home room exist: a living room must connect
to a bedroom, a kitchen, a bathroom and so on. Drop one and the floorplan has no
solution — measured at 23,832 graph attempts and ten minutes of CPU with no
convergence, and no error, because the solver simply keeps retrying. Adding a
kind leaves every existing constraint satisfiable, so it solves as fast as a
plain dwelling.
"""

import runpy
import sys

import gin
from infinigen.core import tags as t

#: Infinigen's own dwelling mix, from ``RoomConstants.home_room_types``. It is
#: repeated rather than read because reading it means constructing
#: RoomConstants, which samples the scene's random parameters as a side effect.
HOME = (
    "kitchen",
    "bedroom",
    "living-room",
    "closet",
    "hallway",
    "bathroom",
    "garage",
    "balcony",
    "dining-room",
    "utility",
    "staircase-room",
)

#: Room kinds Infinigen's shipped constraints know how to furnish. Anything else
#: is laid out and then left empty, which for ray tracing is a bare box bought at
#: the price of a full solve.
FURNISHABLE = frozenset({*HOME, "office"})


@gin.configurable("deepmimo_room_types")
def deepmimo_room_types(add: tuple[str, ...] = ()) -> set:
    """Build the room-kind set for a scene: the dwelling mix plus additions.

    Args:
        add: Extra room kinds, e.g. ``["office"]``.

    Returns:
        The matching :class:`Semantics` members.

    Raises:
        ValueError: If a name is not a room kind Infinigen can furnish.

    """
    unknown = [name for name in add if name not in FURNISHABLE]
    if unknown:
        msg = f"unfurnishable room kinds {unknown}; choose from {sorted(FURNISHABLE)}"
        raise ValueError(msg)
    return {t.Semantics(name) for name in (*HOME, *add)}


def main() -> None:
    """Hand over to Infinigen's generator with the configurable registered."""
    runpy.run_module("infinigen_examples.generate_indoors", run_name="__main__")


if __name__ == "__main__":
    sys.exit(main())
