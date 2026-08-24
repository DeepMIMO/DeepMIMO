"""Infinigen -> Sionna RT -> DeepMIMO pipeline runner.

Takes a furnished Infinigen ``scene.blend`` through to a DeepMIMO v4 scenario:

  1. Export the scene to Mitsuba, decimating per object class (needs ``bpy``)
  2. Place a ceiling-mounted TX and a floor-plane RX grid from the scene bounds
  3. Run Sionna RT ray tracing
  4. Convert to DeepMIMO v4, losslessly

Steps 1 and 3 need different environments - Infinigen pins ``numpy<2`` and
Sionna 2.x needs ``numpy>=2`` - so they are separate subcommands that hand off
through the scene folder on disk.

Usage:
    # in the Infinigen env (bpy importable)
    uv run scripts/pipelines/infinigen_pipeline_runner.py generate outputs/apt_seed7 --seed 7
    uv run scripts/pipelines/infinigen_pipeline_runner.py export \\
        outputs/apt_seed7/scene.blend scenes/apt_seed7

    # in the ray-tracing env
    uv run scripts/pipelines/infinigen_pipeline_runner.py trace \\
        scenes/apt_seed7 --scenario indoor_infinigen_apt_3p5

On Apple Silicon there is no CUDA backend, so Dr.Jit needs LLVM:
    export DRJIT_LIBLLVM_PATH=/opt/homebrew/opt/llvm/lib/libLLVM.dylib
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# Ray-tracing configuration. Indoor scenes are small and enclosed, so more
# reflections matter more than they do outdoors while far fewer sample rays are
# needed to find them: measured against a 1e6 reference, 250k agreed to within
# 0.49 dB peak per receiver at 43% of the solver time.
RT_PARAMS: dict = {
    "carrier_freq": 3.5e9,
    "bandwidth": 10e6,
    "max_reflections": 4,
    "max_paths": 25,
    # Diffraction around furniture edges is a primary indoor mechanism; without
    # it a cluttered interior shows large false outages.
    "max_diffractions": 1,
    "edge_diffraction": True,
    "max_transmissions": 0,
    "ds_enable": False,
    "bs2bs": False,
    "los": True,
    "synthetic_array": True,
    "batch_size": 200,
    "use_builtin_scene": False,
    "builtin_scene_path": "",
    "create_scene_folder": False,
    "path_inspection_func": None,
    "scene_edit_func": None,
    "n_samples_per_src": 250_000,
    "max_paths_per_src": 1_000_000,
    "refraction": True,
    "cpu_offload": True,
    "rx_ori": None,
    "rx_vel": None,
    "tx_ori": None,
    "tx_vel": None,
    "obj_idx": None,
    "obj_pos": None,
    "obj_ori": None,
    "obj_vel": None,
    "pos_prec": 4,
}


def scene_bounds(scene_folder: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the bounding box of an exported scene from its PLY meshes.

    Args:
        scene_folder: Folder holding ``meshes/``.

    Returns:
        Tuple of (lower corner, upper corner).

    """
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for ply in (scene_folder / "meshes").glob("*.ply"):
        lines = ply.read_text().splitlines()
        header_end = lines.index("end_header")
        count = int(
            next(line for line in lines if line.startswith("element vertex")).split()[-1],
        )
        block = np.array(
            [[float(v) for v in lines[header_end + 1 + i].split()] for i in range(count)],
        )
        if len(block):
            lo = np.minimum(lo, block.min(axis=0))
            hi = np.maximum(hi, block.max(axis=0))
    return lo, hi


def parse_positions(text: str | None) -> np.ndarray | None:
    """Parse ``"x,y,z; x,y,z"`` into an array of positions.

    Args:
        text: Semicolon-separated coordinate triples, or None.

    Returns:
        Array of shape (N, 3), or None if nothing was given.

    """
    if not text or not text.strip():
        return None
    rows = [
        [float(v) for v in chunk.split(",")]
        for chunk in text.split(";")
        if chunk.strip()
    ]
    return np.array(rows, dtype=float)


def parse_bounds(text: str | None) -> tuple[float, float, float, float] | None:
    """Parse ``"xmin,ymin,xmax,ymax"`` into a receiver-grid footprint.

    Args:
        text: Four comma-separated numbers, or None.

    Returns:
        Tuple of bounds, or None if nothing was given.

    """
    if not text or not text.strip():
        return None
    values = [float(v) for v in text.split(",")]
    expected = 4
    if len(values) != expected:
        msg = f"expected 'xmin,ymin,xmax,ymax', got {text!r}"
        raise ValueError(msg)
    return values[0], values[1], values[2], values[3]


def place_devices(  # noqa: PLR0913 - placement needs its bounds and spacing
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    spacing: float,
    rx_height: float,
    n_tx: int,
    clearance: float,
    margin: float,
    tx_positions: np.ndarray | None = None,
    rx_bounds: tuple[float, float, float, float] | None = None,
    tx_height: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Lay a receiver grid over the floor and hang transmitters near the ceiling.

    Both can be overridden: pass explicit transmitter coordinates, or a
    footprint for the receiver grid, when the automatic placement from the
    scene's bounding box is not what you want.

    Args:
        lo: Lower corner of the scene.
        hi: Upper corner of the scene.
        spacing: Receiver grid spacing in metres.
        rx_height: Receiver height above the floor.
        n_tx: Number of transmitters.
        clearance: Distance below the ceiling for the transmitters.
        margin: Inset from the bounding box, so receivers do not sit in a wall.
        tx_positions: Explicit transmitter coordinates, or None to place them
            automatically below the ceiling.
        rx_bounds: (xmin, ymin, xmax, ymax) for the receiver grid, or None to
            cover the scene footprint.
        tx_height: Absolute transmitter height, or None to hang them
            ``clearance`` below the ceiling.

    Returns:
        Tuple of (transmitter positions, receiver positions).

    """
    if rx_bounds is not None:
        x0, y0, x1, y1 = rx_bounds
    else:
        x0, y0 = lo[0] + margin, lo[1] + margin
        x1, y1 = hi[0] - margin, hi[1] - margin
    xs = np.arange(x0, x1 + 1e-9, spacing)
    ys = np.arange(y0, y1 + 1e-9, spacing)
    grid_x, grid_y = np.meshgrid(xs, ys)
    rx = np.stack(
        [grid_x.ravel(), grid_y.ravel(), np.full(grid_x.size, lo[2] + rx_height)],
        axis=-1,
    )

    if tx_positions is not None and len(tx_positions):
        return tx_positions, rx

    # `clearance` is a drop from the ceiling; `tx_height` is an absolute height.
    # The latter is what a person means by "put the AP at 2.6 m".
    tx_z = tx_height if tx_height is not None else hi[2] - clearance
    tx_x = np.linspace(lo[0], hi[0], n_tx + 2)[1:-1]
    tx_y = (lo[1] + hi[1]) / 2
    tx = np.array([[float(x), float(tx_y), float(tx_z)] for x in tx_x])
    return tx, rx


#: Building presets. Infinigen ships one constraint script and it describes a
#: dwelling, so what can be varied is the building's *shape*, not its room
#: vocabulary: storeys, storey height and footprint. See infinigen_launcher for
#: why room kinds cannot be swapped out, and INFINIGEN.md for what a genuine
#: office or warehouse would take.
SCENE_TYPES: dict[str, dict] = {
    "home": {},
    "tall_space": {"wall_height": 5.5},
    "multistorey": {"stories": 2},
    "tall_multistorey": {"stories": 2, "wall_height": 4.0},
    "compact": {"aspect": (0.95, 1.0)},
    "elongated": {"aspect": (0.45, 0.65)},
}


def solved_rooms(output: Path) -> list[str]:
    """Read back which kinds of room a generated scene actually contains.

    Adding a room kind to the mix only makes it available: whether the solver
    places one depends on the seed. The solver writes its final state beside the
    blend, and the room keys in it are named ``kind_index/storey``.

    Args:
        output: Generation output folder.

    Returns:
        Sorted room kinds, empty if the state file is missing or unreadable.

    """
    state = output / "solve_state.json"
    if not state.exists():
        return []
    try:
        objs = json.loads(state.read_text(encoding="utf-8")).get("objs", {})
    except (OSError, ValueError):
        return []
    kinds = {name.split("/")[0].rsplit("_", 1)[0] for name in objs if "/" in name}
    return sorted(kinds)


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate a furnished Infinigen scene.

    Infinigen's gin configs use repo-relative includes, so this runs from the
    package directory rather than the working directory.

    The knobs interact in ways worth knowing. ``addition_weight_scalar``
    (default 1.0) raises how eagerly the solver adds objects, so it is the lever
    for "more furniture" rather than "more detail on the same furniture".
    ``fast_solve.gin`` cuts the steps spent per solver stage. ``singleroom.gin``
    cuts the number of stages, and dominates runtime: without it a furnished
    dwelling needed 66 stages and hours, with it the same scene takes minutes
    and still comes out multi-room.

    Args:
        args: Parsed command-line arguments.

    """
    import infinigen_examples  # noqa: PLC0415

    package_dir = Path(infinigen_examples.__file__).parent.parent
    overrides = [
        "compose_indoors.terrain_enabled=False",
        f"solve_objects.addition_weight_scalar={args.furniture}",
    ]
    if args.room_type:
        overrides.append(f'restrict_solving.restrict_parent_rooms=["{args.room_type}"]')

    preset = SCENE_TYPES[args.scene_type]
    added = tuple(args.add_room) or preset.get("add", ())
    if added:
        # Gin cannot write enum members, so the room mix goes through the
        # launcher's configurable rather than straight into RoomConstants.
        names = ",".join(f'"{room}"' for room in added)
        overrides += [
            "RoomConstants.room_type=@deepmimo_room_types()",
            f"deepmimo_room_types.add=[{names}]",
        ]
    stories = args.stories or preset.get("stories")
    height = args.wall_height or preset.get("wall_height")
    aspect = preset.get("aspect")
    if stories:
        overrides.append(f"RoomConstants.n_stories={stories}")
    if height:
        overrides.append(f"RoomConstants.global_params.wall_height={height}")
    if aspect:
        overrides.append(f"RoomConstants.aspect_ratio_range=({aspect[0]},{aspect[1]})")

    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "infinigen_launcher.py"),
        "--seed",
        str(args.seed),
        "--task",
        "coarse",
        "--output_folder",
        str(Path(args.output).resolve()),
        "-p",
        *overrides,
    ]
    # Config presets. singleroom.gin is the big lever on runtime: it cuts the
    # solver's greedy stage count sharply, which is what makes a scene take
    # minutes rather than hours. It still yields a multi-room dwelling.
    presets = []
    if args.fast:
        presets.append("fast_solve.gin")
    if args.single_room:
        presets.append("singleroom.gin")
    if presets:
        command[-len(overrides) - 1 : -len(overrides) - 1] = ["-g", *presets]

    required = tuple(args.require_room) or preset.get("require", ())
    output = Path(args.output)
    for attempt in range(max(1, args.max_attempts)):
        seed = args.seed + attempt
        command[command.index("--seed") + 1] = str(seed)
        print(" ".join(command))
        subprocess.run(command, cwd=package_dir, check=True)  # noqa: S603

        rooms = solved_rooms(output)
        print(f"  rooms: {', '.join(rooms) or 'unknown'}")
        missing = [kind for kind in required if kind not in rooms]
        if not missing:
            break
        # The kind was available to the solver and it chose not to place one, so
        # the only lever left is a different draw.
        print(f"  seed {seed} produced no {', '.join(missing)}; re-rolling")
    else:
        print(
            f"  warning: no {', '.join(required)} after {args.max_attempts} seeds; "
            "keeping the last scene",
        )
    print(f"-> {output / 'scene.blend'}")


def cmd_export(args: argparse.Namespace) -> None:
    """Export a ``.blend`` to a Mitsuba scene folder.

    Args:
        args: Parsed command-line arguments.

    """
    from deepmimo.pipelines.infinigen_to_mitsuba import export_blend  # noqa: PLC0415

    summary = export_blend(
        args.blend,
        args.scene_folder,
        budgets={
            "architecture": args.architecture_budget,
            "furniture": args.furniture_budget,
            "ornament": args.ornament_budget,
        },
        min_size=args.min_size,
        open_doors=not args.keep_doors,
    )
    print(f"  classes: {summary['by_class']}")
    print(f"  -> {args.scene_folder}")


def _warn_on_frequency_change(scene_folder: Path, frequency: float) -> None:
    """Warn when a rebuilt scene is traced away from the frequency it was measured at.

    A scene rebuilt from a scenario carries whichever material constants that
    scenario stored. ITU materials are re-derived by Sionna at the new frequency,
    but a material kept as raw permittivity and conductivity is not - tracing it
    somewhere else silently reuses the old numbers.

    Args:
        scene_folder: Folder that may hold a ``source.json`` from the exporter.
        frequency: Carrier frequency the trace will use, in Hz.

    """
    source = scene_folder / "source.json"
    if not source.exists():
        return
    try:
        record = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return

    original = float(record.get("frequency") or 0.0)
    if not original or abs(original - frequency) < 1.0:
        return
    fixed = [
        name
        for name, treatment in (record.get("materials") or {}).items()
        if not str(treatment).startswith("itu:")
    ]
    print(
        f"note: {record.get('scenario', scene_folder.name)} was traced at "
        f"{original / 1e9:.3f} GHz, now tracing at {frequency / 1e9:.3f} GHz",
    )
    if fixed:
        print(
            f"      {len(fixed)} material(s) keep their original constants and are not "
            f"re-derived: {', '.join(sorted(fixed)[:6])}"
            + (" …" if len(fixed) > 6 else ""),  # noqa: PLR2004
        )


def cmd_from_scenario(args: argparse.Namespace) -> None:
    """Turn an existing DeepMIMO scenario back into a Mitsuba scene.

    Args:
        args: Parsed command-line arguments.

    """
    from deepmimo.exporters.mitsuba_exporter import export_scenario  # noqa: PLC0415

    report = export_scenario(
        args.scenario,
        args.scene_folder,
        group_by=args.group_by,
        material_mode=args.material_mode,
        max_shapes=args.max_shapes,
    )
    print(
        f"  {report['objects']} objects -> {report['shapes']} shapes "
        f"grouped by {report['grouped_by']}, {report['triangles']:,} triangles",
    )
    for name, treatment in report["materials"].items():
        print(f"    {name}: {treatment}")
    if report["frequency"]:
        print(f"  scenario was traced at {report['frequency'] / 1e9:.3f} GHz")
    print(f"  -> {report['scene_xml']}")


def cmd_trace(args: argparse.Namespace) -> None:
    """Trace an exported scene and convert it to a DeepMIMO scenario.

    Args:
        args: Parsed command-line arguments.

    """
    import deepmimo as dm  # noqa: PLC0415
    from deepmimo.pipelines.sionna_rt.sionna_raytracer import raytrace_sionna  # noqa: PLC0415

    rt_params = dict(RT_PARAMS)
    rt_params["carrier_freq"] = args.frequency
    rt_params["max_reflections"] = args.max_reflections
    rt_params["n_samples_per_src"] = args.samples
    rt_params["max_diffractions"] = 0 if args.no_diffraction else 1
    rt_params["edge_diffraction"] = not args.no_diffraction
    rt_params["ds_enable"] = args.diffuse

    scene_folder = Path(args.scene_folder)
    _warn_on_frequency_change(scene_folder, args.frequency)
    lo, hi = scene_bounds(scene_folder)
    print(f"scene bounds: {lo.round(2)} .. {hi.round(2)}")

    tx, rx = place_devices(
        lo,
        hi,
        spacing=args.spacing,
        rx_height=args.rx_height,
        n_tx=args.n_tx,
        clearance=args.clearance,
        margin=args.margin,
        tx_positions=parse_positions(args.tx_pos),
        rx_bounds=parse_bounds(args.rx_bounds),
        tx_height=args.tx_height,
    )
    print(f"{len(tx)} TX, {len(rx)} RX")

    start = time.time()
    rt_folder = raytrace_sionna(
        str(scene_folder),
        np.round(tx, rt_params["pos_prec"]),
        np.round(rx, rt_params["pos_prec"]),
        **rt_params,
    )
    print(f"ray tracing took {time.time() - start:.1f}s")

    # lossless=True is not optional indoors: the convex-hull scene export drops
    # flat walls as collinear, leaving a room with a floor, a ceiling and
    # nothing between them.
    scenario = dm.convert(
        rt_folder, scenario_name=args.scenario, overwrite=True, lossless=True,
    )
    print(f"-> scenario '{scenario}'")

    export = scene_folder / "export.json"
    if export.exists():
        summary = json.loads(export.read_text())
        print(f"  from {summary['polys_in']:,} polys -> {summary['polys_out']:,} triangles")


def main() -> None:
    """Parse arguments and dispatch."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="generate a furnished Infinigen scene")
    generate.add_argument("output")
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument(
        "--furniture",
        type=float,
        default=4.0,
        help="how eagerly the solver adds objects (Infinigen default 1.0)",
    )
    generate.add_argument(
        "--room-type",
        default=None,
        help="restrict to one room type, e.g. LivingRoom; omit for a whole dwelling",
    )
    generate.add_argument(
        "--fast",
        action="store_true",
        help="fast_solve preset: fewer solver steps per stage",
    )
    generate.add_argument(
        "--single-room",
        action="store_true",
        help="singleroom preset: far fewer solver stages, so minutes not hours",
    )
    generate.add_argument(
        "--scene-type",
        choices=sorted(SCENE_TYPES),
        default="home",
        help="building preset (default: Infinigen's dwelling mix)",
    )
    generate.add_argument(
        "--add-room",
        action="append",
        default=[],
        metavar="KIND",
        help=(
            "add a room kind to the mix, e.g. office. Repeatable. Note that "
            "Infinigen's constraints never require a non-home room, so the "
            "solver rarely places one - see INFINIGEN.md"
        ),
    )
    generate.add_argument(
        "--require-room",
        action="append",
        default=[],
        metavar="KIND",
        help=(
            "re-roll the seed until the scene contains this room kind. "
            "Repeatable; overrides the preset's own requirement"
        ),
    )
    generate.add_argument(
        "--max-attempts",
        type=int,
        default=4,
        help="how many seeds to try when a room kind is required",
    )
    generate.add_argument(
        "--stories",
        type=int,
        default=None,
        help="number of storeys; Infinigen samples one if not given",
    )
    generate.add_argument(
        "--wall-height",
        type=float,
        default=None,
        help="storey height in metres; Infinigen samples 2.8-3.2 if not given",
    )
    generate.set_defaults(func=cmd_generate)

    export = sub.add_parser("export", help="Infinigen .blend -> Mitsuba scene (needs bpy)")
    export.add_argument("blend")
    export.add_argument("scene_folder")
    export.add_argument("--architecture-budget", type=int, default=2500)
    export.add_argument("--furniture-budget", type=int, default=1500)
    export.add_argument("--ornament-budget", type=int, default=120)
    export.add_argument(
        "--min-size",
        type=float,
        default=0.10,
        help="drop objects smaller than this (metres); a wave cannot resolve them",
    )
    export.add_argument(
        "--keep-doors",
        action="store_true",
        help="keep door leaves shut instead of leaving the openings clear",
    )
    export.set_defaults(func=cmd_export)

    from_scenario = sub.add_parser(
        "from-scenario",
        help="DeepMIMO scenario -> Mitsuba scene, so it can be traced again",
    )
    from_scenario.add_argument("scenario", help="name of an existing DeepMIMO scenario")
    from_scenario.add_argument("scene_folder", help="folder to write scene.xml into")
    from_scenario.add_argument(
        "--material-mode",
        choices=("auto", "exact"),
        default="auto",
        help=(
            "auto re-derives ITU materials at the new frequency; "
            "exact keeps the constants stored in the scenario"
        ),
    )
    from_scenario.add_argument("--group-by", choices=("object", "material"), default="object")
    from_scenario.add_argument(
        "--max-shapes",
        type=int,
        default=600,
        help="above this many shapes, group by material instead of by object",
    )
    from_scenario.set_defaults(func=cmd_from_scenario)

    trace = sub.add_parser("trace", help="Mitsuba scene -> DeepMIMO scenario (needs Sionna)")
    trace.add_argument("scene_folder")
    trace.add_argument("--scenario", required=True)
    trace.add_argument("--spacing", type=float, default=0.25)
    trace.add_argument("--rx-height", type=float, default=1.2)
    trace.add_argument("--n-tx", type=int, default=2)
    trace.add_argument("--clearance", type=float, default=0.3)
    trace.add_argument("--margin", type=float, default=0.3)
    trace.add_argument(
        "--tx-pos",
        default=None,
        help='explicit transmitters as "x,y,z; x,y,z"; overrides --n-tx',
    )
    trace.add_argument(
        "--rx-bounds",
        default=None,
        help='receiver grid footprint as "xmin,ymin,xmax,ymax"',
    )
    trace.add_argument(
        "--tx-height",
        type=float,
        default=None,
        help="absolute transmitter height; overrides --clearance",
    )
    trace.add_argument("--frequency", type=float, default=3.5e9)
    trace.add_argument("--max-reflections", type=int, default=4)
    trace.add_argument("--samples", type=int, default=250_000)
    trace.add_argument(
        "--no-diffraction",
        action="store_true",
        help="disable diffraction; expect large false outages in cluttered scenes",
    )
    trace.add_argument(
        "--diffuse",
        action="store_true",
        help="enable diffuse reflection (slower)",
    )
    trace.set_defaults(func=cmd_trace)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
