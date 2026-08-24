# Infinigen → DeepMIMO v4 indoor scenarios

Generate a furnished interior, ray trace it, and get a DeepMIMO v4 scenario.

```bash
GEN=../.venv-infinigen/bin/python      # Blender, Infinigen
RT=../.venv-rt/bin/python              # Sionna, Mitsuba, DeepMIMO
R=scripts/pipelines/infinigen_pipeline_runner.py

$GEN $R generate  /tmp/scene --scene-type office --single-room --fast
$GEN $R export    /tmp/scene/scene.blend /tmp/mitsuba
$RT  $R trace     /tmp/mitsuba --scenario my_office_3p5 --frequency 3.5e9
```

`scripts/scenario_dashboard/` puts the same four stages behind a browser UI,
with a progress bar and a 3D view of the result.

## Why two environments

Infinigen pins `numpy<2`; Sionna needs `numpy>=2`. They cannot share an
interpreter, so the stages hand over through files on disk: a `.blend`, then a
Mitsuba `scene.xml`, then a scenario folder. Nothing is passed in memory, which
also means any stage can be re-run on its own.

## The four stages

| stage | needs | produces |
|---|---|---|
| `generate` | Blender + Infinigen | `scene.blend`, a furnished building |
| `export` | Blender | `scene.xml` + `meshes/`, budgeted for ray tracing |
| `from-scenario` | DeepMIMO | `scene.xml` rebuilt from a scenario you already have |
| `trace` | Sionna + DeepMIMO | a DeepMIMO v4 scenario |

`from-scenario` is an alternative entry point, not a step: it takes a converted
scenario and writes the Mitsuba scene back out, so an existing scenario can be
re-traced at another frequency without any of its original source files.

## Budgets

Infinigen's output is far too detailed for ray tracing — millions of triangles,
most of them below any wavelength of interest. `export` classifies each object
and decimates it to a per-class budget:

| class | default | why |
|---|---|---|
| architecture | 2500 | walls, floors, doorframes: these set the propagation |
| furniture | 1500 | tables, shelves, appliances: real blockers, coarse shapes |
| ornament | 120 | bottles, books, trinkets: present, but not resolved |

Objects below `--min-size` (0.10 m) are dropped outright. One apartment went
from 8.5M triangles to 61k with every blocker intact.

Classification is ordered and matches compound words first, because substring
rules quietly do the wrong thing: `CeilingLightFactory` is not a ceiling,
`SimpleBookcaseFactory` is not a book, and `NatureShelfTrinketsFactory` is not
shelving — that last one alone kept 3.6M triangles.

## Buildings other than dwellings

Infinigen ships one constraint script and it describes a home. Room kinds can be
**added** to its mix, never swapped out: the constraints require every home room
to exist — a living room must connect to a bedroom, a kitchen, a bathroom — so
removing one leaves the floorplan with no solution, and the solver responds by
retrying forever rather than failing.

Measured on one seed:

| mix | outcome |
|---|---|
| replaced with an office-only set | 23,832 graph attempts, 10 min CPU, no convergence |
| dwelling mix **plus** `office` | solved on the first attempt |

So `--scene-type office` adds office rooms rather than removing home ones.
Presets: `home`, `office`, `tall_space`, `multistorey`, `office_multistorey`;
underneath are `--add-room KIND`, `--stories N`, `--wall-height M`.

Adding a kind only makes it *available* — whether the solver places one depends
on the seed. Every run prints the room kinds it actually built, read back from
the solver's own state file, and `--require-room KIND` re-rolls the seed until
the kind appears.

Supporting a genuine office or warehouse *building* — one with no bedrooms at
all — means writing a constraint script for it, alongside Infinigen's
`infinigen_examples/constraints/home.py`. That is the next step, not a tweak.

## Runtime

Measured on an M-series laptop, one furnished dwelling:

| setting | time |
|---|---|
| `--single-room --fast` | ~5 min |
| full solve | hours — one run sat on a nine-hour trajectory |

`--single-room` is the dominant lever, not `--fast`: it cuts the solver's greedy
stage count rather than the steps per stage, and the result is still multi-room.
It also turns off open wall cuts, so rooms connect through doorways only.

## Placement and tracing

Transmitters and receivers default to the scene bounds and can be pinned:

```bash
$RT $R trace /tmp/mitsuba --scenario my_office_3p5 \
    --tx-pos '4.3,9.0,2.6; 8.2,9.0,2.6' \
    --rx-bounds '2,2,10,16' --rx-height 1.5 --spacing 0.3
```

Keep diffraction on. A cluttered interior reports large false outages without
it — one warehouse went from 87% served to 53%.
