# Documentation Workflow

## Overview

The documentation site is built with MkDocs and uses stored notebook outputs.
Notebooks are kept as lightweight `.py` source files (Jupytext percent format) and
must be converted to executed `.ipynb` files before the site can display outputs.
`mkdocs build` / `mkdocs serve` never re-executes code — it only renders what is
already stored inside the `.ipynb` files.

---

## Prerequisites by notebook group

| Group | Notebooks | Extra requirements |
|-------|-----------|-------------------|
| **Tutorials 1–8** | `tutorials/1_getting_started` … `8_migration_guide` | DeepMIMO scenario files in `deepmimo_scenarios/` — download any scenario with `dm.load()` or via the [database API](../api/database.md) |
| **Sionna apps 2–4** | `applications/2_sionna_rt_downstream` … `4_osm_pipeline` | `uv sync --extra sionna` (installs PyTorch, Sionna RT, Mitsuba) |
| **App 1 — Channel Prediction** | `applications/1_channel_prediction` | External ML dataset — maintained on Colab; rendered as a code stub locally |

!!! info "OSM Pipeline (App 4)"
    The OSM pipeline notebook queries the public Overpass API to fetch
    OpenStreetMap building data.  The API can be slow or temporarily unavailable.
    If execution fails with a 504 timeout, convert it as a stub
    (`--no-execute`) and run it manually when the API is responsive.

---

## Step-by-step workflow

### Step 1 — Pre-render notebooks

```bash
# Tutorials only (requires downloaded scenario data)
uv run python scripts/pre_render_notebooks.py --tutorials

# Sionna application notebooks (requires deepmimo[sionna])
uv run python scripts/pre_render_notebooks.py --sionna

# Both at once
uv run python scripts/pre_render_notebooks.py

# Code-only stubs — no outputs, no extra deps required
uv run python scripts/pre_render_notebooks.py --no-execute

# Single notebook
uv run python scripts/pre_render_notebooks.py docs/tutorials/3_channel_generation.py
```

### Step 2 — Serve or build

```bash
# Live-reloading dev server at http://127.0.0.1:8000
uv run mkdocs serve

# One-shot static build into site/
uv run mkdocs build
```

That's it.  No Makefile, no separate build step.

---

## Timing reference

Measured on an NVIDIA workstation (CPU execution, single machine).
GPU-accelerated environments will be faster for Sionna notebooks.

### Stub conversion — `--no-execute` (all 11 notebooks)

Converts every `.py` source to a bare `.ipynb` with code cells but no outputs.
No execution or extra dependencies required.

| Step | Time |
|------|------|
| All 11 notebooks → stubs | **~13 s** |

### Tutorial execution — `--tutorials` (8 notebooks, needs scenario data)

| Notebook | Approximate time |
|----------|-----------------|
| 1 — Getting Started | ~30 s |
| 2 — Visualization | ~45 s |
| 3 — Channel Generation | ~60 s |
| 4 — Dataset Manipulation | ~45 s |
| 5 — Doppler & Mobility | ~55 s |
| 6 — Beamforming | ~65 s |
| 7 — Converters | ~55 s |
| 8 — Migration Guide | ~30 s |
| **Total (all 8)** | **~7 min** |

### Sionna app execution — `--sionna` (apps 2 & 3, GPU optional)

| Notebook | Approximate time |
|----------|-----------------|
| 2 — Sionna RT → DeepMIMO | ~25 s |
| 3 — DeepMIMO → Sionna | ~28 s |
| 4 — OSM Pipeline | network-dependent (Overpass API) |
| **Total (apps 2–3)** | **~55 s** |

### Docs build

| Step | Time |
|------|------|
| `mkdocs build` (all pages, stored outputs) | **~17 s** |
| `mkdocs serve` first load | **~17 s**, then incremental |

---

## Notebook source files

| File | Purpose |
|------|---------|
| `docs/tutorials/*.py` | Source of truth — edit this file |
| `docs/tutorials/*.ipynb` | Rendered output — gitignored; regenerate with pre-render script |
| `docs/applications/*.py` | Source of truth |
| `docs/applications/*.ipynb` | Rendered output — gitignored |

!!! warning "`.ipynb` files are gitignored"
    The generated `.ipynb` files are excluded from version control
    (`docs/**/*.ipynb` in `.gitignore`).  Every contributor who wants to
    build docs with outputs must run the pre-render script locally.

---

## Which notebooks skip execution?

Two notebooks are intentionally excluded from the pre-render script:

| Notebook | Reason |
|----------|--------|
| `tutorials/manual.py` | Too large; maintained separately on Colab |
| `applications/1_channel_prediction.py` | Requires an external ML dataset download; rendered as a code stub |

To build with all outputs visible, run `--tutorials` and `--sionna`, then
create a stub for the channel-prediction notebook:

```bash
uv run jupytext --to notebook \
    docs/applications/1_channel_prediction.py \
    -o docs/applications/1_channel_prediction.ipynb \
    --quiet
```
