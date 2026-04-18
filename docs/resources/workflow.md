# Documentation Workflow

## Local preview

Start the live preview server from the repository root:

```bash
mkdocs serve
```

The site rebuilds automatically when you edit Markdown or notebook files.

---

## Notebook pre-rendering (required before publishing)

Tutorials and application notebooks are kept as lightweight `.py` source files
(Jupytext percent format).  Before the docs site can show **cell outputs**,
each notebook must be converted to a `.ipynb` file with stored outputs.
The `mkdocs build` step itself does **not** execute any code (`execute: false`);
it only renders whatever is already stored in the `.ipynb` files.

### Step 1 — run the pre-render script

```bash
# Pre-render everything (requires all optional deps + DeepMIMO scenario data)
uv run python scripts/pre_render_notebooks.py

# Pre-render only Sionna app notebooks (requires deepmimo[sionna])
uv run python scripts/pre_render_notebooks.py --sionna

# Pre-render only core tutorial notebooks (requires downloaded scenario data)
uv run python scripts/pre_render_notebooks.py --tutorials

# Pre-render a single notebook
uv run python scripts/pre_render_notebooks.py docs/applications/4_osm_pipeline.py

# Convert to stub .ipynb without executing (shows code only, no outputs)
uv run python scripts/pre_render_notebooks.py --no-execute
```

The script:

1. Converts each `.py` (Jupytext percent format) to a fresh `.ipynb` via
   `jupytext`.
2. Executes the `.ipynb` with `jupyter nbconvert --execute --inplace`,
   storing all outputs inside the file.

### Step 2 — commit the .ipynb output alongside the .py source

```bash
git add docs/tutorials/*.ipynb docs/applications/*.ipynb
git commit -m "Pre-render notebooks for docs"
```

Both files should be committed:

| File | Purpose |
|------|---------|
| `*.py` | Source of truth — edit this file |
| `*.ipynb` | Rendered output — commit after pre-rendering |

### Step 3 — build the docs

```bash
mkdocs build   # or: mkdocs serve
```

With `execute: false` in `mkdocs.yml`, the build uses the stored `.ipynb`
outputs without re-running any code.  The build completes in seconds.

---

## Which notebooks need which environment?

| Group | Command flag | Extra requirements |
|-------|-------------|-------------------|
| Tutorials 1–8 | `--tutorials` | DeepMIMO scenario files in `deepmimo_scenarios/` |
| Sionna apps 2–4 | `--sionna` | `pip install 'deepmimo[sionna]'` (PyTorch, Sionna RT, Mitsuba) |
| App 1 (channel prediction) | *(skipped)* | External ML dataset — maintained on Colab |
| Complete manual | *(skipped)* | Maintained on Colab; not pre-rendered locally |

---

## Tutorial tests

Tutorial notebooks double as integration tests.  They are excluded from the
default `pytest` run to keep CI fast.

| Command | Description |
|---------|-------------|
| `uv run pytest` | All tests **except** tutorials (fast, default) |
| `uv run pytest tests/tutorials/` | All tutorial tests (~6 min) |
| `uv run pytest tests/tutorials/test_1_getting_started.py` | One tutorial test |
| `uv run pytest -m tutorial` | Same as above via marker |
| `uv run python docs/tutorials/1_getting_started.py` | Run tutorial directly |

---

## Publishing

1. Pre-render notebooks (see above).
2. Run `mkdocs build` to confirm the site builds cleanly.
3. Commit `.ipynb` outputs and any Markdown changes.
4. Push to `main` — GitHub Actions publishes to GitHub Pages automatically.
