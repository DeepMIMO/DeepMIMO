# Documentation Workflow

This page captures the legacy Sphinx workflow so contributors can reproduce the same steps with the new MkDocs toolchain.

## Local preview

1. Open a terminal in the repository root and start the live preview:
   - **MkDocs** (new flow): `mkdocs serve`.
   - **Sphinx** (legacy): `sphinx-build -b html . _build/html` inside `docs_old/`, then serve `_build/html` with `python -m http.server --directory _build/html`.
2. On Windows you can clear previous Sphinx builds with `rmdir /s /q _build`.

## Updating the Jupyter manual

1. Build the updated package and publish a test wheel so it can be installed on Google Colab.
2. Install the test package inside Colab and capture new examples in the notebook.
3. Download the notebook as `manual.ipynb`.
4. Copy the table of contents cells from the tracked `manual.ipynb` so internal Markdown links remain intact.
5. (Legacy) When using Sphinx, strip metadata/output via:
   ```bash
   jupyter nbconvert manual.ipynb --ClearMetadataPreprocessor.enabled=True \
       --clear-output --to notebook --output manual2.ipynb
   ```
6. Replace the committed `docs/manual.ipynb` with the cleaned notebook and re-run the docs build.

## Running Tutorial Tests

Tutorial tests execute each `.py` tutorial file in `docs/tutorials/` to ensure they run without errors. These tests are **excluded by default** from `pytest` runs to save time (tutorials take ~6 minutes).

| Command | Description |
|---------|-------------|
| `uv run pytest` | Run all tests **except** tutorials (default, fast) |
| `uv run pytest tests/tutorials/` | Run **all** tutorial tests (~6 min) |
| `uv run pytest -s tests/tutorials/` | Run **all** tutorial tests with output capture |
| `uv run pytest tests/tutorials/test_1_getting_started.py` | Run **one** specific tutorial test |
| `uv run pytest -m tutorial` | Run all tutorial tests (alternative method) |
| `uv run python docs/tutorials/1_getting_started.py` | Run tutorial directly (not as test) |

**Note**: Tutorials are excluded via `norecursedirs` in `pyproject.toml`, so running `pytest` alone will skip them for fast iteration. Explicitly specify `tests/tutorials/` to include them.

## Building docs with tutorial execution

By default, tutorials are rendered but not executed during docs build (`execute: false` in `mkdocs.yml`).

To build docs with tutorials executed:

1. Temporarily set `execute: true` in `mkdocs.yml`:
   ```yaml
   plugins:
     - mkdocs-jupyter:
         execute: true
         allow_errors: false
   ```
2. Run the build:
   ```bash
   mkdocs build
   ```
3. Revert the change before committing (unless this is for a release build).

**Note**: Executing tutorials during build will:
- Download datasets (requires internet)
- Take significantly longer (5-15 minutes vs. seconds)
- Fail the build if any tutorial has errors

## Publishing changes

1. Run the site build locally:
   - `mkdocs build` (new) or `sphinx-build -b html . _build/html` (legacy).
2. Commit updated Markdown, assets, and notebooks.
3. Push to the `mkdocs` branch and let CI/GitHub Pages deploy the new site.
