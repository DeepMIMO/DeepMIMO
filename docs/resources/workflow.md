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
3. Download the notebook as `manual_full.ipynb`.
4. Copy the table of contents cells from the tracked `manual.ipynb` so internal Markdown links remain intact.
5. (Legacy) When using Sphinx, strip metadata/output via:
   ```bash
   jupyter nbconvert manual_full.ipynb --ClearMetadataPreprocessor.enabled=True \
       --clear-output --to notebook --output manual2.ipynb
   ```
6. Replace the committed `docs/manual.ipynb` with the cleaned notebook and re-run the docs build.

## Publishing changes

1. Run the site build locally:
   - `mkdocs build` (new) or `sphinx-build -b html . _build/html` (legacy).
2. Commit updated Markdown, assets, and notebooks.
3. Push to the `mkdocs` branch and let CI/GitHub Pages deploy the new site.
