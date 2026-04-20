"""Pre-render Jupyter notebooks for documentation.

Converts percent-format .py notebooks to executed .ipynb files so that
`mkdocs build` can display stored outputs without re-executing notebooks
at build time (which would require Sionna, downloaded datasets, etc.).

Workflow
--------
1. Edit / update the .py source notebook.
2. Run this script to produce the corresponding .ipynb with fresh outputs.
3. Commit **both** the .py source and the .ipynb output.
4. `mkdocs build` (with ``execute: false``) picks up the stored outputs.

Usage
-----
    # Pre-render everything (requires all optional deps + data)
    uv run python scripts/pre_render_notebooks.py

    # Pre-render only Sionna application notebooks
    uv run python scripts/pre_render_notebooks.py --sionna

    # Pre-render only the core tutorials (requires DeepMIMO scenario data)
    uv run python scripts/pre_render_notebooks.py --tutorials

    # Pre-render a single notebook
    uv run python scripts/pre_render_notebooks.py docs/applications/4_osm_pipeline.py

Dependencies
------------
The script needs ``jupytext`` and ``jupyter`` (``nbconvert``) in the
active environment::

    uv sync --extra dev
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Notebook groups
# ---------------------------------------------------------------------------

TUTORIALS = [
    ROOT / "docs/tutorials/1_getting_started.py",
    ROOT / "docs/tutorials/2_visualization.py",
    ROOT / "docs/tutorials/3_channel_generation.py",
    ROOT / "docs/tutorials/4_dataset_manipulation.py",
    ROOT / "docs/tutorials/5_doppler_mobility.py",
    ROOT / "docs/tutorials/6_beamforming.py",
    ROOT / "docs/tutorials/7_converters.py",
    ROOT / "docs/tutorials/8_migration_guide.py",
]

# Application notebooks that only need scenario data (no Sionna)
APPS = [
    ROOT / "docs/applications/1_channel_prediction.py",
]

# Sionna RT application notebooks (require `pip install deepmimo[sionna]`)
SIONNA_APPS = [
    ROOT / "docs/applications/2_sionna_rt_downstream.py",
    ROOT / "docs/applications/3_sionna_upstream.py",
    ROOT / "docs/applications/4_osm_pipeline.py",
    ROOT / "docs/applications/5_dynamic_rt.py",
]

# Notebooks intentionally skipped:
#   tutorials/manual.py — too large; maintained separately on Colab
SKIP: set[Path] = {
    ROOT / "docs/tutorials/manual.py",
}

EXECUTION_TIMEOUT = 600  # seconds per notebook


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, capture_output=False)


def convert_py_to_ipynb(py_path: Path) -> Path:
    """Convert a Jupytext percent-format .py file to a .ipynb notebook stub.

    The resulting .ipynb has all code cells but **no outputs** yet.
    Call :func:`execute_ipynb` afterwards to add outputs.
    """
    ipynb_path = py_path.with_suffix(".ipynb")
    _run(["jupytext", "--to", "notebook", str(py_path), "-o", str(ipynb_path), "--quiet"])
    return ipynb_path


def execute_ipynb(ipynb_path: Path, *, timeout: int = EXECUTION_TIMEOUT) -> None:
    """Execute a .ipynb in-place, storing outputs in the file."""
    _run([
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        f"--ExecutePreprocessor.timeout={timeout}",
        str(ipynb_path),
    ])


def pre_render(py_path: Path, *, execute: bool = True) -> None:
    """Convert .py → .ipynb and optionally execute it."""
    py_path = py_path.resolve()
    if py_path in SKIP:
        print(f"  SKIP  {py_path.relative_to(ROOT)}")
        return

    print(f"  → converting  {py_path.relative_to(ROOT)}")
    ipynb_path = convert_py_to_ipynb(py_path)

    if execute:
        print(f"  → executing   {ipynb_path.relative_to(ROOT)}")
        execute_ipynb(ipynb_path)
        print(f"  ✓ done        {ipynb_path.relative_to(ROOT)}")
    else:
        print(f"  ✓ stub only   {ipynb_path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "notebooks",
        nargs="*",
        metavar="NOTEBOOK",
        help="Specific .py notebook paths to pre-render (default: all groups selected below)",
    )
    parser.add_argument("--tutorials", action="store_true", help="Pre-render tutorial notebooks 1–8")
    parser.add_argument("--apps", action="store_true", help="Pre-render application notebooks (no Sionna)")
    parser.add_argument("--sionna", action="store_true", help="Pre-render Sionna application notebooks")
    parser.add_argument(
        "--no-execute",
        action="store_true",
        help="Convert to .ipynb stub without executing (no outputs)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.notebooks:
        notebooks = [Path(nb) for nb in args.notebooks]
    elif args.tutorials or args.apps or args.sionna:
        notebooks = []
        if args.tutorials:
            notebooks += TUTORIALS
        if args.apps:
            notebooks += APPS
        if args.sionna:
            notebooks += SIONNA_APPS
    else:
        # Default: all groups
        notebooks = TUTORIALS + APPS + SIONNA_APPS

    execute = not args.no_execute
    print(f"Pre-rendering {len(notebooks)} notebook(s) (execute={execute})…\n")
    for nb in notebooks:
        pre_render(nb, execute=execute)
    print("\nDone. Commit the generated .ipynb files alongside the .py sources.")


if __name__ == "__main__":
    main()
