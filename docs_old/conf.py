"""Configuration file for the Sphinx documentation builder."""

import inspect
import os
import sys
import tomllib
from pathlib import Path
from typing import Any

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(".."))

# ----------------------------------------------------------------------------------------
# Project info
# ----------------------------------------------------------------------------------------
project = "DeepMIMO"
copyright = "2025, Wireless Intelligence Lab"
author = "João Morais, Umut Demirhan, Ahmed Alkhateeb"

# ----------------------------------------------------------------------------------------
# Version
# ----------------------------------------------------------------------------------------
try:
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        pyproject = tomllib.load(f)
    release = pyproject["project"]["version"]
except FileNotFoundError:
    release = "4.0.0aX"

# ----------------------------------------------------------------------------------------
# Theme selection
# ----------------------------------------------------------------------------------------
# Set this to 'furo' or 'sphinx_rtd_theme' to switch themes
THEME = "sphinx_rtd_theme"

# ----------------------------------------------------------------------------------------
# Extensions
# ----------------------------------------------------------------------------------------
extensions = [
    # Core Sphinx extensions
    "sphinx.ext.autodoc",  # Auto-doc from docstrings
    "sphinx.ext.napoleon",  # NumPy/Google docstring style
    "sphinx.ext.viewcode",  # Link to highlighted source code
    "sphinx.ext.githubpages",  # GitHub Pages .nojekyll
    "sphinx.ext.autosummary",  # Summary tables for modules/classes
    "sphinx.ext.intersphinx",  # Link to other projects' docs
    # 'sphinx.ext.mathjax',                  # Math support via MathJax
    "sphinx.ext.linkcode",  # Source links to GitHub
    # Third-party extensions
    "sphinx_autodoc_typehints",  # Move type hints into doc body
    "sphinx_copybutton",  # Adds copy buttons to code blocks
    "sphinx_design",  # Layout/design components (cards, tabs)
    "matplotlib.sphinxext.plot_directive",  # Matplotlib plot directive
    "sphinxext.opengraph",  # Social metadata (OpenGraph)
    # 'sphinx_remove_toctrees',              # Hide entries from the ToC
    # 'sphinx_plotly_directive',             # Plotly directive support
    # 'sphinxcontrib.bibtex',                # BibTeX citation support
    # MyST extensions - keep these last
    "myst_nb",  # Jupyter notebook support via MyST
]

# ----------------------------------------------------------------------------------------
# Source Suffix Configuration
# ----------------------------------------------------------------------------------------
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

# ----------------------------------------------------------------------------------------
# MyST Extensions
# ----------------------------------------------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "tasklist",
]

# Disable footnote transformation to avoid conflicts
myst_footnote_transition = False
suppress_warnings = ["myst.domains"]

# ----------------------------------------------------------------------------------------
# Templates / static
# ----------------------------------------------------------------------------------------
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "manual.ipynb"]
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
master_doc = "index"

# ----------------------------------------------------------------------------------------
# Theme config
# ----------------------------------------------------------------------------------------
html_theme = THEME
html_show_sidebar = True
html_show_sphinx = False
html_show_sourcelink = False
html_show_copyright = True

# Sidebar
if THEME == "sphinx_rtd_theme":
    html_sidebars = {
        "**": ["globaltoc.html", "relations.html", "sourcelink.html", "searchbox.html"],
    }

# Theme options
if THEME == "furo":
    # Furo theme configuration
    html_theme_options = {
        # Sidebar behavior
        "sidebar_hide_name": False,
        "navigation_with_keys": True,
        # Light mode styling
        "light_css_variables": {
            # Brand colors
            "color-brand-primary": "#0EA5E9",
            "color-brand-content": "#0284C7",
            # Sidebar styling
            "color-sidebar-background": "#F8F9FC",
            "color-sidebar-item-background--hover": "#F2F4F8",
            "color-sidebar-link": "#374151",
            "color-sidebar-link--hover": "#0EA5E9",
            # TOC styling
            "toc-title-font-size": "1rem",
            "toc-spacing-vertical": "1.5rem",
            "toc-font-size": "0.9rem",
            # Content styling
            "font-size--normal": "1rem",
            "font-size--small": "0.9rem",
            "content-padding": "3rem",
            "color-foreground-primary": "#111827",
            "color-foreground-secondary": "#374151",
            "color-foreground-muted": "#4B5563",
            # Admonition styling
            "color-admonition-background": "#F5F7FA",
        },
        # Dark mode styling
        "dark_css_variables": {
            "color-brand-primary": "#4A7DFF",
            "color-brand-content": "#185dbe",
            "color-sidebar-background": "#0F0F0F",
            "color-sidebar-item-background--hover": "#1A1A1A",
            "color-sidebar-link": "#E0E0E0",
            "color-sidebar-link--hover": "#6eafeb",
            "color-foreground-primary": "#FFFFFF",
            "color-foreground-secondary": "#E0E0E0",
            "color-foreground-muted": "#C0C0C0",
            "color-admonition-background": "#141414",
        },
    }

elif THEME == "sphinx_rtd_theme":
    # Read the Docs theme options
    html_theme_options = {
        "navigation_depth": 4,
        "titles_only": False,
        "logo_only": False,
        "style_external_links": True,
        "style_nav_header_background": "#2980B9",
        "collapse_navigation": False,
        "sticky_navigation": True,
        "includehidden": True,
        "prev_next_buttons_location": "both",
    }

# ----------------------------------------------------------------------------------------
# Autodoc config
# ----------------------------------------------------------------------------------------
autodoc_member_order = "bysource"
add_module_names = False
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
autosummary_generate = True

# ----------------------------------------------------------------------------------------
# Notebook execution
# ----------------------------------------------------------------------------------------
nb_execution_mode = "off"  # "auto"
nb_execution_timeout = 600
nb_merge_streams = True
nb_mime_priority_overrides = [("*", "text/html", 0)]

# ----------------------------------------------------------------------------------------
# Math config
# ----------------------------------------------------------------------------------------
# mathjax3_config = {
#     "loader": {"load": ["[tex]/boldsymbol"]},
#     "tex": {"packages": {"[+]": ["boldsymbol"]}},
# }
# numfig = True

# ----------------------------------------------------------------------------------------
# Intersphinx
# ----------------------------------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "plotly": ("https://plotly.com/python-api-reference", None),
}

# ----------------------------------------------------------------------------------------
# OpenGraph
# ----------------------------------------------------------------------------------------
ogp_site_url = "https://deepmimo.net/"
ogp_use_first_image = True

# ----------------------------------------------------------------------------------------
# BibTeX
# ----------------------------------------------------------------------------------------
# bibtex_bibfiles = ["references.bib"]  # currently not used


# ----------------------------------------------------------------------------------------
# GitHub source linking
# ----------------------------------------------------------------------------------------
def linkcode_resolve(domain: str, info: dict[str, Any]) -> str | None:
    if domain != "py" or not info["module"] or not info["fullname"]:
        return None
    try:
        module = sys.modules.get(info["module"])
        obj = eval(f"{info['module']}.{info['fullname']}", {}, {})
        obj = inspect.unwrap(obj)
        filepath = inspect.getsourcefile(obj)
        lineno = inspect.getsourcelines(obj)[1]
    except Exception:
        return None
    if not filepath:
        return None
    relpath = os.path.relpath(filepath, start=Path(__file__).parent.parent)
    return f"https://github.com/DeepMIMO/DeepMIMO/blob/main/{relpath}#L{lineno}"
