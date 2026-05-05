"""Sphinx configuration for CompOpt documentation."""

import os
import sys

# -- Path setup ----------------------------------------------------------------
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -------------------------------------------------------
project = "CompOpt"
copyright = "2026, CompOpt Contributors & Oak Ridge National Laboratory"
author = "CompOpt Contributors"
release = "0.1.0"
version = "0.1"

# -- General configuration -----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_design",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Source file suffixes
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Options for autodoc -------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__, __call__",
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"

# -- Napoleon (Google/NumPy docstrings) ----------------------------------------
napoleon_google_docstrings = True
napoleon_numpy_docstrings = True
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

# -- Intersphinx ---------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "gymnasium": ("https://gymnasium.farama.org/", None),
}

# -- HTML output ---------------------------------------------------------------
html_theme = "furo"
html_title = "CompOpt"
html_static_path = ["_static"]

html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/compopt-benchmark/compopt",
            "html": '<svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>',
            "class": "",
        },
    ],
}

# Suppress missing logo warnings (logos are optional)
html_logo = None

# -- MyST Markdown config ------------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "tasklist",
]
myst_heading_anchors = 3

# -- Copy button config --------------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# -- Suppress harmless warnings ------------------------------------------------
# Duplicate object descriptions arise from dataclass attributes documented
# via both autosummary and automodule.
suppress_warnings = ["autodoc.duplicate_object", "ref.duplicate"]

# Deduplicate: don't re-document attrs imported into __init__
autosummary_imported_members = False
autodoc_class_content = "class"

# -- Custom CSS ----------------------------------------------------------------
html_css_files = ["custom.css"]
