# Configuration file for the Sphinx documentation builder.

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "hydra_tod"
copyright = "2025-2026, Zheng Zhang"
author = "Zheng Zhang"
release = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "numpydoc",
    "myst_parser",
]

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

# Numpydoc settings
numpydoc_show_class_members = False

# Autodoc settings
autodoc_member_order = "bysource"
autodoc_mock_imports = [
    "mpi4py",
    "jax",
    "jaxlib",
    "numpyro",
    "torch",
    "emcee",
    "MomentEmu",
    "pygdsm",
    "healpy",
    "h5py",
    "psutil",
    "mpmath",
    "corner",
    "seaborn",
    "pdf2image",
    "astropy",
    "tqdm",
]

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# Source suffix
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Suppress numpydoc/autodoc warnings for numpy type annotations
# (e.g., np.bool_ is parsed as an RST hyperlink due to trailing underscore)
nitpicky = False
numpydoc_validation_checks = set()

# Suppress unknown reference target warnings from type annotations
# (e.g., NDArray[np.bool_] where trailing _ is parsed as RST hyperlink)
suppress_warnings = ["ref.ref", "docutils"]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Theme options
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
}
