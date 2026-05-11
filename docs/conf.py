"""Sphinx configuration for spora-bench."""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "spora-bench"
copyright = "2026, AIMM"
author = "AIMM"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

try:
    import sphinx_autodoc_typehints  # noqa: F401
except ImportError:
    pass
else:
    extensions.append("sphinx_autodoc_typehints")

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstrings = True
napoleon_numpy_docstrings = True

autodoc_mock_imports = [
    # common heavy ML/data deps used by models and benchmarks
    "torch",
    "torchvision",
    "safetensors",
    "virtues",
    "kronos",
    "astir",
    "skimage",
    "sklearn",
    "einops",
    "loguru",
    "omegaconf",
    "tqdm",
    "cuML",
    "PIL",
    "pandas",
    "numpy",
]

# Also mock heavy I/O or local packages that cause C-extension import errors
for _m in ("spora_io", "zarr", "anndata", "h5py"):
    if _m not in autodoc_mock_imports:
        autodoc_mock_imports.append(_m)
