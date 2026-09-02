# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'spectral_galerkin_heat'
copyright = '2026 Laboratoire de Mécanique des Solides (LMS), École Polytechnique, CNRS UMR 7649, Institut Polytechnique de Paris, Route de Saclay, Palaiseau, 91128, France'
author = 'Théo Andrieux, Jules Dichamp, Manas V. Upadhyay'

# Version comes from the installed package metadata, which setuptools fills from
# spectral_galerkin_heat.__version__ -- never hardcode it here.
try:
    release = _pkg_version('spectral_galerkin_heat')
except PackageNotFoundError:  # docs built against an uninstalled source tree
    from spectral_galerkin_heat import __version__ as release
version = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'myst_parser',
    'sphinxcontrib.mermaid',
    'autoapi.extension',
    'sphinx_design',
]

autoapi_dirs = ['../src/spectral_galerkin_heat']
autoapi_type = 'python'
autoapi_options = [
    'members',
    'show-inheritance',
    'show-module-summary',
]
autoapi_ignore = ['*__pycache__*']
autoapi_root = 'api'

# Lets Markdown pages write {{ version }} instead of repeating the number.
myst_substitutions = {'version': release}

# MyST: enable LaTeX-style math ($...$ and $$...$$) and amsmath environments
myst_enable_extensions = [
    'dollarmath',
    'amsmath',
    'colon_fence',
    'substitution',   # required for the {{ version }} substitution above
]

# Silence autoapi's cyclic-import notices (io_utils <-> compute_L2_error).
suppress_warnings = ['autoapi']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Type hint rendering
autodoc_typehints = 'description'
autodoc_typehints_format = 'short'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

autodoc_mock_imports = ['cupy', 'cupyx']
napoleon_use_ivar = True
napoleon_include_init_method = False
