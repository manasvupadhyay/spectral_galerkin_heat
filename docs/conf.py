# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'fastHeatSolv'
copyright = '2026, Theo Andrieux'
author = 'Theo Andrieux'

version = '0.1.0'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'myst_parser',
    'sphinxcontrib.mermaid',
    'autoapi.extension',
]

autoapi_dirs = ['../src/fast_heat_solv']
autoapi_type = 'python'
autoapi_options = [
    'members',
    'show-inheritance',
    'show-module-summary',
]
autoapi_ignore = ['*__pycache__*']
autoapi_root = 'api'

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
