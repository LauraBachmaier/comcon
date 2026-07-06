"""Sphinx configuration for the ComCon API reference.

Build with:  sphinx-build -b html docs/api docs/api/_build/html
"""
import os
import sys

# Point at the repository root so ``import comcon`` resolves.
sys.path.insert(0, os.path.abspath('../..'))

project = 'ComCon'
author = 'Laura Bachmaier'
release = '1.0.0'
version = '1.0.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',   # parses the NumPy-style docstrings
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
]

autosummary_generate = True
napoleon_numpy_docstring = True
napoleon_google_docstring = False

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}
# Importing comcon prints a welcome banner and builds ipywidgets; it is harmless for
# autodoc (it does not block) but produces noise in the build log, not in the output.

html_theme = 'sphinx_rtd_theme'
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
