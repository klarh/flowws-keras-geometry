
# -- Path setup --------------------------------------------------------------

import sys
from unittest.mock import MagicMock
import os

with open('../../flowws_keras_geometry/version.py') as version_file:
    exec(version_file.read())
sys.path.insert(0, os.path.abspath('../..'))

class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

    @staticmethod
    def __setitem__(key, value):
        pass

autodoc_mock_imports = [
    'freud',
    'keras_gtar',
    'pyriodic',
]

sys.modules.update((mod_name, Mock()) for mod_name in autodoc_mock_imports)

# -- Project information -----------------------------------------------------

project = 'flowws-keras-geometry'
copyright = '2021, Matthew Spellings'
author = 'Matthew Spellings'
version = __version__
release = version

# -- General configuration ---------------------------------------------------

extensions = [
    'm2r2',
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
