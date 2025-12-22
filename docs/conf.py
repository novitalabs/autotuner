# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add src directory to path for autodoc
sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -----------------------------------------------------

project = 'LLM Autotuner'
copyright = '2025, LLM Autotuner Team'
author = 'LLM Autotuner Team'
release = '1.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_autodoc_typehints',
    'sphinx_copybutton',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '.venv', '**/.venv']

# -- MyST Parser options -----------------------------------------------------

myst_enable_extensions = [
    'colon_fence',
    'deflist',
    'tasklist',
]
myst_heading_anchors = 3

# Support both .rst and .md files
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- Options for HTML output -------------------------------------------------

html_theme = 'furo'
html_static_path = ['_static']
html_logo = '_static/logo.svg'
html_favicon = '_static/logo.svg'

html_theme_options = {
    'light_css_variables': {
        'color-brand-primary': '#3ECF8E',
        'color-brand-content': '#3ECF8E',
    },
    'dark_css_variables': {
        'color-brand-primary': '#5edba3',
        'color-brand-content': '#5edba3',
    },
    'sidebar_hide_name': False,
    'navigation_with_keys': True,
}

# -- Autodoc configuration ---------------------------------------------------

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
}

autodoc_typehints = 'description'
autodoc_class_signature = 'separated'

# -- Napoleon configuration --------------------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

# -- Intersphinx configuration -----------------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'fastapi': ('https://fastapi.tiangolo.com/', None),
}
