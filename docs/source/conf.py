# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


import quac
import tomli

with open("../../pyproject.toml", "rb") as fh:
    project = tomli.load(fh)["project"]

author_list = ", ".join([author["name"] for author in project["authors"]])

project = "quac"
author = author_list
version = quac.__version__
release = quac.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "jupyter_sphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.githubpages",
    "sphinx.ext.mathjax",
    "sphinx_rtd_theme",
    "sphinx_togglebutton",
    "sphinxcontrib.jquery",
    "sphinx.ext.intersphinx",
    "sphinxcontrib.email",
    "myst_parser",
    "sphinx_design",
]


myst_enable_extensions = ["colon_fence"]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_logo = "assets/quac.png"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_show_sourcelink = False

togglebutton_hint = ""
togglebutton_hint_hide = ""

pygments_style = "arduino"

# Napoleon settings
napoleon_google_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# Email
email_automode = True
