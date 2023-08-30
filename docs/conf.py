import sys
import os


sys.path.insert(0, os.path.abspath(".."))


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # Optional, for parsing Google-style docstrings
]


# Include both class and module level docstrings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

source_suffix = ".rst"

# The master toctree document.
master_doc = "index"


# Project information
project = "TS-DiffuGen"
author = "Sacha Raffaud"
version = "2023"
release = "2023"


source_suffix = ".rst"
exclude_patterns = ["_build"]


# Options for LaTeX output
latex_elements = {"extraclassoptions": "openany,oneside"}


# Options for manual page output
man_pages = [
    (master_doc, "TS-DiffuGen", "TS-DiffuGen Documentation", [author], 1)
]  # noqa

# Options for Texinfo output
texinfo_documents = [
    (
        master_doc,
        "TS-DiffuGen",
        "TS-DiffuGen Documentation",
        author,
        "TS-DiffuGen",
        "Transition State geometry optimisation",
        "Miscellaneous",
    ),
]
