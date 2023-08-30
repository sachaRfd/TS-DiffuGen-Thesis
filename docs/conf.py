import sys
import os


sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("../apollo"))


extensions = ["sphinx.ext.autodoc"]
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"


# Project information
project = "TS-DiffuGen"
author = "Sacha Raffaud"
version = "2023"
release = "2023"


# General configuration
extensions = [
    "sphinx.ext.autodoc",
]

source_suffix = ".rst"
exclude_patterns = ["_build"]


# Options for LaTeX output
latex_elements = {"extraclassoptions": "openany,oneside"}

# latex_documents = [
#     (master_doc, ".tex", "Apollo Documentation", "Apollo", "manual"),
# ]

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
