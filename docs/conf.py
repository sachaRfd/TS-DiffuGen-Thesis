import sys


sys.path.insert(0, "..")


# Project information
project = "TS-DiffuGen"
author = "Sacha Raffaud"
version = "2023"
release = "2023"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
]

source_suffix = ".rst"
master_doc = "index"
exclude_patterns = ["_build"]
autoclass_content = "both"
