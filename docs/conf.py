import sys
import os

sys.path.insert(0, os.path.abspath(os.sep.join((os.curdir, "../docs"))))  # noqa

# Project information
project = "TS-DiffuGen"
author = "Sacha Raffaud"
version = "1.0"
release = "1.0.0"

master_doc = "index"

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

source_suffix = ".rst"
exclude_patterns = ["_build"]


# Options for LaTeX output
latex_elements = {"extraclassoptions": "openany,oneside"}

# Options for manual page output
man_pages = [(master_doc, "myproject", "My Project Documentation", [author], 1)]  # noqa

# Options for Texinfo output
texinfo_documents = [
    (
        master_doc,
        "MyProject",
        "My Project Documentation",
        author,
        "MyProject",
        "One line description of project.",
        "Miscellaneous",
    ),
]
