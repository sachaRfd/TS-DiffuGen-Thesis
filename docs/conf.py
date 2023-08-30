import sys


sys.path.insert(0, "../src")


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

# # Options for LaTeX output
# latex_elements = {"extraclassoptions": "openany,oneside"}


# # Options for manual page output
# man_pages = [
#     (master_doc, "TS-DiffuGen", "TS-DiffuGen Documentation", [author], 1)
# ]  # noqa

# # Options for Texinfo output
# texinfo_documents = [
#     (
#         master_doc,
#         "TS-DiffuGen",
#         "TS-DiffuGen Documentation",
#         author,
#         "TS-DiffuGen",
#         "Transition State geometry optimisation",
#         "Miscellaneous",
#     ),
# ]
