# Makefile for Sphinx documentation

# Build the PDF documentation
pdf:
	sphinx-build -b latex . _build/latex
	$(MAKE) -C _build/latex all-pdf

# Clean the build directory
clean:
	rm -rf _build