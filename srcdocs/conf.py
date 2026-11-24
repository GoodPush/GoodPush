"""Sphinx configuration. Build (from the repository root):

    python -m sphinx -b html -d build/doctrees srcdocs docs

docs/ is what GitHub Pages serves (Settings > Pages > Deploy from a branch > main, /docs). The guide's pages are
Markdown (MyST), the model's reStructuredText; CONTRIBUTING.md and COSTS.md are included from the repository root.
"""
import goodpush

project = "GoodPush"
author = "Leon Foks"
copyright = "2026, Leon Foks"
release = goodpush.__version__

REPO = "https://github.com/GoodPush/GoodPush"

extensions = ["myst_parser", "sphinx.ext.autodoc", "sphinx.ext.mathjax", "sphinx.ext.viewcode",
              "sphinx.ext.githubpages", "sphinxcontrib.bibtex"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
myst_enable_extensions = ["colon_fence"]
myst_heading_anchors = 3              # [text](#a-heading) links, as on GitHub
autodoc_member_order = "bysource"
autodoc_default_options = {"members": True, "undoc-members": False}
autodoc_typehints = "description"
autodoc_preserve_defaults = True       # defaults as written (xp=np, not the module repr with its install path)
exclude_patterns = ["_build"]
bibtex_bibfiles = ["references.bib"]   # cited as :cite:p:`key` (rst) / {cite:p}`key` (MyST)
bibtex_default_style = "plain"
bibtex_reference_style = "author_year"

html_theme = "furo"
html_title = f"GoodPush {release}"
html_baseurl = "https://goodpush.github.io/GoodPush/"
html_show_sourcelink = False
html_copy_source = False
html_domain_indices = False          # no module index (and its icons)
html_theme_options = {
    "announcement": f'Beta: expect bugs. <a href="{REPO}/issues/new/choose">Tell us what goes wrong</a>.',
    "source_repository": REPO,           # "Edit this page" on every page
    "source_branch": "main",
    "source_directory": "srcdocs/",
}
