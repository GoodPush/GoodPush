"""Sphinx configuration. Build (from the repository root):

    python -m sphinx -b html -d build/doctrees srcdocs docs

docs/ is what GitHub Pages serves (Settings > Pages > Deploy from a branch > main, /docs). The guide's pages are
Markdown (MyST), the model's reStructuredText; CONTRIBUTING.md and COSTS.md are included from the repository root.
"""
import re
from pathlib import Path

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
# The logo: the app's logo.svg with its letters in a fixed colour, one for each theme (an <img> doesn't follow the
# page's theme switch). Written to build/logo/ and served from _static/, so docs/_static/ holds them for the README.
STATIC = Path(goodpush.__file__).parent / "app" / "static"
LOGO_DIR = Path(__file__).parents[1] / "build" / "logo"
LOGO_DIR.mkdir(parents=True, exist_ok=True)
LOGO = (STATIC / "logo.svg").read_text(encoding="utf-8")
for theme, colour in (("light", "#0b0b0b"), ("dark", "#f0efec")):
    svg = re.sub(r"<style>.*?</style>", f"<style>:root {{ color:{colour} }}</style>", LOGO, flags=re.S)
    (LOGO_DIR / f"logo-{theme}.svg").write_text(svg, encoding="utf-8")
html_static_path = [str(LOGO_DIR)]
html_favicon = str(STATIC / "mark.svg")
html_baseurl = "https://goodpush.github.io/GoodPush/"
html_show_sourcelink = False
html_copy_source = False
html_domain_indices = False          # no module index (and its icons)
html_theme_options = {
    "announcement": f'Beta: expect bugs. <a href="{REPO}/issues/new/choose">Tell us what goes wrong</a>.',
    "source_repository": REPO,           # "Edit this page" on every page
    "source_branch": "main",
    "source_directory": "srcdocs/",
    "light_logo": "logo-light.svg",
    "dark_logo": "logo-dark.svg",
    "sidebar_hide_name": True,           # the logo says it
}
