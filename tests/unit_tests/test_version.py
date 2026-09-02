"""The project version is declared once; everything else must agree with it.

``src/spectral_galerkin_heat/__init__.py`` holds the canonical literal.
``pyproject.toml`` picks it up via ``[tool.setuptools.dynamic]`` and the docs read
it back from the installed metadata, so those two cannot drift. ``CITATION.cff``
is a static file that no build step touches, so it is pinned here instead: bump
``__version__`` without bumping the citation and this test fails.
"""

import tomllib
from importlib.metadata import version as installed_version
from pathlib import Path

import yaml

import spectral_galerkin_heat

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_citation_cff_matches_package_version():
    citation = yaml.safe_load((REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert str(citation["version"]) == spectral_galerkin_heat.__version__


def test_pyproject_declares_the_version_dynamically():
    # Guards against someone re-adding a static `version = "..."` alongside the
    # dynamic declaration, which would silently reintroduce a second source.
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    assert "version" in project.get("dynamic", [])
    assert "version" not in project
    assert (
        pyproject["tool"]["setuptools"]["dynamic"]["version"]["attr"]
        == "spectral_galerkin_heat.__version__"
    )


def test_installed_metadata_matches_the_literal():
    assert installed_version("spectral_galerkin_heat") == spectral_galerkin_heat.__version__
