"""hydroline -- planning tools for a zenith-pointing 21 cm drift-scan telescope.

The package answers four questions for a fixed, upward-staring antenna:

  1. Where on the celestial sphere is the beam, as a function of time?
  2. What does that look like (static plots + animation)?
  3. Which catalogued objects drift through the beam?
  4. Which of those can this particular antenna actually detect or resolve?

Everything is driven by two configuration objects -- :class:`Site` and
:class:`Antenna` -- so changing the array location or the horn geometry
re-derives the whole analysis.
"""
from __future__ import annotations

__version__ = "1.0.0"


# ---------------------------------------------------------------------------
# Dependency preflight
# ---------------------------------------------------------------------------
# This runs before any third-party import below, so a wrong interpreter gets a
# sentence explaining what to do instead of a traceback from deep inside a
# submodule.  It is a common failure on machines with several Pythons: a shell
# whose PATH points at the system interpreter, which ships numpy but not pandas
# or astropy, so the import dies halfway through and blames the wrong thing.
_REQUIRED = {
    "numpy": "numpy",
    "pandas": "pandas",
    "astropy": "astropy",
}
#: Needed only for figures and YAML configs; absent is survivable.
_OPTIONAL = {
    "matplotlib": "matplotlib",
    "PIL": "Pillow",
    "yaml": "PyYAML",
}


def missing_dependencies(which: str = "required") -> list[str]:
    """Distribution names of the packages this interpreter cannot import."""
    import importlib.util

    table = _REQUIRED if which == "required" else _OPTIONAL
    out = []
    for module, package in table.items():
        try:
            found = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            out.append(package)
    return out


def _interpreter_with_dependencies() -> str | None:
    """Look for another Python on this machine that can already run hydroline."""
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    seen = {sys.executable}
    candidates: list[str] = []
    for name in ("python3", "python"):
        for found in (shutil.which(name),):
            if found and found not in seen:
                seen.add(found)
                candidates.append(found)
    for guess in ("/opt/anaconda3/bin/python3", "/opt/miniconda3/bin/python3",
                  "/usr/local/bin/python3", "/opt/homebrew/bin/python3",
                  str(Path.home() / "anaconda3/bin/python3"),
                  str(Path.home() / "miniconda3/bin/python3")):
        if guess not in seen and Path(guess).exists():
            seen.add(guess)
            candidates.append(guess)

    probe = "import numpy, pandas, astropy"
    for candidate in candidates[:8]:
        try:
            done = subprocess.run([candidate, "-c", probe], capture_output=True,
                                  timeout=20)
        except (OSError, subprocess.SubprocessError):
            continue
        if done.returncode == 0:
            return candidate
    return None


def _preflight() -> None:
    missing = missing_dependencies()
    if not missing:
        return

    import sys

    version = ".".join(str(n) for n in sys.version_info[:3])
    lines = [
        "",
        "hydroline cannot start: this Python is missing "
        + ", ".join(missing) + ".",
        "",
        f"  running:  {sys.executable}",
        f"            Python {version}",
        "",
    ]
    other = _interpreter_with_dependencies()
    if other:
        lines += [
            "Another interpreter on this machine already has everything:",
            "",
            f"  {other} run.py",
            "",
            "That usually means your shell's PATH is pointing at a different",
            "Python than you expect. Check with 'which python3'; a new terminal",
            "often fixes it.",
            "",
            "Or install the packages into the interpreter you are using now:",
        ]
    else:
        lines.append("Install them with:")
    lines += [
        "",
        f"  {sys.executable} -m pip install -r requirements.txt",
        "",
    ]
    raise ImportError("\n".join(lines))


_preflight()

from .config import Site, Antenna, Observation, Configuration, DUKE_PHYSICS
from .antenna import AntennaModel
from .pointing import zenith_track
from .catalog import load_catalog, solar_system_track
from .analysis import (
    beam_encounters,
    detectability_table,
    galactic_plane_crossings,
)

__all__ = [
    "Site",
    "Antenna",
    "Observation",
    "Configuration",
    "DUKE_PHYSICS",
    "AntennaModel",
    "zenith_track",
    "load_catalog",
    "solar_system_track",
    "beam_encounters",
    "detectability_table",
    "galactic_plane_crossings",
]
