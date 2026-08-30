"""Plot theming.

Two selected palettes -- not one flipped into the other.  Both were checked with
the data-viz validator for lightness band, chroma floor, colour-vision-deficiency
separation on all pairs, normal-vision separation and contrast against their own
surface.  Only three categorical slots are used, which is the documented cap for
scatter-style charts where every pair of colours can end up adjacent.

Colour carries one job here: *beam status* -- in the beam, just outside it, or
elsewhere on the sky.  Magnitude (flux) is carried by marker size, and the object
class by marker shape, so nothing depends on hue alone.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

# matplotlib is deliberately NOT imported at module scope: the CLI imports
# THEMES to validate --theme, and that must keep working on a machine with no
# plotting stack so the CSVs still get written.  The import lives inside
# styled(), the only thing here that actually needs it.


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str          # figure background
    panel: str            # axes background
    ink: str              # primary text
    ink_secondary: str
    ink_muted: str        # de-emphasised marks and context sources
    grid: str
    #: categorical slots 1-3, in fixed order.  Slot 1 = the beam itself,
    #: slot 2 = near-miss sources, slot 3 = in-beam sources.
    series: tuple[str, str, str]
    #: single-hue ramp, light -> dark, for magnitude (time, beam response)
    sequential: tuple[str, ...]
    #: the full eight-slot categorical order, for LINE and BAR charts only.
    #: Validated on the adjacent pairlist; scatter-type charts must stay
    #: inside ``series`` (the first three), which validate on all pairs.
    categorical: tuple[str, ...]
    good: str
    warning: str
    critical: str

    @property
    def beam(self) -> str:
        return self.series[0]

    @property
    def near(self) -> str:
        return self.series[1]

    @property
    def inside(self) -> str:
        return self.series[2]


DARK = Theme(
    name="dark",
    surface="#16181c",
    panel="#1a1c21",
    ink="#ffffff",
    ink_secondary="#c3c2b7",
    ink_muted="#6f7178",
    grid="#2b2e34",
    series=("#3987e5", "#d95926", "#199e70"),
    sequential=("#184f95", "#256abf", "#3987e5", "#5598e7",
                "#86b6ef", "#b7d3f6"),
    categorical=("#3987e5", "#d95926", "#199e70", "#c98500",
                 "#d55181", "#008300", "#9085e9", "#e66767"),
    good="#0ca30c",
    warning="#fab219",
    critical="#d03b3b",
)

LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    panel="#ffffff",
    ink="#0b0b0b",
    ink_secondary="#52514e",
    ink_muted="#9a9892",
    grid="#e6e5e1",
    series=("#2a78d6", "#eb6834", "#1baf7a"),
    sequential=("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                "#256abf", "#104281"),
    categorical=("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                 "#e87ba4", "#008300", "#4a3aa7", "#e34948"),
    good="#0ca30c",
    warning="#fab219",
    critical="#d03b3b",
)

THEMES = {"dark": DARK, "light": LIGHT}


@contextmanager
def styled(theme: Theme):
    """rcParams context: thin marks, hairline recessive grid, roomy padding."""
    import matplotlib as mpl

    rc = {
        "figure.facecolor": theme.surface,
        "savefig.facecolor": theme.surface,
        "axes.facecolor": theme.panel,
        "axes.edgecolor": theme.grid,
        "axes.labelcolor": theme.ink_secondary,
        "axes.titlecolor": theme.ink,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.titlepad": 12,
        "axes.labelsize": 9,
        "axes.labelpad": 6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": theme.grid,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",          # solid hairline: never dashed
        "grid.alpha": 1.0,
        "xtick.color": theme.ink_muted,
        "ytick.color": theme.ink_muted,
        "xtick.labelcolor": theme.ink_secondary,
        "ytick.labelcolor": theme.ink_secondary,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "text.color": theme.ink,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "legend.labelcolor": theme.ink_secondary,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "font.size": 9,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial",
                            "DejaVu Sans"],
        "figure.dpi": 130,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "figure.constrained_layout.use": False,
    }
    with mpl.rc_context(rc):
        yield theme
