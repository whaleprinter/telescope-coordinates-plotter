"""Usage examples for projections.py. Run directly:

    python3 example.py

Writes every image into output/. No sys.path setup needed -- Python puts a
script's own directory on sys.path automatically, and projections.py lives
right next to this file.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from projections import plot_points

OUTPUT_DIR = Path(__file__).parent / "output"


def example_scatter() -> None:
    """The simple case: a short list of named coordinates -> a labeled plot."""
    names = ["Cas A", "Cyg A", "Crab Nebula", "Andromeda Gal.", "Virgo A",
            "Orion Nebula", "Sgr A*", "Vega", "Sirius"]
    ra  = [350.850, 299.868,  83.633,  10.685, 187.706,  83.822, 266.417, 279.234, 101.287]
    dec = [ 58.815,  40.734,  22.015,  41.269,  12.391,  -5.391, -29.008,  38.784, -16.716]


    out = OUTPUT_DIR / "example_scatter_arc.png"
    plot_points(ra, dec, "arc", labels=names, path=out,
               title="Nine well-known sources, arc projection")
    print(f"wrote {out}")


def example_trace() -> None:
    """The real use case: a dense, ordered trace read as a continuous line.

    This mimics a fixed, zenith-pointing antenna's drift scan: the sky
    sweeps past at the sidereal rate (about 15.041 deg/hour in right
    ascension) while declination stays essentially constant. Over 24 hours
    the trace crosses the 0/360 deg wrap exactly once -- kind="line" has to
    get that right, or it draws a spurious line straight across the plot at
    the wrap. 20000 points is deliberately more than you'd need (roughly one
    every 4 seconds for a full day), to make the point that this is fast at
    a size well beyond any realistic drift-scan sampling.
    """
    n = 20_000
    hours = np.linspace(0, 24, n)
    ra = (200.0 + hours * 15.041) % 360.0                    # sidereal drift, wraps once
    dec = np.full_like(ra, 36.0) + 0.05 * np.sin(hours / 24 * 2 * np.pi)  # tiny realistic wobble

    t0 = time.perf_counter()
    out = OUTPUT_DIR / "example_trace_mollweide.png"
    plot_points(ra, dec, "mol", kind="line", path=out,
               title=f"{n:,}-point drift-scan trace, 24 h, Mollweide")
    dt = time.perf_counter() - t0
    print(f"wrote {out}  ({n:,} points projected and plotted in {dt*1000:.0f} ms)")

    # The more useful view for "what does my fixed beam actually sweep
    # through": zenithal equidistant, valid over the whole sky, centred on
    # the beam's own mean position rather than an arbitrary (0, 0).
    out = OUTPUT_DIR / "example_trace_arc.png"
    plot_points(ra, dec, "arc", center_lon=200.0, center_lat=36.0, kind="line",
               path=out, title=f"Same trace, ARC projection centred on the beam")
    print(f"wrote {out}")

    # Before/after: what you get WITHOUT the wrap-break logic. Passing an
    # absurdly high break_factor effectively disables it, using the exact
    # same code path, so this is a fair side-by-side rather than a separate
    # hand-rolled "wrong" version. The jump is real here too (checked
    # numerically in test_projections.py), but a real drift scan's
    # declination barely changes, so the wrong connecting segment lands
    # almost exactly on top of the real curve and is hard to *see*. A path
    # whose declination actually varies makes the same bug obvious instead:
    lon_wobble = np.linspace(0, 360, 4000)
    lat_wobble = 30.0 * np.sin(np.radians(lon_wobble) * 3)
    out = OUTPUT_DIR / "example_wrap_bug_WITH_fix.png"
    plot_points(lon_wobble, lat_wobble, "mol", kind="line", path=out,
               title="A wandering path crossing the wrap -- break-detection ON")
    print(f"wrote {out}")

    out = OUTPUT_DIR / "example_wrap_bug_WITHOUT_fix.png"
    plot_points(lon_wobble, lat_wobble, "mol", kind="line", break_factor=1e12,
               path=out, title="Same path, break-detection OFF -- the "
                              "spurious line across the middle is the bug")
    print(f"wrote {out}  (compare to the previous file)")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    example_scatter()
    example_trace()
