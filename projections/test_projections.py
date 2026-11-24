"""Correctness checks for projections.py. Run directly:

    python3 test_projections.py

Numeric checks print PASS/FAIL for each projection; a plot is saved to
output/projections_check.png so the projections can be checked by eye too --
often the fastest way to catch a wrong one, since a bad projection usually
just looks obviously wrong.

No sys.path setup is needed: Python puts a script's own directory on
sys.path automatically, and projections.py lives right next to this file.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.coordinates import angular_separation
from astropy.wcs import WCS

from projections import PROJECTIONS, _draw_graticule, _wcs, deproject, project

OUTPUT_DIR = Path(__file__).parent / "output"


def _sep(lon1, lat1, lon2, lat2) -> np.ndarray:
    """Wrap-safe angular distance in degrees (never naive subtraction --
    e.g. longitude 359.999 and 0.001 are 0.002 deg apart, not 359.998)."""
    return np.degrees(angular_separation(*np.radians([lon1, lat1, lon2, lat2])))


def verify(plot_path: Path | str | None = None) -> bool:
    ok = True

    print("1. _wcs() actually builds an 8-character CTYPE with the separator "
         "dash (checking the real object, not a re-typed copy of the string):")
    for name, (code, _, _) in PROJECTIONS.items():
        w = _wcs(name, 0.0, 0.0, 1.0)
        s = w.wcs.ctype[0]
        good = len(s) == 8 and s == f"RA---{code}"
        ok &= good
        print(f"   {s}  len={len(s)}  {'OK' if good else 'WRONG'}")

    print("\n2. Prefix used for the CTYPE quantity code does not change the "
         "numeric result (proves it's safe to hard-code RA/DEC for any "
         "frame -- galactic, ecliptic, horizontal, whatever):")
    lons = np.array([0., 45., 90., 179., 271., 359.])
    lats = np.array([-80., -10., 0., 30., 60., 89.])
    for name, (code, needs_c, _) in PROJECTIONS.items():
        center = (10.0, 20.0) if needs_c else (0.0, 0.0)
        ref_x, ref_y = project(lons, lats, name, *center)
        same = True
        for prefix in ("GLON", "GLAT"), ("ELON", "ELAT"):
            w = WCS(naxis=2)
            w.wcs.ctype = [f"{prefix[0]}-{code}", f"{prefix[1]}-{code}"]
            w.wcs.crval, w.wcs.crpix, w.wcs.cdelt = list(center), [0, 0], [1, 1]
            pix = w.wcs_world2pix(np.column_stack([lons, lats]), 1)
            same &= np.array_equal(pix[:, 0], ref_x, equal_nan=True)
            same &= np.array_equal(pix[:, 1], ref_y, equal_nan=True)
        ok &= same
        print(f"   {name:4s}: identical across prefixes = {same}")

    print("\n3. Centre maps to exactly (0, 0), for every projection:")
    for name, (_, needs_c, _) in PROJECTIONS.items():
        center = (123.4, -17.5) if needs_c else (0.0, 0.0)
        x, y = project(*center, name, *center)
        good = abs(x[0]) < 1e-9 and abs(y[0]) < 1e-9
        ok &= good
        print(f"   {name:4s}: ({x[0]:.2e}, {y[0]:.2e})  {'OK' if good else 'WRONG'}")

    print("\n4. project() then deproject() recovers the original point:")
    for name, (_, needs_c, _) in PROJECTIONS.items():
        center = (30.0, 15.0) if needs_c else (0.0, 0.0)
        lon0 = np.array([35., 40., 25., 33.])
        lat0 = np.array([18., 12., 20., 10.])
        x, y = project(lon0, lat0, name, *center)
        lon1, lat1 = deproject(x, y, name, *center)
        err = _sep(lon0, lat0, lon1, lat1).max()
        good = err < 1e-9
        ok &= good
        print(f"   {name:4s}: max round-trip error = {err:.2e} deg  "
             f"{'OK' if good else 'WRONG'}")

    print("\n5. Domain at 90 deg from centre, the boundary case where tan and "
         "sin genuinely differ (radius ~ tan(sep) vs ~ sin(sep) -- tan(90) is "
         "infinite, sin(90) is a perfectly finite point, the visible limb):")
    domain_90deg_nan = {"tan"}          # everything else is finite at 90 deg
    for name, (_, needs_c, _) in PROJECTIONS.items():
        x, _ = project(0.0, 90.0, name, 0.0, 0.0)
        finite = bool(np.isfinite(x[0]))
        good = finite != (name in domain_90deg_nan)
        ok &= good
        print(f"   {name:4s}: finite at 90 deg = {finite}  "
             f"(expected {'NaN' if name in domain_90deg_nan else 'finite'})  "
             f"{'OK' if good else 'WRONG'}")

    print("\n5b. sin specifically goes NaN just PAST 90 deg (beyond the "
         "visible limb, where far- and near-hemisphere points would "
         "otherwise fold onto the same point):")
    x, _ = project(0.0, 90.5, "sin", 0.0, 0.0)
    good = not np.isfinite(x[0])
    ok &= good
    print(f"   sin at 90.5 deg: finite={np.isfinite(x[0])}  "
         f"{'OK' if good else 'WRONG'}")

    print("\n6. A trace crossing a whole-sky map's antimeridian (crval + 180 "
         "deg -- NOT the literal lon=0/360 boundary, which WCSLIB already "
         "handles smoothly when it's near crval, confirmed just below) gets "
         "broken, not drawn as a spurious line straight across the plot:")
    from projections import _break_jumps

    # Near crval itself (0 deg here), the literal 0/360 wrap is a non-event:
    # WCSLIB projects longitude relative to crval, so 359.5 and 0.5 are just
    # two nearby points, continuous in x. Confirm that directly, so the real
    # test below isn't mistaken for testing the same thing twice.
    lon_center = np.linspace(355.0, 365.0, 21) % 360.0
    x_center, _ = project(lon_center, np.full_like(lon_center, 40.0), "mol", 0.0, 0.0)
    smooth_at_crval = np.max(np.abs(np.diff(x_center))) < 1.0
    ok &= smooth_at_crval
    print(f"   at crval itself: max step = {np.max(np.abs(np.diff(x_center))):.3f}  "
         f"(smooth, as expected)  {'OK' if smooth_at_crval else 'WRONG'}")

    # The actual seam is 180 deg away from crval.
    lon_trace = np.linspace(175.0, 185.0, 200) % 360.0
    lat_trace = np.full_like(lon_trace, 40.0)
    xw, yw = project(lon_trace, lat_trace, "mol", 0.0, 0.0)
    xb, yb = _break_jumps(xw, yw, factor=8.0)
    n_breaks = int(np.isnan(xb).sum())
    good = n_breaks == 1 and xb.size == xw.size + n_breaks
    ok &= good
    print(f"   at the antimeridian: inserted {n_breaks} break(s) into a "
         f"{xw.size}-point trace  {'OK' if good else 'WRONG'}")

    print(f"\n{'ALL CHECKS PASS' if ok else 'SOME CHECKS FAILED'}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    out = Path(plot_path) if plot_path else OUTPUT_DIR / "projections_check.png"
    _plot_graticules(out)
    print(f"\nSaved a graticule plot to {out} -- open it and check each panel "
         f"looks like what that projection is supposed to look like.")
    return ok


def _plot_graticules(path: Path) -> None:
    """Plot the sky grid through a few projections, to check by eye."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    panels = [
        (axes[0], "mol", 0.0, 0.0, True, "Mollweide, whole sky"),
        (axes[1], "arc", 0.0, 90.0, True,
         "Zenithal equidistant, centred on the N pole"),
        (axes[2], "tan", 45.0, 30.0, False,
         "Tangent (gnomonic), narrow field around (45, 30)"),
    ]
    for ax, name, clon, clat, whole_sky, title in panels:
        _draw_graticule(ax, name, clon, clat, whole_sky, half_width=40.0)
        ax.set_title(f"{name}: {title}", fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlabel("x (deg)")
        ax.set_ylabel("y (deg)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(0 if verify() else 1)
