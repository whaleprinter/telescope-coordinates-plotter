"""Project spherical sky coordinates onto a 2D plane, for plotting.

Three functions you'll actually call: project(), deproject(), and
plot_points(). Everything else is either the PROJECTIONS lookup table or a
small private helper. Correctness checks live in test_projections.py, next
to this file; a usage example lives in example.py.

The projection math itself is NOT computed here -- it's computed by
astropy.wcs, i.e. WCSLIB, Mark Calabretta's reference implementation of the
FITS-WCS projections standard (Calabretta & Greisen 2002). This file
contains no hand-derived trigonometry; its only job is constructing a WCS
object correctly and calling it. That's a deliberate choice: a projection
formula is exactly the kind of thing that's easy to get subtly wrong by
hand, and there is no reason to take that risk when the reference
implementation is already installed (astropy.wcs ships inside astropy,
already a dependency here).

Two non-obvious facts this code depends on, both confirmed by direct testing
(see test_projections.py, which re-checks them every time it runs):

1. FITS' CTYPE string is 8 characters: a 4-character coordinate-quantity
   code (dash-padded, e.g. "RA--") + a literal "-" + the 3-character
   projection code, e.g. "RA---TAN". The quantity code only labels the axes
   -- WCSLIB's actual projection arithmetic depends only on the projection
   code and the two numeric angles, confirmed by projecting the same
   numbers through "RA--"/"DEC-", "GLON"/"GLAT" and "ELON"/"ELAT" and
   getting bit-for-bit identical output for every projection below. So this
   file always uses "RA"/"DEC" as a generic placeholder -- it works
   regardless of what the two input angles actually represent (equatorial,
   galactic, horizontal, ecliptic -- doesn't matter).

2. With crpix=(0, 0) and origin=1 (FITS' 1-indexed pixel convention) in
   wcs_world2pix/wcs_pix2world, the projection centre maps to exactly
   (0.0, 0.0). Using origin=0 with the same crpix bakes in a spurious -1
   offset, because FITS crpix is defined in 1-indexed coordinates.
"""
from __future__ import annotations

import numpy as np
from astropy.wcs import WCS

#: name -> (3-letter FITS-WCS projection code, needs an explicit centre?, description)
PROJECTIONS = {
    "tan": ("TAN", True,  "Gnomonic: great circles -> straight lines. "
                          "Valid within 90 deg of centre, NaN beyond."),
    "sin": ("SIN", True,  "Orthographic: the sphere as seen from infinity. "
                          "Valid within 90 deg of centre, NaN beyond."),
    "arc": ("ARC", True,  "Zenithal equidistant: radius from centre = true "
                          "angular distance. Valid everywhere."),
    "stg": ("STG", True,  "Stereographic: conformal. Valid everywhere "
                          "except the exact antipode."),
    "zea": ("ZEA", True,  "Zenithal equal-area. Valid everywhere."),
    "mol": ("MOL", False, "Mollweide: whole-sky, equal-area, elliptical."),
    "ait": ("AIT", False, "Hammer-Aitoff: whole-sky, equal-area, elliptical."),
    "car": ("CAR", False, "Plate carree: x=lon, y=lat. Trivial baseline."),
}

#: Projections whose *usable domain* is a narrow patch around their centre
#: (radius <= 90 deg) rather than the whole sphere. This is NOT the same
#: grouping as "needs an explicit centre" -- arc/stg/zea also need a centre,
#: but (like mol/ait/car) are valid over the whole sky *around wherever
#: that centre is*, so they get the same whole-sky-style graticule and
#: unclipped view as the whole-sky projections, just centred differently.
#: Only tan (radius ~ tan(sep), blows up at 90 deg) and sin (folds beyond
#: 90 deg, see test_projections.py) are genuinely narrow-field.
_NARROW_FIELD = {"tan", "sin"}


def _wcs(projection: str, center_lon: float, center_lat: float,
        cdelt: float) -> WCS:
    code = PROJECTIONS[projection][0]
    w = WCS(naxis=2)
    w.wcs.ctype = [f"RA---{code}", f"DEC--{code}"]  # see module docstring, fact 1
    w.wcs.crval = [center_lon, center_lat]
    w.wcs.crpix = [0.0, 0.0]                        # see module docstring, fact 2
    w.wcs.cdelt = [cdelt, cdelt]
    return w


def project(lon, lat, projection: str, center_lon: float, center_lat: float,
           cdelt: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Project native spherical (lon, lat), in degrees, onto a plane.

    (center_lon, center_lat) -> exactly (0.0, 0.0). lon/lat can be a single
    number or an array of any size -- either way this makes exactly one call
    into WCSLIB, so it is just as efficient for a hundred thousand points
    (an hour of a dense telescope trace, easily) as for one. Points outside
    the projection's valid domain (see PROJECTIONS) come back NaN rather
    than raising, since a plot naturally just leaves them out.
    """
    if projection not in PROJECTIONS:
        raise ValueError(f"unknown projection {projection!r}; choose from "
                         f"{sorted(PROJECTIONS)}")
    lon = np.atleast_1d(np.asarray(lon, dtype=float))
    lat = np.atleast_1d(np.asarray(lat, dtype=float))
    w = _wcs(projection, center_lon, center_lat, cdelt)
    pix = w.wcs_world2pix(np.column_stack([lon, lat]), 1)
    return pix[:, 0], pix[:, 1]


def deproject(x, y, projection: str, center_lon: float, center_lat: float,
             cdelt: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of project(): plane (x, y) back to native (lon, lat), degrees."""
    if projection not in PROJECTIONS:
        raise ValueError(f"unknown projection {projection!r}; choose from "
                         f"{sorted(PROJECTIONS)}")
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    w = _wcs(projection, center_lon, center_lat, cdelt)
    world = w.wcs_pix2world(np.column_stack([x, y]), 1)
    return world[:, 0], world[:, 1]


def _draw_graticule(ax, projection: str, center_lon: float, center_lat: float,
                    whole_sky: bool, half_width: float = 40.0,
                    cdelt: float = 1.0) -> None:
    """Draw faint reference meridians/parallels -- what turns a bare scatter
    of projected points into something recognizable as a sky map.

    whole_sky=True draws the full [0,360] x [-90,90] grid (for mol/ait/car,
    and for the zenithal projections that are valid over the whole sky --
    arc/stg/zea -- just centred wherever their centre is). whole_sky=False
    draws a local grid of *offsets* from the centre instead, for tan/sin,
    the two zenithal projections whose usable domain is genuinely limited to
    a modest patch (see PROJECTIONS and test_projections.py).
    """
    if whole_sky:
        for lon0 in range(0, 360, 30):                      # meridians
            lat_line = np.linspace(-89.9, 89.9, 181)
            x, y = project(np.full_like(lat_line, lon0), lat_line, projection,
                           center_lon, center_lat, cdelt)
            ax.plot(x, y, color="0.75", linewidth=0.6, zorder=1)
        for lat0 in range(-60, 90, 30):                      # parallels
            lon_line = np.linspace(0, 360, 361)
            x, y = project(lon_line, np.full_like(lon_line, lat0), projection,
                           center_lon, center_lat, cdelt)
            ax.plot(x, y, color="0.75", linewidth=0.6, zorder=1)
    else:
        step = max(half_width / 4, 5.0)
        offsets = np.arange(-half_width, half_width + 1, step)
        for dlon in offsets:                                 # local meridians
            lat_line = np.linspace(center_lat - half_width, center_lat + half_width, 161)
            x, y = project(np.full_like(lat_line, center_lon + dlon), lat_line,
                           projection, center_lon, center_lat, cdelt)
            ax.plot(x, y, color="0.75", linewidth=0.6, zorder=1)
        for dlat in offsets:                                 # local parallels
            lon_line = np.linspace(center_lon - half_width, center_lon + half_width, 161)
            x, y = project(lon_line, np.full_like(lon_line, center_lat + dlat),
                           projection, center_lon, center_lat, cdelt)
            ax.plot(x, y, color="0.75", linewidth=0.6, zorder=1)


def _break_jumps(x: np.ndarray, y: np.ndarray, factor: float
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Insert a NaN wherever consecutive points jump far more than typical.

    For a dense, ordered trace (many points, small steps between them) this
    is what stops a projection discontinuity -- most commonly the sky
    wrapping from 359.9 deg back to 0 deg, which happens once every sidereal
    day for a drift scan -- from being drawn as a single line shooting
    straight across the plot. It does NOT touch points that are already NaN
    (out of the projection's valid domain): matplotlib already breaks a line
    at a NaN on its own, so those need no help.

    `factor` is a multiple of the *typical* (median) step size between
    consecutive points; a step bigger than that is treated as a discontinuity
    rather than real motion across the sky. For a reasonably evenly-sampled
    trace, a true wrap-around jump is close to the full width of the plot --
    tens to hundreds of times the typical step -- so a default of 8x has
    plenty of margin against flagging normal, real steps by mistake.
    """
    if x.size < 3:
        return x, y
    step = np.hypot(np.diff(x), np.diff(y))
    finite = np.isfinite(step)
    if not finite.any():
        return x, y
    typical = np.median(step[finite])
    threshold = factor * max(typical, 1e-12)
    break_after = np.flatnonzero(finite & (step > threshold))
    if break_after.size == 0:
        return x, y
    return (np.insert(x, break_after + 1, np.nan),
           np.insert(y, break_after + 1, np.nan))


def plot_points(lon, lat, projection: str, center_lon: float = 0.0,
                center_lat: float = 0.0, *, kind: str = "scatter",
                labels=None, path: str | None = None, title: str | None = None,
                graticule: bool = True, half_width: float = 40.0,
                cdelt: float = 1.0, s: float = 18, linewidth: float = 1.3,
                break_factor: float = 8.0):
    """Project a list of (lon, lat) points and plot them, in one call.

    This is the everyday entry point: give it your coordinates and a
    projection name, get an image back. Works identically for one point,
    a handful of named sources, or tens of thousands of points densely
    tracing a path across the sky (a telescope's pointing over time, say) --
    project() does exactly one vectorized computation regardless of size,
    and matplotlib draws either a scatter of markers or a single connected
    line just as cheaply at that scale.

    Parameters
    ----------
    lon, lat
        Your coordinates, in degrees, in whatever frame you've already put
        them in (this function does not know or care which frame -- convert
        first with SkyCoord(...).transform_to(...) if needed). Any
        array-like: a Python list, a numpy array, a pandas column.
    projection
        One of PROJECTIONS, e.g. "mol", "tan", "arc".
    center_lon, center_lat
        Where the projection is centred, in the same frame/units as lon/lat.
        Required (not just recommended) for the zenithal projections
        (tan/sin/arc/stg/zea) -- there's no sensible default centre for an
        arbitrary list of points. Defaults to (0, 0), which is fine for the
        whole-sky projections (mol/ait/car).
    kind
        "scatter" (default) plots each point as a separate marker -- right
        for a handful of named sources. "line" connects the points in the
        order given -- right for a dense, time-ordered trace. In "line" mode,
        a jump much bigger than the typical step between consecutive points
        is treated as a projection discontinuity (see break_factor) and
        breaks the line there instead of drawing a spurious connector across
        the whole plot.
    labels
        Optional list of strings, same length as lon/lat, to annotate each
        point. Sensible for "scatter"; usually skipped for a dense "line".
    path
        If given, save the figure here (any matplotlib-supported extension:
        .png, .pdf, .svg, ...). Either way the Figure is returned, so you
        can also just show it yourself (fig.show(), or in a notebook, plt).
    graticule
        Draw faint reference meridian/parallel lines (default on) -- makes
        the plot readable as a sky map instead of a bare scatter.
    half_width
        Only affects "tan" and "sin", the two genuinely narrow-field
        projections (radius <= 90 deg): how far out, in degrees from the
        centre, to draw the reference grid and set the plot limits. Every
        other projection -- including the zenithal ones valid over the
        whole sky (arc/stg/zea) -- draws the full sky instead, just centred
        wherever you pointed it, so this is ignored for those.
    linewidth
        Line width, only used when kind="line".
    break_factor
        Only used when kind="line". See _break_jumps().

    Returns
    -------
    matplotlib.figure.Figure
    """
    if kind not in ("scatter", "line"):
        raise ValueError(f"kind must be 'scatter' or 'line', got {kind!r}")

    import matplotlib
    if path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lon = np.atleast_1d(np.asarray(lon, dtype=float))
    lat = np.atleast_1d(np.asarray(lat, dtype=float))
    x, y = project(lon, lat, projection, center_lon, center_lat, cdelt)
    dropped = int(np.sum(~np.isfinite(x) | ~np.isfinite(y)))

    # mol/ait/car are naturally ~2:1 wide; the zenithal ones (whole-sky or
    # not) are naturally circular/square. Matching the figure to that avoids
    # a lot of dead white space above and below a Mollweide ellipse.
    figsize = (9, 5) if projection in ("mol", "ait", "car") else (7, 6.5)
    fig, ax = plt.subplots(figsize=figsize)
    whole_sky = projection not in _NARROW_FIELD
    if graticule:
        _draw_graticule(ax, projection, center_lon, center_lat, whole_sky,
                        half_width, cdelt)

    if kind == "line":
        xb, yb = _break_jumps(x, y, break_factor)
        ax.plot(xb, yb, "-", linewidth=linewidth, color="C0", zorder=3)
    else:
        ax.scatter(x, y, s=s, color="C0", zorder=3)
    if labels is not None:
        for xi, yi, label in zip(x, y, labels):
            if np.isfinite(xi) and np.isfinite(yi):
                ax.annotate(str(label), (xi, yi), xytext=(4, 4),
                           textcoords="offset points", fontsize=7)

    ax.set_aspect("equal")
    ax.set_xlabel("x (deg)")
    ax.set_ylabel("y (deg)")
    ax.set_title(title or f"{projection} projection, centred on "
                          f"({center_lon:g}, {center_lat:g})")
    if not whole_sky:
        ax.set_xlim(half_width, -half_width)   # east left, the usual sky convention
        ax.set_ylim(-half_width, half_width)
    if dropped:
        ax.text(0.02, 0.02, f"{dropped} point(s) outside this projection's "
                            f"valid domain, not shown", transform=ax.transAxes,
               fontsize=7, color="0.4")
    fig.tight_layout()

    if path is not None:
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return fig
