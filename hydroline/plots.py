"""Figures: where the beam goes, when, and what it can see.

Every figure has a CSV twin written by :mod:`hydroline.export`, so no value is
reachable only through a picture.

Encoding rules kept throughout: colour carries one job per chart, scatter charts
stay inside the three all-pairs-validated slots, line and bar charts may use the
eight-slot categorical order, magnitude rides on position or marker size rather
than on a colour ramp over nominal categories, and reference lines are neutral
so they never impersonate a series.
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D

import astropy.units as u
from astropy.coordinates import AltAz, SkyCoord
from astropy.utils.exceptions import AstropyWarning

from .analysis import _separation_deg
from .antenna import AntennaModel
from .catalog import Catalog
from .pointing import PointingTrack
from .theme import DARK, Theme, styled

__all__ = [
    "plot_sky_equatorial", "plot_sky_galactic", "plot_time_series",
    "plot_transit_timeline", "plot_separations", "plot_antenna",
    "plot_detectability", "animate_drift", "make_all_figures",
    "spherical_circle",
]


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def spherical_circle(lon0_deg: float, lat0_deg: float, radius_deg: float,
                     n: int = 181) -> tuple[np.ndarray, np.ndarray]:
    """Points on a true small circle of angular radius ``radius_deg``.

    A beam several degrees across is not a circle in any flat projection, so the
    outline is traced on the sphere first and projected afterwards.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, n)
    r = math.radians(radius_deg)
    d0, a0 = math.radians(lat0_deg), math.radians(lon0_deg)
    sin_d = math.sin(d0) * math.cos(r) + math.cos(d0) * math.sin(r) * np.cos(theta)
    lat = np.arcsin(np.clip(sin_d, -1.0, 1.0))
    y = np.sin(theta) * math.sin(r) * math.cos(d0)
    x = math.cos(r) - math.sin(d0) * np.sin(lat)
    return np.degrees(a0 + np.arctan2(y, x)) % 360.0, np.degrees(lat)


def _moll_x(lon_deg) -> np.ndarray:
    """Longitude in radians for a Mollweide axis, increasing leftward.

    All-sky astronomical maps put increasing longitude on the left, as if the
    reader were inside the sphere looking out.
    """
    return -np.radians(((np.asarray(lon_deg, dtype=float) + 180.0) % 360.0) - 180.0)


def _split_wrap(x: np.ndarray, y: np.ndarray, limit: float = np.pi):
    """Break a polyline wherever it wraps around the projection seam."""
    jump = np.flatnonzero(np.abs(np.diff(x)) > limit) + 1
    return zip(np.split(x, jump), np.split(y, jump))


# ---------------------------------------------------------------------------
# label placement
# ---------------------------------------------------------------------------
def _place_labels(ax, items: Sequence[tuple[float, float, str, str]],
                  fontsize: float = 6.8, weight: str = "normal") -> int:
    """Draw point labels, skipping any that would collide with an earlier one.

    Candidate offsets are tried above then below the marker at increasing
    distance; a label that cannot find a clear slot is dropped rather than
    overprinted.  Dropping is the right failure mode -- the full list is always
    in the CSV.
    """
    fig = ax.figure
    fig.canvas.draw()
    placed: list[tuple[float, float, float, float]] = []
    char_w = fontsize * 0.60 * fig.dpi / 72.0
    line_h = fontsize * 1.42 * fig.dpi / 72.0
    offsets = (13, -17, 23, -27, 33, -37, 43, -47)
    drawn = 0

    for x, y, text, color in items:
        px, py = ax.transData.transform((x, y))
        if not (np.isfinite(px) and np.isfinite(py)):
            continue
        half_w = 0.5 * char_w * len(text)
        for dy in offsets:
            box = (px - half_w, px + half_w,
                   py + dy - line_h * 0.5, py + dy + line_h * 0.5)
            if any(box[0] < b[1] and box[1] > b[0]
                   and box[2] < b[3] and box[3] > b[2] for b in placed):
                continue
            ax.annotate(text, (x, y), xytext=(0, dy),
                        textcoords="offset points" if False else "offset pixels",
                        ha="center", va="center", fontsize=fontsize,
                        color=color, weight=weight, annotation_clip=False,
                        zorder=20)
            placed.append(box)
            drawn += 1
            break
    return drawn


def _sequential_cmap(theme: Theme, lo: float = 0.25, hi: float = 0.95):
    """Single-hue ramp restricted to the sub-range actually used by the marks.

    Truncating here rather than at draw time keeps the colourbar and the marks
    on the same mapping -- a colourbar that spans a wider range than the data
    uses is a legend that lies.
    """
    base = matplotlib.colors.LinearSegmentedColormap.from_list(
        "seq", list(theme.sequential)[::-1])
    return matplotlib.colors.LinearSegmentedColormap.from_list(
        "seq_cut", base(np.linspace(lo, hi, 256)))


def _flux_marker_size(flux_jy) -> np.ndarray:
    """Marker area encoding flux density, logarithmically compressed."""
    f = pd.to_numeric(pd.Series(np.asarray(flux_jy).ravel()),
                      errors="coerce").to_numpy(dtype=float)
    f = np.where(np.isfinite(f) & (f > 0), f, 0.05)
    return 8.0 + 26.0 * np.clip(np.log10(f) + 1.5, 0.0, 5.0) ** 0.9


def _classify(det: pd.DataFrame, beam_radius_deg: float) -> np.ndarray:
    """Three-state beam status -- the only thing colour encodes on sky maps."""
    off = det["dec_offset_from_beam_deg"].abs().to_numpy()
    return np.where(off <= beam_radius_deg, "in beam",
                    np.where(off <= 1.6 * beam_radius_deg, "near miss",
                             "off track"))


def _status_colors(theme: Theme) -> dict[str, str]:
    return {"in beam": theme.inside, "near miss": theme.near,
            "off track": theme.ink_muted}


def _sky_legend_handles(theme: Theme, model: AntennaModel):
    """Legend entries shared by every sky map."""
    handles = _sky_handles(theme, model)
    return handles, [h.get_label() for h in handles]


def _sky_handles(theme: Theme, model: AntennaModel):
    return [
        Line2D([], [], color=theme.beam, lw=2,
               label=f"beam track and {model.hpbw_deg:.1f}° HPBW footprint"),
        Line2D([], [], marker="o", color="none", markerfacecolor=theme.inside,
               markeredgecolor=theme.panel, markersize=7,
               label="inside the half-power beam"),
        Line2D([], [], marker="o", color="none", markerfacecolor=theme.near,
               markeredgecolor=theme.panel, markersize=7,
               label="near miss (within 1.6 beam radii)"),
        Line2D([], [], marker="o", color="none",
               markerfacecolor=theme.ink_muted, markeredgecolor=theme.panel,
               markersize=6, label="never in the beam"),
        Line2D([], [], marker="D", color="none",
               markerfacecolor=theme.ink_secondary, markeredgecolor=theme.panel,
               markersize=6, label="21 cm line emitter (diamond)"),
    ]


def _sky_legend(ax, theme: Theme, model: AntennaModel, loc="upper center",
                bbox=(0.5, -0.09), ncol=3) -> None:
    ax.legend(handles=_sky_handles(theme, model), loc=loc,
              bbox_to_anchor=bbox, ncol=ncol, frameon=False, fontsize=7.6,
              labelcolor=theme.ink_secondary, handletextpad=0.5,
              columnspacing=1.6)


def _scatter_sources(ax, lon, lat, det, theme, status, project=True,
                     alpha_off=0.5) -> None:
    """Plot the catalogue with colour = beam status, size = flux, shape = HI."""
    sizes = _flux_marker_size(det["s1400_total_jy"])
    hi = det["hi_emitter"].to_numpy(dtype=bool)
    x = _moll_x(lon) if project else np.asarray(lon, dtype=float)
    y = np.radians(lat) if project else np.asarray(lat, dtype=float)
    for label, z, alpha in (("off track", 2, alpha_off), ("near miss", 5, 0.95),
                            ("in beam", 6, 1.0)):
        m = status == label
        if not m.any():
            continue
        for mm, marker in ((m & ~hi, "o"), (m & hi, "D")):
            if mm.any():
                ax.scatter(x[mm], y[mm], s=sizes[mm], marker=marker,
                           facecolor=_status_colors(theme)[label],
                           edgecolor=theme.panel, linewidths=0.7, alpha=alpha,
                           zorder=z)


def _style_mollweide(ax, theme: Theme, xlabel: str, tick_fmt) -> None:
    ax.set_facecolor(theme.panel)
    ax.grid(True, color=theme.grid, linewidth=0.6, linestyle="-", alpha=0.9)
    ticks = np.array([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150])
    ax.set_xticks(np.radians(ticks))
    ax.set_xticklabels([tick_fmt(-t) for t in ticks], fontsize=7.5,
                       color=theme.ink_muted)
    ax.set_yticks(np.radians([-60, -30, 0, 30, 60]))
    ax.set_yticklabels(["−60°", "−30°", "0°", "+30°", "+60°"], fontsize=7.5,
                       color=theme.ink_muted)
    ax.set_xlabel(xlabel, color=theme.ink_secondary, fontsize=8.5, labelpad=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(theme.grid)


# ---------------------------------------------------------------------------
# 1. equatorial: all-sky context + the drift strip up close
# ---------------------------------------------------------------------------
def plot_sky_equatorial(track: PointingTrack, model: AntennaModel,
                        det: pd.DataFrame, path, theme: Theme = DARK) -> Path:
    """All-sky map of the swept band, plus a zoom on the strip itself."""
    path = Path(path)
    beam_r = model.beam_radius_deg
    dec0 = track.dec_deg
    status = _classify(det, beam_r)
    lon = det["ra_deg"].to_numpy()
    lat = det["dec_deg"].to_numpy()
    hours = track.table["hours_from_start"].to_numpy()
    bra = track.table["ra_icrs_deg"].to_numpy()
    bdec = track.table["dec_icrs_deg"].to_numpy()

    with styled(theme):
        fig = plt.figure(figsize=(11.4, 9.6))
        gs = fig.add_gridspec(2, 1, height_ratios=[1.24, 1.0], hspace=0.34,
                              top=0.895, bottom=0.135, left=0.06, right=0.965)

        # ---------------- all-sky context ------------------------------
        ax = fig.add_subplot(gs[0], projection="mollweide")
        _style_mollweide(ax, theme, "Right ascension (J2000)",
                         lambda t: f"{int(round((t % 360) / 15)) % 24}h")
        grid = np.linspace(-180.0, 180.0, 361)
        # A wide beam near the pole reaches past +/-90: the band becomes a polar
        # cap, so clamp rather than draw a declination that does not exist.
        band_lo = max(dec0 - beam_r, -90.0)
        band_hi = min(dec0 + beam_r, 90.0)
        ax.fill_between(_moll_x(grid),
                        np.radians(np.full_like(grid, band_lo)),
                        np.radians(np.full_like(grid, band_hi)),
                        color=theme.beam, alpha=0.22, linewidth=0, zorder=1)
        for edge in (band_lo, band_hi):
            if abs(edge) < 89.9:
                ax.plot(_moll_x(grid), np.radians(np.full_like(grid, edge)),
                        color=theme.beam, lw=0.9, alpha=0.75, zorder=3)
        ax.plot(_moll_x(grid), np.radians(np.full_like(grid, dec0)),
                color=theme.beam, lw=1.8, zorder=4)
        _scatter_sources(ax, lon, lat, det, theme, status, alpha_off=0.5)
        # No labels on the context map: the strip below carries them all with
        # room to breathe, and duplicating them here only creates collisions.
        ax.set_title(
            f"The {2 * beam_r:.1f}°-wide band is all this antenna will ever see"
            f" — {100 * (math.sin(math.radians(min(90, dec0 + beam_r))) - math.sin(math.radians(max(-90, dec0 - beam_r)))) / 2:.1f}%"
            f" of the sky",
            fontsize=9.5, color=theme.ink_secondary, pad=10)

        # ---------------- the strip, unrolled ---------------------------
        axs = fig.add_subplot(gs[1])
        axs.set_facecolor(theme.panel)
        # Declination is bounded by the sphere: a 60-degree beam must not draw
        # an axis running to +132 deg.
        half = min(max(3.2 * beam_r, 14.0), 60.0)
        strip_lo = max(dec0 - half, -90.0)
        strip_hi = min(dec0 + half, 90.0)
        axs.axhspan(dec0 - beam_r, dec0 + beam_r, color=theme.beam, alpha=0.16,
                    lw=0, zorder=1)
        for edge in (dec0 - beam_r, dec0 + beam_r):
            axs.axhline(edge, color=theme.beam, lw=0.9, alpha=0.8, zorder=2)
        axs.axhline(dec0, color=theme.beam, lw=1.8, zorder=3)
        seen_ra: list[float] = []
        for h in np.arange(0, min(hours.max(), 23.99) + 1e-6, 3.0):
            i = int(np.argmin(np.abs(hours - h)))
            ra_h = bra[i] / 15.0
            if any(abs(ra_h - r) < 0.6 for r in seen_ra):
                continue                       # second lap of a >24 h window
            seen_ra.append(ra_h)
            cx, cy = spherical_circle(bra[i], bdec[i], beam_r, 181)
            cx = ((cx - bra[i] + 180) % 360 - 180) + bra[i]
            axs.plot(cx / 15.0, cy, color=theme.beam, lw=0.9, alpha=0.6,
                     zorder=4)
            axs.annotate(f"+{h:.0f} h", (ra_h, dec0 - beam_r),
                         xytext=(0, -12), textcoords="offset points",
                         ha="center", fontsize=6.6, color=theme.beam)
        near = (lat >= strip_lo) & (lat <= strip_hi)
        _scatter_sources(axs, lon[near] / 15.0, lat[near], det[near], theme,
                         status[near], project=False, alpha_off=0.75)
        order = np.argsort(-np.nan_to_num(
            det["s1400_total_jy"].to_numpy(dtype=float)))
        _place_labels(axs, [
            (lon[i] / 15.0, lat[i], det["name"].iloc[i],
             _status_colors(theme)[status[i]])
            for i in order if near[i] and status[i] != "off track"
        ], fontsize=7.0)
        axs.set_xlim(0, 24)
        axs.set_ylim(strip_lo, strip_hi)
        axs.set_xticks(np.arange(0, 25, 2))
        axs.set_xlabel("right ascension (hours) — also the order in which "
                       "sources transit")
        axs.set_ylabel("declination (degrees)")
        axs.grid(True, color=theme.grid, lw=0.6, ls="-")
        axs.set_title("The drift strip, unrolled", fontsize=9.5,
                      color=theme.ink_secondary, loc="left", pad=8)

        fig.text(0.5, 0.972, "Where the beam looks — equatorial (ICRS / J2000)",
                 ha="center", va="top", fontsize=14, color=theme.ink,
                 weight="semibold")
        fig.text(0.5, 0.938,
                 f"{track.site.name}  ·  zenith declination {dec0:+.2f}°  ·  "
                 f"the sky drifts through a fixed {model.hpbw_deg:.1f}° beam "
                 f"once per sidereal day",
                 ha="center", va="top", fontsize=8.6,
                 color=theme.ink_secondary)
        handles, labels_ = _sky_legend_handles(theme, model)
        fig.legend(handles=handles, labels=labels_, loc="lower center",
                   bbox_to_anchor=(0.5, 0.005), ncol=5, frameon=False,
                   fontsize=7.6, labelcolor=theme.ink_secondary,
                   handletextpad=0.5, columnspacing=1.8)
        fig.text(0.5, 0.048,
                 "marker area scales with log 1.4 GHz flux density",
                 ha="center", fontsize=7.2, color=theme.ink_muted)
        fig.savefig(path, facecolor=theme.surface)
        plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 2. galactic
# ---------------------------------------------------------------------------
def plot_sky_galactic(track: PointingTrack, model: AntennaModel,
                      det: pd.DataFrame, path, theme: Theme = DARK) -> Path:
    """The same track in galactic coordinates, where the HI story lives."""
    path = Path(path)
    beam_r = model.beam_radius_deg
    status = _classify(det, beam_r)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        gal = SkyCoord(det["ra_deg"].to_numpy() * u.deg,
                       det["dec_deg"].to_numpy() * u.deg,
                       frame="icrs").galactic
    lon, lat = gal.l.deg, gal.b.deg
    bl = track.table["gal_l_deg"].to_numpy()
    bb = track.table["gal_b_deg"].to_numpy()

    with styled(theme):
        fig = plt.figure(figsize=(11.4, 6.9))
        ax = fig.add_subplot(111, projection="mollweide")
        _style_mollweide(ax, theme, "Galactic longitude  l",
                         lambda t: f"{int(round(t)) % 360}°")

        ax.plot(_moll_x(np.linspace(-180, 180, 361)), np.zeros(361),
                color=theme.ink_secondary, lw=0.9, ls="--", alpha=0.55,
                zorder=2)
        ax.text(_moll_x(300.0), math.radians(6.0), "galactic plane  b = 0",
                fontsize=7.2, color=theme.ink_secondary, ha="center")

        step = max(1, len(bl) // 260)
        for i in range(0, len(bl), step):
            cl, cb = spherical_circle(bl[i], bb[i], beam_r, 61)
            for xx, yy in _split_wrap(_moll_x(cl), np.radians(cb)):
                ax.plot(xx, yy, color=theme.beam, lw=0.5, alpha=0.09, zorder=1)
        for xx, yy in _split_wrap(_moll_x(bl), np.radians(bb)):
            ax.plot(xx, yy, color=theme.beam, lw=2.0, zorder=4)
        for h in np.arange(0, track.table["hours_from_start"].max() + 1e-6, 3.0):
            i = int(np.argmin(np.abs(
                track.table["hours_from_start"].to_numpy() - h)))
            cl, cb = spherical_circle(bl[i], bb[i], beam_r, 181)
            for xx, yy in _split_wrap(_moll_x(cl), np.radians(cb)):
                ax.plot(xx, yy, color=theme.beam, lw=0.9, alpha=0.6, zorder=3)

        _scatter_sources(ax, lon, lat, det, theme, status, alpha_off=0.45)
        order = np.argsort(-np.nan_to_num(
            det["s1400_total_jy"].to_numpy(dtype=float)))
        _place_labels(ax, [
            (_moll_x(lon[i]), math.radians(lat[i]), det["name"].iloc[i],
             _status_colors(theme)[status[i]])
            for i in order if status[i] != "off track"
        ], fontsize=6.8)
        _sky_legend(ax, theme, model, bbox=(0.5, -0.13))

        fig.text(0.5, 0.972, "Where the beam looks — galactic",
                 ha="center", va="top", fontsize=14, color=theme.ink,
                 weight="semibold")
        fig.text(0.5, 0.934,
                 "the two crossings of b = 0 are the Milky Way's HI disk: "
                 "strongest line, multiple spiral-arm components",
                 ha="center", va="top", fontsize=8.6,
                 color=theme.ink_secondary)
        fig.savefig(path, facecolor=theme.surface)
        plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 3. time series
# ---------------------------------------------------------------------------
def _mask_wrap(y: np.ndarray, period: float) -> np.ndarray:
    """Insert NaN at a modular wrap so the line does not draw a false jump."""
    y = np.asarray(y, dtype=float).copy()
    jump = np.flatnonzero(np.abs(np.diff(y)) > 0.5 * period)
    y[jump] = np.nan
    return y


def plot_time_series(track: PointingTrack, model: AntennaModel, path,
                     theme: Theme = DARK) -> Path:
    """Pointing and Doppler reference vs. time -- one measure per axis."""
    t = track.table
    h = t["hours_from_start"].to_numpy()
    path = Path(path)

    with styled(theme):
        fig, axes = plt.subplots(4, 1, figsize=(10.8, 10.4), sharex=True)

        # -- right ascension -------------------------------------------------
        ax = axes[0]
        ax.plot(h, _mask_wrap(t["ra_icrs_hours"].to_numpy(), 24.0),
                color=theme.beam, lw=2, label="beam RA (ICRS / J2000)")
        ax.plot(h, _mask_wrap(t["lst_apparent_hours"].to_numpy(), 24.0),
                color=theme.near, lw=1.4, alpha=0.9,
                label="local apparent sidereal time")
        ax.set_ylabel("hours")
        ax.set_ylim(0, 24)
        ax.set_yticks([0, 6, 12, 18, 24])
        ax.set_title("Right ascension of the beam centre", loc="left")
        ax.legend(loc="lower right", ncol=2)
        off = float(np.mean(t["ra_icrs_minus_lst_arcmin"]))
        ax.text(0.012, 0.90,
                f"the two differ by {abs(off):.1f}′ — general precession of the "
                f"equinox since J2000",
                transform=ax.transAxes, ha="left", va="top", fontsize=7.6,
                color=theme.ink_muted)

        # -- declination -----------------------------------------------------
        ax = axes[1]
        dec = t["dec_icrs_deg"].to_numpy()
        ax.plot(h, dec, color=theme.beam, lw=2)
        ax.axhline(track.site.latitude_deg, color=theme.ink_secondary, lw=1.0,
                   ls="--", zorder=1)
        ax.set_ylabel("degrees")
        pad = max(np.ptp(dec) * 0.9, 0.01)
        ax.set_ylim(dec.mean() - pad, dec.mean() + pad)
        ax.set_title("Declination of the beam centre", loc="left")
        ax.text(0.012, 0.12,
                f"geodetic latitude {track.site.latitude_deg:+.4f}° (dashed).  "
                f"The ICRS value swings ±{np.ptp(dec) * 1800:.0f}″ over a day "
                f"because the J2000 pole no longer coincides with the pole of "
                f"date; in the apparent frame the declination is the latitude "
                f"to 0.4″.",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=7.6,
                color=theme.ink_muted, wrap=True)

        # -- galactic latitude ------------------------------------------------
        ax = axes[2]
        b = t["gal_b_deg"].to_numpy()
        ax.fill_between(h, -10, 10, color=theme.beam, alpha=0.12, lw=0)
        ax.plot(h, b, color=theme.beam, lw=2)
        ax.axhline(0, color=theme.ink_secondary, lw=1.0, ls="--", zorder=1)
        ax.set_ylabel("degrees")
        ax.set_title("Galactic latitude — shaded band is the HI disk (|b| < 10°)",
                     loc="left")
        for hh, bb in _plane_crossing_marks(h, b):
            ax.plot([hh], [bb], marker="o", ms=6, color=theme.beam,
                    mec=theme.panel, mew=1.2, zorder=5)
            ax.annotate(f"crosses the plane at +{hh:.1f} h", (hh, bb),
                        xytext=(0, 13), textcoords="offset points",
                        ha="center", fontsize=7.2, color=theme.ink_secondary)

        # -- velocity corrections ---------------------------------------------
        ax = axes[3]
        ax.plot(h, t["v_lsr_correction_kms"], color=theme.beam, lw=2,
                label="topocentric to LSR")
        ax.plot(h, t["v_barycentric_correction_kms"], color=theme.near, lw=1.4,
                label="topocentric to barycentric")
        ax.axhline(0, color=theme.grid, lw=0.8, zorder=1)
        ax.set_ylabel("km s⁻¹")
        ax.set_xlabel(f"hours since {track.times[0].isot[:16]} UTC")
        ax.set_title("Velocity correction to add to a measured spectrum",
                     loc="left")
        ax.legend(loc="lower right", ncol=2)
        rng = float(np.ptp(t["v_lsr_correction_kms"]))
        ax.text(0.012, 0.90,
                f"swings {rng:.1f} km s⁻¹ across the window — apply it per "
                f"spectrum, not once per night",
                transform=ax.transAxes, ha="left", va="top", fontsize=7.6,
                color=theme.ink_muted)

        for ax in axes:
            ax.grid(True, color=theme.grid, lw=0.6, ls="-")
            ax.set_xlim(h.min(), h.max())

        fig.suptitle("Beam pointing and Doppler reference over time",
                     fontsize=14, color=theme.ink, weight="semibold", y=0.997)
        fig.tight_layout(rect=(0, 0, 1, 0.975))
        fig.savefig(path, facecolor=theme.surface)
        plt.close(fig)
    return path


def _plane_crossing_marks(h: np.ndarray, b: np.ndarray) -> list[tuple[float, float]]:
    """Local minima of |b| -- the moments the beam is closest to the plane."""
    absb = np.abs(b)
    return [(float(h[i]), float(b[i])) for i in range(1, len(absb) - 1)
            if absb[i] <= absb[i - 1] and absb[i] < absb[i + 1] and absb[i] < 12]


# ---------------------------------------------------------------------------
# 4. transit timeline
# ---------------------------------------------------------------------------
def plot_transit_timeline(encounters: pd.DataFrame, track: PointingTrack,
                          model: AntennaModel, path,
                          theme: Theme = DARK) -> Path:
    """Gantt-style schedule: what is inside the beam, and when."""
    path = Path(path)
    enc = (encounters[encounters["in_beam"]].copy()
           if len(encounters) else encounters)
    with styled(theme):
        if enc is None or enc.empty:
            fig, ax = plt.subplots(figsize=(10.5, 2.6))
            ax.text(0.5, 0.5, "No catalogued source enters the beam "
                              "during this window",
                    ha="center", va="center", color=theme.ink_secondary)
            ax.axis("off")
            fig.savefig(path, facecolor=theme.surface)
            plt.close(fig)
            return path

        enc = enc.sort_values("transit_hours_from_start").reset_index(drop=True)
        t0 = pd.Timestamp(track.times[0].isot)
        start = ((pd.to_datetime(enc["enter_utc"], format="mixed") - t0)
                 .dt.total_seconds() / 3600.0).to_numpy()
        width = enc["duration_minutes"].to_numpy() / 60.0
        y = np.arange(len(enc))
        resp = enc["beam_response"].to_numpy()
        cmap = _sequential_cmap(theme)

        fig, ax = plt.subplots(figsize=(11.0, 0.34 * len(enc) + 3.0))
        for yi, x0, w, r in zip(y, start, width, resp):
            ax.barh(yi, w, left=x0, height=0.34, color=cmap(r), linewidth=0)
            ax.text(x0 + w + 0.22, yi, f"{w * 60:.0f} min · {r * 100:.0f}% gain",
                    va="center", fontsize=7.2, color=theme.ink_muted)
        ax.set_yticks(y)
        ax.set_yticklabels(enc["name"], fontsize=8.2,
                           color=theme.ink_secondary)
        ax.invert_yaxis()
        ax.set_xlabel(f"hours since {track.times[0].isot[:16]} UTC")
        ax.set_xlim(0, track.table["hours_from_start"].max())
        ax.set_ylim(len(enc) - 0.4, -0.6)
        ax.grid(True, axis="x", color=theme.grid, lw=0.6, ls="-")
        ax.grid(False, axis="y")

        sm = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=matplotlib.colors.Normalize(0, 1))
        cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.035,
                          aspect=45, pad=0.20)
        cb.set_label("beam response at closest approach", fontsize=7.4,
                     color=theme.ink_secondary)
        cb.ax.tick_params(labelsize=7, colors=theme.ink_muted)
        cb.outline.set_edgecolor(theme.grid)

        ax.set_title("Transit schedule — time inside the half-power beam",
                     loc="left", fontsize=13, color=theme.ink,
                     weight="semibold", pad=18)
        ax.text(0, 1.02,
                f"{len(enc)} passes · beam fixed at the zenith, HPBW "
                f"{model.hpbw_deg:.1f}° · sky drifts 15.04°/hour",
                transform=ax.transAxes, fontsize=8.2,
                color=theme.ink_secondary)
        fig.tight_layout()
        fig.savefig(path, facecolor=theme.surface)
        plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 5. separation curves
# ---------------------------------------------------------------------------
def plot_separations(track: PointingTrack, catalog: Catalog,
                     model: AntennaModel, det: pd.DataFrame, path,
                     theme: Theme = DARK, max_sources: int = 6) -> Path:
    """Angular distance from the beam axis vs. time, for the nearest sources."""
    path = Path(path)
    h = track.table["hours_from_start"].to_numpy()
    bra = track.table["ra_icrs_deg"].to_numpy()
    bdec = track.table["dec_icrs_deg"].to_numpy()

    cand = det[det["dec_offset_from_beam_deg"].abs()
               <= 1.8 * model.beam_radius_deg].copy()
    cand = cand.reindex(
        cand["dec_offset_from_beam_deg"].abs().sort_values().index
    ).head(max_sources)

    with styled(theme):
        fig, ax = plt.subplots(figsize=(10.8, 5.6))
        if cand.empty:
            ax.text(0.5, 0.5, "No source passes near the beam", ha="center",
                    va="center", color=theme.ink_secondary)
            ax.axis("off")
        else:
            ax.axhspan(0, model.beam_radius_deg, color=theme.beam, alpha=0.14,
                       lw=0, zorder=1)
            ax.axhline(model.beam_radius_deg, color=theme.ink_secondary,
                       lw=1.1, ls="--", zorder=2)
            ax.text(0.995, model.beam_radius_deg + 0.25, "half-power edge",
                    transform=ax.get_yaxis_transform(), ha="right",
                    va="bottom", fontsize=7.6, color=theme.ink_secondary)

            marks = []
            for i, (_, r) in enumerate(cand.iterrows()):
                color = theme.categorical[i % len(theme.categorical)]
                sep = _separation_deg(r["ra_deg"], r["dec_deg"], bra, bdec)
                ax.plot(h, sep, lw=1.9, color=color, label=r["name"], zorder=3)
                j = int(np.argmin(sep))
                ax.plot([h[j]], [sep[j]], marker="o", ms=6, color=color,
                        mec=theme.panel, mew=1.4, zorder=4)
                marks.append((h[j], sep[j], r["name"], color))

            ax.set_xlim(h.min(), h.max())
            ax.set_ylim(0, max(3.0 * model.beam_radius_deg, 12))
            ax.set_xlabel(f"hours since {track.times[0].isot[:16]} UTC")
            ax.set_ylabel("angular separation from beam axis  (degrees)")
            ax.legend(loc="upper center", ncol=min(len(cand), 6),
                      bbox_to_anchor=(0.5, -0.13))
            ax.set_title("How close each source comes to the beam axis",
                         loc="left", fontsize=13, color=theme.ink,
                         weight="semibold", pad=18)
            ax.text(0, 1.02,
                    "a curve dipping into the shaded band is a transit; the "
                    "width of the dip is the integration time it gives you",
                    transform=ax.transAxes, fontsize=8.2,
                    color=theme.ink_secondary)
            _place_labels(ax, marks, fontsize=7.4, weight="semibold")
        fig.tight_layout()
        fig.savefig(path, facecolor=theme.surface)
        plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 6. antenna
# ---------------------------------------------------------------------------
def plot_antenna(model: AntennaModel, path, theme: Theme = DARK) -> Path:
    """Beam shape, the aperture/beamwidth trade, and noise vs. integration."""
    path = Path(path)
    a = model.antenna
    with styled(theme):
        fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5))

        # -- power pattern ----------------------------------------------------
        ax = axes[0]
        off = np.linspace(-1.6 * model.hpbw_deg, 1.6 * model.hpbw_deg, 400)
        resp = model.beam_profile(off)
        ax.plot(off, resp, color=theme.beam, lw=2)
        ax.fill_between(off, 0, resp, where=np.abs(off) <= model.beam_radius_deg,
                        color=theme.beam, alpha=0.18, lw=0)
        ax.axhline(0.5, color=theme.ink_secondary, lw=1.0, ls="--")
        ax.annotate("", xy=(-model.beam_radius_deg, 0.5),
                    xytext=(model.beam_radius_deg, 0.5),
                    arrowprops=dict(arrowstyle="<->", color=theme.ink_secondary,
                                    lw=1.0))
        ax.text(0, 0.545, f"{model.hpbw_deg:.2f}°", ha="center", fontsize=8.5,
                color=theme.ink)
        ax.set_xlabel("offset from beam axis  (degrees)")
        ax.set_ylabel("normalised power response")
        ax.set_ylim(0, 1.06)
        ax.set_title(f"Beam pattern — HPBW {model.hpbw_deg:.2f}°", loc="left")

        # -- aperture vs beamwidth --------------------------------------------
        ax = axes[1]
        d = np.linspace(0.3, 6.0, 240)
        ax.plot(d, a.beam_factor_deg * model.wavelength_m / d,
                color=theme.beam, lw=2)
        ax.set_yscale("log")
        ax.set_xlabel("aperture diameter  (m)")
        ax.set_ylabel("half-power beamwidth  (degrees)")
        ax.set_title("Aperture sets resolution at 21 cm", loc="left")
        for size, label in ((0.5, "Moon and Sun, 0.5°"),
                            (3.2, "M31's HI disk, 3.2°")):
            ax.axhline(size, color=theme.ink_secondary, lw=0.9, ls="--",
                       alpha=0.8)
            ax.text(5.95, size * 1.08, label, ha="right", fontsize=7.2,
                    color=theme.ink_secondary)
        cur = (a.diameter_m if a.kind != "horn"
               else math.sqrt(a.aperture_e_m * a.aperture_h_m))
        if 0.3 <= cur <= 6.0:
            ax.plot([cur], [model.hpbw_deg], marker="o", ms=8,
                    color=theme.inside, mec=theme.panel, mew=1.6, zorder=5)
            ax.annotate(f"this antenna: {cur:.2f} m gives {model.hpbw_deg:.1f}°",
                        (cur, model.hpbw_deg), xytext=(12, 16),
                        textcoords="offset points", fontsize=7.8,
                        color=theme.inside, weight="semibold")
        ax.set_ylim(0.3, 90)

        # -- radiometer --------------------------------------------------------
        ax = axes[2]
        tau = np.logspace(0, 4.6, 240)
        for i, (bw, lab) in enumerate((
            (a.rf_bandwidth_hz, f"continuum, {a.rf_bandwidth_hz / 1e6:.1f} MHz"),
            (a.channel_hz, f"one channel, {a.channel_hz / 1e3:.0f} kHz"),
        )):
            ax.loglog(tau, [model.delta_t_rms_k(bandwidth_hz=bw,
                                                integration_s=x) for x in tau],
                      lw=2, color=(theme.beam if i == 0 else theme.near),
                      label=lab)
        ax.set_xlabel("integration time  (s)")
        ax.set_ylabel("rms noise  ΔT  (K)")
        ax.set_title(f"Radiometer noise — T_sys {a.t_sys_k:.0f} K", loc="left")
        ax.legend(loc="upper right")
        ax.grid(True, which="both", color=theme.grid, lw=0.5, ls="-")
        for x_mark, lab in ((60, "1 min"), (3600, "1 h"), (36000, "10 h")):
            ax.axvline(x_mark, color=theme.ink_secondary, lw=0.8, ls="--",
                       alpha=0.7)
            ax.annotate(lab, xy=(x_mark, 0.03), xycoords=("data",
                                                          "axes fraction"),
                        xytext=(3, 0), textcoords="offset points", fontsize=7,
                        color=theme.ink_secondary)

        for ax in axes:
            ax.grid(True, color=theme.grid, lw=0.6, ls="-")
        fig.suptitle(f"Antenna model — {a.name}", fontsize=14, color=theme.ink,
                     weight="semibold", x=0.007, ha="left", y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(path, facecolor=theme.surface)
        plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 7. detectability
# ---------------------------------------------------------------------------
def plot_detectability(det: pd.DataFrame, model: AntennaModel, path,
                       theme: Theme = DARK, top: int = 18) -> Path:
    """Single-transit signal-to-noise for everything the beam meets.

    A dot plot rather than bars: the scale is logarithmic, and a bar on a log
    axis encodes its length from an arbitrary origin.  The rule from each dot to
    the threshold is meaningful -- it is the ratio to the detection limit.
    """
    path = Path(path)
    d = det[det["enters_beam"]].copy()
    d["snr"] = d[["snr_continuum_single_transit",
                  "snr_line_matched_transit"]].max(axis=1)
    d = d[d["snr"].notna() & (d["snr"] > 0)].sort_values("snr").tail(top)
    thr = model.antenna.snr_threshold

    with styled(theme):
        fig, ax = plt.subplots(figsize=(10.2, 0.36 * max(len(d), 4) + 2.8))
        if d.empty:
            ax.text(0.5, 0.5, "Nothing in the beam has a modelled flux",
                    ha="center", va="center", color=theme.ink_secondary)
            ax.axis("off")
        else:
            y = np.arange(len(d))
            snr = d["snr"].to_numpy()
            colors = [theme.inside if s >= thr else theme.ink_muted
                      for s in snr]
            for yi, s, c in zip(y, snr, colors):
                ax.plot([min(s, thr), max(s, thr)], [yi, yi], lw=1.4, color=c,
                        alpha=0.55, zorder=2, solid_capstyle="butt")
                ax.plot([s], [yi], marker="o", ms=8, color=c,
                        mec=theme.panel, mew=1.4, zorder=4)
                ax.text(s * (1.22 if s >= thr else 0.82), yi,
                        f"{s:,.0f}σ" if s >= 10 else f"{s:.1f}σ",
                        va="center", ha="left" if s >= thr else "right",
                        fontsize=7.4, color=theme.ink_secondary)
            ax.axvline(thr, color=theme.ink_secondary, lw=1.2, ls="--",
                       zorder=3)
            ax.annotate(f"{thr:.0f}σ detection threshold", xy=(thr, 1.0),
                        xycoords=("data", "axes fraction"), xytext=(5, -12),
                        textcoords="offset points", fontsize=7.8,
                        color=theme.ink_secondary)
            ax.set_yticks(y)
            ax.set_yticklabels(d["name"], fontsize=8.2,
                               color=theme.ink_secondary)
            ax.set_xscale("log")
            ax.set_xlim(min(snr.min(), thr) * 0.35, snr.max() * 4.5)
            ax.set_ylim(-0.7, len(d) - 0.3)
            ax.set_xlabel("signal-to-noise in a single transit  (log scale)")
            ax.grid(True, axis="x", which="major", color=theme.grid, lw=0.6,
                    ls="-")
            ax.grid(False, axis="y")
            ax.set_title("What this antenna can detect as it drifts",
                         loc="left", fontsize=13, color=theme.ink,
                         weight="semibold", pad=18)
            ax.text(0, 1.025,
                    "continuum sources use the full RF bandwidth; 21 cm "
                    "emitters use a filter matched to their line width",
                    transform=ax.transAxes, fontsize=8.2,
                    color=theme.ink_secondary)
        fig.tight_layout()
        fig.savefig(path, facecolor=theme.surface)
        plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 8. animation
# ---------------------------------------------------------------------------
def animate_drift(track: PointingTrack, model: AntennaModel, catalog: Catalog,
                  det: pd.DataFrame, path, theme: Theme = DARK,
                  max_frames: int = 96, fps: int = 10) -> Path:
    """Animated GIF: the sky rotating past a fixed, upward-staring beam.

    Left  -- the local sky seen looking up, horizon at the rim, beam at centre.
    Right -- the drift strip in equatorial coordinates, beam held fixed.
    """
    path = Path(path)
    step = max(1, track.n_steps // max_frames)
    idx = np.arange(0, track.n_steps, step)

    keep = det[det["dec_offset_from_beam_deg"].abs() <= 55.0]
    names = keep["name"].tolist()
    sra = keep["ra_deg"].to_numpy()
    sdec = keep["dec_deg"].to_numpy()
    sizes = _flux_marker_size(keep["s1400_total_jy"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        src = SkyCoord(sra * u.deg, sdec * u.deg, frame="icrs")
        aa = src[:, None].transform_to(
            AltAz(obstime=track.times[idx], location=track.location,
                  pressure=0 * u.hPa)[None, :])
        alt, az = aa.alt.deg, aa.az.deg

    tt = track.table
    bra = tt["ra_icrs_deg"].to_numpy()[idx]
    bdec = tt["dec_icrs_deg"].to_numpy()[idx]
    hours = tt["hours_from_start"].to_numpy()[idx]
    utc = tt["time_utc"].to_numpy()[idx]
    gall = tt["gal_l_deg"].to_numpy()[idx]
    galb = tt["gal_b_deg"].to_numpy()[idx]
    beam_r = model.beam_radius_deg

    with styled(theme):
        fig = plt.figure(figsize=(11.8, 6.1))
        axp = fig.add_subplot(121, projection="polar")
        axe = fig.add_subplot(122)
        fig.subplots_adjust(left=0.04, right=0.97, top=0.80, bottom=0.11,
                            wspace=0.22)

        axp.set_theta_zero_location("N")
        axp.set_theta_direction(1)          # looking up: east to the left
        axp.set_rlim(0, 90)
        axp.set_rticks([30, 60, 90])
        axp.set_yticklabels(["60°", "30°", "0°"], fontsize=7,
                            color=theme.ink_muted)
        axp.set_xticks(np.radians([0, 90, 180, 270]))
        axp.set_xticklabels(["N", "E", "S", "W"], fontsize=9,
                            color=theme.ink_secondary)
        axp.set_rlabel_position(112)
        axp.grid(True, color=theme.grid, lw=0.6, ls="-")
        axp.set_facecolor(theme.panel)
        axp.spines["polar"].set_edgecolor(theme.grid)

        # facecolors=, not c=: passing c makes the collection colormap-mapped,
        # and update_scalarmappable() then overwrites whatever we set per frame.
        pts = axp.scatter([], [], s=[], facecolors=theme.ink_secondary,
                          edgecolors=theme.panel, linewidths=0.6, zorder=5)
        pts.set_array(None)
        axp.plot(np.linspace(0, 2 * np.pi, 181), np.full(181, beam_r),
                 color=theme.beam, lw=1.8, zorder=6)
        axp.scatter([0], [0], s=16, color=theme.beam, zorder=7)

        axe.set_facecolor(theme.panel)
        axe.set_xlabel("right-ascension offset from the beam axis  (degrees)")
        axe.set_ylabel("declination  (degrees)")
        axe.grid(True, color=theme.grid, lw=0.6, ls="-")
        axe.set_xlim(34, -34)
        axe.set_ylim(max(track.dec_deg - 25, -90.0),
                     min(track.dec_deg + 25, 90.0))
        axe.set_aspect("equal", adjustable="box")
        axe.axhspan(track.dec_deg - beam_r, track.dec_deg + beam_r,
                    color=theme.beam, alpha=0.13, lw=0)
        cx, cy = spherical_circle(0.0, track.dec_deg, beam_r, 181)
        axe.plot(((cx + 180) % 360) - 180, cy, color=theme.beam, lw=1.8)
        strip = axe.scatter([], [], s=[], facecolors=theme.ink_muted,
                            edgecolors=theme.panel, linewidths=0.6, zorder=5)
        strip.set_array(None)
        labels = [axe.text(0, 0, "", fontsize=7.2, ha="center", va="bottom",
                           color=theme.inside, weight="semibold")
                  for _ in range(6)]

        clock = fig.text(0.5, 0.955, "", ha="center", fontsize=11.5,
                         color=theme.ink, weight="semibold")
        status = fig.text(0.5, 0.912, "", ha="center", fontsize=8.6,
                          color=theme.ink_secondary)
        inbeam = fig.text(0.5, 0.028, "", ha="center", fontsize=8.8,
                          color=theme.inside, weight="semibold")
        axp.set_title("looking up — horizon at the rim", fontsize=9,
                      color=theme.ink_secondary, pad=16)
        axe.set_title("the drift strip — sky moves right to left", fontsize=9,
                      color=theme.ink_secondary, pad=10, loc="left")

        def update(k: int):
            a, z = alt[:, k], az[:, k]
            sep = _separation_deg(sra, sdec, bra[k], bdec[k])
            pts.set_offsets(np.column_stack([np.radians(z), 90.0 - a]))
            pts.set_sizes(np.where(a > 0, sizes, 0.0))
            pts.set_facecolor(list(np.where(sep <= beam_r, theme.inside,
                                            theme.ink_secondary)))

            dra = ((sra - bra[k] + 180.0) % 360.0) - 180.0
            strip.set_offsets(np.column_stack([dra, sdec]))
            strip.set_sizes(sizes)
            strip.set_facecolor(list(np.where(sep <= beam_r, theme.inside,
                                              theme.ink_muted)))

            here = []
            for rank, (lab, i) in enumerate(zip(labels, np.argsort(sep)[:6])):
                if sep[i] <= 1.9 * beam_r and abs(dra[i]) < 31:
                    # stagger by rank so two close sources do not overprint
                    lab.set_position((dra[i], sdec[i] + 1.2 + 2.1 * (rank % 3)))
                    lab.set_text(names[i])
                    lab.set_color(theme.inside if sep[i] <= beam_r
                                  else theme.near)
                else:
                    lab.set_text("")
                if sep[i] <= beam_r:
                    here.append(names[i])

            clock.set_text(f"{str(utc[k])[:16].replace('T', '   ')} UTC"
                           f"    ·    +{hours[k]:.1f} h")
            status.set_text(
                f"beam at RA {bra[k] / 15:5.2f} h   Dec {bdec[k]:+.2f}°"
                f"    ·    galactic  l {gall[k]:6.1f}°   b {galb[k]:+5.1f}°")
            inbeam.set_text("in the beam:  " + ", ".join(here) if here
                            else "in the beam:  Galactic HI only")
            return [pts, strip, clock, status, inbeam, *labels]

        anim = FuncAnimation(fig, update, frames=len(idx),
                             interval=1000 // fps, blit=False)
        anim.save(str(path), writer=PillowWriter(fps=fps), dpi=100,
                  savefig_kwargs={"facecolor": theme.surface})
        plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def make_all_figures(track, model, catalog, det, encounters, outdir,
                     theme: Theme = DARK, animate: bool = True,
                     max_frames: int = 96) -> dict[str, Path]:
    """Render the whole figure set into ``outdir``."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    figs = {
        "sky_equatorial": plot_sky_equatorial(
            track, model, det, outdir / "01_sky_equatorial.png", theme),
        "sky_galactic": plot_sky_galactic(
            track, model, det, outdir / "02_sky_galactic.png", theme),
        "time_series": plot_time_series(
            track, model, outdir / "03_pointing_vs_time.png", theme),
        "transit_timeline": plot_transit_timeline(
            encounters, track, model, outdir / "04_transit_timeline.png", theme),
        "separations": plot_separations(
            track, catalog, model, det, outdir / "05_separations.png", theme),
        "antenna": plot_antenna(
            model, outdir / "06_antenna_model.png", theme),
        "detectability": plot_detectability(
            det, model, outdir / "07_detectability.png", theme),
    }
    if animate:
        figs["animation"] = animate_drift(
            track, model, catalog, det, outdir / "08_drift_animation.gif",
            theme, max_frames=max_frames)
    return figs
