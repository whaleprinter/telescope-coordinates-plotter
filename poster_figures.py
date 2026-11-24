#!/usr/bin/env python3
"""poster_figures.py -- poster versions of hydroline / hardware_model figures.

    python3 poster_figures.py                 # both figures
    python3 poster_figures.py sky             # just the galactic sky map
    python3 poster_figures.py hardware        # just the receiver-chain response
    python3 poster_figures.py sky --offline   # extra flags go straight to hydroline (see run.py --help)

This is a thin layer on top of the two programs: it computes everything with
the same functions run.py and hardware_model/compute_performance.py use (same
config, same catalogue cache, same parts list) and only redraws the result
with the poster's style (../Duke-Radio-Telescope/posters/fall-2026/poster_style.py): its fonts,
colours, transparent background and real-size type.

Figures                                      (poster version of)
  sky_galactic          beam track on the galactic sky     outputs/02_sky_galactic.png
  hardware_response     LNA + filter, sim vs NanoVNA      hardware_model/output/frequency_response.png

Each is written as  outputs/poster/<name>.pdf  and  .png  here, and as
../Duke-Radio-Telescope/posters/fall-2026/figures/<name>.pdf  for LaTeX (skip that with --no-poster).
"""
from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
POSTER_DIR = HERE.parent / "Duke-Radio-Telescope" / "posters" / "fall-2026"
POSTER_FIG_DIR = POSTER_DIR / "figures"
LOCAL_DIR = HERE / "outputs" / "poster"
HARDWARE_FILE = HERE / "hardware_model" / "hardware_parameters.yaml"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(POSTER_DIR))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

import poster_style as ps  # noqa: E402

# Printed width of one poster column (\linewidth, measured from the LaTeX
# log: 1053.9 pt).  poster_style.figsize() counts in fractions of its own
# 32.6 cm block width, so this converts "x of the column" into that fraction,
# letting each figure be drawn at the exact size it prints (fonts at true size).
COLUMN_IN = 37.04 / 2.54


def col_frac(frac_of_column: float) -> float:
    return frac_of_column * COLUMN_IN / ps.BLOCK_WIDTH_IN


GOLD = "#C98500"   # near-miss sources: the one extra colour, off the poster's palette


# ===========================================================================
#  1. Galactic sky map (hydroline)
# ===========================================================================
# Well-known sources the catalogue only lists by survey designation.
KNOWN_NAMES = {(299.868, 40.734): "Cygnus A"}   # (RA, Dec) in degrees


def _friendly_name(row) -> str | None:
    for (ra, dec), name in KNOWN_NAMES.items():
        if abs(row.ra_deg - ra) < 0.1 and abs(row.dec_deg - dec) < 0.1:
            return name
    n = str(row["name"])
    if n.startswith("N") and n[1:].strip().isdigit():          # ALFALFA's "N4214"
        return f"NGC {n[1:].strip()}"
    return None


def hydroline_data(hydroline_argv: list[str]):
    """Run hydroline's analysis exactly as run.py would, without writing outputs."""
    from hydroline.analysis import detectability_table
    from hydroline.antenna import AntennaModel
    from hydroline.catalog import solar_system_track
    from hydroline.cli import _build_catalog, build_parser, configuration_from_args
    from hydroline.pointing import configure_iers, zenith_track

    # Unless told otherwise, draw from the same catalogue the last run.py used
    # (recorded in outputs/summary.json), so the poster matches 02_sky_galactic.png
    # and needs no network.
    if not {"--catalog", "--offline", "--refresh-catalog"} & set(hydroline_argv):
        import json
        try:
            last = json.loads((HERE / "outputs" / "summary.json").read_text())
            cached = Path(last["live_catalog"]["cache_path"])
        except (OSError, KeyError, TypeError, ValueError):
            cached = None
        if cached and cached.exists():
            print(f"Using the catalogue from the last run.py: {cached.name}")
            hydroline_argv = [*hydroline_argv, "--catalog", str(cached)]

    args = build_parser().parse_args(hydroline_argv)
    cfg = configuration_from_args(args)
    configure_iers(offline=args.offline)
    model = AntennaModel(cfg.antenna)
    track = zenith_track(cfg.site, cfg.observation, cfg.antenna)
    frac = cfg.observation.beam_fraction
    catalog, _ = _build_catalog(cfg, args, model, track, frac, print)
    bodies = tuple(b for b in (args.bodies or []) if b.lower() != "none")
    solar = solar_system_track(track.times, track.location, bodies) if bodies else {}
    det = detectability_table(catalog, model, track.dec_deg, solar, beam_fraction=frac)
    return model, track, det


def plot_sky_galactic(model, track, det, font_pt: float, aspect: float, width: float,
                      args_max_labels: int = 3):
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.utils.exceptions import AstropyWarning

    from hydroline.plots import (_classify, _flux_marker_size, _moll_x, _place_labels,
                                 _split_wrap, spherical_circle)

    beam_r = model.beam_radius_deg
    status = _classify(det, beam_r)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        gal = SkyCoord(det["ra_deg"].to_numpy() * u.deg, det["dec_deg"].to_numpy() * u.deg,
                       frame="icrs").galactic
    lon, lat = gal.l.deg, gal.b.deg
    bl = track.table["gal_l_deg"].to_numpy()
    bb = track.table["gal_b_deg"].to_numpy()
    hours = track.table["hours_from_start"].to_numpy()

    ps.use(base_pt=font_pt)
    fig = plt.figure(figsize=ps.figsize(width, aspect))
    ax = fig.add_subplot(111, projection="mollweide")
    ax.set_facecolor("none")
    ax.grid(True, color=ps.HAIRLINE, linewidth=0.8)
    # gridlines every 60° of l (no labels: they would sit on the galactic plane)
    ax.set_xticks(np.radians([-120, -60, 0, 60, 120]))
    ax.set_xticklabels([])
    ax.set_yticks(np.radians([-60, -30, 0, 30, 60]))
    ax.set_yticklabels(["−60°", "−30°", "", "+30°", "+60°"], color=ps.INKSOFT,
                       fontsize=font_pt * 0.82)
    for sp in ax.spines.values():
        sp.set_edgecolor(ps.INKSOFT)

    # galactic plane
    ax.plot(_moll_x(np.linspace(-180, 180, 361)), np.zeros(361), color=ps.INKSOFT,
            lw=1.2, ls="--", zorder=2)

    # beam footprint swept over the day, then the track itself
    step = max(1, len(bl) // 260)
    for i in range(0, len(bl), step):
        cl, cb = spherical_circle(bl[i], bb[i], beam_r, 61)
        for xx, yy in _split_wrap(_moll_x(cl), np.radians(cb)):
            ax.plot(xx, yy, color=ps.ROYAL, lw=0.6, alpha=0.06, zorder=1)
    for xx, yy in _split_wrap(_moll_x(bl), np.radians(bb)):
        ax.plot(xx, yy, color=ps.ROYAL, lw=3.0, zorder=4)

    # sources: colour = beam status, size = flux, diamond = HI galaxy
    # Sources the beam reaches (sources it never reaches, and the Sun/planets,
    # are left off). Colour = in beam / near miss, size = flux, diamond = HI galaxy.
    colors = {"in beam": ps.RUST, "near miss": GOLD}
    sizes = 2.2 * _flux_marker_size(det["s1400_total_jy"])
    hi = det["hi_emitter"].to_numpy(dtype=bool)
    x, y = _moll_x(lon), np.radians(lat)
    for label, z in (("near miss", 5), ("in beam", 6)):
        for m, marker in ((~hi, "o"), (hi, "D")):
            m = m & (status == label)
            if m.any():
                ax.scatter(x[m], y[m], s=sizes[m], marker=marker, facecolor=colors[label],
                           edgecolor="white", linewidths=0.8, zorder=z)

    # A few names, not dozens of survey designations: Cygnus A (brightest
    # continuum source) and the HI galaxies with the most HI flux.
    named, seen_names = [], set()
    for i in range(len(det)):
        n = _friendly_name(det.iloc[i]) if status[i] != "off track" else None
        if n and n not in seen_names:              # Cygnus A's two lobes -> one label
            seen_names.add(n)
            named.append((i, n))
    rank = lambda t: (t[1] != "Cygnus A", -np.nan_to_num(det["hi_flux_jykms"].iloc[t[0]]))
    named.sort(key=rank)
    _place_labels(ax, [(x[i], y[i], n, ps.NAVY) for i, n in named[:args_max_labels]],
                  fontsize=font_pt * 0.82, weight="semibold")
    import matplotlib.patheffects as pe
    for t in ax.texts:                              # white halo so names read over the track
        t.set_path_effects([pe.withStroke(linewidth=4, foreground="white")])

    # Where the track crosses b = 0: the Milky Way's own HI, the strongest line.
    cross_l = []
    for i in np.where(np.diff(np.sign(bb)) != 0)[0]:
        if not any(abs(bl[i] % 360 - s) < 20 for s in cross_l):
            cross_l.append(bl[i] % 360)
            ax.scatter(_moll_x(bl[i]), 0, marker="*", s=700, color=ps.NAVY,
                       edgecolor="white", linewidths=1.2, zorder=8)

    handles = [
        Line2D([], [], color=ps.ROYAL, lw=3, label=f"beam track ({model.hpbw_deg:.1f}° beam)"),
        Line2D([], [], marker="o", ls="none", mfc=ps.RUST, mec="white", ms=12,
               label="radio source in beam"),
        Line2D([], [], marker="D", ls="none", mfc=ps.RUST, mec="white", ms=10,
               label="HI galaxy in beam"),
        Line2D([], [], marker="*", ls="none", mfc=ps.NAVY, mec="white", ms=20,
               label="Milky Way crossing (ℓ ≈ "
                     + ", ".join(f"{v:.0f}°" for v in sorted(cross_l)) + ")"),
    ]
    if (status == "near miss").any():
        handles.append(Line2D([], [], marker="o", ls="none", mfc=GOLD, mec="white", ms=12,
                              label="near miss"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=2,
              fontsize=font_pt * 0.82, handletextpad=0.4, columnspacing=1.5)
    ax.set_xlabel("")
    return fig


# ===========================================================================
#  2. Receiver chain response (hardware_model)
# ===========================================================================
def hardware_data(path: Path):
    from hardware_model.inputs import load_hardware
    from hardware_model.performance import evaluate
    hw = load_hardware(path)
    return evaluate(hw), hw


# The NanoVNA chain sweeps (verify_part.py runs), one per part order.  Both went
# through a 30 dB pad into port 2 but were saved with 0 dB external attenuation,
# so the pad is added back here (--pad-db).
VNA_DIR = HERE.parent / "NanoVNA" / "Parts_Verification"
CHAIN_RUNS = {"lna_filter": "Chain_LNA_VAT30_Filter_*", "filter_lna": "Chain_Filter_LNA_VAT30_*"}


def read_chain_run(pattern: str) -> tuple[np.ndarray, np.ndarray, Path]:
    """Newest verify_part.py folder matching ``pattern`` -> (freq_hz, |S21| dB, csv path)."""
    runs = sorted(VNA_DIR.glob(pattern))
    if not runs:
        raise SystemExit(f"No NanoVNA run matching {pattern!r} in {VNA_DIR}")
    csv = next(runs[-1].glob("S21_*.csv"))
    d = np.genfromtxt(csv, delimiter=",", names=True)
    return d["freq_hz"], d["s21_db"], csv


def plot_hardware_response(perf, hw, font_pt: float, aspect: float, width: float,
                           span_mhz: tuple[float, float], pad_db: float):
    """Two panels, simulated (navy) against measured on the NanoVNA (rust):
    the LNA and filter on the bench, LNA -> filter and filter -> LNA.

    No cables: the feed coax isn't bought yet, so it is left out of both the
    simulation and the measurement.  The model adds part gains in dB, so its
    two orders are identical; any difference between the measured orders is
    mismatch between the parts.
    """
    from hardware_model.performance import HI_REST_HZ, component_gain_db

    sw = perf.sweep
    f = sw.freq_hz / 1e6
    m = (f >= span_mhz[0]) & (f <= span_mhz[1])
    hi = HI_REST_HZ / 1e6

    # simulated LNA + filter, the parts on the bench
    parts_sim = sum(component_gain_db(hw.components[c], sw.freq_hz)
                    for c in ("lna", "bandpass_filter"))

    meas = {}
    for key, pattern in CHAIN_RUNS.items():
        fh, s21, csv = read_chain_run(pattern)
        meas[key] = (fh / 1e6, s21 + pad_db)
        print(f"  measured {key:10s} {csv.parent.name}")

    ps.use(base_pt=font_pt)
    fig, axes = plt.subplots(1, 2, figsize=ps.figsize(width, aspect), layout="constrained",
                             sharey=True)
    fig.get_layout_engine().set(wspace=0.04)
    a_lf, a_fl = axes

    def pair(ax, sim, meas_f, meas_db, title):
        ax.plot(f[m], sim[m], color=ps.NAVY, lw=3, label="Simulated")
        mm = (meas_f >= span_mhz[0]) & (meas_f <= span_mhz[1])
        ax.plot(meas_f[mm], meas_db[mm], color=ps.RUST, lw=2.5, ls="--", label="Measured (VNA)")
        g_sim = float(np.interp(hi, f, sim))
        g_meas = float(np.interp(hi, meas_f, meas_db))
        # numbers at the HI line, centred between the filter skirts
        ax.text(0.5, 0.06, f"at {hi:.1f} MHz\nsim {g_sim:.1f} dB, meas {g_meas:.1f} dB",
                transform=ax.transAxes, ha="center", va="bottom", color=ps.INK,
                fontsize=font_pt * 0.82, linespacing=1.15,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9))
        ps.title(ax, title)

    pair(a_lf, parts_sim, *meas["lna_filter"], r"LNA $\rightarrow$ filter")
    pair(a_fl, parts_sim, *meas["filter_lna"], r"Filter $\rightarrow$ LNA")

    top = float(np.nanmax(parts_sim[m]))
    for ax in axes:
        ax.axvline(hi, color=ps.INKSOFT, lw=1.2, ls=":")
        ax.set_xlim(*span_mhz)
        ax.set_ylim(top - 30, top + 17)
        ax.set_xlabel("Frequency (MHz)")
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(MultipleLocator(50))
    a_lf.set_ylabel("Gain (dB)")
    a_lf.legend(loc="upper left", fontsize=font_pt * 0.82, handlelength=1.6,
                  borderaxespad=0.3)
    return fig


# ===========================================================================
def save(fig, name: str, to_poster: bool) -> list[Path]:
    out = [ps.save(fig, LOCAL_DIR / name), ps.save(fig, LOCAL_DIR / f"{name}.png", dpi=200)]
    if to_poster:
        out.append(ps.save(fig, POSTER_FIG_DIR / name))
    plt.close(fig)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("which", nargs="*", choices=["sky", "hardware", []], default=[],
                   help="which figures to make (default: all)")
    p.add_argument("--font-pt", type=float, default=22,
                   help="axis-label size in points on the printed poster (default 22)")
    p.add_argument("--hardware-file", type=Path, default=HARDWARE_FILE,
                   help="parts list for the hardware figure (default: %(default)s)")
    p.add_argument("--span", type=float, nargs=2, default=(1300, 1550), metavar=("LO", "HI"),
                   help="frequency range of the hardware figure in MHz (default 1300 1550)")
    p.add_argument("--pad-db", type=float, default=30.0,
                   help="attenuator in the NanoVNA chain runs, added back to them (default 30)")
    p.add_argument("--labels", type=int, default=3,
                   help="how many sources to name on the sky map (default 3)")
    p.add_argument("--no-poster", action="store_true",
                   help="only save into outputs/poster/, not the poster's figures folder")
    args, hydroline_argv = p.parse_known_args()
    which = set(args.which) or {"sky", "hardware"}

    written = []
    if "sky" in which:
        print("== galactic sky map (hydroline) ==")
        model, track, det = hydroline_data(hydroline_argv)
        fig = plot_sky_galactic(model, track, det, args.font_pt, aspect=0.6, width=col_frac(0.6),
                                args_max_labels=args.labels)
        written += save(fig, "sky_galactic", not args.no_poster)
    if "hardware" in which:
        print("== receiver chain response (hardware_model) ==")
        perf, hw = hardware_data(args.hardware_file)
        fig = plot_hardware_response(perf, hw, args.font_pt, aspect=0.32, width=col_frac(0.85),
                                     span_mhz=tuple(args.span), pad_db=args.pad_db)
        written += save(fig, "hardware_response", not args.no_poster)

    print("Saved:")
    for w in written:
        print("  ", w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
