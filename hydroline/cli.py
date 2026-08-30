"""Command-line entry point: one run produces every table and figure."""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

from . import __version__
from .analysis import (
    beam_encounters,
    detectability_table,
    galactic_hi_estimate,
    galactic_plane_crossings,
    sky_coverage,
)
from .antenna import AntennaModel
from .catalog import DEFAULT_BODIES, load_catalog, solar_system_track
from .config import (
    Antenna,
    Configuration,
    DUKE_PHYSICS,
    Observation,
    PRESET_ANTENNAS,
    Site,
)
from .export import write_report, write_summary, write_tables
from .pointing import configure_iers, validate_track, zenith_track
from .theme import THEMES


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hydroline",
        description="Plan a zenith-pointing 21 cm drift-scan telescope: sky "
                    "track, transits, and what the antenna can detect.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"hydroline {__version__}")
    p.add_argument("--config", type=Path,
                   help="YAML/JSON config file; flags below override it")
    p.add_argument("--save-config", type=Path,
                   help="write the resolved configuration and exit")
    p.add_argument("--list-antennas", action="store_true",
                   help="print the built-in antenna presets and exit")

    g = p.add_argument_group("site")
    g.add_argument("--site-name", help="label for plots and reports")
    g.add_argument("--lat", type=float, help="geodetic latitude, degrees north")
    g.add_argument("--lon", type=float, help="longitude, degrees east (west is negative)")
    g.add_argument("--elevation", type=float, help="height above the ellipsoid, m")
    g.add_argument("--timezone", help="IANA timezone for local times")

    g = p.add_argument_group("antenna")
    g.add_argument("--antenna", choices=sorted(PRESET_ANTENNAS),
                   help="start from a built-in preset")
    g.add_argument("--antenna-name")
    g.add_argument("--kind", choices=["dish", "horn", "beamwidth"])
    g.add_argument("--diameter", type=float, help="dish diameter, m")
    g.add_argument("--aperture-e", type=float, help="horn E-plane aperture, m")
    g.add_argument("--aperture-h", type=float, help="horn H-plane aperture, m")
    g.add_argument("--hpbw", type=float, help="measured half-power beamwidth, deg")
    g.add_argument("--efficiency", type=float, help="aperture efficiency, 0-1")
    g.add_argument("--beam-factor", type=float,
                   help="HPBW[deg] = factor * lambda / D (58.4 uniform, ~70 tapered)")
    g.add_argument("--tsys", type=float, help="system temperature, K")
    g.add_argument("--bandwidth", type=float, help="RF bandwidth, Hz")
    g.add_argument("--channel", type=float, help="spectrometer channel width, Hz")
    g.add_argument("--integration", type=float, help="integration time, s")
    g.add_argument("--npol", type=int, choices=[1, 2])
    g.add_argument("--snr", type=float, help="detection threshold in sigma")
    g.add_argument("--frequency", type=float, help="centre frequency, Hz")
    g.add_argument("--alt", type=float,
                   help="pointing altitude, deg (90 = straight up)")
    g.add_argument("--az", type=float, help="pointing azimuth, deg")

    g = p.add_argument_group("observation")
    g.add_argument("--start", help="UTC start, ISO 8601, or 'auto'")
    g.add_argument("--hours", type=float, help="window length in hours")
    g.add_argument("--step", type=float, help="time step in minutes")
    g.add_argument("--beam-fraction", type=float,
                   help="beam edge as a multiple of the HPBW (0.5 = half power)")

    g = p.add_argument_group("output")
    g.add_argument("--outdir", type=Path, default=Path("outputs"))
    g.add_argument("--catalog", type=Path, help="alternative source CSV")
    g.add_argument("--theme", choices=sorted(THEMES), default="dark")
    g.add_argument("--bodies", nargs="*", default=list(DEFAULT_BODIES),
                   help="solar-system bodies to track ('none' to skip)")
    g.add_argument("--no-plots", action="store_true")
    g.add_argument("--no-animation", action="store_true")
    g.add_argument("--frames", type=int, default=96,
                   help="maximum animation frames")
    g.add_argument("--include-misses", action="store_true",
                   help="also list sources that never enter the beam")
    g.add_argument("--offline", action="store_true",
                   help="never download IERS Earth-orientation tables")
    g.add_argument("--quiet", action="store_true")
    return p


def configuration_from_args(args: argparse.Namespace) -> Configuration:
    cfg = Configuration.load(args.config) if args.config else Configuration(
        site=DUKE_PHYSICS, antenna=PRESET_ANTENNAS["dish-1m5"],
        observation=Observation(),
    )

    site_map = {
        "name": args.site_name, "latitude_deg": args.lat,
        "longitude_deg": args.lon, "elevation_m": args.elevation,
        "timezone": args.timezone,
    }
    if any(v is not None for v in site_map.values()):
        cfg.site = replace(cfg.site,
                           **{k: v for k, v in site_map.items() if v is not None})

    if args.antenna:
        cfg.antenna = PRESET_ANTENNAS[args.antenna]
    ant_map = {
        "name": args.antenna_name, "kind": args.kind,
        "diameter_m": args.diameter, "aperture_e_m": args.aperture_e,
        "aperture_h_m": args.aperture_h, "hpbw_deg": args.hpbw,
        "aperture_efficiency": args.efficiency,
        "beam_factor_deg": args.beam_factor, "t_sys_k": args.tsys,
        "rf_bandwidth_hz": args.bandwidth, "channel_hz": args.channel,
        "integration_s": args.integration, "n_pol": args.npol,
        "snr_threshold": args.snr, "center_freq_hz": args.frequency,
        "pointing_alt_deg": args.alt, "pointing_az_deg": args.az,
    }
    ant_map = {k: v for k, v in ant_map.items() if v is not None}
    if ant_map:
        if "kind" in ant_map and ant_map["kind"] == "horn":
            ant_map.setdefault("aperture_e_m", cfg.antenna.aperture_e_m or 0.6)
            ant_map.setdefault("aperture_h_m", cfg.antenna.aperture_h_m or 0.5)
        cfg.antenna = replace(cfg.antenna, **ant_map)

    obs_map = {
        "start_utc": args.start, "duration_hours": args.hours,
        "step_minutes": args.step, "beam_fraction": args.beam_fraction,
    }
    obs_map = {k: v for k, v in obs_map.items() if v is not None}
    if obs_map:
        cfg.observation = replace(cfg.observation, **obs_map)
    return cfg


def run(cfg: Configuration, args: argparse.Namespace) -> dict:
    """Execute a full analysis and write everything to ``args.outdir``."""
    say = (lambda *a: None) if args.quiet else print
    configure_iers(offline=args.offline)

    model = AntennaModel(cfg.antenna)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    say(f"\n{model.describe()}\n")
    say(f"Site    : {cfg.site.name}")
    say(f"          {cfg.site.latitude_deg:+.4f}° lat, "
        f"{cfg.site.longitude_deg:+.4f}° lon, {cfg.site.elevation_m:.0f} m")

    track = zenith_track(cfg.site, cfg.observation, cfg.antenna)
    validation = validate_track(track)
    say(f"Window  : {track.times[0].isot[:19]} → {track.times[-1].isot[:19]} UTC"
        f"  ({track.n_steps} steps)")
    say(f"Pointing: Dec {track.dec_deg:+.4f}°  (geodetic latitude "
        f"{cfg.site.latitude_deg:+.4f}°), RA sweeps the full 24 h")
    if validation["passed"] is not None:
        mark = "OK" if validation["passed"] else "FAILED"
        say(f"Check   : zenith identities reproduced to "
            f"{max(validation['max_abs_dec_apparent_minus_latitude_arcmin'], validation['max_abs_ra_apparent_minus_lst_arcmin']) * 60:.2f}″  [{mark}]")

    catalog = load_catalog(args.catalog)
    bodies = tuple(b for b in (args.bodies or []) if b.lower() != "none")
    solar = (solar_system_track(track.times, track.location, bodies)
             if bodies else {})

    frac = cfg.observation.beam_fraction
    enc = beam_encounters(track, catalog, model, solar, beam_fraction=frac,
                          include_misses=args.include_misses)
    det = detectability_table(catalog, model, track.dec_deg, solar,
                              beam_fraction=frac)
    crossings = galactic_plane_crossings(track)
    ghi = galactic_hi_estimate(track, model)
    coverage = sky_coverage(track, model)

    in_beam = det[det["enters_beam"]]
    say(f"\nCatalogue: {len(catalog)} fixed sources + {len(solar)} solar-system "
        f"bodies")
    say(f"In beam  : {len(in_beam)} objects, "
        f"{int(enc['in_beam'].sum()) if len(enc) else 0} transits in this window")
    say(f"Detectable: {int(in_beam['any_detection'].sum())} at "
        f"{cfg.antenna.snr_threshold:.0f}σ in a single transit")
    say(f"Sky covered: {coverage['fraction_of_sky'] * 100:.1f}% "
        f"({coverage['band_solid_angle_deg2']:,.0f} deg²), declination "
        f"{coverage['declination_band_deg'][0]:+.1f}° to "
        f"{coverage['declination_band_deg'][1]:+.1f}°")
    say(f"Galactic HI: peak T_A {ghi['estimated_peak_t_a_k'].max():.0f} K, "
        f"{len(crossings)} plane crossing(s)")

    contam = det[det["contaminates_off_axis"]].sort_values(
        "t_a_at_closest_k", ascending=False)
    if not contam.empty:
        say("\nOff-axis contamination (outside the beam but still strong):")
        for _, r in contam.head(5).iterrows():
            say(f"  {r['name']:<22s} {abs(r['dec_offset_from_beam_deg']):5.1f}° "
                f"off axis  ->  {r['t_a_at_closest_k']:9.3f} K at closest "
                f"approach")
        say("  (Gaussian main beam only; real sidelobes leak more, "
            "so treat these as lower bounds.)")

    tables = {
        "pointing_track": track.table,
        "beam_encounters": enc,
        "detectability": det,
        "sources_in_beam": in_beam,
        "galactic_plane_crossings": crossings,
        "galactic_hi_estimate": ghi,
        "antenna_model": pd.DataFrame([model.summary()]),
    }
    for key, info in solar.items():
        tables[f"solar_system_{key}"] = info["frame"]
    written = write_tables(outdir, tables)

    figures: dict[str, Path] = {}
    if not args.no_plots:
        try:
            from .plots import make_all_figures
        except ImportError as exc:
            from . import missing_dependencies

            need = missing_dependencies("optional")
            reason = (f"{', '.join(need)} not installed" if need
                      else f"the plotting stack failed to import ({exc})")
            say(f"\nSkipping figures: {reason}.")
            say("  Every CSV and summary.json was still written.")
            if need:
                say(f"  Install with: {sys.executable} -m pip install "
                    f"{' '.join(need)}")
            say("  Pass --no-plots to silence this.")
        else:
            say("\nRendering figures…")
            figures = make_all_figures(
                track, model, catalog, det, enc, outdir, THEMES[args.theme],
                animate=not args.no_animation, max_frames=args.frames,
            )

    summary = {
        "hydroline_version": __version__,
        "configuration": cfg.to_dict(),
        "antenna": model.summary(),
        "pointing": track.summary(),
        "validation": validation,
        "sky_coverage": coverage,
        "counts": {
            "catalog_sources": len(catalog),
            "solar_system_bodies": len(solar),
            "objects_in_beam": int(len(in_beam)),
            "transits": int(enc["in_beam"].sum()) if len(enc) else 0,
            "detectable_objects": int(in_beam["any_detection"].sum()),
            "galactic_plane_crossings": int(len(crossings)),
        },
        "galactic_hi": {
            "peak_antenna_temperature_k": float(ghi["estimated_peak_t_a_k"].max()),
            "min_antenna_temperature_k": float(ghi["estimated_peak_t_a_k"].min()),
        },
        "objects_in_beam": [
            {k: v for k, v in row.items()
             if k in ("name", "kind", "dec_deg", "dec_offset_from_beam_deg",
                      "transit_duration_min", "resolution_class",
                      "snr_continuum_single_transit",
                      "snr_line_matched_transit", "detectable_continuum",
                      "detectable_line")}
            for row in in_beam.to_dict("records")
        ],
        "files": {k: str(v) for k, v in {**written, **figures}.items()},
    }
    write_summary(outdir, summary)

    report = write_report(outdir, cfg, model, track, det, enc, crossings,
                          coverage, validation, ghi, figures)
    say(f"\nWrote {len(written)} CSVs, {len(figures)} figures, summary.json")
    say(f"Report: {report}")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_antennas:
        for key, a in sorted(PRESET_ANTENNAS.items()):
            m = AntennaModel(a)
            print(f"{key:16s} {a.name:38s} HPBW {m.hpbw_deg:5.1f}°  "
                  f"A_eff {m.effective_area_m2:5.2f} m²  T_sys {a.t_sys_k:.0f} K")
        return 0

    cfg = configuration_from_args(args)
    if args.save_config:
        cfg.save(args.save_config)
        print(f"wrote {args.save_config}")
        return 0

    run(cfg, args)
    return 0


if __name__ == "__main__":                                  # pragma: no cover
    sys.exit(main())
