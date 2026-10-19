#!/usr/bin/env python3
"""Hardware parameters -> performance parameters.

    python3 compute_performance.py                       # the defaults below
    python3 compute_performance.py my_hardware.yaml      # another input file
    python3 compute_performance.py --order lna,feed_cable,bandpass_filter,coax_run,sdr
    python3 compute_performance.py --compare current lna,bandpass_filter,feed_cable,coax_run,sdr
    python3 compute_performance.py --all-orders          # try every chain order

Reads   hardware/hardware_parameters.yaml   (and observing_goals.yaml beside it)
Writes  hardware/output/performance_parameters.yaml
        hardware/output/detection_thresholds.yaml
        hardware/output/frequency_response.csv
        hardware/output/frequency_response.png, beam_3d.png, noise_budget.png
        hardware/output/chain_comparison.png   (with --compare / --all-orders)

HARDWARE_MODEL.md explains every input, every output, and what to expect.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from hydroline.hardware.params import HardwareConfigError  # noqa: E402
from hydroline.hardware.system import (evaluate, load_goals, load_hardware,  # noqa: E402
                                       thresholds_dict, write_performance_yaml,
                                       write_response_csv, write_thresholds_yaml)
from hydroline.theme import THEMES  # noqa: E402

DEFAULT_HARDWARE = HERE / "hardware" / "hardware_parameters.yaml"


def _order_arg(text: str) -> list[str] | None:
    return None if text == "current" else [s.strip() for s in text.split(",") if s.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("hardware", nargs="?", type=Path, default=DEFAULT_HARDWARE,
                   help="hardware parameters YAML (default: %(default)s)")
    p.add_argument("--goals", type=Path,
                   help="observing goals YAML (default: observing_goals.yaml next to "
                        "the hardware file, if present)")
    p.add_argument("-o", "--outdir", type=Path,
                   help="output folder (default: output/ next to the hardware file)")
    p.add_argument("--order", type=_order_arg,
                   help="comma-separated component ids: run the chain in this order "
                        "instead of chain_position")
    p.add_argument("--compare", nargs="+", metavar="ORDER",
                   help="evaluate several orders and plot them together; each is a "
                        "comma-separated id list, or 'current'")
    p.add_argument("--all-orders", action="store_true",
                   help="try every order of the chain (SDR kept last) and rank them")
    p.add_argument("--snr-target", type=float,
                   help="override the goals file's snr_target")
    p.add_argument("--theme", choices=sorted(THEMES), default="light")
    p.add_argument("--no-plots", action="store_true")
    return p


def describe(perf) -> str:
    c, pb, hw = perf.center, perf.passband, perf.hw
    lines = [
        f"{hw.name}",
        f"  chain        {'  →  '.join(x.id for x in hw.chain)}",
        f"  total gain   {c['gain_to_sdr_db']:.2f} dB to the SDR input",
        f"  passband     {pb.lo_3db_hz/1e6:.3f}-{pb.hi_3db_hz/1e6:.3f} MHz (-3 dB), "
        f"noise bandwidth {pb.noise_bandwidth_hz/1e6:.3f} MHz "
        f"(nominal edges from {pb.nominal_lo_by} / {pb.nominal_hi_by})",
        f"  T_rec        {c['t_rec_k']:.1f} K     T_A {c['t_a_k']:.1f} K     "
        f"T_sys {c['t_sys_k']:.1f} K",
        f"  A_eff        {c['a_eff_m2']:.3f} m²   HPBW {perf.beam.hpbw_e_deg:.2f}° (E) × "
        f"{perf.beam.hpbw_h_deg:.2f}° (H)",
        f"  SEFD         {c['sefd_jy']:.4g} Jy",
        f"  beam         Dec {perf.beam_dec_deg:+.2f}°, strip {perf.hpbw_ns_deg:.2f}° wide, "
        f"transit {perf.transit_s/60:.1f} min",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    hw_path: Path = args.hardware
    outdir: Path = args.outdir or hw_path.parent / "output"
    goals_path = args.goals or hw_path.parent / "observing_goals.yaml"
    theme = THEMES[args.theme]

    try:
        hw = load_hardware(hw_path, order=args.order)
        perf = evaluate(hw)
        goals = load_goals(goals_path) if goals_path.exists() else None
        if goals is None and args.snr_target is not None:
            goals = {"snr_target": args.snr_target, "integrations": ["transit"],
                     "line_width": 200.0, "base_dir": hw_path.parent}
        if goals is not None and args.snr_target is not None:
            goals["snr_target"] = args.snr_target
    except HardwareConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    outdir.mkdir(parents=True, exist_ok=True)
    print(describe(perf))
    for w in hw.warnings:
        print(f"  ! {w}")
    for a in hw.assumptions:
        print(f"  ~ assumed: {a}")
    if hw.placeholders:
        print(f"  ~ {len(hw.placeholders)} inputs are placeholders/estimates "
              f"(listed in performance_parameters.yaml)")

    written = [write_performance_yaml(perf, outdir / "performance_parameters.yaml"),
               write_response_csv(perf, outdir / "frequency_response.csv")]
    if goals is not None:
        written.append(write_thresholds_yaml(perf, goals, outdir / "detection_thresholds.yaml"))
        t = thresholds_dict(perf, goals)
        print(f"\n  For SNR {t['snr_target']:g}  (S = SNR · SEFD / sqrt(n_pol · Δν · τ)):")
        print(f"    {'integration':>14}  {'continuum':>11}  {'HI, 1 chan':>11}  "
              f"{'HI, matched':>11}")
        for r in t["required_flux"]:
            print(f"    {r['integration']:>14}  {r['continuum_flux_jy']:>9.3g} Jy  "
                  f"{r['line_channel_peak_jy']:>9.3g} Jy  {r['line_matched_peak_jy']:>9.3g} Jy")
        db = t["database_index"]
        print(f"    → index the catalogues at NVSS S1.4 ≥ {db['nvss_s1400_min']['value']:.3g} Jy, "
              f"HI peak ≥ {db['hi_peak_min']['value']:.3g} Jy ({db['integration']})")
    snr_t = goals["snr_target"] if goals else 3.0

    if not args.no_plots:
        from hydroline.hardware.plots import (plot_beam, plot_frequency_response,
                                              plot_noise_budget, plot_order_comparison)
        written += [plot_frequency_response(perf, outdir / "frequency_response.png", theme, snr_t),
                    plot_beam(perf, outdir / "beam_3d.png", theme),
                    plot_noise_budget(perf, outdir / "noise_budget.png", theme)]

    comparisons = []
    try:
        if args.compare:
            for spec in args.compare:
                order = _order_arg(spec)
                p = evaluate(load_hardware(hw_path, order=order), reuse=perf)
                comparisons.append((" > ".join(p.hw.order), p))
        elif args.all_orders:
            movable = [c.id for c in hw.chain if c.type != "sdr"]
            tail = [c.id for c in hw.chain if c.type == "sdr"]
            if len(movable) > 7:
                print("error: --all-orders is limited to 7 movable components", file=sys.stderr)
                return 2
            for perm in itertools.permutations(movable):
                order = list(perm) + tail
                try:
                    p = evaluate(load_hardware(hw_path, order=order), reuse=perf)
                except HardwareConfigError:
                    continue                    # e.g. centre falls out of band
                comparisons.append((" > ".join(order), p))
            comparisons.sort(key=lambda lp: lp[1].t_sys_k)
            print(f"\n  {len(comparisons)} orders, best first (T_sys at the line):")
            for k, (lab, p) in enumerate(comparisons):
                if k < 8 or k == len(comparisons) - 1:
                    mark = "   ← current" if p.hw.order == hw.order else ""
                    print(f"    {p.t_sys_k:7.1f} K   {lab}{mark}")
                elif k == 8:
                    print("    ...")
            cur = next((lp for lp in comparisons if lp[1].hw.order == hw.order), None)
            shown = comparisons[:5] + ([cur] if cur and cur not in comparisons[:5] else []) \
                + [comparisons[-1]]
            comparisons = shown
    except HardwareConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if comparisons:
        if args.compare:
            print("\n  Comparison (T_sys at the line, SEFD):")
            for lab, p in comparisons:
                print(f"    {p.t_sys_k:7.1f} K  {p.sefd_jy:10.4g} Jy   {lab}")
        if not args.no_plots:
            written.append(plot_order_comparison(comparisons, outdir / "chain_comparison.png",
                                                 theme))

    print("\n  wrote:")
    for w in written:
        print(f"    {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
