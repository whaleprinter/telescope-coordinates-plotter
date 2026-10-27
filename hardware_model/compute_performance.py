#!/usr/bin/env python3
"""hardware_model: hardware parameters -> performance parameters.

    python3 hardware_model/compute_performance.py            # the files in this folder
    python3 hardware_model/compute_performance.py my_rig.yaml
    python3 hardware_model/compute_performance.py --order lna,feed_cable,bandpass_filter,coax_run,sdr
    python3 hardware_model/compute_performance.py --compare current lna,feed_cable,bandpass_filter,coax_run,sdr
    python3 hardware_model/compute_performance.py --all-orders
    python3 -m hardware_model ...                            # the same thing

Reads   hardware_model/hardware_parameters.yaml   (+ observing_goals.yaml beside it)
Writes  hardware_model/output/performance_parameters.yaml, detection_thresholds.yaml,
        frequency_response.csv, frequency_response.png, beam_3d.png,
        noise_budget.png, chain_comparison.png (with --compare / --all-orders)

All equations are in hardware_model/performance.py; hardware_model/README.md
explains the inputs, outputs and what to expect.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if __package__ in (None, ""):                 # run as a script: make the package importable
    sys.path.insert(0, str(HERE.parent))

from hardware_model.inputs import HardwareConfigError, Goals, load_goals, load_hardware  # noqa: E402
from hardware_model.performance import evaluate, evaluate_bench  # noqa: E402
from hardware_model.report import (describe, describe_bench, thresholds_dict,  # noqa: E402
                                   write_bench, write_performance_yaml, write_response_csv,
                                   write_thresholds_yaml)

DEFAULT_HARDWARE = HERE / "hardware_parameters.yaml"


def _order_arg(text: str) -> list[str] | None:
    return None if text == "current" else [s.strip() for s in text.split(",") if s.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("hardware", nargs="?", type=Path, default=DEFAULT_HARDWARE,
                   help="hardware parameters YAML (default: %(default)s)")
    p.add_argument("--goals", type=Path, help="observing goals YAML (default: "
                   "observing_goals.yaml next to the hardware file, if present)")
    p.add_argument("-o", "--outdir", type=Path,
                   help="output folder (default: output/ next to the hardware file)")
    p.add_argument("--order", type=_order_arg, help="comma-separated component ids: run "
                   "the chain in this order instead of chain_position")
    p.add_argument("--compare", nargs="+", metavar="ORDER", help="evaluate several orders "
                   "and plot them together; each is a comma-separated id list, or 'current'")
    p.add_argument("--all-orders", action="store_true",
                   help="try every order of the chain (SDR kept last) and rank them")
    p.add_argument("--snr-target", type=float, help="override the goals file's snr_target")
    p.add_argument("--theme", choices=["light", "dark"], default="light")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--test", nargs="+", metavar="ID",
                   help="bench files: run only these bench_tests ids")
    return p


def run_bench(hw, args, outdir: Path) -> int:
    """Bench tests: expected VNA/SDR readings for components with no antenna."""
    tests = hw.bench_tests
    if args.test:
        unknown = [t for t in args.test if t not in {x.id for x in tests}]
        if unknown:
            print(f"error: no bench test {', '.join(unknown)} (have: "
                  f"{', '.join(x.id for x in tests)})", file=sys.stderr)
            return 2
        tests = [x for x in tests if x.id in args.test]
    print(f"{hw.name}\n  bench tests -- "
          f"{hw.instrument_model or 'no instrument block'}, "
          f"{hw.sweep_hz[0]/1e6:g}-{hw.sweep_hz[1]/1e6:g} MHz, {hw.sweep_hz[2]} points")
    for w in hw.warnings:
        print(f"  ! {w}")
    if hw.placeholders:
        print(f"  ~ {len(hw.placeholders)} inputs are placeholders/estimates "
              f"(listed in each expected.yaml)")
    written = []
    for t in tests:
        try:
            res = evaluate_bench(hw, t)
        except HardwareConfigError as exc:
            print(f"error in bench test {t.id!r}: {exc}", file=sys.stderr)
            return 2
        print(describe_bench(res))
        d = outdir / "bench" / t.id
        written += write_bench(res, hw, d)
        if not args.no_plots:
            from hardware_model.plots import STYLES, plot_bench
            written.append(plot_bench(res, d / "expected.png", STYLES[args.theme]))
    print("\n  wrote:")
    for w in written:
        print(f"    {w}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    hw_path: Path = args.hardware
    outdir: Path = args.outdir or hw_path.parent / "output"
    goals_path = args.goals or hw_path.parent / "observing_goals.yaml"

    try:
        hw = load_hardware(hw_path, order=args.order)
    except HardwareConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not hw.has_antenna:
        return run_bench(hw, args, outdir)
    if args.test:
        return run_bench(hw, args, outdir)
    try:
        perf = evaluate(hw)
        goals = load_goals(goals_path) if goals_path.exists() else None
        if args.snr_target is not None:
            goals = goals or Goals(args.snr_target, ["transit"], 200.0, hw_path.parent)
            goals.snr_target = args.snr_target
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
        print(f"\n  For SNR {t['snr_target']:g}  (Eq. 34: S = SNR · SEFD / sqrt(n_pol · Δν · τ)):")
        print(f"    {'integration':>14}  {'continuum':>11}  {'HI, 1 chan':>11}  {'HI, matched':>11}")
        for r in t["required_flux"]:
            print(f"    {r['integration']:>14}  {r['continuum_flux_jy']:>9.3g} Jy  "
                  f"{r['line_channel_peak_jy']:>9.3g} Jy  {r['line_matched_peak_jy']:>9.3g} Jy")
        db = t["database_index"]
        print(f"    → index the catalogues at NVSS S1.4 ≥ {db['nvss_s1400_min']['value']:.3g} Jy, "
              f"HI peak ≥ {db['hi_peak_min']['value']:.3g} Jy ({db['integration']})")
    snr_t = goals.snr_target if goals else 3.0

    if not args.no_plots:
        from hardware_model.plots import (STYLES, plot_beam, plot_frequency_response,
                                          plot_noise_budget)
        style = STYLES[args.theme]
        written += [plot_frequency_response(perf, outdir / "frequency_response.png", style, snr_t),
                    plot_beam(perf, outdir / "beam_3d.png", style),
                    plot_noise_budget(perf, outdir / "noise_budget.png", style)]

    comparisons = []
    try:
        if args.compare:
            for spec in args.compare:
                p = evaluate(load_hardware(hw_path, order=_order_arg(spec)), reuse=perf)
                comparisons.append((" > ".join(p.hw.order), p))
            print("\n  Comparison (T_sys at the line, SEFD):")
            for lab, p in comparisons:
                print(f"    {p.t_sys_k:7.1f} K  {p.sefd_jy:10.4g} Jy   {lab}")
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
                    continue                          # e.g. centre falls out of band
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
            comparisons = (comparisons[:5] + ([cur] if cur and cur not in comparisons[:5] else [])
                           + [comparisons[-1]])
    except HardwareConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if comparisons and not args.no_plots:
        from hardware_model.plots import STYLES, plot_order_comparison
        written.append(plot_order_comparison(comparisons, outdir / "chain_comparison.png",
                                             STYLES[args.theme]))

    print("\n  wrote:")
    for w in written:
        print(f"    {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
