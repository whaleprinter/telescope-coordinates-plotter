"""Figures.  No physics happens here: every curve is a quantity computed in
``performance.py`` (the beam map evaluates Eq. 25's pattern on a grid).

The colours match hydroline's light/dark themes but are defined here, so this
program does not depend on hydroline.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .performance import (HI_PROTECTED_BAND_HZ, HI_REST_HZ, Performance,  # noqa: E402
                          required_flux_jy)


@dataclass(frozen=True)
class Style:
    surface: str
    panel: str
    ink: str
    ink_secondary: str
    ink_muted: str
    grid: str
    series: tuple[str, str, str]
    categorical: tuple[str, ...]


STYLES = {
    "light": Style("#fcfcfb", "#ffffff", "#0b0b0b", "#52514e", "#9a9892", "#e6e5e1",
                   ("#2a78d6", "#eb6834", "#1baf7a"),
                   ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
                    "#4a3aa7", "#e34948")),
    "dark": Style("#16181c", "#1a1c21", "#ffffff", "#c3c2b7", "#6f7178", "#2b2e34",
                  ("#3987e5", "#d95926", "#199e70"),
                  ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300",
                   "#9085e9", "#e66767")),
}


@contextmanager
def styled(s: Style):
    rc = {
        "figure.facecolor": s.surface, "savefig.facecolor": s.surface,
        "axes.facecolor": s.panel, "axes.edgecolor": s.grid,
        "axes.labelcolor": s.ink_secondary, "axes.titlecolor": s.ink,
        "axes.grid": True, "axes.axisbelow": True, "axes.titlesize": 11,
        "axes.titleweight": "semibold", "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": s.grid, "grid.linewidth": 0.6,
        "xtick.color": s.ink_muted, "ytick.color": s.ink_muted,
        "xtick.labelcolor": s.ink_secondary, "ytick.labelcolor": s.ink_secondary,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "text.color": s.ink,
        "legend.frameon": False, "legend.fontsize": 8, "legend.labelcolor": s.ink_secondary,
        "font.size": 9, "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "figure.dpi": 130, "savefig.dpi": 160, "savefig.bbox": "tight",
    }
    with matplotlib.rc_context(rc):
        yield s


def _decorate(ax, perf: Performance, s: Style, first: bool = False):
    pb = perf.passband
    ax.axvspan(pb.lo_3db_hz / 1e6, pb.hi_3db_hz / 1e6, color=s.series[0], alpha=0.08, lw=0,
               label="processed band (-3 dB)" if first else None)
    lo, hi = HI_PROTECTED_BAND_HZ
    ax.axvspan(lo / 1e6, hi / 1e6, ymin=0.97, ymax=1.0, color=s.ink_muted, alpha=0.5, lw=0,
               label="protected 1400-1427 MHz" if first else None)
    ax.axvline(HI_REST_HZ / 1e6, color=s.ink_muted, lw=0.9,
               label="HI 1420.406 MHz" if first else None)


def plot_frequency_response(perf: Performance, path: Path, s: Style = STYLES["light"],
                            snr_target: float = 3.0) -> Path:
    sw, hw, i0 = perf.sweep, perf.hw, perf.index
    f = sw.freq_hz / 1e6
    with styled(s):
        fig, axes = plt.subplots(4, 1, figsize=(13, 13), sharex=True)
        cat = s.categorical
        legend = dict(loc="upper left", bbox_to_anchor=(1.01, 1.0))

        ax = axes[0]                                         # gain (Eqs. 5-9)
        _decorate(ax, perf, s, first=True)
        for k, st in enumerate(x for x in sw.stages if x.component.type != "sdr"):
            ax.plot(f, st.cumulative_gain_db, lw=1.0, color=cat[(k + 1) % len(cat)],
                    label=f"after {st.component.id}")
        ax.plot(f, sw.gain_to_sdr_db, color=s.ink, lw=2.2,
                label=f"total to SDR input ({sw.gain_to_sdr_db[i0]:.1f} dB at HI)")
        ax.plot(f, sw.response_db, color=s.series[0], lw=1.4,
                label="incl. feed band + SDR selectivity")
        top = float(np.nanmax(sw.gain_to_sdr_db))
        ax.set_ylim(top - 45, top + 8)
        ax.set_ylabel("gain (dB)")
        ax.set_title("Gain through the chain")
        ax.legend(**legend)

        ax = axes[1]                                         # temperatures (Eqs. 4, 27-30)
        _decorate(ax, perf, s)
        for arr, lab, col, lw in ((sw.t_sys_k, "T_sys", s.ink, 2.2),
                                  (sw.t_rec_k, "T_rec (Friis)", s.series[1], 1.6),
                                  (sw.t_a_k, "T_A (sky + ground + feed loss)", s.series[2], 1.6),
                                  (sw.t_sky_k, "T_sky", s.series[0], 1.2)):
            ax.plot(f, arr, color=col, lw=lw, label=f"{lab}: {arr[i0]:.1f} K")
        ax.set_yscale("log")
        ax.set_ylim(max(1.0, 0.5 * float(sw.t_sky_k.min())), 20 * perf.t_sys_k)
        ax.set_ylabel("temperature (K)")
        ax.set_title("Noise temperatures (referred to the feed terminals)")
        ax.legend(**legend)

        ax = axes[2]                                         # Friis terms (Eq. 4)
        _decorate(ax, perf, s)
        ax.stackplot(f, [st.contribution_k for st in sw.stages],
                     labels=[f"{st.component.id}: {st.contribution_k[i0]:.1f} K"
                             for st in sw.stages],
                     colors=[cat[(k + 1) % len(cat)] for k in range(len(sw.stages))], alpha=0.85)
        ax.set_ylim(0, 2.5 * perf.at_center("t_rec_k"))
        ax.set_ylabel("contribution to T_rec (K)")
        ax.set_title("Where the receiver noise comes from  (T_i / G_1...G_{i-1})")
        ax.legend(**legend)

        ax = axes[3]                                         # sensitivity (Eqs. 31, 34)
        _decorate(ax, perf, s)
        s_req = required_flux_jy(snr_target, sw.sefd_jy, perf.channel_hz, perf.transit_s,
                                 hw.n_pol, hw.k_s)
        ax.plot(f, sw.sefd_jy, color=s.ink, lw=2.0, label=f"SEFD: {sw.sefd_jy[i0]:.3g} Jy")
        ax.plot(f, s_req, color=s.series[1], lw=1.6,
                label=(f"flux for SNR {snr_target:g}:\none {perf.channel_hz/1e3:.2f} kHz "
                       f"channel, one {perf.transit_s/60:.0f}-min\ntransit: {s_req[i0]:.3g} Jy"))
        ax.set_yscale("log")
        ax.set_ylim(0.3 * float(np.nanmin(s_req)), 30 * perf.sefd_jy)
        ax.set_ylabel("flux density (Jy)")
        ax.set_xlabel("frequency (MHz)")
        ax.set_title("Sensitivity")
        ax.legend(**legend)
        ax.set_xlim(f[0], f[-1])

        fig.suptitle(f"{hw.name}\nchain: {'  >  '.join(c.id for c in hw.chain)}",
                     fontsize=11, y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.965))
        fig.savefig(path)
        plt.close(fig)
    return path


def plot_beam(perf: Performance, path: Path, s: Style = STYLES["light"]) -> Path:
    """Eq. 25's pattern in 3-D, as a footprint on the sky, and in its principal planes."""
    ant = perf.antenna
    bm = ant.beam
    hp = max(ant.hpbw_e_deg, ant.hpbw_h_deg)
    r_max = min(2.5 * hp, float(bm.theta_deg[-1]))
    x = np.linspace(-r_max, r_max, 181)                 # toward east (deg)
    X, Y = np.meshgrid(x, x)                            # Y toward north
    TH = np.hypot(X, Y)
    P = bm.power_azimuth(TH, np.degrees(np.arctan2(X, Y)))
    P[TH > bm.theta_deg[-1]] = np.nan
    PdB = np.clip(10 * np.log10(np.maximum(P, 1e-6)), -40, 0)

    with styled(s):
        fig = plt.figure(figsize=(16.5, 5.4))
        ax3 = fig.add_subplot(1, 3, 1, projection="3d")
        ax3.plot_surface(X, Y, PdB, cmap="viridis", rstride=2, cstride=2, lw=0,
                         antialiased=True, vmin=-40, vmax=0)
        ax3.set_xlabel("east offset (deg)")
        ax3.set_ylabel("north offset (deg)")
        ax3.set_zlabel("power (dB)")
        ax3.set_zlim(-40, 0)
        ax3.set_title("Power pattern in 3-D")
        ax3.set_facecolor(s.panel)
        ax3.view_init(elev=28, azim=-60)

        ax = fig.add_subplot(1, 3, 2)
        cs = ax.contourf(X, Y, PdB, levels=np.arange(-40, 1, 2.5), cmap="viridis")
        ax.contour(X, Y, PdB, levels=[-3], colors=[s.ink], linewidths=2.0, linestyles="solid")
        ax.contour(X, Y, PdB, levels=[-20, -10], colors=[s.ink_muted], linewidths=0.8,
                   linestyles="solid")
        ax.set_aspect("equal")
        ax.set_xlabel("east offset (deg)")
        ax.set_ylabel("north offset (deg, = +Dec)")
        ax.set_title("Footprint on the sky (−3 dB contour bold)")
        ax.text(0.02, 0.02,
                f"N–S {perf.hpbw_ns_deg:.2f}°  (declination strip)\n"
                f"E–W {perf.hpbw_ew_deg:.2f}°  (transit {perf.transit_s/60:.1f} min)\n"
                f"E-plane azimuth {bm.e_plane_azimuth_deg:.0f}°",
                transform=ax.transAxes, fontsize=8, color="white", va="bottom")
        fig.colorbar(cs, ax=ax, shrink=0.8, label="dB")

        ax = fig.add_subplot(1, 3, 3)
        th = bm.theta_deg
        m = th <= min(4 * hp, th[-1])
        for phi, lab, col in ((0.0, f"E-plane  HPBW {ant.hpbw_e_deg:.2f}°", s.series[0]),
                              (90.0, f"H-plane  HPBW {ant.hpbw_h_deg:.2f}°", s.series[1])):
            db = 10 * np.log10(np.maximum(bm.cut(phi)[m], 1e-6))
            ax.plot(np.concatenate([-th[m][::-1], th[m]]), np.concatenate([db[::-1], db]),
                    color=col, lw=1.6, label=lab)
        ax.axhline(-3, color=s.ink_muted, lw=0.8)
        ax.set_ylim(-50, 2)
        ax.set_xlabel("angle from beam axis (deg)")
        ax.set_ylabel("power (dB)")
        sl = [v for v in (ant.sidelobe_e_db, ant.sidelobe_h_db) if v is not None]
        ax.set_title("Principal-plane cuts" + (f"  (1st sidelobe {max(sl):.1f} dB)" if sl else ""))
        ax.legend(loc="upper right")
        fig.tight_layout(w_pad=4.0)
        fig.savefig(path)
        plt.close(fig)
    return path


def plot_noise_budget(perf: Performance, path: Path, s: Style = STYLES["light"]) -> Path:
    """T_sys split into the three Eq. 29 terms and the Eq. 4 stage contributions."""
    i = perf.index
    sky, ground, feed = (float(t[i]) for t in perf.sweep.t_a_terms)
    rows = [("sky (CMB + Galaxy + atmosphere)", sky), ("ground pickup (spillover)", ground),
            ("feed ohmic loss", feed)]
    rows += [(f"{st.component.id} ({st.component.type})", float(st.contribution_k[i]))
             for st in perf.sweep.stages]
    total = sum(v for _, v in rows)
    with styled(s):
        fig, ax = plt.subplots(figsize=(9, 0.45 * len(rows) + 1.6))
        y = np.arange(len(rows))[::-1]
        ax.barh(y, [v for _, v in rows], height=0.6,
                color=[s.series[2]] * 3 + [s.series[1]] * (len(rows) - 3))
        for yi, (_, v) in zip(y, rows):
            ax.text(v + total * 0.01, yi, f"{v:.1f} K  ({100 * v / total:.0f}%)",
                    va="center", fontsize=8, color=s.ink_secondary)
        ax.set_yticks(y)
        ax.set_yticklabels([lab for lab, _ in rows])
        ax.set_xlim(0, max(v for _, v in rows) * 1.3)
        ax.set_xlabel("contribution to T_sys (K)")
        ax.set_title(f"Noise budget at {perf.hw.center_hz/1e6:.3f} MHz:  T_sys = {total:.1f} K"
                     f"  (green: antenna side, orange: receiver)")
        ax.grid(axis="y", visible=False)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
    return path


def plot_order_comparison(results: Sequence[tuple[str, Performance]], path: Path,
                          s: Style = STYLES["light"]) -> Path:
    with styled(s):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.4, 1]})
        cat = s.categorical
        for k, (label, perf) in enumerate(results):
            a1.plot(perf.sweep.freq_hz / 1e6, perf.sweep.t_sys_k, color=cat[k % len(cat)],
                    lw=1.6, label=f"{label}:  {perf.t_sys_k:.1f} K")
        a1.axvline(HI_REST_HZ / 1e6, color=s.ink_muted, lw=0.9)
        a1.set_yscale("log")
        a1.set_xlabel("frequency (MHz)")
        a1.set_ylabel("T_sys (K)")
        a1.set_title("System temperature for each chain order")
        a1.legend(loc="upper left", fontsize=7)
        base = results[0][1].t_sys_k
        y = np.arange(len(results))[::-1]
        vals = [p.t_sys_k for _, p in results]
        a2.barh(y, vals, color=[cat[k % len(cat)] for k in range(len(results))], height=0.6)
        for yi, (_, p) in zip(y, results):
            a2.text(p.t_sys_k * 1.01, yi, f"{p.t_sys_k:.1f} K   SEFD {p.sefd_jy:,.0f} Jy   "
                    f"time ×{(p.t_sys_k / base) ** 2:.2f}", va="center", fontsize=7,
                    color=s.ink_secondary)
        a2.set_yticks(y)
        a2.set_yticklabels([lab for lab, _ in results], fontsize=7)
        a2.set_xlim(0, max(vals) * 1.9)
        a2.set_xlabel("T_sys at the HI line (K)")
        a2.set_title("Integration time scales as T_sys²  (Eq. 34)")
        a2.grid(axis="y", visible=False)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
    return path


def plot_bench(res, path: Path, s: Style = STYLES["light"]) -> Path:
    """Expected (and measured) VNA traces for one bench test, with limit lines."""
    t, f = res.test, res.freq_hz / 1e6
    with styled(s):
        fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
        legend = dict(loc="upper left", bbox_to_anchor=(1.01, 1.0))

        ax = axes[0]
        ax.axvline(HI_REST_HZ / 1e6, color=s.ink_muted, lw=0.9, label="HI 1420.406 MHz")
        if t.type == "vna":
            ax.plot(f, res.s21_db, color=s.ink, lw=2.0, label="S21 expected (datasheet model)")
            if res.s21_shown_db is not None:
                ax.plot(f, res.s21_shown_db, color=s.ink_muted, lw=1.2,
                        label="what the VNA can show (Eq. 41 floor)")
            if res.measured_s21_db is not None:
                ax.plot(res.measured_freq_hz / 1e6, res.measured_s21_db, color=s.series[1],
                        lw=1.6, label="S21 measured")
            ax.set_title(f"{t.id}: S21   ({' > '.join(c.id for c in t.chain)})")
        else:
            gain = res.stages[-2].cumulative_gain_db if len(res.stages) > 1 else res.s21_db
            ax.plot(f, gain, color=s.ink, lw=2.0, label="gain up to the SDR input")
            ax.set_title(f"{t.id}: analog gain ahead of the SDR")
        for c in (x for x in res.limits if x.quantity == "s21"):
            for v, lab in ((c.min_db, "min"), (c.max_db, "max")):
                if v is None:
                    continue
                lo, hi = c.lo_hz / 1e6, c.hi_hz / 1e6
                if hi - lo < 1:
                    ax.plot([lo], [v], marker="v" if lab == "max" else "^", ms=9,
                            color=s.series[2], ls="none")
                else:
                    ax.plot([lo, hi], [v, v], color=s.series[2], lw=2.5)
        ax.plot([], [], color=s.series[2], lw=2.5, marker="^", label="limit lines (see expected.yaml)")
        ax.set_ylabel("dB")
        ax.legend(**legend)

        ax = axes[1]
        if t.type == "vna":
            if res.s11_db is not None:
                ax.plot(f, res.s11_db, color=s.series[0], lw=1.8, label="S11 expected (input)")
            if res.s22_db is not None:
                ax.plot(f, res.s22_db, color=s.series[0], lw=1.0, alpha=0.6,
                        label="S22 expected (output; reverse the DUT)")
            if res.measured_s11_db is not None:
                ax.plot(res.measured_freq_hz / 1e6, res.measured_s11_db, color=s.series[1],
                        lw=1.6, label="S11 measured")
            for c in (x for x in res.limits if x.quantity == "s11"):
                if c.max_db is not None:
                    ax.plot([c.lo_hz / 1e6, c.hi_hz / 1e6], [c.max_db] * 2, color=s.series[2], lw=2.5,
                            label="S11 limit")
            ax.set_ylabel("reflection (dB)")
            ax.set_title("Return loss  (Eq. 39, from datasheet VSWR)")
            ax.set_ylim(-40, 0)
        else:
            ax.plot(f, res.noise["ratio_db_vs_f"], color=s.series[1], lw=1.8,
                    label=f"expected noise-floor rise: {res.noise['ratio_db']:.1f} dB at HI")
            ax.set_ylabel("rise (dB)")
            ax.set_title("Noise floor with the chain vs the SDR alone (Eq. 42)")
        ax.set_xlabel("frequency (MHz)")
        ax.set_xlim(f[0], f[-1])
        if ax.get_legend_handles_labels()[0]:
            ax.legend(**legend)
        else:
            ax.text(0.5, 0.5, "no published input/output match for this part:\n"
                    "the VNA's S11 trace is the first measurement of it",
                    transform=ax.transAxes, ha="center", va="center", color=s.ink_muted)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
    return path
