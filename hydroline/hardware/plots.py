"""Figures for the hardware model: frequency response, 3-D beam, noise
budget, and chain-order comparisons."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..constants import HI_PROTECTED_BAND_HZ, HI_REST_HZ  # noqa: E402
from ..theme import LIGHT, Theme, styled  # noqa: E402
from .system import Performance, required_flux_jy  # noqa: E402


def _mhz(f):
    return np.asarray(f) / 1e6


def _decorate_freq(ax, perf: Performance, theme: Theme, first: bool = False):
    pb = perf.passband
    ax.axvspan(pb.lo_3db_hz / 1e6, pb.hi_3db_hz / 1e6, color=theme.series[0], alpha=0.08,
               lw=0, label="processed band (-3 dB)" if first else None)
    lo, hi = HI_PROTECTED_BAND_HZ
    ax.axvspan(lo / 1e6, hi / 1e6, ymin=0.97, ymax=1.0, color=theme.ink_muted, alpha=0.5,
               lw=0, label="protected 1400-1427 MHz" if first else None)
    ax.axvline(HI_REST_HZ / 1e6, color=theme.ink_muted, lw=0.9,
               label="HI 1420.406 MHz" if first else None)


def plot_frequency_response(perf: Performance, path: Path, theme: Theme = LIGHT,
                            snr_target: float = 3.0) -> Path:
    sw, hw = perf.sweep, perf.hw
    f = _mhz(sw.freq_hz)
    i0 = perf.center["index"]
    with styled(theme):
        fig, axes = plt.subplots(4, 1, figsize=(13, 13), sharex=True)
        cat = theme.categorical

        # 1 -- gain ----------------------------------------------------------
        ax = axes[0]
        _decorate_freq(ax, perf, theme, first=True)
        analog = [s for s in sw.cascade.stages if s.component.type != "sdr"]
        for k, s in enumerate(analog):
            ax.plot(f, s.cumulative_gain_db, lw=1.0, color=cat[(k + 1) % len(cat)],
                    label=f"after {s.component.id}")
        ax.plot(f, sw.cascade.gain_to_sdr_db, color=theme.ink, lw=2.2,
                label=f"total to SDR input ({sw.cascade.gain_to_sdr_db[i0]:.1f} dB at HI)")
        ax.plot(f, sw.response_db, color=theme.series[0], lw=1.4,
                label="incl. feed band + SDR selectivity")
        top = float(np.nanmax(sw.cascade.gain_to_sdr_db))
        ax.set_ylim(top - 45, top + 8)
        ax.set_ylabel("gain (dB)")
        ax.set_title("Gain through the chain")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))

        # 2 -- temperatures ---------------------------------------------------
        ax = axes[1]
        _decorate_freq(ax, perf, theme)
        for arr, lab, col, lw in ((sw.t_sys_k, "T_sys", theme.ink, 2.2),
                                  (sw.t_rec_k, "T_rec (Friis)", theme.series[1], 1.6),
                                  (sw.t_a_k, "T_A (sky + ground + feed loss)", theme.series[2], 1.6),
                                  (sw.t_sky_k, "T_sky", theme.series[0], 1.2)):
            ax.plot(f, arr, color=col, lw=lw, label=f"{lab}: {arr[i0]:.1f} K")
        ax.set_yscale("log")
        ax.set_ylim(max(1.0, 0.5 * float(sw.t_sky_k.min())), 20 * perf.t_sys_k)
        ax.set_ylabel("temperature (K)")
        ax.set_title("Noise temperatures (referred to the feed terminals)")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))

        # 3 -- Friis contributions --------------------------------------------
        ax = axes[2]
        _decorate_freq(ax, perf, theme)
        stack = [s.contribution_k for s in sw.cascade.stages]
        labels = [f"{s.component.id}: {s.contribution_k[i0]:.1f} K" for s in sw.cascade.stages]
        ax.stackplot(f, stack, labels=labels,
                     colors=[cat[(k + 1) % len(cat)] for k in range(len(stack))], alpha=0.85)
        ax.set_ylim(0, 2.5 * perf.center["t_rec_k"])
        ax.set_ylabel("contribution to T_rec (K)")
        ax.set_title("Where the receiver noise comes from  (T_i / G_1...G_{i-1})")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))

        # 4 -- sensitivity ----------------------------------------------------
        ax = axes[3]
        _decorate_freq(ax, perf, theme)
        s_req = required_flux_jy(snr_target, sw.sefd_jy, hw.channel_hz, perf.transit_s,
                                 hw.n_pol, hw.k_s)
        ax.plot(f, sw.sefd_jy, color=theme.ink, lw=2.0,
                label=f"SEFD: {sw.sefd_jy[i0]:.3g} Jy")
        ax.plot(f, s_req, color=theme.series[1], lw=1.6,
                label=(f"flux for SNR {snr_target:g}:\none {hw.channel_hz/1e3:.2f} kHz "
                       f"channel, one {perf.transit_s/60:.0f}-min\ntransit: {s_req[i0]:.3g} Jy"))
        ax.set_yscale("log")
        ax.set_ylim(0.3 * float(np.nanmin(s_req)), 30 * perf.sefd_jy)
        ax.set_ylabel("flux density (Jy)")
        ax.set_xlabel("frequency (MHz)")
        ax.set_title("Sensitivity")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
        ax.set_xlim(f[0], f[-1])

        order = "  >  ".join(c.id for c in hw.chain)
        fig.suptitle(f"{hw.name}\nchain: {order}", fontsize=11, y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.965))
        fig.savefig(path)
        plt.close(fig)
    return path


def plot_beam(perf: Performance, path: Path, theme: Theme = LIGHT) -> Path:
    """The beam in 3-D, as a footprint on the sky, and in its principal planes."""
    bm = perf.beam
    hp = max(bm.hpbw_e_deg, bm.hpbw_h_deg)
    r_max = min(2.5 * hp, float(bm.theta_deg[-1]))
    n = 181
    x = np.linspace(-r_max, r_max, n)               # toward east (deg)
    X, Y = np.meshgrid(x, x)                        # Y toward north
    TH = np.hypot(X, Y)
    AZ = np.degrees(np.arctan2(X, Y))               # compass azimuth
    P = bm.power_azimuth(TH, AZ)
    P[TH > bm.theta_deg[-1]] = np.nan
    PdB = np.clip(10 * np.log10(np.maximum(P, 1e-6)), -40, 0)

    with styled(theme):
        fig = plt.figure(figsize=(16.5, 5.4))
        ax3 = fig.add_subplot(1, 3, 1, projection="3d")
        ax3.plot_surface(X, Y, PdB, cmap="viridis", rstride=2, cstride=2, lw=0,
                         antialiased=True, vmin=-40, vmax=0)
        ax3.set_xlabel("east offset (deg)")
        ax3.set_ylabel("north offset (deg)")
        ax3.set_zlabel("power (dB)")
        ax3.set_zlim(-40, 0)
        ax3.set_title("Power pattern in 3-D")
        ax3.set_facecolor(theme.panel)
        ax3.view_init(elev=28, azim=-60)

        ax = fig.add_subplot(1, 3, 2)
        cs = ax.contourf(X, Y, PdB, levels=np.arange(-40, 1, 2.5), cmap="viridis")
        ax.contour(X, Y, PdB, levels=[-3], colors=[theme.ink], linewidths=2.0,
                   linestyles="solid")
        ax.contour(X, Y, PdB, levels=[-20, -10], colors=[theme.ink_muted], linewidths=0.8,
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
        for arr, lab, col in ((bm.p_e, f"E-plane  HPBW {bm.hpbw_e_deg:.2f}°", theme.series[0]),
                              (bm.p_h, f"H-plane  HPBW {bm.hpbw_h_deg:.2f}°", theme.series[1])):
            db = 10 * np.log10(np.maximum(arr[m], 1e-6))
            ax.plot(np.concatenate([-th[m][::-1], th[m]]), np.concatenate([db[::-1], db]),
                    color=col, lw=1.6, label=lab)
        ax.axhline(-3, color=theme.ink_muted, lw=0.8)
        ax.set_ylim(-50, 2)
        ax.set_xlabel("angle from beam axis (deg)")
        ax.set_ylabel("power (dB)")
        sl = [v for v in (bm.sidelobe_db(bm.p_e), bm.sidelobe_db(bm.p_h)) if v is not None]
        ax.set_title("Principal-plane cuts" + (f"  (1st sidelobe {max(sl):.1f} dB)" if sl else ""))
        ax.legend(loc="upper right")
        fig.tight_layout(w_pad=4.0)
        fig.savefig(path)
        plt.close(fig)
    return path


def noise_budget(perf: Performance) -> list[tuple[str, float]]:
    """T_sys at the centre frequency split into every contributor."""
    hw, op = perf.hw, perf.optics
    feed = hw.feed
    i = perf.center["index"]
    t_sky = perf.center["t_sky_k"]
    fg = op.ground_fraction
    rows = [
        ("sky (CMB + Galaxy + atmosphere)", feed.eta_mismatch * feed.eta_ohmic * (1 - fg) * t_sky),
        ("ground pickup (spillover)", feed.eta_mismatch * feed.eta_ohmic * fg * hw.environment.ground_k),
        ("feed ohmic loss", feed.eta_mismatch * (1 - feed.eta_ohmic) * feed.t_phys_k),
    ]
    rows += [(f"{s.component.id} ({s.component.type})", float(s.contribution_k[i]))
             for s in perf.sweep.cascade.stages]
    return rows


def plot_noise_budget(perf: Performance, path: Path, theme: Theme = LIGHT) -> Path:
    rows = noise_budget(perf)
    total = sum(v for _, v in rows)
    with styled(theme):
        fig, ax = plt.subplots(figsize=(9, 0.45 * len(rows) + 1.6))
        y = np.arange(len(rows))[::-1]
        cols = [theme.series[2]] * 3 + [theme.series[1]] * (len(rows) - 3)
        ax.barh(y, [v for _, v in rows], color=cols, height=0.6)
        for yi, (lab, v) in zip(y, rows):
            ax.text(v + total * 0.01, yi, f"{v:.1f} K  ({100 * v / total:.0f}%)",
                    va="center", fontsize=8, color=theme.ink_secondary)
        ax.set_yticks(y)
        ax.set_yticklabels([lab for lab, _ in rows])
        ax.set_xlim(0, max(v for _, v in rows) * 1.3)
        ax.set_xlabel("contribution to T_sys (K)")
        ax.set_title(f"Noise budget at {perf.hw.center_hz/1e6:.3f} MHz:  "
                     f"T_sys = {total:.1f} K  (green: antenna side, orange: receiver)")
        ax.grid(axis="y", visible=False)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
    return path


def plot_order_comparison(results: Sequence[tuple[str, Performance]], path: Path,
                          theme: Theme = LIGHT) -> Path:
    """Overlay T_sys(f) and compare SEFD at the line for several chain orders."""
    with styled(theme):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5),
                                     gridspec_kw={"width_ratios": [1.4, 1]})
        cat = theme.categorical
        for k, (label, perf) in enumerate(results):
            sw = perf.sweep
            a1.plot(_mhz(sw.freq_hz), sw.t_sys_k, color=cat[k % len(cat)], lw=1.6,
                    label=f"{label}:  {perf.t_sys_k:.1f} K")
        a1.axvline(HI_REST_HZ / 1e6, color=theme.ink_muted, lw=0.9)
        a1.set_yscale("log")
        a1.set_xlabel("frequency (MHz)")
        a1.set_ylabel("T_sys (K)")
        a1.set_title("System temperature for each chain order")
        a1.legend(loc="upper left", fontsize=7)
        base = results[0][1].t_sys_k
        y = np.arange(len(results))[::-1]
        vals = [p.t_sys_k for _, p in results]
        a2.barh(y, vals, color=[cat[k % len(cat)] for k in range(len(results))], height=0.6)
        for yi, (lab, p) in zip(y, results):
            a2.text(p.t_sys_k * 1.01, yi,
                    f"{p.t_sys_k:.1f} K   SEFD {p.sefd_jy:,.0f} Jy   "
                    f"time ×{(p.t_sys_k / base) ** 2:.2f}",
                    va="center", fontsize=7, color=theme.ink_secondary)
        a2.set_yticks(y)
        a2.set_yticklabels([lab for lab, _ in results], fontsize=7)
        a2.set_xlim(0, max(vals) * 1.9)
        a2.set_xlabel("T_sys at the HI line (K)")
        a2.set_title("Integration time scales as T_sys²")
        a2.grid(axis="y", visible=False)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
    return path
