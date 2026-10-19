"""Hardware parameters -> performance parameters.

:func:`load_hardware` reads ``hardware_parameters.yaml`` into a
:class:`Hardware`; :func:`evaluate` turns that into a :class:`Performance`
(gain, Friis receiver temperature, passband, antenna temperature, T_sys,
A_eff, beam, SEFD); :func:`write_performance_yaml` writes the intermediate
file.  The hardware stops at SEFD.  Signal-to-noise needs three things the
hardware does not know -- the source flux, the integration time and the
bandwidth you integrate over -- so it is the function :func:`snr` (and its
inverse :func:`required_flux_jy`) rather than a number in the file.

Reference plane: everything is referred to the **feed output terminals**.
Feed ohmic loss and mismatch are therefore part of the antenna (they reduce
A_eff and add to T_A), and the first cable is the first stage of the
receiver (inside T_rec).
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..constants import C_LIGHT, HI_REST_HZ, JY, K_BOLTZMANN, T_CMB_K
from .optics import Beam, FeedModel, Optics, Reflector, build_feed, build_reflector
from .params import HardwareConfigError, ParamSet
from .rf import (CHAIN_TYPES, CascadeResult, Component, Passband, build_component,
                 cascade, lin_to_db, measure_passband, nominal_passband, T0_K)

SIDEREAL_RATE_DEG_PER_HOUR = 360.0 / 23.9344696
STRUCTURE_TYPES = {"reflector", "feed", "mount"}


# ---------------------------------------------------------------------------
# signal-to-noise: the function the hardware hands to the schedule
# ---------------------------------------------------------------------------
def snr(flux_jy, sefd_jy, bandwidth_hz, integration_s, n_pol=1, k_s=1.0):
    """Total-power radiometer: SNR = (S / SEFD) * sqrt(n_pol * B * tau) / K_s."""
    return (np.asarray(flux_jy, float) / np.asarray(sefd_jy, float)
            * np.sqrt(n_pol * np.asarray(bandwidth_hz, float)
                      * np.asarray(integration_s, float)) / k_s)


def required_flux_jy(target_snr, sefd_jy, bandwidth_hz, integration_s,
                     n_pol=1, k_s=1.0):
    """The inverse: flux needed to reach ``target_snr``.

    This is the number to index a catalogue with -- "I want SNR 3, so show me
    only sources brighter than this".
    """
    return (np.asarray(target_snr, float) * k_s * np.asarray(sefd_jy, float)
            / np.sqrt(n_pol * np.asarray(bandwidth_hz, float)
                      * np.asarray(integration_s, float)))


def required_integration_s(target_snr, flux_jy, sefd_jy, bandwidth_hz,
                           n_pol=1, k_s=1.0):
    """And the other inverse: how long to integrate for ``target_snr``."""
    return (np.asarray(target_snr, float) * k_s * np.asarray(sefd_jy, float)
            / np.asarray(flux_jy, float)) ** 2 / (n_pol * np.asarray(bandwidth_hz, float))


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
@dataclass
class Environment:
    latitude_deg: float
    ambient_k: float
    ground_k: float
    cmb_k: float
    t408_k: float
    spectral_index: float
    atmosphere_k: float
    horizon_deg: float

    def t_sky_k(self, f_hz):
        """CMB + Galactic synchrotron (scaled from 408 MHz) + atmosphere."""
        f = np.asarray(f_hz, float)
        return self.cmb_k + self.t408_k * (f / 408e6) ** self.spectral_index + self.atmosphere_k


@dataclass
class Hardware:
    name: str
    path: Path | None
    center_hz: float
    sweep_hz: tuple[float, float, int]
    environment: Environment
    reflector: Reflector | None
    feed: FeedModel
    pointing_alt_deg: float
    pointing_az_deg: float
    mount_type: str
    chain: list[Component]
    disabled: list[str]
    channel_hz: float
    n_pol: int
    k_s: float
    stages_doc: list[dict]
    assumptions: list[str] = field(default_factory=list)
    placeholders: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    lna_p1db_dbm: float | None = None

    @property
    def order(self) -> list[str]:
        return [c.id for c in self.chain]


def _load_yaml(path: Path) -> dict:
    import yaml
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f" (line {mark.line + 1}, column {mark.column + 1})" if mark else ""
        raise HardwareConfigError(
            f"{path} is not valid YAML{where}: {getattr(exc, 'problem', exc)}. "
            f"A common cause is a value starting with ':' -- quote it, e.g. "
            f"unit: \":1\"") from exc
    if not isinstance(data, dict):
        raise HardwareConfigError(f"{path}: expected a mapping at the top level")
    return data


def load_hardware(path: str | Path | None = None, *, data: dict | None = None,
                  order: Sequence[str] | None = None) -> Hardware:
    """Read the hardware file.  ``order`` (a list of component ids) overrides
    every ``chain_position`` -- the quick way to try a different chain."""
    p = Path(path) if path is not None else None
    if data is None:
        if p is None:
            raise HardwareConfigError("give a path or a data mapping")
        data = _load_yaml(p)
    base = p.parent if p is not None else Path.cwd()
    assumptions: list[str] = []
    warnings: list[str] = []

    an = ParamSet.from_yaml("analysis", data.get("analysis"), base)
    center = an.number("center_freq", "Hz", HI_REST_HZ)
    sweep = (an.number("sweep_min", "Hz", center - 40e6),
             an.number("sweep_max", "Hz", center + 40e6),
             int(an.number("sweep_points", "", 1601)))
    if not sweep[0] < center < sweep[1]:
        raise HardwareConfigError("analysis: sweep_min < center_freq < sweep_max is required")

    env_ps = ParamSet.from_yaml("environment", data.get("environment"), base)
    env = Environment(
        latitude_deg=env_ps.number("site_latitude", "deg", 36.0031),
        ambient_k=env_ps.number("ambient_temperature", "K", 290.0),
        ground_k=env_ps.number("ground_temperature", "K", 290.0),
        cmb_k=env_ps.number("cmb_temperature", "K", T_CMB_K),
        t408_k=env_ps.number("galactic_temperature_408", "K", 20.0),
        spectral_index=env_ps.number("galactic_spectral_index", "", -2.7),
        atmosphere_k=env_ps.number("atmosphere_temperature", "K", 2.0),
        horizon_deg=env_ps.number("horizon_elevation", "deg", 0.0),
    )
    placeholders = [f"environment.{n}" for n in env_ps.placeholders()]

    comps_raw = data.get("components") or []
    structure: dict[str, tuple[dict, ParamSet]] = {}
    chain: list[Component] = []
    disabled: list[str] = []
    seen: set[str] = set()
    for i, raw in enumerate(comps_raw):
        if not isinstance(raw, dict):
            raise HardwareConfigError(f"components[{i}] must be a mapping")
        cid = str(raw.get("id") or f"component_{i}")
        if cid in seen:
            raise HardwareConfigError(f"two components are called {cid!r}")
        seen.add(cid)
        ctype = str(raw.get("type") or "").strip().lower()
        enabled = raw.get("enabled", True) is not False
        if ctype in STRUCTURE_TYPES:
            if ctype in structure:
                raise HardwareConfigError(f"only one {ctype} is allowed")
            ps = ParamSet.from_yaml(cid, raw.get("parameters"), base)
            if enabled:
                structure[ctype] = (raw, ps)
                placeholders += [f"{cid}.{n}" for n in ps.placeholders()]
            else:
                disabled.append(cid)
            continue
        if ctype not in CHAIN_TYPES:
            raise HardwareConfigError(
                f"component {cid!r}: type {ctype or '(missing)'!r} is not one of "
                f"{', '.join(sorted(CHAIN_TYPES | STRUCTURE_TYPES))}")
        if not enabled:
            disabled.append(cid)
            continue
        comp = build_component(raw, index=i, base_dir=base,
                               default_t_phys_k=env.ambient_k, center_hz=center)
        chain.append(comp)
        placeholders += [f"{cid}.{n}" for n in comp.params.placeholders()]

    if order:
        by_id = {c.id: c for c in chain}
        missing = [o for o in order if o not in by_id]
        if missing:
            raise HardwareConfigError(
                f"--order names {', '.join(missing)}, which "
                f"{'is' if len(missing) == 1 else 'are'} not enabled chain "
                f"components (have: {', '.join(by_id)})")
        left_out = [c.id for c in chain if c.id not in order]
        if left_out:
            warnings.append(f"--order left out {', '.join(left_out)}; "
                            f"they are dropped from this run")
        chain = [by_id[o] for o in order]
        for pos, c in enumerate(chain, 1):
            c.chain_position = float(pos)
    else:
        positions = [c.chain_position for c in chain]
        dup = {x for x in positions if positions.count(x) > 1}
        if dup:
            raise HardwareConfigError(
                f"chain_position {', '.join(str(int(d)) for d in sorted(dup))} is "
                f"used by more than one component")
        chain.sort(key=lambda c: c.chain_position)
    if not chain:
        raise HardwareConfigError("no enabled RF-chain components")
    sdrs = [i for i, c in enumerate(chain) if c.type == "sdr"]
    if not sdrs:
        warnings.append("no SDR in the chain: T_rec ignores the digitiser's noise")
    elif sdrs[-1] != len(chain) - 1:
        warnings.append(f"{chain[sdrs[-1]].id} is not the last component; anything "
                        f"after the SDR is already digital and has no analog effect")

    refl_ps = structure.get("reflector", (None, None))[1]
    reflector = build_reflector(refl_ps, assumptions)
    if "feed" not in structure:
        raise HardwareConfigError("a feed component is required (type: feed)")
    feed = build_feed(structure["feed"][1], reflector, env.ambient_k, assumptions)

    mount_ps = structure.get("mount", (None, ParamSet("mount")))[1]
    alt = mount_ps.number("pointing_alt", "deg", 90.0)
    az = mount_ps.number("pointing_az", "deg", 0.0)

    sdr = chain[sdrs[-1]] if sdrs else None
    chan = None
    n_pol, k_s = 1, 1.0
    if sdr is not None:
        chan = sdr.params.number("channel_width", "Hz")
        fft = sdr.params.number("fft_size", "")
        if chan is None and fft:
            chan = sdr.sdr_bandwidth_hz() / fft
        n_pol = int(sdr.params.number("n_pol", "", 1))
        k_s = sdr.params.number("radiometer_factor", "", 1.0)
    if chan is None:
        chan = 5e3
        assumptions.append("no SDR fft_size/channel_width: assumed 5 kHz channels")
    if n_pol not in (1, 2):
        raise HardwareConfigError("n_pol must be 1 or 2")

    lna = next((c for c in chain if c.type == "amplifier"), None)
    p1db = lna.params.number("p1db_output", "dBm") if lna else None

    return Hardware(
        name=str((data.get("meta") or {}).get("name") or "unnamed telescope"),
        path=p, center_hz=center, sweep_hz=sweep, environment=env,
        reflector=reflector, feed=feed, pointing_alt_deg=alt, pointing_az_deg=az,
        mount_type=mount_ps.text("type", "fixed"), chain=chain, disabled=disabled,
        channel_hz=chan, n_pol=n_pol, k_s=k_s, stages_doc=data.get("stages") or [],
        assumptions=assumptions, placeholders=placeholders, warnings=warnings,
        lna_p1db_dbm=p1db,
    )


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
@dataclass
class Sweep:
    """Everything as a function of frequency (for the response plot / CSV)."""

    freq_hz: np.ndarray
    cascade: CascadeResult
    feed_response_db: np.ndarray
    t_sky_k: np.ndarray
    t_a_k: np.ndarray
    t_rec_k: np.ndarray
    t_sys_k: np.ndarray
    a_eff_m2: np.ndarray
    sefd_jy: np.ndarray
    response_db: np.ndarray


@dataclass
class Performance:
    hw: Hardware
    optics: Optics
    beam: Beam
    sweep: Sweep
    dense: Sweep
    passband: Passband
    center: dict[str, float]

    # convenient scalars ------------------------------------------------
    @property
    def sefd_jy(self) -> float:
        return self.center["sefd_jy"]

    @property
    def t_sys_k(self) -> float:
        return self.center["t_sys_k"]

    @property
    def noise_bandwidth_hz(self) -> float:
        return self.passband.noise_bandwidth_hz

    @property
    def beam_dec_deg(self) -> float:
        lat = math.radians(self.hw.environment.latitude_deg)
        alt = math.radians(self.hw.pointing_alt_deg)
        az = math.radians(self.hw.pointing_az_deg)
        return math.degrees(math.asin(math.sin(lat) * math.sin(alt)
                                      + math.cos(lat) * math.cos(alt) * math.cos(az)))

    @property
    def hpbw_ns_deg(self) -> float:
        return self.beam.hpbw_azimuth_deg(0.0)

    @property
    def hpbw_ew_deg(self) -> float:
        return self.beam.hpbw_azimuth_deg(90.0)

    @property
    def transit_s(self) -> float:
        rate = SIDEREAL_RATE_DEG_PER_HOUR * math.cos(math.radians(self.beam_dec_deg))
        return self.hpbw_ew_deg / rate * 3600.0 if rate > 0 else float("inf")

    def sigma_jy(self, integration_s, bandwidth_hz=None):
        b = self.noise_bandwidth_hz if bandwidth_hz is None else bandwidth_hz
        return required_flux_jy(1.0, self.sefd_jy, b, integration_s,
                                self.hw.n_pol, self.hw.k_s)

    def snr(self, flux_jy, integration_s, bandwidth_hz=None):
        b = self.noise_bandwidth_hz if bandwidth_hz is None else bandwidth_hz
        return snr(flux_jy, self.sefd_jy, b, integration_s, self.hw.n_pol, self.hw.k_s)

    def required_flux_jy(self, target_snr, integration_s, bandwidth_hz=None):
        b = self.noise_bandwidth_hz if bandwidth_hz is None else bandwidth_hz
        return required_flux_jy(target_snr, self.sefd_jy, b, integration_s,
                                self.hw.n_pol, self.hw.k_s)


def _sweep(hw: Hardware, optics: Optics, f: np.ndarray) -> Sweep:
    cas = cascade(hw.chain, f)
    feed = hw.feed
    feed_resp = feed.response_db(f)
    t_sky = hw.environment.t_sky_k(f)
    fg = optics.ground_fraction
    scene = (1.0 - fg) * t_sky + fg * hw.environment.ground_k
    t_a = feed.eta_mismatch * (feed.eta_ohmic * scene
                               + (1.0 - feed.eta_ohmic) * feed.t_phys_k)
    t_rec = cas.t_rec_k
    t_sys = t_a + t_rec
    a_eff = optics.effective_area_m2(f)
    sefd = 2.0 * K_BOLTZMANN * t_sys / np.maximum(a_eff, 1e-30) / JY
    return Sweep(f, cas, feed_resp, t_sky, t_a, t_rec, t_sys, a_eff, sefd,
                 cas.response_db + feed_resp)


def evaluate(hw: Hardware, *, reuse: Performance | None = None) -> Performance:
    """Everything, at the centre frequency and across the sweep.

    ``reuse`` borrows the optics and beam from an earlier evaluation of the
    same antenna -- they do not depend on the RF chain, so a chain-order
    sweep need not recompute them.
    """
    if reuse is not None:
        optics, beam = reuse.optics, reuse.beam
    else:
        optics = Optics(hw.center_hz, hw.feed, hw.reflector, hw.environment.horizon_deg)
        beam = optics.beam()

    lo, hi, lo_by, hi_by = nominal_passband(
        hw.chain, [("feed", hw.feed.band_lo_hz, hw.feed.band_hi_hz)])
    if not lo < hw.center_hz < hi:
        raise HardwareConfigError(
            f"the centre frequency {hw.center_hz/1e6:.3f} MHz is outside the "
            f"chain's nominal passband {lo/1e6:.2f}-{hi/1e6:.2f} MHz "
            f"(set by {lo_by} and {hi_by})")
    w = hi - lo
    dense_f = np.linspace(max(lo - 0.75 * w, 1e6), hi + 0.75 * w, 30001)
    dense = _sweep(hw, optics, dense_f)
    pb = measure_passband(dense_f, dense.response_db, hw.center_hz, (lo, hi, lo_by, hi_by))

    f0, f1, n = hw.sweep_hz
    grid = np.unique(np.concatenate([np.linspace(f0, f1, n), [hw.center_hz]]))
    sw = _sweep(hw, optics, grid)
    i = int(np.argmin(np.abs(grid - hw.center_hz)))
    center = {k: float(getattr(sw, k)[i]) for k in
              ("t_sky_k", "t_a_k", "t_rec_k", "t_sys_k", "a_eff_m2", "sefd_jy")}
    center["gain_to_sdr_db"] = float(sw.cascade.gain_to_sdr_db[i])
    center["index"] = i
    return Performance(hw, optics, beam, sw, dense, pb, center)


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def _r(x: float, sig: int = 5) -> float | None:
    if x is None or not np.isfinite(x):
        return None
    if x == 0:
        return 0.0
    return float(f"{x:.{sig}g}")


def _q(value, unit, formula="", notes="", sig: int = 5) -> dict[str, Any]:
    num = isinstance(value, (int, float, np.floating)) and not isinstance(value, bool)
    out: dict[str, Any] = {"value": _r(value, sig) if num else value,
                           "unit": unit}
    if formula:
        out["formula"] = formula
    if notes:
        out["notes"] = notes
    return out


def friis_table(perf: Performance) -> list[dict[str, Any]]:
    i = perf.center["index"]
    t_rec = perf.center["t_rec_k"]
    rows = []
    for s in perf.sweep.cascade.stages:
        c = s.component
        row = {
            "position": int(c.chain_position), "id": c.id, "type": c.type,
            "gain_db": _r(float(s.gain_db[i]), 4),
            "noise_temp_k": _r(float(s.noise_temp_k[i]), 4),
            "gain_before_db": _r(float(lin_to_db(s.gain_before_lin[i])), 4),
            "contribution_k": _r(float(s.contribution_k[i]), 4),
            "share_of_t_rec": _r(float(s.contribution_k[i]) / t_rec, 3),
        }
        if c.type in {"amplifier", "sdr"}:
            row["noise_figure_db"] = _r(float(c.params.require("noise_figure", "dB")
                                              .at(perf.hw.center_hz)), 4)
        else:
            row["physical_temperature_k"] = _r(c.t_phys_k, 4)
        rows.append(row)
    return rows


def performance_dict(perf: Performance) -> dict[str, Any]:
    hw, op, bm, pb, c = perf.hw, perf.optics, perf.beam, perf.passband, perf.center
    lam = op.wavelength_m
    feed = hw.feed

    collection: dict[str, Any] = {
        "wavelength": _q(lam * 100, "cm", "c / f"),
        "effective_area": _q(c["a_eff_m2"], "m^2", "A_eff = G lambda^2 / (4 pi)"),
        "gain": _q(10 * math.log10(float(op.gain_lin())), "dBi",
                   "realised gain at the feed terminals, every loss included"),
        "directivity": _q(10 * math.log10(op.directivity_lin()), "dBi",
                          "pattern directivity (no losses)"),
        "sensitivity": _q(c["a_eff_m2"] * JY / (2 * K_BOLTZMANN) * 1e3, "mK/Jy",
                          "A_eff / 2k: antenna temperature per jansky, one pol"),
        "feed_ohmic_efficiency": _q(feed.eta_ohmic, "", "10^(-loss/10)"),
        "feed_mismatch_efficiency": _q(feed.eta_mismatch, "", "1 - |Gamma|^2"),
    }
    if op.reflector is not None:
        r = op.reflector
        collection.update({
            "geometric_area": _q(r.geometric_area_m2, "m^2", "pi D^2 / 4"),
            "aperture_efficiency": _q(float(op.aperture_efficiency()), "",
                                      "product of every efficiency below"),
            "spillover_efficiency": _q(op.eta_spill * op.eta_forward, "",
                                       "feed power the dish intercepts"),
            "taper_efficiency": _q(op.eta_taper, "",
                                   "|int E dA|^2 / (A int |E|^2 dA)"),
            "blockage_efficiency": _q(op.eta_block, ""),
            "surface_efficiency": _q(float(op.eta_surface()), "",
                                     "Ruze: exp(-(4 pi eps / lambda)^2)"),
            "rim_half_angle": _q(math.degrees(r.half_angle_rad), "deg",
                                 "tan(theta_0 / 2) = 1 / (4 f/D)"),
            "edge_illumination_e": _q(_edge_db(feed.q_e, r.half_angle_rad), "dB",
                                      "feed taper + spherical spreading at the rim"),
            "edge_illumination_h": _q(_edge_db(feed.q_h, r.half_angle_rad), "dB"),
            "rule_of_thumb_gain": _q(10 * math.log10(6 * (r.diameter_m / lam) ** 2), "dBi",
                                     "10 log10(6 (D/lambda)^2), for comparison"),
        })

    beam_pattern = {
        "hpbw_e_plane": _q(bm.hpbw_e_deg, "deg"),
        "hpbw_h_plane": _q(bm.hpbw_h_deg, "deg"),
        "hpbw_north_south": _q(perf.hpbw_ns_deg, "deg",
                               notes="sets the width of the declination strip swept"),
        "hpbw_east_west": _q(perf.hpbw_ew_deg, "deg",
                             notes="sets how long a source takes to drift through"),
        "beam_ellipticity": _q(bm.hpbw_e_deg / bm.hpbw_h_deg, "",
                               "HPBW_E / HPBW_H; 1 = circular beam"),
        "e_plane_azimuth": _q(feed.e_plane_azimuth_deg, "deg",
                              notes="compass direction of the feed's E-plane"),
        "directivity_from_pattern": _q(10 * math.log10(bm.directivity_lin), "dBi",
                                       "4 pi / integral(P dOmega); checks the line above"),
        "beam_solid_angle": _q(bm.solid_angle_sr, "sr"),
        "main_beam_solid_angle": _q(bm.main_beam_solid_angle_sr, "sr",
                                    "integral of P out to the first null"),
        "main_beam_efficiency": _q(bm.main_beam_solid_angle_sr / bm.solid_angle_sr, ""),
        "first_sidelobe_e": _q(bm.sidelobe_db(bm.p_e), "dB"),
        "first_sidelobe_h": _q(bm.sidelobe_db(bm.p_h), "dB"),
        "feed_q_e": _q(feed.q_e, "", "cos^q power-pattern exponent"),
        "feed_q_h": _q(feed.q_h, ""),
    }
    if op.reflector is not None:
        beam_pattern["rule_of_thumb_hpbw"] = _q(
            70 * lam / op.reflector.diameter_m, "deg", "70 lambda / D, for comparison")

    stages = perf.sweep.cascade.stages
    noise = {
        "t_receiver": _q(c["t_rec_k"], "K",
                         "Friis: T1 + T2/G1 + T3/(G1 G2) + ..."),
        "t_antenna": _q(c["t_a_k"], "K",
                        "eta_mm [eta_ohm ((1-f_g) T_sky + f_g T_ground) "
                        "+ (1-eta_ohm) T_feed]"),
        "t_sky": _q(c["t_sky_k"], "K", "T_cmb + T_408 (f/408 MHz)^beta + T_atm"),
        "ground_fraction": _q(op.ground_fraction, "",
                              "share of the response that sees warm ground"),
        "t_system": _q(c["t_sys_k"], "K", "T_A + T_rec"),
        "receiver_noise_figure": _q(10 * math.log10(1 + c["t_rec_k"] / T0_K), "dB",
                                    "10 log10(1 + T_rec / 290 K)"),
    }

    p_noise_dbm = 10 * math.log10(K_BOLTZMANN * c["t_sys_k"] * pb.noise_bandwidth_hz
                                  * 10 ** (c["gain_to_sdr_db"] / 10) * 1e3)
    gain = {
        "total_gain": _q(c["gain_to_sdr_db"], "dB",
                         "sum of stage gains in dB, chain input to SDR input"),
        "passband_nominal_low": _q(pb.nominal_lo_hz / 1e6, "MHz",
                                   notes=f"highest lower edge, set by {pb.nominal_lo_by}"),
        "passband_nominal_high": _q(pb.nominal_hi_hz / 1e6, "MHz",
                                    notes=f"lowest upper edge, set by {pb.nominal_hi_by}"),
        "passband_3db_low": _q(pb.lo_3db_hz / 1e6, "MHz"),
        "passband_3db_high": _q(pb.hi_3db_hz / 1e6, "MHz"),
        "bandwidth_3db": _q(pb.bw_3db_hz / 1e6, "MHz",
                            "between the -3 dB points of the multiplied responses"),
        "noise_bandwidth": _q(pb.noise_bandwidth_hz / 1e6, "MHz",
                              "integral(G df) / G_peak -- the radiometer bandwidth"),
        "passband_ripple": _q(pb.ripple_db, "dB", "max - min over the central 80% of the -3 dB band"),
        "noise_power_at_sdr": _q(p_noise_dbm, "dBm",
                                 "k T_sys G B_N: what the ADC has to digitise"),
    }
    lna = next((s for s in stages if s.component.type == "amplifier"), None)
    if lna is not None and hw.lna_p1db_dbm is not None:
        lo_ = lna.component.params.number("freq_min", "Hz")
        hi_ = lna.component.params.number("freq_max", "Hz")
        if lo_ and hi_:
            i = c["index"]
            t_in = c["t_a_k"] + sum(s.contribution_k[i] for s in stages[:stages.index(lna) + 1])
            g_out = float(lna.cumulative_gain_db[i])
            p = 10 * math.log10(K_BOLTZMANN * t_in * (hi_ - lo_) * 10 ** (g_out / 10) * 1e3)
            gain["lna_output_noise_power"] = _q(
                p, "dBm", "k T (f_max - f_min) G at the first amplifier's output",
                "the whole LNA band, since nothing before it filters")
            gain["lna_compression_headroom"] = _q(
                hw.lna_p1db_dbm - p, "dB",
                "P1dB - noise power; RFI has to fit in this margin")

    position = {
        "mount": _q(hw.mount_type, ""),
        "pointing_altitude": _q(hw.pointing_alt_deg, "deg"),
        "pointing_azimuth": _q(hw.pointing_az_deg, "deg"),
        "beam_declination": _q(perf.beam_dec_deg, "deg",
                               "asin(sin lat sin alt + cos lat cos alt cos az)"),
        "declination_strip": _q(perf.hpbw_ns_deg, "deg", "N-S HPBW, centred on the line above"),
        "transit_time": _q(perf.transit_s, "s",
                           "HPBW_EW / (15.04 deg/h cos dec): one drift through the beam"),
    }

    system = {
        "sefd": _q(c["sefd_jy"], "Jy", "2 k T_sys / A_eff"),
        "spectral_channel": _q(hw.channel_hz / 1e3, "kHz", "sample_rate / fft_size"),
        "velocity_resolution": _q(C_LIGHT * hw.channel_hz / hw.center_hz / 1e3, "km/s"),
        "n_pol": _q(hw.n_pol, "", notes="2 = both polarisations summed: sqrt(2) better"),
        "radiometer_factor": _q(hw.k_s, "", notes="K_s: 1 total power, 2 Dicke switched"),
        "noise_per_root_hz_s": _q(c["sefd_jy"] * hw.k_s / math.sqrt(hw.n_pol), "Jy",
                                  "sigma_S = this / sqrt(bandwidth_Hz * time_s)"),
        "sigma_continuum_one_transit": _q(float(perf.sigma_jy(perf.transit_s)), "Jy",
                                          "noise bandwidth, one transit"),
        "sigma_channel_one_transit": _q(
            float(perf.sigma_jy(perf.transit_s, hw.channel_hz)), "Jy",
            "one spectral channel, one transit"),
    }
    return {
        "generated_from": {
            "file": str(hw.path) if hw.path else None,
            "sha256": _sha(hw.path),
            "telescope": hw.name,
            "center_frequency": _q(hw.center_hz / 1e6, "MHz", sig=13),
            "reference_plane": "feed output terminals",
        },
        "chain_order": [f"{int(c_.chain_position)}: {c_.label}" for c_ in hw.chain],
        "disabled_components": hw.disabled,
        "collection": collection,
        "beam_pattern": beam_pattern,
        "noise": noise,
        "gain": gain,
        "position": position,
        "system": system,
        "friis_breakdown": friis_table(perf),
        "assumptions": hw.assumptions,
        "placeholder_inputs": hw.placeholders,
        "warnings": hw.warnings,
    }


def _edge_db(q: float, theta0: float) -> float:
    return 10 * q * math.log10(math.cos(theta0)) + 20 * math.log10(math.cos(theta0 / 2) ** 2)


def _sha(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def write_performance_yaml(perf: Performance, path: Path) -> Path:
    import yaml
    header = ("# performance_parameters.yaml -- GENERATED by compute_performance.py.\n"
              "# Do not edit: change hardware_parameters.yaml and re-run.\n"
              "# Every value is at the centre frequency; frequency_response.csv has\n"
              "# the same quantities across the sweep.\n\n")
    text = yaml.safe_dump(performance_dict(perf), sort_keys=False,
                          default_flow_style=None, width=100, allow_unicode=True)
    path.write_text(header + text)
    return path


def write_response_csv(perf: Performance, path: Path) -> Path:
    import csv
    sw = perf.sweep
    cols = ["freq_mhz", "response_db", "gain_to_sdr_db", "t_sky_k", "t_antenna_k",
            "t_receiver_k", "t_system_k", "a_eff_m2", "sefd_jy",
            "sigma_channel_one_transit_jy"]
    stage_cols = [f"{s.component.id}_gain_db" for s in sw.cascade.stages] + \
                 [f"{s.component.id}_noise_contrib_k" for s in sw.cascade.stages]
    sig = required_flux_jy(1.0, sw.sefd_jy, perf.hw.channel_hz, perf.transit_s,
                           perf.hw.n_pol, perf.hw.k_s)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols + stage_cols)
        for i, f in enumerate(sw.freq_hz):
            row = [f / 1e6, sw.response_db[i], sw.cascade.gain_to_sdr_db[i], sw.t_sky_k[i],
                   sw.t_a_k[i], sw.t_rec_k[i], sw.t_sys_k[i], sw.a_eff_m2[i],
                   sw.sefd_jy[i], sig[i]]
            row += [s.gain_db[i] for s in sw.cascade.stages]
            row += [s.contribution_k[i] for s in sw.cascade.stages]
            w.writerow([f"{v:.6g}" for v in row])
    return path


# ---------------------------------------------------------------------------
# observing goals -> detection thresholds (the "I want SNR 3" step)
# ---------------------------------------------------------------------------
def load_goals(path: Path) -> dict[str, Any]:
    data = _load_yaml(path)
    scalars = {k: v for k, v in data.items() if k != "integrations"}
    ps = ParamSet.from_yaml("goals", scalars, path.parent)
    return {
        "snr_target": ps.number("snr_target", "", 3.0),
        "integrations": data.get("integrations") or ["transit"],
        "line_width": ps.number("hi_line_width", "km/s", 200.0),
        "base_dir": path.parent,
    }


def _integration_s(perf: Performance, item: Any, base: Path) -> tuple[str, float]:
    from .params import parse_param
    if isinstance(item, str) and item.strip().lower().startswith("transit"):
        parts = item.split("x")
        n = float(parts[1]) if len(parts) == 2 else 1.0
        label = "1 transit" if n == 1 else f"{n:g} transits"
        return label, perf.transit_s * n
    p = parse_param("integration", item, where="goals.integrations", base_dir=base)
    if p.unit != "s":
        raise HardwareConfigError(f"goals.integrations: {item!r} is not a time")
    v = float(p.value)
    return (f"{v/3600:g} h" if v >= 3600 else f"{v:g} s"), v


def thresholds_dict(perf: Performance, goals: dict[str, Any]) -> dict[str, Any]:
    """Required flux for the target SNR, per integration and per measurement.

    Three measurements, because they integrate over very different
    bandwidths:

    * continuum -- the whole noise bandwidth (a source's total power);
    * HI line, one channel -- the spectrometer resolution;
    * HI line, matched -- summed over a line of ``hi_line_width`` km/s, which
      is how a galaxy's HI is actually detected.

    ``database_index`` gives the floors to filter catalogues with: NVSS by
    1.4 GHz flux, ALFALFA / HIPASS by peak HI flux density.
    """
    snr_t = float(goals["snr_target"])
    width_kms = float(goals["line_width"])
    b_cont = perf.noise_bandwidth_hz
    b_chan = perf.hw.channel_hz
    b_line = width_kms * 1e3 / C_LIGHT * perf.hw.center_hz
    rows = []
    for item in goals["integrations"]:
        label, tau = _integration_s(perf, item, goals["base_dir"])
        rows.append({
            "integration": label,
            "integration_s": _r(tau, 4),
            "continuum_flux_jy": _r(float(perf.required_flux_jy(snr_t, tau, b_cont)), 4),
            "line_channel_peak_jy": _r(float(perf.required_flux_jy(snr_t, tau, b_chan)), 4),
            "line_matched_peak_jy": _r(float(perf.required_flux_jy(snr_t, tau, b_line)), 4),
        })
    first = rows[0]
    return {
        "snr_target": snr_t,
        "sefd": _q(perf.sefd_jy, "Jy"),
        "bandwidths": {
            "continuum": _q(b_cont / 1e6, "MHz", "noise bandwidth of the chain"),
            "line_channel": _q(b_chan / 1e3, "kHz"),
            "line_matched": _q(b_line / 1e3, "kHz", f"{width_kms:g} km/s at the line"),
        },
        "n_pol": perf.hw.n_pol,
        "formula": "S_required = SNR * K_s * SEFD / sqrt(n_pol * bandwidth * time)",
        "required_flux": rows,
        "database_index": {
            "integration": first["integration"],
            "nvss_s1400_min": _q(first["continuum_flux_jy"], "Jy",
                                 notes="continuum sources fainter than this are below target"),
            "hi_peak_min": _q(first["line_matched_peak_jy"], "Jy",
                              notes=f"HI peak flux density, {width_kms:g} km/s line"),
            "hi_integrated_min": _q(first["line_matched_peak_jy"] * width_kms, "Jy km/s",
                                    notes="the same, as ALFALFA's integrated flux (flat-top line)"),
        },
    }


def write_thresholds_yaml(perf: Performance, goals: dict[str, Any], path: Path) -> Path:
    import yaml
    header = ("# detection_thresholds.yaml -- GENERATED by compute_performance.py\n"
              "# from performance (SEFD, bandwidth) + observing_goals.yaml.\n\n")
    path.write_text(header + yaml.safe_dump(thresholds_dict(perf, goals), sort_keys=False,
                                            default_flow_style=None, width=100))
    return path
