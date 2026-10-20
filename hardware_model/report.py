"""Writing results.  No physics happens here.

Every number below was computed in ``performance.py``; this module only
rounds it, attaches its unit and the equation it came from, and writes
``performance_parameters.yaml``, ``detection_thresholds.yaml`` and
``frequency_response.csv``.  Each output entry carries an ``eq`` field
naming the equation in ``performance.py`` that produced it.
"""
from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

from . import performance as pm
from .inputs import Goals, integration_entry
from .performance import Performance


def _r(x, sig: int = 5):
    if x is None:
        return None
    x = float(x)
    if not np.isfinite(x):
        return None
    return 0.0 if x == 0 else float(f"{x:.{sig}g}")


def _q(value, unit, eq="", notes="", sig: int = 5) -> dict[str, Any]:
    num = isinstance(value, (int, float, np.floating)) and not isinstance(value, bool)
    out: dict[str, Any] = {"value": _r(value, sig) if num else value, "unit": unit}
    if eq:
        out["eq"] = eq
    if notes:
        out["notes"] = notes
    return out


def friis_table(perf: Performance) -> list[dict[str, Any]]:
    i, t_rec = perf.index, perf.at_center("t_rec_k")
    rows = []
    for s in perf.sweep.stages:
        c = s.component
        row = {
            "position": int(c.chain_position), "id": c.id, "type": c.type,
            "gain_db": _r(s.gain_db[i], 4),
            "noise_temp_k": _r(s.noise_temp_k[i], 4),
            "gain_before_db": _r(pm.lin_to_db(s.gain_before_lin[i]), 4),
            "contribution_k": _r(s.contribution_k[i], 4),
            "share_of_t_rec": _r(s.contribution_k[i] / t_rec, 3),
            "eq": "Eq. 2 (active) or Eq. 3 (passive), then Eq. 4",
        }
        if c.type in {"amplifier", "sdr"}:
            row["noise_figure_db"] = _r(c.params.require("noise_figure", "dB")
                                        .at(perf.hw.center_hz), 4)
        else:
            row["physical_temperature_k"] = _r(c.t_phys_k, 4)
        rows.append(row)
    return rows


def performance_dict(perf: Performance) -> dict[str, Any]:
    hw, ant, pb = perf.hw, perf.antenna, perf.passband
    eta, f0 = ant.eta, hw.center_hz
    c = perf.at_center

    collection: dict[str, Any] = {
        "wavelength": _q(ant.wavelength_m * 100, "cm", "Eq. 13"),
        "effective_area": _q(c("a_eff_m2"), "m^2", "Eq. 24"),
        "gain": _q(pm.lin_to_db(c("a_eff_m2") * 4 * math.pi / ant.wavelength_m ** 2), "dBi",
                   "Eq. 24", "realised, every loss included"),
        "directivity": _q(pm.lin_to_db(ant.directivity_lin), "dBi", "Eq. 15 / 22",
                          "aperture theory, no losses"),
        "sensitivity": _q(pm.antenna_temperature_per_jy(c("a_eff_m2")) * 1e3, "mK/Jy",
                          "Eq. 32", "antenna temperature per jansky, one polarisation"),
        "feed_ohmic_efficiency": _q(eta["ohmic"], "", "Eq. 18"),
        "feed_mismatch_efficiency": _q(eta["mismatch"], "", "Eq. 17"),
        "feed_forward_fraction": _q(eta["forward"], "", notes="1 - back radiation"),
        "aperture_efficiency": _q(eta["aperture"], "", notes="product of every efficiency"),
    }
    if hw.reflector is not None:
        r = hw.reflector
        collection.update({
            "geometric_area": _q(math.pi * r.diameter_m ** 2 / 4, "m^2"),
            "spillover_efficiency": _q(eta["spillover"], "", "Eq. 21"),
            "illumination_efficiency": _q(eta["illumination"], "", "Eq. 22"),
            "polarization_efficiency": _q(eta["polarization"], "", "Eq. 22"),
            "blockage_efficiency": _q(eta["blockage"], "", "Eq. 22"),
            "surface_efficiency": _q(eta["surface"], "", "Eq. 23"),
            "rim_half_angle": _q(math.degrees(ant.theta0_rad), "deg", "Eq. 19"),
            "edge_illumination_e": _q(ant.edge_illumination_db[0], "dB", "Eqs. 15, 20",
                                      "feed taper + spherical spreading at the rim"),
            "edge_illumination_h": _q(ant.edge_illumination_db[1], "dB", "Eqs. 15, 20"),
            "rule_of_thumb_gain": _q(pm.rule_of_thumb_gain_dbi(r.diameter_m, f0), "dBi",
                                     "Eq. 38b", "for comparison"),
        })

    beam_pattern = {
        "hpbw_e_plane": _q(ant.hpbw_e_deg, "deg", "Eqs. 25, 26a"),
        "hpbw_h_plane": _q(ant.hpbw_h_deg, "deg", "Eqs. 25, 26a"),
        "hpbw_north_south": _q(perf.hpbw_ns_deg, "deg", "Eq. 26a",
                               "sets the width of the declination strip swept"),
        "hpbw_east_west": _q(perf.hpbw_ew_deg, "deg", "Eq. 26a",
                             "sets how long a source takes to drift through"),
        "beam_ellipticity": _q(ant.hpbw_e_deg / ant.hpbw_h_deg, "", notes="1 = circular"),
        "e_plane_azimuth": _q(hw.feed.e_plane_azimuth_deg, "deg",
                              notes="compass direction of the feed's E-plane (input)"),
        "directivity_from_pattern": _q(pm.lin_to_db(ant.beam_directivity_lin), "dBi",
                                       "Eqs. 26c, 26e", "checks 'directivity' above"),
        "beam_solid_angle": _q(ant.beam_solid_angle_sr, "sr", "Eq. 26c"),
        "main_beam_solid_angle": _q(ant.main_beam_solid_angle_sr, "sr", "Eq. 26d"),
        "main_beam_efficiency": _q(ant.main_beam_solid_angle_sr / ant.beam_solid_angle_sr,
                                   "", "Eq. 26d"),
        "first_sidelobe_e": _q(ant.sidelobe_e_db, "dB", "Eq. 26b"),
        "first_sidelobe_h": _q(ant.sidelobe_h_db, "dB", "Eq. 26b"),
        "feed_q_e": _q(ant.q_e, "", "Eq. 14", "cos^q power-pattern exponent"),
        "feed_q_h": _q(ant.q_h, "", "Eq. 14"),
    }
    if hw.reflector is not None:
        beam_pattern["rule_of_thumb_hpbw"] = _q(
            pm.rule_of_thumb_hpbw_deg(hw.reflector.diameter_m, f0), "deg", "Eq. 38a",
            "for comparison")

    noise = {
        "t_receiver": _q(c("t_rec_k"), "K", "Eq. 4"),
        "t_antenna": _q(c("t_a_k"), "K", "Eq. 29"),
        "t_sky": _q(c("t_sky_k"), "K", "Eq. 27"),
        "ground_fraction": _q(ant.ground_fraction, "", "Eq. 28"),
        "t_system": _q(c("t_sys_k"), "K", "Eq. 30"),
        "receiver_noise_figure": _q(pm.temp_to_nf_db(c("t_rec_k")), "dB", "Eq. 2 inverted"),
    }

    gain = {
        "total_gain": _q(c("gain_to_sdr_db"), "dB", "Eq. 5", "chain input to SDR input"),
        "passband_nominal_low": _q(pb.nominal_lo_hz / 1e6, "MHz", "Eq. 10",
                                   f"set by {pb.nominal_lo_by}"),
        "passband_nominal_high": _q(pb.nominal_hi_hz / 1e6, "MHz", "Eq. 10",
                                    f"set by {pb.nominal_hi_by}"),
        "passband_3db_low": _q(pb.lo_3db_hz / 1e6, "MHz", "Eq. 11"),
        "passband_3db_high": _q(pb.hi_3db_hz / 1e6, "MHz", "Eq. 11"),
        "bandwidth_3db": _q(pb.bw_3db_hz / 1e6, "MHz", "Eq. 11"),
        "noise_bandwidth": _q(pb.noise_bandwidth_hz / 1e6, "MHz", "Eq. 12",
                              "the radiometer bandwidth"),
        "passband_ripple": _q(pb.ripple_db, "dB", "Eq. 11",
                              "max - min over the central 80% of the -3 dB band"),
        "noise_power_at_sdr": _q(perf.noise_power_sdr_dbm, "dBm", "Eq. 33",
                                 "what the ADC has to digitise"),
    }
    if perf.lna_output_noise_dbm is not None:
        gain["lna_output_noise_power"] = _q(perf.lna_output_noise_dbm, "dBm", "Eq. 33",
                                            "over the LNA's whole band")
    if perf.lna_headroom_db is not None:
        gain["lna_compression_headroom"] = _q(perf.lna_headroom_db, "dB", notes=(
            "P1dB - noise power; RFI has to fit in this margin"))

    position = {
        "mount": _q(hw.mount_type, ""),
        "pointing_altitude": _q(hw.pointing_alt_deg, "deg"),
        "pointing_azimuth": _q(hw.pointing_az_deg, "deg"),
        "beam_declination": _q(perf.beam_dec_deg, "deg", "Eq. 36"),
        "declination_strip": _q(perf.hpbw_ns_deg, "deg", "Eq. 26a", "N-S HPBW"),
        "transit_time": _q(perf.transit_s, "s", "Eq. 37"),
    }

    system = {
        "sefd": _q(c("sefd_jy"), "Jy", "Eq. 31"),
        "spectral_channel": _q(perf.channel_hz / 1e3, "kHz", notes="SDR bandwidth / fft_size"),
        "velocity_resolution": _q(perf.channel_hz / pm.doppler_bandwidth_hz(1.0, f0), "km/s",
                                  "Eq. 35"),
        "n_pol": _q(hw.n_pol, "", notes="2 = both polarisations summed: sqrt(2) better"),
        "radiometer_factor": _q(hw.k_s, "", notes="K_s: 1 total power, 2 Dicke switched"),
        "noise_per_root_hz_s": _q(perf.required_flux_jy(1.0, 1.0, 1.0), "Jy", "Eq. 34",
                                  "sigma_S = this / sqrt(bandwidth_Hz * time_s)"),
        "sigma_continuum_one_transit": _q(perf.required_flux_jy(1.0, perf.transit_s), "Jy",
                                          "Eq. 34", "noise bandwidth, one transit"),
        "sigma_channel_one_transit": _q(perf.required_flux_jy(1.0, perf.transit_s,
                                                              perf.channel_hz), "Jy",
                                        "Eq. 34", "one spectral channel, one transit"),
    }
    return {
        "generated_from": {
            "file": str(hw.path) if hw.path else None,
            "sha256": _sha(hw.path),
            "telescope": hw.name,
            "center_frequency": _q(f0 / 1e6, "MHz", sig=13),
            "reference_plane": "feed output terminals",
            "equations": "hardware_model/performance.py (eq numbers below refer to it)",
        },
        "chain_order": [f"{int(x.chain_position)}: {x.label}" for x in hw.chain],
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


def _sha(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def write_performance_yaml(perf: Performance, path: Path) -> Path:
    import yaml
    header = ("# performance_parameters.yaml -- GENERATED by hardware_model.\n"
              "# Do not edit: change hardware_parameters.yaml and re-run.\n"
              "# Values are at the centre frequency; frequency_response.csv has them\n"
              "# across the sweep.  'eq' = equation number in hardware_model/performance.py.\n\n")
    path.write_text(header + yaml.safe_dump(performance_dict(perf), sort_keys=False,
                                            default_flow_style=None, width=100,
                                            allow_unicode=True))
    return path


def write_response_csv(perf: Performance, path: Path) -> Path:
    sw, hw = perf.sweep, perf.hw
    sigma = pm.required_flux_jy(1.0, sw.sefd_jy, perf.channel_hz, perf.transit_s,   # Eq. 34
                                hw.n_pol, hw.k_s)
    cols = ["freq_mhz", "response_db", "gain_to_sdr_db", "t_sky_k", "t_antenna_k",
            "t_receiver_k", "t_system_k", "a_eff_m2", "sefd_jy", "sigma_channel_one_transit_jy"]
    cols += [f"{s.component.id}_gain_db" for s in sw.stages]
    cols += [f"{s.component.id}_noise_contrib_k" for s in sw.stages]
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for i, f in enumerate(sw.freq_hz):
            row = [f / 1e6, sw.response_db[i], sw.gain_to_sdr_db[i], sw.t_sky_k[i], sw.t_a_k[i],
                   sw.t_rec_k[i], sw.t_sys_k[i], sw.a_eff_m2[i], sw.sefd_jy[i], sigma[i]]
            row += [s.gain_db[i] for s in sw.stages] + [s.contribution_k[i] for s in sw.stages]
            w.writerow([f"{v:.6g}" for v in row])
    return path


def thresholds_dict(perf: Performance, goals: Goals) -> dict[str, Any]:
    """Flux needed for the target SNR (Eq. 34 solved for S), per integration,
    for continuum (noise bandwidth, Eq. 12), one channel, and a matched HI line
    (Doppler width, Eq. 35)."""
    snr_t, width = float(goals.snr_target), float(goals.line_width_kms)
    b_cont, b_chan = perf.noise_bandwidth_hz, perf.channel_hz
    b_line = float(pm.doppler_bandwidth_hz(width, perf.hw.center_hz))              # Eq. 35
    rows = []
    for item in goals.integrations:
        label, seconds, n_transits = integration_entry(item, goals.base_dir)
        tau = seconds if seconds is not None else n_transits * perf.transit_s
        rows.append({
            "integration": label, "integration_s": _r(tau, 4),
            "continuum_flux_jy": _r(perf.required_flux_jy(snr_t, tau, b_cont), 4),
            "line_channel_peak_jy": _r(perf.required_flux_jy(snr_t, tau, b_chan), 4),
            "line_matched_peak_jy": _r(perf.required_flux_jy(snr_t, tau, b_line), 4),
        })
    first = rows[0]
    return {
        "snr_target": snr_t,
        "sefd": _q(perf.sefd_jy, "Jy", "Eq. 31"),
        "bandwidths": {
            "continuum": _q(b_cont / 1e6, "MHz", "Eq. 12"),
            "line_channel": _q(b_chan / 1e3, "kHz"),
            "line_matched": _q(b_line / 1e3, "kHz", "Eq. 35", f"{width:g} km/s"),
        },
        "n_pol": perf.hw.n_pol,
        "formula": "Eq. 34: S_required = SNR * K_s * SEFD / sqrt(n_pol * bandwidth * time)",
        "required_flux": rows,
        "database_index": {
            "integration": first["integration"],
            "nvss_s1400_min": _q(first["continuum_flux_jy"], "Jy",
                                 notes="continuum sources fainter than this are below target"),
            "hi_peak_min": _q(first["line_matched_peak_jy"], "Jy",
                              notes=f"HI peak flux density, {width:g} km/s line"),
            "hi_integrated_min": _q(first["line_matched_peak_jy"] * width, "Jy km/s",
                                    notes="the same as an integrated flux (flat-topped line)"),
        },
    }


def write_thresholds_yaml(perf: Performance, goals: Goals, path: Path) -> Path:
    import yaml
    header = ("# detection_thresholds.yaml -- GENERATED by hardware_model\n"
              "# from the performance (SEFD, bandwidth) + observing_goals.yaml.\n\n")
    path.write_text(header + yaml.safe_dump(thresholds_dict(perf, goals), sort_keys=False,
                                            default_flow_style=None, width=100))
    return path


def describe(perf: Performance) -> str:
    pb, ant, c = perf.passband, perf.antenna, perf.at_center
    return "\n".join([
        f"{perf.hw.name}",
        f"  chain        {'  →  '.join(x.id for x in perf.hw.chain)}",
        f"  total gain   {c('gain_to_sdr_db'):.2f} dB to the SDR input",
        f"  passband     {pb.lo_3db_hz/1e6:.3f}-{pb.hi_3db_hz/1e6:.3f} MHz (-3 dB), "
        f"noise bandwidth {pb.noise_bandwidth_hz/1e6:.3f} MHz "
        f"(nominal edges from {pb.nominal_lo_by} / {pb.nominal_hi_by})",
        f"  T_rec        {c('t_rec_k'):.1f} K     T_A {c('t_a_k'):.1f} K     "
        f"T_sys {c('t_sys_k'):.1f} K",
        f"  A_eff        {c('a_eff_m2'):.3f} m²   HPBW {ant.hpbw_e_deg:.2f}° (E) × "
        f"{ant.hpbw_h_deg:.2f}° (H)",
        f"  SEFD         {c('sefd_jy'):.4g} Jy",
        f"  beam         Dec {perf.beam_dec_deg:+.2f}°, strip {perf.hpbw_ns_deg:.2f}° wide, "
        f"transit {perf.transit_s/60:.1f} min",
    ])
