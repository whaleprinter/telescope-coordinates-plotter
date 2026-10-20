"""Checks of hardware_model against hand-computed textbook values.

Every expected number is worked out here from the formula and its source
(equation numbers refer to hardware_model/performance.py), not read back
from the code, so a failure means the model disagrees with the physics.

    python3 -m pytest hardware_model/tests -q
"""
from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hardware_model import performance as pm  # noqa: E402
from hardware_model.inputs import (HardwareConfigError, convert, load_hardware,  # noqa: E402
                                   parse_param, read_touchstone_s21)
from hardware_model.performance import antenna_side, evaluate  # noqa: E402

EXAMPLE = ROOT / "hardware_model" / "hardware_parameters.yaml"
F0 = 1420.405751768e6
LAM = pm.C_LIGHT / F0


def _hw(chain=None, reflector=True, feed=None, **env):
    """A small, fully specified hardware description."""
    comps = []
    if reflector:
        comps.append({"id": "reflector", "type": "reflector", "parameters": {
            "diameter": {"value": 2.0, "unit": "m"}, "focal_ratio": {"value": 0.4, "unit": ""}}})
    default_feed = ({"edge_taper_e": {"value": -10, "unit": "dB"},
                     "edge_taper_h": {"value": -10, "unit": "dB"}} if reflector else
                    {"hpbw_e": {"value": 60, "unit": "deg"}, "hpbw_h": {"value": 60, "unit": "deg"}})
    comps.append({"id": "feed", "type": "feed", "parameters": feed or default_feed})
    comps += chain if chain is not None else [
        {"id": "cable", "type": "cable", "chain_position": 1,
         "parameters": {"loss": {"value": 1.0, "unit": "dB"},
                        "physical_temperature": {"value": 290, "unit": "K"}}},
        {"id": "lna", "type": "amplifier", "chain_position": 2,
         "parameters": {"gain": {"value": 20, "unit": "dB"},
                        "noise_figure": {"value": 0.7, "unit": "dB"},
                        "freq_min": {"value": 1.0, "unit": "GHz"},
                        "freq_max": {"value": 2.0, "unit": "GHz"}}},
        {"id": "sdr", "type": "sdr", "chain_position": 3,
         "parameters": {"sample_rate": {"value": 6, "unit": "MHz"},
                        "noise_figure": {"value": 4.3, "unit": "dB"},
                        "fft_size": {"value": 1000, "unit": ""}}},
    ]
    e = {"ambient_temperature": {"value": 290, "unit": "K"}}
    e.update(env)
    return {"analysis": {"center_freq": {"value": F0 / 1e6, "unit": "MHz"}},
            "environment": e, "components": comps}


# ----------------------------------------------------------- inputs.py (no physics)
def test_unit_conversion():
    assert convert(1.4, "GHz", where="t") == (1.4e9, "Hz")
    assert convert(4.4, "dB/100ft", where="t")[0] == pytest.approx(4.4 / 30.48)
    assert convert(20, "C", where="t") == (293.15, "K")
    with pytest.raises(HardwareConfigError, match="did you mean"):
        convert(1, "GHzz", where="t")


def test_typo_key_is_caught_with_suggestion():
    """The draft file's  {value: 1.8, unt: :1}  (quoted so that it parses)."""
    with pytest.raises(HardwareConfigError, match="did you mean 'unit'"):
        parse_param("output_vswr", {"value": 1.8, "unt": ":1"}, where="lna", base_dir=ROOT)


def test_comma_in_notes_gets_a_hint():
    import yaml
    raw = yaml.safe_load("x: {value: 2.0, unit: K, notes: zenith, 1.4 GHz}")["x"]
    with pytest.raises(HardwareConfigError, match="quote the notes"):
        parse_param("x", raw, where="env", base_dir=ROOT)


def test_frequency_curve_interpolates_and_holds_ends():
    p = parse_param("gain", {"unit": "dB", "freq_unit": "GHz",
                             "curve": [[1.3, 20.0], [1.5, 18.0]]}, where="t", base_dir=ROOT)
    assert p.at(1.4e9) == pytest.approx(19.0)
    assert p.at(2.0e9) == pytest.approx(18.0)


def test_bad_order_and_duplicate_positions_are_errors():
    with pytest.raises(HardwareConfigError, match="not enabled chain"):
        load_hardware(data=_hw(), order=["lna", "nope", "sdr"])
    d = _hw()
    d["components"][-1]["chain_position"] = 2
    with pytest.raises(HardwareConfigError, match="more than one"):
        load_hardware(data=d)


def test_disabled_component_drops_out():
    d = _hw()
    d["components"][2]["enabled"] = False                      # the cable
    hw = load_hardware(data=d)
    assert hw.order == ["lna", "sdr"] and hw.disabled == ["cable"]


def test_touchstone_reader_and_eq9(tmp_path):
    f = tmp_path / "amp.s2p"
    f.write_text("! test\n# MHz S MA R 50\n"
                 "1400 0.1 0 10 10 0.01 0 0.2 0\n1500 0.1 0 5 10 0.01 0 0.2 0\n")
    s = read_touchstone_s21(f)
    assert list(s.freq_hz) == [1.4e9, 1.5e9]
    assert pm.s21_db(s.fmt, s.a, s.b) == pytest.approx([20.0, 20 * math.log10(5)])


# ------------------------------------------------- Eqs. 1-5: noise and Friis
def test_eq2_eq3_noise_temperatures():
    assert pm.nf_to_temp_k(10 * math.log10(2)) == pytest.approx(290.0)     # F = 2 -> T0
    assert pm.passive_temp_k(-10 * math.log10(2), 290.0) == pytest.approx(290.0)  # L = 2
    assert pm.passive_temp_k(0.0, 290.0) == pytest.approx(0.0)


def test_eq4_friis_by_hand():
    perf = evaluate(load_hardware(data=_hw()))
    l_c = 10 ** 0.1
    expect = 290 * (l_c - 1) + 290 * (10 ** 0.07 - 1) * l_c + 290 * (10 ** 0.43 - 1) * l_c / 100
    assert perf.at_center("t_rec_k") == pytest.approx(expect, rel=1e-6)
    assert perf.at_center("gain_to_sdr_db") == pytest.approx(19.0)


def test_eq4_reordering():
    base = evaluate(load_hardware(data=_hw()))
    swapped = evaluate(load_hardware(data=_hw(), order=["lna", "cable", "sdr"]))
    expect = (290 * (10 ** 0.07 - 1) + 290 * (10 ** 0.1 - 1) / 100
              + 290 * (10 ** 0.43 - 1) * 10 ** 0.1 / 100)
    assert swapped.at_center("t_rec_k") == pytest.approx(expect, rel=1e-6)
    assert swapped.t_sys_k < base.t_sys_k


# --------------------------------------------- Eqs. 6, 10-12: passband
def test_eq10_passband_is_the_intersection():
    perf = evaluate(load_hardware(data=_hw()))
    assert perf.passband.nominal_lo_by == "sdr"
    assert perf.passband.nominal_hi_hz - perf.passband.nominal_lo_hz == pytest.approx(6e6)
    d = _hw()
    d["components"][3]["parameters"]["freq_max"] = {"value": 1.3, "unit": "GHz"}
    with pytest.raises(HardwareConfigError, match="no common passband"):
        evaluate(load_hardware(data=d))


def test_eq12_butterworth_noise_bandwidth():
    """B_N / B_3dB = (pi/2n) / sin(pi/2n) for an n-pole Butterworth."""
    n, bw = 3, 1e6
    f = np.linspace(F0 - 40 * bw, F0 + 40 * bw, 400001)
    pb = pm.measure_passband(f, -pm.bandpass_attenuation_db(f, F0, bw, n), F0,
                             (F0 - bw / 2, F0 + bw / 2, "", ""))
    x = math.pi / (2 * n)
    assert pb.noise_bandwidth_hz / bw == pytest.approx(x / math.sin(x), rel=2e-3)
    assert pb.bw_3db_hz == pytest.approx(bw, rel=1e-3)


def test_eq6_chebyshev_ripple_at_band_edge():
    a = pm.bandpass_attenuation_db(np.array([F0 + 0.5e6 * (1 + 1e-9)]), F0, 1e6, 5,
                                   kind="chebyshev", ripple_db=0.5)
    assert float(a[0]) == pytest.approx(0.5, abs=0.01)


# ------------------------------------------------- Eqs. 13-26: antenna
def test_eq14_round_trip():
    q = pm.cosq_exponent_from_hpbw(60.0)
    assert math.cos(math.radians(30)) ** q == pytest.approx(0.5)


def test_uniform_aperture_limits():
    """Near-uniform illumination: HPBW = 1.029 lambda/D (58.9 lambda/D deg),
    first sidelobe -17.6 dB, illumination efficiency 1 [Balanis ch. 12]."""
    hw = load_hardware(data=_hw())
    hw.feed = replace(hw.feed, hpbw_e_deg=179.999, hpbw_h_deg=179.999,
                      edge_taper_e_db=None, edge_taper_h_db=None)
    hw.reflector = replace(hw.reflector, focal_ratio=20.0)
    ant = antenna_side(hw)
    assert ant.eta["illumination"] == pytest.approx(1.0, abs=1e-4)
    assert ant.hpbw_e_deg == pytest.approx(58.9 * LAM / 2.0, rel=0.005)
    assert ant.sidelobe_e_db == pytest.approx(-17.6, abs=0.15)


def test_ten_db_taper_efficiency_is_textbook():
    """-10 dB edge taper: spillover x illumination ~ 0.8 [Balanis 15.4]."""
    ant = antenna_side(load_hardware(data=_hw()))
    assert 0.75 < ant.eta["spillover"] * ant.eta["illumination"] < 0.84


def test_symmetric_feed_has_no_cross_polar_loss_and_round_beam():
    ant = antenna_side(load_hardware(data=_hw()))
    assert ant.eta["polarization"] == pytest.approx(1.0, abs=1e-9)
    assert ant.hpbw_e_deg == pytest.approx(ant.hpbw_h_deg, rel=1e-6)


def test_aperture_and_pattern_directivity_agree():
    """Eq. 22's directivity and Eq. 26e's (from integrating Eq. 25) must agree
    to the accuracy of scalar aperture theory."""
    ant = antenna_side(load_hardware(EXAMPLE))
    gap_db = 10 * math.log10(ant.beam_directivity_lin / ant.directivity_lin)
    assert abs(gap_db) < 0.15


def test_feed_only_is_exact():
    """No reflector: D = C of Eq. 15, and the pattern integral reproduces it."""
    ant = antenna_side(load_hardware(data=_hw(reflector=False)))
    q = pm.cosq_exponent_from_hpbw(60.0)
    assert ant.directivity_lin == pytest.approx(2 * (q + 1))
    assert ant.beam_directivity_lin == pytest.approx(ant.directivity_lin, rel=1e-4)


def test_asymmetric_feed_beam_follows_e_plane_azimuth():
    feed = {"hpbw_e": {"value": 50, "unit": "deg"}, "hpbw_h": {"value": 90, "unit": "deg"},
            "e_plane_azimuth": {"value": 90, "unit": "deg"}}
    perf = evaluate(load_hardware(data=_hw(reflector=False, feed=feed)))
    assert perf.antenna.hpbw_e_deg == pytest.approx(50, rel=1e-3)
    assert perf.hpbw_ew_deg == pytest.approx(50, rel=1e-3)          # E-plane points east
    assert perf.hpbw_ns_deg == pytest.approx(90, rel=1e-3)


def test_eq16_eq23_waveguide_and_ruze():
    assert pm.circular_waveguide_band_hz(0.15)[0] == pytest.approx(
        1.8412 * pm.C_LIGHT / (math.pi * 0.15))
    assert float(pm.ruze_efficiency(0.005, F0)) == pytest.approx(
        math.exp(-(4 * math.pi * 0.005 / LAM) ** 2))


def test_eq17_mismatch():
    assert pm.mismatch_efficiency(2.0) == pytest.approx(1 - (1 / 3) ** 2)


# ------------------------------------------ Eqs. 27-35: noise and sensitivity
def test_eq28_29_ground_pickup():
    cold = evaluate(load_hardware(data=_hw(ground_temperature={"value": 0, "unit": "K"})))
    warm = evaluate(load_hardware(data=_hw(ground_temperature={"value": 300, "unit": "K"})))
    fg = warm.antenna.ground_fraction
    assert warm.at_center("t_a_k") - cold.at_center("t_a_k") == pytest.approx(300 * fg, rel=1e-6)


def test_eq31_sefd_and_eq34_inverses():
    perf = evaluate(load_hardware(data=_hw()))
    assert perf.sefd_jy == pytest.approx(
        2 * pm.K_BOLTZMANN * perf.t_sys_k / perf.a_eff_m2 / pm.JY)
    s = pm.required_flux_jy(3.0, 1e5, 1e6, 100.0)
    assert pm.snr(s, 1e5, 1e6, 100.0) == pytest.approx(3.0)
    assert pm.required_integration_s(3.0, s, 1e5, 1e6) == pytest.approx(100.0)
    assert pm.snr(s, 1e5, 1e6, 100.0, n_pol=2) == pytest.approx(3.0 * math.sqrt(2))


def test_eq24_effective_area_is_eta_times_geometric_area():
    perf = evaluate(load_hardware(data=_hw()))
    assert perf.a_eff_m2 == pytest.approx(perf.antenna.eta["aperture"] * math.pi, rel=1e-6)


def test_eq36_eq37_position():
    assert pm.beam_declination_deg(36.0, 90.0, 0.0) == pytest.approx(36.0)
    assert pm.beam_declination_deg(36.0, 80.0, 0.0) == pytest.approx(46.0)   # 10 deg north
    assert pm.transit_time_s(15.0411, 0.0) == pytest.approx(3600.0, rel=1e-4)


# ------------------------------------------------------ the program end to end
def test_example_file_end_to_end(tmp_path):
    from hardware_model.compute_performance import main
    assert main([str(EXAMPLE), "-o", str(tmp_path), "--no-plots"]) == 0
    import yaml
    out = yaml.safe_load((tmp_path / "performance_parameters.yaml").read_text())
    assert out["generated_from"]["center_frequency"]["value"] == pytest.approx(1420.405751768)
    assert out["noise"]["t_system"]["value"] > out["noise"]["t_receiver"]["value"]
    assert out["system"]["sefd"]["eq"] == "Eq. 31"
    thr = yaml.safe_load((tmp_path / "detection_thresholds.yaml").read_text())
    assert thr["database_index"]["nvss_s1400_min"]["value"] > 0


def test_does_not_import_hydroline():
    """The two programs are separate: nothing here may import hydroline."""
    import re
    src = "".join(p.read_text() for p in (ROOT / "hardware_model").glob("*.py"))
    assert not re.search(r"^\s*(from|import)\s+hydroline", src, re.M)
