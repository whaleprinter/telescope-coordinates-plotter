"""Physics and plumbing checks for hydroline.hardware.

Every expected value here is computed by hand from the textbook formula, not
read back from the code, so a test failing means the model disagrees with
the physics, not merely that it changed.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hydroline.constants import C_LIGHT, JY, K_BOLTZMANN  # noqa: E402
from hydroline.hardware import (HardwareConfigError, evaluate, load_hardware,  # noqa: E402
                                required_flux_jy, required_integration_s, snr)
from hydroline.hardware.optics import Optics, q_from_hpbw  # noqa: E402
from hydroline.hardware.params import convert, parse_param  # noqa: E402
from hydroline.hardware.rf import (bandpass_attenuation_db, measure_passband,  # noqa: E402
                                   nf_db_to_temp_k, passive_temp_k,
                                   read_touchstone_s21)

EXAMPLE = ROOT / "hardware" / "hardware_parameters.yaml"
F0 = 1420.405751768e6


def _hw(chain=None, reflector=True, feed=None, **env):
    """A small, fully-specified hardware description."""
    comps = []
    if reflector:
        comps.append({"id": "reflector", "type": "reflector", "parameters": {
            "diameter": {"value": 2.0, "unit": "m"},
            "focal_ratio": {"value": 0.4, "unit": ""}}})
    comps.append({"id": "feed", "type": "feed", "parameters": feed or {
        "edge_taper_e": {"value": -10, "unit": "dB"},
        "edge_taper_h": {"value": -10, "unit": "dB"}}
        if reflector else feed or {"hpbw_e": {"value": 60, "unit": "deg"},
                                   "hpbw_h": {"value": 60, "unit": "deg"}}})
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


# --------------------------------------------------------------------- units
def test_unit_conversion():
    assert convert(1.4, "GHz", where="t") == (1.4e9, "Hz")
    assert convert(4.4, "dB/100ft", where="t")[0] == pytest.approx(4.4 / 30.48)
    assert convert(20, "C", where="t") == (293.15, "K")
    with pytest.raises(HardwareConfigError, match="did you mean"):
        convert(1, "GHzz", where="t")


def test_typo_key_is_caught_with_suggestion():
    """The original file's  {value: 1.8, unt: :1}  (quoted so it parses)."""
    with pytest.raises(HardwareConfigError, match="did you mean 'unit'"):
        parse_param("output_vswr", {"value": 1.8, "unt": ":1"}, where="lna",
                    base_dir=ROOT)


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


# --------------------------------------------------------------- noise maths
def test_noise_figure_and_passive_temperatures():
    assert nf_db_to_temp_k(3.0103) == pytest.approx(290.0, rel=1e-4)   # NF 3 dB = 290 K
    assert passive_temp_k(-3.0103, 290.0) == pytest.approx(290.0, rel=1e-4)
    assert passive_temp_k(0.0, 290.0) == pytest.approx(0.0)


def test_friis_by_hand():
    perf = evaluate(load_hardware(data=_hw()))
    l_cable = 10 ** 0.1
    t_cable = 290 * (l_cable - 1)
    t_lna = 290 * (10 ** 0.07 - 1)
    t_sdr = 290 * (10 ** 0.43 - 1)
    expected = t_cable + t_lna * l_cable + t_sdr * l_cable / 100.0
    assert perf.center["t_rec_k"] == pytest.approx(expected, rel=1e-6)
    assert perf.center["gain_to_sdr_db"] == pytest.approx(19.0)


def test_reordering_puts_lna_noise_where_friis_says():
    base = evaluate(load_hardware(data=_hw()))
    swapped = evaluate(load_hardware(data=_hw(), order=["lna", "cable", "sdr"]))
    assert swapped.t_sys_k < base.t_sys_k          # LNA first is quieter
    t_lna = 290 * (10 ** 0.07 - 1)
    t_cable = 290 * (10 ** 0.1 - 1)
    t_sdr = 290 * (10 ** 0.43 - 1)
    expect = t_lna + t_cable / 100 + t_sdr * 10 ** 0.1 / 100
    assert swapped.center["t_rec_k"] == pytest.approx(expect, rel=1e-6)


def test_bad_order_and_duplicate_positions_are_errors():
    with pytest.raises(HardwareConfigError, match="not enabled chain"):
        load_hardware(data=_hw(), order=["lna", "nope", "sdr"])
    d = _hw()
    d["components"][-1]["chain_position"] = 2
    with pytest.raises(HardwareConfigError, match="more than one"):
        load_hardware(data=d)


def test_disabled_component_drops_out():
    d = _hw()
    d["components"][2]["enabled"] = False             # the cable
    hw = load_hardware(data=d)
    assert hw.order == ["lna", "sdr"] and hw.disabled == ["cable"]


# ------------------------------------------------------------------ passband
def test_passband_is_the_intersection():
    perf = evaluate(load_hardware(data=_hw()))
    assert perf.passband.nominal_lo_by == "sdr"
    assert perf.passband.nominal_hi_hz - perf.passband.nominal_lo_hz == pytest.approx(6e6)


def test_no_common_passband_is_reported():
    d = _hw()
    d["components"][3]["parameters"]["freq_max"] = {"value": 1.3, "unit": "GHz"}
    with pytest.raises(HardwareConfigError, match="no common passband"):
        evaluate(load_hardware(data=d))


def test_butterworth_noise_bandwidth():
    """B_N / B_3dB = (pi / 2n) / sin(pi / 2n) for an n-pole Butterworth."""
    n, bw = 3, 1e6
    f = np.linspace(F0 - 40 * bw, F0 + 40 * bw, 400001)
    r = -bandpass_attenuation_db(f, F0, bw, n)
    pb = measure_passband(f, r, F0, (F0 - bw / 2, F0 + bw / 2, "", ""))
    x = math.pi / (2 * n)
    assert pb.noise_bandwidth_hz / bw == pytest.approx(x / math.sin(x), rel=2e-3)
    assert pb.bw_3db_hz == pytest.approx(bw, rel=1e-3)


# ------------------------------------------------------------------- optics
def test_q_from_hpbw_round_trip():
    q = q_from_hpbw(60.0)
    assert math.cos(math.radians(30)) ** q == pytest.approx(0.5)


def test_uniform_aperture_limits():
    """Near-uniform illumination: HPBW 58.9 lambda/D, -17.6 dB sidelobe."""
    hw = load_hardware(data=_hw())
    from dataclasses import replace
    hw.feed = replace(hw.feed, q_e=1e-9, q_h=1e-9)
    hw.reflector = replace(hw.reflector, focal_ratio=5.0)
    op = Optics(F0, hw.feed, hw.reflector, 0.0)
    bm = op.beam()
    lam = C_LIGHT / F0
    assert op.eta_taper == pytest.approx(1.0, abs=1e-4)
    assert bm.hpbw_e_deg == pytest.approx(58.9 * lam / 2.0, rel=0.01)
    assert bm.sidelobe_db(bm.p_e) == pytest.approx(-17.6, abs=0.2)


def test_ten_db_taper_efficiency_is_textbook():
    """-10 dB feed taper at f/D 0.4: taper x spillover ~ 0.8 (Balanis fig. 15.20)."""
    perf = evaluate(load_hardware(data=_hw()))
    eta = perf.optics.eta_taper * perf.optics.eta_spill
    assert 0.75 < eta < 0.84


def test_feed_only_directivity_matches_its_pattern():
    perf = evaluate(load_hardware(data=_hw(reflector=False)))
    assert perf.beam.directivity_lin == pytest.approx(perf.optics.directivity_lin(), rel=1e-4)
    # cos^q pattern: D = 2(q + 1)
    q = q_from_hpbw(60.0)
    assert perf.optics.directivity_lin() == pytest.approx(2 * (q + 1))


def test_asymmetric_feed_gives_elliptical_beam_aligned_with_e_plane():
    feed = {"hpbw_e": {"value": 50, "unit": "deg"}, "hpbw_h": {"value": 90, "unit": "deg"},
            "e_plane_azimuth": {"value": 90, "unit": "deg"}}
    perf = evaluate(load_hardware(data=_hw(reflector=False, feed=feed)))
    assert perf.beam.hpbw_e_deg == pytest.approx(50, rel=1e-3)
    assert perf.hpbw_ew_deg == pytest.approx(50, rel=1e-3)     # E-plane points east
    assert perf.hpbw_ns_deg == pytest.approx(90, rel=1e-3)


def test_ruze_and_waveguide_cutoff():
    d = _hw()
    d["components"][0]["parameters"]["surface_rms_error"] = {"value": 5, "unit": "mm"}
    d["components"][1]["parameters"]["diameter"] = {"value": 0.15, "unit": "m"}
    perf = evaluate(load_hardware(data=d))
    lam = C_LIGHT / F0
    assert float(perf.optics.eta_surface()) == pytest.approx(
        math.exp(-(4 * math.pi * 0.005 / lam) ** 2))
    assert perf.hw.feed.band_lo_hz == pytest.approx(1.8412 * C_LIGHT / (math.pi * 0.15))


# --------------------------------------------------------- SEFD and SNR
def test_sefd_definition_and_snr_inverses():
    perf = evaluate(load_hardware(data=_hw()))
    c = perf.center
    assert c["sefd_jy"] == pytest.approx(2 * K_BOLTZMANN * c["t_sys_k"] / c["a_eff_m2"] / JY)
    s = required_flux_jy(3.0, 1e5, 1e6, 100.0)
    assert snr(s, 1e5, 1e6, 100.0) == pytest.approx(3.0)
    assert required_integration_s(3.0, s, 1e5, 1e6) == pytest.approx(100.0)
    # dual polarisation: same flux, sqrt(2) more SNR
    assert snr(s, 1e5, 1e6, 100.0, n_pol=2) == pytest.approx(3.0 * math.sqrt(2))


def test_ground_spillover_raises_antenna_temperature():
    cold = evaluate(load_hardware(data=_hw(ground_temperature={"value": 0, "unit": "K"})))
    warm = evaluate(load_hardware(data=_hw(ground_temperature={"value": 300, "unit": "K"})))
    fg = warm.optics.ground_fraction
    assert warm.center["t_a_k"] - cold.center["t_a_k"] == pytest.approx(300 * fg, rel=1e-6)


# --------------------------------------------------------- files
def test_touchstone_reader(tmp_path):
    f = tmp_path / "amp.s2p"
    f.write_text("! test\n# MHz S DB R 50\n"
                 "1400 -20 0 19.5 10 -40 0 -15 0\n1500 -20 0 18.5 10 -40 0 -15 0\n")
    fr, s21 = read_touchstone_s21(f)
    assert list(fr) == [1.4e9, 1.5e9] and list(s21) == [19.5, 18.5]


def test_example_file_end_to_end(tmp_path):
    import compute_performance
    assert compute_performance.main([str(EXAMPLE), "-o", str(tmp_path), "--no-plots"]) == 0
    import yaml
    out = yaml.safe_load((tmp_path / "performance_parameters.yaml").read_text())
    assert out["generated_from"]["center_frequency"]["value"] == pytest.approx(1420.405751768)
    assert out["noise"]["t_system"]["value"] > out["noise"]["t_receiver"]["value"]
    thr = yaml.safe_load((tmp_path / "detection_thresholds.yaml").read_text())
    assert thr["database_index"]["nvss_s1400_min"]["value"] > 0


def test_bridge_to_planner_preserves_sefd(tmp_path):
    import compute_performance
    compute_performance.main([str(EXAMPLE), "-o", str(tmp_path), "--no-plots"])
    from hydroline.antenna import AntennaModel
    from hydroline.hardware.bridge import antenna_from_performance
    perf = evaluate(load_hardware(EXAMPLE))
    model = AntennaModel(antenna_from_performance(tmp_path / "performance_parameters.yaml"))
    assert model.sefd_jy == pytest.approx(perf.sefd_jy, rel=1e-3)
    assert model.effective_area_m2 == pytest.approx(perf.center["a_eff_m2"], rel=1e-3)
