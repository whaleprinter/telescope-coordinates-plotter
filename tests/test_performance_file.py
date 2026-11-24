"""hydroline's reader for a performance_parameters.yaml (run.py --performance).

The file is the only link between hydroline and the separate hardware_model
program, so the first test uses a hand-written file; the second checks the
two programs agree when hardware_model is present.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hydroline.antenna import AntennaModel  # noqa: E402
from hydroline.constants import JY, K_BOLTZMANN  # noqa: E402
from hydroline.performance_file import PerformanceFileError, antenna_from_performance  # noqa: E402

MINIMAL = """
generated_from: {telescope: test dish, center_frequency: {value: 1420.405751768, unit: MHz}}
collection: {effective_area: {value: 1.0, unit: m^2}}
beam_pattern: {hpbw_e_plane: {value: 9.0, unit: deg}, hpbw_h_plane: {value: 16.0, unit: deg}}
noise: {t_system: {value: 100.0, unit: K}}
gain: {noise_bandwidth: {value: 6.0, unit: MHz}}
system: {spectral_channel: {value: 5.0, unit: kHz}, n_pol: {value: 1, unit: ''},
         radiometer_factor: {value: 1, unit: ''}}
position: {pointing_altitude: {value: 90, unit: deg}, pointing_azimuth: {value: 0, unit: deg}}
"""


def test_reader_maps_every_quantity(tmp_path):
    f = tmp_path / "performance_parameters.yaml"
    f.write_text(MINIMAL)
    ant = antenna_from_performance(f)
    model = AntennaModel(ant)
    assert ant.hpbw_deg == pytest.approx(12.0)                  # sqrt(9 x 16)
    assert model.effective_area_m2 == pytest.approx(1.0)
    assert model.sefd_jy == pytest.approx(2 * K_BOLTZMANN * 100.0 / 1.0 / JY)
    assert ant.rf_bandwidth_hz == pytest.approx(6e6) and ant.channel_hz == pytest.approx(5e3)


def test_reader_names_a_missing_key(tmp_path):
    f = tmp_path / "bad.yaml"
    f.write_text("collection: {}\n")
    with pytest.raises(PerformanceFileError, match="effective_area"):
        antenna_from_performance(f)


def test_round_trip_with_hardware_model(tmp_path):
    hm = pytest.importorskip("hardware_model.compute_performance")
    from hardware_model.inputs import load_hardware
    from hardware_model.performance import evaluate
    example = ROOT / "hardware_model" / "hardware_parameters.yaml"
    assert hm.main([str(example), "-o", str(tmp_path), "--no-plots"]) == 0
    perf = evaluate(load_hardware(example))
    model = AntennaModel(antenna_from_performance(tmp_path / "performance_parameters.yaml"))
    assert model.sefd_jy == pytest.approx(perf.sefd_jy, rel=1e-3)
    assert model.effective_area_m2 == pytest.approx(perf.a_eff_m2, rel=1e-3)
    assert math.isfinite(model.hpbw_deg)
