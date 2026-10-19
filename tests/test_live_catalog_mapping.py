"""Tests for hydroline/live_catalog.py that never touch the network.

Every value here is either a literal row pulled from a real, verified query
made while building this feature, or a small synthetic fixture built to
exercise one specific piece of logic (the merge/dedup rule, the ranking
rule). Nothing here imports astroquery or opens a socket -- the fetch
functions that do are exercised separately in test_live_catalog_network.py,
gated behind the `network` marker (see conftest.py) so a routine
`pytest tests -q` stays exactly as fast and offline as it always has.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hydroline.antenna import AntennaModel                          # noqa: E402
from hydroline.catalog import load_catalog                          # noqa: E402
from hydroline.config import PRESET_ANTENNAS                        # noqa: E402
from hydroline.live_catalog import (                                # noqa: E402
    RAW_COLUMNS,
    _fetch_alfalfa,
    _implied_hi_peak_jy,
    alfalfa_row_to_hydroline,
    antenna_flux_floor_jy,
    enrich_with_simbad,
    kind_from_simbad_otype,
    merge_nvss_alfalfa,
    nvss_row_to_hydroline,
)


# ---------------------------------------------------------------------------
# row mapping -- literal values from real, verified queries
# ---------------------------------------------------------------------------
def test_nvss_row_to_hydroline_units():
    row = {"NVSS": "231940+585440", "RAJ2000": "23 19 40.60",
          "DEJ2000": "+58 54 40.0", "S1.4": 105.1, "MajAxis": 127.1,
          "MinAxis": 87.2}
    out = nvss_row_to_hydroline(row)
    assert out["name"] == "NVSS J231940+585440"
    assert out["kind"] == "continuum_unclassified"
    assert out["ra"] == "23 19 40.60" and out["dec"] == "+58 54 40.0"
    assert out["s1400_jy"] == pytest.approx(0.1051)          # mJy -> Jy
    assert out["maj_arcmin"] == pytest.approx(127.1 / 60.0)  # arcsec -> arcmin
    assert out["min_arcmin"] == pytest.approx(87.2 / 60.0)
    assert out["hi_emitter"] == 0
    assert np.isnan(out["dist_mpc"])
    assert set(out) == set(RAW_COLUMNS)


def test_nvss_row_to_hydroline_missing_size():
    """Unresolved point sources have no MajAxis/MinAxis in real NVSS rows."""
    out = nvss_row_to_hydroline({"NVSS": "X", "RAJ2000": "00 00 00.0",
                                 "DEJ2000": "+00 00 00", "S1.4": 10.0,
                                 "MajAxis": None, "MinAxis": None})
    assert np.isnan(out["maj_arcmin"]) and np.isnan(out["min_arcmin"])


def test_alfalfa_row_to_hydroline_units():
    row = {"AGC": "105367", "Name": "", "RAJ2000": "00 00 00.4",
          "DEJ2000": "+05 26 36", "Vhel": 11983, "W50": 274, "Dist": 166.0,
          "logMHI": 9.87}
    out = alfalfa_row_to_hydroline(row)
    assert out["name"] == "AGC 105367"       # blank Name -> AGC fallback
    assert out["kind"] == "hi_galaxy" and out["hi_emitter"] == 1
    assert out["dist_mpc"] == 166.0
    assert out["hi_mass_msun"] == pytest.approx(10.0 ** 9.87, rel=1e-9)
    assert out["vsys_kms"] == 11983.0 and out["w50_kms"] == 274.0
    assert np.isnan(out["s1400_jy"])
    assert set(out) == set(RAW_COLUMNS)


def test_alfalfa_row_to_hydroline_uses_real_name_when_present():
    out = alfalfa_row_to_hydroline({"AGC": "331060", "Name": "478-009b",
                                    "RAJ2000": "00 00 02.5", "DEJ2000": "+23 05 05",
                                    "Vhel": 4463, "W50": 160, "Dist": 50.6,
                                    "logMHI": 9.07})
    assert out["name"] == "478-009b"
    assert out["aliases"] == "AGC 331060"


def test_kind_from_simbad_otype():
    assert kind_from_simbad_otype("QSO") == "quasar"
    assert kind_from_simbad_otype("SNR") == "snr"
    assert kind_from_simbad_otype("totally-unknown-code") is None


# ---------------------------------------------------------------------------
# the real integration seam: mapped rows -> the actual load_catalog()
# ---------------------------------------------------------------------------
def test_mapped_rows_load_through_the_real_load_catalog(tmp_path):
    """Exercises the actual seam: written CSV -> unmodified load_catalog()."""
    rows = [
        # S1.4 in mJy, MajAxis/MinAxis in arcsec -- real NVSS units. 3600" = 1 deg.
        nvss_row_to_hydroline({"NVSS": "AAA", "RAJ2000": "10 00 00.0",
                               "DEJ2000": "+40 00 00", "S1.4": 500.0,
                               "MajAxis": 3600.0, "MinAxis": 3600.0}),
        alfalfa_row_to_hydroline({"AGC": "1", "Name": "", "RAJ2000": "11 00 00.0",
                                  "DEJ2000": "+41 00 00", "Vhel": 1000, "W50": 200,
                                  "Dist": 50.0, "logMHI": 9.5}),
    ]
    csv_path = tmp_path / "live.csv"
    pd.DataFrame(rows, columns=RAW_COLUMNS).to_csv(csv_path, index=False)

    catalog = load_catalog(csv_path)
    assert len(catalog) == 2
    df = catalog.frame.set_index("name")

    nvss = df.loc["NVSS JAAA"]
    assert nvss["ra_deg"] == pytest.approx(150.0, abs=1e-6)
    assert nvss["dec_deg"] == pytest.approx(40.0, abs=1e-6)
    assert nvss["size_deg"] == pytest.approx(1.0, abs=1e-6)   # 3600"x3600" -> 1 deg

    hi = df.loc["AGC 1"]
    assert hi["hi_emitter"]
    assert hi["hi_flux_jykms"] > 0                            # derived, not imported
    assert hi["hi_peak_jy"] == pytest.approx(hi["hi_flux_jykms"] / 200.0)


# ---------------------------------------------------------------------------
# merge / dedup
# ---------------------------------------------------------------------------
def _nvss(name, ra, dec, flux_mjy):
    """flux_mjy: real NVSS units (mJy) -- nvss_row_to_hydroline converts to Jy."""
    return nvss_row_to_hydroline({"NVSS": name, "RAJ2000": ra, "DEJ2000": dec,
                                  "S1.4": flux_mjy, "MajAxis": None, "MinAxis": None})


def _alfalfa(agc, ra, dec, vhel=1000, w50=100, dist=50.0, logmhi=9.0):
    return alfalfa_row_to_hydroline({"AGC": agc, "Name": "", "RAJ2000": ra,
                                     "DEJ2000": dec, "Vhel": vhel, "W50": w50,
                                     "Dist": dist, "logMHI": logmhi})


def test_merge_nvss_alfalfa_matches_close_pair_keeps_distant_ones_separate():
    nvss = pd.DataFrame([
        _nvss("AAA", "01 00 00.0", "+30 00 00", 200.0),   # will match ALFALFA #1
        _nvss("BBB", "05 00 00.0", "+10 00 00", 300.0),   # far from any ALFALFA row
    ], columns=RAW_COLUMNS)
    alfalfa = pd.DataFrame([
        _alfalfa("1", "01 00 00.5", "+30 00 05"),          # ~8" from NVSS AAA
        _alfalfa("2", "20 00 00.0", "-10 00 00"),           # far from any NVSS row
    ], columns=RAW_COLUMNS)

    merged = merge_nvss_alfalfa(nvss, alfalfa)
    assert len(merged) == 3                                  # 1 merged + 1 + 1

    matched = merged[merged["name"] == "AGC 1"].iloc[0]
    assert matched["s1400_jy"] == pytest.approx(0.2)          # 200 mJy -> 0.2 Jy, carries NVSS flux
    assert matched["hi_mass_msun"] > 0                        # keeps ALFALFA HI data
    assert "NVSS cross-match" in matched["notes"]

    assert (merged["name"] == "NVSS JBBB").any()
    assert (merged["name"] == "AGC 2").any()


def test_merge_nvss_alfalfa_handles_empty_inputs():
    empty = pd.DataFrame(columns=RAW_COLUMNS)
    one = pd.DataFrame([_nvss("X", "01 00 00.0", "+30 00 00", 100.0)],
                       columns=RAW_COLUMNS)
    assert len(merge_nvss_alfalfa(empty, empty)) == 0
    assert len(merge_nvss_alfalfa(one, empty)) == 1
    assert len(merge_nvss_alfalfa(empty, one)) == 1


# ---------------------------------------------------------------------------
# antenna-derived flux floor
# ---------------------------------------------------------------------------
def test_antenna_flux_floor_monotonic_with_dish_size():
    """A bigger/more sensitive dish should need a lower flux to be worth
    fetching -- checked across every real preset, not just two points."""
    order = ["horn-cantenna", "horn-small", "horn-large", "dish-1m2",
            "dish-1m5", "dish-3m", "dish-4m6"]
    floors = [antenna_flux_floor_jy(AntennaModel(PRESET_ANTENNAS[k]))
             for k in order]
    assert all(a >= b for a, b in zip(floors, floors[1:])), floors


def test_antenna_flux_floor_respects_absolute_floor():
    m = AntennaModel(PRESET_ANTENNAS["dish-4m6"])
    floor = antenna_flux_floor_jy(m, multiplier=1e-9, absolute_floor_jy=0.3)
    assert floor == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# implied HI peak flux + ALFALFA ranking (via a monkeypatched fetch)
# ---------------------------------------------------------------------------
def test_implied_hi_peak_jy_matches_catalog_derivation():
    from hydroline.constants import HI_MASS_COEFF

    dist, mass, w50 = 20.0, 1e9, 100.0
    expected = (mass / (HI_MASS_COEFF * dist ** 2)) / w50
    assert _implied_hi_peak_jy(dist, mass, w50) == pytest.approx(expected)


def test_implied_hi_peak_jy_nan_on_bad_input():
    assert np.isnan(_implied_hi_peak_jy(np.nan, 1e9, 100.0))
    assert np.isnan(_implied_hi_peak_jy(20.0, 1e9, 0.0))


def test_fetch_alfalfa_keeps_brightest_by_implied_peak_flux(monkeypatch):
    """Exercises _fetch_alfalfa's ranking/truncation without astroquery --
    monkeypatches astroquery.vizier.Vizier itself (imported lazily inside
    the function), the seam build_live_catalog's orchestration relies on
    for testability.
    """
    class _FakeTable(list):
        colnames = ["Name", "AGC", "RAJ2000", "DEJ2000", "Vhel", "W50",
                   "Dist", "logMHI"]

    rows = [
        # bright: close + massive -> high implied peak flux
        ("", "1", "01 00 00.0", "+30 00 00", 1000, 100, 5.0, 9.5),
        # faint: far + modest mass
        ("", "2", "02 00 00.0", "+31 00 00", 2000, 300, 200.0, 9.0),
        # brightest of the three
        ("", "3", "03 00 00.0", "+32 00 00", 1500, 80, 3.0, 9.8),
    ]
    fake_result = [_FakeTable(rows)]

    class _FakeVizier:
        def __init__(self, *a, **k):
            pass

        def query_constraints(self, *a, **k):
            return fake_result

    fake_module = type(sys)("astroquery.vizier")
    fake_module.Vizier = _FakeVizier
    monkeypatch.setitem(sys.modules, "astroquery.vizier", fake_module)

    out = _fetch_alfalfa(dec_lo=25.0, dec_hi=35.0, max_rows=1000, top_n=2)
    assert len(out) == 2
    # AGC 3 (highest implied peak flux) must be kept; AGC 2 (lowest) dropped.
    assert "AGC 3" in out["name"].tolist()
    assert "AGC 2" not in out["name"].tolist()


# ---------------------------------------------------------------------------
# SIMBAD enrichment orchestration (fetch_fn monkeypatched -- no network)
# ---------------------------------------------------------------------------
def test_enrich_with_simbad_ranks_and_caps(monkeypatch):
    rows = pd.DataFrame([
        _nvss("A", "01 00 00.0", "+30 00 00", 1.0),    # faint
        _nvss("B", "02 00 00.0", "+31 00 00", 100.0),  # brightest
        _nvss("C", "03 00 00.0", "+32 00 00", 10.0),
    ], columns=RAW_COLUMNS)

    calls: list[str] = []

    class _FakeSimbadTable:
        """Minimal stand-in for the real astropy Table query_region returns:
        len() is the row count, table["col"][0] is that column's first
        value, table.colnames lists the columns -- exactly what
        enrich_with_simbad actually calls."""
        colnames = ["main_id", "otype"]

        def __len__(self):
            return 1

        def __getitem__(self, col):
            return {"main_id": ["Match"], "otype": ["QSO"]}[col]

    def fake_fetch(coord_obj, radius):
        calls.append("x")
        return _FakeSimbadTable()          # len()==1 -> unambiguous match

    enriched, notes = enrich_with_simbad(rows, fake_fetch, max_rows=2, sleep_s=0)
    assert len(calls) == 2                              # capped at max_rows
    assert any("limited to the 2" in n for n in notes)
    # the two brightest (B, C) should have been queried, not the faintest (A)
    assert enriched.loc[rows["name"] == "NVSS JA", "kind"].iloc[0] == "continuum_unclassified"


def test_enrich_with_simbad_skips_ambiguous_and_empty_matches():
    rows = pd.DataFrame([_nvss("A", "01 00 00.0", "+30 00 00", 1.0)],
                        columns=RAW_COLUMNS)

    def fetch_none(coord_obj, radius):
        return None

    enriched, notes = enrich_with_simbad(rows, fetch_none, max_rows=10, sleep_s=0)
    assert enriched.iloc[0]["kind"] == "continuum_unclassified"
    assert "matched 0 of 1" in notes[-1]


def test_enrich_with_simbad_survives_a_query_exception():
    rows = pd.DataFrame([_nvss("A", "01 00 00.0", "+30 00 00", 1.0)],
                        columns=RAW_COLUMNS)

    def fetch_raises(coord_obj, radius):
        raise RuntimeError("simulated network failure")

    # must not raise -- a single bad query degrades that one row, not the run
    enriched, notes = enrich_with_simbad(rows, fetch_raises, max_rows=10, sleep_s=0)
    assert len(enriched) == 1


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
