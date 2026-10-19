"""Real queries against NVSS, ALFALFA and SIMBAD.

Every test here is marked `network` and skipped by default -- see
conftest.py. Run explicitly with:

    pytest tests/test_live_catalog_network.py -q --run-network

These check the actual services still return what this project's mapping
code expects (real column names, real units), not the offline logic already
covered in test_live_catalog_mapping.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("astroquery")

pytestmark = pytest.mark.network

from hydroline.antenna import AntennaModel                          # noqa: E402
from hydroline.config import PRESET_ANTENNAS                        # noqa: E402
from hydroline.live_catalog import (                                # noqa: E402
    LiveCatalogError,
    _fetch_alfalfa,
    _fetch_nvss,
    build_live_catalog,
)


def test_nvss_query_returns_expected_columns():
    df = _fetch_nvss(dec_lo=58.0, dec_hi=59.0, flux_floor_jy=1.0, max_rows=50)
    assert not df.empty
    assert (df["s1400_jy"] > 0).all()
    assert df["dec"].str.startswith(("+58", "+59")).all()


def test_alfalfa_query_returns_expected_columns():
    df = _fetch_alfalfa(dec_lo=30.0, dec_hi=36.0, max_rows=5000, top_n=10)
    assert not df.empty
    assert (df["hi_emitter"] == 1).all()
    assert (df["dist_mpc"] > 0).all()


def test_alfalfa_footprint_gap_outside_arecibo_sky():
    """The real, previously-verified limitation: ~0 results well outside
    Dec 0-36 deg."""
    df = _fetch_alfalfa(dec_lo=55.0, dec_hi=65.0, max_rows=5000, top_n=10)
    assert len(df) == 0


def test_build_live_catalog_end_to_end(tmp_path):
    model = AntennaModel(PRESET_ANTENNAS["dish-1m5"])
    result = build_live_catalog(
        dec_deg=36.0011, half_width_deg=model.beam_radius_deg, model=model,
        cache_dir=tmp_path, include_simbad=False,   # SIMBAD covered above; skip for speed
    )
    assert len(result.catalog) > 0
    assert result.counts["total"] == len(result.catalog.frame)
    assert not result.from_cache

    # a second call with the same parameters must hit the cache
    result2 = build_live_catalog(
        dec_deg=36.0011, half_width_deg=model.beam_radius_deg, model=model,
        cache_dir=tmp_path, include_simbad=False,
    )
    assert result2.from_cache


def test_build_live_catalog_bad_host_falls_back_gracefully(tmp_path, monkeypatch):
    """A total, unrecoverable live-query failure with nothing cached must
    raise LiveCatalogError (which cli.py catches and falls back to the
    static catalogue), not some unrelated exception."""
    import hydroline.live_catalog as lc

    monkeypatch.setattr(lc, "_fetch_nvss", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("simulated outage")))
    monkeypatch.setattr(lc, "_fetch_alfalfa", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("simulated outage")))

    model = AntennaModel(PRESET_ANTENNAS["dish-1m5"])
    with pytest.raises(LiveCatalogError):
        build_live_catalog(dec_deg=36.0011, half_width_deg=model.beam_radius_deg,
                           model=model, cache_dir=tmp_path)


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q", "--run-network"]))
