"""Physics and plumbing tests.

Run with ``python3 -m pytest tests -q`` or just ``python3 tests/test_hydroline.py``.

The point of these is not coverage -- it is to pin the numbers that are easy to
get quietly wrong: frame conventions, the factor of two in T_A, the beam
coupling limits, and the sidereal rate.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hydroline.analysis import (                                    # noqa: E402
    _separation_deg,
    beam_encounters,
    detectability_table,
    sky_coverage,
    transit_duration_seconds,
)
from hydroline.antenna import SIDEREAL_RATE_DEG_PER_HOUR, AntennaModel  # noqa: E402
from hydroline.catalog import (                                    # noqa: E402
    flux_from_brightness_temperature,
    load_catalog,
    solar_system_track,
)
from hydroline.config import (                                     # noqa: E402
    Antenna,
    Configuration,
    DUKE_PHYSICS,
    Observation,
    PRESET_ANTENNAS,
    Site,
)
from hydroline.constants import C_LIGHT, HI_REST_HZ, JY, K_BOLTZMANN  # noqa: E402
from hydroline.pointing import (                                   # noqa: E402
    configure_iers,
    lsr_apex,
    validate_track,
    zenith_track,
)

configure_iers()
OBS = Observation(start_utc="2026-09-01T00:00:00", duration_hours=24.0,
                  step_minutes=15.0)


@pytest.fixture(scope="module")
def track():
    return zenith_track(DUKE_PHYSICS, OBS, PRESET_ANTENNAS["dish-1m5"])


@pytest.fixture(scope="module")
def model():
    return AntennaModel(PRESET_ANTENNAS["dish-1m5"])


@pytest.fixture(scope="module")
def catalog():
    return load_catalog()


# ---------------------------------------------------------------------------
# frames
# ---------------------------------------------------------------------------
def test_apparent_declination_is_the_geodetic_latitude(track):
    """The defining identity of a zenith pointing, in the apparent frame."""
    err = np.abs(track.table["dec_apparent_deg"] - DUKE_PHYSICS.latitude_deg)
    assert err.max() * 3600.0 < 1.0, "apparent Dec must equal geodetic latitude"


def test_apparent_ra_is_the_local_apparent_sidereal_time(track):
    err = np.abs(track.table["ra_apparent_minus_lst_arcmin"])
    assert err.max() * 60.0 < 1.0, "apparent RA must equal LAST at the zenith"


def test_validate_track_agrees(track):
    result = validate_track(track)
    assert result["is_zenith_pointing"] and result["passed"]


def test_icrs_differs_from_apparent_by_precession(track):
    """J2000 is ~27 years stale, so ICRS RA must lag LAST by ~20 arcmin.

    General precession is 50.29"/yr in ecliptic longitude, about 46.1"/yr
    projected into right ascension near the equinox.
    """
    lag = -float(np.mean(track.table["ra_icrs_minus_lst_arcmin"]))
    years = (track.times[0].jyear - 2000.0)
    expected = years * 46.1 / 60.0
    assert 0.6 * expected < lag < 1.4 * expected, (
        f"ICRS-to-apparent RA offset {lag:.1f}' is not precession-like "
        f"(expected about {expected:.1f}')"
    )


def test_declination_wander_is_the_pole_offset_not_noise(track):
    """ICRS Dec swings with RA because the J2000 pole is not the pole of date.

    Amplitude should be the pole displacement since J2000, ~20"/yr.
    """
    swing = float(np.ptp(track.table["dec_icrs_deg"])) * 3600.0 / 2.0
    years = track.times[0].jyear - 2000.0
    assert 0.5 * 20.0 * years < swing < 1.6 * 20.0 * years


def test_right_ascension_sweeps_at_the_sidereal_rate(track):
    ra = np.unwrap(track.table["ra_icrs_deg"].to_numpy(), period=360.0)
    hours = track.table["hours_from_start"].to_numpy()
    rate = np.polyfit(hours, ra, 1)[0]
    assert abs(rate - SIDEREAL_RATE_DEG_PER_HOUR) < 0.01


def test_moving_the_site_moves_the_beam_declination():
    for lat in (-33.0, 0.0, 19.8, 52.4):
        site = Site(name="test", latitude_deg=lat, longitude_deg=10.0)
        t = zenith_track(site, Observation(start_utc="2026-03-20T00:00:00",
                                           duration_hours=2.0,
                                           step_minutes=60.0))
        assert abs(float(np.mean(t.table["dec_apparent_deg"])) - lat) < 1 / 60.0


def test_off_zenith_pointing_still_runs():
    ant = Antenna(pointing_alt_deg=60.0, pointing_az_deg=180.0)
    t = zenith_track(DUKE_PHYSICS, Observation(start_utc="2026-09-01T00:00:00",
                                               duration_hours=4.0,
                                               step_minutes=60.0), ant)
    # Pointing 30 deg south of the zenith lands 30 deg south in declination.
    assert abs(t.dec_deg - (DUKE_PHYSICS.latitude_deg - 30.0)) < 0.5
    assert validate_track(t)["passed"] is None      # identity does not apply


# ---------------------------------------------------------------------------
# doppler
# ---------------------------------------------------------------------------
def test_lsr_apex_lands_where_the_convention_says():
    apex = lsr_apex()
    assert abs(apex.ra.hour - 18.06) < 0.05
    assert abs(apex.dec.deg - 30.0) < 0.1


def test_velocity_corrections_are_physically_bounded(track):
    v = track.table["v_lsr_correction_kms"].to_numpy()
    b = track.table["v_barycentric_correction_kms"].to_numpy()
    # Earth's orbit is 29.8 km/s, its spin 0.46 km/s, the solar motion 20 km/s.
    assert np.abs(b).max() < 30.5
    assert np.abs(v).max() < 51.0
    assert np.ptp(v) > 1.0, "the correction must actually vary with time"


# ---------------------------------------------------------------------------
# antenna
# ---------------------------------------------------------------------------
def test_beam_and_aperture_formulas(model):
    a = model.antenna
    lam = C_LIGHT / HI_REST_HZ
    assert math.isclose(model.wavelength_m, lam, rel_tol=1e-12)
    assert math.isclose(model.hpbw_deg,
                        a.beam_factor_deg * lam / a.diameter_m, rel_tol=1e-12)
    assert math.isclose(model.geometric_area_m2,
                        math.pi * (a.diameter_m / 2) ** 2, rel_tol=1e-12)
    assert math.isclose(model.effective_area_m2,
                        a.aperture_efficiency * model.geometric_area_m2,
                        rel_tol=1e-12)
    assert math.isclose(model.gain_dbi, 10 * math.log10(
        4 * math.pi * model.effective_area_m2 / lam ** 2), rel_tol=1e-12)


def test_sefd_and_sensitivity(model):
    expected = (2 * K_BOLTZMANN * model.antenna.t_sys_k
                / model.effective_area_m2 / JY)
    assert math.isclose(model.sefd_jy, expected, rel_tol=1e-12)
    # 1 Jy on a 1 m^2 effective aperture is 0.36 mK.
    assert math.isclose(model.sensitivity_k_per_jy,
                        model.effective_area_m2 * JY / (2 * K_BOLTZMANN),
                        rel_tol=1e-12)


def test_radiometer_equation_scales_as_one_over_root_t(model):
    n1 = model.delta_t_rms_k(integration_s=100.0)
    n2 = model.delta_t_rms_k(integration_s=400.0)
    assert math.isclose(n1 / n2, 2.0, rel_tol=1e-9)
    b1 = model.delta_t_rms_k(bandwidth_hz=1e6)
    b2 = model.delta_t_rms_k(bandwidth_hz=4e6)
    assert math.isclose(b1 / b2, 2.0, rel_tol=1e-9)


def test_flux_and_brightness_temperature_routes_agree(model):
    """The two ways of computing T_A must meet in both limits.

    Point source:      T_A = S * A_eff / 2k
    Fully resolved:    T_A = eta_MB * T_b
    """
    tiny = model.hpbw_deg / 300.0
    flux = 100.0
    assert math.isclose(
        model.antenna_temperature_from_flux_k(flux, tiny),
        flux * model.sensitivity_k_per_jy, rel_tol=2e-4)

    # A source 30 beams across is fully resolved.
    huge = model.hpbw_deg * 30.0
    tb = 80.0
    omega_src = 1.133 * math.radians(huge) ** 2
    s_total = 2 * K_BOLTZMANN * tb * omega_src / model.wavelength_m ** 2 / JY
    via_flux = model.antenna_temperature_from_flux_k(s_total, huge)
    via_tb = model.antenna_temperature_from_tb_k(tb, huge)
    assert math.isclose(via_flux, via_tb, rel_tol=1e-2)
    assert math.isclose(via_tb, model.main_beam_efficiency * tb, rel_tol=1e-2)


def test_beam_coupling_limits(model):
    assert model.flux_coupling(0.0) == pytest.approx(1.0)
    assert model.flux_coupling(model.hpbw_deg) == pytest.approx(0.5)
    assert model.flux_coupling(1e4) < 1e-5
    for size in (0.01, 1.0, 9.0, 100.0):
        assert math.isclose(model.flux_coupling(size)
                            + model.beam_dilution(size), 1.0, rel_tol=1e-12)


def test_horn_beam_uses_both_planes():
    m = AntennaModel(PRESET_ANTENNAS["horn-large"])
    a = m.antenna
    lam = m.wavelength_m
    assert math.isclose(m.hpbw_e_deg, a.horn_factor_e_deg * lam / a.aperture_e_m)
    assert math.isclose(m.hpbw_h_deg, a.horn_factor_h_deg * lam / a.aperture_h_m)
    assert m.hpbw_e_deg < m.hpbw_deg < m.hpbw_h_deg
    assert math.isclose(m.geometric_area_m2, a.aperture_e_m * a.aperture_h_m)


def test_velocity_resolution_matches_the_channel_width(model):
    expected = C_LIGHT * model.antenna.channel_hz / HI_REST_HZ / 1000.0
    assert math.isclose(model.velocity_resolution_kms, expected, rel_tol=1e-12)
    assert math.isclose(
        model.matched_bandwidth_hz(model.velocity_resolution_kms),
        model.antenna.channel_hz, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# catalogue
# ---------------------------------------------------------------------------
def test_catalog_derived_columns_are_sane(catalog):
    df = catalog.frame
    assert len(df) > 40
    assert df["ra_deg"].between(0, 360).all()
    assert df["dec_deg"].between(-90, 90).all()
    assert (df["size_deg"] > 0).all()
    gal = df[df["hi_emitter"] & df["hi_mass_msun"].notna()]
    assert (gal["hi_flux_jykms"] > 0).all()


def test_m31_hi_flux_matches_the_literature(catalog):
    """M_HI = 2.356e5 D^2 S_int, so M31 must come out near 3e4 Jy km/s."""
    m31 = catalog.by_name("M31")
    assert 1.5e4 < m31["hi_flux_jykms"] < 5.0e4
    assert 30.0 < m31["hi_peak_jy"] < 90.0


def test_moon_and_sun_flux_densities():
    """Sanity anchors: the Moon is ~900 Jy and the quiet Sun ~5e5 Jy at 21 cm."""
    moon = flux_from_brightness_temperature(220.0, 0.52)
    assert 700 < moon < 1100
    sun = flux_from_brightness_temperature(1.0e5, 0.533)
    assert 3e5 < sun < 7e5


def test_solar_system_positions_are_reasonable(track):
    ss = solar_system_track(track.times, track.location, ("sun", "moon"))
    assert set(ss) == {"sun", "moon"}
    assert 0.52 < ss["sun"]["mean_size_deg"] < 0.55
    assert 0.48 < ss["moon"]["mean_size_deg"] < 0.57
    # Regression guard.  get_body returns a geocentric position carrying a
    # distance; transforming that straight to ICRS re-origins it at the
    # barycentre and puts the Sun on the opposite side of the sky.  On
    # 2026-09-01 the Sun is at RA 10.66 h, Dec +8.49 deg.
    sun = ss["sun"]["frame"].iloc[0]
    assert abs(sun["ra_icrs_deg"] / 15.0 - 10.658) < 0.02
    assert abs(sun["dec_icrs_deg"] - 8.486) < 0.05
    moon = ss["moon"]["frame"].iloc[0]
    assert abs(moon["ra_icrs_deg"] / 15.0 - 1.484) < 0.02
    assert abs(moon["dec_icrs_deg"] - 13.102) < 0.05


def test_solar_system_separation_from_zenith_matches_altitude(track):
    """A body's distance from a zenith beam must equal 90 deg minus altitude."""
    from hydroline.analysis import _separation_deg

    ss = solar_system_track(track.times, track.location, ("sun", "moon"))
    for key, info in ss.items():
        sep = _separation_deg(info["coords"].ra.deg, info["coords"].dec.deg,
                              track.table["ra_icrs_deg"].to_numpy(),
                              track.table["dec_icrs_deg"].to_numpy())
        zenith_angle = 90.0 - info["frame"]["altitude_deg"].to_numpy()
        # Agreement to well under a degree; the residual is topocentric
        # parallax, which is ~1 deg for the Moon and negligible otherwise.
        tol = 1.2 if key == "moon" else 0.05
        assert np.abs(sep - zenith_angle).max() < tol, key


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------
def test_transit_duration_matches_the_separation_curve(track, model):
    """The analytic chord length must agree with the numerically sampled one."""
    fine = zenith_track(DUKE_PHYSICS,
                        Observation(start_utc="2026-09-01T00:00:00",
                                    duration_hours=26.0, step_minutes=0.5),
                        model.antenna)
    dec_src = track.dec_deg - 2.0
    ra_src = 40.0
    sep = _separation_deg(ra_src, dec_src,
                          fine.table["ra_icrs_deg"].to_numpy(),
                          fine.table["dec_icrs_deg"].to_numpy())
    numeric = float((sep <= model.beam_radius_deg).sum()) * 30.0   # 0.5 min
    analytic = transit_duration_seconds(model, track.dec_deg, dec_src)
    assert abs(numeric - analytic) / analytic < 0.05


def test_transit_duration_edge_cases(model):
    dec0 = 36.0
    # Exactly on the beam axis gives the longest transit.
    on_axis = transit_duration_seconds(model, dec0, dec0)
    off = transit_duration_seconds(model, dec0, dec0 + 3.0)
    assert on_axis > off > 0
    # Outside the beam entirely, nothing.
    assert transit_duration_seconds(model, dec0, dec0 + 45.0) == 0.0
    # The on-axis chord is close to HPBW / (15.04 cos dec).
    naive = model.hpbw_deg / (SIDEREAL_RATE_DEG_PER_HOUR
                              * math.cos(math.radians(dec0))) * 3600.0
    assert abs(on_axis - naive) / naive < 0.05


def test_only_sources_near_the_beam_declination_ever_enter(track, model, catalog):
    det = detectability_table(catalog, model, track.dec_deg)
    inside = det[det["enters_beam"]]
    assert (inside["dec_offset_from_beam_deg"].abs()
            <= model.beam_radius_deg + 1e-9).all()
    outside = det[~det["enters_beam"]]
    assert (outside["transit_duration_s"] == 0).all()
    assert not outside["detectable_continuum"].any()


def test_encounters_agree_with_the_declination_test(track, model, catalog):
    enc = beam_encounters(track, catalog, model)
    passes = enc[enc["in_beam"]]
    assert len(passes) > 0
    for _, r in passes.iterrows():
        assert r["min_separation_deg"] <= model.beam_radius_deg + 0.2
        assert 0 < r["duration_minutes"] < 24 * 60
        assert 0.5 <= r["beam_response"] <= 1.0


def test_southern_sources_are_never_in_a_northern_beam(track, model, catalog):
    det = detectability_table(catalog, model, track.dec_deg)
    for name in ("Large Magellanic Cloud", "Sagittarius A"):
        row = det[det["name"] == name].iloc[0]
        assert not row["enters_beam"]
        assert not row["detectable_continuum"]


def test_off_axis_contamination_catches_the_sun(track):
    """The Sun near solstice is outside a 25-deg beam but still contributes.

    At Durham the zenith sits at Dec +36 and the solstice Sun at +23.4, a
    12.6 deg offset -- just outside the 12.5 deg half-power radius of the small
    horn.  At 4e5 Jy it still injects several kelvin through the beam skirt,
    which is the whole point of tracking off-axis sources.
    """
    from hydroline.catalog import solar_system_track

    wide = AntennaModel(PRESET_ANTENNAS["horn-small"])
    solstice = zenith_track(DUKE_PHYSICS,
                            Observation(start_utc="2026-06-21T00:00:00",
                                        duration_hours=24.0, step_minutes=30.0),
                            wide.antenna)
    ss = solar_system_track(solstice.times, solstice.location, ("sun",))
    det = detectability_table(catalog := load_catalog(), wide,
                              solstice.dec_deg, ss)
    sun = det[det["name"] == "Sun"].iloc[0]
    assert not sun["enters_beam"], "the solstice Sun just misses a 25 deg beam"
    assert sun["contaminates_off_axis"]
    assert sun["t_a_at_closest_k"] > 1.0
    assert 0.0 < sun["beam_response_at_closest"] < 1.0


def test_moving_body_declination_is_the_closest_approach(track):
    """A body's quoted declination must be the one that meets the beam."""
    from hydroline.catalog import solar_system_as_rows, solar_system_track

    ss = solar_system_track(track.times, track.location, ("moon",))
    rows = solar_system_as_rows(ss, beam_dec_deg=track.dec_deg)
    moon = rows.iloc[0]
    decs = ss["moon"]["frame"]["dec_icrs_deg"].to_numpy()
    assert moon["dec_min_deg"] <= moon["dec_deg"] <= moon["dec_max_deg"]
    assert abs(moon["dec_deg"] - track.dec_deg) == pytest.approx(
        float(np.min(np.abs(decs - track.dec_deg))), abs=1e-9)


def test_sky_coverage_is_a_band(track, model):
    cov = sky_coverage(track, model)
    assert 0.0 < cov["fraction_of_sky"] < 0.2
    lo, hi = cov["declination_band_deg"]
    assert math.isclose(hi - lo, model.hpbw_deg, rel_tol=1e-6)
    assert cov["full_circle_covered"]


def test_bigger_dish_sees_fewer_sources(track, catalog):
    """A narrower beam is a smaller net: strictly fewer objects drift through."""
    wide = AntennaModel(PRESET_ANTENNAS["horn-small"])     # 25 deg
    narrow = AntennaModel(PRESET_ANTENNAS["dish-4m6"])     # 3.2 deg
    n_wide = detectability_table(catalog, wide, track.dec_deg)["enters_beam"].sum()
    n_narrow = detectability_table(catalog, narrow,
                                   track.dec_deg)["enters_beam"].sum()
    assert n_wide > n_narrow


def test_longer_integration_lowers_the_detection_limit():
    quick = AntennaModel(Antenna(integration_s=10.0))
    slow = AntennaModel(Antenna(integration_s=1000.0))
    assert slow.continuum_limit_jy < quick.continuum_limit_jy
    assert math.isclose(quick.continuum_limit_jy / slow.continuum_limit_jy,
                        10.0, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
def test_configuration_round_trip(tmp_path):
    cfg = Configuration(site=DUKE_PHYSICS,
                        antenna=PRESET_ANTENNAS["horn-large"],
                        observation=Observation(duration_hours=12.0))
    for suffix in (".json", ".yaml"):
        p = cfg.save(tmp_path / f"cfg{suffix}")
        back = Configuration.load(p)
        assert back.site.latitude_deg == cfg.site.latitude_deg
        assert back.antenna.aperture_e_m == cfg.antenna.aperture_e_m
        assert back.observation.duration_hours == 12.0


def test_yaml_style_exponents_are_coerced(tmp_path):
    """YAML 1.1 reads 2.5e6 as a string; the loader must not care."""
    p = tmp_path / "c.yaml"
    p.write_text("antenna:\n  rf_bandwidth_hz: 2.5e6\n  n_pol: '2'\n"
                 "site:\n  latitude_deg: '41.5'\n")
    cfg = Configuration.load(p)
    assert cfg.antenna.rf_bandwidth_hz == 2.5e6
    assert cfg.antenna.n_pol == 2
    assert cfg.site.latitude_deg == 41.5


def test_bad_configuration_is_rejected():
    with pytest.raises(ValueError):
        Site(latitude_deg=120.0)
    with pytest.raises(ValueError):
        Antenna(kind="horn")                       # no aperture given
    with pytest.raises(ValueError):
        Antenna(kind="beamwidth")                  # no beamwidth given
    with pytest.raises(ValueError):
        Antenna(aperture_efficiency=1.5)
    with pytest.raises(ValueError):
        Observation(step_minutes=0.0)


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
