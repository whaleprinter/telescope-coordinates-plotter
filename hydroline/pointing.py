"""Where a fixed antenna points on the celestial sphere, as a function of time.

The physics is one sentence long: the antenna is bolted to the ground, the
ground rotates, so the beam sweeps a circle of constant declination once per
sidereal day.  For an antenna aimed straight up,

    Dec(beam)  =  geodetic latitude of the site          (constant)
    RA(beam)   =  local apparent sidereal time           (increases 15.04 deg/hr)

That identity is exact in the *apparent* (true equator and equinox of date)
frame.  Catalogues are in ICRS/J2000, so the numbers you compare against a
catalogue differ from the identity above by precession since J2000 (~0.36 deg
of RA by 2026), nutation (~17 arcsec) and annual aberration (~20 arcsec).

This module does the conversion properly with astropy/ERFA rather than by hand,
and reports both frames plus the residuals, so the approximation is visible
instead of hidden.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone as _tz
from typing import Any

import numpy as np
import pandas as pd

import astropy.units as u
from astropy.coordinates import (
    AltAz,
    EarthLocation,
    FK4,
    Galactic,
    SkyCoord,
    TETE,
    get_body,
    solar_system_ephemeris,
)
from astropy.time import Time
from astropy.utils import iers
from astropy.utils.exceptions import AstropyWarning

from .config import Antenna, Observation, Site
from .constants import (
    LSR_APEX_DEC_B1900_DEG,
    LSR_APEX_RA_B1900_HOURS,
    LSR_SOLAR_SPEED_KMS,
)


# ---------------------------------------------------------------------------
# Earth-orientation data
# ---------------------------------------------------------------------------
def configure_iers(offline: bool = False) -> None:
    """Make Earth-orientation handling robust.

    UT1-UTC and polar motion shift the pointing by well under an arcsecond,
    which is nothing next to a beam of degrees.  So we never let a missing or
    stale IERS table stop the run.
    """
    iers.conf.auto_max_age = None          # tolerate an out-of-date table
    if offline:
        iers.conf.auto_download = False
        iers.conf.iers_degraded_accuracy = "ignore"


# ---------------------------------------------------------------------------
# Local standard of rest
# ---------------------------------------------------------------------------
def lsr_apex() -> SkyCoord:
    """Direction of the standard solar motion, converted to ICRS.

    Defined as 20.0 km/s toward RA 18h00m, Dec +30 deg in B1900 coordinates --
    the kinematic LSR that radio astronomers have used since the 1950s and the
    convention behind every published V_LSR for an HI line.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        apex_b1900 = SkyCoord(
            ra=LSR_APEX_RA_B1900_HOURS * u.hourangle,
            dec=LSR_APEX_DEC_B1900_DEG * u.deg,
            frame=FK4(equinox="B1900", obstime="B1900"),
        )
        return apex_b1900.transform_to("icrs")


def lsr_projection_kms(coord: SkyCoord) -> np.ndarray:
    """Component of the Sun's LSR motion along the line of sight, km/s.

    Positive where the Sun moves *toward* the target, which is the sign that
    must be **added** to a barycentric radial velocity to obtain V_LSR.
    """
    sep = coord.separation(lsr_apex())
    return LSR_SOLAR_SPEED_KMS * np.cos(sep.radian)


# ---------------------------------------------------------------------------
# Time grid
# ---------------------------------------------------------------------------
def build_times(observation: Observation) -> Time:
    """Uniform UTC grid from the observation window."""
    if str(observation.start_utc).lower() == "auto":
        now = datetime.now(_tz.utc)
        start = Time(datetime(now.year, now.month, now.day, tzinfo=_tz.utc), scale="utc")
    else:
        start = Time(observation.start_utc, scale="utc")
    n = int(round(observation.duration_hours * 60.0 / observation.step_minutes)) + 1
    offsets = np.arange(n) * observation.step_minutes * u.minute
    return start + offsets


def earth_location(site: Site) -> EarthLocation:
    """WGS84 geodetic position of the array."""
    return EarthLocation.from_geodetic(
        lon=site.longitude_deg * u.deg,
        lat=site.latitude_deg * u.deg,
        height=site.elevation_m * u.m,
        ellipsoid="WGS84",
    )


# ---------------------------------------------------------------------------
# The track
# ---------------------------------------------------------------------------
@dataclass
class PointingTrack:
    """Beam pointing sampled on a time grid, in every frame we care about."""

    site: Site
    antenna: Antenna
    times: Time
    location: EarthLocation
    icrs: SkyCoord            # J2000 / ICRS -- use this against catalogues
    apparent: SkyCoord        # true equator & equinox of date
    galactic: SkyCoord
    table: pd.DataFrame

    # -- convenience ---------------------------------------------------------
    @property
    def dec_deg(self) -> float:
        """Mean ICRS declination of the beam centre (constant to ~arcseconds)."""
        return float(np.mean(self.icrs.dec.deg))

    @property
    def n_steps(self) -> int:
        return len(self.times)

    def local_times(self) -> pd.Series:
        """Wall-clock times at the site, for humans."""
        try:
            from zoneinfo import ZoneInfo

            tz = ZoneInfo(self.site.timezone)
        except Exception:                                   # pragma: no cover
            return pd.to_datetime(self.table["time_utc"])
        return (pd.to_datetime(self.table["time_utc"], utc=True)
                .dt.tz_convert(tz))

    def summary(self) -> dict[str, Any]:
        t = self.table
        return {
            "site": self.site.name,
            "latitude_deg": self.site.latitude_deg,
            "longitude_deg": self.site.longitude_deg,
            "elevation_m": self.site.elevation_m,
            "pointing_alt_deg": self.antenna.pointing_alt_deg,
            "pointing_az_deg": self.antenna.pointing_az_deg,
            "start_utc": str(self.times[0].isot),
            "end_utc": str(self.times[-1].isot),
            "n_steps": self.n_steps,
            "step_minutes": float(
                (self.times[1] - self.times[0]).to_value(u.minute)
            ) if self.n_steps > 1 else 0.0,
            "dec_icrs_mean_deg": float(t["dec_icrs_deg"].mean()),
            "dec_icrs_span_arcsec": float(
                (t["dec_icrs_deg"].max() - t["dec_icrs_deg"].min()) * 3600.0
            ),
            "ra_icrs_min_deg": float(t["ra_icrs_deg"].min()),
            "ra_icrs_max_deg": float(t["ra_icrs_deg"].max()),
            "dec_apparent_mean_deg": float(t["dec_apparent_deg"].mean()),
            "galactic_b_min_deg": float(t["gal_b_deg"].min()),
            "galactic_b_max_deg": float(t["gal_b_deg"].max()),
            "galactic_l_range_deg": [float(t["gal_l_deg"].min()),
                                     float(t["gal_l_deg"].max())],
            "v_lsr_correction_min_kms": float(t["v_lsr_correction_kms"].min()),
            "v_lsr_correction_max_kms": float(t["v_lsr_correction_kms"].max()),
            "dec_minus_latitude_arcmin_mean": float(
                t["dec_minus_latitude_arcmin"].mean()
            ),
        }


def zenith_track(
    site: Site,
    observation: Observation,
    antenna: Antenna | None = None,
    ephemeris: str = "builtin",
) -> PointingTrack:
    """Compute where the antenna looks, over the observation window.

    Parameters
    ----------
    site
        Array location.  Latitude is geodetic.
    observation
        Start time, duration and step.
    antenna
        Only ``pointing_alt_deg`` / ``pointing_az_deg`` are used; the default is
        straight up (alt = 90).
    ephemeris
        Solar-system ephemeris for the Sun/Moon columns.  ``"builtin"`` needs no
        downloads and is accurate to arcseconds -- far better than a beam of
        degrees requires.

    Returns
    -------
    PointingTrack
        With ``.table`` a tidy DataFrame, one row per time step.
    """
    antenna = antenna or Antenna()
    times = build_times(observation)
    loc = earth_location(site)

    # --- the pointing, in the topocentric horizontal frame -----------------
    # pressure = 0 switches refraction off.  At alt = 90 refraction is exactly
    # zero anyway; this only matters if the mount is tilted.
    altaz = AltAz(
        alt=np.full(len(times), antenna.pointing_alt_deg) * u.deg,
        az=np.full(len(times), antenna.pointing_az_deg) * u.deg,
        obstime=times,
        location=loc,
        pressure=site.pressure_hpa * u.hPa,
        temperature=site.temperature_c * u.deg_C,
        relative_humidity=site.relative_humidity,
        obswl=0.211 * u.m,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        # ICRS: the modern inertial reference frame, and what every modern
        # catalogue (Gaia, NVSS, 2MASS, SIMBAD) is expressed in.  Astropy runs
        # the full ERFA chain: Earth rotation -> polar motion -> nutation ->
        # precession -> frame bias, plus diurnal and annual aberration and
        # gravitational light deflection.
        icrs = SkyCoord(altaz).transform_to("icrs")

        # Apparent place: true equator and equinox of date.  This is the frame
        # in which "RA of the zenith == local apparent sidereal time" holds
        # exactly, so it is our correctness check.
        apparent = SkyCoord(altaz).transform_to(TETE(obstime=times, location=loc))

        galactic = icrs.transform_to(Galactic())

        lst_apparent = Time(times, location=loc).sidereal_time("apparent")
        lst_mean = Time(times, location=loc).sidereal_time("mean")

        # --- Doppler reference frames -------------------------------------
        # Correction to ADD to a topocentric radial velocity to put a spectrum
        # on the barycentric scale, then on the kinematic LSR scale.
        # A bare ICRS direction: the coord returned by the AltAz transform
        # carries obstime/location attributes, which radial_velocity_correction
        # refuses to combine with explicit arguments.
        bare = SkyCoord(ra=icrs.ra, dec=icrs.dec, frame="icrs")
        v_bary = bare.radial_velocity_correction(
            kind="barycentric", obstime=times, location=loc
        ).to(u.km / u.s).value
        v_apex = lsr_projection_kms(bare)

        # --- Sun and Moon --------------------------------------------------
        with solar_system_ephemeris.set(ephemeris):
            sun = get_body("sun", times, loc)
            moon = get_body("moon", times, loc)
        sun_sep = sun.separation(icrs).deg
        moon_sep = moon.separation(icrs).deg
        sun_altaz = sun.transform_to(
            AltAz(obstime=times, location=loc, pressure=0 * u.hPa)
        )

    # --- residuals: how far the exact answer is from the textbook one ------
    dec_minus_lat = (icrs.dec.deg - site.latitude_deg) * 60.0        # arcmin
    dec_app_minus_lat = (apparent.dec.deg - site.latitude_deg) * 60.0
    ra_app_minus_lst = _wrap180(apparent.ra.deg - lst_apparent.deg) * 60.0
    ra_icrs_minus_lst = _wrap180(icrs.ra.deg - lst_apparent.deg) * 60.0

    table = pd.DataFrame(
        {
            "time_utc": times.isot,
            "mjd": times.mjd,
            "jd": times.jd,
            "hours_from_start": (times - times[0]).to_value(u.hour),
            "lst_apparent_hours": lst_apparent.hour,
            "lst_mean_hours": lst_mean.hour,
            "ra_icrs_deg": icrs.ra.deg,
            "dec_icrs_deg": icrs.dec.deg,
            "ra_icrs_hours": icrs.ra.hour,
            "ra_apparent_deg": apparent.ra.deg,
            "dec_apparent_deg": apparent.dec.deg,
            "gal_l_deg": galactic.l.deg,
            "gal_b_deg": galactic.b.deg,
            "v_barycentric_correction_kms": v_bary,
            "v_lsr_apex_projection_kms": v_apex,
            "v_lsr_correction_kms": v_bary + v_apex,
            "sun_separation_deg": sun_sep,
            "moon_separation_deg": moon_sep,
            "sun_altitude_deg": sun_altaz.alt.deg,
            "is_night": sun_altaz.alt.deg < -6.0,
            "dec_minus_latitude_arcmin": dec_minus_lat,
            "dec_apparent_minus_latitude_arcmin": dec_app_minus_lat,
            "ra_apparent_minus_lst_arcmin": ra_app_minus_lst,
            "ra_icrs_minus_lst_arcmin": ra_icrs_minus_lst,
        }
    )

    # Human-friendly local time column, if the timezone resolves.
    try:
        from zoneinfo import ZoneInfo

        tz = ZoneInfo(site.timezone)
        table.insert(
            1,
            "time_local",
            pd.to_datetime(table["time_utc"], utc=True)
            .dt.tz_convert(tz)
            .dt.strftime("%Y-%m-%d %H:%M:%S %Z"),
        )
    except Exception:                                        # pragma: no cover
        pass

    # Sexagesimal columns, because that is what you type into a catalogue.
    table["ra_icrs_hms"] = icrs.ra.to_string(unit=u.hour, sep=":", precision=1,
                                             pad=True)
    table["dec_icrs_dms"] = icrs.dec.to_string(unit=u.deg, sep=":", precision=0,
                                               pad=True, alwayssign=True)

    return PointingTrack(
        site=site,
        antenna=antenna,
        times=times,
        location=loc,
        icrs=icrs,
        apparent=apparent,
        galactic=galactic,
        table=table,
    )


def _wrap180(deg: np.ndarray) -> np.ndarray:
    """Wrap an angle difference into (-180, 180]."""
    return (np.asarray(deg) + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------
def validate_track(track: PointingTrack, tolerance_arcmin: float = 1.0) -> dict:
    """Verify the computed track against the analytic zenith identities.

    For a zenith pointing the apparent declination must equal the geodetic
    latitude and the apparent RA must equal the local apparent sidereal time.
    Agreement to well under an arcminute means the frame chain is wired up
    correctly.
    """
    t = track.table
    zenith = abs(track.antenna.pointing_alt_deg - 90.0) < 1e-9
    result = {
        "is_zenith_pointing": zenith,
        "max_abs_dec_apparent_minus_latitude_arcmin": float(
            np.max(np.abs(t["dec_apparent_minus_latitude_arcmin"]))
        ),
        "max_abs_ra_apparent_minus_lst_arcmin": float(
            np.max(np.abs(t["ra_apparent_minus_lst_arcmin"]))
        ),
        "mean_dec_icrs_minus_latitude_arcmin": float(
            np.mean(t["dec_minus_latitude_arcmin"])
        ),
        "mean_ra_icrs_minus_lst_arcmin": float(
            np.mean(t["ra_icrs_minus_lst_arcmin"])
        ),
        "tolerance_arcmin": tolerance_arcmin,
    }
    if zenith:
        result["passed"] = bool(
            result["max_abs_dec_apparent_minus_latitude_arcmin"] < tolerance_arcmin
            and result["max_abs_ra_apparent_minus_lst_arcmin"] < tolerance_arcmin
        )
    else:
        result["passed"] = None      # identity does not apply off-zenith
    return result
