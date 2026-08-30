"""The source list: fixed catalogue entries plus moving solar-system bodies.

Everything is stored in ICRS (J2000), which is what modern catalogues use and
what :mod:`hydroline.pointing` produces, so no frame juggling is needed to
compare a beam position with a source position.

Flux densities are for 1.4 GHz and are approximate -- good to tens of percent
for the calibrators, worse for the extended Galactic sources whose "total flux"
depends on how much sky you integrate over.  They are here to answer "can this
antenna see it at all", not to calibrate anything.  Check NED, SIMBAD or the
NVSS before quoting a number.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_body, solar_system_ephemeris
from astropy.time import Time
from astropy.utils.exceptions import AstropyWarning

from .constants import HI_MASS_COEFF, HI_REST_M, JY, K_BOLTZMANN

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_CATALOG = DATA_DIR / "sources_1420.csv"


# ---------------------------------------------------------------------------
# Solar-system bodies
# ---------------------------------------------------------------------------
#: Disk-equivalent brightness temperature near 1.4 GHz, kelvin.
#: Jupiter's value is inflated well above its ~150 K thermal disk because the
#: synchrotron radiation belts dominate at metre-to-decimetre wavelengths; the
#: quoted number reproduces the observed ~5 Jy at 4.04 AU.
#: The Sun's quiet-level T_b can rise by an order of magnitude during a flare.
SOLAR_SYSTEM_TB_K = {
    "sun": 1.0e5,
    "moon": 220.0,
    "mercury": 500.0,
    "venus": 600.0,
    "mars": 200.0,
    "jupiter": 2500.0,
    "saturn": 140.0,
    "uranus": 200.0,
    "neptune": 200.0,
}

#: Equatorial radii, km.
SOLAR_SYSTEM_RADIUS_KM = {
    "sun": 695_700.0,
    "moon": 1_737.4,
    "mercury": 2_439.7,
    "venus": 6_051.8,
    "mars": 3_389.5,
    "jupiter": 69_911.0,
    "saturn": 58_232.0,
    "uranus": 25_362.0,
    "neptune": 24_622.0,
}

DEFAULT_BODIES = ("sun", "moon", "jupiter", "venus", "mars", "saturn")


def flux_from_brightness_temperature(tb_k: float, diameter_deg: float,
                                     wavelength_m: float = HI_REST_M) -> float:
    """Rayleigh-Jeans flux density of a uniform disk, in janskys.

    S = 2 k T_b Omega / lambda^2, with Omega = pi theta^2 / 4 for a disk of
    angular diameter theta.
    """
    theta = math.radians(diameter_deg)
    omega = math.pi * theta ** 2 / 4.0
    return 2.0 * K_BOLTZMANN * tb_k * omega / wavelength_m ** 2 / JY


def brightness_temperature_from_flux(flux_jy: float, diameter_deg: float,
                                     wavelength_m: float = HI_REST_M) -> float:
    """Inverse of :func:`flux_from_brightness_temperature`."""
    theta = math.radians(diameter_deg)
    omega = math.pi * theta ** 2 / 4.0
    if omega <= 0:
        return float("nan")
    return flux_jy * JY * wavelength_m ** 2 / (2.0 * K_BOLTZMANN * omega)


# ---------------------------------------------------------------------------
# Catalogue container
# ---------------------------------------------------------------------------
@dataclass
class Catalog:
    """Fixed sources plus their ICRS coordinates."""

    frame: pd.DataFrame
    coords: SkyCoord
    source_path: Path | None = None

    def __len__(self) -> int:
        return len(self.frame)

    def filter_declination(self, dec_deg: float, half_width_deg: float) -> "Catalog":
        """Keep only sources whose declination could ever enter the beam."""
        keep = np.abs(self.frame["dec_deg"].to_numpy() - dec_deg) <= half_width_deg
        return Catalog(self.frame[keep].reset_index(drop=True),
                       self.coords[keep], self.source_path)

    def by_name(self, name: str) -> pd.Series:
        hit = self.frame[self.frame["name"].str.lower() == name.lower()]
        if hit.empty:
            hit = self.frame[
                self.frame["aliases"].fillna("").str.lower().str.contains(name.lower())
            ]
        if hit.empty:
            raise KeyError(f"no catalogue source matching {name!r}")
        return hit.iloc[0]


def load_catalog(path: str | Path | None = None,
                 wavelength_m: float = HI_REST_M) -> Catalog:
    """Read the source CSV and derive everything that can be derived.

    Added columns
    -------------
    ``ra_deg``, ``dec_deg``
        Parsed ICRS coordinates.
    ``size_deg``
        Circular-equivalent angular diameter, sqrt(major * minor).
    ``hi_flux_jykms``
        Velocity-integrated HI line flux from M_HI and distance:
        S_int = M_HI / (2.356e5 D_Mpc^2).
    ``hi_peak_jy``
        Profile-averaged line flux density, S_int / W50.  A real double-horned
        profile peaks ~1.3-1.5x above this, so treat it as conservative.
    ``tb_continuum_k``
        Continuum brightness temperature implied by the flux and the source
        size -- the number that matters for sources larger than the beam.
    """
    path = Path(path) if path else DEFAULT_CATALOG
    df = pd.read_csv(path)

    required = {"name", "kind", "ra", "dec"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"catalogue {path} is missing columns: {sorted(missing)}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        coords = SkyCoord(df["ra"].astype(str), df["dec"].astype(str),
                          unit=(u.hourangle, u.deg), frame="icrs")

    df = df.copy()
    df["ra_deg"] = coords.ra.deg
    df["dec_deg"] = coords.dec.deg
    df["gal_l_deg"] = coords.galactic.l.deg
    df["gal_b_deg"] = coords.galactic.b.deg

    for col in ("maj_arcmin", "min_arcmin", "s1400_jy", "dist_mpc",
                "hi_mass_msun", "vsys_kms", "w50_kms"):
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "hi_emitter" not in df.columns:
        df["hi_emitter"] = 0
    df["hi_emitter"] = df["hi_emitter"].fillna(0).astype(int).astype(bool)
    df["aliases"] = df.get("aliases", pd.Series(dtype=object)).fillna("")
    df["notes"] = df.get("notes", pd.Series(dtype=object)).fillna("")

    # Circular-equivalent diameter in degrees.
    maj = df["maj_arcmin"].fillna(0.1)
    mnr = df["min_arcmin"].fillna(maj)
    df["size_deg"] = np.sqrt(maj * mnr) / 60.0

    # 21 cm line quantities.
    with np.errstate(divide="ignore", invalid="ignore"):
        df["hi_flux_jykms"] = df["hi_mass_msun"] / (
            HI_MASS_COEFF * df["dist_mpc"] ** 2
        )
        df["hi_peak_jy"] = df["hi_flux_jykms"] / df["w50_kms"]
        df["tb_continuum_k"] = [
            brightness_temperature_from_flux(s, d, wavelength_m)
            if np.isfinite(s) and s > 0 and d > 0 else np.nan
            for s, d in zip(df["s1400_jy"], df["size_deg"])
        ]

    df["is_solar_system"] = False
    df["moving"] = False

    order = [
        "name", "aliases", "kind", "ra", "dec", "ra_deg", "dec_deg",
        "gal_l_deg", "gal_b_deg", "maj_arcmin", "min_arcmin", "size_deg",
        "s1400_jy", "tb_continuum_k", "hi_emitter", "dist_mpc",
        "hi_mass_msun", "vsys_kms", "w50_kms", "hi_flux_jykms", "hi_peak_jy",
        "is_solar_system", "moving", "notes",
    ]
    df = df[[c for c in order if c in df.columns]]
    return Catalog(frame=df.reset_index(drop=True), coords=coords, source_path=path)


# ---------------------------------------------------------------------------
# Moving bodies
# ---------------------------------------------------------------------------
def solar_system_track(times: Time,
                       location: EarthLocation,
                       bodies: tuple[str, ...] = DEFAULT_BODIES,
                       ephemeris: str = "builtin",
                       wavelength_m: float = HI_REST_M) -> dict[str, dict]:
    """Positions, apparent sizes and fluxes of solar-system bodies over time.

    Returns a dict keyed by body name, each value holding:

    ``coords``      SkyCoord (ICRS) sampled on ``times``
    ``altaz``       SkyCoord in the topocentric horizontal frame
    ``frame``       DataFrame with distance, angular diameter, flux
    ``mean_flux_jy``, ``mean_size_deg``, ``tb_k``

    Fluxes are Rayleigh-Jeans disk estimates from a fixed brightness
    temperature; they are order-of-magnitude for anything but the Moon.
    """
    out: dict[str, dict] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        with solar_system_ephemeris.set(ephemeris):
            altaz_frame = AltAz(obstime=times, location=location,
                                pressure=0 * u.hPa)
            for body in bodies:
                key = body.lower()
                if key not in SOLAR_SYSTEM_RADIUS_KM:
                    raise KeyError(f"unknown solar-system body: {body!r}")
                gcrs = get_body(key, times, location)

                # Direction only.  Transforming a distance-carrying geocentric
                # position to ICRS re-origins it at the solar-system
                # barycentre, which is meaningless for a body a light-second
                # away -- it puts the Sun on the far side of the sky.  Dropping
                # the distance makes the transform a pure rotation of axes
                # (plus removal of aberration), which is the direction a
                # catalogue comparison actually wants.
                direction = SkyCoord(ra=gcrs.ra, dec=gcrs.dec,
                                     frame=gcrs.frame.replicate_without_data())
                icrs = direction.transform_to("icrs")

                # AltAz keeps the distance: topocentric parallax is real and
                # matters for the Moon (up to a degree).
                altaz = gcrs.transform_to(altaz_frame)

                dist_km = gcrs.distance.to(u.km).value
                radius = SOLAR_SYSTEM_RADIUS_KM[key]
                diam_deg = np.degrees(2.0 * np.arcsin(
                    np.clip(radius / dist_km, -1.0, 1.0)
                ))
                tb = SOLAR_SYSTEM_TB_K[key]
                flux = np.array([
                    flux_from_brightness_temperature(tb, d, wavelength_m)
                    for d in diam_deg
                ])

                out[key] = {
                    "name": key.capitalize(),
                    "coords": icrs,
                    "altaz": altaz,
                    "tb_k": tb,
                    "mean_size_deg": float(np.mean(diam_deg)),
                    "mean_flux_jy": float(np.mean(flux)),
                    "frame": pd.DataFrame({
                        "time_utc": times.isot,
                        "ra_icrs_deg": icrs.ra.deg,
                        "dec_icrs_deg": icrs.dec.deg,
                        "ra_apparent_deg": gcrs.ra.deg,
                        "dec_apparent_deg": gcrs.dec.deg,
                        "altitude_deg": altaz.alt.deg,
                        "azimuth_deg": altaz.az.deg,
                        "distance_au": gcrs.distance.to(u.au).value,
                        "angular_diameter_deg": diam_deg,
                        "flux_1420_jy": flux,
                    }),
                }
    return out


def solar_system_as_rows(tracks: dict[str, dict],
                         beam_dec_deg: float | None = None) -> pd.DataFrame:
    """Flatten :func:`solar_system_track` into catalogue-shaped rows.

    Parameters
    ----------
    beam_dec_deg
        When given, each body's reported declination is the one it reaches
        *closest to the beam* during the window rather than its average.  This
        matters for the Moon, whose declination swings by several degrees a day
        and by 28 degrees a month: an averaged declination would report a miss
        for a body that in fact drives straight through the beam, disagreeing
        with the time-resolved transit search.
    """
    rows = []
    for key, info in tracks.items():
        f = info["frame"]
        decs = f["dec_icrs_deg"].to_numpy()
        if beam_dec_deg is None:
            i = len(decs) // 2
            dec = float(np.mean(decs))
        else:
            i = int(np.argmin(np.abs(decs - beam_dec_deg)))
            dec = float(decs[i])
        rows.append({
            "name": info["name"],
            "aliases": "",
            "kind": "solar_system",
            "ra_deg": float(f["ra_icrs_deg"].to_numpy()[i]),
            "dec_deg": dec,
            "dec_min_deg": float(decs.min()),
            "dec_max_deg": float(decs.max()),
            "size_deg": info["mean_size_deg"],
            "s1400_jy": info["mean_flux_jy"],
            "tb_continuum_k": info["tb_k"],
            "hi_emitter": False,
            "is_solar_system": True,
            "moving": True,
            "notes": (f"moves {float(np.ptp(f['ra_icrs_deg'])):.2f} deg in RA "
                      f"and {float(np.ptp(decs)):.2f} deg in Dec over the "
                      f"window; declination quoted is the closest approach to "
                      f"the beam; flux from a {info['tb_k']:.0f} K disk"),
        })
    return pd.DataFrame(rows)
