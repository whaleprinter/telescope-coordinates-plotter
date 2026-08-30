"""What drifts through the beam, and what the antenna can do about it.

Three questions, three tables:

``beam_encounters``      when each object crosses the beam, and for how long
``detectability_table``  signal-to-noise and resolution for every object
``galactic_plane_crossings``  when the beam sweeps the Milky Way's HI disk
"""
from __future__ import annotations

import math
import warnings
from typing import Any, Iterable

import numpy as np
import pandas as pd

import astropy.units as u
from astropy.coordinates import SkyCoord, angular_separation
from astropy.time import Time
from astropy.utils.exceptions import AstropyWarning

from .antenna import SIDEREAL_RATE_DEG_PER_HOUR, AntennaModel
from .catalog import Catalog
from .constants import (
    HI_TB_HIGH_LAT_K,
    HI_TB_PLANE_K,
    HI_TB_SCALE_HEIGHT_DEG,
)
from .pointing import PointingTrack


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _separation_deg(ra1, dec1, ra2, dec2) -> np.ndarray:
    """Great-circle separation in degrees, broadcasting numpy arrays."""
    return np.degrees(
        angular_separation(
            np.radians(ra1), np.radians(dec1), np.radians(ra2), np.radians(dec2)
        )
    )


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs of a boolean array as inclusive (start, end) pairs."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    idx = np.flatnonzero(
        np.diff(np.concatenate(([False], mask, [False])).astype(np.int8))
    )
    return [(int(a), int(b - 1)) for a, b in zip(idx[::2], idx[1::2])]


def _refine_minimum(x: np.ndarray, y: np.ndarray, i: int) -> tuple[float, float]:
    """Parabolic vertex through three samples around index ``i``.

    Recovers the true transit time and minimum separation to far better than
    the sampling step, so a coarse time grid still gives a sharp transit time.
    """
    if i <= 0 or i >= len(x) - 1:
        return float(x[i]), float(y[i])
    y0, y1, y2 = float(y[i - 1]), float(y[i]), float(y[i + 1])
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-12:
        return float(x[i]), float(y[i])
    frac = 0.5 * (y0 - y2) / denom
    frac = max(-1.0, min(1.0, frac))
    step = float(x[i + 1] - x[i]) if frac >= 0 else float(x[i] - x[i - 1])
    return float(x[i] + frac * step), float(y1 - 0.25 * (y0 - y2) * frac)


def _crossing_time(x: np.ndarray, y: np.ndarray, i_in: int, i_out: int,
                   level: float) -> float:
    """Linear interpolation of the time at which y crosses ``level``."""
    y0, y1 = float(y[i_out]), float(y[i_in])
    if y0 == y1:
        return float(x[i_in])
    f = (level - y1) / (y0 - y1)
    return float(x[i_in] + f * (x[i_out] - x[i_in]))


def transit_duration_seconds(model: AntennaModel, beam_dec_deg: float,
                             source_dec_deg: float,
                             beam_fraction: float = 0.5) -> float:
    """Seconds a source spends inside the beam contour, per sidereal day.

    Solves the spherical triangle exactly rather than using the small-angle
    HPBW/(15 cos d) shortcut, because a small-antenna beam can be tens of
    degrees wide, where the shortcut is badly wrong.
    """
    r = math.radians(beam_fraction * model.hpbw_deg)
    db, ds = math.radians(beam_dec_deg), math.radians(source_dec_deg)
    denom = math.cos(db) * math.cos(ds)
    if abs(denom) < 1e-12:
        return 0.0
    arg = (math.cos(r) - math.sin(db) * math.sin(ds)) / denom
    if arg >= 1.0:
        return 0.0                      # never reaches the beam
    if arg <= -1.0:
        return 24.0 * 3600.0            # circumpolar inside the beam
    delta_ra_deg = math.degrees(math.acos(arg))
    return 2.0 * delta_ra_deg / SIDEREAL_RATE_DEG_PER_HOUR * 3600.0


# ---------------------------------------------------------------------------
# 1. encounters
# ---------------------------------------------------------------------------
def beam_encounters(track: PointingTrack,
                    catalog: Catalog,
                    model: AntennaModel,
                    solar_tracks: dict[str, dict] | None = None,
                    beam_fraction: float = 0.5,
                    include_misses: bool = False) -> pd.DataFrame:
    """Every pass of a catalogued object through the beam, with times.

    Parameters
    ----------
    beam_fraction
        Multiplier on the HPBW defining the beam edge.  0.5 (default) means the
        half-power contour: the source is inside the -3 dB circle.  Use 1.0 to
        include the shoulders down to roughly the first null.

    Returns
    -------
    DataFrame
        One row per pass, sorted by transit time.  Columns include the entry,
        transit and exit times (UTC and local), duration, minimum separation
        and the beam response at closest approach.
    """
    radius = beam_fraction * model.hpbw_deg
    hours = track.table["hours_from_start"].to_numpy()
    beam_ra = track.table["ra_icrs_deg"].to_numpy()
    beam_dec = track.table["dec_icrs_deg"].to_numpy()
    t0 = track.times[0]

    entries: list[dict[str, Any]] = []

    def _scan(name: str, kind: str, ra, dec, extra: dict[str, Any]) -> None:
        sep = _separation_deg(ra, dec, beam_ra, beam_dec)
        inside = sep <= radius
        runs = _runs(inside)
        if not runs:
            if include_misses:
                i = int(np.argmin(sep))
                t_min, s_min = _refine_minimum(hours, sep, i)
                entries.append({
                    "name": name, "kind": kind, "in_beam": False,
                    "min_separation_deg": s_min,
                    "transit_hours_from_start": t_min,
                    "transit_utc": (t0 + t_min * u.hour).isot,
                    "duration_minutes": 0.0, "beam_response": 0.0,
                    "truncated": False, **extra,
                })
            return
        for a, b in runs:
            i = a + int(np.argmin(sep[a:b + 1]))
            t_min, s_min = _refine_minimum(hours, sep, i)
            t_in = (_crossing_time(hours, sep, a, a - 1, radius)
                    if a > 0 else float(hours[a]))
            t_out = (_crossing_time(hours, sep, b, b + 1, radius)
                     if b < len(hours) - 1 else float(hours[b]))
            truncated = (a == 0) or (b == len(hours) - 1)
            entries.append({
                "name": name, "kind": kind, "in_beam": True,
                "min_separation_deg": max(s_min, 0.0),
                "transit_hours_from_start": t_min,
                "transit_utc": (t0 + t_min * u.hour).isot,
                "enter_utc": (t0 + t_in * u.hour).isot,
                "exit_utc": (t0 + t_out * u.hour).isot,
                "duration_minutes": (t_out - t_in) * 60.0,
                "beam_response": float(model.beam_profile(max(s_min, 0.0))),
                "truncated": bool(truncated),
                **extra,
            })

    # -- fixed sources ---------------------------------------------------
    cf = catalog.frame
    for _, row in cf.iterrows():
        _scan(
            row["name"], row["kind"],
            row["ra_deg"], row["dec_deg"],
            {
                "ra_deg": row["ra_deg"], "dec_deg": row["dec_deg"],
                "size_deg": row.get("size_deg", np.nan),
                "s1400_jy": row.get("s1400_jy", np.nan),
                "hi_emitter": bool(row.get("hi_emitter", False)),
                "moving": False,
            },
        )

    # -- moving bodies ---------------------------------------------------
    for key, info in (solar_tracks or {}).items():
        c = info["coords"]
        _scan(
            info["name"], "solar_system", c.ra.deg, c.dec.deg,
            {
                "ra_deg": float(np.mean(c.ra.deg)),
                "dec_deg": float(np.mean(c.dec.deg)),
                "size_deg": info["mean_size_deg"],
                "s1400_jy": info["mean_flux_jy"],
                "hi_emitter": False,
                "moving": True,
            },
        )

    if not entries:
        return pd.DataFrame(columns=[
            "name", "kind", "in_beam", "transit_utc", "duration_minutes",
            "min_separation_deg", "beam_response",
        ])

    out = pd.DataFrame(entries).sort_values(
        "transit_hours_from_start"
    ).reset_index(drop=True)

    # Local wall-clock transit, and the sidereal time to point at.
    out = _add_local_and_lst(out, track)

    cols = [
        "name", "kind", "in_beam", "transit_utc", "transit_local",
        "transit_lst_hours", "enter_utc", "exit_utc", "duration_minutes",
        "min_separation_deg", "beam_response", "ra_deg", "dec_deg",
        "size_deg", "s1400_jy", "hi_emitter", "moving", "truncated",
        "transit_hours_from_start",
    ]
    return out[[c for c in cols if c in out.columns]]


def _add_local_and_lst(out: pd.DataFrame, track: PointingTrack,
                       hours_col: str = "transit_hours_from_start",
                       utc_col: str = "transit_utc",
                       prefix: str = "transit") -> pd.DataFrame:
    """Attach local wall-clock time and sidereal time to a table of events."""
    hours = track.table["hours_from_start"].to_numpy()
    lst = np.unwrap(track.table["lst_apparent_hours"].to_numpy(), period=24.0)
    out[f"{prefix}_lst_hours"] = np.interp(out[hours_col], hours, lst) % 24.0
    try:
        from zoneinfo import ZoneInfo

        tz = ZoneInfo(track.site.timezone)
        out[f"{prefix}_local"] = (
            pd.to_datetime(out[utc_col], utc=True, format="mixed")
            .dt.tz_convert(tz).dt.strftime("%Y-%m-%d %H:%M %Z")
        )
    except Exception:                                       # pragma: no cover
        out[f"{prefix}_local"] = out[utc_col]
    return out


# ---------------------------------------------------------------------------
# 2. detectability and resolution
# ---------------------------------------------------------------------------
def detectability_table(catalog: Catalog,
                        model: AntennaModel,
                        beam_dec_deg: float,
                        solar_tracks: dict[str, dict] | None = None,
                        beam_fraction: float = 0.5) -> pd.DataFrame:
    """Per-source sensitivity and angular-resolution assessment.

    "Resolvable" is reported two ways, because the word means two things:

    ``resolution_class``
        Strictly angular: is the source bigger than the beam, so that scanning
        across it would show structure?  For a metre-class antenna at 21 cm the
        beam is many degrees, so almost nothing is resolved.
    ``detectable_continuum`` / ``detectable_line``
        Whether there is enough signal to see the source at all, which is
        usually what people mean.

    Integration times are quoted three ways: one transit (what a drift scan
    gives you for free each day), the configured integration, and the time
    actually needed to reach the SNR threshold.
    """
    rows: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = [catalog.frame]
    if solar_tracks:
        from .catalog import solar_system_as_rows

        frames.append(solar_system_as_rows(solar_tracks, beam_dec_deg))
    allsrc = pd.concat(frames, ignore_index=True, sort=False)

    beam_r = beam_fraction * model.hpbw_deg
    a = model.antenna

    for _, s in allsrc.iterrows():
        size = float(s.get("size_deg") or 0.0)
        dec = float(s["dec_deg"])
        dec_off = dec - beam_dec_deg
        enters = abs(dec_off) <= beam_r
        transit_s = transit_duration_seconds(model, beam_dec_deg, dec,
                                             beam_fraction)

        coupling = float(model.flux_coupling(size))
        beams_across = size / model.hpbw_deg if model.hpbw_deg else np.nan

        # --- continuum ---
        flux = float(s.get("s1400_jy") or np.nan)
        row: dict[str, Any] = {
            "name": s["name"],
            "kind": s["kind"],
            "ra_deg": s["ra_deg"],
            "dec_deg": dec,
            "dec_offset_from_beam_deg": dec_off,
            "enters_beam": bool(enters),
            "hi_emitter": bool(s.get("hi_emitter", False)),
            "transit_duration_s": transit_s,
            "transit_duration_min": transit_s / 60.0,
            "size_deg": size,
            "size_arcmin": size * 60.0,
            "beams_across_source": beams_across,
            "resolution_class": _resolution_class(beams_across),
            "flux_coupling": coupling,
            "s1400_total_jy": flux,
        }

        if np.isfinite(flux) and flux > 0:
            s_beam = flux * coupling
            # Response of the beam at closest approach.  For a drift scan the
            # right ascension sweeps through every value, so the minimum
            # separation a source ever reaches is exactly its declination
            # offset from the beam centre.
            resp = float(model.beam_profile(abs(dec_off)))
            row["beam_response_at_closest"] = resp
            row["t_a_at_closest_k"] = (s_beam * resp
                                       * model.sensitivity_k_per_jy)
            row["s1400_in_beam_jy"] = s_beam
            row["t_a_continuum_k"] = model.antenna_temperature_from_flux_k(
                flux, size
            )
            row["snr_continuum_single_transit"] = (
                s_beam / model.delta_s_rms_jy(integration_s=transit_s)
                if transit_s > 0 else 0.0
            )
            row["snr_continuum_configured"] = s_beam / model.delta_s_rms_jy()
            row["integration_for_threshold_s"] = _required_integration(
                model, s_beam, a.rf_bandwidth_hz
            )
            row["detectable_continuum"] = bool(
                enters and row["snr_continuum_single_transit"]
                >= a.snr_threshold
            )
        else:
            row.update({
                "beam_response_at_closest": np.nan,
                "t_a_at_closest_k": np.nan,
                "s1400_in_beam_jy": np.nan, "t_a_continuum_k": np.nan,
                "snr_continuum_single_transit": np.nan,
                "snr_continuum_configured": np.nan,
                "integration_for_threshold_s": np.nan,
                "detectable_continuum": False,
            })

        # --- 21 cm line ---
        peak = float(s.get("hi_peak_jy") or np.nan)
        w50 = float(s.get("w50_kms") or np.nan)
        if bool(s.get("hi_emitter", False)) and np.isfinite(peak) and peak > 0:
            line_beam = peak * coupling
            matched_bw = (model.matched_bandwidth_hz(w50)
                          if np.isfinite(w50) and w50 > 0 else a.channel_hz)
            row["hi_flux_jykms"] = s.get("hi_flux_jykms", np.nan)
            row["hi_peak_jy"] = peak
            row["hi_peak_in_beam_jy"] = line_beam
            row["hi_w50_kms"] = w50
            row["hi_matched_bandwidth_khz"] = matched_bw / 1e3
            row["t_a_line_k"] = line_beam * model.sensitivity_k_per_jy
            row["snr_line_per_channel"] = line_beam / model.delta_s_rms_jy(
                bandwidth_hz=a.channel_hz
            )
            row["snr_line_matched_transit"] = (
                line_beam / model.delta_s_rms_jy(bandwidth_hz=matched_bw,
                                                 integration_s=transit_s)
                if transit_s > 0 else 0.0
            )
            row["snr_line_matched_configured"] = (
                line_beam / model.delta_s_rms_jy(bandwidth_hz=matched_bw)
            )
            row["line_integration_for_threshold_s"] = _required_integration(
                model, line_beam, matched_bw
            )
            row["detectable_line"] = bool(
                enters and row["snr_line_matched_transit"] >= a.snr_threshold
            )
            row["nights_to_detect_line"] = (
                row["line_integration_for_threshold_s"] / transit_s
                if transit_s > 0 else np.inf
            )
        else:
            row.update({
                "hi_flux_jykms": np.nan, "hi_peak_jy": np.nan,
                "hi_peak_in_beam_jy": np.nan, "hi_w50_kms": np.nan,
                "hi_matched_bandwidth_khz": np.nan, "t_a_line_k": np.nan,
                "snr_line_per_channel": np.nan,
                "snr_line_matched_transit": np.nan,
                "snr_line_matched_configured": np.nan,
                "line_integration_for_threshold_s": np.nan,
                "detectable_line": False, "nights_to_detect_line": np.nan,
            })

        row["vsys_kms"] = s.get("vsys_kms", np.nan)
        row["notes"] = s.get("notes", "")
        rows.append(row)

    out = pd.DataFrame(rows)
    out["any_detection"] = out["detectable_continuum"] | out["detectable_line"]

    # Off-axis contamination.  A source can sit well outside the half-power
    # circle and still dominate the total power: the Sun is 5e5 Jy, so even a
    # few per cent of beam response is an enormous signal.  The Gaussian used
    # here has no far sidelobes, so these numbers are a LOWER bound -- a real
    # feed leaks at the -20 dB level in directions this model calls zero.
    noise = model.delta_t_rms_k()
    out["contaminates_off_axis"] = (
        (~out["enters_beam"])
        & out["t_a_at_closest_k"].notna()
        & (out["t_a_at_closest_k"] > model.antenna.snr_threshold * noise)
    )
    return out.sort_values(
        ["enters_beam", "any_detection", "snr_continuum_single_transit"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _resolution_class(beams_across: float) -> str:
    """Angular resolution verdict for a source of the given size in beams."""
    if not np.isfinite(beams_across):
        return "unknown"
    if beams_across >= 2.0:
        return "resolved"
    if beams_across >= 1.0:
        return "marginally resolved"
    if beams_across >= 0.5:
        return "beam-scale"
    return "unresolved"


def _required_integration(model: AntennaModel, flux_jy: float,
                          bandwidth_hz: float) -> float:
    """Integration time to reach the SNR threshold on ``flux_jy``.

    Inverting the radiometer equation: tau = (K_s * threshold * SEFD
    / (S * sqrt(n_pol * B)))^2.
    """
    a = model.antenna
    if flux_jy <= 0 or bandwidth_hz <= 0:
        return float("inf")
    num = a.radiometer_factor * a.snr_threshold * model.sefd_jy
    return (num / (flux_jy * math.sqrt(a.n_pol * bandwidth_hz))) ** 2


# ---------------------------------------------------------------------------
# 3. the Milky Way
# ---------------------------------------------------------------------------
def galactic_hi_estimate(track: PointingTrack,
                         model: AntennaModel) -> pd.DataFrame:
    """Coarse expected Galactic HI signal along the beam track.

    The brightness model is a crude exponential in galactic latitude anchored to
    typical LAB/EBHIS peak values (~100 K in the plane, ~12 K at high latitude).
    It is here to set expectations -- "you will see tens of kelvin, always" --
    not to predict a spectrum.  For real predicted profiles, query the LAB
    survey.  The Milky Way fills the beam completely at any latitude, so the
    only beam correction is the main-beam efficiency.
    """
    b = np.abs(track.table["gal_b_deg"].to_numpy())
    tb = HI_TB_HIGH_LAT_K + (HI_TB_PLANE_K - HI_TB_HIGH_LAT_K) * np.exp(
        -b / HI_TB_SCALE_HEIGHT_DEG
    )
    t_a = tb * model.main_beam_efficiency          # fills the beam
    sigma = model.delta_t_rms_k(bandwidth_hz=model.antenna.channel_hz)
    return pd.DataFrame({
        "time_utc": track.table["time_utc"],
        "hours_from_start": track.table["hours_from_start"],
        "gal_l_deg": track.table["gal_l_deg"],
        "gal_b_deg": track.table["gal_b_deg"],
        "estimated_peak_tb_k": tb,
        "estimated_peak_t_a_k": t_a,
        "estimated_peak_snr_per_channel": t_a / sigma if sigma > 0 else np.inf,
        "v_lsr_correction_kms": track.table["v_lsr_correction_kms"],
    })


def galactic_plane_crossings(track: PointingTrack,
                             threshold_deg: float = 10.0) -> pd.DataFrame:
    """Windows when the beam centre is within ``threshold_deg`` of b = 0.

    These are the best times to observe: the HI column density peaks in the
    plane, the line splits into multiple spiral-arm components, and the rotation
    curve is measurable.
    """
    hours = track.table["hours_from_start"].to_numpy()
    b = track.table["gal_b_deg"].to_numpy()
    absb = np.abs(b)
    t0 = track.times[0]

    out = []
    for a, z in _runs(absb <= threshold_deg):
        i = a + int(np.argmin(absb[a:z + 1]))
        t_min, b_min = _refine_minimum(hours, absb, i)
        t_in = (_crossing_time(hours, absb, a, a - 1, threshold_deg)
                if a > 0 else float(hours[a]))
        t_out = (_crossing_time(hours, absb, z, z + 1, threshold_deg)
                 if z < len(hours) - 1 else float(hours[z]))
        out.append({
            "enter_utc": (t0 + t_in * u.hour).isot,
            "closest_utc": (t0 + t_min * u.hour).isot,
            "exit_utc": (t0 + t_out * u.hour).isot,
            "duration_hours": t_out - t_in,
            "min_abs_galactic_latitude_deg": b_min,
            "galactic_longitude_at_closest_deg": float(
                np.interp(t_min, hours,
                          np.unwrap(track.table["gal_l_deg"].to_numpy(),
                                    period=360.0)) % 360.0
            ),
            "hours_from_start": t_min,
            "truncated": bool(a == 0 or z == len(hours) - 1),
        })
    df = pd.DataFrame(out)
    if not df.empty:
        df = _add_local_and_lst(df, track, hours_col="hours_from_start",
                                utc_col="closest_utc", prefix="closest")
        cols = ["enter_utc", "closest_utc", "closest_local", "closest_lst_hours",
                "exit_utc", "duration_hours", "min_abs_galactic_latitude_deg",
                "galactic_longitude_at_closest_deg", "hours_from_start",
                "truncated"]
        df = df[[c for c in cols if c in df.columns]]
    return df


def sky_coverage(track: PointingTrack, model: AntennaModel) -> dict[str, Any]:
    """Fraction of the celestial sphere the drift scan sweeps."""
    dec = track.dec_deg
    r = math.radians(model.beam_radius_deg)
    d = math.radians(dec)
    # Area of the band swept by a circle of radius r along a parallel of dec:
    # a spherical band from dec-r to dec+r (plus small caps, ignored).
    hi = min(math.pi / 2, d + r)
    lo = max(-math.pi / 2, d - r)
    band = 2.0 * math.pi * (math.sin(hi) - math.sin(lo))
    full = 4.0 * math.pi
    ra_span = float(
        np.ptp(np.unwrap(track.table["ra_icrs_deg"].to_numpy(), period=360.0))
    )
    return {
        "beam_declination_deg": dec,
        "declination_band_deg": [math.degrees(lo), math.degrees(hi)],
        "band_solid_angle_sr": band,
        "band_solid_angle_deg2": band * (180.0 / math.pi) ** 2,
        "fraction_of_sky": band / full,
        "ra_swept_deg": min(ra_span, 360.0),
        "full_circle_covered": ra_span >= 360.0,
        "sidereal_day_hours": 23.9344696,
    }
