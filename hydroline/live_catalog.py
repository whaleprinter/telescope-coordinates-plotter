"""Build hydroline's source catalog from real astronomical databases.

Three services, queried over a declination band around the beam (never the
whole sky -- a fixed antenna only ever sees one declination):

- **NVSS** (VizieR ``VIII/65/nvss``, Condon+1998): the actual 1.4 GHz
  continuum survey hydroline's own observing frequency matches. Dense --
  verified empirically at ~14,000 sources per 12-degree-wide declination
  strip above 50 mJy -- so it is queried with a flux floor derived from the
  *specific antenna's* own sensitivity (:func:`antenna_flux_floor_jy`),
  not one hardcoded number that would be wrong for a 0.3 m horn and a 4.6 m
  dish alike.
- **ALFALFA** (VizieR ``J/ApJ/861/49``, Haynes+2018): an HI-selected galaxy
  catalog that gives exactly the columns hydroline's HI-emitter schema
  wants -- distance, HI mass, systemic velocity, line width -- verified
  against real sample rows. Its real limitation, also verified empirically
  (zero results at declination -5, 40 and 60 degrees; results at 0, 16, 36):
  it is an Arecibo survey, so it only has data for roughly declination 0 to
  +36 degrees. Outside that band this module correctly returns zero HI
  emitters, and says so in ``LiveCatalogResult.warnings`` -- it does not
  pretend to have full-sky coverage.
- **SIMBAD**: used only to enrich a position NVSS/ALFALFA already found
  with a common name and an object-type classification, never as a primary
  source of positions or flux.

This module is the only place in hydroline that imports ``astroquery`` --
everything else stays importable and fully offline without it, matching the
optional-matplotlib pattern already used for figures in ``cli.py``.

The integration seam with the rest of the package is deliberately thin:
every row this module produces, from any of the three services, is shaped
into the exact pre-derivation schema :func:`hydroline.catalog.load_catalog`
already reads from the static CSV (``name, aliases, kind, ra, dec,
maj_arcmin, min_arcmin, s1400_jy, hi_emitter, dist_mpc, hi_mass_msun,
vsys_kms, w50_kms, notes``). The merged result is written to a cache CSV and
handed to the real, unmodified ``load_catalog()`` -- so every derived
quantity downstream (``ra_deg``, ``size_deg``, ``hi_flux_jykms``, ...) comes
from the exact same code path as the static catalog always has, with zero
duplicated logic.
"""
from __future__ import annotations

import hashlib
import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .antenna import AntennaModel
from .catalog import DATA_DIR, DEFAULT_CATALOG, Catalog, load_catalog
from .constants import HI_MASS_COEFF

LIVE_CACHE_DIR = DATA_DIR / "live_cache"

#: Bump this whenever RAW_COLUMNS or a mapping function's output changes --
#: it invalidates every existing cache entry by changing the cache key.
SCHEMA_VERSION = 1

NVSS_CATALOG_ID = "VIII/65/nvss"
ALFALFA_CATALOG_ID = "J/ApJ/861/49"

#: The exact pre-derivation schema load_catalog() reads from a CSV.
RAW_COLUMNS = [
    "name", "aliases", "kind", "ra", "dec", "maj_arcmin", "min_arcmin",
    "s1400_jy", "hi_emitter", "dist_mpc", "hi_mass_msun", "vsys_kms",
    "w50_kms", "notes",
]

#: SIMBAD otype -> hydroline kind. Deliberately small and conservative:
#: anything not recognised keeps whatever kind NVSS/ALFALFA-only assigned it
#: (continuum_unclassified / hi_galaxy) rather than guessing.
_SIMBAD_KIND_MAP: dict[str, str] = {
    "AGN": "radio_galaxy", "Sy1": "radio_galaxy", "Sy2": "radio_galaxy",
    "RadioG": "radio_galaxy", "LINER": "radio_galaxy",
    "QSO": "quasar", "Blazar": "quasar", "BLLac": "quasar",
    "SNR": "snr", "SNR?": "snr",
    "HII": "hii_region", "HIIG": "hii_region",
    "PN": "planetary_nebula", "PN?": "planetary_nebula",
    "Psr": "pulsar",
    "G": "galaxy", "GiG": "galaxy", "GiP": "galaxy", "GiC": "galaxy",
    "*": "star", "**": "star", "PM*": "star", "V*": "star",
}

# Cross-match radii. NVSS positions are compact continuum cores; ALFALFA
# positions are single-dish (Arecibo, ~3.5 arcmin beam) HI centroids, hence
# the much looser radius for enrichment and for the NVSS<->ALFALFA merge.
_SIMBAD_MATCH_RADIUS_NVSS_ARCSEC = 15.0
_SIMBAD_MATCH_RADIUS_ALFALFA_ARCSEC = 45.0
_NVSS_ALFALFA_MATCH_RADIUS_ARCSEC = 60.0


class LiveCatalogError(RuntimeError):
    """Raised only when no usable catalog -- live or cached -- could be built."""


@dataclass
class LiveCatalogResult:
    catalog: Catalog
    cache_path: Path
    from_cache: bool
    counts: dict[str, int]
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# flux floor
# ---------------------------------------------------------------------------
def antenna_flux_floor_jy(model: AntennaModel, multiplier: float = 0.5,
                          reference_hours: float = 24.0,
                          absolute_floor_jy: float = 0.05) -> float:
    """Minimum NVSS 1.4 GHz flux worth fetching for this antenna.

    Deliberately *not* based on ``model.continuum_limit_jy`` -- that
    property uses the antenna's configured ``integration_s``, which
    defaults to a single 300 s pointing. Measured directly: at that basis
    every preset from a 1.2 m dish up gives a floor of several Jy to
    hundreds of Jy, which the empirically-measured NVSS density (12
    sources above 5 Jy, 224 above 1 Jy, in one 12-degree strip) shows would
    return almost nothing for most antennas. A drift-scan instrument
    doesn't observe for one 300 s pointing -- it accumulates sensitivity
    over many transits across a real campaign, so the floor is instead
    based on a much longer reference integration (default 24 h, i.e. roughly
    one day of accumulated transits), then further relaxed by `multiplier`
    to stay generously inclusive of near-threshold sources -- the real
    per-source SNR/coupling/multi-night math already happens correctly
    downstream in :func:`hydroline.analysis.detectability_table`; this
    function's only job is keeping the live-query row count sane, never to
    decide what counts as "detectable".
    """
    rms_at_reference = model.delta_s_rms_jy(integration_s=reference_hours * 3600.0)
    threshold_at_reference = model.antenna.snr_threshold * rms_at_reference
    return max(absolute_floor_jy, multiplier * threshold_at_reference)


def _implied_hi_peak_jy(dist_mpc: float, hi_mass_msun: float,
                        w50_kms: float) -> float:
    """Same formula catalog.py's load_catalog() uses to derive hi_peak_jy --
    duplicated here (not imported) because load_catalog derives it from a
    full DataFrame post-parse, and this needs it on raw fetch results
    before they ever reach load_catalog, to filter before merging/caching.
    """
    if not (_finite(dist_mpc) and _finite(hi_mass_msun) and _finite(w50_kms)
           and dist_mpc > 0 and w50_kms > 0):
        return float("nan")
    hi_flux_jykms = hi_mass_msun / (HI_MASS_COEFF * dist_mpc ** 2)
    return hi_flux_jykms / w50_kms


# ---------------------------------------------------------------------------
# row mapping -- the highest-risk code here, so every field is explicit
# ---------------------------------------------------------------------------
def _finite(x: Any) -> bool:
    try:
        return x is not None and np.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def nvss_row_to_hydroline(row: dict[str, Any]) -> dict[str, Any]:
    """One VizieR VIII/65/nvss row -> a raw hydroline catalogue row.

    Verified columns/units (real query, this session): ``NVSS`` (name
    string, e.g. "231940+585440"), ``RAJ2000``/``DEJ2000`` (sexagesimal
    strings), ``S1.4`` (mJy), ``MajAxis``/``MinAxis`` (arcsec, absent/masked
    for unresolved point sources).
    """
    maj, minr = row.get("MajAxis"), row.get("MinAxis")
    return {
        "name": f"NVSS J{row['NVSS']}",
        "aliases": "",
        "kind": "continuum_unclassified",
        "ra": str(row["RAJ2000"]),
        "dec": str(row["DEJ2000"]),
        "maj_arcmin": float(maj) / 60.0 if _finite(maj) else np.nan,
        "min_arcmin": float(minr) / 60.0 if _finite(minr) else np.nan,
        "s1400_jy": float(row["S1.4"]) / 1000.0,
        "hi_emitter": 0,
        "dist_mpc": np.nan,
        "hi_mass_msun": np.nan,
        "vsys_kms": np.nan,
        "w50_kms": np.nan,
        "notes": "NVSS continuum source (1.4 GHz), live-queried",
    }


def alfalfa_row_to_hydroline(row: dict[str, Any]) -> dict[str, Any]:
    """One VizieR J/ApJ/861/49 (Haynes+2018 ALFALFA) row -> a raw row.

    Verified columns/units (real query, this session): ``Name`` (often
    blank -- many ALFALFA entries have no separate name, only an AGC
    catalogue number), ``AGC`` (catalogue number), ``RAJ2000``/``DEJ2000``
    (sexagesimal strings), ``Vhel`` (km/s), ``W50`` (km/s), ``Dist`` (Mpc),
    ``logMHI`` (log10 solar masses).

    Feeds ``Dist``/``logMHI`` through -- not the catalogue's own tabulated
    ``HIflux`` -- so :func:`hydroline.catalog.load_catalog`'s existing
    ``hi_mass_msun / (2.356e5 * dist_mpc**2)`` derivation produces a
    self-consistent flux, rather than importing a second number that could
    quietly disagree with it.
    """
    agc = row.get("AGC")
    name = str(row["Name"]).strip() if row.get("Name") else ""
    return {
        "name": name or f"AGC {agc}",
        "aliases": f"AGC {agc}" if agc else "",
        "kind": "hi_galaxy",
        "ra": str(row["RAJ2000"]),
        "dec": str(row["DEJ2000"]),
        "maj_arcmin": np.nan,
        "min_arcmin": np.nan,
        "s1400_jy": np.nan,
        "hi_emitter": 1,
        "dist_mpc": float(row["Dist"]) if _finite(row.get("Dist")) else np.nan,
        "hi_mass_msun": (10.0 ** float(row["logMHI"])
                         if _finite(row.get("logMHI")) else np.nan),
        "vsys_kms": float(row["Vhel"]) if _finite(row.get("Vhel")) else np.nan,
        "w50_kms": float(row["W50"]) if _finite(row.get("W50")) else np.nan,
        "notes": "ALFALFA HI detection (Haynes+2018), live-queried",
    }


def kind_from_simbad_otype(otype: str) -> str | None:
    """SIMBAD object-type code -> hydroline kind, or None if unrecognised."""
    return _SIMBAD_KIND_MAP.get(str(otype).strip())


# ---------------------------------------------------------------------------
# merge NVSS + ALFALFA
# ---------------------------------------------------------------------------
def merge_nvss_alfalfa(nvss_rows: pd.DataFrame, alfalfa_rows: pd.DataFrame,
                       match_radius_arcsec: float = _NVSS_ALFALFA_MATCH_RADIUS_ARCSEC
                       ) -> pd.DataFrame:
    """Cross-match by position and combine same-object rows.

    A matched pair keeps ALFALFA's position/name/HI columns (the HI mass,
    velocity and line width are tied to that specific radio detection) and
    NVSS's flux/size columns, with ``kind="hi_galaxy"``. Unmatched rows from
    either side pass through unchanged. Returns one combined, schema-shaped
    DataFrame -- never raises on empty input.
    """
    if nvss_rows.empty:
        return alfalfa_rows.reset_index(drop=True)
    if alfalfa_rows.empty:
        return nvss_rows.reset_index(drop=True)

    import astropy.units as u
    from astropy.coordinates import SkyCoord

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nvss_c = SkyCoord(nvss_rows["ra"].astype(str), nvss_rows["dec"].astype(str),
                          unit=(u.hourangle, u.deg), frame="icrs")
        alfalfa_c = SkyCoord(alfalfa_rows["ra"].astype(str), alfalfa_rows["dec"].astype(str),
                             unit=(u.hourangle, u.deg), frame="icrs")
        idx, sep, _ = alfalfa_c.match_to_catalog_sky(nvss_c)

    matched_nvss_idx: set[int] = set()
    merged_rows: list[dict[str, Any]] = []
    for i, (nvss_i, sep_i) in enumerate(zip(idx, sep)):
        arow = alfalfa_rows.iloc[i].to_dict()
        if sep_i.arcsec <= match_radius_arcsec:
            nrow = nvss_rows.iloc[int(nvss_i)]
            arow["s1400_jy"] = nrow["s1400_jy"]
            arow["maj_arcmin"] = nrow["maj_arcmin"]
            arow["min_arcmin"] = nrow["min_arcmin"]
            arow["notes"] = arow["notes"] + "; NVSS cross-match " \
                f"{sep_i.arcsec:.0f}\" away"
            matched_nvss_idx.add(int(nvss_i))
        merged_rows.append(arow)

    nvss_only = nvss_rows.iloc[[i for i in range(len(nvss_rows))
                                if i not in matched_nvss_idx]]
    return pd.concat([pd.DataFrame(merged_rows), nvss_only], ignore_index=True)


# ---------------------------------------------------------------------------
# SIMBAD enrichment
# ---------------------------------------------------------------------------
def enrich_with_simbad(rows: pd.DataFrame, fetch_fn: Callable[..., Any],
                       max_rows: int = 40, sleep_s: float = 0.05,
                       progress: Callable[[int, int], None] | None = None,
                       ) -> tuple[pd.DataFrame, list[str]]:
    """Attach a common name (into ``aliases``) and a ``kind`` from SIMBAD.

    Queries one position at a time -- SIMBAD's batch query_region does not
    cleanly report which input position each result belongs to, and getting
    that attribution wrong would silently mislabel sources, which is worse
    than the extra network round-trips. Measured directly: real per-query
    latency to the actual service is several seconds, not the sub-second
    figure a single warmed-up request suggests, so `max_rows` defaults low
    (40) to keep this to well under a minute rather than let a large
    catalogue turn a live run into a multi-minute wait; raise it
    deliberately (`--no-simbad`'s opposite -- there's no flag to raise it
    yet, that's a natural follow-up) if you want deeper enrichment.

    Candidates are ranked before truncating -- flux for continuum sources,
    HI mass for HI emitters -- so a capped run enriches the sources someone
    actually looking at the report would care about first, not whatever
    order the NVSS/ALFALFA/merge step happened to leave them in.

    A radius that matches zero or more than one SIMBAD object is left
    alone -- ambiguous crowded-field matches are expected, not something
    this function should guess through.
    """
    notes: list[str] = []
    rows = rows.copy()
    n = len(rows)
    rank = rows["s1400_jy"].fillna(0) + rows["hi_mass_msun"].fillna(0) / 1e10
    ranked_idx = rank.sort_values(ascending=False).index
    if n > max_rows:
        notes.append(f"SIMBAD enrichment limited to the {max_rows} brightest/"
                     f"most massive of {n} sources")
        enrich_idx = ranked_idx[:max_rows]
    else:
        enrich_idx = ranked_idx

    import astropy.units as u
    from astropy.coordinates import SkyCoord

    matched = 0
    for done, i in enumerate(enrich_idx, start=1):
        row = rows.loc[i]
        radius = (_SIMBAD_MATCH_RADIUS_ALFALFA_ARCSEC
                 if row["hi_emitter"] else _SIMBAD_MATCH_RADIUS_NVSS_ARCSEC)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c = SkyCoord(str(row["ra"]), str(row["dec"]),
                        unit=(u.hourangle, u.deg), frame="icrs")
        try:
            result = fetch_fn(c, radius * u.arcsec)
        except Exception:                                     # noqa: BLE001
            result = None
        if progress:
            progress(done, len(enrich_idx))
        if result is None or len(result) != 1:
            continue                                            # no / ambiguous match
        name = str(result["main_id"][0])
        kind = kind_from_simbad_otype(result["otype"][0]) if "otype" in result.colnames else None
        rows.at[i, "aliases"] = (f"{row['aliases']}|{name}".strip("|")
                                 if row["aliases"] else name)
        if kind:
            rows.at[i, "kind"] = kind
        matched += 1
        if sleep_s:
            time.sleep(sleep_s)

    notes.append(f"SIMBAD matched {matched} of {len(enrich_idx)} queried sources")
    return rows, notes


# ---------------------------------------------------------------------------
# fetch functions -- the only place astroquery is actually imported/called.
# Kept as separate, small, monkeypatchable seams so build_live_catalog's
# orchestration (caching, merging, fallback) is testable without a network.
# ---------------------------------------------------------------------------
def _fetch_nvss(dec_lo: float, dec_hi: float, flux_floor_jy: float,
                max_rows: int) -> pd.DataFrame:
    from astroquery.vizier import Vizier

    v = Vizier(columns=["NVSS", "RAJ2000", "DEJ2000", "S1.4", "MajAxis", "MinAxis"],
              row_limit=max_rows)
    result = v.query_constraints(
        catalog=NVSS_CATALOG_ID,
        **{"S1.4": f">{flux_floor_jy * 1000.0}", "DEJ2000": f"{dec_lo}..{dec_hi}"},
    )
    if not result:
        return pd.DataFrame(columns=RAW_COLUMNS)
    rows = [nvss_row_to_hydroline(dict(zip(result[0].colnames, r)))
           for r in result[0]]
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def _fetch_alfalfa(dec_lo: float, dec_hi: float, max_rows: int,
                   top_n: int = 40) -> pd.DataFrame:
    """Fetch ALFALFA detections in the Dec band, keeping the `top_n`
    brightest by implied peak line flux.

    A hard antenna-sensitivity gate here was tried and measured to be the
    wrong tool: even a generous 24 h matched-filter integration puts a
    1.5 m dish's HI sensitivity around 3 Jy peak, and essentially no real
    external galaxy clears that (Arecibo, which ALFALFA was observed with,
    has ~41,000x the collecting area of a 1.5 m dish) -- a strict floor
    zeroes out every external HI galaxy for every realistic small-dish
    preset, which defeats the point. The static catalogue this replaces
    included bright, well-known galaxies (M31, M33, ...) specifically as
    informational/aspirational targets even though a small dish can't
    detect them either -- "what's really out there" is deliberately a
    broader question than "what's detectable", and
    detectability_table() already draws that distinction downstream, per
    source, correctly. A rank-based cap instead just keeps this module's
    job -- not exploding a report meant for a few hundred sources into one
    with several thousand -- separate from that detectability judgement.
    """
    from astroquery.vizier import Vizier

    v = Vizier(columns=["Name", "AGC", "RAJ2000", "DEJ2000", "Vhel", "W50",
                        "Dist", "logMHI"], row_limit=max_rows)
    result = v.query_constraints(catalog=ALFALFA_CATALOG_ID,
                                 DEJ2000=f"{dec_lo}..{dec_hi}")
    if not result:
        return pd.DataFrame(columns=RAW_COLUMNS)
    scored: list[tuple[float, dict]] = []
    for r in result[0]:
        raw = dict(zip(result[0].colnames, r))
        peak = _implied_hi_peak_jy(
            raw.get("Dist"),
            10.0 ** float(raw["logMHI"]) if _finite(raw.get("logMHI")) else float("nan"),
            raw.get("W50"))
        scored.append((peak if np.isfinite(peak) else -1.0, raw))
    scored.sort(key=lambda t: t[0], reverse=True)
    rows = [alfalfa_row_to_hydroline(raw) for _, raw in scored[:top_n]]
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def _make_simbad_fetcher() -> Callable[..., Any]:
    """One configured Simbad client, reused across every enrichment query.

    Constructing Simbad() and calling add_votable_fields() has real,
    measured setup cost; doing that fresh on every row (the first version of
    this code did) was a needless multiplier on top of per-query network
    latency, which is itself the dominant, unavoidable cost of this
    (measured directly: several seconds per query against the real service,
    not the sub-second figure one already-warm request suggests).
    """
    from astroquery.simbad import Simbad

    sim = Simbad()
    sim.add_votable_fields("otype")
    return sim.query_region


def _safe_fetch(fn: Callable[[], pd.DataFrame], label: str,
                warnings_out: list[str]) -> pd.DataFrame:
    try:
        return fn()
    except Exception as exc:                                    # noqa: BLE001
        warnings_out.append(f"{label} query failed ({type(exc).__name__}: {exc})")
        return pd.DataFrame(columns=RAW_COLUMNS)


# ---------------------------------------------------------------------------
# caching
# ---------------------------------------------------------------------------
def _cache_key(dec_lo: float, dec_hi: float, flux_floor_jy: float,
              alfalfa_top_n: int, include_simbad: bool) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION, "nvss": NVSS_CATALOG_ID,
        "alfalfa": ALFALFA_CATALOG_ID, "dec_lo": round(dec_lo, 2),
        "dec_hi": round(dec_hi, 2), "flux_floor_jy": round(flux_floor_jy, 4),
        "alfalfa_top_n": alfalfa_top_n,
        "include_simbad": include_simbad,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    return f"live_{payload['dec_lo']:+05.1f}_{payload['dec_hi']:+05.1f}_{digest}"


def _read_cache(cache_dir: Path, key: str, max_age_days: float
               ) -> tuple[pd.DataFrame, dict] | None:
    csv_path = cache_dir / f"{key}.csv"
    meta_path = cache_dir / f"{key}.meta.json"
    if not (csv_path.exists() and meta_path.exists()):
        return None
    try:
        meta = json.loads(meta_path.read_text())
        df = pd.read_csv(csv_path)
        if set(RAW_COLUMNS) - set(df.columns):
            return None                                          # corrupt: self-heal
        age_days = (time.time() - meta["fetched_unix"]) / 86400.0
        if age_days > max_age_days:
            return None
        return df, meta
    except (json.JSONDecodeError, KeyError, pd.errors.ParserError, OSError):
        return None                                               # corrupt: self-heal


def _write_cache(cache_dir: Path, key: str, df: pd.DataFrame, meta: dict) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / f"{key}.csv"
    df.to_csv(csv_path, index=False)
    (cache_dir / f"{key}.meta.json").write_text(json.dumps(meta, indent=2))
    return csv_path


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------
def build_live_catalog(
    dec_deg: float, half_width_deg: float, model: AntennaModel, *,
    flux_floor_jy: float | None = None, flux_floor_multiplier: float = 0.5,
    alfalfa_top_n: int = 40,
    include_simbad: bool = True, simbad_max_rows: int = 40,
    cache_dir: Path | str = LIVE_CACHE_DIR,
    max_age_days: float = 90.0, refresh: bool = False,
    max_rows: int = 20_000, say: Callable[[str], None] = lambda msg: None,
) -> LiveCatalogResult:
    """Build a Catalog for the declination band around the beam.

    Tries NVSS + ALFALFA (+ SIMBAD enrichment), caches the merged result to
    `cache_dir`, and returns it wrapped exactly like `load_catalog()` would.
    Raises :class:`LiveCatalogError` only when neither a live query nor a
    cache hit produced anything usable -- callers (see `cli.py`) are
    expected to catch this and fall back to the static catalog. `say`, if
    given, receives short progress messages -- this can take tens of
    seconds (SIMBAD enrichment in particular queries one position at a
    time), and a live run should not sit silent that whole time.
    """
    cache_dir = Path(cache_dir)
    query_half_width = min(half_width_deg, 40.0)
    clamp_note = (
        [f"beam half-width {half_width_deg:.1f} deg exceeds the practical "
         f"live-query bound of 40 deg; results cover only "
         f"+/-{query_half_width:.0f} deg"] if query_half_width < half_width_deg
        else []
    )
    dec_lo = max(dec_deg - query_half_width, -90.0)
    dec_hi = min(dec_deg + query_half_width, 90.0)

    floor = (flux_floor_jy if flux_floor_jy is not None
            else antenna_flux_floor_jy(model, flux_floor_multiplier))
    key = _cache_key(dec_lo, dec_hi, floor, alfalfa_top_n, include_simbad)

    if not refresh:
        cached = _read_cache(cache_dir, key, max_age_days)
        if cached is not None:
            df, meta = cached
            df.to_csv(cache_dir / f"{key}.csv", index=False)  # re-touch, cheap
            catalog = load_catalog(cache_dir / f"{key}.csv")
            say(f"Catalogue: served from cache ({meta.get('counts', {}).get('total', len(df))} sources)")
            return LiveCatalogResult(
                catalog=catalog.filter_declination(dec_deg, half_width_deg),
                cache_path=cache_dir / f"{key}.csv", from_cache=True,
                counts=meta.get("counts", {}), warnings=meta.get("warnings", []),
            )

    run_warnings: list[str] = list(clamp_note)
    say(f"Catalogue: querying NVSS (Dec {dec_lo:+.1f}° to {dec_hi:+.1f}°, "
        f"flux floor {floor:.3f} Jy)…")
    nvss_df = _safe_fetch(
        lambda: _fetch_nvss(dec_lo, dec_hi, floor, max_rows), "NVSS", run_warnings)
    say(f"Catalogue: querying ALFALFA (same Dec band, keeping the "
        f"{alfalfa_top_n} brightest by implied HI peak flux)…")
    alfalfa_df = _safe_fetch(
        lambda: _fetch_alfalfa(dec_lo, dec_hi, max_rows, alfalfa_top_n),
        "ALFALFA", run_warnings)
    say(f"Catalogue: {len(nvss_df)} NVSS + {len(alfalfa_df)} ALFALFA sources fetched")

    if len(nvss_df) >= max_rows:
        run_warnings.append(f"NVSS query hit the {max_rows}-row cap; some "
                            f"sources may be missing -- raise --flux-floor")
    if len(alfalfa_df) >= max_rows:
        run_warnings.append(f"ALFALFA query hit the {max_rows}-row cap")
    if dec_lo > 36.0 or dec_hi < 0.0:
        run_warnings.append(
            "ALFALFA only covers roughly declination 0 to +36 deg (it is an "
            "Arecibo survey); this beam's declination band falls outside "
            "that footprint, so 0 HI emitters is expected here, not a bug"
        )

    if nvss_df.empty and alfalfa_df.empty:
        cached = _read_cache(cache_dir, key, max_age_days=float("inf"))
        if cached is not None:
            df, meta = cached
            run_warnings.append("live query returned nothing; served a "
                                "stale cache entry instead")
            catalog = load_catalog(cache_dir / f"{key}.csv")
            return LiveCatalogResult(
                catalog=catalog.filter_declination(dec_deg, half_width_deg),
                cache_path=cache_dir / f"{key}.csv", from_cache=True,
                counts=meta.get("counts", {}), warnings=run_warnings, )
        raise LiveCatalogError(
            "no usable live catalogue: NVSS and ALFALFA both returned "
            "nothing (network unreachable, both services down, or an "
            "empty declination band) and no cache entry exists for these "
            "parameters. Pass --catalog to use the bundled static catalogue, "
            "or --offline to skip live querying entirely."
        )

    merged = merge_nvss_alfalfa(nvss_df, alfalfa_df)
    counts = {"nvss": int(len(nvss_df)), "alfalfa": int(len(alfalfa_df)),
             "merged": int(len(merged)), "simbad_matched": 0}

    if include_simbad and not merged.empty:
        n_enrich = min(simbad_max_rows, len(merged))
        say(f"Catalogue: enriching {n_enrich} of {len(merged)} sources via "
            f"SIMBAD (one query per source, ~a few seconds each)…")

        def _progress(done: int, total: int) -> None:
            if done == total or done % 10 == 0:
                say(f"Catalogue: SIMBAD {done}/{total}")

        try:
            fetch = _make_simbad_fetcher()
            merged, simbad_notes = enrich_with_simbad(
                merged, fetch, max_rows=simbad_max_rows, progress=_progress)
            run_warnings.extend(simbad_notes)
        except Exception as exc:                                # noqa: BLE001
            run_warnings.append(f"SIMBAD enrichment skipped "
                                f"({type(exc).__name__}: {exc})")
        for note in simbad_notes:
            if note.startswith("SIMBAD matched"):
                counts["simbad_matched"] = int(note.split()[2])

    counts["total"] = int(len(merged))
    meta = {"fetched_unix": time.time(), "counts": counts, "warnings": run_warnings,
           "dec_lo": dec_lo, "dec_hi": dec_hi, "flux_floor_jy": floor}
    csv_path = _write_cache(cache_dir, key, merged[RAW_COLUMNS], meta)

    catalog = load_catalog(csv_path)
    return LiveCatalogResult(
        catalog=catalog.filter_declination(dec_deg, half_width_deg),
        cache_path=csv_path, from_cache=False, counts=counts,
        warnings=run_warnings,
    )
