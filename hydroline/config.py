"""User-facing configuration: where the array is, what the antenna is, when to look.

Everything downstream reads these three dataclasses, so a change here (move the
array to a different roof, swap the horn for a dish) propagates through the
pointing track, the beam size, the sensitivity and the source lists.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
from typing import Any

from .constants import HI_REST_HZ


# ---------------------------------------------------------------------------
# Site
# ---------------------------------------------------------------------------
@dataclass
class Site:
    """An observing location on the Earth's surface.

    Latitude is *geodetic* (what a GPS reports).  That matters: the local
    vertical -- which is where a level antenna points -- is normal to the
    reference ellipsoid, not to a sphere, so the declination of the zenith
    equals the geodetic latitude, differing from the geocentric latitude by
    up to ~11.5 arcmin.
    """

    name: str = "Unnamed site"
    latitude_deg: float = 0.0            # geodetic, +N
    longitude_deg: float = 0.0           # +E (west longitudes are negative)
    elevation_m: float = 0.0             # height above the WGS84 ellipsoid
    timezone: str = "UTC"                # IANA name, for human-readable output
    horizon_deg: float = 10.0            # roof/tree obstruction, for context plots

    # Refraction model.  pressure_hpa = 0 disables refraction entirely, which is
    # the right choice for a zenith-pointing antenna: atmospheric refraction is
    # identically zero at the zenith.  Set a real pressure only if you tilt the
    # mount and care about sub-arcminute pointing near the horizon.
    pressure_hpa: float = 0.0
    temperature_c: float = 15.0
    relative_humidity: float = 0.4

    def __post_init__(self) -> None:
        if not -90.0 <= self.latitude_deg <= 90.0:
            raise ValueError(f"latitude out of range: {self.latitude_deg}")
        if not -180.0 <= self.longitude_deg <= 360.0:
            raise ValueError(f"longitude out of range: {self.longitude_deg}")
        # Normalise to (-180, 180] so plots and printouts are unsurprising.
        if self.longitude_deg > 180.0:
            self.longitude_deg -= 360.0


#: Default site: rooftop of the Physics building, Duke University, Durham NC.
#: Coordinates are campus-accurate to ~50 m; replace with a GPS fix from the
#: actual mounting point before you trust the sub-degree numbers.
DUKE_PHYSICS = Site(
    name="Duke University Physics rooftop, Durham NC",
    latitude_deg=36.0021,
    longitude_deg=-78.9406,
    elevation_m=140.0,
    timezone="America/New_York",
    horizon_deg=15.0,
)


# ---------------------------------------------------------------------------
# Antenna
# ---------------------------------------------------------------------------
@dataclass
class Antenna:
    """Antenna + receiver parameters.

    Three ways to specify the beam, checked in this order:

    ``kind="dish"``
        Circular aperture of ``diameter_m``.  HPBW = ``beam_factor_deg`` *
        lambda / D.  The factor is 58.4 for uniform illumination and ~70 for a
        realistically tapered feed; 70 is the default because real dishes are
        tapered.

    ``kind="horn"``
        Rectangular aperture ``aperture_e_m`` x ``aperture_h_m``.  Separate
        E- and H-plane beamwidths (54 lambda/a_E and 78 lambda/a_H degrees for
        an optimum-gain pyramidal horn).

    ``kind="beamwidth"``
        You measured or simulated the pattern: give ``hpbw_deg`` directly.
        Collecting area then comes from ``diameter_m`` if set, otherwise from
        the beam solid angle (which assumes *all* the power is in the main
        beam and so is optimistic).
    """

    name: str = "1.5 m prime-focus dish"
    kind: str = "dish"                      # dish | horn | beamwidth

    # --- geometry ---
    diameter_m: float = 1.5                 # circular aperture
    aperture_e_m: float | None = None       # horn E-plane opening
    aperture_h_m: float | None = None       # horn H-plane opening
    hpbw_deg: float | None = None           # explicit half-power beamwidth

    beam_factor_deg: float = 70.0           # HPBW[deg] = factor * lambda / D
    horn_factor_e_deg: float = 54.0
    horn_factor_h_deg: float = 78.0
    aperture_efficiency: float = 0.60       # A_eff / A_geometric

    # --- receiver ---
    center_freq_hz: float = HI_REST_HZ
    t_sys_k: float = 150.0                  # LNA + spillover + sky + losses
    rf_bandwidth_hz: float = 2.5e6          # total processed bandwidth
    channel_hz: float = 5.0e3               # spectrometer channel width
    integration_s: float = 300.0            # per-pointing integration
    n_pol: int = 1                          # 1 = single pol, 2 = dual summed
    radiometer_factor: float = 1.0          # K_s: >1 for switched / correlation
    snr_threshold: float = 5.0              # what counts as a detection

    # --- pointing ---
    # Straight up is the design case; leave these alone unless you tilt the mount.
    pointing_alt_deg: float = 90.0
    pointing_az_deg: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in {"dish", "horn", "beamwidth"}:
            raise ValueError(f"unknown antenna kind: {self.kind!r}")
        if self.kind == "horn" and not (self.aperture_e_m and self.aperture_h_m):
            raise ValueError("horn requires aperture_e_m and aperture_h_m")
        if self.kind == "beamwidth" and not self.hpbw_deg:
            raise ValueError("kind='beamwidth' requires hpbw_deg")
        if not 0.0 < self.aperture_efficiency <= 1.0:
            raise ValueError("aperture_efficiency must be in (0, 1]")
        if self.n_pol not in (1, 2):
            raise ValueError("n_pol must be 1 or 2")
        if not 0.0 <= self.pointing_alt_deg <= 90.0:
            raise ValueError("pointing_alt_deg must be in [0, 90]")


# A few ready-made antennas covering the range a student group actually builds.
PRESET_ANTENNAS: dict[str, Antenna] = {
    "horn-cantenna": Antenna(
        name="Cylindrical 'cantenna' feed, no reflector",
        kind="beamwidth", hpbw_deg=60.0, diameter_m=0.30,
        aperture_efficiency=0.5, t_sys_k=200.0,
    ),
    "horn-small": Antenna(
        name="0.6 x 0.5 m pyramidal horn",
        kind="horn", aperture_e_m=0.60, aperture_h_m=0.50,
        aperture_efficiency=0.51, t_sys_k=180.0,
    ),
    "horn-large": Antenna(
        name="1.2 x 0.9 m pyramidal horn",
        kind="horn", aperture_e_m=1.20, aperture_h_m=0.90,
        aperture_efficiency=0.51, t_sys_k=150.0,
    ),
    "dish-1m2": Antenna(
        name="1.2 m offset satellite dish",
        kind="dish", diameter_m=1.2, aperture_efficiency=0.55, t_sys_k=150.0,
    ),
    "dish-1m5": Antenna(
        name="1.5 m prime-focus dish",
        kind="dish", diameter_m=1.5, aperture_efficiency=0.60, t_sys_k=150.0,
    ),
    "dish-3m": Antenna(
        name="3 m mesh dish",
        kind="dish", diameter_m=3.0, aperture_efficiency=0.60, t_sys_k=120.0,
    ),
    "dish-4m6": Antenna(
        name="4.6 m dish (SRT class)",
        kind="dish", diameter_m=4.6, aperture_efficiency=0.55, t_sys_k=110.0,
    ),
}


# ---------------------------------------------------------------------------
# Observation window
# ---------------------------------------------------------------------------
@dataclass
class Observation:
    """The time grid to evaluate the pointing on."""

    start_utc: str = "auto"      # ISO 8601, or "auto" = next midnight UTC
    duration_hours: float = 24.0
    step_minutes: float = 5.0
    #: Separation below which a source counts as "in the beam".  Default is
    #: half the HPBW (i.e. inside the half-power contour).
    beam_fraction: float = 0.5

    def __post_init__(self) -> None:
        if self.duration_hours <= 0:
            raise ValueError("duration_hours must be positive")
        if self.step_minutes <= 0:
            raise ValueError("step_minutes must be positive")
        if self.duration_hours * 60.0 / self.step_minutes > 500_000:
            raise ValueError("time grid too large; increase step_minutes")


# ---------------------------------------------------------------------------
# Bundle + serialisation
# ---------------------------------------------------------------------------
@dataclass
class Configuration:
    site: Site = field(default_factory=lambda: DUKE_PHYSICS)
    antenna: Antenna = field(default_factory=Antenna)
    observation: Observation = field(default_factory=Observation)

    # ---- serialisation ----
    def to_dict(self) -> dict[str, Any]:
        return {
            "site": asdict(self.site),
            "antenna": asdict(self.antenna),
            "observation": asdict(self.observation),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Configuration":
        cfg = cls()
        if "site" in data:
            cfg.site = replace(cfg.site, **_filter(Site, data["site"]))
        if "antenna" in data:
            preset = data["antenna"].get("preset")
            base = PRESET_ANTENNAS[preset] if preset else cfg.antenna
            fields = _filter(Antenna, data["antenna"])
            cfg.antenna = replace(base, **fields) if fields else base
        if "observation" in data:
            cfg.observation = replace(
                cfg.observation, **_filter(Observation, data["observation"])
            )
        return cfg

    @classmethod
    def load(cls, path: str | Path) -> "Configuration":
        """Read a YAML or JSON configuration file."""
        path = Path(path)
        text = path.read_text()
        if path.suffix.lower() in {".yaml", ".yml"}:
            import yaml  # optional dependency, only needed for YAML configs

            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text)
        return cls.from_dict(data)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        if path.suffix.lower() in {".yaml", ".yml"}:
            import yaml

            path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
        else:
            path.write_text(json.dumps(self.to_dict(), indent=2))
        return path


def _filter(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    """Keep the keys that are real fields of ``cls``, coerced to their type.

    Coercion matters because YAML 1.1 only reads ``2.5e6`` as a float when it is
    written ``2.5e+6``; bare ``2.5e6`` arrives as the string "2.5e6" and would
    otherwise blow up several layers later with an unhelpful message.
    """
    fields = cls.__dataclass_fields__
    out: dict[str, Any] = {}
    for key, value in data.items():
        if key not in fields:
            continue                       # e.g. 'preset', handled by caller
        out[key] = _coerce(value, str(fields[key].type), key)
    return out


def _coerce(value: Any, annotation: str, key: str) -> Any:
    """Best-effort cast of a config value to its annotated type."""
    if value is None:
        return None
    try:
        if "float" in annotation and not isinstance(value, bool):
            return float(value)
        if "int" in annotation and not isinstance(value, bool):
            return int(float(value))
        if "bool" in annotation and not isinstance(value, bool):
            return str(value).strip().lower() in {"1", "true", "yes", "on"}
        if annotation.startswith("str") and not isinstance(value, str):
            return str(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"configuration value for {key!r} is {value!r}, which is not a "
            f"valid {annotation}"
        ) from exc
    return value
