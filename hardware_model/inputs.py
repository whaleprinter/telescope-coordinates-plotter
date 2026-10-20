"""Reading the input files.  No physics happens here.

This module turns ``hardware_parameters.yaml`` and ``observing_goals.yaml``
into plain Python objects.  It does two things only:

1. **Unit conversion.**  Every number is written the way a datasheet gives it
   (``{value: 0.7, unit: GHz}``) and converted once, here, to the unit the
   maths uses (Hz, m, s, K, deg, dB).  Conversion is multiplication by an
   exact factor (1 GHz = 1e9 Hz, 1 ft = 0.3048 m exactly, 0 C = 273.15 K).
2. **Validation.**  Unknown keys, unknown units, missing required values,
   duplicate chain positions -- each is reported with the file location.

Every equation lives in ``performance.py``.
"""
from __future__ import annotations

import csv
import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np


class HardwareConfigError(ValueError):
    """The input file is malformed; the message says where and how."""


# ---------------------------------------------------------------------------
# units: (lower-case unit) -> (canonical unit, exact multiplier)
# ---------------------------------------------------------------------------
_UNITS: dict[str, tuple[str, float]] = {
    "hz": ("Hz", 1.0), "khz": ("Hz", 1e3), "mhz": ("Hz", 1e6), "ghz": ("Hz", 1e9),
    "sps": ("Hz", 1.0), "ksps": ("Hz", 1e3), "msps": ("Hz", 1e6), "ms/s": ("Hz", 1e6),
    "m": ("m", 1.0), "cm": ("m", 1e-2), "mm": ("m", 1e-3),
    "ft": ("m", 0.3048), "in": ("m", 0.0254),
    "s": ("s", 1.0), "min": ("s", 60.0), "h": ("s", 3600.0), "hr": ("s", 3600.0),
    "k": ("K", 1.0),
    "deg": ("deg", 1.0), "arcmin": ("deg", 1.0 / 60.0), "rad": ("deg", 180.0 / np.pi),
    "db": ("dB", 1.0), "dbi": ("dBi", 1.0), "dbm": ("dBm", 1.0),
    "db/m": ("dB/m", 1.0), "db/100m": ("dB/m", 0.01),
    "db/ft": ("dB/m", 1.0 / 0.3048), "db/100ft": ("dB/m", 1.0 / 30.48),
    "jy": ("Jy", 1.0), "mjy": ("Jy", 1e-3),
    "km/s": ("km/s", 1.0), "m/s": ("km/s", 1e-3),
    "w": ("W", 1.0), "mw": ("W", 1e-3), "v": ("V", 1.0), "a": ("A", 1.0),
    "bit": ("bit", 1.0), "bits": ("bit", 1.0),
    "": ("", 1.0), ":1": ("", 1.0), "ratio": ("", 1.0),
}
_CELSIUS = {"c", "degc", "°c"}

PARAM_KEYS = {"value", "unit", "status", "notes", "curve", "file", "freq_unit"}
STATUSES = {"datasheet", "measured", "estimate", "placeholder", "computed", "default"}
CHAIN_TYPES = {"amplifier", "filter", "cable", "attenuator", "sdr"}
STRUCTURE_TYPES = {"reflector", "feed", "mount"}


def convert(value: float, unit: str, *, where: str) -> tuple[float, str]:
    """``value`` in ``unit`` -> (value in the canonical unit, canonical unit)."""
    u = (unit or "").strip().lower()
    if u in _CELSIUS:
        return float(value) + 273.15, "K"
    if u not in _UNITS:
        close = difflib.get_close_matches(u, list(_UNITS), n=3)
        hint = f" (did you mean {', '.join(close)}?)" if close else ""
        raise HardwareConfigError(f"{where}: unknown unit {unit!r}{hint}")
    canon, mult = _UNITS[u]
    return float(value) * mult, canon


# ---------------------------------------------------------------------------
# one {value, unit, status, notes} parameter
# ---------------------------------------------------------------------------
@dataclass
class Param:
    """A parameter in canonical units: a number, a string, or a curve vs frequency."""

    name: str
    value: Any = None
    unit: str = ""
    status: str = "datasheet"
    notes: str = ""
    curve_hz: np.ndarray | None = None
    curve_val: np.ndarray | None = None
    source_unit: str = ""

    @property
    def is_set(self) -> bool:
        return self.value is not None or self.curve_hz is not None

    def at(self, freq_hz: Any) -> np.ndarray:
        """Value at ``freq_hz``.  A curve is linearly interpolated between its
        points and held flat beyond its ends (a lookup, not a physical model)."""
        f = np.asarray(freq_hz, dtype=float)
        if self.curve_hz is not None:
            return np.interp(f, self.curve_hz, self.curve_val)
        if self.value is None:
            raise HardwareConfigError(f"parameter {self.name!r} has no value")
        return np.full_like(f, float(self.value), dtype=float)

    def scalar(self, freq_hz: float | None = None) -> float:
        if self.curve_hz is not None:
            if freq_hz is None:
                raise HardwareConfigError(
                    f"{self.name!r} is a frequency curve; a frequency is needed")
            return float(self.at(freq_hz))
        if self.value is None:
            raise HardwareConfigError(f"parameter {self.name!r} has no value")
        return float(self.value)


def parse_param(name: str, raw: Any, *, where: str, base_dir: Path) -> Param:
    """One YAML entry -> :class:`Param` (a bare scalar is a descriptive entry)."""
    loc = f"{where}.{name}"
    if not isinstance(raw, dict):
        return Param(name=name, value=raw, status="datasheet")

    unknown = set(raw) - PARAM_KEYS
    if unknown:
        msgs = []
        for key in sorted(unknown):
            close = difflib.get_close_matches(key, PARAM_KEYS, n=1)
            msgs.append(f"{key!r}" + (f" (did you mean {close[0]!r}?)" if close else ""))
        hint = ""
        if any(raw[k] is None for k in unknown):
            hint = (". If that is the tail of your notes, the comma inside "
                    "{...} split it: quote the notes, e.g. notes: \"a, b\"")
        raise HardwareConfigError(
            f"{loc}: unknown key(s) {', '.join(msgs)}; allowed: "
            f"{', '.join(sorted(PARAM_KEYS))}{hint}")

    status = str(raw.get("status") or "datasheet").strip().lower()
    if status not in STATUSES:
        raise HardwareConfigError(
            f"{loc}: status {status!r} is not one of {', '.join(sorted(STATUSES))}")
    unit = "" if raw.get("unit") is None else str(raw.get("unit"))
    p = Param(name=name, status=status, notes=str(raw.get("notes") or ""), source_unit=unit)

    value = raw.get("value")
    if isinstance(value, str):
        try:                         # YAML 1.1 reads a bare 2.5e6 as text
            value = float(value)
        except ValueError:
            p.value, p.unit = value, unit
            return p
    if value is not None:
        p.value, p.unit = convert(value, unit, where=loc)
    else:
        p.unit = convert(1.0, unit, where=loc)[1]

    pairs = None
    if raw.get("curve") is not None:
        pairs = raw["curve"]
    elif raw.get("file"):
        pairs = _read_curve_file(base_dir / str(raw["file"]), where=loc)
    if pairs is not None:
        try:
            arr = np.asarray(pairs, dtype=float)
        except (TypeError, ValueError) as exc:
            raise HardwareConfigError(
                f"{loc}: curve must be a list of [frequency, value] pairs") from exc
        if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < 2:
            raise HardwareConfigError(f"{loc}: curve needs at least two [frequency, value] pairs")
        f_mult, _ = convert(1.0, str(raw.get("freq_unit") or "MHz"), where=f"{loc}.freq_unit")
        order = np.argsort(arr[:, 0])
        p.curve_hz = arr[order, 0] * f_mult
        p.curve_val = np.array([convert(v, unit, where=loc)[0] for v in arr[order, 1]])
    return p


def _read_curve_file(path: Path, *, where: str) -> list[list[float]]:
    if not path.exists():
        raise HardwareConfigError(f"{where}: curve file not found: {path}")
    rows = []
    with path.open(newline="") as fh:
        for row in csv.reader(fh):
            if not row or row[0].strip().startswith("#"):
                continue
            try:
                rows.append([float(row[0]), float(row[1])])
            except (ValueError, IndexError):
                continue                                   # header line
    return rows


@dataclass
class ParamSet:
    """The ``parameters:`` block of one component."""

    owner: str
    params: dict[str, Param] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, owner: str, raw: dict | None, base_dir: Path) -> "ParamSet":
        ps = cls(owner=owner)
        for name, entry in (raw or {}).items():
            ps.params[name] = parse_param(name, entry, where=owner, base_dir=base_dir)
        return ps

    def get(self, name: str) -> Param | None:
        p = self.params.get(name)
        return p if p is not None and p.is_set else None

    def require(self, name: str, unit: str) -> Param:
        p = self.get(name)
        if p is None:
            raise HardwareConfigError(
                f"{self.owner}: parameter {name!r} is required (unit {unit or 'none'})")
        self._check_unit(p, unit)
        return p

    def number(self, name: str, unit: str, default: float | None = None,
               freq_hz: float | None = None) -> float | None:
        p = self.get(name)
        if p is None:
            return default
        self._check_unit(p, unit)
        return p.scalar(freq_hz)

    def text(self, name: str, default: str = "") -> str:
        p = self.params.get(name)
        return default if p is None or p.value is None else str(p.value)

    def _check_unit(self, p: Param, unit: str) -> None:
        if p.unit != unit and not (unit == "dB" and p.unit == "dBi"):
            raise HardwareConfigError(
                f"{self.owner}.{p.name}: expected a quantity in {unit or 'no unit'}, "
                f"got {p.source_unit!r}")

    def placeholders(self) -> list[str]:
        return [n for n, p in self.params.items()
                if p.is_set and p.status in {"placeholder", "estimate"}]


# ---------------------------------------------------------------------------
# Touchstone (.s2p) files: read the raw S21 numbers, no conversion
# ---------------------------------------------------------------------------
@dataclass
class TouchstoneS21:
    freq_hz: np.ndarray
    fmt: str                 # "DB", "MA" or "RI" -- the file's own data format
    a: np.ndarray            # first number of the S21 pair
    b: np.ndarray            # second number of the S21 pair


def read_touchstone_s21(path: Path) -> TouchstoneS21:
    """S21 from a 2-port Touchstone file, as stored (Touchstone v1.1 layout:
    ``f S11 S21 S12 S22``, two numbers each, after a ``# <unit> S <fmt> R z``
    option line).  ``performance.s21_db`` converts it to dB."""
    if not path.exists():
        raise HardwareConfigError(f"Touchstone file not found: {path}")
    f_mult, fmt = 1e9, "MA"                       # Touchstone defaults
    data: list[float] = []
    for line in path.read_text().splitlines():
        line = line.split("!", 1)[0].strip()
        if not line:
            continue
        if line.startswith("#"):
            for tok in line[1:].upper().split():
                if tok in {"HZ", "KHZ", "MHZ", "GHZ"}:
                    f_mult = {"HZ": 1.0, "KHZ": 1e3, "MHZ": 1e6, "GHZ": 1e9}[tok]
                elif tok in {"DB", "MA", "RI"}:
                    fmt = tok
            continue
        data.extend(float(x) for x in line.split())
    if not data or len(data) % 9:
        raise HardwareConfigError(f"{path}: not a 2-port Touchstone file")
    arr = np.asarray(data).reshape(-1, 9)
    return TouchstoneS21(arr[:, 0] * f_mult, fmt, arr[:, 3], arr[:, 4])


# ---------------------------------------------------------------------------
# the hardware description
# ---------------------------------------------------------------------------
@dataclass
class ComponentSpec:
    """One RF-chain part exactly as the file describes it."""

    id: str
    type: str
    params: ParamSet
    chain_position: float
    t_phys_k: float
    model: str = ""
    description: str = ""
    s21: TouchstoneS21 | None = None
    center_hz: float = 1.42e9          # the analysis centre (SDR default tuning)

    @property
    def label(self) -> str:
        return self.id + (f" ({self.model})" if self.model else "")


@dataclass
class ReflectorSpec:
    diameter_m: float
    focal_ratio: float
    surface_rms_m: float
    blockage_fraction: float


@dataclass
class FeedSpec:
    kind: str
    hpbw_e_deg: float | None
    hpbw_h_deg: float | None
    edge_taper_e_db: float | None
    edge_taper_h_db: float | None
    waveguide_diameter_m: float | None
    freq_min_hz: float | None
    freq_max_hz: float | None
    rolloff_db_per_octave: float
    ohmic_loss_db: float
    vswr: float
    back_fraction: float
    e_plane_azimuth_deg: float
    t_phys_k: float


@dataclass
class Environment:
    latitude_deg: float
    ambient_k: float
    ground_k: float
    cmb_k: float
    t408_k: float
    spectral_index: float
    atmosphere_k: float
    horizon_deg: float


@dataclass
class Hardware:
    name: str
    path: Path | None
    center_hz: float
    sweep_hz: tuple[float, float, int]
    environment: Environment
    reflector: ReflectorSpec | None
    feed: FeedSpec
    pointing_alt_deg: float
    pointing_az_deg: float
    mount_type: str
    chain: list[ComponentSpec]
    disabled: list[str]
    fft_size: float | None
    channel_width_hz: float | None
    n_pol: int
    k_s: float
    assumptions: list[str] = field(default_factory=list)
    placeholders: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def order(self) -> list[str]:
        return [c.id for c in self.chain]

    @property
    def sdr(self) -> ComponentSpec | None:
        sdrs = [c for c in self.chain if c.type == "sdr"]
        return sdrs[-1] if sdrs else None


def load_yaml(path: Path) -> dict:
    import yaml
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f" (line {mark.line + 1}, column {mark.column + 1})" if mark else ""
        raise HardwareConfigError(
            f"{path} is not valid YAML{where}: {getattr(exc, 'problem', exc)}. "
            f"A common cause is a value starting with ':' -- quote it, e.g. "
            f"unit: \":1\"") from exc
    if not isinstance(data, dict):
        raise HardwareConfigError(f"{path}: expected a mapping at the top level")
    return data


def _component(raw: dict, *, index: int, base_dir: Path, default_t_phys_k: float,
               center_hz: float) -> ComponentSpec:
    cid = str(raw.get("id") or f"component_{index}")
    ctype = str(raw.get("type") or "").strip().lower()
    if raw.get("chain_position") is None:
        raise HardwareConfigError(f"component {cid!r}: chain_position is required")
    params = ParamSet.from_yaml(cid, raw.get("parameters"), base_dir)
    s21 = read_touchstone_s21(base_dir / raw["sparameters"]) if raw.get("sparameters") else None
    return ComponentSpec(
        id=cid, type=ctype, params=params, chain_position=float(raw["chain_position"]),
        t_phys_k=params.number("physical_temperature", "K", default_t_phys_k),
        model=str(raw.get("model") or ""), description=str(raw.get("description") or ""),
        s21=s21, center_hz=center_hz)


def _feed(ps: ParamSet, has_reflector: bool, default_t_phys_k: float,
          assumptions: list[str]) -> FeedSpec:
    spec = {}
    for plane in ("e", "h"):
        hpbw = ps.number(f"hpbw_{plane}", "deg")
        taper = ps.number(f"edge_taper_{plane}", "dB")
        if hpbw is None and taper is None:
            if has_reflector:
                taper = -10.0
                assumptions.append(f"feed: no hpbw_{plane}/edge_taper_{plane}; assumed "
                                   f"the classic -10 dB edge taper")
            else:
                hpbw = 60.0
                assumptions.append(f"feed: no hpbw_{plane}; assumed 60 deg")
        if taper is not None and hpbw is None and not has_reflector:
            raise HardwareConfigError(
                f"feed.edge_taper_{plane} needs a reflector to define the rim; "
                f"with no reflector give hpbw_{plane} instead")
        if taper is not None and taper >= 0:
            raise HardwareConfigError("edge taper is a level below the peak: use a "
                                      "negative number of dB, e.g. -10")
        if hpbw is not None and not 0.0 < hpbw < 180.0:
            raise HardwareConfigError(f"feed.hpbw_{plane} must be in (0, 180) deg")
        spec[plane] = (hpbw, taper)
    vswr = ps.number("vswr", "", 1.0)
    if vswr < 1.0:
        raise HardwareConfigError("feed.vswr must be >= 1")
    back = ps.number("back_radiation_fraction", "", 0.0)
    if not 0.0 <= back < 1.0:
        raise HardwareConfigError("feed.back_radiation_fraction must be in [0, 1)")
    return FeedSpec(
        kind=ps.text("type", "unspecified"),
        hpbw_e_deg=spec["e"][0], hpbw_h_deg=spec["h"][0],
        edge_taper_e_db=spec["e"][1], edge_taper_h_db=spec["h"][1],
        waveguide_diameter_m=ps.number("diameter", "m"),
        freq_min_hz=ps.number("freq_min", "Hz"), freq_max_hz=ps.number("freq_max", "Hz"),
        rolloff_db_per_octave=ps.number("rolloff", "dB", 40.0),
        ohmic_loss_db=ps.number("ohmic_loss", "dB", 0.0), vswr=vswr, back_fraction=back,
        e_plane_azimuth_deg=ps.number("e_plane_azimuth", "deg", 0.0),
        t_phys_k=ps.number("physical_temperature", "K", default_t_phys_k),
    )


def _reflector(ps: ParamSet, assumptions: list[str]) -> ReflectorSpec:
    d = ps.number("diameter", "m")
    if d is None:
        raise HardwareConfigError(
            "reflector: diameter is required (or set the reflector's "
            "'enabled: false' to model the feed on its own)")
    fd = ps.number("focal_ratio", "")
    if fd is None:
        fd = 0.4
        assumptions.append("reflector: focal_ratio not given; assumed f/D = 0.4")
    return ReflectorSpec(d, fd, ps.number("surface_rms_error", "m", 0.0),
                         ps.number("blockage_fraction", "", 0.0))


def load_hardware(path: str | Path | None = None, *, data: dict | None = None,
                  order: Sequence[str] | None = None) -> Hardware:
    """Read ``hardware_parameters.yaml``.  ``order`` (component ids) overrides
    every ``chain_position``."""
    p = Path(path) if path is not None else None
    if data is None:
        if p is None:
            raise HardwareConfigError("give a path or a data mapping")
        data = load_yaml(p)
    base = p.parent if p is not None else Path.cwd()
    assumptions: list[str] = []
    warnings: list[str] = []

    an = ParamSet.from_yaml("analysis", data.get("analysis"), base)
    center = an.number("center_freq", "Hz", 1_420_405_751.768)
    sweep = (an.number("sweep_min", "Hz", center - 40e6),
             an.number("sweep_max", "Hz", center + 40e6),
             int(an.number("sweep_points", "", 1601)))
    if not sweep[0] < center < sweep[1]:
        raise HardwareConfigError("analysis: sweep_min < center_freq < sweep_max is required")

    env_ps = ParamSet.from_yaml("environment", data.get("environment"), base)
    env = Environment(
        latitude_deg=env_ps.number("site_latitude", "deg", 36.003147),
        ambient_k=env_ps.number("ambient_temperature", "K", 290.0),
        ground_k=env_ps.number("ground_temperature", "K", 290.0),
        cmb_k=env_ps.number("cmb_temperature", "K", 2.725),
        t408_k=env_ps.number("galactic_temperature_408", "K", 20.0),
        spectral_index=env_ps.number("galactic_spectral_index", "", -2.7),
        atmosphere_k=env_ps.number("atmosphere_temperature", "K", 2.0),
        horizon_deg=env_ps.number("horizon_elevation", "deg", 0.0),
    )
    placeholders = [f"environment.{n}" for n in env_ps.placeholders()]

    structure: dict[str, ParamSet] = {}
    chain: list[ComponentSpec] = []
    disabled: list[str] = []
    seen: set[str] = set()
    for i, raw in enumerate(data.get("components") or []):
        if not isinstance(raw, dict):
            raise HardwareConfigError(f"components[{i}] must be a mapping")
        cid = str(raw.get("id") or f"component_{i}")
        if cid in seen:
            raise HardwareConfigError(f"two components are called {cid!r}")
        seen.add(cid)
        ctype = str(raw.get("type") or "").strip().lower()
        enabled = raw.get("enabled", True) is not False
        if ctype not in CHAIN_TYPES | STRUCTURE_TYPES:
            raise HardwareConfigError(
                f"component {cid!r}: type {ctype or '(missing)'!r} is not one of "
                f"{', '.join(sorted(CHAIN_TYPES | STRUCTURE_TYPES))}")
        if not enabled:
            disabled.append(cid)
            continue
        if ctype in STRUCTURE_TYPES:
            if ctype in structure:
                raise HardwareConfigError(f"only one {ctype} is allowed")
            structure[ctype] = ParamSet.from_yaml(cid, raw.get("parameters"), base)
            placeholders += [f"{cid}.{n}" for n in structure[ctype].placeholders()]
            continue
        comp = _component(raw, index=i, base_dir=base, default_t_phys_k=env.ambient_k,
                          center_hz=center)
        chain.append(comp)
        placeholders += [f"{cid}.{n}" for n in comp.params.placeholders()]

    if order:
        by_id = {c.id: c for c in chain}
        missing = [o for o in order if o not in by_id]
        if missing:
            raise HardwareConfigError(
                f"--order names {', '.join(missing)}, which "
                f"{'is' if len(missing) == 1 else 'are'} not enabled chain "
                f"components (have: {', '.join(by_id)})")
        left_out = [c.id for c in chain if c.id not in order]
        if left_out:
            warnings.append(f"--order left out {', '.join(left_out)}; "
                            f"they are dropped from this run")
        chain = [by_id[o] for o in order]
        for pos, c in enumerate(chain, 1):
            c.chain_position = float(pos)
    else:
        positions = [c.chain_position for c in chain]
        dup = sorted({x for x in positions if positions.count(x) > 1})
        if dup:
            raise HardwareConfigError(
                f"chain_position {', '.join(str(int(d)) for d in dup)} is "
                f"used by more than one component")
        chain.sort(key=lambda c: c.chain_position)
    if not chain:
        raise HardwareConfigError("no enabled RF-chain components")
    sdr_idx = [i for i, c in enumerate(chain) if c.type == "sdr"]
    if not sdr_idx:
        warnings.append("no SDR in the chain: T_rec ignores the digitiser's noise")
    elif sdr_idx[-1] != len(chain) - 1:
        warnings.append(f"{chain[sdr_idx[-1]].id} is not the last component; anything "
                        f"after the SDR is already digital and has no analog effect")

    reflector = _reflector(structure["reflector"], assumptions) if "reflector" in structure else None
    if "feed" not in structure:
        raise HardwareConfigError("a feed component is required (type: feed)")
    feed = _feed(structure["feed"], reflector is not None, env.ambient_k, assumptions)
    mount = structure.get("mount", ParamSet("mount"))

    sdr = chain[sdr_idx[-1]] if sdr_idx else None
    fft = sdr.params.number("fft_size", "") if sdr else None
    chan = sdr.params.number("channel_width", "Hz") if sdr else None
    if sdr is None or (fft is None and chan is None):
        chan = 5e3
        assumptions.append("no SDR fft_size/channel_width: assumed 5 kHz channels")
    n_pol = int(sdr.params.number("n_pol", "", 1)) if sdr else 1
    if n_pol not in (1, 2):
        raise HardwareConfigError("n_pol must be 1 or 2")

    return Hardware(
        name=str((data.get("meta") or {}).get("name") or "unnamed telescope"),
        path=p, center_hz=center, sweep_hz=sweep, environment=env,
        reflector=reflector, feed=feed,
        pointing_alt_deg=mount.number("pointing_alt", "deg", 90.0),
        pointing_az_deg=mount.number("pointing_az", "deg", 0.0),
        mount_type=mount.text("type", "fixed"), chain=chain, disabled=disabled,
        fft_size=fft, channel_width_hz=chan, n_pol=n_pol,
        k_s=sdr.params.number("radiometer_factor", "", 1.0) if sdr else 1.0,
        assumptions=assumptions, placeholders=placeholders, warnings=warnings,
    )


# ---------------------------------------------------------------------------
# observing goals
# ---------------------------------------------------------------------------
@dataclass
class Goals:
    snr_target: float
    integrations: list[Any]          # "transit", "transit x N", or a time entry
    line_width_kms: float
    base_dir: Path


def load_goals(path: Path) -> Goals:
    data = load_yaml(path)
    scalars = {k: v for k, v in data.items() if k != "integrations"}
    ps = ParamSet.from_yaml("goals", scalars, path.parent)
    return Goals(snr_target=ps.number("snr_target", "", 3.0),
                 integrations=data.get("integrations") or ["transit"],
                 line_width_kms=ps.number("hi_line_width", "km/s", 200.0),
                 base_dir=path.parent)


def integration_entry(item: Any, base_dir: Path) -> tuple[str, float | None, float]:
    """An ``integrations`` entry -> (label, seconds or None, number of transits).

    ``None`` seconds means "a multiple of the transit time", which only the
    performance model knows."""
    if isinstance(item, str) and item.strip().lower().startswith("transit"):
        parts = item.split("x")
        n = float(parts[1]) if len(parts) == 2 else 1.0
        return ("1 transit" if n == 1 else f"{n:g} transits"), None, n
    p = parse_param("integration", item, where="goals.integrations", base_dir=base_dir)
    if p.unit != "s":
        raise HardwareConfigError(f"goals.integrations: {item!r} is not a time")
    v = float(p.value)
    return (f"{v/3600:g} h" if v >= 3600 else f"{v:g} s"), v, 0.0
