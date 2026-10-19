"""Reading ``{value, unit, status, notes}`` parameters out of the hardware YAML.

Every number in ``hardware_parameters.yaml`` is written the way a datasheet
gives it -- ``{value: 0.7, unit: GHz}`` -- and converted here, once, to the
canonical unit the maths uses (Hz, m, s, K, deg, dB).  Nothing downstream ever
sees a raw YAML number, so a ``MHz`` / ``GHz`` slip cannot silently propagate.

A parameter may also vary with frequency.  Instead of ``value`` give a
``curve`` (a list of ``[frequency, value]`` pairs, frequency in ``freq_unit``)
or a ``file`` (two-column CSV of the same).  The curve is linearly
interpolated and held flat beyond its ends.
"""
from __future__ import annotations

import csv
import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


class HardwareConfigError(ValueError):
    """The hardware file is malformed; the message says where and how."""


# unit (lower-case) -> (canonical unit, multiplier)
_UNITS: dict[str, tuple[str, float]] = {
    "hz": ("Hz", 1.0), "khz": ("Hz", 1e3), "mhz": ("Hz", 1e6), "ghz": ("Hz", 1e9),
    "sps": ("Hz", 1.0), "ksps": ("Hz", 1e3), "msps": ("Hz", 1e6),
    "ms/s": ("Hz", 1e6),
    "m": ("m", 1.0), "cm": ("m", 1e-2), "mm": ("m", 1e-3),
    "ft": ("m", 0.3048), "in": ("m", 0.0254),
    "s": ("s", 1.0), "min": ("s", 60.0), "h": ("s", 3600.0), "hr": ("s", 3600.0),
    "k": ("K", 1.0),
    "deg": ("deg", 1.0), "arcmin": ("deg", 1.0 / 60.0),
    "rad": ("deg", 180.0 / np.pi),
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


@dataclass
class Param:
    """One parameter after unit conversion.

    ``value`` is a float in ``unit`` (canonical) or a string for descriptive
    entries; ``curve_hz`` / ``curve_val`` hold a frequency-dependent version.
    """

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
        """The value at ``freq_hz`` (array-friendly); curves are interpolated."""
        f = np.asarray(freq_hz, dtype=float)
        if self.curve_hz is not None:
            return np.interp(f, self.curve_hz, self.curve_val)
        if self.value is None:
            raise HardwareConfigError(f"parameter {self.name!r} has no value")
        return np.full_like(f, float(self.value), dtype=float)

    def scalar(self, freq_hz: float | None = None) -> float:
        """A single number: the value, or the curve at ``freq_hz``."""
        if self.curve_hz is not None:
            if freq_hz is None:
                raise HardwareConfigError(
                    f"{self.name!r} is a frequency curve; a frequency is needed")
            return float(self.at(freq_hz))
        if self.value is None:
            raise HardwareConfigError(f"parameter {self.name!r} has no value")
        return float(self.value)


def convert(value: float, unit: str, *, where: str) -> tuple[float, str]:
    """Convert ``value`` in ``unit`` to the canonical unit."""
    u = (unit or "").strip().lower()
    if u in _CELSIUS:
        return float(value) + 273.15, "K"
    if u not in _UNITS:
        close = difflib.get_close_matches(u, list(_UNITS), n=3)
        hint = f" (did you mean {', '.join(close)}?)" if close else ""
        raise HardwareConfigError(f"{where}: unknown unit {unit!r}{hint}")
    canon, mult = _UNITS[u]
    return float(value) * mult, canon


def parse_param(name: str, raw: Any, *, where: str, base_dir: Path) -> Param:
    """Turn one YAML entry into a :class:`Param`.

    Accepts the full ``{value, unit, ...}`` mapping, or a bare scalar for
    unit-less descriptive entries (``type: cantenna``).
    """
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
    p = Param(name=name, status=status, notes=str(raw.get("notes") or ""),
              source_unit=unit)

    value = raw.get("value")
    if isinstance(value, str):
        # Descriptive strings stay strings; numeric strings (YAML 1.1 reads a
        # bare 2.5e6 as text) become numbers.
        try:
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
            raise HardwareConfigError(
                f"{loc}: curve needs at least two [frequency, value] pairs")
        f_unit = str(raw.get("freq_unit") or "MHz")
        f_mult, _ = convert(1.0, f_unit, where=f"{loc}.freq_unit")
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
                continue                    # header line
    return rows


@dataclass
class ParamSet:
    """The ``parameters:`` block of one component, with typed accessors."""

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
        if p.unit != unit and not (unit == "dB" and p.unit in {"dBi"}):
            raise HardwareConfigError(
                f"{self.owner}.{p.name}: expected a quantity in {unit or 'no unit'}, "
                f"got {p.source_unit!r}")

    def placeholders(self) -> list[str]:
        return [n for n, p in self.params.items()
                if p.is_set and p.status in {"placeholder", "estimate"}]
