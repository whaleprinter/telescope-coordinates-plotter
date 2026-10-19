"""The RF chain: per-component gain and noise vs frequency, and the cascade.

Every chain component is a two-port with two curves:

* ``gain_db(f)`` -- available gain (a loss is a negative gain), and
* ``noise_temp_k(f)`` -- input-referred effective noise temperature.

Active parts (amplifier, SDR) get T from their noise figure,
``T = 290 K * (10^(NF/10) - 1)``.  Passive lossy parts (cable, filter,
attenuator) get it from their loss and physical temperature,
``T = T_phys * (L - 1)`` with ``L = 10^(loss_dB/10)``.  That is the same
expression whether the loss is a cable's resistance or a filter's skirt.

:func:`cascade` then applies Friis in whatever order ``chain_position`` puts
the parts, which is what lets you move components around and watch what
happens to the receiver temperature.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .params import HardwareConfigError, ParamSet

T0_K = 290.0                     # IEEE reference temperature for noise figure

CHAIN_TYPES = {"amplifier", "filter", "cable", "attenuator", "sdr"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def db_to_lin(db):
    return 10.0 ** (np.asarray(db, dtype=float) / 10.0)


def lin_to_db(lin):
    return 10.0 * np.log10(np.maximum(np.asarray(lin, dtype=float), 1e-300))


def nf_db_to_temp_k(nf_db):
    """Noise figure (dB) -> effective noise temperature (K)."""
    return T0_K * (db_to_lin(nf_db) - 1.0)


def passive_temp_k(gain_db, t_phys_k):
    """T = T_phys (L - 1) for a passive part with available gain ``gain_db``."""
    loss_lin = 1.0 / db_to_lin(np.minimum(gain_db, 0.0))
    return t_phys_k * (loss_lin - 1.0)


def edge_rolloff_db(f, lo_hz, hi_hz, db_per_octave):
    """Attenuation (positive dB) outside a datasheet's [lo, hi] spec range.

    Zero inside the range -- the datasheet gain already describes the
    in-band behaviour, so nothing is subtracted there -- and a constant
    dB/octave slope beyond each edge.  A deliberate simplification: replace
    it with a ``curve`` or an ``sparameters`` file once you have real data.
    """
    f = np.asarray(f, dtype=float)
    att = np.zeros_like(f)
    if lo_hz:
        att += np.where(f < lo_hz, db_per_octave * np.log2(lo_hz / np.maximum(f, 1.0)), 0.0)
    if hi_hz:
        att += np.where(f > hi_hz, db_per_octave * np.log2(f / hi_hz), 0.0)
    return att


def bandpass_attenuation_db(f, center_hz, bandwidth_hz, order, kind="butterworth",
                            ripple_db=0.5):
    """Selectivity of an ideal bandpass prototype (positive dB, 0 at centre).

    Lowpass-to-bandpass mapping ``W = (f/f0 - f0/f) f0/B`` puts the band
    edges at ``W = +/-1``.  Butterworth: ``1 + W^2n`` (3 dB at the edges).
    Chebyshev: ``1 + eps^2 T_n(W)^2`` (``ripple_db`` at the edges, so
    ``bandwidth`` is the ripple bandwidth).
    """
    f = np.maximum(np.asarray(f, dtype=float), 1.0)
    w = (f / center_hz - center_hz / f) * center_hz / bandwidth_hz
    n = int(order)
    if kind == "butterworth":
        return 10.0 * np.log10(1.0 + np.abs(w) ** (2 * n))
    if kind == "chebyshev":
        eps2 = 10.0 ** (ripple_db / 10.0) - 1.0
        aw = np.abs(w)
        tn = np.where(aw <= 1.0, np.cos(n * np.arccos(np.minimum(aw, 1.0))),
                      np.cosh(n * np.arccosh(np.maximum(aw, 1.0))))
        return 10.0 * np.log10(1.0 + eps2 * tn ** 2)
    raise HardwareConfigError(f"unknown filter response {kind!r} "
                              f"(use butterworth or chebyshev)")


def read_touchstone_s21(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """S21 magnitude (dB) vs frequency (Hz) from a 2-port Touchstone file.

    Handles the ``# <Hz|kHz|MHz|GHz> S <DB|MA|RI> R <z>`` option line that
    manufacturers' .s2p downloads use.  Only S21 is read: that is the gain.
    """
    if not path.exists():
        raise HardwareConfigError(f"Touchstone file not found: {path}")
    f_mult, fmt = 1e9, "MA"                 # Touchstone defaults
    data: list[float] = []
    for line in path.read_text().splitlines():
        line = line.split("!", 1)[0].strip()
        if not line:
            continue
        if line.startswith("#"):
            tokens = line[1:].upper().split()
            for tok in tokens:
                if tok in {"HZ", "KHZ", "MHZ", "GHZ"}:
                    f_mult = {"HZ": 1.0, "KHZ": 1e3, "MHZ": 1e6, "GHZ": 1e9}[tok]
                elif tok in {"DB", "MA", "RI"}:
                    fmt = tok
            continue
        data.extend(float(x) for x in line.split())
    if len(data) % 9:
        raise HardwareConfigError(f"{path}: not a 2-port Touchstone file")
    arr = np.asarray(data).reshape(-1, 9)
    f = arr[:, 0] * f_mult
    a, b = arr[:, 3], arr[:, 4]             # S21 pair
    if fmt == "DB":
        s21_db = a
    elif fmt == "MA":
        s21_db = 20.0 * np.log10(np.maximum(a, 1e-30))
    else:
        s21_db = 20.0 * np.log10(np.maximum(np.hypot(a, b), 1e-30))
    return f, s21_db


# ---------------------------------------------------------------------------
# components
# ---------------------------------------------------------------------------
@dataclass
class Component:
    """One RF-chain part, as read from the hardware file."""

    id: str
    type: str
    params: ParamSet
    chain_position: float
    t_phys_k: float
    model: str = ""
    description: str = ""
    base_dir: Path = field(default_factory=Path)
    sparameters: str | None = None
    center_hz: float = 1.42e9               # analysis centre (SDR default tune)
    _s21: tuple[np.ndarray, np.ndarray] | None = field(default=None, repr=False)

    # ------------------------------------------------------------- band edges
    def band_hz(self) -> tuple[float | None, float | None]:
        """Nominal band edges used by the "intersect every band" rule."""
        p = self.params
        if self.type == "filter":
            c = p.require("center_freq", "Hz").scalar()
            bw = p.require("bandwidth", "Hz").scalar()
            return c - bw / 2.0, c + bw / 2.0
        if self.type == "sdr":
            c = p.number("tuned_freq", "Hz", self.center_hz)
            bw = self.sdr_bandwidth_hz()
            return c - bw / 2.0, c + bw / 2.0
        return p.number("freq_min", "Hz"), p.number("freq_max", "Hz")

    def sdr_bandwidth_hz(self) -> float:
        p = self.params
        bw = p.number("if_bandwidth", "Hz")
        if bw is None:
            bw = p.require("sample_rate", "Hz").scalar()
        return bw

    # ------------------------------------------------------------------ gain
    def gain_db(self, f) -> np.ndarray:
        f = np.asarray(f, dtype=float)
        p, t = self.params, self.type
        if t == "amplifier":
            g = self._s21_at(f) if self.sparameters else p.require("gain", "dB").at(f)
            lo, hi = p.number("freq_min", "Hz"), p.number("freq_max", "Hz")
            slope = p.number("rolloff", "dB", 12.0)
            return g - edge_rolloff_db(f, lo, hi, slope)
        if t == "filter":
            if self.sparameters:
                return self._s21_at(f)
            il = p.get("insertion_loss")
            il_arr = p.require("insertion_loss", "dB").at(f) if il else 0.0
            c = p.require("center_freq", "Hz").scalar()
            bw = p.require("bandwidth", "Hz").scalar()
            att = bandpass_attenuation_db(
                f, c, bw, p.number("order", "", 5.0),
                kind=p.text("response", "butterworth").lower(),
                ripple_db=p.number("ripple", "dB", 0.5))
            rej = p.number("out_of_band_rejection", "dB")
            if rej is not None:
                att = np.minimum(att, rej)
            return -(il_arr + att)
        if t in {"cable", "attenuator"}:
            return -self.loss_db(f)
        if t == "sdr":
            # The SDR digitises; its "gain" here is only its selectivity, which
            # sets the band actually integrated.  Its noise is in noise_temp_k.
            c = p.number("tuned_freq", "Hz", self.center_hz)
            att = bandpass_attenuation_db(f, c, self.sdr_bandwidth_hz(),
                                          p.number("order", "", 12.0))
            return -np.minimum(att, p.number("stopband", "dB", 80.0))
        raise HardwareConfigError(f"{self.id}: unknown chain type {t!r}")

    def loss_db(self, f) -> np.ndarray:
        """Cable / attenuator loss (positive dB)."""
        p = self.params
        f = np.asarray(f, dtype=float)
        loss = p.get("loss")
        if loss is not None:
            if loss.curve_hz is not None or self.type == "attenuator":
                return loss.at(f)
            ref = p.number("loss_ref_freq", "Hz")
            if ref is None:
                return loss.at(f)
            return loss.scalar() * np.sqrt(f / ref)
        per_m = p.get("loss_per_length")
        if per_m is None:
            raise HardwareConfigError(
                f"{self.id}: give either 'loss' (dB) or 'loss_per_length' + 'length'")
        length = p.require("length", "m").scalar()
        if per_m.curve_hz is not None:
            return per_m.at(f) * length
        ref = p.require("loss_ref_freq", "Hz").scalar()
        # Coax loss is dominated by conductor (skin-effect) loss, ~ sqrt(f).
        return per_m.scalar() * length * np.sqrt(f / ref)

    # ----------------------------------------------------------------- noise
    def noise_temp_k(self, f) -> np.ndarray:
        f = np.asarray(f, dtype=float)
        if self.type in {"amplifier", "sdr"}:
            return nf_db_to_temp_k(self.params.require("noise_figure", "dB").at(f))
        return passive_temp_k(self.gain_db(f), self.t_phys_k)

    # ---------------------------------------------------------------- misc
    def _s21_at(self, f):
        if self._s21 is None:
            self._s21 = read_touchstone_s21(self.base_dir / self.sparameters)
        fs, s = self._s21
        return np.interp(f, fs, s)

    @property
    def label(self) -> str:
        return f"{self.id}" + (f" ({self.model})" if self.model else "")


# ---------------------------------------------------------------------------
# the cascade
# ---------------------------------------------------------------------------
@dataclass
class StageResult:
    component: Component
    gain_db: np.ndarray            # this stage
    noise_temp_k: np.ndarray       # this stage, at its own input
    gain_before_lin: np.ndarray    # product of all earlier gains
    contribution_k: np.ndarray     # T_i / (G_1 ... G_{i-1}): its share of T_rec
    cumulative_gain_db: np.ndarray # gain from chain input to this stage's output


@dataclass
class CascadeResult:
    freq_hz: np.ndarray
    stages: list[StageResult]

    @property
    def t_rec_k(self) -> np.ndarray:
        return sum(s.contribution_k for s in self.stages)

    @property
    def gain_to_sdr_db(self) -> np.ndarray:
        """Total analog gain: chain input to the SDR's input."""
        total = np.zeros_like(self.freq_hz)
        for s in self.stages:
            if s.component.type == "sdr":
                break
            total = total + s.gain_db
        return total

    @property
    def response_db(self) -> np.ndarray:
        """Product of every stage's response, SDR selectivity included."""
        return sum(s.gain_db for s in self.stages)


def cascade(components: list[Component], freq_hz) -> CascadeResult:
    """Friis: T_rec = T1 + T2/G1 + T3/(G1 G2) + ... in the given order."""
    f = np.asarray(freq_hz, dtype=float)
    g_before = np.ones_like(f)
    cum_db = np.zeros_like(f)
    out = []
    for comp in components:
        g = comp.gain_db(f)
        t = comp.noise_temp_k(f)
        contrib = t / g_before
        cum_db = cum_db + g
        out.append(StageResult(comp, g, t, g_before.copy(), contrib, cum_db.copy()))
        g_before = g_before * db_to_lin(g)
    return CascadeResult(f, out)


def nominal_passband(components: list[Component],
                     extra: list[tuple[str, float | None, float | None]] = ()
                     ) -> tuple[float, float, str, str]:
    """Intersect every part's band: (lo, hi, who set lo, who set hi).

    Lower edge = the highest lower edge; upper edge = the lowest upper edge.
    """
    lo, hi, lo_by, hi_by = -math.inf, math.inf, "", ""
    entries = [(c.id, *c.band_hz()) for c in components] + list(extra)
    for name, a, b in entries:
        if a is not None and a > lo:
            lo, lo_by = a, name
        if b is not None and b < hi:
            hi, hi_by = b, name
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise HardwareConfigError("no component declares band edges, so the "
                                  "passband is undefined")
    if lo >= hi:
        raise HardwareConfigError(
            f"the chain has no common passband: {lo_by} starts at "
            f"{lo/1e6:.2f} MHz but {hi_by} ends at {hi/1e6:.2f} MHz")
    return lo, hi, lo_by, hi_by


@dataclass
class Passband:
    nominal_lo_hz: float
    nominal_hi_hz: float
    nominal_lo_by: str
    nominal_hi_by: str
    lo_3db_hz: float
    hi_3db_hz: float
    noise_bandwidth_hz: float
    ripple_db: float
    peak_gain_db: float

    @property
    def bw_3db_hz(self) -> float:
        return self.hi_3db_hz - self.lo_3db_hz


def measure_passband(freq_hz: np.ndarray, response_db: np.ndarray,
                     center_hz: float, nominal: tuple) -> Passband:
    """-3 dB points, noise-equivalent bandwidth and ripple of ``response_db``.

    The -3 dB band is the contiguous region around ``center_hz`` within 3 dB
    of the peak.  Noise bandwidth is the width of the rectangle with the
    same area and the same peak: ``B_N = integral(G df) / G_peak``.  It is
    the right bandwidth for the radiometer equation; the -3 dB width is not.
    """
    f, r = np.asarray(freq_hz), np.asarray(response_db)
    i0 = int(np.argmin(np.abs(f - center_hz)))
    peak = float(r.max())
    above = r >= peak - 3.0
    if not above[i0]:
        raise HardwareConfigError(
            f"the analysis centre {center_hz/1e6:.3f} MHz is more than 3 dB "
            f"down the chain's response -- is a filter or the SDR tuned elsewhere?")
    lo_i = i0
    while lo_i > 0 and above[lo_i - 1]:
        lo_i -= 1
    hi_i = i0
    while hi_i < len(f) - 1 and above[hi_i + 1]:
        hi_i += 1

    def _cross(i_in, i_out):
        # linear interpolation of the -3 dB crossing between two grid points
        if i_out < 0 or i_out >= len(f):
            return float(f[i_in])
        y0, y1 = r[i_in] - (peak - 3.0), r[i_out] - (peak - 3.0)
        t = y0 / (y0 - y1) if y0 != y1 else 0.0
        return float(f[i_in] + t * (f[i_out] - f[i_in]))

    lo3, hi3 = _cross(lo_i, lo_i - 1), _cross(hi_i, hi_i + 1)
    g = db_to_lin(r - peak)
    bn = float(np.trapezoid(g, f)) if hasattr(np, "trapezoid") else float(np.trapz(g, f))
    # Ripple over the central 80 % of the -3 dB band: across the whole band
    # it would just be the 3 dB the edges fall by, by definition.
    margin = 0.1 * (hi3 - lo3)
    band = r[(f >= lo3 + margin) & (f <= hi3 - margin)]
    if len(band) == 0:
        band = r[i0:i0 + 1]
    return Passband(nominal[0], nominal[1], nominal[2], nominal[3], lo3, hi3, bn,
                    float(band.max() - band.min()), peak)


def build_component(raw: dict, *, index: int, base_dir: Path,
                    default_t_phys_k: float, center_hz: float) -> Component:
    cid = str(raw.get("id") or f"component_{index}")
    ctype = str(raw.get("type") or "").strip().lower()
    if ctype not in CHAIN_TYPES:
        raise HardwareConfigError(
            f"component {cid!r}: type {ctype or '(missing)'!r} is not one of "
            f"{', '.join(sorted(CHAIN_TYPES))}")
    if raw.get("chain_position") is None:
        raise HardwareConfigError(f"component {cid!r}: chain_position is required")
    params = ParamSet.from_yaml(cid, raw.get("parameters"), base_dir)
    t_phys = params.number("physical_temperature", "K", default_t_phys_k)
    return Component(
        id=cid, type=ctype, params=params,
        chain_position=float(raw["chain_position"]), t_phys_k=t_phys,
        model=str(raw.get("model") or ""), description=str(raw.get("description") or ""),
        base_dir=base_dir, sparameters=raw.get("sparameters"), center_hz=center_hz,
    )
