"""Antenna side: feed pattern, reflector efficiencies and the 3-D beam.

The feed is modelled as the standard ``cos^q`` power pattern, with separate
exponents in its E- and H-planes,

    G_f(theta, phi) = C * [cos^qE(theta) cos^2(phi) + cos^qH(theta) sin^2(phi)]

(``phi`` measured from the E-plane, ``C`` normalising to 4 pi over the
forward hemisphere).  ``qE != qH`` is what makes a real feed -- a cantenna,
a dipole over a ground plane -- asymmetric, and that asymmetry is carried
all the way through to an elliptical beam on the sky.

With a reflector, the feed illuminates a paraboloid and the beam is the
Fourier (Hankel) transform of the resulting aperture field, computed
separately in each principal plane.  With no reflector the feed pattern
*is* the beam.

References: Balanis, *Antenna Theory* 4th ed., sec. 15.4 (aperture,
spillover and taper efficiency of a paraboloid); Ruze (1966) for surface
error; Kildal, *Foundations of Antenna Engineering*, ch. 2.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from scipy.special import j0

from ..constants import C_LIGHT
from .params import HardwareConfigError, ParamSet
from .rf import edge_rolloff_db

#: Cut-off of the TE11 and TM01 modes of a circular waveguide:
#: f_c = x' c / (pi d), with x'11 = 1.8412 and x01 = 2.4048.
TE11_X, TM01_X = 1.8412, 2.4048


def q_from_hpbw(hpbw_deg: float) -> float:
    """cos^q power pattern whose half-power beamwidth is ``hpbw_deg``."""
    half = math.radians(hpbw_deg) / 2.0
    if not 0.0 < half < math.pi / 2:
        raise HardwareConfigError(f"feed HPBW {hpbw_deg} deg must be in (0, 180)")
    return math.log(0.5) / math.log(math.cos(half))


def q_from_edge_taper(taper_db: float, half_angle_rad: float) -> float:
    """cos^q pattern that is ``taper_db`` (negative) down at the dish rim."""
    if taper_db >= 0:
        raise HardwareConfigError("edge taper is a level below the peak: use a "
                                  "negative number of dB, e.g. -10")
    return taper_db / (10.0 * math.log10(math.cos(half_angle_rad)))


@dataclass
class FeedModel:
    q_e: float
    q_h: float
    ohmic_loss_db: float
    vswr: float
    back_fraction: float
    band_lo_hz: float | None
    band_hi_hz: float | None
    rolloff_db_per_octave: float
    e_plane_azimuth_deg: float
    t_phys_k: float
    kind: str
    notes: list[str] = field(default_factory=list)

    @property
    def norm(self) -> float:
        """C such that the pattern integrates to 4 pi over the hemisphere."""
        return 4.0 / (1.0 / (self.q_e + 1.0) + 1.0 / (self.q_h + 1.0))

    @property
    def eta_ohmic(self) -> float:
        return 10.0 ** (-self.ohmic_loss_db / 10.0)

    @property
    def eta_mismatch(self) -> float:
        g = (self.vswr - 1.0) / (self.vswr + 1.0)
        return 1.0 - g * g

    def plane_power(self, theta_rad, q):
        c = np.cos(np.clip(theta_rad, 0.0, math.pi / 2))
        return np.where(np.asarray(theta_rad) <= math.pi / 2, c ** q, 0.0)

    def response_db(self, f_hz):
        """Feed's own band (e.g. waveguide cut-off) as a gain curve, 0 dB in band."""
        return -edge_rolloff_db(f_hz, self.band_lo_hz, self.band_hi_hz,
                                self.rolloff_db_per_octave)


@dataclass
class Reflector:
    diameter_m: float
    focal_ratio: float
    surface_rms_m: float
    blockage_fraction: float

    @property
    def focal_length_m(self) -> float:
        return self.focal_ratio * self.diameter_m

    @property
    def half_angle_rad(self) -> float:
        """Angle the dish rim subtends at the focus: tan(t0/2) = 1/(4 f/D)."""
        return 2.0 * math.atan(1.0 / (4.0 * self.focal_ratio))

    @property
    def geometric_area_m2(self) -> float:
        return math.pi * (self.diameter_m / 2.0) ** 2


def build_feed(ps: ParamSet, reflector: Reflector | None, default_t_phys_k: float,
               assumptions: list[str]) -> FeedModel:
    """Feed pattern from HPBW, edge taper, or a sensible stated default."""
    kind = ps.text("type", "unspecified")
    qs = []
    for plane in ("e", "h"):
        hpbw = ps.number(f"hpbw_{plane}", "deg")
        taper = ps.number(f"edge_taper_{plane}", "dB")
        if hpbw is not None:
            qs.append(q_from_hpbw(hpbw))
        elif taper is not None:
            if reflector is None:
                raise HardwareConfigError(
                    f"feed.edge_taper_{plane} needs a reflector to define the rim; "
                    f"with no reflector give hpbw_{plane} instead")
            qs.append(q_from_edge_taper(taper, reflector.half_angle_rad))
        elif reflector is not None:
            qs.append(q_from_edge_taper(-10.0, reflector.half_angle_rad))
            assumptions.append(f"feed: no hpbw_{plane}/edge_taper_{plane}; assumed "
                               f"the classic -10 dB edge taper")
        else:
            qs.append(q_from_hpbw(60.0))
            assumptions.append(f"feed: no hpbw_{plane}; assumed 60 deg")

    lo, hi = ps.number("freq_min", "Hz"), ps.number("freq_max", "Hz")
    d = ps.number("diameter", "m")
    if d is not None:
        # A circular-waveguide feed (cantenna) propagates between the TE11 and
        # TM01 cut-offs; outside that it is either evanescent or multimode.
        lo = lo if lo is not None else TE11_X * C_LIGHT / (math.pi * d)
        hi = hi if hi is not None else TM01_X * C_LIGHT / (math.pi * d)

    vswr = ps.number("vswr", "", 1.0)
    if vswr < 1.0:
        raise HardwareConfigError("feed.vswr must be >= 1")
    back = ps.number("back_radiation_fraction", "", 0.0)
    if not 0.0 <= back < 1.0:
        raise HardwareConfigError("feed.back_radiation_fraction must be in [0, 1)")
    return FeedModel(
        q_e=qs[0], q_h=qs[1],
        ohmic_loss_db=ps.number("ohmic_loss", "dB", 0.0),
        vswr=vswr, back_fraction=back, band_lo_hz=lo, band_hi_hz=hi,
        rolloff_db_per_octave=ps.number("rolloff", "dB", 40.0),
        e_plane_azimuth_deg=ps.number("e_plane_azimuth", "deg", 0.0),
        t_phys_k=ps.number("physical_temperature", "K", default_t_phys_k),
        kind=kind,
    )


def build_reflector(ps: ParamSet | None, assumptions: list[str]) -> Reflector | None:
    if ps is None:
        return None
    d = ps.number("diameter", "m")
    if d is None:
        raise HardwareConfigError(
            "reflector: diameter is required (or set the reflector's "
            "'enabled: false' to model the feed on its own)")
    fd = ps.number("focal_ratio", "")
    if fd is None:
        fd = 0.4
        assumptions.append("reflector: focal_ratio not given; assumed f/D = 0.4")
    return Reflector(
        diameter_m=d, focal_ratio=fd,
        surface_rms_m=ps.number("surface_rms_error", "m", 0.0),
        blockage_fraction=ps.number("blockage_fraction", "", 0.0),
    )


# ---------------------------------------------------------------------------
# efficiencies and the beam
# ---------------------------------------------------------------------------
@dataclass
class Optics:
    freq_hz: float
    feed: FeedModel
    reflector: Reflector | None
    horizon_deg: float

    @property
    def wavelength_m(self) -> float:
        return C_LIGHT / self.freq_hz

    # --------------------------------------------------- reflector-only terms
    def _aperture_field(self, n_rho=600, n_phi=181):
        r = self.reflector
        a = r.diameter_m / 2.0
        rho = np.linspace(0.0, a, n_rho)
        phi = np.linspace(0.0, 2 * math.pi, n_phi)
        th = 2.0 * np.arctan(rho / (2.0 * r.focal_length_m))
        P, R = np.meshgrid(phi, rho)
        TH = np.broadcast_to(th[:, None], R.shape)
        g = self.feed.norm * (self.feed.plane_power(TH, self.feed.q_e) * np.cos(P) ** 2
                              + self.feed.plane_power(TH, self.feed.q_h) * np.sin(P) ** 2)
        e = np.sqrt(g) * np.cos(TH / 2.0) ** 2
        return rho, phi, R, e

    @cached_property
    def eta_spill(self) -> float:
        """Fraction of the feed's forward power the dish intercepts."""
        if self.reflector is None:
            return 1.0
        c0 = math.cos(self.reflector.half_angle_rad)
        qe, qh = self.feed.q_e, self.feed.q_h
        num = (1 - c0 ** (qe + 1)) / (qe + 1) + (1 - c0 ** (qh + 1)) / (qh + 1)
        den = 1 / (qe + 1) + 1 / (qh + 1)
        return num / den

    def _taper(self, with_blockage: bool) -> float:
        rho, phi, R, e = self._aperture_field()
        a = self.reflector.diameter_m / 2.0
        mask = np.ones_like(e)
        if with_blockage and self.reflector.blockage_fraction > 0:
            mask = (R >= a * math.sqrt(self.reflector.blockage_fraction)).astype(float)
        w = R                                         # dA = rho drho dphi
        num = _int2(e * mask * w, rho, phi) ** 2
        den = (math.pi * a * a) * _int2(e * e * w, rho, phi)
        return float(num / den)

    @cached_property
    def eta_taper(self) -> float:
        """Illumination (taper) efficiency: |integral E dA|^2 / (A integral |E|^2 dA)."""
        return 1.0 if self.reflector is None else self._taper(False)

    @cached_property
    def eta_block(self) -> float:
        """Loss from the feed/struts shadowing the aperture centre."""
        if self.reflector is None:
            return 1.0
        return self._taper(True) / self._taper(False)

    def eta_surface(self, freq_hz=None) -> np.ndarray:
        """Ruze: exp(-(4 pi eps / lambda)^2)."""
        if self.reflector is None:
            return np.ones_like(np.asarray(self.freq_hz if freq_hz is None else freq_hz, float))
        lam = C_LIGHT / np.asarray(self.freq_hz if freq_hz is None else freq_hz, float)
        return np.exp(-(4 * math.pi * self.reflector.surface_rms_m / lam) ** 2)

    # ------------------------------------------------------- headline numbers
    @property
    def eta_forward(self) -> float:
        """Fraction of feed power radiated forward (the rest is back radiation)."""
        return 1.0 - self.feed.back_fraction

    def directivity_lin(self) -> float:
        """Directivity of the radiated pattern (no loss terms)."""
        if self.reflector is None:
            return self.feed.norm
        return (4 * math.pi / self.wavelength_m ** 2
                * self.reflector.geometric_area_m2 * self.eta_taper * self.eta_block)

    def aperture_efficiency(self, freq_hz=None) -> np.ndarray:
        """eta_a = A_eff / A_geom, every loss included (reflector case)."""
        return (self.eta_spill * self.eta_forward * self.eta_taper * self.eta_block
                * self.eta_surface(freq_hz) * self.feed.eta_ohmic
                * self.feed.eta_mismatch)

    def gain_lin(self, freq_hz=None) -> np.ndarray:
        """Realised gain at the feed output terminals."""
        f = self.freq_hz if freq_hz is None else freq_hz
        if self.reflector is None:
            g = (self.feed.norm * self.eta_forward * self.feed.eta_ohmic
                 * self.feed.eta_mismatch)
            return np.full_like(np.asarray(f, float), g) * db_to_lin(self.feed.response_db(f))
        lam = C_LIGHT / np.asarray(f, float)
        return (4 * math.pi * self.reflector.geometric_area_m2 / lam ** 2
                * self.aperture_efficiency(f) * db_to_lin(self.feed.response_db(f)))

    def effective_area_m2(self, freq_hz=None) -> np.ndarray:
        f = self.freq_hz if freq_hz is None else freq_hz
        lam = C_LIGHT / np.asarray(f, float)
        return self.gain_lin(f) * lam ** 2 / (4 * math.pi)

    @property
    def ground_fraction(self) -> float:
        """Fraction of the antenna's response that looks at warm ground.

        Reflector: the spillover past the rim (a prime-focus feed looks down,
        so everything that misses the dish heads for the ground); the feed's
        back lobe looks up at cold sky.  Feed alone, pointing up: the part of
        its pattern below the local horizon line, plus its back lobe.
        """
        if self.reflector is not None:
            return self.eta_forward * (1.0 - self.eta_spill)
        th1 = math.radians(90.0 - self.horizon_deg)
        c1 = math.cos(th1)
        qe, qh = self.feed.q_e, self.feed.q_h
        below = ((c1 ** (qe + 1)) / (qe + 1) + (c1 ** (qh + 1)) / (qh + 1)) / (
            1 / (qe + 1) + 1 / (qh + 1))
        return self.eta_forward * below + self.feed.back_fraction

    # ------------------------------------------------------------ the beam
    def beam(self) -> "Beam":
        return Beam.compute(self)


def db_to_lin(db):
    return 10.0 ** (np.asarray(db, dtype=float) / 10.0)


def _int2(z, rho, phi):
    t = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return t(t(z, phi, axis=1), rho)


def _trap(y, x):
    t = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return t(y, x)


@dataclass
class Beam:
    """Normalised power pattern P(theta, phi), phi measured from the E-plane."""

    theta_deg: np.ndarray
    p_e: np.ndarray
    p_h: np.ndarray
    e_plane_azimuth_deg: float

    @classmethod
    def compute(cls, optics: Optics, n_theta: int | None = None) -> "Beam":
        feed, refl = optics.feed, optics.reflector
        if refl is None:
            th = np.linspace(0.0, 90.0, n_theta or 1801)
            t = np.radians(th)
            return cls(th, feed.plane_power(t, feed.q_e), feed.plane_power(t, feed.q_h),
                       feed.e_plane_azimuth_deg)
        # Fine enough to resolve the main lobe and several sidelobes.
        lam = optics.wavelength_m
        first_null_deg = math.degrees(1.22 * lam / refl.diameter_m) * 1.3
        th = np.linspace(0.0, min(90.0, 12 * first_null_deg), n_theta or 2401)
        t = np.radians(th)
        k = 2 * math.pi / lam
        a = refl.diameter_m / 2.0
        rho = np.linspace(0.0, a, 500)
        thp = 2.0 * np.arctan(rho / (2.0 * refl.focal_length_m))
        block = rho >= a * math.sqrt(refl.blockage_fraction)
        arg = k * np.outer(np.sin(t), rho)              # (n_theta, n_rho)
        kern = j0(arg) * rho
        out = []
        for q in (feed.q_e, feed.q_h):
            e = np.sqrt(feed.plane_power(thp, q)) * np.cos(thp / 2) ** 2 * block
            f = _trap(kern * e, rho) * (1 + np.cos(t)) / 2   # obliquity factor
            out.append((f / f[0]) ** 2)
        return cls(th, out[0], out[1], feed.e_plane_azimuth_deg)

    def power(self, theta_deg, phi_deg) -> np.ndarray:
        """P at (theta, phi), phi from the E-plane; cos^2/sin^2 blending."""
        th = np.asarray(theta_deg, float)
        ph = np.radians(np.asarray(phi_deg, float))
        pe = np.interp(th, self.theta_deg, self.p_e, right=0.0)
        ph_ = np.interp(th, self.theta_deg, self.p_h, right=0.0)
        return pe * np.cos(ph) ** 2 + ph_ * np.sin(ph) ** 2

    def power_azimuth(self, theta_deg, az_deg) -> np.ndarray:
        """P with the cut given as a compass azimuth (N = 0, E = 90)."""
        return self.power(theta_deg, np.asarray(az_deg) - self.e_plane_azimuth_deg)

    # ------------------------------------------------------------ measures
    @staticmethod
    def _hpbw(th, p) -> float:
        below = np.nonzero(p < 0.5)[0]
        if len(below) == 0:
            return float("nan")
        i = below[0]
        x = th[i - 1] + (0.5 - p[i - 1]) / (p[i] - p[i - 1]) * (th[i] - th[i - 1])
        return 2.0 * float(x)

    @staticmethod
    def _first_null_index(p) -> int | None:
        d = np.diff(p)
        rising = np.nonzero((d[:-1] < 0) & (d[1:] >= 0))[0]
        return int(rising[0] + 1) if len(rising) else None

    def hpbw_deg(self, phi_deg: float) -> float:
        return self._hpbw(self.theta_deg, self.power(self.theta_deg, phi_deg))

    @property
    def hpbw_e_deg(self) -> float:
        return self._hpbw(self.theta_deg, self.p_e)

    @property
    def hpbw_h_deg(self) -> float:
        return self._hpbw(self.theta_deg, self.p_h)

    def hpbw_azimuth_deg(self, az_deg: float) -> float:
        return self.hpbw_deg(az_deg - self.e_plane_azimuth_deg)

    def sidelobe_db(self, p) -> float | None:
        i = self._first_null_index(p)
        if i is None:
            return None
        return float(10 * np.log10(max(p[i:].max(), 1e-12)))

    def _grid(self, n_phi=181):
        phi = np.linspace(0.0, 360.0, n_phi)
        P = self.power(self.theta_deg[:, None], phi[None, :])
        return phi, P

    @property
    def solid_angle_sr(self) -> float:
        """Omega_A = integral of P dOmega over the computed range."""
        phi, P = self._grid()
        t = np.radians(self.theta_deg)
        return float(_trap(_trap(P * np.sin(t)[:, None], np.radians(phi)), t))

    @property
    def main_beam_solid_angle_sr(self) -> float:
        """Integral out to the first null of each phi cut."""
        phi, P = self._grid()
        t = np.radians(self.theta_deg)
        tot = []
        for j in range(P.shape[1]):
            col = P[:, j]
            i = self._first_null_index(col) or len(col)
            tot.append(_trap(col[:i] * np.sin(t[:i]), t[:i]))
        return float(_trap(np.asarray(tot), np.radians(phi)))

    @property
    def directivity_lin(self) -> float:
        return 4 * math.pi / self.solid_angle_sr
