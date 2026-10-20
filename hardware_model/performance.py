"""ALL of the mathematics that turns hardware parameters into performance parameters.

Nothing else in ``hardware_model`` computes physics:
``inputs.py`` only reads files and converts units, ``report.py`` only formats
numbers that were computed here, and ``plots.py`` only draws them.

How to read this file
=====================
* Sections 0-9 define the equations, one small function each, numbered
  ``[Eq. N]``.  Every one gives the formula, the meaning and unit of each
  symbol, and a reference (short keys below).
* Section 10, :func:`evaluate`, is the computation itself: Steps 1-12 apply the
  equations in order, each step naming the equations it uses.  Read
  :func:`evaluate` top to bottom to follow the whole calculation.

Conventions: frequencies in Hz, lengths in m, temperatures in K, angles in
degrees at the interfaces (radians inside), gains in dB unless "_lin".
Reference plane: the feed output terminals (see Step 9).

References (short key -> full citation)
---------------------------------------
[SI]       BIPM, *The International System of Units (SI Brochure)*, 9th ed., 2019.
           Exact values of c and k_B (sec. 2.2).
[Pozar]    D. M. Pozar, *Microwave Engineering*, 4th ed., Wiley, 2012.
           Ch. 1.7 (surface resistance), 2.3 (reflection, VSWR), 2.7 (lossy
           lines), 3.4 (circular waveguide), 4.3 (scattering matrix),
           8.3-8.4 (filter prototypes and transformations), 10.1-10.2 (noise
           temperature, noise figure, lossy lines, cascades).
[Friis]    H. T. Friis, "Noise figures of radio receivers," Proc. IRE 32(7),
           419-422, 1944.
[Nyquist]  H. Nyquist, "Thermal agitation of electric charge in conductors,"
           Phys. Rev. 32, 110-113, 1928.
[Balanis]  C. A. Balanis, *Antenna Theory: Analysis and Design*, 4th ed., Wiley,
           2016.  Ch. 2 (beamwidth, beam solid angle, directivity, efficiency,
           effective area), Ch. 12 (circular apertures), Ch. 15.4 (parabolic
           reflector: geometry, aperture distribution, spillover and taper
           efficiency for cos^n feeds).
[Kildal]   P.-S. Kildal, *Foundations of Antenna Engineering*, 2015, chapters on
           sub-efficiencies of aperture/reflector antennas and on BOR1 feeds.
[Ludwig]   A. C. Ludwig, "The definition of cross polarization," IEEE Trans.
           Antennas Propag. 21(1), 116-119, 1973.
[Ruze]     J. Ruze, "Antenna tolerance theory -- a review," Proc. IEEE 54(4),
           633-640, 1966.
[DLMF]     NIST Digital Library of Mathematical Functions, https://dlmf.nist.gov/
           eq. 10.9.2 (integral form of J_n) and 10.12.2 (Jacobi-Anger expansion).
[ERA]      J. J. Condon & S. M. Ransom, *Essential Radio Astronomy*, Princeton,
           2016.  Free online: https://www.cv.nrao.edu/~sransom/web/Ch3.html
           (Ch. 3: antenna temperature, SEFD, radiometer equation).
[Kraus]    J. D. Kraus, *Radio Astronomy*, 2nd ed., Cygnus-Quasar, 1986.
           Ch. 6 (antenna temperature, beam efficiency), Ch. 7 (radiometer
           sensitivity constant K_s).
[Dicke]    R. H. Dicke, "The measurement of thermal radiation at microwave
           frequencies," Rev. Sci. Instrum. 17, 268-275, 1946.
[Haslam]   C. G. T. Haslam et al., "A 408 MHz all-sky continuum survey. II,"
           A&AS 47, 1-143, 1982.
[Fixsen]   D. J. Fixsen, "The temperature of the cosmic microwave background,"
           ApJ 707, 916-920, 2009.
[ITU-372]  ITU-R Recommendation P.372, *Radio noise* (sky and atmospheric noise).
[ITU-RR]   ITU Radio Regulations, footnote 5.340 (1400-1427 MHz passive band).
[Hellwig]  H. Hellwig et al., "Measurement of the unperturbed hydrogen hyperfine
           transition frequency," IEEE Trans. Instrum. Meas. 19, 200-209, 1970.
[Meeus]    J. Meeus, *Astronomical Algorithms*, 2nd ed., Willmann-Bell, 1998.
           Ch. 12 (sidereal time), Ch. 13 (coordinate transformations).
[Touchstone] IBIS Open Forum, *Touchstone File Format Specification*, v1.1/2.0.
[TMS]      A. R. Thompson, J. M. Moran, G. W. Swenson, *Interferometry and
           Synthesis in Radio Astronomy*, 3rd ed., Springer (open access),
           2017 -- noise-equivalent bandwidth, Ch. 1/6.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.special import j0, jv

from .inputs import ComponentSpec, FeedSpec, Hardware, HardwareConfigError, ReflectorSpec

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


# =============================================================================
# 0. Constants
# =============================================================================
C_LIGHT = 299_792_458.0          # speed of light, m/s, exact            [SI]
K_BOLTZMANN = 1.380_649e-23      # Boltzmann constant, J/K, exact        [SI]
JY = 1e-26                       # 1 jansky in W m^-2 Hz^-1              [ERA ch. 2]
T0_K = 290.0                     # reference temperature of noise figure [Pozar 10.2]
HI_REST_HZ = 1_420_405_751.768   # HI hyperfine line                     [Hellwig]
HI_PROTECTED_BAND_HZ = (1.400e9, 1.427e9)                              # [ITU-RR]
SIDEREAL_DAY_H = 23.934_469_6    # mean sidereal day, hours              [Meeus ch. 12]
SIDEREAL_RATE_DEG_PER_H = 360.0 / SIDEREAL_DAY_H   # = 15.0411 deg/h the sky drifts


# =============================================================================
# 1. Decibels and noise temperature
# =============================================================================
def db_to_lin(x_db):
    """[Eq. 1a]  G = 10^(G_dB / 10).

    Power ratio from decibels.  Losses are negative gains: a part with
    1 dB loss has G_dB = -1, G = 0.794.  (Definition of the decibel.)"""
    return 10.0 ** (np.asarray(x_db, dtype=float) / 10.0)


def lin_to_db(x):
    """[Eq. 1b]  G_dB = 10 log10(G)."""
    return 10.0 * np.log10(np.maximum(np.asarray(x, dtype=float), 1e-300))


def nf_to_temp_k(nf_db):
    """[Eq. 2]  T_e = T0 (F - 1),  F = 10^(NF/10),  T0 = 290 K.

    Effective input noise temperature of an *active* part (LNA, SDR) from
    its datasheet noise figure NF (dB).  [Pozar 10.2, "noise figure";
    IEEE definition of F at T0 = 290 K]"""
    return T0_K * (db_to_lin(nf_db) - 1.0)


def temp_to_nf_db(t_k):
    """[Eq. 2, inverted]  NF = 10 log10(1 + T_e / T0)."""
    return lin_to_db(1.0 + np.asarray(t_k, dtype=float) / T0_K)


def passive_temp_k(gain_db, t_phys_k):
    """[Eq. 3]  T_e = T_phys (L - 1),  L = 10^(loss_dB/10) = 1/G.

    Input-referred noise temperature of a *passive* lossy part (cable,
    filter, attenuator) at physical temperature T_phys.  A lossy part in
    thermal equilibrium radiates as much as it absorbs, so a warm cable adds
    noise as well as attenuating.  [Pozar 10.2, "noise figure of a lossy
    transmission line"]"""
    loss_lin = 1.0 / db_to_lin(np.minimum(gain_db, 0.0))
    return t_phys_k * (loss_lin - 1.0)


def friis_cascade(temps_k: Sequence[np.ndarray], gains_db: Sequence[np.ndarray]):
    """[Eq. 4]  T_rec = T1 + T2/G1 + T3/(G1 G2) + T4/(G1 G2 G3) + ...

    Receiver temperature of stages in order.  Stage i's own noise T_i is
    divided by the total gain G_1...G_(i-1) ahead of it, which is why noise
    after a high-gain LNA matters little and noise before it matters fully.
    Returns (contribution of each stage, gain ahead of each stage).
    [Friis; Pozar 10.2, "noise figure of a cascaded system"]"""
    gain_before = np.ones_like(np.asarray(temps_k[0], dtype=float))
    contributions, befores = [], []
    for t, g in zip(temps_k, gains_db):
        befores.append(gain_before.copy())
        contributions.append(np.asarray(t, dtype=float) / gain_before)
        gain_before = gain_before * db_to_lin(g)
    return contributions, befores


def cascade_gain_db(gains_db: Sequence[np.ndarray]):
    """[Eq. 5]  G_total,dB = sum_i G_i,dB   (i.e. G_total = prod_i G_i).

    Gains multiply in linear units, so they add in dB.  (Definition.)"""
    return sum(np.asarray(g, dtype=float) for g in gains_db)


# =============================================================================
# 2. How each kind of component responds to frequency
# =============================================================================
def bandpass_attenuation_db(f, f0, bw, order, kind="butterworth", ripple_db=0.5):
    """[Eq. 6]  Ideal bandpass prototype (positive dB, 0 at the centre).

    Lowpass-to-bandpass frequency mapping
        W = (f0 / B) (f/f0 - f0/f)          (W = +/-1 at the band edges)
    Butterworth ("maximally flat"):   A(W) = 10 log10(1 + W^(2n))
    Chebyshev  (equal ripple):        A(W) = 10 log10(1 + eps^2 T_n(W)^2),
                                      eps^2 = 10^(ripple/10) - 1,
                                      T_n = Chebyshev polynomial
                                          = cos(n acos W) for |W| <= 1,
                                            cosh(n acosh|W|) for |W| > 1.
    n = order (number of resonators), B = bandwidth (3 dB for Butterworth,
    ripple bandwidth for Chebyshev), f0 = centre.
    [Pozar 8.3, "insertion loss method" (maximally flat and equal-ripple
    responses); 8.4, "bandpass transformation"]"""
    f = np.maximum(np.asarray(f, dtype=float), 1.0)
    w = (f / f0 - f0 / f) * f0 / bw
    n = int(order)
    if kind == "butterworth":
        return 10.0 * np.log10(1.0 + np.abs(w) ** (2 * n))
    if kind == "chebyshev":
        eps2 = 10.0 ** (ripple_db / 10.0) - 1.0
        aw = np.abs(w)
        tn = np.where(aw <= 1.0, np.cos(n * np.arccos(np.minimum(aw, 1.0))),
                      np.cosh(n * np.arccosh(np.maximum(aw, 1.0))))
        return 10.0 * np.log10(1.0 + eps2 * tn ** 2)
    raise HardwareConfigError(f"unknown filter response {kind!r} (butterworth or chebyshev)")


def coax_loss_db(loss_ref_db, f, f_ref):
    """[Eq. 7]  L(f) = L(f_ref) sqrt(f / f_ref).

    Coax loss scales as the square root of frequency because conductor
    (skin-effect) loss dominates: the surface resistance is
    R_s = sqrt(omega mu0 / 2 sigma) and the attenuation constant
    alpha_c is proportional to R_s.  Dielectric loss (~ f) is neglected; at
    1.4 GHz in foam-dielectric cable such as LMR-400 it is small.
    [Pozar 1.7 (R_s), 2.7 (alpha_c of a lossy line)]"""
    return loss_ref_db * np.sqrt(np.asarray(f, dtype=float) / f_ref)


def datasheet_edge_rolloff_db(f, lo_hz, hi_hz, db_per_octave):
    """[Eq. 8]  A(f) = s * log2(f_lo / f) below f_lo,  s * log2(f / f_hi) above f_hi,
    0 in between   (positive dB; s = slope in dB per octave).

    MODEL ASSUMPTION, NOT FROM A REFERENCE.  A datasheet gives a gain and a
    "frequency range"; it does not say how the part behaves outside it.  This
    leaves the datasheet value untouched inside the range and falls off at a
    fixed slope beyond it, so out-of-band numbers are illustrative only.
    Replace with a measured curve or a Touchstone file when you have one."""
    f = np.asarray(f, dtype=float)
    att = np.zeros_like(f)
    if lo_hz:
        att += np.where(f < lo_hz, db_per_octave * np.log2(lo_hz / np.maximum(f, 1.0)), 0.0)
    if hi_hz:
        att += np.where(f > hi_hz, db_per_octave * np.log2(f / hi_hz), 0.0)
    return att


def s21_db(fmt: str, a, b):
    """[Eq. 9]  |S21|_dB from the numbers stored in a Touchstone file.

    DB format: already 20 log10|S21|.   MA (magnitude-angle): 20 log10(a).
    RI (real-imaginary): 20 log10 sqrt(a^2 + b^2).  |S21|^2 is the power
    gain into a matched load.  [Pozar 4.3; Touchstone]"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if fmt == "DB":
        return a
    if fmt == "MA":
        return 20.0 * np.log10(np.maximum(a, 1e-30))
    return 20.0 * np.log10(np.maximum(np.hypot(a, b), 1e-30))


def sdr_band_hz(comp: ComponentSpec) -> tuple[float, float]:
    """SDR band: tuned frequency +/- half its IF bandwidth (default: the
    complex sample rate, which is the Nyquist bandwidth of I/Q sampling)."""
    p = comp.params
    c = p.number("tuned_freq", "Hz", comp.center_hz)
    bw = p.number("if_bandwidth", "Hz") or p.require("sample_rate", "Hz").scalar()
    return c - bw / 2.0, c + bw / 2.0


def component_gain_db(comp: ComponentSpec, f) -> np.ndarray:
    """Gain (dB, losses negative) of one chain component vs frequency,
    choosing the equation by component type."""
    f = np.asarray(f, dtype=float)
    p = comp.params
    if comp.type == "amplifier":
        g = (np.interp(f, comp.s21.freq_hz, s21_db(comp.s21.fmt, comp.s21.a, comp.s21.b))
             if comp.s21 else p.require("gain", "dB").at(f))                     # Eq. 9
        return g - datasheet_edge_rolloff_db(                                     # Eq. 8
            f, p.number("freq_min", "Hz"), p.number("freq_max", "Hz"),
            p.number("rolloff", "dB", 12.0))
    if comp.type == "filter":
        if comp.s21:
            return np.interp(f, comp.s21.freq_hz, s21_db(comp.s21.fmt, comp.s21.a, comp.s21.b))
        il = p.require("insertion_loss", "dB").at(f) if p.get("insertion_loss") else 0.0
        att = bandpass_attenuation_db(                                            # Eq. 6
            f, p.require("center_freq", "Hz").scalar(), p.require("bandwidth", "Hz").scalar(),
            p.number("order", "", 5.0), kind=p.text("response", "butterworth").lower(),
            ripple_db=p.number("ripple", "dB", 0.5))
        rej = p.number("out_of_band_rejection", "dB")
        if rej is not None:
            att = np.minimum(att, rej)          # a real filter's stopband has a floor
        return -(il + att)
    if comp.type in {"cable", "attenuator"}:
        return -_passive_loss_db(comp, f)
    if comp.type == "sdr":
        # The SDR digitises; its only "gain" here is its selectivity, which
        # sets the band actually integrated.  Modelled as a steep (order-12)
        # Butterworth across its I/Q bandwidth -- a stand-in for its
        # anti-alias + digital decimation filters.                             Eq. 6
        lo, hi = sdr_band_hz(comp)
        att = bandpass_attenuation_db(f, (lo + hi) / 2, hi - lo, p.number("order", "", 12.0))
        return -np.minimum(att, p.number("stopband", "dB", 80.0))
    raise HardwareConfigError(f"{comp.id}: unknown chain type {comp.type!r}")


def _passive_loss_db(comp: ComponentSpec, f) -> np.ndarray:
    p = comp.params
    loss = p.get("loss")
    if loss is not None:
        ref = p.number("loss_ref_freq", "Hz")
        if loss.curve_hz is not None or comp.type == "attenuator" or ref is None:
            return loss.at(f)
        return coax_loss_db(loss.scalar(), f, ref)                               # Eq. 7
    per_m = p.get("loss_per_length")
    if per_m is None:
        raise HardwareConfigError(
            f"{comp.id}: give either 'loss' (dB) or 'loss_per_length' + 'length'")
    length = p.require("length", "m").scalar()
    if per_m.curve_hz is not None:
        return per_m.at(f) * length
    return coax_loss_db(per_m.scalar() * length, f,                               # Eq. 7
                        p.require("loss_ref_freq", "Hz").scalar())


def component_noise_temp_k(comp: ComponentSpec, f, gain_db=None) -> np.ndarray:
    """Input-referred noise temperature of one component: Eq. 2 for active
    parts, Eq. 3 for passive ones."""
    if comp.type in {"amplifier", "sdr"}:
        return nf_to_temp_k(comp.params.require("noise_figure", "dB").at(f))      # Eq. 2
    g = component_gain_db(comp, f) if gain_db is None else gain_db
    return passive_temp_k(g, comp.t_phys_k)                                       # Eq. 3


def component_band_hz(comp: ComponentSpec) -> tuple[float | None, float | None]:
    """Nominal band edges a component declares (used by Eq. 10)."""
    p = comp.params
    if comp.type == "filter":
        c, bw = p.require("center_freq", "Hz").scalar(), p.require("bandwidth", "Hz").scalar()
        return c - bw / 2.0, c + bw / 2.0
    if comp.type == "sdr":
        return sdr_band_hz(comp)
    return p.number("freq_min", "Hz"), p.number("freq_max", "Hz")


# =============================================================================
# 3. Passband
# =============================================================================
def intersect_bands(bands: Sequence[tuple[str, float | None, float | None]]):
    """[Eq. 10]  f_lo = max_i(f_lo,i),   f_hi = min_i(f_hi,i).

    The chain passes only what every part passes: the lower edge is the
    highest lower edge, the upper edge the lowest upper edge.  (Set
    intersection of the components' bands.)  Returns (lo, hi, who set lo,
    who set hi)."""
    lo, hi, lo_by, hi_by = -math.inf, math.inf, "", ""
    for name, a, b in bands:
        if a is not None and a > lo:
            lo, lo_by = a, name
        if b is not None and b < hi:
            hi, hi_by = b, name
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise HardwareConfigError("no component declares band edges, so the passband is undefined")
    if lo >= hi:
        raise HardwareConfigError(
            f"the chain has no common passband: {lo_by} starts at {lo/1e6:.2f} MHz "
            f"but {hi_by} ends at {hi/1e6:.2f} MHz")
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


def measure_passband(f, response_db, center_hz, nominal) -> Passband:
    """[Eq. 11]  -3 dB band: the contiguous region around f_c where
                 R(f) >= R_peak - 3 dB, edges linearly interpolated.
    [Eq. 12]  Noise-equivalent bandwidth  B_N = integral G(f) df / G_peak.
              The width of a rectangular filter with the same peak gain that
              passes the same noise power.  This, not the -3 dB width, is the
              bandwidth in the radiometer equation (Eq. 34).  [TMS; Pozar 10.1]
    Ripple: max - min of R over the central 80 % of the -3 dB band (over the
    whole band it would just be the 3 dB the edges fall by)."""
    f, r = np.asarray(f), np.asarray(response_db)
    i0 = int(np.argmin(np.abs(f - center_hz)))
    peak = float(r.max())
    above = r >= peak - 3.0
    if not above[i0]:
        raise HardwareConfigError(
            f"the analysis centre {center_hz/1e6:.3f} MHz is more than 3 dB down the "
            f"chain's response -- is a filter or the SDR tuned elsewhere?")
    lo_i = i0
    while lo_i > 0 and above[lo_i - 1]:
        lo_i -= 1
    hi_i = i0
    while hi_i < len(f) - 1 and above[hi_i + 1]:
        hi_i += 1

    def cross(i_in, i_out):                                    # Eq. 11 interpolation
        if i_out < 0 or i_out >= len(f):
            return float(f[i_in])
        y0, y1 = r[i_in] - (peak - 3.0), r[i_out] - (peak - 3.0)
        t = y0 / (y0 - y1) if y0 != y1 else 0.0
        return float(f[i_in] + t * (f[i_out] - f[i_in]))

    lo3, hi3 = cross(lo_i, lo_i - 1), cross(hi_i, hi_i + 1)
    b_n = float(_trapz(db_to_lin(r - peak), f))                                   # Eq. 12
    m = 0.1 * (hi3 - lo3)
    band = r[(f >= lo3 + m) & (f <= hi3 - m)]
    if len(band) == 0:
        band = r[i0:i0 + 1]
    return Passband(nominal[0], nominal[1], nominal[2], nominal[3], lo3, hi3, b_n,
                    float(band.max() - band.min()), peak)


# =============================================================================
# 4. The feed
# =============================================================================
def wavelength_m(f_hz):
    """[Eq. 13]  lambda = c / f."""
    return C_LIGHT / np.asarray(f_hz, dtype=float)


def cosq_exponent_from_hpbw(hpbw_deg: float) -> float:
    """[Eq. 14a]  q = ln(1/2) / ln(cos(HPBW/2)).

    The feed's power pattern in each principal plane is modelled as cos^q(theta)
    for theta <= 90 deg (the standard "cos^n" feed model [Balanis 15.4]).
    Solving cos^q(HPBW/2) = 1/2 for q gives this."""
    return math.log(0.5) / math.log(math.cos(math.radians(hpbw_deg) / 2.0))


def cosq_exponent_from_taper(taper_db: float, theta0_rad: float) -> float:
    """[Eq. 14b]  q = T_dB / (10 log10 cos(theta_0)).

    The q whose cos^q pattern is T_dB (negative) down at the rim angle
    theta_0 (Eq. 18).  Solving 10 log10(cos^q theta_0) = T_dB for q."""
    return taper_db / (10.0 * math.log10(math.cos(theta0_rad)))


def feed_pattern_norm(q_e: float, q_h: float) -> float:
    """[Eq. 15]  G_f(theta, phi) = C [cos^qE(theta) cos^2(phi) + cos^qH(theta) sin^2(phi)],
                 C = 4 / (1/(qE+1) + 1/(qH+1)).

    Total (co- + cross-polar) power pattern of a feed whose E-plane (phi = 0)
    and H-plane (phi = 90 deg) cuts differ.  For a "BOR1" feed (any feed fed
    by a single TE11-like mode: cantenna, horn, dipole-over-disc) the
    co- and cross-polar fields in Ludwig's 3rd definition are
        E_co = E_E(theta) cos^2 phi + E_H(theta) sin^2 phi
        E_xp = (E_E(theta) - E_H(theta)) sin phi cos phi
    and |E_co|^2 + |E_xp|^2 = |E_E|^2 cos^2 phi + |E_H|^2 sin^2 phi exactly,
    which is the form above with |E_E|^2 = cos^qE, |E_H|^2 = cos^qH.
    C makes the pattern integrate to 4 pi over the forward hemisphere
    (integral of cos^q(theta) sin(theta) dtheta from 0 to 90 deg = 1/(q+1),
    integral of cos^2 phi dphi over 2 pi = pi), i.e. C is the peak directivity.
    For qE = qH = q it reduces to the familiar 2(q+1) cos^q [Balanis 15.4].
    [Kildal, BOR1 antennas; Ludwig]"""
    return 4.0 / (1.0 / (q_e + 1.0) + 1.0 / (q_h + 1.0))


def circular_waveguide_band_hz(diameter_m: float) -> tuple[float, float]:
    """[Eq. 16]  f_c,TE11 = 1.8412 c / (pi d),   f_c,TM01 = 2.4048 c / (pi d).

    A circular-waveguide feed (cantenna) carries only the TE11 mode between
    these two cut-offs: below TE11 nothing propagates, above TM01 a second
    mode spoils the pattern.  1.8412 = first zero of J1', 2.4048 = first zero
    of J0.  [Pozar 3.4, circular waveguide mode tables]"""
    return (1.8412 * C_LIGHT / (math.pi * diameter_m),
            2.4048 * C_LIGHT / (math.pi * diameter_m))


def mismatch_efficiency(vswr: float) -> float:
    """[Eq. 17]  |Gamma| = (VSWR - 1)/(VSWR + 1),   eta_mm = 1 - |Gamma|^2.

    Fraction of the power the antenna collects that is delivered to the
    receiver rather than reflected.  [Pozar 2.3; Balanis ch. 2, "antenna
    efficiency" (reflection efficiency e_r)]"""
    g = (vswr - 1.0) / (vswr + 1.0)
    return 1.0 - g * g


def ohmic_efficiency(loss_db: float) -> float:
    """[Eq. 18]  eta_ohm = 10^(-L_dB/10).

    Radiation (conduction-dielectric) efficiency of the feed from its ohmic
    loss.  [Balanis ch. 2, "antenna efficiency" (e_cd)]"""
    return float(db_to_lin(-loss_db))


# =============================================================================
# 5. The reflector
# =============================================================================
def rim_half_angle_rad(focal_ratio: float) -> float:
    """[Eq. 19]  tan(theta_0 / 2) = 1 / (4 f/D).

    Half-angle the dish rim subtends at the focus of a paraboloid.
    [Balanis 15.4, front-fed paraboloid geometry]"""
    return 2.0 * math.atan(1.0 / (4.0 * focal_ratio))


def aperture_mapping(rho, focal_length_m):
    """[Eq. 20]  theta'(rho) = 2 atan(rho / 2f),   amplitude factor cos^2(theta'/2).

    A ray leaving the focus at angle theta' from the axis reflects off the
    paraboloid and crosses the aperture plane at radius rho = 2f tan(theta'/2).
    The path from the focus to the dish is r = 2f / (1 + cos theta'), so the
    spherical-wave amplitude falls as 1/r, proportional to cos^2(theta'/2):
        E_ap(rho, phi) = sqrt(G_f(theta', phi)) * cos^2(theta'/2)    (up to a constant).
    (Equivalently: power per unit aperture area = G_f cos^4(theta'/2) / f^2.)
    [Balanis 15.4, aperture-field distribution; Kildal, reflector chapter]"""
    th = 2.0 * np.arctan(np.asarray(rho, float) / (2.0 * focal_length_m))
    return th, np.cos(th / 2.0) ** 2


def aperture_fields(feed_q: tuple[float, float], refl: ReflectorSpec, n_rho: int = 2000):
    """Radial parts of the co-polar aperture field (Eqs. 15, 20).

    With E_E, E_H from Eq. 15 mapped through Eq. 20,
        E_co(rho, phi) = A(rho) cos^2 phi + B(rho) sin^2 phi
                       = (A + B)/2 + (A - B)/2 cos 2phi
        E_xp(rho, phi) = (A - B)/2 sin 2phi
    with A = sqrt(C cos^qE theta') cos^2(theta'/2), B likewise with qH.
    For a paraboloid the aperture co/cross-polar fields are the feed's
    Ludwig-3 co/cross fields carried along the rays.  [Kildal; Ludwig]
    Returns rho, A, B."""
    q_e, q_h = feed_q
    c = feed_pattern_norm(q_e, q_h)
    rho = np.linspace(0.0, refl.diameter_m / 2.0, n_rho)
    th, amp = aperture_mapping(rho, refl.focal_ratio * refl.diameter_m)
    ct = np.cos(th)
    return rho, np.sqrt(c * ct ** q_e) * amp, np.sqrt(c * ct ** q_h) * amp


def spillover_efficiency(q_e: float, q_h: float, theta0_rad: float) -> float:
    """[Eq. 21]  eta_sp = [ (1 - cos^(qE+1) theta_0)/(qE+1) + (1 - cos^(qH+1) theta_0)/(qH+1) ]
                         / [ 1/(qE+1) + 1/(qH+1) ].

    Fraction of the feed's (forward) power that hits the dish: the integral
    of Eq. 15 over theta < theta_0 divided by its integral over the hemisphere.
    Uses integral_0^theta_0 cos^q(t) sin(t) dt = (1 - cos^(q+1) theta_0)/(q+1).
    [Balanis 15.4, spillover efficiency; Kildal, sub-efficiencies]"""
    c0 = math.cos(theta0_rad)
    num = (1 - c0 ** (q_e + 1)) / (q_e + 1) + (1 - c0 ** (q_h + 1)) / (q_h + 1)
    return num / (1 / (q_e + 1) + 1 / (q_h + 1))


def illumination_polarization_blockage(rho, a_field, b_field, blockage_fraction: float):
    """[Eq. 22]  Aperture sub-efficiencies, from the fields of aperture_fields().

    illumination (taper):  eta_ill = |integral E_co dA|^2 / (A_geom integral |E_co|^2 dA)
    polarization:          eta_pol = integral |E_co|^2 dA / integral (|E_co|^2 + |E_xp|^2) dA
    blockage:              eta_bl  = ( integral_unblocked E_co dA / integral E_co dA )^2

    eta_ill is 1 only for a perfectly uniform aperture; tapering the edge to
    cut spillover lowers it.  eta_pol < 1 when the E- and H-plane patterns
    differ (cross-polar power is wasted for a single-polarisation receiver).
    eta_bl: the feed and struts shadow a central disc of area fraction
    `blockage_fraction`; that field is removed from the sum.
    shadow: s_b = integral_blocked (|E_co|^2+|E_xp|^2) dA / integral (|E_co|^2+|E_xp|^2) dA,
    the share of the intercepted power that hits the blockage and is
    scattered instead of radiated in the beam.  The beam's directivity
    counts only radiated power, so D carries 1/(1 - s_b) and the gain
    carries (1 - s_b); their product, and so A_eff, is unchanged.
    With the cos 2phi decomposition the phi integrals are done exactly
    (integral of cos 2phi over 2 pi = 0, of cos^2 2phi = pi):
        integral E_co dA          = 2 pi integral (A+B)/2 rho drho
        integral |E_co|^2 dA      = integral [2 pi ((A+B)/2)^2 + pi ((A-B)/2)^2] rho drho
        integral (|E_co|^2+|E_xp|^2) dA = pi integral (A^2 + B^2) rho drho
    [Kildal, sub-efficiencies of aperture antennas; Balanis 15.4 (taper
    efficiency); Ludwig (co/cross-polar split)]"""
    a_geom = math.pi * rho[-1] ** 2
    s, d = (a_field + b_field) / 2.0, (a_field - b_field) / 2.0
    sum_co = 2 * math.pi * _trapz(s * rho, rho)
    pow_co = _trapz((2 * math.pi * s ** 2 + math.pi * d ** 2) * rho, rho)
    pow_tot = math.pi * _trapz((a_field ** 2 + b_field ** 2) * rho, rho)
    eta_ill = float(sum_co ** 2 / (a_geom * pow_co))
    eta_pol = float(pow_co / pow_tot)
    unblocked = rho >= rho[-1] * math.sqrt(blockage_fraction)
    sum_ub = 2 * math.pi * _trapz(np.where(unblocked, s, 0.0) * rho, rho)
    eta_bl = float((sum_ub / sum_co) ** 2)
    pow_ub = math.pi * _trapz(np.where(unblocked, a_field ** 2 + b_field ** 2, 0.0) * rho, rho)
    shadow = float(1.0 - pow_ub / pow_tot)
    return eta_ill, eta_pol, eta_bl, shadow


def ruze_efficiency(surface_rms_m: float, f_hz):
    """[Eq. 23]  eta_s = exp( -(4 pi eps / lambda)^2 ).

    Gain lost to random surface errors of rms eps (measured normal to the
    surface).  3 mm on a 21 cm wavelength costs 3 %; it matters far more at
    higher frequencies.  [Ruze]"""
    lam = wavelength_m(f_hz)
    return np.exp(-(4.0 * math.pi * surface_rms_m / lam) ** 2)


def gain_and_effective_area(directivity_lin, efficiency, f_hz):
    """[Eq. 24]  G = eta D,   A_eff = G lambda^2 / (4 pi).

    Realised gain is directivity times every efficiency; effective area
    follows from the reciprocity relation between gain and capture area.
    For a reflector D = 4 pi A_geom / lambda^2 times the aperture
    sub-efficiencies of Eq. 22, so A_eff = eta_a A_geom.
    [Balanis ch. 2, "maximum effective area" and "gain"; ERA ch. 3]"""
    g = np.asarray(efficiency, float) * np.asarray(directivity_lin, float)
    return g, g * wavelength_m(f_hz) ** 2 / (4.0 * math.pi)


# =============================================================================
# 6. The beam, in 3-D
# =============================================================================
@dataclass
class Beam:
    """Far-field pattern as two radial functions (see Eq. 25):

        P(theta, phi) = [ (f0 - f2 cos 2phi)^2 + (f2 sin 2phi)^2 ] / f0(0)^2

    phi measured from the feed's E-plane; ``e_plane_azimuth_deg`` places
    the E-plane on the compass."""

    theta_deg: np.ndarray
    f0: np.ndarray
    f2: np.ndarray
    e_plane_azimuth_deg: float

    def power(self, theta_deg, phi_deg) -> np.ndarray:
        th = np.asarray(theta_deg, float)
        ph = np.radians(np.asarray(phi_deg, float))
        f0 = np.interp(th, self.theta_deg, self.f0, right=0.0)
        f2 = np.interp(th, self.theta_deg, self.f2, right=0.0)
        return ((f0 - f2 * np.cos(2 * ph)) ** 2 + (f2 * np.sin(2 * ph)) ** 2) / self.f0[0] ** 2

    def power_azimuth(self, theta_deg, az_deg) -> np.ndarray:
        """P with the direction given as a compass azimuth (N = 0, E = 90)."""
        return self.power(theta_deg, np.asarray(az_deg) - self.e_plane_azimuth_deg)

    def cut(self, phi_deg: float) -> np.ndarray:
        return self.power(self.theta_deg, phi_deg)


def far_field(feed_q, refl: ReflectorSpec | None, f_hz: float, e_plane_az: float,
              n_theta: int | None = None) -> Beam:
    """[Eq. 25]  Far field of the aperture, exactly separated in phi.

    Scalar aperture theory: the far field is the 2-D Fourier transform of the
    aperture field,  F(theta, phi) = (1 + cos theta)/2 * integral E(rho, phi') e^{i k rho sin(theta) cos(phi' - phi)} dA,
    with the Huygens obliquity factor (1 + cos theta)/2.  [Balanis ch. 12]
    Expanding the exponential with the Jacobi-Anger series [DLMF 10.12.2] and
    doing the phi' integral [DLMF 10.9.2], a cos(2phi') field term transforms to
    -2 pi J2(u) cos(2phi), and a constant term to 2 pi J0(u), u = k rho sin theta.
    With the fields of aperture_fields():
        F_co = (1+cos theta)/2 * 2 pi [ integral (A+B)/2 J0(u) rho drho
                                        - cos 2phi integral (A-B)/2 J2(u) rho drho ]
        F_xp = -(1+cos theta)/2 * 2 pi sin 2phi integral (A-B)/2 J2(u) rho drho
        P    = |F_co|^2 + |F_xp|^2.
    For A = B (a symmetric feed) this reduces to the textbook circular-aperture
    Hankel transform (uniform illumination gives the Airy pattern,
    HPBW 1.03 lambda/D, first sidelobe -17.6 dB) [Balanis ch. 12].
    The blocked central disc contributes no field.

    No reflector: the feed is the antenna, and its own fields are already the
    far field:  f0 = (E_E + E_H)/2,  f2 = -(E_E - E_H)/2  with E = sqrt(cos^q).
    """
    q_e, q_h = feed_q
    if refl is None:
        th = np.linspace(0.0, 90.0, n_theta or 1801)
        c = np.cos(np.radians(th))
        ee, eh = c ** (q_e / 2), c ** (q_h / 2)
        return Beam(th, (ee + eh) / 2, -(ee - eh) / 2, e_plane_az)
    lam = float(wavelength_m(f_hz))
    first_null = math.degrees(1.22 * lam / refl.diameter_m)       # Airy first null
    th = np.linspace(0.0, min(90.0, 16 * first_null), n_theta or 3001)
    t = np.radians(th)
    rho, a, b = aperture_fields(feed_q, refl, n_rho=600)
    keep = rho >= rho[-1] * math.sqrt(refl.blockage_fraction)
    s = np.where(keep, (a + b) / 2, 0.0)
    d = np.where(keep, (a - b) / 2, 0.0)
    u = (2 * math.pi / lam) * np.outer(np.sin(t), rho)
    obliq = (1 + np.cos(t)) / 2
    f0 = obliq * 2 * math.pi * _trapz(j0(u) * s * rho, rho, axis=1)
    f2 = obliq * 2 * math.pi * _trapz(jv(2, u) * d * rho, rho, axis=1)
    return Beam(th, f0, f2, e_plane_az)


def half_power_beamwidth_deg(theta_deg, p) -> float:
    """[Eq. 26a]  HPBW = 2 theta_half,  where P(theta_half) = 1/2.

    Full width between the half-power points of a cut through the beam,
    crossing linearly interpolated.  [Balanis ch. 2, "beamwidth"]"""
    below = np.nonzero(p < 0.5)[0]
    if len(below) == 0:
        return float("nan")
    i = below[0]
    x = theta_deg[i - 1] + (0.5 - p[i - 1]) / (p[i] - p[i - 1]) * (theta_deg[i] - theta_deg[i - 1])
    return 2.0 * float(x)


def first_null_index(p) -> int | None:
    d = np.diff(p)
    rising = np.nonzero((d[:-1] < 0) & (d[1:] >= 0))[0]
    return int(rising[0] + 1) if len(rising) else None


def first_sidelobe_db(p) -> float | None:
    """[Eq. 26b]  SLL = 10 log10( max P beyond the first null ).  [Balanis ch. 2]"""
    i = first_null_index(p)
    return None if i is None else float(10 * np.log10(max(p[i:].max(), 1e-12)))


def beam_solid_angle_sr(beam: Beam) -> float:
    """[Eq. 26c]  Omega_A = integral P dOmega = 2 pi integral (f0^2 + f2^2) sin(theta) dtheta / f0(0)^2.

    Beam solid angle; the phi integral of Eq. 25's P is done exactly
    (cross term integrates to zero).  [Balanis ch. 2, "beam solid angle"]"""
    t = np.radians(beam.theta_deg)
    return float(2 * math.pi * _trapz((beam.f0 ** 2 + beam.f2 ** 2) * np.sin(t), t)
                 / beam.f0[0] ** 2)


def main_beam_solid_angle_sr(beam: Beam, n_phi: int = 181) -> float:
    """[Eq. 26d]  Omega_M = integral of P over the main lobe (to the first null
    of each phi cut).  Main-beam efficiency eta_MB = Omega_M / Omega_A.
    [Kraus ch. 6; Balanis ch. 2]"""
    phi = np.linspace(0.0, 360.0, n_phi)
    t = np.radians(beam.theta_deg)
    per_cut = []
    for ph in phi:
        col = beam.cut(ph)
        i = first_null_index(col) or len(col)
        per_cut.append(_trapz(col[:i] * np.sin(t[:i]), t[:i]))
    return float(_trapz(np.asarray(per_cut), np.radians(phi)))


def directivity_from_solid_angle(omega_a_sr: float) -> float:
    """[Eq. 26e]  D = 4 pi / Omega_A.  [Balanis ch. 2, "directivity"]"""
    return 4.0 * math.pi / omega_a_sr


# =============================================================================
# 7. Noise from the sky and the ground
# =============================================================================
def sky_temperature_k(f_hz, t_cmb, t408, beta, t_atm):
    """[Eq. 27]  T_sky(f) = T_CMB + T_408 (f / 408 MHz)^beta + T_atm.

    Cosmic microwave background (2.7255 K [Fixsen]) + Galactic synchrotron
    scaled from the 408 MHz all-sky map [Haslam] with spectral index beta
    (about -2.7 near 1 GHz) + atmospheric emission at the zenith (about 2 K
    at 1.4 GHz [ITU-372]).  T_408 is the map value along the beam: ~20 K off
    the Galactic plane, 50-200 K in it."""
    f = np.asarray(f_hz, float)
    return t_cmb + t408 * (f / 408e6) ** beta + t_atm


def ground_fraction(feed_q, back_fraction: float, theta0_rad: float | None,
                    horizon_deg: float) -> float:
    """[Eq. 28]  f_g = share of the antenna's response that looks at warm ground.

    Reflector (prime focus, pointing up):  f_g = (1 - b)(1 - eta_sp).
        The feed looks *down* at the dish, so everything that misses the dish
        (the spillover, Eq. 21) continues to the ground; the feed's back lobe
        (fraction b) looks up at the sky.
    Feed alone, pointing up:  f_g = (1 - b) x_h + b,
        x_h = part of the Eq. 15 pattern more than 90 - h deg from the zenith
        (below the horizon line at elevation h: roofline, trees), from
        integral_theta1^90 cos^q sin = cos^(q+1)(theta1)/(q+1); the back lobe
        looks down at the roof.
    [Kraus ch. 6 and ERA ch. 3: antenna temperature is the pattern-weighted
    brightness of everything the antenna sees]"""
    q_e, q_h = feed_q
    if theta0_rad is not None:
        return (1 - back_fraction) * (1 - spillover_efficiency(q_e, q_h, theta0_rad))
    c1 = math.cos(math.radians(90.0 - horizon_deg))
    x_h = ((c1 ** (q_e + 1)) / (q_e + 1) + (c1 ** (q_h + 1)) / (q_h + 1)) / (
        1 / (q_e + 1) + 1 / (q_h + 1))
    return (1 - back_fraction) * x_h + back_fraction


def antenna_temperature_terms(t_sky, t_ground, f_g, eta_ohm, eta_mm, t_feed):
    """[Eq. 29]  T_A = eta_mm [ eta_ohm ((1 - f_g) T_sky + f_g T_ground) + (1 - eta_ohm) T_feed ].

    Antenna temperature at the feed output terminals.  The scene the beam sees
    is weighted by Eq. 28; a lossy feed (efficiency eta_ohm, Eq. 18) passes
    eta_ohm of it and adds its own thermal noise (1 - eta_ohm) T_feed (Eq. 3
    referred to the output); a mismatch (Eq. 17) passes eta_mm of all of it.
    Returns the three terms (sky, ground, feed loss); T_A is their sum.
    [Kraus ch. 6; ERA ch. 3; Pozar 10.2 (lossy line at T_phys)]"""
    sky = eta_mm * eta_ohm * (1 - f_g) * np.asarray(t_sky, float)
    ground = np.full_like(sky, eta_mm * eta_ohm * f_g * t_ground)
    feed = np.full_like(sky, eta_mm * (1 - eta_ohm) * t_feed)
    return sky, ground, feed


def system_temperature_k(t_a, t_rec):
    """[Eq. 30]  T_sys = T_A + T_rec, both at the same reference plane.
    [ERA ch. 3]"""
    return np.asarray(t_a, float) + np.asarray(t_rec, float)


# =============================================================================
# 8. Sensitivity
# =============================================================================
def sefd_jy(t_sys_k, a_eff_m2):
    """[Eq. 31]  SEFD = 2 k T_sys / A_eff   (divided by 1e-26 for Jy).

    System equivalent flux density: the flux of a source that would double
    the noise power.  The 2 is because a single-polarisation receiver
    collects half of an unpolarised source's flux.  T_sys and A_eff must be
    referred to the same plane (here: the feed terminals).  [ERA ch. 3]"""
    return 2.0 * K_BOLTZMANN * np.asarray(t_sys_k, float) / np.maximum(a_eff_m2, 1e-30) / JY


def antenna_temperature_per_jy(a_eff_m2):
    """[Eq. 32]  T_A / S = A_eff / (2 k)   (K per Jy).  [ERA ch. 3]"""
    return np.asarray(a_eff_m2, float) * JY / (2.0 * K_BOLTZMANN)


def noise_power_dbm(t_k, bandwidth_hz, gain_db=0.0):
    """[Eq. 33]  P = k T B G   (in dBm: 10 log10(P / 1 mW)).

    Thermal noise power in bandwidth B, amplified by G.  [Nyquist]"""
    p_w = K_BOLTZMANN * np.asarray(t_k, float) * bandwidth_hz * db_to_lin(gain_db)
    return 10.0 * np.log10(p_w * 1e3)


def snr(flux_jy, sefd, bandwidth_hz, integration_s, n_pol=1, k_s=1.0):
    """[Eq. 34]  SNR = (S / SEFD) sqrt(n_pol B tau) / K_s.

    The radiometer equation, sigma_T = K_s T_sys / sqrt(B tau), expressed in
    flux units (divide by A_eff/2k, Eq. 32).  B = bandwidth integrated over
    (Eq. 12 for continuum, a channel or a line width for spectral work),
    tau = integration time, n_pol = independent polarisations averaged
    (2 gives sqrt 2), K_s = receiver constant (1 total power, 2 Dicke-switched).
    [Dicke; ERA ch. 3, "the radiometer equation"; Kraus ch. 7 (K_s)]"""
    return (np.asarray(flux_jy, float) / np.asarray(sefd, float)
            * np.sqrt(n_pol * np.asarray(bandwidth_hz, float) * np.asarray(integration_s, float))
            / k_s)


def required_flux_jy(target_snr, sefd, bandwidth_hz, integration_s, n_pol=1, k_s=1.0):
    """[Eq. 34, solved for S]  S = SNR K_s SEFD / sqrt(n_pol B tau).

    "I want SNR 3": the flux a source needs -- the number to filter a
    catalogue with."""
    return (np.asarray(target_snr, float) * k_s * np.asarray(sefd, float)
            / np.sqrt(n_pol * np.asarray(bandwidth_hz, float) * np.asarray(integration_s, float)))


def required_integration_s(target_snr, flux_jy, sefd, bandwidth_hz, n_pol=1, k_s=1.0):
    """[Eq. 34, solved for tau]  tau = (SNR K_s SEFD / S)^2 / (n_pol B)."""
    return ((np.asarray(target_snr, float) * k_s * np.asarray(sefd, float)
             / np.asarray(flux_jy, float)) ** 2 / (n_pol * np.asarray(bandwidth_hz, float)))


def doppler_bandwidth_hz(velocity_kms, f0_hz=HI_REST_HZ):
    """[Eq. 35]  Delta f = (Delta v / c) f0.

    First-order Doppler width of a line of velocity width Delta v (also the
    velocity resolution of a channel, read backwards).  Exact to 1e-3 for
    the |v| < 300 km/s of Galactic and nearby-galaxy HI.  [ERA ch. 7]"""
    return np.asarray(velocity_kms, float) * 1e3 / C_LIGHT * f0_hz


# =============================================================================
# 9. Where the beam points
# =============================================================================
def beam_declination_deg(lat_deg, alt_deg, az_deg):
    """[Eq. 36]  sin(dec) = sin(lat) sin(alt) + cos(lat) cos(alt) cos(az).

    Declination of a fixed horizontal pointing, azimuth measured from north
    through east.  (Meeus measures azimuth from the south, which flips the
    sign of the last term.)  At the zenith (alt = 90) dec = latitude.
    [Meeus ch. 13]"""
    la, al, az = (math.radians(x) for x in (lat_deg, alt_deg, az_deg))
    return math.degrees(math.asin(math.sin(la) * math.sin(al)
                                  + math.cos(la) * math.cos(al) * math.cos(az)))


def transit_time_s(width_ew_deg, dec_deg):
    """[Eq. 37]  t = theta_EW / (15.041 deg/h * cos dec).

    Time for a source to drift across the beam's east-west width: the sky
    turns 360 deg per sidereal day, which moves a source at declination dec
    across the sky at 15.041 cos(dec) deg/h.  At the zenith, local north is
    +dec and east is +RA, and a fixed mount never rotates the beam, so the
    beam's east-west width sets this and its north-south width sets the
    width of the declination strip swept.  [Meeus ch. 12]"""
    rate = SIDEREAL_RATE_DEG_PER_H * math.cos(math.radians(dec_deg))
    return width_ew_deg / rate * 3600.0 if rate > 0 else float("inf")


def rule_of_thumb_hpbw_deg(diameter_m: float, f_hz: float) -> float:
    """[Eq. 38a]  HPBW ~ 70 lambda / D  (degrees).

    Quick estimate for a dish with a typical (~ -10 dB) edge taper, about
    1.2 lambda/D radians; uniform illumination gives 58.9 lambda/D.  Printed
    next to the computed beam for comparison only.  [Kraus ch. 6; ERA ch. 3]"""
    return 70.0 * float(wavelength_m(f_hz)) / diameter_m


def rule_of_thumb_gain_dbi(diameter_m: float, f_hz: float) -> float:
    """[Eq. 38b]  G ~ 6 (D / lambda)^2 = eta (pi D / lambda)^2 with eta ~ 0.6.

    The estimate in the draft performance_parameters notes.  For comparison
    only.  [Balanis ch. 2 & 15 (G = eta 4 pi A / lambda^2)]"""
    return float(lin_to_db(6.0 * (diameter_m / wavelength_m(f_hz)) ** 2))


# =============================================================================
# 10. The computation
# =============================================================================
@dataclass
class AntennaSide:
    """Everything that depends on the antenna only (not the RF chain)."""

    wavelength_m: float
    q_e: float
    q_h: float
    feed_band: tuple[float | None, float | None]
    theta0_rad: float | None
    edge_illumination_db: tuple[float, float] | None
    eta: dict[str, float]
    directivity_lin: float
    beam: Beam
    beam_directivity_lin: float
    beam_solid_angle_sr: float
    main_beam_solid_angle_sr: float
    hpbw_e_deg: float
    hpbw_h_deg: float
    sidelobe_e_db: float | None
    sidelobe_h_db: float | None
    ground_fraction: float


@dataclass
class StageResult:
    component: ComponentSpec
    gain_db: np.ndarray
    noise_temp_k: np.ndarray
    gain_before_lin: np.ndarray
    contribution_k: np.ndarray
    cumulative_gain_db: np.ndarray


@dataclass
class Spectrum:
    """Every frequency-dependent quantity on one frequency grid."""

    freq_hz: np.ndarray
    stages: list[StageResult]
    feed_response_db: np.ndarray
    response_db: np.ndarray        # product of every response, feed + SDR included
    gain_to_sdr_db: np.ndarray
    t_rec_k: np.ndarray
    t_sky_k: np.ndarray
    t_a_terms: tuple[np.ndarray, np.ndarray, np.ndarray]
    t_a_k: np.ndarray
    t_sys_k: np.ndarray
    a_eff_m2: np.ndarray
    sefd_jy: np.ndarray


@dataclass
class Performance:
    hw: Hardware
    antenna: AntennaSide
    sweep: Spectrum                 # the plotting range
    dense: Spectrum                 # fine grid used for the passband
    passband: Passband
    index: int                      # position of the centre frequency in sweep
    channel_hz: float
    beam_dec_deg: float
    hpbw_ns_deg: float
    hpbw_ew_deg: float
    transit_s: float
    noise_power_sdr_dbm: float
    lna_output_noise_dbm: float | None
    lna_headroom_db: float | None

    def at_center(self, name: str) -> float:
        return float(getattr(self.sweep, name)[self.index])

    @property
    def t_sys_k(self) -> float:
        return self.at_center("t_sys_k")

    @property
    def sefd_jy(self) -> float:
        return self.at_center("sefd_jy")

    @property
    def a_eff_m2(self) -> float:
        return self.at_center("a_eff_m2")

    @property
    def noise_bandwidth_hz(self) -> float:
        return self.passband.noise_bandwidth_hz

    def required_flux_jy(self, target_snr, integration_s, bandwidth_hz=None):
        b = self.noise_bandwidth_hz if bandwidth_hz is None else bandwidth_hz
        return required_flux_jy(target_snr, self.sefd_jy, b, integration_s,       # Eq. 34
                                self.hw.n_pol, self.hw.k_s)

    def snr(self, flux_jy, integration_s, bandwidth_hz=None):
        b = self.noise_bandwidth_hz if bandwidth_hz is None else bandwidth_hz
        return snr(flux_jy, self.sefd_jy, b, integration_s, self.hw.n_pol, self.hw.k_s)


def antenna_side(hw: Hardware) -> AntennaSide:
    """Steps 1-5: the antenna, at the centre frequency."""
    feed, refl, f0 = hw.feed, hw.reflector, hw.center_hz

    # Step 1 -- wavelength.                                                Eq. 13
    lam = float(wavelength_m(f0))

    # Step 2 -- feed pattern exponents, one per principal plane.     Eqs. 19, 14
    theta0 = rim_half_angle_rad(refl.focal_ratio) if refl else None
    q = []
    for hpbw, taper in ((feed.hpbw_e_deg, feed.edge_taper_e_db),
                        (feed.hpbw_h_deg, feed.edge_taper_h_db)):
        q.append(cosq_exponent_from_hpbw(hpbw) if hpbw is not None
                 else cosq_exponent_from_taper(taper, theta0))
    q_e, q_h = q

    # Step 3 -- the feed's band: explicit edges, else waveguide cut-offs. Eq. 16
    lo, hi = feed.freq_min_hz, feed.freq_max_hz
    if feed.waveguide_diameter_m is not None:
        te11, tm01 = circular_waveguide_band_hz(feed.waveguide_diameter_m)
        lo, hi = (lo if lo is not None else te11), (hi if hi is not None else tm01)

    # Step 4 -- efficiencies and directivity.          Eqs. 15, 17, 18, 20-24
    eta = {"ohmic": ohmic_efficiency(feed.ohmic_loss_db),                       # Eq. 18
           "mismatch": mismatch_efficiency(feed.vswr),                          # Eq. 17
           "forward": 1.0 - feed.back_fraction}
    edge, shadow = None, 0.0
    if refl is not None:
        rho, a_f, b_f = aperture_fields((q_e, q_h), refl)                       # Eqs. 15, 20
        ill, pol, blk, shadow = illumination_polarization_blockage(             # Eq. 22
            rho, a_f, b_f, refl.blockage_fraction)
        eta.update(spillover=spillover_efficiency(q_e, q_h, theta0),            # Eq. 21
                   illumination=ill, polarization=pol, blockage=blk,
                   surface=float(ruze_efficiency(refl.surface_rms_m, f0)))      # Eq. 23
        # Directivity of the radiated beam: 4 pi A / lambda^2 x the parts of
        # Eq. 22 that shape the beam, per unit of power actually radiated.
        a_geom = math.pi * (refl.diameter_m / 2) ** 2
        directivity = 4 * math.pi * a_geom / lam ** 2 * ill * pol * blk / (1 - shadow)
        # Edge illumination = feed taper + spherical spreading at the rim.  Eqs. 15, 20
        edge = tuple(10 * qq * math.log10(math.cos(theta0))
                     + 20 * math.log10(math.cos(theta0 / 2) ** 2) for qq in (q_e, q_h))
    else:
        directivity = feed_pattern_norm(q_e, q_h)                               # Eq. 15
    # Gain = directivity x the efficiencies that do NOT shape the beam
    # (illumination, polarization and blockage are already inside the
    # directivity above, so they must not be applied twice).              Eq. 24
    eta["gain_factor"] = float(np.prod([eta[k] for k in
                                        ("ohmic", "mismatch", "forward", "spillover", "surface")
                                        if k in eta]) * (1 - shadow))
    # Aperture efficiency A_eff / A_geom = every sub-efficiency multiplied.
    eta["aperture"] = float(np.prod([v for k, v in eta.items() if k != "gain_factor"]))

    # Step 5 -- the 3-D beam and its measures.                          Eqs. 25, 26
    beam = far_field((q_e, q_h), refl, f0, feed.e_plane_azimuth_deg)
    omega_a = beam_solid_angle_sr(beam)
    p_e, p_h = beam.cut(0.0), beam.cut(90.0)
    return AntennaSide(
        wavelength_m=lam, q_e=q_e, q_h=q_h, feed_band=(lo, hi), theta0_rad=theta0,
        edge_illumination_db=edge, eta=eta, directivity_lin=directivity, beam=beam,
        beam_directivity_lin=directivity_from_solid_angle(omega_a),
        beam_solid_angle_sr=omega_a, main_beam_solid_angle_sr=main_beam_solid_angle_sr(beam),
        hpbw_e_deg=half_power_beamwidth_deg(beam.theta_deg, p_e),
        hpbw_h_deg=half_power_beamwidth_deg(beam.theta_deg, p_h),
        sidelobe_e_db=first_sidelobe_db(p_e), sidelobe_h_db=first_sidelobe_db(p_h),
        ground_fraction=ground_fraction((q_e, q_h), feed.back_fraction, theta0,  # Eq. 28
                                        hw.environment.horizon_deg),
    )


def spectrum(hw: Hardware, ant: AntennaSide, f: np.ndarray) -> Spectrum:
    """Steps 6-10: everything that varies with frequency, on grid ``f``."""
    env, feed = hw.environment, hw.feed

    # Step 6 -- each component's gain and noise temperature.     Eqs. 2, 3, 6-9
    gains = [component_gain_db(c, f) for c in hw.chain]
    temps = [component_noise_temp_k(c, f, g) for c, g in zip(hw.chain, gains)]

    # Step 7 -- Friis cascade and cumulative gain, in chain order.       Eqs. 4, 5
    contributions, befores = friis_cascade(temps, gains)
    stages, cum = [], np.zeros_like(f)
    for c, g, t, gb, ct in zip(hw.chain, gains, temps, befores, contributions):
        cum = cum + g
        stages.append(StageResult(c, g, t, gb, ct, cum.copy()))
    t_rec = sum(contributions)
    # Total analog gain: everything up to (not including) the SDR.            Eq. 5
    analog = []
    for c, g in zip(hw.chain, gains):
        if c.type == "sdr":
            break
        analog.append(g)
    gain_to_sdr = cascade_gain_db(analog) if analog else np.zeros_like(f)

    # Step 8 -- overall response: feed band x every stage (incl. SDR).    Eqs. 5, 8
    feed_resp = -datasheet_edge_rolloff_db(f, *ant.feed_band, feed.rolloff_db_per_octave)
    response = cascade_gain_db(gains) + feed_resp

    # Step 9 -- antenna temperature at the feed terminals.            Eqs. 27-29
    t_sky = sky_temperature_k(f, env.cmb_k, env.t408_k, env.spectral_index, env.atmosphere_k)
    terms = antenna_temperature_terms(t_sky, env.ground_k, ant.ground_fraction,
                                      ant.eta["ohmic"], ant.eta["mismatch"], feed.t_phys_k)
    t_a = sum(terms)

    # Step 10 -- T_sys, effective area and SEFD.                    Eqs. 30, 23, 24, 31
    t_sys = system_temperature_k(t_a, t_rec)
    eff = ant.eta["gain_factor"] * db_to_lin(feed_resp)
    if hw.reflector is not None:
        # Ruze (Eq. 23) and the dish directivity (4 pi A / lambda^2) both
        # change with frequency; re-evaluate them across the grid.
        eff = eff / ant.eta["surface"] * ruze_efficiency(hw.reflector.surface_rms_m, f)
        d = ant.directivity_lin * (f / hw.center_hz) ** 2
    else:
        d = np.full_like(f, ant.directivity_lin)        # cos^q pattern: no lambda dependence
    _, a_eff = gain_and_effective_area(d, eff, f)                                 # Eq. 24
    return Spectrum(f, stages, feed_resp, response, gain_to_sdr, t_rec, t_sky, terms,
                    t_a, t_sys, a_eff, sefd_jy(t_sys, a_eff))                      # Eq. 31


def evaluate(hw: Hardware, *, reuse: Performance | None = None) -> Performance:
    """The whole calculation: hardware parameters -> performance parameters.

    ``reuse`` borrows the antenna side (Steps 1-5) from an earlier evaluation
    of the same antenna; a chain-order sweep only changes Steps 6-12."""
    # Steps 1-5 -- antenna (wavelength, feed, efficiencies, 3-D beam).
    ant = reuse.antenna if reuse is not None else antenna_side(hw)

    # Step 11a -- nominal passband: intersect every part's band.             Eq. 10
    bands = [(c.id, *component_band_hz(c)) for c in hw.chain] + [("feed", *ant.feed_band)]
    lo, hi, lo_by, hi_by = intersect_bands(bands)
    if not lo < hw.center_hz < hi:
        raise HardwareConfigError(
            f"the centre frequency {hw.center_hz/1e6:.3f} MHz is outside the chain's "
            f"nominal passband {lo/1e6:.2f}-{hi/1e6:.2f} MHz (set by {lo_by} and {hi_by})")

    # Step 11b -- the rigorous passband on a fine grid spanning it.     Eqs. 11, 12
    w = hi - lo
    dense = spectrum(hw, ant, np.linspace(max(lo - 0.75 * w, 1e6), hi + 0.75 * w, 30001))
    pb = measure_passband(dense.freq_hz, dense.response_db, hw.center_hz, (lo, hi, lo_by, hi_by))

    # Steps 6-10 again on the plotting grid (with the exact centre included).
    f0, f1, n = hw.sweep_hz
    grid = np.unique(np.concatenate([np.linspace(f0, f1, n), [hw.center_hz]]))
    sw = spectrum(hw, ant, grid)
    i = int(np.argmin(np.abs(grid - hw.center_hz)))

    # Step 12 -- where the beam points, and drift geometry.            Eqs. 36, 37
    dec = beam_declination_deg(hw.environment.latitude_deg, hw.pointing_alt_deg,
                               hw.pointing_az_deg)
    ns = half_power_beamwidth_deg(ant.beam.theta_deg, ant.beam.power_azimuth(        # Eq. 26a
        ant.beam.theta_deg, 0.0))
    ew = half_power_beamwidth_deg(ant.beam.theta_deg, ant.beam.power_azimuth(
        ant.beam.theta_deg, 90.0))

    # Spectrometer channel: SDR bandwidth / FFT length (or as given).
    sdr = hw.sdr
    chan = hw.channel_width_hz
    if chan is None:
        lo_s, hi_s = sdr_band_hz(sdr)
        chan = (hi_s - lo_s) / hw.fft_size

    # Dynamic range: noise power at the SDR input, and at the first
    # amplifier's output over that amplifier's whole band (nothing ahead of
    # it filters).                                                        Eq. 33
    p_sdr = float(noise_power_dbm(sw.t_sys_k[i], pb.noise_bandwidth_hz, sw.gain_to_sdr_db[i]))
    p_lna = headroom = None
    k = next((k for k, s in enumerate(sw.stages) if s.component.type == "amplifier"), None)
    if k is not None:
        lna = sw.stages[k].component
        a, b = lna.params.number("freq_min", "Hz"), lna.params.number("freq_max", "Hz")
        p1db = lna.params.number("p1db_output", "dBm")
        if a and b:
            t_in = sw.t_a_k[i] + sum(s.contribution_k[i] for s in sw.stages[:k + 1])
            p_lna = float(noise_power_dbm(t_in, b - a, sw.stages[k].cumulative_gain_db[i]))
            headroom = None if p1db is None else p1db - p_lna

    return Performance(hw=hw, antenna=ant, sweep=sw, dense=dense, passband=pb, index=i,
                       channel_hz=chan, beam_dec_deg=dec, hpbw_ns_deg=ns, hpbw_ew_deg=ew,
                       transit_s=transit_time_s(ew, dec),                          # Eq. 37
                       noise_power_sdr_dbm=p_sdr, lna_output_noise_dbm=p_lna,
                       lna_headroom_db=headroom)
