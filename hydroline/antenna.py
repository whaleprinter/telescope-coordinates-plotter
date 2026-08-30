"""Beam geometry and radiometric sensitivity derived from :class:`Antenna`.

The formulas here are the standard single-dish relations (Kraus, *Radio
Astronomy*; Condon & Ransom, *Essential Radio Astronomy* ch. 3 & 8).  Every
derived quantity is a property, so nothing is stale after you edit the config.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import Antenna
from .constants import C_LIGHT, JY, K_BOLTZMANN

#: Apparent sidereal drift rate of the sky in right ascension, deg / hour.
SIDEREAL_RATE_DEG_PER_HOUR = 360.0 / 23.9344696  # = 15.0411 deg/hr


@dataclass(frozen=True)
class AntennaModel:
    """Everything derivable from an :class:`Antenna` at its centre frequency."""

    antenna: Antenna

    # ---------------------------------------------------------------- basics
    @property
    def frequency_hz(self) -> float:
        return self.antenna.center_freq_hz

    @property
    def wavelength_m(self) -> float:
        return C_LIGHT / self.frequency_hz

    # ------------------------------------------------------------ beam shape
    @property
    def hpbw_e_deg(self) -> float:
        """Half-power beamwidth in the E-plane (or the only plane, if circular)."""
        a = self.antenna
        lam = self.wavelength_m
        if a.kind == "beamwidth":
            return float(a.hpbw_deg)
        if a.kind == "horn":
            return a.horn_factor_e_deg * lam / a.aperture_e_m
        return a.beam_factor_deg * lam / a.diameter_m

    @property
    def hpbw_h_deg(self) -> float:
        """Half-power beamwidth in the H-plane."""
        a = self.antenna
        lam = self.wavelength_m
        if a.kind == "horn":
            return a.horn_factor_h_deg * lam / a.aperture_h_m
        return self.hpbw_e_deg

    @property
    def hpbw_deg(self) -> float:
        """Circularly-equivalent HPBW: the geometric mean of the two planes.

        This is the single number used for "is the source in the beam?".
        """
        return math.sqrt(self.hpbw_e_deg * self.hpbw_h_deg)

    @property
    def beam_radius_deg(self) -> float:
        """Half-power *radius* -- the usual definition of the beam edge."""
        return 0.5 * self.hpbw_deg

    @property
    def main_beam_solid_angle_sr(self) -> float:
        """Main-beam solid angle for a Gaussian beam: 1.133 * theta_E * theta_H."""
        te = math.radians(self.hpbw_e_deg)
        th = math.radians(self.hpbw_h_deg)
        return 1.133 * te * th

    # ------------------------------------------------------------ collecting
    @property
    def geometric_area_m2(self) -> float:
        a = self.antenna
        if a.kind == "horn":
            return a.aperture_e_m * a.aperture_h_m
        if a.kind == "beamwidth" and not a.diameter_m:
            return float("nan")
        return math.pi * (a.diameter_m / 2.0) ** 2

    @property
    def effective_area_m2(self) -> float:
        """A_eff = eta_a * A_geom, or from the beam solid angle if no aperture.

        The fallback A_eff = lambda^2 / Omega_MB assumes *all* radiated power
        sits in the main beam, so it overestimates A_eff (typically by 1/0.7)
        for a real feed.  Give a physical aperture when you can.
        """
        a = self.antenna
        geom = self.geometric_area_m2
        if a.kind == "beamwidth" and (not a.diameter_m or math.isnan(geom)):
            return self.wavelength_m ** 2 / self.main_beam_solid_angle_sr
        return a.aperture_efficiency * geom

    @property
    def beam_solid_angle_sr(self) -> float:
        """Total (all-sky) beam solid angle: Omega_A = lambda^2 / A_eff."""
        return self.wavelength_m ** 2 / self.effective_area_m2

    @property
    def main_beam_efficiency(self) -> float:
        """eta_MB = Omega_MB / Omega_A -- the fraction of power in the main lobe."""
        return self.main_beam_solid_angle_sr / self.beam_solid_angle_sr

    @property
    def gain_dbi(self) -> float:
        g = 4.0 * math.pi * self.effective_area_m2 / self.wavelength_m ** 2
        return 10.0 * math.log10(g)

    @property
    def sensitivity_k_per_jy(self) -> float:
        """Antenna temperature produced by 1 Jy of unresolved flux (K/Jy).

        T_A = S * A_eff / (2 k).  The 2 is because a single receiver channel
        sees one polarisation of an unpolarised source.
        """
        return self.effective_area_m2 * JY / (2.0 * K_BOLTZMANN)

    # ----------------------------------------------------------- sensitivity
    @property
    def sefd_jy(self) -> float:
        """System equivalent flux density: the flux that doubles the noise power."""
        return 2.0 * K_BOLTZMANN * self.antenna.t_sys_k / self.effective_area_m2 / JY

    def delta_t_rms_k(self, bandwidth_hz: float | None = None,
                      integration_s: float | None = None) -> float:
        """Radiometer equation: dT = K_s * T_sys / sqrt(n_pol * B * tau)."""
        a = self.antenna
        b = a.rf_bandwidth_hz if bandwidth_hz is None else bandwidth_hz
        t = a.integration_s if integration_s is None else integration_s
        return a.radiometer_factor * a.t_sys_k / math.sqrt(max(a.n_pol * b * t, 1e-30))

    def delta_s_rms_jy(self, bandwidth_hz: float | None = None,
                       integration_s: float | None = None) -> float:
        """Flux-density noise: SEFD / sqrt(n_pol * B * tau)."""
        a = self.antenna
        b = a.rf_bandwidth_hz if bandwidth_hz is None else bandwidth_hz
        t = a.integration_s if integration_s is None else integration_s
        return (a.radiometer_factor * self.sefd_jy
                / math.sqrt(max(a.n_pol * b * t, 1e-30)))

    @property
    def continuum_limit_jy(self) -> float:
        """Minimum detectable continuum flux at the configured SNR threshold."""
        return self.antenna.snr_threshold * self.delta_s_rms_jy()

    @property
    def line_limit_jy(self) -> float:
        """Minimum detectable flux in ONE spectral channel."""
        return self.antenna.snr_threshold * self.delta_s_rms_jy(
            bandwidth_hz=self.antenna.channel_hz
        )

    @property
    def line_limit_k(self) -> float:
        """Minimum detectable brightness temperature in one channel."""
        return self.antenna.snr_threshold * self.delta_t_rms_k(
            bandwidth_hz=self.antenna.channel_hz
        )

    # ------------------------------------------------------------- spectral
    @property
    def velocity_resolution_kms(self) -> float:
        """Channel width expressed as a Doppler velocity at the HI line."""
        return C_LIGHT * self.antenna.channel_hz / self.frequency_hz / 1000.0

    @property
    def velocity_coverage_kms(self) -> float:
        """Total velocity span covered by the RF bandwidth (+/- half of this)."""
        return C_LIGHT * self.antenna.rf_bandwidth_hz / self.frequency_hz / 1000.0

    @property
    def n_channels(self) -> int:
        return int(round(self.antenna.rf_bandwidth_hz / self.antenna.channel_hz))

    # ------------------------------------------------------------ drift scan
    def transit_seconds(self, dec_deg: float) -> float:
        """How long a source at ``dec_deg`` stays inside the HPBW as it drifts.

        The sky moves through the fixed beam at 15.04 deg/hr in right ascension,
        which projects to 15.04*cos(dec) deg/hr on the sky.  This is the natural
        integration time for a transit observation.
        """
        rate = SIDEREAL_RATE_DEG_PER_HOUR * math.cos(math.radians(dec_deg))
        if rate <= 0:
            return float("inf")
        return self.hpbw_deg / rate * 3600.0

    def beam_dilution(self, source_size_deg: float) -> float:
        """Fraction of a source's *brightness temperature* that survives.

        Gaussian source convolved with a Gaussian beam:
        f = theta_s^2 / (theta_s^2 + theta_b^2).  Approaches 1 for sources much
        larger than the beam, and falls as (theta_s/theta_b)^2 for small ones.
        """
        s2 = np.asarray(source_size_deg, dtype=float) ** 2
        return s2 / (s2 + self.hpbw_deg ** 2)

    def flux_coupling(self, source_size_deg: float) -> float:
        """Fraction of a source's *total flux* that lands inside the main beam.

        The complement of :meth:`beam_dilution`:
        theta_b^2 / (theta_b^2 + theta_s^2).  It is 1 for a source much smaller
        than the beam (all the flux is collected) and falls as (theta_b/theta_s)^2
        once the source overflows the beam.  Flux is what is conserved, so this
        is the factor to apply before any sensitivity comparison.
        """
        s2 = np.asarray(source_size_deg, dtype=float) ** 2
        b2 = self.hpbw_deg ** 2
        return b2 / (b2 + s2)

    def antenna_temperature_from_flux_k(self, flux_jy: float,
                                        source_size_deg: float = 0.0) -> float:
        """T_A from a source's total flux density.

        T_A = S * f_coupling * A_eff / (2k).  The factor of 2 is because one
        receiver channel sees one polarisation of an unpolarised source.
        """
        s_in_beam = np.asarray(flux_jy, dtype=float) * self.flux_coupling(
            source_size_deg
        )
        return s_in_beam * self.sensitivity_k_per_jy

    def antenna_temperature_from_tb_k(self, tb_k: float,
                                      source_size_deg: float) -> float:
        """T_A for an extended source of brightness temperature ``tb_k``.

        A fully-resolved source gives T_A = eta_MB * T_b, not T_b: the power
        that lands in the sidelobes never reaches the main beam.  This is the
        same physics as :meth:`antenna_temperature_from_flux_k` -- the two agree
        exactly in both the point-source and fully-resolved limits.
        """
        return (np.asarray(tb_k, dtype=float)
                * self.beam_dilution(source_size_deg)
                * self.main_beam_efficiency)

    def matched_bandwidth_hz(self, line_width_kms: float) -> float:
        """Bandwidth spanned by a spectral line of the given velocity width.

        Integrating a line over its own width is the matched filter for
        detecting it, and is how a real HI detection is actually made -- one
        5 kHz channel of an M31 profile is hopeless, the summed profile is not.
        """
        return (abs(float(line_width_kms)) * 1000.0 / C_LIGHT
                * self.frequency_hz)

    # ------------------------------------------------------------- confusion
    @property
    def confusion_noise_jy(self) -> float:
        """Classical confusion noise from unresolved background sources.

        Condon's 1.4 GHz relation, sigma_c ~ 0.2 mJy * (theta/arcmin)^2, which
        is calibrated for beams of arcseconds to a few arcminutes.  For the
        many-degree beams typical of a student HI instrument this is a wild
        extrapolation and should be read as "continuum source photometry is
        hopeless, the beam is full of blended sources" rather than as a number.
        """
        theta_arcmin = self.hpbw_deg * 60.0
        return 0.2e-3 * theta_arcmin ** 2

    @property
    def confusion_limited(self) -> bool:
        return self.confusion_noise_jy > self.delta_s_rms_jy()

    # ------------------------------------------------------------ reporting
    def beam_profile(self, offset_deg: np.ndarray) -> np.ndarray:
        """Normalised Gaussian power pattern vs. offset from the beam axis."""
        sigma = self.hpbw_deg / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        return np.exp(-0.5 * (np.asarray(offset_deg) / sigma) ** 2)

    def summary(self) -> dict[str, Any]:
        """Flat dict of every derived quantity, ready for JSON/CSV export."""
        a = self.antenna
        return {
            "name": a.name,
            "kind": a.kind,
            "center_frequency_mhz": self.frequency_hz / 1e6,
            "wavelength_cm": self.wavelength_m * 100.0,
            "aperture_description": self._aperture_text(),
            "hpbw_deg": self.hpbw_deg,
            "hpbw_e_deg": self.hpbw_e_deg,
            "hpbw_h_deg": self.hpbw_h_deg,
            "beam_radius_deg": self.beam_radius_deg,
            "geometric_area_m2": self.geometric_area_m2,
            "aperture_efficiency": a.aperture_efficiency,
            "effective_area_m2": self.effective_area_m2,
            "gain_dbi": self.gain_dbi,
            "main_beam_solid_angle_sr": self.main_beam_solid_angle_sr,
            "beam_solid_angle_sr": self.beam_solid_angle_sr,
            "main_beam_efficiency": self.main_beam_efficiency,
            "sensitivity_k_per_jy": self.sensitivity_k_per_jy,
            "t_sys_k": a.t_sys_k,
            "sefd_jy": self.sefd_jy,
            "n_pol": a.n_pol,
            "rf_bandwidth_mhz": a.rf_bandwidth_hz / 1e6,
            "channel_khz": a.channel_hz / 1e3,
            "n_channels": self.n_channels,
            "integration_s": a.integration_s,
            "velocity_resolution_kms": self.velocity_resolution_kms,
            "velocity_coverage_kms": self.velocity_coverage_kms,
            "delta_t_rms_continuum_k": self.delta_t_rms_k(),
            "delta_t_rms_channel_k": self.delta_t_rms_k(bandwidth_hz=a.channel_hz),
            "delta_s_rms_continuum_jy": self.delta_s_rms_jy(),
            "delta_s_rms_channel_jy": self.delta_s_rms_jy(bandwidth_hz=a.channel_hz),
            "snr_threshold": a.snr_threshold,
            "continuum_limit_jy": self.continuum_limit_jy,
            "line_limit_jy": self.line_limit_jy,
            "line_limit_k": self.line_limit_k,
            "confusion_noise_jy": self.confusion_noise_jy,
            "confusion_limited": self.confusion_limited,
            "pointing_alt_deg": a.pointing_alt_deg,
            "pointing_az_deg": a.pointing_az_deg,
        }

    def _aperture_text(self) -> str:
        a = self.antenna
        if a.kind == "horn":
            return f"{a.aperture_e_m:.2f} x {a.aperture_h_m:.2f} m rectangular"
        if a.kind == "beamwidth" and not a.diameter_m:
            return "specified by beamwidth only"
        return f"{a.diameter_m:.2f} m circular"

    def describe(self) -> str:
        """Human-readable block for the console."""
        s = self.summary()
        lines = [
            f"Antenna : {s['name']}  ({s['aperture_description']})",
            f"  Frequency        {s['center_frequency_mhz']:.4f} MHz"
            f"   (lambda = {s['wavelength_cm']:.2f} cm)",
            f"  HPBW             {s['hpbw_deg']:.2f} deg"
            + (f"   (E {s['hpbw_e_deg']:.2f} x H {s['hpbw_h_deg']:.2f})"
               if abs(s['hpbw_e_deg'] - s['hpbw_h_deg']) > 1e-6 else ""),
            f"  Effective area   {s['effective_area_m2']:.3f} m^2"
            f"   (gain {s['gain_dbi']:.1f} dBi, {s['sensitivity_k_per_jy']*1e3:.3f} mK/Jy)",
            f"  Main-beam eff.   {s['main_beam_efficiency']:.2f}",
            f"  T_sys / SEFD     {s['t_sys_k']:.0f} K / {s['sefd_jy']:.3e} Jy",
            f"  Spectrometer     {s['rf_bandwidth_mhz']:.2f} MHz in "
            f"{s['n_channels']} x {s['channel_khz']:.2f} kHz channels"
            f"  ({s['velocity_resolution_kms']:.2f} km/s res,"
            f" {s['velocity_coverage_kms']:.0f} km/s span)",
            f"  Noise ({s['integration_s']:.0f} s)  continuum "
            f"{s['delta_t_rms_continuum_k']*1e3:.3f} mK"
            f" / channel {s['delta_t_rms_channel_k']:.4f} K",
            f"  {s['snr_threshold']:.0f}-sigma limits  continuum "
            f"{s['continuum_limit_jy']:.3f} Jy"
            f" / line {s['line_limit_jy']:.1f} Jy ({s['line_limit_k']:.3f} K)",
        ]
        if s["confusion_limited"]:
            lines.append(
                "  NOTE: beam is confusion-limited for continuum work -- "
                "individual continuum sources cannot be separated."
            )
        return "\n".join(lines)
