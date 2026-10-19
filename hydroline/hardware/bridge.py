"""Hand a computed performance_parameters.yaml to the hydroline planner.

The planner's :class:`~hydroline.config.Antenna` describes a beam and a
T_sys; the hardware model computes both from parts.  This maps one onto the
other, so ``run.py --performance hardware/output/performance_parameters.yaml``
plans with the beam and noise your actual hardware produces.
"""
from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

from ..config import Antenna
from .params import HardwareConfigError


def _v(block: dict, key: str, path: Path) -> float:
    try:
        return float(block[key]["value"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HardwareConfigError(
            f"{path}: missing {key!r} -- regenerate it with compute_performance.py") from exc


def antenna_from_performance(path: str | Path, base: Antenna | None = None) -> Antenna:
    """An :class:`Antenna` with the beam, A_eff, T_sys and bandwidth of the file.

    The planner models a circular beam, so the HPBW is the geometric mean of
    the E- and H-plane widths; the exact elliptical footprint is in the
    hardware model's beam figure.  A_eff is carried exactly by giving the
    equivalent diameter sqrt(4 A_eff / pi) with an efficiency of 1.
    """
    import yaml

    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    beam, coll = data.get("beam_pattern", {}), data.get("collection", {})
    noise, gain = data.get("noise", {}), data.get("gain", {})
    sysb, pos = data.get("system", {}), data.get("position", {})
    gen = data.get("generated_from", {})

    a_eff = _v(coll, "effective_area", path)
    hpbw = math.sqrt(_v(beam, "hpbw_e_plane", path) * _v(beam, "hpbw_h_plane", path))
    return replace(
        base or Antenna(),
        name=str(gen.get("telescope") or "hardware model"),
        kind="beamwidth",
        hpbw_deg=hpbw,
        diameter_m=math.sqrt(4.0 * a_eff / math.pi),
        aperture_efficiency=1.0,
        center_freq_hz=_v(gen, "center_frequency", path) * 1e6,
        t_sys_k=_v(noise, "t_system", path),
        rf_bandwidth_hz=_v(gain, "noise_bandwidth", path) * 1e6,
        channel_hz=_v(sysb, "spectral_channel", path) * 1e3,
        n_pol=int(_v(sysb, "n_pol", path)),
        radiometer_factor=_v(sysb, "radiometer_factor", path),
        pointing_alt_deg=_v(pos, "pointing_altitude", path),
        pointing_az_deg=_v(pos, "pointing_azimuth", path),
    )
