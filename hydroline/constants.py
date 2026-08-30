"""Physical constants and 21 cm specific numbers.

All SI unless the name says otherwise.
"""
from __future__ import annotations

# --- fundamental -----------------------------------------------------------
C_LIGHT = 299_792_458.0          # m / s                (exact, SI)
K_BOLTZMANN = 1.380_649e-23      # J / K                (exact, SI 2019)
JY = 1e-26                       # W / m^2 / Hz per jansky

# --- the hydrogen line -----------------------------------------------------
# Rest frequency of the neutral-hydrogen hyperfine (spin-flip) transition.
# Value adopted by the IAU; good to ~1 mHz.
HI_REST_HZ = 1_420_405_751.768
HI_REST_M = C_LIGHT / HI_REST_HZ  # 0.2110611405 m  (21.1 cm)

# Protected radio-astronomy band around the line (ITU-R RR 5.340):
# 1400-1427 MHz is allocated to radio astronomy with all emissions prohibited.
HI_PROTECTED_BAND_HZ = (1.400e9, 1.427e9)

# --- local standard of rest ------------------------------------------------
# "Standard solar motion" (the kinematic LSR used throughout HI astronomy):
# the Sun moves at 20.0 km/s toward RA 18h00m, Dec +30 deg in B1900 coords.
LSR_SOLAR_SPEED_KMS = 20.0
LSR_APEX_RA_B1900_HOURS = 18.0
LSR_APEX_DEC_B1900_DEG = 30.0

# --- rough sky brightness --------------------------------------------------
# 21 cm continuum background contributions, for T_sys budgeting.
T_CMB_K = 2.725
# Galactic synchrotron at 1.4 GHz: ~1-3 K off the plane, tens of K in the plane.
T_GALACTIC_HIGH_LAT_K = 2.0

# Order-of-magnitude Galactic HI peak brightness temperature (LAB / EBHIS
# survey scale).  Used only for a coarse "what will my spectrum look like"
# estimate -- NOT a substitute for a real survey lookup.
HI_TB_PLANE_K = 100.0        # |b| ~ 0, optically thick, self-absorbed
HI_TB_HIGH_LAT_K = 12.0      # |b| > 40 deg typical peak
HI_TB_SCALE_HEIGHT_DEG = 9.0 # exponential falloff scale in |b|

# --- HI mass conversion ----------------------------------------------------
# M_HI [Msun] = 2.356e5 * D[Mpc]^2 * S_int[Jy km/s]   (optically thin)
HI_MASS_COEFF = 2.356e5
