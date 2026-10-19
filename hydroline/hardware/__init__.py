"""Hardware parameters -> performance parameters.

    hardware_parameters.yaml --compute_performance.py--> performance_parameters.yaml
                                                     +--> frequency_response.csv, figures
    + observing_goals.yaml   ------------------------->  detection_thresholds.yaml

See HARDWARE_MODEL.md at the repository root for how to drive it.
"""
from .params import HardwareConfigError
from .system import (Hardware, Performance, evaluate, load_hardware, required_flux_jy,
                     required_integration_s, snr)

__all__ = ["HardwareConfigError", "Hardware", "Performance", "evaluate", "load_hardware",
           "required_flux_jy", "required_integration_s", "snr"]
