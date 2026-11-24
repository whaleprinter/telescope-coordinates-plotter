"""hardware_model -- a separate program from hydroline.

    hardware_parameters.yaml --compute_performance.py--> output/performance_parameters.yaml
    observing_goals.yaml     ------------------------->  output/detection_thresholds.yaml
                                                     +->  output/frequency_response.csv, *.png

Modules
    inputs.py       read the YAML, convert units, validate        (no physics)
    performance.py  EVERY equation, numbered and referenced       (all the physics)
    report.py       write the YAML/CSV outputs                    (no physics)
    plots.py        draw the figures                              (no physics)
    compute_performance.py  the command-line entry point

It does not import hydroline; hydroline reads the performance file it writes.
See README.md in this folder.
"""
from .inputs import HardwareConfigError, load_goals, load_hardware
from .performance import (Performance, evaluate, required_flux_jy, required_integration_s,
                          snr)

__all__ = ["HardwareConfigError", "Performance", "evaluate", "load_goals", "load_hardware",
           "required_flux_jy", "required_integration_s", "snr"]
