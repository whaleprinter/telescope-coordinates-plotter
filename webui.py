#!/usr/bin/env python3
"""Convenience entry point: `python3 webui.py [options]`.

Starts a local browser-based editor for the hydroline configuration: load
site/antenna/observation parameters from a config file, edit them in a form,
and hit Run to re-execute the pipeline and view the regenerated report in
place. See `python3 webui.py --help` for options; --config picks the file the
editor reads from and saves to (default config.local.yaml).

Needs Flask in addition to hydroline's normal dependencies:

    python3 -m pip install flask
"""
import sys


def main() -> int:
    try:
        from hydroline.webapp import main as webapp_main
    except ImportError as exc:
        print(exc, file=sys.stderr)
        return 1
    return webapp_main()


if __name__ == "__main__":
    sys.exit(main())
