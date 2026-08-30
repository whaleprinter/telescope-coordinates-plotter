#!/usr/bin/env python3
"""Convenience entry point: `python3 run.py [options]`.

Identical to `python3 -m hydroline`.  See --help for every option.
"""
import sys


def main() -> int:
    try:
        from hydroline.cli import main as cli_main
    except ImportError as exc:
        # hydroline's preflight raises a formatted, multi-line explanation.
        # Print it plainly -- a traceback here only buries the useful part.
        print(exc, file=sys.stderr)
        return 1
    return cli_main()


if __name__ == "__main__":
    sys.exit(main())
