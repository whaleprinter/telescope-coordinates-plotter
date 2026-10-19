"""Local interactive editor for the hydroline configuration.

Serves a form pre-filled from a config file, lets you edit site / antenna /
observation parameters in the browser, and a "Run" button saves the edits and
re-executes the same pipeline as ``run.py``, then shows the regenerated
``report.html`` in place.  Everything happens on localhost -- there is no
authentication and it is not meant to be exposed off your machine.

    python3 webui.py

needs Flask, which is not in the core dependency list (see requirements.txt).
"""
from __future__ import annotations

import contextlib
import io
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .catalog import DEFAULT_BODIES
from .cli import run as run_pipeline
from .config import Antenna, Configuration, DUKE_PHYSICS, Observation, PRESET_ANTENNAS
from .theme import THEMES

REPO_ROOT = Path(__file__).resolve().parent.parent


def _default_args(outdir: Path, catalog: Path | None, theme: str) -> SimpleNamespace:
    """The fields hydroline.cli.run() reads off ``args``, with run.py's defaults."""
    return SimpleNamespace(
        outdir=outdir,
        catalog=catalog,
        theme=theme,
        bodies=list(DEFAULT_BODIES),
        no_plots=False,
        no_animation=False,
        frames=96,
        include_misses=False,
        offline=False,
        quiet=True,
    )


def _form_to_config_dict(form: dict[str, str]) -> dict[str, Any]:
    """Turn ``site.latitude_deg`` style form field names back into a nested dict.

    Blank strings are dropped so an empty optional field (e.g. a horn aperture
    when kind=dish) comes through as unset rather than as an empty string that
    would fail float coercion.
    """
    out: dict[str, dict[str, str]] = {"site": {}, "antenna": {}, "observation": {}}
    for key, value in form.items():
        if "." not in key or value == "":
            continue
        section, field = key.split(".", 1)
        if section in out:
            out[section][field] = value
    return out


def create_app(config_path: Path, outdir: Path) -> "Flask":
    from flask import Flask, Response, abort, redirect, render_template, request, url_for

    app = Flask(__name__)
    app.config["CONFIG_PATH"] = Path(config_path)
    app.config["OUTDIR"] = Path(outdir)

    def _load_config() -> Configuration:
        path = app.config["CONFIG_PATH"]
        if path.exists():
            return Configuration.load(path)
        return Configuration(site=DUKE_PHYSICS, antenna=PRESET_ANTENNAS["dish-1m5"],
                             observation=Observation())

    @app.get("/")
    def editor():
        cfg = _load_config()
        return render_template(
            "editor.html",
            cfg=cfg.to_dict(),
            presets=PRESET_ANTENNAS,
            preset_fields={k: asdict(v) for k, v in PRESET_ANTENNAS.items()},
            themes=sorted(THEMES),
            default_bodies=DEFAULT_BODIES,
            config_path=str(app.config["CONFIG_PATH"]),
            error=None,
            report_url=None,
        )

    @app.post("/run")
    def run_route():
        data = _form_to_config_dict(request.form)
        try:
            cfg = Configuration.from_dict(data)
        except (ValueError, KeyError) as exc:
            cfg_for_display = _load_config().to_dict()
            # Show what the user typed, not the last-saved file, so nothing
            # they entered gets silently lost behind the error.
            for section, fields in data.items():
                cfg_for_display.setdefault(section, {}).update(fields)
            return render_template(
                "editor.html", cfg=cfg_for_display, presets=PRESET_ANTENNAS,
                preset_fields={k: asdict(v) for k, v in PRESET_ANTENNAS.items()},
                themes=sorted(THEMES), default_bodies=DEFAULT_BODIES,
                config_path=str(app.config["CONFIG_PATH"]), error=str(exc),
                report_url=None,
            ), 400

        cfg.save(app.config["CONFIG_PATH"])
        theme = request.form.get("theme", "light")
        args = _default_args(app.config["OUTDIR"], None, theme)

        log = io.StringIO()
        try:
            with contextlib.redirect_stdout(log):
                summary = run_pipeline(cfg, args)
        except Exception as exc:  # noqa: BLE001 -- surface it in the browser, not a 500 traceback
            return render_template(
                "editor.html", cfg=cfg.to_dict(), presets=PRESET_ANTENNAS,
                preset_fields={k: asdict(v) for k, v in PRESET_ANTENNAS.items()},
                themes=sorted(THEMES), default_bodies=DEFAULT_BODIES,
                config_path=str(app.config["CONFIG_PATH"]),
                error=f"Run failed: {exc}", report_url=None,
            ), 400

        return render_template(
            "result.html", cfg=cfg.to_dict(), summary=summary, log=log.getvalue(),
            report_url=url_for("outputs", filename="report.html"),
        )

    @app.get("/outputs/<path:filename>")
    def outputs(filename):
        from flask import send_from_directory

        outdir = app.config["OUTDIR"]
        if not (outdir / filename).resolve().is_relative_to(outdir.resolve()):
            abort(404)
        return send_from_directory(outdir, filename)

    return app


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        description="Interactive local editor for the hydroline configuration.")
    p.add_argument("--config", type=Path, default=REPO_ROOT / "config.local.yaml",
                   help="working config file the editor reads from and saves to "
                        "(default: config.local.yaml, created on first Run)")
    p.add_argument("--seed", type=Path, default=REPO_ROOT / "config.example.yaml",
                   help="config to seed --config from if it doesn't exist yet")
    p.add_argument("--outdir", type=Path, default=REPO_ROOT / "outputs")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args(argv)

    if not args.config.exists() and args.seed.exists():
        Configuration.load(args.seed).save(args.config)
        print(f"seeded {args.config} from {args.seed}")

    app = create_app(args.config, args.outdir)
    print(f"hydroline web UI: http://{args.host}:{args.port}  "
          f"(editing {args.config})")
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    sys.exit(main())
