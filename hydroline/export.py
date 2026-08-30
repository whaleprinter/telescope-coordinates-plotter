"""Getting the numbers out.

Every table the program computes is written as CSV, every scalar as JSON, and
the whole run as one self-contained HTML report with the figures inlined.  The
CSVs are the source of truth -- the report and the figures are views of them.
"""
from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .antenna import AntennaModel
from .config import Configuration
from .pointing import PointingTrack


def _jsonable(obj: Any) -> Any:
    """Make numpy / pandas scalars JSON-serialisable."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def write_tables(outdir: Path, tables: dict[str, pd.DataFrame]) -> dict[str, Path]:
    """Write each DataFrame to ``outdir/<name>.csv``."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, df in tables.items():
        if df is None:
            continue
        p = outdir / f"{name}.csv"
        df.to_csv(p, index=False)
        written[name] = p
    return written


def write_summary(outdir: Path, payload: dict[str, Any]) -> Path:
    """Write the machine-readable run summary."""
    p = Path(outdir) / "summary.json"
    p.write_text(json.dumps(_jsonable(payload), indent=2))
    return p


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------
_CSS = """
:root{--bg:#16181c;--panel:#1a1c21;--line:#2b2e34;--ink:#fff;--ink2:#c3c2b7;
--muted:#8a8c93;--blue:#3987e5;--green:#199e70;--orange:#d95926}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:48px 28px 90px}
h1{font-size:30px;line-height:1.2;margin:0 0 6px;letter-spacing:-.02em}
h2{font-size:19px;margin:56px 0 14px;letter-spacing:-.01em;
border-top:1px solid var(--line);padding-top:26px}
h3{font-size:14px;margin:26px 0 8px;color:var(--ink2);font-weight:600}
p{color:var(--ink2);margin:10px 0}
.sub{color:var(--muted);font-size:13.5px;margin-bottom:26px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(162px,1fr));
gap:12px;margin:22px 0 6px}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:10px;
padding:14px 16px}
.tile .k{font-size:11px;text-transform:uppercase;letter-spacing:.07em;
color:var(--muted)}
.tile .v{font-size:24px;font-weight:600;margin-top:3px;letter-spacing:-.02em}
.tile .u{font-size:12px;color:var(--muted);margin-top:2px}
figure{margin:22px 0}
figure img{width:100%;border:1px solid var(--line);border-radius:10px;display:block}
figcaption{color:var(--muted);font-size:12.5px;margin-top:9px}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:10px;
background:var(--panel)}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th{text-align:left;padding:9px 12px;color:var(--muted);font-weight:600;
border-bottom:1px solid var(--line);white-space:nowrap;font-size:11px;
text-transform:uppercase;letter-spacing:.05em}
td{padding:8px 12px;border-bottom:1px solid var(--line);color:var(--ink2);
white-space:nowrap;font-variant-numeric:tabular-nums}
tr:last-child td{border-bottom:none}
td.name{color:var(--ink);font-weight:500;font-variant-numeric:normal}
.yes{color:var(--green);font-weight:600}
.no{color:var(--muted)}
code{background:var(--panel);border:1px solid var(--line);border-radius:5px;
padding:1.5px 6px;font-size:12.5px}
.note{background:var(--panel);border:1px solid var(--line);border-left:3px solid
var(--orange);border-radius:8px;padding:13px 17px;margin:20px 0;font-size:13.5px;
color:var(--ink2)}
.note b{color:var(--ink)}
"""


def _fig_tag(path: Path, caption: str) -> str:
    if not path or not Path(path).exists():
        return ""
    mime = "image/gif" if Path(path).suffix == ".gif" else "image/png"
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    return (f'<figure><img src="data:{mime};base64,{b64}" alt="{caption}">'
            f"<figcaption>{caption}</figcaption></figure>")


def _table_html(df: pd.DataFrame, cols: dict[str, str], limit: int = 60) -> str:
    if df is None or df.empty:
        return '<p class="sub">nothing to show</p>'
    d = df.head(limit)
    head = "".join(f"<th>{v}</th>" for v in cols.values())
    rows = []
    for _, r in d.iterrows():
        cells = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, (bool, np.bool_)):
                cells.append(f'<td class="{"yes" if v else "no"}">'
                             f'{"yes" if v else "—"}</td>')
            elif isinstance(v, (float, np.floating)):
                cells.append("<td>—</td>" if not np.isfinite(v) else
                             f"<td>{v:,.4g}</td>")
            elif c == "name":
                cells.append(f'<td class="name">{v}</td>')
            else:
                cells.append(f"<td>{v}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    more = (f'<p class="sub">showing {limit} of {len(df)} rows — '
            f"full table in the CSV</p>" if len(df) > limit else "")
    return (f'<div class="scroll"><table><thead><tr>{head}</tr></thead>'
            f"<tbody>{''.join(rows)}</tbody></table></div>{more}")


def _tile(k: str, v: str, u: str = "") -> str:
    return (f'<div class="tile"><div class="k">{k}</div>'
            f'<div class="v">{v}</div><div class="u">{u}</div></div>')


def write_report(outdir: Path, config: Configuration, model: AntennaModel,
                 track: PointingTrack, det: pd.DataFrame,
                 encounters: pd.DataFrame, crossings: pd.DataFrame,
                 coverage: dict, validation: dict, ghi: pd.DataFrame,
                 figures: dict[str, Path]) -> Path:
    """One self-contained HTML file with the figures and tables inlined."""
    outdir = Path(outdir)
    a = model.antenna
    s = model.summary()
    in_beam = det[det["enters_beam"]]
    detectable = in_beam[in_beam["any_detection"]]
    near = det[(~det["enters_beam"]) &
               (det["dec_offset_from_beam_deg"].abs()
                <= 1.6 * model.beam_radius_deg)]

    tiles = "".join([
        _tile("Zenith declination", f"{track.dec_deg:+.2f}°",
              "= geodetic latitude"),
        _tile("Beamwidth", f"{model.hpbw_deg:.1f}°", "half power"),
        _tile("Sky covered", f"{coverage['fraction_of_sky'] * 100:.1f}%",
              f"{coverage['band_solid_angle_deg2']:,.0f} deg²"),
        _tile("Sources in beam", f"{len(in_beam)}",
              f"{len(detectable)} detectable"),
        _tile("Transits", f"{int(encounters['in_beam'].sum()) if len(encounters) else 0}",
              "per window"),
        _tile("Galactic HI", f"{ghi['estimated_peak_t_a_k'].max():.0f} K",
              "peak expected T_A"),
    ])

    contam = det[det.get("contaminates_off_axis", False) == True]  # noqa: E712
    contam_note = ""
    if not contam.empty:
        c = contam.sort_values("t_a_at_closest_k", ascending=False).head(4)
        items = "; ".join(
            f"<b>{r['name']}</b> at {abs(r['dec_offset_from_beam_deg']):.1f}° "
            f"off axis contributes {r['t_a_at_closest_k']:.3g} K"
            for _, r in c.iterrows()
        )
        contam_note = (
            f'<div class="note"><b>Off-axis contamination.</b> These sources '
            f"never enter the half-power beam but are still strong enough to "
            f"matter: {items}. The beam model here is a pure Gaussian with no "
            f"far sidelobes, so these are <i>lower</i> bounds — a real feed "
            f"leaks at the −20 dB level in directions this model calls zero. "
            f"If the Sun appears in this list, expect a daytime baseline "
            f"offset and plan to observe at night or to switch against a "
            f"reference position.</div>"
        )

    near_note = ""
    if not near.empty:
        top = near.sort_values("dec_offset_from_beam_deg", key=abs).head(4)
        items = ", ".join(
            f"{r['name']} ({r['dec_offset_from_beam_deg']:+.1f}°)"
            for _, r in top.iterrows()
        )
        tilt = float(near.iloc[
            near["dec_offset_from_beam_deg"].abs().argmin()
        ]["dec_offset_from_beam_deg"])
        near_note = (
            f'<div class="note"><b>Near misses.</b> These sit just outside the '
            f"half-power circle: {items}. Tilting the mount {abs(tilt):.1f}° "
            f"toward the {'north' if tilt > 0 else 'south'}, or widening the "
            f"beam with a smaller aperture, brings the nearest of them in.</div>"
        )

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Zenith drift survey — {config.site.name}</title><style>{_CSS}</style>
</head><body><div class="wrap">

<h1>Zenith drift survey at 21 cm</h1>
<p class="sub">{config.site.name} &nbsp;·&nbsp;
{config.site.latitude_deg:+.4f}°, {config.site.longitude_deg:+.4f}°,
{config.site.elevation_m:.0f} m &nbsp;·&nbsp;
{track.times[0].isot[:16]} → {track.times[-1].isot[:16]} UTC &nbsp;·&nbsp;
generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC</p>

<div class="tiles">{tiles}</div>

<h2>1 · Where the beam points</h2>
<p>The antenna is fixed and the Earth turns, so the beam traces a circle of
constant declination once per sidereal day. In the apparent frame the
declination equals the geodetic latitude exactly and the right ascension equals
the local apparent sidereal time. This run reproduces both identities to
{max(validation['max_abs_dec_apparent_minus_latitude_arcmin'],
     validation['max_abs_ra_apparent_minus_lst_arcmin']) * 60:.2f} arcseconds,
which confirms the frame chain.</p>
{_fig_tag(figures.get('sky_equatorial'), 'The swept band in equatorial coordinates, with the catalogue overlaid.')}
{_fig_tag(figures.get('sky_galactic'), 'The same track in galactic coordinates. Crossings of b = 0 are the Milky Way HI disk.')}
{_fig_tag(figures.get('time_series'), 'Pointing and Doppler reference against time.')}

<h2>2 · The drift, animated</h2>
{_fig_tag(figures.get('animation'), 'The sky rotating past the fixed zenith beam over the observing window.')}

<h2>3 · What crosses the beam</h2>
{_table_html(encounters[encounters['in_beam']] if len(encounters) else encounters, {
    'name': 'source', 'kind': 'type', 'transit_local': 'transit (local)',
    'transit_lst_hours': 'LST (h)', 'duration_minutes': 'minutes in beam',
    'min_separation_deg': 'closest (°)', 'beam_response': 'gain',
    's1400_jy': 'S₁․₄ (Jy)'})}
{_fig_tag(figures.get('transit_timeline'), 'When each source is inside the half-power beam.')}
{_fig_tag(figures.get('separations'), 'Angular distance from the beam axis for the nearest sources.')}

<h3>Galactic plane crossings</h3>
{_table_html(crossings, {
    'closest_local': 'closest approach (local)', 'closest_lst_hours': 'LST (h)',
    'duration_hours': 'hours within 10° of the plane',
    'min_abs_galactic_latitude_deg': 'min |b| (°)',
    'galactic_longitude_at_closest_deg': 'l at closest (°)'})}

<h2>4 · What this antenna can detect</h2>
{_fig_tag(figures.get('detectability'), 'Single-transit signal-to-noise for everything the beam meets.')}
{_table_html(in_beam, {
    'name': 'source', 'kind': 'type', 'size_arcmin': 'size (′)',
    'resolution_class': 'resolution', 'flux_coupling': 'beam coupling',
    's1400_in_beam_jy': 'S in beam (Jy)',
    'snr_continuum_single_transit': 'continuum SNR',
    'snr_line_matched_transit': 'line SNR',
    'transit_duration_min': 'transit (min)',
    'detectable_continuum': 'continuum?', 'detectable_line': '21 cm?'})}
{near_note}
{contam_note}

<h2>5 · The antenna</h2>
{_fig_tag(figures.get('antenna'), 'Beam pattern, the aperture/beamwidth trade, and radiometer noise.')}
<div class="tiles">
{_tile("Effective area", f"{s['effective_area_m2']:.2f} m²", f"{s['gain_dbi']:.1f} dBi")}
{_tile("Sensitivity", f"{s['sensitivity_k_per_jy'] * 1e3:.2f} mK/Jy", f"SEFD {s['sefd_jy']:,.0f} Jy")}
{_tile("System temp", f"{s['t_sys_k']:.0f} K", f"{s['n_pol']} polarisation(s)")}
{_tile("Velocity res.", f"{s['velocity_resolution_kms']:.2f} km/s", f"{s['channel_khz']:.1f} kHz channels")}
{_tile("Velocity span", f"±{s['velocity_coverage_kms'] / 2:.0f} km/s", f"{s['rf_bandwidth_mhz']:.1f} MHz")}
{_tile("Line limit", f"{s['line_limit_k']:.2f} K", f"{s['snr_threshold']:.0f}σ in {s['integration_s']:.0f} s")}
</div>

<div class="note"><b>Reading the sensitivity numbers.</b> Continuum
signal-to-noise assumes the full RF bandwidth; 21 cm line values use a filter
matched to the source's own line width, which is how a real HI detection is
made. In practice small telescopes are limited by baseline ripple rather than by
thermal noise, so treat these as upper bounds and use frequency or position
switching. {'This beam is confusion-limited for continuum work: many background '
'sources sit inside it at once, so continuum flux cannot be attributed to a '
'single object.' if model.confusion_limited else ''}</div>

<h2>6 · Files</h2>
<p>Every table above is a CSV in this directory:
<code>pointing_track.csv</code> (one row per time step, all frames),
<code>beam_encounters.csv</code>, <code>detectability.csv</code>,
<code>galactic_plane_crossings.csv</code>, <code>galactic_hi_estimate.csv</code>,
<code>antenna_model.csv</code>, and <code>summary.json</code> for the scalars.</p>

</div></body></html>"""

    p = outdir / "report.html"
    p.write_text(html)
    return p
