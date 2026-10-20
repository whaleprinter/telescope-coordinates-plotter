# Duke 21 cm telescope software

This repository holds **three separate programs**. Each has its own folder,
entry point and tests, and none imports another. Where one program uses
another's result, it reads a file the other wrote.

| program | question it answers | folder | run it with | docs |
|---|---|---|---|---|
| **hardware_model** | Given our parts, how well does the telescope work (gain, T_sys, beam, SEFD), and how bright must a source be? | [`hardware_model/`](hardware_model/) | `python3 hardware_model/compute_performance.py` | [hardware_model/README.md](hardware_model/README.md) |
| **hydroline** | Where does the zenith beam point over time, what drifts through it, and what can we detect? | [`hydroline/`](hydroline/) (+ `run.py`, `webui.py`) | `python3 run.py` | this file, from the next heading |
| **projections** | How does a set of sky coordinates look under a given map projection? | [`projections/`](projections/) | `python3 projections/example.py` | docstring of `projections/projections.py` |

```
 hardware_model/hardware_parameters.yaml        (you edit: the parts list)
            │
            ▼  hardware_model  (all equations in hardware_model/performance.py)
 hardware_model/output/performance_parameters.yaml   ← the only link between the two programs
            │
            ▼  hydroline   python3 run.py --performance <that file> --snr-target 3
 outputs/  (pointing track, transits, detectability, figures, report)
```

hydroline can also run without hardware_model: it then uses one of its preset
antennas or the antenna in `config.example.yaml`.

---

# hydroline — zenith drift-scan planner for a 21 cm telescope

Point an antenna straight up, bolt it down, and let the Earth do the scanning.
This program works out what that buys you: where the beam is on the sky at every
instant, which catalogued objects drift through it, and which of those your
particular antenna can actually detect or resolve.

Built for a hydrogen-line array on a rooftop at Duke University in Durham, NC,
but the site and antenna are both inputs — change either and everything
downstream re-derives.

```bash
python3 run.py
```

That writes thirteen CSVs, eight figures, a `summary.json` and a standalone
`report.html` into `outputs/`.

---

## The idea in one paragraph

A fixed antenna sweeps a circle of constant declination once per sidereal day.
For a zenith pointing the geometry collapses to two identities:

```
Dec(beam) = geodetic latitude of the site        (constant)
RA(beam)  = local apparent sidereal time         (advances 15.041°/hour)
```

Both are exact in the *apparent* frame — true equator and equinox of date.
Catalogues are in ICRS/J2000, so the numbers you compare against a catalogue
differ from the identities above by precession since J2000 (about 20 arcmin of
RA by 2026), nutation (~17″) and annual aberration (~20″). The program computes
the conversion properly through astropy/ERFA and reports both frames plus the
residuals, so the approximation is visible instead of hidden. It checks itself
on every run: the apparent declination reproduces the geodetic latitude and the
apparent RA reproduces sidereal time to **0.4 arcseconds**.

---

## Install

```bash
python3 -m pip install -r requirements.txt
```

Needs numpy, pandas and astropy; matplotlib and Pillow for the figures, PyYAML
for YAML configs, astroquery for the live source catalogue. Without the
plotting stack the run still writes every CSV and `summary.json`, and says so.
Tested on Python 3.12; 3.10 or newer is expected to work.

By default a run needs network access — the source catalogue comes from real
astronomical databases (see below), and astropy also reaches for
Earth-orientation tables. Pass `--offline` for a fully offline run: it skips
both, and the catalogue falls back to a small bundled static CSV. Once a
declination band has been queried once, it's cached locally
(`data/live_cache/`), so repeat runs at the same site don't need network
either, `--offline` or not.

### If it cannot find pandas or astropy

You almost certainly have more than one Python and your shell picked the wrong
one. macOS ships a system Python 3.9 that has numpy but *not* pandas, which is
why the failure often points at pandas and looks like a missing install rather
than a wrong interpreter. Check with:

```bash
which python3
```

`hydroline` detects this before it imports anything and prints the interpreter
it is running under, what is missing, and the path to another Python on the
machine that already works. A fresh terminal usually fixes it; otherwise call
the working interpreter by its full path.

---

## Using it

### The four questions

| Question | Where the answer is |
|---|---|
| Where does the beam look, over time? | `pointing_track.csv`, figures 1–3 |
| What does that look like? | figures 1–3 and the animated GIF |
| What falls in the path? | `beam_encounters.csv`, `sources_in_beam.csv`, figures 4–5 |
| What can we detect or resolve? | `detectability.csv`, figures 6–7 |

### Common runs

```bash
python3 run.py --antenna dish-3m --hours 48 --step 2
```

```bash
python3 run.py --lat 42.3601 --lon -71.0942 --elevation 40 --timezone America/New_York --site-name "MIT rooftop"
```

```bash
python3 run.py --kind horn --aperture-e 0.9 --aperture-h 0.7 --tsys 120 --integration 1800
```

```bash
python3 run.py --config config.example.yaml --theme dark
```

```bash
python3 run.py --offline                    # bundled static catalogue, no network
python3 run.py --refresh-catalog             # force a fresh NVSS/ALFALFA/SIMBAD fetch
python3 run.py --flux-floor 0.5 --no-simbad  # more NVSS sources, skip name enrichment
```

`python3 run.py --list-antennas` prints the built-in presets; `--help` lists
every flag. `--save-config my.yaml` writes out the fully resolved configuration
so a run is reproducible.

### Interactive web UI

```bash
python3 -m pip install flask   # not needed for the CLI, only this
python3 webui.py
```

Opens a form at `http://127.0.0.1:5000` pre-filled from `config.local.yaml`
(seeded from `config.example.yaml` the first time you run it). Edit any site,
antenna or observation field and hit **Run**: it saves the edits back to that
file and re-executes the full pipeline, then shows the regenerated
`report.html` in place. `--config` points it at a different working file;
`--port` and `--host` control where it listens. It's a local dev tool with no
authentication — don't point `--host` at anything but `127.0.0.1` unless you
know what that exposes.

### Configuration file

`config.example.yaml` is commented line by line. Command-line flags override it,
so the file holds what rarely changes (where the array is) and the flags carry
what you sweep. Numeric strings are coerced, so YAML's `2.5e6` footgun — which
YAML 1.1 reads as a *string* unless you write `2.5e+6` — will not bite you.

### Antenna from real parts (the separate hardware_model program)

Instead of a preset, hydroline can take its antenna from the
`performance_parameters.yaml` that the **hardware_model** program computes
from a parts list. hydroline only reads that file (`hydroline/performance_file.py`);
it never runs or imports hardware_model.

```bash
python3 hardware_model/compute_performance.py
python3 run.py --performance hardware_model/output/performance_parameters.yaml --snr-target 3
```

`--snr-target 3` sets the detection threshold and queries the live catalogues
only for sources bright enough to reach SNR 3 in one transit. See
[hardware_model/README.md](hardware_model/README.md).

---

## What the antenna model computes

Give it a dish diameter, a horn's two aperture dimensions, or a measured
beamwidth, plus a system temperature and a spectrometer configuration.

| Quantity | Relation |
|---|---|
| Beamwidth | `HPBW = k·λ/D`, k = 58.4 uniform, ~70 tapered (configurable) |
| Effective area | `A_eff = η_a · A_geom` |
| Gain | `G = 4π A_eff / λ²` |
| Sensitivity | `T_A/S = A_eff / 2k_B` |
| SEFD | `2 k_B T_sys / A_eff` |
| Noise | `ΔT = K_s T_sys / √(n_pol · B · τ)` |
| Beam coupling | `f = θ_b² / (θ_b² + θ_s²)` — the fraction of a source's total flux inside the main beam |
| Transit time | solved on the sphere, not with the small-angle `HPBW/(15 cos δ)` shortcut |

The two routes to antenna temperature — from flux density and from brightness
temperature — agree exactly in both the point-source limit (`T_A = S·A_eff/2k_B`)
and the fully-resolved limit (`T_A = η_MB·T_b`). That agreement is a test, not a
comment.

Horns get separate E- and H-plane beamwidths (54λ/a_E and 78λ/a_H degrees for an
optimum-gain pyramidal horn); everything downstream uses their geometric mean as
a circular-equivalent beam. **If your horn's two planes differ a lot, a source
can be inside the wide plane and outside the narrow one depending on how the
horn is rolled — the circular-equivalent number will not tell you that.**

---

## What "resolvable" means here

The word means two different things, so the program reports both.

**Angular resolution** — `resolution_class` asks whether the source is bigger
than the beam, so that scanning across it would show structure. At 21 cm a
metre-class aperture has a beam of roughly `14.8°/D(m)`, so a 1.5 m dish
resolves essentially nothing: the Milky Way, and that is the list. This is
physics, not a limitation of the code.

**Detectability** — `detectable_continuum` and `detectable_line` ask whether
there is enough signal to see the source at all, which is usually what people
mean. Continuum uses the full RF bandwidth. Line detections use a filter matched
to the source's own velocity width, because one 5 kHz channel of an M31 profile
is hopeless while the summed profile is not. Three integration times are quoted:
one transit (what a drift scan gives you free each day), the configured
integration, and the time actually needed to reach threshold.

---

## The source catalogue

By default, sources come from real astronomical databases, queried over the
declination band the beam actually sweeps (never the whole sky — a fixed
antenna only ever sees one declination):

- **NVSS** (VizieR `VIII/65/nvss`, Condon+1998) — the 1.4 GHz continuum
  survey, matching hydroline's own observing frequency exactly. It's dense
  (thousands of sources per beam-width strip above a few tens of mJy), so
  it's queried with a flux floor derived from *this antenna's own
  sensitivity* over a realistic 24 h cumulative integration — not a fixed
  number that would be wrong for a 0.3 m horn and a 4.6 m dish alike.
- **ALFALFA** (VizieR `J/ApJ/861/49`, Haynes+2018) — an HI-selected galaxy
  survey, giving distance, HI mass, systemic velocity and line width
  directly. Its real limitation: it's an Arecibo survey, so it only covers
  roughly declination 0° to +36°. Outside that band you'll correctly get
  zero HI galaxies from it — the report says why, rather than silently
  looking complete. Kept to a bounded number of the brightest (by implied
  peak flux), not the antenna's detection threshold: even a generous 24 h
  integration on a small dish can't match a 305 m telescope's sensitivity,
  so a strict cutoff would return nothing for most presets. The point of
  this list is "what's really out there" (the same reason the old
  hand-picked catalogue included M31 and M33 despite neither being
  detectable with modest gear) — `detectability_table()` still correctly
  works out what's actually detectable, per source.
- **SIMBAD** — enriches a position NVSS/ALFALFA already found with a common
  name and object-type classification. One query per source, so it's capped
  at 40 (skip entirely with `--no-simbad` for a faster run) and ranked to
  enrich the brightest/most massive sources first.

Results are cached locally (`data/live_cache/`, gitignored) keyed by the
query parameters — NVSS and ALFALFA are finished, static survey products,
so there's no real staleness risk; `--refresh-catalog` forces a re-fetch
anyway. `--catalog PATH` uses exactly that static CSV instead and never
touches the network; `--offline` does the same with the bundled
`data/sources_1420.csv`. If live querying fails for any reason (no network,
a service down), it falls back to that bundled CSV automatically and says
so loudly — a run never silently uses different data than what actually
happened. `--flux-floor` / `--flux-floor-multiplier` tune the NVSS cutoff
directly if the antenna-derived default pulls in too much or too little.

---

## Things the program will tell you that are easy to miss

- **Your latitude picks your targets.** At Durham the zenith sits at Dec +36.0°,
  and M31 (+41.3°) and M33 (+30.7°) both sit about 5.3° away — straddling the
  zenith, and both just *outside* the 4.9° half-power radius of a 1.5 m dish.
  The two best extragalactic HI targets in the northern sky are near misses.
  Widen the beam (smaller aperture, or a horn) or tilt the mount a few degrees
  and both come in. The report calls these out in a "near misses" box.
- **The Sun does not have to be in the beam to ruin your day.** At the solstice
  it passes 12.6° from Durham's zenith. For a 25° horn that is outside the
  half-power circle, and it still injects ~11 K. The `contaminates_off_axis`
  column flags this. Since the model is a pure Gaussian with no far sidelobes,
  those numbers are *lower* bounds.
- **The Galactic HI line is always there**, at an expected 10–80 K depending on
  galactic latitude, hundreds of sigma per channel. The beam crosses `b = 0`
  twice a day; from Durham those crossings are Cygnus (l ≈ 74°) and the
  anticenter (l ≈ 172°) — the two best windows for a rotation curve.
- **A wide beam is confusion-limited in continuum.** Many background sources sit
  inside it at once, so a continuum flux cannot be attributed to one object.
  Flagged automatically.
- **The velocity correction moves by ~20 km/s across a single night.** Apply it
  per spectrum, not once per session.

---

## Doppler reference frames

You cannot read a 21 cm spectrum without knowing what the frequency axis is
relative to. `pointing_track.csv` carries, for every time step, the correction
**to add to a topocentric radial velocity**:

- `v_barycentric_correction_kms` — removes the Earth's spin and orbit.
- `v_lsr_apex_projection_kms` — removes the Sun's motion relative to the local
  standard of rest, using the kinematic LSR convention every published HI
  velocity uses: 20.0 km/s toward RA 18ʰ, Dec +30° in B1900 coordinates.
- `v_lsr_correction_kms` — the sum, which is what you want.

---

## Layout

```
run.py                     entry point (same as python3 -m hydroline)
webui.py                   interactive config editor (python3 webui.py)
config.example.yaml        commented configuration
data/sources_1420.csv      bundled static catalogue -- the --offline/fallback source
data/live_cache/           cached live-query results (gitignored)
hydroline/
  config.py                Site, Antenna, Observation dataclasses
  constants.py             physical constants and 21 cm numbers
  antenna.py               beam geometry and radiometric sensitivity
  pointing.py              the coordinate conversion, and its self-check
  catalog.py               static-CSV catalogue loading + solar-system ephemeris
  live_catalog.py          NVSS/ALFALFA/SIMBAD queries, merging, caching
  analysis.py              transits, detectability, galactic crossings
  plots.py                 figures and the animation
  webapp.py                Flask app behind webui.py
  templates/                 its HTML form and result pages
  theme.py                 plot palettes
  export.py                CSV / JSON / HTML output
  cli.py                   argument parsing
  performance_file.py      reads hardware_model's performance_parameters.yaml
tests/
  test_hydroline.py          35 tests, mostly on the physics
  test_live_catalog_mapping.py  catalogue-building tests, no network
  test_live_catalog_network.py  real NVSS/ALFALFA/SIMBAD queries, opt-in
  test_performance_file.py   reading a performance_parameters.yaml
  conftest.py                 registers the network-test opt-in
outputs/                   everything a run produces
```

The two other programs live in `hardware_model/` and `projections/`, each with
its own tests; `python3 -m pytest` from the repository root runs all of them.

---

## Output files

| File | Contents |
|---|---|
| `pointing_track.csv` | one row per time step: UTC and local time, MJD, sidereal time, RA/Dec in ICRS *and* apparent, sexagesimal strings, galactic l/b, both velocity corrections, Sun/Moon separations, and the residuals against the analytic identities |
| `beam_encounters.csv` | one row per pass: entry / transit / exit times in UTC, local and sidereal time, minutes inside the beam, closest approach, beam response |
| `detectability.csv` | every object: angular size in beams, resolution class, beam coupling, in-beam flux, antenna temperature, continuum and line SNR, integration needed to reach threshold, off-axis contamination |
| `sources_in_beam.csv` | the subset that actually enters the beam |
| `galactic_plane_crossings.csv` | when the beam is within 10° of b = 0 |
| `galactic_hi_estimate.csv` | expected Galactic HI brightness along the track |
| `antenna_model.csv` | every derived antenna quantity |
| `solar_system_*.csv` | per-body position, distance, angular size and flux |
| `summary.json` | all the scalars, machine-readable |
| `report.html` | self-contained: figures inlined, tables, no assets needed |

---

## Accuracy, and where to be careful

**Trustworthy.** The coordinate conversion is astropy/ERFA end to end —
precession, nutation, polar motion, Earth rotation, annual and diurnal
aberration, gravitational light deflection. Self-checked on every run to sub-
arcsecond agreement with the analytic zenith identities. Solar-system positions
come from astropy's built-in ephemeris, good to arcseconds. The antenna formulas
are the standard single-dish relations and are unit-tested against each other.

**Approximate, by design.** NVSS/ALFALFA flux densities and HI masses are the
published survey values, real but with their own published uncertainties and
angular-resolution effects — a single-dish NVSS flux for an extended source can
differ from a total-power measurement. The bundled static CSV's fluxes are
rougher still: hand-typed estimates for the standard calibrators, worse for
extended Galactic sources whose "total flux" depends on how much sky you
integrate over. Check NED or SIMBAD before quoting any number in a write-up.

**Deliberately crude.** The Galactic HI brightness model is an exponential in
galactic latitude anchored to typical LAB-survey peak values. It answers "will I
see tens of kelvin?" (yes, always) and nothing finer. For predicted profiles,
query the LAB or EBHIS survey.

**Optimistic.** The sensitivity numbers are pure radiometer-equation limits. Real
small telescopes are limited by baseline ripple and standing waves long before
they reach thermal noise — plan on frequency or position switching, and treat the
quoted integration times as lower bounds. The Gaussian beam has no far sidelobes,
so off-axis contamination is understated.

**Editable.** `data/sources_1420.csv` (and any live-query cache CSV, which is
in the exact same format) is a plain CSV. Add a row with a name, a kind,
sexagesimal RA and Dec, an angular size and either a 1.4 GHz flux density or
an HI mass and distance, and it flows through the whole analysis. Point at a
different file with `--catalog`.

---

## Tests

```bash
python3 -m pytest tests -q
```

They pin the things that are easy to get quietly wrong: the frame identities,
the sidereal rate, the factor of two in `T_A = S·A_eff/2k_B`, the beam-coupling
limits, the transit-chord geometry against a numerically sampled separation
curve, and the solar-system frame handling — where transforming a
distance-carrying geocentric position straight to ICRS re-origins it at the
solar-system barycentre and puts the Sun on the opposite side of the sky.

That command never touches the network — the live-catalogue tests are hand-fed
fixture rows, not real queries. To also run the ones that hit NVSS/ALFALFA/
SIMBAD for real:

```bash
python3 -m pytest tests -q --run-network
```
