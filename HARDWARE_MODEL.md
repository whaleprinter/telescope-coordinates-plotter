# Hardware model: from parts list to sensitivity

This is the part of the project that answers *"given the parts we bought, how
well will the telescope work, and what can it see?"*  It is a small pipeline
of plain files, so each step can be checked by hand and each intermediate
result can be read, committed and compared:

```
hardware/hardware_parameters.yaml  ─┐
                                    ├─ compute_performance.py ─▶ hardware/output/performance_parameters.yaml
hardware/observing_goals.yaml  ─────┘                         ├▶ hardware/output/detection_thresholds.yaml
                                                              ├▶ hardware/output/frequency_response.csv
                                                              └▶ frequency_response.png, beam_3d.png,
                                                                 noise_budget.png, chain_comparison.png

performance_parameters.yaml ─ run.py --performance … --snr-target 3 ─▶ the drift-scan plan (outputs/)
```

1. **You edit** `hardware_parameters.yaml`: datasheet facts about each part.
2. **The script computes** `performance_parameters.yaml`: total gain, passband,
   receiver temperature (Friis), antenna temperature, T_sys, effective area,
   the beam, and the SEFD. The hardware calculation ends at SEFD.
3. **Your goals** (`observing_goals.yaml`: "I want SNR 3") turn SEFD into the
   flux a source needs (`detection_thresholds.yaml`). That is the number to
   filter a catalogue by.
4. **The planner** (`run.py`) can take the computed antenna instead of a
   preset. With `--snr-target` it queries the live catalogues only for
   sources bright enough to matter.

---

## Quick start

```bash
python3 compute_performance.py
```

Use `/opt/anaconda3/bin/python3` if plain `python3` picks up the system
Python 3.9. It runs in about 2 s and prints:

```
Duke 21 cm zenith telescope (Fitzpatrick roof)
  chain        feed_cable  →  lna  →  bandpass_filter  →  coax_run  →  sdr
  total gain   17.36 dB to the SDR input
  passband     1417.386-1423.432 MHz (-3 dB), noise bandwidth 6.064 MHz (nominal edges from sdr / sdr)
  T_rec        65.4 K     T_A 21.2 K     T_sys 86.6 K
  A_eff        1.152 m²   HPBW 9.79° (E) × 9.29° (H)
  SEFD         2.075e+05 Jy
  beam         Dec +36.00°, strip 9.79° wide, transit 45.8 min
  ~ 19 inputs are placeholders/estimates (listed in performance_parameters.yaml)

  For SNR 3  (S = SNR · SEFD / sqrt(n_pol · Δν · τ)):
       integration    continuum   HI, 1 chan  HI, matched
         1 transit       4.82 Jy        154 Jy       12.2 Jy
       30 transits       0.88 Jy       28.2 Jy       2.23 Jy
    ...
    → index the catalogues at NVSS S1.4 ≥ 4.82 Jy, HI peak ≥ 12.2 Jy (1 transit)
```

**Read the "placeholders" line first.** The reflector, feed and filter in your
draft file had no values yet, so the example uses stated guesses, each marked
`status: placeholder` or `status: estimate`. Every such input is listed under
`placeholder_inputs:` in the output. Until that list is empty, the numbers are
a model of *a* telescope like yours, not of yours.

### Common commands

| You want to… | Run |
|---|---|
| compute everything from the default files | `python3 compute_performance.py` |
| use a different hardware file | `python3 compute_performance.py my_rig.yaml` |
| try a different chain order once | `python3 compute_performance.py --order lna,feed_cable,bandpass_filter,coax_run,sdr` |
| compare specific orders side by side | `python3 compute_performance.py --compare current lna,feed_cable,bandpass_filter,coax_run,sdr` |
| rank every possible order | `python3 compute_performance.py --all-orders` |
| change the SNR you insist on | `python3 compute_performance.py --snr-target 5` |
| plan observations with this hardware | `python3 run.py --performance hardware/output/performance_parameters.yaml --snr-target 3` |
| dark figures | add `--theme dark` |
| run the tests | `python3 -m pytest tests/test_hardware.py -q` |

---

## 1. The input: `hardware_parameters.yaml`

It uses the format from your draft. Every number is written the way the
datasheet gives it:

```yaml
noise_figure: {value: 0.7, unit: dB, status: datasheet, notes: at 1.4 GHz}
```

| key | meaning |
|---|---|
| `value` | the number (or text, for things like `type`) |
| `unit` | `Hz kHz MHz GHz`, `m cm mm ft in`, `K C`, `deg`, `dB dBm dBi`, `dB/m dB/100ft`, `s min h`, `W V`, `bit`, `""` (no unit), `":1"` (ratios such as VSWR) |
| `status` | `datasheet` (default), `measured`, `estimate`, `placeholder` |
| `notes` | free text. **Quote it if it contains a comma**: `notes: "selectable 12, 10 or 8"` |

The loader converts every unit once, when it reads the file, so a MHz/GHz
mix-up can't creep in further down. It also refuses anything it doesn't
understand and says why. Two mistakes from the draft file are now caught:

* `{value: 1.8, unt: :1}` → `unknown key(s) 'unt' (did you mean 'unit'?)`. A
  bare `:1` is also not valid YAML, so write `unit: ":1"`.
* `{..., notes: selctable 12, 10, or 8}` → YAML splits the entry at the
  commas. The error tells you to quote the notes.

### Values that change with frequency

Any numeric parameter can be a curve instead of a single value. The curve is
interpolated linearly and held flat past its last point:

```yaml
gain:         {unit: dB, freq_unit: GHz, curve: [[1.3, 20.6], [1.4, 19.81], [1.5, 19.1]]}
noise_figure: {unit: dB, freq_unit: MHz, file: lna_nf.csv}     # two columns: freq, value
```

An amplifier or filter can take its whole gain curve from the manufacturer's
Touchstone file. Mini-Circuits publishes one for the ZX60-P162LN+. Save it
next to the YAML and add `sparameters: ZX60-P162LN+.s2p` to the component; S21
replaces `gain`. **This is the single most valuable upgrade:** the model
currently assumes flat gain inside each part's band.

### Sections

**`analysis`**: `center_freq` (default: the HI rest frequency) and the
frequency range for the response plot (`sweep_min`, `sweep_max`,
`sweep_points`).

**`environment`**: the inputs your note said were missing (the "environment
block" for T_A):

| parameter | used for | default |
|---|---|---|
| `site_latitude` | beam declination, transit time | 36.003147° |
| `ambient_temperature` | physical temperature of every part that doesn't give its own | 290 K |
| `ground_temperature` | what the spillover sees | 290 K |
| `cmb_temperature` | sky | 2.725 K |
| `galactic_temperature_408`, `galactic_spectral_index` | Galactic synchrotron, scaled as T₄₀₈·(f/408 MHz)^β | 20 K, −2.7 |
| `atmosphere_temperature` | sky | 2 K |
| `horizon_elevation` | ground pickup of a bare feed (roofline, trees) | 0° |

The latitude in the file is 36.003147°, which is 36°00′11.33″ converted
properly. See [a note on the site coordinates](#a-note-on-the-site-coordinates).

**`components`**: one list holding both structure and RF chain, as in your
draft. `type` decides how each component is modelled:

| type | role | parameters it reads |
|---|---|---|
| `reflector` | collecting area, beam | `diameter`, `focal_ratio`, `surface_rms_error`, `blockage_fraction`. Set `enabled: false` to point the feed at the sky by itself |
| `feed` | illumination, beam shape, band | `hpbw_e`, `hpbw_h` (feed beamwidths) **or** `edge_taper_e`, `edge_taper_h` (dB at the dish rim); `diameter` (circular waveguide → TE11/TM01 band edges); `freq_min`/`freq_max`; `ohmic_loss`; `vswr`; `back_radiation_fraction`; `e_plane_azimuth` |
| `mount` | pointing | `pointing_alt`, `pointing_az` |
| `amplifier` | gain + noise | `gain`, `noise_figure`, `freq_min`, `freq_max`, optional `rolloff` (dB/octave beyond the band, default 12), `p1db_output` |
| `filter` | selectivity + loss | `center_freq`, `bandwidth`, `insertion_loss`, `out_of_band_rejection`, `response` (`butterworth`/`chebyshev`), `order`, `ripple` |
| `cable` | loss | `length` + `loss_per_length` + `loss_ref_freq` (scaled as √f), or a total `loss` |
| `attenuator` | flat loss | `loss` |
| `sdr` | digitiser | `sample_rate`, `if_bandwidth` (default = sample rate), `noise_figure`, `fft_size` (→ channel width), `n_pol`, `radiometer_factor`, `tuned_freq` |

Any passive part also takes `physical_temperature`. This is the T_phys in
T = T_phys(L − 1). A cable in the summer sun on the roof (≈320 K) is noisier
than the same cable indoors. The `stages:` block is documentation. The output
file uses the same headings.

---

## 2. Rearranging the chain

The RF parts run in `chain_position` order. To change the behaviour:

* **renumber** the `chain_position`s and re-run, or
* **override** without editing the file: `--order lna,feed_cable,bandpass_filter,coax_run,sdr`, or
* **remove** a part: `enabled: false` (an attenuator you're unsure about, a
  second LNA, the reflector itself).

`--compare` puts several orders on one figure (`chain_comparison.png`).
`--all-orders` tries every permutation (the SDR stays last) and ranks them:

```
  24 orders, best first (T_sys at the line):
       83.2 K   lna > feed_cable > bandpass_filter > coax_run > sdr
       ...
       86.6 K   feed_cable > lna > bandpass_filter > coax_run > sdr   ← current
    ...
      342.5 K   bandpass_filter > feed_cable > coax_run > lna > sdr
```

**What to expect.** The results follow from the 1/G divisors in Friis:

* **Whatever comes before the LNA sets the noise floor.** 1 dB of loss ahead
  of the LNA adds about 75 K at room temperature and raises T_sys by the same
  amount. Behind 20 dB of gain, the same loss costs under 1 K. Moving the
  10 m coax run (1.4 dB) ahead of the LNA takes T_sys from 87 K to 214 K.
  Integration time grows as T_sys², so the same detection takes 6 times
  longer. With everything ahead of the LNA (the worst order, 342 K) it takes
  17 times longer.
* **Once the LNA is first, the order of the rest barely matters.** Six orders
  tie at 83.2 K. The current order (0.3 m pigtail, then the LNA) costs 3.4 K
  over mounting the LNA right at the feed.
* **The ranking only measures noise.** A filter *before* the LNA costs about
  85 K, but it protects the LNA from strong out-of-band signals (cell towers,
  radar, 1.5–1.6 GHz GNSS). The `lna_compression_headroom` line in the output
  is the first check on that: the LNA's P1dB minus the noise power across its
  whole 0.7–1.6 GHz input band. On a city rooftop the real answer is often a
  low-loss filter first, or an LNA module with a filter built in. Only an RF
  survey of the roof can settle that, and the model can't do it for you.

---

## 3. What the script computes, and how

Everything is referred to the **feed output terminals**. Feed ohmic loss and
mismatch belong to the antenna: they reduce A_eff and add to T_A. The first
cable is the first stage of the receiver, so it counts inside T_rec. T_sys and
A_eff are therefore at the same reference plane, as SEFD requires.

### `gain` stage

* **Total gain** is the sum of each stage's gain in dB, from the chain input
  to the SDR input. A loss counts as negative gain.
* **Nominal passband**: lower edge = the highest lower edge of any part,
  upper edge = the lowest upper edge. The output says which part set each
  edge. In the example both come from the SDR's 6 MHz.
* **Rigorous passband**: every part's response curve is multiplied (added in
  dB) on a fine grid. That gives the −3 dB points, the **noise bandwidth**
  (B_N = ∫G df / G_peak, the bandwidth the radiometer equation needs, which
  differs from the 3 dB width), and the **ripple** over the central 80% of the
  band.
* **Noise power at the SDR** (k·T_sys·G·B_N, here −94 dBm) and **LNA headroom**
  (above).

### `noise` stage

* **Receiver temperature (Friis)**: T_rec = T₁ + T₂/G₁ + T₃/(G₁G₂) + …
  * active parts: T = 290 K × (10^(NF/10) − 1)
  * passive parts: L = 10^(loss/10), G = 1/L, T = T_phys × (L − 1)

  `friis_breakdown` in the output lists each stage's own T, the gain ahead of
  it, its contribution in K, and its share of T_rec. The example's LNA is
  78% of T_rec and the SDR is 14%: 4.3 dB NF divided by only 17 dB of gain.
  More gain ahead of the SDR would shrink that.
* **Antenna temperature**:
  T_A = η_mm [ η_ohm ((1 − f_g) T_sky + f_g T_ground) + (1 − η_ohm) T_feed ]
  * T_sky = T_CMB + T₄₀₈(f/408 MHz)^β + T_atm
  * f_g is the fraction of the response that looks at warm ground. With a
    reflector it is the spillover past the rim: a prime-focus feed looks
    down, so everything that misses the dish hits the ground. With a bare
    feed pointing up, it is the part of the feed pattern below
    `horizon_elevation`, plus the back lobe.
* **T_sys = T_A + T_rec**. `noise_budget.png` splits it into every
  contributor.

### `collection` and `beam_pattern` stages: the 3-D part

The feed is modelled as a cos^q power pattern with **separate E- and H-plane
exponents**, set from your feed HPBWs or edge tapers:

  G_f(θ, φ) ∝ cos^qE(θ)·cos²φ + cos^qH(θ)·sin²φ

This is what lets the model "think in 3-D". A real feed (cantenna, dipole) is
not circularly symmetric. Its asymmetry follows the light path: feed pattern
→ illumination on the dish → aperture field → far-field beam. The result is
an elliptical beam.

* **Efficiencies** (paraboloid, Balanis §15.4), computed by integrating the
  aperture field over the dish in 2-D: spillover, taper (illumination),
  blockage, surface (Ruze, exp(−(4πε/λ)²)), feed ohmic loss, mismatch, and
  back radiation. Their product is the aperture efficiency η_a. A_eff = η_a ×
  πD²/4, and gain = 4πA_eff/λ².
* **The beam**: the Hankel transform of the aperture field in each principal
  plane, blended as P(θ,φ) = P_E cos²φ + P_H sin²φ. The outputs are HPBW in
  the E and H planes, first sidelobes, beam solid angle, main-beam solid angle
  and efficiency, and directivity from integrating the pattern.
  `rule_of_thumb_hpbw` (70λ/D) and `rule_of_thumb_gain` (10·log₁₀(6(D/λ)²)),
  the formulas from your notes, are printed next to the computed values for
  comparison.
* **Orientation on the sky matters for a drift scan.** At the zenith, local
  north points to +Dec and east points to +RA, and a fixed mount never
  rotates the beam. So an elliptical beam has a **fixed footprint**:
  * its **north–south** width sets how wide a **declination strip** you sweep
    (`declination_strip`);
  * its **east–west** width sets how long a source takes to drift through,
    which is your integration time per transit (`transit_time`).

  `e_plane_azimuth` sets which way the feed is turned. Rotating a strongly
  asymmetric feed by 90° trades strip width against integration time. With
  `hpbw_e: 50, hpbw_h: 90` on a bare feed, 0° gives a 50°-wide strip with 90°
  of drift, and 90° gives the reverse. `beam_3d.png` shows the 3-D pattern,
  the footprint with N/E axes and the −3 dB contour, and the two principal
  cuts.

With the example's placeholder cantenna (66° × 80°) on a 1.5 m dish, the beam
comes out at 9.8° × 9.3°, only 5% elliptical. The dish smooths out most of the
feed's asymmetry. With `reflector: enabled: false` the same feed gives a
66° × 80° beam, A_eff ≈ 0.03 m² and SEFD ≈ 8.5 MJy, about 40 times less
sensitive. That is the trade-off between a feed-only design and a dish.

### `position` stage

Beam declination = asin(sin φ sin alt + cos φ cos alt cos az), which equals
the latitude for a zenith mount. The output also gives the strip width and the
transit time, HPBW_EW / (15.04°/h × cos δ).

### `system`: SEFD, and where the hardware stops

* **SEFD = 2kT_sys / A_eff** (×10²⁶ for Jy): the source flux that would double
  the noise power. Lower is better. The example gives 2.1 × 10⁵ Jy.
* `noise_per_root_hz_s` = K_s·SEFD/√n_pol, so the 1σ noise in any
  measurement is that divided by √(Δν·τ).
* `spectral_channel` = sample_rate / fft_size (5.9 kHz = 1.25 km/s here).

**SNR is not stored in the hardware file.** As your note says, it needs three
inputs the hardware doesn't know. In code it is three functions in
`hydroline.hardware`:

```python
from hydroline.hardware import snr, required_flux_jy, required_integration_s
snr(flux_jy, sefd_jy, bandwidth_hz, integration_s, n_pol=1, k_s=1)          # forward
required_flux_jy(target_snr, sefd_jy, bandwidth_hz, integration_s, n_pol=1)  # "I want SNR 3"
required_integration_s(target_snr, flux_jy, sefd_jy, bandwidth_hz, n_pol=1) # "how long?"
```

`n_pol=2` (both RSPduo tuners, one per polarisation, summed) gives the
expected factor of √2.

---

## 4. "I want an SNR of 3": `observing_goals.yaml` → `detection_thresholds.yaml`

This is your professor's suggestion. You state the SNR you want, the
integrations you care about, and a typical HI line width:

```yaml
snr_target:    {value: 3, unit: ""}
integrations:  [transit, transit x 30, {value: 1, unit: h}, {value: 24, unit: h}]
hi_line_width: {value: 200, unit: km/s}
```

The script inverts the radiometer equation, S = SNR·K_s·SEFD/√(n_pol·Δν·τ),
for three kinds of measurement with very different Δν:

| measurement | Δν | one transit, SNR 3 |
|---|---|---|
| **continuum** (a source's total power) | noise bandwidth, 6.06 MHz | **4.8 Jy** |
| **HI, one channel** | 5.9 kHz | 154 Jy |
| **HI, matched** (summed over a 200 km/s line, which is how galaxies are actually detected) | 948 kHz | **12 Jy** peak |

`database_index` turns the first integration into catalogue floors
(`nvss_s1400_min`, `hi_peak_min`, `hi_integrated_min`). The planner does the
same in one step:

```bash
python3 run.py --performance hardware/output/performance_parameters.yaml --snr-target 3
# Catalogue: SNR target 3 in one 47-min transit needs S ≥ 4.76 Jy (SEFD 2.08e+05 Jy, 6.06 MHz)
#            -- querying above that
```

`--snr-target` sets the planner's detection threshold and also the NVSS flux
floor of the live catalogue query, so the database is searched only for
sources this hardware could reach. Pass `--flux-floor` to override it. The
planner's 4.76 Jy differs slightly from the 4.82 Jy above because the planner
models a circular beam (the geometric-mean HPBW), so its transit is a little
longer.

**What to expect from those numbers:**

* **Integration per source is set by the sky, not by you.** A fixed beam sees
  a given source for one transit a day (46 min here). "24 h" is only
  meaningful for something that fills the beam all day, such as Galactic HI.
  For a point source, the realistic way to go deeper is "transit × N days"
  of stacking, which gains only √N.
* **Galactic HI is easy; other galaxies are hard.** Galactic HI gives tens of
  K of antenna temperature (the planner's `galactic_hi_estimate.csv`). A
  12 Jy HI peak, by contrast, is brighter than any galaxy in the northern
  sky. ALFALFA's brightest have peaks of a few Jy at most, and M31 is the
  exception only because it is huge. M31 and M33 lie ~5° off the beam at this latitude
  (see the planner's notes).
* **Continuum.** The 4.8 Jy floor leaves a handful of NVSS sources per transit
  strip. A 10° beam is also *confusion-limited*: many faint sources blend
  together, and the planner prints that warning. Individual continuum sources
  other than the very brightest (Cas A, Cyg A, Tau A) are not realistic
  targets.

---

## 5. The frequency-response graph (`frequency_response.png`)

Four panels share one frequency axis (1380–1460 MHz by default). The
processed band is shaded, the 1400–1427 MHz protected band is a grey bar, and
the HI line is a vertical line.

1. **Gain through the chain**: cumulative gain after each part, the total to
   the SDR input (bold), and the total including the feed's band and the SDR's
   selectivity. The filter sets the analog shape; the SDR cuts out the 6 MHz
   that is actually digitised.
2. **Noise temperatures**: T_sys, T_rec, T_A and T_sky across the band. The
   analog chain is shown over the whole sweep because the SDR can be retuned:
   this panel shows where the hardware *could* observe.
3. **Where the receiver noise comes from**: Friis contributions stacked. In
   band the LNA dominates. Outside the filter's band, the filter's own loss
   (orange) dominates. That is Friis applied to a part that passes nothing.
4. **Sensitivity**: SEFD, and the flux needed for your SNR target in one
   channel during one transit.

The same numbers per frequency, with each stage's gain and noise
contribution, are in `frequency_response.csv` for anything you'd rather do in
a spreadsheet.

---

## 6. Limits of the model

* **Flat in-band gain** unless you supply a curve or `.s2p`. Outside a datasheet
  `freq_min`/`freq_max` it rolls off at a fixed dB/octave. Fine for the HI
  band, crude far outside it.
* **Filters are ideal prototypes** (Butterworth/Chebyshev) until you load a
  measured curve.
* **Mismatch** is treated as an efficiency only. The LNA's noise actually
  depends on the source impedance the feed presents (its noise parameters).
  For a 1.5:1 VSWR this can be worth several K, and the model doesn't
  capture it.
* **Out-of-band T_rec** uses T = T_phys(L − 1) for reflective filter skirts.
  That is the standard engineering cascade. In-band numbers don't depend on
  it.
* **Beam**: the principal-plane blend and the scalar aperture theory agree
  with the exact uniform-aperture result (HPBW 58.9λ/D, −17.6 dB sidelobe) to
  under 1%. The pattern-integrated directivity is within 0.1–0.3 dB of the
  aperture-integral value, which is what gain and A_eff use. Feed
  cross-polarisation, strut scattering and the far sidelobes past ~90° are
  not modelled.
* **The sky is uniform**: one T_sky. When the Galactic plane drifts through, T_A
  rises by tens of K. The planner's Galactic HI estimate shows when.
* **A tilted mount** gets the right declination, but the beam's rotation on
  the sky (parallactic angle) is only exact at the zenith.

---

## 7. Files and code

| path | what |
|---|---|
| `compute_performance.py` | the script; `--help` lists every option |
| `hardware/hardware_parameters.yaml` | **input**: edit this |
| `hardware/observing_goals.yaml` | **input**: SNR target and integrations |
| `hardware/output/` | **generated**, git-ignored; re-run to refresh |
| `hydroline/hardware/params.py` | `{value, unit, status}` parsing, unit conversion, curves, typo detection |
| `hydroline/hardware/rf.py` | component gain/noise models, Friis cascade, passband, Touchstone reader |
| `hydroline/hardware/optics.py` | feed pattern, reflector efficiencies, 3-D beam |
| `hydroline/hardware/system.py` | loading, T_A, T_sys, SEFD, SNR functions, output writers |
| `hydroline/hardware/plots.py` | the four figures |
| `hydroline/hardware/bridge.py` | performance file → planner antenna (`run.py --performance`) |
| `tests/test_hardware.py` | 23 checks against hand-computed physics |

The tests are the best place to see that the maths matches the textbook.
Friis is checked by hand for two chain orders; the Butterworth noise
bandwidth against (π/2n)/sin(π/2n); the uniform aperture against
58.9λ/D and −17.6 dB; the −10 dB-taper efficiency against Balanis; Ruze,
waveguide cut-offs, SEFD, the SNR inverses and √2 for two polarisations are
each checked directly.

## A note on the site coordinates

`config.example.yaml` gives the Fitzpatrick roof as `latitude_deg: 36.0011`,
`longitude_deg: -78.5623`. The comment beside them, 36°00′11.33″N
78°56′23.53″W, converts to **36.003147°, −78.939869°**. The file seems to have
taken the minutes and seconds as decimal digits. The latitude error is only
7″, but the **longitude is 0.38° east of the roof**. That shifts every
predicted transit by about **1.5 minutes**, and the planner sees the wrong
patch of sky by that much. The hardware model uses the corrected latitude.
`config.example.yaml` has not been changed; that's left for you to decide.
