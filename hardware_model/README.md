# hardware_model: from parts list to sensitivity

`hardware_model` is **its own program**, separate from the hydroline planner.
It answers *"given the parts we bought, how well will the telescope work, and
how bright must a source be for us to detect it?"*

It does not import hydroline, and hydroline does not import it. The only link
is a file: hardware_model writes `output/performance_parameters.yaml`, and
hydroline can read it (`python3 run.py --performance …`). A test
(`test_does_not_import_hydroline`) enforces this.

```
hardware_parameters.yaml ─┐
                          ├─ compute_performance.py ─▶ output/performance_parameters.yaml ──▶ (hydroline, optional)
observing_goals.yaml  ────┘                          ├▶ output/detection_thresholds.yaml
                                                     ├▶ output/frequency_response.csv
                                                     └▶ output/frequency_response.png, beam_3d.png,
                                                        noise_budget.png, chain_comparison.png
```

---

## Quick start

From the repository root:

```bash
python3 hardware_model/compute_performance.py
```

`python3 -m hardware_model` does the same. Use `/opt/anaconda3/bin/python3` if
plain `python3` picks up the system Python 3.9. It runs in about 2 s:

```
Duke 21 cm zenith telescope (Fitzpatrick roof)
  chain        feed_cable  →  lna  →  bandpass_filter  →  coax_run  →  sdr
  total gain   17.36 dB to the SDR input
  passband     1417.386-1423.432 MHz (-3 dB), noise bandwidth 6.064 MHz (nominal edges from sdr / sdr)
  T_rec        65.4 K     T_A 21.2 K     T_sys 86.6 K
  A_eff        1.142 m²   HPBW 9.91° (E) × 9.15° (H)
  SEFD         2.092e+05 Jy
  beam         Dec +36.00°, strip 9.91° wide, transit 45.1 min
  ~ 19 inputs are placeholders/estimates (listed in performance_parameters.yaml)

  For SNR 3  (Eq. 34: S = SNR · SEFD / sqrt(n_pol · Δν · τ)):
       integration    continuum   HI, 1 chan  HI, matched
         1 transit        4.9 Jy        157 Jy       12.4 Jy
       30 transits      0.894 Jy       28.7 Jy       2.26 Jy
    ...
```

**Read the "placeholders" line first.** The reflector, feed and filter had no
values in the draft, so the example uses stated guesses marked
`status: placeholder` / `estimate`. All of them are listed under
`placeholder_inputs:` in the output. Until that list is empty, the numbers
describe *a* telescope like yours, not yours.

| You want to… | Run |
|---|---|
| compute everything | `python3 hardware_model/compute_performance.py` |
| use another hardware file | `python3 hardware_model/compute_performance.py my_rig.yaml` |
| try another chain order | `… --order lna,feed_cable,bandpass_filter,coax_run,sdr` |
| compare orders side by side | `… --compare current lna,feed_cable,bandpass_filter,coax_run,sdr` |
| rank every order | `… --all-orders` |
| change the SNR you insist on | `… --snr-target 5` |
| plan observations with this hardware (hydroline) | `python3 run.py --performance hardware_model/output/performance_parameters.yaml --snr-target 3` |
| run the tests | `python3 -m pytest hardware_model/tests -q` |

---

## How the code is organised

Every equation lives in **one file, `performance.py`**. The other modules
either read inputs or present results, and contain no physics.

| file | job | physics? |
|---|---|---|
| `inputs.py` | Read the YAML. Convert every unit once, by an exact factor (GHz→Hz, ft→m, °C→K). Validate keys, units and chain positions. Read Touchstone files. Build plain data objects (`Hardware`, `ComponentSpec`, `FeedSpec`, `ReflectorSpec`, `Environment`, `Goals`). | none |
| **`performance.py`** | **All the maths.** Sections 0–9 define 38 numbered equations, each with a reference. Section 10 runs them, from `evaluate()` through Steps 1–12. | **all of it** |
| `report.py` | Round each computed number, attach its unit and its `eq` number, and write the YAML and CSV outputs. | none |
| `plots.py` | Draw the four figures from computed arrays. Colours are defined here, so it doesn't depend on hydroline's theme. | none |
| `compute_performance.py` | Command line: parse arguments, call `evaluate()`, write files, run order comparisons. | none |
| `hardware_parameters.yaml`, `observing_goals.yaml` | The inputs you edit. | — |
| `tests/test_hardware_model.py` | 28 checks against hand-computed textbook values. | — |

Each entry in `performance_parameters.yaml` has an `eq:` field, for example
`sefd: {value: 209200.0, unit: Jy, eq: Eq. 31}`. Search `performance.py` for
`[Eq. 31]` to find the formula, its explanation and its source.

---

## The mathematics: hardware parameters → performance parameters

`evaluate()` in `performance.py` runs these steps in order. Equation numbers
are the `[Eq. N]` tags in that file. Each docstring there has the full
derivation and the meaning and unit of every symbol.

| Step | What | Equations | Formula | Source |
|---|---|---|---|---|
| 1 | Wavelength | 13 | λ = c/f | definition; c exact [SI] |
| 2 | Feed pattern exponents (E and H plane) | 14a, 14b, 19 | q = ln(½)/ln cos(HPBW/2), or q = T_dB / (10 log cos θ₀) | cos^q feed model [Balanis 15.4] |
| 3 | Feed band (cantenna) | 16 | f_TE11 = 1.8412c/(πd), f_TM01 = 2.4048c/(πd) | [Pozar 3.4] |
| 4 | Feed losses | 17, 18 | η_mm = 1 − ((VSWR−1)/(VSWR+1))², η_ohm = 10^(−L/10) | [Pozar 2.3], [Balanis ch. 2] |
| 4 | Feed pattern | 15 | G_f = C[cos^qE θ cos²φ + cos^qH θ sin²φ], C = 4/(1/(qE+1)+1/(qH+1)) | BOR1 feeds [Kildal], co/cross-pol [Ludwig 1973] |
| 4 | Rim angle | 19 | tan(θ₀/2) = 1/(4 f/D) | [Balanis 15.4] |
| 4 | Aperture field | 20 | ρ = 2f tan(θ′/2); E_ap ∝ √G_f · cos²(θ′/2) | [Balanis 15.4] |
| 4 | Spillover | 21 | η_sp = Σ(1−cos^(q+1)θ₀)/(q+1) / Σ 1/(q+1) | [Balanis 15.4], [Kildal] |
| 4 | Illumination, polarisation, blockage | 22 | η_ill = \|∫E_co dA\|²/(A∫\|E_co\|²dA), η_pol = ∫\|E_co\|²/∫(\|E_co\|²+\|E_xp\|²), η_bl = (∫_unblocked E_co/∫E_co)² | [Kildal], [Balanis 15.4], [Ludwig] |
| 4 | Surface accuracy | 23 | η_s = exp(−(4πε/λ)²) | [Ruze 1966] |
| 4 | Gain, effective area | 24 | G = ηD, A_eff = Gλ²/4π | [Balanis ch. 2], [ERA ch. 3] |
| 5 | **The 3-D beam** | 25 | F_co = (1+cosθ)/2 · 2π[∫(A+B)/2 J₀(u)ρdρ − cos2φ ∫(A−B)/2 J₂(u)ρdρ], u = kρ sinθ | Fourier transform of the aperture [Balanis ch. 12]; Jacobi–Anger [DLMF 10.12.2, 10.9.2] |
| 5 | HPBW, sidelobes, Ω_A, Ω_M, D | 26a–e | P(θ_½) = ½; Ω_A = ∫P dΩ; D = 4π/Ω_A; η_MB = Ω_M/Ω_A | [Balanis ch. 2], [Kraus ch. 6] |
| 6 | Each part's gain vs frequency | 6, 7, 8, 9 | Butterworth/Chebyshev bandpass; coax L ∝ √f; datasheet roll-off; Touchstone S21 | [Pozar 8.3–8.4], [Pozar 1.7, 2.7], *model assumption*, [Pozar 4.3] |
| 6 | Each part's noise temperature | 2, 3 | active: T = 290(10^(NF/10) − 1); passive: T = T_phys(L − 1) | [Pozar 10.2] |
| 7 | **Receiver temperature (Friis)** | 4, 5 | T_rec = T₁ + T₂/G₁ + T₃/(G₁G₂) + …; G_total,dB = Σ G_i,dB | [Friis 1944], [Pozar 10.2] |
| 8 | Overall response | 5, 8 | feed band × every stage, SDR included | — |
| 9 | Sky temperature | 27 | T_sky = T_CMB + T₄₀₈(f/408 MHz)^β + T_atm | [Fixsen 2009], [Haslam 1982], [ITU-R P.372] |
| 9 | Ground pickup | 28 | reflector: f_g = (1−b)(1−η_sp); bare feed: pattern below the horizon line + back lobe | [Kraus ch. 6], [ERA ch. 3] |
| 9 | **Antenna temperature** | 29 | T_A = η_mm[η_ohm((1−f_g)T_sky + f_g T_ground) + (1−η_ohm)T_feed] | [Kraus ch. 6], [ERA ch. 3], [Pozar 10.2] |
| 10 | **System temperature** | 30 | T_sys = T_A + T_rec (same reference plane) | [ERA ch. 3] |
| 10 | **SEFD** | 31 | SEFD = 2kT_sys/A_eff (/10⁻²⁶ for Jy) | [ERA ch. 3] |
| 11 | Passband | 10, 11, 12 | edges = max of lower, min of upper; −3 dB points; B_N = ∫G df / G_peak | definition; [TMS], [Pozar 10.1] |
| 12 | Where the beam points | 36, 37 | sin δ = sin φ sin a + cos φ cos a cos A; t = θ_EW/(15.041°/h · cos δ) | [Meeus ch. 13, ch. 12] |
| — | Dynamic range | 33 | P = kTBG | [Nyquist 1928] |
| — | **SNR, and "I want SNR 3"** | 34 | SNR = (S/SEFD)√(n_pol B τ)/K_s; inverted for S or τ | [Dicke 1946], [ERA ch. 3], [Kraus ch. 7] |
| — | Line width → bandwidth | 35 | Δf = (Δv/c) f₀ | first-order Doppler [ERA ch. 7] |
| — | Rules of thumb (for comparison only) | 38a, 38b | HPBW ≈ 70λ/D; G ≈ 6(D/λ)² | [Kraus ch. 6], [ERA ch. 3] |

The **reference plane** is the feed output terminals. Feed ohmic loss and
mismatch therefore belong to the antenna (they reduce A_eff and add to T_A),
and the first cable is inside T_rec. That puts T_sys and A_eff at the same
plane, as Eq. 31 requires.

**Where the model is not textbook-exact**, it says so in the code:

* **Eq. 8, gain outside a part's datasheet range.** A datasheet gives a gain
  and a frequency range but not how the part behaves outside that range. The
  model keeps the datasheet value inside the range and falls off at a fixed
  dB/octave beyond it. This is an assumption, labelled as such, and it doesn't
  affect results in band.
* **Eq. 25, scalar aperture theory.** It is exact for the uniform aperture:
  HPBW 1.029λ/D and a −17.6 dB first sidelobe, both checked in the tests. The
  directivity from integrating the pattern agrees with aperture theory to
  0.07 dB for the example dish.
* **SDR selectivity** is modelled as a steep (order-12) Butterworth across its
  I/Q bandwidth, standing in for its real anti-alias and decimation filters.

### Full references

| key | reference |
|---|---|
| SI | BIPM, *The International System of Units*, 9th ed., 2019 |
| Pozar | D. M. Pozar, *Microwave Engineering*, 4th ed., Wiley, 2012 |
| Friis | H. T. Friis, "Noise figures of radio receivers," *Proc. IRE* 32, 419–422, 1944 |
| Nyquist | H. Nyquist, "Thermal agitation of electric charge in conductors," *Phys. Rev.* 32, 110–113, 1928 |
| Balanis | C. A. Balanis, *Antenna Theory: Analysis and Design*, 4th ed., Wiley, 2016 |
| Kildal | P.-S. Kildal, *Foundations of Antenna Engineering*, 2015 |
| Ludwig | A. C. Ludwig, "The definition of cross polarization," *IEEE Trans. AP* 21, 116–119, 1973 |
| Ruze | J. Ruze, "Antenna tolerance theory — a review," *Proc. IEEE* 54, 633–640, 1966 |
| DLMF | NIST Digital Library of Mathematical Functions, <https://dlmf.nist.gov/> (10.9.2, 10.12.2) |
| ERA | J. J. Condon & S. M. Ransom, *Essential Radio Astronomy*, Princeton, 2016; free at <https://www.cv.nrao.edu/~sransom/web/Ch3.html> |
| Kraus | J. D. Kraus, *Radio Astronomy*, 2nd ed., Cygnus-Quasar, 1986 |
| Dicke | R. H. Dicke, *Rev. Sci. Instrum.* 17, 268–275, 1946 |
| Haslam | C. G. T. Haslam et al., "A 408 MHz all-sky continuum survey. II," *A&AS* 47, 1–143, 1982 |
| Fixsen | D. J. Fixsen, "The temperature of the CMB," *ApJ* 707, 916–920, 2009 |
| ITU-R P.372 | ITU-R Recommendation P.372, *Radio noise* |
| ITU-RR | ITU Radio Regulations, footnote 5.340 (1400–1427 MHz) |
| Hellwig | H. Hellwig et al., *IEEE Trans. Instrum. Meas.* 19, 200–209, 1970 (HI rest frequency) |
| Meeus | J. Meeus, *Astronomical Algorithms*, 2nd ed., Willmann-Bell, 1998 |
| TMS | A. R. Thompson, J. M. Moran, G. W. Swenson, *Interferometry and Synthesis in Radio Astronomy*, 3rd ed., Springer (open access), 2017 |
| Touchstone | IBIS Open Forum, *Touchstone File Format Specification* |

Book references are given by chapter or section title. Section numbers can
move between editions, so check them against the edition you have.

---

## The input: `hardware_parameters.yaml`

Every number is written the way a datasheet gives it:

```yaml
noise_figure: {value: 0.7, unit: dB, status: datasheet, notes: at 1.4 GHz}
```

| key | meaning |
|---|---|
| `value` | the number (or text, for things like `type`) |
| `unit` | `Hz kHz MHz GHz`, `m cm mm ft in`, `K C`, `deg`, `dB dBm dBi`, `dB/m dB/100ft`, `s min h`, `W V`, `bit`, `""`, `":1"` (ratios such as VSWR) |
| `status` | `datasheet` (default), `measured`, `estimate`, `placeholder` |
| `notes` | free text. **Quote it if it contains a comma**: `notes: "selectable 12, 10 or 8"` |

The loader rejects anything it doesn't understand and says why. Both
mistakes in the draft file are caught. `{value: 1.8, unt: :1}` gives
"unknown key 'unt' (did you mean 'unit'?)"; a bare `:1` is also invalid YAML,
so write `unit: ":1"`. An unquoted comma inside `notes:` gives a hint to
quote it.

**Values that change with frequency** can be given as a curve
(`gain: {unit: dB, freq_unit: GHz, curve: [[1.3, 20.6], [1.4, 19.81]]}`), as a
two-column CSV (`file: lna_gain.csv`), or for an amplifier or filter as a
manufacturer's Touchstone file (`sparameters: ZX60-P162LN+.s2p` on the
component). **The LNA's `.s2p` is the most valuable upgrade:** the model
currently assumes flat gain inside each part's band.

**`environment`** holds the inputs T_A needs (Eqs. 27–29):
`site_latitude` (36.003147° by default), `ambient_temperature` (the default
physical temperature of every part), `ground_temperature`, `cmb_temperature`,
`galactic_temperature_408` and `galactic_spectral_index`,
`atmosphere_temperature`, and `horizon_elevation`.

**`components`** is one list of structure and RF chain. `type` decides the
model:

| type | parameters it reads |
|---|---|
| `reflector` | `diameter`, `focal_ratio`, `surface_rms_error`, `blockage_fraction`. Set `enabled: false` to point the feed at the sky by itself |
| `feed` | `hpbw_e`, `hpbw_h` **or** `edge_taper_e`, `edge_taper_h`; `diameter` (circular waveguide band edges); `freq_min`/`freq_max`; `ohmic_loss`; `vswr`; `back_radiation_fraction`; `e_plane_azimuth` |
| `mount` | `pointing_alt`, `pointing_az` |
| `amplifier` | `gain`, `noise_figure`, `freq_min`, `freq_max`, `rolloff` (dB/octave beyond the band, default 12), `p1db_output` |
| `filter` | `center_freq`, `bandwidth`, `insertion_loss`, `out_of_band_rejection`, `response` (`butterworth`/`chebyshev`), `order`, `ripple` |
| `cable` | `length` + `loss_per_length` + `loss_ref_freq` (scaled as √f, Eq. 7), or a total `loss` |
| `attenuator` | `loss` |
| `sdr` | `sample_rate`, `if_bandwidth`, `noise_figure`, `fft_size`, `n_pol`, `radiometer_factor`, `tuned_freq` |

Any passive part also takes `physical_temperature`, the T_phys in Eq. 3. A
cable in the summer sun on the roof is noisier than the same cable indoors.

---

## Rearranging the chain

RF parts run in `chain_position` order. Renumber them, or override the order
with `--order a,b,c`. `enabled: false` removes a part. `--compare` puts several
orders on one figure, and `--all-orders` ranks every permutation (the SDR
stays last).

| order | T_sys | SEFD |
|---|---|---|
| lna > feed_cable > filter > coax_run > sdr (best, with five ties) | 83.2 K | 2.01 × 10⁵ Jy |
| feed_cable > lna > filter > coax_run > sdr (**current**) | 86.6 K | 2.09 × 10⁵ Jy |
| filter > lna > … | 171.9 K | 4.16 × 10⁵ Jy |
| coax_run > lna > … | 214.1 K | 5.17 × 10⁵ Jy |
| filter > feed_cable > coax_run > lna > sdr (worst) | 342.5 K | — |

These follow from the 1/G divisors in Friis (Eq. 4):

* **Whatever comes before the LNA sets the noise floor.** 1 dB of loss ahead
  of the LNA adds about 75 K. Behind 20 dB of gain it costs under 1 K. Moving
  the 10 m coax ahead of the LNA takes T_sys from 87 K to 214 K. Integration
  time scales as T_sys² (Eq. 34), so the same detection takes 6× longer.
* **Once the LNA is first, the order of the rest barely matters.** Six orders
  tie at 83.2 K. The 0.3 m pigtail ahead of the LNA costs 3.4 K.
* **The ranking only measures noise.** A filter before the LNA costs about
  85 K, but it protects the LNA from strong out-of-band signals.
  `lna_compression_headroom` (the LNA's P1dB minus the noise power over its
  whole band) is the first check on that trade-off. Only an RF survey of the
  roof can settle it.

---

## The 3-D beam, and why its orientation matters

When the feed's E- and H-plane patterns differ (Eq. 15), the beam is
elliptical (Eq. 25). The example cantenna (66° × 80°, placeholder values) on a
1.5 m dish gives a 9.9° × 9.2° beam, 8% elliptical. The 0.4% of the feed's
power that goes cross-polar is lost (η_pol, Eq. 22).

At the zenith, local north is +Dec and east is +RA, and a fixed mount never
rotates the beam. So:

* the beam's **north–south** width sets how wide a **declination strip** is
  swept, and
* its **east–west** width sets how long a source takes to drift through
  (Eq. 37), which is the integration time per transit.

`e_plane_azimuth` sets which way the feed is turned. Rotating a strongly
asymmetric feed by 90° swaps the two. `beam_3d.png` shows the pattern in 3-D,
the footprint on the sky with the −3 dB contour, and the two principal cuts.

Without the reflector (`enabled: false`), the same feed gives a 66° × 80° beam,
A_eff ≈ 0.027 m² and SEFD ≈ 8.5 × 10⁶ Jy, about 40 times less sensitive.

---

## "I want an SNR of 3": `observing_goals.yaml` → `detection_thresholds.yaml`

You state the SNR you want, the integrations to tabulate, and a typical HI
line width. Eq. 34 is then solved for the flux each kind of measurement needs:

| measurement | bandwidth | one transit, SNR 3 |
|---|---|---|
| continuum | noise bandwidth, 6.06 MHz (Eq. 12) | **4.9 Jy** |
| HI, one channel | 5.9 kHz | 157 Jy |
| HI, matched to a 200 km/s line | 948 kHz (Eq. 35) | **12.4 Jy** peak |

`database_index` turns the first integration into catalogue floors. hydroline
applies the same idea with
`run.py --performance … --snr-target 3`: it queries the live NVSS catalogue
only above the flux that reaches SNR 3 in one transit.

What to expect:

* **A fixed beam sees each source for one transit a day** (45 min here). For a
  point source, "24 h" isn't available. Stacking N days gains only √N. The
  long integrations apply to emission that fills the beam all day, such as
  Galactic HI.
* **Galactic HI is easy; other galaxies are hard.** A 12 Jy HI peak is
  brighter than any galaxy in the northern sky except M31, and M31 lies about
  5° outside the beam at this latitude.
* **Continuum**: a 10° beam is confusion-limited. Only the very brightest
  sources (Cas A, Cyg A, Tau A) stand out individually.

---

## The frequency-response graph (`frequency_response.png`)

Four panels over 1380–1460 MHz:

1. gain after each part, and the total, with and without the SDR's
   selectivity;
2. T_sys, T_rec, T_A and T_sky;
3. each stage's Friis contribution, stacked;
4. SEFD, and the flux for your SNR target in one channel during one transit.

The analog chain is shown across the whole sweep because the SDR can be
retuned. `frequency_response.csv` has every curve, plus each stage's gain and
noise contribution, at each frequency.

---

## Limits

* Flat in-band gain unless you give a curve or an `.s2p` file. Out of band,
  the roll-off is a labelled assumption (Eq. 8).
* Filters are ideal Butterworth/Chebyshev prototypes (Eq. 6) until measured.
* Mismatch is treated as an efficiency only (Eq. 17). The LNA's noise
  actually depends on the source impedance the feed presents (its noise
  parameters), which can be worth several K at 1.5:1 VSWR.
* Reflective out-of-band filter skirts use T = T_phys(L−1) (Eq. 3), the
  standard engineering cascade. In-band numbers don't depend on this.
* One uniform T_sky (Eq. 27). When the Galactic plane drifts through, T_A
  rises by tens of K; hydroline's Galactic HI estimate shows when.
* For a tilted mount the declination is right (Eq. 36), but the beam's
  rotation on the sky (parallactic angle) is only exact at the zenith.

## A note on the site coordinates

`config.example.yaml` (hydroline's config) gives the Fitzpatrick roof as
`36.0011, -78.5623`. The comment beside it, 36°00′11.33″N 78°56′23.53″W,
converts to **36.003147, −78.939869**: the minutes and seconds were typed in
as decimals. The longitude error, 0.38°, shifts hydroline's transit
predictions by about 1.5 min. This program uses the corrected latitude.
hydroline's config has not been changed.
