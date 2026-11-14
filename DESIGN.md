# ToxicElement15 — Design Notes

A modular re-implementation of the original single-file `te15.c` SDR
waterfall analyzer. Same stack (C11, GTK3, FFTW3f, Cairo, mmap'd gqrx raw
IQ), same behaviour, cleaner structure.

## Modules (`src/`)

| File | Responsibility |
|------|----------------|
| `iq_source.h/.c`  | Open/mmap a gqrx raw IQ file, parse `gqrx_*_FREQ_RATE_fc.raw` names (incl. start wall-clock), RFC 3339 formatting |
| `colormap.h/.c`   | Viridis 256-entry LUT (canonical anchors, linear interpolation) |
| `dsp.h/.c`        | Hann window, FFT→dB spectra, entropy, PRF estimate, FftPlan wrapper |
| `signals.h/.c`    | Known-signal database, confidence matching, ranked candidates |
| `fit.h/.c`        | Aspect-preserving contain-fit geometry (pure, unit-tested) |
| `region.h/.c`     | (freq, time) selection rect + normalisation, analysis sample cap |
| `detect.h/.c`     | automatic signal detection over a spectrogram window (noise-floor thresholding, run growth, time tightening) |
| `waterfall_cache.h/.c` | Background spectrogram worker + LRU window cache |
| `analysis.h/.c`   | Region spectrogram + statistics (entropy, PRF, power, classification) |
| `metrics.h/.c`    | Extended feature suite, modulation classification, demodulators |
| `wav.h/.c`        | 16-bit mono PCM WAV writer |
| `audio.h/.c`      | CoreAudio AudioQueue playback (16-bit PCM, atomic position, seek) |
| `cyclic.h/.c`     | FAM spectral correlation: SCF, cyclic domain profile, harmonic peak extraction |
| `fluct.h/.c`      | Burel fluctuation-of-autocorrelation DSSS detector (windowed variance of the ACF estimate, normalised F(tau) = T·Var/sigma^4) |
| `cpar.h/.c`       | Cumulative Peak-to-Average Ratio DSSS detector (FFT windowed ACF, global-baseline PAR per peak, CPAR criterion, code-period + chip-period recovery) |
| `hos.h/.c`        | Higher-order statistics: 2nd-6th order cumulants (Swami & Sadler), normalised |C40|/|C41|/|C42| over C21^2, Gaussian-suppression verdict, bicoherence bispectrum |
| `dagcore.h/.c`    | Generic analysis-DAG engine: node registry, dependency ordering, statuses, skip rationales, timings, cancellation, progress; used by both the Spectrum and Frequency Counter panes |
| `dag.h/.c`        | Spectrum analysis node registry (A0..A9, post-DAG aggregations) and typed façade over `dagcore` |
| `fcdataset.h/.c`  | Measurement datasets per `schema/fc_dataset.json`: schema-validated JSON load/save (json-c), dataset→session→round→subject→measurements model, statistics, deep copy |
| `fcdag.h/.c`      | Frequency Counter analysis registry on `dagcore` (empty until the algorithms are defined) |
| `fcreport.h/.c`   | Frequency Counter report rendering (text/Markdown/HTML from one document model) |
| `ui.h/.c`         | GTK3 UI: waterfall widget, controls, selection, timeline scrollbar + RFC 3339 scrub tooltip, analysis popup, demod player, cyclostationary window, fluctuation window, menu bar |
| `main.c`          | CLI parsing, wiring, `--check` headless verification mode, File>Open integration |

## Concurrency model

* `WaterfallCache` owns one worker thread and a fixed number of window
  slots. Each slot holds a contiguous run of computed spectra and its own
  mutex.
* The worker computes into a slot marked invalid, then flips it valid.
  Readers (`wc_lock`/`wc_unlock`) hold the slot mutex while reading, so
  eviction (which frees under the same mutex) can never free data being
  drawn.
* Everything else is main-thread only. The UI never touches cache fields
  directly; progress is polled via `wc_progress()` from a 50 ms timer.

## Rendering

* The waterfall maps time to the vertical axis (oldest at the bottom) and
  frequency to the horizontal axis (full sample-rate span).
* Rows are decimated when more FFT frames than pixel rows are visible,
  so every draw pass costs O(pixels), not O(frames × bins).
* Colours come from the shared viridis LUT; dB values are normalised to
  [0,1] over the visible range (auto) or the slider range (manual).

## The "Selected Region" fit rule (the fix)

When the analysis popup opens, the "Selected Region" box renders the
region's spectrogram with an aspect-preserving *contain* fit
(`fit_contain`): the image is scaled by
`min(box_w/img_w, box_h/img_h)` and centred, so the selection always
fills the box as much as possible in both axes. A region wider than tall
is width-constrained and centred vertically; a region taller than wide is
height-constrained and centred horizontally. Resizing the popup re-fits
automatically (the draw handler reads the current allocation).

## Timeline scrollbar & scrub tooltip

A vertical `GtkScrollbar` sits right of the waterfall. Its adjustment
maps `value = scroll_position`, `upper = duration - visible_seconds`,
`page_size = visible_seconds` (thumb length ∝ visible fraction). Model →
adjustment updates go through `sync_scrollbar_from_model()` with a
`syncing_scrollbar` guard; user drags flow back through "value-changed"
into the model, the cache request, and the redraw. While scrubbing, a
`GTK_WINDOW_POPUP` tooltip follows the pointer and shows the position as
an RFC 3339 local timestamp (`iq_source_format_time`, start wall-clock
parsed from the gqrx filename).

## Auto-detection of signals in view

Pressing Analyze without a manual selection runs `detect_signals()` over
the cached window intersecting the visible view: frames are averaged into
a PSD, the noise floor is the median bin, and contiguous above-threshold
runs (floor+6 dB, gaps ≤ 2 bins merged) become candidates. Runs are then
merged iteratively — a short gap (≤ 16 bins) that never falls below the
floor bridges wide structures such as FM comb spectra, and fragments
within 128 bins whose per-frame band power is strongly correlated
(r ≥ 0.8) merge pulsed carriers with their side lobes. Marginal remnants
below floor+14 dB are dropped and the top 16 by peak power survive. Each
candidate is tightened in time via its per-frame band power (longest
on-run). The UI draws them as green clickable rectangles; clicking one
clears all rectangles, turns it into the selection, and opens the
analysis popup. Rectangles are discarded on scroll/zoom or when a manual
drag starts.

## Analysis on a private segment buffer

`analysis_run()` and `metrics_compute()` first derive the sample range
from the selection, copy that part of the recording out of the memory
map into a private heap buffer (capped at 32 M samples ≈ 256 MB), and
run entirely on the buffer — the mmap is never touched during analysis.

## Extended analysis suite & demodulation

`metrics_compute()` measures, over the selected region:
occupied (99%) bandwidth, spectral flatness, sideband symmetry, tone
purity, SNR, AM depth, envelope crest factor, duty cycle, FM deviation
(robust MAD), FSK level count (with a peak-concentration gate that
rejects swept-FM arcsine histograms), symbol/envelope rate, and a chirp
trend. A threshold decision tree classifies CW / AM / FM / FSK / PSK /
OOK / OFDM / CHIRP / NOISE — thresholds are regression-tested against
synthetic recordings of each class. `signals_rank()` re-ranks the known-
signal database by base confidence plus measured-parameter agreement
(occupied-bandwidth containment ±15, class-keyword +15), giving a ranked
top-3. FM/AM/CW selections expose "Demodulate Audio": mixdown to the
region centre, discriminator/envelope/beat-note demodulation, box-
filter decimation to 32 kHz mono, and a 16-bit WAV via `wav_write`.

## Demod playback, saving, and File > Open

The demodulated audio (32 kHz mono floats) is kept in the UI after the
first computation. "Play Demod" opens a minimal player window (seek
bar, Play/Pause, m:ss.t position) driving an AudioQueue player; the
queue callback advances an atomic frame position, so seeking/position
reads are lock-free from the UI thread. "Save Demod" opens a
GtkFileChooserDialog pre-named `demod_<...>.wav` and writes it via
`wav_write`. The main window has a menu bar: File > Open… (callback into
main.c, which swaps the heap-owned IqSource and rebuilds the waterfall
cache, then calls `ui_reset_source`), File > Settings… (placeholder
window), and an empty Analyze menu reserved for future work.

## Cyclostationary analysis

`cyclic_analyze()` runs a FAM (FFT Accumulation Method) spectral
correlation over the selection: FIR-decimated mixdowned data (256 kHz
effective rate), 128 Hamming frames of 1024 bins, coherence-normalised
SCF on a 1024×1024 grid, and a cyclic domain profile (max coherence per
alpha). Odd-alpha rows use dual half-bin averaging; the alpha≈0
self-coherence zone is masked for detection; peaks must clear both the
statistical and median-relative thresholds, and wall-hugging peaks
require a corroborating harmonic (rejecting a pure tone's decay tail).
Reported peaks are sub-bin refined and sorted with the fundamental
(symbol/baud/chip rate) first. The UI shows the SCF image, the CDP plot
with marked peaks, and the ranked cyclic feature list.

## About screen

**File > About** opens a modal screen whose background is the
dark-industrial artwork with **Toxic Element 15**,
**Daniel R. Azulay | Radio Defense** and **2026** overlaid in bold
pale-green text with a black border. The artwork (`media/about.gif`) is
converted to a C array by `tools/bin2c.c` at build time and **embedded in
the executable**, so the screen never reads it from disk at run time; it
closes only on an explicit user action (Escape/Return or the window's
close button).

## Main window layout

Below the menu bar the main window is a GtkNotebook with two tabs. The
**Spectrum** tab holds the whole view - the HPaned with the waterfall +
timeline scrollbar on the left and the dockable analysis pane on the
right, plus the bottom toolbar and status bar. The second tab,
**Frequency Counter**, is split by an HPaned: the left side loads series
of frequency measurements (one frequency in Hz per line, from text files),
keeps them in a list view and lets the user remove sessions and edit each
session's label, location (latitude/longitude), date/time and notes; the
right side is a notebook with an **Analysis** page (Analyze button,
progress bar, Text/Markdown/HTML report selector, Save report and a
read-only report view, mirroring the Selection tab) and an empty
**Settings** page.

### Subcarrier-lattice analyses

The Frequency Counter analysis DAG (`fcdag.c`) runs a chain of tests over
the loaded measurements: blocks are built per session round (all subjects
combined) and per subject. The lattice tests use the candidate spacings
5, 15, 30, 60, 120 and 240 kHz (LTE/NR numerologies plus the general 5 kHz
case):

* **Rayleigh** test on phases modulo df (unknown lattice offset) with the
  asymptotic p and a permutation p (uniform frequencies inside the block).
* **V test** directed at phase 0 (measurements on the assumed lattice),
  permutation p; ignores clustering at other phases.
* **Phase dispersion** (circular variance 1-R, small = clustering) with a
  lower-tail permutation p.
* **Periodogram scan** of df from 2 to 300 kHz in 1 kHz steps (top five
  spacings per block; not corrected for the scan).
* **Chi-square** test of lattice-centred bins (width 0.25 df, uniform
  expectation; only cells with at least 5 expected counts).
* **KDE enrichment** at the lattice points (Silverman bandwidth capped at
  df/4) with a permutation p.
* **Cluster-based permutation** test over 64 block bins (cyclic clusters,
  uniform null).
* **Silverman** critical-bandwidth bootstrap for multimodality
  (Hartigan's dip is not implemented in this build).
* **Multiple testing**: Benjamini-Hochberg and Bonferroni correction per
  test family with the rejection counts.
* **Summary**: the best spacing per block with a rose phase histogram and
  a lattice comb plot rendered into the report.

### K-subcarrier averaging and spacing synthesis

A second analysis chain tests whether the measurements are equal-weight
averages of K latent subcarriers on the candidate grid (induced lattice
df/K): an **induced-lattice scan** over K = 1..6 with permutation p-values
(Bonferroni over K), a **phase-dispersion multiplicity** diagnostic, a
**characteristic-function slope** estimate, a **grid-constrained Gaussian
mixture fit** with EM and BIC/ICL model selection (the primary
identifiability route), an **MDL/eigen-spectrum** node that is skipped with
an explicit rationale (frequency lists carry no signal covariance), and a
**K model-selection** table combining the methods with a confidence count.
The **evidence screen** then discards analysis results that show little
evidence (p >= 0.05, or DeltaBIC <= 2 for the mixture model) and keeps the
significant ones, and the **synthesis** node aggregates the retained
evidence per subcarrier spacing and per block, ranks the spacings
(weighted -log10 p, family weights), reports the implied K and prints the
overall ranking with the most likely subcarrier spacing. The DAG skips
whole branches when their premise is absent: the K chain is skipped when
no subcarrier clustering was found, the mixture fit when no
induced-lattice clustering was detected, the CF slope when the K=1
residual dispersion cannot serve as a single-subcarrier reference, and the
MDL node always (no signal data).

The Frequency Counter **Settings** tab configures the analysis scope:
analyse every dataset separately (session rounds and subjects) or merge all
loaded measurements into one list. The choice is applied to the next
Analyze run and stated in the report header.

Every test is deterministic (fixed RNG seed), reports effect sizes (R, V,
theta, enrichment, cluster mass) as well as p-values, and the report
renders the tables as text, Markdown and HTML.

Datasets are loaded and written with json-c against the JSON schema:
required properties are enforced, unknown properties are rejected
(`additionalProperties: false`), coordinates require both latitude and
longitude, ids may be strings or integers, and every failure message names
the JSON path of the offending value.

The Analysis page is driven by the shared DAG engine in `dagcore.h/.c`:
node identities, dependency ordering, statuses with skip rationales,
timings, cancellation and progress all live there, and both panes use it
the same way. The Spectrum pane keeps its imperative worker (its branching
is data dependent) and records each step through the engine's explicit
begin/end/skip API via its `dag.h` façade; the Frequency Counter defines
its own registry (`fcdag.h/.c`, still empty - analyses to be defined) and
lets the engine walk it with an executor callback. Report rendering is a
separate concern: `report.h/.c` renders the Spectrum document and
`fcreport.h/.c` the frequency-counter document, each in text/Markdown/HTML. The docked pane can be detached into its own window and
re-attached, and its Settings tab comes last.

## Docked analysis pane

The main window is an HPaned: waterfall + timeline scrollbar on the
left, a GtkNotebook dock pane on the right with five dockable analysis
pages. Selection Details is live — it re-runs the analysis whenever the
selection changes (rectangle click or drag release) and shows a no-data
state otherwise. The other four pages are Run-button driven: they
analyse the current selection, or the entire visible part of the
waterfall when nothing is selected. The Analyze menu is gone (only
File > Open… / Settings… remain). Detected rectangles are widened to a
minimum on-screen width so narrow pure tones stay visible and
clickable.

## Find Signals workflow

The "Find Signals" button (and `A`) highlights probable signals with
yellow rectangles. Clicking one selects it (its rectangle turns red
under the selection overlay) while the other rectangles stay; clicking
empty space clears the rectangles and starts a manual drag. Analyze →
Selection Details opens the analysis window ("Selection Details") for
the current selection; Analyze → Cyclostationary Analysis opens the
cyclostationary window.

## Fluctuations of autocorrelation estimators

`fluct_analyze()` implements Burel, Bouder & Berder (ISPACS 2000): the
selection is FIR-decimated to ~100 kHz and split into M windows of 512
samples; per window and lag the autocorrelation estimate is computed,
and per lag its variance V(tau) across windows is normalised to
F(tau) = T·V(tau)/sigma^4. Noise alone gives F ≈ 1 (chi-square, mean 1,
std 1/sqrt(M)); a hidden DSSS signal raises F at the chip-period lag by
the signal's fourth-order statistics, far below the SNR at which the
mean peak emerges (verified at -10 dB chip SNR: F = 6.2 vs noise 1.3).
The detection threshold is max(1 + 4/sqrt(M), 2.0); the peak lag yields
the chip period and a rough chip-SNR. The window shows the (flat) mean
autocorrelation and the spiked fluctuation curve with baseline,
threshold and peak markers.

## Cumulative Peak-to-Average Ratio

`cpar_analyze()` implements the CPAR DSSS detector (Djaouad & Ait
Hemiche, IEEE): the selection is FIR-decimated to ~500 kHz and split
into 16384-sample windows whose FFT-based autocorrelations are averaged
and normalised. Candidate peaks are local maxima above 2x the median;
each peak's PAR is its power over the mean squared autocorrelation
across the whole lag range (the global noise baseline), and the CPAR
criterion sums the PARs of significant peaks (PAR > 25). A repeating
spreading code produces a peak at the code period with a large PAR
(regression: CPAR 136 at the exact 12.7 ms m-sequence period, noise
CPAR 0); the chip period is recovered from where the chip-correlation
triangle first crosses the noise floor (9.7 kHz estimated vs 10 kHz
true). The window shows the normalised ACF with peak markers, the PAR
per lag with the threshold, and the verdict.

## Higher-order statistics

`hos_analyze()` computes the standard higher-order cumulants of the
centralised mixdowned selection (Swami & Sadler definitions): C20, C21,
C30, C31, C40, C41, C42, C60, C63, plus the normalised magnitudes
|C40|/C21², |C41|/C21², |C42|/C21² and the complex kurtosis. Gaussian
noise has zero cumulants beyond second order (Mendel, Proc. IEEE 1991),
so |C42|/C21² (asymptotic std sqrt(24/N) under Gaussianity) suppresses
noise and reveals non-Gaussian signals: the regression shows 0.72 for
the BPSK DSSS vs 0.0000 for equal-power Gaussian noise at a 5-sigma
threshold, with a BPSK/QPSK/QAM-style hint from the classic cumulant
features. A windowed bicoherence bispectrum (principal domain) is also
estimated and displayed.

## Spooner CSP suite (SSCA, cycle detectors, cyclic cumulants)

`ssca.c` implements the Strip Spectral Correlation Analyzer (N'=64
channelizer hopped by L=16, 32 strips, per-strip demodulate products,
P-length strip FFTs, Spooner coarse-PSD coherence, CFAR-thresholded CF
list with the q/(P·L) alpha mapping). `cycledet.c` implements the CFAR
cycle detectors: the lagged cyclic autocorrelation x(n)x*(n-tau) at 7
lags, DC-removed per frame, whitened with H_w = 1/sqrt(S(f)), with an
empirically calibrated Gaussian threshold (100/sqrt(np·n)) and a
single-cycle verdict. `ccc.c` implements modulation recognition via
cyclic temporal cumulants: |C40(4fc)|, |C41(2fc)|, |C42(2fc)| / P^2
with hierarchical BPSK/QPSK/8PSK/16QAM verdicts. `cyclic.c` gained the
conjugate SCF (cyclic_analyze_ex), the coherence CF list and its
threshold. `dsp.c` gained dsp_srrc_pulse (SRRC test-signal generator)
and dsp_whiten.

## Verification

* `make test` — unit tests (no GUI): filename parsing, contain-fit
  geometry, colormap endpoints, entropy, PRF, signal matching, and an
  end-to-end spectrum/analysis run on a synthetic recording.
* `./te15 --check <file>` — headless end-to-end: loads the recording,
  computes a cache window, runs a region analysis, prints results.
* `make gen-test-data` — writes `data/test_iq.raw` (2 MS/s, 5 s, CW +
  FM + pulse-train signals) via `tools/gen_test_iq.py`.
