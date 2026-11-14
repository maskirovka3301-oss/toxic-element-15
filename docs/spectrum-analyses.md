# Spectrum tab — the implemented analyses

The Spectrum tab runs one analysis suite over the user's selection (or the
visible waterfall when nothing is selected) and renders the result as the
Selection report. The suite is organised as a DAG: a cheap baseline, three
exploratory probes, then progressively more expensive detectors that are only
run when their precondition holds. The node set is declared in
`src/dag.h` (`DAG_A0` … `DAG_SSCLASS`) and registered in `src/dag.c`; the
execution order and the gating decisions live in the analysis worker in
`src/ui.c`; the per-analysis result schemas live in `src/aregistry.c`.

```
SOT → A0 (analysis + metrics)
    → A1 classic → A2 pulse → A3 HOS
    → wideband ? A5 SSCA : A4 FAM [→ A5 if FAM finds no cycles]
    → A6 cycle detectors + CPAR      (only if cycle frequencies found)
    → A8 cyclic cumulants            (only if cycle frequencies found)
    → A9 intra-burst envelope        (only if bursty/pulsed)
    → A7 Burel fluctuation           (only if cycles were not validated)
    → MODCONF → signal DB match → SSCLASS → Director
```

Each node records *executed / failed / skipped* plus a reason, and the report
prints that provenance. A skipped node therefore never silently counts as a
negative result: the rationale (e.g. "no candidate cycle frequencies") is part
of the output.

The sections below document every implemented analysis in execution order. The
same algorithms also back the live Selection pane and the Selection report;
this document describes each one once, from the report's point of view.

---

### Common source of truth (SOT)

**Description.** Before anything else, one shared buffer is prepared from the
recording (`sot_prepare`, `src/sot.c`): the selection's time range is copied
across the *full* original band at the original sample rate, mixed down at the
selection centre, and band-filtered. The pass/stop edges are derived from the
wideband context (the noise floor, the selection's own spectral extent, and
the nearest out-of-selection emission). A uniform real gain normalises the
slice, and the buffer is exposed as a pseudo-`IqSource` so every analysis reads
the same data through the existing API.

**Rationale.** Without a single prepared source, each analysis re-extracts,
re-mixes and re-filters the region with slightly different parameters — the
detectors then disagree for reasons that have nothing to do with the signal.
Preparing once also keeps the OOK/ASK edge sidelobes (needed by the pulse and
intra-burst stages) while rejecting neighbouring emissions, and it prevents a
fixed low-rate decimation from aliasing away the resolution the later stages
need. Each analysis may still decimate for cost, but only by a factor
`dsp_decimation_for()` allows.

**What it can prove about a signal.** By itself, nothing: SOT is the measuring
instrument, not a measurement. But its parameters are evidence in their own
right — `pass_hz`/`stop_hz` bound how wide the emission is and how far away the
next emitter sits, and `noise_floor_db` is the wideband reference against which
every detector's thresholds are set. More importantly, it makes the
cross-analysis agreement reported by the Director meaningful, because all
detectors are then looking at identical complex baseband data.

---

### A0 — Baseline region analysis

**Description.** `analysis_run()` (`src/analysis.c`) computes a spectrogram of
the selection (a `frames × bins` image, with `bin_hz` and `dt`), the centre
frequency and bandwidth, the duration, average and peak power, the Shannon
entropy of the spectrogram, a frame-level fluctuation rate (`prf`), and coarse
`signal_type` / `modulation` labels.

**Rationale.** This is the cheapest stage and the one every other stage
depends on: it answers "where is the energy, how wide is it, how strong is it,
and does it look structured or noise-like" without committing to a modulation
hypothesis. It anchors the rest: the measured occupied bandwidth, spectral
flatness, crest factor and classifier labels are what let the DAG decide
whether the input is wideband/OFDM-like (go to SSCA) or narrowband (go to FAM),
and the measured duty cycle and AM depth are part of what decides whether the
intra-burst stage has a burst to zoom into.

**What it can prove about a signal.** The emission's centre and occupied extent,
its power level, whether its spectrum is ordered or noise-like (low entropy =
concentrated/structured; high entropy = spread/noise-like), whether its power
fluctuates frame-to-frame (pulsed/bursty) or is stationary, and a first coarse
class (CW/AM/FM/FSK/PSK/OOK/OFDM/CHIRP/NOISE). It is the stable "what, where and
how strong" layer that the later claims refer back to.

---

### A0 — Signal metrics (extended feature suite)

**Description.** `metrics_compute()` (`src/metrics.c`) measures, over the
selection: occupied (99 %) bandwidth and its absolute edges, spectral SNR,
spectral flatness (Wiener entropy), band symmetry around the centre, PAPR
(envelope crest factor), tone purity (fraction of power in the strongest bin),
AM depth, duty cycle, FM deviation (robust MAD of the instantaneous frequency),
FSK level count (with a peak-concentration gate that rejects swept-FM arcsine
histograms), symbol/envelope rate with a confidence, and a chirp trend. It also
derives a modulation-class hypothesis and a demodulation recommendation.

**Rationale.** The classifier labels alone are coarse; a claim such as "FM" is
only defensible when it is backed by a measured deviation and a low AM depth.
Presenting physical parameters alongside the class lets the reader check the
label against the numbers, and lets the Director and the modulation consensus
score hypotheses from feature consistency rather than from a single string.
The FSK gate exists because a swept FM signal produces the same bimodal
instantaneous-frequency histogram as a 2-FSK signal.

**What it can prove about a signal.** The quantitative identity card: how much
bandwidth it really occupies (99 % containment, not the selection width), how
far it stands above the noise, whether the spectrum is flat (spread/multicarrier)
or peaky (tone), whether the modulation is symmetric (AM/FM) or leans
(SSB/offset), how deep the envelope modulation is, what fraction of the time it
is on (OOK/pulsed), the FM deviation and FSK level count, the symbol/burst rate,
and whether the IF drifts linearly with time (chirp). This is the layer that
separates "a carrier" from "a carrier modulated in this specific way", and it
determines whether a demodulator applies.

---

### A0 — Signal database match and anomaly check

**Description.** `signals_rank_conf()` (`src/signals.c`) scores every entry of
the known-signal database against the measured centre frequency and bandwidth,
then re-ranks by measured-parameter agreement: occupied-bandwidth containment
(±15), modulation-family match (+20) or mismatch (−15), and family-specific
consistency (e.g. Carson's rule for FM, duty cycle for OOK, crest factor for
OFDM, symbol-rate tables for known digital modes, +10 each). Each match carries
a validity verdict — in the licensed band, bandwidth inside the documented
channel, modulation family conforming to the specification — and
`signals_anomaly_check()` reports an anomaly when no candidate is valid. The
registry files this stage under `DAG_A0`, but it is evaluated after MODCONF
because it consumes that consensus as well as the metrics.

**Rationale.** Frequency and bandwidth alone produce many plausible hits; the
physical measurements break the tie. The separate validity gate matters because
"the best match" and "a legal, conforming emission" are different claims: a
signal can resemble a system while sitting outside its band or occupying the
wrong channel width. Reporting the anomaly explicitly is often more useful than
reporting a weak match.

**What it can prove about a signal.** Whether the emission is *consistent with*
a documented system — its band plan, channel width, modulation family and
symbol rate — and, when nothing conforms, that it is anomalous with respect to
the database (possibly unlicensed, out-of-band, mis-tuned, or simply not
tabulated). It does not prove identity: an unlisted emitter, an interferer, or
a signal distorted by multipath can match nothing while still being perfectly
legitimate. The report therefore exposes both the confidence and the
`VALID`/`INVALID` reason rather than a single name.

---

### A1 — Classic envelope analysis

**Description.** `classic_analyze()` (`src/classic.c`, SPTK-derived) runs, on a
decimated complex envelope (≈50 kHz): a real cepstrum of the log-magnitude
spectrum (peak quefrency → period), an AMDF period estimate, an envelope-ACF
period estimate, an LPC spectral envelope, a group-delay curve, and a Goertzel
single-bin probe. It reports each estimate and a consensus period.

**Rationale.** These are model-free, well-understood periodicity and spectral
estimators that do not share the pulse module's thresholding assumptions. The
cepstrum in particular separates a periodic excitation from the spectral
envelope that shapes it (so an echo or a keying period shows up as a quefrency
peak even when the spectrum itself is smeared), while the AMDF is a robust
time-domain period detector. Running several independent estimators and
reporting their consensus is a cheap way to avoid trusting a single method's
failure mode. The Goertzel probe answers "is there energy exactly at this
frequency" without a full FFT.

**What it can prove about a signal.** That the emission has a periodic
structure at the envelope time scale — a symbol, burst, chip, scan or echo
period — independently of the pulse segmenter; how linear the phase is
(group delay flat vs tilted, i.e. clean vs distorted); the shape of the
spectral envelope (LPC); and the presence of a tone at a probed frequency. The
consensus period is cross-checked in the report against the pulse-module `f0`
and the frame-level fluctuation rate, so agreement between three unrelated
estimators is strong evidence for a real keying/burst rate, while disagreement
is a warning that one of them has locked onto a harmonic or a noise artifact.

---

### A2 — Pulse-train analysis

**Description.** `pulse_analyze()` (`src/pulse.c`) follows the standard
pulse-descriptor workflow: envelope extraction (magnitude + smoothing),
hysteresis thresholding, edge detection, false-trigger rejection, pulse
grouping, and per-train metrics (width τ, period T, PRF, duty cycle, inter-pulse
interval, rise/fall times, peak/RMS amplitude, jitter). It also computes the
envelope line spectrum (fundamental `f0 = 1/T`, sinc envelope and the `1/τ`
nulls), intra-pulse chirp statistics, discrete interval/width levels
(PPM/PIM/PWM evidence), and a modulation-style verdict. A credibility gate
rejects a "train" built from threshold crossings of envelope noise (using duty,
envelope CV, period jitter and line-spectrum consistency).

**Rationale.** Pulse parameters are the most direct description of a radar,
beacon, telemetry burst or switching emission, and the envelope line spectrum
gives an independent estimate of the burst rate that does not depend on
edge detection. The credibility gate is essential: at low SNR a threshold on a
noisy envelope always produces "pulses", and reporting their PRF as if it were
measured would be a false positive. Chirp and interval-level analysis extend
the same segmentation to variable waveforms and interval coding.

**What it can prove about a signal.** Whether it is genuinely pulsed (credible
train) or merely noisy/continuous; the pulse width, period, PRF, duty cycle and
jitter; whether the PRF is stable (clock/radar-like) or jittered/random; whether
the pulses are chirped (and in which direction); whether widths or intervals
take discrete levels (PWM/PPM/PIM); and the burst/modulation rate `f0`. This is
the primary evidence for time-structure claims in the report, and it feeds the
DSSS/FHSS/THSS/CSS hypotheses downstream.

---

### A3 — Higher-order statistics

**Description.** `hos_analyze()` (`src/hos.c`) computes the second- through
sixth-order cumulants of the centralised, mixdowned selection following Swami &
Sadler — C20, C21, C30, C31, C40, C41, C42, C60, C63 — plus the normalised
magnitudes |C40|/C21², |C41|/C21², |C42|/C21², the complex kurtosis, and a
windowed bicoherence bispectrum over the principal domain. A Gaussianity test
compares |C42|/C21² with the asymptotic 5σ baseline `5·sqrt(24/N)`.

**Rationale.** Gaussian noise has identically zero cumulants beyond second
order (Mendel, Proc. IEEE 1991). Modulated signals — including spread-spectrum
signals — do not. Measuring fourth-order cumulants therefore suppresses the
noise and reveals a non-Gaussian signal at SNRs where the mean spectrum is
still buried. The normalised ratios are the classical constellation features,
so the same calculation yields both a detection statistic and a
constellation-style hint. The bispectrum adds a check for quadratic phase
coupling, which harmonics and some nonlinear emissions exhibit and Gaussian
noise does not.

**What it can prove about a signal.** That a non-Gaussian, structured signal is
present even when the spectrum looks noise-like (the key claim at low SNR);
a constellation-style hint (BPSK/QPSK/QAM-like symmetry classes) that
corroborates or contradicts the demodulator's guess; and, from the bispectrum,
whether there is phase coupling (harmonic generation, nonlinearity, some
modulation types). It does **not** prove a specific modulation on its own:
non-Gaussianity is necessary but not sufficient, which is why the Director
treats it as a vote rather than a verdict.

---

### A4 — FAM spectral correlation

**Description.** `cyclic_analyze_ex()` (`src/cyclic.c`) implements the FFT
Accumulation Method (Gardner): the selection is mixed to the centre, FIR-
decimated toward a 256 kHz effective rate (kept higher when the signal's own
bandwidth demands it), and split into up to 512 Hamming-windowed blocks of
1024 samples, hopped by 256 samples. The coherence-normalised spectral
correlation function (SCF) is estimated on a 1024×1024 grid. A cyclic-domain
profile (max coherence per cycle frequency α) is derived; odd-α rows use dual
half-bin averaging; the α ≈ 0 self-coherence
zone is masked for detection; peaks must clear both the statistical and the
median-relative thresholds; wall-hugging peaks require a corroborating
harmonic (which rejects a pure tone's leakage tail). Reported peaks are
sub-bin refined, sorted with the fundamental (symbol/baud/chip rate) first, and
the coherence-thresholded cycle-frequency list drives the A6 validation stage;
a non-empty list is also the precondition for A8, which takes the region centre
as its carrier.

**Rationale.** A modulated signal is cyclostationary: its statistics vary
periodically at the symbol, baud or chip rate, producing spectral correlation
at specific cycle frequencies and their harmonics. The SCF exposes those
frequencies even when the power spectrum is flat and uninformative — exactly
the spread-spectrum and digital-modulation case. Coherence normalisation makes
the peaks comparable across signal powers, the median-relative threshold keeps
the detector from firing on strong but unstructured signals, and the harmonic
corroboration rule prevents a pure carrier's spectral skirt from being
mistaken for a cycle frequency.

**What it can prove about a signal.** That the signal is cyclostationary (i.e.
modulated/structured, not stationary noise), and *at which rates*: the
fundamental symbol/baud/chip rate and its harmonics, sub-bin refined. This is
the strongest non-demodulating evidence for a digital modulation rate, and the
candidate list is what the cycle detectors later validate and the cyclic
cumulants classify. A pure tone produces no surviving peak after the
wall-harmonic rule, so the analysis also discriminates "carrier" from
"modulated carrier".

---

### A5 — SSCA (Strip Spectral Correlation Analyzer)

**Description.** `ssca_analyze()` (`src/ssca.c`, Spooner/Roberts) is the
exhaustive blind cycle-frequency search: the input is channelised by short FFTs
(N′ = 64 points, hopped by L = 16), each channeliser output is multiplied by the
original signal, and the products are transformed by long strip FFTs. Each
strip gives SCF point estimates along a line parallel to α = −2f; normalising by
the power spectrum yields the spectral coherence, which is CFAR-thresholded and
binned into a cycle-frequency list (α resolution set by the q/(P·L) mapping).

**Rationale.** FAM trades α resolution against cost and is tuned for narrowband
inputs; for wideband, OFDM-like or noise-like emissions the conventional
estimate can miss or smear the cycle frequencies. SSCA searches the cycle-
frequency axis over the channelised band with uniform sensitivity and is the
better tool when the baseline says "wideband" (high flatness + high PAPR,
QAM/OFDM classifier, or occupied bandwidth > 20 % of the sample rate). Running
it only in that branch, or when FAM finds no cycles, keeps the heavy
computation off the common path.

**What it can prove about a signal.** The cycle-frequency content of a
wideband/noise-like emission — i.e. that a spread or multicarrier signal is
cyclostationary and at what rates — including cases where FAM returned nothing.
It broadens coverage of the cyclic evidence rather than duplicating FAM: two
algorithms agreeing on a cycle frequency (the Director's cross-check) is much
stronger than either alone, and their disagreement is itself diagnostic.

---

### A6 — Cycle detectors (CFAR)

**Description.** `cycledet_analyze()` (`src/cycledet.c`, Spooner's cycle
detectors) computes the lagged cyclic autocorrelation `x(n)·x*(n−τ)` at seven
lags, removes the DC per frame, optionally whitens with `H_w = 1/sqrt(S(f))`,
and compares the normalised statistic against an empirically calibrated
Gaussian threshold (`100/sqrt(np·n)`). Individual values are combined into a
multi-cycle statistic (sum of the top-four squared statistics); the node reports
the candidates, the threshold, whether whitening was used, and a
detected/not-detected verdict with the best cycle frequency.

**Rationale.** An SCF peak can be a genuine cycle frequency or an artifact of
finite averaging, windowing or a strong neighbouring carrier. A CFAR test with a
calibrated Gaussian baseline answers the question the SCF cannot: *is this peak
significant given the amount of data?* Whitening extends that test to non-flat
noise, which is common in real recordings; the multi-cycle combination exploits
the fact that a true cyclostationary signal shows several mutually consistent
cycles, whereas noise artifacts do not.

**What it can prove about a signal.** Whether the candidate cycle frequencies
survive a false-alarm-controlled statistical test — i.e. that the signal is
cyclostationary *and* the specific cycle frequency is trustworthy — and which
one is strongest. It is a validation stage, not a discovery stage: it is only
run when A4/A5 produced candidates, so a negative result means "candidates not
confirmed", not "no cyclostationarity". The report uses this as the detector
validation line and as one of the four presence votes.

---

### A6 — CPAR detector (cumulative peak-to-average ratio)

**Description.** `cpar_analyze()` (`src/cpar.c`) implements the CPAR family of
blind DSSS detectors: the selection is FIR-decimated to ≈500 kHz and split into
16384-sample windows whose FFT-based autocorrelations are averaged and
normalised. Candidate peaks are local maxima above twice the median; each
peak's PAR is its power over the mean squared autocorrelation across the whole
lag range (a global noise baseline); the CPAR criterion sums the PARs of
significant peaks (PAR > 25). The peak lag is the code period for a repeating
spreading code, and the chip period is recovered from where the
chip-correlation triangle first crosses the noise floor.

**Rationale.** A direct-sequence signal multiplies the data by a fast code, so
its autocorrelation has a sharp, repeating chip/code-period structure even when
the spectrum is flat and the signal is below the noise in the mean spectrum.
CPAR measures how far each autocorrelation peak stands above the *global* lag
baseline, which is precisely what distinguishes a repeating code from noise
(where all lags are comparable, PAR ≈ 1). It needs no prior knowledge of the
code or chip rate and is robust at low SNR. CPAR is deliberately not used to
validate cycle frequencies: it is a chip-rate detector, and it can fire on
ordinary burst periodicity.

**What it can prove about a signal.** That a repeating spreading code is
present (DSSS-like structure), its code period (highest-PAR peak lag) and an
estimate of the chip rate. Agreement with the Burel fluctuation detector is the
strongest single piece of DSSS evidence in the suite; disagreement leaves the
chip rate unvalidated and is reported as such rather than being averaged away.

---

### A7 — Autocorrelation fluctuation (Burel)

**Description.** `fluct_analyze()` (`src/fluct.c`) implements Burel, Bouder &
Berder (ISPACS 2000): the selection is FIR-decimated to ≈100 kHz and split into
M windows of 512 samples; per window and lag the autocorrelation estimate is
computed, and per lag its variance `V(τ)` across windows is normalised to
`F(τ) = T·V(τ)/σ⁴`. Noise alone gives `F ≈ 1` (chi-square, mean 1, standard
deviation `1/sqrt(M)`); a hidden DSSS signal raises `F` at the chip-period lag
by the signal's fourth-order statistics. The detection threshold is
`max(1 + 4/sqrt(M), 2.0)`; the peak lag yields the chip period and a rough
chip-SNR.

**Rationale.** A DSSS signal can be far too weak for its *mean* autocorrelation
peak to emerge, but the *variance* of the autocorrelation estimate across short
windows responds to the code's fourth-order structure long before the mean
does. Measuring fluctuation instead of the peak is therefore a low-SNR
detection strategy, and it is statistically calibrated: under Gaussian noise
the normalised statistic has a known mean and spread, so the threshold has a
defensible false-alarm interpretation. It runs when the cycle detectors failed
or found nothing, i.e. exactly when a hidden spread signal would otherwise be
missed.

**What it can prove about a signal.** That periodic correlation structure is
present below the SNR at which the mean ACF peak appears — the "hidden DSSS"
claim — with a chip-rate estimate and a rough chip SNR, even when the spectrum
is flat and no cycle frequency survived. It is the strongest low-SNR evidence
for DSSS, but it only proves *chip-rate periodicity*: it does not identify the
modulation or the family, and strong non-stationarity (drifting power, hopping)
can raise the statistic without a spreading code.

---

### A8 — Cyclic cumulants (AMR)

**Description.** `ccc_analyze()` (`src/ccc.c`, Spooner's cyclic-cumulant
modulation recognition) computes the normalised magnitudes of the fourth-order
cyclic temporal cumulants |C40(4fc)|, |C41(2fc)| and |C42(2fc)| / P² from delay
products of the bandpass signal, without removing the carrier. The
constellation classes have well-separated ideal feature values — BPSK (2, 2, 2),
QPSK (1, 0, 1), 8PSK (0, 0, 1), 16QAM (≈0.68, 0, ≈0.68) — and the implemented
hierarchical verdict returns BPSK-like, QPSK-like, OOK/ASK-like burst emission
(C42 = 2P² with C40/C41 absent), circular PSK/QAM-like (C40 = C41 = 0, weak
C42), 16QAM-like, or "ambiguous" with the measured features. The carrier is
taken as the region centre.

**Rationale.** Once the cyclic stages have established that a modulated signal
is present and where its carrier is, the remaining question is *which*
constellation. Cyclic cumulants answer it from the phase structure of the
signal at multiples of the carrier, without needing symbol synchronisation or a
demodulator, and the classes above are the ones a fourth-order feature set
separates reliably. The OOK/ASK branch is the same statistic applied to a
one-dimensional amplitude modulation: C42 = 2P² with C40/C41 absent is the
signature of a real (one-dimensional) burst train rather than a circular
PSK/QAM constellation, so the analysis does not mislabel a pulsed emission as a
PSK signal.

**What it can prove about a signal.** The constellation family of a
single-carrier digital emission — BPSK, QPSK, circular PSK/QAM-like,
16QAM-like — or that the emission is an OOK/ASK burst rather than a phase-
modulated one. This is the strongest modulation-identification evidence in the
suite, and it feeds both the modulation consensus and the Director's vote. It
does not prove the bit rate, the coding, or the protocol; it needs adequate
SNR/samples to be valid; and because it assumes the signal is centred at the
region centre, a carrier offset or a multi-carrier signal degrades or
invalidates the verdict (the `valid` flag says when it is unusable).

---

### A9 — Intra-burst envelope

**Description.** `intraburst_analyze_sot()` (`src/intraburst.c`) inspects the
envelope *inside* individual bursts at the finest resolution the data allows.
It uses the prepared SOT slice (already mixed and band-filtered), applies a
residual-CFO de-rotation estimated from the cyclic-cumulant carrier, and builds
the envelope at the full sample rate without decimation or smoothing, so edges
stay as sharp as 1/fs. It detects bursts on a short RMS envelope, derives the
burst period from the envelope spectrum, folds the bursts coherently (phase-
coherent complex fold when possible, otherwise incoherent |x|), fits a
two-level on/off amplitude model, segments the folded average with a
multi-detector scheme (levels, hysteresis, change-point, HMM, multiscale edges;
Delta-BIC statistic, surrogate phase-randomised false-alarm rate), and then
characterises each sub-pulse (width, spacing, 10–90 % rise, duty, SNR, IF span,
FSK steps, bandwidth, matched-filter correlation). Uniform widths and spacings
indicate a clock/preamble/PFM-supply signature; varying widths indicate encoded
data; noise-like envelopes indicate no coherent intra-burst structure. A
per-burst classifier votes on PPM/PIM/PWM/PFM/Manchester-style hypotheses and
combines the votes with a likelihood/BIC criterion.

**Rationale.** The coarse analyses stop at the burst boundary; the question
"what is inside a burst" is where clock, preamble and coding structure live.
Re-reading at full rate (rather than reusing the decimated envelope) is
deliberate: decimation and smoothing blur exactly the edge structure this stage
is meant to measure. Folding many bursts recovers SNR for the sub-pulse
segmentation, and the surrogate false-alarm rate keeps the segmentation from
finding structure in noise. The stage runs only for a bursty/pulsed signature
and after A8 so the residual carrier estimate is available.

**What it can prove about a signal.** Whether the emission has coherent
intra-burst structure at all; whether that structure is a uniform clock/preamble
train or variable pulse-position/width coding; the sub-pulse timing (widths,
spacings, rise times, duty) at the sample-rate limit; the intra-pulse frequency
span and FSK step count; and a ranked intra-burst modulation hypothesis with
per-burst support. It is the analysis that distinguishes "a burst" from "a
burst containing a specific kind of keying", and it is the finest time-domain
evidence in the suite. It is gated on a bursty/pulsed signature, so its absence
is not evidence about a continuous signal.

---

### MODCONF — Modulation consensus

**Description.** `modconf_rank()` (`src/modconf.c`) scores every modulation
type the suite can detect on a 0–100 scale, assembling the evidence from the
demodulator's class, the analysis classifier's modulation string, the cyclic-
cumulant verdict, the HOS constellation hint, the pulse-train structure (PPM,
PIM, PWM, chirped, constant-PRF) and feature consistency (duty cycle, crest
factor, spectral flatness, FM deviation, FSK levels, chirp trend, DSSS detector
agreement). Duplicate names are merged and the list is sorted.

**Rationale.** No single stage is authoritative: the classifier can be fooled
by an unusual pulse shape, the cumulants need SNR, the HOS hint is only a
family, and the demodulator sees only what its own model expects. Scoring each
hypothesis from *all* the evidence, and showing the supporting reason with each
score, makes the disagreements explicit (two near-tied candidates mean the
evidence is ambiguous) instead of forcing an early binary decision.

**What it can prove about a signal.** Which modulation hypothesis is best
supported and by how much, with the specific reasons; that several independent
stages converge on (or diverge from) a given class; and, from the ranking
shape, whether the identification is confident or ambiguous. The score is a
weighted evidence tally, not a probability — it is designed for ranking and
explanation, not for calibration.

---

### SSCLASS — Spread-spectrum classification

**Description.** `spread_classify()` (`src/ssclass.c`) combines every analysis
into scored hypotheses for the spread-spectrum families: **DSSS** (a chip rate
from the Burel and/or CPAR detectors, agreeing or not, plus a flat spectrum and
cycle frequencies consistent with the chip rate), **FHSS** (bursts whose centre
frequency changes from burst to burst, low duty), **THSS** (irregular burst
intervals / no stable PRF, low duty), **CSS** (chirped pulses, fixed or variable
sweep rate, LoRa-like), the **FH-DSSS hybrid** (chip rate plus hopping), and
**Not spread spectrum**. Each hypothesis carries a 0–100 confidence and the
evidence string that produced it.

**Rationale.** "Spread spectrum present" is too coarse an answer for an analyst
who needs to know *which* family, and the families are distinguished by
different observables (chip rate vs hop rate vs irregular timing vs chirp). A
scored-hypothesis layer also encodes the detector-agreement policy: two chip-
rate detectors that agree score DSSS strongly, a single detector is only
suggestive, and two disagreeing detectors explicitly leave the chip rate
unvalidated. The explicit "Not spread spectrum" candidate prevents a weak
positive from being read as a detection.

**What it can prove about a signal.** Whether the emission is spread spectrum
at all, and if so which family — DSSS, FHSS, THSS, CSS or a hybrid — with a
confidence and the supporting evidence, including which chip-rate detector(s)
fired and whether they agreed. It is a relative ranking: the top score is the
best-supported hypothesis, and a low best score means no strong spread-spectrum
evidence. Like MODCONF, the scores are evidence tallies, not probabilities.

---

### Director — cross-analysis synthesis

**Description.** `director_build()` (`src/director.c`) consumes the per-analysis
envelopes plus the DAG status and produces the consolidated view the report
prints: a four-way modulation vote (demodulator class, cyclic-cumulant verdict,
HOS hint, pulse modulation), a cycle-frequency cross-check (FAM vs SSCA vs the
cycle detector, agreeing within 10 %), a chip-rate/DSSS consistency check
(Burel vs CPAR within 1.5×), a detector-presence count out of four (cycle
detector, HOS non-Gaussian, ≥5 pulses, a chip rate), the best-supported
characterisation and final verdict, and the full provenance/rationale trail
(which nodes ran, which were skipped and why, how the hypotheses were ranked).

**Rationale.** The individual analyses answer local questions; the value of the
suite comes from their agreement. Making the synthesis explicit and auditable —
votes, cross-checks and provenance rather than a single opaque verdict — is what
lets the reader see *why* a characterisation was chosen, which analyses
contradicted it, and which were skipped, so a confident-sounding sentence can
be checked against the evidence that produced it.

**What it can prove about a signal.** A single, cross-checked characterisation
that is stronger than any one analysis: modulation supported by multiple
independent stages, cycle frequencies confirmed by more than one estimator, a
chip rate supported by two detectors, and a documented account of what was and
was not measured. It also proves the *negative* claims honestly: a skipped
stage is listed with its reason, and a disagreement between detectors is
reported rather than averaged out.

---

## What the suite as a whole can prove

1. **Existence, location and strength.** Centre, occupied bandwidth (99 %
   containment), power, SNR and noise floor — the A0 analysis, metrics and SOT
   parameters.
2. **Time structure.** Whether the emission is continuous, pulsed or bursty;
   its pulse width, period, PRF, duty cycle and jitter; its burst rate and
   whether the intra-burst structure is a clock/preamble or encoded data — A1,
   A2 and A9.
3. **Spectral character.** Whether the spectrum is flat/spread or peaky/tone,
   symmetric or leaning, how much power is in the strongest bin, whether the IF
   drifts (chirp) — the metrics suite.
4. **Modulation.** A modulation class, a ranked set of candidates with
   confidences, a PSK/QAM constellation family from cyclic cumulants, and a
   consensus across the classifier, HOS, cumulants, pulse structure and feature
   consistency — the metrics classifier, A3, A8 and MODCONF.
5. **Cyclostationarity.** Whether the signal has periodic statistics, at which
   cycle frequencies (symbol/baud/chip rate and harmonics), whether the
   cycle-frequency estimate is statistically significant, and whether several
   independent estimators agree — A4, A5, A6 and the Director's cross-check.
6. **Spread spectrum.** Whether the emission is spread spectrum and which family
   (DSSS, FHSS, THSS, CSS, hybrid), with the chip rate, hop evidence and
   detector agreement — A6/CPAR, A7 and SSCLASS.
7. **Identity and anomaly.** Whether the emission is consistent with a
   documented system's band, channel width, modulation family and rate, and
   whether it is anomalous with respect to the database — the signal database
   match.
8. **Auditability.** For every claim, which analyses ran, which were skipped and
   why, how long they took, and how the hypotheses were ranked — the DAG
   provenance and Director rationale.

## What it cannot prove

* **A specific emitter, protocol or payload.** The suite identifies structure,
  not intent; a matching database entry is a consistency claim, not an
  identification, and no stage decodes bits.
* **That a skipped analysis would have been negative.** Gating is deliberate:
  A4/A5, A6, A7, A8 and A9 run only under their preconditions, so a missing
  result means "not warranted/not run", never "not present". The report states
  the skip reason for exactly this reason.
* **Calibrated probabilities.** The MODCONF and SSCLASS scores, and the
  Director's weights, are evidence tallies designed for ranking and
  explanation — not posterior probabilities.
* **Robustness to arbitrary signal conditions.** Overlapping emissions,
  multipath, strong non-stationarity, carrier offset and very low SNR each
  break specific assumptions (FAM resolution, cumulant validity, CPAR
  false alarms, intra-burst folding). The per-analysis caveats above name the
  assumption; the `valid`/`credible`/`detected` flags and the skip reasons are
  where the suite admits it.

## Source map

| Stage | Implementation | Report location |
|---|---|---|
| SOT | `src/sot.c` | "Common source: …" line |
| A0 analysis | `src/analysis.c` | §1 "Frequency, power and classifier" |
| A0 metrics | `src/metrics.c` | §1–§3 |
| A0 signal DB | `src/signals.c` | §5 "System identification" |
| A1 classic | `src/classic.c` | "Classic analysis", §2 |
| A2 pulse | `src/pulse.c` | "Pulse train", §2–§3 |
| A3 HOS | `src/hos.c` | "Higher-order statistics", §3 |
| A4 FAM | `src/cyclic.c` | "Spectral correlation — FAM", §4 |
| A5 SSCA | `src/ssca.c` | "Spectral correlation — SSCA", §4 |
| A6 cycle detectors | `src/cycledet.c` | "Cycle detectors (CFAR)", §4 |
| A6 CPAR | `src/cpar.c` | "CPAR", §4 |
| A7 Burel | `src/fluct.c` | "Fluctuations (Burel)", §4 |
| A8 cyclic cumulants | `src/ccc.c` | "Cyclic temporal cumulants", §3–§4 |
| A9 intra-burst | `src/intraburst.c` | "Intra-burst envelope", §2–§3 |
| MODCONF | `src/modconf.c` | §3 candidates |
| SSCLASS | `src/ssclass.c` | §4 spread-spectrum classification |
| Director | `src/director.c` | §4–§7, provenance |
