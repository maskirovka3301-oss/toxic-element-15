/*
 * apublish.c — result struct -> AnalysisResult envelope adapters.
 */
#include "apublish.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

void apub_set_confidence(AnalysisResult *r, double confidence)
{
    if (r == NULL) return;
    if (!isfinite(confidence)) confidence = 0.0;
    if (confidence < 0.0) confidence = 0.0;
    if (confidence > 1.0) confidence = 1.0;
    r->confidence = confidence;
}

void apub_analysis(AnalysisResult *r, const Analysis *a)
{
    if (r == NULL || a == NULL) return;
    ar_num(r, "center_hz", a->center_freq, "Hz", "region centre");
    ar_num(r, "bandwidth_hz", a->bandwidth, "Hz", "region bandwidth");
    ar_num(r, "avg_power_db", a->avg_power, "dB", "average power");
    ar_num(r, "max_power_db", a->max_power, "dB", "peak power");
    ar_num(r, "entropy_bits", a->entropy, "bits", "spectrogram entropy");
    ar_num(r, "prf_hz", a->prf, "Hz", "frame-level fluctuation");
    ar_text(r, "signal_type", a->signal_type, "classifier label");
    ar_text(r, "modulation", a->modulation, "classifier modulation");
}

void apub_metrics(AnalysisResult *r, const Metrics *m)
{
    if (r == NULL || m == NULL) return;
    ar_num(r, "occupied_bw_hz", m->occupied_bw_hz, "Hz", "99% containment");
    ar_num(r, "occupied_lo_hz", m->occupied_lo_hz, "Hz", "occupied lower edge");
    ar_num(r, "occupied_hi_hz", m->occupied_hi_hz, "Hz", "occupied upper edge");
    ar_num(r, "snr_db", m->snr_db, "dB", "spectral SNR");
    ar_num(r, "spectral_flatness", m->spectral_flatness, NULL, "Wiener entropy");
    ar_num(r, "band_symmetry", m->band_symmetry, NULL, "spectrum symmetry");
    ar_num(r, "papr_db", m->papr_db, "dB", "crest factor");
    ar_num(r, "tone_purity", m->tone_purity, NULL, "strongest-bin fraction");
    ar_num(r, "demodulable", m->demodulable ? 1.0 : 0.0, NULL, NULL);
    ar_text(r, "demod_mode", m->demod_mode, "selected demodulator");
    ar_text(r, "demod_note", m->demod_note, "classification reason");
    ar_text(r, "mod_class", m->mod_class, "modulation class");
    ar_num(r, "duty_cycle", m->duty_cycle, NULL, "on-time fraction");
    ar_num(r, "am_depth", m->am_depth, NULL, "AM depth");
    ar_num(r, "fm_deviation_hz", m->fm_deviation_hz, "Hz", "IF spread");
    ar_num(r, "fsk_levels", (double)m->fsk_levels, NULL, "FSK levels");
    ar_num(r, "sym_rate_hz", m->sym_rate_hz, "Hz", "envelope symbol rate");
    ar_num(r, "sym_rate_conf", m->sym_rate_conf, NULL, "rate confidence");
    ar_num(r, "chirp_trend", m->chirp_trend, NULL, "IF-vs-time trend");
}

void apub_classic(AnalysisResult *r, const ClassicResult *c)
{
    if (r == NULL || c == NULL) return;
    ar_num(r, "period_hz", c->period_hz, "Hz", "cepstrum/AMDF consensus");
    ar_text(r, "note", c->note, NULL);
}

void apub_pulse(AnalysisResult *r, const PulseResult *p)
{
    if (r == NULL || p == NULL) return;
    ar_num(r, "num_pulses", (double)p->num_pulses, NULL, "detected pulses");
    ar_num(r, "pulse_width_s", p->pulse_width_s, "s", "mean width");
    ar_num(r, "period_s", p->period_s, "s", "mean period");
    ar_num(r, "prf_hz", p->prf_hz, "Hz", "pulse repetition frequency");
    ar_num(r, "duty_cycle", p->duty_cycle, NULL, "pulse duty");
    ar_num(r, "jitter_s", p->jitter_s, "s", "timing jitter");
    ar_num(r, "f0_hz", p->f0_hz, "Hz", "envelope line-spectrum fundamental");
    ar_num(r, "env_cv", p->env_cv, NULL, "envelope CV");
    ar_num(r, "credible", p->credible ? 1.0 : 0.0, NULL, "credibility gate");
    ar_text(r, "modulation", p->modulation, "pulse modulation class");
    ar_text(r, "mod_detail", p->mod_detail, NULL);
    ar_text(r, "credibility_note", p->credibility_note, NULL);
    ar_num(r, "pulse_freq_cv", p->pulse_freq_cv, NULL, "burst centre-freq CV");
    ar_num(r, "chirp_pulses", (double)p->chirp_pulses, NULL, "chirped pulses");
    ar_num(r, "chirp_bw_mean_hz", 0.0, "Hz", "chirp sweep (see note)");
    ar_num(r, "chirp_rate_cv", 0.0, NULL, "chirp-rate variation");
    ar_num(r, "interval_cv", p->interval_cv, NULL, "interval variation");
    ar_num(r, "jitter_frac", p->jitter_frac, NULL, "jittered fraction");
    ar_num(r, "interval_levels", (double)p->interval_levels, NULL,
           "discrete interval levels");
    ar_num(r, "width_cv", p->width_cv, NULL, "pulse-width CV");
}

void apub_hos(AnalysisResult *r, const HosResult *h)
{
    if (r == NULL || h == NULL) return;
    ar_num(r, "non_gaussian", h->non_gaussian ? 1.0 : 0.0, NULL,
           "|C42|/C21^2 above the Gaussian baseline");
    ar_num(r, "n40", h->n40, NULL, "normalised |C40|");
    ar_num(r, "n41", h->n41, NULL, "normalised |C41|");
    ar_num(r, "n42", h->n42, NULL, "normalised |C42|");
    ar_num(r, "kurtosis", h->kurtosis, NULL, "excess kurtosis");
    ar_text(r, "hint", h->hint, "constellation-style hint");
    ar_text(r, "note", h->note, NULL);
}

void apub_cyclic(AnalysisResult *r, const CyclicResult *c)
{
    if (r == NULL || c == NULL) return;
    double alpha = 0.0;

    if (c->peak_count > 0) alpha = c->peaks[0].alpha;
    else if (c->cf_count > 0) alpha = c->cf_list[0][0];
    ar_num(r, "peak_count", (double)c->peak_count, NULL, "cycle-frequency peaks");
    ar_num(r, "strongest_alpha_hz", alpha, "Hz", "strongest cycle frequency");
    ar_num(r, "cf_count", (double)c->cf_count, NULL, "candidate cycle frequencies");
    ar_num(r, "fs_eff", c->fs_eff, "Hz", "effective rate after decimation");
}

void apub_ssca(AnalysisResult *r, const SscaResult *s)
{
    if (r == NULL || s == NULL) return;
    double cf = s->cf_count > 0 ? s->cf_list[0][0] : 0.0;

    ar_num(r, "cf_count", (double)s->cf_count, NULL, "cycle-frequency candidates");
    ar_num(r, "strongest_cf_hz", cf, "Hz", "strongest cycle frequency");
    ar_text(r, "note", s->note, NULL);
}

void apub_cycledet(AnalysisResult *r, const CycleDetResult *d)
{
    if (r == NULL || d == NULL) return;
    ar_num(r, "detected", d->detected ? 1.0 : 0.0, NULL, "CFAR verdict");
    ar_num(r, "best_cf_hz", d->best_cf, "Hz", "detected cycle frequency");
    ar_num(r, "num_cf", (double)d->num_cf, NULL, "candidate cycle frequencies");
    ar_num(r, "threshold", d->threshold, NULL, "5-sigma Gaussian baseline");
    ar_text(r, "note", d->note, NULL);
}

void apub_fluct(AnalysisResult *r, const FluctResult *f)
{
    if (r == NULL || f == NULL) return;
    ar_num(r, "detected", f->detected ? 1.0 : 0.0, NULL, "Burel criterion");
    ar_num(r, "chip_rate_hz", f->chip_rate_hz, "Hz", "1/peak lag");
    ar_num(r, "peak_lag_s", f->peak_lag, "s", "lag of the maximum F");
    ar_num(r, "snr_db", f->snr_db, "dB", "chip-SNR estimate");
    ar_num(r, "threshold", f->threshold, NULL, "F(tau) threshold");
    ar_text(r, "note", f->note, NULL);
}

void apub_cpar(AnalysisResult *r, const CparResult *c)
{
    if (r == NULL || c == NULL) return;
    ar_num(r, "detected", c->detected ? 1.0 : 0.0, NULL, "CPAR verdict");
    ar_num(r, "chip_rate_hz", c->chip_rate_hz, "Hz", "chip rate");
    ar_num(r, "n_peaks", (double)c->n_peaks, NULL, "candidate peaks");
    ar_text(r, "note", c->note, NULL);
}

void apub_ccc(AnalysisResult *r, const CccResult *c)
{
    if (r == NULL || c == NULL) return;
    ar_num(r, "valid", c->valid ? 1.0 : 0.0, NULL, NULL);
    ar_num(r, "carrier_hz", c->carrier_hz, "Hz", "carrier estimate");
    ar_num(r, "n40", c->n40, NULL, "|C40(4fc)|/P^2");
    ar_num(r, "n41", c->n41, NULL, "|C41(2fc)|/P^2");
    ar_num(r, "n42", c->n42, NULL, "|C42(2fc)|/P^2");
    ar_text(r, "verdict", c->verdict, "modulation verdict");
    ar_text(r, "note", c->note, NULL);
}

void apub_intraburst(AnalysisResult *r, const IntraBurstResult *i)
{
    if (r == NULL || i == NULL) return;
    ar_num(r, "valid", i->valid ? 1.0 : 0.0, NULL, NULL);
    ar_num(r, "bursts_detected", (double)i->bursts_detected, NULL, NULL);
    ar_num(r, "folded_bursts", (double)i->folded_bursts, NULL, NULL);
    ar_num(r, "period_hz", i->period_hz, "Hz", "envelope burst rate");
    ar_num(r, "coherence", i->coherence, NULL, "fold coherence");
    ar_num(r, "rms_window_s", i->rms_window_s, "s", "RMS window");
    ar_num(r, "sub_pulses", (double)i->sub_pulses, NULL, NULL);
    ar_num(r, "width_mean_s", i->width_mean_s, "s", "mean sub-pulse width");
    ar_num(r, "width_cv", i->width_cv, NULL, "width CV");
    ar_num(r, "period_mean_s", i->period_mean_s, "s", "mean spacing");
    ar_num(r, "period_cv", i->period_cv, NULL, "spacing CV");
    ar_num(r, "rise_mean_s", i->rise_mean_s, "s", "10-90% rise");
    ar_num(r, "duty", i->duty, NULL, "intra-burst duty");
    ar_num(r, "env_cv", i->env_cv, NULL, "envelope CV");
    ar_num(r, "snr_db", i->snr_db, "dB", "per-pulse SNR");
    ar_num(r, "on_level", i->on_level, NULL, "on level");
    ar_num(r, "off_level", i->off_level, NULL, "off level");
    ar_num(r, "delta_bic", (double)i->delta_bic, NULL, "two-level Delta-BIC");
    ar_num(r, "surrogate_fpr", i->surrogate_fpr, NULL, "surrogate FPR");
    ar_num(r, "detectors_confirming", (double)i->detectors_confirming, NULL,
           "confirming detectors");
    ar_num(r, "resolution_s", i->resolution_s, "s", "edge resolution");
    ar_num(r, "residual_cfo_hz", i->residual_cfo_hz, "Hz", "residual CFO");
    ar_num(r, "bandwidth_hz", i->bandwidth_hz, "Hz", "mean pulse bandwidth");
    ar_num(r, "if_span_hz", i->if_span_hz, "Hz", "IF span");
    ar_num(r, "fsk_steps", (double)i->fsk_steps, NULL, "FSK steps");
    ar_num(r, "matched_filter", i->matched_filter, NULL, "matched-filter response");
    ar_num(r, "uniform", i->uniform ? 1.0 : 0.0, NULL, "uniform train flag");
    ar_num(r, "width_levels", (double)i->width_levels, NULL, "width levels");
    ar_num(r, "interval_levels", (double)i->interval_levels, NULL,
           "interval levels");
    ar_text(r, "burst_mod", i->burst_mod, "modulation hypothesis");
    ar_text(r, "burst_mod_ev", i->burst_mod_ev, NULL);
    ar_text(r, "burst_mod_cand0", i->burst_mod_cand[0], "per-burst candidate");
    ar_num(r, "burst_mod_support0", i->burst_mod_support[0], NULL, NULL);
    ar_text(r, "burst_mod_cand1", i->burst_mod_cand[1], NULL);
    ar_num(r, "burst_mod_support1", i->burst_mod_support[1], NULL, NULL);
    ar_text(r, "burst_mod_combo", i->burst_mod_combo, NULL);
    ar_text(r, "burst_mod_choice", i->burst_mod_choice, NULL);
    ar_text(r, "verdict", i->verdict, "intra-burst verdict");
}

void apub_modconf(AnalysisResult *r, const ModCandidate *cand, size_t n)
{
    char key[32];

    if (r == NULL || cand == NULL) return;
    for (size_t i = 0; i < n && i < 5; i++) {
        snprintf(key, sizeof(key), "cand%zu_name", i + 1);
        ar_text(r, key, cand[i].name, cand[i].why);
        snprintf(key, sizeof(key), "cand%zu_score", i + 1);
        ar_num(r, key, (double)cand[i].score, NULL, NULL);
    }
}

void apub_ssclass(AnalysisResult *r, const SpreadCandidate *cand, size_t n)
{
    char key[32];

    if (r == NULL || cand == NULL) return;
    ar_text(r, "best_name", cand[0].name, cand[0].evidence);
    ar_num(r, "best_score", (double)cand[0].score, NULL, NULL);
    for (size_t i = 0; i < n && i < 3; i++) {
        snprintf(key, sizeof(key), "cand%zu_name", i + 1);
        ar_text(r, key, cand[i].name, cand[i].evidence);
        snprintf(key, sizeof(key), "cand%zu_score", i + 1);
        ar_num(r, key, (double)cand[i].score, NULL, NULL);
    }
}

void apub_signals(AnalysisResult *r, const RankedMatch *m, size_t n,
                  bool anomaly, const char *anomaly_note)
{
    char key[32];
    size_t valid = 0;

    if (r == NULL) return;
    for (size_t i = 0; i < n; i++) {
        if (m[i].valid) valid++;
        if (i < 3) {
            snprintf(key, sizeof(key), "rank%zu_name", i + 1);
            ar_text(r, key, m[i].signal != NULL ? m[i].signal->name : "?",
                    m[i].why);
            snprintf(key, sizeof(key), "rank%zu_score", i + 1);
            ar_num(r, key, (double)m[i].confidence_pct, NULL, NULL);
            snprintf(key, sizeof(key), "rank%zu_valid", i + 1);
            ar_num(r, key, m[i].valid ? 1.0 : 0.0, NULL, NULL);
        }
    }
    ar_num(r, "anomaly", anomaly ? 1.0 : 0.0, NULL, NULL);
    ar_text(r, "anomaly_note", anomaly_note, NULL);
    ar_num(r, "validity_n", (double)valid, NULL, NULL);
    ar_num(r, "validity_total", (double)n, NULL, NULL);
}

void apub_sot(AnalysisResult *r, double pass_hz, double stop_hz, double gain,
              double noise_floor_db, size_t samples, double rate_hz)
{
    if (r == NULL) return;
    ar_num(r, "pass_hz", pass_hz, "Hz", "context filter passband edge");
    ar_num(r, "stop_hz", stop_hz, "Hz", "context filter stopband edge");
    ar_num(r, "gain", gain, NULL, "uniform normalisation gain");
    ar_num(r, "noise_floor_db", noise_floor_db, "dB", "wideband floor");
    ar_num(r, "samples", (double)samples, NULL, "complex samples in the SoT");
    ar_num(r, "rate_hz", rate_hz, "Hz", "original sample rate");
}
