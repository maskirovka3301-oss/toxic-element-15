/*
 * aregistry.c — the static analysis registry (see aregistry.h).
 */
#include "aregistry.h"

#include <string.h>

size_t aregistry_count(void);
const ArAnalysisDesc *aregistry_at(size_t i);

static const ArParamDesc analysis_params[] = {
    { "center_hz", "Hz", "region centre reported by the analysis image" },
    { "bandwidth_hz", "Hz", "region bandwidth" },
    { "avg_power_db", "dB", "average spectrogram power" },
    { "max_power_db", "dB", "peak spectrogram power" },
    { "entropy_bits", "bits", "Shannon entropy of the spectrogram" },
    { "prf_hz", "Hz", "frame-level fluctuation rate" },
    { "signal_type", NULL, "classifier label" },
    { "modulation", NULL, "classifier modulation label" },
    { "mod_class", NULL, "metrics modulation class" },
};

static const ArParamDesc metrics_params[] = {
    { "occupied_bw_hz", "Hz", "99% power containment bandwidth" },
    { "snr_db", "dB", "spectral SNR" },
    { "spectral_flatness", NULL, "Wiener entropy (0 tone .. 1 noise)" },
    { "band_symmetry", NULL, "spectrum symmetry around the centre" },
    { "papr_db", "dB", "crest factor" },
    { "tone_purity", NULL, "fraction of power in the strongest bin" },
    { "demodulable", NULL, "1 when a demodulator applies" },
    { "demod_mode", NULL, "selected demodulator" },
    { "duty_cycle", NULL, "fraction of time the envelope is on" },
    { "am_depth", NULL, "AM modulation depth" },
    { "fm_deviation_hz", "Hz", "IF spread" },
    { "fsk_levels", NULL, "detected FSK levels" },
    { "sym_rate_hz", "Hz", "symbol/envelope rate estimate" },
    { "sym_rate_conf", NULL, "symbol-rate confidence 0..1" },
    { "chirp_trend", NULL, "IF-vs-time trend" },
};

static const ArParamDesc classic_params[] = {
    { "period_hz", "Hz", "cepstrum/AMDF period estimate" },
    { "note", NULL, "human-readable summary" },
};

static const ArParamDesc pulse_params[] = {
    { "num_pulses", NULL, "detected pulses" },
    { "pulse_width_s", "s", "mean pulse width" },
    { "period_s", "s", "mean pulse period" },
    { "prf_hz", "Hz", "pulse repetition frequency" },
    { "duty_cycle", NULL, "pulse duty cycle" },
    { "jitter_s", "s", "timing jitter" },
    { "f0_hz", "Hz", "envelope line-spectrum fundamental" },
    { "env_cv", NULL, "envelope coefficient of variation" },
    { "credible", NULL, "1 when the pulse train passed the credibility gate" },
    { "modulation", NULL, "pulse modulation class" },
    { "mod_detail", NULL, "pulse modulation detail" },
    { "credibility_note", NULL, "why the train was rejected" },
    { "pulse_freq_cv", NULL, "burst centre-frequency CV" },
    { "chirp_pulses", NULL, "number of chirped pulses" },
    { "chirp_bw_mean_hz", "Hz", "mean chirp sweep" },
    { "chirp_rate_cv", NULL, "chirp-rate variation" },
    { "interval_cv", NULL, "interval variation" },
    { "jitter_frac", NULL, "fraction of pulses with jitter" },
};

static const ArParamDesc hos_params[] = {
    { "non_gaussian", NULL, "1 when the HOS tests reject Gaussianity" },
    { "hint", NULL, "constellation-style hint" },
    { "note", NULL, "summary" },
};

static const ArParamDesc cyclic_params[] = {
    { "peak_count", NULL, "cycle-frequency peaks" },
    { "strongest_alpha_hz", "Hz", "strongest cycle frequency" },
    { "note", NULL, "summary" },
};

static const ArParamDesc ssca_params[] = {
    { "cf_count", NULL, "cycle-frequency candidates" },
    { "strongest_cf_hz", "Hz", "strongest cycle frequency" },
    { "note", NULL, "summary" },
};

static const ArParamDesc cycledet_params[] = {
    { "detected", NULL, "1 when cyclic features were detected" },
    { "best_cf_hz", "Hz", "detected cycle frequency" },
    { "note", NULL, "summary" },
};

static const ArParamDesc fluct_params[] = {
    { "detected", NULL, "1 when a chip rate was detected" },
    { "chip_rate_hz", "Hz", "chip rate (Burel criterion)" },
    { "snr_db", "dB", "detector SNR" },
    { "note", NULL, "summary" },
};

static const ArParamDesc cpar_params[] = {
    { "detected", NULL, "1 when a chip rate was detected" },
    { "chip_rate_hz", "Hz", "chip rate (CPAR)" },
    { "note", NULL, "summary" },
};

static const ArParamDesc ccc_params[] = {
    { "valid", NULL, "1 when the cumulants were usable" },
    { "carrier_hz", "Hz", "carrier estimate from the cumulants" },
    { "n40", NULL, "normalised |C40|" },
    { "n41", NULL, "normalised |C41|" },
    { "n42", NULL, "normalised |C42|" },
    { "verdict", NULL, "modulation verdict" },
    { "note", NULL, "summary" },
};

static const ArParamDesc intraburst_params[] = {
    { "valid", NULL, "1 when the burst analysis produced segments" },
    { "bursts_detected", NULL, "bursts found on the RMS envelope" },
    { "folded_bursts", NULL, "bursts aligned and folded" },
    { "period_hz", "Hz", "burst rate from the envelope spectrum" },
    { "coherence", NULL, "fold coherence 0..1" },
    { "rms_window_s", "s", "burst-detection RMS window" },
    { "sub_pulses", NULL, "segmented sub-pulses" },
    { "width_mean_s", "s", "mean sub-pulse width" },
    { "width_cv", NULL, "sub-pulse width CV" },
    { "period_mean_s", "s", "mean sub-pulse period" },
    { "period_cv", NULL, "sub-pulse period CV" },
    { "rise_mean_s", "s", "10-90% rise time" },
    { "duty", NULL, "intra-burst duty" },
    { "env_cv", NULL, "envelope CV" },
    { "snr_db", "dB", "per-pulse SNR" },
    { "on_level", NULL, "on level" },
    { "off_level", NULL, "off level" },
    { "delta_bic", NULL, "two-level Delta-BIC" },
    { "surrogate_fpr", NULL, "surrogate false-alarm rate" },
    { "detectors_confirming", NULL, "detectors confirming the segmentation" },
    { "resolution_s", "s", "edge resolution" },
    { "residual_cfo_hz", "Hz", "residual carrier removed" },
    { "burst_mod", NULL, "modulation hypothesis" },
    { "burst_mod_ev", NULL, "modulation evidence" },
    { "burst_mod_cand0", NULL, "top per-burst candidate" },
    { "burst_mod_support0", NULL, "support of candidate 0" },
    { "burst_mod_cand1", NULL, "second per-burst candidate" },
    { "burst_mod_support1", NULL, "support of candidate 1" },
    { "burst_mod_combo", NULL, "combination verdict" },
    { "burst_mod_choice", NULL, "likelihood-based choice" },
    { "verdict", NULL, "intra-burst verdict" },
};

static const ArParamDesc modconf_params[] = {
    { "cand1_name", NULL, "top modulation candidate" },
    { "cand1_score", NULL, "top candidate score" },
    { "cand2_name", NULL, "second candidate" },
    { "cand2_score", NULL, "second score" },
    { "cand3_name", NULL, "third candidate" },
    { "cand3_score", NULL, "third score" },
    { "cand4_name", NULL, "fourth candidate" },
    { "cand4_score", NULL, "fourth score" },
    { "cand5_name", NULL, "fifth candidate" },
    { "cand5_score", NULL, "fifth score" },
};

static const ArParamDesc ssclass_params[] = {
    { "best_name", NULL, "best spread-spectrum hypothesis" },
    { "best_score", NULL, "its confidence" },
    { "cand1_name", NULL, "top candidate" },
    { "cand1_score", NULL, "top score" },
    { "cand2_name", NULL, "second candidate" },
    { "cand2_score", NULL, "second score" },
    { "cand3_name", NULL, "third candidate" },
    { "cand3_score", NULL, "third score" },
};

static const ArSettingDesc pulse_settings[] = {
    { "jitter_frac_reject", AR_SET_NUMBER, 0.60, 0, false, NULL,
      "reject a train when more pulses jitter than this", 0.05, 2.0, 0.05 },
    { "env_cv_reject", AR_SET_NUMBER, 0.12, 0, false, NULL,
      "reject a train when the envelope CV is below this", 0.01, 1.0, 0.01 },
};

static const ArSettingDesc classic_settings[] = {
    { "envelope_rate_hz", AR_SET_NUMBER, 50000.0, 0, false, NULL,
      "effective envelope rate after decimation", 5000.0, 1.0e6, 1000.0 },
};

static const ArSettingDesc cyclic_settings[] = {
    { "decimation_target_hz", AR_SET_NUMBER, 256000.0, 0, false, NULL,
      "effective sample rate target", 16000.0, 5.0e6, 8000.0 },
};

static const ArSettingDesc ssca_settings[] = {
    { "cf_threshold", AR_SET_NUMBER, 0.030, 0, false, NULL,
      "cycle-frequency detection threshold (not yet wired)", 0.001, 0.5, 0.001 },
};

static const ArSettingDesc cycledet_settings[] = {
    { "fs_target_hz", AR_SET_NUMBER, 400000.0, 0, false, NULL,
      "effective sample rate target", 16000.0, 5.0e6, 8000.0 },
};

static const ArSettingDesc fluct_settings[] = {
    { "fs_target_hz", AR_SET_NUMBER, 100000.0, 0, false, NULL,
      "effective sample rate target (calibrated)", 8000.0, 2.0e6, 4000.0 },
};

static const ArSettingDesc cpar_settings[] = {
    { "fs_target_hz", AR_SET_NUMBER, 500000.0, 0, false, NULL,
      "effective sample rate target", 16000.0, 5.0e6, 8000.0 },
};

static const ArSettingDesc hos_settings[] = {
    { "fs_target_hz", AR_SET_NUMBER, 100000.0, 0, false, NULL,
      "effective sample rate target", 8000.0, 2.0e6, 4000.0 },
};

static const ArSettingDesc intraburst_settings[] = {
    { "rms_window_us", AR_SET_NUMBER, 10.0, 0, false, NULL,
      "burst-detection RMS window", 0.5, 200.0, 0.5 },
    { "max_bursts", AR_SET_INT, 24, 24, false, NULL,
      "maximum bursts folded", 1, 24, 1 },
};

static const ArSettingDesc modconf_settings[] = {
    { "max_candidates", AR_SET_INT, 12, 12, false, NULL,
      "candidates kept in the ranking", 1, 12, 1 },
};

static const ArSettingDesc ssclass_settings[] = {
    { "present_score", AR_SET_INT, 40, 40, false, NULL,
      "minimum score to call a spread-spectrum family present", 0, 100, 1 },
    { "append_to_verdict", AR_SET_BOOL, 0, 0, true, NULL,
      "append a confident spread-spectrum verdict to the characterisation",
      0, 0, 0 },
};

static const ArAnalysisDesc registry[] = {
    { "analysis", "Baseline analysis", DAG_A0, analysis_params,
      sizeof(analysis_params) / sizeof(analysis_params[0]), NULL, 0 },
    { "metrics", "Signal metrics", DAG_A0, metrics_params,
      sizeof(metrics_params) / sizeof(metrics_params[0]), NULL, 0 },
    { "classic", "Classic envelope analysis", DAG_A1, classic_params,
      sizeof(classic_params) / sizeof(classic_params[0]), classic_settings,
      sizeof(classic_settings) / sizeof(classic_settings[0]) },
    { "pulse", "Pulse-train analysis", DAG_A2, pulse_params,
      sizeof(pulse_params) / sizeof(pulse_params[0]), pulse_settings,
      sizeof(pulse_settings) / sizeof(pulse_settings[0]) },
    { "hos", "Higher-order statistics", DAG_A3, hos_params,
      sizeof(hos_params) / sizeof(hos_params[0]), hos_settings,
      sizeof(hos_settings) / sizeof(hos_settings[0]) },
    { "cyclic", "FAM spectral correlation", DAG_A4, cyclic_params,
      sizeof(cyclic_params) / sizeof(cyclic_params[0]), cyclic_settings,
      sizeof(cyclic_settings) / sizeof(cyclic_settings[0]) },
    { "ssca", "SSCA", DAG_A5, ssca_params,
      sizeof(ssca_params) / sizeof(ssca_params[0]), ssca_settings,
      sizeof(ssca_settings) / sizeof(ssca_settings[0]) },
    { "cycledet", "Cycle detectors (CFAR)", DAG_A6, cycledet_params,
      sizeof(cycledet_params) / sizeof(cycledet_params[0]), cycledet_settings,
      sizeof(cycledet_settings) / sizeof(cycledet_settings[0]) },
    { "cpar", "CPAR detector", DAG_A6, cpar_params,
      sizeof(cpar_params) / sizeof(cpar_params[0]), cpar_settings,
      sizeof(cpar_settings) / sizeof(cpar_settings[0]) },
    { "fluct", "Autocorrelation fluctuation", DAG_A7, fluct_params,
      sizeof(fluct_params) / sizeof(fluct_params[0]), fluct_settings,
      sizeof(fluct_settings) / sizeof(fluct_settings[0]) },
    { "ccc", "Cyclic cumulants (AMR)", DAG_A8, ccc_params,
      sizeof(ccc_params) / sizeof(ccc_params[0]), NULL, 0 },
    { "intraburst", "Intra-burst envelope", DAG_A9, intraburst_params,
      sizeof(intraburst_params) / sizeof(intraburst_params[0]),
      intraburst_settings,
      sizeof(intraburst_settings) / sizeof(intraburst_settings[0]) },
    { "modconf", "Modulation consensus", DAG_MODCONF, modconf_params,
      sizeof(modconf_params) / sizeof(modconf_params[0]), modconf_settings,
      sizeof(modconf_settings) / sizeof(modconf_settings[0]) },
    { "ssclass", "Spread-spectrum classification", DAG_SSCLASS, ssclass_params,
      sizeof(ssclass_params) / sizeof(ssclass_params[0]), ssclass_settings,
      sizeof(ssclass_settings) / sizeof(ssclass_settings[0]) },
    { "sot", "Common source of truth", DAG_SOT,
      (const ArParamDesc[]){ { "pass_hz", "Hz", "filter passband edge" },
                             { "stop_hz", "Hz", "filter stopband edge" },
                             { "gain", NULL, "uniform normalisation gain" },
                             { "noise_floor_db", "dB", "wideband floor" } },
      4, NULL, 0 },
};

size_t aregistry_count(void)
{
    return sizeof(registry) / sizeof(registry[0]);
}

const ArAnalysisDesc *aregistry_at(size_t i)
{
    if (i >= aregistry_count()) return NULL;
    return &registry[i];
}

const ArAnalysisDesc *aregistry_find(const char *id)
{
    if (id == NULL) return NULL;
    for (size_t i = 0; i < aregistry_count(); i++) {
        if (strcmp(registry[i].id, id) == 0) return &registry[i];
    }
    return NULL;
}

void aregistry_apply_settings(AnalysisResult *r)
{
    const ArAnalysisDesc *d;

    if (r == NULL) return;
    d = aregistry_find(r->id);
    if (d == NULL) return;
    for (size_t i = 0; i < d->n_settings; i++) {
        const ArSettingDesc *s = &d->settings[i];

        switch (s->kind) {
        case AR_SET_NUMBER: ar_setting_num(r, s->key, s->number); break;
        case AR_SET_INT:    ar_setting_int(r, s->key, s->integer); break;
        case AR_SET_BOOL:   ar_setting_bool(r, s->key, s->boolean); break;
        case AR_SET_TEXT:   ar_setting_text(r, s->key, s->text); break;
        }
    }
}
