/* ------------------------------------------------------------------ */
/* report.c — the Reporter                                             */
/*                                                                     */
/* Turns the consolidated analysis data (the envelopes, the Director's */
/* view and the raw results the sections print) into one document and  */
/* renders it as plain text, Markdown or HTML. The plain-text output   */
/* is byte-identical to the historical report; Markdown and HTML are   */
/* post-processed presentations of the same document.                  */
/* ------------------------------------------------------------------ */

#include "report.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* growable output buffer (no glib: unit-testable)                     */
/* ------------------------------------------------------------------ */

typedef struct {
    char  *s;
    size_t n, cap;
} RepBuf;

static bool rb_reserve(RepBuf *b, size_t extra)
{
    if (b->n + extra + 1 > b->cap) {
        size_t ncap = b->cap ? b->cap : 4096;
        char *ns;

        while (ncap < b->n + extra + 1) ncap *= 2;
        ns = realloc(b->s, ncap);
        if (ns == NULL) return false;
        b->s = ns;
        b->cap = ncap;
    }
    return true;
}

static void rb_puts(RepBuf *b, const char *s)
{
    size_t len;

    if (b == NULL || s == NULL) return;
    len = strlen(s);
    if (!rb_reserve(b, len)) return;
    memcpy(b->s + b->n, s, len);
    b->n += len;
    b->s[b->n] = '\0';
}

static void rb_putc(RepBuf *b, char ch)
{
    if (b == NULL) return;
    if (!rb_reserve(b, 1)) return;
    b->s[b->n++] = ch;
    b->s[b->n] = '\0';
}

static void rb_printf(RepBuf *b, const char *fmt, ...)
{
    va_list ap;
    int need;

    if (b == NULL || fmt == NULL) return;
    va_start(ap, fmt);
    need = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (need < 0) return;
    if (!rb_reserve(b, (size_t)need)) return;
    va_start(ap, fmt);
    vsnprintf(b->s + b->n, (size_t)need + 1, fmt, ap);
    va_end(ap);
    b->n += (size_t)need;
}

static char *rb_take(RepBuf *b)
{
    char *s = b->s;

    if (s == NULL) {
        s = malloc(1);
        if (s != NULL) s[0] = '\0';
    }
    b->s = NULL;
    b->n = b->cap = 0;
    return s;
}

/* ------------------------------------------------------------------ */
/* helpers used by the section formatters                              */
/* ------------------------------------------------------------------ */

static void report_rule(RepBuf *b, char ch, int len)
{
    for (int i = 0; i < len; i++) rb_putc(b, ch);
    rb_putc(b, '\n');
}

static void report_section(RepBuf *b, const char *title)
{
    size_t len = strlen(title);

    rb_printf(b, "\n%s\n", title);
    for (size_t i = 0; i < len; i++) rb_putc(b, '-');
    rb_putc(b, '\n');
}

static void hook_advance(ReportInput *in, const char *label)
{
    if (in->hooks.advance != NULL) in->hooks.advance(in->hooks.ctx, label);
}

static bool hook_cancelled(ReportInput *in)
{
    if (in->hooks.cancelled != NULL) return in->hooks.cancelled(in->hooks.ctx);
    return false;
}

const char *report_format_label(IqFormat f)
{
    switch (f) {
    case IQ_FORMAT_FLOAT32: return "Float32";
    case IQ_FORMAT_INT16:   return "Int16";
    case IQ_FORMAT_INT8:    return "Int8";
    case IQ_FORMAT_UINT8:   return "Uint8";
    }
    return "Unknown";
}

const char *report_window_name(DspWindow w)
{
    switch (w) {
    case DSP_WINDOW_HANN:        return "Hann";
    case DSP_WINDOW_HAMMING:     return "Hamming";
    case DSP_WINDOW_BLACKMAN:    return "Blackman";
    case DSP_WINDOW_RECTANGULAR: return "Rectangular";
    }
    return "Unknown";
}

/* ------------------------------------------------------------------ */
/* the document (plain text; Markdown/HTML are derived presentations)  */
/* ------------------------------------------------------------------ */

struct ReportDoc {
    char *text;
    bool  valid;
    bool  cancelled;   /* the build stopped early (cancellation) */
};

/* ------------------------------------------------------------------ */
/* build                                                               */
/* ------------------------------------------------------------------ */

static void report_header(ReportInput *in, RepBuf *b)
{
    const IqSource *src = in->src;
    const char *base = strrchr(src->path, '/');

    base = (base != NULL) ? base + 1 : src->path;
    rb_puts(b, "SELECTION REPORT\n");
    report_rule(b, '=', (int)strlen("SELECTION REPORT"));
    rb_printf(b, "Generated: %s\n", in->stamp != NULL ? in->stamp : "");
    rb_printf(b,
        "Recording: %s | %.3f s at %.3f MHz | center %.6f MHz | %s\n",
        base, src->duration, src->sample_rate / 1e6,
        src->center_freq / 1e6, report_format_label(src->format));
    rb_printf(b,
        "Selection: %.6f - %.6f MHz | %.3f - %.3f s | bandwidth %.2f kHz | "
        "duration %.3f s | samples %.0f\n",
        in->region.freq_min / 1e6, in->region.freq_max / 1e6,
        in->region.time_min, in->region.time_max,
        (in->region.freq_max - in->region.freq_min) / 1e3,
        in->region.time_max - in->region.time_min,
        (in->region.time_max - in->region.time_min) * src->sample_rate);
    rb_printf(b,
        "Settings: %.3f MHz, %s, center %.6f MHz, FFT %d, window %s\n",
        in->settings_rate / 1e6, report_format_label(in->settings_format),
        in->settings_center / 1e6, in->settings_fft_size,
        report_window_name(in->settings_window));
    if (!in->ok.a0) rb_puts(b, "\nFAILED: baseline analysis\n");
    if (in->sot_failed) {
        rb_puts(b, "\nFAILED: could not prepare the common analysis source\n");
    }
    if (in->cancelled) {
        rb_printf(b, "\nANALYSIS CANCELLED (PARTIAL REPORT)\n"
                     "-----------------------------------\n"
                     "%d of %d DAG stages completed; cancelled during stage "
                     "'%s' after %.1f s.\nThe results above are what finished "
                     "before the cancellation.\n",
                  in->stages_done, REPORT_TOTAL_MAX,
                  in->stage_name != NULL ? in->stage_name : "unknown",
                  in->elapsed_s);
    }
}

static void report_sot(ReportInput *in, RepBuf *b)
{
    const Sot *sot_p = in->sot;

    if (sot_p != NULL && sot_p->valid) {
        Sot sot = *sot_p;

        rb_printf(b,
            "Common source: full-band slice of the selection (%.3f s, %zu "
            "complex samples at %.3f MHz, no decimation), normalised with a "
            "uniform real gain x%.4f (%+.2f dB, raw RMS %.4f -> target "
            "%.2f: phase and frequency preserved, divide by the gain to "
            "recover the original amplitudes), mixed at %.6f MHz",
            (double)sot.n / sot.view.sample_rate, sot.n,
            sot.view.sample_rate / 1e6, sot.gain, sot.gain_db, sot.rms_in,
            0.1, sot.view.center_freq / 1e6);
        if (sot.pass_hz > 0.0) {
            rb_printf(b,
                ", band filter +/-%.1f kHz pass / +/-%.1f kHz stop (%.1f kHz "
                "transition, zero-phase mask, wideband floor %.1f dB)",
                sot.pass_hz / 1e3, sot.stop_hz / 1e3, sot.trans_hz / 1e3,
                sot.noise_floor_db);
        }
        rb_puts(b, "\n");
    }
}

static void report_body(ReportInput *in, RepBuf *b, ReportDoc *doc)
{
    /* aliases: the sections below are the historical report code, moved
     * verbatim out of the UI worker so the text stays identical */

    const Analysis *analysis_p = in->analysis;
    const Metrics *metrics_p = in->metrics;
    const ClassicResult *classic_p = in->classic;
    const PulseResult *pulse_p = in->pulse;
    const HosResult *hos_p = in->hos;
    const CyclicResult *cyclic_p = in->cyclic;
    const SscaResult *ssca_p = in->ssca;
    const CycleDetResult *cycledet_p = in->cycledet;
    const CparResult *cpar_p = in->cpar;
    const FluctResult *fluct_p = in->fluct;
    const CccResult *ccc_p = in->ccc;
    const IntraBurstResult *intraburst_p = in->intraburst;
    Analysis analysis = analysis_p != NULL ? *analysis_p : (Analysis){0};
    Metrics metrics = metrics_p != NULL ? *metrics_p : (Metrics){0};
    ClassicResult classic = classic_p != NULL ? *classic_p : (ClassicResult){0};
    PulseResult pulse = pulse_p != NULL ? *pulse_p : (PulseResult){0};
    HosResult hos = hos_p != NULL ? *hos_p : (HosResult){0};
    CyclicResult cyclic = cyclic_p != NULL ? *cyclic_p : (CyclicResult){0};
    SscaResult ssca = ssca_p != NULL ? *ssca_p : (SscaResult){0};
    CycleDetResult cycledet = cycledet_p != NULL ? *cycledet_p : (CycleDetResult){0};
    CparResult cpar = cpar_p != NULL ? *cpar_p : (CparResult){0};
    FluctResult fluct = fluct_p != NULL ? *fluct_p : (FluctResult){0};
    CccResult ccc = ccc_p != NULL ? *ccc_p : (CccResult){0};
    IntraBurstResult intraburst = intraburst_p != NULL ? *intraburst_p
                                                      : (IntraBurstResult){0};
    bool ran_a1 = in->ran.a1, ran_a2 = in->ran.a2, ran_a3 = in->ran.a3;
    bool ran_a4 = in->ran.a4, ran_a5 = in->ran.a5, ran_a6 = in->ran.a6;
    bool ran_a7 = in->ran.a7, ran_a8 = in->ran.a8, ran_a9 = in->ran.a9;
    bool ok_classic = in->ok.a1, ok_pulse = in->ok.a2, ok_hos = in->ok.a3;
    bool ok_fam = in->ok.a4, ok_ssca = in->ok.a5, ok_cycledet = in->ok.a6a;
    bool ok_cpar = in->ok.a6b, ok_fluct = in->ok.a7, ok_ccc = in->ok.a8;
    bool ok_intraburst = in->ok.a9;
    const char *skip_a4 = in->skip_a4 != NULL ? in->skip_a4 : "";
    const char *skip_a5 = in->skip_a5 != NULL ? in->skip_a5 : "";
    const char *skip_a6 = in->skip_a6 != NULL ? in->skip_a6 : "";
    const char *skip_a7 = in->skip_a7 != NULL ? in->skip_a7 : "";
    const char *skip_a8 = in->skip_a8 != NULL ? in->skip_a8 : "";
    const char *skip_a9 = in->skip_a9 != NULL ? in->skip_a9 : "";
    const SpreadCandidate *spread = in->spread;
    size_t spread_n = in->n_spread;
    const IqSource *src = in->src;

    /* 2. FAM (A4) */
    if (!ran_a4) {
        report_section(b, "SPECTRAL CORRELATION - FAM (SKIPPED)");
        rb_printf(b, "Not executed: %s\n", skip_a4);
    } else if (!ok_fam) {
        rb_puts(b, "\nFAILED: Spectral Correlation - FAM\n");
    } else {
        report_section(b, "SPECTRAL CORRELATION - FAM");
        rb_printf(b,
            "Grid: %d x %d | alpha resolution: %.2f Hz | effective rate: %.1f Hz\n",
            cyclic.n, cyclic.n, cyclic.alpha_res, cyclic.fs_eff);
        rb_printf(b, "Peaks: %d\n", cyclic.peak_count);
        for (int i = 0; i < cyclic.peak_count; i++) {
            rb_printf(b, "%d. %+.1f Hz (strength %.3f)\n",
                                   i + 1, cyclic.peaks[i].alpha,
                                   cyclic.peaks[i].strength);
        }
        rb_printf(b,
            "Coherence threshold: %.3f (P_FA threshold: %.3f)\n",
            cyclic.cf_threshold, cyclic.cf_pfa_threshold);
        rb_printf(b, "Significant cycle frequencies: %zu\n",
                               cyclic.cf_count);
        for (size_t i = 0; i < cyclic.cf_count; i++) {
            rb_printf(b, "%zu. %+.1f Hz (coherence %.3f)\n",
                                   i + 1, cyclic.cf_list[i][0],
                                   cyclic.cf_list[i][1]);
        }
    }
    hook_advance(in, "report: FAM section");
    if (hook_cancelled(in)) {
        doc->cancelled = true;
        goto done;
    }

    /* 3. SSCA */
    if (!ran_a5) {
        report_section(b, "SPECTRAL CORRELATION - SSCA (SKIPPED)");
        rb_printf(b, "Not executed: %s\n", skip_a5);
    } else if (!ok_ssca) {
        rb_puts(b, "\nFAILED: Spectral Correlation - SSCA\n");
    } else {
        report_section(b, "SPECTRAL CORRELATION - SSCA");
        rb_printf(b,
            "Channel FFT: %d | Strip FFT: %d | Strips: %d | CF bins: %d\n",
            ssca.channel_fft, ssca.strip_fft, ssca.num_strips, ssca.num_cf);
        rb_printf(b,
            "Threshold: %.3f | Significant cycle frequencies: %zu\n",
            ssca.cf_threshold, ssca.cf_count);
        for (size_t i = 0; i < ssca.cf_count; i++) {
            rb_printf(b, "%zu. %+.1f Hz (coherence %.3f)\n",
                                   i + 1, ssca.cf_list[i][0],
                                   ssca.cf_list[i][1]);
        }
        rb_printf(b, "Note: %s\n", ssca.note);
    }
    hook_advance(in, "report: SSCA section");
    if (hook_cancelled(in)) {
        doc->cancelled = true;
        goto done;
    }

    /* 4. Cycle detector */
    if (!ran_a6) {
        report_section(b, "CYCLE DETECTORS (SKIPPED)");
        rb_printf(b, "Not executed: %s\n", skip_a6);
    } else if (!ok_cycledet) {
        rb_puts(b, "\nFAILED: Cycle Detectors (CFAR)\n");
    } else {
        report_section(b, "CYCLE DETECTORS (CFAR)");
        rb_printf(b,
            "Detected: %s | Best CF: %+.1f Hz (stat %.3f)\n",
            cycledet.detected ? "yes" : "no",
            cycledet.best_cf, cycledet.best_stat);
        rb_printf(b,
            "Candidates: %d | Threshold: %.3f | Multi-cycle: %.3f vs %.3f | Whitened: %s\n",
            cycledet.num_cf, cycledet.threshold, cycledet.multi_cycle,
            cycledet.multi_threshold, cycledet.whitened ? "on" : "off");
        rb_printf(b, "Note: %s\n", cycledet.note);
    }
    hook_advance(in, "report: cycle-detector section");
    if (hook_cancelled(in)) {
        doc->cancelled = true;
        goto done;
    }

    /* 5. Burel */
    if (!ran_a7) {
        report_section(b, "AUTOCORRELATION FLUCTUATION (SKIPPED)");
        rb_printf(b, "Not executed: %s\n", skip_a7);
    } else if (!ok_fluct) {
        rb_puts(b, "\nFAILED: Fluctuations (Burel)\n");
    } else {
        report_section(b, "FLUCTUATIONS (BUREL)");
        rb_printf(b,
            "Detected: %s\n"
            "Lags: %d | Peak lag: %.6f s | Peak F: %.3f | Peak ACF: %.3f\n"
            "Chip rate: %.1f Hz | SNR: %.1f dB | Threshold: %.3f | Noise power: %.3e\n",
            fluct.detected ? "yes" : "no",
            fluct.n_lags, fluct.peak_lag, fluct.peak_fluct, fluct.peak_acf,
            fluct.chip_rate_hz, fluct.snr_db, fluct.threshold,
            fluct.noise_power);
        rb_printf(b, "Note: %s\n", fluct.note);
    }
    hook_advance(in, "report: fluctuation section");
    if (hook_cancelled(in)) {
        doc->cancelled = true;
        goto done;
    }

    /* 6. CPAR */
    if (!ran_a6) {
        report_section(b, "CPAR DETECTOR (SKIPPED)");
        rb_printf(b, "Not executed: %s\n", skip_a6);
    } else if (!ok_cpar) {
        rb_puts(b, "\nFAILED: CPAR\n");
    } else {
        report_section(b, "CPAR");
        rb_printf(b,
            "Detected: %s\n"
            "Lags: %d | CPAR: %.1f | CPAR (all): %.1f | Threshold: %.1f\n"
            "Code period: %.4f ms | Chip rate: %.1f Hz\n",
            cpar.detected ? "yes" : "no",
            cpar.n_lags, cpar.cpar, cpar.cpar_all, cpar.par_threshold,
            cpar.chip_period_s * 1e3, cpar.chip_rate_hz);
        rb_printf(b, "Note: %s\n", cpar.note);
    }
    hook_advance(in, "report: CPAR section");
    if (hook_cancelled(in)) {
        doc->cancelled = true;
        goto done;
    }

    /* 7. HOS */
    /* A9. Intra-burst envelope inspection */
    if (!ran_a9) {
        report_section(b, "INTRA-BURST ENVELOPE (SKIPPED)");
        if (skip_a9[0] != '\0') {
            rb_printf(b, "Not executed: %s\n", skip_a9);
        }
    } else if (!ok_intraburst) {
        report_section(b, "INTRA-BURST ENVELOPE");
        rb_printf(b, "Inspection failed: %s\n",
                               intraburst.note);
    } else {
        report_section(b, "INTRA-BURST ENVELOPE");
        rb_printf(b,
            "Synchronisation: %d bursts detected on a %.1f us RMS envelope; "
            "burst period %.2f Hz from the envelope spectrum; %d bursts "
            "aligned and folded (coherence %.2f -> %s average)\n",
            intraburst.bursts_detected, intraburst.rms_window_s * 1e6,
            intraburst.period_hz, intraburst.folded_bursts,
            intraburst.coherence,
            intraburst.fold_coherent ? "phase-coherent complex" : "incoherent |x|");
        rb_printf(b,
            "Folded burst %.3f ms | amplitude model: on %.3f / off %.3f | "
            "segmentation: %d sub-pulses, %d/4 detectors confirming, "
            "Delta-BIC %d, surrogate false-alarm %.0f%%\n",
            intraburst.burst_len_s * 1e3, intraburst.on_level,
            intraburst.off_level, intraburst.sub_pulses,
            intraburst.detectors_confirming, intraburst.delta_bic,
            intraburst.surrogate_fpr * 100.0);
        rb_printf(b,
            "Sub-pulse: width %.2f us (CV %.0f%%) | spacing %.2f us "
            "(CV %.0f%%) | rise %.2f us | duty %.0f%% | SNR %.1f dB | "
            "matched-filter %.2f\n",
            intraburst.width_mean_s * 1e6, intraburst.width_cv * 100.0,
            intraburst.period_mean_s * 1e6, intraburst.period_cv * 100.0,
            intraburst.rise_mean_s * 1e6, intraburst.duty * 100.0,
            intraburst.snr_db, intraburst.matched_filter);
        if (intraburst.if_span_hz > 0.0) {
            rb_printf(b,
                "Intra-pulse frequency span %.2f kHz | FSK steps %d\n",
                intraburst.if_span_hz / 1e3, intraburst.fsk_steps);
        }
        if (intraburst.burst_mod[0] != '\0') {
            rb_printf(b,
                "Modulation hypotheses: %s%s%s\n", intraburst.burst_mod,
                intraburst.burst_mod_ev[0] != '\0' ? " - " : "",
                intraburst.burst_mod_ev);
        }
        if (intraburst.bursts_classified > 0) {
            rb_printf(b,
                "Per-burst classification (%d bursts isolated): %s %.0f%%",
                intraburst.bursts_classified,
                intraburst.burst_mod_cand[0],
                intraburst.burst_mod_support[0] * 100.0);
            if (intraburst.burst_mod_cand[1][0] != '\0') {
                rb_printf(b, ", %s %.0f%%",
                                       intraburst.burst_mod_cand[1],
                                       intraburst.burst_mod_support[1] *
                                           100.0);
            }
            if (intraburst.burst_mod_combo[0] != '\0') {
                rb_printf(b, " -> %s",
                                       intraburst.burst_mod_combo);
            }
            rb_puts(b, "\n");
            if (intraburst.burst_mod_choice[0] != '\0') {
                rb_printf(b, "Candidate fit: %s\n",
                                       intraburst.burst_mod_choice);
            }
        }
        rb_printf(b, "Verdict: %s\n", intraburst.verdict);
    }
    hook_advance(in, "report: HOS section");
    if (hook_cancelled(in)) {
        doc->cancelled = true;
        goto done;
    }

    if (!ran_a3) {
        report_section(b, "HIGHER-ORDER STATISTICS (SKIPPED)");
        rb_puts(b, "Not executed: not selected by the DAG\n");
    } else if (!ok_hos) {
        rb_puts(b, "\nFAILED: Higher-Order Statistics\n");
    } else {
        report_section(b, "HIGHER-ORDER STATISTICS");
        rb_printf(b,
            "C20: %+.6f %+.6fj | C21: %.6f\n"
            "C30: %+.6f %+.6fj | C31: %+.6f %+.6fj\n"
            "C40: %+.6f %+.6fj | C41: %+.6f %+.6fj | C42: %.6f\n"
            "C60: %+.6f %+.6fj | C63: %.6f\n",
            hos.c20_re, hos.c20_im, hos.c21,
            hos.c30_re, hos.c30_im, hos.c31_re, hos.c31_im,
            hos.c40_re, hos.c40_im, hos.c41_re, hos.c41_im, hos.c42,
            hos.c60_re, hos.c60_im, hos.c63);
        rb_printf(b,
            "Normalised: |C40|/C21^2 %.4f | |C41|/C21^2 %.4f | |C42|/C21^2 %.4f | Kurtosis %.3f\n",
            hos.n40, hos.n41, hos.n42, hos.kurtosis);
        rb_printf(b,
            "Gaussian threshold: %.4f (N = %zu) | Non-Gaussian: %s\n"
            "Hint: %s\nNote: %s\n",
            hos.n42_threshold, hos.samples,
            hos.non_gaussian ? "yes" : "no", hos.hint, hos.note);
    }
    hook_advance(in, "report: cyclic-cumulant section");
    if (hook_cancelled(in)) {
        doc->cancelled = true;
        goto done;
    }

    /* 8. Cyclic cumulants */
    if (!ran_a8) {
        report_section(b, "CYCLIC CUMULANTS / AMR (SKIPPED)");
        rb_printf(b, "Not executed: %s\n", skip_a8);
    } else if (!ok_ccc) {
        rb_puts(b, "\nFAILED: Cyclic Temporal Cumulants\n");
    } else {
        report_section(b, "CYCLIC TEMPORAL CUMULANTS");
        rb_printf(b,
            "|C40(4fc)|/P^2: %.4f | |C41(2fc)|/P^2: %.4f | |C42(2fc)|/P^2: %.4f\n"
            "Power: %.3e | Carrier: %.1f Hz | Samples: %zu\n",
            ccc.n40, ccc.n41, ccc.n42, ccc.power, ccc.carrier_hz, ccc.samples);
        rb_printf(b, "Verdict: %s\nNote: %s\n",
                               ccc.verdict, ccc.note);
    }
    hook_advance(in, "report: classic section");
    if (hook_cancelled(in)) {
        doc->cancelled = true;
        goto done;
    }

    /* 9. Classic */
    if (!ran_a1) {
        report_section(b, "CLASSIC ANALYSIS (SKIPPED)");
        rb_puts(b, "Not executed: not selected by the DAG\n");
    } else if (!ok_classic) {
        rb_puts(b, "\nFAILED: Classic Analysis\n");
    } else {
        report_section(b, "CLASSIC ANALYSIS");
        rb_printf(b, "Period: %.3f Hz (consensus)\n",
                               classic.period_hz);
        rb_printf(b,
            "Cepstrum: peak quefrency %.3f samples (mag %.3f) -> %.3f Hz\n",
            classic.cep_peak_quef, classic.cep_peak_mag,
            classic.cep_period_hz);
        rb_printf(b,
            "AMDF: period %.3f samples -> %.3f Hz\n",
            classic.amdf_period, classic.amdf_period_hz);
        rb_printf(b,
            "LPC order: %d | Goertzel probe: %.0f Hz, power %.3f\n",
            classic.lpc_order, classic.probe_hz, classic.goertzel_power);
        rb_printf(b, "Note: %s\n", classic.note);
    }
    hook_advance(in, "report: pulse section");
    if (hook_cancelled(in)) {
        doc->cancelled = true;
        goto done;
    }

    /* 10. Pulse train */
    if (!ran_a2) {
        report_section(b, "PULSE TRAIN ANALYSIS (SKIPPED)");
        rb_puts(b, "Not executed: not selected by the DAG\n");
    } else if (!ok_pulse) {
        rb_puts(b, "\nFAILED: Pulse Train\n");
    } else {
        report_section(b, "PULSE TRAIN");
        if (pulse.num_pulses <= 0) {
            rb_puts(b, "No pulse train detected.\n");
        } else {
            rb_printf(b,
                "Pulses: %d | tau: %.3f ms | T: %.3f ms | PRF: %.1f Hz | Duty: %.0f %%\n",
                pulse.num_pulses, pulse.pulse_width_s * 1e3,
                pulse.period_s * 1e3, pulse.prf_hz,
                pulse.duty_cycle * 100.0);
            rb_printf(b,
                "IPI: %.3f ms | Rise: %.1f us | Fall: %.1f us | Jitter: %.1f us | IPI std: %.1f us\n",
                pulse.ipi_s * 1e3, pulse.rise_s * 1e6, pulse.fall_s * 1e6,
                pulse.jitter_s * 1e6, pulse.ipi_std_s * 1e6);
            rb_printf(b,
                "Peak amplitude: %.3f | RMS: %.3f\n",
                pulse.von_peak, pulse.arms);
            rb_printf(b,
                "f0: %.1f Hz | Sinc null: %.1f Hz | Tau (sinc): %.3f ms\n",
                pulse.f0_hz, pulse.sinc_null_hz, pulse.tau_from_sinc_s * 1e3);
            rb_printf(b, "Modulation: %s\n", pulse.modulation);
        }
        rb_printf(b, "Note: %s\n", pulse.note);
    }
    hook_advance(in, "report: intra-burst section");
    if (hook_cancelled(in)) {
        doc->cancelled = true;
        goto done;
    }

    /* --- DIRECTOR: consolidate the analysis envelopes -----------------
     * The Director ranks the hypotheses produced by every analysis, records
     * which DAG nodes were skipped/failed, and emits the best-supported
     * characterisation + verdict. The formatting below only prints its
     * consolidated view. */
    {
        DirectorInput dir_in;
        Director dir;
        const char *base = src->path != NULL ? strrchr(src->path, '/') : NULL;

        memset(&dir_in, 0, sizeof(dir_in));
        dir_in.file = base != NULL ? base + 1 : (src->path != NULL
                                                      ? src->path : "");
        dir_in.center_hz = src->center_freq;
        dir_in.rate_hz = src->sample_rate;
        dir_in.duration = src->duration;
        dir_in.region = in->region;
        dir_in.fft_size = in->settings_fft_size;
        dir_in.format = report_format_label(in->settings_format);
        dir_in.window_name = report_window_name(in->settings_window);
        dir_in.analyses = in->envelopes;
        dir_in.dag = in->dag;
        dir_in.mods = in->mods;
        dir_in.n_mods = in->n_mods;
        dir_in.spread = spread;
        dir_in.n_spread = spread_n;
        director_build(&dir, &dir_in);

        /* --- derived facts shared by several paragraphs --- */
        char vote_line[220];
        char majority[32];
        int maj;
        const char *cf_verdict;
        double fam_cf = 0.0, ssca_cf = 0.0, det_cf = 0.0;
        double bchip;
        double cchip;
        const char *dsss_label;
        int pres_k;
        char best[512];

        snprintf(vote_line, sizeof(vote_line), "%s", dir.vote_line);
        snprintf(majority, sizeof(majority), "%s", dir.majority);
        maj = dir.majority_count;
        cf_verdict = dir.cf_summary;
        bchip = dir.bchip;
        cchip = dir.cchip;
        dsss_label = dir.dsss_label;
        pres_k = dir.presence;
        snprintf(best, sizeof(best), "%s", dir.characterisation);

        /* cycle frequencies for the cross-check line (formatting inputs) */
        if (ran_a4 && ok_fam && cyclic.peak_count > 0) {
            fam_cf = cyclic.peaks[0].alpha;
        } else if (ran_a4 && ok_fam && cyclic.cf_count > 0) {
            fam_cf = cyclic.cf_list[0][0];
        }
        ssca_cf = (ran_a5 && ok_ssca && ssca.cf_count > 0) ? ssca.cf_list[0][0]
                                                            : 0.0;
        det_cf = (ran_a6 && ok_cycledet && cycledet.detected)
                     ? cycledet.best_cf : 0.0;

        report_section(b, "SIGNAL CHARACTERISATION (SYNTHESIS)");
        rb_printf(b,
            "Selection %.3f s at %.3f MHz | analysis image %d x %d "
            "bins/frames (bin %.2f Hz, dt %.6f s)\n\n",
            analysis.duration, src->sample_rate / 1e6, analysis.bins,
            analysis.frames, analysis.bin_hz, analysis.dt);

        /* 0. candidate-first identification flow */
        rb_puts(b, "0. CANDIDATE-FIRST IDENTIFICATION FLOW [database "
                             "pre-selection -> modulation -> candidate "
                             "filter -> DAG decision]\n");
        rb_printf(b,
            "   Pre-selection: %zu candidate%s %s possible from the "
            "measured centre and approximate bandwidth alone "
            "(no analysis, no modulation evidence)\n",
            in->n_preselect, in->n_preselect == 1 ? "" : "s",
            in->n_preselect == 1 ? "is" : "are");
        for (size_t i = 0; i < in->n_preselect && i < 6; i++) {
            rb_printf(b, "     %zu. %s\n", i + 1, in->preselect_names[i]);
        }
        rb_printf(b,
            "   Modulation determination: %s (confidence %d %%) -> %s\n",
            in->mod_family != NULL ? in->mod_family : "unknown",
            in->mod_family_conf,
            in->mod_reliable
                ? "RELIABLE - the family is decisive"
                : "not reliable - ambiguous family / weak margin");
        rb_printf(b,
            "   Candidate filter: %zu of the %zu preselected candidates "
            "match the detected modulation\n", in->n_filtered,
            in->n_preselect);
        for (size_t i = 0; i < in->n_filtered && i < 6; i++) {
            rb_printf(b, "     %zu. %s\n", i + 1, in->filtered_names[i]);
        }
        rb_printf(b,
            "   DAG decision: remaining analyses %s - %s\n",
            in->rest_warranted ? "WANTED" : "NOT WANTED",
            in->rest_reason != NULL ? in->rest_reason
                                    : "no rationale recorded");
        if (!in->rest_warranted) {
            rb_puts(b, "     The deeper analyses (A1..A9) were skipped: "
                               "the candidate filter already identified the "
                               "modulation family and the matching bands.\n");
        } else if (in->ran.a1) {
            rb_puts(b, "     They ran in increasing specialisation: "
                               "classic envelope -> pulse train -> HOS -> "
                               "cycle search -> detectors -> cumulants/"
                               "fluctuation -> intra-burst, each only when "
                               "the accumulated evidence warranted it.\n");
        }
        rb_puts(b, "\n");

        /* 1. frequency, power, classifier */
        rb_puts(b, "1. FREQUENCY, POWER AND CLASSIFIER "
                             "[analysis + metrics]\n");
        rb_printf(b,
            "   Occupied bandwidth %.1f kHz (%.3f-%.3f MHz) [99%% power "
            "containment of the selection spectrum]; measured selection "
            "bandwidth %.1f kHz\n",
            metrics.occupied_bw_hz / 1e3, metrics.occupied_lo_hz / 1e6,
            metrics.occupied_hi_hz / 1e6,
            (in->region.freq_max - in->region.freq_min) / 1e3);
        rb_printf(b,
            "   Power: average %.2f dB, peak %.2f dB; entropy %.3f bits "
            "[Shannon entropy of the spectrogram]\n",
            analysis.avg_power, analysis.max_power, analysis.entropy);
        rb_printf(b,
            "   SNR %.1f dB [spectral], flatness %.3f [Wiener entropy], band "
            "symmetry %+.3f, PAPR %.1f dB, tone purity %.3f\n",
            metrics.snr_db, metrics.spectral_flatness, metrics.band_symmetry,
            metrics.papr_db, metrics.tone_purity);
        rb_printf(b,
            "   Classifier: '%s' / '%s'; demodulable %s (%s)%s%s\n\n",
            analysis.signal_type, analysis.modulation,
            metrics.demodulable ? "yes" : "no",
            metrics.demod_mode[0] != '\0' ? metrics.demod_mode : "n/a",
            metrics.demod_note[0] != '\0' ? " - " : "", metrics.demod_note);

        /* 2. time structure */
        rb_puts(b, "2. TIME STRUCTURE [envelope, classic analysis, "
                             "spectrogram frames]\n");
        rb_printf(b,
            "   Envelope character: %s [smoothed |x| variation CV %.0f%%, "
            "duty at the mean %.0f%%, AM depth %.0f%%]\n",
            (ran_a2 && ok_pulse && pulse.num_pulses >= 2 && pulse.credible)
                ? "pulsed" : "bursty (no segmentable flat-top pulses)",
            ran_a2 && ok_pulse ? pulse.env_cv * 100.0 : 0.0,
            metrics.duty_cycle * 100.0, metrics.am_depth * 100.0);
        if (ran_a2 && ok_pulse && pulse.num_pulses >= 2) {
            if (pulse.credible) {
                rb_printf(b,
                    "   Pulse-train segmentation: ACCEPTED - %d pulses, width "
                    "%.3f ms, period %.3f ms, duty %.0f%%, jitter %.1f us "
                    "[threshold + hysteresis on the envelope]\n",
                    pulse.num_pulses, pulse.pulse_width_s * 1e3,
                    pulse.period_s * 1e3,
                    pulse.duty_cycle > 1.0 ? 100.0 : pulse.duty_cycle * 100.0,
                    pulse.jitter_s * 1e6);
            } else {
                rb_printf(b,
                    "   Pulse-train segmentation: REJECTED - %s "
                    "[credibility gate: duty, envelope CV, period jitter and "
                    "line-spectrum consistency]\n",
                    pulse.credibility_note);
            }
        }
        if (ran_a2 && ok_pulse && pulse.f0_hz > 0.0) {
            rb_printf(b,
                "   Burst/modulation rate %.2f Hz (%.3f ms period) [envelope "
                "line-spectrum fundamental]\n",
                pulse.f0_hz, 1e3 / pulse.f0_hz);
            if (analysis.prf > 0.0) {
                double r = analysis.prf / pulse.f0_hz;
                double dev = (r > 1.0 ? r : 1.0 / r) - 1.0;

                rb_printf(b,
                    "   Corroboration: frame-level fluctuation %.2f Hz "
                    "[spectrogram frame power, dt-limited] %s (%.0f%% "
                    "difference)\n",
                    analysis.prf, dev <= 0.25 ? "agrees" : "does NOT agree",
                    dev * 100.0);
            }
        }
        if (ran_a9 && ok_intraburst) {
            rb_printf(b,
                "   Intra-burst envelope: %s [%d bursts folded at full "
                "rate (coherence %.2f), %d sub-pulses, width %.2f us "
                "(CV %.0f%%), Delta-BIC %d, %d/4 detectors]\n",
                intraburst.verdict, intraburst.folded_bursts,
                intraburst.coherence, intraburst.sub_pulses,
                intraburst.width_mean_s * 1e6, intraburst.width_cv * 100.0,
                intraburst.delta_bic, intraburst.detectors_confirming);
        } else if (ran_a9) {
            rb_puts(b, "   Intra-burst envelope: not available"
                                 " (no usable burst structure)\n");
        } else {
            rb_printf(b,
                "   Intra-burst envelope: SKIPPED (%s)\n", skip_a9);
        }
        if (ran_a1 && ok_classic && classic.period_hz > 0.0) {
            if (ran_a2 && ok_pulse && pulse.f0_hz > 0.0) {
                double r = classic.period_hz / pulse.f0_hz;
                double dev = (r > 1.0 ? r : 1.0 / r) - 1.0;

                rb_printf(b,
                    "   Corroboration: classic period %.3f Hz [cepstrum/AMDF "
                    "on the 50 kHz complex envelope] %s (%.0f%% difference)\n",
                    classic.period_hz,
                    dev <= 0.25 ? "agrees" : "does NOT agree", dev * 100.0);
            } else {
                rb_printf(b,
                    "   Corroboration: classic period %.3f Hz [cepstrum/AMDF "
                    "on the 50 kHz complex envelope]\n", classic.period_hz);
            }
        }
        rb_puts(b, "\n");

        /* 3. modulation evidence */
        rb_puts(b, "3. MODULATION EVIDENCE [classifier, "
                             "higher-order statistics, cyclic cumulants]\n");
        rb_printf(b,
            "   FM deviation %.1f kHz [IF spread], FSK levels %d, symbol rate "
            "%.1f Hz (confidence %.0f%%), chirp trend %+.3f\n",
            metrics.fm_deviation_hz / 1e3, metrics.fsk_levels,
            metrics.sym_rate_hz, metrics.sym_rate_conf * 100.0,
            metrics.chirp_trend);
        rb_puts(b, "   Candidates (confidence, evidence):\n");
        for (size_t i = 0; i < in->n_mods && i < 3; i++) {
            rb_printf(b, "     %zu. %s - %d %% (%s)\n", i + 1,
                                   in->mods[i].name,
                                   in->mods[i].score,
                                   in->mods[i].why);
        }
        rb_printf(b,
            "   Votes: %s -> majority %s (%d/4)\n", vote_line,
            majority[0] ? majority : "none", maj);
        if (ran_a9 && ok_intraburst) {
            const char *ib;

            if (intraburst.uniform) {
                ib = "uniform intra-burst train -> clock/preamble/PFM-"
                     "supply signature rather than data";
            } else if (strstr(intraburst.verdict, "varying") != NULL) {
                ib = "varying intra-burst widths -> encoded data "
                     "(pulse-width/position coding)";
            } else if (strstr(intraburst.verdict, "noise-like") != NULL) {
                ib = "noise-like intra-burst envelope -> no coherent "
                     "intra-burst structure";
            } else if (strstr(intraburst.verdict, "constant-envelope") != NULL) {
                ib = "constant-envelope burst -> no intra-burst modulation";
            } else {
                ib = intraburst.verdict;
            }
            if (intraburst.burst_mod[0] != '\0') {
                rb_printf(b, "   Intra-burst modulation: %s",
                                       intraburst.burst_mod);
                if (intraburst.bursts_classified > 0) {
                    rb_printf(b, " (per-burst: %s %.0f%%",
                                           intraburst.burst_mod_cand[0],
                                           intraburst.burst_mod_support[0] *
                                               100.0);
                    if (intraburst.burst_mod_cand[1][0] != '\0') {
                        rb_printf(b, ", %s %.0f%%",
                                               intraburst.burst_mod_cand[1],
                                               intraburst.burst_mod_support[1] *
                                                   100.0);
                    }
                    rb_puts(b, ")");
                }
                rb_puts(b, "\n");
            }
            rb_printf(b, "   Intra-burst evidence: %s\n", ib);
        }
        rb_puts(b, "\n");

        /* 4. cyclostationary / spread spectrum */
        rb_puts(b, "4. CYCLOSTATIONARY AND SPREAD SPECTRUM "
                             "[SSCA, cycle detectors, cyclic cumulants]\n");
        {
            char cf1[32], cf2[32], cf3[32];

            if (fam_cf > 0.0) {
                snprintf(cf1, sizeof(cf1), "%.1f Hz", fam_cf);
            } else {
                snprintf(cf1, sizeof(cf1), "n/a");
            }
            if (ssca_cf > 0.0) {
                snprintf(cf2, sizeof(cf2), "%.1f Hz", ssca_cf);
            } else {
                snprintf(cf2, sizeof(cf2), "n/a");
            }
            if (det_cf > 0.0) {
                snprintf(cf3, sizeof(cf3), "%.1f Hz", det_cf);
            } else {
                snprintf(cf3, sizeof(cf3), "n/a");
            }
            rb_printf(b,
                "   Cycle frequencies (strongest): FAM %s, SSCA %s (%zu "
                "candidates), cycle detector %s -> cross-check %s\n", cf1,
                cf2, (ran_a5 && ok_ssca) ? ssca.cf_count : (size_t)0, cf3,
                cf_verdict);
        }

        rb_printf(b,
            "   Detector validation: %s [CFAR / multi-cycle cyclic "
            "autocorrelation]\n",
            (ran_a6 && ok_cycledet && cycledet.detected) ? "positive"
                                                         : "negative");
        /* spread-spectrum presence and type (all analyses) */
        if (spread_n > 0) {
            rb_puts(b,
                "   Spread-spectrum classification (confidence):\n");
            for (size_t i = 0; i < spread_n && i < 3; i++) {
                rb_printf(b, "     %zu. %s - %d %% (%s)\n",
                                       i + 1, spread[i].name,
                                       spread[i].score,
                                       spread[i].evidence);
            }
            rb_printf(b,
                "     -> %s (best hypothesis %s at %d %%); chip-rate "
                "evidence: Burel %.0f Hz, CPAR %.0f Hz\n",
                spread[0].score >= 40
                    ? (strcmp(spread[0].name, "Not spread spectrum") == 0
                           ? "not spread spectrum"
                           : "spread spectrum present")
                    : "no strong spread-spectrum evidence",
                spread[0].name, spread[0].score, bchip, cchip);
        }
        rb_printf(b,
            "   Cyclic cumulants: %s [C40/C42 at the candidate cycle "
            "frequency]\n",
            (ran_a8 && ok_ccc) ? ccc.verdict : "not executed");
        rb_printf(b,
            "   Detector presence: %d/4 (cycle detector %s, HOS %s, pulse "
            "train %d pulses, DSSS %s)\n\n",
            pres_k,
            ran_a6 ? (cycledet.detected ? "yes" : "no") : "n/a",
            ran_a3 ? (hos.non_gaussian ? "non-Gaussian" : "Gaussian") : "n/a",
            ran_a2 ? pulse.num_pulses : 0,
            (bchip > 0.0 && cchip > 0.0 && dsss_label[0] == 's') ? "yes"
                                                                : "no");

        /* 5. system identification (synthesis step) */
        rb_puts(b, "5. SYSTEM IDENTIFICATION (database match, synthesis "
                   "step only)\n");
        rb_puts(b, "   The analyses are prior-agnostic; this catalogue match "
                   "is computed after the DAG from the measured results and "
                   "never feeds back into any analysis.\n");
        {
            size_t n_valid = 0;

            for (size_t i = 0; i < in->n_ids; i++) {
                if (in->ids[i].valid) {
                    n_valid++;
                }
            }
            for (size_t i = 0; i < in->n_ids && i < 4; i++) {
                const RankedMatch *m = &in->ids[i];

                rb_printf(b, "   %zu. %s - %d %% %s", i + 1,
                                       m->signal->name, m->confidence_pct,
                                       m->valid ? "VALID" : "INVALID");
                if (m->why[0] != '\0') {
                    rb_printf(b, " (%s)", m->why);
                }
                rb_puts(b, "\n");
                if (!m->valid && m->invalid_reason[0] != '\0') {
                    rb_printf(b, "      reason: %s\n",
                                           m->invalid_reason);
                }
            }
            if (in->n_ids == 0) {
                rb_puts(b, "   (no catalogue candidate matched the "
                           "measured parameters)\n");
            } else if (in->anomaly) {
                rb_printf(b,
                    "   Validity %zu/%zu -> ANOMALY: %s\n", n_valid,
                    in->n_ids,
                    in->anomaly_explain[0] != '\0'
                        ? (strncmp(in->anomaly_explain, "ANOMALY: ",
                                   strlen("ANOMALY: ")) == 0
                               ? in->anomaly_explain + strlen("ANOMALY: ")
                               : in->anomaly_explain)
                        : "no candidate conforms to its specification");
            } else {
                rb_printf(b,
                    "   Validity %zu/%zu -> no anomaly: at least one "
                    "candidate complies with the measured frequency, "
                    "bandwidth and modulation\n", n_valid,
                    in->n_ids);
            }
        }

        /* 6. verdict */
        rb_printf(b,
            "\n6. VERDICT\n   Best-supported characterisation: %s\n",
            dir.verdict);

        /* 7. analysis provenance + the Director's rationale --------------
         * Additive section: it states which DAG nodes ran, which were
         * skipped/failed and why, and how the Director ranked the
         * hypotheses, so the consolidated view above is auditable. */
        if (1 && dir.n_provenance > 0) {
            report_section(b, "ANALYSIS PROVENANCE AND DIRECTOR RATIONALE");
            rb_puts(b,
                "DAG execution (node: state (elapsed) - reason):\n");
            for (size_t i = 0; i < dir.n_provenance; i++) {
                rb_printf(b, "   %s\n",
                                       dir.provenance[i]);
            }
            if (dir.n_rationale > 0) {
                rb_puts(b, "Director rationale:\n");
                for (size_t i = 0; i < dir.n_rationale; i++) {
                    rb_printf(b, "   %zu. %s\n", i + 1,
                                           dir.rationale[i]);
                }
            }
            if (dir.n_cand > 0) {
                rb_puts(b, "Ranked hypotheses (weight, evidence):\n");
                for (size_t i = 0; i < dir.n_cand; i++) {
                    rb_printf(b, "   %zu. %s - %.0f %% (%s)\n",
                                           i + 1, dir.cand[i].name,
                                           dir.cand[i].weight,
                                           dir.cand[i].evidence);
                }
            }
            rb_puts(b, "\n");
        }
    }



done:
    rb_puts(b, "\nReport generated by ToxicElement15\n");
}

ReportDoc *report_build(ReportInput *in)
{
    ReportDoc *doc = calloc(1, sizeof(*doc));
    RepBuf b = {0, 0, 0};

    if (doc == NULL) return NULL;
    report_header(in, &b);
    report_sot(in, &b);
    report_body(in, &b, doc);
    doc->text = rb_take(&b);
    doc->valid = doc->text != NULL;
    return doc;
}

const char *report_text(const ReportDoc *doc)
{
    return doc != NULL ? doc->text : "";
}

const char *report_format_name(ReportFormat fmt)
{
    switch (fmt) {
    case REPORT_TEXT:     return "text";
    case REPORT_MARKDOWN: return "markdown";
    case REPORT_HTML:     return "html";
    }
    return "text";
}

const char *report_format_suffix(ReportFormat fmt)
{
    switch (fmt) {
    case REPORT_TEXT:     return ".txt";
    case REPORT_MARKDOWN: return ".md";
    case REPORT_HTML:     return ".html";
    }
    return ".txt";
}

/* ------------------------------------------------------------------ */
/* Markdown presentation                                              */
/* ------------------------------------------------------------------ */

/* A section title is a non-empty, non-indented line that directly follows a
 * blank line and is followed by a rule line made only of '-' or '='. */
static bool is_rule_line(const char *s)
{
    bool any = false;

    while (*s == ' ' || *s == '\t') s++;
    while (*s == '-' || *s == '=') {
        any = true;
        s++;
    }
    while (*s == ' ' || *s == '\t' || *s == '\r') s++;
    return any && *s == '\0';
}

static bool is_title_line(const char *s)
{
    if (s[0] == '\0' || s[0] == ' ' || s[0] == '\t') return false;
    if (is_rule_line(s)) return false;
    return true;
}

static char *to_markdown(const char *text)
{
    RepBuf b = {0, 0, 0};
    const char *p = text;
    bool prev_blank = true;
    char line[1024];

    while (*p != '\0') {
        const char *eol = strchr(p, '\n');
        size_t len = eol != NULL ? (size_t)(eol - p) : strlen(p);
        const char *next = eol != NULL ? eol + 1 : p + len;
        size_t nlen = 0;
        const char *neol = strchr(next, '\n');

        if (len >= sizeof(line)) len = sizeof(line) - 1;
        memcpy(line, p, len);
        line[len] = '\0';
        if (neol != NULL) nlen = (size_t)(neol - next);
        else nlen = strlen(next);

        if (prev_blank && is_title_line(line) && nlen > 0) {
            char nbuf[256];
            size_t cn = nlen < sizeof(nbuf) - 1 ? nlen : sizeof(nbuf) - 1;

            memcpy(nbuf, next, cn);
            nbuf[cn] = '\0';
            if (is_rule_line(nbuf)) {
                if (strcmp(line, "SELECTION REPORT") == 0) {
                    rb_printf(&b, "# %s\n", line);
                } else {
                    rb_printf(&b, "\n## %s\n", line);
                }
                /* skip the rule line */
                p = neol != NULL ? neol + 1 : next + nlen;
                prev_blank = true;
                continue;
            }
        }
        rb_printf(&b, "%s\n", line);
        prev_blank = (line[0] == '\0');
        p = next;
    }
    return rb_take(&b);
}

/* ------------------------------------------------------------------ */
/* HTML presentation                                                  */
/* ------------------------------------------------------------------ */

static void html_escape(RepBuf *b, const char *s, size_t len)
{
    for (size_t i = 0; i < len; i++) {
        switch (s[i]) {
        case '&':  rb_puts(b, "&amp;"); break;
        case '<':  rb_puts(b, "&lt;"); break;
        case '>':  rb_puts(b, "&gt;"); break;
        case '"':  rb_puts(b, "&quot;"); break;
        default:   rb_putc(b, s[i]); break;
        }
    }
}

static char *to_html(const char *text, const char *title)
{
    RepBuf b = {0, 0, 0};
    const char *p = text;
    bool prev_blank = true;
    bool in_pre = false;
    char line[1024];

    rb_puts(&b,
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n");
    rb_printf(&b, "<title>%s</title>\n", title != NULL ? title : "Analysis report");
    rb_puts(&b,
        "<style>\n"
        "  :root { color-scheme: light dark; }\n"
        "  body { font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;\n"
        "         max-width: 62rem; margin: 2rem auto; padding: 0 1.25rem;\n"
        "         line-height: 1.5; }\n"
        "  h1 { font-size: 1.6rem; border-bottom: 2px solid currentColor;\n"
        "       padding-bottom: .35rem; }\n"
        "  h2 { font-size: 1.15rem; margin-top: 2rem; text-transform: uppercase;\n"
        "       letter-spacing: .04em; }\n"
        "  pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas,\n"
        "        monospace; font-size: .82rem; line-height: 1.35;\n"
        "        padding: .75rem 1rem; border-radius: .4rem;\n"
        "        background: rgba(127,127,127,.12); overflow-x: auto;\n"
        "        white-space: pre-wrap; }\n"
        "  p { margin: .6rem 0; }\n"
        "</style>\n</head>\n<body>\n");

    while (*p != '\0') {
        const char *eol = strchr(p, '\n');
        size_t len = eol != NULL ? (size_t)(eol - p) : strlen(p);
        const char *next = eol != NULL ? eol + 1 : p + len;
        size_t nlen;
        const char *neol = strchr(next, '\n');

        if (len >= sizeof(line)) len = sizeof(line) - 1;
        memcpy(line, p, len);
        line[len] = '\0';
        nlen = neol != NULL ? (size_t)(neol - next) : strlen(next);

        if (prev_blank && is_title_line(line) && nlen > 0) {
            char nbuf[256];
            size_t cn = nlen < sizeof(nbuf) - 1 ? nlen : sizeof(nbuf) - 1;

            memcpy(nbuf, next, cn);
            nbuf[cn] = '\0';
            if (is_rule_line(nbuf)) {
                if (in_pre) {
                    rb_puts(&b, "</pre>\n");
                    in_pre = false;
                }
                rb_printf(&b, "<h%s>", strcmp(line, "SELECTION REPORT") == 0
                                           ? "1" : "2");
                html_escape(&b, line, strlen(line));
                rb_printf(&b, "</h%s>\n", strcmp(line, "SELECTION REPORT") == 0
                                              ? "1" : "2");
                p = neol != NULL ? neol + 1 : next + nlen;
                prev_blank = true;
                continue;
            }
        }
        if (line[0] == '\0') {
            if (in_pre) {
                rb_puts(&b, "</pre>\n");
                in_pre = false;
            }
        } else {
            if (!in_pre) {
                rb_puts(&b, "<pre>");
                in_pre = true;
            }
            html_escape(&b, line, strlen(line));
            rb_putc(&b, '\n');
        }
        prev_blank = (line[0] == '\0');
        p = next;
    }
    if (in_pre) rb_puts(&b, "</pre>\n");
    rb_puts(&b, "</body>\n</html>\n");
    return rb_take(&b);
}

/* ------------------------------------------------------------------ */
/* public render                                                      */
/* ------------------------------------------------------------------ */

char *report_render(const ReportDoc *doc, ReportFormat fmt)
{
    if (doc == NULL || doc->text == NULL) return NULL;
    switch (fmt) {
    case REPORT_TEXT:
        {
            char *copy = malloc(strlen(doc->text) + 1);

            if (copy != NULL) strcpy(copy, doc->text);
            return copy;
        }
    case REPORT_MARKDOWN:
        return to_markdown(doc->text);
    case REPORT_HTML:
        return to_html(doc->text, "SELECTION REPORT");
    }
    return NULL;
}

void report_doc_free(ReportDoc *doc)
{
    if (doc == NULL) return;
    free(doc->text);
    free(doc);
}
