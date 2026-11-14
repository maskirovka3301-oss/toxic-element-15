/*
 * filters.c — practical filter toolbox (see header).
 */
#include "filters.h"
#include "debug.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsp.h"
#include "metrics.h"
#include "region.h"
#include "sptk.h"

void filter_clear(FilterResult *r)
{
    if (r == NULL) return;
    free(r->response);
    free(r->input_psd);
    free(r->output_psd);
    memset(r, 0, sizeof(*r));
}

bool filter_analyze(const IqSource *src, const Region *region, int kind,
                    double f1_hz, double f2_hz, int taps, FilterResult *out)
{
    filter_clear(out);
    if (src == NULL || region == NULL || kind < 0 || kind > 3 ||
        taps < 3 || (taps % 2) == 0 || f1_hz <= 0.0 ||
        f1_hz >= src->sample_rate / 2.0) {
        return false;
    }

    Region r = *region;
    if (!region_normalize(&r, src)) return false;

    size_t s0 = (size_t)(r.time_min * src->sample_rate);
    size_t s1 = (size_t)(r.time_max * src->sample_rate);
    if (s1 > src->total_samples) s1 = src->total_samples;
    if (s0 >= s1) return false;
    size_t n = s1 - s0;
    if (n < (size_t)taps) return false;
    if (n > TE15_MAX_ANALYSIS_SAMPLES) n = TE15_MAX_ANALYSIS_SAMPLES;

    float *seg = malloc(2 * n * sizeof(float));
    if (seg == NULL) return false;
    iq_source_read_complex(src, s0, seg, n);
    double offset = ((r.freq_min + r.freq_max) / 2.0) - src->center_freq;
    metrics_mixdown(seg, n, src->sample_rate, -offset, seg);

    /* input power */
    double p_in = 0.0;
    for (size_t i = 0; i < n; i++) {
        p_in += (double)seg[2 * i] * seg[2 * i] +
                (double)seg[2 * i + 1] * seg[2 * i + 1];
    }
    p_in /= (double)n;
    out->power_in_db = 10.0 * log10(p_in + 1e-30);

    /* design the filter */
    float *t = malloc((size_t)taps * sizeof(float));
    if (t == NULL) {
        free(seg);
        return false;
    }
    if (sptk_fir_design(kind, f1_hz / src->sample_rate,
                        f2_hz / src->sample_rate, t, (size_t)taps) == 0) {
        free(t);
        free(seg);
        return false;
    }

    /* keep a copy for the input PSD */
    float *seg_in = malloc(2 * n * sizeof(float));
    if (seg_in == NULL) {
        free(t);
        free(seg);
        return false;
    }
    memcpy(seg_in, seg, 2 * n * sizeof(float));
    sptk_fir_apply(seg, n, t, (size_t)taps);

    double p_out = 0.0;
    for (size_t i = 0; i < n; i++) {
        p_out += (double)seg[2 * i] * seg[2 * i] +
                 (double)seg[2 * i + 1] * seg[2 * i + 1];
    }
    p_out /= (double)n;
    out->power_out_db = 10.0 * log10(p_out + 1e-30);

    out->kind = kind;
    out->f1_hz = f1_hz;
    out->f2_hz = (kind == 2 || kind == 3) ? f2_hz : 0.0;
    out->taps = taps;

    /* PSD before/after over FILTER_N via Welch-like averaging */
    const int N = FILTER_N;
    out->n = N;
    out->response = calloc((size_t)N, sizeof(double));
    out->input_psd = calloc((size_t)N, sizeof(double));
    out->output_psd = calloc((size_t)N, sizeof(double));
    if (out->response == NULL || out->input_psd == NULL ||
        out->output_psd == NULL) {
        filter_clear(out);
        free(seg);
        return false;
    }

    /* response from the taps (zero-padded FFT) */
    {
        float *taps_z = calloc(2 * (size_t)N, sizeof(float));
        if (taps_z == NULL) {
            filter_clear(out);
            free(seg_in);
            free(seg);
            return false;
        }
        for (int i = 0; i < taps; i++) {
            taps_z[i] = t[i];
        }
        FftPlan *plan = fft_plan_create(N);
        fftwf_complex *H = fftwf_malloc(sizeof(fftwf_complex) * (size_t)N);
        if (plan == NULL || H == NULL) {
            fft_plan_destroy(plan);
            fftwf_free(H);
            free(taps_z);
            filter_clear(out);
            free(seg_in);
            free(seg);
            return false;
        }
        fft_plan_transform(plan, taps_z, NULL, H);
        for (int b = 0; b < N; b++) {
            out->response[b] = sqrt((double)H[b][0] * H[b][0] +
                                    (double)H[b][1] * H[b][1]);
        }
        fft_plan_destroy(plan);
        fftwf_free(H);
        free(taps_z);
    }
    free(t);

    /* Welch PSDs: input and filtered output, dB normalised */
    {
        FftPlan *plan = fft_plan_create(N);
        float *win = malloc((size_t)N * sizeof(float));
        fftwf_complex *X = fftwf_malloc(sizeof(fftwf_complex) * (size_t)N);
        if (plan == NULL || win == NULL || X == NULL) {
            fft_plan_destroy(plan);
            free(win);
            fftwf_free(X);
            filter_clear(out);
            free(seg_in);
            free(seg);
            return false;
        }
        dsp_hann(win, (size_t)N);
        size_t frames = n / (size_t)N;
        if (frames > 64) frames = 64;
        if (frames < 1) frames = 1;
        for (int pass = 0; pass < 2; pass++) {
            const float *data = (pass == 0) ? seg_in : seg;
            double *acc = calloc((size_t)N, sizeof(double));
            if (acc == NULL) {
                filter_clear(out);
                free(win);
                fftwf_free(X);
                fft_plan_destroy(plan);
                free(seg_in);
                free(seg);
                return false;
            }
            for (size_t f = 0; f < frames; f++) {
                fft_plan_transform(plan, data + 2 * f * (size_t)N, win, X);
                for (int b = 0; b < N; b++) {
                    acc[b] += (double)X[b][0] * X[b][0] +
                              (double)X[b][1] * X[b][1];
                }
            }
            double mx = 0.0;
            for (int b = 0; b < N; b++) {
                if (acc[b] > mx) mx = acc[b];
            }
            for (int b = 0; b < N; b++) {
                double v = 10.0 * log10(acc[b] / (double)frames + 1e-30) -
                           10.0 * log10(mx + 1e-30);
                if (pass == 0) out->input_psd[b] = v;
                else out->output_psd[b] = v;
            }
            free(acc);
        }
        fft_plan_destroy(plan);
        free(win);
        fftwf_free(X);
    }
    free(seg_in);
    free(seg);

    snprintf(out->note, sizeof(out->note),
             "Filter: %s, cutoffs %.1f kHz %s, %d taps. Power %.1f dB -> "
             "%.1f dB (change %.1f dB).",
             kind == 0 ? "lowpass" : kind == 1 ? "highpass" :
             kind == 2 ? "bandpass" : "bandstop",
             f1_hz / 1e3, "",
             taps, out->power_in_db, out->power_out_db,
             out->power_out_db - out->power_in_db);
    TE15_LOG("\nPRACTICAL FILTERS\n%s\n", out->note);
    return true;
}
