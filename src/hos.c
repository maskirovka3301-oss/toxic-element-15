/*
 * hos.c — higher-order statistics (see header).
 *
 * Cumulants are estimated from the raw sample moments of the centralised
 * data (mean removed), using the moment-to-cumulant relations of Swami
 * & Sadler (2000). The detection statistic is the normalised fourth-
 * order cumulant |C42|/C21^2: zero in expectation for Gaussian noise
 * with an asymptotic standard deviation sqrt(24/N), so the threshold is
 * 5 * sqrt(24/N). The bispectrum is estimated by windowed averaging of
 * X(f1) X(f2) X*(f1+f2), normalised into a bicoherence magnitude.
 */
#include "hos.h"
#include "debug.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsp.h"
#include "aconfig.h"
#include "metrics.h"
#include "region.h"

#define HOS_BS_FFT      256    /* bispectrum window FFT size      */
#define HOS_BS_WINDOWS  256    /* max averaged windows           */
#define HOS_FS_TARGET   100000.0

void hos_clear(HosResult *r)
{
    if (r == NULL) return;
    free(r->bisp);
    memset(r, 0, sizeof(*r));
}

bool hos_analyze(const IqSource *src, const Region *region, HosResult *out)
{
    hos_clear(out);
    if (src == NULL || region == NULL) return false;

    Region r = *region;
    if (!region_normalize(&r, src)) return false;

    size_t s0 = (size_t)(r.time_min * src->sample_rate);
    size_t s1 = (size_t)(r.time_max * src->sample_rate);
    if (s1 > src->total_samples) s1 = src->total_samples;
    if (s0 >= s1) return false;
    size_t n = s1 - s0;
    if (n < 1024) return false;
    if (n > TE15_MAX_ANALYSIS_SAMPLES) n = TE15_MAX_ANALYSIS_SAMPLES;

    float *seg = malloc(2 * n * sizeof(float));
    if (seg == NULL) return false;
    iq_source_read_complex(src, s0, seg, n);
    double offset = ((r.freq_min + r.freq_max) / 2.0) - src->center_freq;
    metrics_mixdown(seg, n, src->sample_rate, -offset, seg);

    /* centralise (remove the sample mean) */
    {
        double mr = 0.0, mi = 0.0;
        for (size_t i = 0; i < n; i++) {
            mr += seg[2 * i];
            mi += seg[2 * i + 1];
        }
        mr /= (double)n;
        mi /= (double)n;
        for (size_t i = 0; i < n; i++) {
            seg[2 * i]     = (float)(seg[2 * i] - mr);
            seg[2 * i + 1] = (float)(seg[2 * i + 1] - mi);
        }
    }

    /* raw moments */
    double M20r = 0.0, M20i = 0.0, M21 = 0.0;
    double M30r = 0.0, M30i = 0.0, M31r = 0.0, M31i = 0.0;
    double M40 = 0.0, M41r = 0.0, M41i = 0.0, M42 = 0.0;
    double M60r = 0.0, M60i = 0.0, M63 = 0.0;
    double m4_raw = 0.0, m2_raw = 0.0;
    for (size_t i = 0; i < n; i++) {
        double a = seg[2 * i], b = seg[2 * i + 1];
        double ab2 = a * a + b * b;
        double x2r = a * a - b * b, x2i = 2.0 * a * b;
        double x3r = x2r * a - x2i * b, x3i = x2r * b + x2i * a;
        double x4r = x2r * x2r - x2i * x2i;
        double x4i = 2.0 * x2r * x2i;
        double x6r = x4r * x2r - x4i * x2i;
        double x6i = x4r * x2i + x4i * x2r;

        M20r += x2r; M20i += x2i; M21 += ab2;
        M30r += x3r; M30i += x3i;
        M31r += x2r * a + x2i * b;  /* Re(x^2 x*) = Re(x^2 * conj(x)) */
        M31i += x2i * a - x2r * b;
        M40 += x4r;
        M41r += x3r * a + x3i * b;
        M41i += x3i * a - x3r * b;
        M42 += ab2 * ab2;
        M60r += x6r; M60i += x6i;
        M63 += ab2 * ab2 * ab2;
        m4_raw += ab2 * ab2;
        m2_raw += ab2;
    }
    M20r /= (double)n; M20i /= (double)n; M21 /= (double)n;
    M30r /= (double)n; M30i /= (double)n; M31r /= (double)n; M31i /= (double)n;
    M40  /= (double)n; M41r /= (double)n; M41i /= (double)n; M42 /= (double)n;
    M60r /= (double)n; M60i /= (double)n; M63 /= (double)n;
    m4_raw /= (double)n; m2_raw /= (double)n;

    /* moment -> cumulant (Swami & Sadler) */
    out->c20_re = M20r;
    out->c20_im = M20i;
    out->c21 = M21;
    out->c30_re = M30r;
    out->c30_im = M30i;
    out->c31_re = M31r;
    out->c31_im = M31i;
    out->c40_re = M40 - 3.0 * (M20r * M20r - M20i * M20i);
    out->c40_im = 0.0 - 3.0 * (2.0 * M20r * M20i);
    {
        /* C41 = M41 - 3 M20 M21 (complex) */
        out->c41_re = M41r - 3.0 * (M20r * M21);
        out->c41_im = M41i - 3.0 * (M20i * M21);
    }
    out->c42 = M42 - (M20r * M20r + M20i * M20i) - 2.0 * M21 * M21;
    {
        /* C60 = M60 - 15 M40 M20 + 30 M20^3 (M40 real, M60 complex) */
        double M20_2r = M20r * M20r - M20i * M20i;
        double M20_2i = 2.0 * M20r * M20i;
        double M20_3r = M20_2r * M20r - M20_2i * M20i;
        double M20_3i = M20_2r * M20i + M20_2i * M20r;
        out->c60_re = M60r - 15.0 * M40 * M20r + 30.0 * M20_3r;
        out->c60_im = M60i - 15.0 * M40 * M20i + 30.0 * M20_3i;
    }
    out->c63 = M63 - 9.0 * out->c42 * out->c21 - 6.0 * out->c21 * out->c21 * out->c21;
    out->kurtosis = (m2_raw > 0.0) ? m4_raw / (m2_raw * m2_raw) - 2.0 : 0.0;

    /* normalised magnitudes */
    {
        double c21sq = M21 * M21;
        if (c21sq <= 0.0) c21sq = 1e-30;
        out->n40 = sqrt(out->c40_re * out->c40_re + out->c40_im * out->c40_im) / c21sq;
        out->n41 = sqrt(out->c41_re * out->c41_re + out->c41_im * out->c41_im) / c21sq;
        out->n42 = fabs(out->c42) / c21sq;
    }

    /* detection: Gaussian noise gives C42 = 0 with std sqrt(24/N) */
    out->samples = n;
    out->n42_threshold = 5.0 * sqrt(24.0 / (double)n);
    out->non_gaussian = out->n42 > out->n42_threshold;

    /* constellation-style hint from the classic cumulant features */
    if (out->non_gaussian) {
        if (out->c42 < 0.0 && out->n40 > 0.9) {
            snprintf(out->hint, sizeof(out->hint),
                     "BPSK-like (C42 < 0, |C40|/C21^2 = %.2f ~ 2)",
                     out->n40);
        } else if (out->c42 < 0.0 && out->n40 < 0.35) {
            snprintf(out->hint, sizeof(out->hint),
                     "QPSK-like (C42 < 0, |C40|/C21^2 = %.2f ~ 0)",
                     out->n40);
        } else if (out->c42 > 0.0) {
            snprintf(out->hint, sizeof(out->hint),
                     "higher-order QAM / multi-level (C42 = %.3f > 0)",
                     out->c42 / (out->c21 * out->c21));
        } else {
            snprintf(out->hint, sizeof(out->hint),
                     "non-Gaussian constellation (C42 = %.3f)",
                     out->c42 / (out->c21 * out->c21));
        }
        snprintf(out->note, sizeof(out->note),
                 "Non-Gaussian signal detected: |C42|/C21^2 = %.4f "
                 "vs Gaussian threshold %.4f (N = %zu). %s.",
                 out->n42, out->n42_threshold, n, out->hint);
    } else {
        snprintf(out->hint, sizeof(out->hint), "Gaussian-consistent");
        snprintf(out->note, sizeof(out->note),
                 "No significant higher-order cumulant: |C42|/C21^2 = "
                 "%.4f below the Gaussian threshold %.4f (N = %zu) - "
                 "noise-like, no non-Gaussian signal detected",
                 out->n42, out->n42_threshold, n);
    }

    /* bispectrum: windowed bicoherence over the principal domain */
    {
    size_t D = dsp_decimation_for(
        src->sample_rate, r.freq_max - r.freq_min,
        aconfig_get_num("hos", "fs_target_hz", HOS_FS_TARGET));
        if (D < 1) D = 1;
        size_t nd = n / D;
        float *dec = malloc(2 * nd * sizeof(float));
        if (dec != NULL) {
            size_t got = dsp_decimate_staged(seg, n, D, dec);
            nd = got < nd ? got : nd;

            const int N = HOS_BS_FFT;
            const int half = N / 2;
            size_t K = nd / (size_t)N;
            if (K > HOS_BS_WINDOWS) K = HOS_BS_WINDOWS;
            if (K >= 8) {
                out->bs_n = half;
                out->bisp = calloc((size_t)half * (size_t)half, sizeof(float));
                if (out->bisp != NULL) {
                    FftPlan *plan = fft_plan_create(N);
                    float *win = malloc((size_t)N * sizeof(float));
                    fftwf_complex *X = fftwf_malloc(sizeof(fftwf_complex) * (size_t)N);
                    double *S_r = calloc((size_t)half * (size_t)half, sizeof(double));
                    double *S_i = calloc((size_t)half * (size_t)half, sizeof(double));
                    double *P12 = calloc((size_t)half * (size_t)half, sizeof(double));
                    double *P3 = calloc((size_t)half, sizeof(double));
                    if (plan != NULL && win != NULL && X != NULL &&
                        S_r != NULL && S_i != NULL && P12 != NULL && P3 != NULL) {
                        dsp_hann(win, (size_t)N);
                        for (size_t k = 0; k < K; k++) {
                            fft_plan_transform(plan, dec + 2 * k * (size_t)N,
                                               win, X);
                            for (int f1 = 0; f1 < half; f1++) {
                                for (int f2 = 0; f2 < half; f2++) {
                                    int f3 = f1 + f2;
                                    if (f3 >= half) continue;
                                    size_t idx = (size_t)f1 * (size_t)half + (size_t)f2;
                                    double ar = X[f1][0], ai = X[f1][1];
                                    double br = X[f2][0], bi = X[f2][1];
                                    double cr = X[f3][0], ci = -X[f3][1];
                                    /* X(f1) * X(f2) * X*(f3) */
                                    double pr = ar * br - ai * bi;
                                    double pi = ar * bi + ai * br;
                                    double sr = pr * cr - pi * ci;
                                    double si = pr * ci + pi * cr;
                                    S_r[idx] += sr;
                                    S_i[idx] += si;
                                    P12[idx] += pr * pr + pi * pi;
                                    P3[f3] += cr * cr + ci * ci;
                                }
                            }
                        }
                        for (int f1 = 0; f1 < half; f1++) {
                            for (int f2 = 0; f2 < half; f2++) {
                                int f3 = f1 + f2;
                                if (f3 >= half) continue;
                                size_t idx = (size_t)f1 * (size_t)half + (size_t)f2;
                                double mag = sqrt(S_r[idx] * S_r[idx] +
                                                  S_i[idx] * S_i[idx]);
                                double denom = sqrt(P12[idx] * P3[f3]);
                                double v = denom > 0.0 ? mag / denom : 0.0;
                                out->bisp[idx] = (float)(v > 1.0 ? 1.0 : v);
                            }
                        }
                    }
                    fft_plan_destroy(plan);
                    free(win);
                    fftwf_free(X);
                    free(S_r); free(S_i); free(P12); free(P3);
                }
            }
            free(dec);
        }
    }
    free(seg);

    TE15_LOG("\nHIGHER-ORDER STATISTICS\n"
           "N = %zu samples\n"
           "C20 = %.6f %+.6fj   C21 = %.6f\n"
           "C40 = %.6f %+.6fj   C41 = %.6f %+.6fj   C42 = %.6f\n"
           "C60 = %.6f %+.6fj   C63 = %.6f\n"
           "|C40|/C21^2 = %.4f  |C41|/C21^2 = %.4f  |C42|/C21^2 = %.4f\n"
           "%s\n",
           n, out->c20_re, out->c20_im, out->c21,
           out->c40_re, out->c40_im, out->c41_re, out->c41_im, out->c42,
           out->c60_re, out->c60_im, out->c63,
           out->n40, out->n41, out->n42, out->note);
    return true;
}
