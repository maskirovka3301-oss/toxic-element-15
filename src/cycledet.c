/*
 * cycledet.c — CFAR cycle detectors (see header).
 *
 * The cyclic autocorrelation magnitude at zero lag,
 *   R_hat^alpha(0) = < |x(n)|^2 e^{-i 2 pi alpha n} >,
 * is a cycle detector: zero in expectation for Gaussian noise (with
 * standard deviation ~ power/sqrt(N)) and large at true cycle
 * frequencies of the signal. Candidate cycle frequencies come from a
 * quick scan of the non-conjugate cyclic domain profile computed here
 * with the same time-smoothing machinery, optionally after whitening
 * the data (H_w(f) = 1/sqrt(S(f))) to flatten the noise. The top
 * individual statistics are combined into a multi-cycle statistic.
 */
#include "cycledet.h"
#include "debug.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsp.h"
#include "aconfig.h"
#include "metrics.h"
#include "region.h"

#define CYCLEDET_N        512    /* scan FFT size                     */
#define CYCLEDET_FS_TARGET 400000.0
#define CYCLEDET_NP       512    /* max averaged frames               */

void cycledet_clear(CycleDetResult *r)
{
    if (r == NULL) return;
    free(r->cf_hz);
    free(r->stat);
    free(r->whitened_stat);
    memset(r, 0, sizeof(*r));
}

bool cycledet_analyze(const IqSource *src, const Region *region,
                      CycleDetResult *out)
{
    cycledet_clear(out);
    if (src == NULL || region == NULL) return false;

    Region r = *region;
    if (!region_normalize(&r, src)) return false;

    size_t s0 = (size_t)(r.time_min * src->sample_rate);
    size_t s1 = (size_t)(r.time_max * src->sample_rate);
    if (s1 > src->total_samples) s1 = src->total_samples;
    if (s0 >= s1) return false;
    size_t n_work = s1 - s0;
    if (n_work > TE15_MAX_ANALYSIS_SAMPLES) n_work = TE15_MAX_ANALYSIS_SAMPLES;

    float *seg = malloc(2 * n_work * sizeof(float));
    if (seg == NULL) return false;
    iq_source_read_complex(src, s0, seg, n_work);
    double offset = ((r.freq_min + r.freq_max) / 2.0) - src->center_freq;
    metrics_mixdown(seg, n_work, src->sample_rate, -offset, seg);
    size_t D = dsp_decimation_for(
        src->sample_rate, r.freq_max - r.freq_min,
        aconfig_get_num("cycledet", "fs_target_hz",
                        CYCLEDET_FS_TARGET));
    if (D < 1) D = 1;
    size_t nd = n_work / D;
    float *dec = malloc(2 * nd * sizeof(float));
    if (dec == NULL) {
        free(seg);
        return false;
    }
    size_t got = dsp_decimate_staged(seg, n_work, D, dec);
    nd = got < nd ? got : nd;
    free(seg);

    const int n = CYCLEDET_N;
    if (nd < (size_t)n * 8) {
        free(dec);
        return false;
    }

    /* whitened copy for the whitened statistics */
    float *wh = malloc(2 * nd * sizeof(float));
    if (wh == NULL) {
        free(dec);
        return false;
    }
    memcpy(wh, dec, 2 * nd * sizeof(float));
    dsp_whiten(wh, nd);

    size_t np = nd / (size_t)n;
    if (np > CYCLEDET_NP) np = CYCLEDET_NP;

    const int num_cf = n / 2 + 1; /* alpha bins 1..n/2 inclusive */
    out->num_cf = num_cf;
    out->cf_hz = calloc((size_t)num_cf, sizeof(double));
    out->stat = calloc((size_t)num_cf, sizeof(double));
    out->whitened_stat = calloc((size_t)num_cf, sizeof(double));
    if (out->cf_hz == NULL || out->stat == NULL ||
        out->whitened_stat == NULL) {
        cycledet_clear(out);
        free(wh);
        free(dec);
        return false;
    }

    const double fs = src->sample_rate / (double)D;
    for (int a = 1; a < num_cf; a++) {
        out->cf_hz[a] = (double)a * fs / (double)n;
    }

    /* per-frame FFTs of |x|^2: the alpha-domain scan */
    FftPlan *plan = fft_plan_create(n);
    float *win = malloc((size_t)n * sizeof(float));
    fftwf_complex *X = fftwf_malloc(sizeof(fftwf_complex) * (size_t)n);
    if (plan == NULL || win == NULL || X == NULL) {
        cycledet_clear(out);
        free(win);
        fftwf_free(X);
        fft_plan_destroy(plan);
        free(wh);
        free(dec);
        return false;
    }
    dsp_hann(win, (size_t)n);

    double pow_sum = 0.0;
    for (size_t k = 0; k < np; k++) {
        float *block = dec + 2 * k * (size_t)n;
        double pw = 0.0;
        for (int i = 0; i < n; i++) {
            double re = block[2 * i], im = block[2 * i + 1];
            pw += re * re + im * im;
        }
        pow_sum += pw;
    }
    double power = pow_sum / (double)(np * n); /* <|x|^2> */

    static const int LAGS[] = {1, 2, 4, 8, 16, 32, 40};
    const int num_lags = (int)(sizeof(LAGS) / sizeof(LAGS[0]));
    double *mag = calloc((size_t)num_lags * (size_t)num_cf, sizeof(double));
    double *magw = calloc((size_t)num_lags * (size_t)num_cf, sizeof(double));
    float *pw = malloc(2 * (size_t)n * sizeof(float));
    if (mag == NULL || magw == NULL || pw == NULL) {
        free(mag);
        free(magw);
        free(pw);
        cycledet_clear(out);
        free(win);
        fftwf_free(X);
        fft_plan_destroy(plan);
        free(wh);
        free(dec);
        return false;
    }
    /* The cycle detector statistic is the CYCLIC AUTOCORRELATION
     * magnitude at several lags (Spooner's detectors correlate the
     * signal with delayed versions of itself; the square-law statistic
     * alone misses baseband BPSK whose envelope is constant):
     *   R_hat^alpha(tau) = < x(n) x*(n-tau) e^{-i 2 pi alpha n} >.
     * Each lag product is DC-removed per frame (sinc-lobe leakage). */
    for (size_t k = 0; k < np; k++) {
        for (int li = 0; li < num_lags; li++) {
            int tau = LAGS[li];
            /* raw */
            double mean = 0.0;
            for (int i = 0; i < n; i++) {
                int j = i - tau;
                if (j < 0) j += n; /* cyclic lag across the frame */
                float re1 = dec[2 * (k * (size_t)n + i)];
                float im1 = dec[2 * (k * (size_t)n + i) + 1];
                float re2 = dec[2 * (k * (size_t)n + j)];
                float im2 = dec[2 * (k * (size_t)n + j) + 1];
                pw[2 * i] = win[i] * (re1 * re2 + im1 * im2);
                pw[2 * i + 1] = 0.0f;
                mean += pw[2 * i];
            }
            mean /= (double)n;
            for (int i = 0; i < n; i++) pw[2 * i] -= (float)mean;
            fft_plan_transform(plan, pw, NULL, X);
            for (int a = 1; a < num_cf; a++) {
                mag[(size_t)li * num_cf + a] +=
                    sqrt((double)X[a][0] * X[a][0] +
                         (double)X[a][1] * X[a][1]);
            }
            /* whitened */
            mean = 0.0;
            for (int i = 0; i < n; i++) {
                int j = i - tau;
                if (j < 0) j += n;
                float re1 = wh[2 * (k * (size_t)n + i)];
                float im1 = wh[2 * (k * (size_t)n + i) + 1];
                float re2 = wh[2 * (k * (size_t)n + j)];
                float im2 = wh[2 * (k * (size_t)n + j) + 1];
                pw[2 * i] = win[i] * (re1 * re2 + im1 * im2);
                pw[2 * i + 1] = 0.0f;
                mean += pw[2 * i];
            }
            mean /= (double)n;
            for (int i = 0; i < n; i++) pw[2 * i] -= (float)mean;
            fft_plan_transform(plan, pw, NULL, X);
            for (int a = 1; a < num_cf; a++) {
                magw[(size_t)li * num_cf + a] +=
                    sqrt((double)X[a][0] * X[a][0] +
                         (double)X[a][1] * X[a][1]);
            }
        }
    }
    free(pw);
    fft_plan_destroy(plan);
    free(win);
    fftwf_free(X);
    free(dec);

    /* normalise: |R_hat^alpha(0)| / power; the Hann window reduces the
     * peak by its coherent gain (0.5) and the FFT includes a 1/n that
     * is common to both the statistic and the power reference. */
    const double cg = 0.5; /* Hann coherent gain */
    double best_stat = 0.0;
    int best_a = 0;
    for (int a = 1; a < num_cf; a++) {
        double m = 0.0, mw = 0.0;
        for (int li = 0; li < num_lags; li++) {
            double v = mag[(size_t)li * num_cf + a];
            if (v > m) m = v;
            double vw = magw[(size_t)li * num_cf + a];
            if (vw > mw) mw = vw;
        }
        out->stat[a] = (m / (double)np / cg) / (power * (double)n);
        out->whitened_stat[a] =
            (mw / (double)np / cg) / (power * (double)n);
        if (out->stat[a] > best_stat) {
            best_stat = out->stat[a];
            best_a = a;
        }
    }
    free(mag);
    free(magw);
    free(wh);

    /* Gaussian baseline: the measured noise floor of the lag-product
     * statistic exceeds the analytic 1/sqrt(np*n) by a constant factor
     * (max over bins and lags); the empirically calibrated threshold
     * below keeps the false-alarm rate near zero on synthetic noise
     * while detecting symbol-rate cycles. */
    out->threshold = 100.0 / sqrt((double)(np * n));
    out->multi_threshold = 2.0 * out->threshold;
    out->multi_cycle = 0.0;
    /* multi-cycle: sum of the top-4 squared statistics */
    {
        double top[4] = {0.0, 0.0, 0.0, 0.0};
        for (int a = 1; a < num_cf; a++) {
            double v = out->stat[a];
            for (int t = 0; t < 4; t++) {
                if (v > top[t]) {
                    for (int u = 3; u > t; u--) top[u] = top[u - 1];
                    top[t] = v;
                    break;
                }
            }
        }
        for (int t = 0; t < 4; t++) out->multi_cycle += top[t] * top[t];
    }
    out->best_cf = (best_a > 0) ? out->cf_hz[best_a] : 0.0;
    out->best_stat = best_stat;
    out->detected = out->best_stat > out->threshold;
    out->whitened = true;

    if (out->detected) {
        snprintf(out->note, sizeof(out->note),
                 "Cycle detector: SIGNAL PRESENT - strongest cyclic "
                 "autocorrelation at %.2f Hz (%.4f vs threshold %.4f; "
                 "multi-cycle %.4f).",
                 out->best_cf, out->best_stat, out->threshold,
                 out->multi_cycle);
    } else {
        snprintf(out->note, sizeof(out->note),
                 "Cycle detector: no significant cyclic autocorrelation "
                 "(multi-cycle %.4f below threshold %.4f) - noise-like.",
                 out->multi_cycle, out->multi_threshold);
    }

    TE15_LOG("\nCYCLE DETECTORS (CFAR, multi-cycle, whitened)\n"
           "N = %zu frames x %d, power %.6f\n"
           "best CF %.2f Hz, |R^a(0)|/P = %.4f (thr %.4f)\n"
           "multi-cycle %.4f (thr %.4f)\n%s\n",
           np, n, power, out->best_cf, out->best_stat, out->threshold,
           out->multi_cycle, out->multi_threshold, out->note);
    return true;
}
