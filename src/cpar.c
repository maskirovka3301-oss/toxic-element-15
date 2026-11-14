/*
 * cpar.c — Cumulative Peak-to-Average Ratio DSSS detector (see header).
 *
 * Pipeline:
 *   1. Copy the selection, mix down to the region centre, FIR decimate
 *      to ~500 kHz effective rate.
 *   2. Split into M windows of T samples (T = 16384 so realistic code
 *      periods fit inside the lag window); average the FFT-based
 *      autocorrelation estimates into Rbar(tau), normalised to
 *      r(tau) = |Rbar(tau)| / |Rbar(0)|.
 *   3. Candidate peaks: local maxima of r above 2x the median, at least
 *      4 lags apart, outside the near-zero-lag chip triangle.
 *   4. PAR per peak: r(peak)^2 / mean(r^2 over +-32 lags, excluding the
 *      peak). Noise: PAR ~ 1; the code-period peak of a DSSS signal:
 *      PAR >> 1.
 *   5. CPAR = sum of the PARs of significant peaks (PAR > 25); a
 *      positive CPAR is the detection verdict. The peak lag is the
 *      code period; the chip period is recovered from the initial
 *      linear slope of r(tau) (the chip correlation triangle).
 */
#include "cpar.h"
#include "debug.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsp.h"
#include "aconfig.h"
#include "metrics.h"
#include "region.h"

#define CPAR_FS_TARGET   500000.0
#define CPAR_WINDOW      16384
#define CPAR_MAX_WINDOWS 128
#define CPAR_MIN_WINDOWS 4
#define CPAR_TAU_MIN     8      /* skip the chip-triangle zone       */
#define CPAR_PEAK_SEP    4      /* minimum peak separation, lags     */
#define CPAR_PEAK_FLOOR  2.0    /* candidate floor: r > 2 * median   */
#define CPAR_PAR_THRESH  25.0   /* significance threshold on PAR     */

static double median_of(const double *v, size_t n)
{
    if (n == 0) return 0.0;
    double *tmp = malloc(n * sizeof(double));
    if (tmp == NULL) return 0.0;
    memcpy(tmp, v, n * sizeof(double));
    for (size_t i = 1; i < n; i++) {
        double key = tmp[i];
        size_t j = i;
        while (j > 0 && tmp[j - 1] > key) {
            tmp[j] = tmp[j - 1];
            j--;
        }
        tmp[j] = key;
    }
    double m = tmp[n / 2];
    free(tmp);
    return m;
}

void cpar_clear(CparResult *r)
{
    if (r == NULL) return;
    free(r->lag_s);
    free(r->acf);
    free(r->par);
    free(r->is_peak);
    memset(r, 0, sizeof(*r));
}

bool cpar_analyze(const IqSource *src, const Region *region, CparResult *out)
{
    cpar_clear(out);
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
        aconfig_get_num("cpar", "fs_target_hz", CPAR_FS_TARGET));
    if (D < 1) D = 1;
    size_t nd = n_work / D;
    const double fs = src->sample_rate / (double)D;
    const size_t T = CPAR_WINDOW;
    const size_t tau_max = CPAR_MAX_LAGS;
    if (nd < T * CPAR_MIN_WINDOWS) {
        free(seg);
        return false;
    }
    float *dec = malloc(2 * nd * sizeof(float));
    if (dec == NULL) {
        free(seg);
        return false;
    }
    {
        size_t got = dsp_decimate_staged(seg, n_work, D, dec);
        nd = got < nd ? got : nd;
    }
    free(seg);

    size_t M = nd / T;
    if (M > CPAR_MAX_WINDOWS) M = CPAR_MAX_WINDOWS;
    if (M < CPAR_MIN_WINDOWS) {
        free(dec);
        return false;
    }

    /* windowed mean autocorrelation (FFT-based) */
    size_t L = tau_max + 1;
    double *Rr = calloc(L, sizeof(double));
    double *Ri = calloc(L, sizeof(double));
    if (Rr == NULL || Ri == NULL) {
        free(Rr); free(Ri);
        free(dec);
        return false;
    }
    {
        double *bre = malloc(L * sizeof(double));
        double *bim = malloc(L * sizeof(double));
        if (bre == NULL || bim == NULL) {
            free(bre); free(bim);
            free(Rr); free(Ri);
            free(dec);
            return false;
        }
        for (size_t m = 0; m < M; m++) {
            dsp_autocorr(dec + 2 * m * T, T, tau_max, bre, bim);
            for (size_t tau = 0; tau < L; tau++) {
                Rr[tau] += bre[tau];
                Ri[tau] += bim[tau];
            }
        }
        free(bre);
        free(bim);
    }
    free(dec);
    for (size_t tau = 0; tau < L; tau++) {
        Rr[tau] /= (double)M;
        Ri[tau] /= (double)M;
    }

    out->n_lags = (int)L;
    out->lag_s = malloc(L * sizeof(double));
    out->acf = malloc(L * sizeof(double));
    out->par = calloc(L, sizeof(double));
    out->is_peak = calloc(L, sizeof(int));
    if (out->lag_s == NULL || out->acf == NULL ||
        out->par == NULL || out->is_peak == NULL) {
        free(Rr); free(Ri);
        cpar_clear(out);
        return false;
    }

    double r0 = sqrt(Rr[0] * Rr[0] + Ri[0] * Ri[0]);
    if (r0 <= 0.0) {
        free(Rr); free(Ri);
        cpar_clear(out);
        return false;
    }
    for (size_t tau = 0; tau < L; tau++) {
        double m = sqrt(Rr[tau] * Rr[tau] + Ri[tau] * Ri[tau]);
        out->lag_s[tau] = (double)tau / fs;
        out->acf[tau] = m / r0;
    }
    free(Rr);
    free(Ri);

    /* chip period from the first floor crossing of the chip
     * correlation triangle: r(tau) ~ (a^2/power)(1 - tau/Tc) falls to
     * the noise floor at tau ~ Tc; the FIR corner-rounding shifts the
     * crossing slightly early, compensated by the divisor below */
    {
        const double cross = 0.02;
        for (size_t tau = CPAR_TAU_MIN; tau < L; tau++) {
            if (out->acf[tau] < cross) {
                out->chip_period_s = (double)tau / fs / 0.95;
                out->chip_rate_hz = 1.0 / out->chip_period_s;
                break;
            }
        }
    }

    /* candidate peaks: local maxima above 2x the median */
    {
        size_t n_lags_use = L - CPAR_TAU_MIN;
        double *vals = malloc(n_lags_use * sizeof(double));
        if (vals == NULL) {
            cpar_clear(out);
            return false;
        }
        for (size_t i = 0; i < n_lags_use; i++) {
            vals[i] = out->acf[CPAR_TAU_MIN + i];
        }
        double med = median_of(vals, n_lags_use);
        free(vals);
        double floor_val = med * CPAR_PEAK_FLOOR;

        int last_peak = -CPAR_PEAK_SEP - 1;
        for (size_t tau = CPAR_TAU_MIN; tau + 1 < L; tau++) {
            if (out->acf[tau] < floor_val) continue;
            if (out->acf[tau] <= out->acf[tau - 1] ||
                out->acf[tau] <= out->acf[tau + 1]) {
                continue;
            }
            if ((int)tau - last_peak < CPAR_PEAK_SEP) continue;
            out->is_peak[tau] = 1;
            last_peak = (int)tau;
        }
    }

    /* PAR per peak and the cumulative criteria */
    out->par_threshold = CPAR_PAR_THRESH;
    {
        double cpar_sig = 0.0, cpar_all = 0.0;
        int n_peaks = 0, n_sig = 0;
        double best_par = 0.0, best_lag = 0.0;

        for (size_t tau = CPAR_TAU_MIN; tau + 1 < L; tau++) {
            if (!out->is_peak[tau]) continue;
            n_peaks++;

            /* peak-to-average: peak power vs the mean squared ACF
             * over the whole lag range (the global noise baseline) */
            double sum = 0.0;
            int cnt = 0;
            for (size_t k = CPAR_TAU_MIN; k < L; k++) {
                if (k == tau) continue;
                sum += out->acf[k] * out->acf[k];
                cnt++;
            }
            double baseline = cnt > 0 ? sum / (double)cnt : 0.0;
            if (baseline < 1e-20) baseline = 1e-20;
            double pk = out->acf[tau];
            double par = (pk * pk) / baseline;
            out->par[tau] = par;
            cpar_all += par;
            if (par > CPAR_PAR_THRESH) {
                cpar_sig += par;
                n_sig++;
            }
            if (par > best_par) {
                best_par = par;
                best_lag = out->lag_s[tau];
            }
        }

        out->n_peaks = n_peaks;
        out->cpar_all = cpar_all;
        out->cpar = cpar_sig;
        out->peak_par = best_par;
        out->peak_lag = best_lag;
        out->detected = cpar_sig > 0.0;

        if (out->detected) {
            snprintf(out->note, sizeof(out->note),
                     "DSSS-like signal detected: CPAR %.1f over %d "
                     "significant peak(s); strongest PAR %.1f at lag "
                     "%.2f ms (code period), chip period ~%.1f us "
                     "(chip rate ~%.1f kHz)",
                     cpar_sig, n_sig, best_par, best_lag * 1e3,
                     out->chip_period_s * 1e6, out->chip_rate_hz / 1e3);
        } else {
            snprintf(out->note, sizeof(out->note),
                     "No significant cumulative peak-to-average ratio "
                     "(CPAR 0 over %d candidate peak(s), strongest PAR "
                     "%.1f vs threshold %.1f) - noise-like autocorrelation",
                     n_peaks, best_par, CPAR_PAR_THRESH);
        }
    }

    TE15_LOG("\nCUMULATIVE PEAK-TO-AVERAGE RATIO\n"
           "windows: %zu x %zu samples at %.0f Hz\n"
           "candidate peaks: %d, CPAR(all)=%.1f, CPAR(sig)=%.1f "
           "(threshold %.1f per peak)\n"
           "%s\n",
           M, T, fs, out->n_peaks, out->cpar_all, out->cpar,
           out->par_threshold, out->note);
    return true;
}
