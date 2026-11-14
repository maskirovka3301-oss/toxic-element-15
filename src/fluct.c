/*
 * fluct.c — fluctuation-of-autocorrelation DSSS detector (see header).
 *
 * Pipeline (Burel et al., ISPACS 2000):
 *   1. Copy the selection, mix down to the region centre, and FIR
 *      decimate to ~100 kHz effective rate.
 *   2. Split into M windows of T samples. For each window and each lag
 *      tau compute R_m(tau) = (1/T) sum x[n] x*[n-tau].
 *   3. Per lag: mean Rbar(tau) and variance V(tau) over the windows.
 *   4. Normalised fluctuation F(tau) = T * V(tau) / sigma^4
 *      (sigma^2 = measured power). Noise only: F(tau) ~ chi^2, mean 1,
 *      std 1/sqrt(M); detection threshold = 1 + 4/sqrt(M).
 *   5. The lag maximising F (outside the near-zero-lag bias zone) gives
 *      the chip period; the chip-SNR is recovered from the excess
 *      F(peak) - 1 ~ (chip SNR)^2 * chips_per_window.
 */
#include "fluct.h"
#include "debug.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsp.h"
#include "aconfig.h"
#include "metrics.h"
#include "region.h"

#define FLUCT_FS_TARGET 100000.0 /* effective rate after decimation */
#define FLUCT_WINDOW    512      /* samples per autocorrelation window */
#define FLUCT_MAX_WINDOWS 128
#define FLUCT_MIN_WINDOWS 8
#define FLUCT_TAU_MIN   4        /* skip the near-zero-lag bias zone  */

void fluct_clear(FluctResult *r)
{
    if (r == NULL) return;
    free(r->lag_s);
    free(r->mean_acf);
    free(r->fluct);
    memset(r, 0, sizeof(*r));
}

bool fluct_analyze(const IqSource *src, const Region *region, FluctResult *out)
{
    fluct_clear(out);
    if (src == NULL || region == NULL) return false;

    Region r = *region;
    if (!region_normalize(&r, src)) return false;

    size_t s0 = (size_t)(r.time_min * src->sample_rate);
    size_t s1 = (size_t)(r.time_max * src->sample_rate);
    if (s1 > src->total_samples) s1 = src->total_samples;
    if (s0 >= s1) return false;
    size_t n_work = s1 - s0;
    if (n_work > TE15_MAX_ANALYSIS_SAMPLES) n_work = TE15_MAX_ANALYSIS_SAMPLES;

    /* private copy of the segment, mixdowned to the centre */
    float *seg = malloc(2 * n_work * sizeof(float));
    if (seg == NULL) return false;
    iq_source_read_complex(src, s0, seg, n_work);
    double offset = ((r.freq_min + r.freq_max) / 2.0) - src->center_freq;
    metrics_mixdown(seg, n_work, src->sample_rate, -offset, seg);

    /* decimate to ~FLUCT_FS_TARGET (keeps chip rates up to ~25 kHz) */
    /* The Burel fluctuation criterion is calibrated for its fixed low
     * rate (the per-window statistics and the F(tau) expectation depend
     * on it); a higher rate measurably shifts the detected lag. It
     * therefore decimates to FLUCT_FS_TARGET, raised only by the
     * selection bandwidth to avoid aliasing. */
    size_t D = dsp_decimation_for_x(src->sample_rate,
                                    r.freq_max - r.freq_min,
                                    aconfig_get_num("fluct",
                                                    "fs_target_hz",
                                                    FLUCT_FS_TARGET),
                                    1.0);
    if (D < 1) D = 1;
    size_t nd = n_work / D;
    const double fs = src->sample_rate / (double)D;
    const size_t T = FLUCT_WINDOW;
    const size_t tau_max = FLUCT_MAX_LAGS;
    if (nd < T * FLUCT_MIN_WINDOWS) {
        free(seg);
        return false;
    }
    float *dec = malloc(2 * nd * sizeof(float));
    if (dec == NULL) {
        free(seg);
        return false;
    }
    {
        size_t got = dsp_fir_decimate(seg, n_work, D, dec);
        nd = got < nd ? got : nd;
    }
    free(seg);

    size_t M = nd / T; /* windows */
    if (M > FLUCT_MAX_WINDOWS) M = FLUCT_MAX_WINDOWS;
    if (M < FLUCT_MIN_WINDOWS) {
        free(dec);
        return false;
    }

    /* measured power (noise + signal) */
    double sigma2 = 0.0;
    for (size_t i = 0; i < (size_t)T * M; i++) {
        sigma2 += (double)dec[2 * i] * dec[2 * i] +
                  (double)dec[2 * i + 1] * dec[2 * i + 1];
    }
    sigma2 /= (double)((size_t)T * M);
    if (sigma2 <= 0.0) {
        free(dec);
        return false;
    }
    const double sigma4 = sigma2 * sigma2;

    /* per-lag accumulators over windows */
    size_t L = tau_max + 1;
    double *sum_re = calloc(L, sizeof(double));
    double *sum_im = calloc(L, sizeof(double));
    double *sum_abs2 = calloc(L, sizeof(double));
    if (sum_re == NULL || sum_im == NULL || sum_abs2 == NULL) {
        free(sum_re); free(sum_im); free(sum_abs2);
        free(dec);
        return false;
    }

    for (size_t m = 0; m < M; m++) {
        const float *x = dec + 2 * m * T;
        for (size_t tau = 1; tau <= tau_max; tau++) {
            double re = 0.0, im = 0.0;
            for (size_t n = tau; n < T; n++) {
                double a = x[2 * n], b = x[2 * n + 1];
                double c = x[2 * (n - tau)], d = x[2 * (n - tau) + 1];
                re += a * c + b * d;
                im += b * c - a * d;
            }
            re /= (double)T;
            im /= (double)T;
            sum_re[tau] += re;
            sum_im[tau] += im;
            sum_abs2[tau] += re * re + im * im;
        }
    }

    out->n_lags = (int)L;
    out->lag_s = malloc(L * sizeof(double));
    out->mean_acf = malloc(L * sizeof(double));
    out->fluct = malloc(L * sizeof(double));
    if (out->lag_s == NULL || out->mean_acf == NULL || out->fluct == NULL) {
        free(sum_re); free(sum_im); free(sum_abs2);
        free(dec);
        fluct_clear(out);
        return false;
    }

    /* mean and variance per lag; F(tau) = T * V / sigma^4 */
    double peak_fluct = 0.0, peak_lag = 0.0, peak_acf = 0.0;
    for (size_t tau = 0; tau < L; tau++) {
        double rm = sum_re[tau] / (double)M;
        double im = sum_im[tau] / (double)M;
        double var = sum_abs2[tau] / (double)M - (rm * rm + im * im);
        if (var < 0.0) var = 0.0;

        double acf = sqrt(rm * rm + im * im) / sigma2;
        double f = (double)T * var / sigma4;

        out->lag_s[tau] = (double)tau / fs;
        out->mean_acf[tau] = acf;
        out->fluct[tau] = f;

        if (tau >= FLUCT_TAU_MIN && f > peak_fluct) {
            peak_fluct = f;
            peak_lag = out->lag_s[tau];
            peak_acf = acf;
        }
    }
    free(sum_re);
    free(sum_im);
    free(sum_abs2);
    free(dec);

    out->noise_power = sigma2;
    /* noise-only F follows ~chi^2 with mean 1 and std 1/sqrt(M); the
     * maximum over the 256 (correlated) lags empirically sits higher,
     * so a floor of 2.0 keeps the false-alarm rate negligible while
     * leaving the DSSS excess (typically >= 3 at -10 dB chip SNR)
     * well above it */
    {
        double t = 1.0 + 4.0 / sqrt((double)M);
        out->threshold = t > 2.0 ? t : 2.0;
    }
    out->peak_lag = peak_lag;
    out->peak_fluct = peak_fluct;
    out->peak_acf = peak_acf;
    out->detected = peak_fluct > out->threshold;

    if (out->detected && peak_lag > 0.0) {
        out->chip_rate_hz = 1.0 / peak_lag;
        /* F(peak) - 1 ~ (chip SNR)^2 * chips_per_window */
        double chips_per_window = (double)T / (peak_lag * fs);
        double snr2 = (peak_fluct - 1.0) / (chips_per_window > 0.0
                                               ? chips_per_window : 1.0);
        out->snr_db = snr2 > 0.0 ? 5.0 * log10(snr2) : -99.0;
        snprintf(out->note, sizeof(out->note),
                 "Spread-spectrum-like signal detected: fluctuation "
                 "peak %.1f at lag %.1f us (chip rate ~%.1f kHz, "
                 "chip SNR ~%.1f dB)",
                 peak_fluct, peak_lag * 1e6, out->chip_rate_hz / 1e3,
                 out->snr_db);
    } else {
        out->chip_rate_hz = 0.0;
        out->snr_db = -99.0;
        snprintf(out->note, sizeof(out->note),
                 "No significant fluctuation of the autocorrelation "
                 "estimator (peak %.1f vs threshold %.1f) — no hidden "
                 "spread-spectrum signal detected",
                 peak_fluct, out->threshold);
    }

    TE15_LOG("\nFLUCTUATIONS OF AUTOCORRELATION ESTIMATORS\n"
           "windows: %zu x %zu samples at %.0f Hz\n"
           "fluctuation peak: %.1f at %.1f us (threshold %.1f)\n"
           "mean-ACF peak at that lag: %.3f (normalised)\n"
           "%s\n",
           M, T, fs, peak_fluct, peak_lag * 1e6, out->threshold,
           peak_acf, out->note);
    return true;
}
