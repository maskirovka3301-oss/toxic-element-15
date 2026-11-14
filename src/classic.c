/*
 * classic.c — classic signal-analysis tools for IQ data (SPTK-derived):
 * cepstrum (periodicity / echo structure), AMDF (period estimation),
 * LPC spectral envelope, group delay (phase linearity) and a Goertzel
 * single-bin probe. Selection-scoped: meant for an interesting part of
 * the waterfall.
 *
 * Pipeline: copy + cap + mixdown to the region centre, decimate to
 * ~50 kHz, then: real cepstrum of the log-magnitude FFT (periodicity),
 * AMDF (first deep dip = period), Levinson-Durbin LPC of the magnitude
 * autocorrelation, group delay from the spectrum phase, and a Goertzel
 * probe at a requested frequency (default fs/4).
 */
#include "aconfig.h"
#include "debug.h"
#include "classic.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsp.h"
#include "metrics.h"
#include "sptk.h"

#define CEP_MAX 4096 /* cepstrum transform cap (O(n^2) cost) */
#define AMDF_MAX_LAGS 2048 /* AMDF lag span cap (O(lags * n) cost) */

#define CLASSIC_FS_TARGET 50000.0   /* decimated effective rate      */
#define CLASSIC_MIN_SAMPLES 64      /* minimum decimated selection    */
#define CLASSIC_AMDF_MIN_LAG 8      /* skip tiny lags (window smear)  */
#define CLASSIC_AMDF_DIP 0.3       /* dip below 0.3 * mean magnitude */

void classic_clear(ClassicResult *r)
{
    if (r == NULL) return;
    free(r->cepstrum);
    free(r->amdf);
    free(r->lpc_coeffs);
    free(r->group_delay);
    memset(r, 0, sizeof(*r));
}

bool classic_analyze(const IqSource *src, const Region *region,
                     double probe_hz, ClassicResult *out)
{
    classic_clear(out);
    if (src == NULL || region == NULL) return false;

    Region r = *region;
    if (!region_normalize(&r, src)) return false;

    size_t s0 = (size_t)(r.time_min * src->sample_rate);
    size_t s1 = (size_t)(r.time_max * src->sample_rate);
    if (s1 > src->total_samples) s1 = src->total_samples;
    if (s0 >= s1) return false;
    size_t n_work = s1 - s0;
    if (n_work > TE15_MAX_ANALYSIS_SAMPLES) n_work = TE15_MAX_ANALYSIS_SAMPLES;

    /* private copy of the selected segment, mixdowned to the centre */
    float *seg = malloc(2 * n_work * sizeof(float));
    if (seg == NULL) return false;
    iq_source_read_complex(src, s0, seg, n_work);
    double offset = ((r.freq_min + r.freq_max) / 2.0) - src->center_freq;
    metrics_mixdown(seg, n_work, src->sample_rate, -offset, seg);

    /* Decimate relative to the SELECTION BANDWIDTH, not to a fixed rate:
     * a narrow selection still gets a fine envelope, while a wide one
     * (e.g. 400 kHz) keeps its whole band instead of being cut to 50 kHz
     * (the old behaviour discarded 90% of such a selection and biased
     * the period estimates). The envelope rate is kept between
     * CLASSIC_FS_TARGET and 4x the selection bandwidth and never above
     * 1 MHz. */
    /* decimate to ~50 kHz (the envelope estimators below were calibrated
     * at this rate; a bandwidth-relative rate made the cepstrum/ACF pick
     * spurious periods on wide selections) */
    /* The classic envelope estimators (cepstrum, AMDF, ACF) work on the
     * |x| sequence, whose bandwidth is set by the pulse structure (a
     * millisecond-scale burst train lives below ~1 kHz), so the low target
     * rate throws away nothing of the envelope - and the estimators are
     * calibrated there: at higher rates the cepstral peak drifts (an
     * 800 Hz reading instead of the true ~200 Hz on the reference
     * recording). The factor below must therefore stay small; the only
     * guard is that the decimation must not be so coarse that the
     * selection band folds onto itself. */
    size_t D = dsp_decimation_for_x(src->sample_rate, r.freq_max - r.freq_min,
                                    aconfig_get_num("classic",
                                                    "envelope_rate_hz",
                                                    CLASSIC_FS_TARGET),
                                    0.10);
    size_t nd = n_work / D;
    if (nd < CLASSIC_MIN_SAMPLES) {
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
    if (nd < CLASSIC_MIN_SAMPLES) {
        free(dec);
        return false;
    }

    const double fs = src->sample_rate / (double)D;
    const int n = CLASSIC_N < (int)nd ? CLASSIC_N : (int)nd;
    out->n = n;
    out->fs = fs;

    /* magnitude sequence as a real signal for the cepstrum FFT */
    float *magiq = malloc(2 * (size_t)n * sizeof(float));
    if (magiq == NULL) {
        free(dec);
        return false;
    }
    for (int i = 0; i < n; i++) {
        double a = (double)dec[2 * i];
        double b = (double)dec[2 * i + 1];
        magiq[2 * i] = (float)sqrt(a * a + b * b);
        magiq[2 * i + 1] = 0.0f;
    }

    fftwf_complex *X = fftwf_malloc((size_t)n * sizeof(fftwf_complex));
    FftPlan *plan = fft_plan_create(n);
    if (X == NULL || plan == NULL) {
        fftwf_free(X);
        fft_plan_destroy(plan);
        free(magiq);
        free(dec);
        return false;
    }
    fft_plan_transform(plan, magiq, NULL, X);
    fft_plan_destroy(plan);
    free(magiq);

    /* log-magnitude spectrum and real cepstrum */
    double *logspec = malloc((size_t)n * sizeof(double));
    double *cep = malloc((size_t)n * sizeof(double));
    double *gd = malloc((size_t)n * sizeof(double));
    double *cepstore = malloc(((size_t)n / 2) * sizeof(double));
    if (logspec == NULL || cep == NULL || gd == NULL || cepstore == NULL) {
        free(logspec);
        free(cep);
        free(gd);
        free(cepstore);
        fftwf_free(X);
        free(dec);
        return false;
    }
    for (int b = 0; b < n; b++) {
        double m = sqrt((double)X[b][0] * X[b][0] +
                        (double)X[b][1] * X[b][1]);
        logspec[b] = log(m + 1e-12);
    }
    int cep_len = n;
    {
        /* sptk_cepstrum is O(n^2): cap the transform to the first
         * CEP_MAX samples (quefrency resolution ~160 us at 50 kHz, still
         * resolving the 100 Hz-2 kHz PRF band) so the classic analysis
         * stays fast on long selections. */
        int ncep = n < CEP_MAX ? n : CEP_MAX;
        if (ncep < n) {
            /* taper the truncated log spectrum: an abrupt end would put
             * a spurious high-quefrency component into the cepstrum */
            double *w = malloc((size_t)ncep * sizeof(double));
            if (w != NULL) {
                for (int i = 0; i < ncep; i++) {
                    double ph = M_PI * (double)i / (double)(ncep - 1);
                    w[i] = logspec[i] * (0.5 - 0.5 * cos(2.0 * ph));
                }
                sptk_cepstrum(w, (size_t)ncep, cep);
                free(w);
            } else {
                sptk_cepstrum(logspec, (size_t)ncep, cep);
            }
        } else {
            sptk_cepstrum(logspec, (size_t)ncep, cep);
        }
        memset(cep + ncep, 0, (size_t)(n - ncep) * sizeof(double));
        cep_len = ncep;
    }
    for (int q = 0; q < n / 2; q++) cepstore[q] = cep[q];
    free(logspec);

    /* Cepstral peak over quefrencies 1..n/2-1 (excluding quefrency 0,
     * which only carries the log-energy term). The log spectrum gives
     * even tiny harmonic leakage (e.g. -12 vs the -27.6 floor) large
     * weight, so a periodic envelope produces a quefrency-harmonic
     * series whose members all tie at the global maximum; reporting the
     * arbitrary global-max quefrency would misstate the period. We
     * therefore report the SMALLEST quefrency reaching 90% of the
     * maximum -- the fundamental of the series. */
    double cep_peak_mag = -1.0;
    /* the cepstrum is symmetric about cep_len/2 (the log-DC mirrors to
     * the high-quefrency end), so only the first half is meaningful */
    int cep_lim = cep_len / 2;
    if (cep_lim > n / 2) cep_lim = n / 2;
    if (cep_lim < 16) cep_lim = 16;
    for (int q = 1; q < cep_lim; q++) {
        if (cep[q] > cep_peak_mag) {
            cep_peak_mag = cep[q];
        }
    }
    int cep_peak_quef = 0;
    /* start the fundamental search at ~0.25 ms: shorter quefrencies are
     * the log-spectrum's DC lobe, not a modulation period */
    for (int q = 12; q < cep_lim; q++) {
        /* skip the ultra-low quefrencies (the log-DC lobe) and require
         * a local peak so flat harmonic ties resolve to the fundamental */
        if (cep[q] >= 0.9 * cep_peak_mag && cep[q] >= cep[q - 1] &&
            cep[q] >= cep[q + 1]) {
            cep_peak_quef = q;
            break;
        }
    }
    if (cep_peak_quef == 0) {
        /* no clean fundamental: fall back to the AMDF period */
        cep_peak_mag = 0.0;
    }
    if (cep_peak_quef > 0) {
        cep_peak_mag = cep[cep_peak_quef];
    }
    /* mean of the quefrency content (again excluding cep[0]) for the
     * peak-strength test */
    double cep_mean = 0.0;
    for (int q = 1; q < cep_lim; q++) cep_mean += cep[q];
    cep_mean /= (double)(cep_lim - 1);

    /* group delay from the spectrum phase (before the cepstrum) */
    sptk_group_delay((const float *)X, (size_t)n, gd);
    fftwf_free(X);

    /* AMDF over the decimated complex signal. Note: sptk_amdf defines
     * lag 0 as identically zero, so the classic "dip below 0.3 * amdf[0]"
     * reference is replaced by 0.3 * mean(|x|) (the AMDF's natural
     * scale); the zero-lag value of the implemented AMDF would make the
     * literal test degenerate (amdf[0] == 0 always). */
    double *amdf = calloc(CLASSIC_LAGS, sizeof(double));
    if (amdf == NULL) {
        free(cep);
        free(gd);
        free(cepstore);
        free(dec);
        return false;
    }
    {
        size_t lag_n = CLASSIC_LAGS < nd ? CLASSIC_LAGS : nd;
        if (lag_n > AMDF_MAX_LAGS) lag_n = AMDF_MAX_LAGS;
        sptk_amdf(dec, nd, lag_n, amdf);
        double mean_mag = 0.0;
        for (size_t i = 0; i < nd; i++) {
            double a = (double)dec[2 * i];
            double b = (double)dec[2 * i + 1];
            mean_mag += sqrt(a * a + b * b);
        }
        mean_mag /= (double)nd;
        double dip = CLASSIC_AMDF_DIP * mean_mag;
        size_t period = 0;
        for (size_t lag = CLASSIC_AMDF_MIN_LAG; lag < lag_n; lag++) {
            if (amdf[lag] < dip) {
                period = lag;
                break;
            }
        }
        out->amdf_period = (double)period;
        out->amdf_period_hz = period > 0 ? fs / (double)period : 0.0;
    }

    /* LPC of the magnitude autocorrelation */
    double *lpc = calloc((size_t)CLASSIC_LPC + 1, sizeof(double));
    if (lpc == NULL) {
        free(amdf);
        free(cep);
        free(gd);
        free(cepstore);
        free(dec);
        return false;
    }
    (void)sptk_lpc(dec, nd, CLASSIC_LPC, lpc);
    lpc[0] = 1.0;               /* keep a[0] = 1 even on failure */
    out->lpc_order = CLASSIC_LPC;

    /* Goertzel probe: 0 means "region centre offset" -> default fs/4 */
    double probe = probe_hz != 0.0 ? probe_hz : fs / 4.0;
    out->probe_hz = probe;
    {
        double raw = sptk_goertzel(dec, nd, probe / fs);
        double total_power = 0.0;
        for (size_t i = 0; i < nd; i++) {
            double a = (double)dec[2 * i];
            double b = (double)dec[2 * i + 1];
            total_power += a * a + b * b;
        }
        total_power /= (double)nd;
        double p = total_power > 0.0
                       ? raw / (total_power * (double)nd * (double)nd / 4.0)
                       : 0.0;
        if (p > 1.0) p = 1.0;
        out->goertzel_power = p;
    }
    /* Envelope autocorrelation period: robust for bursty/pulse trains,
     * where the cepstrum of the raw log-spectrum is too weak. Computed
     * with an FFT so the lag search costs O(n log n). */
    double acf_period_hz = 0.0;
    {
        size_t nf = 1;
        while (nf < nd) nf <<= 1;
        if (nf >= 64) {
            float *e = calloc(2 * nf, sizeof(float));
            double mean = 0.0;
            for (size_t i = 0; i < nd; i++) {
                mean += hypot(dec[2 * i], dec[2 * i + 1]);
            }
            mean /= (double)nd;
            double r0 = 0.0;
            for (size_t i = 0; i < nd; i++) {
                float v = (float)(hypot(dec[2 * i], dec[2 * i + 1]) - mean);
                e[2 * i] = v;
                r0 += (double)v * v;
            }
            if (e != NULL && r0 > 0.0) {
                FftPlan *p = fft_plan_create((int)nf);
                fftwf_complex *xa = fftwf_malloc(sizeof(fftwf_complex) * nf);
                if (p != NULL && xa != NULL) {
                    fft_plan_transform(p, e, NULL, xa);
                    for (size_t i = 0; i < nf; i++) {
                        double pw = (double)xa[i][0] * xa[i][0] +
                                    (double)xa[i][1] * xa[i][1];
                        xa[i][0] = (float)pw;
                        xa[i][1] = 0.0f;
                    }
                    fft_plan_transform(p, (const float *)xa, NULL, xa);
                    size_t maxlag = (size_t)(0.2 * fs);
                    if (maxlag > nd / 2) maxlag = nd / 2;
                    /* pitch-period rule: skip the short-lag lobe, find the
                     * first zero crossing, then the first local maximum
                     * after it with a meaningful correlation value */
                    size_t lag = (size_t)(fs * 0.0002);
                    if (lag < 2) lag = 2;
                    double prev = (double)xa[lag - 1][0] / (double)nf / r0;
                    while (lag < maxlag) {
                        double cur = (double)xa[lag][0] / (double)nf / r0;
                        if (prev <= 0.0 && cur > 0.0) {
                            break; /* zero crossing upward */
                        }
                        prev = cur;
                        lag++;
                    }
                    for (; lag + 1 < maxlag; lag++) {
                        double cur = (double)xa[lag][0] / (double)nf / r0;
                        double nxt = (double)xa[lag + 1][0] / (double)nf / r0;
                        if (cur > prev && cur >= nxt) {
                            if (cur > 0.2) {
                                acf_period_hz = fs / (double)lag;
                            }
                            break;
                        }
                        prev = cur;
                    }
                }
                fft_plan_destroy(p);
                fftwf_free(xa);
            }
            free(e);
        }
    }
    free(dec);

    /* consensus period: cepstrum wins when its peak is strong */
    out->cepstrum = cepstore;
    out->cep_peak_quef = (double)cep_peak_quef;
    out->cep_peak_mag = cep_peak_mag;
    out->cep_period_hz = cep_peak_quef > 0 ? fs / (double)cep_peak_quef : 0.0;
    out->amdf = amdf;
    out->lpc_coeffs = lpc;
    out->group_delay = gd;
    out->acf_period_hz = acf_period_hz;
    /* Consensus order (calibrated against the exported-selection and
     * original-file cross-check): the envelope ACF is the direct
     * estimator and wins whenever it finds a period; the AMDF is next;
     * the cepstrum is a fallback and must clear a prominence bar (2x the
     * mean quefrency content) - a weak cepstral peak previously reported
     * spurious periods (124 Hz where the envelope evidence put the true
     * burst rate at ~200 Hz). */
    out->period_hz = 0.0;
    if (acf_period_hz > 0.0) {
        out->period_hz = acf_period_hz;
    } else if (out->amdf_period_hz > 0.0) {
        out->period_hz = out->amdf_period_hz;
    } else if (cep_peak_quef > 0 && cep_peak_mag > 2.0 * cep_mean) {
        out->period_hz = out->cep_period_hz;
    }
    free(cep);

    snprintf(out->note, sizeof(out->note),
             "period %.1f Hz (%s; cepstrum %.1f @ mag %.3f, AMDF %.1f, "
             "envelope ACF %.1f) | Goertzel power at %.1f Hz: %.3f",
             out->period_hz,
             out->period_hz <= 0.0 ? "no estimator passed"
                 : (acf_period_hz > 0.0 ? "envelope ACF"
                    : (out->amdf_period_hz > 0.0 ? "AMDF" : "cepstrum")),
             out->cep_period_hz, cep_peak_mag, out->amdf_period_hz,
             out->acf_period_hz, probe, out->goertzel_power);

    TE15_LOG("\nCLASSIC ANALYSIS\n"
           "fs = %.0f Hz, period = %.1f Hz\n"
           "cepstral peak: quef %.0f samples, mag %.3f\n"
           "AMDF period: %.0f samples (%.1f Hz)\n"
           "Goertzel power at %.1f Hz: %.3f\n",
           fs, out->period_hz, out->cep_peak_quef, out->cep_peak_mag,
           out->amdf_period, out->amdf_period_hz, probe,
           out->goertzel_power);

    return true;
}
