/*
 * metrics.c — extended signal-analysis suite and demodulators.
 *
 * Feature battery (all measured from the selected region):
 *   spectral : occupied (99%) bandwidth, flatness, symmetry, tone purity,
 *              SNR (tone vs median floor), Welch-averaged PSD
 *   time     : envelope AM depth, crest factor (PAPR), OOK duty cycle,
 *              instantaneous-frequency FM deviation (robust MAD),
 *              FSK level count, symbol/envelope rate, chirp trend
 *   class    : rule-based modulation hypothesis + demod recommendation
 *
 * The decision tree is deliberately simple and threshold-driven; the
 * thresholds are validated by the unit tests in tests/test_runner.c.
 */
#include "metrics.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsp.h"
#include "region.h"

#define METRICS_FFT           2048
#define METRICS_MAX_PSD_FRAMES 2048
#define METRICS_MAX_DEC_SAMPLES 262144
#define METRICS_MIN_SAMPLES   1024
#define DEMOD_OUT_RATE        32000.0
#define CW_BEAT_HZ            700.0

/* ------------------------------------------------------------------ */
/* small numeric helpers                                               */
/* ------------------------------------------------------------------ */

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* Percentile of v[0..n) at fraction q in [0,1]; sorts a copy. */
static double percentile(const float *v, size_t n, double q)
{
    if (n == 0) return 0.0;
    double *tmp = malloc(n * sizeof(double));
    if (tmp == NULL) return v[0];
    for (size_t i = 0; i < n; i++) tmp[i] = v[i];
    qsort(tmp, n, sizeof(double), cmp_double);
    size_t idx = (size_t)(q * (double)(n - 1) + 0.5);
    if (idx >= n) idx = n - 1;
    double r = tmp[idx];
    free(tmp);
    return r;
}

static double median_of(const float *v, size_t n)
{
    return percentile(v, n, 0.5);
}

/* ------------------------------------------------------------------ */
/* mixdown                                                             */
/* ------------------------------------------------------------------ */

void metrics_mixdown(const float *iq, size_t n, double src_rate,
                     double shift_hz, float *out)
{
    if (iq == NULL || out == NULL || n == 0) return;
    if (out != iq) memcpy(out, iq, 2 * n * sizeof(float));

    double dphi = 2.0 * M_PI * shift_hz / src_rate;
    double phi = 0.0;
    for (size_t i = 0; i < n; i++) {
        double c = cos(phi), s = sin(phi);
        float re = out[2 * i], im = out[2 * i + 1];
        out[2 * i]     = (float)(re * c - im * s);
        out[2 * i + 1] = (float)(re * s + im * c);
        phi += dphi;
        if (phi > 2.0 * M_PI) phi -= 2.0 * M_PI;
    }
}

/* ------------------------------------------------------------------ */
/* decimation (box filter)                                             */
/* ------------------------------------------------------------------ */

static size_t box_decimate(const float *in, size_t n, size_t K,
                           float *out, size_t max_out)
{
    size_t out_n = 0;
    if (K < 1) K = 1;
    for (size_t i = 0; i + K <= n && out_n < max_out; i += K) {
        double acc = 0.0;
        for (size_t k = 0; k < K; k++) acc += in[i + k];
        out[out_n++] = (float)(acc / (double)K);
    }
    return out_n;
}

/* ------------------------------------------------------------------ */
/* demodulators (input already mixdowned to DC)                        */
/* ------------------------------------------------------------------ */

size_t demod_fm(const float *iq, size_t n, double src_rate, double out_rate,
                double deviation_hz, float *out, size_t max_out)
{
    if (iq == NULL || out == NULL || n < 2 || max_out == 0) return 0;

    float *freq = malloc(n * sizeof(float));
    if (freq == NULL) return 0;

    const double dt = 1.0 / src_rate;
    const double scale = 1.0 / (2.0 * M_PI * dt); /* Hz per radian */
    double mean = 0.0;
    freq[0] = 0.0f;
    for (size_t i = 1; i < n; i++) {
        double re = (double)iq[2 * i] * iq[2 * i - 2] +
                    (double)iq[2 * i + 1] * iq[2 * i - 1];
        double im = (double)iq[2 * i + 1] * iq[2 * i - 2] -
                    (double)iq[2 * i] * iq[2 * i - 1];
        freq[i] = (float)(atan2(im, re) * scale);
        mean += freq[i];
    }
    mean /= (double)(n - 1);
    double dev = deviation_hz > 0.0 ? deviation_hz : 1.0;

    size_t K = (size_t)(src_rate / out_rate + 0.5);
    if (K < 1) K = 1;
    size_t out_n = 0;
    for (size_t i = 0; i + K <= n && out_n < max_out; i += K) {
        double acc = 0.0;
        for (size_t k = 0; k < K; k++) acc += freq[i + k] - mean;
        double v = acc / ((double)K * dev); /* dimensionless, ~+-1 */
        if (v > 1.5) v = 1.5;
        if (v < -1.5) v = -1.5;
        out[out_n++] = (float)(0.6 * v);
    }
    free(freq);
    return out_n;
}

size_t demod_am(const float *iq, size_t n, double src_rate, double out_rate,
                float *out, size_t max_out)
{
    if (iq == NULL || out == NULL || n == 0 || max_out == 0) return 0;

    float *env = malloc(n * sizeof(float));
    if (env == NULL) return 0;
    double mean = 0.0;
    for (size_t i = 0; i < n; i++) {
        env[i] = sqrtf(iq[2 * i] * iq[2 * i] + iq[2 * i + 1] * iq[2 * i + 1]);
        mean += env[i];
    }
    mean /= (double)n;
    double peak = 0.0;
    for (size_t i = 0; i < n; i++) {
        env[i] = (float)((double)env[i] - mean);
        double a = fabs(env[i]);
        if (a > peak) peak = a;
    }
    if (peak < 1e-9) {
        free(env);
        return 0;
    }
    for (size_t i = 0; i < n; i++) {
        double v = env[i] / peak;
        if (v > 1.0) v = 1.0;
        if (v < -1.0) v = -1.0;
        env[i] = (float)v;
    }

    size_t K = (size_t)(src_rate / out_rate + 0.5);
    if (K < 1) K = 1;
    size_t out_n = box_decimate(env, n, K, out, max_out);
    free(env);
    return out_n;
}

size_t demod_cw(const float *iq, size_t n, double src_rate, double out_rate,
                float *out, size_t max_out)
{
    if (iq == NULL || out == NULL || n == 0 || max_out == 0) return 0;

    float *beat = malloc(n * sizeof(float));
    if (beat == NULL) return 0;

    /* Beat the DC carrier up to an audible tone; keying rides on the
     * envelope of the real part. */
    double dphi = 2.0 * M_PI * CW_BEAT_HZ / src_rate;
    double phi = 0.0, mean = 0.0;
    for (size_t i = 0; i < n; i++) {
        beat[i] = (float)(iq[2 * i] * cos(phi) - iq[2 * i + 1] * sin(phi));
        mean += beat[i];
        phi += dphi;
        if (phi > 2.0 * M_PI) phi -= 2.0 * M_PI;
    }
    mean /= (double)n;
    double peak = 0.0;
    for (size_t i = 0; i < n; i++) {
        beat[i] = (float)((double)beat[i] - mean);
        double a = fabs(beat[i]);
        if (a > peak) peak = a;
    }
    if (peak < 1e-9) {
        free(beat);
        return 0;
    }
    for (size_t i = 0; i < n; i++) beat[i] = (float)(beat[i] / peak);

    size_t K = (size_t)(src_rate / out_rate + 0.5);
    if (K < 1) K = 1;
    size_t out_n = box_decimate(beat, n, K, out, max_out);
    free(beat);
    return out_n;
}

/* ------------------------------------------------------------------ */
/* feature computation                                                 */
/* ------------------------------------------------------------------ */

static void classify(Metrics *m, double region_bw)
{
    double bw = m->occupied_bw_hz;
    if (bw <= 0.0) bw = region_bw;

    m->demodulable = false;
    m->demod_mode[0] = '\0';

    if (m->tone_purity > 0.3 && m->snr_db > 12.0 && bw < 3000.0 &&
        m->sym_rate_conf < 0.3) {
        strcpy(m->mod_class, "CW");
        strcpy(m->demod_mode, "cw");
        m->demodulable = true;
        snprintf(m->demod_note, sizeof(m->demod_note),
                 "Narrow continuous tone — beat-note audio demodulation");
    } else if (m->chirp_trend > 0.6 && m->fm_deviation_hz > 0.2 * bw) {
        strcpy(m->mod_class, "CHIRP");
        snprintf(m->demod_note, sizeof(m->demod_note),
                 "Linear frequency sweep (radar/lidar style) — no audio demod");
    } else if (m->fsk_levels >= 2) {
        strcpy(m->mod_class, "FSK");
        snprintf(m->demod_note, sizeof(m->demod_note),
                 "%d-level frequency shift keying — digital, no audio demod",
                 m->fsk_levels);
    } else if (m->fm_deviation_hz > 0.10 * bw && m->am_depth < 0.30) {
        strcpy(m->mod_class, "FM");
        strcpy(m->demod_mode, "fm");
        m->demodulable = true;
        snprintf(m->demod_note, sizeof(m->demod_note),
                 "Frequency modulation, deviation ~%.1f kHz",
                 m->fm_deviation_hz / 1e3);
    } else if (m->am_depth >= 0.30 && m->duty_cycle > 0.85 &&
               fabs(m->band_symmetry) < 0.45 &&
               m->fm_deviation_hz < 0.3 * bw) {
        strcpy(m->mod_class, "AM");
        strcpy(m->demod_mode, "am");
        m->demodulable = true;
        snprintf(m->demod_note, sizeof(m->demod_note),
                 "Amplitude modulation, depth ~%.0f%%", m->am_depth * 100.0);
    } else if (m->duty_cycle > 0.02 && m->duty_cycle < 0.92 &&
               m->am_depth > 0.60 && m->spectral_flatness < 0.85) {
        /* Strongly pulsed envelope (bursts above the noise floor) with a
         * structured spectrum - a burst/pulse train. The in-burst
         * modulation may well be wideband; the Pulse Trains tab
         * resolves the burst timing (tau, PRF, jitter). Pure noise is
         * excluded by the spectral-flatness bound. */
        strcpy(m->mod_class, "OOK");
        snprintf(m->demod_note, sizeof(m->demod_note),
                 "Pulsed emission (duty %.1f%%, envelope depth %.0f%%) - "
                 "bursty train; check the Pulse Trains tab for tau/PRF",
                 m->duty_cycle * 100.0, m->am_depth * 100.0);
    } else if (m->papr_db > 7.0 && m->spectral_flatness > 0.35 &&
               m->am_depth < 0.55) {
        strcpy(m->mod_class, "OFDM");
        snprintf(m->demod_note, sizeof(m->demod_note),
                 "High crest factor + flat spectrum — multi-carrier");
    } else if (m->spectral_flatness > 0.85 && m->snr_db < 8.0 &&
               m->fm_deviation_hz > 0.5 * bw) {
        strcpy(m->mod_class, "NOISE");
        snprintf(m->demod_note, sizeof(m->demod_note),
                 "No detectable structure");
    } else if (m->am_depth < 0.20 && m->papr_db < 3.5 &&
               m->spectral_flatness > 0.30 && m->fm_deviation_hz < 0.2 * bw) {
        strcpy(m->mod_class, "PSK");
        snprintf(m->demod_note, sizeof(m->demod_note),
                 "Constant envelope, phase-modulated — digital, no audio demod");
    } else {
        strcpy(m->mod_class, "NOISE");
        snprintf(m->demod_note, sizeof(m->demod_note),
                 "Unclassifiable structure");
    }
}

bool metrics_compute(const IqSource *src, const Region *region, Metrics *m)
{
    memset(m, 0, sizeof(*m));
    if (src == NULL || region == NULL) return false;

    Region r = *region;
    if (!region_normalize(&r, src)) return false;

    size_t s0 = (size_t)(r.time_min * src->sample_rate);
    size_t s1 = (size_t)(r.time_max * src->sample_rate);
    if (s1 > src->total_samples) s1 = src->total_samples;
    if (s0 >= s1 || s1 - s0 < METRICS_MIN_SAMPLES) return false;

    const size_t n = s1 - s0;
    const double rate = src->sample_rate;
    const double region_bw = r.freq_max - r.freq_min;
    const double center_rf = (r.freq_min + r.freq_max) / 2.0;
    const double offset = center_rf - src->center_freq; /* baseband */

    /*
     * Copy the selected part of the recording out of the memory map
     * into a private buffer; all measurements below run on this buffer.
     */
    size_t n_work = n;
    if (n_work > TE15_MAX_ANALYSIS_SAMPLES) {
        fprintf(stderr,
                "metrics: selection is %.1f s long; truncating to the "
                "first %.1f s\n",
                (double)n_work / rate,
                (double)TE15_MAX_ANALYSIS_SAMPLES / rate);
        n_work = TE15_MAX_ANALYSIS_SAMPLES;
    }
    float *segment = malloc(2 * n_work * sizeof(float));
    if (segment == NULL) {
        fprintf(stderr, "metrics: out of memory for the segment buffer\n");
        return false;
    }
    iq_source_read_complex(src, s0, segment, n_work);

    /* ---------------- spectral (Welch PSD over the full segment) ---- */
    const int fft = METRICS_FFT;
    FftPlan *plan = fft_plan_create(fft);
    float *win = malloc((size_t)fft * sizeof(float));
    float *db = malloc((size_t)fft * sizeof(float));
    float *frame_mix = malloc(2 * (size_t)fft * sizeof(float));
    double *psd = calloc((size_t)fft, sizeof(double));
    if (plan == NULL || win == NULL || db == NULL ||
        frame_mix == NULL || psd == NULL) {
        fft_plan_destroy(plan);
        free(win); free(db); free(frame_mix); free(psd);
        free(segment);
        return false;
    }
    dsp_hann(win, (size_t)fft);

    size_t hop = (size_t)fft / 2;
    size_t stride = hop;
    if (n_work > (size_t)fft) {
        size_t frames_full = (n_work - (size_t)fft) / hop + 1;
        if (frames_full > METRICS_MAX_PSD_FRAMES) {
            stride = hop * ((frames_full + METRICS_MAX_PSD_FRAMES - 1) /
                            METRICS_MAX_PSD_FRAMES);
        }
    }
    size_t frame_count = 0;
    for (size_t st = 0; st + (size_t)fft <= n_work; st += stride) {
        metrics_mixdown(segment + 2 * st, (size_t)fft, rate,
                        -offset, frame_mix);
        fft_plan_spectrum(plan, frame_mix, win, db);
        for (int b = 0; b < fft; b++) {
            psd[b] += pow(10.0, db[b] / 10.0);
        }
        frame_count++;
    }
    if (frame_count == 0) {
        fft_plan_destroy(plan);
        free(win); free(db); free(frame_mix); free(psd);
        free(segment);
        return false;
    }
    for (int b = 0; b < fft; b++) psd[b] /= (double)frame_count;

    const double bin_hz = rate / (double)fft;
    int b_lo = (int)((-region_bw / 2.0) / bin_hz) + fft / 2;
    int b_hi = (int)((region_bw / 2.0) / bin_hz) + fft / 2;
    if (b_lo < 0) b_lo = 0;
    if (b_hi > fft) b_hi = fft;
    if (b_hi - b_lo < 2) { b_lo = 0; b_hi = fft; }

    double total = 0.0;
    for (int b = b_lo; b < b_hi; b++) total += psd[b];
    if (total <= 0.0) {
        fft_plan_destroy(plan);
        free(win); free(db); free(frame_mix); free(psd);
        free(segment);
        return false;
    }

    /* occupied bandwidth: 99% of in-band power */
    double cum = 0.0;
    int occ_lo = b_lo, occ_hi = b_hi - 1;
    int found_lo = 0;
    for (int b = b_lo; b < b_hi; b++) {
        cum += psd[b];
        if (!found_lo && cum >= 0.005 * total) { occ_lo = b; found_lo = 1; }
        if (cum >= 0.995 * total) { occ_hi = b; break; }
    }
    if (occ_hi <= occ_lo) occ_hi = occ_lo + 1;
    m->occupied_lo_hz = center_rf + (occ_lo - fft / 2) * bin_hz;
    m->occupied_hi_hz = center_rf + (occ_hi - fft / 2) * bin_hz;
    m->occupied_bw_hz = m->occupied_hi_hz - m->occupied_lo_hz;
    if (m->occupied_bw_hz < bin_hz) m->occupied_bw_hz = bin_hz;

    /* flatness, symmetry, purity, snr over the occupied range */
    {
        double log_sum = 0.0, arith_sum = 0.0;
        int cnt = occ_hi - occ_lo + 1;
        double mx = -1.0;
        for (int b = occ_lo; b <= occ_hi; b++) {
            double p = psd[b] + 1e-30;
            log_sum += log(p);
            arith_sum += p;
            if (p > mx) mx = p;
        }
        double geo = exp(log_sum / (double)cnt);
        m->spectral_flatness = geo / (arith_sum / (double)cnt);
        if (m->spectral_flatness > 1.0) m->spectral_flatness = 1.0;
        m->tone_purity = mx / (arith_sum / (double)cnt) / (double)cnt;

        /* median over the FULL region band is the noise floor (the
         * occupied range alone is too narrow to contain a median) */
        double *sorted = malloc(((size_t)(b_hi - b_lo)) * sizeof(double));
        double med = 0.0;
        if (sorted != NULL) {
            size_t k = 0;
            for (int b = b_lo; b < b_hi; b++) sorted[k++] = psd[b];
            qsort(sorted, k, sizeof(double), cmp_double);
            med = sorted[k / 2];
            free(sorted);
        }
        m->snr_db = med > 0.0 ? 10.0 * log10(mx / med) : 99.0;

        /* symmetry of the sidebands: the DC bin (the region centre,
         * where an AM/FM carrier sits) is excluded from both halves so
         * a centred carrier cannot skew the metric */
        double lo_p = 0.0, hi_p = 0.0;
        for (int b = occ_lo; b <= occ_hi; b++) {
            if (b < fft / 2 - 1) lo_p += psd[b];
            else if (b > fft / 2 + 1) hi_p += psd[b];
        }
        m->band_symmetry = (lo_p + hi_p > 0.0)
                               ? (hi_p - lo_p) / (lo_p + hi_p)
                               : 0.0;
    }

    fft_plan_destroy(plan);
    free(win); free(db); free(frame_mix); free(psd);

    /* ---------------- time-domain on a decimated stream ------------ */
    size_t D = (n_work + METRICS_MAX_DEC_SAMPLES - 1) / METRICS_MAX_DEC_SAMPLES;
    if (D < 1) D = 1;
    size_t nd = n_work / D;
    if (nd < 64) {
        free(segment);
        return false;
    }
    const double drate = rate / (double)D;

    float *mix = malloc(2 * nd * sizeof(float));
    float *env = malloc(nd * sizeof(float));
    float *ifreq = malloc(nd * sizeof(float));
    if (mix == NULL || env == NULL || ifreq == NULL) {
        free(mix); free(env); free(ifreq);
        free(segment);
        return false;
    }

    /* decimated mixdown to the region centre */
    {
        double dphi = -2.0 * M_PI * offset * (double)D / rate;
        double phi = 0.0;
        for (size_t i = 0; i < nd; i++) {
            double c = cos(phi), s = sin(phi);
            float re = segment[2 * (i * D)];
            float im = segment[2 * (i * D) + 1];
            mix[2 * i]     = (float)(re * c - im * s);
            mix[2 * i + 1] = (float)(re * s + im * c);
            phi += dphi;
            if (phi > 2.0 * M_PI) phi -= 2.0 * M_PI;
        }
    }

    double env_sum2 = 0.0;
    for (size_t i = 0; i < nd; i++) {
        env[i] = sqrtf(mix[2 * i] * mix[2 * i] + mix[2 * i + 1] * mix[2 * i + 1]);
        env_sum2 += (double)env[i] * env[i];
    }
    const double env_rms = sqrt(env_sum2 / (double)nd);

    /* AM depth from robust percentiles */
    double p05 = percentile(env, nd, 0.05);
    double p95 = percentile(env, nd, 0.95);
    m->am_depth = (p95 + p05 > 1e-12) ? (p95 - p05) / (p95 + p05) : 0.0;

    /* crest factor of the envelope */
    double env_max = 0.0;
    for (size_t i = 0; i < nd; i++)
        if (env[i] > env_max) env_max = env[i];
    m->papr_db = env_rms > 1e-12 ? 20.0 * log10(env_max / env_rms) : 0.0;

    /* duty cycle: fraction of time above 35% of the 95th percentile */
    {
        double thr = 0.35 * p95;
        size_t on = 0;
        for (size_t i = 0; i < nd; i++)
            if (env[i] > thr) on++;
        m->duty_cycle = (double)on / (double)nd;
    }

    /* instantaneous frequency */
    {
        const double scale = drate / (2.0 * M_PI);
        for (size_t i = 0; i + 1 < nd; i++) {
            double re = (double)mix[2 * (i + 1)] * mix[2 * i] +
                        (double)mix[2 * (i + 1) + 1] * mix[2 * i + 1];
            double im = (double)mix[2 * (i + 1) + 1] * mix[2 * i] -
                        (double)mix[2 * (i + 1)] * mix[2 * i + 1];
            ifreq[i] = (float)(atan2(im, re) * scale);
        }
        size_t nf = nd > 1 ? nd - 1 : 0;

        /* robust deviation via MAD (resists phase-jump spikes) */
        double med = median_of(ifreq, nf);
        double *absdev = malloc(nf * sizeof(double));
        double mad = 0.0;
        if (absdev != NULL) {
            for (size_t i = 0; i < nf; i++)
                absdev[i] = fabs((double)ifreq[i] - med);
            qsort(absdev, nf, sizeof(double), cmp_double);
            mad = absdev[nf / 2];
            free(absdev);
        }
        m->fm_deviation_hz = 1.4826 * mad;

        /* chirp trend: Pearson correlation of IF with time */
        double t_mean = (double)(nf - 1) / 2.0;
        double f_mean = 0.0;
        for (size_t i = 0; i < nf; i++) f_mean += ifreq[i];
        f_mean /= (double)nf;
        double s_tt = 0.0, s_ff = 0.0, s_tf = 0.0;
        for (size_t i = 0; i < nf; i++) {
            double dt_ = (double)i - t_mean;
            double df_ = (double)ifreq[i] - f_mean;
            s_tt += dt_ * dt_;
            s_ff += df_ * df_;
            s_tf += dt_ * df_;
        }
        m->chirp_trend = (s_tt > 0.0 && s_ff > 0.0)
                             ? s_tf / sqrt(s_tt * s_ff)
                             : 0.0;

        /* FSK level count from the IF histogram */
        double half = region_bw / 2.0;
        double dr_half = drate / 2.0;
        if (half > dr_half) half = dr_half;
        if (half > 0.0 && nf >= 256) {
            enum { HB = 64 };
            double hist[HB];
            memset(hist, 0, sizeof(hist));
            for (size_t i = 0; i < nf; i++) {
                double t = (ifreq[i] + half) / (2.0 * half);
                int idx = (int)(t * HB);
                if (idx >= 0 && idx < HB) hist[idx] += 1.0;
            }
            /* light 3-tap smoothing */
            double sm[HB];
            for (int i = 0; i < HB; i++) {
                double a = hist[i];
                double b = (i > 0) ? hist[i - 1] : hist[i];
                double c = (i < HB - 1) ? hist[i + 1] : hist[i];
                sm[i] = 0.25 * b + 0.5 * a + 0.25 * c;
            }
            double hmax = 0.0;
            for (int i = 0; i < HB; i++)
                if (sm[i] > hmax) hmax = sm[i];
            if (hmax > 0.0) {
                int peaks[8];
                int np = 0;
                for (int i = 1; i < HB - 1 && np < 8; i++) {
                    if (sm[i] >= sm[i - 1] && sm[i] >= sm[i + 1] &&
                        sm[i] >= 0.30 * hmax &&
                        (np == 0 || i - peaks[np - 1] >= 4)) {
                        peaks[np++] = i;
                    }
                }
                /* levels = peaks separated by a real valley */
                int levels = 0;
                int accepted[8];
                for (int p = 0; p < np; p++) {
                    bool separate = true;
                    if (levels > 0) {
                        int prev = accepted[levels - 1];
                        double vmin = 1e300;
                        for (int i = prev; i <= peaks[p]; i++)
                            if (sm[i] < vmin) vmin = sm[i];
                        double vpk = sm[prev] < sm[peaks[p]]
                                         ? sm[prev] : sm[peaks[p]];
                        separate = vmin < 0.5 * vpk;
                    }
                    if (separate) accepted[levels++] = peaks[p];
                }
                /* FSK dwells at its tone frequencies, so most samples
                 * cluster at the detected peaks. A continuously swept
                 * FM signal shows the same two edge peaks (arcsine
                 * distribution) but almost no mass at them, which this
                 * concentration test rejects. */
                double conc = 0.0;
                for (int p = 0; p < levels; p++) {
                    for (int b = accepted[p] - 2; b <= accepted[p] + 2; b++) {
                        if (b >= 0 && b < HB) conc += hist[b];
                    }
                }
                conc /= (double)nf;
                m->fsk_levels =
                    (levels >= 2 && levels <= 4 && conc >= 0.5) ? levels : 0;
            }
        }
    }

    /* symbol / envelope rate: FFT of the detrended envelope and |x|^2 */
    {
        size_t N = 2;
        while (N < nd) N <<= 1;
        if (N > 65536) {
            /* subsample the envelope so the FFT stays bounded */
            size_t step = (nd + 32767) / 32768;
            if (step < 2) step = 2;
            size_t nn = 0;
            for (size_t i = 0; i < nd; i += step) {
                env[nn++] = env[i];
                if (nn >= 32768) break;
            }
            nd = nn;
            N = 32768;
        }
        float *x = malloc(2 * N * sizeof(float));
        if (x != NULL) {
            FftPlan *ep = fft_plan_create((int)N);
            if (ep != NULL) {
                double best_conf = 0.0;
                for (int pass = 0; pass < 2 && best_conf < 0.5; pass++) {
                    /* pass 0: envelope; pass 1: squared envelope */
                    double mean = 0.0;
                    for (size_t i = 0; i < nd; i++) {
                        double v = pass == 0 ? env[i] : (double)env[i] * env[i];
                        mean += v;
                    }
                    mean /= (double)nd;
                    for (size_t i = 0; i < N; i++) {
                        if (i < nd) {
                            double v = pass == 0 ? env[i]
                                                 : (double)env[i] * env[i];
                            x[2 * i] = (float)(v - mean);
                        } else {
                            x[2 * i] = 0.0f;
                        }
                        x[2 * i + 1] = 0.0f;
                    }
                    float *mag = malloc(N * sizeof(float));
                    if (mag == NULL) continue;
                    fft_plan_spectrum(ep, x, NULL, mag);
                    double spec_total = 0.0, peak = 0.0;
                    int peak_bin = 0;
                    for (size_t b = 1; b < (size_t)N / 2 - 1; b++) { /* skip DC */
                        double lin = pow(10.0, mag[b] / 10.0);
                        spec_total += lin;
                        if (lin > peak) { peak = lin; peak_bin = (int)b; }
                    }
                    if (spec_total > 0.0) {
                        double conf = peak / spec_total;
                        if (conf > best_conf) {
                            best_conf = conf;
                            m->sym_rate_hz = (double)peak_bin * drate / (double)N;
                            m->sym_rate_conf = conf;
                        }
                    }
                    free(mag);
                }
                fft_plan_destroy(ep);
            }
            free(x);
        }
        /* if the envelope rate exceeds half the decimated rate it is
         * meaningless; cap the confidence instead */
        if (m->sym_rate_hz > drate / 2.0) m->sym_rate_conf = 0.0;
    }

    free(mix);
    free(env);
    free(ifreq);
    free(segment);

    classify(m, region_bw);
    return true;
}
