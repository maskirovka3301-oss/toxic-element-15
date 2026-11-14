/*
 * pulse.c — pulse-train analysis (see header).
 */
#include "pulse.h"
#include "debug.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsp.h"
#include "aconfig.h"
#include "metrics.h"
#include "region.h"

static int cmp_double(const void *a, const void *b)
{
    double d = *(const double *)a - *(const double *)b;
    return (d > 0.0) - (d < 0.0);
}

void pulse_clear(PulseResult *r)
{
    if (r == NULL) return;
    free(r->env);
    free(r->line_spectrum);
    memset(r, 0, sizeof(*r));
}

bool pulse_analyze(const IqSource *src, const Region *region, PulseResult *out)
{
    pulse_clear(out);
    if (src == NULL || region == NULL) return false;

    Region r = *region;
    if (!region_normalize(&r, src)) return false;

    size_t s0 = (size_t)(r.time_min * src->sample_rate);
    size_t s1 = (size_t)(r.time_max * src->sample_rate);
    if (s1 > src->total_samples) s1 = src->total_samples;
    if (s0 >= s1) return false;
    size_t n = s1 - s0;
    if (n < 64) return false;
    if (n > TE15_MAX_ANALYSIS_SAMPLES) n = TE15_MAX_ANALYSIS_SAMPLES;

    float *seg = malloc(2 * n * sizeof(float));
    if (seg == NULL) return false;
    iq_source_read_complex(src, s0, seg, n);
    double offset = ((r.freq_min + r.freq_max) / 2.0) - src->center_freq;
    metrics_mixdown(seg, n, src->sample_rate, -offset, seg);

    /* Envelope processing rate: bursts of a millisecond or shorter are
     * unresolved at 16 MS/s per-sample smoothing; decimate the selected
     * band to PULSE_FS_TARGET first (pulse widths then resolve to ~2 us
     * and the per-sample envelope loops stay cheap). */
    size_t D = dsp_decimation_for(src->sample_rate, r.freq_max - r.freq_min, PULSE_FS_TARGET);
    if (D < 1) D = 1;
    size_t nd = n / D;
    float *dec = malloc(2 * nd * sizeof(float));
    if (dec == NULL) {
        free(seg);
        return false;
    }
    {
        size_t got = dsp_decimate_staged(seg, n, D, dec);
        nd = got < nd ? got : nd;
    }
    free(seg);
    n = nd;
    if (n < 64) {
        free(dec);
        return false;
    }
    const double env_fs = src->sample_rate / (double)D;
    out->fs = env_fs;
    seg = dec; /* the envelope block below uses seg at the decimated rate */

    /* envelope: smoothed magnitude */
    double *env = malloc(n * sizeof(double));
    if (env == NULL) {
        free(seg);
        return false;
    }
    double arms2 = 0.0;
    for (size_t i = 0; i < n; i++) {
        double m = sqrt((double)seg[2 * i] * seg[2 * i] +
                        (double)seg[2 * i + 1] * seg[2 * i + 1]);
        arms2 += m * m;
        env[i] = m;
    }
    out->arms = sqrt(arms2 / (double)n);
    /* seg (the decimated complex samples) stays alive for the per-pulse
     * instantaneous-frequency analysis below */

    /* smoothing: ~20 us at the envelope rate (min 3 taps) */
    size_t smw = (size_t)(0.00002 * env_fs);
    if (smw < 3) smw = 3;
    if (smw > 63) smw = 63;
    double *smooth = malloc(n * sizeof(double));
    if (smooth == NULL) {
        free(env);
        return false;
    }
    {
        double acc = 0.0;
        size_t cnt = 0;
        for (size_t i = 0; i < n; i++) {
            acc += env[i];
            if (cnt >= smw) acc -= env[i - smw];
            cnt++;
            smooth[i] = acc / (double)(cnt < smw ? cnt : smw);
        }
    }
    free(env);

    /* threshold with hysteresis (the decimated complex samples stay
     * alive for the per-pulse instantaneous-frequency analysis) */
    double mean = 0.0, var = 0.0;
    for (size_t i = 0; i < n; i++) mean += smooth[i];
    mean /= (double)n;
    for (size_t i = 0; i < n; i++) {
        double d = smooth[i] - mean;
        var += d * d;
    }
    double sd = sqrt(var / (double)n);
    double mx = 0.0;
    for (size_t i = 0; i < n; i++) {
        if (smooth[i] > mx) mx = smooth[i];
    }
    /* Otsu threshold on the envelope histogram: pulse-train envelopes
     * are bimodal (floor vs pulses), and Otsu locates the valley
     * robustly even when the pulse amplitudes vary a lot - the sigma
     * and midpoint-capped rules both miss weak pulses of bursty
     * trains. */
    double thr_on = mean + 4.0 * sd;
    {
        enum { NB = 256 };
        double hist[NB];
        memset(hist, 0, sizeof(hist));
        if (mx > 0.0) {
            for (size_t i = 0; i < n; i++) {
                int b = (int)(smooth[i] / mx * (NB - 1));
                if (b < 0) b = 0;
                if (b >= NB) b = NB - 1;
                hist[b] += 1.0;
            }
            double total = (double)n;
            double sum_all = 0.0, sum_bg = 0.0, w_bg = 0.0;
            double best_var = -1.0;
            int best = 0;
            for (int b = 0; b < NB; b++) sum_all += (double)b * hist[b];
            for (int b = 0; b < NB; b++) {
                w_bg += hist[b];
                if (w_bg <= 0.0) continue;
                double w_fg = total - w_bg;
                if (w_fg <= 0.0) break;
                sum_bg += (double)b * hist[b];
                double m_bg = sum_bg / w_bg;
                double m_fg = (sum_all - sum_bg) / w_fg;
                double var_b = w_bg * w_fg * (m_bg - m_fg) * (m_bg - m_fg);
                if (var_b > best_var) {
                    best_var = var_b;
                    best = b;
                }
            }
            thr_on = (double)best / (double)(NB - 1) * mx;
            /* the Otsu level can sit close to the floor for very low
             * duty cycles: do not go below mean + 2 sigma */
            if (thr_on < mean + 2.0 * sd) thr_on = mean + 2.0 * sd;
        }
    }
    double thr_off = 0.7 * thr_on;
    if (thr_off < mean + 0.5 * sd) thr_off = mean + 0.5 * sd;
    if (thr_on < 1e-12) thr_on = 1e-12;
    out->threshold = thr_on;

    /* edge detection with hysteresis + false-trigger rejection */
    int np = 0;
    bool in_pulse = false;
    size_t pstart = 0;
    for (size_t i = 0; i < n; i++) {
        if (!in_pulse && smooth[i] >= thr_on) {
            in_pulse = true;
            pstart = i;
        } else if (in_pulse && smooth[i] < thr_off) {
            size_t pend = i;
            if (pend - pstart >= 2) { /* reject 1-sample triggers */
                if (np < PULSE_MAX_PULSES) {
                    out->pulse_start[np] = (double)pstart;
                    out->pulse_end[np] = (double)pend;
                    np++;
                }
            }
            in_pulse = false;
        }
    }
    if (in_pulse && n - pstart >= 2 && np < PULSE_MAX_PULSES) {
        out->pulse_start[np] = (double)pstart;
        out->pulse_end[np] = (double)n - 1;
        np++;
    }
    out->num_pulses = np;
    out->env = smooth;
    out->env_len = n;
    out->fs = env_fs;

    if (np < 2) {
        snprintf(out->note, sizeof(out->note),
                 "Pulse analysis: only %d pulse(s) found - the selection "
                 "may not contain a pulse train (threshold %.4f).",
                 np, thr_on);
        TE15_LOG("\nPULSE TRAIN ANALYSIS\n%s\n", out->note);
        out->modulation[0] = '\0';
        free(seg);
        return true;
    }

    /* Per-pulse metrics. Use median-based robust aggregates and reject
     * outlier detections (noise triggers): the raw mean is dominated by
     * stray single-sample triggers, which produced nonsense values such
     * as a jitter several times the period before. */
    double *widths = malloc((size_t)np * sizeof(double));
    double *periods = malloc((size_t)(np - 1) * sizeof(double));
    double *ipis = malloc((size_t)(np - 1) * sizeof(double));
    double *rises = malloc((size_t)np * sizeof(double));
    double *falls = malloc((size_t)np * sizeof(double));
    if (widths == NULL || periods == NULL || ipis == NULL ||
        rises == NULL || falls == NULL) {
        free(widths); free(periods); free(ipis); free(rises); free(falls);
        return true;
    }
    double tamp_sum = 0.0;
    for (int i = 0; i < np; i++) {
        size_t ps = (size_t)out->pulse_start[i];
        size_t pe = (size_t)out->pulse_end[i];
        double amp = 0.0;
        for (size_t j = ps; j <= pe; j++) {
            if (smooth[j] > amp) amp = smooth[j];
        }
        out->pulse_amp[i] = amp;
        tamp_sum += amp;
        widths[i] = (out->pulse_end[i] - out->pulse_start[i]) / env_fs;
        /* rise 10%-90%, fall 90%-10% of the pulse peak (signed, clamped) */
        double lo = 0.1 * amp, hi = 0.9 * amp;
        size_t r10 = ps, r90 = pe, f90 = ps, f10 = pe;
        for (size_t j = ps; j <= pe; j++) {
            if (smooth[j] >= lo) { r10 = j; break; }
        }
        for (size_t j = r10; j <= pe; j++) {
            if (smooth[j] >= hi) { r90 = j; break; }
        }
        for (size_t j = ps; j <= pe; j++) {
            if (smooth[j] >= hi) { f90 = j; break; }
        }
        for (size_t j = f90; j <= pe; j++) {
            if (smooth[j] <= lo) { f10 = j; break; }
        }
        /* unsigned subtraction must not wrap: use signed differences */
        rises[i] = (r90 > r10) ? (double)((long)r90 - (long)r10) / env_fs
                               : 1.0 / env_fs;
        falls[i] = (f10 > f90) ? (double)((long)f10 - (long)f90) / env_fs
                               : 1.0 / env_fs;
    }
    for (int i = 0; i + 1 < np; i++) {
        periods[i] = (out->pulse_start[i + 1] - out->pulse_start[i]) /
                     env_fs;
        ipis[i] = (out->pulse_start[i + 1] - out->pulse_end[i]) /
                  env_fs;
    }
    /* medians for robust aggregation and outlier rejection */
    {
        double *tmp = malloc((size_t)np * sizeof(double));
        if (tmp != NULL) {
            memcpy(tmp, widths, (size_t)np * sizeof(double));
            qsort(tmp, (size_t)np, sizeof(double), cmp_double);
            double med_w = tmp[np / 2];
            memcpy(tmp, periods, (size_t)(np - 1) * sizeof(double));
            qsort(tmp, (size_t)(np - 1), sizeof(double), cmp_double);
            double med_p = tmp[(np - 1) / 2];
            free(tmp);
            /* accept pulses whose width and period are within 5x/0.2x
             * of the medians (drops stray noise triggers) */
            int inl = 0;
            double w_sum = 0.0, p_sum = 0.0, i_sum = 0.0;
            double r_sum = 0.0, f_sum = 0.0;
            for (int i = 0; i < np; i++) {
                if (med_w > 0.0 &&
                    (widths[i] < 0.2 * med_w || widths[i] > 5.0 * med_w)) {
                    continue;
                }
                w_sum += widths[i];
                r_sum += rises[i];
                f_sum += falls[i];
                inl++;
            }
            int p_cnt = 0;
            for (int i = 0; i + 1 < np; i++) {
                if (med_p > 0.0 &&
                    (periods[i] < 0.2 * med_p || periods[i] > 5.0 * med_p)) {
                    continue;
                }
                p_sum += periods[i];
                i_sum += ipis[i];
                p_cnt++;
            }
            if (inl > 0) {
                out->pulse_width_s = w_sum / (double)inl;
                out->rise_s = r_sum / (double)inl;
                out->fall_s = f_sum / (double)inl;
            }
            if (p_cnt > 0) {
                out->period_s = p_sum / (double)p_cnt;
                out->ipi_s = i_sum / (double)p_cnt;
            }
            out->prf_hz = out->period_s > 0.0 ? 1.0 / out->period_s : 0.0;
            out->duty_cycle = out->period_s > 0.0
                                  ? out->pulse_width_s / out->period_s : 0.0;
            /* jitter: RMS deviation of the inlier periods, never above
             * half the period (anything larger is a detection failure) */
            double pvar = 0.0;
            int jc = 0;
            for (int i = 0; i + 1 < np; i++) {
                if (med_p > 0.0 &&
                    (periods[i] < 0.2 * med_p || periods[i] > 5.0 * med_p)) {
                    continue;
                }
                double dp = periods[i] - out->period_s;
                pvar += dp * dp;
                jc++;
            }
            out->jitter_s = jc > 0 ? sqrt(pvar / (double)jc) : 0.0;
            if (out->period_s > 0.0 && out->jitter_s > 0.5 * out->period_s) {
                out->jitter_s = 0.5 * out->period_s;
            }
            out->ipi_std_s = out->jitter_s;
        }
    }
    out->von_peak = np > 0 ? tamp_sum / (double)np : 0.0;
    free(rises);
    free(falls); /* widths stays alive for the interval-modulation analysis */

    /* frequency comb: FFT of the mean-removed envelope */
    {
        size_t nf = n < PULSE_ENV_FFT ? n : PULSE_ENV_FFT;
        double emean = 0.0;
        for (size_t i = 0; i < nf; i++) emean += smooth[i];
        emean /= (double)nf;
        float *ew = malloc(2 * nf * sizeof(float));
        if (ew != NULL) {
            FftPlan *plan = fft_plan_create((int)nf);
            fftwf_complex *X = fftwf_malloc(sizeof(fftwf_complex) * nf);
            if (plan != NULL && X != NULL) {
                for (size_t i = 0; i < nf; i++) {
                    ew[2 * i] = (float)(smooth[i] - emean);
                    ew[2 * i + 1] = 0.0f;
                }
                fft_plan_transform(plan, ew, NULL, X);
                out->line_n = (int)(nf / 2);
                out->line_spectrum = malloc((size_t)out->line_n * sizeof(double));
                if (out->line_spectrum != NULL) {
                    double smx = 0.0;
                    for (int b = 0; b < out->line_n; b++) {
                        out->line_spectrum[b] =
                            sqrt((double)X[b][0] * X[b][0] +
                                 (double)X[b][1] * X[b][1]);
                        if (out->line_spectrum[b] > smx) {
                            smx = out->line_spectrum[b];
                        }
                    }
                    for (int b = 0; b < out->line_n; b++) {
                        out->line_spectrum[b] =
                            20.0 * log10(out->line_spectrum[b] / smx + 1e-12);
                    }
                    /* f0: strongest local peak above 3 bins */
                    double best = -1e30;
                    int best_b = 0;
                    for (int b = 3; b < out->line_n - 1; b++) {
                        if (out->line_spectrum[b] > best &&
                            out->line_spectrum[b] > out->line_spectrum[b - 1] &&
                            out->line_spectrum[b] >= out->line_spectrum[b + 1] &&
                            out->line_spectrum[b] > -40.0) {
                            best = out->line_spectrum[b];
                            best_b = b;
                        }
                    }
                    out->f0_hz = best_b > 0
                                     ? (double)best_b * env_fs / (double)nf
                                     : 0.0;
                    /* first sinc null after the main lobe */
                    int nb = best_b > 0 ? best_b : 1;
                    out->sinc_null_hz = 0.0;
                    for (int b = nb + 2; b < out->line_n - 1; b++) {
                        if (out->line_spectrum[b] < -30.0 &&
                            out->line_spectrum[b + 1] <
                                out->line_spectrum[b]) {
                            out->sinc_null_hz =
                                (double)b * env_fs / (double)nf;
                            break;
                        }
                    }
                }
            }
            fft_plan_destroy(plan);
            fftwf_free(X);
            free(ew);
        }
    }
    out->tau_from_sinc_s = out->sinc_null_hz > 0.0 ? 1.0 / out->sinc_null_hz
                                                   : 0.0;

    /* Intra-pulse analysis: instantaneous frequency slope (chirp rate),
     * swept bandwidth, and centre-frequency diversity across pulses -
     * this is what distinguishes variable-waveform (chirped / agile)
     * bursts from plain amplitude pulses. */
    double chirp_rate_mean = 0.0, chirp_rate_cv = 0.0, chirp_bw_mean = 0.0;
    double pulse_freq_cv = 0.0;
    int chirp_pulses = 0, up_chirps = 0, down_chirps = 0;
    {
        double *centres = malloc((size_t)np * sizeof(double));
        double *sweeps = malloc((size_t)np * sizeof(double));
        double *slopes = malloc((size_t)np * sizeof(double));
        int n_cent = 0, n_chirp = 0;
        if (centres != NULL && sweeps != NULL && slopes != NULL) {
            for (int i = 0; i < np; i++) {
                long ps = (long)out->pulse_start[i];
                long pe = (long)out->pulse_end[i];
                long len = pe - ps + 1;
                if (len < 8) continue;
                /* Unwrapped phase -> instantaneous frequency. The
                 * leading/trailing 20% of the burst is excluded from the
                 * fit: the phase discontinuities at the pulse edges
                 * otherwise dominate the chirp-slope estimate. */
                long margin = len / 5;
                long lo_i = margin, hi_i = len - 1 - margin;
                if (hi_i - lo_i < 4) {
                    lo_i = 1;
                    hi_i = len - 1;
                }
                double prev_ph = atan2((double)seg[2 * (ps + lo_i) + 1],
                                       (double)seg[2 * (ps + lo_i)]);
                long cnt = hi_i - lo_i;
                double *ifr = malloc((size_t)cnt * sizeof(double));
                if (ifr == NULL) continue;
                double fsum = 0.0;
                for (long j = lo_i + 1; j <= hi_i; j++) {
                    double ph = atan2((double)seg[2 * (ps + j) + 1],
                                      (double)seg[2 * (ps + j)]);
                    double d = ph - prev_ph;
                    while (d > M_PI) d -= 2.0 * M_PI;
                    while (d < -M_PI) d += 2.0 * M_PI;
                    prev_ph = ph;
                    ifr[j - lo_i - 1] = d / (2.0 * M_PI) * env_fs;
                    fsum += ifr[j - lo_i - 1];
                }
                centres[n_cent++] = fsum / (double)cnt;
                /* least-squares slope of IF vs time (Hz/s) */
                double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
                for (long j = 0; j < cnt; j++) {
                    double t = (double)j / env_fs;
                    sx += t;
                    sy += ifr[j];
                    sxx += t * t;
                    sxy += t * ifr[j];
                }
                double den = (double)cnt * sxx - sx * sx;
                double slope = den > 1e-30
                                   ? ((double)cnt * sxy - sx * sy) / den
                                   : 0.0;
                /* Linearity (R^2) of the IF ramp: only a linear IF
                 * sweep is a chirp - the erratic IF of a wideband
                 * noise-like burst also fits a slope but with a poor
                 * R^2, which must not be reported as a chirp. */
                double r2 = 0.0;
                {
                    double ybar = sy / (double)cnt;
                    double ss_tot = 0.0, ss_res = 0.0;
                    for (long j = 0; j < cnt; j++) {
                        double t = (double)j / env_fs;
                        double fit = ybar - slope * (sx / (double)cnt) +
                                     slope * t;
                        double dy = ifr[j] - ybar;
                        ss_tot += dy * dy;
                        double dr = ifr[j] - fit;
                        ss_res += dr * dr;
                    }
                    if (ss_tot > 1e4 * (double)cnt) {
                        r2 = 1.0 - ss_res / ss_tot;
                        if (r2 < 0.0) r2 = 0.0;
                    }
                }
                double t_dur = (double)(len - 1) / env_fs;
                double sweep = fabs(slope) * t_dur;
                if (sweep > 10000.0 && r2 > 0.7) {
                    /* significant, linear chirp: > 10 kHz swept */
                    chirp_pulses++;
                    sweeps[n_chirp] = sweep;
                    slopes[n_chirp] = slope;
                    if (slope > 0.0) {
                        up_chirps++;
                    } else {
                        down_chirps++;
                    }
                    n_chirp++;
                }
                free(ifr);
            }
            if (n_chirp > 0) {
                double m_sweep = 0.0, m_rate = 0.0;
                for (int i = 0; i < n_chirp; i++) {
                    m_sweep += sweeps[i];
                    m_rate += fabs(slopes[i]);
                }
                m_sweep /= (double)n_chirp;
                m_rate /= (double)n_chirp;
                chirp_bw_mean = m_sweep;
                chirp_rate_mean = m_rate;
                double v = 0.0;
                for (int i = 0; i < n_chirp; i++) {
                    double d = fabs(slopes[i]) - m_rate;
                    v += d * d;
                }
                v = sqrt(v / (double)n_chirp);
                chirp_rate_cv = m_rate > 0.0 ? v / m_rate : 0.0;
            }
            if (n_cent > 1) {
                double m = 0.0, v = 0.0;
                for (int i = 0; i < n_cent; i++) m += centres[i];
                m /= (double)n_cent;
                for (int i = 0; i < n_cent; i++) {
                    double d = centres[i] - m;
                    v += d * d;
                }
                v = sqrt(v / (double)n_cent);
                /* the CV is meaningless when the pulses sit at DC (the
                 * relative spread would explode): require the mean
                 * centre frequency to be a real fraction of the band */
                if (fabs(m) > 0.01 * env_fs) {
                    pulse_freq_cv = v / fabs(m);
                }
            }
        }
        free(centres);
        free(sweeps);
        free(slopes);
    }
    out->chirp_rate_mean = chirp_rate_mean;
    out->chirp_rate_cv = chirp_rate_cv;
    out->chirp_bw_mean = chirp_bw_mean;
    out->chirp_pulses = chirp_pulses;
    out->up_chirps = up_chirps;
    out->down_chirps = down_chirps;
    out->pulse_freq_cv = pulse_freq_cv;

    /* Interval-modulation analysis: cluster the inlier pulse periods to
     * find discrete (PPM) levels; a continuous spread is PIM. */
    double interval_cv = 0.0, width_cv = 0.0;
    int interval_levels = 0;
    {
        size_t n_in = 0;
        double *srt = malloc((size_t)(np - 1) * sizeof(double));
        if (srt != NULL) {
            for (int i = 0; i + 1 < np; i++) srt[n_in++] = periods[i];
            qsort(srt, n_in, sizeof(double), cmp_double);
            /* cluster when the period leaves the current cluster by 5%
             * (tight discrete levels); an irregular train with
             * continuously varying intervals produces many clusters */
            double lo = 0.0;
            int cnt = 0;
            for (size_t i = 0; i < n_in; i++) {
                if (cnt == 0) {
                    lo = srt[i];
                    cnt = 1;
                } else if (srt[i] <= 1.05 * lo) {
                    cnt++;
                } else {
                    if (cnt >= 2) interval_levels++;
                    lo = srt[i];
                    cnt = 1;
                }
            }
            if (cnt >= 2) interval_levels++;
            free(srt);
            if (interval_levels > 8) interval_levels = 8;
        }
        if (out->period_s > 0.0) {
            double pv = 0.0;
            int c = 0;
            for (int i = 0; i + 1 < np; i++) {
                double d = periods[i] - out->period_s;
                pv += d * d;
                c++;
            }
            interval_cv = c > 0 ? sqrt(pv / (double)c) / out->period_s : 0.0;
        }
        if (out->pulse_width_s > 0.0) {
            double wv = 0.0;
            int c = 0;
            for (int i = 0; i < np; i++) {
                double d = widths[i] - out->pulse_width_s;
                wv += d * d;
                c++;
            }
            width_cv = c > 0 ? sqrt(wv / (double)c) / out->pulse_width_s : 0.0;
        }
    }
    out->interval_cv = interval_cv;
    out->width_cv = width_cv;
    out->interval_levels = interval_levels;
    out->mod_detail[0] = '\0';

    /* Modulation verdict, most specific structure first:
     * variable chirps > constant chirp bursts > PPM (discrete interval
     * levels) > PWM (width modulation) > PIM (continuous interval
     * modulation) > frequency-agile > constant-PRF > irregular. */
    {
        double pcv = out->jitter_s / (out->period_s + 1e-30);
        if (out->chirp_pulses >= 3 &&
            out->chirp_pulses * 2 >= np) {
            if (out->chirp_rate_cv > 0.30) {
                snprintf(out->modulation, sizeof(out->modulation),
                         "variable chirps (%s, rate CV %.0f%%)",
                         out->up_chirps >= out->down_chirps ? "up" : "down",
                         out->chirp_rate_cv * 100.0);
                snprintf(out->mod_detail, sizeof(out->mod_detail),
                         "chirped pulses: %.0f%% swept %.1f kHz on average, "
                         "sweep rate varies between pulses",
                         (double)out->chirp_pulses / np * 100.0,
                         out->chirp_bw_mean / 1e3);
            } else {
                snprintf(out->modulation, sizeof(out->modulation),
                         "chirp bursts (%s, %.1f kHz sweep)",
                         out->up_chirps >= out->down_chirps ? "up" : "down",
                         out->chirp_bw_mean / 1e3);
            }
        } else if (out->interval_levels >= 2 && out->interval_levels <= 6 &&
                   out->interval_cv > 0.05 && out->width_cv < 0.25) {
            snprintf(out->modulation, sizeof(out->modulation),
                     "PPM-like (%d discrete interval levels)",
                     out->interval_levels);
            snprintf(out->mod_detail, sizeof(out->mod_detail),
                     "pulse-position modulation: %d discrete intervals, "
                     "interval CV %.0f%%",
                     out->interval_levels, out->interval_cv * 100.0);
        } else if (out->width_cv > 0.20 && out->interval_cv < 0.12) {
            snprintf(out->modulation, sizeof(out->modulation),
                     "PWM-like (width CV %.0f%%)", out->width_cv * 100.0);
        } else if (out->interval_cv > 0.15 && out->width_cv < 0.25) {
            /* continuously varying intervals (no tight discrete levels):
             * pulse-interval modulation rather than a random jittering
             * train */
            snprintf(out->modulation, sizeof(out->modulation),
                     "PIM-like (interval-modulated, CV %.0f%%)",
                     out->interval_cv * 100.0);
            snprintf(out->mod_detail, sizeof(out->mod_detail),
                     "pulse-interval modulation: continuous interval "
                     "variation, interval CV %.0f%%",
                     out->interval_cv * 100.0);
        } else if (out->pulse_freq_cv > 0.20) {
            snprintf(out->modulation, sizeof(out->modulation),
                     "frequency-agile bursts (freq CV %.0f%%)",
                     out->pulse_freq_cv * 100.0);
        } else if (pcv > 0.25) {
            snprintf(out->modulation, sizeof(out->modulation),
                     "irregular burst train (period CV %.0f%%)",
                     pcv * 100.0);
        } else {
            snprintf(out->modulation, sizeof(out->modulation),
                     "constant-PRF train");
        }
    }
    /* --- pulse-train credibility -------------------------------------
     * Reject trains that are really threshold crossings of envelope
     * noise: a near-continuous envelope (duty > 92 %), a flat envelope
     * (CV < 12 %) or a period jitter comparable to the period all make
     * the PRF meaningless. An irregular train with moderate jitter stays
     * credible but is reported as such by the modulation verdict. */
    {
        double env_mean = 0.0;
        double env_var = 0.0;
        int credible = 1;

        out->credibility_note[0] = '\0';
        if (out->env != NULL && out->env_len > 0) {
            for (size_t i = 0; i < out->env_len; i++) {
                env_mean += out->env[i];
            }
            env_mean /= (double)out->env_len;
            for (size_t i = 0; i < out->env_len; i++) {
                double d = out->env[i] - env_mean;

                env_var += d * d;
            }
            env_var /= (double)out->env_len;
        }
        out->env_cv = env_mean > 0.0 ? sqrt(env_var) / env_mean : 0.0;
        out->jitter_frac = out->jitter_s / (out->period_s + 1e-30);
        if (np >= 2) {
            if (out->duty_cycle > 0.92) {
                credible = 0;
                snprintf(out->credibility_note, sizeof(out->credibility_note),
                         "duty %.0f%%: the envelope is nearly continuous, so "
                         "the %d 'pulses' are threshold crossings of envelope "
                         "noise, not bursts",
                         out->duty_cycle > 1.0 ? 100.0
                                               : out->duty_cycle * 100.0, np);
            } else if (out->env_cv < aconfig_get_num("pulse",
                                                    "env_cv_reject", 0.12)) {
                credible = 0;
                snprintf(out->credibility_note, sizeof(out->credibility_note),
                         "envelope CV %.0f%%: no pulsed/AM structure to "
                         "support a train", out->env_cv * 100.0);
            } else if (out->jitter_frac > aconfig_get_num(
                                             "pulse",
                                             "jitter_frac_reject",
                                             0.60)) {
                credible = 0;
                snprintf(out->credibility_note, sizeof(out->credibility_note),
                         "period jitter %.0f%% of the period: the spacing is "
                         "essentially random, no stable PRF",
                         out->jitter_frac * 100.0);
            } else if (out->f0_hz > 0.0 && out->prf_hz > 0.0) {
                /* A genuine periodic train puts its fundamental line at
                 * 1/T in the envelope's line spectrum; threshold crossings
                 * of noise put the envelope energy somewhere else. */
                double fr = out->f0_hz / out->prf_hz;
                /* the line spectrum's strongest line can be a harmonic or
                 * sub-harmonic of the PRF (a 50%-duty train has a strong
                 * 2f line), so accept any small rational ratio */
                static const double ratios[] = { 1.0, 2.0, 3.0, 4.0,
                                                 0.5, 1.0 / 3.0, 0.25,
                                                 1.5, 2.0 / 3.0 };
                bool consistent = false;

                for (size_t ri = 0; ri < sizeof(ratios) / sizeof(ratios[0]);
                     ri++) {
                    if (fabs(fr - ratios[ri]) <= 0.12 * ratios[ri]) {
                        consistent = true;
                        break;
                    }
                }
                if (!consistent) {
                    credible = 0;
                    snprintf(out->credibility_note,
                             sizeof(out->credibility_note),
                             "the envelope line spectrum peaks at %.1f Hz, "
                             "not at the reported PRF %.1f Hz: the pulses are "
                             "threshold crossings, not a periodic train",
                             out->f0_hz, out->prf_hz);
                }
            }
        }
        out->credible = credible;
        if (!credible) {
            out->prf_hz = 0.0;
            /* Strong structural verdicts survive the rate rejection: PPM
             * (discrete interval levels) and per-pulse chirps are
             * evidence about the pulse structure itself, not about a
             * stable PRF. Weaker verdicts (PIM/PWM/frequency-agile/
             * irregular/constant-PRF) would be claims about the rate
             * and are suppressed. */
            if (strstr(out->modulation, "PPM") != NULL ||
                strstr(out->modulation, "chirp") != NULL) {
                size_t l = strlen(out->modulation);

                snprintf(out->modulation + l,
                         sizeof(out->modulation) - l,
                         " [rate unvalidated]");
            } else {
                snprintf(out->modulation, sizeof(out->modulation),
                         "continuous (no credible pulse train)");
                out->mod_detail[0] = '\0';
            }
        }
    }

    (void)cmp_double;
    free(widths);
    free(periods);
    free(ipis);

    snprintf(out->note, sizeof(out->note),
             "Pulse train: %d pulses, tau %.3f ms, T %.3f ms, PRF %.1f Hz, "
             "duty %.1f%%, IPI %.3f ms, rise %.1f us, fall %.1f us, "
             "Vpeak %.3f, Arms %.3f, jitter %.2f us, f0 %.1f Hz, %s%s%s",
             np, out->pulse_width_s * 1e3, out->period_s * 1e3, out->prf_hz,
             out->duty_cycle * 100.0, out->ipi_s * 1e3, out->rise_s * 1e6,
             out->fall_s * 1e6, out->von_peak, out->arms, out->jitter_s * 1e6,
             out->f0_hz, out->modulation,
             out->mod_detail[0] != '\0' ? "; " : "",
             out->mod_detail);
    if (!out->credible && out->credibility_note[0] != '\0') {
        size_t l = strlen(out->note);

        snprintf(out->note + l, sizeof(out->note) - l,
                 " [not credible: %s]", out->credibility_note);
    }

    TE15_LOG("\nPULSE TRAIN ANALYSIS\n%s\n", out->note);
    free(seg);
    return true;
}
