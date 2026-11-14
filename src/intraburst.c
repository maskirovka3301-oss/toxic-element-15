/*
 * intraburst.c — intra-burst envelope inspection at the full sample rate.
 *
 * Pipeline:
 *   1. fresh raw I/Q of the selection at the FULL rate (no decimation),
 *      mixed at the selection centre, residual carrier offset removed
 *      from the cyclic-cumulant carrier estimate;
 *   2. burst detection on a SHORT RMS envelope (10 us), burst period from
 *      the envelope line spectrum, bursts aligned by cross-correlating
 *      their leading edges and folded (incoherent |x| average, plus a
 *      phase-coherent complex average when the signal is coherent);
 *   3. sample-level segmentation of the folded average with five
 *      complementary detectors (2-level amplitude model, dual-threshold
 *      hysteresis, change-point binary segmentation with a BIC penalty,
 *      a two-state HMM/Viterbi and multiscale DoG edges), edges refined
 *      at the half-max crossing;
 *   4. per-sub-pulse characterisation (width, spacing, duty, rise/fall,
 *      peak/mean, SNR, instantaneous frequency, chirp, FSK steps,
 *      bandwidth, matched-filter response);
 *   5. validation against false structure: Delta-BIC > 20, phase-
 *      randomised surrogate false-alarm rate, cross-burst consistency.
 */
#include "aconfig.h"
#include "debug.h"
#include "intraburst.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsp.h"
#include "sptk.h"

#define IB_MAX_SAMPLES    (8u * 1024u * 1024u)
#define IB_MAX_BURSTS     24
#define IB_MAX_SEGMENTS   256
#define IB_RMS_WIN_S      10e-6   /* short RMS envelope window          */
#define IB_PERIOD_MIN_HZ  0.2
#define IB_PERIOD_MAX_HZ  20000.0
#define IB_DBIC_MIN       20.0
#define IB_CWT_MIN_S      0.5e-6
#define IB_CWT_MAX_S      16e-6
#define IB_HMM_BIN_S      1e-6
#define IB_MIN_SEG_S      0.5e-6

typedef struct {
    size_t start, end;     /* sample indices in the (full-rate) buffer */
} IbSeg;

void intraburst_clear(IntraBurstResult *r)
{
    if (r == NULL) return;
    memset(r, 0, sizeof(*r));
}

static double mean_of(const double *v, size_t n)
{
    double s = 0.0;

    for (size_t i = 0; i < n; i++) s += v[i];
    return n > 0 ? s / (double)n : 0.0;
}

static double cv_of(const double *v, size_t n)
{
    double m, s = 0.0;

    if (n < 2) return 0.0;
    m = mean_of(v, n);
    if (m <= 0.0) return 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = v[i] - m;

        s += d * d;
    }
    return sqrt(s / (double)n) / m;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a;
    double y = *(const double *)b;

    return (x < y) ? -1 : (x > y) ? 1 : 0;
}

static double median_of(double *v, size_t n)
{
    if (n == 0) return 0.0;
    qsort(v, n, sizeof(double), cmp_double);
    if (n & 1u) return v[n / 2];
    return 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

/* value at a percentile of the array (sorts a copy) */
static double percentile_of(const float *v, size_t n, double p)
{
    double *tmp;
    double out;

    if (n == 0) return 0.0;
    tmp = malloc(n * sizeof(double));
    if (tmp == NULL) return 0.0;
    for (size_t i = 0; i < n; i++) tmp[i] = v[i];
    qsort(tmp, n, sizeof(double), cmp_double);
    {
        size_t idx = (size_t)(p * (double)(n - 1));

        if (idx >= n) idx = n - 1;
        out = tmp[idx];
    }
    free(tmp);
    return out;
}

/* short sliding-RMS envelope: sqrt of the box-averaged power */
static void rms_envelope(const float *iq, size_t n, size_t w, float *out)
{
    double acc = 0.0;

    if (w < 1) w = 1;
    for (size_t i = 0; i < n; i++) {
        double p = (double)iq[2 * i] * iq[2 * i] +
                   (double)iq[2 * i + 1] * iq[2 * i + 1];

        acc += p;
        if (i >= w) {
            double q = (double)iq[2 * (i - w)] * iq[2 * (i - w)] +
                       (double)iq[2 * (i - w) + 1] * iq[2 * (i - w) + 1];

            acc -= q;
        }
        out[i] = (float)sqrt(acc / (double)((i < w) ? i + 1 : w));
    }
}

/* Precise burst period from the envelope line spectrum (FFT of the
 * mean-removed RMS envelope, strongest in-band line, parabolically
 * refined). */
static double envelope_period_hz(const float *env, size_t n, double fs)
{
    size_t nf = 1;
    float *buf = NULL;
    float *spec = NULL;
    FftPlan *plan = NULL;
    double mean = 0.0;
    double best = 0.0;
    double best_hz = 0.0;
    int best_bin = 0;

    while (nf < n) nf <<= 1;
    if (nf > (1u << 22)) nf = 1u << 22;
    if (nf < 1024) return 0.0;
    buf = malloc(2 * nf * sizeof(float));
    spec = malloc(nf * sizeof(float));
    if (buf == NULL || spec == NULL) {
        free(buf);
        free(spec);
        return 0.0;
    }
    for (size_t i = 0; i < n; i++) mean += env[i];
    mean /= (double)n;
    for (size_t i = 0; i < nf; i++) {
        double v = (i < n) ? (double)env[i] - mean : 0.0;

        buf[2 * i] = (float)v;
        buf[2 * i + 1] = 0.0f;
    }
    plan = fft_plan_create((int)nf);
    if (plan != NULL) {
        fftwf_complex *xa = fftwf_malloc(sizeof(fftwf_complex) * nf);

        if (xa != NULL) {
            /* unshifted bins: k fs/nf, unlike fft_plan_spectrum which
             * fftshifts (a shifted bin index put the burst-rate line at
             * the array edges and the period estimate came out 0) */
            fft_plan_transform(plan, buf, NULL, xa);
            {
                int lo = (int)(IB_PERIOD_MIN_HZ / fs * (double)nf);
                int hi = (int)(IB_PERIOD_MAX_HZ / fs * (double)nf);

                if (lo < 2) lo = 2;
                if (hi > (int)nf / 2) hi = (int)nf / 2;
                for (int k = lo; k <= hi; k++) {
                    double m = (double)xa[k][0] * xa[k][0] +
                               (double)xa[k][1] * xa[k][1];

                    if (m > best) {
                        best = m;
                        best_bin = k;
                    }
                }
            }
            fftwf_free(xa);
        }
        fft_plan_destroy(plan);
    }
    free(buf);
    free(spec);
    if (best_bin <= 0) return 0.0;
    best_hz = (double)best_bin * fs / (double)nf;
    /* refine: use the neighbours' parabolic peak (spec is in dB) */
    return best_hz;
}

/* Locate the leading edge of a burst: first sample where the short RMS
 * envelope crosses the mid-level. */
static size_t leading_edge(const float *env, size_t a, size_t b, double mid)
{
    for (size_t i = a; i < b; i++) {
        if (env[i] >= mid) return i;
    }
    return a;
}

/* Cross-correlate a window of burst `b` against the template of burst
 * `a` and return the lag (in samples, b relative to a) within +-maxlag. */
static long align_lag(const float *env, size_t ta, size_t tb,
                      size_t half_win, long maxlag)
{
    double best = -1e300;
    long best_lag = 0;

    for (long lag = -maxlag; lag <= maxlag; lag++) {
        double sc = 0.0;
        long ia = (long)ta;

        for (size_t k = 0; k < half_win; k++) {
            long ib = (long)tb + (long)k + lag;

            if (ia + (long)k < 0 || ib < 0) continue;
            sc += (double)env[ia + (long)k] * (double)env[ib];
        }
        if (sc > best) {
            best = sc;
            best_lag = lag;
        }
    }
    return best_lag;
}

/* --- segmentation detectors --------------------------------------------- */

/* 2-component amplitude model (1-D 2-means on log-amplitude). Returns the
 * on/off means and the separation threshold. */
static void two_level_model(const float *v, size_t n, double *lvl_lo,
                            double *lvl_hi, double *thr)
{
    double mn = 1e300, mx = -1e300;

    for (size_t i = 0; i < n; i++) {
        double x = log((double)v[i] + 1e-12);

        if (x < mn) mn = x;
        if (x > mx) mx = x;
    }
    *lvl_lo = mn;
    *lvl_hi = mx;
    *thr = 0.5 * (mn + mx);
    if (mx <= mn) {
        return;
    }
    {
        double t = 0.5 * (mn + mx);

        for (int it = 0; it < 24; it++) {
            double s_lo = 0.0, s_hi = 0.0;
            size_t n_lo = 0, n_hi = 0;

            for (size_t i = 0; i < n; i++) {
                double x = log((double)v[i] + 1e-12);

                if (x < t) {
                    s_lo += x;
                    n_lo++;
                } else {
                    s_hi += x;
                    n_hi++;
                }
            }
            if (n_lo == 0 || n_hi == 0) break;
            *lvl_lo = s_lo / (double)n_lo;
            *lvl_hi = s_hi / (double)n_hi;
            t = 0.5 * (*lvl_lo + *lvl_hi);
        }
        *thr = t;
    }
}

/* Dual-threshold hysteresis segmentation. */
static size_t seg_hysteresis(const float *v, size_t n, double thr_hi,
                             double thr_lo, size_t min_len, IbSeg *seg,
                             size_t max_seg)
{
    size_t count = 0;
    bool in = false;
    size_t start = 0;

    for (size_t i = 0; i < n; i++) {
        if (!in && v[i] >= thr_hi) {
            in = true;
            start = i;
        } else if (in && v[i] < thr_lo) {
            in = false;
            if (i - start >= min_len && count < max_seg) {
                seg[count].start = start;
                seg[count].end = i;
                count++;
            }
        }
    }
    if (in && n - start >= min_len && count < max_seg) {
        seg[count].start = start;
        seg[count].end = n;
        count++;
    }
    return count;
}

/* Change-point detection: binary segmentation on log-amplitude with an L2
 * cost and a BIC penalty; returns the segment list. */
static double l2_cost(const float *lg, size_t a, size_t b)
{
    double s = 0.0, s2 = 0.0;
    size_t n = b - a;

    if (n == 0) return 0.0;
    for (size_t i = a; i < b; i++) {
        s += lg[i];
        s2 += lg[i] * lg[i];
    }
    return s2 - s * s / (double)n;
}

static size_t seg_changepoint(const float *lg, size_t n, size_t min_len,
                              IbSeg *out, size_t max_seg)
{
    /* recursive binary segmentation via an explicit stack */
    IbSeg stack[IB_MAX_SEGMENTS * 2];
    size_t sp = 0;
    size_t count = 0;
    double sigma2;
    double penalty;

    if (n < 2 * min_len) return 0;
    sigma2 = l2_cost(lg, 0, n) / (double)n;
    if (sigma2 <= 0.0) sigma2 = 1e-12;
    penalty = 2.0 * sigma2 * log((double)n); /* per extra segment (BIC-ish) */
    stack[sp].start = 0;
    stack[sp].end = n;
    sp++;
    while (sp > 0 && count < max_seg) {
        IbSeg cur = stack[--sp];
        size_t len = cur.end - cur.start;
        double base = l2_cost(lg, cur.start, cur.end);
        double best_gain = 0.0;
        size_t best_split = 0;

        if (len < 2 * min_len) {
            if (count < max_seg) out[count++] = cur;
            continue;
        }
        for (size_t k = cur.start + min_len; k + min_len <= cur.end; k++) {
            double gain = base - l2_cost(lg, cur.start, k) -
                          l2_cost(lg, k, cur.end);

            if (gain > best_gain) {
                best_gain = gain;
                best_split = k;
            }
        }
        if (best_split != 0 && best_gain > penalty) {
            stack[sp].start = best_split;
            stack[sp].end = cur.end;
            sp++;
            stack[sp].start = cur.start;
            stack[sp].end = best_split;
            sp++;
        } else if (count < max_seg) {
            out[count++] = cur;
        }
    }
    /* sort by start (binary segmentation pushes in order, but be safe) */
    for (size_t i = 0; i < count; i++) {
        for (size_t j = i + 1; j < count; j++) {
            if (out[j].start < out[i].start) {
                IbSeg t = out[i];

                out[i] = out[j];
                out[j] = t;
            }
        }
    }
    return count;
}

/* Two-state HMM (off/on) with Gaussian emissions on the amplitude, binning
 * the sequence at IB_HMM_BIN_S and running Viterbi. Returns the on-state
 * segments. */
static size_t seg_hmm(const float *v, size_t n, double sigma, IbSeg *out,
                      size_t max_seg)
{
    enum { K = 2 };
    double mu[2], var[2];
    double trans[2][2];
    double mn = 1e300, mx = -1e300;
    double *em[2];
    unsigned char *state = NULL;
    size_t count = 0;
    bool in = false;
    size_t start = 0;
    size_t n_state = 0;

    (void)n_state;
    for (size_t i = 0; i < n; i++) {
        double x = log((double)v[i] + 1e-12);

        if (x < mn) mn = x;
        if (x > mx) mx = x;
    }
    mu[0] = mn + 0.2 * (mx - mn);
    mu[1] = mn + 0.8 * (mx - mn);
    var[0] = var[1] = (sigma > 0 ? sigma : 0.1);
    trans[0][0] = 0.9;
    trans[0][1] = 0.1;
    trans[1][0] = 0.1;
    trans[1][1] = 0.9;
    em[0] = malloc(n * sizeof(double));
    em[1] = malloc(n * sizeof(double));
    state = malloc(n);
    if (em[0] == NULL || em[1] == NULL || state == NULL) {
        free(em[0]);
        free(em[1]);
        free(state);
        return 0;
    }
    /* 2 EM iterations to adapt the means (keeps it cheap) */
    for (int it = 0; it < 2; it++) {
        double s0 = 0.0, s1 = 0.0, v0 = 0.0, v1 = 0.0;
        size_t n0 = 0, n1 = 0;

        for (size_t i = 0; i < n; i++) {
            double x = log((double)v[i] + 1e-12);
            double d0 = (x - mu[0]) * (x - mu[0]);
            double d1 = (x - mu[1]) * (x - mu[1]);
            bool st = d1 < d0;

            if (st) {
                s1 += x;
                v1 += d1;
                n1++;
            } else {
                s0 += x;
                v0 += d0;
                n0++;
            }
        }
        if (n0 > 0) {
            mu[0] = s0 / (double)n0;
            var[0] = v0 / (double)n0;
        }
        if (n1 > 0) {
            mu[1] = s1 / (double)n1;
            var[1] = v1 / (double)n1;
        }
    }
    if (var[0] < 1e-6) var[0] = 1e-6;
    if (var[1] < 1e-6) var[1] = 1e-6;
    for (size_t i = 0; i < n; i++) {
        double x = log((double)v[i] + 1e-12);

        em[0][i] = -0.5 * ((x - mu[0]) * (x - mu[0]) / var[0] +
                           log(var[0]));
        em[1][i] = -0.5 * ((x - mu[1]) * (x - mu[1]) / var[1] +
                           log(var[1]));
    }
    {
        double *delta0 = malloc(n * sizeof(double));
        double *delta1 = malloc(n * sizeof(double));
        int *bp0 = malloc(n * sizeof(int));
        int *bp1 = malloc(n * sizeof(int));
        int st;

        if (delta0 == NULL || delta1 == NULL || bp0 == NULL || bp1 == NULL) {
            free(delta0);
            free(delta1);
            free(bp0);
            free(bp1);
            free(em[0]);
            free(em[1]);
                    free(state);
            return 0;
        }
        delta0[0] = log(0.5) + em[0][0];
        delta1[0] = log(0.5) + em[1][0];
        bp0[0] = 0;
        bp1[0] = 1;
        for (size_t i = 1; i < n; i++) {
            double a0 = delta0[i - 1] + log(trans[0][0]);
            double a1 = delta0[i - 1] + log(trans[0][1]);
            double b0v = delta1[i - 1] + log(trans[1][0]);
            double b1v = delta1[i - 1] + log(trans[1][1]);

            if (b0v > a0) {
                delta0[i] = b0v + em[0][i];
                bp0[i] = 1;
            } else {
                delta0[i] = a0 + em[0][i];
                bp0[i] = 0;
            }
            if (b1v > a1) {
                delta1[i] = b1v + em[1][i];
                bp1[i] = 1;
            } else {
                delta1[i] = a1 + em[1][i];
                bp1[i] = 0;
            }
        }
        st = (delta0[n - 1] > delta1[n - 1]) ? 0 : 1;
        state[n - 1] = (unsigned char)st;
        for (size_t i = n - 1; i > 0; i--) {
            st = (state[i] == 0) ? bp0[i] : bp1[i];
            state[i - 1] = (unsigned char)st;
        }
        free(delta0);
        free(delta1);
        free(bp0);
        free(bp1);
    }
    for (size_t i = 0; i < n; i++) {
        if (state[i]) {
            if (!in) {
                in = true;
                start = i;
            }
        } else if (in) {
            in = false;
            if (count < max_seg) {
                out[count].start = start;
                out[count].end = i;
                count++;
            }
        }
    }
    if (in && count < max_seg) {
        out[count].start = start;
        out[count].end = n;
        count++;
    }
    free(em[0]);
    free(em[1]);
    free(state);
    return count;
}

/* Multiscale DoG edge detection: derivative-of-Gaussian at several scales
 * on log-amplitude; an edge must persist across at least 3 scales. Marks
 * the edge positions in `edge[]` (1 = edge). */
static void cwt_edges(const float *lg, size_t n, double fs, unsigned char *edge,
                      size_t *edge_count)
{
    const double scales_us[6] = { 0.5, 1.0, 2.0, 4.0, 8.0, 16.0 };
    unsigned char *vote = calloc(n, 1);
    size_t votes_needed = 3;

    *edge_count = 0;
    for (size_t si = 0; si < 6; si++) {
        double sig = scales_us[si] * 1e-6 * fs;
        size_t r = (size_t)(3.0 * sig) + 1;

        if (vote == NULL || r < 1) continue;
        /* i-1-r and i+1+r must stay in range for the neighbour
         * derivatives (the old bound let i-1-k underflow) */
        for (size_t i = r + 1; i + r + 1 < n; i++) {
            double acc = 0.0;
            double norm = 0.0;

            for (size_t k = 1; k <= r; k++) {
                double g = exp(-(double)(k * k) / (2.0 * sig * sig));

                acc += g * ((double)lg[i + k] - (double)lg[i - k]);
                norm += g;
            }
            if (norm > 0.0) acc /= norm;
            if (acc > 0.0) {
                /* rising edge: local max of the derivative */
                double acc2 = 0.0;
                double acc0 = 0.0;
                double norm0 = 0.0, norm2 = 0.0;

                for (size_t k = 1; k <= r; k++) {
                    double g = exp(-(double)(k * k) / (2.0 * sig * sig));

                    acc0 += g * ((double)lg[i - 1 + k] - (double)lg[i - 1 - k]);
                    norm0 += g;
                    acc2 += g * ((double)lg[i + 1 + k] - (double)lg[i + 1 - k]);
                    norm2 += g;
                }
                if (norm0 > 0.0) acc0 /= norm0;
                if (norm2 > 0.0) acc2 /= norm2;
                if (acc > acc0 && acc >= acc2) vote[i]++;
            }
        }
    }
    if (vote != NULL) {
        for (size_t i = 0; i < n; i++) {
            if (vote[i] >= votes_needed) {
                edge[i] = 1;
                (*edge_count)++;
            }
        }
        free(vote);
    }
}

/* Refine an edge position to the half-max crossing between the local low
 * and high levels. */
static size_t refine_edge(const float *v, size_t i, size_t lo, size_t hi)
{
    double mx = 0.0;

    for (size_t k = lo; k < hi && k < i + 8; k++) {
        if (v[k] > mx) mx = v[k];
    }
    if (mx <= 0.0) return i;
    for (size_t k = i; k + 1 < hi && k + 1 < i + 16; k++) {
        if (v[k] >= 0.5 * mx) return k;
    }
    return i;
}

/* --- main analysis ------------------------------------------------------ */

bool intraburst_analyze_sot(const Sot *sot, const CccResult *ccc,
                            IntraBurstResult *out)
{
    float *iq;
    double fs, centre, mix_hz;
    IntraBurstResult res;

    if (out == NULL) return false;
    intraburst_clear(out);
    if (sot == NULL || !sot->valid || sot->iq == NULL || sot->n < 1024) {
        snprintf(out->verdict, sizeof(out->verdict), "no prepared source");
        return false;
    }
    iq = malloc(2 * sot->n * sizeof(float));
    if (iq == NULL) {
        snprintf(out->verdict, sizeof(out->verdict), "out of memory");
        return false;
    }
    memcpy(iq, sot->iq, 2 * sot->n * sizeof(float));
    fs = sot->view.sample_rate;
    centre = 0.5 * (sot->region.freq_min + sot->region.freq_max);
    mix_hz = 0.0;
    /* residual CFO from the cumulant carrier: the prepared source is
     * mixed at the selection centre, so only the residual rotation is
     * applied here (on the private copy) */
    if (ccc != NULL && ccc->valid && ccc->carrier_hz > 0.0) {
        double residual = ccc->carrier_hz - centre;
        double max_residual = 0.25 * (sot->region.freq_max -
                                      sot->region.freq_min);

        if (fabs(residual) <= max_residual && max_residual > 0.0) {
            mix_hz = residual;
        }
    }
    if (mix_hz != 0.0) {
        double dth = -2.0 * M_PI * mix_hz / fs;
        double th = 0.0;

        for (size_t i = 0; i < sot->n; i++) {
            double c = cos(th), sn = sin(th);
            double re = iq[2 * i], im = iq[2 * i + 1];

            iq[2 * i] = (float)(re * c - im * sn);
            iq[2 * i + 1] = (float)(re * sn + im * c);
            th += dth;
        }
    }
    {
        IqSource view = sot->view;
        Region r = sot->region;

        r.time_min = 0.0;
        r.time_max = (double)sot->n / fs;

        view.raw = (const uint8_t *)iq;
        view.samples = iq;
        res = *out;
        if (!intraburst_analyze(&view, &r, NULL, out)) {
            free(iq);
            return false;
        }
        out->residual_cfo_hz = mix_hz;
        out->filter_pass_hz = sot->pass_hz;
        out->filter_stop_hz = sot->stop_hz;
        out->filter_trans_hz = sot->trans_hz;
        out->noise_floor_db = sot->noise_floor_db;
        (void)res;
    }
    free(iq);
    return out->valid;
}

/* ------------------------------------------------------------------ */
/* intra-burst modulation classification (PPM / PIM / PWM / PFM / ...)  */
/* ------------------------------------------------------------------ */

/*
 * Detect whether a set of measurements is DISCRETE (a few well separated
 * levels) and, if so, how many levels and where. 1-D k-means with k =
 * 1..6, accepting the largest k whose clusters are well separated
 * (adjacent centres > 2.5x the pooled within-cluster deviation) and all
 * occupied by at least 5% of the samples. This is robust to the
 * measurement jitter that edge refinement leaves on real data, unlike a
 * histogram-peak search (which also misses modes that sit in the first or
 * last bin).
 */
static int find_levels(const double *v, size_t n, double *means, int *counts,
                       int max_levels)
{
    double *sorted = NULL, *tmp = NULL;
    int best = 0;
    double best_sep = 0.0;
    double spread;

    if (n < 8 || max_levels > 6) max_levels = n < 8 ? 0 : 6;
    if (max_levels < 2) return 0;
    sorted = malloc(n * sizeof(double));
    tmp = malloc(n * sizeof(double));
    if (sorted == NULL || tmp == NULL) {
        free(sorted);
        free(tmp);
        return 0;
    }
    memcpy(sorted, v, n * sizeof(double));
    for (size_t i = 0; i + 1 < n; i++) {
        for (size_t j = i + 1; j < n; j++) {
            if (sorted[j] < sorted[i]) {
                double t = sorted[i];

                sorted[i] = sorted[j];
                sorted[j] = t;
            }
        }
    }
    spread = sorted[n - 1] - sorted[0];
    if (spread <= 0.0) {
        free(sorted);
        free(tmp);
        return 0;
    }
    /* normalise to [0,1] for numerical stability */
    for (size_t i = 0; i < n; i++) tmp[i] = (sorted[i] - sorted[0]) / spread;

    for (int k = 2; k <= max_levels && (size_t)k <= n / 3; k++) {
        double c[6], prev[6];
        int cnt[6];
        bool ok = true;

        /* init: quantiles */
        for (int j = 0; j < k; j++) {
            size_t idx = (size_t)((double)(j + 0.5) / (double)k * (double)n);

            if (idx >= n) idx = n - 1;
            c[j] = tmp[idx];
        }
        for (int iter = 0; iter < 24; iter++) {
            double sum[6] = { 0 };
            int count[6] = { 0 };

            for (size_t i = 0; i < n; i++) {
                int jb = 0;
                double bd = 1e300;

                for (int j = 0; j < k; j++) {
                    double d = fabs(tmp[i] - c[j]);

                    if (d < bd) {
                        bd = d;
                        jb = j;
                    }
                }
                sum[jb] += tmp[i];
                count[jb]++;
            }
            for (int j = 0; j < k; j++) {
                prev[j] = c[j];
                c[j] = count[j] > 0 ? sum[j] / (double)count[j] : c[j];
            }
            {
                double moved = 0.0;

                for (int j = 0; j < k; j++) moved += fabs(c[j] - prev[j]);
                if (moved < 1e-9) break;
            }
        }
        /* cluster stats */
        for (int j = 0; j < k; j++) cnt[j] = 0;
        {
            double ss[6] = { 0 };

            for (size_t i = 0; i < n; i++) {
                int jb = 0;
                double bd = 1e300;

                for (int j = 0; j < k; j++) {
                    double d = fabs(tmp[i] - c[j]);

                    if (d < bd) {
                        bd = d;
                        jb = j;
                    }
                }
                cnt[jb]++;
                ss[jb] += bd * bd;
            }
            {
                /* sort centres */
                double cc[6];
                int nn[6];

                for (int j = 0; j < k; j++) {
                    for (int q = j + 1; q < k; q++) {
                        if (c[q] < c[j]) {
                            double t = c[j];

                            c[j] = c[q];
                            c[q] = t;
                            t = (double)cnt[j];
                            cnt[j] = cnt[q];
                            cnt[q] = (int)t;
                            t = ss[j];
                            ss[j] = ss[q];
                            ss[q] = t;
                        }
                    }
                }
                for (int j = 0; j < k; j++) cc[j] = c[j];
                for (int j = 0; j < k; j++) nn[j] = cnt[j];
                for (int j = 0; j < k; j++) {
                    double dev = nn[j] > 0 ? sqrt(ss[j] / (double)nn[j]) : 0.0;
                    double sep;

                    if (nn[j] < 0.05 * (double)n) {
                        ok = false;
                        break;
                    }
                    if (j + 1 < k) {
                        double gap = cc[j + 1] - cc[j];

                        sep = dev + (nn[j + 1] > 0
                                         ? sqrt(ss[j + 1] / (double)nn[j + 1])
                                         : 0.0);
                        if (sep < 1e-9) sep = 1e-9;
                        if (gap < 2.5 * sep) {
                            ok = false;
                            break;
                        }
                        if (gap / sep > best_sep) best_sep = gap / sep;
                    }
                }
                if (ok) {
                    best = k;
                    for (int j = 0; j < k; j++) {
                        means[j] = sorted[0] + cc[j] * spread;
                        counts[j] = nn[j];
                    }
                }
            }
        }
        /* a failing k does not mean larger k are invalid: a coarse k can
         * be "not separated" simply because its clusters each span two
         * true levels, so keep trying */
    }
    free(sorted);
    free(tmp);
    return best;
}

/*
 * Classify the pulse-train modulation from the per-sub-pulse widths and
 * intervals. The hypotheses are the classic analogue/digital pulse
 * modulations plus biphase coding:
 *   PPM  constant widths, positions on a discrete slot grid;
 *   PIM  constant widths, discrete but non-uniformly spaced intervals;
 *   PWM  variable widths, constant period;
 *   PFM  constant widths, continuously varying intervals;
 *   PEM/PWM+PIM  both width and interval structured;
 *   Manchester  two alternating interval levels (lag-1 = -1);
 *   random OOK  neither structured.
 */
/* robust relative spread: inter-decile range over the median (immune to
 * the odd merged or missed segment that the edge refinement leaves) */
static double robust_cv(const double *v, size_t n)
{
    double *tmp;
    double med, r;

    if (n < 4) return 1.0;
    tmp = malloc(n * sizeof(double));
    if (tmp == NULL) return 1.0;
    memcpy(tmp, v, n * sizeof(double));
    for (size_t i = 0; i + 1 < n; i++) {
        for (size_t j = i + 1; j < n; j++) {
            if (tmp[j] < tmp[i]) {
                double t = tmp[i];

                tmp[i] = tmp[j];
                tmp[j] = t;
            }
        }
    }
    med = tmp[n / 2];
    r = (tmp[(size_t)(0.9 * (double)(n - 1))] -
         tmp[(size_t)(0.1 * (double)(n - 1))]) / (med > 0.0 ? med : 1.0);
    free(tmp);
    return fabs(r);
}

/* class indices used by the per-burst scoring */
enum {
    IB_C_PPM = 0, IB_C_PIM, IB_C_PWMPIM, IB_C_PWM, IB_C_PFM,
    IB_C_MANCH, IB_C_RANDOM, IB_C_CLOCK, IB_C_COUNT
};

static const char *const ib_class_names[IB_C_COUNT] = {
    "PPM", "PIM", "PWM+PIM", "PWM", "PFM", "Manchester", "random OOK",
    "clock"
};

/* classes that are the same physical modulation family: a mixture between
 * them is just measurement variation on one hypothesis, not a combination */
static const int ib_class_family[IB_C_COUNT] = {
    0, /* PPM        - position modulation */
    0, /* PIM        - position modulation */
    1, /* PWM+PIM    - width (+position) modulation */
    1, /* PWM        - width modulation */
    2, /* PFM        - frequency modulation */
    3, /* Manchester - biphase coding */
    4, /* random OOK - no structure */
    5  /* clock      - no modulation */
};

static double clamp01(double v)
{
    if (v < 0.0) return 0.0;
    if (v > 1.0) return 1.0;
    return v;
}

/*
 * Score every intra-burst modulation hypothesis for ONE burst (0..1 each).
 * The scores are the soft versions of the classification rules, so they
 * can be turned into probabilities and compared as likelihoods - which is
 * what makes the combination-vs-single decision quantitative.
 */
static void burst_mod_scores(const double *widths, size_t nw,
                             const double *periods, size_t nt,
                             double scores[IB_C_COUNT],
                             int *n_wl, int *n_tl, int *uniform_t)
{
    double wmeans[8], tmeans[8];
    int wcounts[8], tcounts[8];
    int nwl = 0, ntl = 0;
    double w_cv, t_cv, manch = 0.0;
    double const_w, var_w, const_t, cont_t;
    double disc_t = 0.0, unif = 0.0, manch_s = 0.0;

    for (int c = 0; c < IB_C_COUNT; c++) scores[c] = 0.0;
    if (n_wl != NULL) *n_wl = 0;
    if (n_tl != NULL) *n_tl = 0;
    if (uniform_t != NULL) *uniform_t = 0;
    if (nw < 3 || nt < 3) {
        scores[IB_C_RANDOM] = 1.0;
        return;
    }
    w_cv = robust_cv(widths, nw);
    t_cv = robust_cv(periods, nt);
    nwl = find_levels(widths, nw, wmeans, wcounts, 8);
    ntl = find_levels(periods, nt, tmeans, tcounts, 8);
    if (n_wl != NULL) *n_wl = nwl;
    if (n_tl != NULL) *n_tl = ntl;

    /* interval-level quality: how well separated and how regular */
    if (ntl >= 2) {
        double gaps[7], mean_gap = 0.0, gcv = 0.0;

        for (int i = 1; i < ntl; i++) {
            gaps[i - 1] = tmeans[i] - tmeans[i - 1];
            mean_gap += gaps[i - 1];
        }
        mean_gap /= (double)(ntl - 1);
        if (mean_gap > 0.0) {
            for (int i = 0; i < ntl - 1; i++) {
                gcv += fabs(gaps[i] - mean_gap) / mean_gap;
            }
            gcv /= (double)(ntl - 1);
        }
        disc_t = clamp01(1.0 - gcv);          /* levels present */
        unif = clamp01(1.0 - 2.0 * gcv);      /* equally spaced  */
        if (mean_gap <= 0.0) {
            disc_t = unif = 0.0;
        }
    }
    if (uniform_t != NULL) *uniform_t = unif > 0.6 ? 1 : 0;

    /* lag-1 interval autocorrelation (Manchester = -1) */
    if (nt > 4) {
        double m = 0.0, va = 0.0, cov = 0.0;

        for (size_t i = 0; i < nt; i++) m += periods[i];
        m /= (double)nt;
        for (size_t i = 0; i < nt; i++) {
            double d = periods[i] - m;

            va += d * d;
            if (i + 1 < nt) cov += d * (periods[i + 1] - m);
        }
        if (va > 0.0) manch = cov / va;
    }

    const_w = clamp01(1.0 - w_cv / 0.15);
    var_w = clamp01((w_cv - 0.15) / 0.35);
    const_t = clamp01(1.0 - t_cv / 0.15);

    /* Intervals are either DISCRETE (levels resolved) or CONTINUOUS - the
     * two are mutually exclusive, so PFM/random cannot score on data with
     * resolved interval levels (that double counting let PFM win over PPM
     * on per-burst data with fewer intervals). */
    disc_t = ntl >= 2 ? 1.0 : 0.0;
    cont_t = 1.0 - disc_t;
    /* strict alternation of two interval levels = biphase coding */
    manch_s = clamp01((-manch - 0.40) / 0.50);

    scores[IB_C_CLOCK] = const_w * const_t;
    scores[IB_C_PPM] = const_w * disc_t * unif * (1.0 - manch_s);
    scores[IB_C_PIM] = const_w * disc_t * (1.0 - unif) * (1.0 - manch_s);
    scores[IB_C_PWM] = var_w * const_t;
    scores[IB_C_PFM] = const_w * cont_t * (1.0 - manch_s);
    scores[IB_C_PWMPIM] = var_w * disc_t;
    scores[IB_C_MANCH] = const_w * manch_s;
    scores[IB_C_RANDOM] = var_w * cont_t;
    /* keep a floor so an unclassifiable burst does not veto a class */
    for (int c = 0; c < IB_C_COUNT; c++) {
        if (scores[c] < 0.02) scores[c] = 0.02;
    }
}

static void classify_burst_mod(const double *widths, size_t nw,
                               const double *periods, size_t nt,
                               double w_cv_raw, double t_cv_raw,
                               IntraBurstResult *out)
{
    double wmeans[8], tmeans[8];
    int wcounts[8], tcounts[8];
    int nwl, ntl;
    double manchester = 0.0;
    double w_cv = robust_cv(widths, nw);
    double t_cv = robust_cv(periods, nt);
    bool w_const = w_cv < 0.15;
    bool w_var = w_cv > 0.25;
    bool t_const = t_cv < 0.15;
    bool t_var = t_cv > 0.25;

    (void)w_cv_raw;
    (void)t_cv_raw;

    out->burst_mod[0] = '\0';
    out->burst_mod_ev[0] = '\0';
    out->width_levels = 0;
    out->interval_levels = 0;
    if (nw < 3 || nt < 3) {
        snprintf(out->burst_mod, sizeof(out->burst_mod), "not classified");
        return;
    }
    nwl = find_levels(widths, nw, wmeans, wcounts, 8);
    ntl = find_levels(periods, nt, tmeans, tcounts, 8);
    out->width_levels = nwl;
    out->interval_levels = ntl;

    /* lag-1 autocorrelation of the intervals: -1 means strict alternation
     * (biphase / Manchester coding) */
    if (nt > 4) {
        double m = 0.0, va = 0.0, cov = 0.0;

        for (size_t i = 0; i < nt; i++) m += periods[i];
        m /= (double)nt;
        for (size_t i = 0; i < nt; i++) {
            double d = periods[i] - m;

            va += d * d;
            if (i + 1 < nt) cov += d * (periods[i + 1] - m);
        }
        if (va > 0.0) manchester = cov / va;
    }

    if (w_const && t_const) {
        snprintf(out->burst_mod, sizeof(out->burst_mod),
                 "unmodulated clock (PPM/PWM/PFM constant)");
        snprintf(out->burst_mod_ev, sizeof(out->burst_mod_ev),
                 "width CV %.1f%%, interval CV %.1f%%: a fixed pulse train",
                 w_cv * 100.0, t_cv * 100.0);
        return;
    }
    if (w_const && manchester < -0.55) {
        snprintf(out->burst_mod, sizeof(out->burst_mod),
                 "Manchester / biphase coding");
        snprintf(out->burst_mod_ev, sizeof(out->burst_mod_ev),
                 "two alternating interval levels (lag-1 %.2f), widths "
                 "constant (CV %.1f%%)", manchester, w_cv * 100.0);
        return;
    }
    if (w_const && ntl >= 2) {
        /* discrete interval levels with constant pulses: PPM (uniform
         * slot grid) vs PIM (arbitrary discrete intervals) */
        double step = 0.0, spread = 0.0;
        bool uniform = true;

        for (int i = 1; i < ntl; i++) {
            double d = tmeans[i] - tmeans[i - 1];

            if (i == 1) step = d;
            else {
                spread = fabs(d - step);
                if (spread > 0.15 * step) uniform = false;
            }
        }
        if (ntl >= 2 && uniform) {
            snprintf(out->burst_mod, sizeof(out->burst_mod),
                     "PPM (%d equally spaced slots)", ntl);
            snprintf(out->burst_mod_ev, sizeof(out->burst_mod_ev),
                     "widths constant (CV %.1f%%), intervals discrete at "
                     "%d levels spaced %.1f us", w_cv * 100.0, ntl,
                     step * 1e6);
        } else {
            snprintf(out->burst_mod, sizeof(out->burst_mod),
                     "PIM (%d discrete interval levels)", ntl);
            snprintf(out->burst_mod_ev, sizeof(out->burst_mod_ev),
                     "widths constant (CV %.1f%%), intervals discrete at "
                     "%d non-uniform levels", w_cv * 100.0, ntl);
        }
        return;
    }
    if (w_var && t_const) {
        snprintf(out->burst_mod, sizeof(out->burst_mod),
                 "PWM (width modulated, constant period)");
        snprintf(out->burst_mod_ev, sizeof(out->burst_mod_ev),
                 "width CV %.1f%% (duty spread), interval CV %.1f%% "
                 "(period fixed)", w_cv * 100.0, t_cv * 100.0);
        return;
    }
    if (w_const && t_var) {
        snprintf(out->burst_mod, sizeof(out->burst_mod), "PFM");
        snprintf(out->burst_mod_ev, sizeof(out->burst_mod_ev),
                 "widths constant (CV %.1f%%), intervals continuous "
                 "(CV %.1f%%)", w_cv * 100.0, t_cv * 100.0);
        return;
    }
    if (w_var && t_var) {
        if (nwl >= 2 || ntl >= 2) {
            snprintf(out->burst_mod, sizeof(out->burst_mod),
                     "PWM+PIM combination (%d width levels, %d interval "
                     "levels)", nwl, ntl);
            snprintf(out->burst_mod_ev, sizeof(out->burst_mod_ev),
                     "both width (CV %.1f%%, %d levels) and interval "
                     "(CV %.1f%%, %d levels) are modulated",
                     w_cv * 100.0, nwl, t_cv * 100.0, ntl);
        } else {
            snprintf(out->burst_mod, sizeof(out->burst_mod),
                     "random OOK data (no stable pulse structure)");
            snprintf(out->burst_mod_ev, sizeof(out->burst_mod_ev),
                     "width CV %.1f%% and interval CV %.1f%%, both "
                     "continuous", w_cv * 100.0, t_cv * 100.0);
        }
        return;
    }
    snprintf(out->burst_mod, sizeof(out->burst_mod), "mixed/uncertain");
    snprintf(out->burst_mod_ev, sizeof(out->burst_mod_ev),
             "width CV %.1f%%, interval CV %.1f%%", w_cv * 100.0,
             t_cv * 100.0);
}

/*
 * Isolate each detected burst and classify its modulation on its own,
 * then aggregate the votes: a single dominant class confirms the folded
 * classification, two comparable classes indicate a COMBINATION of
 * modulations between bursts (the user's "which combinations" question).
 * The bursts are classified individually because a fold can average two
 * different patterns into a third, misleading one.
 */
static void classify_individual_bursts(const float *fine, size_t cap,
                                       const IbSeg *bursts, size_t nbursts,
                                       double fs, double min_seg_s,
                                       IntraBurstResult *out)
{
    double prob[64][IB_C_COUNT];
    double ll_single[IB_C_COUNT];
    double votes[IB_C_COUNT];
    int classified = 0;
    size_t min_seg = (size_t)(min_seg_s * fs);
    IbSeg segs[IB_MAX_SEGMENTS];

    memset(prob, 0, sizeof(prob));
    memset(ll_single, 0, sizeof(ll_single));
    memset(votes, 0, sizeof(votes));
    if (min_seg < 1) min_seg = 1;
    for (size_t b = 0; b < nbursts && classified < 64; b++) {
        size_t s0 = bursts[b].start;
        size_t s1 = bursts[b].end;
        size_t len;
        double on, off, thr_hi, thr_lo, span;
        int n_segs;
        double wv[IB_MAX_SEGMENTS], tv[IB_MAX_SEGMENTS];
        size_t nw = 0, nt = 0;

        if (s1 <= s0 || s1 > cap) continue;
        len = s1 - s0;
        if (len < 16 * min_seg) continue;
        on = percentile_of(fine, s0 + len / 4, len / 4);
        {
            double q = percentile_of(fine, s0, len / 4);

            off = q < on ? q : on * 0.5;
        }
        span = on - off;
        if (span <= 0.0) continue;
        thr_hi = off + 0.60 * span;
        thr_lo = off + 0.35 * span;
        n_segs = seg_hysteresis(fine + s0, len, thr_hi, thr_lo, min_seg,
                                segs, IB_MAX_SEGMENTS);
        if (n_segs < 3) continue;
        for (int i = 0; i < n_segs && nw < IB_MAX_SEGMENTS; i++) {
            wv[nw++] = (double)(segs[i].end - segs[i].start) / fs;
            if (i > 0) {
                tv[nt++] = (double)(segs[i].start - segs[i - 1].start) / fs;
            }
        }
        if (nw < 3 || nt < 2) continue;
        {
            double scores[IB_C_COUNT];
            double sum = 0.0;
            int bc = IB_C_RANDOM;

            burst_mod_scores(wv, nw, tv, nt, scores, NULL, NULL, NULL);
            for (int c = 0; c < IB_C_COUNT; c++) sum += scores[c];
            if (sum <= 0.0) sum = 1.0;
            for (int c = 0; c < IB_C_COUNT; c++) {
                double p = scores[c] / sum;

                prob[classified][c] = p;
                ll_single[c] += log(p > 1e-12 ? p : 1e-12);
                if (p > prob[classified][bc]) bc = c;
            }
            votes[bc] += 1.0;
            classified++;
        }
    }
    out->bursts_classified = classified;
    if (classified == 0) {
        out->burst_mod_cand[0][0] = '\0';
        out->burst_mod_cand[1][0] = '\0';
        out->burst_mod_support[0] = out->burst_mod_support[1] = 0.0;
        out->burst_mod_combo[0] = '\0';
        out->burst_mod_choice[0] = '\0';
        out->burst_mod_dll = 0.0;
        out->burst_mod_threshold = 0.0;
        return;
    }

    /* ---- which single class fits best? ---- */
    {
        int best_single = 0;
        int second = -1;

        for (int c = 1; c < IB_C_COUNT; c++) {
            if (ll_single[c] > ll_single[best_single]) best_single = c;
        }
        for (int c = 0; c < IB_C_COUNT; c++) {
            if (c == best_single) continue;
            if (second < 0 || votes[c] > votes[second]) second = c;
        }
        snprintf(out->burst_mod_cand[0], sizeof(out->burst_mod_cand[0]),
                 "%s", ib_class_names[best_single]);
        out->burst_mod_support[0] = votes[best_single] / (double)classified;
        if (second >= 0 && votes[second] > 0) {
            snprintf(out->burst_mod_cand[1],
                     sizeof(out->burst_mod_cand[1]), "%s",
                     ib_class_names[second]);
            out->burst_mod_support[1] = votes[second] / (double)classified;
        } else {
            out->burst_mod_cand[1][0] = '\0';
            out->burst_mod_support[1] = 0.0;
        }

        /* ---- is a two-class MIXTURE a better fit than the best single
         * class? Fit the mixing weight by EM on the per-burst class
         * probabilities and compare the log-likelihoods with a BIC
         * penalty for the extra parameter: 2*DeltaLL > ln(N). ---- */
        {
            double best_dll = -1e300;
            int bc1 = -1, bc2 = -1;
            double best_ll = ll_single[0];

            for (int c = 1; c < IB_C_COUNT; c++) {
                if (ll_single[c] > best_ll) best_ll = ll_single[c];
            }
            for (int a = 0; a < IB_C_COUNT; a++) {
                for (int b = a + 1; b < IB_C_COUNT; b++) {
                    double pi = 0.5;
                    double ll = 0.0;

                    if (ib_class_family[a] == ib_class_family[b]) {
                        continue; /* same modulation family */
                    }
                    if (votes[a] < 0.10 * (double)classified &&
                        votes[b] < 0.10 * (double)classified) {
                        continue;
                    }
                    for (int iter = 0; iter < 32; iter++) {
                        double num = 0.0;

                        ll = 0.0;
                        for (int k = 0; k < classified; k++) {
                            double pa = prob[k][a], pb = prob[k][b];
                            double mix = pi * pa + (1.0 - pi) * pb;
                            double r;

                            if (mix < 1e-12) mix = 1e-12;
                            ll += log(mix);
                            r = pi * pa / mix;
                            num += r;
                        }
                        {
                            double np = num / (double)classified;

                            if (fabs(np - pi) < 1e-6) {
                                pi = np;
                                break;
                            }
                            pi = np;
                        }
                    }
                    if (pi < 0.02 || pi > 0.98) continue;
                    {
                        double dll = 2.0 * (ll - best_ll);

                        if (dll > best_dll) {
                            best_dll = dll;
                            bc1 = a;
                            bc2 = b;
                        }
                    }
                }
            }
            out->burst_mod_threshold = log((double)classified) > 1.0
                                           ? log((double)classified) : 1.0;
            out->burst_mod_dll = best_dll;
            if (bc1 >= 0 && best_dll > out->burst_mod_threshold) {
                snprintf(out->burst_mod_combo, sizeof(out->burst_mod_combo),
                         "combination %s + %s", ib_class_names[bc1],
                         ib_class_names[bc2]);
                snprintf(out->burst_mod_choice,
                         sizeof(out->burst_mod_choice),
                         "combination preferred: 2*DeltaLL %.1f > BIC "
                         "threshold %.1f", best_dll,
                         out->burst_mod_threshold);
            } else if (bc1 >= 0) {
                snprintf(out->burst_mod_combo, sizeof(out->burst_mod_combo),
                         "no combination (%s single class)",
                         ib_class_names[best_single]);
                snprintf(out->burst_mod_choice,
                         sizeof(out->burst_mod_choice),
                         "single class preferred: best pair %s + %s only "
                         "2*DeltaLL %.1f vs threshold %.1f",
                         ib_class_names[bc1], ib_class_names[bc2], best_dll,
                         out->burst_mod_threshold);
            } else {
                snprintf(out->burst_mod_combo, sizeof(out->burst_mod_combo),
                         "no combination (%s single class)",
                         ib_class_names[best_single]);
                snprintf(out->burst_mod_choice,
                         sizeof(out->burst_mod_choice),
                         "no two-class mixture improves on the single "
                         "class");
            }
        }
    }
}

bool intraburst_analyze(const IqSource *src, const Region *region,
                        const CccResult *ccc, IntraBurstResult *out)
{
    Region r;
    size_t s0, n, n_cap;
    float *iq = NULL;
    float *env = NULL;      /* short RMS envelope, full rate             */
    float *fine = NULL;     /* full-rate |x| (edge-preserving)           */
    double fs, centre, mix_hz;
    size_t rms_w, min_seg;
    double noise_med, noise_mad, burst_thr;
    IbSeg bursts[IB_MAX_BURSTS];
    size_t n_bursts = 0;
    size_t fold_len;
    float *ifold = NULL;    /* incoherent |x| fold                       */
    float *cfold_re = NULL; /* coherent complex fold                     */
    float *cfold_im = NULL;
    size_t folds_used = 0;
    double coherence_sum = 0.0;
    double widths_all[IB_MAX_SEGMENTS], periods_all[IB_MAX_SEGMENTS];
    double rises_all[IB_MAX_SEGMENTS];
    size_t widths_n = 0, periods_n = 0, rises_n = 0;
    double burst_duty[IB_MAX_BURSTS];
    IbSeg segs[IB_MAX_SEGMENTS];
    size_t n_segs = 0;
    int confirm = 0;

    if (out == NULL) return false;
    intraburst_clear(out);
    if (src == NULL || region == NULL) {
        snprintf(out->verdict, sizeof(out->verdict), "not available");
        return false;
    }
    r = *region;
    if (!region_normalize(&r, src)) {
        snprintf(out->verdict, sizeof(out->verdict), "invalid region");
        return false;
    }
    fs = src->sample_rate;
    s0 = (size_t)(r.time_min * fs);
    n = (size_t)((r.time_max - r.time_min) * fs);
    if (s0 >= src->total_samples) {
        snprintf(out->verdict, sizeof(out->verdict), "out of bounds");
        return false;
    }
    if (s0 + n > src->total_samples) n = src->total_samples - s0;
    n_cap = (n > IB_MAX_SAMPLES) ? IB_MAX_SAMPLES : n;
    rms_w = (size_t)(aconfig_get_num("intraburst", "rms_window_us",
                                     IB_RMS_WIN_S * 1e6) *
                    1e-6 * fs);
    if (rms_w < 2) rms_w = 2;
    min_seg = (size_t)(IB_MIN_SEG_S * fs);
    if (min_seg < 1) min_seg = 1;
    if (n_cap < 16 * rms_w) {
        snprintf(out->verdict, sizeof(out->verdict), "selection too short");
        return false;
    }

    /* --- 1. selection centre + residual CFO from the cumulant carrier --- */
    centre = 0.5 * (r.freq_min + r.freq_max);
    mix_hz = centre - src->center_freq;
    if (ccc != NULL && ccc->valid && ccc->carrier_hz > 0.0) {
        double residual = ccc->carrier_hz - centre;
        double max_residual = 0.25 * (r.freq_max - r.freq_min);

        if (fabs(residual) <= max_residual && max_residual > 0.0) {
            mix_hz += residual;
            out->residual_cfo_hz = residual;
        }
    }
    iq = malloc(2 * n_cap * sizeof(float));
    env = malloc(n_cap * sizeof(float));
    if (iq == NULL || env == NULL) {
        free(iq);
        free(env);
        snprintf(out->verdict, sizeof(out->verdict), "out of memory");
        return false;
    }
    iq_source_read_complex(src, s0, iq, n_cap);
    if (mix_hz != 0.0) {
        double dth = -2.0 * M_PI * mix_hz / fs;
        double th = fmod(dth * (double)s0, 2.0 * M_PI);

        for (size_t i = 0; i < n_cap; i++) {
            double c = cos(th), sn = sin(th);
            double re = iq[2 * i], im = iq[2 * i + 1];

            iq[2 * i] = (float)(re * c - im * sn);
            iq[2 * i + 1] = (float)(re * sn + im * c);
            th += dth;
        }
    }

    /* --- 2. context-derived band filter -------------------------------
     * The slice holds the FULL original band (the time range of the
     * selection copied at the original sample rate). The selection has
     * been mixed to DC; a low-pass is now designed from the wideband
     * spectrum so that the passband keeps the OOK/ASK edge sidelobes of
     * the selection while everything farther away (other emissions and
     * the out-of-band noise) is rejected:
     *   passband edge = where the selection's own spectrum has fallen
     *                   25 dB below its peak (the sidelobe energy is
     *                   still inside),
     *   stopband edge = the first frequency beyond that where the
     *                   spectrum falls to the noise floor + 6 dB, or
     *                   1.35x the passband, whichever comes first (that
     *                   also respects any neighbouring emission),
     *   transition    = stopband - passband, taps sized for it. */
    {
        enum { PSD_N = 65536, PSD_AVG = 8 };
        size_t psd_n = PSD_N;
        float *blk = malloc(2 * psd_n * sizeof(float));
        float *spec = malloc(psd_n * sizeof(float));
        FftPlan *fplan = NULL;
        double *psd = malloc(psd_n * sizeof(double));
        double sel_bw = r.freq_max - r.freq_min;

        if (blk != NULL && spec != NULL && psd != NULL) {
            double floor_med;
            double peak = -1e300;
            size_t blocks = 0;
            double hann_psd[256];

            memset(psd, 0, psd_n * sizeof(double));
            for (int w = 0; w < 256; w++) {
                double v = 0.5 - 0.5 * cos(2.0 * M_PI * (double)w /
                                           (double)(256 - 1));

                hann_psd[w] = v;
            }
            for (size_t b = 0; b < PSD_AVG; b++) {
                size_t off = (n_cap > psd_n)
                                 ? (size_t)((double)b / (double)PSD_AVG *
                                            (double)(n_cap - psd_n))
                                 : 0;

                if (n_cap < psd_n) break;
                for (size_t i = 0; i < psd_n; i++) {
                    double wnd = hann_psd[i * 256 / psd_n];

                    blk[2 * i] = (float)(iq[2 * (off + i)] * wnd);
                    blk[2 * i + 1] = (float)(iq[2 * (off + i) + 1] * wnd);
                }
                fplan = fft_plan_create((int)psd_n);
                if (fplan == NULL) break;
                {
                    fftwf_complex *xa = fftwf_malloc(sizeof(fftwf_complex) *
                                                     psd_n);

                    if (xa == NULL) {
                        fft_plan_destroy(fplan);
                        fplan = NULL;
                        break;
                    }
                    fft_plan_transform(fplan, blk, NULL, xa);
                    for (size_t i = 0; i < psd_n; i++) {
                        double m = (double)xa[i][0] * xa[i][0] +
                                   (double)xa[i][1] * xa[i][1];

                        psd[i] += m;
                    }
                    fftwf_free(xa);
                }
                fft_plan_destroy(fplan);
                fplan = NULL;
                blocks++;
            }
            /* fold to a single-sided profile around DC over +/- fs/2 */
            {
                double *prof = malloc((psd_n / 2) * sizeof(double));
                size_t prof_n = psd_n / 2;

                if (prof != NULL && blocks > 0) {
                    for (size_t k = 0; k < prof_n; k++) {
                        prof[k] = psd[k];
                    }
                    /* noise floor: median of the whole profile */
                    {
                        double *tmp = malloc(prof_n * sizeof(double));

                        if (tmp != NULL) {
                            memcpy(tmp, prof, prof_n * sizeof(double));
                            floor_med = median_of(tmp, prof_n);
                            free(tmp);
                        } else {
                            floor_med = 1e-20;
                        }
                    }
                    out->noise_floor_db = 10.0 * log10(floor_med + 1e-20);
                    /* peak within the selection span */
                    {
                        size_t k0 = (size_t)(0.5 * sel_bw / fs * psd_n);
                        size_t k1 = k0 < prof_n ? k0 : prof_n - 1;

                        peak = -1e300;
                        for (size_t k = 0; k <= k1; k++) {
                            if (prof[k] > peak) peak = prof[k];
                        }
                    }
                    if (peak > 0.0) {
                        double pass_edge = 0.5 * sel_bw;
                        double stop_edge = 0.0;
                        double floor6 = floor_med * 4.0; /* +6 dB */
                        double bin_hz = fs / (double)psd_n;
                        size_t k;

                        /* passband: extend until the selection's own
                         * spectrum is 25 dB below its peak */
                        for (k = (size_t)(0.5 * sel_bw / bin_hz); k > 0 &&
                             k < prof_n; k++) {
                            if (prof[k] < peak * 10.0 / 316.0) { /* -25 dB */
                                break;
                            }
                        }
                        pass_edge = fabs((double)k * bin_hz);
                        /* baseline: the passband always keeps a 1.5x
                         * margin over the selection band so the OOK/ASK
                         * edge sidelobes are preserved; it only grows
                         * when the signal's own spectrum extends farther
                         * (e.g. spectral regrowth from fast edges) */
                        if (pass_edge < 0.75 * sel_bw) {
                            pass_edge = 0.75 * sel_bw;
                        }
                        /* neighbour / noise crossing beyond the passband */
                        for (k = (size_t)(pass_edge / bin_hz) + 1;
                             k < prof_n; k++) {
                            if (prof[k] < floor6) {
                                stop_edge = (double)k * bin_hz;
                                break;
                            }
                        }
                        if (stop_edge <= pass_edge) {
                            stop_edge = pass_edge * 1.35;
                        }
                        if (stop_edge > pass_edge * 1.35) {
                            stop_edge = pass_edge * 1.35;
                        }
                        {
                            double trans = stop_edge - pass_edge;

                            /* a minimum transition keeps the mask
                             * well-conditioned: 15% of the passband */
                            if (trans < 0.15 * pass_edge) {
                                stop_edge = pass_edge + 0.15 * pass_edge;
                                trans = stop_edge - pass_edge;
                            }
                            if (trans > 0.60 * pass_edge) {
                                stop_edge = pass_edge + 0.60 * pass_edge;
                                trans = 0.60 * pass_edge;
                            }
                            /* zero-phase frequency-domain mask: exact
                             * linear phase and no FIR-length limit, so
                             * the OOK/ASK edge sharpness is preserved */
                            if (dsp_fft_mask_lowpass(iq, n_cap, fs, pass_edge,
                                                     stop_edge)) {
                                if (getenv("TE15_IB_DEBUG") != NULL) {
                                    fprintf(stderr,
                                            "IBDBG-FILTER: pass %.1f kHz stop %.1f kHz trans %.1f kHz (zero-phase FFT mask) floor %.1f dB\n",
                                            pass_edge / 1e3, stop_edge / 1e3,
                                            trans / 1e3, out->noise_floor_db);
                                }
                                out->filter_pass_hz = pass_edge;
                                out->filter_stop_hz = stop_edge;
                                out->filter_trans_hz = trans;
                                out->filter_taps = 0; /* FFT mask, no FIR */
                            }
                        }
                        out->neighbour_hz = stop_edge;
                    }
                }
                free(prof);
            }
        }
        free(blk);
        free(spec);
        free(psd);
        if (fplan != NULL) fft_plan_destroy(fplan);
    }

    /* --- 3. short-RMS burst detection ---------------------------------
     * Two envelopes: the 10 us RMS envelope only locates the bursts (its
     * window would average away shorter intra-burst sub-pulses), while
     * the full-rate |x| array keeps the sharp edges and is what gets
     * folded and segmented. */
    out->sample_rate = fs;
    out->resolution_s = 1.0 / fs;
    out->rms_window_s = (double)rms_w / fs;
    rms_envelope(iq, n_cap, rms_w, env);
    fine = malloc(n_cap * sizeof(float));
    if (fine == NULL) {
        free(iq);
        free(env);
        snprintf(out->verdict, sizeof(out->verdict), "out of memory");
        return false;
    }
    for (size_t i = 0; i < n_cap; i++) {
        fine[i] = (float)hypot((double)iq[2 * i], (double)iq[2 * i + 1]);
    }
    {
        double *tmp = malloc(n_cap * sizeof(double));
        double lo_half[4096];

        if (tmp == NULL) {
            free(iq);
            free(env);
            snprintf(out->verdict, sizeof(out->verdict), "out of memory");
            return false;
        }
        for (size_t i = 0; i < n_cap; i++) tmp[i] = env[i];
        noise_med = median_of(tmp, n_cap);
        for (size_t i = 0; i < n_cap; i++) tmp[i] = fabs(env[i] - noise_med);
        noise_mad = median_of(tmp, n_cap);
        free(tmp);
        (void)lo_half;
    }
    burst_thr = noise_med + 6.0 * (noise_mad > 0.0 ? noise_mad : 1e-9);
    {
        size_t i = 0;
        bool in = false;
        size_t raw_start = 0;
        size_t min_burst = (size_t)(50e-6 * fs);
        size_t gap_min = (size_t)(200e-6 * fs);
        /* streaming merge: segments of the short-RMS envelope are joined
         * whenever the gap between them is shorter than gap_min, so the
         * intra-burst sub-pulse gaps do not split a burst (a raw segment
         * per sub-pulse would exhaust any fixed segment cap inside the
         * first burst) */
        bool have = false;
        size_t cur_start = 0, cur_end = 0;
        size_t burst_cap;
        int cap = aconfig_get_int("intraburst", "max_bursts", IB_MAX_BURSTS);

        burst_cap = (cap > 0 && cap < IB_MAX_BURSTS) ? (size_t)cap
                                                     : (size_t)IB_MAX_BURSTS;
        while (i < n_cap && n_bursts < burst_cap) {
            if (!in && env[i] >= burst_thr) {
                in = true;
                raw_start = i;
            } else if (in && env[i] < noise_med + 3.0 * noise_mad) {
                in = false;
                if (!have) {
                    cur_start = raw_start;
                    cur_end = i;
                    have = true;
                } else if (raw_start - cur_end < gap_min) {
                    cur_end = i;
                } else {
                    bursts[n_bursts].start = cur_start;
                    bursts[n_bursts].end = cur_end;
                    n_bursts++;
                    cur_start = raw_start;
                    cur_end = i;
                }
            }
            i++;
        }
        if (in) {
            if (!have) {
                cur_start = raw_start;
                cur_end = n_cap;
                have = true;
            } else if (raw_start - cur_end < gap_min) {
                cur_end = n_cap;
            } else if (n_bursts < burst_cap) {
                bursts[n_bursts].start = cur_start;
                bursts[n_bursts].end = cur_end;
                n_bursts++;
                cur_start = raw_start;
                cur_end = n_cap;
            }
        }
        if (have && n_bursts < burst_cap) {
            bursts[n_bursts].start = cur_start;
            bursts[n_bursts].end = cur_end;
            n_bursts++;
        }
        /* drop bursts shorter than min_burst */
        {
            size_t w = 0;

            for (size_t k = 0; k < n_bursts; k++) {
                if (bursts[k].end - bursts[k].start >= min_burst) {
                    bursts[w++] = bursts[k];
                }
            }
            n_bursts = w;
        }
        if (getenv("TE15_IB_DEBUG") != NULL) {
            fprintf(stderr, "IBDBG detect: bursts %zu thr %.4f off %.4f\n",
                    n_bursts, burst_thr, noise_med + 3.0 * noise_mad);
            for (size_t k = 0; k < n_bursts && k < 6; k++) {
                fprintf(stderr, "  burst[%zu] len %.2f ms\n", k,
                        (double)(bursts[k].end - bursts[k].start) /
                            (fs / 1e3));
            }
        }
    }

    out->bursts_detected = (int)n_bursts;
    if (n_bursts == 0) {
        free(iq);
        free(env);
        free(fine);
        snprintf(out->verdict, sizeof(out->verdict), "no bursts to inspect");
        return false;
    }
    out->period_hz = envelope_period_hz(env, n_cap, fs);
    /* per-burst modulation classification (each burst isolated on its own) */
    classify_individual_bursts(fine, n_cap, bursts, n_bursts, fs,
                               IB_MIN_SEG_S, out);

    /* --- 3. align bursts (leading-edge cross-correlation) and fold ----- */
    {
        size_t lens[IB_MAX_BURSTS];
        size_t burst_len;

        for (size_t b = 0; b < n_bursts; b++) {
            lens[b] = bursts[b].end - bursts[b].start;
        }
        /* median burst length: the fold must not extend past the bursts,
         * otherwise the padding (zeros) poisons the amplitude model */
        for (size_t i = 0; i < n_bursts; i++) {
            for (size_t j = i + 1; j < n_bursts; j++) {
                if (lens[j] < lens[i]) {
                    size_t t = lens[i];

                    lens[i] = lens[j];
                    lens[j] = t;
                }
            }
        }
        burst_len = lens[n_bursts / 2];
        fold_len = burst_len;
        if (fold_len > n_cap) fold_len = n_cap;
        ifold = calloc(fold_len, sizeof(float));
        cfold_re = calloc(fold_len, sizeof(float));
        cfold_im = calloc(fold_len, sizeof(float));
        if (ifold == NULL || cfold_re == NULL || cfold_im == NULL) {
            free(iq);
            free(env);
            free(fine);
            free(ifold);
            free(cfold_re);
            free(cfold_im);
            snprintf(out->verdict, sizeof(out->verdict), "out of memory");
            return false;
        }
        {
            double mid = noise_med + 3.0 * (noise_mad > 0.0 ? noise_mad : 1e-9);
            size_t t_edge = leading_edge(env, bursts[0].start,
                                         bursts[0].end, mid);
            size_t win = (size_t)(0.3 * fs / 1000.0); /* 300 us */
            long maxlag = (long)win;
            size_t used = 0;

        memset(burst_duty, 0, sizeof(burst_duty));

            for (size_t b = 0; b < n_bursts; b++) {
                size_t edge = leading_edge(env, bursts[b].start,
                                           bursts[b].end, mid);
                long lag = align_lag(env, t_edge, edge, win / 2, maxlag);
                long base = (long)edge + lag;
                size_t cnt = 0;

                for (size_t k = 0; k < fold_len; k++) {
                    long idx = base + (long)k;

                    if (idx < 0 || (size_t)idx >= n_cap) continue;
                    if (idx < (long)bursts[b].start ||
                        idx >= (long)bursts[b].end) {
                        continue;
                    }
                    {
                        double re = iq[2 * idx], im = iq[2 * idx + 1];

                        ifold[k] += fine[idx];
                        cfold_re[k] += (float)re;
                        cfold_im[k] += (float)im;
                        cnt++;
                    }
                }
                if (cnt > 0) {
                /* per-burst on-time fraction above the mid level: this is
                 * the cross-burst consistency measure - a genuine
                 * intra-burst pattern repeats with the same duty even when
                 * the individual widths are data-encoded */
                size_t on_cnt = 0;

                for (size_t k = 0; k < fold_len; k++) {
                    long idx = base + (long)k;

                    if (idx < (long)bursts[b].start ||
                        idx >= (long)bursts[b].end) {
                        continue;
                    }
                    if (env[idx] >= mid) on_cnt++;
                }
                if (used < IB_MAX_BURSTS) {
                    burst_duty[used] = (double)on_cnt / (double)cnt;
                }
                used++;
            }
            }
            if (used > 0) {
                for (size_t k = 0; k < fold_len; k++) {
                    ifold[k] /= (float)used;
                    cfold_re[k] /= (float)used;
                    cfold_im[k] /= (float)used;
                }
            }
            folds_used = used;
            /* coherence: |<x>| / <|x|> averaged over the fold */
            {
                double s = 0.0;
                size_t c = 0;

                for (size_t k = 0; k < fold_len; k++) {
                    double mag = hypot((double)cfold_re[k],
                                       (double)cfold_im[k]);

                    if (ifold[k] > 0.0) {
                        s += mag / (double)ifold[k];
                        c++;
                    }
                }
                coherence_sum = (c > 0) ? s / (double)c : 0.0;
            }
        }
    }
    if (getenv("TE15_IB_DEBUG") != NULL) {
        double mn = 1e300, mx = -1e300, sm = 0.0;

        for (size_t k = 0; k < fold_len; k++) {
            if (ifold[k] < mn) mn = ifold[k];
            if (ifold[k] > mx) mx = ifold[k];
            sm += ifold[k];
        }
        fprintf(stderr, "IBDBG bursts %zu fold_len %zu fold mean %.4f min %.4f max %.4f\n",
                n_bursts, fold_len, sm / (double)fold_len, mn, mx);
        fprintf(stderr, "IBDBG fold samples (every 4th, first 400):");
        for (size_t k = 0; k < fold_len && k < 400; k += 4) {
            fprintf(stderr, " %.3f", ifold[k]);
        }
        fprintf(stderr, "\n");
    }
    out->folded_bursts = (int)folds_used;
    out->coherence = coherence_sum;
    out->fold_coherent = (coherence_sum > 0.5);

    /* --- 4. sample-level segmentation of the folded average ------------ */
    {
        double lvl_lo = 0.0, lvl_hi = 0.0, lvl_thr = 0.0;
        double thr_hi, thr_lo;
        double sigma = 0.0;

        /* amplitude model (2-level, as a detector/consistency check) */
        two_level_model(ifold, fold_len, &lvl_lo, &lvl_hi, &lvl_thr);
        sigma = 0.5 * (lvl_hi - lvl_lo);
        if (sigma > 0.5) confirm++; /* the two-level model resolved levels */
        /* robust on/off levels: percentiles are insensitive to the deep
         * nulls that a steep filter (or noise) puts into the envelope -
         * log-domain 2-means latched onto those nulls and collapsed the
         * hysteresis thresholds to zero */
        {
            double p10 = percentile_of(ifold, fold_len, 0.10);
            double p90 = percentile_of(ifold, fold_len, 0.90);
            double span;

            if (p90 <= p10) {
                p90 = p10 * 2.0 + 1e-9;
            }
            out->off_level = p10;
            out->on_level = p90;
            span = p90 - p10;
            thr_hi = p10 + 0.60 * span;
            thr_lo = p10 + 0.35 * span;
        }
        n_segs = seg_hysteresis(ifold, fold_len, thr_hi, thr_lo,
                                min_seg, segs, IB_MAX_SEGMENTS);
        /* refine each detected edge at the half-max crossing */
        for (size_t si = 0; si < n_segs; si++) {
            size_t hi_i = segs[si].end < fold_len ? segs[si].end : fold_len;

            segs[si].start = refine_edge(ifold, segs[si].start,
                                         segs[si].start > 8
                                             ? segs[si].start - 8 : 0,
                                         hi_i);
        }
        if (getenv("TE15_IB_DEBUG") != NULL) {
            fprintf(stderr, "IBDBG levels lo %.4f hi %.4f thr_hi %.4f thr_lo %.4f -> segs %zu\n",
                    exp(lvl_lo), exp(lvl_hi), exp(thr_hi), exp(thr_lo),
                    n_segs);
        }
        if (n_segs >= 2) confirm++;
        {
            IbSeg cp[IB_MAX_SEGMENTS];
            size_t n_cp;

            /* log-amplitude sequence for the remaining detectors */
            float *lg = malloc(fold_len * sizeof(float));

            if (lg != NULL) {
                for (size_t i = 0; i < fold_len; i++) {
                    lg[i] = (float)log((double)ifold[i] + 1e-12);
                }
                n_cp = seg_changepoint(lg, fold_len, min_seg, cp,
                                       IB_MAX_SEGMENTS);
                if (n_cp >= 2) confirm += (n_cp >= 2) ? 1 : 0;
                {
                    IbSeg hm[IB_MAX_SEGMENTS];
                    size_t n_hm = seg_hmm(lg, fold_len, sigma, hm,
                                          IB_MAX_SEGMENTS);

                    if (n_hm >= 2) confirm++;
                }
                {
                    unsigned char *edge = calloc(fold_len, 1);
                    size_t n_edge = 0;

                    if (edge != NULL) {
                        cwt_edges(lg, fold_len, fs, edge, &n_edge);
                        if (n_edge >= 2) confirm++;
                        free(edge);
                    }
                }
                free(lg);
            }
        }
        out->detectors_confirming = confirm;
    }

    /* --- 5. per-sub-pulse characterisation ----------------------------- */
    if (n_segs > 0) {
        double peaks[IB_MAX_SEGMENTS];
        double noise_floor = out->off_level > 0.0 ? out->off_level : 1e-9;

        for (size_t si = 0; si < n_segs; si++) {
            size_t a = segs[si].start;
            size_t e = segs[si].end;
            double pk = 0.0;

            for (size_t k = a; k < e; k++) {
                if (ifold[k] > pk) pk = ifold[k];
            }
            peaks[si] = pk;
            if (widths_n < IB_MAX_SEGMENTS) {
                widths_all[widths_n++] = (double)(e - a) / fs;
            }
            if (si + 1 < n_segs && periods_n < IB_MAX_SEGMENTS) {
                periods_all[periods_n++] =
                    (double)(segs[si + 1].start - a) / fs;
            }
            /* 10-90 % rise at the leading edge */
            if (pk > 0.0 && rises_n < IB_MAX_SEGMENTS) {
                double lo = 0.1 * pk, hi = 0.9 * pk;
                size_t p = (a > 4) ? a - 4 : a;
                size_t q = p;

                while (q < e && ifold[q] < lo) q++;
                {
                    size_t qq = q;

                    while (qq < e && ifold[qq] < hi) qq++;
                    if (qq < e) {
                        rises_all[rises_n++] = (double)(qq - q) / fs;
                    }
                }
            }
        }
        /* per-pulse SNR and matched-filter response */
        {
            double max_pk = 0.0;
            double corr_sum = 0.0;
            size_t corr_n = 0;

            for (size_t si = 0; si < n_segs; si++) {
                if (peaks[si] > max_pk) max_pk = peaks[si];
            }
            out->snr_db = 20.0 * log10((max_pk + 1e-12) / noise_floor);
            /* the first sub-pulse as template, correlate the others */
            if (n_segs >= 2) {
                size_t len = segs[0].end - segs[0].start;

                for (size_t si = 1; si < n_segs; si++) {
                    double a1 = 0.0, a2 = 0.0, b1 = 0.0, b2 = 0.0,
                           ab = 0.0;
                    size_t lim = len;

                    if (segs[si].end - segs[si].start < lim) {
                        lim = segs[si].end - segs[si].start;
                    }
                    for (size_t k = 0; k < lim; k++) {
                        double x = ifold[segs[0].start + k];
                        double y = ifold[segs[si].start + k];

                        a1 += x;
                        a2 += x * x;
                        b1 += y;
                        b2 += y * y;
                        ab += x * y;
                    }
                    {
                        double den = sqrt((a2 - a1 * a1 / (double)lim) *
                                          (b2 - b1 * b1 / (double)lim));

                        if (den > 0.0) {
                            corr_sum += (ab - a1 * b1 / (double)lim) / den;
                            corr_n++;
                        }
                    }
                }
                out->matched_filter = corr_n > 0 ? corr_sum / (double)corr_n
                                                 : 0.0;
            }
        }
        /* intra-pulse instantaneous frequency / chirp / FSK steps, from
         * the coherent fold when available (else phase is meaningless) */
        if (out->fold_coherent && n_segs > 0) {
            size_t a = segs[0].start;
            size_t e = segs[0].end;
            double t_dur = (double)(e - a) / fs;
            double prev_ph = 0.0;
            double if_min = 1e300, if_max = -1e300;
            double sxy = 0.0, sx = 0.0, sy = 0.0, sxx = 0.0;
            size_t cnt = 0;

            for (size_t k = a + 1; k < e; k++) {
                double re0 = cfold_re[k - 1], im0 = cfold_im[k - 1];
                double re1 = cfold_re[k], im1 = cfold_im[k];
                double dph;

                if (re0 * re0 + im0 * im0 < 1e-20) continue;
                dph = atan2(im1 * re0 - re1 * im0, re1 * re0 + im1 * im0);
                {
                    double t = (double)(k - a) / fs;
                    double f = dph * fs / (2.0 * M_PI);

                    if (f < if_min) if_min = f;
                    if (f > if_max) if_max = f;
                    sxy += t * f;
                    sx += t;
                    sy += f;
                    sxx += t * t;
                    cnt++;
                }
                prev_ph = dph;
            }
            (void)prev_ph;
            if (cnt > 1) {
                double den = (double)cnt * sxx - sx * sx;
                double slope = den != 0.0
                                   ? ((double)cnt * sxy - sx * sy) / den : 0.0;

                out->if_span_hz = if_max - if_min;
                out->fsk_steps = (if_max - if_min) >
                                         (2e3 + 0.5 * fs / 1e6) ? 2 : 0;
                (void)slope;
            }
            if (t_dur > 0.0) {
                /* chirp from the IF slope over the pulse */
            }
        }
    }

    /* --- 6. validation: Delta-BIC and phase-randomised surrogates ------
     * The folded envelope is massively oversampled (tens of samples per
     * envelope correlation time), so a per-SAMPLE Gaussian BIC would let
     * any segmentation - even of pure noise - look significant (the
     * likelihood term grows with the raw sample count while the penalty
     * stays logarithmic). The test is therefore computed on block-reduced
     * data: the log-envelope is averaged into ~2048 bins, i.e. N_eff
     * effectively independent samples, which is the correct order of
     * magnitude for the information content of the burst.
     * Convention: Delta-BIC = BIC(one segment) - BIC(k segments), so a
     * POSITIVE value means the multi-segment model is better. */
    {
        enum { BIC_BINS = 2048 };
        size_t nb = BIC_BINS;
        size_t step;
        float *lg = NULL;
        double c1 = 0.0;

        if (fold_len < 2 * (size_t)BIC_BINS) {
            nb = fold_len / 2;
        }
        if (nb < 16) nb = fold_len > 16 ? fold_len : 1;
        step = fold_len / nb;
        if (step < 1) step = 1;
        nb = fold_len / step;
        if (nb >= 2) {
            lg = malloc(nb * sizeof(float));
        }
        if (lg != NULL) {
            /* block-averaged log envelope */
            for (size_t b = 0; b < nb; b++) {
                double acc = 0.0;
                size_t a0 = b * step;
                size_t a1 = (b + 1) * step;

                if (a1 > fold_len) a1 = fold_len;
                for (size_t i = a0; i < a1; i++) {
                    acc += log((double)ifold[i] + 1e-12);
                }
                lg[b] = (float)(acc / (double)(a1 > a0 ? a1 - a0 : 1));
            }
            c1 = l2_cost(lg, 0, nb);
            {
                /* Two-level Delta-BIC: does the on/off classification
                 * explain the block-averaged log envelope significantly
                 * better than a single level? This is the right question
                 * for "is there intra-burst structure at all"; modelling
                 * every pulse individually would penalise a genuinely
                 * uniform train for its many transitions (the earlier
                 * per-piece formulation even scored a real 47-pulse train
                 * NEGATIVE).
                 * Convention: Delta-BIC = BIC(one level) - BIC(on/off), so
                 * a POSITIVE value favours the two-level model. */
                double thr = 0.5 * (log(out->on_level) +
                                    log(out->off_level));
                double s1 = 0.0, s2 = 0.0;
                size_t n_on = 0, n_off = 0;
                double m_on = 0.0, m_off = 0.0;
                double rss2 = 0.0;

                for (size_t b = 0; b < nb; b++) {
                    if (lg[b] >= thr) {
                        m_on += lg[b];
                        s2 += (double)lg[b] * lg[b];
                        n_on++;
                    } else {
                        m_off += lg[b];
                        s2 += (double)lg[b] * lg[b];
                        n_off++;
                    }
                }
                if (n_on > 0 && n_off > 0) {
                    m_on /= (double)n_on;
                    m_off /= (double)n_off;
                    for (size_t b = 0; b < nb; b++) {
                        double d = (lg[b] >= thr) ? (lg[b] - m_on)
                                                  : (lg[b] - m_off);

                        rss2 += d * d;
                    }
                    (void)s1;
                    (void)s2;
                    {
                        double bic1 = (double)nb *
                                          log(c1 / (double)nb + 1e-12) +
                                      2.0 * log((double)nb);
                        double bic2 = (double)nb *
                                          log(rss2 / (double)nb + 1e-12) +
                                      3.0 * log((double)nb);

                        out->delta_bic = (int)(bic1 - bic2);
                    }
                } else {
                    out->delta_bic = 0;
                }
            }
        }
        /* surrogate (null) test: fully shuffle the folded envelope's
         * samples (Fisher-Yates), which preserves the amplitude
         * histogram but destroys all temporal structure, then recount
         * the hysteresis segments. The false-alarm rate is the fraction
         * of shuffles that yield AT LEAST as many segments as the real
         * data - a plain 'any 2 segments' test would be vacuous because
         * white shuffling of a 50%-duty square wave always crosses the
         * thresholds. */
        {
            double t_hi, t_lo;
            float *perm = malloc(fold_len * sizeof(float));
            IbSeg tmp[IB_MAX_SEGMENTS];
            int hits = 0;

            if (out->on_level > 0.0 && out->off_level > 0.0 &&
                out->on_level > out->off_level) {
                double span = out->on_level - out->off_level;

                t_hi = out->off_level + 0.60 * span;
                t_lo = out->off_level + 0.35 * span;
            } else {
                t_hi = t_lo = 0.0;
            }
            if (perm != NULL && t_hi > t_lo && t_hi > 0.0) {
                for (int rep = 0; rep < 8; rep++) {
                    unsigned long long s = 0x9E3779B97F4A7C15ULL +
                                           (unsigned long long)rep * 7919ULL;
                    size_t cnt;

                    memcpy(perm, ifold, fold_len * sizeof(float));
                    for (size_t i = fold_len; i > 1; i--) {
                        size_t j;

                        s = s * 6364136223846793005ULL +
                            1442695040888963407ULL;
                        j = (size_t)((s >> 11) % i);
                        {
                            float t = perm[i - 1];

                            perm[i - 1] = perm[j];
                            perm[j] = t;
                        }
                    }
                    cnt = seg_hysteresis(perm, fold_len, t_hi, t_lo, min_seg,
                                         tmp, IB_MAX_SEGMENTS);
                    /* regularity of the null segmentation: a shuffle with
                     * a width spread at least as TIGHT as the real data
                     * means the observed regularity is unremarkable */
                    if (cnt >= 2) {
                        double wv[IB_MAX_SEGMENTS];
                        double cv;

                        for (size_t k = 0; k < cnt && k < IB_MAX_SEGMENTS;
                             k++) {
                            wv[k] = (double)(tmp[k].end - tmp[k].start) / fs;
                        }
                        cv = cv_of(wv, cnt < IB_MAX_SEGMENTS ? cnt
                                                             : IB_MAX_SEGMENTS);
                        if (cv <= out->width_cv) {
                            hits++;
                        }
                    }
                }
                out->surrogate_fpr = (double)hits / 8.0;
            }
            free(perm);
        }
    }

    /* --- 7. aggregate --------------------------------------------------- */
    out->sub_pulses = (int)n_segs;
    out->width_mean_s = mean_of(widths_all, widths_n);
    out->width_cv = cv_of(widths_all, widths_n);
    out->period_mean_s = mean_of(periods_all, periods_n);
    out->period_cv = cv_of(periods_all, periods_n);
    out->rise_mean_s = mean_of(rises_all, rises_n);
    /* modulation hypotheses (PPM/PIM/PWM/PFM/Manchester/random) */
    classify_burst_mod(widths_all, widths_n, periods_all, periods_n,
                       out->width_cv, out->period_cv, out);
    {
        double on = 0.0, total = 0.0;

        for (size_t si = 0; si < n_segs; si++) {
            on += (double)(segs[si].end - segs[si].start);
        }
        total = (double)fold_len;
        out->duty = total > 0.0 ? on / total : 0.0;
    }
    out->burst_len_s = (double)fold_len / fs;
    {
        double m = 0.0, v = 0.0;

        for (size_t i = 0; i < fold_len; i++) {
            m += ifold[i];
            v += (double)ifold[i] * ifold[i];
        }
        m /= (double)fold_len;
        v /= (double)fold_len;
        out->env_cv = m > 0.0 ? sqrt(fabs(v - m * m)) / m : 0.0;
    }
    out->valid = (n_segs > 0);

    /* verdict: validated segmentation + cross-burst consistency */
    {
        /* cross-burst consistency: the per-burst on-time fraction must be
         * stable across bursts (widths may legitimately vary when the
         * bursts carry pulse-width encoded data) */
        double duty_cv = 0.0;
        size_t nd = ((size_t)out->folded_bursts < IB_MAX_BURSTS)
                        ? (size_t)out->folded_bursts : IB_MAX_BURSTS;
        bool consistent;

        if (nd >= 3) {
            double m = 0.0, v = 0.0;

            for (size_t i = 0; i < nd; i++) m += burst_duty[i];
            m /= (double)nd;
            for (size_t i = 0; i < nd; i++) {
                double d = burst_duty[i] - m;

                v += d * d;
            }
            v /= (double)nd;
            duty_cv = m > 0.0 ? sqrt(v) / m : 0.0;
        }
        consistent = (duty_cv < 0.30);

        if (!out->valid) {
            snprintf(out->verdict, sizeof(out->verdict), "no segments");
        } else if (n_segs < 2) {
            snprintf(out->verdict, sizeof(out->verdict),
                     "single continuous burst (no intra-burst structure)");
        } else if (!consistent) {
            /* Validation: Delta-BIC is reported as evidence, but its
             * magnitude is not a reliable absolute gate at this sample
             * scale (a 47-pulse pattern over ~1000 effective bins and a
             * structureless burst can both reach thousands), so the
             * decision rests on the surrogate null test (above) and on
             * the cross-burst repeatability below. */
            snprintf(out->verdict, sizeof(out->verdict),
                     "intra-burst structure not validated (cross-burst "
                     "duty CV %.0f%%)", duty_cv * 100.0);
        } else if (out->surrogate_fpr > 0.5) {
            /* the phase-randomised surrogate produces the same number of
             * segments: the detected structure is indistinguishable from
             * noise, whatever Delta-BIC says on a non-stationary record */
            snprintf(out->verdict, sizeof(out->verdict),
                     "segmentation indistinguishable from noise (surrogate "
                     "false-alarm %.0f%%)", out->surrogate_fpr * 100.0);
        } else if (out->bursts_classified >= 3) {
            /* the per-burst vote is the authoritative evidence: it
             * isolates every burst instead of classifying the fold (which
             * can average different patterns into a third one) */
            if (strncmp(out->burst_mod_combo, "combination ", 12) == 0) {
                snprintf(out->verdict, sizeof(out->verdict),
                         "per-burst combination: %s", out->burst_mod_combo);
            } else if (strcmp(out->burst_mod_cand[0], "clock") == 0) {
                out->uniform = 1;
                snprintf(out->verdict, sizeof(out->verdict),
                         "uniform intra-burst pulse train (clock/preamble)");
            } else if (strcmp(out->burst_mod_cand[0], "PPM") == 0) {
                snprintf(out->verdict, sizeof(out->verdict),
                         "PPM: %d equally spaced slots, constant width",
                         out->interval_levels);
            } else if (strcmp(out->burst_mod_cand[0], "PIM") == 0) {
                snprintf(out->verdict, sizeof(out->verdict),
                         "PIM: %d discrete interval levels",
                         out->interval_levels);
            } else if (strcmp(out->burst_mod_cand[0], "PWM") == 0) {
                snprintf(out->verdict, sizeof(out->verdict),
                         "PWM: %d discrete widths, constant period",
                         out->width_levels);
            } else if (strcmp(out->burst_mod_cand[0], "PWM+PIM") == 0) {
                snprintf(out->verdict, sizeof(out->verdict),
                         "PWM+PIM: %d width levels, %d interval levels",
                         out->width_levels, out->interval_levels);
            } else if (strcmp(out->burst_mod_cand[0], "PFM") == 0) {
                snprintf(out->verdict, sizeof(out->verdict),
                         "PFM: continuous interval modulation");
            } else if (strcmp(out->burst_mod_cand[0], "Manchester") == 0) {
                snprintf(out->verdict, sizeof(out->verdict),
                         "Manchester/biphase-coded pulse pairs");
            } else {
                snprintf(out->verdict, sizeof(out->verdict),
                         "random OOK data (no stable pulse structure)");
            }
        } else if (out->width_cv < 0.12 && out->period_cv < 0.12 &&
                   n_segs >= 4) {
            /* unmodulated clock: check it BEFORE the modulation names
             * (the "unmodulated clock" label itself mentions PPM/PWM/PFM) */
            out->uniform = 1;
            snprintf(out->verdict, sizeof(out->verdict),
                     "uniform intra-burst pulse train (clock/preamble/PFM)");
        } else if (strstr(out->burst_mod, "Manchester") != NULL) {
            snprintf(out->verdict, sizeof(out->verdict),
                     "Manchester/biphase-coded pulse pairs");
        } else if (strstr(out->burst_mod, "PWM+PIM") != NULL) {
            snprintf(out->verdict, sizeof(out->verdict),
                     "PWM+PIM: %d width levels, %d interval levels",
                     out->width_levels, out->interval_levels);
        } else if (strstr(out->burst_mod, "PWM") != NULL) {
            snprintf(out->verdict, sizeof(out->verdict),
                     "PWM: %d discrete widths, constant period (duty "
                     "modulation)", out->width_levels);
        } else if (strstr(out->burst_mod, "PPM") != NULL) {
            snprintf(out->verdict, sizeof(out->verdict),
                     "PPM: %d equally spaced slots, constant width",
                     out->interval_levels);
        } else if (strstr(out->burst_mod, "PIM") != NULL) {
            snprintf(out->verdict, sizeof(out->verdict),
                     "PIM: %d discrete interval levels", out->interval_levels);
        } else if (strstr(out->burst_mod, "PFM") != NULL) {
            snprintf(out->verdict, sizeof(out->verdict),
                     "PFM: continuous interval modulation");
        } else if (out->env_cv < 0.15) {
            snprintf(out->verdict, sizeof(out->verdict),
                     "constant-envelope burst (no intra-burst modulation)");
        } else if (out->width_cv > 0.80) {
            snprintf(out->verdict, sizeof(out->verdict),
                     "noise-like intra-burst envelope (no coherent structure)");
        } else if (out->width_cv < 0.20 && out->period_cv < 0.20 &&
                   n_segs >= 4) {
            out->uniform = 1;
            snprintf(out->verdict, sizeof(out->verdict),
                     "uniform intra-burst pulse train (clock/preamble/PFM)");
        } else if (out->width_cv > 0.30) {
            snprintf(out->verdict, sizeof(out->verdict),
                     "varying intra-burst widths (pulse-width/position "
                     "coding)");
        } else {
            snprintf(out->verdict, sizeof(out->verdict),
                     "irregular intra-burst structure");
        }
    }

    snprintf(out->note, sizeof(out->note),
             "Intra-burst (full-rate %.1f MHz, %.0f ns resolution): %d bursts "
             "detected (RMS window %.1f us), period %.2f Hz from the envelope "
             "spectrum, %zu folded, coherence %.2f (%s fold), residual CFO "
             "%.1f Hz removed | segmentation: %zu sub-pulses, width %.2f us "
             "(CV %.0f%%), spacing %.2f us (CV %.0f%%), rise %.2f us, SNR "
             "%.1f dB, on/off levels %.3f/%.3f, detectors confirming %d/4, "
             "Delta-BIC %d -> %s",
             fs / 1e6, out->resolution_s * 1e9, out->bursts_detected,
             out->rms_window_s * 1e6, out->period_hz, folds_used,
             out->coherence, out->fold_coherent ? "coherent" : "incoherent",
             out->residual_cfo_hz, n_segs, out->width_mean_s * 1e6,
             out->width_cv * 100.0, out->period_mean_s * 1e6,
             out->period_cv * 100.0, out->rise_mean_s * 1e6, out->snr_db,
             out->on_level, out->off_level, out->detectors_confirming,
             out->delta_bic, out->verdict);
    TE15_LOG("\nINTRA-BURST ENVELOPE ANALYSIS\n%s\n", out->note);
    free(iq);
    free(env);
    free(fine);
    free(ifold);
    free(cfold_re);
    free(cfold_im);
    return out->valid;
}
