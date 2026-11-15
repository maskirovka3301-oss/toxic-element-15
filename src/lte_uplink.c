/*
 * lte_uplink.c — LTE uplink PUSCH / PUCCH demodulator (see lte_uplink.h).
 *
 * Both channels use the LTE uplink numerology: 15 kHz subcarrier spacing
 * with subcarriers at (m + 1/2) * 15 kHz (no subcarrier at DC). The
 * receiver therefore resamples the selection to N * 15 kHz, synchronises on
 * the cyclic prefix, and removes the half-subcarrier offset per symbol with
 * a phase reset at the useful-symbol start (3GPP TS 36.211 §5.6).
 *
 *   PUSCH (TS 36.211 §5.3–5.5; D. Estévez, "Demodulation of the LTE
 *   uplink", 2022): SC-FDMA data symbols with an OFDMA Zadoff-Chu DMRS in
 *   the middle of each slot. The DMRS group/sequence (u, v) and cyclic
 *   shift are found by correlation; the DMRS gives a per-subcarrier channel
 *   estimate, the equalised data subcarriers are taken through an M-point
 *   IDFT (undoing the precoder) and the QAM constellation is demapped.
 *
 *   PUCCH (TS 36.211 §5.4, Table 5.5.1.2-1; D. Estévez, "Demodulation of
 *   LTE PUCCH", 2022): one resource block hopped between the two slots,
 *   formats 1/1a/1b (one BPSK control symbol spread with an orthogonal
 *   cover code) and 2/2a/2b (QPSK CQI symbols) on a computer-optimised QPSK
 *   base sequence. The base-sequence group is found by DFT correlation, the
 *   per-symbol cyclic shift is estimated from the peak lag, and the format
 *   is classified with the fourth-power metric.
 */
#include "lte_uplink.h"

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <fftw3.h>

#include "debug.h"

#define LTE_SCS_HZ 15000.0
#define LTE_CP0_RATIO (160.0 / 2048.0)   /* first symbol of a slot */
#define LTE_CP_RATIO  (144.0 / 2048.0)
#define LTE_SYMBOLS_PER_SLOT 7
#define LTE_MAX_SLOTS 16
#define LTE_MAX_SUBC 1200

const char *lte_channel_name(LteChannel ch)
{
    switch (ch) {
    case LTE_CH_PUSCH: return "PUSCH";
    case LTE_CH_PUCCH: return "PUCCH";
    case LTE_CH_AUTO:
    default: return "auto";
    }
}

/* PUCCH computer-optimised QPSK base sequences, TS 36.211 Table 5.5.1.2-1 */
static const signed char pucch_phi[30][12] = {
    {-1, 1, 3, -3, 3, 3, 1, 1, 3, 1, -3, 3},
    {1, 1, 3, 3, 3, -1, 1, -3, -3, 1, -3, 3},
    {1, 1, -3, -3, -3, -1, -3, -3, 1, -3, 1, -1},
    {-1, 1, 1, 1, 1, -1, -3, -3, 1, -3, 3, -1},
    {-1, 3, 1, -1, 1, -1, -3, -1, 1, -1, 1, 3},
    {1, -3, 3, -1, -1, 1, 1, -1, -1, 3, -3, 1},
    {-1, 3, -3, -3, -3, 3, 1, -1, 3, 3, -3, 1},
    {-3, -1, -1, -1, 1, -3, 3, -1, 1, -3, 3, 1},
    {1, -3, 3, 1, -1, -1, -1, 1, 1, 3, -1, 1},
    {1, -3, -1, 3, 3, -1, -3, 1, 1, 1, 1, 1},
    {-1, 3, -1, 1, 1, -3, -3, -1, -3, -3, 3, -1},
    {3, 1, -1, -1, 3, 3, -3, 1, 3, 1, 3, 3},
    {1, -3, 1, 1, -3, 1, 1, 1, -3, -3, -3, 1},
    {3, 3, -3, 3, -3, 1, 1, 3, -1, -3, 3, 3},
    {-3, 1, -1, -3, -1, 3, 1, 3, 3, 3, -1, 1},
    {3, -1, 1, -3, -1, -1, 1, 1, 3, 1, -1, -3},
    {1, 3, 1, -1, 1, 3, 3, 3, -1, -1, 3, -1},
    {-3, 1, 1, 3, -3, 3, -3, -3, 3, 1, 3, -1},
    {-3, 3, 1, 1, -3, 1, -3, -3, -1, -1, 1, -3},
    {-1, 3, 1, 3, 1, -1, -1, 3, -3, -1, -3, -1},
    {-1, -3, 1, 1, 1, 1, 3, 1, -1, 1, -3, -1},
    {-1, 3, -1, 1, -3, -3, -3, -3, -3, 1, -1, -3},
    {1, 1, -3, -3, -3, -3, -1, 3, -3, 1, -3, 3},
    {1, 1, -1, -3, -1, -3, 1, -1, 1, 3, -1, 1},
    {1, 1, 3, 1, 3, 3, -1, 1, -1, -3, -3, 1},
    {1, -3, 3, 3, 1, 3, 3, 1, -3, -1, -1, 3},
    {1, 3, -3, -3, 3, -3, 1, -1, -1, 3, -1, -3},
    {-3, -1, -3, -1, -3, 3, 1, -1, 1, 3, -3, -3},
    {-1, 3, -3, 3, -1, 3, 3, -3, 3, 3, -1, -1},
    {3, -3, -3, -1, -1, -3, -1, 3, -3, 3, 1, -1},
};


static float complex pucch_base(int u, int n);

int lte_pucch_phi(int u, int n)
{
    if (u < 0 || u > 29 || n < 0 || n > 11) return 0;
    return (int)pucch_phi[u][n];
}

float lte_pucch_r_re(int u, int n)
{
    return crealf(pucch_base(u, n));
}

float lte_pucch_r_im(int u, int n)
{
    return cimagf(pucch_base(u, n));
}

/* r_u(n) = exp(i phi_u(n) pi / 4) */
static float complex pucch_base(int u, int n)
{
    return cexpf(I * (float)pucch_phi[u][n] * (float)M_PI / 4.0f);
}

/* ------------------------------------------------------------------ */
/* small helpers                                                      */
/* ------------------------------------------------------------------ */

static bool is_prime(int n)
{
    if (n < 2) return false;
    for (int i = 2; (long)i * i <= n; i++) {
        if (n % i == 0) return false;
    }
    return true;
}

static int largest_prime_below(int n)
{
    for (int p = n - 1; p > 2; p--) {
        if (is_prime(p)) return p;
    }
    return 3;
}

/* Zadoff-Chu sequence x_q(n), periodic in N_ZC (TS 36.211 §5.5.1.1). */
static float complex zc_value(int q, int nzc, int n)
{
    int m = ((n % nzc) + nzc) % nzc;

    return cexpf(-I * (float)M_PI * (float)q * (float)m * (float)(m + 1) /
                 (float)nzc);
}

/* DFT of arbitrary length; dir > 0 is the forward transform. */
static void dft(const float complex *in, float complex *out, int n, int dir)
{
    fftwf_complex *a = fftwf_malloc(sizeof(fftwf_complex) * (size_t)n);
    fftwf_complex *b = fftwf_malloc(sizeof(fftwf_complex) * (size_t)n);
    fftwf_plan p;

    if (a == NULL || b == NULL) {
        fftwf_free(a);
        fftwf_free(b);
        memset(out, 0, (size_t)n * sizeof(*out));
        return;
    }
    for (int i = 0; i < n; i++) a[i] = in[i];
    p = fftwf_plan_dft_1d(n, a, b, dir > 0 ? FFTW_FORWARD : FFTW_BACKWARD,
                          FFTW_ESTIMATE);
    fftwf_execute(p);
    fftwf_destroy_plan(p);
    for (int i = 0; i < n; i++) out[i] = b[i];
    fftwf_free(a);
    fftwf_free(b);
}

/* Sample index of the start of the cyclic prefix of slot symbol `l`. */
static int symbol_cp_start(int slot_base, int n, int cp0, int cp, int l)
{
    if (l == 0) return slot_base;
    return slot_base + cp0 + n + (l - 1) * (cp + n);
}

/* FFT the useful part of one OFDM symbol with the uplink half-subcarrier
 * offset removed (phase reset at the useful-symbol start) and the carrier
 * offset corrected. */
static void ofdm_symbol(const float complex *x, int cp_start, int cp, int n,
                        double cfo_hz, double fs, float complex *out)
{
    float complex *buf = malloc((size_t)n * sizeof(*buf));

    if (buf == NULL) {
        memset(out, 0, (size_t)n * sizeof(*out));
        return;
    }
    for (int t = 0; t < n; t++) {
        float complex half = cexpf(-I * (float)M_PI * (float)t / (float)n);
        float complex cfo = cexpf(-I * 2.0f * (float)M_PI *
                                  (float)cfo_hz * (float)t / (float)fs);

        buf[t] = x[cp_start + cp + t] * half * cfo;
    }
    dft(buf, out, n, 1);
    free(buf);
}

/* Average magnitude spectrum over the seven symbols of a slot. */
static void slot_spectrum(const float complex *x, int slot_base, int n,
                          int cp0, int cp, double cfo, double fs,
                          float *power)
{
    float complex *X = malloc((size_t)n * sizeof(*X));

    if (X == NULL) return;
    for (int l = 0; l < LTE_SYMBOLS_PER_SLOT; l++) {
        int cs = symbol_cp_start(slot_base, n, cp0, cp, l);
        int cpl = l == 0 ? cp0 : cp;

        ofdm_symbol(x, cs, cpl, n, cfo, fs, X);
        for (int b = 0; b < n; b++) power[b] += cabsf(X[b]);
    }
    free(X);
}

/* Constellation slicing: nearest normalised QAM point (unit average power)
 * plus its Gray-coded bits. */
static float complex qam_slice(float complex z, int order, unsigned *bits)
{
    if (order <= 1) {
        *bits = crealf(z) < 0.0f ? 1u : 0u;
        return crealf(z) < 0.0f ? -1.0f : 1.0f;
    }
    if (order == 2) {
        int ri = crealf(z) < 0.0f ? 1 : 0;
        int ii = cimagf(z) < 0.0f ? 1 : 0;

        /* QPSK points (1+i)/sqrt2 with bit0 = real, bit1 = imag */
        *bits = (unsigned)((ri << 1) | ii);
        return (ri ? -0.70710678f : 0.70710678f) +
               I * (ii ? -0.70710678f : 0.70710678f);
    }
    {
        int bits_axis = order / 2;
        int levels = 1 << bits_axis;
        float step = 2.0f / (float)(levels - 1);
        float best_r = 1e9f, best_i = 1e9f;
        int ri = 0, ii = 0;

        for (int k = 0; k < levels; k++) {
            float v = -1.0f + step * (float)k;

            if (fabsf(crealf(z) - v) < best_r) {
                best_r = fabsf(crealf(z) - v);
                ri = k;
            }
            if (fabsf(cimagf(z) - v) < best_i) {
                best_i = fabsf(cimagf(z) - v);
                ii = k;
            }
        }
        *bits = (unsigned)((ri << bits_axis) | ii);
        return (-1.0f + step * (float)ri) + I * (-1.0f + step * (float)ii);
    }
}

static void pack_bits(LteDemod *out, const unsigned *values, size_t count,
                      int order)
{
    for (size_t i = 0; i < count; i++) {
        for (int b = order - 1; b >= 0; b--) {
            size_t idx = out->n_bits;

            if (idx >= LTE_MAX_BITS) return;
            if ((values[i] >> b) & 1u) {
                out->bits[idx >> 3] |= (unsigned char)(1u << (idx & 7u));
            }
            out->n_bits++;
        }
    }
}

/* Linear resampling that keeps absolute frequencies (no mixing needed: the
 * FFT bins are offsets from the recording centre). */
static size_t resample(const float *in, size_t n_in, size_t n_out,
                       float complex *out)
{
    for (size_t i = 0; i < n_out; i++) {
        double pos = (double)i * (double)n_in / (double)n_out;
        size_t k = (size_t)pos;
        double frac = pos - (double)k;

        if (k + 1 >= n_in) break;
        out[i] = (in[2 * k] + I * in[2 * k + 1]) * (1.0 - frac) +
                 (in[2 * (k + 1)] + I * in[2 * (k + 1) + 1]) * frac;
    }
    return n_out;
}

/* ------------------------------------------------------------------ */
/* Analysis hint                                                      */
/* ------------------------------------------------------------------ */

void lte_hint_clear(LteHint *hint)
{
    if (hint != NULL) memset(hint, 0, sizeof(*hint));
}

static int mod_order_from_text(const char *text)
{
    if (text == NULL) return 0;
    if (strstr(text, "64QAM") != NULL || strstr(text, "64-QAM") != NULL) return 6;
    if (strstr(text, "16QAM") != NULL || strstr(text, "16-QAM") != NULL) return 4;
    if (strstr(text, "QPSK") != NULL) return 2;
    if (strstr(text, "BPSK") != NULL) return 1;
    return 0;
}

bool lte_hint_from_analysis(LteHint *hint, const RankedMatch *ids, size_t n,
                            double measured_bw_hz, const Metrics *m,
                            const ModCandidate *mods, size_t nmods,
                            const Region *region)
{
    const RankedMatch *best = NULL;

    if (hint == NULL) return false;
    lte_hint_clear(hint);
    /* the highest-scoring ranked match that is an LTE uplink channel */
    for (size_t i = 0; i < n; i++) {
        const KnownSignal *sig = ids[i].signal;

        if (sig == NULL) continue;
        /* only the LTE uplink channels: the exhaustive database also
         * carries 5G NR PUSCH/PUCCH signatures, which are not LTE */
        if (strncmp(sig->name, "LTE ", 4) != 0) continue;
        if (strstr(sig->name, "PUSCH") == NULL &&
            strstr(sig->name, "PUCCH") == NULL) {
            continue;
        }
        if (best == NULL || ids[i].total > best->total ||
            (ids[i].total == best->total && ids[i].valid && !best->valid)) {
            best = &ids[i];
        }
    }
    if (best == NULL) return false;
    hint->valid = true;
    hint->from_analysis = true;
    hint->channel = strstr(best->signal->name, "PUCCH") != NULL
                        ? LTE_CH_PUCCH : LTE_CH_PUSCH;
    hint->band_min_hz = best->signal->freq_min;
    hint->band_max_hz = best->signal->freq_max;
    /* band label: the part of the name after the channel keyword */
    {
        const char *name = best->signal->name;
        const char *kw = hint->channel == LTE_CH_PUCCH ? "PUCCH" : "PUSCH";
        const char *p = strstr(name, kw);

        if (p != NULL) {
            p += strlen(kw);
            while (*p == ' ') p++;
            snprintf(hint->band, sizeof(hint->band), "%s", p);
        }
    }
    /* measured occupied bandwidth (fall back to the band width) */
    hint->bandwidth_hz = measured_bw_hz > 0.0 ? measured_bw_hz
                                              : hint->band_max_hz -
                                                hint->band_min_hz;
    hint->center_hz = 0.5 * (hint->band_min_hz + hint->band_max_hz);
    if (region != NULL) {
        hint->center_hz = 0.5 * (region->freq_min + region->freq_max);
        hint->time_min = region->time_min;
        hint->time_max = region->time_max;
    }
    /* resource-block estimate from the measured bandwidth */
    if (hint->bandwidth_hz > 0.0) {
        int rb = (int)llround(hint->bandwidth_hz / 180000.0);
        int rb_max = hint->channel == LTE_CH_PUCCH ? 4 : 100;

        if (rb < 1) rb = 1;
        if (rb > rb_max) rb = rb_max;
        hint->rb_estimate = rb;
    }
    /* modulation: the analysis' ranked modulation class first */
    hint->mod_order = 0;
    for (size_t i = 0; i < nmods && i < 3 && hint->mod_order == 0; i++) {
        hint->mod_order = mod_order_from_text(mods[i].name);
        if (hint->mod_order == 0) {
            hint->mod_order = mod_order_from_text(mods[i].why);
        }
    }
    if (hint->mod_order == 0 && m != NULL) {
        hint->mod_order = mod_order_from_text(m->mod_class);
    }
    switch (hint->mod_order) {
    case 1: snprintf(hint->mod_name, sizeof(hint->mod_name), "%s", "BPSK");
        break;
    case 2: snprintf(hint->mod_name, sizeof(hint->mod_name), "%s", "QPSK");
        break;
    case 4: snprintf(hint->mod_name, sizeof(hint->mod_name), "%s", "16QAM");
        break;
    case 6: snprintf(hint->mod_name, sizeof(hint->mod_name), "%s", "64QAM");
        break;
    default: hint->mod_name[0] = '\0'; break;
    }
    snprintf(hint->source, sizeof(hint->source),
             "analyse: %s in band %s (%.3f-%.3f MHz), %.1f kHz occupied "
             "(~%d RB)%s%s", hint->channel == LTE_CH_PUCCH ? "PUCCH" : "PUSCH",
             hint->band[0] != '\0' ? hint->band : "?",
             hint->band_min_hz / 1e6, hint->band_max_hz / 1e6,
             hint->bandwidth_hz / 1e3, hint->rb_estimate,
             hint->mod_name[0] != '\0' ? ", " : "",
             hint->mod_name[0] != '\0' ? hint->mod_name : "");
    return true;
}

/* ------------------------------------------------------------------ */
/* PUSCH                                                              */
/* ------------------------------------------------------------------ */

/* Normalised DMRS correlation for (u, v): 1 for the matching sequence,
 * about 1/M otherwise. */
static double dmrs_metric(const float complex *X, int b0, int M, int u, int v,
                          int *lag)
{
    int nzc = largest_prime_below(M);
    double qbar = (double)nzc * (double)(u + 1) / 31.0;
    int q = (int)floor(qbar + 0.5) +
            ((int)floor(2.0 * qbar) % 2 == 0 ? v : -v);
    float complex *z = malloc((size_t)M * sizeof(*z));
    float complex *Z = malloc((size_t)M * sizeof(*Z));
    double peak = 0.0, pw = 0.0;
    int best_l = 0;

    if (z == NULL || Z == NULL) {
        free(z);
        free(Z);
        return 0.0;
    }
    for (int n = 0; n < M; n++) {
        z[n] = X[b0 + n] * conjf(zc_value(q, nzc, n));
        pw += (double)(crealf(X[b0 + n]) * crealf(X[b0 + n]) +
                       cimagf(X[b0 + n]) * cimagf(X[b0 + n]));
    }
    dft(z, Z, M, 1);
    for (int l = 0; l < M; l++) {
        double p = (double)(crealf(Z[l]) * crealf(Z[l]) +
                            cimagf(Z[l]) * cimagf(Z[l]));

        if (p > peak) {
            peak = p;
            best_l = l;
        }
    }
    free(z);
    free(Z);
    if (pw <= 0.0) return 0.0;
    if (lag != NULL) *lag = best_l;
    return peak / ((double)M * pw);
}

/* Blind modulation classification: the lowest order that already explains
 * the constellation within the measurement noise wins. (A pure EVM minimum
 * is degenerate: a 64QAM constellation also contains every QPSK point, so
 * its residual is never larger.) */
static int best_modulation(const float complex *z, size_t n, double *evm_out,
                           int prefer_order)
{
    static const int orders[3] = {2, 4, 6};
    double evm[3];

    for (int i = 0; i < 3; i++) {
        double num = 0.0, den = 0.0;

        for (size_t k = 0; k < n; k++) {
            unsigned bits;
            float complex ref = qam_slice(z[k], orders[i], &bits);
            float dr = crealf(z[k]) - crealf(ref);
            float di = cimagf(z[k]) - cimagf(ref);

            num += (double)(dr * dr + di * di);
            den += (double)(crealf(z[k]) * crealf(z[k]) +
                            cimagf(z[k]) * cimagf(z[k]));
        }
        evm[i] = sqrt(num / (den + 1e-12));
    }
    /* a hint from the analysis is trusted when its constellation fits the
     * recovered symbols with a reasonable margin */
    if (prefer_order > 0) {
        for (int i = 0; i < 3; i++) {
            if (orders[i] == prefer_order && evm[i] < 0.25) {
                if (evm_out != NULL) *evm_out = evm[i];
                return prefer_order;
            }
        }
    }
    for (int i = 0; i < 3; i++) {
        if (evm[i] < 0.12) {
            if (evm_out != NULL) *evm_out = evm[i];
            return orders[i];
        }
    }
    if (evm_out != NULL) *evm_out = evm[2];
    return 6;
}

static bool pusch_demod(const float complex *x, int n_slots, int slot_len,
                        int n, int cp0, int cp, double fs, double f_lo,
                        double f_hi, LteDemod *out, double *quality,
                        int expect_subc, int prefer_order)
{
    float *power = calloc((size_t)n, sizeof(float));
    float complex *X = NULL;
    float complex *H = NULL;
    float complex *eq = NULL;
    float complex *Y = NULL;
    float complex *cloud = NULL;
    unsigned *vals = NULL;
    size_t n_cloud = 0, n_vals = 0, cloud_cap;
    int b0 = 0, M = 0, u = 0, v = 0, alpha = 0, order = 2;
    double evm = 0.0;
    bool ok = false;
    int bin_lo, bin_hi;

    if (power == NULL) return false;
    slot_spectrum(x, 0, n, cp0, cp, 0.0, fs, power);
    X = malloc((size_t)n * sizeof(*X));
    if (X == NULL) {
        free(power);
        return false;
    }
    bin_lo = (int)floor(f_lo / LTE_SCS_HZ - 0.5);
    bin_hi = (int)ceil(f_hi / LTE_SCS_HZ - 0.5);
    {
        float peak = 0.0f;

        for (int k = 0; k < n; k++) if (power[k] > peak) peak = power[k];
        if (peak <= 0.0f) {
            free(power);
            free(X);
            return false;
        }
        /* Detected span, then the resource-block alignment and the DMRS
         * sequence are chosen jointly: the alignment whose extracted DMRS
         * correlates best with a Zadoff-Chu sequence is the true one. */
        {
            int lo = bin_lo, hi = bin_hi;
            double best_score = -1.0;
            int best_lag = 0, best_u = 0, best_v = 0;

            for (int k = bin_lo; k < bin_hi; k++) {
                if (power[((k % n) + n) % n] > 0.08f * peak) {
                    lo = k;
                    break;
                }
            }
            for (int k = bin_hi; k > bin_lo; k--) {
                if (power[(((k - 1) % n) + n) % n] > 0.08f * peak) {
                    hi = k;
                    break;
                }
            }
            if (hi - lo < 12) hi = lo + 12;
            {
                int cs_dmrs = symbol_cp_start(0, n, cp0, cp, 3);
                double best_close = -1.0;
                int close_b0 = 0, close_M = 0, close_u = 0, close_v = 0;
                int close_lag = 0;

                ofdm_symbol(x, cs_dmrs, cp, n, 0.0, fs, X);
                for (int a = 0; a < 12; a++) {
                    int l2 = ((lo - a) / 12) * 12 + a;
                    int h2 = ((hi - a + 11) / 12) * 12 + a;
                    int m2 = h2 - l2;
                    bool near_expected = expect_subc > 0 &&
                        abs(m2 - expect_subc) <=
                        (expect_subc / 5 > 12 ? expect_subc / 5 : 12);

                    if (m2 < 12 || m2 > LTE_MAX_SUBC) continue;
                    if (l2 < bin_lo - 12 || h2 > bin_hi + 12) continue;
                    for (int cv = 0; cv < 2; cv++) {
                        for (int cu = 0; cu < 30; cu++) {
                            int lag = 0;
                            double m = dmrs_metric(X, l2, m2, cu, cv, &lag);

                            if (m > best_score) {
                                best_score = m;
                                b0 = l2;
                                M = m2;
                                best_u = cu;
                                best_v = cv;
                                best_lag = lag;
                            }
                            if (near_expected && m > best_close) {
                                best_close = m;
                                close_b0 = l2;
                                close_M = m2;
                                close_u = cu;
                                close_v = cv;
                                close_lag = lag;
                            }
                        }
                    }
                }
                /* the analysis' measured allocation wins when its DMRS
                 * correlation is (almost) as good as the global best */
                if (best_close > 0.0 && best_close >= 0.7 * best_score) {
                    best_score = best_close;
                    b0 = close_b0;
                    M = close_M;
                    best_u = close_u;
                    best_v = close_v;
                    best_lag = close_lag;
                }
            }
            if (best_score < 0.5 || M < 12) {
                free(power);
                free(X);
                return false;
            }
            u = best_u;
            v = best_v;
            alpha = best_lag % 12;
        }
    }
    H = malloc((size_t)M * sizeof(*H));
    eq = malloc((size_t)M * sizeof(*eq));
    Y = malloc((size_t)M * sizeof(*Y));
    cloud_cap = (size_t)n_slots * 6 * (size_t)M;
    cloud = malloc(cloud_cap * sizeof(*cloud));
    vals = malloc(cloud_cap * sizeof(*vals));
    if (H == NULL || eq == NULL || Y == NULL || cloud == NULL || vals == NULL) {
        free(power);
        free(X);
        free(H);
        free(eq);
        free(Y);
        free(cloud);
        free(vals);
        return false;
    }
    /* DMRS: symbol 3 of every slot. The sequence group (u, v) is common to
     * the slots while the cyclic shift changes per slot, so the channel is
     * estimated once per slot from that slot's own DMRS. */
    {
        double best_metric = 0.0;
        int best_u = 0, best_v = 0, first_lag = 0;

        for (int s = 0; s < n_slots; s++) {
            int cs = symbol_cp_start(s * slot_len, n, cp0, cp, 3);
            int lag = 0;

            ofdm_symbol(x, cs, cp, n, 0.0, fs, X);
            for (int k = 0; k < M && k < 8; k++) {
                TE15_LOG(" %.1f", (double)cabsf(X[b0 + k]));
            }
            TE15_LOG("\n");
            for (int cv = 0; cv < 2; cv++) {
                for (int cu = 0; cu < 30; cu++) {
                    double m = dmrs_metric(X, b0, M, cu, cv, &lag);

                    if (m > best_metric) {
                        best_metric = m;
                        best_u = cu;
                        best_v = cv;
                        first_lag = lag;
                    }
                }
            }
        }
        if (best_metric < 0.35) {
            free(power);
            free(X);
            free(H);
            free(eq);
            free(Y);
            free(cloud);
            free(vals);
            return false;
        }
        u = best_u;
        v = best_v;
        alpha = first_lag % 12;
        if (quality != NULL) *quality = best_metric;
        for (int s = 0; s < n_slots; s++) {
            int cs = symbol_cp_start(s * slot_len, n, cp0, cp, 3);
            int nzc = largest_prime_below(M);
            double qbar = (double)nzc * (double)(u + 1) / 31.0;
            int q = (int)floor(qbar + 0.5) +
                    ((int)floor(2.0 * qbar) % 2 == 0 ? v : -v);
            int lag = 0;

            ofdm_symbol(x, cs, cp, n, 0.0, fs, X);
            (void)dmrs_metric(X, b0, M, u, v, &lag);
            for (int k = 0; k < M; k++) {
                float complex ref =
                    cexpf(I * 2.0f * (float)M_PI * (float)lag * (float)k /
                          (float)M) *
                    zc_value(q, nzc, k);

                H[k] = X[b0 + k] * conjf(ref);
            }
            /* equalise and undo the precoder for this slot's data symbols */
            for (int l = 0; l < LTE_SYMBOLS_PER_SLOT; l++) {
                int cpl = l == 0 ? cp0 : cp;

                if (l == 3) continue;
                cs = symbol_cp_start(s * slot_len, n, cp0, cp, l);
                ofdm_symbol(x, cs, cpl, n, 0.0, fs, X);
                for (int k = 0; k < M; k++) {
                    float hp = crealf(H[k]) * crealf(H[k]) +
                               cimagf(H[k]) * cimagf(H[k]);

                    eq[k] = X[b0 + k] * conjf(H[k]) / (hp + 1e-12f);
                }
                dft(eq, Y, M, -1);
                {
                    double pw = 0.0;

                    for (int k = 0; k < M; k++) {
                        pw += (double)(crealf(Y[k]) * crealf(Y[k]) +
                                       cimagf(Y[k]) * cimagf(Y[k]));
                    }
                    pw = sqrt(pw / (double)M);
                    if (pw < 1e-9) pw = 1.0;
                    for (int k = 0; k < M; k++) {
                        if (n_cloud < cloud_cap) {
                            cloud[n_cloud++] = Y[k] / (float)pw;
                        }
                    }
                }
            }
        }
    }
    order = best_modulation(cloud, n_cloud, &evm, prefer_order);
    for (size_t k = 0; k < n_cloud; k++) {
        unsigned bits;
        (void)qam_slice(cloud[k], order, &bits);
        vals[n_vals++] = bits;
    }
    pack_bits(out, vals, n_vals, order);
    out->channel = LTE_CH_PUSCH;
    out->channel_name = "PUSCH";
    out->fft_size = n;
    out->cp_first = cp0;
    out->cp_normal = cp;
    out->slots = n_slots;
    out->subcarriers = M;
    out->rb_count = M / 12;
    out->bw_hz = (double)M * LTE_SCS_HZ;
    out->mod_order = order;
    snprintf(out->mod_name, sizeof(out->mod_name), "%s",
             order == 2 ? "QPSK" : (order == 4 ? "16QAM" : "64QAM"));
    out->dmrs_u = u;
    out->dmrs_v = v;
    out->dmrs_alpha = alpha;
    out->evm_pct = 100.0 * evm;
    out->snr_db = evm > 0.0 ? -20.0 * log10(evm + 1e-6) : 40.0;
    /* constellation points (normalised) */
    for (size_t k = 0; k < n_cloud && out->n_points < LTE_MAX_POINTS; k += 3) {
        double mag = cabsf(cloud[k]);

        if (mag < 1e-9) continue;
        out->pt_re[out->n_points] = crealf(cloud[k] / (float)mag);
        out->pt_im[out->n_points] = cimagf(cloud[k] / (float)mag);
        out->n_points++;
    }
    ok = true;
    free(power);
    free(X);
    free(H);
    free(eq);
    free(Y);
    free(cloud);
    free(vals);
    return ok;
}

/* ------------------------------------------------------------------ */
/* PUCCH                                                              */
/* ------------------------------------------------------------------ */

static double pucch_metric(const float complex *sym, int u, int *lag)
{
    float complex z[12], Z[12];
    double peak = 0.0;
    double energy = 0.0;

    for (int n = 0; n < 12; n++) {
        z[n] = sym[n] * conjf(pucch_base(u, n));
        energy += (double)(crealf(sym[n]) * crealf(sym[n]) +
                           cimagf(sym[n]) * cimagf(sym[n]));
    }
    dft(z, Z, 12, 1);
    for (int l = 0; l < 12; l++) {
        double p = (double)(crealf(Z[l]) * crealf(Z[l]) +
                            cimagf(Z[l]) * cimagf(Z[l]));

        if (p > peak) {
            peak = p;
            if (lag != NULL) *lag = l;
        }
    }
    /* scale-invariant: the DFT peak of a matching sequence is 12x the
     * symbol energy, so normalise to 0..1 (a noise-dominated symbol can
     * then no longer outvote the real sequence) */
    if (energy > 1e-20) return peak / (12.0 * energy);
    return 0.0;
}

static bool pucch_demod(const float complex *x, int n_slots, int slot_len,
                        int n, int cp0, int cp, double fs, double f_lo,
                        double f_hi, LteDemod *out, double *quality,
                        double expect_center_hz, double src_center_hz)
{
    float *power = calloc((size_t)n, sizeof(float));
    float complex *X = NULL;
    float complex sym[LTE_MAX_SLOTS * LTE_SYMBOLS_PER_SLOT][12];
    int total = n_slots * LTE_SYMBOLS_PER_SLOT;
    int b0 = 0, u = 0, n_used = 0, format = 1;
    double metric4 = 1.0;

    if (power == NULL) return false;
    slot_spectrum(x, 0, n, cp0, cp, 0.0, fs, power);
    X = malloc((size_t)n * sizeof(*X));
    if (X == NULL) {
        free(power);
        return false;
    }
    {
        float peak = 0.0f;
        int bin_lo, bin_hi, best_lo = -1;
        float best_sum = 0.0f;

        for (int k = 0; k < n; k++) if (power[k] > peak) peak = power[k];
        if (peak <= 0.0f) {
            free(power);
            free(X);
            return false;
        }
        bin_lo = (int)floor(f_lo / LTE_SCS_HZ - 0.5);
        bin_hi = (int)ceil(f_hi / LTE_SCS_HZ - 0.5);
        if (bin_hi - bin_lo < 12) bin_hi = bin_lo + 12;
        {
            int expect_bin = -1;

            if (expect_center_hz > 0.0 && src_center_hz > 0.0) {
                expect_bin = (int)llround((expect_center_hz - src_center_hz) /
                                          15000.0 - 0.5) - 6;
            }
            for (int k = bin_lo; k + 12 <= bin_hi; k++) {
                float sum = 0.0f;

                for (int i = 0; i < 12; i++) sum += power[((k + i) % n + n) % n];
                if (sum > best_sum) {
                    best_sum = sum;
                    best_lo = k;
                }
            }
            /* among strongly occupied blocks, the one nearest the centre the
             * analysis measured (avoids locking onto a neighbour PUCCH) */
            if (expect_bin >= 0 && best_sum > 0.0f) {
                int best_near = -1;
                int best_dist = 1 << 30;

                for (int k = bin_lo; k + 12 <= bin_hi; k++) {
                    float sum = 0.0f;

                    for (int i = 0; i < 12; i++) {
                        sum += power[((k + i) % n + n) % n];
                    }
                    if (sum < 0.5f * best_sum) continue;
                    {
                        int dist = abs(k - expect_bin);

                        if (dist < best_dist) {
                            best_dist = dist;
                            best_near = k;
                        }
                    }
                }
                if (best_near >= 0) {
                    best_lo = best_near;
                    best_sum = 0.0f;
                    for (int i = 0; i < 12; i++) {
                        best_sum += power[((best_lo + i) % n + n) % n];
                    }
                }
            }
        }
        if (best_lo < 0 || best_sum < 0.1f * peak * 12.0f) {
            free(power);
            free(X);
            return false;
        }
        b0 = best_lo;
    }
    for (int s = 0; s < n_slots; s++) {
        for (int l = 0; l < LTE_SYMBOLS_PER_SLOT; l++) {
            int cs = symbol_cp_start(s * slot_len, n, cp0, cp, l);
            int cpl = l == 0 ? cp0 : cp;
            int idx = s * LTE_SYMBOLS_PER_SLOT + l;

            if (idx >= total) break;
            ofdm_symbol(x, cs, cpl, n, 0.0, fs, X);
            for (int k = 0; k < 12; k++) {
                sym[idx][k] = X[((b0 + k) % n + n) % n];
            }
            n_used = idx + 1;
        }
    }
    for (int k = 0; k < 12; k++) {
        TE15_LOG(" %.2f", (double)cabsf(sym[0][k]));
    }
    TE15_LOG("\n");
    /* Base sequence group: search over the 30 sequences using the DMRS
     * symbols only. Data symbols carry QPSK payloads, so their de-rotated
     * DFT is not a single peak and would let wrong candidates win by
     * chance (u detection was data-dependent before this change). */
    {
        double best = 0.0;

        for (int cand = 0; cand < 30; cand++) {
            /* PUCCH formats 1/1a/1b carry DMRS at symbols 2,3,4 of each
             * slot; formats 2/2a/2b at 1 and 5. Try both pilot patterns and
             * keep the better one: this makes the sequence-group search
             * independent of the format guess (which is made later). */
            static const int pat_a[] = { 1, 5 };
            static const int pat_b[] = { 2, 3, 4 };
            double score_a = 1e9, score_b = 1e9;
            int na = 0, nb = 0;

            for (int s = 0; s < n_slots; s++) {
                for (int p = 0; p < 2; p++) {
                    int idx = s * LTE_SYMBOLS_PER_SLOT + pat_a[p];

                    if (idx < n_used) {
                        double m = pucch_metric(sym[idx], cand, NULL);

                        if (m < score_a) score_a = m;
                        na++;
                    }
                }
                for (int p = 0; p < 3; p++) {
                    int idx = s * LTE_SYMBOLS_PER_SLOT + pat_b[p];

                    if (idx < n_used) {
                        double m = pucch_metric(sym[idx], cand, NULL);

                        if (m < score_b) score_b = m;
                        nb++;
                    }
                }
            }
            {
                double score = (na > 0 ? score_a : -1.0);

                if (nb > 0 && score_b > score) score = score_b;
                if (score > best) {
                    best = score;
                    u = cand;
                }
            }
        }
    }
    /* wipe the base sequence and the per-symbol cyclic shift; each symbol
     * then reduces to a single complex value */
    {
        float complex val[LTE_MAX_SLOTS * LTE_SYMBOLS_PER_SLOT];
        int lag[LTE_MAX_SLOTS * LTE_SYMBOLS_PER_SLOT];

        for (int i = 0; i < n_used; i++) {
            float complex acc = 0.0f;

            (void)pucch_metric(sym[i], u, &lag[i]);
            for (int k = 0; k < 12; k++) {
                acc += sym[i][k] * conjf(pucch_base(u, k)) *
                       cexpf(-I * 2.0f * (float)M_PI * (float)lag[i] *
                             (float)k / 12.0f);
            }
            val[i] = acc / 12.0f;
        }
        out->pucch_u = u;
        out->fft_size = n;
        out->cp_first = cp0;
        out->cp_normal = cp;
        out->slots = n_slots;
        out->subcarriers = 12;
        out->rb_count = 1;
        out->bw_hz = 12.0 * LTE_SCS_HZ;
        out->channel = LTE_CH_PUCCH;
        out->channel_name = "PUCCH";
        for (int s = 0; s < n_slots && s < 4; s++) {
            out->pucch_alpha[s] = lag[s * LTE_SYMBOLS_PER_SLOT];
        }
        /* format 2/2a/2b: DMRS at 1 and 5, QPSK data at 0,2,3,4,6 */
        {
            static const int data_idx[5] = {0, 2, 3, 4, 6};
            double sum = 0.0;
            int count = 0;

            for (int s = 0; s < n_slots; s++) {
                float complex ref = val[s * 7 + 1] /
                                    (cabsf(val[s * 7 + 1]) + 1e-12f);

                for (int i = 0; i < 5; i++) {
                    float complex w = val[s * 7 + data_idx[i]] /
                                      (cabsf(val[s * 7 + data_idx[i]]) +
                                       1e-12f);
                    float complex q = w / ref;
                    float complex q4 = q * q * q * q;
                    double dr = crealf(q4) + 1.0;
                    double di = cimagf(q4);

                    sum += dr * dr + di * di;
                    count++;
                }
            }
            metric4 = count > 0 ? sqrt(sum / (double)count) : 1.0;
            format = (metric4 < 0.6) ? 2 : 1;
            if (quality != NULL) {
                *quality = (format == 2) ? (1.0 / (0.05 + metric4)) : 0.0;
            }
            for (int s = 0; s < n_slots; s++) {
                TE15_LOG("  slot %d lags:", s);
                for (int l = 0; l < LTE_SYMBOLS_PER_SLOT; l++) {
                    TE15_LOG(" %d", lag[s * LTE_SYMBOLS_PER_SLOT + l]);
                }
                TE15_LOG("  |val|:");
                for (int l = 0; l < LTE_SYMBOLS_PER_SLOT; l++) {
                    TE15_LOG(" %.3f", (double)cabsf(val[s * LTE_SYMBOLS_PER_SLOT + l]));
                }
                TE15_LOG("\n");
            }
        }
        if (format == 2) {
            unsigned vals[LTE_MAX_SLOTS * 10];
            size_t nv = 0;
            static const int data_idx[5] = {0, 2, 3, 4, 6};

            for (int s = 0; s < n_slots; s++) {
                float complex ref = val[s * 7 + 1] /
                                    (cabsf(val[s * 7 + 1]) + 1e-12f);

                for (int i = 0; i < 5; i++) {
                    float complex q = val[s * 7 + data_idx[i]] / ref;
                    int ri = crealf(q) < 0.0f ? 1 : 0;
                    int ii = cimagf(q) < 0.0f ? 1 : 0;
                    unsigned bits = (unsigned)((ri << 1) | ii);

                    vals[nv++] = bits;
                    if (out->n_points < LTE_MAX_POINTS) {
                        out->pt_re[out->n_points] =
                            ri ? -0.70710678f : 0.70710678f;
                        out->pt_im[out->n_points] =
                            ii ? -0.70710678f : 0.70710678f;
                        out->n_points++;
                    }
                }
            }
            out->pucch_format = 2;
            out->mod_order = 2;
            snprintf(out->mod_name, sizeof(out->mod_name), "%s", "QPSK");
            out->evm_pct = 100.0 * metric4;
            out->snr_db = metric4 > 0.0 ? -20.0 * log10(metric4 + 1e-6) : 40.0;
            pack_bits(out, vals, nv, 2);
        } else {
            /* format 1/1a/1b: DMRS at 2,3,4; data at 0,1,5,6 */
            float complex w3[3][3];
            static const float complex w4[3][4] = {
                {1.0f, 1.0f, 1.0f, 1.0f},
                {1.0f, -1.0f, 1.0f, -1.0f},
                {1.0f, -1.0f, -1.0f, 1.0f},
            };
            static const int dmrs_idx[3] = {2, 3, 4};
            static const int data_idx[4] = {0, 1, 5, 6};
            float d0_total = 0.0f;
            int slots_seen = 0;

            w3[0][0] = 1.0f; w3[0][1] = 1.0f; w3[0][2] = 1.0f;
            w3[1][0] = 1.0f;
            w3[1][1] = cexpf(I * 2.0f * (float)M_PI / 3.0f);
            w3[1][2] = cexpf(-I * 2.0f * (float)M_PI / 3.0f);
            w3[2][0] = 1.0f;
            w3[2][1] = cexpf(-I * 2.0f * (float)M_PI / 3.0f);
            w3[2][2] = cexpf(I * 2.0f * (float)M_PI / 3.0f);
            for (int s = 0; s < n_slots; s++) {
                int c3 = 0, c4 = 0;
                float complex acc, ref;

                {
                    double best = -1.0;

                    for (int c = 0; c < 3; c++) {
                        float complex a = 0.0f;

                        for (int i = 0; i < 3; i++) {
                            a += val[s * 7 + dmrs_idx[i]] * conjf(w3[c][i]);
                        }
                        if (cabsf(a) > best) {
                            best = cabsf(a);
                            c3 = c;
                        }
                    }
                }
                {
                    double best = -1.0;

                    for (int c = 0; c < 3; c++) {
                        float complex a = 0.0f;

                        for (int i = 0; i < 4; i++) {
                            a += val[s * 7 + data_idx[i]] * conjf(w4[c][i]);
                        }
                        if (cabsf(a) > best) {
                            best = cabsf(a);
                            c4 = c;
                        }
                    }
                }
                acc = 0.0f;
                for (int i = 0; i < 4; i++) {
                    acc += val[s * 7 + data_idx[i]] * conjf(w4[c4][i]);
                }
                acc /= 4.0f;
                ref = 0.0f;
                for (int i = 0; i < 3; i++) {
                    ref += val[s * 7 + dmrs_idx[i]] * conjf(w3[c3][i]);
                }
                ref /= 3.0f;
                acc *= conjf(ref);
                {
                    double re = crealf(acc * acc);
                    float complex S = re >= 0.0 ? 1.0f : I;

                    d0_total += crealf(acc / S);
                    slots_seen++;
                }
            }
            out->pucch_format = 1;
            out->mod_order = 1;
            snprintf(out->mod_name, sizeof(out->mod_name), "%s", "BPSK");
            out->control_bits = 1;
            out->n_bits = 1;
            out->bits[0] = d0_total < 0.0f ? 1u : 0u;
            snprintf(out->control_text, sizeof(out->control_text), "%s",
                     d0_total < 0.0f ? "ACK (d0 = -1)"
                                     : "NACK/SR (d0 = +1)");
            out->evm_pct = 0.0;
            out->snr_db = 0.0;
            if (quality != NULL) {
                double pw = 0.0;

                for (int i = 0; i < n_used; i++) {
                    pw += (double)(crealf(val[i]) * crealf(val[i]) +
                                   cimagf(val[i]) * cimagf(val[i]));
                }
                pw = sqrt(pw / (double)(n_used > 0 ? n_used : 1));
                *quality = 0.0;
                if (pw > 0.0) {
                    for (int s = 0; s < n_slots; s++) {
                        float complex a = 0.0f;

                        for (int i = 0; i < 3; i++) {
                            a += val[s * 7 + dmrs_idx[i]] * conjf(w3[0][i]);
                        }
                        *quality += (double)cabsf(a) / (3.0 * pw);
                    }
                }
            }
            (void)slots_seen;
        }
    }
    free(power);
    free(X);
    return true;
}

/* ------------------------------------------------------------------ */
/* entry point                                                        */
/* ------------------------------------------------------------------ */

bool lte_demodulate(const IqSource *src, const Region *region,
                    LteChannel want, const LteHint *hint, LteDemod *out)
{
    double fs_src, f_lo, f_hi;
    int n, cp0, cp, slot_len, n_slots, offset = 0;
    size_t in_n;
    float *raw = NULL;
    float complex *x = NULL;
    bool ok;

    if (src == NULL || region == NULL || out == NULL) return false;
    memset(out, 0, sizeof(*out));
    if (region->freq_max <= region->freq_min) return false;
    /* a valid analysis hint decides the channel when the caller left it
     * on auto, and narrows the frequency search below */
    if (hint != NULL && hint->valid && want == LTE_CH_AUTO) {
        want = hint->channel;
    }
    fs_src = src->sample_rate;
    {
        double target = fs_src / LTE_SCS_HZ;
        int nfft = 64;

        while (nfft < target && nfft < 8192) nfft <<= 1;
        if (nfft > 8192) nfft = 8192;
        n = nfft;
    }
    cp0 = (int)llround((double)n * LTE_CP0_RATIO);
    cp = (int)llround((double)n * LTE_CP_RATIO);
    slot_len = 7 * n + cp0 + 6 * cp;
    {
        double dur = region->time_max - region->time_min;
        int slots = (int)floor(dur * fs_src / (double)slot_len);

        if (slots < 1) return false;
        if (slots > LTE_MAX_SLOTS) slots = LTE_MAX_SLOTS;
        n_slots = slots;
        if (want == LTE_CH_PUCCH && n_slots > 4) n_slots = 4;
        if (n_slots > 8) n_slots = 8;
    }
    in_n = (size_t)n_slots * (size_t)slot_len;
    raw = malloc(2 * in_n * sizeof(float));
    x = malloc(in_n * sizeof(*x));
    if (raw == NULL || x == NULL) {
        free(raw);
        free(x);
        return false;
    }
    iq_source_read_complex(src, (size_t)(region->time_min * fs_src), raw, in_n);
    resample(raw, in_n, in_n, x);
    /* Coarse symbol timing: the cyclic prefixes of ALL seven symbols of a
     * slot are correlated at once, so a spurious single-symbol peak cannot
     * win (the uplink has no dedicated synchronisation signal). */
    {
        double best = -1.0;
        int max_cand = (int)in_n - slot_len;

        if (max_cand < 0) max_cand = 0;
        for (int cand = 0; cand <= max_cand; cand++) {
            double score = 0.0;
            bool complete = true;

            for (int l = 0; l < LTE_SYMBOLS_PER_SLOT; l++) {
                int cs = symbol_cp_start(cand, n, cp0, cp, l);
                int cpl = l == 0 ? cp0 : cp;
                float complex acc = 0.0f;

                if (cs + cpl + n > (int)in_n) {
                    complete = false;
                    break;
                }
                for (int k = 0; k < cpl; k++) {
                    acc += x[cs + k] * conjf(x[cs + k + n]);
                }
                /* normalised per CP length: attributing the longer cp0
                 * window to a normal symbol makes its correlation (and the
                 * score) smaller, so the true slot start wins */
                score += (double)cabsf(acc) / (double)cpl;
            }
            /* only candidates where the whole slot fits are comparable */
            if (complete && score > best) {
                best = score;
                offset = cand;
            }
        }
        /* carrier offset from the strongest symbol's correlation phase */
        {
            int best_l = 0;
            double best_p = 0.0;

            for (int l = 0; l < LTE_SYMBOLS_PER_SLOT; l++) {
                int cs = symbol_cp_start(offset, n, cp0, cp, l);
                int cpl = l == 0 ? cp0 : cp;
                float complex acc = 0.0f;

                if (cs + cpl + n >= (int)in_n) break;
                for (int k = 0; k < cpl; k++) {
                    acc += x[cs + k] * conjf(x[cs + k + n]);
                }
                if (cabsf(acc) > best_p) {
                    best_p = cabsf(acc);
                    best_l = l;
                }
            }
            {
                int cs = symbol_cp_start(offset, n, cp0, cp, best_l);
                float complex acc = 0.0f;

                for (int k = 0; k < (best_l == 0 ? cp0 : cp); k++) {
                    acc += x[cs + k] * conjf(x[cs + k + n]);
                }
                out->cfo_hz = cargf(acc) / (2.0 * M_PI) * fs_src /
                              (double)n;
            }
        }
    }
    /* the demodulator's FFT bins are offsets from the recording centre
     * after the half-subcarrier removal: bin k holds (k + 1/2) * 15 kHz */
    f_lo = region->freq_min - src->center_freq;
    f_hi = region->freq_max - src->center_freq;
    if (hint != NULL && hint->valid) {
        /* stay inside the band's legal uplink range and around the
         * allocation the analysis measured (50 % dilation keeps room for
         * the real edge of the burst) */
        double lo = f_lo, hi = f_hi;

        lo = fmax(lo, hint->band_min_hz - src->center_freq);
        hi = fmin(hi, hint->band_max_hz - src->center_freq);
        if (hi > lo) {
            f_lo = lo;
            f_hi = hi;
        }
        if (hint->bandwidth_hz > 0.0 && hint->center_hz > 0.0) {
            double half = 0.75 * hint->bandwidth_hz;

            lo = fmax(f_lo, hint->center_hz - half - src->center_freq);
            hi = fmin(f_hi, hint->center_hz + half - src->center_freq);
            if (hi > lo) {
                f_lo = lo;
                f_hi = hi;
            }
        }
    }
    out->scs_hz = LTE_SCS_HZ;
    out->center_hz = 0.5 * (region->freq_min + region->freq_max);
    /* The cyclic-prefix correlation locks onto *a* symbol boundary; the slot
     * start (and therefore which symbol carries the DMRS / the control
     * symbols) is resolved by trying the seven possible rotations and
     * keeping the one whose channel-specific signature is strongest. */
    {
        int max_slots = (int)((in_n - (size_t)offset) / (size_t)slot_len);
        double best_quality = -1.0;
        LteDemod best;
        bool best_ok = false;

        if (max_slots < 1) max_slots = 1;
        if (max_slots > n_slots) max_slots = n_slots;
        memset(&best, 0, sizeof(best));
        for (int rot = 0; rot < 2 * LTE_SYMBOLS_PER_SLOT; rot++) {
            int shift = (rot < LTE_SYMBOLS_PER_SLOT)
                            ? symbol_cp_start(0, n, cp0, cp, rot)
                            : slot_len + symbol_cp_start(0, n, cp0, cp,
                                                         rot -
                                                         LTE_SYMBOLS_PER_SLOT);
            int slot_start = offset - shift;
            double quality = 0.0;
            LteDemod trial;
            bool trial_ok = false;

            if (slot_start < 0) continue;
            {
                int avail = (int)((in_n - (size_t)slot_start) /
                                  (size_t)slot_len);

                if (avail < 1) continue;
                if (avail > n_slots) avail = n_slots;
                max_slots = avail;
            }
            memset(&trial, 0, sizeof(trial));
            trial.scs_hz = LTE_SCS_HZ;
            trial.cfo_hz = out->cfo_hz;
            trial.center_hz = 0.5 * (region->freq_min + region->freq_max);
            int expect_subc = (hint != NULL && hint->valid &&
                               hint->rb_estimate > 0)
                                  ? hint->rb_estimate * 12 : 0;
            int prefer_order = (hint != NULL && hint->valid)
                                   ? hint->mod_order : 0;

            if (want == LTE_CH_PUSCH) {
                trial_ok = pusch_demod(x + slot_start, max_slots, slot_len, n,
                                       cp0, cp, fs_src, f_lo, f_hi, &trial,
                                       &quality, expect_subc, prefer_order);
            } else if (want == LTE_CH_PUCCH) {
                trial_ok = pucch_demod(x + slot_start, max_slots > 4 ? 4 :
                                                                     max_slots,
                                       slot_len, n, cp0, cp, fs_src, f_lo,
                                       f_hi, &trial, &quality,
                                       hint != NULL ? hint->center_hz : 0.0,
                                       src->center_freq);
            } else {
                trial_ok = pusch_demod(x + slot_start, max_slots, slot_len, n,
                                       cp0, cp, fs_src, f_lo, f_hi, &trial,
                                       &quality, expect_subc, prefer_order);
                if (!trial_ok) {
                    memset(&trial, 0, sizeof(trial));
                    trial.scs_hz = LTE_SCS_HZ;
                    trial.cfo_hz = out->cfo_hz;
                    trial.center_hz = 0.5 * (region->freq_min +
                                             region->freq_max);
                    trial_ok = pucch_demod(x + slot_start,
                                           max_slots > 4 ? 4 : max_slots,
                                           slot_len, n, cp0, cp, fs_src,
                                           f_lo, f_hi, &trial, &quality,
                                           hint != NULL ? hint->center_hz
                                                        : 0.0,
                                           src->center_freq);
                }
            }
            /* automatic detection needs the strong evidence: a PUSCH
             * DMRS correlation well above the noise floor, or a PUCCH
             * format-2 QPSK closure (a format-1 control-only burst is
             * ambiguous against noise, so it needs an explicit request) */
            if (trial_ok && want == LTE_CH_AUTO &&
                trial.channel == LTE_CH_PUCCH &&
                !(trial.pucch_format == 2 && trial.evm_pct < 50.0)) {
                trial_ok = false;
            }
            if (trial_ok &&
                (!best_ok || trial.slots > best.slots ||
                 (trial.slots == best.slots &&
                  quality > best_quality + 1e-6))) {
                best_quality = quality;
                best = trial;
                best_ok = true;
                best.cfo_hz = out->cfo_hz;
            }
        }
        *out = best;
        out->scs_hz = LTE_SCS_HZ;
        if (!best_ok) {
            out->center_hz = 0.5 * (region->freq_min + region->freq_max);
        }
        ok = best_ok;
    }
    out->valid = ok;
    if (ok) {
        size_t used;

        if (hint != NULL && hint->valid) {
            snprintf(out->summary, sizeof(out->summary), "%s", hint->source);
            used = strlen(out->summary);
            if (used + 3 < sizeof(out->summary)) {
                snprintf(out->summary + used, sizeof(out->summary) - used,
                         " | ");
            }
        } else {
            out->summary[0] = '\0';
        }
        used = strlen(out->summary);
        snprintf(out->summary + used, sizeof(out->summary) - used,
                 "%s | %.3f MHz centre | %.1f kHz occupied (%d RB, %d "
                 "subcarriers) | 15 kHz SCS | %d slot(s), %d symbols | %s | "
                 "EVM %.1f%%, SNR %.1f dB",
                 out->channel_name, out->center_hz / 1e6, out->bw_hz / 1e3,
                 out->rb_count, out->subcarriers, out->slots,
                 out->slots * LTE_SYMBOLS_PER_SLOT, out->mod_name,
                 out->evm_pct, out->snr_db);
        used = strlen(out->summary);
        if (out->channel == LTE_CH_PUCCH) {
            snprintf(out->summary + used, sizeof(out->summary) - used,
                     " | base sequence u=%d | format %s%d%s%s",
                     out->pucch_u, "2", out->pucch_format,
                     out->control_text[0] != '\0' ? " | " : "",
                     out->control_text);
        } else {
            snprintf(out->summary + used, sizeof(out->summary) - used,
                     " | DMRS Zadoff-Chu u=%d v=%d alpha=%d/12 (%.1f kHz "
                     "subcarrier spacing)", out->dmrs_u, out->dmrs_v,
                     out->dmrs_alpha, out->scs_hz / 1e3);
        }
        if (out->n_bits > 0) {
            used = strlen(out->summary);
            snprintf(out->summary + used, sizeof(out->summary) - used,
                     " | %zu demodulated bits", out->n_bits);
        }
    } else {
        snprintf(out->summary, sizeof(out->summary),
                 "No %s burst detected in the selection "
                 "(%s): the cyclic-prefix timing and the channel-specific "
                 "signature did not match.",
                 lte_channel_name(want == LTE_CH_AUTO ? LTE_CH_PUSCH : want),
                 want == LTE_CH_AUTO ? "auto" : "requested");
    }
    free(raw);
    free(x);
    return out->valid;
}
