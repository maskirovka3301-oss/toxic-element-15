/*
 * dsp.c — small, reusable DSP primitives: windowing, FFT spectra in dB,
 * entropy, pulse-repetition-frequency estimation, and an FftPlan wrapper
 * around single-precision FFTW3 plans.
 */
#include "dsp.h"

#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

/* FFTW planning is not thread-safe in this build (no
 * fftwf_make_planner_thread_safe): the waterfall cache worker and the
 * async analysis worker can create plans concurrently, which deadlocks
 * or corrupts the planner. Serialise plan creation/destruction with one
 * process-wide mutex. Execution of distinct plans stays concurrent. */
static pthread_mutex_t fftw_plan_mutex = PTHREAD_MUTEX_INITIALIZER;

#ifndef M_PI
#define TE15_DSP_PI 3.14159265358979323846
#else
#define TE15_DSP_PI M_PI
#endif

void dsp_hann(float *w, size_t n)
{
    if (n <= 1) {
        if (n == 1) {
            w[0] = 1.0f;
        }
        return;
    }

    for (size_t i = 0; i < n; i++) {
        double phase = 2.0 * TE15_DSP_PI * (double)i / (double)(n - 1);
        w[i] = (float)(0.5 * (1.0 - cos(phase)));
    }
}

void dsp_bins_to_db(const fftwf_complex *bins, size_t n, float *db_out)
{
    size_t half = n / 2;

    for (size_t j = 0; j < n; j++) {
        size_t idx = (j + half) % n;
        float re = bins[idx][0];
        float im = bins[idx][1];
        float mag = sqrtf(re * re + im * im) + 1e-10f;
        db_out[j] = 20.0f * log10f(mag);
    }
}

double dsp_entropy(const float *v, size_t n)
{
    if (n == 0) {
        return 0.0;
    }

    /* Pass 1: min/max of the input. */
    float mn = v[0];
    float mx = v[0];
    for (size_t i = 1; i < n; i++) {
        if (v[i] < mn) {
            mn = v[i];
        }
        if (v[i] > mx) {
            mx = v[i];
        }
    }

    float range = mx - mn;
    if (range <= 0.0f) {
        return 0.0;             /* empty or constant input */
    }

    /* Pass 2a: normalise to u[i] in [0,1] and accumulate the total. */
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += (double)((v[i] - mn) / range);
    }
    if (sum <= 0.0) {
        return 0.0;
    }

    /* Pass 2b: Shannon entropy in bits of the normalised distribution. */
    double h = 0.0;
    for (size_t i = 0; i < n; i++) {
        double p = (double)((v[i] - mn) / range) / sum;
        if (p > 1e-12) {
            h -= p * log2(p);
        }
    }
    return h;
}

double dsp_estimate_prf(const float *power, size_t n, double dt)
{
    if (n < 4 || dt <= 0.0) {
        return 0.0;
    }

    /* Mean and standard deviation of the frame power series. */
    double mean = 0.0;
    for (size_t i = 0; i < n; i++) {
        mean += (double)power[i];
    }
    mean /= (double)n;

    double variance = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)power[i] - mean;
        variance += d * d;
    }
    variance /= (double)n;
    double std = sqrt(variance);
    double threshold = mean + 1.0 * std;

    int *peaks = malloc(n * sizeof(int));
    if (!peaks) {
        return 0.0;
    }

    /* Rising-edge pulse detection with a 2-sample minimum gap. */
    int peak_count = 0;
    int above = 0;
    const int min_gap = 2;
    for (size_t i = 0; i < n; i++) {
        if (power[i] > threshold && !above) {
            if (peak_count == 0 ||
                ((int)i - peaks[peak_count - 1]) >= min_gap) {
                peaks[peak_count++] = (int)i;
                above = 1;
            }
        } else if (power[i] < mean) {
            above = 0;
        }
    }

    double prf = 0.0;
    if (peak_count >= 2) {
        double interval_sum = 0.0;
        int interval_count = 0;
        for (int i = 1; i < peak_count; i++) {
            double interval = (peaks[i] - peaks[i - 1]) * dt;
            if (interval > 0.001 && interval < 10.0) {
                interval_sum += interval;
                interval_count++;
            }
        }
        if (interval_count > 0) {
            double avg_interval = interval_sum / interval_count;
            if (avg_interval > 0.0) {
                prf = 1.0 / avg_interval;
            }
        }
    }

    free(peaks);
    return prf;
}

void dsp_normalize_to_unit(const float *v, size_t n, float *out,
                           float *min_out, float *max_out)
{
    if (n == 0) {
        *min_out = 0.0f;
        *max_out = 0.0f;
        return;
    }

    float mn = v[0];
    float mx = v[0];
    for (size_t i = 1; i < n; i++) {
        if (v[i] < mn) {
            mn = v[i];
        }
        if (v[i] > mx) {
            mx = v[i];
        }
    }

    *min_out = mn;
    *max_out = mx;

    float range = mx - mn;
    if (range < 1e-6f) {
        for (size_t i = 0; i < n; i++) {
            out[i] = 0.5f;
        }
        return;
    }

    for (size_t i = 0; i < n; i++) {
        float t = (v[i] - mn) / range;
        if (t < 0.0f) {
            t = 0.0f;
        } else if (t > 1.0f) {
            t = 1.0f;
        }
        out[i] = t;
    }
}

/* ---------------------------------------------------------------------
 * FftPlan — reusable single-threaded forward FFT (single precision).
 * ------------------------------------------------------------------ */

struct FftPlan {
    int n;
    fftwf_complex *in;
    fftwf_complex *out;
    fftwf_plan plan;
};

FftPlan *fft_plan_create(int n)
{
    if (n <= 0) {
        return NULL;
    }

    FftPlan *p = malloc(sizeof(*p));
    if (!p) {
        return NULL;
    }

    p->n = n;
    p->in = fftwf_malloc((size_t)n * sizeof(fftwf_complex));
    if (!p->in) {
        free(p);
        return NULL;
    }

    p->out = fftwf_malloc((size_t)n * sizeof(fftwf_complex));
    if (!p->out) {
        fftwf_free(p->in);
        free(p);
        return NULL;
    }

    pthread_mutex_lock(&fftw_plan_mutex);
    p->plan = fftwf_plan_dft_1d(n, p->in, p->out, FFTW_FORWARD,
                                FFTW_ESTIMATE);
    pthread_mutex_unlock(&fftw_plan_mutex);
    if (!p->plan) {
        fftwf_free(p->out);
        fftwf_free(p->in);
        free(p);
        return NULL;
    }

    return p;
}

void fft_plan_destroy(FftPlan *p)
{
    if (!p) {
        return;
    }
    fftwf_destroy_plan(p->plan);
    fftwf_free(p->in);
    fftwf_free(p->out);
    free(p);
}

int fft_plan_size(const FftPlan *p)
{
    return p->n;
}

void fft_plan_spectrum(const FftPlan *p, const float *iq,
                       const float *window, float *db_out)
{
    for (int j = 0; j < p->n; j++) {
        float w = window ? window[j] : 1.0f;
        p->in[j][0] = iq[2 * j] * w;
        p->in[j][1] = iq[2 * j + 1] * w;
    }

    fftwf_execute(p->plan);

    dsp_bins_to_db(p->out, (size_t)p->n, db_out);
}

void fft_plan_transform(const FftPlan *p, const float *iq,
                        const float *window, fftwf_complex *out)
{
    for (int j = 0; j < p->n; j++) {
        float w = window ? window[j] : 1.0f;
        p->in[j][0] = iq[2 * j] * w;
        p->in[j][1] = iq[2 * j + 1] * w;
    }
    fftwf_execute(p->plan);
    memcpy(out, p->out, (size_t)p->n * sizeof(fftwf_complex));
}

size_t dsp_fir_decimate(const float *iq, size_t n, size_t D, float *out)
{
    if (iq == NULL || out == NULL || n == 0 || D == 0) return 0;
    if (D == 1) {
        memcpy(out, iq, 2 * n * sizeof(float));
        return n;
    }
    enum { T = 64 }; /* taps */
    const double fc = 0.4 / (double)D; /* normalised cutoff */
    float h[T];
    double hsum = 0.0;
    for (int i = 0; i < T; i++) {
        double t = (double)i - (double)(T - 1) / 2.0;
        double w = 0.54 - 0.46 * cos(2.0 * M_PI * (double)i / (double)(T - 1));
        double v = (t == 0.0) ? 2.0 * fc : sin(2.0 * M_PI * fc * t) / (M_PI * t);
        h[i] = (float)(v * w);
        hsum += h[i];
    }
    for (int i = 0; i < T; i++) h[i] = (float)(h[i] / hsum);

    size_t out_i = 0;
    for (size_t m = 0; m + (size_t)T <= n; m += D) {
        double sr = 0.0, si = 0.0;
        for (int i = 0; i < T; i++) {
            sr += (double)h[i] * iq[2 * (m + (size_t)i)];
            si += (double)h[i] * iq[2 * (m + (size_t)i) + 1];
        }
        out[2 * out_i]     = (float)sr;
        out[2 * out_i + 1] = (float)si;
        out_i++;
    }
    return out_i;
}

void dsp_autocorr(const float *iq, size_t n, size_t max_lag,
                  double *out_re, double *out_im)
{
    if (iq == NULL || n < 2 || max_lag >= n ||
        out_re == NULL || out_im == NULL) {
        return;
    }
    size_t N = 1;
    while (N < 2 * n) N <<= 1;

    fftwf_complex *a = fftwf_malloc(sizeof(fftwf_complex) * N);
    fftwf_complex *b = fftwf_malloc(sizeof(fftwf_complex) * N);
    if (a == NULL || b == NULL) {
        fftwf_free(a);
        fftwf_free(b);
        return;
    }
    memset(a, 0, sizeof(fftwf_complex) * N);
    for (size_t i = 0; i < n; i++) {
        a[i][0] = iq[2 * i];
        a[i][1] = iq[2 * i + 1];
    }
    fftwf_plan fwd = fftwf_plan_dft_1d((int)N, a, b, FFTW_FORWARD,
                                       FFTW_ESTIMATE);
    fftwf_plan bwd = fftwf_plan_dft_1d((int)N, b, a, FFTW_BACKWARD,
                                       FFTW_ESTIMATE);
    if (fwd == NULL || bwd == NULL) {
        fftwf_destroy_plan(fwd);
        fftwf_destroy_plan(bwd);
        fftwf_free(a);
        fftwf_free(b);
        return;
    }
    fftwf_execute(fwd);
    for (size_t k = 0; k < N; k++) {
        b[k][0] = b[k][0] * b[k][0] + b[k][1] * b[k][1];
        b[k][1] = 0.0f;
    }
    fftwf_execute(bwd);
    for (size_t tau = 0; tau <= max_lag; tau++) {
        out_re[tau] = (double)a[tau][0] / (double)N;
        out_im[tau] = (double)a[tau][1] / (double)N;
    }
    fftwf_destroy_plan(fwd);
    fftwf_destroy_plan(bwd);
    fftwf_free(a);
    fftwf_free(b);
}

void dsp_window(DspWindow type, float *w, size_t n)
{
    if (w == NULL || n == 0) return;
    if (n == 1) {
        w[0] = 1.0f;
        return;
    }
    switch (type) {
    case DSP_WINDOW_RECTANGULAR:
        for (size_t i = 0; i < n; i++) w[i] = 1.0f;
        break;
    case DSP_WINDOW_HAMMING:
        for (size_t i = 0; i < n; i++) {
            w[i] = (float)(0.54 - 0.46 * cos(2.0 * M_PI * i / (double)(n - 1)));
        }
        break;
    case DSP_WINDOW_BLACKMAN:
        for (size_t i = 0; i < n; i++) {
            double x = 2.0 * M_PI * i / (double)(n - 1);
            w[i] = (float)(0.42 - 0.5 * cos(x) + 0.08 * cos(2.0 * x));
        }
        break;
    case DSP_WINDOW_HANN:
    default:
        dsp_hann(w, n);
        break;
    }
}

size_t dsp_srrc_pulse(float *taps, size_t taps_len, int samples_per_sym,
                      double rolloff)
{
    if (taps == NULL || taps_len == 0 || samples_per_sym < 1 ||
        rolloff < 0.0 || rolloff > 1.0) {
        return 0;
    }
    const double sps = samples_per_sym;
    double sum2 = 0.0;
    for (size_t i = 0; i < taps_len; i++) {
        double t = ((double)i - (double)(taps_len - 1) / 2.0) / sps;
        double v;
        if (fabs(fabs(t) - 0.25 / rolloff) < 1e-12 && rolloff > 0.0) {
            /* limit of the sinc form at the singular points */
            v = rolloff / sqrt(2.0) *
                ((1.0 + 2.0 / M_PI) * sin(M_PI / (4.0 * rolloff)) +
                 (1.0 - 2.0 / M_PI) * cos(M_PI / (4.0 * rolloff)));
        } else if (fabs(t) < 1e-12) {
            v = 1.0 - rolloff + 4.0 * rolloff / M_PI;
        } else {
            double num = sin(M_PI * t * (1.0 - rolloff)) +
                         4.0 * rolloff * t * cos(M_PI * t * (1.0 + rolloff));
            double den = M_PI * t * (1.0 - (4.0 * rolloff * t) *
                                     (4.0 * rolloff * t));
            v = num / den;
        }
        taps[i] = (float)v;
        sum2 += v * v;
    }
    double norm = sqrt(sum2);
    if (norm > 0.0) {
        for (size_t i = 0; i < taps_len; i++) taps[i] = (float)(taps[i] / norm);
    }
    return taps_len;
}

void dsp_whiten(float *iq, size_t n)
{
    if (iq == NULL || n < 16) return;
    fftwf_complex *X = fftwf_malloc(sizeof(fftwf_complex) * (size_t)n);
    double *p = malloc((size_t)n * sizeof(double));
    if (X == NULL || p == NULL) {
        fftwf_free(X);
        free(p);
        return;
    }
    fftwf_plan fwd = fftwf_plan_dft_1d((int)n, X, X, FFTW_FORWARD,
                                       FFTW_ESTIMATE);
    fftwf_plan inv = fftwf_plan_dft_1d((int)n, X, X, FFTW_BACKWARD,
                                       FFTW_ESTIMATE);
    if (fwd == NULL || inv == NULL) {
        if (fwd) fftwf_destroy_plan(fwd);
        if (inv) fftwf_destroy_plan(inv);
        fftwf_free(X);
        free(p);
        return;
    }
    for (size_t i = 0; i < n; i++) {
        X[i][0] = iq[2 * i];
        X[i][1] = iq[2 * i + 1];
    }
    fftwf_execute(fwd);
    /* 5-bin moving-average power */
    for (size_t i = 0; i < n; i++) {
        double pw = 0.0;
        for (int d = -2; d <= 2; d++) {
            size_t j = ((int)i + d + (int)n) % (size_t)n;
            pw += X[j][0] * X[j][0] + X[j][1] * X[j][1];
        }
        p[i] = pw / 5.0;
    }
    for (size_t i = 0; i < n; i++) {
        double g = p[i] > 1e-20 ? 1.0 / sqrt(p[i]) : 0.0;
        X[i][0] = (float)(X[i][0] * g);
        X[i][1] = (float)(X[i][1] * g);
    }
    fftwf_execute(inv);
    for (size_t i = 0; i < n; i++) {
        iq[2 * i]     = X[i][0] / (float)n;
        iq[2 * i + 1] = X[i][1] / (float)n;
    }
    fftwf_destroy_plan(fwd);
    fftwf_destroy_plan(inv);
    fftwf_free(X);
    free(p);
}

size_t dsp_decimate_staged(const float *iq, size_t n, size_t D, float *out)
{
    if (iq == NULL || out == NULL || n == 0 || D == 0) {
        return 0;
    }
    if (D == 1) {
        memcpy(out, iq, 2 * n * sizeof(float));
        return n;
    }
    {
        float *cur = malloc(2 * n * sizeof(float));
        size_t cur_n = n;
        size_t factor = D;

        if (cur == NULL) {
            return 0;
        }
        memcpy(cur, iq, 2 * n * sizeof(float));
        while (factor > 1) {
            /* pick the largest stage factor <= 8 that DIVIDES the
             * remaining factor exactly; a non-dividing stage (e.g. 8 of
             * 13) would leave the total decimation short and the caller's
             * output buffer too small */
            size_t f = 1;

            for (size_t cand = 8; cand >= 2; cand--) {
                if (factor % cand == 0) {
                    f = cand;
                    break;
                }
            }
            if (f < 2) {
                f = factor; /* prime remainder: one final pass */
            }
            {
                size_t cap = cur_n / f + 8;
                float *next = malloc(2 * cap * sizeof(float));
                size_t got;

                if (next == NULL) {
                    free(cur);
                    return 0;
                }
                got = dsp_fir_decimate(cur, cur_n, f, next);
                free(cur);
                cur = next;
                cur_n = got;
                factor /= f;
                if (got == 0) {
                    free(cur);
                    return 0;
                }
            }
        }
        memcpy(out, cur, 2 * cur_n * sizeof(float));
        free(cur);
        return cur_n;
    }
}

bool dsp_fft_mask_lowpass(float *iq, size_t n, double fs, double pass_hz,
                          double stop_hz)
{
    size_t nf = 1;
    fftwf_complex *buf = NULL;
    fftwf_plan fwd = NULL;
    fftwf_plan bwd = NULL;
    bool ok = false;

    if (iq == NULL || n < 8 || fs <= 0.0 || pass_hz <= 0.0 ||
        stop_hz <= pass_hz) {
        return false;
    }
    while (nf < n) nf <<= 1;
    buf = fftwf_malloc(sizeof(fftwf_complex) * nf);
    if (buf == NULL) return false;
    for (size_t i = 0; i < nf; i++) {
        if (i < n) {
            buf[i][0] = iq[2 * i];
            buf[i][1] = iq[2 * i + 1];
        } else {
            buf[i][0] = 0.0f;
            buf[i][1] = 0.0f;
        }
    }
    pthread_mutex_lock(&fftw_plan_mutex);
    fwd = fftwf_plan_dft_1d((int)nf, buf, buf, FFTW_FORWARD, FFTW_ESTIMATE);
    bwd = fftwf_plan_dft_1d((int)nf, buf, buf, FFTW_BACKWARD, FFTW_ESTIMATE);
    pthread_mutex_unlock(&fftw_plan_mutex);
    if (fwd == NULL || bwd == NULL) {
        goto out;
    }
    fftwf_execute(fwd);
    {
        double bin = fs / (double)nf;
        for (size_t k = 0; k < nf; k++) {
            double f = ((double)k <= (double)nf / 2.0)
                           ? (double)k * bin
                           : ((double)nf - (double)k) * bin;
            double h;

            if (f <= pass_hz) {
                h = 1.0;
            } else if (f >= stop_hz) {
                h = 0.0;
            } else {
                double t = (f - pass_hz) / (stop_hz - pass_hz);

                h = 0.5 * (1.0 + cos(M_PI * t));
            }
            buf[k][0] = (float)(buf[k][0] * h);
            buf[k][1] = (float)(buf[k][1] * h);
        }
    }
    fftwf_execute(bwd);
    for (size_t i = 0; i < n; i++) {
        iq[2 * i] = (float)(buf[i][0] / (double)nf);
        iq[2 * i + 1] = (float)(buf[i][1] / (double)nf);
    }
    ok = true;
out:
    pthread_mutex_lock(&fftw_plan_mutex);
    if (fwd != NULL) fftwf_destroy_plan(fwd);
    if (bwd != NULL) fftwf_destroy_plan(bwd);
    pthread_mutex_unlock(&fftw_plan_mutex);
    fftwf_free(buf);
    return ok;
}

size_t dsp_decimation_for_x(double fs, double signal_bw, double min_rate,
                            double factor)
{
    double target = min_rate;
    double need = factor * (signal_bw > 0.0 ? signal_bw : 0.0);
    size_t D;

    if (need > target) target = need;
    if (target <= 0.0 || fs <= 0.0) return 1;
    if (target > fs) return 1;
    D = (size_t)(fs / target);
    if (D < 1) D = 1;
    return D;
}

size_t dsp_decimation_for(double fs, double signal_bw, double min_rate)
{
    return dsp_decimation_for_x(fs, signal_bw, min_rate, 2.5);
}
