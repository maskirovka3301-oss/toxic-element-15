/*
 * sptk.c — SPTK-derived primitives (see header).
 */
#include "sptk.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static double sinc(double x)
{
    if (fabs(x) < 1e-12) return 1.0;
    return sin(M_PI * x) / (M_PI * x);
}

void sptk_charfun(const float *x, size_t n, const double *omega, int m,
                  double *out_re, double *out_im)
{
    if (x == NULL || n == 0 || omega == NULL || m <= 0 ||
        out_re == NULL || out_im == NULL) {
        return;
    }
    for (int j = 0; j < m; j++) {
        double sr = 0.0, si = 0.0;
        double w = omega[j];
        for (size_t i = 0; i < n; i++) {
            double ph = w * (double)x[i];
            sr += cos(ph);
            si += sin(ph);
        }
        out_re[j] = sr / (double)n;
        out_im[j] = si / (double)n;
    }
}

void sptk_mgf(const float *x, size_t n, const double *s, int m, double *out)
{
    if (x == NULL || n == 0 || s == NULL || m <= 0 || out == NULL) return;
    for (int j = 0; j < m; j++) {
        double v = s[j];
        double mx = 0.0;
        for (size_t i = 0; i < n; i++) {
            double t = v * (double)x[i];
            if (t > mx) mx = t;
        }
        double shift = (mx > 30.0) ? mx - 30.0 : 0.0;
        double sum = 0.0;
        for (size_t i = 0; i < n; i++) {
            sum += exp(v * (double)x[i] - shift);
        }
        out[j] = sum / (double)n * exp(shift);
    }
}

void sptk_cepstrum(const double *log_spectrum, size_t n, double *out)
{
    if (log_spectrum == NULL || n < 2 || out == NULL) return;
    for (size_t q = 0; q < n; q++) {
        double acc = 0.0;
        for (size_t b = 0; b < n; b++) {
            acc += log_spectrum[b] * cos(2.0 * M_PI * (double)(q * b) /
                                         (double)n);
        }
        out[q] = acc / (double)n;
    }
}

bool sptk_lpc(const float *iq, size_t n, int order, double *coeffs)
{
    if (iq == NULL || n < 8 || order < 1 || order > 64 || coeffs == NULL) {
        return false;
    }
    double *r = malloc((size_t)(order + 1) * sizeof(double));
    double *e = malloc((size_t)(order + 1) * sizeof(double));
    double *k = malloc((size_t)(order + 1) * sizeof(double));
    double *a = malloc((size_t)(order + 1) * sizeof(double));
    double *a2 = malloc((size_t)(order + 1) * sizeof(double));
    double *mag = malloc(n * sizeof(double));
    if (!r || !e || !k || !a || !a2 || !mag) {
        free(r); free(e); free(k); free(a); free(a2); free(mag);
        return false;
    }
    /* magnitude sequence with the sample mean removed: a dominant DC
     * term makes the normal equations near-singular and the
     * Levinson-Durbin recursion explodes */
    double mean = 0.0;
    for (size_t i = 0; i < n; i++) {
        mag[i] = sqrt((double)iq[2 * i] * iq[2 * i] +
                      (double)iq[2 * i + 1] * iq[2 * i + 1]);
        mean += mag[i];
    }
    mean /= (double)n;
    for (int j = 0; j <= order; j++) {
        double acc = 0.0;
        for (size_t i = 0; i + (size_t)j < n; i++) {
            acc += (mag[i] - mean) * (mag[i + (size_t)j] - mean);
        }
        r[j] = acc / (double)n;
    }
    free(mag);

    e[0] = r[0];
    a[0] = 1.0;
    for (int i = 1; i <= order; i++) {
        double num = r[i];
        for (int j = 1; j < i; j++) num += a[j] * r[i - j];
        double ki = e[i - 1] > 1e-20 ? num / e[i - 1] : 0.0;
        k[i] = ki;
        for (int j = 0; j <= i; j++) a2[j] = a[j];
        for (int j = 1; j < i; j++) a[j] = a2[j] - ki * a2[i - j];
        a[i] = -ki;
        e[i] = e[i - 1] * (1.0 - ki * ki);
        if (e[i] < 1e-20) e[i] = 1e-20;
    }
    memcpy(coeffs, a, (size_t)(order + 1) * sizeof(double));
    free(r); free(e); free(k); free(a); free(a2);
    return true;
}

void sptk_amdf(const float *iq, size_t n, size_t max_lag, double *out)
{
    if (iq == NULL || n == 0 || max_lag == 0 || out == NULL) return;
    for (size_t lag = 0; lag < max_lag; lag++) {
        double acc = 0.0;
        size_t cnt = n - lag;
        for (size_t i = 0; i < cnt; i++) {
            double d1 = sqrt((double)iq[2 * i] * iq[2 * i] +
                             (double)iq[2 * i + 1] * iq[2 * i + 1]);
            double d2 = sqrt((double)iq[2 * (i + lag)] * iq[2 * (i + lag)] +
                             (double)iq[2 * (i + lag) + 1] * iq[2 * (i + lag) + 1]);
            acc += fabs(d1 - d2);
        }
        out[lag] = acc / (double)cnt;
    }
}

void sptk_group_delay(const float *X, size_t n, double *out)
{
    if (X == NULL || n < 3 || out == NULL) return;
    for (size_t b = 0; b < n; b++) {
        size_t b1 = (b + 1) % n;
        double p0 = atan2((double)X[2 * b + 1], (double)X[2 * b]);
        double p1 = atan2((double)X[2 * b1 + 1], (double)X[2 * b1]);
        double d = p0 - p1;
        while (d > M_PI) d -= 2.0 * M_PI;
        while (d <= -M_PI) d += 2.0 * M_PI;
        out[b] = d * (double)n / (2.0 * M_PI);
    }
}

double sptk_goertzel(const float *iq, size_t n, double f)
{
    /* classic Goertzel on the COMPLEX signal: the single-bin DFT
     * S(f) = sum x(n) e^{-i 2 pi f n}, power = |S|^2 */
    if (iq == NULL || n == 0) return 0.0;
    double sr = 0.0, si = 0.0;
    double w = 2.0 * M_PI * f;
    for (size_t i = 0; i < n; i++) {
        double ph = w * (double)i;
        double c = cos(ph), s = sin(ph);
        double re = (double)iq[2 * i], im = (double)iq[2 * i + 1];
        sr += re * c + im * s;
        si += im * c - re * s;
    }
    return sr * sr + si * si;
}

size_t sptk_fir_design(int kind, double f1, double f2, float *taps,
                       size_t taps_len)
{
    if (taps == NULL || taps_len < 3 || (taps_len % 2) == 0 ||
        kind < 0 || kind > 3 || f1 <= 0.0 || f1 >= 0.5) {
        return 0;
    }
    const double m = (double)(taps_len - 1) / 2.0;
    for (size_t i = 0; i < taps_len; i++) {
        double t = (double)i - m;
        double h = 0.0;
        switch (kind) {
        case 0:
            h = 2.0 * f1 * sinc(2.0 * f1 * t);
            break;
        case 1:
            h = (fabs(t) < 1e-12 ? 1.0 - 2.0 * f1 : 0.0) -
                2.0 * f1 * sinc(2.0 * f1 * t);
            break;
        case 2:
            if (f2 <= f1 || f2 >= 0.5) return 0;
            h = 2.0 * f2 * sinc(2.0 * f2 * t) - 2.0 * f1 * sinc(2.0 * f1 * t);
            break;
        case 3:
            if (f2 <= f1 || f2 >= 0.5) return 0;
            h = (fabs(t) < 1e-12 ? 1.0 : 0.0) -
                (2.0 * f2 * sinc(2.0 * f2 * t) -
                 2.0 * f1 * sinc(2.0 * f1 * t));
            break;
        }
        taps[i] = (float)(h * (0.54 - 0.46 * cos(2.0 * M_PI * i /
                                                   (double)(taps_len - 1))));
    }
    return taps_len;
}

void sptk_fir_apply(float *iq, size_t n, const float *taps, size_t taps_len)
{
    if (iq == NULL || n == 0 || taps == NULL || taps_len == 0) return;
    float *out = malloc(2 * n * sizeof(float));
    if (out == NULL) return;
    const size_t half = taps_len / 2;
    for (size_t i = 0; i < n; i++) {
        double re = 0.0, im = 0.0;
        for (size_t j = 0; j < taps_len; j++) {
            size_t idx = (i + j >= half) ? i + j - half : 0;
            if (idx >= n) idx = n - 1;
            re += (double)taps[j] * iq[2 * idx];
            im += (double)taps[j] * iq[2 * idx + 1];
        }
        out[2 * i] = (float)re;
        out[2 * i + 1] = (float)im;
    }
    memcpy(iq, out, 2 * n * sizeof(float));
    free(out);
}

void sptk_median_filter(float *iq, size_t n)
{
    if (iq == NULL || n < 5) return;
    float *mag = malloc(n * sizeof(float));
    if (mag == NULL) return;
    for (size_t i = 0; i < n; i++) {
        mag[i] = sqrtf(iq[2 * i] * iq[2 * i] + iq[2 * i + 1] * iq[2 * i + 1]);
    }
    for (size_t i = 2; i + 2 < n; i++) {
        float v[5];
        for (int j = 0; j < 5; j++) v[j] = mag[i - 2 + (size_t)j];
        for (int a = 0; a < 4; a++) {
            for (int b = a + 1; b < 5; b++) {
                if (v[b] < v[a]) {
                    float t = v[a];
                    v[a] = v[b];
                    v[b] = t;
                }
            }
        }
        float med = v[2];
        if (mag[i] > 1e-12) {
            float g = med / mag[i];
            iq[2 * i] *= g;
            iq[2 * i + 1] *= g;
        }
    }
    free(mag);
}
