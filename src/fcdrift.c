/*
 * fcdrift.c — warm-up drift estimation and correction (see fcdrift.h).
 */
#include "fcdrift.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fcdataset.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const char *fcd_model_name(FcdModel model)
{
    switch (model) {
    case FCD_MODEL_NONE:     return "none";
    case FCD_MODEL_CONSTANT: return "constant";
    case FCD_MODEL_LINEAR:   return "linear";
    case FCD_MODEL_EXP:      return "exponential";
    case FCD_MODEL_EXP2:     return "double exponential";
    }
    return "unknown";
}

/* ------------------------------------------------------------------ */
/* observation-time stub                                              */
/* ------------------------------------------------------------------ */

bool fcd_timestamps_available(void)
{
    /* The current schema (schema/fc_dataset.json) has no per-measurement
     * times. When it gains a `times`/`timestamps` array the loader will
     * expose it through fcd_round_times() and this returns true. */
    return false;
}

const double *fcd_round_times(const void *round, size_t *n_out)
{
    (void)round;
    if (n_out != NULL) *n_out = 0;
    return NULL;
}

/* ------------------------------------------------------------------ */
/* small numerics                                                     */
/* ------------------------------------------------------------------ */

static double median_copy(double *v, size_t n)
{
    double m;

    if (n == 0) return 0.0;
    for (size_t i = 1; i < n; i++) {
        double key = v[i];
        size_t j = i;

        while (j > 0 && v[j - 1] > key) {
            v[j] = v[j - 1];
            j--;
        }
        v[j] = key;
    }
    if (n % 2 == 1) return v[n / 2];
    m = 0.5 * (v[n / 2 - 1] + v[n / 2]);
    return m;
}

/* Weighted least squares for y = c + a1*e1 + a2*e2 (e2 may be NULL).
 * `w` may be NULL (unit weights). Returns false on a singular system. */
static bool wls_exp(const double *y, const double *e1, const double *e2,
                    const double *w, size_t n, double *c, double *a1,
                    double *a2)
{
    double s00 = 0, s01 = 0, s02 = 0, s11 = 0, s12 = 0, s22 = 0;
    double b0 = 0, b1 = 0, b2 = 0;
    double det;

    for (size_t i = 0; i < n; i++) {
        double wi = w != NULL ? w[i] : 1.0;
        double x1 = e1[i];
        double x2 = e2 != NULL ? e2[i] : 0.0;

        s00 += wi;
        s01 += wi * x1;
        s02 += wi * x2;
        s11 += wi * x1 * x1;
        s12 += wi * x1 * x2;
        s22 += wi * x2 * x2;
        b0 += wi * y[i];
        b1 += wi * x1 * y[i];
        b2 += wi * x2 * y[i];
    }
    if (e2 == NULL) {
        det = s00 * s11 - s01 * s01;
        if (getenv("TE15_DEBUG_DRIFT") != NULL) {
            fprintf(stderr, "  wls: n=%zu s00=%.3g s01=%.3g s11=%.3g det=%.3g\n",
                    n, s00, s01, s11, det);
        }
        if (fabs(det) < 1e-30) return false;
        *c = (b0 * s11 - b1 * s01) / det;
        *a1 = (b1 * s00 - b0 * s01) / det;
        if (a2 != NULL) *a2 = 0.0;
        return true;
    }
    /* 3x3 normal equations */
    {
        double m[3][4] = {
            { s00, s01, s02, b0 },
            { s01, s11, s12, b1 },
            { s02, s12, s22, b2 }
        };

        for (int col = 0; col < 3; col++) {
            int piv = col;
            double best = fabs(m[col][col]);

            for (int r = col + 1; r < 3; r++) {
                if (fabs(m[r][col]) > best) {
                    best = fabs(m[r][col]);
                    piv = r;
                }
            }
            if (best < 1e-30) return false;
            if (piv != col) {
                for (int c2 = col; c2 < 4; c2++) {
                    double t = m[col][c2];

                    m[col][c2] = m[piv][c2];
                    m[piv][c2] = t;
                }
            }
            for (int r = 0; r < 3; r++) {
                double f;

                if (r == col) continue;
                f = m[r][col] / m[col][col];
                for (int c2 = col; c2 < 4; c2++) m[r][c2] -= f * m[col][c2];
            }
        }
        *c = m[0][3] / m[0][0];
        *a1 = m[1][3] / m[1][1];
        *a2 = m[2][3] / m[2][2];
        return true;
    }
}

/* Linear fit y = c + s*x (weighted). */
static bool wls_linear(const double *y, const double *x, const double *w,
                       size_t n, double *c, double *s)
{
    double sw = 0, sx = 0, sxx = 0, sy = 0, sxy = 0;

    for (size_t i = 0; i < n; i++) {
        double wi = w != NULL ? w[i] : 1.0;

        sw += wi;
        sx += wi * x[i];
        sxx += wi * x[i] * x[i];
        sy += wi * y[i];
        sxy += wi * x[i] * y[i];
    }
    {
        double det = sw * sxx - sx * sx;

        if (fabs(det) < 1e-30 || sw <= 0.0) return false;
        *c = (sy * sxx - sxy * sx) / det;
        *s = (sxy * sw - sy * sx) / det;
    }
    return true;
}

/* Residual RMS of a fitted exponential/linear model, plus the residual
 * vector (for IRLS down-weighting). */
static double residual_rms(const double *y, const double *e1, const double *e2,
                           const double *x, size_t n, int model, double c,
                           double a1, double a2, double s, double *resid)
{
    double sum = 0.0;

    for (size_t i = 0; i < n; i++) {
        double fit;

        if (model == FCD_MODEL_LINEAR) {
            fit = c + s * x[i];
        } else if (model == FCD_MODEL_CONSTANT) {
            fit = c;
        } else {
            fit = c + a1 * e1[i] + (e2 != NULL ? a2 * e2[i] : 0.0);
        }
        resid[i] = y[i] - fit;
        sum += resid[i] * resid[i];
    }
    return sqrt(sum / (double)n);
}

/* Robust scale of residuals (MAD-based). */
static double robust_sigma(const double *resid, size_t n)
{
    double *absr = malloc(n * sizeof(*absr));
    double m;

    if (absr == NULL) return 1.0;
    for (size_t i = 0; i < n; i++) absr[i] = fabs(resid[i]);
    m = median_copy(absr, n);
    free(absr);
    return m * 1.4826 > 1e-12 ? m * 1.4826 : 1e-12;
}

/* Fit y = c + a*exp(-x/tau) with an IRLS robust loop; returns the BIC and
 * leaves the fit in the out parameters. */
static bool fit_exp(const double *y, const double *x, size_t n, double tau,
                    double *c, double *a, double *rms, double *resid)
{
    double *e1 = malloc(n * sizeof(*e1));
    double *w = calloc(n, sizeof(*w));
    double *r = malloc(n * sizeof(*r));
    bool ok = false;

    if (e1 == NULL || w == NULL || r == NULL) {
        free(e1);
        free(w);
        free(r);
        return false;
    }
    for (size_t i = 0; i < n; i++) {
        e1[i] = exp(-x[i] / tau);
        w[i] = 1.0;
    }
    for (int it = 0; it < 4; it++) {
        {
            double a2_dummy = 0.0;
            bool okw = wls_exp(y, e1, NULL, w, n, c, a, &a2_dummy);

            if (!okw) break;
        }
        {
            double rms0 = residual_rms(y, e1, NULL, x, n, FCD_MODEL_EXP, *c,
                                       *a, 0.0, 0.0, r);
            double sigma = robust_sigma(r, n);

            for (size_t i = 0; i < n; i++) {
                double z = fabs(r[i]) / (1.345 * sigma);

                w[i] = z > 1.0 ? 1.0 / z : 1.0;
            }
            if (it == 3 && rms0 >= 0.0) {
                *rms = rms0;
                ok = true;
            }
        }
    }
    if (ok && resid != NULL) memcpy(resid, r, n * sizeof(*resid));
    free(e1);
    free(w);
    free(r);
    return ok;
}

/* ------------------------------------------------------------------ */
/* estimation                                                         */
/* ------------------------------------------------------------------ */

bool fcd_estimate(const double *values, const double *times, size_t n,
                  FcdResult *out)
{
    double *x, *resid;
    double mean = 0.0, best_bic = 0.0;
    double c_const, c_lin = 0, s_lin = 0;
    double rms_const;
    /* exponential candidates */
    double exp_tau = 0.0, exp_c = 0.0, exp_a = 0.0, exp_rms = 0.0;
    double e2_t1 = 0.0, e2_t2 = 0.0, e2_c = 0.0, e2_a1 = 0.0, e2_a2 = 0.0;
    double e2_rms = 0.0, e2_bic = 0.0;

    if (out == NULL) return false;
    memset(out, 0, sizeof(*out));
    out->n = n;
    out->model = FCD_MODEL_NONE;
    if (values == NULL || n < 8) return false;
    x = malloc(n * sizeof(*x));
    resid = calloc(n, sizeof(*resid));
    if (x == NULL || resid == NULL) {
        free(x);
        free(resid);
        return false;
    }
    out->timestamps = times != NULL;
    for (size_t i = 0; i < n; i++) {
        x[i] = times != NULL ? times[i] : (double)i;
        mean += values[i];
    }
    mean /= (double)n;

    /* constant model */
    c_const = mean;
    rms_const = residual_rms(values, NULL, NULL, x, n, FCD_MODEL_CONSTANT,
                             c_const, 0, 0, 0, resid);
    best_bic = (double)n * log((rms_const * rms_const + 1e-30)) +
               1.0 * log((double)n);

    /* linear model */
    if (wls_linear(values, x, NULL, n, &c_lin, &s_lin)) {
        double rms = residual_rms(values, NULL, NULL, x, n, FCD_MODEL_LINEAR,
                                  c_lin, 0, 0, s_lin, resid);
        double bic = (double)n * log((rms * rms + 1e-30)) +
                     2.0 * log((double)n);

        if (bic < best_bic - 6.0) {
            out->model = FCD_MODEL_LINEAR;
            out->level = c_lin;
            out->slope = s_lin;
            out->rms_before = rms_const;
            out->rms_after = rms;
            best_bic = bic;
        } else if (out->model == FCD_MODEL_NONE) {
            out->level = mean;
            out->rms_before = rms_const;
            out->rms_after = rms_const;
        }
    }

    /* single exponential: grid search over tau with a robust least-squares
     * fit for (level, amplitude) — variable projection. */
    {
        double tau_min = 1.5, tau_max = (double)n * 2.5;
        double best_exp_bic = 1e300;
        double lo = log(tau_min), hi = log(tau_max);

        for (int gi = 0; gi < 32; gi++) {
            double tau = exp(lo + (hi - lo) * (double)gi / 31.0);
            double cc, aa, rr;

            if (!fit_exp(values, x, n, tau, &cc, &aa, &rr, NULL)) continue;
            {
                double bic = (double)n * log((rr * rr + 1e-30)) +
                             3.0 * log((double)n);

                if (bic < best_exp_bic) {
                    best_exp_bic = bic;
                    exp_tau = tau;
                    exp_c = cc;
                    exp_a = aa;
                    exp_rms = rr;
                }
            }
        }
        if (exp_tau > 0.0 && best_exp_bic < best_bic - 6.0) {
            out->model = FCD_MODEL_EXP;
            out->tau1 = exp_tau;
            out->amp1 = exp_a;
            out->level = exp_c;
            out->rms_before = rms_const;
            out->rms_after = exp_rms;
            best_bic = best_exp_bic;
        }
    }

    /* double exponential: fast + slow thermal transients. Grid over the
     * two time constants, linear least squares for the amplitudes. */
    if (out->model == FCD_MODEL_EXP) {
        double t1_lo = log(1.5), t1_hi = log((double)n / 3.0 + 1.0);
        double t2_lo = log(exp(t1_hi)), t2_hi = log((double)n * 6.0);
        double best2 = 1e300;

        for (int i1 = 0; i1 < 6; i1++) {
            double tau1 = exp(t1_lo + (t1_hi - t1_lo) * (double)i1 / 5.0);

            for (int i2 = 0; i2 < 6; i2++) {
                double tau2 = exp(t2_lo + (t2_hi - t2_lo) * (double)i2 / 5.0);
                double *e1 = malloc(n * sizeof(*e1));
                double *e2 = malloc(n * sizeof(*e2));
                double cc, a1, a2, rms, bic;

                if (e1 == NULL || e2 == NULL) {
                    free(e1);
                    free(e2);
                    continue;
                }
                for (size_t k = 0; k < n; k++) {
                    e1[k] = exp(-x[k] / tau1);
                    e2[k] = exp(-x[k] / tau2);
                }
                if (!wls_exp(values, e1, e2, NULL, n, &cc, &a1, &a2)) {
                    free(e1);
                    free(e2);
                    continue;
                }
                rms = residual_rms(values, e1, e2, x, n, FCD_MODEL_EXP2, cc,
                                   a1, a2, 0.0, resid);
                bic = (double)n * log((rms * rms + 1e-30)) +
                      5.0 * log((double)n);
                if (bic < best2) {
                    best2 = bic;
                    e2_t1 = tau1;
                    e2_t2 = tau2;
                    e2_c = cc;
                    e2_a1 = a1;
                    e2_a2 = a2;
                    e2_rms = rms;
                    e2_bic = bic;
                }
                free(e1);
                free(e2);
            }
        }
        if (e2_t2 > 0.0 && e2_bic < best_bic - 6.0) {
            out->model = FCD_MODEL_EXP2;
            out->tau1 = e2_t1;
            out->tau2 = e2_t2;
            out->amp1 = e2_a1;
            out->amp2 = e2_a2;
            out->level = e2_c;
            out->rms_before = rms_const;
            out->rms_after = e2_rms;
            best_bic = e2_bic;
        }
    }

    /* settling time (95 % of the slow transient) */
    switch (out->model) {
    case FCD_MODEL_EXP:
        out->settle_time = out->tau1 * log(20.0);
        break;
    case FCD_MODEL_EXP2:
        out->settle_time = out->tau2 * log(20.0);
        break;
    case FCD_MODEL_LINEAR:
        out->settle_time = 0.0;
        break;
    default:
        out->settle_time = 0.0;
        break;
    }
    /* only correct when the drift is larger than the noise */
    if (out->model == FCD_MODEL_EXP || out->model == FCD_MODEL_EXP2) {
        double amp = fabs(out->amp1) + fabs(out->amp2);

        if (amp < 2.0 * (rms_const > 0.0 ? rms_const : 1e-9)) {
            out->corrected = false;
        } else {
            out->corrected = true;
        }
    } else if (out->model == FCD_MODEL_LINEAR) {
        double span = x[n - 1] - x[0];
        double amp = fabs(out->slope) * (span > 0.0 ? span : 1.0);

        out->corrected = amp >= 2.0 * (rms_const > 0.0 ? rms_const : 1e-9);
    }
    if (out->corrected && out->rms_after >= out->rms_before) {
        out->corrected = false;
    }
    free(x);
    free(resid);
    return out->model != FCD_MODEL_NONE;
}

void fcd_correct(const double *values, const double *times, size_t n,
                 const FcdResult *result, double *corrected)
{
    if (values == NULL || result == NULL || corrected == NULL) return;
    for (size_t i = 0; i < n; i++) {
        double t = times != NULL ? times[i] : (double)i;
        double v = values[i];

        switch (result->model) {
        case FCD_MODEL_EXP:
            v -= result->amp1 * exp(-t / result->tau1);
            break;
        case FCD_MODEL_EXP2:
            v -= result->amp1 * exp(-t / result->tau1);
            v -= result->amp2 * exp(-t / result->tau2);
            break;
        case FCD_MODEL_LINEAR: {
            double tref = times != NULL ? times[n - 1] : (double)(n - 1);

            v -= result->slope * (t - tref);
            break;
        }
        default:
            break;
        }
        corrected[i] = v;
    }
}

void fcd_pool(const FcdResult *results, size_t n, FcdGlobal *global)
{
    double *taus = malloc(n * sizeof(*taus));
    double *t95 = malloc(n * sizeof(*t95));
    double *amps = malloc(n * sizeof(*amps));
    size_t k = 0, corrected = 0;
    bool any_ts = false;

    if (global == NULL) {
        free(taus);
        free(t95);
        free(amps);
        return;
    }
    memset(global, 0, sizeof(*global));
    if (results == NULL || n == 0 || taus == NULL || t95 == NULL ||
        amps == NULL) {
        free(taus);
        free(t95);
        free(amps);
        return;
    }
    for (size_t i = 0; i < n; i++) {
        if (results[i].model != FCD_MODEL_EXP &&
            results[i].model != FCD_MODEL_EXP2) {
            continue;
        }
        taus[k] = results[i].model == FCD_MODEL_EXP2 ? results[i].tau2
                                                     : results[i].tau1;
        t95[k] = results[i].settle_time;
        amps[k] = fabs(results[i].amp1) + fabs(results[i].amp2);
        if (results[i].timestamps) any_ts = true;
        if (results[i].corrected) corrected++;
        k++;
    }
    if (k > 0) {
        global->tau_median = median_copy(taus, k);
        global->t95_median = median_copy(t95, k);
        global->amp_median = median_copy(amps, k);
        global->n_rounds = k;
        global->n_corrected = corrected;
        global->timestamps = any_ts;
    }
    free(taus);
    free(t95);
    free(amps);
}
