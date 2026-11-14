/*
 * fclk.c — K-subcarrier averaging analyses (see fclk.h).
 */
#include "fclk.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ------------------------------------------------------------------ */
/* induced lattice scan                                               */
/* ------------------------------------------------------------------ */

static double cell_z(const double *values, size_t n, double delta)
{
    double R = 0.0;

    fcl_circular_stats(values, n, 0.0, delta, NULL, &R);
    return (double)n * R * R;
}

void fclk_scan_block(const double *values, size_t n, double min_hz,
                     double max_hz, double df_hz, int kmax, size_t reps,
                     FclRng *rng, FclkScanCell *cells)
{
    double width = max_hz - min_hz;
    double *sample = malloc(n * sizeof(*sample));

    if (cells == NULL || kmax < 1) return;
    if (kmax > FCLK_MAX) kmax = FCLK_MAX;
    if (values == NULL || n < 3 || width <= 0.0 || sample == NULL) {
        free(sample);
        for (int K = 1; K <= kmax; K++) {
            cells[K - 1].df_hz = df_hz;
            cells[K - 1].K = K;
            cells[K - 1].valid = false;
            cells[K - 1].p_perm = 1.0;
            cells[K - 1].p_adj = 1.0;
        }
        return;
    }
    for (int K = 1; K <= kmax; K++) {
        FclkScanCell *c = &cells[K - 1];
        double delta = df_hz / (double)K;
        double observed, hits = 0.0;
        double R = 0.0;

        c->df_hz = df_hz;
        c->K = K;
        c->delta_hz = delta;
        c->n = n;
        fcl_circular_stats(values, n, 0.0, delta, NULL, &R);
        c->R = R;
        c->z = (double)n * R * R;
        observed = c->z;
        for (size_t r = 0; r < reps; r++) {
            for (size_t i = 0; i < n; i++) {
                sample[i] = min_hz + width * fcl_rng_uniform(rng);
            }
            if (cell_z(sample, n, delta) >= observed) hits += 1.0;
        }
        c->p_perm = (hits + 1.0) / ((double)reps + 1.0);
        c->p_adj = c->p_perm * (double)kmax;
        if (c->p_adj > 1.0) c->p_adj = 1.0;
        c->valid = true;
    }
    free(sample);
}

int fclk_finest_significant(const FclkScanCell *cells, int kmax, double alpha)
{
    int best = 0;

    if (cells == NULL) return 0;
    for (int K = 1; K <= kmax && K <= FCLK_MAX; K++) {
        if (cells[K - 1].valid && cells[K - 1].p_adj < alpha) best = K;
    }
    return best;
}

/* ------------------------------------------------------------------ */
/* latent grid mixture                                                */
/* ------------------------------------------------------------------ */

/* Build the induced lattice points inside the block. */
static size_t lattice_points(double f0_hz, double delta, double min_hz,
                             double max_hz, double **means_out)
{
    long l_min = (long)ceil((min_hz - delta * 0.5 - f0_hz) / delta);
    long l_max = (long)floor((max_hz + delta * 0.5 - f0_hz) / delta);
    size_t m;
    double *means;

    if (delta <= 0.0 || l_max < l_min) return 0;
    m = (size_t)(l_max - l_min + 1);
    if (m > 256) return 0;   /* too fine for a stable mixture fit */
    means = malloc(m * sizeof(*means));
    if (means == NULL) return 0;
    for (size_t i = 0; i < m; i++) {
        means[i] = f0_hz + (double)(l_min + (long)i) * delta;
    }
    *means_out = means;
    return m;
}

bool fclk_mixture_fit(const double *values, size_t n, double f0_hz,
                      double df_hz, int K, double min_hz, double max_hz,
                      FclkMixtureFit *out)
{
    double delta, *means = NULL, *pi = NULL, *resp = NULL, *sum_r = NULL;
    size_t m, it;
    double sigma, loglik = 0.0, entropy = 0.0;

    if (out == NULL) return false;
    memset(out, 0, sizeof(*out));
    out->K = K;
    if (values == NULL || n < 10 || K < 1 || K > FCLK_MAX) return false;
    delta = df_hz / (double)K;
    out->delta_hz = delta;
    m = lattice_points(f0_hz, delta, min_hz, max_hz, &means);
    if (m < 2 || m == 0) {
        free(means);
        return false;
    }
    pi = malloc(m * sizeof(*pi));
    resp = malloc(m * n * sizeof(*resp));
    sum_r = malloc(m * sizeof(*sum_r));
    if (pi == NULL || resp == NULL || sum_r == NULL) {
        free(means);
        free(pi);
        free(resp);
        free(sum_r);
        return false;
    }
    /* initialise: uniform component weights, sigma from nearest-lattice
     * residual spread */
    sigma = 0.05 * delta;
    {
        double sum2 = 0.0;

        for (size_t i = 0; i < n; i++) {
            double best = 1e300;

            for (size_t j = 0; j < m; j++) {
                double d = fabs(values[i] - means[j]);

                if (d < best) best = d;
            }
            sum2 += best * best;
        }
        {
            double s = sqrt(sum2 / (double)n);

            if (s > 1e-9) sigma = s;
        }
        if (sigma < 1e-6) sigma = delta * 1e-3;
    }
    for (size_t j = 0; j < m; j++) pi[j] = 1.0 / (double)m;
    for (it = 0; it < 80; it++) {
        double new_sigma2 = 0.0, change = 0.0;
        double inv2s2 = 1.0 / (2.0 * sigma * sigma);

        for (size_t i = 0; i < n; i++) {
            double sum = 0.0;

            for (size_t j = 0; j < m; j++) {
                double d = values[i] - means[j];
                double v = pi[j] * exp(-d * d * inv2s2);

                resp[j * n + i] = v;
                sum += v;
            }
            if (sum <= 1e-300) {
                /* fall back to the nearest component */
                double best = 1e300;
                size_t bj = 0;

                for (size_t j = 0; j < m; j++) {
                    double d = fabs(values[i] - means[j]);

                    if (d < best) {
                        best = d;
                        bj = j;
                    }
                }
                for (size_t j = 0; j < m; j++) resp[j * n + i] = 0.0;
                resp[bj * n + i] = 1.0;
                sum = 1.0;
            }
            for (size_t j = 0; j < m; j++) resp[j * n + i] /= sum;
        }
        memset(sum_r, 0, m * sizeof(*sum_r));
        for (size_t j = 0; j < m; j++) {
            for (size_t i = 0; i < n; i++) {
                double r = resp[j * n + i];

                sum_r[j] += r;
                {
                    double d = values[i] - means[j];

                    new_sigma2 += r * d * d;
                }
            }
        }
        new_sigma2 /= (double)n;
        for (size_t j = 0; j < m; j++) {
            double new_pi = sum_r[j] / (double)n;

            change += fabs(new_pi - pi[j]);
            pi[j] = new_pi;
        }
        sigma = sqrt(new_sigma2 > 1e-12 ? new_sigma2 : 1e-12);
        if (change < 1e-6 && it > 5) break;
        (void)change;
    }
    /* log-likelihood and the classification entropy */
    for (size_t i = 0; i < n; i++) {
        double sum = 0.0;

        for (size_t j = 0; j < m; j++) {
            double d = values[i] - means[j];
            double v = pi[j] *
                       exp(-d * d / (2.0 * sigma * sigma)) /
                       (sigma * sqrt(2.0 * M_PI));

            sum += v;
        }
        if (sum <= 1e-300) sum = 1e-300;
        loglik += log(sum);
        for (size_t j = 0; j < m; j++) {
            if (resp[j * n + i] > 1e-12) {
                entropy -= resp[j * n + i] * log(resp[j * n + i]);
            }
        }
    }
    out->n_components = m;
    out->n_params = m;   /* m-1 weights + sigma */
    out->loglik = loglik;
    out->bic = -2.0 * loglik + (double)out->n_params * log((double)n);
    out->entropy = entropy;
    out->icl = out->bic + 2.0 * entropy;
    out->sigma = sigma;
    out->iterations = (int)it;
    out->valid = true;
    free(means);
    free(pi);
    free(resp);
    free(sum_r);
    return true;
}

int fclk_bic_select(const FclkMixtureFit *fits, int n_fits,
                    double *delta_bic_vs_k1)
{
    int best = 0;
    double best_bic = 0.0, bic_k1 = 0.0;
    bool have = false;

    if (delta_bic_vs_k1 != NULL) *delta_bic_vs_k1 = 0.0;
    for (int i = 0; i < n_fits; i++) {
        if (!fits[i].valid) continue;
        if (fits[i].K == 1) bic_k1 = fits[i].bic;
        if (!have || fits[i].bic < best_bic) {
            best = fits[i].K;
            best_bic = fits[i].bic;
            have = true;
        }
    }
    if (have && delta_bic_vs_k1 != NULL) *delta_bic_vs_k1 = bic_k1 - best_bic;
    return have ? best : 0;
}

int fclk_icl_select(const FclkMixtureFit *fits, int n_fits)
{
    int best = 0;
    double best_icl = 0.0;
    bool have = false;

    for (int i = 0; i < n_fits; i++) {
        if (!fits[i].valid) continue;
        if (!have || fits[i].icl < best_icl) {
            best = fits[i].K;
            best_icl = fits[i].icl;
            have = true;
        }
    }
    return have ? best : 0;
}

/* ------------------------------------------------------------------ */
/* dispersion and CF diagnostics                                      */
/* ------------------------------------------------------------------ */

bool fclk_dispersion_estimate(const double *values, size_t n, double df_hz,
                              int kmax, int *K_hat, double *sigma1_2,
                              double *sigma_best_2)
{
    double R1 = 0.0, best_sigma = 1e300;
    int best_k = 0;

    if (K_hat != NULL) *K_hat = 0;
    if (sigma1_2 != NULL) *sigma1_2 = 0.0;
    if (sigma_best_2 != NULL) *sigma_best_2 = 0.0;
    if (values == NULL || n < 8 || df_hz <= 0.0) return false;
    if (kmax > FCLK_MAX) kmax = FCLK_MAX;
    fcl_circular_stats(values, n, 0.0, df_hz, NULL, &R1);
    {
        double s1 = 1.0 - R1;

        if (sigma1_2 != NULL) *sigma1_2 = s1;
        if (s1 < 0.005 || s1 > 0.6) {
            /* The reference must be a *concentrated but not degenerate*
             * single-subcarrier distribution: a dispersion near 0 means the
             * data are already on the grid (K is then not identifiable from
             * dispersion), and a dispersion near 1 means the K=1 phases are
             * effectively uniform (the reference carries no information). */
            return false;
        }
        for (int K = 2; K <= kmax; K++) {
            double RK = 0.0, sK;

            fcl_circular_stats(values, n, 0.0, df_hz / (double)K, NULL, &RK);
            sK = 1.0 - RK;
            if (sK < best_sigma) {
                best_sigma = sK;
                best_k = K;
            }
        }
    }
    if (best_k == 0) return false;
    if (sigma_best_2 != NULL) *sigma_best_2 = best_sigma;
    {
        double ratio = (sigma1_2 != NULL ? *sigma1_2 : (1.0 - R1)) /
                       (best_sigma > 1e-12 ? best_sigma : 1e-12);
        int k = (int)floor(ratio + 0.5);

        if (k < 1) k = 1;
        if (k > kmax) k = kmax;
        if (K_hat != NULL) *K_hat = k;
    }
    return true;
}

/* Signed distance to the nearest induced lattice point, in Hz. */
double *fclk_residuals(const double *values, size_t n, double delta_hz,
                       double f0_hz)
{
    double *r;

    if (values == NULL || n == 0 || delta_hz <= 0.0) return NULL;
    r = malloc(n * sizeof(*r));
    if (r == NULL) return NULL;
    for (size_t i = 0; i < n; i++) {
        double l = floor((values[i] - f0_hz) / delta_hz + 0.5);

        r[i] = values[i] - (f0_hz + l * delta_hz);
    }
    return r;
}

int fclk_cf_slope(const double *values, size_t n, double df_hz, double f0_hz,
                  int kmax, double min_hz, double max_hz, double *r2_out)
{
    (void)min_hz;
    (void)max_hz;
    double *res_ref, *res_k;
    int best_k = 0;
    double best_r2 = 0.0;

    if (r2_out != NULL) *r2_out = 0.0;
    if (values == NULL || n < 12 || df_hz <= 0.0) return 0;
    if (kmax > FCLK_MAX) kmax = FCLK_MAX;
    res_ref = fclk_residuals(values, n, df_hz, f0_hz);
    if (res_ref == NULL) return 0;
    /* normalise reference residuals to cycles of the induced spacing */
    for (size_t i = 0; i < n; i++) res_ref[i] /= df_hz;
    for (int K = 1; K <= kmax; K++) {
        double delta_k = df_hz / (double)K;
        double num = 0.0, den = 0.0, syy = 0.0, sxx = 0.0, sxy = 0.0;
        size_t used = 0;
        double slope = 0.0, r2 = 0.0;

        res_k = fclk_residuals(values, n, delta_k, f0_hz);
        if (res_k == NULL) break;
        for (size_t i = 0; i < n; i++) res_k[i] /= delta_k;
        /* t sampling: cycles over the residual range */
        for (int ti = 1; ti <= 12; ti++) {
            double t = 0.05 * (double)ti;
            double cr = 0.0, ci = 0.0, ck = 0.0, sk = 0.0;
            double lr, lk;

            for (size_t i = 0; i < n; i++) {
                double a = 2.0 * M_PI * t * res_ref[i];
                double b = 2.0 * M_PI * t * res_k[i];

                cr += cos(a);
                ci += sin(a);
                ck += cos(b);
                sk += sin(b);
            }
            lr = hypot(cr / (double)n, ci / (double)n);
            lk = hypot(ck / (double)n, sk / (double)n);
            if (lr < 1e-3 || lk < 1e-3 || lr > 1.0 || lk > 1.0) continue;
            lr = log(lr);
            lk = log(lk);
            sxx += lr * lr;
            sxy += lr * lk;
            syy += lk * lk;
            used++;
        }
        if (used >= 3 && sxx > 1e-12) {
            slope = sxy / sxx;
            r2 = (syy > 1e-12) ? (sxy * sxy) / (sxx * syy) : 0.0;
            if (slope > 0.0 && r2 > best_r2) {
                best_r2 = r2;
                best_k = (int)floor(slope + 0.5);
                if (best_k < 1) best_k = 1;
                if (best_k > kmax) best_k = kmax;
            }
        }
        free(res_k);
        (void)num;
        (void)den;
    }
    free(res_ref);
    if (r2_out != NULL) *r2_out = best_r2;
    return best_k;
}
