/*
 * sdist.c — distribution analysis via the empirical characteristic
 * function and the Laplace transform (moment-generating function).
 *
 * The mixdowned selection's real (I) component is treated as the
 * sequence under test: its empirical characteristic function is compared
 * against the Gaussian CF e^{i mu w - sigma^2 w^2/2} on an omega grid
 * spanning +-8/sigma, giving a Gaussian-likeness verdict, and its
 * empirical MGF is evaluated on s in [-3/sigma, +3/sigma] to probe the
 * tail behaviour. Whole-file statistic with long averaging gain.
 */
#include "sdist.h"
#include "debug.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "metrics.h"
#include "sptk.h"

#define SDIST_MAX_SAMPLES 262144 /* CF/MGF estimation sample cap */

void sdist_clear(SdistResult *r)
{
    if (r == NULL) return;
    free(r->omega);
    free(r->cf_re);
    free(r->cf_im);
    free(r->mgf_s);
    free(r->mgf);
    memset(r, 0, sizeof(*r));
}

bool sdist_analyze(const IqSource *src, const Region *region, SdistResult *out)
{
    sdist_clear(out);
    if (src == NULL || region == NULL) return false;

    Region r = *region;
    if (!region_normalize(&r, src)) return false;

    size_t s0 = (size_t)(r.time_min * src->sample_rate);
    size_t s1 = (size_t)(r.time_max * src->sample_rate);
    if (s1 > src->total_samples) s1 = src->total_samples;
    if (s0 >= s1) return false;
    size_t n_work = s1 - s0;
    if (n_work > TE15_MAX_ANALYSIS_SAMPLES) n_work = TE15_MAX_ANALYSIS_SAMPLES;
    if (n_work < 64) {
        return false;           /* too short for a distribution estimate */
    }

    /* private copy of the selected segment, mixdowned to the centre */
    float *seg = malloc(2 * n_work * sizeof(float));
    if (seg == NULL) return false;
    iq_source_read_complex(src, s0, seg, n_work);
    double offset = ((r.freq_min + r.freq_max) / 2.0) - src->center_freq;
    metrics_mixdown(seg, n_work, src->sample_rate, -offset, seg);

    /* x = the real (I) component sequence */
    float *x = malloc(n_work * sizeof(float));
    if (x == NULL) {
        free(seg);
        return false;
    }
    for (size_t i = 0; i < n_work; i++) x[i] = seg[2 * i];
    free(seg);

    /* sample mean and standard deviation */
    double mu = 0.0;
    for (size_t i = 0; i < n_work; i++) mu += (double)x[i];
    mu /= (double)n_work;
    double var = 0.0;
    for (size_t i = 0; i < n_work; i++) {
        double d = (double)x[i] - mu;
        var += d * d;
    }
    var /= (double)n_work;
    double sigma = sqrt(var);
    if (!(sigma > 0.0)) sigma = 1.0;    /* constant input: fall back */

    const int grid = SDIST_GRID;
    double *omega = malloc((size_t)grid * sizeof(double));
    double *cf_re = malloc((size_t)grid * sizeof(double));
    double *cf_im = malloc((size_t)grid * sizeof(double));
    double *mgf_s = malloc((size_t)grid * sizeof(double));
    double *mgf = malloc((size_t)grid * sizeof(double));
    if (omega == NULL || cf_re == NULL || cf_im == NULL ||
        mgf_s == NULL || mgf == NULL) {
        free(omega);
        free(cf_re);
        free(cf_im);
        free(mgf_s);
        free(mgf);
        free(x);
        return false;
    }

    const double omax = 8.0 / sigma;
    const double smax = 3.0 / sigma;
    for (int i = 0; i < grid; i++) {
        omega[i] = (double)i * omax / (double)(grid - 1);
        mgf_s[i] = -smax + (double)i * 2.0 * smax / (double)(grid - 1);
    }

    /* The empirical CF/MGF are simple averages: a capped sample count
     * (SDIST_MAX_SAMPLES) gives statistically equivalent estimates while
     * keeping the O(n x grid) trig loops fast on long selections (a
     * 13 M-sample selection took >10 s uncapped). */
    {
        size_t n_cf = n_work < SDIST_MAX_SAMPLES ? n_work : SDIST_MAX_SAMPLES;

        sptk_charfun(x, n_cf, omega, grid, cf_re, cf_im);
        sptk_mgf(x, n_cf, mgf_s, grid, mgf);
    }

    /* L2 distance to the Gaussian characteristic function */
    double acc = 0.0;
    for (int i = 0; i < grid; i++) {
        double decay = exp(-0.5 * sigma * sigma * omega[i] * omega[i]);
        double gr = decay * cos(mu * omega[i]);
        double gi = decay * sin(mu * omega[i]);
        double dr = cf_re[i] - gr;
        double di = cf_im[i] - gi;
        acc += dr * dr + di * di;
    }
    out->gauss_dist = sqrt(acc / (double)grid);
    out->gaussian_like = out->gauss_dist < 0.05;

    out->grid = grid;
    out->omega = omega;
    out->cf_re = cf_re;
    out->cf_im = cf_im;
    out->mgf_s = mgf_s;
    out->mgf = mgf;
    out->samples = n_work;

    snprintf(out->verdict, sizeof(out->verdict),
             out->gaussian_like ? "Gaussian-like (CF distance %.4f)"
                                : "Non-Gaussian (CF distance %.4f)",
             out->gauss_dist);
    snprintf(out->note, sizeof(out->note),
             "N = %zu samples, sigma = %.6f", n_work, sigma);

    TE15_LOG("\nDISTRIBUTION ANALYSIS\n"
           "N = %zu, sigma = %.6f\n"
           "CF-vs-Gaussian distance = %.4f -> %s\n"
           "MGF at s=+/-S_MAX: %.6e / %.6e\n",
           n_work, sigma, out->gauss_dist,
           out->gaussian_like ? "Gaussian-like" : "Non-Gaussian",
           mgf[0], mgf[grid - 1]);

    free(x);
    return true;
}
