/*
 * fclk.h — can the measured frequencies be averages of K subcarriers?
 *
 * The observed Y_i are modelled as equal-weight averages of K latent
 * subcarriers drawn from a candidate grid f0 + m*df:
 *
 *     Y_i = f0 + (df/K) * sum_{k=1..K} m_ik + noise
 *
 * so the averages live on the *induced* lattice f0 + l*(df/K). For every
 * candidate (df, K) this module answers:
 *
 *   - does the induced lattice show clustering (Rayleigh/permutation),
 *   - which K best explains the data by a grid-constrained Gaussian
 *     mixture (EM fitted component weights p_j on the induced lattice,
 *     with BIC and ICL model selection),
 *   - what does the phase dispersion say about the multiplicity,
 *   - what does the characteristic-function slope say (using the K=1
 *     residuals as a single-subcarrier reference),
 *   - which K is preferred overall, with the identifiability caveat that
 *     K is only identifiable under a structural assumption (here: equal
 *     weights, latent subcarriers on the candidate grid, shared Gaussian
 *     measurement noise).
 */
#ifndef TE15_FCLK_H
#define TE15_FCLK_H

#include <stdbool.h>
#include <stddef.h>

#include "fclattice.h"

#define FCLK_MAX 6   /* largest candidate averaging multiplicity */

/* One induced-lattice cell: spacing df averaged over K subcarriers. */
typedef struct {
    double df_hz;
    int    K;
    double delta_hz;      /* df/K                                     */
    size_t n;
    double R;             /* mean resultant length of phases mod delta */
    double z;             /* n R^2                                    */
    double p_perm;        /* permutation p (uniform frequencies)      */
    double p_adj;         /* Bonferroni-adjusted over K candidates    */
    bool   valid;
} FclkScanCell;

/* Fill cells[0..kmax-1] (K = 1..kmax) for one block and spacing.
 * `reps` permutation replicates (>=100 recommended). */
void fclk_scan_block(const double *values, size_t n, double min_hz,
                     double max_hz, double df_hz, int kmax, size_t reps,
                     FclRng *rng, FclkScanCell *cells);

/* Largest K with a significant cell (alpha, Bonferroni over K), or 0. */
int fclk_finest_significant(const FclkScanCell *cells, int kmax, double alpha);

/* Grid-constrained Gaussian mixture on the induced lattice: component
 * means at f0 + l*delta inside the block, shared sigma, weights p_j
 * estimated by EM. */
typedef struct {
    int    K;
    double delta_hz;
    size_t n_components;
    size_t n_params;
    double loglik;
    double bic;
    double icl;          /* BIC + 2*entropy (classification uncertainty) */
    double entropy;
    double sigma;
    int    iterations;
    bool   valid;
} FclkMixtureFit;

bool fclk_mixture_fit(const double *values, size_t n, double f0_hz,
                      double df_hz, int K, double min_hz, double max_hz,
                      FclkMixtureFit *out);

/* Minimum BIC across fits[0..n_fits-1] (smaller K wins ties); 0 when no
 * fit is valid. *delta_bic_vs_k1 is filled when K=1 is among the fits. */
int fclk_bic_select(const FclkMixtureFit *fits, int n_fits,
                    double *delta_bic_vs_k1);
/* Minimum ICL. */
int fclk_icl_select(const FclkMixtureFit *fits, int n_fits);

/* Phase-dispersion multiplicity diagnostic: residual circular variance of
 * the phases modulo the induced spacing for each K (cells), and the
 * dispersion-based K estimate. The K=1 dispersion is the reference; the
 * estimate is rounded(sigma_1^2 / sigma_K^2) where sigma_K^2 is smallest.
 * valid=false when the reference dispersion is negligible (K is then not
 * identifiable from dispersion alone — e.g. already single-subcarrier
 * data) or n < 8. */
bool fclk_dispersion_estimate(const double *values, size_t n, double df_hz,
                              int kmax, int *K_hat, double *sigma1_2,
                              double *sigma_best_2);

/* Characteristic-function slope: the K-average residual CF is the K-th
 * power of the single-subcarrier residual CF, so log|phi_K(t)| is
 * proportional to log|phi_ref(t)| with slope K. The reference is the
 * empirical residual CF at K=1 (single-subcarrier assumption). Returns the
 * rounded slope (0 when unusable). */
int fclk_cf_slope(const double *values, size_t n, double df_hz, double f0_hz,
                  int kmax, double min_hz, double max_hz, double *r2_out);

/* Signed distance of each measurement to the nearest induced lattice
 * point, in Hz (malloc'd; caller frees). */
double *fclk_residuals(const double *values, size_t n, double delta_hz,
                       double f0_hz);

#endif /* TE15_FCLK_H */
