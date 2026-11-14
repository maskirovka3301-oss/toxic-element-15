/*
 * fclattice.h — subcarrier-lattice statistics for the Frequency Counter.
 *
 * The loaded datasets give lists of measured frequencies (Hz). This module
 * asks whether they cluster around the subcarriers of a candidate lattice
 * f0 + k*df (e.g. 4G/LTE 15 kHz, 5G NR 15/30/60/120/240 kHz), using:
 *
 *   - circular statistics on the folded phases (Rayleigh and V tests),
 *   - phase-dispersion minimisation (PDM) and a periodogram scan over df,
 *   - chi-square testing of lattice-centred bins,
 *   - Gaussian KDE enrichment at the lattice points,
 *   - a cluster-based permutation test over bins of the block,
 *   - Silverman's critical-bandwidth multimodality test,
 *   - permutation p-values throughout (robust for small n and for blocks
 *     that do not span an integer number of lattice cycles),
 *   - Benjamini-Hochberg and Bonferroni correction across tests,
 *   - ASCII phase histograms and lattice comb plots for interpretation.
 *
 * Everything is deterministic for a given seed so results are reproducible
 * and testable.
 */
#ifndef TE15_FCLATTICE_H
#define TE15_FCLATTICE_H

#include <stdbool.h>
#include <stddef.h>

#include "fcdataset.h"

/* Deterministic RNG (xorshift64*). */
typedef struct { unsigned long long state; } FclRng;

void   fcl_rng_seed(FclRng *rng, unsigned long long seed);
double fcl_rng_uniform(FclRng *rng);          /* [0,1)                    */
double fcl_rng_normal(FclRng *rng);           /* N(0,1) via Box-Muller    */

/* A block of measurements inside one frequency span [min_hz, max_hz]. */
typedef struct {
    char        label[160];   /* e.g. "dataset/round ..."                */
    char        dataset[96];
    const double *values;     /* borrowed from the dataset model         */
    size_t       n;
    double       min_hz;
    double       max_hz;
    bool         owns_values;  /* true when values were allocated here    */
} FclBlock;

/* Candidate lattice: f0 + k*df. */
typedef struct {
    double df_hz;
    double f0_hz;
} FclCandidate;

/* Generic rendered table (used as a DAG node payload). */
typedef struct {
    char   *title;
    char  **columns;
    size_t  n_columns;
    char ***cells;            /* cells[row][column]                      */
    size_t  n_rows;
    size_t  rows_cap;
    char   *notes;            /* optional free-form note / plot text     */
} FclTable;

FclTable *fcl_table_new(const char *title, const char *const *columns,
                        size_t n_columns);
void      fcl_table_free(void *table);
/* Append one row; the values are copied. */
void      fcl_table_row(FclTable *t, const char *const *cells);
/* Append a note line (printf style). */
void      fcl_table_note(FclTable *t, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));

/* ---- block construction --------------------------------------------- */

/* How the analyses treat multiple loaded datasets. */
typedef enum {
    FCL_SCOPE_PER_DATASET = 0,  /* blocks inside each dataset (default) */
    FCL_SCOPE_MERGED            /* all measurements in one list         */
} FclScope;

/* Build the analysis blocks. Per-dataset scope: every session round (all
 * its subjects combined) and every subject. Merged scope: one block with
 * every measurement of every dataset. The returned array is calloc'd (with
 * a zero terminator) and owns the combined copies; free with
 * fcl_blocks_free(). */
FclBlock *fcl_blocks_build(const FcDatasetList *list, FclScope scope,
                           size_t *n_blocks_out);
FclBlock *fcl_blocks_from_datasets(const FcDatasetList *list,
                                   size_t *n_blocks_out);
void      fcl_blocks_free(FclBlock *blocks);
/* Human-readable scope name for reports. */
const char *fcl_scope_name(FclScope scope);

/* 3GPP LTE / 5G NR numerology families. */
typedef enum {
    FCL_NUM_UNKNOWN = 0,
    FCL_NUM_LTE_PRACH,   /* 1.25 kHz, long PRACH formats           */
    FCL_NUM_NB_IOT,      /* 3.75 kHz, NB-IoT single-tone uplink    */
    FCL_NUM_LTE_EMBMS,   /* 7.5 kHz, LTE eMBMS                     */
    FCL_NUM_NR           /* 15/30/60/120/240 kHz (15 * 2^mu)       */
} FclNumerology;

typedef struct {
    double        df_hz;
    FclNumerology family;
    const char   *name;
} FclNumerologySpec;

/* The 3GPP numerologies in priority order (15..240 kHz first, then the
 * dedicated 7.5 / 3.75 / 1.25 kHz channels). 5 kHz is deliberately absent:
 * it is not a 3GPP numerology. */
extern const FclNumerologySpec fcl_numerologies[];
extern const size_t fcl_n_numerologies;
const FclNumerologySpec *fcl_numerology_for(double df_hz, double tol_hz);
bool        fcl_is_3gpp_spacing(double df_hz);
const char *fcl_numerology_name(double df_hz);

/* The 3GPP candidate spacings (Hz) used by every analysis. */
extern const double fcl_spacings_hz[];
extern const size_t fcl_n_spacings;

/* Non-3GPP fallback candidate spacings, evaluated only when no 3GPP
 * numerology can explain the measurements. */
extern const double fcl_non3gpp_spacings_hz[];
extern const size_t fcl_n_non3gpp_spacings;
extern const double fcl_offsets_hz[];
extern const size_t fcl_n_offsets;

/* ---- circular tests -------------------------------------------------- */

/* Fold a frequency to [0,1): ((f - f0) mod df) / df. */
double fcl_fold(double hz, double f0_hz, double df_hz);

/* Mean resultant length and mean phase (radians) of folded values. */
void fcl_circular_stats(const double *values, size_t n, double f0_hz,
                        double df_hz, double *mean_angle, double *R);
double fcl_rayleigh_p(size_t n, double R);          /* asymptotic           */
double fcl_v_statistic(size_t n, double R, double mean_angle,
                       double direction);           /* effect (0..1)        */

/* Permutation p-values (uniform frequencies inside [min_hz, max_hz],
 * folded exactly like the data). `replicates` >= 100 recommended. */
double fcl_perm_rayleigh(const double *values, size_t n, double f0_hz,
                         double df_hz, double min_hz, double max_hz,
                         size_t replicates, FclRng *rng);
double fcl_perm_v(const double *values, size_t n, double f0_hz, double df_hz,
                  double min_hz, double max_hz, double direction,
                  size_t replicates, FclRng *rng);
/* Lower-tail permutation p for a small-is-significant statistic. */
double fcl_perm_pdm(const double *values, size_t n, double f0_hz,
                    double df_hz, double min_hz, double max_hz,
                    size_t bins, size_t replicates, FclRng *rng);

/* Phase-dispersion minimisation statistic (small = clustering; ~1 = none). */
double fcl_pdm(const double *values, size_t n, double f0_hz, double df_hz,
               size_t bins);

/* Phase histogram (equal-width phase bins) rendered as an ASCII rose. */
char *fcl_rose_plot(const double *values, size_t n, double f0_hz,
                    double df_hz, int n_bins, int height);

/* Lattice comb plot: distribution of measurements relative to the lattice
 * points, rendered as ASCII. */
char *fcl_comb_plot(const double *values, size_t n, double f0_hz,
                    double df_hz, int n_bins, int height);

/* ---- binning / density ---------------------------------------------- */

/* Chi-square test of lattice-centred bins: bins of width frac*df centred
 * on the lattice; expected counts from a uniform distribution over the
 * block. Returns the statistic and df; *valid is false when the expected
 * counts are too small. */
double fcl_chi2_lattice(const double *values, size_t n, double f0_hz,
                        double df_hz, double min_hz, double max_hz,
                        double frac, size_t *df_out, bool *valid);

/* Regularised upper incomplete gamma Q(a,x) — chi-square tail helper. */
double fcl_gamma_q(double a, double x);
double fcl_chi2_p(double chi2, size_t df);

/* Gaussian KDE enrichment: density at the lattice points relative to the
 * mean density in the block (1 = no enrichment). */
double fcl_kde_enrichment(const double *values, size_t n, double f0_hz,
                          double df_hz, double min_hz, double max_hz);
double fcl_perm_enrichment(const double *values, size_t n, double f0_hz,
                           double df_hz, double min_hz, double max_hz,
                           size_t replicates, FclRng *rng);

/* Cluster-based permutation test over `bins` equal-width bins of the block.
 * Returns the maximum cluster mass, its extent (bins) and a permutation
 * p-value. */
void fcl_cluster_test(const double *values, size_t n, double min_hz,
                      double max_hz, size_t bins, size_t replicates,
                      FclRng *rng, double *max_mass_out,
                      size_t *extent_out, double *p_out);

/* ---- multimodality --------------------------------------------------- */

/* Silverman's critical-bandwidth test for k modes (k=1: unimodality).
 * *n_modes receives the number of modes at the critical bandwidth.
 * Returns the bootstrap p-value (small: more than k modes). Blocks with
 * n > 400 are skipped (p = -1). */
double fcl_silverman(const double *values, size_t n, size_t k_modes,
                     size_t replicates, FclRng *rng, size_t *n_modes_out);

/* ---- multiple testing ------------------------------------------------ */

/* In-place Benjamini-Hochberg adjustment; returns the number of rejections
 * at level alpha. */
size_t fcl_bh_adjust(double *p, size_t n, double alpha);
/* Bonferroni-adjusted p. */
double fcl_bonferroni(double p, size_t n);

#endif /* TE15_FCLATTICE_H */
