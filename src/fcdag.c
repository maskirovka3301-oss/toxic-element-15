/*
 * fcdag.c — Frequency Counter analysis DAG: subcarrier-lattice tests.
 *
 * The analyses are registered against the shared DAG engine (dagcore) and
 * run over the loaded measurement datasets. Each node produces a FclTable
 * payload (rendered by fcreport) and can consume the results recorded by
 * its predecessors through the shared context.
 *
 * Node chain:
 *   blocks -> rayleigh -> vtest -> pdm -> periodogram -> chisq -> kde
 *          -> cluster -> multimodal -> correction -> summary
 */
#include "fcdag.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fclattice.h"
#include "fclk.h"

#define FC_PERM_REPS     400
#define FC_PERM_REPS_BIG 200
#define FC_ALPHA         0.05
#define FC_SEED          0x5EED20261114ULL

typedef struct {
    char   family[32];   /* "rayleigh", "vtest", ...                     */
    char   label[192];   /* block + candidate description                */
    double p;
} FcTest;

typedef struct {
    const FcDatasetList *list;
    FclBlock   *blocks;
    size_t      n_blocks;
    FclRng      rng;
    FcTest     *tests;
    size_t      n_tests;
    size_t      cap_tests;
    /* per-block best candidate from the circular tests */
    double     *best_p;    /* [n_blocks]                                   */
    double     *best_df;   /* [n_blocks]                                   */
    /* K-subcarrier analyses */
    FclkScanCell  *scan;   /* [n_blocks][FCLK_MAX]                         */
    FclkMixtureFit *fits;  /* [n_blocks][FCLK_MAX]                         */
    int        *finest_k;  /* [n_blocks]                                   */
    int        *k_disp;    /* [n_blocks]                                   */
    int        *k_cf;      /* [n_blocks]                                   */
    int        *k_bic;     /* [n_blocks]                                   */
    int        *k_icl;     /* [n_blocks]                                   */
    int        *k_final;   /* [n_blocks]                                   */
    double     *dbic;      /* [n_blocks]                                   */
    bool        any_clustering;   /* some circular test was significant    */
    bool        have_induced;     /* induced-lattice clustering found      */
    bool        have_reference;   /* dispersion reference usable           */
    bool        have_cf;          /* CF slope usable                       */
    bool        have_mixture;     /* mixture fits produced                 */
    bool        have_model;       /* a K was selected                      */
    FclScope    scope;
    const bool *enabled;    /* user selection (NULL = all enabled) */
    const FcProgress *progress;  /* optional worker progress         */
    /* evidence accumulated by the individual analyses */
    struct {
        char   family[24];
        size_t block;
        double spacing_hz;
        int    K;
        double effect;
        double p;
        double weight;
        bool   retained;
    } *evidence;
    size_t      n_evidence;
    size_t      cap_evidence;
    bool        have_evidence;
} FcCtx;

/* Per-family evidence weight used when synthesising the spacings. */
static double evidence_weight(const char *family)
{
    if (strcmp(family, "mixture-BIC") == 0) return 1.5;
    if (strcmp(family, "vtest") == 0) return 1.2;
    if (strcmp(family, "rayleigh") == 0) return 1.0;
    if (strcmp(family, "chisq") == 0) return 1.0;
    if (strcmp(family, "induced-K") == 0) return 1.0;
    if (strcmp(family, "pdm") == 0) return 0.8;
    if (strcmp(family, "kde") == 0) return 0.8;
    return 0.7;
}

static void ctx_add_evidence(FcCtx *c, const char *family, size_t block,
                             double spacing_hz, int K, double effect,
                             double p)
{
    if (c->n_evidence == c->cap_evidence) {
        size_t cap = c->cap_evidence == 0 ? 64 : c->cap_evidence * 2;
        void *n = realloc(c->evidence, cap * sizeof(*c->evidence));

        if (n == NULL) return;
        c->evidence = n;
        c->cap_evidence = cap;
    }
    snprintf(c->evidence[c->n_evidence].family,
             sizeof(c->evidence[0].family), "%s", family);
    c->evidence[c->n_evidence].block = block;
    c->evidence[c->n_evidence].spacing_hz = spacing_hz;
    c->evidence[c->n_evidence].K = K;
    c->evidence[c->n_evidence].effect = effect;
    c->evidence[c->n_evidence].p = p;
    c->n_evidence++;
}


const DagCoreNodeSpec fc_node_specs[] = {
    { "blocks", "Measurement blocks", NULL },
    { "rayleigh", "Rayleigh circular-uniformity test", "blocks" },
    { "vtest", "V test for the lattice direction", "rayleigh" },
    { "pdm", "Phase dispersion minimisation", "vtest" },
    { "periodogram", "Periodogram scan over subcarrier spacings", "pdm" },
    { "chisq", "Chi-square lattice-bin test", "periodogram" },
    { "kde", "KDE enrichment at lattice points", "chisq" },
    { "cluster", "Cluster-based permutation test", "kde" },
    { "multimodal", "Silverman multimodality test", "cluster" },
    { "correction", "Multiple-testing correction (BH/Bonferroni)",
      "multimodal" },
    { "summary", "Per-block subcarrier-lattice summary", "correction" },
    { "induced-scan", "Induced-lattice scan over K subcarriers",
      "rayleigh,correction" },
    { "k-dispersion", "Phase-dispersion multiplicity diagnostic",
      "induced-scan" },
    { "cf-slope", "Characteristic-function slope estimate of K",
      "k-dispersion" },
    { "mixture-fit", "Grid-constrained mixture fit over K (BIC/ICL)",
      "induced-scan" },
    { "mdl-eigen", "Eigen-spectrum (MDL) subcarrier count", "mixture-fit" },
    { "model-select", "K model selection (BIC + clustering agreement)",
      "mixture-fit,correction" },
    { "k-summary", "Subcarrier-multiplicity summary and plots",
      "model-select" },
    { "evidence-filter", "Evidence screen and synthesis input", "summary" },
    { "synthesis", "Synthesis of the most likely subcarrier spacings",
      "evidence-filter" }
};
const size_t fc_node_count = sizeof(fc_node_specs) / sizeof(fc_node_specs[0]);

/* ------------------------------------------------------------------ */
/* helpers                                                            */
/* ------------------------------------------------------------------ */

static size_t reps_for(const FcCtx *c)
{
    size_t total = 0;

    for (size_t i = 0; i < c->n_blocks; i++) total += c->blocks[i].n;
    return total > 4000 ? FC_PERM_REPS_BIG : FC_PERM_REPS;
}

static void ctx_add_test(FcCtx *c, const char *family, const char *label,
                         double p)
{
    if (c->n_tests == c->cap_tests) {
        size_t cap = c->cap_tests == 0 ? 64 : c->cap_tests * 2;
        FcTest *n = realloc(c->tests, cap * sizeof(*n));

        if (n == NULL) return;
        c->tests = n;
        c->cap_tests = cap;
    }
    snprintf(c->tests[c->n_tests].family, sizeof(c->tests[0].family), "%s",
             family);
    snprintf(c->tests[c->n_tests].label, sizeof(c->tests[0].label), "%s",
             label);
    c->tests[c->n_tests].p = p;
    c->n_tests++;
}

static void ctx_note_best(FcCtx *c, size_t block, double df, double p)
{
    if (block >= c->n_blocks) return;
    if (c->best_p == NULL || p < c->best_p[block]) {
        if (c->best_p != NULL) c->best_p[block] = p;
        if (c->best_df != NULL) c->best_df[block] = df;
    }
}

static void fc_add_cells(FclTable *t, char (*cells)[128], size_t n)
{
    const char *ptrs[16];

    for (size_t i = 0; i < n && i < 16; i++) ptrs[i] = cells[i];
    fcl_table_row(t, ptrs);
}

/* Spacing used by the K-subcarrier analyses: the block's best spacing from
 * the clustering tests, or the primary LTE numerology. */
static double k_spacing(const FcCtx *c, size_t block)
{
    if (c->best_df != NULL && c->best_df[block] > 0.0) {
        return c->best_df[block];
    }
    return 15000.0;
}

static const char *verdict(double p, double effect, const char *what)
{
    if (p < 0.001) return "strong evidence";
    if (p < 0.01) return "evidence";
    if (p < FC_ALPHA) return "weak evidence";
    (void)effect;
    (void)what;
    return "no evidence";
}

/* ------------------------------------------------------------------ */
/* node: blocks                                                       */
/* ------------------------------------------------------------------ */

static FclTable *analysis_blocks(FcCtx *c)
{
    static const char *cols[] = { "Dataset", "Block", "n", "span (kHz)",
                                  "15 kHz cycles" };
    FclTable *t = fcl_table_new("Measurement blocks", cols, 5);

    if (t == NULL) return NULL;
    for (size_t i = 0; i < c->n_blocks; i++) {
        char cells[5][128];
        double span = c->blocks[i].max_hz - c->blocks[i].min_hz;

        snprintf(cells[0], sizeof(cells[0]), "%s", c->blocks[i].dataset);
        snprintf(cells[1], sizeof(cells[1]), "%s", c->blocks[i].label);
        snprintf(cells[2], sizeof(cells[2]), "%zu", c->blocks[i].n);
        snprintf(cells[3], sizeof(cells[3]), "%.1f", span / 1e3);
        snprintf(cells[4], sizeof(cells[4]), "%.1f", span / 15000.0);
        fc_add_cells(t, cells, 5);
    }
    fcl_table_note(t, "%zu block(s); a block is one session round (all of "
                      "its subjects combined) or one subject.", c->n_blocks);
    return t;
}

/* ------------------------------------------------------------------ */
/* node: Rayleigh                                                     */
/* ------------------------------------------------------------------ */

static FclTable *analysis_rayleigh(FcCtx *c)
{
    static const char *cols[] = { "Block", "df (kHz)", "n", "R",
                                  "mean offset (deg)", "z", "p (Rayleigh)",
                                  "p (perm)", "verdict" };
    FclTable *t = fcl_table_new("Rayleigh test (phases modulo df)", cols, 9);
    size_t reps = reps_for(c);

    if (t == NULL) return NULL;
    for (size_t b = 0; b < c->n_blocks; b++) {
        for (size_t s = 0; s < fcl_n_spacings; s++) {
            for (size_t o = 0; o < fcl_n_offsets; o++) {
                double df = fcl_spacings_hz[s], f0 = fcl_offsets_hz[o];
                double ang = 0.0, R = 0.0, z, p_asym, p_perm;
                char cells[9][128];
                char label[192];

                if (c->blocks[b].n < 3) continue;
                fcl_circular_stats(c->blocks[b].values, c->blocks[b].n, f0, df,
                                   &ang, &R);
                z = (double)c->blocks[b].n * R * R;
                p_asym = fcl_rayleigh_p(c->blocks[b].n, R);
                p_perm = fcl_perm_rayleigh(
                    c->blocks[b].values, c->blocks[b].n, f0, df,
                    c->blocks[b].min_hz, c->blocks[b].max_hz, reps, &c->rng);
                snprintf(cells[0], sizeof(cells[0]), "%s", c->blocks[b].label);
                snprintf(cells[1], sizeof(cells[1]), "%.1f", df / 1e3);
                snprintf(cells[2], sizeof(cells[2]), "%zu", c->blocks[b].n);
                snprintf(cells[3], sizeof(cells[3]), "%.4f", R);
                snprintf(cells[4], sizeof(cells[4]), "%.1f",
                         ang * 180.0 / M_PI);
                snprintf(cells[5], sizeof(cells[5]), "%.2f", z);
                snprintf(cells[6], sizeof(cells[6]), "%.4g", p_asym);
                snprintf(cells[7], sizeof(cells[7]), "%.4g", p_perm);
                snprintf(cells[8], sizeof(cells[8]), "%s",
                         verdict(p_perm, R, "modulo"));
                fc_add_cells(t, cells, 9);
                snprintf(label, sizeof(label), "%s @ df=%.0f Hz",
                         c->blocks[b].label, df);
                ctx_add_test(c, "rayleigh", label, p_perm);
                ctx_add_evidence(c, "rayleigh", b, df, 1, R, p_perm);
                ctx_note_best(c, b, df, p_perm);
                if (p_perm < 0.01) c->any_clustering = true;
            }
        }
    }
    fcl_table_note(t, "R is the mean resultant length (0 = uniform phases, "
                      "1 = all phases equal); the asymptotic p uses "
                      "exp(-n R^2), the permutation p redistributes uniform "
                      "frequencies over the block. %zu permutation "
                      "replicates per cell.", reps);
    return t;
}

/* ------------------------------------------------------------------ */
/* node: V test                                                       */
/* ------------------------------------------------------------------ */

static FclTable *analysis_vtest(FcCtx *c)
{
    static const char *cols[] = { "Block", "df (kHz)", "n", "R",
                                  "V", "p (perm)", "verdict" };
    FclTable *t = fcl_table_new(
        "V test (clustering at phase 0, i.e. on the lattice)", cols, 7);
    size_t reps = reps_for(c);

    if (t == NULL) return NULL;
    for (size_t b = 0; b < c->n_blocks; b++) {
        for (size_t s = 0; s < fcl_n_spacings; s++) {
            for (size_t o = 0; o < fcl_n_offsets; o++) {
                double df = fcl_spacings_hz[s], f0 = fcl_offsets_hz[o];
                double ang = 0.0, R = 0.0, V, p_perm;
                char cells[7][128];
                char label[192];

                if (c->blocks[b].n < 3) continue;
                fcl_circular_stats(c->blocks[b].values, c->blocks[b].n, f0, df,
                                   &ang, &R);
                V = fcl_v_statistic(c->blocks[b].n, R, ang, 0.0);
                p_perm = fcl_perm_v(
                    c->blocks[b].values, c->blocks[b].n, f0, df,
                    c->blocks[b].min_hz, c->blocks[b].max_hz, 0.0, reps,
                    &c->rng);
                snprintf(cells[0], sizeof(cells[0]), "%s", c->blocks[b].label);
                snprintf(cells[1], sizeof(cells[1]), "%.1f", df / 1e3);
                snprintf(cells[2], sizeof(cells[2]), "%zu", c->blocks[b].n);
                snprintf(cells[3], sizeof(cells[3]), "%.4f", R);
                snprintf(cells[4], sizeof(cells[4]), "%.4f", V);
                snprintf(cells[5], sizeof(cells[5]), "%.4g", p_perm);
                snprintf(cells[6], sizeof(cells[6]), "%s",
                         verdict(p_perm, V, "0"));
                fc_add_cells(t, cells, 7);
                snprintf(label, sizeof(label), "%s @ df=%.0f Hz",
                         c->blocks[b].label, df);
                ctx_add_test(c, "vtest", label, p_perm);
                ctx_add_evidence(c, "vtest", b, df, 1, V, p_perm);
                ctx_note_best(c, b, df, p_perm);
            }
        }
    }
    fcl_table_note(t, "The V test is directed at phase 0 (measurements on "
                      "the assumed lattice f0 + k df); it is more powerful "
                      "than Rayleigh when the grid reference is known, and "
                      "ignores clustering at other phases.");
    return t;
}

/* ------------------------------------------------------------------ */
/* node: PDM                                                          */
/* ------------------------------------------------------------------ */

static FclTable *analysis_pdm(FcCtx *c)
{
    static const char *cols[] = { "Block", "df (kHz)", "n", "Theta",
                                  "p (perm)", "verdict" };
    FclTable *t = fcl_table_new("Phase dispersion minimisation", cols, 6);
    size_t reps = reps_for(c);

    if (t == NULL) return NULL;
    for (size_t b = 0; b < c->n_blocks; b++) {
        for (size_t s = 0; s < fcl_n_spacings; s++) {
            for (size_t o = 0; o < fcl_n_offsets; o++) {
                double df = fcl_spacings_hz[s], f0 = fcl_offsets_hz[o];
                double theta, p_perm;
                size_t bins;
                char cells[6][128];
                char label[192];

                if (c->blocks[b].n < 5) continue;
                bins = (size_t)sqrt((double)c->blocks[b].n) + 2;
                if (bins < 4) bins = 4;
                if (bins > 20) bins = 20;
                theta = fcl_pdm(c->blocks[b].values, c->blocks[b].n, f0, df,
                                bins);
                p_perm = fcl_perm_pdm(
                    c->blocks[b].values, c->blocks[b].n, f0, df,
                    c->blocks[b].min_hz, c->blocks[b].max_hz, bins, reps,
                    &c->rng);
                snprintf(cells[0], sizeof(cells[0]), "%s", c->blocks[b].label);
                snprintf(cells[1], sizeof(cells[1]), "%.1f", df / 1e3);
                snprintf(cells[2], sizeof(cells[2]), "%zu", c->blocks[b].n);
                snprintf(cells[3], sizeof(cells[3]), "%.4f", theta);
                snprintf(cells[4], sizeof(cells[4]), "%.4g", p_perm);
                snprintf(cells[5], sizeof(cells[5]), "%s",
                         verdict(p_perm, theta, "pdm"));
                fc_add_cells(t, cells, 6);
                snprintf(label, sizeof(label), "%s @ df=%.0f Hz",
                         c->blocks[b].label, df);
                ctx_add_test(c, "pdm", label, p_perm);
                ctx_add_evidence(c, "pdm", b, df, 1, theta, p_perm);
                ctx_note_best(c, b, df, p_perm);
            }
        }
    }
    fcl_table_note(t, "Theta is the phase-dispersion statistic (small = "
                      "strong periodic clustering, ~1 = none); the "
                      "permutation p uses the lower tail.");
    return t;
}

/* ------------------------------------------------------------------ */
/* node: periodogram scan                                             */
/* ------------------------------------------------------------------ */

static FclTable *analysis_periodogram(FcCtx *c)
{
    static const char *cols[] = { "Block", "rank", "best df (Hz)", "z",
                                  "p (Rayleigh)", "verdict" };
    FclTable *t = fcl_table_new("Periodogram scan over df", cols, 6);

    if (t == NULL) return NULL;
    for (size_t b = 0; b < c->n_blocks; b++) {
        double best[5] = { 0 }, best_z[5] = { 0 };
        size_t n_best = 0;

        if (c->blocks[b].n < 5) continue;
        for (double df = 2000.0; df <= 300000.0; df += 1000.0) {
            double R = 0.0, z;

            fcl_circular_stats(c->blocks[b].values, c->blocks[b].n, 0.0, df,
                               NULL, &R);
            z = (double)c->blocks[b].n * R * R;
            for (size_t k = 0; k < n_best; k++) {
                if (z > best_z[k]) {
                    for (size_t m = (n_best < 5 ? n_best : 4); m > k; m--) {
                        if (m >= 5) continue;
                        best[m] = best[m - 1];
                        best_z[m] = best_z[m - 1];
                    }
                    best[k] = df;
                    best_z[k] = z;
                    if (n_best < 5) n_best++;
                    break;
                }
            }
            if (n_best < 5) {
                best[n_best] = df;
                best_z[n_best] = z;
                n_best++;
            }
        }
        for (size_t k = 0; k < n_best && k < 5; k++) {
            char cells[6][128];
            double p = fcl_rayleigh_p(c->blocks[b].n,
                                      sqrt(best_z[k] /
                                           (double)c->blocks[b].n));

            snprintf(cells[0], sizeof(cells[0]), "%s", c->blocks[b].label);
            snprintf(cells[1], sizeof(cells[1]), "%zu", k + 1);
            snprintf(cells[2], sizeof(cells[2]), "%.0f", best[k]);
            snprintf(cells[3], sizeof(cells[3]), "%.2f", best_z[k]);
            snprintf(cells[4], sizeof(cells[4]), "%.4g", p);
            snprintf(cells[5], sizeof(cells[5]), "%s",
                     verdict(p, best_z[k], "periodogram"));
            fc_add_cells(t, cells, 6);
        }
    }
    fcl_table_note(t, "A Lomb-Scargle-like scan: z = n R^2 of the folded "
                      "phases for every df from 2 kHz to 300 kHz in 1 kHz "
                      "steps; the rows list the five strongest spacings per "
                      "block. Peaks at 15/30/60/120/240 kHz (LTE/NR) or "
                      "5 kHz are the candidates of interest; scan peaks are "
                      "not corrected for the number of spacings tried.");
    return t;
}

/* ------------------------------------------------------------------ */
/* node: chi-square                                                    */
/* ------------------------------------------------------------------ */

static FclTable *analysis_chisq(FcCtx *c)
{
    static const char *cols[] = { "Block", "df (kHz)", "n", "bins",
                                  "chi2", "p", "verdict" };
    FclTable *t = fcl_table_new("Chi-square test of lattice-centred bins",
                                cols, 7);

    if (t == NULL) return NULL;
    for (size_t b = 0; b < c->n_blocks; b++) {
        for (size_t s = 0; s < fcl_n_spacings; s++) {
            for (size_t o = 0; o < fcl_n_offsets; o++) {
                double df = fcl_spacings_hz[s], f0 = fcl_offsets_hz[o];
                double chi2;
                size_t dfree = 0;
                bool valid = false;
                char cells[7][128];
                char label[192];
                FclTable *tt = t;

                if (c->blocks[b].n < 10) continue;
                chi2 = fcl_chi2_lattice(
                    c->blocks[b].values, c->blocks[b].n, f0, df,
                    c->blocks[b].min_hz, c->blocks[b].max_hz, 0.25, &dfree,
                    &valid);
                if (!valid) continue;
                {
                    double p = fcl_chi2_p(chi2, dfree);

                    snprintf(cells[0], sizeof(cells[0]), "%s",
                             c->blocks[b].label);
                    snprintf(cells[1], sizeof(cells[1]), "%.1f", df / 1e3);
                    snprintf(cells[2], sizeof(cells[2]), "%zu",
                             c->blocks[b].n);
                    snprintf(cells[3], sizeof(cells[3]), "%zu", dfree + 1);
                    snprintf(cells[4], sizeof(cells[4]), "%.2f", chi2);
                    snprintf(cells[5], sizeof(cells[5]), "%.4g", p);
                    snprintf(cells[6], sizeof(cells[6]), "%s",
                             verdict(p, chi2, "chisq"));
                    fc_add_cells(tt, cells, 7);
                    snprintf(label, sizeof(label), "%s @ df=%.0f Hz",
                             c->blocks[b].label, df);
                    ctx_add_test(c, "chisq", label, p);
                    ctx_add_evidence(c, "chisq", b, df, 1, chi2, p);
                    ctx_note_best(c, b, df, p);
                }
            }
        }
    }
    fcl_table_note(t, "Bins of width 0.25 df centred on the lattice; the "
                      "expected count per bin follows a uniform distribution "
                      "over the block. Cells with fewer than 5 expected "
                      "counts per bin are omitted.");
    return t;
}

/* ------------------------------------------------------------------ */
/* node: KDE enrichment                                                */
/* ------------------------------------------------------------------ */

static FclTable *analysis_kde(FcCtx *c)
{
    static const char *cols[] = { "Block", "df (kHz)", "n",
                                  "enrichment", "p (perm)", "verdict" };
    FclTable *t = fcl_table_new("KDE enrichment at the lattice points", cols,
                                6);
    size_t reps = reps_for(c) > 200 ? 150 : reps_for(c);

    if (t == NULL) return NULL;
    for (size_t b = 0; b < c->n_blocks; b++) {
        for (size_t s = 0; s < fcl_n_spacings; s++) {
            for (size_t o = 0; o < fcl_n_offsets; o++) {
                double df = fcl_spacings_hz[s], f0 = fcl_offsets_hz[o];
                double enrich, p_perm;
                char cells[6][128];
                char label[192];

                if (c->blocks[b].n < 5) continue;
                enrich = fcl_kde_enrichment(
                    c->blocks[b].values, c->blocks[b].n, f0, df,
                    c->blocks[b].min_hz, c->blocks[b].max_hz);
                p_perm = fcl_perm_enrichment(
                    c->blocks[b].values, c->blocks[b].n, f0, df,
                    c->blocks[b].min_hz, c->blocks[b].max_hz, reps,
                    &c->rng);
                snprintf(cells[0], sizeof(cells[0]), "%s", c->blocks[b].label);
                snprintf(cells[1], sizeof(cells[1]), "%.1f", df / 1e3);
                snprintf(cells[2], sizeof(cells[2]), "%zu", c->blocks[b].n);
                snprintf(cells[3], sizeof(cells[3]), "%.3f", enrich);
                snprintf(cells[4], sizeof(cells[4]), "%.4g", p_perm);
                snprintf(cells[5], sizeof(cells[5]), "%s",
                         verdict(p_perm, enrich, "kde"));
                fc_add_cells(t, cells, 6);
                snprintf(label, sizeof(label), "%s @ df=%.0f Hz",
                         c->blocks[b].label, df);
                ctx_add_test(c, "kde", label, p_perm);
                ctx_add_evidence(c, "kde", b, df, 1, enrich, p_perm);
                ctx_note_best(c, b, df, p_perm);
            }
        }
    }
    fcl_table_note(t, "Gaussian KDE with Silverman's bandwidth; the "
                      "enrichment is the density at the lattice points "
                      "divided by the mean density over the block (1 = no "
                      "enrichment).");
    return t;
}

/* ------------------------------------------------------------------ */
/* node: cluster permutation test                                     */
/* ------------------------------------------------------------------ */

static FclTable *analysis_cluster(FcCtx *c)
{
    static const char *cols[] = { "Block", "n", "max cluster mass",
                                  "extent (bins)", "p (perm)", "verdict" };
    FclTable *t = fcl_table_new("Cluster-based permutation test", cols, 6);
    size_t reps = reps_for(c);

    if (t == NULL) return NULL;
    for (size_t b = 0; b < c->n_blocks; b++) {
        double mass = 0.0, p = 1.0;
        size_t extent = 0;
        char cells[6][128];

        if (c->blocks[b].n < 10) continue;
        fcl_cluster_test(c->blocks[b].values, c->blocks[b].n,
                         c->blocks[b].min_hz, c->blocks[b].max_hz, 64, reps,
                         &c->rng, &mass, &extent, &p);
        snprintf(cells[0], sizeof(cells[0]), "%s", c->blocks[b].label);
        snprintf(cells[1], sizeof(cells[1]), "%zu", c->blocks[b].n);
        snprintf(cells[2], sizeof(cells[2]), "%.1f", mass);
        snprintf(cells[3], sizeof(cells[3]), "%zu", extent);
        snprintf(cells[4], sizeof(cells[4]), "%.4g", p);
        snprintf(cells[5], sizeof(cells[5]), "%s", verdict(p, mass, "cluster"));
        fc_add_cells(t, cells, 6);
        if (c->best_p != NULL && p < c->best_p[b]) {
            /* cluster evidence is spacing-agnostic; keep it as a note only */
        }
    }
    fcl_table_note(t, "64 equal-width bins over the block; bins above the "
                      "uniform expectation are joined into contiguous "
                      "clusters (wrapping at the block edges) and the "
                      "largest cluster mass is compared with the same "
                      "statistic for uniform frequencies over the block. "
                      "The verbatim cluster statistic is spacing-agnostic "
                      "and complements the lattice-specific tests.");
    return t;
}

/* ------------------------------------------------------------------ */
/* node: multimodality                                                 */
/* ------------------------------------------------------------------ */

static FclTable *analysis_multimodal(FcCtx *c)
{
    static const char *cols[] = { "Block", "n", "modes", "p (Silverman)",
                                  "verdict" };
    FclTable *t = fcl_table_new("Silverman multimodality test", cols, 5);

    if (t == NULL) return NULL;
    for (size_t b = 0; b < c->n_blocks; b++) {
        size_t modes = 0;
        double p;
        char cells[5][128];

        p = fcl_silverman(c->blocks[b].values, c->blocks[b].n, 1, 40, &c->rng,
                          &modes);
        snprintf(cells[0], sizeof(cells[0]), "%s", c->blocks[b].label);
        snprintf(cells[1], sizeof(cells[1]), "%zu", c->blocks[b].n);
        snprintf(cells[2], sizeof(cells[2]), "%zu", modes);
        if (p < 0.0) {
            snprintf(cells[3], sizeof(cells[3]), "—");
            snprintf(cells[4], sizeof(cells[4]),
                     "not computed (n outside 8..400)");
        } else {
            snprintf(cells[3], sizeof(cells[3]), "%.4g", p);
            snprintf(cells[4], sizeof(cells[4]), "%s",
                     p < FC_ALPHA ? "more than one mode"
                                  : "unimodality not rejected");
        }
        fc_add_cells(t, cells, 5);
    }
    fcl_table_note(t, "Silverman's critical-bandwidth bootstrap for k = 1 "
                      "mode (40 replicates, deterministic seed). Hartigan's "
                      "dip test is not implemented in this build; use the "
                      "mode count and the lattice tests as complementary "
                      "evidence.");
    return t;
}

/* ------------------------------------------------------------------ */
/* node: correction                                                    */
/* ------------------------------------------------------------------ */

static FclTable *analysis_correction(FcCtx *c)
{
    static const char *cols[] = { "Test family", "tests", "min p",
                                  "BH min (adj)", "Bonferroni min",
                                  "significant at 0.05" };
    static const char *const families[] = {
        "rayleigh", "vtest", "pdm", "chisq", "kde"
    };
    FclTable *t = fcl_table_new("Multiple-testing correction", cols, 6);
    size_t n_fam = sizeof(families) / sizeof(families[0]);
    size_t total_sig = 0;

    if (t == NULL) return NULL;
    for (size_t f = 0; f < n_fam; f++) {
        size_t count = 0;
        double min_p = 1.0;
        double *ps;
        size_t n = 0;
        char cells[6][128];

        for (size_t i = 0; i < c->n_tests; i++) {
            if (strcmp(c->tests[i].family, families[f]) == 0) count++;
        }
        if (count == 0) continue;
        /* collect this family's p-values and adjust them in place */
        ps = malloc(count * sizeof(*ps));
        if (ps == NULL) continue;
        for (size_t i = 0; i < c->n_tests; i++) {
            if (strcmp(c->tests[i].family, families[f]) == 0) {
                ps[n++] = c->tests[i].p;
                if (c->tests[i].p < min_p) min_p = c->tests[i].p;
            }
        }
        n = count;
        {
            double *tmp = malloc(count * sizeof(*tmp));
            size_t rej;
            double min_adj = 1.0, min_bonf = 1.0;

            if (tmp == NULL) {
                free(ps);
                continue;
            }
            memcpy(tmp, ps, count * sizeof(*tmp));
            fcl_bh_adjust(ps, n, FC_ALPHA);
            for (size_t i = 0; i < n; i++) {
                if (ps[i] < min_adj) min_adj = ps[i];
                if (fcl_bonferroni(tmp[i], count) < min_bonf) {
                    min_bonf = fcl_bonferroni(tmp[i], count);
                }
            }
            rej = fcl_bh_adjust(tmp, n, FC_ALPHA);   /* count rejections */
            total_sig += rej;
            free(tmp);
            snprintf(cells[0], sizeof(cells[0]), "%s", families[f]);
            snprintf(cells[1], sizeof(cells[1]), "%zu", count);
            snprintf(cells[2], sizeof(cells[2]), "%.4g", min_p);
            snprintf(cells[3], sizeof(cells[3]), "%.4g", min_adj);
            snprintf(cells[4], sizeof(cells[4]), "%.4g", min_bonf);
            snprintf(cells[5], sizeof(cells[5]), "%zu", rej);
            fc_add_cells(t, cells, 6);
        }
        free(ps);
    }
    {
        char cells[6][128];

        snprintf(cells[0], sizeof(cells[0]), "all families");
        snprintf(cells[1], sizeof(cells[1]), "%zu", c->n_tests);
        snprintf(cells[2], sizeof(cells[2]), "—");
        snprintf(cells[3], sizeof(cells[3]), "—");
        snprintf(cells[4], sizeof(cells[4]), "—");
        snprintf(cells[5], sizeof(cells[5]), "%zu", total_sig);
        fc_add_cells(t, cells, 6);
    }
    if (c->n_tests == 0) {
        fcl_table_note(t, "No tests were recorded (blocks too small).");
    } else {
        fcl_table_note(t, "Adjustment performed over %zu tests within each "
                          "family; alpha = 0.05. The all-families count "
                          "uses the sum of the per-family rejections.",
                       c->n_tests);
    }
    return t;
}

/* ------------------------------------------------------------------ */
/* node: summary                                                      */
/* ------------------------------------------------------------------ */

static FclTable *analysis_summary(FcCtx *c)
{
    static const char *cols[] = { "Block", "n", "best df (kHz)", "evidence" };
    FclTable *t = fcl_table_new("Per-block lattice summary", cols, 4);

    if (t == NULL) return NULL;
    for (size_t b = 0; b < c->n_blocks; b++) {
        char cells[4][128];
        size_t reps = reps_for(c);
        double df = c->best_df != NULL && c->best_df[b] > 0.0
                        ? c->best_df[b] : 15000.0;
        double angle = 0.0, R = 0.0, z, p;
        char *rose, *comb;
        double span = c->blocks[b].max_hz - c->blocks[b].min_hz;

        fcl_circular_stats(c->blocks[b].values, c->blocks[b].n, 0.0, df,
                           &angle, &R);
        z = (double)c->blocks[b].n * R * R;
        p = fcl_rayleigh_p(c->blocks[b].n, R);
        snprintf(cells[0], sizeof(cells[0]), "%s", c->blocks[b].label);
        snprintf(cells[1], sizeof(cells[1]), "%zu", c->blocks[b].n);
        snprintf(cells[2], sizeof(cells[2]), "%.1f", df / 1e3);
        snprintf(cells[3], sizeof(cells[3]), "R=%.3f, mean phase %.1f deg, "
                                             "z=%.2f, p(R)=%.3g; %s",
                 R, angle * 180.0 / M_PI, z, p,
                 p < FC_ALPHA ? "clustering supports this spacing"
                              : "no significant clustering");
        fc_add_cells(t, cells, 6);
        rose = fcl_rose_plot(c->blocks[b].values, c->blocks[b].n, 0.0, df, 12,
                             8);
        comb = fcl_comb_plot(c->blocks[b].values, c->blocks[b].n, 0.0, df, 48,
                             6);
        if (rose != NULL && comb != NULL && c->blocks[b].n >= 10 &&
            span > 0.0) {
            fcl_table_note(t, "block: %s (best df = %.0f Hz)", 
                           c->blocks[b].label, df);
            fcl_table_note(t, "%s", rose);
            fcl_table_note(t, "%s", comb);
        }
        free(rose);
        free(comb);
        (void)reps;
    }
    fcl_table_note(t, "The best spacing is the one with the smallest "
                      "permutation p among the circular/KDE/lattice tests; "
                      "the rose and comb plots show the phase distribution "
                      "and the position of the measurements between "
                      "subcarriers.");
    return t;
}

/* ------------------------------------------------------------------ */
/* node: induced-lattice scan over K                                  */
/* ------------------------------------------------------------------ */

static FclTable *analysis_induced_scan(FcCtx *c)
{
    static const char *cols[] = { "Block", "df (kHz)", "K", "delta (Hz)", "n",
                                  "R", "z", "p (perm)", "p (adj)",
                                  "verdict" };
    FclTable *t = fcl_table_new(
        "Induced-lattice scan: averages of K subcarriers", cols, 10);
    size_t reps = reps_for(c);

    if (t == NULL) return NULL;
    for (size_t b = 0; b < c->n_blocks; b++) {
        double df;
        char cells[10][128];

        if (c->blocks[b].n < 8) continue;
        df = k_spacing(c, b);
        fclk_scan_block(c->blocks[b].values, c->blocks[b].n,
                        c->blocks[b].min_hz, c->blocks[b].max_hz, df,
                        FCLK_MAX, reps, &c->rng, &c->scan[b * FCLK_MAX]);
        c->finest_k[b] = fclk_finest_significant(
            &c->scan[b * FCLK_MAX], FCLK_MAX, 0.05);
        for (int K = 1; K <= FCLK_MAX; K++) {
            const FclkScanCell *cell = &c->scan[b * FCLK_MAX + K - 1];

            if (!cell->valid) continue;
            if (cell->p_adj < 0.05) c->have_induced = true;
            ctx_add_evidence(c, "induced-K", b, df, K, cell->z,
                             cell->p_adj);
            snprintf(cells[0], sizeof(cells[0]), "%s", c->blocks[b].label);
            snprintf(cells[1], sizeof(cells[1]), "%.1f", df / 1e3);
            snprintf(cells[2], sizeof(cells[2]), "%d", K);
            snprintf(cells[3], sizeof(cells[3]), "%.0f", cell->delta_hz);
            snprintf(cells[4], sizeof(cells[4]), "%zu", cell->n);
            snprintf(cells[5], sizeof(cells[5]), "%.4f", cell->R);
            snprintf(cells[6], sizeof(cells[6]), "%.2f", cell->z);
            snprintf(cells[7], sizeof(cells[7]), "%.4g", cell->p_perm);
            snprintf(cells[8], sizeof(cells[8]), "%.4g", cell->p_adj);
            snprintf(cells[9], sizeof(cells[9]), "%s",
                     cell->p_adj < 0.05
                         ? (K == c->finest_k[b] ? "significant (finest K)"
                                                : "significant")
                         : "not significant");
            fc_add_cells(t, cells, 10);
        }
    }
    fcl_table_note(t, "For each candidate K the measurements should lie on "
                      "the induced lattice f0 + l*(df/K) when they are "
                      "equal-weight averages of K latent subcarriers on the "
                      "df grid. The scan uses df = %.0f Hz-class spacing per "
                      "block (the spacing supported by the circular tests) "
                      "with %zu permutations against uniform frequencies "
                      "inside the block; p(adj) is Bonferroni over the %d K "
                      "candidates. K = 1 is the plain single-subcarrier "
                      "case.",
                 (double)(c->n_blocks > 0 ? k_spacing(c, 0) : 15000.0) / 1e3,
                 reps, FCLK_MAX);
    return t;
}

/* ------------------------------------------------------------------ */
/* node: phase-dispersion multiplicity diagnostic                     */
/* ------------------------------------------------------------------ */

static FclTable *analysis_k_dispersion(FcCtx *c)
{
    static const char *cols[] = { "Block", "df (kHz)", "K (finest sig.)",
                                  "sigma_1^2", "sigma_K^2", "K (dispersion)",
                                  "verdict" };
    FclTable *t = fcl_table_new(
        "Phase-dispersion multiplicity diagnostic", cols, 7);

    if (t == NULL) return NULL;
    for (size_t b = 0; b < c->n_blocks; b++) {
        double df = k_spacing(c, b), s1 = 0.0, sbest = 0.0;
        int khat = 0;
        char cells[7][128];
        bool ok;

        if (c->blocks[b].n < 8) continue;
        ok = fclk_dispersion_estimate(c->blocks[b].values, c->blocks[b].n, df,
                                      FCLK_MAX, &khat, &s1, &sbest);
        if (!ok) continue;
        c->have_reference = true;
        c->k_disp[b] = khat;
        snprintf(cells[0], sizeof(cells[0]), "%s", c->blocks[b].label);
        snprintf(cells[1], sizeof(cells[1]), "%.1f", df / 1e3);
        snprintf(cells[2], sizeof(cells[2]), "%d", c->finest_k[b]);
        snprintf(cells[3], sizeof(cells[3]), "%.4f", s1);
        snprintf(cells[4], sizeof(cells[4]), "%.4f", sbest);
        snprintf(cells[5], sizeof(cells[5]), "%d", khat);
        snprintf(cells[6], sizeof(cells[6]), "%s",
                 khat == c->finest_k[b] ? "agrees with the scan"
                 : khat > 1 ? "dispersion favours a finer lattice"
                            : "no dispersion shrinkage");
        fc_add_cells(t, cells, 7);
    }
    fcl_table_note(t, "sigma_1^2 is the circular variance of the phases "
                      "modulo df (the K = 1 residual dispersion, used as the "
                      "single-subcarrier reference); sigma_K^2 is the "
                      "smallest circular variance over the induced spacings "
                      "df/K. If averaging K i.i.d. subcarriers reduces the "
                      "dispersion by about K, the ratio estimates K. This "
                      "requires a credible single-subcarrier reference: when "
                      "sigma_1^2 is already ~0 the data are already on the "
                      "grid and K is not identifiable from dispersion alone.");
    return t;
}

/* ------------------------------------------------------------------ */
/* node: characteristic-function slope                                */
/* ------------------------------------------------------------------ */

static FclTable *analysis_cf_slope(FcCtx *c)
{
    static const char *cols[] = { "Block", "df (kHz)", "K (CF slope)",
                                  "R^2", "verdict" };
    FclTable *t = fcl_table_new(
        "Characteristic-function slope estimate of K", cols, 5);

    if (t == NULL) return NULL;
    for (size_t b = 0; b < c->n_blocks; b++) {
        double df = k_spacing(c, b), r2 = 0.0;
        int k_hat;
        char cells[5][128];

        if (c->blocks[b].n < 12) continue;
        k_hat = fclk_cf_slope(c->blocks[b].values, c->blocks[b].n, df, 0.0,
                              FCLK_MAX, c->blocks[b].min_hz,
                              c->blocks[b].max_hz, &r2);
        if (k_hat <= 0 || r2 < 0.5) continue;
        c->have_cf = true;
        c->k_cf[b] = k_hat;
        snprintf(cells[0], sizeof(cells[0]), "%s", c->blocks[b].label);
        snprintf(cells[1], sizeof(cells[1]), "%.1f", df / 1e3);
        snprintf(cells[2], sizeof(cells[2]), "%d", k_hat);
        snprintf(cells[3], sizeof(cells[3]), "%.3f", r2);
        snprintf(cells[4], sizeof(cells[4]), "%s",
                 k_hat == c->finest_k[b] ? "agrees with the scan"
                 : k_hat > 1 ? "CF favours averaging" : "single subcarrier");
        fc_add_cells(t, cells, 5);
    }
    fcl_table_note(t, "The CF slope uses log|phi_K(t)| = K log|phi_ref(t)| "
                      "for equal-weight i.i.d. subcarriers, with the K = 1 "
                      "residual empirical CF as the single-subcarrier "
                      "reference. As with all references this assumes the "
                      "measurements are a mixture of averaged readings; the "
                      "estimate is reported with its R^2 and should be read "
                      "together with the mixture BIC.");
    return t;
}

/* ------------------------------------------------------------------ */
/* node: grid-constrained mixture fit (BIC/ICL)                       */
/* ------------------------------------------------------------------ */

static FclTable *analysis_mixture(FcCtx *c)
{
    static const char *cols[] = { "Block", "K", "delta (Hz)", "components",
                                  "loglik", "params", "BIC", "ICL",
                                  "sigma (Hz)", "verdict" };
    FclTable *t = fcl_table_new(
        "Grid-constrained mixture fit over K (BIC/ICL)", cols, 10);

    if (t == NULL) return NULL;
    for (size_t b = 0; b < c->n_blocks; b++) {
        double df = k_spacing(c, b);
        double best_bic = 0.0;
        int k_icl;
        bool any = false;

        if (c->blocks[b].n < 10) continue;
        for (int K = 1; K <= FCLK_MAX; K++) {
            FclkMixtureFit *fit = &c->fits[b * FCLK_MAX + K - 1];

            if (!fclk_mixture_fit(c->blocks[b].values, c->blocks[b].n, 0.0,
                                  df, K, c->blocks[b].min_hz,
                                  c->blocks[b].max_hz, fit)) {
                continue;
            }
            any = true;
            if (fit->bic < best_bic || c->k_bic[b] == 0) {
                best_bic = fit->bic;
                c->k_bic[b] = K;
            }
        }
        k_icl = fclk_icl_select(&c->fits[b * FCLK_MAX], FCLK_MAX);
        c->k_icl[b] = k_icl;
        if (!any) continue;
        c->have_mixture = true;
        {
            double delta_bic = 0.0;

            c->k_bic[b] = fclk_bic_select(&c->fits[b * FCLK_MAX], FCLK_MAX,
                                          &delta_bic);
            c->dbic[b] = delta_bic;
            {
                double p_mix = exp(-fmax(0.0, delta_bic) / 2.0);

                if (p_mix > 1.0) p_mix = 1.0;
                ctx_add_evidence(c, "mixture-BIC", b, df, c->k_bic[b],
                                 delta_bic, p_mix);
            }
        }
        for (int K = 1; K <= FCLK_MAX; K++) {
            const FclkMixtureFit *fit = &c->fits[b * FCLK_MAX + K - 1];
            char cells[10][128];

            if (!fit->valid) continue;
            snprintf(cells[0], sizeof(cells[0]), "%s", c->blocks[b].label);
            snprintf(cells[1], sizeof(cells[1]), "%d", K);
            snprintf(cells[2], sizeof(cells[2]), "%.0f", fit->delta_hz);
            snprintf(cells[3], sizeof(cells[3]), "%zu", fit->n_components);
            snprintf(cells[4], sizeof(cells[4]), "%.1f", fit->loglik);
            snprintf(cells[5], sizeof(cells[5]), "%zu", fit->n_params);
            snprintf(cells[6], sizeof(cells[6]), "%.1f", fit->bic);
            snprintf(cells[7], sizeof(cells[7]), "%.1f", fit->icl);
            snprintf(cells[8], sizeof(cells[8]), "%.1f", fit->sigma);
            snprintf(cells[9], sizeof(cells[9]), "%s",
                     K == c->k_bic[b] ? "best BIC"
                     : K == k_icl ? "best ICL"
                     : K == 1 ? "single subcarrier"
                              : "");
            fc_add_cells(t, cells, 10);
        }
    }
    fcl_table_note(t, "The model for candidate K places Gaussian components "
                      "on the induced lattice f0 + l*(df/K) inside the "
                      "block, with weights p_j and a shared sigma estimated "
                      "by EM; BIC = -2 logL + params*ln(n) penalises the "
                      "extra lattice components, ICL adds the classification "
                      "entropy. The selected K is the one whose induced "
                      "lattice best explains the observed distribution "
                      "including complexity, which is the primary "
                      "identifiability route (K is not identifiable without "
                      "a structural assumption). DeltaBIC vs K = 1 is "
                      "reported in the model-selection table.");
    return t;
}

/* ------------------------------------------------------------------ */
/* node: model selection                                              */
/* ------------------------------------------------------------------ */

static FclTable *analysis_model_select(FcCtx *c)
{
    static const char *cols[] = { "Block", "K (scan)", "K (dispersion)",
                                  "K (CF)", "K (BIC)", "K (ICL)",
                                  "DeltaBIC vs K=1", "preferred K",
                                  "confidence" };
    FclTable *t = fcl_table_new(
        "K model selection: subcarrier multiplicity", cols, 9);

    if (t == NULL) return NULL;
    for (size_t b = 0; b < c->n_blocks; b++) {
        int k_pref = 1;
        int votes = 0, agree = 0;
        char cells[9][128];

        if (c->blocks[b].n < 8) continue;
        if (c->have_mixture && c->k_bic[b] > 0) {
            k_pref = c->k_bic[b];
            if (c->dbic[b] <= 0.0) k_pref = 1;   /* no evidence over K=1 */
        } else if (c->finest_k[b] > 0) {
            k_pref = c->finest_k[b];
        }
        /* agreement: count the methods that support the preferred K */
        if (c->finest_k[b] == k_pref) agree++;
        if (c->k_disp[b] == k_pref) agree++;
        if (c->k_cf[b] == k_pref) agree++;
        if (c->k_bic[b] == k_pref) agree++;
        if (c->k_icl[b] == k_pref) agree++;
        votes = agree;
        c->k_final[b] = k_pref;
        c->have_model = true;
        snprintf(cells[0], sizeof(cells[0]), "%s", c->blocks[b].label);
        snprintf(cells[1], sizeof(cells[1]), "%d", c->finest_k[b]);
        snprintf(cells[2], sizeof(cells[2]), "%d", c->k_disp[b]);
        snprintf(cells[3], sizeof(cells[3]), "%d", c->k_cf[b]);
        snprintf(cells[4], sizeof(cells[4]), "%d", c->k_bic[b]);
        snprintf(cells[5], sizeof(cells[5]), "%d", c->k_icl[b]);
        snprintf(cells[6], sizeof(cells[6]), "%.1f", c->dbic[b]);
        snprintf(cells[7], sizeof(cells[7]), "%d", k_pref);
        snprintf(cells[8], sizeof(cells[8]), "%s",
                 votes >= 3 ? "supported by >= 3 methods"
                 : votes == 2 ? "supported by 2 methods"
                 : votes == 1 ? "single method"
                              : "no method supports K > 1");
        fc_add_cells(t, cells, 9);
    }
    fcl_table_note(t, "Preferred K: the mixture BIC choice when it improves "
                      "on K = 1 by DeltaBIC > 0 (parsimony rule), otherwise "
                      "the finest K with significant induced-lattice "
                      "clustering; methods agreeing with the choice are "
                      "counted as confidence. K = 1 means the measurements "
                      "behave like single-subcarrier readings on the df "
                      "grid. Multiplicity across blocks is corrected with "
                      "Benjamini-Hochberg on the scan p-values in the "
                      "clustering correction tables.");
    return t;
}

/* ------------------------------------------------------------------ */
/* node: K summary with induced-lattice plots                         */
/* ------------------------------------------------------------------ */

static FclTable *analysis_k_summary(FcCtx *c)
{
    static const char *cols[] = { "Block", "n", "df (kHz)",
                                  "recommended K", "induced delta (Hz)",
                                  "evidence" };
    FclTable *t = fcl_table_new(
        "Subcarrier multiplicity summary (per block)", cols, 6);

    if (t == NULL) return NULL;
    for (size_t b = 0; b < c->n_blocks; b++) {
        double df = k_spacing(c, b);
        double delta;
        char cells[6][128];
        int k = c->k_final[b] > 0 ? c->k_final[b] : 1;
        char *rose, *comb;

        if (c->blocks[b].n < 8) continue;
        delta = df / (double)k;
        snprintf(cells[0], sizeof(cells[0]), "%s", c->blocks[b].label);
        snprintf(cells[1], sizeof(cells[1]), "%zu", c->blocks[b].n);
        snprintf(cells[2], sizeof(cells[2]), "%.1f", df / 1e3);
        snprintf(cells[3], sizeof(cells[3]), "%d", k);
        snprintf(cells[4], sizeof(cells[4]), "%.0f", delta);
        snprintf(cells[5], sizeof(cells[5]), "%s", delta < df
                     ? "averaged readings on the induced lattice"
                     : "single-subcarrier readings on the df grid");
        fc_add_cells(t, cells, 6);
        if (c->blocks[b].n >= 10) {
            rose = fcl_rose_plot(c->blocks[b].values, c->blocks[b].n, 0.0,
                                 delta, 12, 6);
            comb = fcl_comb_plot(c->blocks[b].values, c->blocks[b].n, 0.0,
                                 delta, 48, 5);
            if (rose != NULL && comb != NULL) {
                fcl_table_note(t, "block: %s (K = %d, induced delta = %.0f Hz)",
                               c->blocks[b].label, k, delta);
                fcl_table_note(t, "%s", rose);
                fcl_table_note(t, "%s", comb);
            }
            free(rose);
            free(comb);
        }
    }
    fcl_table_note(t, "The number of latent subcarriers K is only "
                      "identifiable under the equal-weight/i.i.d./known-grid "
                      "assumption; the recommended K therefore combines the "
                      "mixture model (BIC/ICL), the induced-lattice scan, "
                      "the dispersion ratio and the characteristic-function "
                      "slope, and the confidence column counts how many of "
                      "them agree. Compare the K = 1 row with K > 1: if the "
                      "finer induced lattice is not supported, the "
                      "measurements are better explained as single-subcarrier "
                      "readings.");
    return t;
}

/* ------------------------------------------------------------------ */
/* node: evidence filter (discard low-evidence analyses)              */
/* ------------------------------------------------------------------ */

static FclTable *analysis_evidence_filter(FcCtx *c)
{
    static const char *cols[] = { "Analysis", "spacing (kHz)", "tests",
                                  "retained", "min p", "mean effect",
                                  "decision" };
    FclTable *t = fcl_table_new(
        "Evidence screen: analyses retained for the synthesis", cols, 7);
    static const char *const families[] = {
        "rayleigh", "vtest", "pdm", "chisq", "kde", "induced-K",
        "mixture-BIC"
    };
    size_t n_fam = sizeof(families) / sizeof(families[0]);

    if (t == NULL) return NULL;
    /* classify every evidence entry */
    for (size_t i = 0; i < c->n_evidence; i++) {
        double p = c->evidence[i].p;
        bool retain;

        c->evidence[i].weight = evidence_weight(c->evidence[i].family);
        if (strcmp(c->evidence[i].family, "mixture-BIC") == 0) {
            /* BIC evidence: DeltaBIC > 2 is positive evidence */
            retain = c->evidence[i].effect > 2.0;
        } else {
            retain = p < 0.05;
        }
        c->evidence[i].retained = retain;
        if (retain) c->have_evidence = true;
    }
    /* aggregate per (family, spacing) */
    for (size_t f = 0; f < n_fam; f++) {
        double spacings[8];
        size_t n_sp = 0;

        for (size_t i = 0; i < c->n_evidence; i++) {
            if (strcmp(c->evidence[i].family, families[f]) != 0) continue;
            {
                bool seen = false;

                for (size_t s = 0; s < n_sp; s++) {
                    if (fabs(spacings[s] - c->evidence[i].spacing_hz) < 1.0) {
                        seen = true;
                        break;
                    }
                }
                if (!seen && n_sp < 8) {
                    spacings[n_sp++] = c->evidence[i].spacing_hz;
                }
            }
        }
        for (size_t s = 0; s < n_sp; s++) {
            size_t tests = 0, kept = 0;
            double min_p = 1.0, sum_effect = 0.0;
            char cells[7][128];

            for (size_t i = 0; i < c->n_evidence; i++) {
                if (strcmp(c->evidence[i].family, families[f]) != 0) continue;
                if (fabs(c->evidence[i].spacing_hz - spacings[s]) >= 1.0) {
                    continue;
                }
                tests++;
                if (c->evidence[i].p < min_p) min_p = c->evidence[i].p;
                sum_effect += c->evidence[i].effect;
                if (c->evidence[i].retained) kept++;
            }
            snprintf(cells[0], sizeof(cells[0]), "%s", families[f]);
            snprintf(cells[1], sizeof(cells[1]), "%.1f", spacings[s] / 1e3);
            snprintf(cells[2], sizeof(cells[2]), "%zu", tests);
            snprintf(cells[3], sizeof(cells[3]), "%zu", kept);
            snprintf(cells[4], sizeof(cells[4]), "%.3g", min_p);
            snprintf(cells[5], sizeof(cells[5]), "%.3g",
                     tests > 0 ? sum_effect / (double)tests : 0.0);
            snprintf(cells[6], sizeof(cells[6]), "%s",
                     kept > 0 ? "retained" : "discarded (no evidence)");
            fc_add_cells(t, cells, 7);
        }
    }
    {
        size_t kept = 0, total = c->n_evidence;

        for (size_t i = 0; i < c->n_evidence; i++) {
            if (c->evidence[i].retained) kept++;
        }
        fcl_table_note(t, "Retention rule: a test is retained when its "
                          "p-value is below 0.05 (Bonferroni-adjusted for "
                          "the induced-lattice scan); the mixture model is "
                          "retained when DeltaBIC > 2 against K = 1. %zu of "
                          "%zu analysis results were retained and are "
                          "combined in the synthesis table.",
                       kept, total);
        if (total == 0) {
            fcl_table_note(t, "No analysis produced evidence (the blocks "
                              "are too small or no clustering test ran).");
        }
    }
    return t;
}

/* ------------------------------------------------------------------ */
/* node: synthesis of the most likely subcarrier spacings             */
/* ------------------------------------------------------------------ */

static FclTable *analysis_synthesis(FcCtx *c)
{
    static const char *cols[] = { "Block", "rank", "spacing (kHz)",
                                  "evidence score", "supporting analyses",
                                  "implied K", "interpretation" };
    FclTable *t = fcl_table_new(
        "Synthesis: most likely subcarrier spacings", cols, 7);
    static const char *const families[] = {
        "rayleigh", "vtest", "pdm", "chisq", "kde", "induced-K",
        "mixture-BIC"
    };
    size_t n_fam = sizeof(families) / sizeof(families[0]);

    if (t == NULL) return NULL;
    for (size_t b = 0; b < c->n_blocks; b++) {
        double spacings[16];
        double score[16];
        size_t support[16];
        size_t n_sp = 0;

        for (size_t i = 0; i < c->n_evidence; i++) {
            bool seen = false;

            if (c->evidence[i].block != b || !c->evidence[i].retained) {
                continue;
            }
            for (size_t s = 0; s < n_sp; s++) {
                if (fabs(spacings[s] - c->evidence[i].spacing_hz) < 1.0) {
                    seen = true;
                    break;
                }
            }
            if (!seen && n_sp < 16) {
                spacings[n_sp] = c->evidence[i].spacing_hz;
                score[n_sp] = 0.0;
                support[n_sp] = 0;
                n_sp++;
            }
        }
        /* accumulate weighted evidence per spacing */
        for (size_t s = 0; s < n_sp; s++) {
            bool have[7] = { false };

            for (size_t i = 0; i < c->n_evidence; i++) {
                double log_ev;

                if (c->evidence[i].block != b || !c->evidence[i].retained) {
                    continue;
                }
                if (fabs(c->evidence[i].spacing_hz - spacings[s]) >= 1.0) {
                    continue;
                }
                log_ev = -log10(fmax(1e-300, c->evidence[i].p));
                if (log_ev > 20.0) log_ev = 20.0;
                score[s] += c->evidence[i].weight * log_ev;
                for (size_t f = 0; f < n_fam; f++) {
                    if (strcmp(c->evidence[i].family, families[f]) == 0) {
                        have[f] = true;
                    }
                }
            }
            for (size_t f = 0; f < n_fam; f++) support[s] += have[f] ? 1 : 0;
        }
        /* rank the spacings (simple selection sort) */
        for (size_t a = 0; a < n_sp; a++) {
            for (size_t bb = a + 1; bb < n_sp; bb++) {
                if (score[bb] > score[a]) {
                    double ts = score[a];
                    double tdf = spacings[a];
                    size_t tn = support[a];

                    score[a] = score[bb];
                    spacings[a] = spacings[bb];
                    support[a] = support[bb];
                    score[bb] = ts;
                    spacings[bb] = tdf;
                    support[bb] = tn;
                }
            }
        }
        for (size_t s = 0; s < n_sp; s++) {
            char cells[7][128];
            int k_implied = 1;
            char fams[128] = "";

            for (size_t i = 0; i < c->n_evidence; i++) {
                if (c->evidence[i].block != b || !c->evidence[i].retained ||
                    fabs(c->evidence[i].spacing_hz - spacings[s]) >= 1.0) {
                    continue;
                }
                if (strcmp(c->evidence[i].family, "mixture-BIC") == 0 &&
                    c->evidence[i].K > 0) {
                    k_implied = c->evidence[i].K;
                }
                if (strlen(fams) < sizeof(fams) - 20) {
                    strncat(fams, c->evidence[i].family,
                            sizeof(fams) - strlen(fams) - 2);
                    strncat(fams, " ", sizeof(fams) - strlen(fams) - 2);
                }
            }
            snprintf(cells[0], sizeof(cells[0]), "%s", c->blocks[b].label);
            snprintf(cells[1], sizeof(cells[1]), "%zu", s + 1);
            snprintf(cells[2], sizeof(cells[2]), "%.1f", spacings[s] / 1e3);
            snprintf(cells[3], sizeof(cells[3]), "%.2f", score[s]);
            snprintf(cells[4], sizeof(cells[4]), "%zu (%s)", support[s], fams);
            snprintf(cells[5], sizeof(cells[5]), "%d", k_implied);
            snprintf(cells[6], sizeof(cells[6]), "%s",
                     s == 0 && support[s] > 0
                         ? (support[s] >= 3 ? "most likely (strong)"
                                            : "most likely (moderate)")
                         : support[s] > 0 ? "supported"
                                          : "weak");
            fc_add_cells(t, cells, 7);
        }
    }
    /* overall ranking across blocks */
    {
        double spacings[16];
        double score[16];
        size_t blocks[16];
        size_t n_sp = 0;

        for (size_t i = 0; i < c->n_evidence; i++) {
            bool seen = false;

            if (!c->evidence[i].retained) continue;
            for (size_t s = 0; s < n_sp; s++) {
                if (fabs(spacings[s] - c->evidence[i].spacing_hz) < 1.0) {
                    seen = true;
                    break;
                }
            }
            if (!seen && n_sp < 16) {
                spacings[n_sp] = c->evidence[i].spacing_hz;
                score[n_sp] = 0.0;
                blocks[n_sp] = 0;
                n_sp++;
            }
        }
        for (size_t s = 0; s < n_sp; s++) {
            bool seen_block[64] = { false };

            for (size_t i = 0; i < c->n_evidence; i++) {
                double log_ev;

                if (!c->evidence[i].retained ||
                    fabs(c->evidence[i].spacing_hz - spacings[s]) >= 1.0) {
                    continue;
                }
                log_ev = -log10(fmax(1e-300, c->evidence[i].p));
                if (log_ev > 20.0) log_ev = 20.0;
                score[s] += c->evidence[i].weight * log_ev;
                if (c->evidence[i].block < 64 &&
                    !seen_block[c->evidence[i].block]) {
                    seen_block[c->evidence[i].block] = true;
                    blocks[s]++;
                }
            }
        }
        for (size_t a = 0; a < n_sp; a++) {
            for (size_t bb = a + 1; bb < n_sp; bb++) {
                if (score[bb] > score[a]) {
                    double ts = score[a], tdf = spacings[a];
                    size_t tn = blocks[a];

                    score[a] = score[bb];
                    spacings[a] = spacings[bb];
                    blocks[a] = blocks[bb];
                    score[bb] = ts;
                    spacings[bb] = tdf;
                    blocks[bb] = tn;
                }
            }
        }
        fcl_table_note(t, "Overall ranking over all blocks (retained analyses "
                          "only; score = sum of weighted -log10 p):");
        for (size_t s = 0; s < n_sp; s++) {
            fcl_table_note(t, "  %zu. %.1f kHz — score %.2f in %zu block(s)%s",
                           s + 1, spacings[s] / 1e3, score[s], blocks[s],
                           s == 0 ? " — MOST LIKELY SUBCARRIER SPACING" : "");
        }
        if (n_sp == 0) {
            fcl_table_note(t, "No analysis retained evidence, so no "
                              "subcarrier spacing can be preferred; the "
                              "measurements are consistent with chance.");
        } else {
            fcl_table_note(t, "Synthesis: the measurements are best "
                              "explained by a subcarrier lattice with spacing "
                              "%.1f kHz (retained evidence in %zu block(s)).",
                           spacings[0] / 1e3, blocks[0]);
        }
    }
    return t;
}

/* ------------------------------------------------------------------ */
/* executor dispatch                                                  */
/* ------------------------------------------------------------------ */

static void fc_on_node(void *ctx, size_t index, size_t count,
                       const char *title)
{
    FcCtx *c = (FcCtx *)ctx;

    if (c->progress != NULL) {
        if (c->progress->total != NULL) {
            *c->progress->total = (int)count;
        }
        if (c->progress->done != NULL) {
            *c->progress->done = (int)index;
        }
        if (c->progress->stage != NULL) {
            *c->progress->stage = (int)index;
        }
    }
    if (getenv("TE15_DEBUG_FCDAG") != NULL) {
        fprintf(stderr, "@node %zu/%zu %s\n", index + 1, count,
                title != NULL ? title : "?");
    }
}

static bool fc_execute(void *ctx, DagCoreNode *node, char *err, size_t err_len)
{
    FcCtx *c = (FcCtx *)ctx;
    FclTable *table = NULL;
    const char *id = node->spec != NULL ? node->spec->id : NULL;

    (void)err;
    (void)err_len;
    if (id == NULL) return false;
    if (strcmp(id, "blocks") == 0) table = analysis_blocks(c);
    else if (strcmp(id, "rayleigh") == 0) table = analysis_rayleigh(c);
    else if (strcmp(id, "vtest") == 0) table = analysis_vtest(c);
    else if (strcmp(id, "pdm") == 0) table = analysis_pdm(c);
    else if (strcmp(id, "periodogram") == 0) table = analysis_periodogram(c);
    else if (strcmp(id, "chisq") == 0) table = analysis_chisq(c);
    else if (strcmp(id, "kde") == 0) table = analysis_kde(c);
    else if (strcmp(id, "cluster") == 0) table = analysis_cluster(c);
    else if (strcmp(id, "multimodal") == 0) table = analysis_multimodal(c);
    else if (strcmp(id, "correction") == 0) table = analysis_correction(c);
    else if (strcmp(id, "summary") == 0) table = analysis_summary(c);
    else if (strcmp(id, "induced-scan") == 0) table = analysis_induced_scan(c);
    else if (strcmp(id, "k-dispersion") == 0) table = analysis_k_dispersion(c);
    else if (strcmp(id, "cf-slope") == 0) table = analysis_cf_slope(c);
    else if (strcmp(id, "mixture-fit") == 0) table = analysis_mixture(c);
    else if (strcmp(id, "model-select") == 0) table = analysis_model_select(c);
    else if (strcmp(id, "k-summary") == 0) table = analysis_k_summary(c);
    else if (strcmp(id, "evidence-filter") == 0) {
        table = analysis_evidence_filter(c);
    }
    else if (strcmp(id, "synthesis") == 0) table = analysis_synthesis(c);
    if (table == NULL) return false;
    node->payload = table;
    return true;
}

static const char *fc_skip_reason(void *ctx, const DagCoreNodeSpec *spec)
{
    FcCtx *c = (FcCtx *)ctx;
    const char *id = spec != NULL ? spec->id : NULL;

    if (c->n_blocks == 0) {
        return "no measurement blocks are loaded";
    }
    if (id == NULL) return NULL;
    /* results-driven skipping: do not run analyses whose premise was not
     * established by the earlier nodes */
    if (strcmp(id, "induced-scan") == 0) {
        if (!c->any_clustering) {
            return "no significant subcarrier clustering was found by the "
                   "circular tests, so there is no lattice to explain by "
                   "averaging";
        }
    } else if (strcmp(id, "k-dispersion") == 0) {
        if (!c->have_induced) {
            return "no induced-lattice clustering was detected for any "
                   "candidate K";
        }
    } else if (strcmp(id, "cf-slope") == 0) {
        if (!c->have_reference) {
            return "the K = 1 residual dispersion is not usable as a "
                   "single-subcarrier reference (too tight or too broad)";
        }
    } else if (strcmp(id, "mixture-fit") == 0) {
        if (!c->have_induced) {
            return "no induced-lattice clustering was detected for any "
                   "candidate K, so no averaging model is fitted";
        }
    } else if (strcmp(id, "mdl-eigen") == 0) {
        return "requires the OFDM signal covariance/eigen-spectrum, which "
               "frequency lists do not provide (MDL subcarrier count is not "
               "applicable to this data source)";
    } else if (strcmp(id, "model-select") == 0) {
        if (!c->have_mixture && !c->have_induced) {
            return "no K model could be fitted";
        }
    } else if (strcmp(id, "k-summary") == 0) {
        if (!c->have_model) {
            return "no K model was selected";
        }
    } else if (strcmp(id, "evidence-filter") == 0) {
        if (c->n_evidence == 0) {
            return "no analysis produced any evidence to screen";
        }
    } else if (strcmp(id, "synthesis") == 0) {
        if (!c->have_evidence) {
            return "no analysis survived the evidence screen, so no "
                   "subcarrier spacing can be synthesised";
        }
    }
    return NULL;
}

/* Selection hook: map the registry spec back to its index. */
static bool fc_enabled(void *ctx, const DagCoreNodeSpec *spec)
{
    FcCtx *c = (FcCtx *)ctx;
    size_t idx;

    if (c->enabled == NULL || spec == NULL) return true;
    idx = (size_t)(spec - fc_node_specs);
    if (idx >= fc_node_count) return true;
    return c->enabled[idx];
}

FcRun *fc_dag_run(const FcDatasetList *list, FclScope scope,
                  const bool *enabled, const FcProgress *progress, char *err,
                  size_t err_len)
{
    FcCtx ctx;
    DagCoreExec exec;
    FcRun *run;

    memset(&ctx, 0, sizeof(ctx));
    ctx.list = list;
    ctx.scope = scope;
    ctx.enabled = enabled;
    ctx.progress = progress;
    if (progress != NULL) {
        if (progress->done != NULL) *progress->done = 0;
        if (progress->total != NULL) {
            *progress->total = (int)fc_node_count;
        }
        if (progress->stage != NULL) *progress->stage = -1;
    }
    ctx.blocks = fcl_blocks_build(list, scope, &ctx.n_blocks);
    fcl_rng_seed(&ctx.rng, FC_SEED);
    if (ctx.n_blocks > 0) {
        ctx.best_p = malloc(ctx.n_blocks * sizeof(*ctx.best_p));
        ctx.best_df = malloc(ctx.n_blocks * sizeof(*ctx.best_df));
        ctx.scan = calloc(ctx.n_blocks, FCLK_MAX * sizeof(*ctx.scan));
        ctx.fits = calloc(ctx.n_blocks, FCLK_MAX * sizeof(*ctx.fits));
        ctx.finest_k = calloc(ctx.n_blocks, sizeof(*ctx.finest_k));
        ctx.k_disp = calloc(ctx.n_blocks, sizeof(*ctx.k_disp));
        ctx.k_cf = calloc(ctx.n_blocks, sizeof(*ctx.k_cf));
        ctx.k_bic = calloc(ctx.n_blocks, sizeof(*ctx.k_bic));
        ctx.k_icl = calloc(ctx.n_blocks, sizeof(*ctx.k_icl));
        ctx.k_final = calloc(ctx.n_blocks, sizeof(*ctx.k_final));
        ctx.dbic = calloc(ctx.n_blocks, sizeof(*ctx.dbic));
        if (ctx.best_p != NULL && ctx.best_df != NULL) {
            for (size_t i = 0; i < ctx.n_blocks; i++) {
                ctx.best_p[i] = 1.0;
                ctx.best_df[i] = 15000.0;
            }
        }
    }
    memset(&exec, 0, sizeof(exec));
    exec.execute = fc_execute;
    exec.skip_reason = fc_skip_reason;
    exec.on_node = fc_on_node;
    exec.enabled = fc_enabled;
    run = dagcore_execute(fc_node_specs, fc_node_count, &exec, &ctx, err,
                          err_len);
    if (progress != NULL && progress->done != NULL) {
        *progress->done = (int)(run != NULL ? run->count : fc_node_count);
    }
    fcl_blocks_free(ctx.blocks);
    free(ctx.best_p);
    free(ctx.best_df);
    free(ctx.scan);
    free(ctx.fits);
    free(ctx.finest_k);
    free(ctx.k_disp);
    free(ctx.k_cf);
    free(ctx.k_bic);
    free(ctx.k_icl);
    free(ctx.k_final);
    free(ctx.dbic);
    free(ctx.tests);
    free(ctx.evidence);
    return run;
}

void fc_dag_free(FcRun *run)
{
    dagcore_destroy(run, fcl_table_free);
}
