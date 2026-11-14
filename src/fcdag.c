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
} FcCtx;


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
    { "summary", "Per-block subcarrier-lattice summary", "correction" }
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
                ctx_note_best(c, b, df, p_perm);
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
/* executor dispatch                                                  */
/* ------------------------------------------------------------------ */

static void fc_on_node(void *ctx, size_t index, size_t count,
                       const char *title)
{
    (void)ctx;
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
    if (table == NULL) return false;
    node->payload = table;
    return true;
}

static const char *fc_skip_reason(void *ctx, const DagCoreNodeSpec *spec)
{
    FcCtx *c = (FcCtx *)ctx;

    (void)spec;
    if (c->n_blocks == 0) {
        return "no measurement blocks are loaded";
    }
    return NULL;
}

FcRun *fc_dag_run(const FcDatasetList *list, char *err, size_t err_len)
{
    FcCtx ctx;
    DagCoreExec exec;
    FcRun *run;

    memset(&ctx, 0, sizeof(ctx));
    ctx.list = list;
    ctx.blocks = fcl_blocks_from_datasets(list, &ctx.n_blocks);
    fcl_rng_seed(&ctx.rng, FC_SEED);
    if (ctx.n_blocks > 0) {
        ctx.best_p = malloc(ctx.n_blocks * sizeof(*ctx.best_p));
        ctx.best_df = malloc(ctx.n_blocks * sizeof(*ctx.best_df));
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
    run = dagcore_execute(fc_node_specs, fc_node_count, &exec, &ctx, err,
                          err_len);
    fcl_blocks_free(ctx.blocks);
    free(ctx.best_p);
    free(ctx.best_df);
    free(ctx.tests);
    return run;
}

void fc_dag_free(FcRun *run)
{
    dagcore_destroy(run, fcl_table_free);
}
