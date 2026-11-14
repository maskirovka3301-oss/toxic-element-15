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
#include "fcdrift.h"
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
    /* warm-up drift correction (applied before the analyses) */
    double     **drift_values;   /* [n_blocks] corrected series      */
    double      *drift_min, *drift_max;
    FcdResult   *drift_results;  /* [n_blocks]                       */
    FcdGlobal    drift_global;
    bool         drift_applied;
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
        bool   is_3gpp;   /* spacing is an LTE/NR numerology */
    } *evidence;
    size_t      n_evidence;
    size_t      cap_evidence;
    bool        have_evidence;
    bool        non3gpp_used;   /* the non-3GPP fallback scan ran */
    bool        any_primary_clustering; /* a 15..240 kHz NR/LTE numerology
                                         * showed significant clustering */
} FcCtx;

/* The values every analysis uses: drift-corrected when the drift node
 * produced a correction, otherwise the raw measurements. */
static const double *block_values(const FcCtx *c, size_t b)
{
    if (c->drift_values != NULL && c->drift_values[b] != NULL) {
        return c->drift_values[b];
    }
    return c->blocks[b].values;
}

static double block_min(const FcCtx *c, size_t b)
{
    if (c->drift_values != NULL && c->drift_values[b] != NULL) {
        return c->drift_min[b];
    }
    return c->blocks[b].min_hz;
}

static double block_max(const FcCtx *c, size_t b)
{
    if (c->drift_values != NULL && c->drift_values[b] != NULL) {
        return c->drift_max[b];
    }
    return c->blocks[b].max_hz;
}

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
    c->evidence[c->n_evidence].is_3gpp = fcl_is_3gpp_spacing(spacing_hz);
    c->n_evidence++;
}


const DagCoreNodeSpec fc_node_specs[] = {
    { "blocks", "Measurement blocks", NULL },
    { "drift-correction", "Device warm-up drift estimation and correction",
      "blocks" },
    { "numerology-screen", "3GPP LTE/NR numerology screen", "blocks" },
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
    { "non3gpp-fallback", "Non-3GPP spacing fallback scan", "summary" },
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
      "evidence-filter" },
    { "hypothesis", "Source test: subcarrier lattice vs chance (combined)",
      "blocks" },
    { "subcarrier-map", "Reconstructed subcarriers per round",
      "hypothesis" }
};
const size_t fc_node_count = sizeof(fc_node_specs) / sizeof(fc_node_specs[0]);

/* ------------------------------------------------------------------ */
/* registry integrity                                                 */
/* ------------------------------------------------------------------ */

/* Every registry node that has an executor (mdl-eigen is the documented
 * deliberate exception: it is always skipped because frequency lists carry
 * no signal covariance). Kept in registry order. */
static const char *const fc_exec_ids[] = {
    "blocks", "drift-correction", "numerology-screen", "rayleigh", "vtest",
    "pdm",
    "periodogram", "chisq", "kde", "cluster", "multimodal", "correction",
    "summary", "induced-scan", "k-dispersion", "cf-slope", "mixture-fit",
    "model-select", "k-summary", "evidence-filter", "synthesis",
    "hypothesis", "subcarrier-map", "non3gpp-fallback"
};

static bool fc_has_executor(const char *id)
{
    for (size_t i = 0; i < sizeof(fc_exec_ids) / sizeof(fc_exec_ids[0]); i++) {
        if (strcmp(fc_exec_ids[i], id) == 0) return true;
    }
    return false;
}

static int fc_node_index(const char *id)
{
    for (size_t i = 0; i < fc_node_count; i++) {
        if (strcmp(fc_node_specs[i].id, id) == 0) return (int)i;
    }
    return -1;
}

/* Verify the registry is a complete, well-formed description of the
 * frequency-counter analyses: unique ids/titles, dependencies that point at
 * earlier entries, an executor for every node except the documented MDL
 * exception, no executor without a node, and room for every node in the
 * Settings checkbox array. Returns true when the registry is consistent;
 * otherwise fills `err` with the first problem. */
bool fc_registry_check(char *err, size_t err_len)
{
    if (fc_node_count == 0) {
        if (err && err_len) snprintf(err, err_len, "the registry is empty");
        return false;
    }
    if (fc_node_count > FC_MAX_NODES) {
        if (err && err_len) {
            snprintf(err, err_len, "registry has %zu nodes but only %d "
                     "checkbox slots", fc_node_count, FC_MAX_NODES);
        }
        return false;
    }
    for (size_t i = 0; i < fc_node_count; i++) {
        const DagCoreNodeSpec *n = &fc_node_specs[i];
        bool mdl = n->id != NULL && strcmp(n->id, "mdl-eigen") == 0;

        if (n->id == NULL || n->id[0] == '\0' ||
            n->title == NULL || n->title[0] == '\0') {
            if (err && err_len) snprintf(err, err_len,
                                         "node %zu has no id or title", i);
            return false;
        }
        for (size_t j = 0; j < i; j++) {
            if (strcmp(fc_node_specs[j].id, n->id) == 0) {
                if (err && err_len) snprintf(err, err_len,
                                             "duplicate node id '%s'",
                                             n->id);
                return false;
            }
        }
        if (!mdl && !fc_has_executor(n->id)) {
            if (err && err_len) snprintf(err, err_len,
                                         "node '%s' has no executor", n->id);
            return false;
        }
        if (mdl && fc_has_executor(n->id)) {
            if (err && err_len) snprintf(err, err_len,
                                         "node '%s' has an executor although "
                                         "it is documented as always "
                                         "skipped", n->id);
            return false;
        }
        if (n->requires != NULL && n->requires[0] != '\0') {
            char deps[256];
            char *save = NULL;
            char *tok;

            snprintf(deps, sizeof(deps), "%s", n->requires);
            for (tok = strtok_r(deps, ",", &save); tok != NULL;
                 tok = strtok_r(NULL, ",", &save)) {
                int dep = fc_node_index(tok);

                if (dep < 0) {
                    if (err && err_len) snprintf(err, err_len,
                                                 "node '%s' requires unknown "
                                                 "node '%s'", n->id, tok);
                    return false;
                }
                if ((size_t)dep >= i) {
                    if (err && err_len) snprintf(err, err_len,
                                                 "node '%s' requires '%s' "
                                                 "which does not run before "
                                                 "it", n->id, tok);
                    return false;
                }
            }
        }
    }
    for (size_t i = 0; i < sizeof(fc_exec_ids) / sizeof(fc_exec_ids[0]); i++) {
        if (fc_node_index(fc_exec_ids[i]) < 0) {
            if (err && err_len) snprintf(err, err_len,
                                         "executor '%s' has no registry node",
                                         fc_exec_ids[i]);
            return false;
        }
    }
    return true;
}

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
/* node: device warm-up drift estimation and correction               */
/* ------------------------------------------------------------------ */

static FclTable *analysis_drift(FcCtx *c)
{
    static const char *cols[] = { "Block", "n", "time base", "model",
                                  "tau (units)", "settle t95",
                                  "amplitude (Hz)", "plateau (MHz)",
                                  "RMS before (Hz)", "RMS after (Hz)",
                                  "corrected" };
    FclTable *t = fcl_table_new(
        "Device warm-up drift and settling time", cols, 11);

    if (t == NULL) return NULL;
    for (size_t b = 0; b < c->n_blocks; b++) {
        const double *times = NULL;
        size_t n_times = 0;
        FcdResult *res = &c->drift_results[b];
        char cells[11][128];

        /* observation times: the schema has no per-measurement timestamps
         * yet, so the sample index is the time base (see fcdrift.h); the
         * stub returns NULL until the schema provides seconds-since-video */
        if (fcd_timestamps_available()) {
            times = fcd_round_times(NULL, &n_times);
            if (n_times != c->blocks[b].n) times = NULL;
        }
        memset(res, 0, sizeof(*res));
        if (c->blocks[b].n < 8) {
            snprintf(cells[0], sizeof(cells[0]), "%s", c->blocks[b].label);
            snprintf(cells[1], sizeof(cells[1]), "%zu", c->blocks[b].n);
            snprintf(cells[2], sizeof(cells[2]), "%s",
                     times != NULL ? "seconds" : "index (stub)");
            snprintf(cells[3], sizeof(cells[3]), "too short");
            snprintf(cells[4], sizeof(cells[4]), "-");
            snprintf(cells[5], sizeof(cells[5]), "-");
            snprintf(cells[6], sizeof(cells[6]), "-");
            snprintf(cells[7], sizeof(cells[7]), "-");
            snprintf(cells[8], sizeof(cells[8]), "-");
            snprintf(cells[9], sizeof(cells[9]), "-");
            snprintf(cells[10], sizeof(cells[10]), "no");
            fc_add_cells(t, cells, 11);
            continue;
        }
        fcd_estimate(c->blocks[b].values, times, c->blocks[b].n, res);
        snprintf(cells[0], sizeof(cells[0]), "%s", c->blocks[b].label);
        snprintf(cells[1], sizeof(cells[1]), "%zu", c->blocks[b].n);
        snprintf(cells[2], sizeof(cells[2]), "%s",
                 times != NULL ? "seconds" : "index (stub)");
        snprintf(cells[3], sizeof(cells[3]), "%s", fcd_model_name(res->model));
        snprintf(cells[4], sizeof(cells[4]), "%s%.2f",
                 res->model == FCD_MODEL_EXP2 ? "fast " : "",
                 res->model == FCD_MODEL_EXP2 ? res->tau1 : res->tau1);
        snprintf(cells[5], sizeof(cells[5]), "%.1f", res->settle_time);
        snprintf(cells[6], sizeof(cells[6]), "%.2f",
                 fabs(res->amp1) + fabs(res->amp2));
        snprintf(cells[7], sizeof(cells[7]), "%.6f",
                 (res->model == FCD_MODEL_NONE ? c->blocks[b].values[0]
                                               : res->level) / 1e6);
        snprintf(cells[8], sizeof(cells[8]), "%.2f", res->rms_before);
        snprintf(cells[9], sizeof(cells[9]), "%.2f", res->rms_after);
        snprintf(cells[10], sizeof(cells[10]), "%s",
                 res->corrected ? "yes" : "no");
        if (res->corrected) {
            double *corr = malloc(c->blocks[b].n * sizeof(*corr));

            if (corr != NULL) {
                fcd_correct(c->blocks[b].values, times, c->blocks[b].n, res,
                            corr);
                c->drift_values[b] = corr;
                c->drift_min[b] = corr[0];
                c->drift_max[b] = corr[0];
                for (size_t i = 1; i < c->blocks[b].n; i++) {
                    if (corr[i] < c->drift_min[b]) c->drift_min[b] = corr[i];
                    if (corr[i] > c->drift_max[b]) c->drift_max[b] = corr[i];
                }
                c->drift_applied = true;
            }
        }
        fc_add_cells(t, cells, 11);
    }
    fcd_pool(c->drift_results, c->n_blocks, &c->drift_global);
    fcl_table_note(t, "The warm-up drift of a frequency counter is modelled "
                      "as a constant, linear, exponential or double-"
                      "exponential approach to the settled frequency "
                      "(f(t) = f_inf + A*exp(-t/tau), optional fast+slow "
                      "thermal terms), fitted robustly (IRLS) with a "
                      "variable-projection grid search over tau and selected "
                      "by BIC. Corrected measurements subtract the fitted "
                      "transient, keeping the settled level. Without "
                      "per-measurement timestamps the sample index is the "
                      "time base (the stub fcd_round_times() will provide "
                      "seconds-since-video once the schema has it).");
    if (c->drift_global.n_rounds > 0) {
        fcl_table_note(t, "Pooled device parameters across all rounds/"
                          "datasets: tau = %.1f %s, t95 settling = %.1f %s, "
                          "median peak drift = %.2f Hz (%zu round(s) fitted, "
                          "%zu corrected).", c->drift_global.tau_median,
                       c->drift_global.timestamps ? "s" : "samples",
                       c->drift_global.t95_median,
                       c->drift_global.timestamps ? "s" : "samples",
                       c->drift_global.amp_median, c->drift_global.n_rounds,
                       c->drift_global.n_corrected);
    } else {
        fcl_table_note(t, "No round produced a drift model: the measurements "
                          "are too short or show no significant drift.");
    }
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
                fcl_circular_stats(block_values(c, b), c->blocks[b].n, f0, df,
                                   &ang, &R);
                z = (double)c->blocks[b].n * R * R;
                p_asym = fcl_rayleigh_p(c->blocks[b].n, R);
                p_perm = fcl_perm_rayleigh(
                    block_values(c, b), c->blocks[b].n, f0, df,
                    block_min(c, b), block_max(c, b), reps, &c->rng);
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
                if (p_perm < 0.01) {
                    const FclNumerologySpec *ns =
                        fcl_numerology_for(df, 1.0);

                    c->any_clustering = true;
                    if (ns != NULL && ns->family == FCL_NUM_NR) {
                        c->any_primary_clustering = true;
                    }
                }
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
                fcl_circular_stats(block_values(c, b), c->blocks[b].n, f0, df,
                                   &ang, &R);
                V = fcl_v_statistic(c->blocks[b].n, R, ang, 0.0);
                p_perm = fcl_perm_v(
                    block_values(c, b), c->blocks[b].n, f0, df,
                    block_min(c, b), block_max(c, b), 0.0, reps,
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
                theta = fcl_pdm(block_values(c, b), c->blocks[b].n, f0, df,
                                bins);
                p_perm = fcl_perm_pdm(
                    block_values(c, b), c->blocks[b].n, f0, df,
                    block_min(c, b), block_max(c, b), bins, reps,
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

            fcl_circular_stats(block_values(c, b), c->blocks[b].n, 0.0, df,
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
                    block_values(c, b), c->blocks[b].n, f0, df,
                    block_min(c, b), block_max(c, b), 0.25, &dfree,
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
                    block_values(c, b), c->blocks[b].n, f0, df,
                    block_min(c, b), block_max(c, b));
                p_perm = fcl_perm_enrichment(
                    block_values(c, b), c->blocks[b].n, f0, df,
                    block_min(c, b), block_max(c, b), reps,
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
        fcl_cluster_test(block_values(c, b), c->blocks[b].n,
                         block_min(c, b), block_max(c, b), 64, reps,
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

        p = fcl_silverman(block_values(c, b), c->blocks[b].n, 1, 40, &c->rng,
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
        double span = block_max(c, b) - block_min(c, b);

        fcl_circular_stats(block_values(c, b), c->blocks[b].n, 0.0, df,
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
        rose = fcl_rose_plot(block_values(c, b), c->blocks[b].n, 0.0, df, 12,
                             8);
        comb = fcl_comb_plot(block_values(c, b), c->blocks[b].n, 0.0, df, 48,
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
        fclk_scan_block(block_values(c, b), c->blocks[b].n,
                        block_min(c, b), block_max(c, b), df,
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
        ok = fclk_dispersion_estimate(block_values(c, b), c->blocks[b].n, df,
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
        k_hat = fclk_cf_slope(block_values(c, b), c->blocks[b].n, df, 0.0,
                              FCLK_MAX, block_min(c, b),
                              block_max(c, b), &r2);
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

            if (!fclk_mixture_fit(block_values(c, b), c->blocks[b].n, 0.0,
                                  df, K, block_min(c, b),
                                  block_max(c, b), fit)) {
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
            rose = fcl_rose_plot(block_values(c, b), c->blocks[b].n, 0.0,
                                 delta, 12, 6);
            comb = fcl_comb_plot(block_values(c, b), c->blocks[b].n, 0.0,
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
                                  "implied K", "numerology",
                                  "interpretation" };
    FclTable *t = fcl_table_new(
        "Synthesis: most likely subcarrier spacings", cols, 8);
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
            char cells[8][128];
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
                     fcl_numerology_name(spacings[s]));
            snprintf(cells[7], sizeof(cells[7]), "%s",
                     s == 0 && support[s] > 0
                         ? (support[s] >= 3 ? "most likely (strong)"
                                            : "most likely (moderate)")
                         : support[s] > 0 ? "supported"
                                          : "weak");
            fc_add_cells(t, cells, 8);
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
/* node: hypothesis (combined source test)                            */
/* ------------------------------------------------------------------ */

/* Permutation replicates of the combined source test. 999 with the
 * (count+1)/(reps+1) correction keeps the permutation p-value valid
 * (Phipson & Smyth 2010) while resolving the tail reasonably. */
#define FC_HYP_REPS 999

/* von Mises concentration from the mean resultant length (Fisher, 1993,
 * "Statistical Analysis of Circular Data", section 4.5.5 approximation). */
static double vm_kappa(double R)
{
    if (R < 1e-6) return 0.0;
    if (R < 0.53) return 2.0 * R + R * R * R + 5.0 * pow(R, 5) / 6.0;
    if (R < 0.85) return -0.4 + 1.39 * R + 0.43 / (1.0 - R);
    if (R > 0.999) return 1e6;
    return 1.0 / (R * R * R - 4.0 * R * R + 3.0 * R);
}

/* Max over the candidate spacings of the Rayleigh statistic n*R^2 for one
 * value set: the Westfall-Young max statistic, so the p-value accounts for
 * having scanned several candidate spacings (the winner is chosen by the
 * data). */
static double max_lattice_z(const double *v, size_t n, const double *cands,
                            size_t n_cand)
{
    double best = 0.0;

    for (size_t i = 0; i < n_cand; i++) {
        double mean_angle = 0.0, R = 0.0;
        double z;

        fcl_circular_stats(v, n, 0.0, cands[i], &mean_angle, &R);
        z = (double)n * R * R;
        if (z > best) best = z;
    }
    return best;
}

/* The source hypothesis test: using the analysis results that survived the
 * evidence screen, decide whether the measurements are better explained by
 * one or more subcarriers with a common spacing than by no lattice at all.
 *
 * For every block the spacing with the strongest retained evidence is
 * tested directly on the (drift corrected) measurements with a
 * Rayleigh/von Mises isotropy test; the p-value uses the Westfall-Young max
 * statistic over all candidate spacings, so the multiplicity of the spacing
 * scan is controlled. The per-block permutation p-values are combined
 * across independent blocks with Fisher's method (chi2 = -2 sum ln p,
 * df = 2m). K (the number of averaged subcarriers) comes from the
 * mixture/BIC chain. The verdict is affirmative only when the combined p is
 * below alpha, more than one independent test family supports the retained
 * spacing, and enough measurements are available; otherwise the table says
 * that the data cannot conclude. */
static FclTable *analysis_hypothesis(FcCtx *c)
{
    static const char *cols[] = { "Block", "n", "spacing (kHz)", "R",
                                  "kappa", "PDM", "max z", "p (perm)",
                                  "BH q" };
    FclTable *t = fcl_table_new(
        "Source test: subcarrier lattice vs chance (combined)", cols, 9);
    size_t m = 0;
    size_t total_n = 0;
    size_t total_families = 0;
    double top_spacing = 0.0;
    int top_K = 1;
    double chi2_sum = 0.0;
    double p_comb = 1.0;
    char verdict[640];
    bool consistent = false;
    bool enough_data = false;
    size_t n_blocks = 0;
    double *p_raw = NULL;
    double *R_sel = NULL;
    double *df_sel = NULL;
    size_t *n_sel = NULL;
    double *kappa_sel = NULL;
    double *pdm_sel = NULL;
    double *maxz_sel = NULL;
    int *K_sel = NULL;
    size_t *fam_sel = NULL;

    if (t == NULL) return NULL;
    if (c->n_blocks == 0) {
        fcl_table_note(t, "No measurement blocks are loaded.");
        return t;
    }
    n_blocks = c->n_blocks;
    p_raw = calloc(n_blocks, sizeof(*p_raw));
    R_sel = calloc(n_blocks, sizeof(*R_sel));
    df_sel = calloc(n_blocks, sizeof(*df_sel));
    n_sel = calloc(n_blocks, sizeof(*n_sel));
    kappa_sel = calloc(n_blocks, sizeof(*kappa_sel));
    pdm_sel = calloc(n_blocks, sizeof(*pdm_sel));
    maxz_sel = calloc(n_blocks, sizeof(*maxz_sel));
    K_sel = calloc(n_blocks, sizeof(*K_sel));
    fam_sel = calloc(n_blocks, sizeof(*fam_sel));
    if (p_raw == NULL || R_sel == NULL || df_sel == NULL ||
        n_sel == NULL || kappa_sel == NULL || pdm_sel == NULL ||
        maxz_sel == NULL || K_sel == NULL || fam_sel == NULL) {
        goto out;
    }

    for (size_t b = 0; b < n_blocks; b++) {
        static const char *const fam_ids[] = {
            "rayleigh", "vtest", "pdm", "chisq", "kde",
            "induced-K", "mixture-BIC", "cf-slope"
        };
        const size_t n_fam = sizeof(fam_ids) / sizeof(fam_ids[0]);
        const double *values = block_values(c, b);
        size_t n = c->blocks[b].n;
        double cands[24];
        size_t n_cand = 0;
        double score = -1.0;
        bool have[8] = { false };
        size_t families = 0;
        double min_hz = block_min(c, b);
        double max_hz = block_max(c, b);
        double mean_angle = 0.0, R = 0.0;
        double obs_max_z;
        size_t hits = 0;
        FclRng rng;
        double *perm;

        if (values == NULL || n < 4) continue;
        /* the spacing that received the strongest retained evidence */
        for (size_t i = 0; i < c->n_evidence; i++) {
            double log_ev;

            if (c->evidence[i].block != b || !c->evidence[i].retained) {
                continue;
            }
            log_ev = -log10(fmax(1e-300, c->evidence[i].p)) *
                     c->evidence[i].weight;
            if (log_ev > 20.0) log_ev = 20.0;
            if (df_sel[b] <= 0.0) {
                df_sel[b] = c->evidence[i].spacing_hz;
                score = log_ev;
            } else if (fabs(c->evidence[i].spacing_hz - df_sel[b]) < 1.0) {
                score += log_ev;
            } else if (log_ev > score) {
                df_sel[b] = c->evidence[i].spacing_hz;
                score = log_ev;
            }
        }
        if (df_sel[b] <= 0.0) {
            if (c->best_df != NULL && c->best_df[b] > 0.0) {
                df_sel[b] = c->best_df[b];
            } else {
                continue;
            }
        }
        for (size_t i = 0; i < c->n_evidence; i++) {
            if (c->evidence[i].block != b || !c->evidence[i].retained) {
                continue;
            }
            if (fabs(c->evidence[i].spacing_hz - df_sel[b]) >= 1.0) {
                continue;
            }
            for (size_t f = 0; f < n_fam; f++) {
                if (strcmp(c->evidence[i].family, fam_ids[f]) == 0) {
                    have[f] = true;
                }
            }
        }
        for (size_t f = 0; f < n_fam; f++) {
            if (have[f]) families++;
        }
        /* Candidate spacings: the numerology grid plus every retained
         * spacing of this block. Only spacings that complete at least two
         * lattice cycles inside the block span are eligible: a narrow band
         * folded at a much larger spacing is trivially concentrated, which
         * would make the max statistic meaningless (and is why the
         * permutation null is folded over the same span). */
        {
            double span_hz = max_hz - min_hz;

            for (size_t s = 0; s < fcl_n_spacings && n_cand < 24; s++) {
                if (span_hz >= 2.0 * fcl_spacings_hz[s]) {
                    cands[n_cand++] = fcl_spacings_hz[s];
                }
            }
            for (size_t i = 0; i < c->n_evidence && n_cand < 24; i++) {
                bool seen = false;

                if (c->evidence[i].block != b ||
                    !c->evidence[i].retained) {
                    continue;
                }
                if (span_hz < 2.0 * c->evidence[i].spacing_hz) {
                    continue;
                }
                for (size_t s = 0; s < n_cand; s++) {
                    if (fabs(cands[s] - c->evidence[i].spacing_hz) < 1.0) {
                        seen = true;
                        break;
                    }
                }
                if (!seen) cands[n_cand++] = c->evidence[i].spacing_hz;
            }
        }
        if (n_cand == 0) {
            /* no spacing completes two cycles: the block is too narrow to
             * test any lattice */
            df_sel[b] = 0.0;
            continue;
        }
        /* the selected spacing must itself be eligible; when it is not,
         * test the best eligible candidate instead */
        {
            bool seen = false;

            for (size_t s = 0; s < n_cand; s++) {
                if (fabs(cands[s] - df_sel[b]) < 1.0) seen = true;
            }
            if (!seen) {
                /* pick the eligible candidate with the strongest retained
                 * evidence for this block */
                double best_score = -1.0;
                double best_df = cands[0];

                for (size_t i = 0; i < c->n_evidence; i++) {
                    double log_ev;

                    if (c->evidence[i].block != b ||
                        !c->evidence[i].retained) {
                        continue;
                    }
                    for (size_t s = 0; s < n_cand; s++) {
                        if (fabs(cands[s] - c->evidence[i].spacing_hz) >=
                            1.0) {
                            continue;
                        }
                        log_ev = -log10(fmax(1e-300, c->evidence[i].p)) *
                                 c->evidence[i].weight;
                        if (log_ev > 20.0) log_ev = 20.0;
                        if (log_ev > best_score) {
                            best_score = log_ev;
                            best_df = cands[s];
                        }
                    }
                }
                df_sel[b] = best_df;
            }
        }

        fcl_circular_stats(values, n, 0.0, df_sel[b], &mean_angle, &R);
        obs_max_z = max_lattice_z(values, n, cands, n_cand);

        perm = malloc(n * sizeof(*perm));
        if (perm == NULL) continue;
        fcl_rng_seed(&rng, 0x9E3779B97F4A7C15ULL ^
                            (unsigned long long)b);
        for (size_t r = 0; r < FC_HYP_REPS; r++) {
            double z;

            for (size_t i = 0; i < n; i++) {
                perm[i] = min_hz + (max_hz - min_hz) *
                                       fcl_rng_uniform(&rng);
            }
            z = max_lattice_z(perm, n, cands, n_cand);
            if (z >= obs_max_z) hits++;
        }
        free(perm);

        p_raw[b] = ((double)hits + 1.0) / ((double)FC_HYP_REPS + 1.0);
        R_sel[b] = R;
        n_sel[b] = n;
        maxz_sel[b] = obs_max_z;
        kappa_sel[b] = vm_kappa(R);
        pdm_sel[b] = fcl_pdm(values, n, 0.0, df_sel[b], 16);
        K_sel[b] = (c->k_final != NULL && c->k_final[b] > 0)
                       ? c->k_final[b] : 1;
        fam_sel[b] = families;
        total_n += n;
        total_families += families;
        m++;
        if (top_spacing <= 0.0) {
            top_spacing = df_sel[b];
            top_K = K_sel[b];
        }
    }
    /* the overall best-supported spacing: the retained evidence summed over
     * all blocks (the same weighting the synthesis node uses) */
    {
        double spacings[24];
        double scores[24];
        size_t n_sp = 0;
        int best_K = 1;

        for (size_t i = 0; i < c->n_evidence; i++) {
            bool seen = false;

            if (!c->evidence[i].retained) continue;
            for (size_t s = 0; s < n_sp; s++) {
                if (fabs(spacings[s] - c->evidence[i].spacing_hz) < 1.0) {
                    seen = true;
                    break;
                }
            }
            if (!seen && n_sp < 24) {
                spacings[n_sp] = c->evidence[i].spacing_hz;
                scores[n_sp] = 0.0;
                n_sp++;
            }
        }
        for (size_t s = 0; s < n_sp; s++) {
            for (size_t i = 0; i < c->n_evidence; i++) {
                double log_ev;

                if (!c->evidence[i].retained) continue;
                if (fabs(c->evidence[i].spacing_hz - spacings[s]) >= 1.0) {
                    continue;
                }
                log_ev = -log10(fmax(1e-300, c->evidence[i].p)) *
                         c->evidence[i].weight;
                if (log_ev > 20.0) log_ev = 20.0;
                scores[s] += log_ev;
                if (c->evidence[i].K > best_K &&
                    strcmp(c->evidence[i].family, "mixture-BIC") == 0) {
                    /* noted below once the winner is known */
                }
            }
        }
        if (n_sp > 0) {
            size_t best = 0;

            for (size_t s = 1; s < n_sp; s++) {
                if (scores[s] > scores[best]) best = s;
            }
            top_spacing = spacings[best];
            /* implied K from the mixture evidence for the winning spacing,
             * or from the K chain of a block that selected it */
            top_K = 1;
            for (size_t i = 0; i < c->n_evidence; i++) {
                if (!c->evidence[i].retained ||
                    strcmp(c->evidence[i].family, "mixture-BIC") != 0) {
                    continue;
                }
                if (fabs(c->evidence[i].spacing_hz - top_spacing) < 1.0 &&
                    c->evidence[i].K > top_K) {
                    top_K = c->evidence[i].K;
                }
            }
            if (top_K <= 1 && c->k_final != NULL) {
                for (size_t i = 0; i < n_blocks; i++) {
                    if (n_sel[i] > 0 &&
                        fabs(df_sel[i] - top_spacing) < 1.0 &&
                        c->k_final[i] > top_K) {
                        top_K = c->k_final[i];
                    }
                }
            }
        }
    }

    if (m == 0) {
        fcl_table_note(t, "No block retained evidence that can be tested "
                          "directly; the data cannot support a conclusion.");
        goto out;
    }

    /* Fisher's method over the independent blocks */
    for (size_t i = 0; i < n_blocks; i++) {
        if (n_sel[i] > 0 && p_raw[i] > 0.0 && p_raw[i] <= 1.0) {
            chi2_sum += -2.0 * log(p_raw[i]);
        }
    }
    p_comb = fcl_chi2_p(chi2_sum, 2 * m);

    /* per-block rows, with the BH adjustment of the permutation p-values */
    {
        double *adj = malloc(m * sizeof(*adj));
        size_t k = 0;
        size_t used = 0;

        if (adj != NULL) {
            for (size_t i = 0; i < n_blocks; i++) {
                if (n_sel[i] > 0) adj[used++] = p_raw[i];
            }
            if (used > 0) fcl_bh_adjust(adj, used, FC_ALPHA);
        }
        for (size_t i = 0; i < n_blocks; i++) {
            char cells[9][128];
            char qbuf[32];

            if (n_sel[i] == 0) continue;
            snprintf(qbuf, sizeof(qbuf), "%.3g",
                     adj != NULL ? adj[k] : p_raw[i]);
            k++;
            snprintf(cells[0], sizeof(cells[0]), "%s",
                     c->blocks[i].label);
            snprintf(cells[1], sizeof(cells[1]), "%zu", n_sel[i]);
            snprintf(cells[2], sizeof(cells[2]), "%.1f",
                     df_sel[i] / 1e3);
            snprintf(cells[3], sizeof(cells[3]), "%.3f", R_sel[i]);
            snprintf(cells[4], sizeof(cells[4]), "%.2f", kappa_sel[i]);
            snprintf(cells[5], sizeof(cells[5]), "%.3f", pdm_sel[i]);
            snprintf(cells[6], sizeof(cells[6]), "%.2f", maxz_sel[i]);
            snprintf(cells[7], sizeof(cells[7]), "%.4g", p_raw[i]);
            snprintf(cells[8], sizeof(cells[8]), "%s", qbuf);
            fc_add_cells(t, cells, 9);
        }
        free(adj);
    }
    {
        char cells[9][128];

        snprintf(cells[0], sizeof(cells[0]), "COMBINED");
        snprintf(cells[1], sizeof(cells[1]), "%zu", total_n);
        snprintf(cells[2], sizeof(cells[2]), "%.1f", top_spacing / 1e3);
        snprintf(cells[3], sizeof(cells[3]), "-");
        snprintf(cells[4], sizeof(cells[4]), "-");
        snprintf(cells[5], sizeof(cells[5]), "-");
        snprintf(cells[6], sizeof(cells[6]), "chi2 %.3f", chi2_sum);
        snprintf(cells[7], sizeof(cells[7]), "%.4g", p_comb);
        snprintf(cells[8], sizeof(cells[8]), "df %zu", 2 * m);
        fc_add_cells(t, cells, 9);
    }

    consistent = total_families >= 2 && m >= 1;
    enough_data = total_n >= 12;
    {
    const FclNumerologySpec *top_spec = fcl_numerology_for(top_spacing, 1.0);
    const bool top_3gpp = top_spec != NULL;
    const bool top_nr = top_spec != NULL && top_spec->family == FCL_NUM_NR;
    const char *top_family = fcl_numerology_name(top_spacing);

    if (p_comb < FC_ALPHA && consistent && enough_data) {
        if (top_3gpp && !top_nr) {
            snprintf(verdict, sizeof(verdict),
                     "No LTE/NR broadband numerology explains the data; the "
                     "measurements are consistent only with the 3GPP "
                     "special-channel grid %s (%.3g kHz%s), which is defined "
                     "for dedicated channels (eMBMS/NB-IoT/PRACH), not for "
                     "broadband subcarrier data (combined p = %.3g < alpha "
                     "%.2f).", top_family, top_spacing / 1e3,
                     top_K > 1 ? ", multiple subcarriers" : "", p_comb,
                     FC_ALPHA);
        } else if (!top_3gpp) {
            snprintf(verdict, sizeof(verdict),
                     "No 3GPP LTE/NR numerology explains the data: the only "
                     "supported explanation is the non-3GPP fallback spacing "
                     "of %.3g kHz%s (combined p = %.3g < alpha %.2f; %zu "
                     "measurements).", top_spacing / 1e3,
                     top_K > 1 ? " (multiple subcarriers)" : "", p_comb,
                     FC_ALPHA, total_n);
        } else if (top_K > 1) {
            snprintf(verdict, sizeof(verdict),
                     "The data are consistent with 4G-LTE/5G-NR numerology "
                     "%s (%.3g kHz subcarrier spacing): equal-weight "
                     "averages of K = %d latent subcarriers on that lattice "
                     "fit (combined p = %.3g < alpha %.2f; %zu test "
                     "families; %zu measurements).", top_family,
                     top_spacing / 1e3, top_K, p_comb, FC_ALPHA,
                     total_families, total_n);
        } else {
            snprintf(verdict, sizeof(verdict),
                     "The data are consistent with 4G-LTE/5G-NR numerology "
                     "%s (%.3g kHz subcarrier spacing, single subcarrier): "
                     "combined p = %.3g < alpha %.2f; %zu test families; "
                     "%zu measurements.", top_family, top_spacing / 1e3,
                     p_comb, FC_ALPHA, total_families, total_n);
        }
    } else if (!enough_data) {
        snprintf(verdict, sizeof(verdict),
                 "Cannot conclude: too few measurements survived the "
                 "evidence screen (%zu measurements; at least 12 needed).",
                 total_n);
    } else if (p_comb >= FC_ALPHA) {
        snprintf(verdict, sizeof(verdict),
                 "Cannot conclude that the source is better explained by "
                 "one or more subcarriers with a common spacing: the "
                 "combined isotropy test gives p = %.3g (alpha %.2f).",
                 p_comb, FC_ALPHA);
    } else {
        snprintf(verdict, sizeof(verdict),
                 "Cannot conclude: the spacing evidence is not consistent "
                 "across independent test families (combined p = %.3g, "
                 "alpha %.2f).", p_comb, FC_ALPHA);
    }
    fcl_table_note(t, "Numerology family of the best-supported spacing: %s "
                      "(3GPP LTE/NR compatible: %s).", top_family,
                   top_3gpp ? "yes" : "no: fallback spacing only");
    }
    fcl_table_note(t, "Alpha: %.2f; per-block p-values use the "
                      "Westfall-Young max statistic over all candidate "
                      "spacings (999 permutations, +1 correction); blocks "
                      "are combined with Fisher's method (chi2 = -2 sum "
                      "ln p, df = 2m).", FC_ALPHA);
    fcl_table_note(t, "Blocks tested: %zu; measurements: %zu; independent "
                      "test families supporting the retained spacing: %zu.",
                   m, total_n, total_families);
    fcl_table_note(t, "Best-supported spacing: %.3g kHz; implied K: %d.",
                   top_spacing / 1e3, top_K);
    fcl_table_note(t, "Combined Fisher test: chi2 = %.3f, df = %zu, "
                      "p = %.4g.", chi2_sum, 2 * m, p_comb);
    fcl_table_note(t, "Verdict: %s", verdict);
    fcl_table_note(t, "Caveats: blocks are treated as independent samples "
                      "of the same device, and a common lattice offset is "
                      "not identifiable from frequencies alone; the "
                      "retained analyses' own p-values are not re-used in "
                      "the combination (a direct per-block isotropy test "
                      "is), so dependent family evidence is summarised but "
                      "not double-counted.");

out:
    free(p_raw);
    free(R_sel);
    free(df_sel);
    free(n_sel);
    free(kappa_sel);
    free(pdm_sel);
    free(maxz_sel);
    free(K_sel);
    free(fam_sel);
    return t;
}

/* ------------------------------------------------------------------ */
/* node: subcarrier-map (reconstructed subcarriers per round)         */
/* ------------------------------------------------------------------ */

/* log I0(kappa) (Bessel function of the first kind, order 0), needed for
 * the von Mises density normalisation. Abramowitz & Stegun 9.8.1/9.8.4. */
static double log_bessel_i0(double k)
{
    if (k < 1e-8) return 0.0;
    if (k < 3.75) {
        double t = k / 3.75;
        double t2 = t * t;
        double y = t2 * (3.5156229 +
                         t2 * (3.0899424 +
                               t2 * (1.2067492 +
                                     t2 * (0.2659732 +
                                           t2 * (0.0360768 +
                                                 t2 * 0.0045813)))));
        return log1p(y);
    }
    {
        double y = 3.75 / k;

        return k - 0.5 * log(k) - 0.5 * log(2.0 * M_PI) +
               log1p((0.39894228 +
                      y * (0.01328592 +
                           y * (0.00225319 +
                                y * (-0.00157565 +
                                     y * (0.00916281 +
                                          y * (-0.02057706 +
                                               y * (0.02635537 +
                                                    y * (-0.01647633 +
                                                         y * 0.00392377)))))))) *
                     y);
    }
}

/* One reconstructed subcarrier set for a measurement round. */
typedef struct {
    double spacing_hz;
    int    K;
    double phase;        /* global phase offset in [0, 1) */
    double kappa;        /* von Mises concentration of the fit */
    double R;            /* mean resultant length at the spacing */
    double bic;
    double loglik;
    size_t n;
    size_t counts[FCLK_MAX];
    bool   valid;
} SubcarrierMap;

/* K-component equal-weight von Mises mixture whose component means sit at
 * phase + j/K on the unit circle (j = 0..K-1). EM with a shared
 * concentration; returns the BIC (lower is better) and fills `out`. */
static bool fit_subcarrier_phases(const double *phases, size_t n, int K,
                                  double spacing_hz, SubcarrierMap *out)
{
    double phase;
    double kappa;
    double bic;
    double prev_ll = -INFINITY;
    double *post = malloc(n * (size_t)K * sizeof(*post));

    if (post == NULL || n == 0 || K < 1 || K > FCLK_MAX) {
        free(post);
        return false;
    }
    /* initial phase: circular mean of the phases folded onto 1/K */
    {
        double sx = 0.0, sy = 0.0;

        for (size_t i = 0; i < n; i++) {
            double q = phases[i] * (double)K;
            double frac = q - floor(q);

            sx += cos(2.0 * M_PI * frac);
            sy += sin(2.0 * M_PI * frac);
        }
        phase = atan2(sy, sx) / (2.0 * M_PI * (double)K);
        if (phase < 0.0) phase += 1.0 / (double)K;
    }
    kappa = 1.0;
    bic = 0.0;
    for (int it = 0; it < 64; it++) {
        double ll = 0.0;
        double mx = 0.0, my = 0.0, wsum = 0.0;

        for (size_t i = 0; i < n; i++) {
            double dens[FCLK_MAX];
            double maxl = -INFINITY;
            double sum = 0.0;

            for (int j = 0; j < K; j++) {
                double mu = phase + (double)j / (double)K;
                double th = 2.0 * M_PI * (phases[i] - mu);

                dens[j] = kappa * cos(th);
                if (dens[j] > maxl) maxl = dens[j];
            }
            for (int j = 0; j < K; j++) {
                dens[j] = exp(dens[j] - maxl);
                sum += dens[j];
            }
            for (int j = 0; j < K; j++) {
                double p = dens[j] / sum;

                post[i * (size_t)K + (size_t)j] = p;
            }
            ll += maxl + log(sum) - log((double)K);
            (void)prev_ll;
        }
        ll -= (double)n * log(2.0 * M_PI * exp(log_bessel_i0(kappa)));
        /* M-step: phase from the circular mean of (p_i - j/K) weighted */
        for (size_t i = 0; i < n; i++) {
            for (int j = 0; j < K; j++) {
                double w = post[i * (size_t)K + (size_t)j];
                double ang = 2.0 * M_PI * (double)K *
                             (phases[i] - (double)j / (double)K);

                mx += w * cos(ang);
                my += w * sin(ang);
                wsum += w;
            }
        }
        if (wsum > 0.0 && (mx != 0.0 || my != 0.0)) {
            double ang = atan2(my, mx);
            double newp = ang / (2.0 * M_PI * (double)K);

            newp -= floor(newp * (double)K) / (double)K;
            phase = newp;
        }
        /* M-step: kappa from the shared residual resultant length */
        {
            double rx = 0.0, ry = 0.0;

            for (size_t i = 0; i < n; i++) {
                for (int j = 0; j < K; j++) {
                    double w = post[i * (size_t)K + (size_t)j];
                    double th = 2.0 * M_PI *
                                (phases[i] - phase - (double)j / (double)K);

                    rx += w * cos(th);
                    ry += w * sin(th);
                }
            }
            if (wsum > 0.0) {
                double Rb = sqrt(rx * rx + ry * ry) / wsum;

                kappa = vm_kappa(Rb > 0.999999 ? 0.999999 : Rb);
                if (kappa < 1e-3) kappa = 1e-3;
            }
        }
        if (it > 3 && ll - prev_ll < 1e-7 && ll < prev_ll) break;
        prev_ll = ll;
        bic = -2.0 * ll + (double)(K + 2) * log((double)n);
    }
    if (!isfinite(bic)) {
        free(post);
        return false;
    }
    /* usage counts: component with the largest posterior responsibility */
    memset(out, 0, sizeof(*out));
    out->spacing_hz = spacing_hz;
    out->K = K;
    out->phase = phase;
    out->kappa = kappa;
    out->bic = bic;
    out->n = n;
    out->valid = true;
    {
        double sx = 0.0, sy = 0.0;

        for (size_t i = 0; i < n; i++) {
            int best = 0;

            for (int j = 1; j < K; j++) {
                if (post[i * (size_t)K + (size_t)j] >
                    post[i * (size_t)K + (size_t)best]) {
                    best = j;
                }
            }
            out->counts[best]++;
            sx += sin(2.0 * M_PI * phases[i]);
            sy += cos(2.0 * M_PI * phases[i]);
        }
        out->R = sqrt(sx * sx + sy * sy) / (double)n;
    }
    free(post);
    return true;
}

/* Split a block label "dataset / session X / round Y[/ subject Z]" into
 * its parts. Returns false for subject-level or unparsable labels. */
static bool parse_round_label(const char *label, char *dataset,
                              size_t dataset_len, char *session,
                              size_t session_len, char *round,
                              size_t round_len)
{
    const char *sess, *sround, *subj;
    size_t n;

    if (label == NULL) return false;
    sess = strstr(label, " / session ");
    if (sess == NULL) return false;
    sround = strstr(sess + 1, " / round ");
    if (sround == NULL) return false;
    subj = strstr(sround + 1, " / subject ");
    if (subj != NULL) return false;   /* subject-level block */
    n = (size_t)(sess - label);
    if (dataset_len > 0) {
        if (n >= dataset_len) n = dataset_len - 1;
        memcpy(dataset, label, n);
        dataset[n] = '\0';
    }
    n = (size_t)(sround - sess - 11);
    if (session_len > 0) {
        if (n >= session_len) n = session_len - 1;
        memcpy(session, sess + 11, n);
        session[n] = '\0';
    }
    n = (size_t)((subj != NULL ? subj : label + strlen(label)) -
                 (sround + 9));
    if (round_len > 0) {
        if (n >= round_len) n = round_len - 1;
        memcpy(round, sround + 9, n);
        round[n] = '\0';
    }
    return true;
}

/* Reconstruct, for every measurement round in every session of every
 * dataset, which subcarriers were used: the lattice spacing is selected per
 * round from the candidate numerologies (plus the spacing with the
 * strongest retained evidence); for each spacing a K-component equal-weight
 * von Mises mixture on the folded phases (positions j/K + phase) is fitted
 * by EM for K = 1..FCLK_MAX and the spacing/K pair with the smallest BIC
 * wins. The used subcarrier indices, their phase offsets inside one spacing,
 * the per-subcarrier measurement counts and a reliability verdict are
 * reported per round. The lattice phase offset is not identifiable from
 * absolute frequencies alone, so indices are relative to the fitted phase. */
static FclTable *analysis_subcarrier_map(FcCtx *c)
{
    static const char *cols[] = { "Dataset", "Session", "Round", "n",
                                  "spacing (kHz)", "K", "used subcarriers",
                                  "phase offsets (kHz)", "counts", "R",
                                  "numerology", "verdict" };
    FclTable *t = fcl_table_new("Reconstructed subcarriers per round", cols,
                                12);
    size_t rounds = 0, multi = 0, clear = 0;
    double evidence_df = 0.0;

    if (t == NULL) return NULL;
    if (c->n_blocks == 0) {
        fcl_table_note(t, "No measurement blocks are loaded.");
        return t;
    }
    /* the spacing with the strongest retained evidence is always a
     * candidate, even when a round's own scan would prefer another one */
    {
        double best = -1.0;

        for (size_t i = 0; i < c->n_evidence; i++) {
            double score;

            if (!c->evidence[i].retained) continue;
            score = -log10(fmax(1e-300, c->evidence[i].p)) *
                    c->evidence[i].weight;
            if (score > best) {
                best = score;
                evidence_df = c->evidence[i].spacing_hz;
            }
        }
    }
    for (size_t b = 0; b < c->n_blocks; b++) {
        char dataset[128], session[64], round[64];
        const double *values = block_values(c, b);
        size_t n = c->blocks[b].n;
        double span, raw_span;
        double best_bic = INFINITY;
        SubcarrierMap best;
        bool found = false;
        bool used_fallback = false;

        if (values == NULL || n < 6) continue;
        if (!parse_round_label(c->blocks[b].label, dataset, sizeof(dataset),
                               session, sizeof(session), round,
                               sizeof(round))) {
            continue;
        }
        span = block_max(c, b) - block_min(c, b);
        raw_span = c->blocks[b].max_hz - c->blocks[b].min_hz;
        /* The warm-up drift model can absorb a genuine frequency ramp when
         * the measurements happen to be ordered by frequency; the corrected
         * series then collapses and no lattice can be reconstructed. Fall
         * back to the raw round values in that case (the reconstruction
         * only looks at the phase structure, for which the slow trend is
         * irrelevant). */
        if (c->drift_values != NULL && c->drift_values[b] != NULL &&
            raw_span > 0.0 && span < 0.25 * raw_span) {
            values = c->blocks[b].values;
            span = raw_span;
        }
        memset(&best, 0, sizeof(best));
        {
            double cands[32];
            size_t n_cand = 0;
            int pass;

            /* Pass 0 tests the 3GPP numerologies (plus any retained 3GPP
             * evidence spacing); only when nothing fits there, pass 1 adds
             * the non-3GPP fallback grid (which includes 5 kHz). */
            for (pass = 0; pass < 2 && !found; pass++) {
                n_cand = 0;
                if (pass == 0) {
                    for (size_t s = 0;
                         s < fcl_n_spacings && n_cand < 32; s++) {
                        cands[n_cand++] = fcl_spacings_hz[s];
                    }
                } else {
                    for (size_t s = 0;
                         s < fcl_n_non3gpp_spacings && n_cand < 32; s++) {
                        cands[n_cand++] = fcl_non3gpp_spacings_hz[s];
                    }
                }
                for (size_t i = 0; i < c->n_evidence && n_cand < 32; i++) {
                    bool seen = false;

                    if (!c->evidence[i].retained) continue;
                    if ((pass == 0) != c->evidence[i].is_3gpp) continue;
                    for (size_t s = 0; s < n_cand; s++) {
                        if (fabs(cands[s] - c->evidence[i].spacing_hz) <
                            1.0) {
                            seen = true;
                        }
                    }
                    if (!seen) cands[n_cand++] = c->evidence[i].spacing_hz;
                }
                if (evidence_df > 0.0 &&
                    ((pass == 0) == fcl_is_3gpp_spacing(evidence_df))) {
                    bool seen = false;

                    for (size_t s = 0; s < n_cand; s++) {
                        if (fabs(cands[s] - evidence_df) < 1.0) seen = true;
                    }
                    if (!seen && n_cand < 32) cands[n_cand++] = evidence_df;
                }
                for (size_t s = 0; s < n_cand; s++) {
                    double phases[2048];
                    double df = cands[s];
                    size_t m = n < 2048 ? n : 2048;

                    /* at least 1.5 lattice cycles must fit in the span */
                    if (df <= 0.0 || span < 1.5 * df) continue;
                    for (size_t i = 0; i < m; i++) {
                        phases[i] = fcl_fold(values[i], 0.0, df);
                    }
                    for (int K = 1; K <= FCLK_MAX; K++) {
                        SubcarrierMap fit;

                        if (!fit_subcarrier_phases(phases, m, K, df,
                                                   &fit)) {
                            continue;
                        }
                        if (!found || fit.bic < best_bic) {
                            best_bic = fit.bic;
                            best = fit;
                            found = true;
                        }
                    }
                }
                if (found && pass == 1) used_fallback = true;
            }
        }
        if (!found) {
            char cells[12][128];

            snprintf(cells[0], sizeof(cells[0]), "%s", dataset);
            snprintf(cells[1], sizeof(cells[1]), "%s", session);
            snprintf(cells[2], sizeof(cells[2]), "%s", round);
            snprintf(cells[3], sizeof(cells[3]), "%zu", n);
            snprintf(cells[4], sizeof(cells[4]), "-");
            snprintf(cells[5], sizeof(cells[5]), "-");
            snprintf(cells[6], sizeof(cells[6]), "-");
            snprintf(cells[7], sizeof(cells[7]), "-");
            snprintf(cells[8], sizeof(cells[8]), "-");
            snprintf(cells[9], sizeof(cells[9]), "-");
            snprintf(cells[10], sizeof(cells[10]), "-");
            snprintf(cells[11], sizeof(cells[11]),
                     "cannot reconstruct (span too narrow)");
            fc_add_cells(t, cells, 12);
            rounds++;
            continue;
        }
        {
            char cells[12][128];
            char used[128] = "";
            char offsets[128] = "";
            char counts[128] = "";
            bool all_used = true;
            double sep_hz = best.spacing_hz / (double)best.K;
            double sigma_hz;

            for (int j = 0; j < best.K; j++) {
                char tmp[32];

                if (best.counts[j] == 0) {
                    all_used = false;
                    continue;
                }
                snprintf(tmp, sizeof(tmp), "%s%d", used[0] ? "," : "", j);
                strncat(used, tmp, sizeof(used) - strlen(used) - 1);
                snprintf(tmp, sizeof(tmp), "%s%.2f",
                         offsets[0] ? "," : "",
                         ((best.phase + (double)j / (double)best.K) -
                          floor(best.phase + (double)j / (double)best.K)) *
                             best.spacing_hz / 1e3);
                strncat(offsets, tmp, sizeof(offsets) - strlen(offsets) - 1);
                snprintf(tmp, sizeof(tmp), "%s%zu",
                         counts[0] ? "," : "", best.counts[j]);
                strncat(counts, tmp, sizeof(counts) - strlen(counts) - 1);
            }
            (void)all_used;
            if (best.K > 1) multi++;
            sigma_hz = (best.kappa > 1e-6)
                           ? best.spacing_hz /
                                 (2.0 * M_PI * sqrt(2.0 * best.kappa))
                           : INFINITY;
            if (best.K > 1 && sep_hz > 2.0 * sigma_hz) clear++;
            snprintf(cells[0], sizeof(cells[0]), "%s", dataset);
            snprintf(cells[1], sizeof(cells[1]), "%s", session);
            snprintf(cells[2], sizeof(cells[2]), "%s", round);
            snprintf(cells[3], sizeof(cells[3]), "%zu", n);
            snprintf(cells[4], sizeof(cells[4]), "%.1f",
                     best.spacing_hz / 1e3);
            snprintf(cells[5], sizeof(cells[5]), "%d", best.K);
            snprintf(cells[6], sizeof(cells[6]), "%s", used);
            snprintf(cells[7], sizeof(cells[7]), "%s", offsets);
            snprintf(cells[8], sizeof(cells[8]), "%s", counts);
            snprintf(cells[9], sizeof(cells[9]), "%.3f", best.R);
            snprintf(cells[10], sizeof(cells[10]), "%s",
                     fcl_numerology_name(best.spacing_hz));
            snprintf(cells[11], sizeof(cells[11]), "%s%s",
                     used_fallback ? "non-3GPP fallback: " : "",
                     best.K > 1 && sep_hz > 2.0 * sigma_hz
                         ? "clear multi-subcarrier structure"
                         : best.K > 1 ? "multiple subcarriers (overlapping)"
                                      : "single subcarrier per spacing");
            fc_add_cells(t, cells, 12);
            rounds++;
        }
    }
    fcl_table_note(t, "Rounds reconstructed: %zu; with more than one "
                      "subcarrier: %zu (clearly separated: %zu).", rounds,
                   multi, clear);
    fcl_table_note(t, "Method: per round the spacing is selected from the "
                      "LTE/NR numerology grid (plus the best retained "
                      "evidence spacing) by BIC of an equal-weight von Mises "
                      "mixture on the phases folded at that spacing; the "
                      "number of subcarriers K is the same BIC minimisation "
                      "(K = 1..%d) over the component positions j/K + phase.",
                   FCLK_MAX);
    fcl_table_note(t, "Subcarrier indices and phase offsets are relative to "
                      "the fitted lattice phase: the absolute lattice offset "
                      "is not identifiable from frequencies alone. 'counts' "
                      "is the number of measurements assigned to each "
                      "subcarrier index.");
    return t;
}

/* ------------------------------------------------------------------ */
/* node: numerology-screen (3GPP LTE/NR candidates first)             */
/* ------------------------------------------------------------------ */

/* Screen the candidate grid: for every block the best-fitting 3GPP
 * numerology (Rayleigh statistic on the folded phases) and its family. The
 * report makes the "3GPP first" policy explicit: only the LTE/NR
 * numerologies are candidates here, and 5 kHz is not one of them. */
static FclTable *analysis_numerology_screen(FcCtx *c)
{
    static const char *cols[] = { "Block", "n", "span (kHz)",
                                  "best 3GPP spacing (kHz)", "numerology",
                                  "R", "p (asymptotic)", "cycles" };
    FclTable *t = fcl_table_new("3GPP LTE/NR numerology screen", cols, 8);
    size_t rows = 0;

    if (t == NULL) return NULL;
    fcl_table_note(t, "Candidate numerologies (3GPP only): 15/30/60/120/240 "
                      "kHz (Delta f = 15*2^mu), 7.5 kHz (LTE eMBMS), "
                      "3.75 kHz (NB-IoT uplink), 1.25 kHz (long PRACH). "
                      "5 kHz is deliberately NOT a candidate: it is not a "
                      "3GPP numerology (not a 15*2^mu value) and would break "
                      "OFDM orthogonality with LTE/NR hardware. Non-3GPP "
                      "spacings are only evaluated by the fallback node when "
                      "no 3GPP numerology explains the clustering.");
    for (size_t b = 0; b < c->n_blocks; b++) {
        const double *values = block_values(c, b);
        size_t n = c->blocks[b].n;
        double span, best_df = 0.0, best_R = 0.0, best_p = 1.0;
        const FclNumerologySpec *best_spec = NULL;
        double cycles = 0.0;

        if (values == NULL || n < 4) continue;
        span = block_max(c, b) - block_min(c, b);
        for (size_t s = 0; s < fcl_n_spacings; s++) {
            double df = fcl_spacings_hz[s];
            double mean_angle = 0.0, R = 0.0, p;

            if (df <= 0.0 || span < 1.5 * df) continue;
            fcl_circular_stats(values, n, 0.0, df, &mean_angle, &R);
            if (R <= best_R) continue;
            p = fcl_rayleigh_p(n, R);
            best_R = R;
            best_p = p;
            best_df = df;
            best_spec = fcl_numerology_for(df, 1.0);
            cycles = span / df;
        }
        if (best_df <= 0.0) {
            char cells[8][128];

            snprintf(cells[0], sizeof(cells[0]), "%s", c->blocks[b].label);
            snprintf(cells[1], sizeof(cells[1]), "%zu", n);
            snprintf(cells[2], sizeof(cells[2]), "%.1f", span / 1e3);
            snprintf(cells[3], sizeof(cells[3]), "-");
            snprintf(cells[4], sizeof(cells[4]), "-");
            snprintf(cells[5], sizeof(cells[5]), "-");
            snprintf(cells[6], sizeof(cells[6]), "-");
            snprintf(cells[7], sizeof(cells[7]), "%s",
                     "span too narrow for any 3GPP numerology");
            fc_add_cells(t, cells, 8);
            rows++;
            continue;
        }
        {
            char cells[8][128];

            snprintf(cells[0], sizeof(cells[0]), "%s", c->blocks[b].label);
            snprintf(cells[1], sizeof(cells[1]), "%zu", n);
            snprintf(cells[2], sizeof(cells[2]), "%.1f", span / 1e3);
            snprintf(cells[3], sizeof(cells[3]), "%.2f", best_df / 1e3);
            snprintf(cells[4], sizeof(cells[4]), "%s",
                     best_spec != NULL ? best_spec->name : "3GPP");
            snprintf(cells[5], sizeof(cells[5]), "%.3f", best_R);
            snprintf(cells[6], sizeof(cells[6]), "%.3g", best_p);
            snprintf(cells[7], sizeof(cells[7]), "%.1f", cycles);
            fc_add_cells(t, cells, 8);
            rows++;
        }
    }
    fcl_table_note(t, "Rows screened: %zu. A 3GPP spacing with a small "
                      "asymptotic p is followed up by the permutation tests "
                      "of the Rayleigh/V/PDM nodes; if none of the 3GPP "
                      "numerologies shows clustering, the non-3GPP fallback "
                      "node is evaluated instead.", rows);
    return t;
}

/* ------------------------------------------------------------------ */
/* node: non3gpp-fallback                                             */
/* ------------------------------------------------------------------ */

/* Fallback spacing scan: only evaluated when NO 3GPP numerology shows
 * significant subcarrier clustering. Tests the non-3GPP grid with the same
 * circular statistics and records it as fallback evidence (flagged as
 * non-3GPP in the evidence structures). */
static FclTable *analysis_non3gpp_fallback(FcCtx *c)
{
    static const char *cols[] = { "Block", "n", "non-3GPP spacing (kHz)",
                                  "R", "p (permutation)", "PDM p", "cycles" };
    FclTable *t = fcl_table_new("Non-3GPP spacing fallback scan", cols, 7);
    size_t significant = 0;
    size_t rows = 0;

    if (t == NULL) return NULL;
    if (c->any_clustering) {
        fcl_table_note(t, "A 3GPP special-channel grid (LTE eMBMS / NB-IoT / "
                          "PRACH) shows clustering, but no LTE/NR broadband "
                          "numerology (15..240 kHz) does; the non-3GPP "
                          "fallback is therefore evaluated as well.");
    }
    fcl_table_note(t, "Fallback policy: the primary goal is to test whether "
                      "the measurements could have been generated by 4G-LTE/"
                      "5G-NR. The 3GPP numerologies (15/30/60/120/240 kHz, "
                      "7.5 kHz eMBMS, 3.75 kHz NB-IoT, 1.25 kHz PRACH) are "
                      "always evaluated first; non-3GPP spacings - including "
                      "5 kHz - are only considered when no 3GPP numerology "
                      "explains the clustering. Results here are exploratory "
                      "fallback evidence, not cellular numerologies.");
    for (size_t b = 0; b < c->n_blocks; b++) {
        const double *values = block_values(c, b);
        size_t n = c->blocks[b].n;
        double span;
        size_t reps = reps_for(c);
        FclRng rng;
        double best_df = 0.0, best_R = 0.0, best_p = 1.0, best_pdm = 1.0;
        double best_cycles = 0.0;

        if (values == NULL || n < 4) continue;
        span = block_max(c, b) - block_min(c, b);
        fcl_rng_seed(&rng, 0xC0FFEE123456789ULL ^
                            (unsigned long long)b);
        for (size_t s = 0; s < fcl_n_non3gpp_spacings; s++) {
            double df = fcl_non3gpp_spacings_hz[s];
            double mean_angle = 0.0, R = 0.0, p, pdm_p;

            if (df <= 0.0 || span < 2.0 * df) continue;
            fcl_circular_stats(values, n, 0.0, df, &mean_angle, &R);
            if (R <= best_R) continue;
            p = fcl_perm_rayleigh(values, n, 0.0, df, block_min(c, b),
                                  block_max(c, b), reps, &rng);
            pdm_p = fcl_perm_pdm(values, n, 0.0, df, block_min(c, b),
                                 block_max(c, b), 16, reps, &rng);
            best_R = R;
            best_p = p;
            best_pdm = pdm_p;
            best_df = df;
            best_cycles = span / df;
        }
        if (best_df <= 0.0) {
            char cells[7][128];

            snprintf(cells[0], sizeof(cells[0]), "%s", c->blocks[b].label);
            snprintf(cells[1], sizeof(cells[1]), "%zu", n);
            snprintf(cells[2], sizeof(cells[2]),
                     "span too narrow for any non-3GPP candidate");
            snprintf(cells[3], sizeof(cells[3]), "-");
            snprintf(cells[4], sizeof(cells[4]), "-");
            snprintf(cells[5], sizeof(cells[5]), "-");
            snprintf(cells[6], sizeof(cells[6]), "-");
            fc_add_cells(t, cells, 7);
            rows++;
            continue;
        }
        if (best_p < FC_ALPHA) {
            ctx_add_evidence(c, "non3gpp", b, best_df, 1, best_R, best_p);
            ctx_note_best(c, b, best_df, best_p);
            c->non3gpp_used = true;
            c->any_clustering = true;
            significant++;
        }
        if (best_pdm < FC_ALPHA) {
            ctx_add_evidence(c, "non3gpp-pdm", b, best_df, 1, best_R,
                             best_pdm);
            c->non3gpp_used = true;
            c->any_clustering = true;
        }
        {
            char cells[7][128];

            snprintf(cells[0], sizeof(cells[0]), "%s", c->blocks[b].label);
            snprintf(cells[1], sizeof(cells[1]), "%zu", n);
            snprintf(cells[2], sizeof(cells[2]), "%.2f", best_df / 1e3);
            snprintf(cells[3], sizeof(cells[3]), "%.3f", best_R);
            snprintf(cells[4], sizeof(cells[4]), "%.3g", best_p);
            snprintf(cells[5], sizeof(cells[5]), "%.3g", best_pdm);
            snprintf(cells[6], sizeof(cells[6]), "%.1f", best_cycles);
            fc_add_cells(t, cells, 7);
            rows++;
        }
    }
    fcl_table_note(t, "Fallback spacings tested per block: the best of "
                      "5/10/20/25/40/50/80/100/200 kHz (a non-3GPP grid); "
                      "blocks listed: %zu. Significant blocks: %zu. %s",
                   rows, significant,
                   significant > 0
                       ? "The 3GPP numerologies could not explain the data, "
                         "so non-3GPP spacings are reported as a fallback "
                         "only."
                       : "Neither the 3GPP nor the non-3GPP spacings explain "
                         "the measurements.");
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
    else if (strcmp(id, "drift-correction") == 0) table = analysis_drift(c);
    else if (strcmp(id, "numerology-screen") == 0) {
        table = analysis_numerology_screen(c);
    }
    else if (strcmp(id, "non3gpp-fallback") == 0) {
        table = analysis_non3gpp_fallback(c);
    }
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
    else if (strcmp(id, "hypothesis") == 0) table = analysis_hypothesis(c);
    else if (strcmp(id, "subcarrier-map") == 0) {
        table = analysis_subcarrier_map(c);
    }
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

    {
        char reg_err[192];

        if (!fc_registry_check(reg_err, sizeof(reg_err))) {
            if (err != NULL && err_len > 0) {
                snprintf(err, err_len, "analysis registry error: %s",
                         reg_err);
            }
            return NULL;
        }
    }
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
        ctx.drift_values = calloc(ctx.n_blocks, sizeof(*ctx.drift_values));
        ctx.drift_min = calloc(ctx.n_blocks, sizeof(*ctx.drift_min));
        ctx.drift_max = calloc(ctx.n_blocks, sizeof(*ctx.drift_max));
        ctx.drift_results = calloc(ctx.n_blocks, sizeof(*ctx.drift_results));
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
    if (ctx.drift_values != NULL) {
        for (size_t i = 0; i < ctx.n_blocks; i++) free(ctx.drift_values[i]);
    }
    free(ctx.drift_values);
    free(ctx.drift_min);
    free(ctx.drift_max);
    free(ctx.drift_results);
    free(ctx.tests);
    free(ctx.evidence);
    return run;
}

void fc_dag_free(FcRun *run)
{
    dagcore_destroy(run, fcl_table_free);
}
