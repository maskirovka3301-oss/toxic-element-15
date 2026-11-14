/*
 * fclattice.c — subcarrier-lattice statistics (see fclattice.h).
 */
#include "fclattice.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const double fcl_spacings_hz[] = {
    5000.0, 15000.0, 30000.0, 60000.0, 120000.0, 240000.0
};
const size_t fcl_n_spacings = sizeof(fcl_spacings_hz) / sizeof(fcl_spacings_hz[0]);

const double fcl_offsets_hz[] = { 0.0 };
const size_t fcl_n_offsets = sizeof(fcl_offsets_hz) / sizeof(fcl_offsets_hz[0]);

/* ------------------------------------------------------------------ */
/* RNG                                                                */
/* ------------------------------------------------------------------ */

void fcl_rng_seed(FclRng *rng, unsigned long long seed)
{
    rng->state = seed == 0 ? 0x9e3779b97f4a7c15ULL : seed;
}

static unsigned long long rng_next(FclRng *rng)
{
    unsigned long long x = rng->state;

    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    rng->state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

double fcl_rng_uniform(FclRng *rng)
{
    return (double)(rng_next(rng) >> 11) * (1.0 / 9007199254740992.0);
}

double fcl_rng_normal(FclRng *rng)
{
    double u1 = fcl_rng_uniform(rng);
    double u2 = fcl_rng_uniform(rng);

    if (u1 < 1e-300) u1 = 1e-300;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/* ------------------------------------------------------------------ */
/* tables                                                             */
/* ------------------------------------------------------------------ */

FclTable *fcl_table_new(const char *title, const char *const *columns,
                        size_t n_columns)
{
    FclTable *t = calloc(1, sizeof(*t));

    if (t == NULL) return NULL;
    t->title = title != NULL ? strdup(title) : NULL;
    t->n_columns = n_columns;
    if (n_columns > 0) {
        t->columns = calloc(n_columns, sizeof(*t->columns));
        if (t->columns == NULL) {
            fcl_table_free(t);
            return NULL;
        }
        for (size_t i = 0; i < n_columns; i++) {
            t->columns[i] = columns[i] != NULL ? strdup(columns[i]) : strdup("");
        }
    }
    return t;
}

void fcl_table_free(void *table)
{
    FclTable *t = (FclTable *)table;

    if (t == NULL) return;
    free(t->title);
    for (size_t i = 0; i < t->n_columns; i++) free(t->columns[i]);
    free(t->columns);
    for (size_t r = 0; r < t->n_rows; r++) {
        if (t->cells[r] != NULL) {
            for (size_t c = 0; c < t->n_columns; c++) free(t->cells[r][c]);
            free(t->cells[r]);
        }
    }
    free(t->cells);
    free(t->notes);
    free(t);
}

void fcl_table_row(FclTable *t, const char *const *cells)
{
    size_t *row;

    if (t == NULL || t->n_columns == 0) return;
    if (t->n_rows == t->rows_cap) {
        size_t cap = t->rows_cap == 0 ? 8 : t->rows_cap * 2;
        size_t **n = realloc(t->cells, cap * sizeof(*n));

        if (n == NULL) return;
        t->cells = (char ***)n;
        t->rows_cap = cap;
    }
    t->cells[t->n_rows] = calloc(t->n_columns, sizeof(char *));
    if (t->cells[t->n_rows] == NULL) return;
    for (size_t c = 0; c < t->n_columns; c++) {
        t->cells[t->n_rows][c] = strdup(cells[c] != NULL ? cells[c] : "");
    }
    t->n_rows++;
    (void)row;
}

void fcl_table_note(FclTable *t, const char *fmt, ...)
{
    va_list ap;
    char line[512];
    size_t len, add;

    if (t == NULL) return;
    va_start(ap, fmt);
    vsnprintf(line, sizeof(line), fmt, ap);
    va_end(ap);
    len = t->notes != NULL ? strlen(t->notes) : 0;
    add = strlen(line);
    {
        char *n = realloc(t->notes, len + add + 2);

        if (n == NULL) return;
        t->notes = n;
        memcpy(t->notes + len, line, add);
        t->notes[len + add] = '\n';
        t->notes[len + add + 1] = '\0';
    }
}

/* ------------------------------------------------------------------ */
/* blocks                                                             */
/* ------------------------------------------------------------------ */

static size_t block_count_for(const FcDatasetList *list)
{
    size_t n = 0;

    for (size_t d = 0; d < list->count; d++) {
        const FcDataset *ds = &list->items[d];

        n += fcd_count_rounds(ds) + fcd_count_subjects(ds);
    }
    return n;
}

const char *fcl_scope_name(FclScope scope)
{
    return scope == FCL_SCOPE_MERGED ? "merged (all datasets in one list)"
                                     : "per dataset (session rounds and "
                                       "subjects)";
}

/* One block containing every measurement of every dataset. */
static FclBlock *blocks_merged(const FcDatasetList *list, size_t *n_blocks_out)
{
    size_t total = fcd_list_count_measurements(list);
    FclBlock *blocks;
    double *values;
    size_t k = 0;

    if (n_blocks_out != NULL) *n_blocks_out = 0;
    if (list == NULL || list->count == 0 || total == 0) return NULL;
    blocks = calloc(2, sizeof(*blocks));
    values = malloc(total * sizeof(*values));
    if (blocks == NULL || values == NULL) {
        free(blocks);
        free(values);
        return NULL;
    }
    for (size_t d = 0; d < list->count; d++) {
        const FcDataset *ds = &list->items[d];

        for (size_t s = 0; s < ds->n_sessions; s++) {
            const FcSession *ss = &ds->sessions[s];

            for (size_t r = 0; r < ss->n_rounds; r++) {
                const FcRound *rr = &ss->rounds[r];

                for (size_t j = 0; j < rr->n_subjects; j++) {
                    const FcSubject *sj = &rr->subjects[j];

                    memcpy(values + k, sj->measurements,
                           sj->count * sizeof(*values));
                    k += sj->count;
                }
            }
        }
        if (k < total && ds->n_unassigned > 0) {
            memcpy(values + k, ds->unassigned,
                   ds->n_unassigned * sizeof(*values));
            k += ds->n_unassigned;
        }
    }
    if (k == 0) {
        free(blocks);
        free(values);
        return NULL;
    }
    snprintf(blocks[0].label, sizeof(blocks[0].label),
             "merged: all %zu dataset(s)", list->count);
    snprintf(blocks[0].dataset, sizeof(blocks[0].dataset), "%s", "merged");
    blocks[0].values = values;
    blocks[0].owns_values = true;
    blocks[0].n = k;
    blocks[0].min_hz = values[0];
    blocks[0].max_hz = values[0];
    for (size_t i = 1; i < k; i++) {
        if (values[i] < blocks[0].min_hz) blocks[0].min_hz = values[i];
        if (values[i] > blocks[0].max_hz) blocks[0].max_hz = values[i];
    }
    if (n_blocks_out != NULL) *n_blocks_out = 1;
    return blocks;
}

FclBlock *fcl_blocks_build(const FcDatasetList *list, FclScope scope,
                           size_t *n_blocks_out)
{
    if (scope == FCL_SCOPE_MERGED) return blocks_merged(list, n_blocks_out);
    return fcl_blocks_from_datasets(list, n_blocks_out);
}

FclBlock *fcl_blocks_from_datasets(const FcDatasetList *list,
                                   size_t *n_blocks_out)
{
    size_t n_blocks = list != NULL ? block_count_for(list) : 0;
    FclBlock *blocks;
    size_t used = 0;

    if (n_blocks_out != NULL) *n_blocks_out = 0;
    if (n_blocks == 0) return NULL;
    /* one extra zeroed entry acts as a terminator for the free loop */
    blocks = calloc(n_blocks + 1, sizeof(*blocks));
    if (blocks == NULL) return NULL;
    for (size_t d = 0; d < list->count; d++) {
        const FcDataset *ds = &list->items[d];

        for (size_t s = 0; s < ds->n_sessions; s++) {
            const FcSession *ss = &ds->sessions[s];

            for (size_t r = 0; r < ss->n_rounds; r++) {
                const FcRound *rr = &ss->rounds[r];
                size_t total = 0;
                double *vals;
                size_t k = 0;

                for (size_t j = 0; j < rr->n_subjects; j++) {
                    total += rr->subjects[j].count;
                }
                if (total == 0) continue;
                vals = malloc(total * sizeof(*vals));
                if (vals == NULL) continue;
                for (size_t j = 0; j < rr->n_subjects; j++) {
                    memcpy(vals + k, rr->subjects[j].measurements,
                           rr->subjects[j].count * sizeof(*vals));
                    k += rr->subjects[j].count;
                }
                blocks[used].label[0] = '\0';
                snprintf(blocks[used].label, sizeof(blocks[used].label),
                         "%s / session %s / round %s",
                         ds->title != NULL ? ds->title : ds->dataset_id,
                         ss->session_id.text, rr->round_id.text);
                snprintf(blocks[used].dataset, sizeof(blocks[used].dataset),
                         "%s", ds->title != NULL ? ds->title : ds->dataset_id);
                blocks[used].values = vals;
                blocks[used].owns_values = true;
                blocks[used].n = total;
                blocks[used].min_hz = vals[0];
                blocks[used].max_hz = vals[0];
                for (size_t i = 1; i < total; i++) {
                    if (vals[i] < blocks[used].min_hz) {
                        blocks[used].min_hz = vals[i];
                    }
                    if (vals[i] > blocks[used].max_hz) {
                        blocks[used].max_hz = vals[i];
                    }
                }
                used++;
            }
        }
        for (size_t s = 0; s < ds->n_sessions; s++) {
            const FcSession *ss = &ds->sessions[s];

            for (size_t r = 0; r < ss->n_rounds; r++) {
                const FcRound *rr = &ss->rounds[r];

                for (size_t j = 0; j < rr->n_subjects; j++) {
                    const FcSubject *sj = &rr->subjects[j];

                    if (sj->count == 0) continue;
                    blocks[used].label[0] = '\0';
                    snprintf(blocks[used].label, sizeof(blocks[used].label),
                             "%s / session %s / round %s / subject %s",
                             ds->title != NULL ? ds->title : ds->dataset_id,
                             ss->session_id.text, rr->round_id.text, sj->name);
                    snprintf(blocks[used].dataset,
                             sizeof(blocks[used].dataset), "%s",
                             ds->title != NULL ? ds->title : ds->dataset_id);
                    blocks[used].values = sj->measurements;
                    blocks[used].owns_values = false;
                    blocks[used].n = sj->count;
                    blocks[used].min_hz = sj->measurements[0];
                    blocks[used].max_hz = sj->measurements[0];
                    for (size_t i = 1; i < sj->count; i++) {
                        if (sj->measurements[i] < blocks[used].min_hz) {
                            blocks[used].min_hz = sj->measurements[i];
                        }
                        if (sj->measurements[i] > blocks[used].max_hz) {
                            blocks[used].max_hz = sj->measurements[i];
                        }
                    }
                    used++;
                }
            }
        }
    }
    if (n_blocks_out != NULL) *n_blocks_out = used;
    return blocks;
}

void fcl_blocks_free(FclBlock *blocks)
{
    if (blocks == NULL) return;
    /* combined round arrays were allocated by the builder; subject blocks
     * only borrow pointers into the datasets */
    for (size_t i = 0; blocks[i].n > 0 || blocks[i].values != NULL; i++) {
        if (blocks[i].owns_values) free((void *)blocks[i].values);
    }
    free(blocks);
}

/* ------------------------------------------------------------------ */
/* circular tests                                                     */
/* ------------------------------------------------------------------ */

double fcl_fold(double hz, double f0_hz, double df_hz)
{
    double x;

    if (df_hz <= 0.0) return 0.0;
    x = (hz - f0_hz) / df_hz;
    x -= floor(x);
    return x;
}

void fcl_circular_stats(const double *values, size_t n, double f0_hz,
                        double df_hz, double *mean_angle, double *R)
{
    double c = 0.0, s = 0.0;

    if (n == 0) {
        if (mean_angle != NULL) *mean_angle = 0.0;
        if (R != NULL) *R = 0.0;
        return;
    }
    for (size_t i = 0; i < n; i++) {
        double a = 2.0 * M_PI * fcl_fold(values[i], f0_hz, df_hz);

        c += cos(a);
        s += sin(a);
    }
    c /= (double)n;
    s /= (double)n;
    if (mean_angle != NULL) *mean_angle = atan2(s, c);
    if (R != NULL) *R = hypot(c, s);
}

double fcl_rayleigh_p(size_t n, double R)
{
    double z;

    if (n == 0) return 1.0;
    z = (double)n * R * R;
    return exp(-z);
}

double fcl_v_statistic(size_t n, double R, double mean_angle, double direction)
{
    if (n == 0) return 0.0;
    return R * cos(mean_angle - direction);
}

static double rayleigh_z(const double *values, size_t n, double f0_hz,
                         double df_hz)
{
    double R = 0.0;

    fcl_circular_stats(values, n, f0_hz, df_hz, NULL, &R);
    return (double)n * R * R;
}

double fcl_perm_rayleigh(const double *values, size_t n, double f0_hz,
                         double df_hz, double min_hz, double max_hz,
                         size_t replicates, FclRng *rng)
{
    double observed, hits = 0.0;
    double width = max_hz - min_hz;
    double *sample;

    if (n == 0 || replicates == 0) return 1.0;
    if (width <= 0.0) return 1.0;
    sample = malloc(n * sizeof(*sample));
    if (sample == NULL) return 1.0;
    observed = rayleigh_z(values, n, f0_hz, df_hz);
    for (size_t r = 0; r < replicates; r++) {
        for (size_t i = 0; i < n; i++) {
            sample[i] = min_hz + width * fcl_rng_uniform(rng);
        }
        if (rayleigh_z(sample, n, f0_hz, df_hz) >= observed) hits += 1.0;
    }
    free(sample);
    return (hits + 1.0) / ((double)replicates + 1.0);
}

double fcl_perm_v(const double *values, size_t n, double f0_hz, double df_hz,
                  double min_hz, double max_hz, double direction,
                  size_t replicates, FclRng *rng)
{
    double observed, hits = 0.0;
    double width = max_hz - min_hz;
    double *sample;
    double R = 0.0, ang = 0.0;

    if (n == 0 || replicates == 0 || width <= 0.0) return 1.0;
    sample = malloc(n * sizeof(*sample));
    if (sample == NULL) return 1.0;
    fcl_circular_stats(values, n, f0_hz, df_hz, &ang, &R);
    observed = fcl_v_statistic(n, R, ang, direction);
    for (size_t r = 0; r < replicates; r++) {
        for (size_t i = 0; i < n; i++) {
            sample[i] = min_hz + width * fcl_rng_uniform(rng);
        }
        fcl_circular_stats(sample, n, f0_hz, df_hz, &ang, &R);
        if (fcl_v_statistic(n, R, ang, direction) >= observed) hits += 1.0;
    }
    free(sample);
    return (hits + 1.0) / ((double)replicates + 1.0);
}

/* Phase-scatter statistic: the circular variance 1 - R of the folded
 * phases. Small values mean the phases cluster (periodic structure at the
 * candidate spacing), ~1 means no periodicity. `bins` is accepted for API
 * symmetry with the literature but the statistic is bin-free (exactly the
 * phase dispersion), which avoids bin-width artefacts for small n. */
double fcl_pdm(const double *values, size_t n, double f0_hz, double df_hz,
               size_t bins)
{
    double R = 0.0;

    (void)bins;
    if (n < 3) return 1.0;
    fcl_circular_stats(values, n, f0_hz, df_hz, NULL, &R);
    if (R > 1.0) R = 1.0;
    return 1.0 - R;
}

double fcl_perm_pdm(const double *values, size_t n, double f0_hz,
                    double df_hz, double min_hz, double max_hz,
                    size_t bins, size_t replicates, FclRng *rng)
{
    double observed, hits = 0.0, width = max_hz - min_hz;
    double *sample;

    if (n < 3 || replicates == 0 || width <= 0.0) return 1.0;
    sample = malloc(n * sizeof(*sample));
    if (sample == NULL) return 1.0;
    observed = fcl_pdm(values, n, f0_hz, df_hz, bins);
    for (size_t r = 0; r < replicates; r++) {
        for (size_t i = 0; i < n; i++) {
            sample[i] = min_hz + width * fcl_rng_uniform(rng);
        }
        if (fcl_pdm(sample, n, f0_hz, df_hz, bins) <= observed) hits += 1.0;
    }
    free(sample);
    return (hits + 1.0) / ((double)replicates + 1.0);
}

/* ------------------------------------------------------------------ */
/* ASCII plots                                                        */
/* ------------------------------------------------------------------ */

/* Small, bounded line buffer for the ASCII plots (plots are tiny). */
typedef struct {
    char   text[8192];
    size_t len;
} PlotBuf;

static void plot_init(PlotBuf *pb)
{
    pb->text[0] = '\0';
    pb->len = 0;
}

static void plot_raw(PlotBuf *pb, const char *line)
{
    size_t add = strlen(line);

    if (pb->len + add + 2 > sizeof(pb->text)) return;
    memcpy(pb->text + pb->len, line, add);
    pb->len += add;
    pb->text[pb->len++] = '\n';
    pb->text[pb->len] = '\0';
}

static void plot_line(PlotBuf *pb, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));

static void plot_line(PlotBuf *pb, const char *fmt, ...)
{
    char line[2048];
    va_list ap;

    va_start(ap, fmt);
    vsnprintf(line, sizeof(line), fmt, ap);
    va_end(ap);
    plot_raw(pb, line);
}

char *fcl_rose_plot(const double *values, size_t n, double f0_hz,
                    double df_hz, int n_bins, int height)
{
    size_t *count;
    PlotBuf pb;
    double max_count = 1.0;

    if (n == 0 || n_bins < 4 || height < 1) return NULL;
    count = calloc((size_t)n_bins, sizeof(*count));
    if (count == NULL) return NULL;
    for (size_t i = 0; i < n; i++) {
        size_t b = (size_t)(fcl_fold(values[i], f0_hz, df_hz) *
                            (double)n_bins);

        if (b >= (size_t)n_bins) b = (size_t)n_bins - 1;
        count[b]++;
    }
    for (int b = 0; b < n_bins; b++) {
        if ((double)count[b] > max_count) max_count = (double)count[b];
    }
    plot_init(&pb);
    plot_line(&pb, "phase rose (0 deg = lattice), %d bins, %d samples:",
              n_bins, (int)n);
    for (int row = height; row >= 1; row--) {
        char line[2048];
        size_t pos = 0;

        line[pos++] = '|';
        line[pos++] = ' ';
        for (int b = 0; b < n_bins; b++) {
            double frac = (double)count[b] / max_count;
            int level = (int)(frac * (double)height + 0.5);

            line[pos++] = level >= row ? '#' : ' ';
            line[pos++] = level >= row ? '#' : ' ';
        }
        line[pos] = '\0';
        plot_raw(&pb, line);
    }
    {
        char line[2048];
        size_t pos = 0;

        line[pos++] = '+';
        for (int b = 0; b < n_bins * 2; b++) line[pos++] = '-';
        line[pos] = '\0';
        plot_raw(&pb, line);
        plot_raw(&pb, "   0 deg                     180 deg");
    }
    free(count);
    return strdup(pb.text);
}

char *fcl_comb_plot(const double *values, size_t n, double f0_hz,
                    double df_hz, int n_bins, int height)
{
    size_t *count;
    PlotBuf pb;
    double max_count = 1.0;

    if (n == 0 || n_bins < 4 || height < 1) return NULL;
    count = calloc((size_t)n_bins, sizeof(*count));
    if (count == NULL) return NULL;
    for (size_t i = 0; i < n; i++) {
        size_t b = (size_t)(fcl_fold(values[i], f0_hz, df_hz) *
                            (double)n_bins);

        if (b >= (size_t)n_bins) b = (size_t)n_bins - 1;
        count[b]++;
    }
    for (int b = 0; b < n_bins; b++) {
        if ((double)count[b] > max_count) max_count = (double)count[b];
    }
    plot_init(&pb);
    plot_line(&pb, "lattice comb (phase between subcarriers), df = %.0f Hz:",
              df_hz);
    for (int row = height; row >= 1; row--) {
        char line[2048];
        size_t pos = 0;

        line[pos++] = '|';
        for (int b = 0; b < n_bins; b++) {
            double frac = (double)count[b] / max_count;
            int level = (int)(frac * (double)height + 0.5);

            line[pos++] = level >= row ? '*' : ' ';
        }
        line[pos] = '\0';
        plot_raw(&pb, line);
    }
    plot_raw(&pb, "   0 deg                     360 deg");
    free(count);
    return strdup(pb.text);
}

/* ------------------------------------------------------------------ */
/* chi-square / gamma                                                 */
/* ------------------------------------------------------------------ */

/* Regularised lower incomplete gamma P(a,x) (series) and Q via continued
 * fraction (Numerical Recipes); accuracy ~1e-10 for the ranges used. */
static double gamma_p_series(double a, double x)
{
    double sum = 1.0 / a, term = sum, ap = a;

    for (int n = 0; n < 500; n++) {
        ap += 1.0;
        term *= x / ap;
        sum += term;
        if (fabs(term) < fabs(sum) * 1e-14) break;
    }
    return sum * exp(-x + a * log(x) - lgamma(a));
}

static double gamma_q_cf(double a, double x)
{
    double b = x + 1.0 - a, c = 1e300, d = 1.0 / b, h = d;

    for (int i = 1; i <= 500; i++) {
        double an = -(double)i * ((double)i - a);

        b += 2.0;
        d = an * d + b;
        if (fabs(d) < 1e-300) d = 1e-300;
        c = b + an / c;
        if (fabs(c) < 1e-300) c = 1e-300;
        d = 1.0 / d;
        {
            double del = d * c;

            h *= del;
            if (fabs(del - 1.0) < 1e-14) break;
        }
    }
    return exp(-x + a * log(x) - lgamma(a)) * h;
}

double fcl_gamma_q(double a, double x)
{
    if (x < 0.0 || a <= 0.0) return 1.0;
    if (x == 0.0) return 1.0;
    if (x < a + 1.0) return 1.0 - gamma_p_series(a, x);
    return gamma_q_cf(a, x);
}

double fcl_chi2_p(double chi2, size_t df)
{
    if (chi2 <= 0.0) return 1.0;
    return fcl_gamma_q((double)df / 2.0, chi2 / 2.0);
}

double fcl_chi2_lattice(const double *values, size_t n, double f0_hz,
                        double df_hz, double min_hz, double max_hz,
                        double frac, size_t *df_out, bool *valid)
{
    double width = max_hz - min_hz;
    double bin_w, expected, chi2 = 0.0;
    long k_min, k_max;
    size_t n_bins;

    if (valid != NULL) *valid = false;
    if (df_out != NULL) *df_out = 0;
    if (n < 5 || df_hz <= 0.0 || width <= 0.0) return 0.0;
    if (frac <= 0.0 || frac > 1.0) frac = 0.25;
    bin_w = frac * df_hz;
    if (bin_w <= 0.0) return 0.0;
    k_min = (long)ceil((min_hz - f0_hz) / df_hz);
    k_max = (long)floor((max_hz - f0_hz) / df_hz);
    n_bins = (size_t)(k_max - k_min + 1);
    if (n_bins < 2) return 0.0;
    expected = (double)n * bin_w / width;
    if (expected < 5.0) {
        /* too few expected counts per bin: fall back to a coarser bin width
         * and require at least 5 expected per bin */
        return 0.0;
    }
    for (long k = k_min; k <= k_max; k++) {
        double centre = f0_hz + (double)k * df_hz;
        size_t obs = 0;

        for (size_t i = 0; i < n; i++) {
            if (fabs(values[i] - centre) <= bin_w / 2.0) obs++;
        }
        {
            double z = ((double)obs - expected) / sqrt(expected);

            chi2 += z * z;
        }
    }
    if (df_out != NULL) *df_out = n_bins > 1 ? n_bins - 1 : 1;
    if (valid != NULL) *valid = true;
    return chi2;
}

/* ------------------------------------------------------------------ */
/* KDE                                                                */
/* ------------------------------------------------------------------ */

static double kde_bandwidth_max(const double *values, size_t n, double max_h)
{
    double mean = 0.0, sd = 0.0, h;

    if (n < 2) return 1.0;
    for (size_t i = 0; i < n; i++) mean += values[i];
    mean /= (double)n;
    for (size_t i = 0; i < n; i++) {
        double d = values[i] - mean;

        sd += d * d;
    }
    sd = sqrt(sd / (double)(n - 1));
    if (sd <= 0.0) sd = 1.0;
    h = 0.9 * sd * pow((double)n, -0.2);
    if (max_h > 0.0 && h > max_h) h = max_h;
    return h > 0.0 ? h : sd * 0.1;
}

static double kde_at(const double *values, size_t n, double h, double x)
{
    double sum = 0.0, inv = 1.0 / (h * sqrt(2.0 * M_PI));

    for (size_t i = 0; i < n; i++) {
        double z = (x - values[i]) / h;

        sum += exp(-0.5 * z * z);
    }
    return sum * inv / (double)n;
}

double fcl_kde_enrichment(const double *values, size_t n, double f0_hz,
                          double df_hz, double min_hz, double max_hz)
{
    double width = max_hz - min_hz, h, lat_sum = 0.0, mean_density = 0.0;
    long k_min, k_max;
    size_t n_lat, grid = 64;

    if (n < 3 || df_hz <= 0.0 || width <= 0.0) return 1.0;
    h = kde_bandwidth_max(values, n, df_hz / 4.0);
    k_min = (long)ceil((min_hz - f0_hz) / df_hz);
    k_max = (long)floor((max_hz - f0_hz) / df_hz);
    n_lat = (size_t)(k_max - k_min + 1);
    if (n_lat == 0) return 1.0;
    for (long k = k_min; k <= k_max; k++) {
        lat_sum += kde_at(values, n, h, f0_hz + (double)k * df_hz);
    }
    for (size_t g = 0; g < grid; g++) {
        double x = min_hz + width * ((double)g + 0.5) / (double)grid;

        mean_density += kde_at(values, n, h, x);
    }
    mean_density /= (double)grid;
    if (mean_density <= 1e-300) return 1.0;
    return (lat_sum / (double)n_lat) / mean_density;
}

double fcl_perm_enrichment(const double *values, size_t n, double f0_hz,
                           double df_hz, double min_hz, double max_hz,
                           size_t replicates, FclRng *rng)
{
    double observed, hits = 0.0, width = max_hz - min_hz;
    double *sample;
    double h = kde_bandwidth_max(values, n, df_hz / 4.0);
    long k_min, k_max;
    size_t grid = 64;

    if (n < 3 || replicates == 0 || width <= 0.0) return 1.0;
    sample = malloc(n * sizeof(*sample));
    if (sample == NULL) return 1.0;
    observed = fcl_kde_enrichment(values, n, f0_hz, df_hz, min_hz, max_hz);
    k_min = (long)ceil((min_hz - f0_hz) / df_hz);
    k_max = (long)floor((max_hz - f0_hz) / df_hz);
    for (size_t r = 0; r < replicates; r++) {
        double lat_sum = 0.0, mean_density = 0.0, enrich;
        size_t n_lat = (size_t)(k_max - k_min + 1);

        for (size_t i = 0; i < n; i++) {
            sample[i] = min_hz + width * fcl_rng_uniform(rng);
        }
        for (long k = k_min; k <= k_max; k++) {
            lat_sum += kde_at(sample, n, h, f0_hz + (double)k * df_hz);
        }
        for (size_t g = 0; g < grid; g++) {
            double x = min_hz + width * ((double)g + 0.5) / (double)grid;

            mean_density += kde_at(sample, n, h, x);
        }
        mean_density /= (double)grid;
        if (mean_density <= 1e-300) n_lat = 1;
        enrich = n_lat > 0 && mean_density > 0.0
                     ? (lat_sum / (double)n_lat) / mean_density
                     : 1.0;
        if (enrich >= observed) hits += 1.0;
    }
    free(sample);
    return (hits + 1.0) / ((double)replicates + 1.0);
}

/* ------------------------------------------------------------------ */
/* cluster test                                                       */
/* ------------------------------------------------------------------ */

void fcl_cluster_test(const double *values, size_t n, double min_hz,
                      double max_hz, size_t bins, size_t replicates,
                      FclRng *rng, double *max_mass_out, size_t *extent_out,
                      double *p_out)
{
    double width = max_hz - min_hz;
    double *counts, *null_mass;
    double *sample;
    size_t max_extent = 0;
    double max_mass = 0.0, hits = 0.0;
    double expected;

    if (max_mass_out != NULL) *max_mass_out = 0.0;
    if (extent_out != NULL) *extent_out = 0;
    if (p_out != NULL) *p_out = 1.0;
    if (n < 5 || bins < 4 || width <= 0.0 || replicates == 0) return;
    counts = calloc(bins, sizeof(*counts));
    null_mass = calloc(replicates, sizeof(*null_mass));
    sample = malloc(n * sizeof(*sample));
    if (counts == NULL || null_mass == NULL || sample == NULL) {
        free(counts);
        free(null_mass);
        free(sample);
        return;
    }
    expected = (double)n / (double)bins;
    for (size_t i = 0; i < n; i++) {
        size_t b = (size_t)((values[i] - min_hz) / width * (double)bins);

        if (b >= bins) b = bins - 1;
        counts[b]++;
    }
    {
        double best = 0.0;
        size_t best_extent = 0;
        double run = 0.0;
        size_t run_extent = 0;

        for (size_t b = 0; b < bins; b++) {
            double excess = counts[b] - expected;

            if (excess > 0.0) {
                run += excess;
                run_extent++;
                if (run > best) {
                    best = run;
                    best_extent = run_extent;
                }
            } else {
                run = 0.0;
                run_extent = 0;
            }
        }
        max_mass = best;
        max_extent = best_extent;
        /* wrap-around cluster (cyclic in the block) */
        {
            size_t lead = 0, tail = 0;
            double sum = 0.0;

            while (lead < bins && counts[lead] > expected) {
                sum += counts[lead] - expected;
                lead++;
            }
            while (tail < bins - lead &&
                   counts[bins - 1 - tail] > expected) {
                sum += counts[bins - 1 - tail] - expected;
                tail++;
            }
            if (lead > 0 && tail > 0 && sum > max_mass) {
                max_mass = sum;
                max_extent = lead + tail;
            }
        }
    }
    for (size_t r = 0; r < replicates; r++) {
        double best = 0.0, run = 0.0;

        memset(counts, 0, bins * sizeof(*counts));
        for (size_t i = 0; i < n; i++) {
            sample[i] = min_hz + width * fcl_rng_uniform(rng);
        }
        for (size_t i = 0; i < n; i++) {
            size_t b = (size_t)((sample[i] - min_hz) / width * (double)bins);

            if (b >= bins) b = bins - 1;
            counts[b]++;
        }
        for (size_t b = 0; b < bins; b++) {
            double excess = counts[b] - expected;

            if (excess > 0.0) {
                run += excess;
                if (run > best) best = run;
            } else {
                run = 0.0;
            }
        }
        null_mass[r] = best;
        if (best >= max_mass) hits += 1.0;
    }
    if (max_mass_out != NULL) *max_mass_out = max_mass;
    if (extent_out != NULL) *extent_out = max_extent;
    if (p_out != NULL) *p_out = (hits + 1.0) / ((double)replicates + 1.0);
    free(counts);
    free(null_mass);
    free(sample);
}

/* ------------------------------------------------------------------ */
/* multimodality                                                      */
/* ------------------------------------------------------------------ */

static int count_modes(const double *values, size_t n, double h, size_t *best)
{
    /* evaluate the KDE on a grid of 256 points across the data range and
     * count local maxima above a small fraction of the peak */
    double lo = values[0], hi = values[0], peak = 0.0;
    double dens[256];
    int modes = 0;
    size_t peak_at = 0;

    for (size_t i = 1; i < n; i++) {
        if (values[i] < lo) lo = values[i];
        if (values[i] > hi) hi = values[i];
    }
    if (hi - lo <= 0.0) hi = lo + 1.0;
    for (int g = 0; g < 256; g++) {
        double x = lo + (hi - lo) * ((double)g + 0.5) / 256.0;

        dens[g] = kde_at(values, n, h, x);
        if (dens[g] > peak) {
            peak = dens[g];
            peak_at = (size_t)g;
        }
    }
    if (peak <= 0.0) return 1;
    for (int g = 1; g < 255; g++) {
        if (dens[g] > dens[g - 1] && dens[g] > dens[g + 1] &&
            dens[g] > peak * 0.05) {
            modes++;
        }
    }
    if (modes == 0) modes = 1;
    if (best != NULL) *best = peak_at;
    (void)lo;
    (void)hi;
    return modes;
}

/* Smallest bandwidth with at most k modes (bisection). */
static double critical_bandwidth(const double *values, size_t n, size_t k)
{
    double lo = values[0], hi = values[0];
    double h_lo, h_hi, mid;
    int modes;

    for (size_t i = 1; i < n; i++) {
        if (values[i] < lo) lo = values[i];
        if (values[i] > hi) hi = values[i];
    }
    if (hi - lo <= 0.0) return 1.0;
    h_lo = (hi - lo) * 1e-4;
    h_hi = (hi - lo) * 2.0;
    modes = count_modes(values, n, h_lo, NULL);
    if ((size_t)modes <= k) return h_lo;
    modes = count_modes(values, n, h_hi, NULL);
    if ((size_t)modes > k) return h_hi;
    for (int it = 0; it < 40; it++) {
        mid = 0.5 * (h_lo + h_hi);
        modes = count_modes(values, n, mid, NULL);
        if ((size_t)modes > k) h_lo = mid;
        else h_hi = mid;
    }
    return h_hi;
}

double fcl_silverman(const double *values, size_t n, size_t k_modes,
                     size_t replicates, FclRng *rng, size_t *n_modes_out)
{
    double h_crit, *copy, hits = 0.0;

    if (n_modes_out != NULL) *n_modes_out = 0;
    if (n < 8 || n > 400 || replicates == 0) return -1.0;
    copy = malloc(n * sizeof(*copy));
    if (copy == NULL) return -1.0;
    memcpy(copy, values, n * sizeof(*copy));
    h_crit = critical_bandwidth(values, n, k_modes);
    if (n_modes_out != NULL) {
        int m = count_modes(values, n, h_crit * 0.999, n_modes_out);

        *n_modes_out = (size_t)m;
    }
    {
        double *sample = malloc(n * sizeof(*sample));

        if (sample == NULL) {
            free(copy);
            return -1.0;
        }
        for (size_t r = 0; r < replicates; r++) {
            double crit;

            for (size_t i = 0; i < n; i++) {
                size_t pick = (size_t)(fcl_rng_uniform(rng) * (double)n);

                if (pick >= n) pick = n - 1;
                sample[i] = copy[pick] + h_crit * fcl_rng_normal(rng);
            }
            crit = critical_bandwidth(sample, n, k_modes);
            if (crit <= h_crit) hits += 1.0;
        }
        free(sample);
    }
    free(copy);
    return (hits + 1.0) / ((double)replicates + 1.0);
}

/* ------------------------------------------------------------------ */
/* multiple testing                                                   */
/* ------------------------------------------------------------------ */

typedef struct { double p; size_t idx; } BhEnt;

static int cmp_bh(const void *a, const void *b)
{
    const BhEnt *x = (const BhEnt *)a, *y = (const BhEnt *)b;

    return x->p < y->p ? -1 : x->p > y->p ? 1 : 0;
}

size_t fcl_bh_adjust(double *p, size_t n, double alpha)
{
    BhEnt *ents;
    double *adj;
    size_t rejections = 0;

    if (n == 0) return 0;
    ents = malloc(n * sizeof(*ents));
    adj = malloc(n * sizeof(*adj));
    if (ents == NULL || adj == NULL) {
        free(ents);
        free(adj);
        return 0;
    }
    for (size_t i = 0; i < n; i++) {
        ents[i].p = p[i];
        ents[i].idx = i;
    }
    qsort(ents, n, sizeof(*ents), cmp_bh);
    /* raw BH values, then enforce monotonicity from the largest p down */
    for (size_t i = 0; i < n; i++) {
        adj[i] = ents[i].p * (double)n / (double)(i + 1);
        if (adj[i] > 1.0) adj[i] = 1.0;
    }
    for (size_t i = n; i > 1; i--) {
        if (adj[i - 2] > adj[i - 1]) adj[i - 2] = adj[i - 1];
    }
    for (size_t i = 0; i < n; i++) {
        p[ents[i].idx] = adj[i];
        if (adj[i] <= alpha) rejections++;
    }
    free(ents);
    free(adj);
    return rejections;
}

double fcl_bonferroni(double p, size_t n)
{
    double adj = p * (double)(n > 0 ? n : 1);

    return adj > 1.0 ? 1.0 : adj;
}
