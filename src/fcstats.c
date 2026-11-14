/*
 * fcstats.c — advanced statistics and population estimates for the
 * Frequency Counter tab.
 *
 * The report is deliberately self-contained plain text so it can be shown
 * in the Statistics tab and saved like the analysis report. Bootstrap
 * resampling uses the deterministic FclRng from fclattice.c, so repeated
 * runs on the same data give identical numbers.
 */
#include "fcstats.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    double *v;
    size_t  n;
} Cluster;

static double *copy_values(const FcDatasetList *list, size_t *n_out,
                           Cluster **clusters_out, size_t *n_clusters_out,
                           size_t *rounds_out)
{
    size_t n = 0, n_lists = 0, n_rounds = 0;
    double *v;
    Cluster *cl;
    size_t used = 0, used_lists = 0;

    for (size_t d = 0; d < list->count; d++) {
        const FcDataset *ds = &list->items[d];

        for (size_t s = 0; s < ds->n_sessions; s++) {
            const FcSession *ss = &ds->sessions[s];

            for (size_t r = 0; r < ss->n_rounds; r++) {
                const FcRound *rr = &ss->rounds[r];

                n_rounds++;
                for (size_t j = 0; j < rr->n_subjects; j++) {
                    n += rr->subjects[j].count;
                    n_lists++;
                }
            }
        }
        n += ds->n_unassigned;
        if (ds->n_unassigned > 0) n_lists++;
    }
    if (n == 0) return NULL;
    v = malloc(n * sizeof(*v));
    cl = calloc(n_lists, sizeof(*cl));
    if (v == NULL || cl == NULL) {
        free(v);
        free(cl);
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

                    if (sj->count > 0) {
                        cl[used_lists].v = sj->measurements;
                        cl[used_lists].n = sj->count;
                        memcpy(v + used, sj->measurements,
                               sj->count * sizeof(*v));
                        used += sj->count;
                        used_lists++;
                    }
                }
            }
        }
        if (ds->n_unassigned > 0) {
            cl[used_lists].v = ds->unassigned;
            cl[used_lists].n = ds->n_unassigned;
            memcpy(v + used, ds->unassigned, ds->n_unassigned * sizeof(*v));
            used += ds->n_unassigned;
            used_lists++;
        }
    }
    *n_out = n;
    *clusters_out = cl;
    *n_clusters_out = used_lists;
    *rounds_out = n_rounds;
    return v;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a;
    double y = *(const double *)b;

    return (x < y) ? -1 : (x > y) ? 1 : 0;
}

static double mean_of(const double *v, size_t n)
{
    double s = 0.0;

    for (size_t i = 0; i < n; i++) s += v[i];
    return n > 0 ? s / (double)n : 0.0;
}

/* sample variance (n-1) */
static double var_of(const double *v, size_t n, double mean)
{
    double s = 0.0;

    if (n < 2) return 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = v[i] - mean;

        s += d * d;
    }
    return s / (double)(n - 1);
}

static double quantile_sorted(const double *s, size_t n, double p)
{
    double h;
    size_t lo;

    if (n == 0) return 0.0;
    if (n == 1) return s[0];
    if (p <= 0.0) return s[0];
    if (p >= 1.0) return s[n - 1];
    h = (double)(n - 1) * p;
    lo = (size_t)floor(h);
    if (lo + 1 >= n) return s[n - 1];
    return s[lo] + (h - (double)lo) * (s[lo + 1] - s[lo]);
}

static double median_sorted(const double *s, size_t n)
{
    return quantile_sorted(s, n, 0.5);
}

static double trimmed_mean(const double *sorted, size_t n, double trim)
{
    size_t k = (size_t)floor((double)n * trim);
    size_t lo = k, hi = n - k;
    double s = 0.0;

    if (hi <= lo) {
        lo = 0;
        hi = n;
    }
    for (size_t i = lo; i < hi; i++) s += sorted[i];
    return s / (double)(hi - lo);
}

static double mad_sorted(const double *s, size_t n, double med)
{
    double *dev = malloc(n * sizeof(*dev));
    double out;

    if (dev == NULL) return 0.0;
    for (size_t i = 0; i < n; i++) dev[i] = fabs(s[i] - med);
    qsort(dev, n, sizeof(*dev), cmp_double);
    out = median_sorted(dev, n);
    free(dev);
    return out;
}

static double skewness(const double *v, size_t n, double mean, double sd)
{
    double s = 0.0;
    double g1;

    if (n < 3 || sd <= 0.0) return 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = (v[i] - mean) / sd;

        s += d * d * d;
    }
    g1 = s / (double)n;
    return g1 * sqrt((double)n * (double)(n - 1)) / (double)(n - 2);
}

static double kurtosis_excess(const double *v, size_t n, double mean,
                              double sd)
{
    double s = 0.0;
    double g2;

    if (n < 4 || sd <= 0.0) return 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = (v[i] - mean) / sd;

        s += d * d * d * d;
    }
    g2 = s / (double)n - 3.0;
    return ((double)(n - 1) / (double)((n - 2) * (n - 3))) *
           ((double)(n + 1) * g2 + 6.0);
}

/* Bootstrap percentile interval for a statistic over plain values. */
typedef double (*StatFn)(const double *v, size_t n);

static double stat_mean(const double *v, size_t n) { return mean_of(v, n); }

static double stat_median(const double *v, size_t n)
{
    double *s = malloc(n * sizeof(*s));
    double out;

    if (s == NULL) return 0.0;
    memcpy(s, v, n * sizeof(*s));
    qsort(s, n, sizeof(*s), cmp_double);
    out = median_sorted(s, n);
    free(s);
    return out;
}

static double stat_sd(const double *v, size_t n)
{
    return sqrt(var_of(v, n, mean_of(v, n)));
}

static double stat_trimmed(const double *v, size_t n)
{
    double *s = malloc(n * sizeof(*s));
    double out;

    if (s == NULL) return 0.0;
    memcpy(s, v, n * sizeof(*s));
    qsort(s, n, sizeof(*s), cmp_double);
    out = trimmed_mean(s, n, 0.1);
    free(s);
    return out;
}

static void bootstrap_ci(const double *v, size_t n, StatFn fn,
                         size_t reps, double conf, FclRng *rng, double *lo,
                         double *hi)
{
    double *stats;
    double *sample;

    if (n == 0 || reps == 0) {
        *lo = *hi = 0.0;
        return;
    }
    stats = malloc(reps * sizeof(*stats));
    sample = malloc(n * sizeof(*sample));
    if (stats == NULL || sample == NULL) {
        free(stats);
        free(sample);
        *lo = *hi = fn(v, n);
        return;
    }
    for (size_t b = 0; b < reps; b++) {
        for (size_t i = 0; i < n; i++) {
            size_t k = (size_t)(fcl_rng_uniform(rng) * (double)n);

            if (k >= n) k = n - 1;
            sample[i] = v[k];
        }
        stats[b] = fn(sample, n);
    }
    qsort(stats, reps, sizeof(*stats), cmp_double);
    *lo = quantile_sorted(stats, reps, (1.0 - conf) / 2.0);
    *hi = quantile_sorted(stats, reps, 1.0 - (1.0 - conf) / 2.0);
    free(stats);
    free(sample);
}

/* Cluster bootstrap: resample whole frequency lists (clusters). */
static void cluster_bootstrap_mean(const Cluster *cl, size_t k, size_t reps,
                                   double conf, FclRng *rng, double *lo,
                                   double *hi)
{
    double *stats = malloc(reps * sizeof(*stats));
    double *picked;
    size_t max_n = 0;

    if (stats == NULL || k == 0) {
        free(stats);
        *lo = *hi = 0.0;
        return;
    }
    for (size_t i = 0; i < k; i++) {
        if (cl[i].n > max_n) max_n = cl[i].n;
    }
    /* a resample draws k clusters with replacement, so the total can be up
     * to k times the largest cluster */
    picked = malloc(k * max_n * sizeof(*picked));
    if (picked == NULL) {
        free(stats);
        *lo = *hi = 0.0;
        return;
    }
    for (size_t b = 0; b < reps; b++) {
        size_t used = 0;

        for (size_t i = 0; i < k; i++) {
            size_t pick = (size_t)(fcl_rng_uniform(rng) * (double)k);

            if (pick >= k) pick = k - 1;
            memcpy(picked + used, cl[pick].v,
                   cl[pick].n * sizeof(*picked));
            used += cl[pick].n;
        }
        stats[b] = mean_of(picked, used);
    }
    qsort(stats, reps, sizeof(*stats), cmp_double);
    *lo = quantile_sorted(stats, reps, (1.0 - conf) / 2.0);
    *hi = quantile_sorted(stats, reps, 1.0 - (1.0 - conf) / 2.0);
    free(stats);
    free(picked);
}

/* Two-level variance decomposition (frequency lists within rounds). */
static void variance_components(const Cluster *cl, size_t k, size_t n,
                                double grand, double *sigma_b2,
                                double *sigma_w2, double *icc, double *deff,
                                double *n_eff)
{
    double within_ss = 0.0, between_ss = 0.0, n0;
    double sum_n2 = 0.0;
    size_t df_w = n > k ? n - k : 0;
    double msb, msw;
    double n_bar = k > 0 ? (double)n / (double)k : 0.0;

    for (size_t i = 0; i < k; i++) {
        double m = mean_of(cl[i].v, cl[i].n);

        for (size_t j = 0; j < cl[i].n; j++) {
            double d = cl[i].v[j] - m;

            within_ss += d * d;
        }
        between_ss += (double)cl[i].n * (m - grand) * (m - grand);
        sum_n2 += (double)cl[i].n * (double)cl[i].n;
    }
    msw = df_w > 0 ? within_ss / (double)df_w : 0.0;
    msb = k > 1 ? between_ss / (double)(k - 1) : 0.0;
    n0 = (k > 1 && n > 0)
             ? (double)n - sum_n2 / (double)n : 0.0;
    n0 = k > 1 ? n0 / (double)(k - 1) : 0.0;
    if (n0 < 1e-9) n0 = n_bar > 0.0 ? n_bar : 1.0;
    *sigma_w2 = msw;
    *sigma_b2 = msb > msw ? (msb - msw) / n0 : 0.0;
    {
        double total = *sigma_b2 + *sigma_w2;

        *icc = total > 0.0 ? *sigma_b2 / total : 0.0;
    }
    *deff = 1.0 + (n_bar > 1.0 ? (n_bar - 1.0) * *icc : 0.0);
    if (*deff < 1.0) *deff = 1.0;
    *n_eff = *deff > 0.0 ? (double)n / *deff : (double)n;
}

/* Allan deviation of the round means (ordered as stored). */
static double allan_deviation(const double *means, size_t k)
{
    double sum = 0.0;
    size_t terms = 0;

    for (size_t i = 0; i + 1 < k; i++) {
        double d = means[i + 1] - means[i];

        sum += d * d;
        terms++;
    }
    return terms > 0 ? sqrt(0.5 * sum / (double)terms) : 0.0;
}

typedef struct {
    char  *text;
    size_t len;
    size_t cap;
} Buf;

static void buf_add(Buf *b, const char *fmt, ...)
{
    va_list ap;
    char tmp[1024];
    int n;

    va_start(ap, fmt);
    n = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    if (n < 0) return;
    if (b->len + (size_t)n + 1 > b->cap) {
        size_t cap = b->cap == 0 ? 4096 : b->cap * 2;

        while (cap < b->len + (size_t)n + 1) cap *= 2;
        {
            char *nt = realloc(b->text, cap);

            if (nt == NULL) return;
            b->text = nt;
            b->cap = cap;
        }
    }
    memcpy(b->text + b->len, tmp, (size_t)n + 1);
    b->len += (size_t)n;
}

static const char *confidence_suffix(int confidence)
{
    if (confidence >= 99) return "99%";
    if (confidence >= 90 && confidence < 95) return "90%";
    return "95%";
}

char *fc_stats_report(const FcDatasetList *list, FclScope scope,
                      const FcStatsOptions *options)
{
    FcStatsOptions opt;
    double *v = NULL;
    Cluster *cl = NULL;
    size_t n = 0, k = 0, rounds = 0;
    Buf b;
    FclRng rng;
    double mean, sd, se, med, mad, trimmed, skew, kurt, jb, jb_p;
    double mean_lo, mean_hi, med_lo, med_hi, sd_lo, sd_hi;
    double trim_lo, trim_hi;
    double sigma_b2 = 0.0, sigma_w2 = 0.0, icc = 0.0, deff = 1.0, n_eff = 0.0;
    double cluster_lo = 0.0, cluster_hi = 0.0;
    double *sorted = NULL;

    if (options != NULL) {
        opt = *options;
    } else {
        opt.confidence = 95;
        opt.bootstrap = 1000;
        opt.per_round = true;
    }
    if (opt.confidence != 90 && opt.confidence != 95 &&
        opt.confidence != 99) {
        opt.confidence = 95;
    }
    if (opt.bootstrap < 200) opt.bootstrap = 200;
    memset(&b, 0, sizeof(b));
    if (list == NULL || list->count == 0) {
        buf_add(&b, "No datasets loaded.\n");
        return b.text;
    }
    v = copy_values(list, &n, &cl, &k, &rounds);
    if (v == NULL) {
        buf_add(&b, "No measurements selected for the statistics.\n");
        return b.text;
    }
    fcl_rng_seed(&rng, 0x5EED5EEDULL);
    mean = mean_of(v, n);
    sd = sqrt(var_of(v, n, mean));
    se = n > 0 ? sd / sqrt((double)n) : 0.0;
    {
        double *s = malloc(n * sizeof(*s));

        if (s != NULL) {
            memcpy(s, v, n * sizeof(*s));
            qsort(s, n, sizeof(*s), cmp_double);
            sorted = s;
        }
    }
    med = sorted != NULL ? median_sorted(sorted, n) : mean;
    mad = sorted != NULL ? mad_sorted(sorted, n, med) * 1.4826 : 0.0;
    trimmed = sorted != NULL ? trimmed_mean(sorted, n, 0.1) : mean;
    skew = skewness(v, n, mean, sd);
    kurt = kurtosis_excess(v, n, mean, sd);
    jb = n > 2 ? (double)n / 6.0 * (skew * skew + kurt * kurt / 4.0) : 0.0;
    jb_p = fcl_chi2_p(jb, 2);
    bootstrap_ci(v, n, stat_mean, opt.bootstrap,
                 1.0 - (1.0 - (double)opt.confidence / 100.0) , &rng,
                 &mean_lo, &mean_hi);
    bootstrap_ci(v, n, stat_median, opt.bootstrap,
                 1.0 - (1.0 - (double)opt.confidence / 100.0) , &rng,
                 &med_lo, &med_hi);
    bootstrap_ci(v, n, stat_sd, opt.bootstrap,
                 1.0 - (1.0 - (double)opt.confidence / 100.0) , &rng,
                 &sd_lo, &sd_hi);
    bootstrap_ci(v, n, stat_trimmed, opt.bootstrap,
                 1.0 - (1.0 - (double)opt.confidence / 100.0) , &rng,
                 &trim_lo, &trim_hi);
    variance_components(cl, k, n, mean, &sigma_b2, &sigma_w2, &icc, &deff,
                        &n_eff);
    cluster_bootstrap_mean(cl, k, opt.bootstrap,
                           1.0 - (1.0 - (double)opt.confidence / 100.0),
                           &rng, &cluster_lo, &cluster_hi);

    buf_add(&b, "FREQUENCY COUNTER STATISTICS\n");
    buf_add(&b, "============================\n");
    buf_add(&b, "Datasets: %zu | sessions: %zu | rounds: %zu | frequency "
              "lists: %zu | measurements: %zu\n",
            list->count, fcd_list_count_sessions(list),
            fcd_list_count_rounds(list), k, n);
    buf_add(&b, "Analysis scope: %s\n\n", fcl_scope_name(scope));
    buf_add(&b, "Descriptive statistics (measurement values, Hz)\n");
    buf_add(&b, "-----------------------------------------------\n");
    buf_add(&b, "  n                 %zu\n", n);
    buf_add(&b, "  mean              %.6f\n", mean);
    buf_add(&b, "  standard error    %.6f\n", se);
    buf_add(&b, "  std. deviation    %.6f\n", sd);
    buf_add(&b, "  minimum           %.6f\n", sorted != NULL ? sorted[0] : 0.0);
    buf_add(&b, "  Q1                %.6f\n",
            sorted != NULL ? quantile_sorted(sorted, n, 0.25) : 0.0);
    buf_add(&b, "  median            %.6f\n", med);
    buf_add(&b, "  Q3                %.6f\n",
            sorted != NULL ? quantile_sorted(sorted, n, 0.75) : 0.0);
    buf_add(&b, "  maximum           %.6f\n",
            sorted != NULL ? sorted[n - 1] : 0.0);
    buf_add(&b, "  IQR               %.6f\n",
            sorted != NULL ? quantile_sorted(sorted, n, 0.75) -
                                 quantile_sorted(sorted, n, 0.25) : 0.0);
    buf_add(&b, "  MAD (scaled)      %.6f\n", mad);
    buf_add(&b, "  10%% trimmed mean  %.6f\n", trimmed);
    buf_add(&b, "  skewness          %.4f\n", skew);
    buf_add(&b, "  excess kurtosis   %.4f\n", kurt);
    buf_add(&b, "\nNormality diagnostic\n");
    buf_add(&b, "--------------------\n");
    buf_add(&b, "  Jarque-Bera       %.4f (chi2 df=2, p = %.4g)\n", jb, jb_p);
    buf_add(&b, "  verdict           %s\n",
            jb_p < 0.05 ? "non-normal tails/skew"
                        : "consistent with a normal distribution");
    buf_add(&b, "\nBootstrap confidence intervals (%s, %zu replicates)\n",
            confidence_suffix(opt.confidence), opt.bootstrap);
    buf_add(&b, "---------------------------------------------------\n");
    buf_add(&b, "  mean              [%.6f, %.6f]\n", mean_lo, mean_hi);
    buf_add(&b, "  median            [%.6f, %.6f]\n", med_lo, med_hi);
    buf_add(&b, "  std. deviation    [%.6f, %.6f]\n", sd_lo, sd_hi);
    buf_add(&b, "  10%% trimmed mean  [%.6f, %.6f]\n", trim_lo, trim_hi);
    buf_add(&b, "\nPopulation estimates (two-level model: frequency lists "
                "within rounds)\n");
    buf_add(&b, "-------------------------------------------------------------"
                "-----\n");
    buf_add(&b, "  cluster count     %zu\n", k);
    buf_add(&b, "  between-list var  %.6f\n", sigma_b2);
    buf_add(&b, "  within-list var   %.6f\n", sigma_w2);
    buf_add(&b, "  total variance    %.6f\n", sigma_b2 + sigma_w2);
    buf_add(&b, "  ICC (list effect) %.4f\n", icc);
    buf_add(&b, "  design effect     %.3f\n", deff);
    buf_add(&b, "  effective n       %.1f\n", n_eff);
    buf_add(&b, "  population mean   %.6f [%.6f, %.6f] (cluster bootstrap "
                "%s)\n", mean, cluster_lo, cluster_hi,
            confidence_suffix(opt.confidence));
    buf_add(&b, "  population sd     %.6f\n", sqrt(sigma_b2 + sigma_w2));
    if (rounds >= 1 && opt.per_round) {
        size_t dcount = 0;

        buf_add(&b, "\nRound means (per dataset, ordered as stored) and "
                    "Allan deviation\n");
        buf_add(&b, "----------------------------------------------------"
                    "------\n");
        for (size_t d = 0; d < list->count; d++) {
            const FcDataset *ds = &list->items[d];
            double means[256];
            size_t rn = 0, idx = 0, valid = 0;
            size_t total_meas = 0;

            for (size_t s = 0; s < ds->n_sessions; s++) {
                rn += ds->sessions[s].n_rounds;
            }
            if (rn == 0) {
                buf_add(&b, "  %-32s no rounds in the selection\n",
                        ds->title != NULL ? ds->title : ds->dataset_id);
                continue;
            }
            for (size_t s = 0; s < ds->n_sessions; s++) {
                const FcSession *ss = &ds->sessions[s];

                for (size_t r = 0; r < ss->n_rounds; r++) {
                    const FcRound *rr = &ss->rounds[r];
                    size_t cnt = 0;
                    double sum = 0.0;

                    for (size_t j = 0; j < rr->n_subjects; j++) {
                        for (size_t i = 0; i < rr->subjects[j].count; i++) {
                            sum += rr->subjects[j].measurements[i];
                            cnt++;
                        }
                    }
                    total_meas += cnt;
                    if (cnt > 0 && idx < sizeof(means) / sizeof(means[0])) {
                        means[idx++] = sum / (double)cnt;
                        valid++;
                    }
                }
            }
            if (valid == 0) {
                buf_add(&b, "  %-32s rounds %2zu  no measurements in the "
                            "selected rounds\n",
                        ds->title != NULL ? ds->title : ds->dataset_id, rn);
                dcount++;
                continue;
            }
            {
                double mn = mean_of(means, idx);
                double m2 = var_of(means, idx, mn);

                /* Allan deviation needs at least two round means; shorter
                 * selections are still listed so every dataset with
                 * selected rounds appears in the report */
                if (valid >= 2) {
                    double adev = allan_deviation(means, idx);

                    buf_add(&b, "  %-32s rounds %2zu  mean %.6f  sd %.6f  "
                                "ADEV(tau=1 round) %.6f\n",
                            ds->title != NULL ? ds->title : ds->dataset_id,
                            rn, mn, sqrt(m2), adev);
                } else {
                    buf_add(&b, "  %-32s rounds %2zu  mean %.6f  sd n/a  "
                                "ADEV(tau=1 round) n/a (needs >= 2 rounds)\n",
                            ds->title != NULL ? ds->title : ds->dataset_id,
                            rn, mn);
                }
                (void)total_meas;
                dcount++;
            }
        }
        if (dcount == 0) {
            buf_add(&b, "  (no dataset has selected rounds)\n");
        }
    }
    buf_add(&b, "\nNotes\n");
    buf_add(&b, "-----\n");
    buf_add(&b, "  Measurements are treated as independent samples of a "
                "two-level process: frequency lists are the sampling units, "
                "rounds group them. The ICC and design effect quantify how "
                "much of the variance is between lists; the effective n is "
                "the independent-sample equivalent of the selected data. "
                "The cluster bootstrap resamples whole frequency lists, "
                "giving a population-mean interval that is robust to the "
                "within-round correlation. Jarque-Bera p-values test "
                "normality (skewness/kurtosis). The Allan deviation of the "
                "round means is a stability measure of the device over one "
                "round.\n");
    free(v);
    free(cl);
    free(sorted);
    return b.text;
}
