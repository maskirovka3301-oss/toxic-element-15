/*
 * fcexport.c — CSV export of the selected Frequency Counter measurements.
 *
 * Each row aggregates the checked frequency lists of one round, session or
 * dataset: the row identity columns, the number of frequency lists, the
 * number of measurements and the descriptive statistics of the values (all
 * in Hz). Rows with no measurements are omitted. Rows are ordered by
 * dataset, session and round as stored; fields are CSV-quoted when needed.
 */
#include "fcexport.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char *fc_csv_granularity_name(FcCsvGranularity granularity)
{
    switch (granularity) {
    case FC_CSV_ROUND: return "round";
    case FC_CSV_SESSION: return "session";
    default: return "dataset";
    }
}

typedef struct {
    char  *text;
    size_t len;
    size_t cap;
} CsvBuf;

static void csv_add(CsvBuf *b, const char *fmt, ...)
{
    va_list ap;
    char tmp[2048];
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

/* Append a CSV-escaped text field. */
static void csv_field(CsvBuf *b, const char *s, int last)
{
    bool quote = false;
    const char *p;

    if (s == NULL) s = "";
    for (p = s; *p != '\0'; p++) {
        if (*p == ',' || *p == '"' || *p == '\n' || *p == '\r') {
            quote = true;
            break;
        }
    }
    if (quote) csv_add(b, "\"");
    for (p = s; *p != '\0'; p++) {
        if (*p == '"') csv_add(b, "\"\"");
        else csv_add(b, "%c", *p);
    }
    if (quote) csv_add(b, "\"");
    if (last) csv_add(b, "\n");
    else csv_add(b, ",");
}

static void csv_number(CsvBuf *b, double v, int last)
{
    csv_add(b, "%.6f%s", v, last ? "\n" : ",");
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a;
    double y = *(const double *)b;

    return (x < y) ? -1 : (x > y) ? 1 : 0;
}

/* Aggregate one unit's measurements and append the statistics columns. */
static void csv_stats(CsvBuf *b, const double *v, size_t n, int last)
{
    double sum = 0.0, mean = 0.0, sd = 0.0, min = 0.0, max = 0.0, med = 0.0;
    double *sorted = NULL;

    if (n > 0) {
        min = max = v[0];
        for (size_t i = 0; i < n; i++) {
            sum += v[i];
            if (v[i] < min) min = v[i];
            if (v[i] > max) max = v[i];
        }
        mean = sum / (double)n;
        if (n > 1) {
            double ss = 0.0;

            for (size_t i = 0; i < n; i++) {
                double d = v[i] - mean;

                ss += d * d;
            }
            sd = sqrt(ss / (double)(n - 1));
        }
        sorted = malloc(n * sizeof(*sorted));
        if (sorted != NULL) {
            memcpy(sorted, v, n * sizeof(*sorted));
            qsort(sorted, n, sizeof(*sorted), cmp_double);
            med = (n % 2 == 1) ? sorted[n / 2]
                               : 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
        }
    }
    csv_add(b, "%zu,", n);
    csv_number(b, mean, 0);
    csv_number(b, sd, 0);
    csv_number(b, min, 0);
    csv_number(b, med, 0);
    if (last) {
        csv_add(b, "%.6f\n", max);
    } else {
        csv_add(b, "%.6f,", max);
    }
    free(sorted);
}

/* Collect the measurements of the selected frequency lists of a
 * dataset/session/round range into `*vals`; returns the count. */
static bool collect(const FcDataset *ds, long only_session, long only_round,
                    double **vals, size_t *n_out, size_t *lists_out)
{
    size_t n = 0, used = 0, lists = 0;

    for (size_t s = 0; s < ds->n_sessions; s++) {
        const FcSession *ss = &ds->sessions[s];

        if (only_session >= 0 && (size_t)only_session != s) continue;
        for (size_t r = 0; r < ss->n_rounds; r++) {
            const FcRound *rr = &ss->rounds[r];

            if (only_round >= 0 && (size_t)only_round != r) continue;
            for (size_t j = 0; j < rr->n_subjects; j++) {
                if (rr->subjects[j].count == 0) continue;
                n += rr->subjects[j].count;
                lists++;
            }
        }
    }
    *vals = n > 0 ? malloc(n * sizeof(**vals)) : NULL;
    *n_out = 0;
    *lists_out = 0;
    if (n > 0 && *vals == NULL) return false;
    for (size_t s = 0; s < ds->n_sessions; s++) {
        const FcSession *ss = &ds->sessions[s];

        if (only_session >= 0 && (size_t)only_session != s) continue;
        for (size_t r = 0; r < ss->n_rounds; r++) {
            const FcRound *rr = &ss->rounds[r];

            if (only_round >= 0 && (size_t)only_round != r) continue;
            for (size_t j = 0; j < rr->n_subjects; j++) {
                if (rr->subjects[j].count == 0) continue;
                memcpy(*vals + used, rr->subjects[j].measurements,
                       rr->subjects[j].count * sizeof(**vals));
                used += rr->subjects[j].count;
                lists++;
            }
        }
    }
    *n_out = used;
    *lists_out = lists;
    return true;
}

char *fc_csv_export(const FcDatasetList *list, FcCsvGranularity granularity)
{
    CsvBuf b;
    bool header_done = false;

    memset(&b, 0, sizeof(b));
    if (list == NULL) return NULL;
    for (size_t d = 0; d < list->count; d++) {
        const FcDataset *ds = &list->items[d];
        const char *title = ds->title != NULL ? ds->title : ds->dataset_id;

        if (!header_done) {
            csv_field(&b, "dataset", 0);
            if (granularity != FC_CSV_DATASET) {
                csv_field(&b, "session", 0);
            }
            if (granularity == FC_CSV_ROUND) {
                csv_field(&b, "round", 0);
            }
            csv_field(&b, "frequency_lists", 0);
            csv_field(&b, "measurements", 0);
            csv_field(&b, "mean_hz", 0);
            csv_field(&b, "sd_hz", 0);
            csv_field(&b, "min_hz", 0);
            csv_field(&b, "median_hz", 0);
            csv_field(&b, "max_hz", 1);
            header_done = true;
        }
        if (granularity == FC_CSV_DATASET) {
            double *vals = NULL;
            size_t n = 0, lists = 0;

            if (!collect(ds, -1, -1, &vals, &n, &lists)) continue;
            if (n == 0) {
                free(vals);
                continue;
            }
            csv_field(&b, title, 0);
            csv_add(&b, "%zu,", lists);
            csv_stats(&b, vals, n, 1);
            free(vals);
            continue;
        }
        for (size_t s = 0; s < ds->n_sessions; s++) {
            const FcSession *ss = &ds->sessions[s];

            if (granularity == FC_CSV_SESSION) {
                double *vals = NULL;
                size_t n = 0, lists = 0;

                if (!collect(ds, (long)s, -1, &vals, &n, &lists)) continue;
                if (n == 0) {
                    free(vals);
                    continue;
                }
                csv_field(&b, title, 0);
                csv_field(&b, ss->session_id.text, 0);
                csv_add(&b, "%zu,", lists);
                csv_stats(&b, vals, n, 1);
                free(vals);
                continue;
            }
            for (size_t r = 0; r < ss->n_rounds; r++) {
                const FcRound *rr = &ss->rounds[r];
                double *vals = NULL;
                size_t n = 0, lists = 0;

                if (!collect(ds, (long)s, (long)r, &vals, &n, &lists)) {
                    continue;
                }
                if (n == 0) {
                    free(vals);
                    continue;
                }
                csv_field(&b, title, 0);
                csv_field(&b, ss->session_id.text, 0);
                csv_field(&b, rr->round_id.text, 0);
                csv_add(&b, "%zu,", lists);
                csv_stats(&b, vals, n, 1);
                free(vals);
            }
        }
    }
    if (!header_done) {
        /* nothing loaded: still produce a usable header for the granularity */
        csv_field(&b, "dataset", 0);
        if (granularity != FC_CSV_DATASET) csv_field(&b, "session", 0);
        if (granularity == FC_CSV_ROUND) csv_field(&b, "round", 0);
        csv_field(&b, "frequency_lists", 0);
        csv_field(&b, "measurements", 0);
        csv_field(&b, "mean_hz", 0);
        csv_field(&b, "sd_hz", 0);
        csv_field(&b, "min_hz", 0);
        csv_field(&b, "median_hz", 0);
        csv_field(&b, "max_hz", 1);
    }
    return b.text;
}
