/*
 * fcexport.c — CSV export of the selected Frequency Counter measurements.
 *
 * The exported table has one row per MEASUREMENT (not per aggregate): the
 * identity columns say which dataset, session, round and frequency list the
 * value belongs to, followed by the position inside the list and the
 * measured frequency in Hz. The granularity chosen by the caller decides
 * how finely the rows are labelled:
 *
 *   round:   dataset,session,round,frequency_list,index,measurement_hz
 *   session: dataset,session,frequency_list,index,measurement_hz
 *   dataset: dataset,frequency_list,index,measurement_hz
 *
 * Only the checked frequency lists of the analysis snapshot are exported.
 * Dataset-level unassigned measurements are exported with empty
 * session/round fields and the frequency-list name "unassigned".
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
        size_t cap = b->cap == 0 ? 8192 : b->cap * 2;

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
static void csv_field(CsvBuf *b, const char *s)
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
    csv_add(b, ",");
}

static void csv_header(CsvBuf *b, FcCsvGranularity granularity)
{
    csv_field(b, "dataset");
    if (granularity != FC_CSV_DATASET) csv_field(b, "session");
    if (granularity == FC_CSV_ROUND) csv_field(b, "round");
    csv_field(b, "frequency_list");
    csv_field(b, "index");
    csv_add(b, "measurement_hz\n");
}

/* One row per measurement of one frequency list. */
static void csv_list_rows(CsvBuf *b, FcCsvGranularity granularity,
                          const char *dataset, const char *session,
                          const char *round, const char *list_name,
                          const double *values, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        csv_field(b, dataset);
        if (granularity != FC_CSV_DATASET) csv_field(b, session);
        if (granularity == FC_CSV_ROUND) csv_field(b, round);
        csv_field(b, list_name);
        csv_add(b, "%zu,%.6f\n", i + 1, values[i]);
    }
}

char *fc_csv_export(const FcDatasetList *list, FcCsvGranularity granularity)
{
    CsvBuf b;

    memset(&b, 0, sizeof(b));
    if (list == NULL) return NULL;
    csv_header(&b, granularity);
    for (size_t d = 0; d < list->count; d++) {
        const FcDataset *ds = &list->items[d];
        const char *title = ds->title != NULL ? ds->title : ds->dataset_id;

        for (size_t s = 0; s < ds->n_sessions; s++) {
            const FcSession *ss = &ds->sessions[s];

            for (size_t r = 0; r < ss->n_rounds; r++) {
                const FcRound *rr = &ss->rounds[r];

                for (size_t j = 0; j < rr->n_subjects; j++) {
                    const FcSubject *sj = &rr->subjects[j];

                    if (sj->count == 0) continue;
                    csv_list_rows(&b, granularity, title, ss->session_id.text,
                                  rr->round_id.text,
                                  sj->name != NULL ? sj->name : "",
                                  sj->measurements, sj->count);
                }
            }
        }
        if (ds->n_unassigned > 0) {
            csv_list_rows(&b, granularity, title, "", "", "unassigned",
                          ds->unassigned, ds->n_unassigned);
        }
    }
    return b.text;
}
