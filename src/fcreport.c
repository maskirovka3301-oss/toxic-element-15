#include "fcreport.h"

#include "fcdag.h"
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ------------------------------------------------------------------ */
/* Report rendering (same document, three presentations)              */
/* ------------------------------------------------------------------ */

typedef struct {
    char  *buf;
    size_t len, cap;
    bool   failed;
} Str;

static void str_add(Str *s, const char *text)
{
    size_t n;

    if (s->failed || text == NULL) return;
    n = strlen(text);
    if (s->len + n + 1 > s->cap) {
        size_t cap = s->cap == 0 ? 1024 : s->cap;
        char *nb;

        while (cap < s->len + n + 1) cap *= 2;
        nb = realloc(s->buf, cap);
        if (nb == NULL) {
            s->failed = true;
            return;
        }
        s->buf = nb;
        s->cap = cap;
    }
    memcpy(s->buf + s->len, text, n + 1);
    s->len += n;
}

static void str_addf(Str *s, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));

static void str_addf(Str *s, const char *fmt, ...)
{
    char tmp[1024];
    va_list ap;

    va_start(ap, fmt);
    vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    str_add(s, tmp);
}

static void html_escape(Str *s, const char *text)
{
    for (const char *p = text; p != NULL && *p != '\0'; p++) {
        switch (*p) {
        case '&': str_add(s, "&amp;"); break;
        case '<': str_add(s, "&lt;"); break;
        case '>': str_add(s, "&gt;"); break;
        case '"': str_add(s, "&quot;"); break;
        default: {
            char one[2] = { *p, '\0' };

            str_add(s, one);
        }
        }
    }
}

static char *str_take(Str *s)
{
    if (s->failed) {
        free(s->buf);
        return NULL;
    }
    if (s->buf == NULL) return strdup("");
    return s->buf;
}

/* Append the schema-aware dataset listing used by all three renderers.
 * `markdown`/`html` select the bullet syntax; plain text uses indentation. */
typedef enum { LIST_TEXT = 0, LIST_MD, LIST_HTML } ListStyle;

static void list_begin(Str *out, ListStyle style)
{
    if (style == LIST_MD) str_add(out, "\n");
    else if (style == LIST_HTML) str_add(out, "<ul>\n");
}

static void list_end(Str *out, ListStyle style)
{
    if (style == LIST_HTML) str_add(out, "</ul>\n");
}

static void list_item_begin(Str *out, ListStyle style, int depth)
{
    if (style == LIST_TEXT) {
        for (int i = 0; i < depth; i++) str_add(out, "  ");
        str_add(out, "- ");
    } else if (style == LIST_MD) {
        for (int i = 0; i < depth; i++) str_add(out, "  ");
        str_add(out, "- ");
    } else {
        str_add(out, "<li>");
    }
}

static void list_item_end(Str *out, ListStyle style)
{
    if (style == LIST_HTML) str_add(out, "</li>\n");
    else str_add(out, "\n");
}

static void add_escaped(Str *out, ListStyle style, const char *text)
{
    if (text == NULL) return;
    if (style == LIST_HTML) html_escape(out, text);
    else str_add(out, text);
}

/* Summarise the measurement statistics of one value series. */
static void add_stats(Str *out, ListStyle style, const double *v, size_t n)
{
    double lo = 0.0, hi = 0.0, mean = 0.0;
    char buf[256];

    if (!fc_stats(v, n, &lo, &hi, &mean)) {
        add_escaped(out, style, "no measurements");
        return;
    }
    snprintf(buf, sizeof(buf), "%zu measurements, %.3f … %.3f Hz "
                               "(mean %.3f Hz)", n, lo, hi, mean);
    add_escaped(out, style, buf);
}

/* Render every dataset, session, round and subject of the loaded list. */
static void render_datasets(Str *out, ListStyle style,
                            const FcDatasetList *list)
{
    if (list == NULL || list->count == 0) {
        if (style == LIST_HTML) {
            str_add(out, "<p><em>No datasets loaded.</em></p>\n");
        } else if (style == LIST_MD) {
            str_add(out, "*No datasets loaded.*\n\n");
        } else {
            str_add(out, "  (no datasets loaded)\n");
        }
        return;
    }
    list_begin(out, style);
    for (size_t d = 0; d < list->count; d++) {
        const FcDataset *ds = &list->items[d];
        char buf[512];

        list_item_begin(out, style, 0);
        add_escaped(out, style, ds->title != NULL ? ds->title : "(untitled)");
        snprintf(buf, sizeof(buf), " [id %s, source %s] — %zu sessions, "
                                   "%zu rounds, %zu subjects, %zu "
                                   "measurements",
                 ds->dataset_id != NULL ? ds->dataset_id : "?",
                 ds->source_file != NULL ? ds->source_file : "?",
                 ds->n_sessions, fcd_count_rounds(ds), fcd_count_subjects(ds),
                 fcd_count_measurements(ds));
        add_escaped(out, style, buf);
        if (style == LIST_TEXT) str_add(out, "\n");
        for (size_t i = 0; i < ds->n_notes; i++) {
            list_item_begin(out, style, 1);
            add_escaped(out, style, "note: ");
            add_escaped(out, style, ds->notes[i]);
            list_item_end(out, style);
        }
        for (size_t s = 0; s < ds->n_sessions; s++) {
            const FcSession *ss = &ds->sessions[s];

            list_item_begin(out, style, 1);
            snprintf(buf, sizeof(buf), "session %s — %zu rounds",
                     ss->session_id.text, ss->n_rounds);
            add_escaped(out, style, buf);
            if (style == LIST_TEXT) str_add(out, "\n");
            for (size_t r = 0; r < ss->n_rounds; r++) {
                const FcRound *rr = &ss->rounds[r];

                list_item_begin(out, style, 2);
                snprintf(buf, sizeof(buf), "round %s", rr->round_id.text);
                add_escaped(out, style, buf);
                if (rr->has_coordinates) {
                    snprintf(buf, sizeof(buf), " (%s, %s)", "", "");
                    add_escaped(out, style, " — coordinates ");
                    snprintf(buf, sizeof(buf), "%.6f, %.6f", rr->latitude,
                             rr->longitude);
                    add_escaped(out, style, buf);
                }
                if (rr->location != NULL) {
                    add_escaped(out, style, " — ");
                    add_escaped(out, style, rr->location);
                }
                if (rr->date != NULL) {
                    add_escaped(out, style, " — ");
                    add_escaped(out, style, rr->date);
                }
                snprintf(buf, sizeof(buf), " — %zu subjects", rr->n_subjects);
                add_escaped(out, style, buf);
                if (style == LIST_TEXT) str_add(out, "\n");
                for (size_t j = 0; j < rr->n_subjects; j++) {
                    const FcSubject *sj = &rr->subjects[j];

                    list_item_begin(out, style, 3);
                    add_escaped(out, style, sj->name);
                    add_escaped(out, style, " — ");
                    add_stats(out, style, sj->measurements, sj->count);
                    list_item_end(out, style);
                }
                list_item_end(out, style);
            }
            list_item_end(out, style);
        }
        if (ds->n_unassigned > 0) {
            list_item_begin(out, style, 1);
            add_escaped(out, style, "unassigned — ");
            add_stats(out, style, ds->unassigned, ds->n_unassigned);
            list_item_end(out, style);
        }
        list_item_end(out, style);
    }
    list_end(out, style);
}

static void render_text(Str *out, const FcDatasetList *list,
                        const FcRun *run)
{
    char stamp[64];
    time_t now = time(NULL);
    struct tm tmv;

    if (localtime_r(&now, &tmv) != NULL) {
        strftime(stamp, sizeof(stamp), "%Y-%m-%d %H:%M:%S", &tmv);
    } else {
        snprintf(stamp, sizeof(stamp), "%s", "unknown");
    }
    str_add(out, "FREQUENCY COUNTER ANALYSIS REPORT\n");
    str_add(out, "=================================\n");
    str_addf(out, "Generated: %s\n", stamp);
    str_addf(out, "Datasets: %zu | sessions: %zu | rounds: %zu | subjects: %zu "
                  "| measurements: %zu\n\n",
             list != NULL ? list->count : 0, fcd_list_count_sessions(list),
             fcd_list_count_rounds(list), fcd_list_count_subjects(list),
             fcd_list_count_measurements(list));
    str_add(out, "INPUT (schema: fc_dataset.json)\n---------------------------------\n");
    render_datasets(out, LIST_TEXT, list);
    str_add(out, "\nDAG\n---\n");
    if (fc_node_count == 0) {
        str_add(out, "  Registry: 0 analyses defined.\n");
        str_add(out, "  The analysis framework (dependency-ordered execution,\n"
                     "  per-node status/skip notes and this report) is in place;\n"
                     "  the analysis algorithms are not implemented yet.\n");
    } else {
        str_addf(out, "  Registry: %zu analyses, executed %zu\n", fc_node_count,
                 run != NULL ? run->count : 0);
        if (run != NULL) {
            for (size_t i = 0; i < run->count; i++) {
                const DagCoreNode *r = &run->nodes[i];

                str_addf(out, "  %zu. %s — %s", i + 1,
                         r->spec != NULL && r->spec->title != NULL
                             ? r->spec->title : "(untitled)",
                         dagcore_status_name(r->status));
                if (r->note[0] != '\0') str_addf(out, " (%s)", r->note);
                str_addf(out, "  %.3f s\n", r->elapsed_s);
            }
        }
    }
    str_add(out, "\nRESULTS\n-------\n");
    str_add(out, "  (no analysis produced any parameters yet)\n");
    str_add(out, "\nSUMMARY\n-------\n");
    str_addf(out, "  Executed 0 of %zu analyses; every registry entry is "
                  "pending an implementation.\n", fc_node_count);
    if (run != NULL) str_addf(out, "  DAG elapsed: %.3f s\n", run->elapsed_s);
}

static void render_markdown(Str *out, const FcDatasetList *list,
                            const FcRun *run)
{
    char stamp[64];
    time_t now = time(NULL);
    struct tm tmv;

    if (localtime_r(&now, &tmv) != NULL) {
        strftime(stamp, sizeof(stamp), "%Y-%m-%d %H:%M:%S", &tmv);
    } else {
        snprintf(stamp, sizeof(stamp), "%s", "unknown");
    }
    str_add(out, "# Frequency Counter Analysis Report\n\n");
    str_addf(out, "*Generated: %s*  \n", stamp);
    str_addf(out, "*Datasets: %zu, sessions: %zu, rounds: %zu, subjects: %zu, "
                  "measurements: %zu*\n\n",
             list != NULL ? list->count : 0, fcd_list_count_sessions(list),
             fcd_list_count_rounds(list), fcd_list_count_subjects(list),
             fcd_list_count_measurements(list));
    str_add(out, "## Input (schema: fc_dataset.json)\n\n");
    render_datasets(out, LIST_MD, list);
    str_add(out, "\n## DAG\n\n");
    if (fc_node_count == 0) {
        str_add(out, "The registry is empty: 0 analyses are defined. The "
                     "execution framework and this report are in place; the "
                     "analysis algorithms are not implemented yet.\n\n");
    } else {
        str_addf(out, "Registry: %zu analyses, executed %zu\n\n",
                 fc_node_count, run != NULL ? run->count : 0);
    }
    str_add(out, "## Results\n\n*No analysis produced any parameters yet.*\n\n");
    str_add(out, "## Summary\n\n");
    str_addf(out, "Executed 0 of %zu analyses; every registry entry is "
                  "pending an implementation.\n", fc_node_count);
}

static void render_html(Str *out, const FcDatasetList *list, const FcRun *run)
{
    char stamp[64];
    time_t now = time(NULL);
    struct tm tmv;

    if (localtime_r(&now, &tmv) != NULL) {
        strftime(stamp, sizeof(stamp), "%Y-%m-%d %H:%M:%S", &tmv);
    } else {
        snprintf(stamp, sizeof(stamp), "%s", "unknown");
    }
    str_add(out, "<!DOCTYPE html>\n<html><head><meta charset=\"utf-8\">"
                 "<title>Frequency Counter Analysis Report</title></head>"
                 "<body>\n<h1>Frequency Counter Analysis Report</h1>\n");
    str_addf(out, "<p><em>Generated: %s</em><br>\n", stamp);
    str_addf(out, "<em>Datasets: %zu, sessions: %zu, rounds: %zu, subjects: "
                  "%zu, measurements: %zu</em></p>\n",
             list != NULL ? list->count : 0, fcd_list_count_sessions(list),
             fcd_list_count_rounds(list), fcd_list_count_subjects(list),
             fcd_list_count_measurements(list));
    str_add(out, "<h2>Input (schema: fc_dataset.json)</h2>\n");
    render_datasets(out, LIST_HTML, list);
    str_add(out, "<h2>DAG</h2>\n");
    if (fc_node_count == 0) {
        str_add(out, "<p>The registry is empty: 0 analyses are defined. The "
                     "execution framework and this report are in place; the "
                     "analysis algorithms are not implemented yet.</p>\n");
    } else {
        str_addf(out, "<p>Registry: %zu analyses, executed %zu</p>\n",
                 fc_node_count, run != NULL ? run->count : 0);
    }
    str_add(out, "<h2>Results</h2>\n<p><em>No analysis produced any "
                 "parameters yet.</em></p>\n");
    str_add(out, "<h2>Summary</h2>\n");
    str_addf(out, "<p>Executed 0 of %zu analyses; every registry entry is "
                  "pending an implementation.</p>\n</body></html>\n",
             fc_node_count);
}

void fc_report(const FcDatasetList *list, const DagCoreRun *run,
                   char **text_out, char **md_out, char **html_out)
{
    Str t;
    Str m;
    Str h;

    memset(&t, 0, sizeof(t));
    memset(&m, 0, sizeof(m));
    memset(&h, 0, sizeof(h));
    render_text(&t, list, run);
    render_markdown(&m, list, run);
    render_html(&h, list, run);
    if (text_out != NULL) *text_out = str_take(&t);
    if (md_out != NULL) *md_out = str_take(&m);
    if (html_out != NULL) *html_out = str_take(&h);
}
