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

static void render_text(Str *out, const FreqSeriesList *list, const DagCoreRun *run)
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
    str_addf(out, "Sessions: %zu | measurements: %zu\n\n",
             list != NULL ? list->count : 0, fsl_total_points(list));
    str_add(out, "INPUT\n-----\n");
    if (list == NULL || list->count == 0) {
        str_add(out, "  (no sessions loaded)\n");
    } else {
        for (size_t i = 0; i < list->count; i++) {
            const FreqSession *s = &list->items[i];

            str_addf(out, "  %zu. %s", i + 1,
                     s->label != NULL ? s->label : "(unnamed)");
            if (s->datetime != NULL && s->datetime[0] != '\0') {
                str_addf(out, " — %s", s->datetime);
            }
            if (s->has_location) {
                str_addf(out, " — %.6f, %.6f", s->lat, s->lon);
            } else {
                str_add(out, " — no location");
            }
            str_addf(out, " — %zu measurements\n", s->count);
            if (s->notes != NULL && s->notes[0] != '\0') {
                str_addf(out, "     notes: %s\n", s->notes);
            }
        }
    }
    str_add(out, "\nDAG\n---\n");
    if (fc_node_count == 0) {
        str_add(out, "  Registry: 0 analyses defined.\n");
        str_add(out, "  The analysis framework (dependency-ordered execution,\n"
                     "  per-node status/skip notes and this report) is in place;\n"
                     "  the analysis algorithms are not implemented yet.\n");
    } else {
        str_addf(out, "  Registry: %zu analyses, executed %zu\n",
                 fc_node_count, run != NULL ? run->count : 0);
        if (run != NULL) {
            for (size_t i = 0; i < run->count; i++) {
                const DagCoreNode *r = &run->nodes[i];

                str_addf(out, "  %zu. %s — %s", i + 1,
                         r->spec != NULL && r->spec != NULL && r->spec->title != NULL ? r->spec->title : "(untitled)" != NULL
                             ? r->spec != NULL && r->spec->title != NULL ? r->spec->title : "(untitled)" : "(untitled)",
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

static void render_markdown(Str *out, const FreqSeriesList *list,
                            const DagCoreRun *run)
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
    str_addf(out, "*Sessions: %zu, measurements: %zu*\n\n",
             list != NULL ? list->count : 0, fsl_total_points(list));
    str_add(out, "## Input\n\n");
    if (list == NULL || list->count == 0) {
        str_add(out, "*No sessions loaded.*\n\n");
    } else {
        for (size_t i = 0; i < list->count; i++) {
            const FreqSession *s = &list->items[i];

            str_addf(out, "- **%s**", s->label != NULL ? s->label : "(unnamed)");
            if (s->datetime != NULL && s->datetime[0] != '\0') {
                str_addf(out, " — %s", s->datetime);
            }
            if (s->has_location) {
                str_addf(out, " — %.6f, %.6f", s->lat, s->lon);
            } else {
                str_add(out, " — no location");
            }
            str_addf(out, " — %zu measurements\n", s->count);
            if (s->notes != NULL && s->notes[0] != '\0') {
                str_addf(out, "  - notes: %s\n", s->notes);
            }
        }
        str_add(out, "\n");
    }
    str_add(out, "## DAG\n\n");
    if (fc_node_count == 0) {
        str_add(out, "The registry is empty: 0 analyses are defined. The "
                     "execution framework and this report are in place; the "
                     "analysis algorithms are not implemented yet.\n\n");
    } else {
        str_addf(out, "Registry: %zu analyses, executed %zu\n\n",
                 fc_node_count, run != NULL ? run->count : 0);
        if (run != NULL) {
            for (size_t i = 0; i < run->count; i++) {
                const DagCoreNode *r = &run->nodes[i];

                str_addf(out, "- %s — %s", 
                         r->spec != NULL && r->spec != NULL && r->spec->title != NULL ? r->spec->title : "(untitled)" != NULL
                             ? r->spec != NULL && r->spec->title != NULL ? r->spec->title : "(untitled)" : "(untitled)",
                         dagcore_status_name(r->status));
                if (r->note[0] != '\0') str_addf(out, " (%s)", r->note);
                str_add(out, "\n");
            }
            str_add(out, "\n");
        }
    }
    str_add(out, "## Results\n\n*No analysis produced any parameters yet.*\n\n");
    str_add(out, "## Summary\n\n");
    str_addf(out, "Executed 0 of %zu analyses; every registry entry is "
                  "pending an implementation.\n", fc_node_count);
}

static void render_html(Str *out, const FreqSeriesList *list, const DagCoreRun *run)
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
    str_addf(out, "<em>Sessions: %zu, measurements: %zu</em></p>\n",
             list != NULL ? list->count : 0, fsl_total_points(list));
    str_add(out, "<h2>Input</h2>\n");
    if (list == NULL || list->count == 0) {
        str_add(out, "<p>No sessions loaded.</p>\n");
    } else {
        str_add(out, "<ul>\n");
        for (size_t i = 0; i < list->count; i++) {
            const FreqSession *s = &list->items[i];

            str_add(out, "<li><strong>");
            html_escape(out, s->label != NULL ? s->label : "(unnamed)");
            str_add(out, "</strong>");
            if (s->datetime != NULL && s->datetime[0] != '\0') {
                str_add(out, " — ");
                html_escape(out, s->datetime);
            }
            if (s->has_location) {
                str_addf(out, " — %.6f, %.6f", s->lat, s->lon);
            } else {
                str_add(out, " — no location");
            }
            str_addf(out, " — %zu measurements", s->count);
            if (s->notes != NULL && s->notes[0] != '\0') {
                str_add(out, "<br><small>notes: ");
                html_escape(out, s->notes);
                str_add(out, "</small>");
            }
            str_add(out, "</li>\n");
        }
        str_add(out, "</ul>\n");
    }
    str_add(out, "<h2>DAG</h2>\n");
    if (fc_node_count == 0) {
        str_add(out, "<p>The registry is empty: 0 analyses are defined. The "
                     "execution framework and this report are in place; the "
                     "analysis algorithms are not implemented yet.</p>\n");
    } else {
        str_addf(out, "<p>Registry: %zu analyses, executed %zu</p>\n<ul>\n",
                 fc_node_count, run != NULL ? run->count : 0);
        if (run != NULL) {
            for (size_t i = 0; i < run->count; i++) {
                const DagCoreNode *r = &run->nodes[i];

                str_add(out, "<li>");
                html_escape(out, r->spec != NULL && r->spec != NULL && r->spec->title != NULL ? r->spec->title : "(untitled)" != NULL
                                      ? r->spec != NULL && r->spec->title != NULL ? r->spec->title : "(untitled)" : "(untitled)");
                str_add(out, " — ");
                html_escape(out,
                            dagcore_status_name(r->status));
                if (r->note[0] != '\0') {
                    str_add(out, " (");
                    html_escape(out, r->note);
                    str_add(out, ")");
                }
                str_add(out, "</li>\n");
            }
        }
        str_add(out, "</ul>\n");
    }
    str_add(out, "<h2>Results</h2>\n<p><em>No analysis produced any "
                 "parameters yet.</em></p>\n");
    str_add(out, "<h2>Summary</h2>\n");
    str_addf(out, "<p>Executed 0 of %zu analyses; every registry entry is "
                  "pending an implementation.</p>\n</body></html>\n",
             fc_node_count);
}

void fc_report(const FreqSeriesList *list, const DagCoreRun *run,
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
