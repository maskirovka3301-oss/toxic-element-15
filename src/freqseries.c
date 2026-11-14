#include "freqseries.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

static char *dup_str(const char *s)
{
    char *d;

    if (s == NULL) return NULL;
    d = strdup(s);
    return d;
}

static void session_clear(FreqSession *s)
{
    if (s == NULL) return;
    free(s->label);
    free(s->path);
    free(s->datetime);
    free(s->notes);
    free(s->freqs);
    memset(s, 0, sizeof(*s));
}

void fsl_init(FreqSeriesList *list)
{
    if (list != NULL) memset(list, 0, sizeof(*list));
}

void fsl_clear(FreqSeriesList *list)
{
    if (list == NULL) return;
    for (size_t i = 0; i < list->count; i++) session_clear(&list->items[i]);
    free(list->items);
    memset(list, 0, sizeof(*list));
}

/* File stem (without directory and extension) -> default label. */
static char *stem_of(const char *path)
{
    const char *base = strrchr(path, '/');
    char *out;
    char *dot;

    base = (base != NULL) ? base + 1 : path;
    out = dup_str(base);
    if (out == NULL) return NULL;
    dot = strrchr(out, '.');
    if (dot != NULL && dot != out) *dot = '\0';
    return out;
}

/* File modification time -> the default date/time text. */
static char *mtime_text(const char *path)
{
    struct stat st;
    struct tm tmv;
    char buf[64];

    if (stat(path, &st) != 0) return dup_str("");
    if (localtime_r(&st.st_mtime, &tmv) == NULL) return dup_str("");
    if (strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M", &tmv) == 0) {
        return dup_str("");
    }
    return dup_str(buf);
}

static bool session_append_freq(FreqSession *s, double hz)
{
    if (s->count == s->cap) {
        size_t cap = s->cap == 0 ? 256 : s->cap * 2;
        double *n = realloc(s->freqs, cap * sizeof(*n));

        if (n == NULL) return false;
        s->freqs = n;
        s->cap = cap;
    }
    s->freqs[s->count++] = hz;
    return true;
}

bool fsl_load_file(FreqSeriesList *list, const char *path, char *err,
                   size_t err_len)
{
    FILE *f;
    char line[512];
    FreqSession s;
    size_t line_no = 0;

    if (list == NULL || path == NULL) return false;
    if (err != NULL && err_len > 0) err[0] = '\0';
    f = fopen(path, "r");
    if (f == NULL) {
        if (err != NULL) {
            snprintf(err, err_len, "cannot open %s: %s", path,
                     strerror(errno));
        }
        return false;
    }
    memset(&s, 0, sizeof(s));
    s.label = stem_of(path);
    s.path = dup_str(path);
    s.datetime = mtime_text(path);
    s.notes = dup_str("");
    if (s.label == NULL || s.path == NULL || s.datetime == NULL ||
        s.notes == NULL) {
        fclose(f);
        session_clear(&s);
        if (err != NULL) snprintf(err, err_len, "out of memory");
        return false;
    }
    while (fgets(line, sizeof(line), f) != NULL) {
        char *p = line;
        char *end = NULL;
        double hz;

        line_no++;
        while (isspace((unsigned char)*p)) p++;
        if (*p == '\0' || *p == '#') continue;   /* blank / comment */
        errno = 0;
        hz = strtod(p, &end);
        if (end == p || errno == ERANGE || !isfinite(hz)) {
            continue;   /* not a measurement line */
        }
        while (isspace((unsigned char)*end)) end++;
        if (*end != '\0' && *end != '#') continue;   /* junk after the value */
        if (!session_append_freq(&s, hz)) {
            fclose(f);
            session_clear(&s);
            if (err != NULL) snprintf(err, err_len, "out of memory");
            return false;
        }
    }
    fclose(f);
    if (s.count == 0) {
        session_clear(&s);
        if (err != NULL) {
            snprintf(err, err_len,
                     "%s contains no frequency measurements (one value in "
                     "Hz per line expected)", path);
        }
        return false;
    }
    /* append */
    if (list->count == list->cap) {
        size_t cap = list->cap == 0 ? 8 : list->cap * 2;
        FreqSession *n = realloc(list->items, cap * sizeof(*n));

        if (n == NULL) {
            session_clear(&s);
            if (err != NULL) snprintf(err, err_len, "out of memory");
            return false;
        }
        list->items = n;
        list->cap = cap;
    }
    list->items[list->count++] = s;
    (void)line_no;
    return true;
}

bool fsl_clone(const FreqSeriesList *src, FreqSeriesList *dst)
{
    if (src == NULL || dst == NULL) return false;
    fsl_init(dst);
    for (size_t i = 0; i < src->count; i++) {
        const FreqSession *s = &src->items[i];
        FreqSession c;

        memset(&c, 0, sizeof(c));
        c.label = dup_str(s->label);
        c.path = dup_str(s->path);
        c.datetime = dup_str(s->datetime);
        c.notes = dup_str(s->notes);
        c.lat = s->lat;
        c.lon = s->lon;
        c.has_location = s->has_location;
        if (s->count > 0) {
            c.freqs = malloc(s->count * sizeof(*c.freqs));
            if (c.freqs != NULL) {
                memcpy(c.freqs, s->freqs, s->count * sizeof(*c.freqs));
                c.count = s->count;
                c.cap = s->count;
            }
        }
        if (dst->count == dst->cap) {
            size_t cap = dst->cap == 0 ? 8 : dst->cap * 2;
            FreqSession *n = realloc(dst->items, cap * sizeof(*n));

            if (n == NULL) {
                fsl_clear(dst);
                return false;
            }
            dst->items = n;
            dst->cap = cap;
        }
        dst->items[dst->count++] = c;
    }
    return true;
}

void fsl_remove(FreqSeriesList *list, size_t index)
{
    if (list == NULL || index >= list->count) return;
    session_clear(&list->items[index]);
    if (index + 1 < list->count) {
        memmove(&list->items[index], &list->items[index + 1],
                (list->count - index - 1) * sizeof(*list->items));
    }
    list->count--;
    memset(&list->items[list->count], 0, sizeof(*list->items));
}

void fsl_set_label(FreqSession *s, const char *label)
{
    if (s == NULL) return;
    free(s->label);
    s->label = dup_str(label != NULL ? label : "");
}

void fsl_set_datetime(FreqSession *s, const char *datetime)
{
    if (s == NULL) return;
    free(s->datetime);
    s->datetime = dup_str(datetime != NULL ? datetime : "");
}

void fsl_set_notes(FreqSession *s, const char *notes)
{
    if (s == NULL) return;
    free(s->notes);
    s->notes = dup_str(notes != NULL ? notes : "");
}

void fsl_set_location(FreqSession *s, bool has_location, double lat,
                      double lon)
{
    if (s == NULL) return;
    s->has_location = has_location;
    s->lat = has_location ? lat : 0.0;
    s->lon = has_location ? lon : 0.0;
}

size_t fsl_total_points(const FreqSeriesList *list)
{
    size_t n = 0;

    if (list == NULL) return 0;
    for (size_t i = 0; i < list->count; i++) n += list->items[i].count;
    return n;
}
