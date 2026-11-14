/*
 * fcdataset.c — load/validate/save measurement datasets (schema/fc_dataset.json).
 */
#include "fcdataset.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <json-c/json.h>

/* ------------------------------------------------------------------ */
/* small helpers                                                      */
/* ------------------------------------------------------------------ */

static void set_err(char *err, size_t err_len, const char *fmt, ...)
    __attribute__((format(printf, 3, 4)));

static void set_err(char *err, size_t err_len, const char *fmt, ...)
{
    va_list ap;

    if (err == NULL || err_len == 0) return;
    va_start(ap, fmt);
    vsnprintf(err, err_len, fmt, ap);
    va_end(ap);
}

static void *xcalloc(size_t n, size_t size)
{
    void *p = calloc(n > 0 ? n : 1, size);

    return p;
}

static char *xstrdup(const char *s)
{
    char *d;

    if (s == NULL) return NULL;
    d = strdup(s);
    return d;
}

static bool id_from_json(struct json_object *obj, FcId *id, const char *path,
                         char *err, size_t err_len)
{
    if (json_object_is_type(obj, json_type_string)) {
        const char *s = json_object_get_string(obj);

        snprintf(id->text, sizeof(id->text), "%s", s != NULL ? s : "");
        id->numeric = false;
        id->number = 0;
        return true;
    }
    if (json_object_is_type(obj, json_type_int)) {
        long long v = json_object_get_int64(obj);

        snprintf(id->text, sizeof(id->text), "%lld", v);
        id->numeric = true;
        id->number = v;
        return true;
    }
    set_err(err, err_len, "%s must be a string or an integer", path);
    return false;
}

static struct json_object *get_required(struct json_object *obj,
                                        const char *key, const char *path,
                                        char *err, size_t err_len)
{
    struct json_object *v = NULL;

    if (!json_object_object_get_ex(obj, key, &v)) {
        set_err(err, err_len, "missing required property '%s' at %s", key,
                path);
        return NULL;
    }
    return v;
}

static bool check_additional(struct json_object *obj, const char *path,
                             const char *const *allowed, char *err,
                             size_t err_len)
{
    json_object_object_foreach(obj, key, val)
    {
        bool ok = false;

        (void)val;
        for (const char *const *a = allowed; *a != NULL; a++) {
            if (strcmp(key, *a) == 0) {
                ok = true;
                break;
            }
        }
        if (!ok) {
            set_err(err, err_len,
                    "unsupported property '%s' at %s (additionalProperties "
                    "is false)", key, path);
            return false;
        }
    }
    return true;
}

static bool string_or_null(struct json_object *obj, const char *key,
                           const char *path, char **out, char *err,
                           size_t err_len)
{
    struct json_object *v = NULL;

    *out = NULL;
    if (!json_object_object_get_ex(obj, key, &v) ||
        json_object_is_type(v, json_type_null)) {
        return true;
    }
    if (!json_object_is_type(v, json_type_string)) {
        set_err(err, err_len, "%s.%s must be a string or null", path, key);
        return false;
    }
    *out = xstrdup(json_object_get_string(v));
    return true;
}

/* array of strings -> char** */
static bool string_array(struct json_object *obj, const char *key,
                         const char *path, char ***out, size_t *n_out,
                         bool *present, char *err, size_t err_len)
{
    struct json_object *v = NULL;
    size_t n = 0;

    *out = NULL;
    *n_out = 0;
    if (present != NULL) *present = false;
    if (!json_object_object_get_ex(obj, key, &v)) return true;
    if (present != NULL) *present = true;
    if (json_object_is_type(v, json_type_null)) return true;
    if (!json_object_is_type(v, json_type_array)) {
        set_err(err, err_len, "%s.%s must be an array of strings", path, key);
        return false;
    }
    n = (size_t)json_object_array_length(v);
    if (n == 0) return true;
    *out = xcalloc(n, sizeof(**out));
    if (*out == NULL) {
        set_err(err, err_len, "out of memory");
        return false;
    }
    for (size_t i = 0; i < n; i++) {
        struct json_object *item = json_object_array_get_idx(v, i);

        if (!json_object_is_type(item, json_type_string)) {
            set_err(err, err_len, "%s.%s[%zu] must be a string", path, key, i);
            for (size_t j = 0; j < i; j++) free((*out)[j]);
            free(*out);
            *out = NULL;
            *n_out = 0;
            return false;
        }
        (*out)[i] = xstrdup(json_object_get_string(item));
    }
    *n_out = n;
    return true;
}

/* closest_towers_enb_ids: array of string|integer, or null */
static bool tower_array(struct json_object *obj, const char *key,
                        const char *path, FcRound *round, char *err,
                        size_t err_len)
{
    struct json_object *v = NULL;
    size_t n;

    if (!json_object_object_get_ex(obj, key, &v)) {
        round->towers_null = false;
        return true;
    }
    if (json_object_is_type(v, json_type_null)) {
        round->towers_null = true;
        return true;
    }
    if (!json_object_is_type(v, json_type_array)) {
        set_err(err, err_len, "%s.%s must be an array or null", path, key);
        return false;
    }
    n = (size_t)json_object_array_length(v);
    if (n == 0) return true;
    round->towers = xcalloc(n, sizeof(*round->towers));
    if (round->towers == NULL) {
        set_err(err, err_len, "out of memory");
        return false;
    }
    for (size_t i = 0; i < n; i++) {
        struct json_object *item = json_object_array_get_idx(v, i);
        FcId id;

        if (!id_from_json(item, &id, path, err, err_len)) {
            for (size_t j = 0; j < i; j++) free(round->towers[j]);
            free(round->towers);
            round->towers = NULL;
            return false;
        }
        round->towers[i] = xstrdup(id.text);
    }
    round->n_towers = n;
    return true;
}

static bool measurement_array(struct json_object *obj, const char *key,
                              const char *path, double **out, size_t *n_out,
                              bool required, char *err, size_t err_len)
{
    struct json_object *v = NULL;
    size_t n;

    *out = NULL;
    *n_out = 0;
    if (required) {
        v = get_required(obj, key, path, err, err_len);
        if (v == NULL) return false;
    } else if (!json_object_object_get_ex(obj, key, &v) ||
               json_object_is_type(v, json_type_null)) {
        return true;
    }
    if (!json_object_is_type(v, json_type_array)) {
        set_err(err, err_len, "%s.%s must be an array", path, key);
        return false;
    }
    n = (size_t)json_object_array_length(v);
    if (n == 0) return true;
    *out = xcalloc(n, sizeof(**out));
    if (*out == NULL) {
        set_err(err, err_len, "out of memory");
        return false;
    }
    for (size_t i = 0; i < n; i++) {
        struct json_object *item = json_object_array_get_idx(v, i);

        if (!json_object_is_type(item, json_type_double) &&
            !json_object_is_type(item, json_type_int)) {
            set_err(err, err_len, "%s.%s[%zu] must be a number", path, key, i);
            free(*out);
            *out = NULL;
            return false;
        }
        (*out)[i] = json_object_get_double(item);
    }
    *n_out = n;
    return true;
}

/* ------------------------------------------------------------------ */
/* parsing per schema level                                           */
/* ------------------------------------------------------------------ */

static void subject_free(FcSubject *s)
{
    free(s->name);
    free(s->measurements);
    memset(s, 0, sizeof(*s));
}

static void round_free(FcRound *r)
{
    free(r->location);
    free(r->date);
    for (size_t i = 0; i < r->n_towers; i++) free(r->towers[i]);
    free(r->towers);
    free(r->context);
    for (size_t i = 0; i < r->n_listed; i++) free(r->subjects_listed[i]);
    free(r->subjects_listed);
    for (size_t i = 0; i < r->n_subjects; i++) subject_free(&r->subjects[i]);
    free(r->subjects);
    memset(r, 0, sizeof(*r));
}

static void session_free(FcSession *s)
{
    for (size_t i = 0; i < s->n_rounds; i++) round_free(&s->rounds[i]);
    free(s->rounds);
    memset(s, 0, sizeof(*s));
}

static void dataset_free(FcDataset *d)
{
    free(d->dataset_id);
    free(d->source_file);
    free(d->title);
    for (size_t i = 0; i < d->n_notes; i++) free(d->notes[i]);
    free(d->notes);
    for (size_t i = 0; i < d->n_sessions; i++) session_free(&d->sessions[i]);
    free(d->sessions);
    free(d->unassigned);
    memset(d, 0, sizeof(*d));
}

static bool parse_subject(struct json_object *obj, FcSubject *subj,
                          const char *path, char *err, size_t err_len)
{
    static const char *const allowed[] = { "name", "measurements", NULL };
    struct json_object *v;

    memset(subj, 0, sizeof(*subj));
    if (!json_object_is_type(obj, json_type_object)) {
        set_err(err, err_len, "%s must be an object", path);
        return false;
    }
    if (!check_additional(obj, path, allowed, err, err_len)) return false;
    v = get_required(obj, "name", path, err, err_len);
    if (v == NULL) return false;
    if (!json_object_is_type(v, json_type_string)) {
        set_err(err, err_len, "%s.name must be a string", path);
        return false;
    }
    subj->name = xstrdup(json_object_get_string(v));
    if (!measurement_array(obj, "measurements", path, &subj->measurements,
                           &subj->count, true, err, err_len)) {
        subject_free(subj);
        return false;
    }
    return true;
}

static bool parse_round(struct json_object *obj, FcRound *round,
                        const char *path, char *err, size_t err_len)
{
    static const char *const allowed[] = {
        "round_id", "location", "date", "coordinates",
        "closest_towers_enb_ids", "context", "subjects_listed", "subjects",
        NULL
    };
    struct json_object *v, *coords;

    memset(round, 0, sizeof(*round));
    if (!json_object_is_type(obj, json_type_object)) {
        set_err(err, err_len, "%s must be an object", path);
        return false;
    }
    if (!check_additional(obj, path, allowed, err, err_len)) return false;
    v = get_required(obj, "round_id", path, err, err_len);
    if (v == NULL || !id_from_json(v, &round->round_id, path, err, err_len)) {
        if (v != NULL) {
            set_err(err, err_len, "%s.round_id must be a string or integer",
                    path);
        }
        return false;
    }
    if (!string_or_null(obj, "location", path, &round->location, err,
                        err_len) ||
        !string_or_null(obj, "date", path, &round->date, err, err_len) ||
        !string_or_null(obj, "context", path, &round->context, err, err_len)) {
        return false;
    }
    if (json_object_object_get_ex(obj, "coordinates", &coords) &&
        !json_object_is_type(coords, json_type_null)) {
        static const char *const callowed[] = { "latitude", "longitude", NULL };
        struct json_object *lat, *lon;
        char cpath[256];

        snprintf(cpath, sizeof(cpath), "%s.coordinates", path);
        if (!json_object_is_type(coords, json_type_object)) {
            set_err(err, err_len, "%s.coordinates must be an object or null",
                    path);
            return false;
        }
        if (!check_additional(coords, cpath, callowed, err, err_len)) {
            return false;
        }
        lat = get_required(coords, "latitude", cpath, err, err_len);
        lon = get_required(coords, "longitude", cpath, err, err_len);
        if (lat == NULL || lon == NULL) return false;
        if ((!json_object_is_type(lat, json_type_double) &&
             !json_object_is_type(lat, json_type_int)) ||
            (!json_object_is_type(lon, json_type_double) &&
             !json_object_is_type(lon, json_type_int))) {
            set_err(err, err_len, "%s.coordinates latitude/longitude must be "
                                  "numbers", path);
            return false;
        }
        round->has_coordinates = true;
        round->latitude = json_object_get_double(lat);
        round->longitude = json_object_get_double(lon);
    }
    if (!tower_array(obj, "closest_towers_enb_ids", path, round, err,
                     err_len)) {
        return false;
    }
    if (!string_array(obj, "subjects_listed", path, &round->subjects_listed,
                      &round->n_listed, NULL, err, err_len)) {
        return false;
    }
    v = get_required(obj, "subjects", path, err, err_len);
    if (v == NULL) return false;
    if (!json_object_is_type(v, json_type_array)) {
        set_err(err, err_len, "%s.subjects must be an array", path);
        return false;
    }
    {
        size_t n = (size_t)json_object_array_length(v);

        if (n > 0) {
            round->subjects = xcalloc(n, sizeof(*round->subjects));
            if (round->subjects == NULL) {
                set_err(err, err_len, "out of memory");
                return false;
            }
            for (size_t i = 0; i < n; i++) {
                char spath[256];

                snprintf(spath, sizeof(spath), "%s.subjects[%zu]", path, i);
                if (!parse_subject(json_object_array_get_idx(v, i),
                                   &round->subjects[i], spath, err, err_len)) {
                    round_free(round);
                    return false;
                }
                round->n_subjects++;
            }
        }
    }
    return true;
}

static bool parse_session(struct json_object *obj, FcSession *session,
                          const char *path, char *err, size_t err_len)
{
    static const char *const allowed[] = { "session_id", "rounds", NULL };
    struct json_object *v;

    memset(session, 0, sizeof(*session));
    if (!json_object_is_type(obj, json_type_object)) {
        set_err(err, err_len, "%s must be an object", path);
        return false;
    }
    if (!check_additional(obj, path, allowed, err, err_len)) return false;
    v = get_required(obj, "session_id", path, err, err_len);
    if (v == NULL || !id_from_json(v, &session->session_id, path, err,
                                   err_len)) {
        if (v != NULL) {
            set_err(err, err_len,
                    "%s.session_id must be a string or integer", path);
        }
        return false;
    }
    v = get_required(obj, "rounds", path, err, err_len);
    if (v == NULL) return false;
    if (!json_object_is_type(v, json_type_array)) {
        set_err(err, err_len, "%s.rounds must be an array", path);
        return false;
    }
    {
        size_t n = (size_t)json_object_array_length(v);

        if (n > 0) {
            session->rounds = xcalloc(n, sizeof(*session->rounds));
            if (session->rounds == NULL) {
                set_err(err, err_len, "out of memory");
                return false;
            }
            for (size_t i = 0; i < n; i++) {
                char rpath[256];

                snprintf(rpath, sizeof(rpath), "%s.rounds[%zu]", path, i);
                if (!parse_round(json_object_array_get_idx(v, i),
                                 &session->rounds[i], rpath, err, err_len)) {
                    session_free(session);
                    return false;
                }
                session->n_rounds++;
            }
        }
    }
    return true;
}

static bool parse_dataset(struct json_object *obj, FcDataset *d, char *err,
                          size_t err_len)
{
    static const char *const allowed[] = {
        "dataset_id", "source_file", "title", "notes", "sessions",
        "unassigned_measurements", NULL
    };
    struct json_object *v;

    memset(d, 0, sizeof(*d));
    if (!json_object_is_type(obj, json_type_object)) {
        set_err(err, err_len, "the dataset must be a JSON object");
        return false;
    }
    if (!check_additional(obj, "$", allowed, err, err_len)) return false;
    v = get_required(obj, "dataset_id", "$", err, err_len);
    if (v == NULL) return false;
    if (!json_object_is_type(v, json_type_string)) {
        set_err(err, err_len, "$.dataset_id must be a string");
        return false;
    }
    d->dataset_id = xstrdup(json_object_get_string(v));
    v = get_required(obj, "source_file", "$", err, err_len);
    if (v == NULL) return false;
    if (!json_object_is_type(v, json_type_string)) {
        set_err(err, err_len, "$.source_file must be a string");
        return false;
    }
    d->source_file = xstrdup(json_object_get_string(v));
    v = get_required(obj, "title", "$", err, err_len);
    if (v == NULL) return false;
    if (!json_object_is_type(v, json_type_string)) {
        set_err(err, err_len, "$.title must be a string");
        return false;
    }
    d->title = xstrdup(json_object_get_string(v));
    if (!string_array(obj, "notes", "$", &d->notes, &d->n_notes, NULL, err,
                      err_len)) {
        return false;
    }
    v = get_required(obj, "sessions", "$", err, err_len);
    if (v == NULL) return false;
    if (!json_object_is_type(v, json_type_array)) {
        set_err(err, err_len, "$.sessions must be an array");
        return false;
    }
    {
        size_t n = (size_t)json_object_array_length(v);

        if (n > 0) {
            d->sessions = xcalloc(n, sizeof(*d->sessions));
            if (d->sessions == NULL) {
                set_err(err, err_len, "out of memory");
                return false;
            }
            for (size_t i = 0; i < n; i++) {
                char spath[256];

                snprintf(spath, sizeof(spath), "$.sessions[%zu]", i);
                if (!parse_session(json_object_array_get_idx(v, i),
                                   &d->sessions[i], spath, err, err_len)) {
                    dataset_free(d);
                    return false;
                }
                d->n_sessions++;
            }
        }
    }
    if (!measurement_array(obj, "unassigned_measurements", "$", &d->unassigned,
                           &d->n_unassigned, false, err, err_len)) {
        dataset_free(d);
        return false;
    }
    return true;
}

/* ------------------------------------------------------------------ */
/* loading                                                            */
/* ------------------------------------------------------------------ */

bool fcd_load_file(FcDatasetList *list, const char *path, char *err,
                   size_t err_len)
{
    struct json_object *doc;
    struct json_tokener *tok;
    char *text = NULL;
    long len = 0;
    FILE *f;
    FcDataset d;
    bool ok = false;

    if (list == NULL || path == NULL) return false;
    if (err != NULL && err_len > 0) err[0] = '\0';
    f = fopen(path, "rb");
    if (f == NULL) {
        set_err(err, err_len, "cannot open %s", path);
        return false;
    }
    if (fseek(f, 0, SEEK_END) != 0 || (len = ftell(f)) < 0 ||
        fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        set_err(err, err_len, "cannot read %s", path);
        return false;
    }
    text = malloc((size_t)len + 1);
    if (text == NULL) {
        fclose(f);
        set_err(err, err_len, "out of memory");
        return false;
    }
    if (fread(text, 1, (size_t)len, f) != (size_t)len) {
        fclose(f);
        free(text);
        set_err(err, err_len, "cannot read %s", path);
        return false;
    }
    fclose(f);
    text[len] = '\0';
    tok = json_tokener_new();
    if (tok == NULL) {
        free(text);
        set_err(err, err_len, "out of memory");
        return false;
    }
    doc = json_tokener_parse_ex(tok, text, (int)len);
    if (doc == NULL || json_tokener_get_error(tok) != json_tokener_success) {
        set_err(err, err_len, "%s is not valid JSON (offset %zu)", path,
                json_tokener_get_parse_end(tok));
        if (doc != NULL) json_object_put(doc);
        json_tokener_free(tok);
        free(text);
        return false;
    }
    json_tokener_free(tok);
    free(text);
    ok = parse_dataset(doc, &d, err, err_len);
    json_object_put(doc);
    if (!ok) return false;
    if (list->count == list->cap) {
        size_t cap = list->cap == 0 ? 8 : list->cap * 2;
        FcDataset *n = realloc(list->items, cap * sizeof(*n));

        if (n == NULL) {
            dataset_free(&d);
            set_err(err, err_len, "out of memory");
            return false;
        }
        list->items = n;
        list->cap = cap;
    }
    list->items[list->count++] = d;
    return true;
}

void fcd_list_init(FcDatasetList *list)
{
    if (list != NULL) memset(list, 0, sizeof(*list));
}

void fcd_list_clear(FcDatasetList *list)
{
    if (list == NULL) return;
    for (size_t i = 0; i < list->count; i++) dataset_free(&list->items[i]);
    free(list->items);
    memset(list, 0, sizeof(*list));
}

void fcd_list_remove(FcDatasetList *list, size_t index)
{
    if (list == NULL || index >= list->count) return;
    dataset_free(&list->items[index]);
    if (index + 1 < list->count) {
        memmove(&list->items[index], &list->items[index + 1],
                (list->count - index - 1) * sizeof(*list->items));
    }
    list->count--;
    memset(&list->items[list->count], 0, sizeof(*list->items));
}

/* ------------------------------------------------------------------ */
/* deep copy                                                          */
/* ------------------------------------------------------------------ */

static bool copy_subject(const FcSubject *s, FcSubject *d)
{
    memset(d, 0, sizeof(*d));
    d->name = xstrdup(s->name);
    if (s->count > 0) {
        d->measurements = malloc(s->count * sizeof(*d->measurements));
        if (d->measurements == NULL) return false;
        memcpy(d->measurements, s->measurements,
               s->count * sizeof(*d->measurements));
        d->count = d->cap = s->count;
    }
    return true;
}

static bool copy_round(const FcRound *s, FcRound *d)
{
    memset(d, 0, sizeof(*d));
    d->round_id = s->round_id;
    d->location = xstrdup(s->location);
    d->date = xstrdup(s->date);
    d->has_coordinates = s->has_coordinates;
    d->latitude = s->latitude;
    d->longitude = s->longitude;
    d->towers_null = s->towers_null;
    if (s->n_towers > 0) {
        d->towers = xcalloc(s->n_towers, sizeof(*d->towers));
        if (d->towers == NULL) return false;
        for (size_t i = 0; i < s->n_towers; i++) {
            d->towers[i] = xstrdup(s->towers[i]);
        }
        d->n_towers = s->n_towers;
    }
    d->context = xstrdup(s->context);
    if (s->n_listed > 0) {
        d->subjects_listed = xcalloc(s->n_listed, sizeof(*d->subjects_listed));
        if (d->subjects_listed == NULL) return false;
        for (size_t i = 0; i < s->n_listed; i++) {
            d->subjects_listed[i] = xstrdup(s->subjects_listed[i]);
        }
        d->n_listed = s->n_listed;
    }
    if (s->n_subjects > 0) {
        d->subjects = xcalloc(s->n_subjects, sizeof(*d->subjects));
        if (d->subjects == NULL) return false;
        for (size_t i = 0; i < s->n_subjects; i++) {
            if (!copy_subject(&s->subjects[i], &d->subjects[i])) return false;
            d->n_subjects++;
        }
    }
    return true;
}

static bool copy_dataset(const FcDataset *s, FcDataset *d)
{
    memset(d, 0, sizeof(*d));
    d->dataset_id = xstrdup(s->dataset_id);
    d->source_file = xstrdup(s->source_file);
    d->title = xstrdup(s->title);
    if (s->n_notes > 0) {
        d->notes = xcalloc(s->n_notes, sizeof(*d->notes));
        if (d->notes == NULL) return false;
        for (size_t i = 0; i < s->n_notes; i++) {
            d->notes[i] = xstrdup(s->notes[i]);
        }
        d->n_notes = s->n_notes;
    }
    if (s->n_sessions > 0) {
        d->sessions = xcalloc(s->n_sessions, sizeof(*d->sessions));
        if (d->sessions == NULL) return false;
        for (size_t i = 0; i < s->n_sessions; i++) {
            const FcSession *ss = &s->sessions[i];
            FcSession *sd = &d->sessions[i];

            sd->session_id = ss->session_id;
            if (ss->n_rounds > 0) {
                sd->rounds = xcalloc(ss->n_rounds, sizeof(*sd->rounds));
                if (sd->rounds == NULL) return false;
                for (size_t r = 0; r < ss->n_rounds; r++) {
                    if (!copy_round(&ss->rounds[r], &sd->rounds[r])) {
                        return false;
                    }
                    sd->n_rounds++;
                }
            }
            d->n_sessions++;
        }
    }
    if (s->n_unassigned > 0) {
        d->unassigned = malloc(s->n_unassigned * sizeof(*d->unassigned));
        if (d->unassigned == NULL) return false;
        memcpy(d->unassigned, s->unassigned,
               s->n_unassigned * sizeof(*d->unassigned));
        d->n_unassigned = s->n_unassigned;
    }
    return true;
}

bool fcd_list_clone(const FcDatasetList *src, FcDatasetList *dst)
{
    if (src == NULL || dst == NULL) return false;
    fcd_list_init(dst);
    for (size_t i = 0; i < src->count; i++) {
        FcDataset d;

        if (!copy_dataset(&src->items[i], &d)) {
            fcd_list_clear(dst);
            return false;
        }
        if (dst->count == dst->cap) {
            size_t cap = dst->cap == 0 ? 8 : dst->cap * 2;
            FcDataset *n = realloc(dst->items, cap * sizeof(*n));

            if (n == NULL) {
                dataset_free(&d);
                fcd_list_clear(dst);
                return false;
            }
            dst->items = n;
            dst->cap = cap;
        }
        dst->items[dst->count++] = d;
    }
    return true;
}

/* ------------------------------------------------------------------ */
/* saving                                                             */
/* ------------------------------------------------------------------ */

static struct json_object *id_to_json(const FcId *id)
{
    if (id->numeric) return json_object_new_int64(id->number);
    return json_object_new_string(id->text);
}

static void put_string_or_null(struct json_object *obj, const char *key,
                               const char *value)
{
    if (value == NULL) {
        json_object_object_add(obj, key, NULL);   /* creates JSON null */
    } else {
        json_object_object_add(obj, key, json_object_new_string(value));
    }
}

static struct json_object *subject_to_json(const FcSubject *s)
{
    struct json_object *o = json_object_new_object();
    struct json_object *arr = json_object_new_array();

    json_object_object_add(o, "name", json_object_new_string(s->name));
    for (size_t i = 0; i < s->count; i++) {
        json_object_array_add(arr, json_object_new_double(s->measurements[i]));
    }
    json_object_object_add(o, "measurements", arr);
    return o;
}

static struct json_object *round_to_json(const FcRound *r)
{
    struct json_object *o = json_object_new_object();
    struct json_object *arr = json_object_new_array();

    json_object_object_add(o, "round_id", id_to_json(&r->round_id));
    put_string_or_null(o, "location", r->location);
    put_string_or_null(o, "date", r->date);
    if (r->has_coordinates) {
        struct json_object *c = json_object_new_object();

        json_object_object_add(c, "latitude",
                               json_object_new_double(r->latitude));
        json_object_object_add(c, "longitude",
                               json_object_new_double(r->longitude));
        json_object_object_add(o, "coordinates", c);
    } else {
        json_object_object_add(o, "coordinates", NULL);
    }
    if (r->towers_null) {
        json_object_object_add(o, "closest_towers_enb_ids", NULL);
    } else {
        struct json_object *towers = json_object_new_array();

        for (size_t i = 0; i < r->n_towers; i++) {
            json_object_array_add(towers,
                                  json_object_new_string(r->towers[i]));
        }
        json_object_object_add(o, "closest_towers_enb_ids", towers);
    }
    put_string_or_null(o, "context", r->context);
    arr = json_object_new_array();
    for (size_t i = 0; i < r->n_listed; i++) {
        json_object_array_add(arr,
                              json_object_new_string(r->subjects_listed[i]));
    }
    json_object_object_add(o, "subjects_listed", arr);
    arr = json_object_new_array();
    for (size_t i = 0; i < r->n_subjects; i++) {
        json_object_array_add(arr, subject_to_json(&r->subjects[i]));
    }
    json_object_object_add(o, "subjects", arr);
    return o;
}

static struct json_object *dataset_to_json(const FcDataset *d)
{
    struct json_object *o = json_object_new_object();
    struct json_object *arr = json_object_new_array();

    json_object_object_add(o, "dataset_id", json_object_new_string(d->dataset_id));
    json_object_object_add(o, "source_file",
                           json_object_new_string(d->source_file));
    json_object_object_add(o, "title", json_object_new_string(d->title));
    arr = json_object_new_array();
    for (size_t i = 0; i < d->n_notes; i++) {
        json_object_array_add(arr, json_object_new_string(d->notes[i]));
    }
    json_object_object_add(o, "notes", arr);
    arr = json_object_new_array();
    for (size_t s = 0; s < d->n_sessions; s++) {
        const FcSession *ss = &d->sessions[s];
        struct json_object *so = json_object_new_object();
        struct json_object *rounds = json_object_new_array();

        json_object_object_add(so, "session_id", id_to_json(&ss->session_id));
        for (size_t r = 0; r < ss->n_rounds; r++) {
            json_object_array_add(rounds, round_to_json(&ss->rounds[r]));
        }
        json_object_object_add(so, "rounds", rounds);
        json_object_array_add(arr, so);
    }
    json_object_object_add(o, "sessions", arr);
    arr = json_object_new_array();
    for (size_t i = 0; i < d->n_unassigned; i++) {
        json_object_array_add(arr,
                              json_object_new_double(d->unassigned[i]));
    }
    json_object_object_add(o, "unassigned_measurements", arr);
    return o;
}

bool fcd_save_file(const FcDataset *dataset, const char *path, char *err,
                   size_t err_len)
{
    struct json_object *doc;
    const char *text;
    FILE *f;
    bool ok;

    if (dataset == NULL || path == NULL) return false;
    if (err != NULL && err_len > 0) err[0] = '\0';
    doc = dataset_to_json(dataset);
    if (doc == NULL) {
        set_err(err, err_len, "out of memory");
        return false;
    }
    text = json_object_to_json_string_ext(doc, JSON_C_TO_STRING_PRETTY |
                                                  JSON_C_TO_STRING_NOSLASHESCAPE);
    f = fopen(path, "wb");
    if (f == NULL) {
        set_err(err, err_len, "cannot write %s", path);
        json_object_put(doc);
        return false;
    }
    ok = fputs(text, f) >= 0 && fputc('\n', f) != EOF;
    if (fclose(f) != 0) ok = false;
    json_object_put(doc);
    if (!ok) set_err(err, err_len, "cannot write %s", path);
    return ok;
}

/* ------------------------------------------------------------------ */
/* statistics                                                         */
/* ------------------------------------------------------------------ */

size_t fcd_count_rounds(const FcDataset *dataset)
{
    size_t n = 0;

    if (dataset == NULL) return 0;
    for (size_t s = 0; s < dataset->n_sessions; s++) {
        n += dataset->sessions[s].n_rounds;
    }
    return n;
}

size_t fcd_count_subjects(const FcDataset *dataset)
{
    size_t n = 0;

    if (dataset == NULL) return 0;
    for (size_t s = 0; s < dataset->n_sessions; s++) {
        const FcSession *ss = &dataset->sessions[s];

        for (size_t r = 0; r < ss->n_rounds; r++) {
            n += ss->rounds[r].n_subjects;
        }
    }
    return n;
}

size_t fcd_count_measurements(const FcDataset *dataset)
{
    size_t n = 0;

    if (dataset == NULL) return 0;
    n += dataset->n_unassigned;
    for (size_t s = 0; s < dataset->n_sessions; s++) {
        const FcSession *ss = &dataset->sessions[s];

        for (size_t r = 0; r < ss->n_rounds; r++) {
            const FcRound *rr = &ss->rounds[r];

            for (size_t j = 0; j < rr->n_subjects; j++) {
                n += rr->subjects[j].count;
            }
        }
    }
    return n;
}

size_t fcd_list_count_sessions(const FcDatasetList *list)
{
    size_t n = 0;

    if (list == NULL) return 0;
    for (size_t i = 0; i < list->count; i++) n += list->items[i].n_sessions;
    return n;
}

size_t fcd_list_count_rounds(const FcDatasetList *list)
{
    size_t n = 0;

    if (list == NULL) return 0;
    for (size_t i = 0; i < list->count; i++) n += fcd_count_rounds(&list->items[i]);
    return n;
}

size_t fcd_list_count_subjects(const FcDatasetList *list)
{
    size_t n = 0;

    if (list == NULL) return 0;
    for (size_t i = 0; i < list->count; i++) n += fcd_count_subjects(&list->items[i]);
    return n;
}

size_t fcd_list_count_measurements(const FcDatasetList *list)
{
    size_t n = 0;

    if (list == NULL) return 0;
    for (size_t i = 0; i < list->count; i++) {
        n += fcd_count_measurements(&list->items[i]);
    }
    return n;
}

bool fc_stats(const double *values, size_t n, double *min_hz, double *max_hz,
              double *mean_hz)
{
    double sum = 0.0;

    if (values == NULL || n == 0) return false;
    for (size_t i = 0; i < n; i++) sum += values[i];
    if (min_hz != NULL) {
        double v = values[0];

        for (size_t i = 1; i < n; i++) {
            if (values[i] < v) v = values[i];
        }
        *min_hz = v;
    }
    if (max_hz != NULL) {
        double v = values[0];

        for (size_t i = 1; i < n; i++) {
            if (values[i] > v) v = values[i];
        }
        *max_hz = v;
    }
    if (mean_hz != NULL) *mean_hz = sum / (double)n;
    return true;
}
