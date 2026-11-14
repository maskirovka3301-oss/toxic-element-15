/*
 * aresult.c — envelope, arena and parameter storage (see aresult.h).
 */
#include "aresult.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

const char *ar_status_name(ArStatus s)
{
    switch (s) {
    case AR_EXECUTED:     return "executed";
    case AR_SKIPPED_USER: return "skipped";
    case AR_SKIPPED_DEP:  return "skipped (dependency)";
    case AR_FAILED:       return "failed";
    }
    return "unknown";
}

/* ---- arena: 4 KB blocks, aligned ------------------------------------ */

typedef struct ArBlock {
    struct ArBlock *next;
    size_t          used, cap;
    unsigned char   data[1];
} ArBlock;

struct ArArena {
    ArBlock *head;
};

#define AR_BLOCK_BYTES 4096u

ArArena *ar_arena_new(void)
{
    ArArena *a = calloc(1, sizeof(*a));

    return a;
}

static void *arena_alloc_raw(ArArena *a, size_t bytes)
{
    ArBlock *b;
    void *out;

    bytes = (bytes + 15u) & ~(size_t)15u;
    if (a == NULL) return NULL;
    if (a->head == NULL || a->head->used + bytes > a->head->cap) {
        size_t cap = bytes > AR_BLOCK_BYTES ? bytes : AR_BLOCK_BYTES;
        ArBlock *nb = calloc(1, sizeof(ArBlock) + cap);

        if (nb == NULL) return NULL;
        nb->cap = cap;
        nb->next = a->head;
        a->head = nb;
    }
    b = a->head;
    out = b->data + b->used;
    b->used += bytes;
    return out;
}

void *ar_arena_alloc(ArArena *a, size_t bytes)
{
    return arena_alloc_raw(a, bytes);
}

char *ar_arena_strdup(ArArena *a, const char *s)
{
    size_t n;
    char *p;

    if (s == NULL) return NULL;
    n = strlen(s) + 1;
    p = arena_alloc_raw(a, n);
    if (p != NULL) memcpy(p, s, n);
    return p;
}

void ar_arena_free(ArArena *a)
{
    if (a == NULL) return;
    {
        ArBlock *b = a->head;

        while (b != NULL) {
            ArBlock *next = b->next;

            free(b);
            b = next;
        }
    }
    free(a);
}

/* ---- envelope collection -------------------------------------------- */

void ar_set_init(ArSet *set, ArArena *arena)
{
    memset(set, 0, sizeof(*set));
    set->arena = arena;
}

void ar_set_clear(ArSet *set)
{
    if (set == NULL) return;
    memset(set, 0, sizeof(*set));
}

AnalysisResult *ar_find(const ArSet *set, const char *id)
{
    if (set == NULL || id == NULL) return NULL;
    for (size_t i = 0; i < set->n; i++) {
        if (strcmp(set->v[i].id, id) == 0) return &set->v[i];
    }
    return NULL;
}

AnalysisResult *ar_add(ArSet *set, const char *id, const char *title,
                       int dag_node)
{
    AnalysisResult *r = ar_find(set, id);

    if (r != NULL) return r;
    if (set == NULL || set->arena == NULL) return NULL;
    if (set->n >= set->cap) {
        size_t ncap = set->cap ? set->cap * 2 : 16;
        AnalysisResult *nv = ar_arena_alloc(set->arena,
                                            ncap * sizeof(AnalysisResult));

        if (nv == NULL) return NULL;
        if (set->n > 0) memcpy(nv, set->v, set->n * sizeof(AnalysisResult));
        set->v = nv;
        set->cap = ncap;
    }
    r = &set->v[set->n++];
    memset(r, 0, sizeof(*r));
    r->id = ar_arena_strdup(set->arena, id);
    r->title = ar_arena_strdup(set->arena, title != NULL ? title : id);
    r->dag_node = dag_node;
    r->status = AR_EXECUTED;
    r->arena = set->arena;
    return r;
}

/* ---- parameters ------------------------------------------------------ */

static ArParam *param_add(AnalysisResult *r, const char *key, ArValKind kind,
                          const char *note)
{
    ArParam *p;

    if (r == NULL || r->arena == NULL || key == NULL) return NULL;
    if (r->n_params >= r->cap_params) {
        size_t ncap = r->cap_params ? r->cap_params * 2 : 24;
        ArParam *nv = ar_arena_alloc(r->arena, ncap * sizeof(ArParam));

        if (nv == NULL) return NULL;
        if (r->n_params > 0) {
            memcpy(nv, r->params, r->n_params * sizeof(ArParam));
        }
        r->params = nv;
        r->cap_params = ncap;
    }
    p = &r->params[r->n_params++];
    memset(p, 0, sizeof(*p));
    p->key = ar_arena_strdup(r->arena, key);
    p->kind = kind;
    p->note = ar_arena_strdup(r->arena, note);
    return p;
}

void ar_num(AnalysisResult *r, const char *key, double value,
            const char *unit, const char *note)
{
    ArParam *p = param_add(r, key, AR_VAL_NUMBER, note);

    if (p == NULL) return;
    p->number = value;
    p->unit = ar_arena_strdup(r->arena, unit);
}

void ar_text(AnalysisResult *r, const char *key, const char *value,
             const char *note)
{
    ArParam *p = param_add(r, key, AR_VAL_TEXT, note);

    if (p == NULL) return;
    p->text = ar_arena_strdup(r->arena, value);
}

static const ArParam *param_find(const AnalysisResult *r, const char *key)
{
    if (r == NULL || key == NULL) return NULL;
    for (size_t i = 0; i < r->n_params; i++) {
        if (strcmp(r->params[i].key, key) == 0) return &r->params[i];
    }
    return NULL;
}

bool ar_get_num(const AnalysisResult *r, const char *key, double *out)
{
    const ArParam *p = param_find(r, key);

    if (p == NULL || p->kind != AR_VAL_NUMBER) return false;
    if (out != NULL) *out = p->number;
    return true;
}

bool ar_get_text(const AnalysisResult *r, const char *key, const char **out)
{
    const ArParam *p = param_find(r, key);

    if (p == NULL || p->kind != AR_VAL_TEXT) return false;
    if (out != NULL) *out = p->text;
    return true;
}

/* ---- settings -------------------------------------------------------- */

static ArSetting *setting_add(AnalysisResult *r, const char *key,
                              ArSetKind kind)
{
    ArSetting *s;

    if (r == NULL || r->arena == NULL || key == NULL) return NULL;
    if (r->n_settings >= r->cap_settings) {
        size_t ncap = r->cap_settings ? r->cap_settings * 2 : 16;
        ArSetting *nv = ar_arena_alloc(r->arena, ncap * sizeof(ArSetting));

        if (nv == NULL) return NULL;
        if (r->n_settings > 0) {
            memcpy(nv, r->settings, r->n_settings * sizeof(ArSetting));
        }
        r->settings = nv;
        r->cap_settings = ncap;
    }
    s = &r->settings[r->n_settings++];
    memset(s, 0, sizeof(*s));
    s->key = ar_arena_strdup(r->arena, key);
    s->kind = kind;
    return s;
}

void ar_setting_num(AnalysisResult *r, const char *key, double v)
{
    ArSetting *s = setting_add(r, key, AR_SET_NUMBER);

    if (s != NULL) s->number = v;
}

void ar_setting_int(AnalysisResult *r, const char *key, int v)
{
    ArSetting *s = setting_add(r, key, AR_SET_INT);

    if (s != NULL) s->integer = v;
}

void ar_setting_bool(AnalysisResult *r, const char *key, bool v)
{
    ArSetting *s = setting_add(r, key, AR_SET_BOOL);

    if (s != NULL) s->boolean = v;
}

void ar_setting_text(AnalysisResult *r, const char *key, const char *v)
{
    ArSetting *s = setting_add(r, key, AR_SET_TEXT);

    if (s != NULL) s->text = ar_arena_strdup(r->arena, v);
}

const ArSetting *ar_find_setting(const AnalysisResult *r, const char *key)
{
    if (r == NULL || key == NULL) return NULL;
    for (size_t i = 0; i < r->n_settings; i++) {
        if (strcmp(r->settings[i].key, key) == 0) return &r->settings[i];
    }
    return NULL;
}

void ar_set_status(AnalysisResult *r, ArStatus st, const char *note)
{
    if (r == NULL) return;
    r->status = st;
    r->status_note = ar_arena_strdup(r->arena, note);
}
