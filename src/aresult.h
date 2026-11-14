/*
 * aresult.h — the common envelope every analysis publishes its output into.
 *
 * One AnalysisResult is allocated per analysis run (arena-backed) and holds:
 *   - the analysis identity (registry id/title) and its DAG node,
 *   - how it ended (executed / skipped / failed) with the reason,
 *   - its wall time and a 0..1 confidence the Director ranks with,
 *   - the estimated parameters as typed parameter→value entries (with the
 *     unit and a bracketed derivation note), and
 *   - its tunable settings (schema defaults now, user overrides later).
 *
 * The struct is deliberately GTK/UI free: analysis modules never touch it
 * directly (thin publishers in apublish.c do), and the Director consumes the
 * envelopes without knowing any analysis' private result struct.
 */
#ifndef TE15_ARESULT_H
#define TE15_ARESULT_H

#include <stdbool.h>
#include <stddef.h>

/* How an analysis ended. */
typedef enum {
    AR_EXECUTED = 0,
    AR_SKIPPED_USER,   /* not applicable / excluded by the DAG condition */
    AR_SKIPPED_DEP,    /* skipped because a dependency failed            */
    AR_FAILED          /* ran but produced no usable output              */
} ArStatus;

const char *ar_status_name(ArStatus s);

/* One estimated parameter. */
typedef enum { AR_VAL_NUMBER = 0, AR_VAL_TEXT } ArValKind;

typedef struct {
    const char *key;   /* "width_mean_us"                                  */
    ArValKind   kind;
    double      number;/* valid for AR_VAL_NUMBER                           */
    const char *text;  /* valid for AR_VAL_TEXT                             */
    const char *unit;  /* "us", "Hz", "dB", "%", NULL for dimensionless     */
    const char *note;  /* bracketed derivation, e.g. "envelope line spectrum" */
} ArParam;

/* One tunable setting (schema default now; user overrides later). */
typedef enum { AR_SET_NUMBER = 0, AR_SET_INT, AR_SET_BOOL, AR_SET_TEXT } ArSetKind;

typedef struct {
    const char *key;
    ArSetKind   kind;
    double      number;
    int         integer;
    bool        boolean;
    const char *text;
} ArSetting;

/* Forward declaration: the arena that owns the envelope's storage. */
typedef struct ArArena ArArena;

/* The envelope. */
typedef struct {
    const char *id;          /* registry id ("classic", "intraburst", ...)  */
    const char *title;       /* human title for reports                     */
    int         dag_node;    /* DAG node index, -1 when unassigned          */
    ArStatus    status;
    const char *status_note; /* why skipped/failed (NULL when executed)     */
    double      elapsed_s;
    double      confidence;  /* 0..1, the Director's weight                 */
    ArParam    *params;
    size_t      n_params, cap_params;
    ArSetting  *settings;
    size_t      n_settings, cap_settings;
    ArArena    *arena;       /* internal: parameter storage for this run */
} AnalysisResult;

/* ---- arenas --------------------------------------------------------- */

/* A small bump allocator: everything published during one analysis run is
 * freed in one call (no per-parameter frees, no leak paths on cancellation). */

ArArena *ar_arena_new(void);
void     ar_arena_free(ArArena *a);
void    *ar_arena_alloc(ArArena *a, size_t bytes);
char    *ar_arena_strdup(ArArena *a, const char *s);

/* ---- the collection of envelopes for one run ------------------------ */

typedef struct {
    AnalysisResult *v;
    size_t          n, cap;
    ArArena        *arena;
} ArSet;

void ar_set_init(ArSet *set, ArArena *arena);
void ar_set_clear(ArSet *set);

/* Add (or return the existing) envelope for an analysis id. */
AnalysisResult *ar_add(ArSet *set, const char *id, const char *title,
                       int dag_node);
AnalysisResult *ar_find(const ArSet *set, const char *id);

/* Publish parameters. Both helpers copy the strings into the arena. */
void ar_num(AnalysisResult *r, const char *key, double value,
            const char *unit, const char *note);
void ar_text(AnalysisResult *r, const char *key, const char *value,
             const char *note);

/* Read parameters back (used by the Director / tests). */
bool  ar_get_num(const AnalysisResult *r, const char *key, double *out);
bool  ar_get_text(const AnalysisResult *r, const char *key,
                  const char **out);

/* Settings (schema defaults from the registry, overrides later). */
void ar_setting_num(AnalysisResult *r, const char *key, double v);
void ar_setting_int(AnalysisResult *r, const char *key, int v);
void ar_setting_bool(AnalysisResult *r, const char *key, bool v);
void ar_setting_text(AnalysisResult *r, const char *key, const char *v);
const ArSetting *ar_find_setting(const AnalysisResult *r, const char *key);

/* Status bookkeeping. */
void ar_set_status(AnalysisResult *r, ArStatus st, const char *note);

#endif /* TE15_ARESULT_H */
