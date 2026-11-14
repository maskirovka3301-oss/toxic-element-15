/*
 * dagcore.h — generic, reusable analysis-DAG engine.
 *
 * A DAG is a registry of nodes (id, title, optional prerequisite id) plus
 * one run of that registry. Two usage modes share the same state model:
 *
 *   - explicit: the caller drives an imperative pipeline and records each
 *     step with dagcore_begin()/dagcore_end()/dagcore_skip() (the Spectrum
 *     tab's worker does this: its branching is data dependent);
 *   - registry: dagcore_execute() walks the registry, checks prerequisites,
 *     asks the executor callback to run each node and records the outcome
 *     (the Frequency Counter tab's backend does this).
 *
 * Neither the payload nor the meaning of a node is known here: this module
 * only owns node identities, dependency order, statuses, skip rationales,
 * timings, cancellation and progress, so every analysis front end shares
 * exactly one implementation.
 */
#ifndef TE15_DAGCORE_H
#define TE15_DAGCORE_H

#include <stdbool.h>
#include <stddef.h>

typedef enum {
    DAGCORE_PENDING = 0,
    DAGCORE_RUNNING,
    DAGCORE_DONE,
    DAGCORE_SKIPPED,
    DAGCORE_FAILED
} DagCoreStatus;

/* One analysis in a registry. */
typedef struct {
    const char *id;        /* unique identifier, e.g. "a0-baseline"       */
    const char *title;     /* human-readable name (reports/UI)            */
    const char *requires;  /* prerequisite node id(s), comma separated,
                            * or NULL when the node has no prerequisite    */
} DagCoreNodeSpec;

/* One node of a run: its registry entry, outcome and free-form note. */
typedef struct {
    const DagCoreNodeSpec *spec;
    DagCoreStatus          status;
    char                   note[192];  /* skip/fail reason, "" on success */
    double                 elapsed_s;
    void                  *payload;    /* opaque, owned by the caller     */
} DagCoreNode;

/* A complete run of a registry. */
typedef struct {
    DagCoreNode *nodes;
    size_t       count;
    double       elapsed_s;
    bool         cancelled;
} DagCoreRun;

/* Hooks for registry mode (all optional except execute). */
typedef struct {
    /* Run one node; return true when it executed. On false fill `err` with
     * the failure reason (recorded in the node's note). */
    bool (*execute)(void *ctx, DagCoreNode *node, char *err, size_t err_len);
    /* Data-dependent skip: return a reason to skip the node, NULL to run. */
    const char *(*skip_reason)(void *ctx, const DagCoreNodeSpec *spec);
    /* Called before each node with (index, count, title). */
    void (*on_node)(void *ctx, size_t index, size_t count, const char *title);
    /* Cooperative cancellation. */
    bool (*cancelled)(void *ctx);
} DagCoreExec;

/* --- run state (both modes) ------------------------------------------ */

DagCoreRun *dagcore_create(const DagCoreNodeSpec *specs, size_t n_specs);
/* Frees the run; `free_payload` (optional) is called for every node that
 * has a non-NULL payload. */
void dagcore_destroy(DagCoreRun *run, void (*free_payload)(void *));

const char *dagcore_status_name(DagCoreStatus st);

/* Look a node id up in a run; returns its index or (size_t)-1. */
size_t dagcore_find(const DagCoreRun *run, const char *id);

void dagcore_begin(DagCoreRun *run, size_t index);
void dagcore_end(DagCoreRun *run, size_t index, bool ok, double elapsed_s,
                 const char *note);
void dagcore_skip(DagCoreRun *run, size_t index, const char *why);

/* --- registry mode --------------------------------------------------- */

/* Execute every registry node in order. Nodes whose prerequisite did not
 * complete are skipped with a rationale; nodes the hooks skip are recorded
 * as skipped; executor failures are recorded as failed. Cancellation stops
 * the walk and marks the run cancelled. Returns NULL only on allocation
 * failure (err filled). */
DagCoreRun *dagcore_execute(const DagCoreNodeSpec *specs, size_t n_specs,
                            const DagCoreExec *exec, void *ctx, char *err,
                            size_t err_len);

#endif /* TE15_DAGCORE_H */
