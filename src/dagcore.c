/*
 * dagcore.c — see dagcore.h.
 */
#include "dagcore.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void)
{
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

const char *dagcore_status_name(DagCoreStatus st)
{
    switch (st) {
    case DAGCORE_PENDING: return "pending";
    case DAGCORE_RUNNING: return "running";
    case DAGCORE_DONE:    return "executed";
    case DAGCORE_SKIPPED: return "skipped";
    case DAGCORE_FAILED:  return "failed";
    }
    return "unknown";
}

DagCoreRun *dagcore_create(const DagCoreNodeSpec *specs, size_t n_specs)
{
    DagCoreRun *run = calloc(1, sizeof(*run));

    if (run == NULL) return NULL;
    if (n_specs > 0) {
        run->nodes = calloc(n_specs, sizeof(*run->nodes));
        if (run->nodes == NULL) {
            free(run);
            return NULL;
        }
        for (size_t i = 0; i < n_specs; i++) {
            run->nodes[i].spec = &specs[i];
            run->nodes[i].status = DAGCORE_PENDING;
            run->nodes[i].note[0] = '\0';
        }
    }
    run->count = n_specs;
    return run;
}

void dagcore_destroy(DagCoreRun *run, void (*free_payload)(void *))
{
    if (run == NULL) return;
    if (free_payload != NULL) {
        for (size_t i = 0; i < run->count; i++) {
            if (run->nodes[i].payload != NULL) {
                free_payload(run->nodes[i].payload);
                run->nodes[i].payload = NULL;
            }
        }
    }
    free(run->nodes);
    free(run);
}

size_t dagcore_find(const DagCoreRun *run, const char *id)
{
    if (run == NULL || id == NULL) return (size_t)-1;
    for (size_t i = 0; i < run->count; i++) {
        if (run->nodes[i].spec != NULL && run->nodes[i].spec->id != NULL &&
            strcmp(run->nodes[i].spec->id, id) == 0) {
            return i;
        }
    }
    return (size_t)-1;
}

void dagcore_begin(DagCoreRun *run, size_t index)
{
    if (run == NULL || index >= run->count) return;
    run->nodes[index].status = DAGCORE_RUNNING;
    run->nodes[index].note[0] = '\0';
}

void dagcore_end(DagCoreRun *run, size_t index, bool ok, double elapsed_s,
                 const char *note)
{
    if (run == NULL || index >= run->count) return;
    run->nodes[index].status = ok ? DAGCORE_DONE : DAGCORE_FAILED;
    run->nodes[index].elapsed_s = elapsed_s;
    if (note != NULL) {
        snprintf(run->nodes[index].note, sizeof(run->nodes[index].note), "%s",
                 note);
    }
}

void dagcore_skip(DagCoreRun *run, size_t index, const char *why)
{
    if (run == NULL || index >= run->count) return;
    run->nodes[index].status = DAGCORE_SKIPPED;
    if (why != NULL) {
        snprintf(run->nodes[index].note, sizeof(run->nodes[index].note), "%s",
                 why);
    }
}

DagCoreRun *dagcore_execute(const DagCoreNodeSpec *specs, size_t n_specs,
                            const DagCoreExec *exec, void *ctx, char *err,
                            size_t err_len)
{
    DagCoreRun *run;
    double t0;

    if (err != NULL && err_len > 0) err[0] = '\0';
    if (n_specs > 0 && specs == NULL) {
        if (err != NULL) snprintf(err, err_len, "empty DAG registry");
        return NULL;
    }
    run = dagcore_create(specs, n_specs);
    if (run == NULL) {
        if (err != NULL) snprintf(err, err_len, "out of memory");
        return NULL;
    }
    t0 = now_s();
    for (size_t i = 0; i < run->count; i++) {
        DagCoreNode *node = &run->nodes[i];
        const DagCoreNodeSpec *spec = node->spec;
        double n0;
        char ex_err[192] = "";
        bool ok;

        if (exec != NULL && exec->cancelled != NULL && exec->cancelled(ctx)) {
            run->cancelled = true;
            break;
        }
        if (exec != NULL && exec->on_node != NULL) {
            exec->on_node(ctx, i, run->count,
                          spec != NULL ? spec->title : NULL);
        }
        if (exec != NULL && exec->enabled != NULL &&
            !exec->enabled(ctx, spec)) {
            dagcore_skip(run, i, "disabled by the analysis selection");
            continue;
        }
        /* dependency ordering: every prerequisite (comma-separated list)
         * must have completed */
        if (spec != NULL && spec->requires != NULL) {
            const char *p = spec->requires;
            bool blocked = false;

            while (*p != '\0' && !blocked) {
                const char *end = strchr(p, ',');
                size_t len = end != NULL ? (size_t)(end - p) : strlen(p);
                char id[64];
                size_t idx;

                while (len > 0 && (*p == ' ' || *p == '\t')) {
                    p++;
                    len--;
                }
                while (len > 0 && (p[len - 1] == ' ' || p[len - 1] == '\t')) {
                    len--;
                }
                if (len == 0) {
                    p = end != NULL ? end + 1 : p + strlen(p);
                    continue;
                }
                if (len >= sizeof(id)) len = sizeof(id) - 1;
                memcpy(id, p, len);
                id[len] = '\0';
                idx = dagcore_find(run, id);
                if (idx == (size_t)-1) {
                    char note[192];

                    snprintf(note, sizeof(note),
                             "prerequisite analysis '%s' is not in the "
                             "registry", id);
                    dagcore_skip(run, i, note);
                    blocked = true;
                } else if (run->nodes[idx].status != DAGCORE_DONE) {
                    char note[192];

                    snprintf(note, sizeof(note),
                             "prerequisite analysis '%s' did not run", id);
                    dagcore_skip(run, i, note);
                    blocked = true;
                }
                p = end != NULL ? end + 1 : p + strlen(p);
            }
            if (blocked) continue;
        }
        if (exec != NULL && exec->skip_reason != NULL) {
            const char *why = exec->skip_reason(ctx, spec);

            if (why != NULL) {
                dagcore_skip(run, i, why);
                continue;
            }
        }
        if (exec == NULL || exec->execute == NULL) {
            dagcore_skip(run, i, "no executor is attached to this DAG");
            continue;
        }
        n0 = now_s();
        dagcore_begin(run, i);
        ok = exec->execute(ctx, node, ex_err, sizeof(ex_err));
        dagcore_end(run, i, ok, now_s() - n0,
                    ok ? NULL : (ex_err[0] != '\0' ? ex_err
                                                   : "analysis failed"));
    }
    run->elapsed_s = now_s() - t0;
    return run;
}
