/*
 * fcdag.c — Frequency Counter analysis registry on the shared DAG engine.
 *
 * The engine in dagcore owns node identities, dependency ordering,
 * statuses, skip rationales, timings and the registry walk; this file only
 * says which analyses exist and how to execute one. No algorithm is
 * implemented yet, so the registry is empty and the executor reports that
 * an analysis is pending.
 */
#include "fcdag.h"

#include <stdio.h>

/* Add analyses here when they are specified. The engine executes them in
 * this order and enforces `requires`. */
const DagCoreNodeSpec fc_node_specs[] = {
    { NULL, NULL, NULL }   /* keep the array valid ISO C (count is 0) */
};
const size_t fc_node_count = 0;

/* Execute one frequency-counter analysis. No algorithm exists yet, so a
 * future node would be skipped by the skip_reason hook before reaching
 * here; the stub keeps the executor contract explicit. */
static bool fc_execute(void *ctx, DagCoreNode *node, char *err, size_t err_len)
{
    (void)ctx;
    (void)node;
    if (err != NULL && err_len > 0) {
        snprintf(err, err_len, "analysis algorithm is not implemented yet");
    }
    return false;
}

static const char *fc_skip_reason(void *ctx, const DagCoreNodeSpec *spec)
{
    (void)ctx;
    (void)spec;
    /* every registered analysis is still pending an algorithm */
    return "no algorithm is implemented for this analysis yet";
}

FcRun *fc_dag_run(const FcDatasetList *list, char *err, size_t err_len)
{
    static const DagCoreExec exec = {
        fc_execute, fc_skip_reason, NULL, NULL
    };

    return dagcore_execute(fc_node_specs, fc_node_count, &exec,
                           (void *)list, err, err_len);
}

void fc_dag_free(FcRun *run)
{
    dagcore_destroy(run, NULL);
}
