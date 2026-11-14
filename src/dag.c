/*
 * dag.c — the Spectrum analysis DAG: node registry and typed façade over
 * the shared engine (dagcore).
 */
#include "dag.h"

#include <string.h>

static const DagCoreNodeSpec dag_specs[DAG_NODES] = {
    { "a0-baseline", "A0 baseline analysis", NULL },
    { "sot", "SOT common source", NULL },
    { "a1-classic", "A1 classic analysis", NULL },
    { "a2-pulse", "A2 pulse-train analysis", NULL },
    { "a3-hos", "A3 higher-order statistics", NULL },
    { "a4-fam", "A4 FAM spectral correlation", NULL },
    { "a5-ssca", "A5 SSCA", NULL },
    { "a6-cycledet", "A6 cycle detectors", NULL },
    { "a7-fluct", "A7 autocorrelation fluctuation", NULL },
    { "a8-ccc", "A8 cyclic cumulants", NULL },
    { "a9-intraburst", "A9 intra-burst envelope", NULL },
    { "modconf", "modulation consensus", NULL },
    { "ssclass", "spread-spectrum classification", NULL }
};

const char *dag_node_name(int node)
{
    if (node < 0 || node >= DAG_NODES) return "unknown";
    return dag_specs[node].title;
}

const char *dag_state_name(DagState st)
{
    return dagcore_status_name(st);
}

int dag_node_count(void)
{
    return DAG_NODES;
}

DagRun *dag_run_new(void)
{
    return dagcore_create(dag_specs, DAG_NODES);
}

void dag_run_free(DagRun *run)
{
    dagcore_destroy(run, NULL);
}

void dag_reset(DagRun *run)
{
    if (run == NULL) return;
    memset(run->nodes, 0, run->count * sizeof(*run->nodes));
    for (size_t i = 0; i < run->count; i++) {
        run->nodes[i].spec = &dag_specs[i];
        run->nodes[i].status = DAGCORE_PENDING;
        run->nodes[i].note[0] = '\0';
    }
    run->elapsed_s = 0.0;
    run->cancelled = false;
}

void dag_begin(DagRun *run, int node)
{
    if (node < 0 || node >= DAG_NODES) return;
    dagcore_begin(run, (size_t)node);
}

void dag_end(DagRun *run, int node, bool ok, double elapsed_s)
{
    if (node < 0 || node >= DAG_NODES) return;
    dagcore_end(run, (size_t)node, ok, elapsed_s, NULL);
}

void dag_skip(DagRun *run, int node, const char *why)
{
    if (node < 0 || node >= DAG_NODES) return;
    dagcore_skip(run, (size_t)node, why);
}

DagState dag_state(const DagRun *run, int node)
{
    if (run == NULL || node < 0 || (size_t)node >= run->count) {
        return DAGCORE_PENDING;
    }
    return run->nodes[node].status;
}

double dag_elapsed(const DagRun *run, int node)
{
    if (run == NULL || node < 0 || (size_t)node >= run->count) return 0.0;
    return run->nodes[node].elapsed_s;
}

const char *dag_skip_reason(const DagRun *run, int node)
{
    if (run == NULL || node < 0 || (size_t)node >= run->count) return NULL;
    return run->nodes[node].note[0] != '\0' ? run->nodes[node].note : NULL;
}
