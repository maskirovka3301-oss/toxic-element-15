/*
 * dag.c — DAG node names and per-run status bookkeeping.
 */
#include "dag.h"

#include <string.h>

static const char *const dag_names[DAG_NODES] = {
    "A0 baseline analysis",
    "SOT common source",
    "A1 classic analysis",
    "A2 pulse-train analysis",
    "A3 higher-order statistics",
    "A4 FAM spectral correlation",
    "A5 SSCA",
    "A6 cycle detectors",
    "A7 autocorrelation fluctuation",
    "A8 cyclic cumulants",
    "A9 intra-burst envelope",
    "modulation consensus",
    "spread-spectrum classification"
};

const char *dag_node_name(int node)
{
    if (node < 0 || node >= DAG_NODES) return "unknown";
    return dag_names[node];
}

const char *dag_state_name(DagState st)
{
    switch (st) {
    case DAG_PENDING: return "pending";
    case DAG_RUNNING: return "running";
    case DAG_DONE:    return "executed";
    case DAG_SKIPPED: return "skipped";
    case DAG_FAILED:  return "failed";
    }
    return "unknown";
}

void dag_reset(DagRun *run)
{
    if (run == NULL) return;
    memset(run, 0, sizeof(*run));
    for (int i = 0; i < DAG_NODES; i++) run->state[i] = DAG_PENDING;
}

void dag_begin(DagRun *run, int node)
{
    if (run == NULL || node < 0 || node >= DAG_NODES) return;
    run->state[node] = DAG_RUNNING;
}

void dag_end(DagRun *run, int node, bool ok, double elapsed_s)
{
    if (run == NULL || node < 0 || node >= DAG_NODES) return;
    run->state[node] = ok ? DAG_DONE : DAG_FAILED;
    run->elapsed[node] = elapsed_s;
}

void dag_skip(DagRun *run, int node, const char *why)
{
    if (run == NULL || node < 0 || node >= DAG_NODES) return;
    run->state[node] = DAG_SKIPPED;
    run->skip[node] = why;
}
