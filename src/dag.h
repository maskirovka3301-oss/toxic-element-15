/*
 * dag.h — the analysis DAG: node identities and per-run execution status.
 *
 * The flow itself stays imperative in the analysis worker (the branching is
 * data dependent: A9 needs A8's cumulant carrier, A8 is conditional on
 * cyclic candidates, SSCA/FAM are alternatives), but every step records
 * begin/end/skip here so the Director knows what ran, what failed and why.
 */
#ifndef TE15_DAG_H
#define TE15_DAG_H

#include <stdbool.h>
#include <stddef.h>

typedef enum {
    DAG_A0 = 0, /* baseline analysis + metrics + signal ranking        */
    DAG_SOT,    /* common source of truth preparation                 */
    DAG_A1,     /* classic analysis                                   */
    DAG_A2,     /* pulse-train analysis                               */
    DAG_A3,     /* higher-order statistics                            */
    DAG_A4,     /* cyclic spectral correlation (FAM)                  */
    DAG_A5,     /* SSCA                                               */
    DAG_A6,     /* cycle detectors (cycledet + CPAR)                  */
    DAG_A7,     /* autocorrelation fluctuation (Burel)                */
    DAG_A8,     /* cyclic cumulants (AMR)                             */
    DAG_A9,     /* intra-burst envelope                               */
    DAG_MODCONF,/* modulation consensus ranking (post-DAG aggregation) */
    DAG_SSCLASS,/* spread-spectrum classification (post-DAG)          */
    DAG_NODES
} DagNode;

typedef enum {
    DAG_PENDING = 0,
    DAG_RUNNING,
    DAG_DONE,
    DAG_SKIPPED,
    DAG_FAILED
} DagState;

typedef struct {
    DagState    state[DAG_NODES];
    const char *skip[DAG_NODES];   /* reason for SKIPPED/FAILED          */
    double      elapsed[DAG_NODES];/* wall time in seconds               */
} DagRun;

const char *dag_node_name(int node);
const char *dag_state_name(DagState st);

void dag_reset(DagRun *run);
void dag_begin(DagRun *run, int node);
void dag_end(DagRun *run, int node, bool ok, double elapsed_s);
void dag_skip(DagRun *run, int node, const char *why);

#endif /* TE15_DAG_H */
