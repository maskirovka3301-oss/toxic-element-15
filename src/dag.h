/*
 * dag.h — the Spectrum tab's analysis DAG registry.
 *
 * The generic engine lives in dagcore.h: node identities, dependency
 * ordering, statuses, skip rationales, timings and the registry walk are
 * shared with the Frequency Counter tab. This module only defines the
 * Spectrum node set (A0..A9 plus the post-DAG aggregations) and a thin,
 * typed façade over the engine for the imperative analysis worker.
 */
#ifndef TE15_DAG_H
#define TE15_DAG_H

#include <stdbool.h>
#include <stddef.h>

#include "dagcore.h"

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

typedef DagCoreStatus DagState;
typedef DagCoreRun DagRun;

const char *dag_node_name(int node);
const char *dag_state_name(DagState st);
int         dag_node_count(void);

/* Create/reset the run of the Spectrum registry. */
DagRun *dag_run_new(void);
void    dag_run_free(DagRun *run);
void    dag_reset(DagRun *run);

void dag_begin(DagRun *run, int node);
void dag_end(DagRun *run, int node, bool ok, double elapsed_s);
void dag_skip(DagRun *run, int node, const char *why);

/* Accessors (the run is shared with the engine, so callers never touch the
 * engine's fields directly). */
DagState    dag_state(const DagRun *run, int node);
double      dag_elapsed(const DagRun *run, int node);
const char *dag_skip_reason(const DagRun *run, int node);

#endif /* TE15_DAG_H */
