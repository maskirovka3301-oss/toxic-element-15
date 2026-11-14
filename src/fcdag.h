/*
 * fcdag.h — analysis DAG backend for the Frequency Counter tab.
 *
 * This mirrors the Spectrum tab's analysis pipeline (a DAG of registry
 * entries executed in dependency order, each node reporting a status and a
 * skip rationale, followed by a consolidated Reporter document) but it
 * works on the loaded frequency-measurement sessions instead of IQ data.
 *
 * The list of analyses is deliberately EMPTY for now: the framework is in
 * place (registry, execution loop with dependency ordering, per-node
 * results, text/Markdown/HTML report rendering) and analyses can be added
 * later by extending fc_nodes[].
 */
#ifndef TE15_FCDAG_H
#define TE15_FCDAG_H

#include <stdbool.h>
#include <stddef.h>

#include "freqseries.h"

typedef enum {
    FC_NODE_SKIPPED = 0,   /* not executed (prerequisites missing)     */
    FC_NODE_RUN,           /* executed                                 */
    FC_NODE_FAILED         /* executed but failed                      */
} FcNodeStatus;

/* One analysis in the registry. `requires` names the node id that must have
 * run first (NULL when the node has no prerequisite). */
typedef struct {
    const char *id;
    const char *title;
    const char *requires;
} FcNodeSpec;

/* One executed (or skipped) node in a run. */
typedef struct {
    const FcNodeSpec *spec;
    FcNodeStatus      status;
    char              note[192];   /* why skipped/failed (empty on success) */
    double            elapsed_s;
    char             *params;      /* rendered parameters (may be NULL)     */
} FcNodeResult;

/* A complete DAG run over a snapshot of the loaded sessions. */
typedef struct {
    FcNodeResult *nodes;
    size_t        count;
    double        elapsed_s;
} FcRun;

/* The analysis registry (currently empty). */
extern const FcNodeSpec fc_nodes[];
extern const size_t     fc_node_count;

/* Execute all registry nodes in order over `list`. Never returns NULL
 * unless out of memory (err filled). Safe to call from a worker thread:
 * it only reads `list`. */
FcRun *fc_dag_run(const FreqSeriesList *list, char *err, size_t err_len);
void   fc_dag_free(FcRun *run);

/* Render the consolidated report from the same document model in the three
 * formats (the caller frees each non-NULL string). `run` may be NULL. */
void fc_dag_report(const FreqSeriesList *list, const FcRun *run,
                   char **text_out, char **md_out, char **html_out);

#endif /* TE15_FCDAG_H */
