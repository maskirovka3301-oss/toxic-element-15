/*
 * fcdag.h — the Frequency Counter tab's analysis DAG registry.
 *
 * The DAG engine itself is dagcore.h (shared with the Spectrum tab). This
 * module registers EVERY frequency-counter analysis as one node; the
 * registry below is the single source of truth: the Settings tab builds its
 * per-analysis checkboxes directly from it, the runner walks it in order and
 * the report renders each node's payload table. fc_registry_check() verifies
 * at runtime that the registry is complete (every analysis has an executor,
 * every executor has a node, dependencies point backwards, there is room for
 * every node in the checkbox array).
 */
#ifndef TE15_FCDAG_H
#define TE15_FCDAG_H

#include <stdbool.h>
#include <stddef.h>

#include "dagcore.h"
#include "fcdataset.h"
#include "fclattice.h"

#define FC_MAX_NODES 32

/* Progress reported from the worker thread while the DAG runs. All fields
 * are written by the worker and read by the UI timer (atomics). */
typedef struct {
    volatile int *done;        /* analyses completed so far               */
    volatile int *total;       /* size of the registry                    */
    volatile int *stage;       /* index of the analysis currently running */
} FcProgress;

typedef DagCoreRun FcRun;

/* Validate the complete registry (see above). Fills `err` on failure. */
bool fc_registry_check(char *err, size_t err_len);

/* The analysis registry (the subcarrier-lattice test chain). */
extern const DagCoreNodeSpec fc_node_specs[];
extern const size_t          fc_node_count;

/* Run the registry over a snapshot of the loaded datasets. Callable from a
 * worker thread (it only reads `list`). Returns NULL only on allocation
 * failure (err filled). */
/* `scope` selects per-dataset blocks or one merged measurement list.
 * `enabled` (optional, NULL = all) is an array of length fc_node_count that
 * selects which registry nodes run; disabled nodes are recorded as skipped
 * with a reason and their dependants are skipped by the prerequisite rule. */
FcRun *fc_dag_run(const FcDatasetList *list, FclScope scope,
                  const bool *enabled, const FcProgress *progress,
                  char *err, size_t err_len);
void   fc_dag_free(FcRun *run);

#endif /* TE15_FCDAG_H */
