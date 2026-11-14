/*
 * fcdag.h — the Frequency Counter tab's analysis DAG registry.
 *
 * The DAG engine itself is dagcore.h (shared with the Spectrum tab). This
 * module only registers the frequency-counter analyses and exposes a typed
 * runner; the algorithms are still to be defined, so the registry is empty
 * for now and the engine records every future node as skipped until an
 * executor for it exists.
 */
#ifndef TE15_FCDAG_H
#define TE15_FCDAG_H

#include <stdbool.h>
#include <stddef.h>

#include "dagcore.h"
#include "fcdataset.h"
#include "fclattice.h"

#define FC_MAX_NODES 32

typedef DagCoreRun FcRun;

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
                  const bool *enabled, char *err, size_t err_len);
void   fc_dag_free(FcRun *run);

#endif /* TE15_FCDAG_H */
