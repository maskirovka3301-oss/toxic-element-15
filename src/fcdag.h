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
#include "freqseries.h"

typedef DagCoreRun FcRun;

/* The analysis registry (currently empty: analyses to be defined later). */
extern const DagCoreNodeSpec fc_node_specs[];
extern const size_t          fc_node_count;

/* Run the registry over a snapshot of the loaded sessions. Callable from a
 * worker thread (it only reads `list`). Returns NULL only on allocation
 * failure (err filled). */
FcRun *fc_dag_run(const FreqSeriesList *list, char *err, size_t err_len);
void   fc_dag_free(FcRun *run);

#endif /* TE15_FCDAG_H */
