/*
 * fcreport.h — report rendering for the Frequency Counter analyses.
 *
 * The DAG engine (dagcore) and the registry (fcdag) are separate concerns;
 * this module turns a run of the frequency-counter registry plus the loaded
 * sessions into a consolidated report, rendered as plain text, Markdown or
 * HTML from the same document model (the Spectrum tab's Reporter provides
 * the same three presentations for its own document).
 */
#ifndef TE15_FCREPORT_H
#define TE15_FCREPORT_H

#include <stddef.h>

#include "dagcore.h"
#include "fcdataset.h"

/* Render the report; the caller frees each non-NULL string. `run` may be
 * NULL (nothing executed yet). */
void fc_report(const FcDatasetList *list, const DagCoreRun *run,
               char **text_out, char **md_out, char **html_out);

#endif /* TE15_FCREPORT_H */
