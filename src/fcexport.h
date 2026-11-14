/*
 * fcexport.h — export the selected Frequency Counter measurements as CSV.
 *
 * The exported table has one row per selected round, session or dataset
 * (the caller chooses the granularity); only the checked frequency lists
 * are used because the caller passes the filtered analysis snapshot.
 */
#ifndef TE15_FCEXPORT_H
#define TE15_FCEXPORT_H

#include "fcdataset.h"

typedef enum {
    FC_CSV_ROUND = 0,    /* one row per selected round    */
    FC_CSV_SESSION,      /* one row per selected session  */
    FC_CSV_DATASET       /* one row per selected dataset  */
} FcCsvGranularity;

/* Human-readable name of a granularity ("round", "session", "dataset"). */
const char *fc_csv_granularity_name(FcCsvGranularity granularity);

/* Build the CSV text (newly allocated, free with free()). Returns NULL on
 * allocation failure; an empty selection yields just the header line. */
char *fc_csv_export(const FcDatasetList *list, FcCsvGranularity granularity);

#endif /* TE15_FCEXPORT_H */
