/*
 * fcstats.h — advanced statistics and population estimates for the
 * Frequency Counter tab.
 *
 * The measurements of the selected datasets are treated as a two-level
 * sample: frequency lists (subjects) are the sampling units and rounds
 * group the lists. The report gives descriptive and robust statistics,
 * normality diagnostics, bootstrap confidence intervals, a variance-
 * decomposition / design-effect view for the population estimates, and
 * (when rounds are ordered) the Allan deviation of the round means.
 */
#ifndef TE15_FCSTATS_H
#define TE15_FCSTATS_H

#include <stddef.h>

#include "fcdataset.h"
#include "fclattice.h"

typedef struct {
    int    confidence;   /* 90, 95 or 99 (percent)                 */
    size_t bootstrap;    /* resampling replicates (>= 200)         */
    bool   per_round;    /* include the per-round descriptive table */
    bool   drift_correct;/* statistics on drift-corrected values   */
} FcStatsOptions;

/* Fill `*out` with a newly allocated text report (free with free()); NULL
 * on allocation failure. `list` is only read. */
char *fc_stats_report(const FcDatasetList *list, FclScope scope,
                      const FcStatsOptions *options);

#endif /* TE15_FCSTATS_H */
