/*
 * cycledet.h — CFAR cycle detectors (Spooner, "The Cycle Detectors").
 *
 * Signal-presence detection based on cyclostationarity: the cyclic
 * autocorrelation magnitude at candidate cycle frequencies is compared
 * against a false-alarm-rate threshold (Gaussian-noise baseline), and
 * the individual values are combined into a multi-cycle statistic.
 * The data is optionally pre-whitened (H_w(f) = 1/sqrt(S(f))) so the
 * test is robust to non-flat noise.
 */
#ifndef TE15_CYCLEDET_H
#define TE15_CYCLEDET_H

#include <stdbool.h>
#include "iq_source.h"
#include "region.h"

#define CYCLEDET_MAX_CF 128

typedef struct {
    int     num_cf;
    double *cf_hz;     /* candidate cycle frequencies, Hz        */
    double *stat;      /* |cyclic autocorrelation|/power per CF  */
    double *whitened_stat; /* same, after whitening              */
    double  threshold; /* 5 sigma Gaussian baseline              */
    double  multi_cycle; /* sum of the top-4 squared statistics  */
    double  multi_threshold;
    bool    detected;
    double  best_cf;
    double  best_stat;
    bool    whitened;
    char    note[256];
} CycleDetResult;

bool cycledet_analyze(const IqSource *src, const Region *region,
                      CycleDetResult *out);
void cycledet_clear(CycleDetResult *r);

#endif /* TE15_CYCLEDET_H */
