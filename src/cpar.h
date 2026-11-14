/*
 * cpar.h — Cumulative Peak-to-Average Ratio DSSS detector.
 *
 * Based on the CPAR family of blind DSSS detectors, e.g. Djaouad &
 * Ait Hemiche, "Real-Time Implementation of DSSS Signal Detection
 * Based on Autocorrelation Estimation and Cumulative Peak-to-Average
 * Ratio Using SDR Platforms" (IEEE). The autocorrelation is estimated
 * (windowed averaging), candidate peaks are located, each peak's
 * peak-to-average ratio PAR = peak^2 / mean(neighbourhood^2) is
 * computed, and the cumulative CPAR = sum of the PARs of significant
 * peaks discriminates a DSSS signal (a sharp chip-period peak with a
 * large PAR) from noise (all lags comparable, PAR ~ 1). Robust at low
 * SNR and requires no prior knowledge of the signal.
 */
#ifndef TE15_CPAR_H
#define TE15_CPAR_H

#include <stdbool.h>
#include "iq_source.h"
#include "region.h"

#define CPAR_MAX_LAGS 8192

typedef struct {
    int      n_lags;      /* number of lags evaluated          */
    double  *lag_s;       /* lag in seconds                    */
    double  *acf;         /* normalised |R(tau)| / |R(0)|      */
    double  *par;         /* PAR per lag (0 where not a peak)  */
    int     *is_peak;     /* 1 where a candidate peak sits     */

    double   cpar;        /* cumulative PAR, significant peaks */
    double   cpar_all;    /* cumulative PAR, all candidates    */
    int      n_peaks;     /* candidate peak count              */
    double   par_threshold; /* significance threshold          */

    double   peak_lag;    /* lag (s) of the highest-PAR peak
                             * (the code period for a repeating code) */
    double   peak_par;
    double   chip_period_s; /* chip period from the initial ACF slope */
    double   chip_rate_hz;  /* 1/chip_period_s when available          */
    bool     detected;
    char     note[256];
} CparResult;

/* Run the CPAR analysis over `region` of `src` (private buffer copy,
 * like the rest of the suite). Returns false for degenerate regions. */
bool cpar_analyze(const IqSource *src, const Region *region, CparResult *out);

void cpar_clear(CparResult *r);

#endif /* TE15_CPAR_H */
