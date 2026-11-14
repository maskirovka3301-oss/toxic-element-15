/*
 * fluct.h — "fluctuations of autocorrelation estimators" analysis.
 *
 * Based on Burel, Bouder, Berder: "Detection of Spread Spectrum
 * Transmissions using fluctuations of correlation estimators"
 * (ISPACS 2000). Instead of looking for a peak in the autocorrelation,
 * the detector measures the VARIANCE of the autocorrelation estimate
 * over many short windows: for noise alone the normalised fluctuation
 * F(tau) = T * Var[R(tau)] / sigma^4 is ~1 at every lag, while a hidden
 * DSSS signal raises F(tau) at the chip-period lag by roughly
 * (chip SNR)^2 * (chips per window) — detectable far below the SNR at
 * which the mean peak would emerge from the noise.
 */
#ifndef TE15_FLUCT_H
#define TE15_FLUCT_H

#include <stdbool.h>
#include "iq_source.h"
#include "region.h"

#define FLUCT_MAX_LAGS 256

typedef struct {
    int      n_lags;       /* number of lags evaluated (>= 1)  */
    double  *lag_s;        /* lag in seconds, length n_lags    */
    double  *mean_acf;     /* |Rbar(tau)| / sigma^2, normalised */
    double  *fluct;        /* F(tau) = T * Var[R(tau)]/sigma^4  */

    double   noise_power;  /* sigma^2 estimate                  */
    double   threshold;    /* detection threshold on F          */
    double   peak_lag;     /* lag (s) of the maximum F          */
    double   peak_fluct;   /* maximum F value                   */
    double   peak_acf;     /* normalised |Rbar| at that lag     */
    double   snr_db;       /* rough chip-SNR estimate, dB       */
    double   chip_rate_hz; /* 1/peak_lag (0 when not detected)  */
    bool     detected;     /* fluctuation criterion verdict     */
    char     note[256];    /* human-readable explanation        */
} FluctResult;

/*
 * Run the fluctuation analysis over `region` of `src` and fill `out`
 * (previous contents released). The selection is copied into a private
 * buffer first, like the rest of the analysis suite. Returns false for
 * degenerate regions or too little data.
 */
bool fluct_analyze(const IqSource *src, const Region *region, FluctResult *out);

/* Free heap storage held by a result. */
void fluct_clear(FluctResult *r);

#endif /* TE15_FLUCT_H */
