/*
 * classic.h — classic signal-analysis tools for IQ data (SPTK-derived):
 * cepstrum (periodicity / echo structure), AMDF (period estimation),
 * LPC spectral envelope, group delay (phase linearity), and a Goertzel
 * single-bin probe. Selection-scoped: meant for an interesting part of
 * the waterfall.
 */
#ifndef TE15_CLASSIC_H
#define TE15_CLASSIC_H

#include <stdbool.h>
#include "iq_source.h"
#include "region.h"

#define CLASSIC_N     32768 /* FFT size used (0.65 s at 50 kHz) */
#define CLASSIC_LAGS  8192  /* AMDF lag count (0.16 s at 50 kHz) */
#define CLASSIC_LPC   16    /* LPC order                      */

typedef struct {
    int     n;              /* FFT size used                  */
    double  fs;             /* effective sample rate          */
    double *cepstrum;       /* n/2 quefrencies                */
    double  cep_peak_quef;  /* quefrency of the peak (samples)*/
    double  cep_peak_mag;
    double  cep_period_hz;  /* fs / quefrency                 */
    double *amdf;           /* CLASSIC_LAGS values            */
    double  amdf_period;    /* samples                        */
    double  amdf_period_hz;
    double  acf_period_hz;  /* envelope-ACF period estimate   */
    double *lpc_coeffs;     /* CLASSIC_LPC + 1 coefficients   */
    int     lpc_order;
    double *group_delay;    /* n values (samples)             */
    double  probe_hz;       /* Goertzel probe frequency       */
    double  goertzel_power; /* normalised (0..1)              */
    double  period_hz;      /* consensus period estimate      */
    char    note[256];
} ClassicResult;

bool classic_analyze(const IqSource *src, const Region *region,
                     double probe_hz, ClassicResult *out);
void classic_clear(ClassicResult *r);

#endif /* TE15_CLASSIC_H */
