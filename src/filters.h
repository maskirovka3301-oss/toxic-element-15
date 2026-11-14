/*
 * filters.h — practical filter toolbox (CSP Blog: practical filters;
 * SPTK windowing/filter utilities): windowed-sinc lowpass / highpass /
 * bandpass / bandstop design applied to a selection, with the filter
 * response and the PSD before/after. Selection-scoped.
 */
#ifndef TE15_FILTERS_H
#define TE15_FILTERS_H

#include <stdbool.h>
#include "iq_source.h"
#include "region.h"

#define FILTER_N 2048

typedef struct {
    int     kind;        /* 0 LP, 1 HP, 2 BP, 3 BS          */
    double  f1_hz;       /* low cutoff                     */
    double  f2_hz;       /* high cutoff (LP/HP unused)     */
    int     taps;        /* filter length                  */
    int     n;           /* PSD FFT size                   */
    double *response;    /* |H(f)|, n bins                 */
    double *input_psd;   /* selection PSD, n bins (dB)     */
    double *output_psd;  /* filtered PSD, n bins (dB)      */
    double  power_in_db;
    double  power_out_db;
    char    note[256];
} FilterResult;

bool filter_analyze(const IqSource *src, const Region *region, int kind,
                    double f1_hz, double f2_hz, int taps, FilterResult *out);
void filter_clear(FilterResult *r);

#endif /* TE15_FILTERS_H */
