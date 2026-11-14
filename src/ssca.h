/*
 * ssca.h — Strip Spectral Correlation Analyzer (Spooner / Roberts et al.).
 *
 * Exhaustive blind cycle-frequency estimation: the input is channelized
 * by short FFTs (N' points, hopped by L), each channelizer output is
 * multiplied by the original signal, and the products are Fourier
 * transformed with long "strip" FFTs (N points). Each strip produces
 * point estimates of the spectral correlation function along a line
 * parallel to alpha = -2f. Normalising by the power spectrum yields the
 * spectral coherence, which is thresholded and binned into a list of
 * significant cycle frequencies.
 */
#ifndef TE15_SSCA_H
#define TE15_SSCA_H

#include <stdbool.h>
#include "iq_source.h"
#include "region.h"

#define SSCA_MAX_CF 64

typedef struct {
    int    channel_fft; /* N' channelizer FFT size       */
    int    strip_fft;   /* N strip FFT size              */
    int    num_strips;  /* channelizer bins kept         */
    int    num_cf;      /* strip FFT bins kept           */
    float *scf;         /* |SCF| grid: num_strips x num_cf, row-major
                         * (row = strip k = channel f_k)  */
    float *coh;         /* |coherence| grid, same layout */
    double *cdp;        /* cyclic domain profile: max coherence over
                         * strips, per cycle-frequency bin (num_cf) */
    double cf_list[SSCA_MAX_CF][2]; /* (alpha Hz, coherence) detected */
    size_t cf_count;
    double cf_threshold;
    char   note[256];
} SscaResult;

bool ssca_analyze(const IqSource *src, const Region *region, SscaResult *out);
void ssca_clear(SscaResult *r);

#endif /* TE15_SSCA_H */
