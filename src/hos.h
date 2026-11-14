/*
 * hos.h — Higher-Order Statistics analysis.
 *
 * Noise is typically Gaussian, whose cumulants beyond second order are
 * identically zero (Mendel, "Tutorial on higher-order statistics
 * (spectra) in signal processing and system theory", Proc. IEEE 1991).
 * Modulated signals — including spread spectrum — have non-zero
 * higher-order cumulants, so the fourth-order cumulant of the IQ data
 * suppresses Gaussian noise and reveals non-Gaussian signals even at
 * low SNR.
 *
 * The cumulant definitions follow Swami & Sadler, "Hierarchical digital
 * modulation classification using cumulants", IEEE Trans. Comm. 2000:
 * C20..C63 on the centralised samples, plus the normalised magnitudes
 * |C40|/C21^2, |C41|/C21^2, |C42|/C21^2 and the bispectrum estimate.
 */
#ifndef TE15_HOS_H
#define TE15_HOS_H

#include <stdbool.h>
#include "iq_source.h"
#include "region.h"

typedef struct {
    /* --- cumulants (central, on the mixdowned selection) --- */
    double c20_re, c20_im; /* E[x^2]                            */
    double c21;            /* E[|x|^2]                          */
    double c30_re, c30_im; /* E[x^3]                            */
    double c31_re, c31_im; /* E[x^2 x*]                         */
    double c40_re, c40_im; /* E[x^4] - 3 E[x^2]^2               */
    double c41_re, c41_im; /* E[x^3 x*] - 3 E[x^2] E[|x|^2]     */
    double c42;            /* E[|x|^4] - |E[x^2]|^2 - 2 E[|x|^2]^2 */
    double c60_re, c60_im; /* E[x^6] - 15 E[x^4] E[x^2] + 30 E[x^2]^3 */
    double c63;            /* E[|x|^6] - 9 C42 C21 - 6 C21^3     */

    /* --- normalised (classification-standard) --- */
    double n40, n41, n42;  /* |C40|/C21^2, |C41|/C21^2, |C42|/C21^2 */
    double kurtosis;       /* E[|x|^4]/E[|x|^2]^2 - 2             */

    /* --- detection --- */
    size_t samples;        /* number of samples used             */
    double n42_threshold;  /* 5 * sqrt(24/N), Gaussian baseline  */
    bool   non_gaussian;   /* |C42|/C21^2 above the threshold    */
    char   hint[128];      /* constellation-style hint           */
    char   note[256];      /* verdict text                       */

    /* --- bispectrum (bicoherence magnitude, principal domain) --- */
    int    bs_n;           /* grid: bs_n x bs_n, 0..1             */
    float *bisp;
} HosResult;

/* Run the HOS analysis over `region` of `src` (private buffer copy,
 * like the rest of the suite). Returns false for degenerate regions. */
bool hos_analyze(const IqSource *src, const Region *region, HosResult *out);

void hos_clear(HosResult *r);

#endif /* TE15_HOS_H */
