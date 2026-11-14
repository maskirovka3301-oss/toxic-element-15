/*
 * sptk.h — SPTK-derived signal-processing primitives (the Speech Signal
 * Processing Toolkit's generally useful tools, applied to RF/IQ data),
 * plus the practical filter toolbox from the CSP Blog's practical-filters
 * post and the characteristic-function / Laplace-transform (MGF)
 * estimators from the SPTK posts.
 */
#ifndef TE15_SPTK_H
#define TE15_SPTK_H

#include <stdbool.h>
#include <stddef.h>

/* Empirical characteristic function Phi(omega) = E[e^{i omega x}] of a
 * real-valued sequence over a grid of `m` omegas. */
void sptk_charfun(const float *x, size_t n, const double *omega, int m,
                  double *out_re, double *out_im);

/* Empirical moment-generating function M(s) = E[e^{s x}] (real s). */
void sptk_mgf(const float *x, size_t n, const double *s, int m, double *out);

/* Real cepstrum of a log-magnitude spectrum (n values in, n out);
 * quefrency is in samples. */
void sptk_cepstrum(const double *log_spectrum, size_t n, double *out);

/* Levinson-Durbin LPC over the autocorrelation of a complex sequence;
 * writes order+1 coefficients with a[0] = 1. Returns false on failure. */
bool sptk_lpc(const float *iq, size_t n, int order, double *coeffs);

/* Average magnitude difference function over lags 0..max_lag-1. */
void sptk_amdf(const float *iq, size_t n, size_t max_lag, double *out);

/* Group delay in samples from the phase of a complex spectrum
 * (interleaved re/im, 2*n floats). */
void sptk_group_delay(const float *X, size_t n, double *out);

/* Goertzel single-bin power at normalized frequency f (cycles/sample). */
double sptk_goertzel(const float *iq, size_t n, double f);

/* Windowed-sinc FIR design: kind 0 = lowpass (f1 cutoff, f2 unused),
 * 1 = highpass, 2 = bandpass (f1..f2), 3 = bandstop. Normalized
 * cutoffs in cycles/sample; taps_len must be odd. */
size_t sptk_fir_design(int kind, double f1, double f2, float *taps,
                       size_t taps_len);

/* Apply a real FIR to a complex signal in place. */
void sptk_fir_apply(float *iq, size_t n, const float *taps, size_t taps_len);

/* 5-point median filter on the complex magnitude in place. */
void sptk_median_filter(float *iq, size_t n);

#endif /* TE15_SPTK_H */
