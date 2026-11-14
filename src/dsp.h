/*
 * dsp.h — small, reusable DSP primitives: windowing, FFT spectra in dB,
 * entropy, and pulse-repetition-frequency estimation.
 */
#ifndef TE15_DSP_H
#define TE15_DSP_H

#include <stddef.h>
#include <stdbool.h>
#include <fftw3.h>

/* Fill `w` (length n) with a periodic Hann window. */
void dsp_hann(float *w, size_t n);

/* Window types for the waterfall FFT. */
typedef enum {
    DSP_WINDOW_HANN = 0,
    DSP_WINDOW_HAMMING,
    DSP_WINDOW_BLACKMAN,
    DSP_WINDOW_RECTANGULAR,
} DspWindow;

/* Fill `w` (length n) with the chosen window. */
void dsp_window(DspWindow type, float *w, size_t n);

/*
 * Convert FFT output bins (length n) to dB magnitudes in `db_out`,
 * applying an FFT shift so that bin 0 is the most negative frequency.
 */
void dsp_bins_to_db(const fftwf_complex *bins, size_t n, float *db_out);

/*
 * Shannon entropy in bits of a dB-magnitude array.
 * Implemented as a two-pass min/max normalisation followed by the
 * distribution entropy; returns 0.0 for empty or constant input.
 */
double dsp_entropy(const float *v, size_t n);

/*
 * Estimate a pulse repetition frequency (Hz) from a band-integrated
 * per-frame power series. `power` has `n` frames spaced `dt` seconds
 * apart. Returns 0.0 when no periodic structure is detected.
 */
double dsp_estimate_prf(const float *power, size_t n, double dt);

/*
 * Normalise a dB array to [0,1] by linear min/max scaling, writing the
 * results to `out` (may alias `v`). Also reports the range actually used.
 */
void dsp_normalize_to_unit(const float *v, size_t n, float *out,
                           float *min_out, float *max_out);

/*
 * Reusable single-threaded FFT plan of size n (forward, ESTIMATE).
 * The plan owns its input/output buffers; use one plan per thread.
 */
typedef struct FftPlan FftPlan;

FftPlan *fft_plan_create(int n);
void     fft_plan_destroy(FftPlan *p);
int      fft_plan_size(const FftPlan *p);

/*
 * Compute the spectrum of one windowed frame:
 *   - `iq` points at 2*n interleaved float32 I/Q samples
 *   - `window` holds n window coefficients (NULL = rectangular)
 *   - `db_out` receives the fftshifted dB magnitudes, length n
 */
void fft_plan_spectrum(const FftPlan *p, const float *iq,
                       const float *window, float *db_out);

/*
 * As fft_plan_spectrum, but returns the raw (unshifted) complex FFT
 * bins in `out` (length n) for algorithms that need phase information
 * (e.g. spectral correlation).
 */
void fft_plan_transform(const FftPlan *p, const float *iq,
                        const float *window, fftwf_complex *out);

/*
 * Windowed-sinc FIR low-pass (cutoff 0.4/D of the input rate) followed
 * by decimation by D: out[i] = sum_k h[k] * iq[i*D + k], on both I and Q.
 * Returns the number of output samples written to `out` (2*that floats).
 * D >= 1.
 */
size_t dsp_fir_decimate(const float *iq, size_t n, size_t D, float *out);

/*
 * Decimate by D in iterative small stages (each factor <= 8) so the
 * combined anti-alias transition stays narrow. A single large-D pass
 * through the 64-tap FIR aliases badly: decimating 16 MHz to 50 kHz in
 * one step leaves a ~250 kHz transition, which folds wideband noise into
 * the envelope and corrupts the envelope/cepstrum period estimates.
 * The result is written to `out` (at least n/D + 8 complex samples);
 * returns the number of complex samples produced.
 */
size_t dsp_decimate_staged(const float *iq, size_t n, size_t D, float *out);

/*
 * Decimation factor that never throws away more resolution than the
 * analysed signal needs: the effective rate stays at least
 * 2.5 x the signal bandwidth and at least `min_rate`. Analyses used to
 * decimate to fixed low rates (50-500 kHz) regardless of the selection
 * bandwidth, which aliased and discarded most of a wide selection; with
 * a single pre-filtered source the only legitimate reason to decimate is
 * cost, so this keeps the rate proportional to the signal.
 */
size_t dsp_decimation_for(double fs, double signal_bw, double min_rate);

/* As dsp_decimation_for with an explicit bandwidth factor (the effective
 * rate stays >= factor x signal_bw). Rate-sensitive detectors that were
 * calibrated at a fixed rate use a smaller factor. */
size_t dsp_decimation_for_x(double fs, double signal_bw, double min_rate,
                            double factor);

/*
 * Zero-phase frequency-domain low-pass: H(f) = 1 for |f| <= pass_hz, a
 * raised-cosine transition down to 0 at |f| >= stop_hz (a low-pass on
 * the complex baseband keeps both signs of the frequency axis). The mask
 * is applied with a single FFT pair, so the transition can be arbitrarily
 * narrow without an FIR-length limit (edges are preserved exactly:
 * linear phase, no pre-ringing beyond the mask's transition).
 * `iq` is filtered in place (n complex samples).
 */
bool dsp_fft_mask_lowpass(float *iq, size_t n, double fs, double pass_hz,
                          double stop_hz);

/*
 * Biased autocorrelation via FFT (zero-padded, linear): for tau in
 * [0, max_lag]: out_re[tau] + j*out_im[tau] = sum_i x[i] * conj(x[i-tau]).
 * Requires max_lag < n.
 */
void dsp_autocorr(const float *iq, size_t n, size_t max_lag,
                  double *out_re, double *out_im);

#endif /* TE15_DSP_H */

/*
 * Square-root raised-cosine pulse (length `taps`, `samples_per_sym`
 * samples per symbol, roll-off in [0,1]), normalized to unit energy
 * per symbol. Used to synthesize spectrally efficient PSK/QAM test
 * signals. Returns the nominal pulse span in taps.
 */
size_t dsp_srrc_pulse(float *taps, size_t taps_len, int samples_per_sym,
                      double rolloff);

/* Whiten a complex signal in place: X <- X / sqrt(smoothed |X|^2),
 * via FFT filtering with a moving-average kernel (width 5 bins). */
void dsp_whiten(float *iq, size_t n);
