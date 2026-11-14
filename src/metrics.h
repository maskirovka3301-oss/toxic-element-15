/*
 * metrics.h — extended signal-analysis suite and demodulators.
 *
 * metrics_compute() measures a battery of spectral and time-domain
 * features of a region and derives a modulation-class hypothesis plus a
 * demodulation recommendation. The demodulators turn the mixdowned
 * region into a mono audio stream (16-bit WAV via wav.h).
 */
#ifndef TE15_METRICS_H
#define TE15_METRICS_H

#include <stdbool.h>
#include <stddef.h>
#include "iq_source.h"
#include "region.h"

typedef struct Metrics {
    /* --- spectral features --- */
    double occupied_bw_hz;      /* 99% power bandwidth of the region  */
    double occupied_lo_hz;      /* lower edge, absolute RF Hz          */
    double occupied_hi_hz;      /* upper edge, absolute RF Hz          */
    double spectral_flatness;   /* 0..1, geometric/arithmetic PSD mean */
    double band_symmetry;       /* -1..+1; + = symmetric, sign = lean  */
    double tone_purity;         /* 0..1, strongest-bin power fraction  */
    double snr_db;              /* tone peak vs median noise floor     */

    /* --- time-domain features (on a decimated complex stream) --- */
    double am_depth;            /* 0..1 robust envelope depth          */
    double papr_db;             /* envelope crest factor, dB           */
    double duty_cycle;          /* 0..1 on-time fraction (OOK-like)    */
    double fm_deviation_hz;     /* std of instantaneous frequency      */
    int    fsk_levels;          /* 0 = unknown/none, else 2..4         */
    double sym_rate_hz;         /* envelope symbol-rate estimate       */
    double sym_rate_conf;       /* 0..1 confidence in the estimate     */
    double chirp_trend;         /* -1..1 IF vs time correlation        */

    /* --- classification output --- */
    char mod_class[16];         /* CW/AM/FM/FSK/PSK/OOK/OFDM/CHIRP/NOISE */
    bool demodulable;
    char demod_mode[8];         /* "fm", "am", "cw" or ""               */
    char demod_note[128];       /* human-readable classification reason */
} Metrics;

/*
 * Measure the feature suite for `region` of `src` and fill `m`.
 * Returns false for degenerate regions. `m` is zeroed on entry.
 */
bool metrics_compute(const IqSource *src, const Region *region, Metrics *m);

/*
 * Mix `n` interleaved complex samples by `shift_hz` (relative to the
 * current centre): out[i] = in[i] * exp(j 2 pi shift t). `out` needs
 * room for 2*n floats and may alias `iq`.
 */
void metrics_mixdown(const float *iq, size_t n, double src_rate,
                     double shift_hz, float *out);

/*
 * Demodulate `n` interleaved complex samples already mixdowned so the
 * signal of interest sits at DC. Output is mono float audio at
 * `out_rate` Hz (decimated with a box filter), written to `out`
 * (max_out samples). `deviation_hz` (fm) scales the discriminator.
 * Returns the number of samples written.
 */
size_t demod_fm(const float *iq, size_t n, double src_rate, double out_rate,
                double deviation_hz, float *out, size_t max_out);
size_t demod_am(const float *iq, size_t n, double src_rate, double out_rate,
                float *out, size_t max_out);
size_t demod_cw(const float *iq, size_t n, double src_rate, double out_rate,
                float *out, size_t max_out);

#endif /* TE15_METRICS_H */
