/*
 * intraburst.h — intra-burst time-domain envelope inspection (finest
 * resolution).
 *
 * When the signal has a bursty/pulsed signature this zooms into
 * individual bursts and inspects the envelope INSIDE a burst:
 *
 *   - a uniform pulse train inside the burst (constant widths and
 *     spacings) indicates a clock, a preamble, or a PFM/switching
 *     power-supply signature;
 *   - a sequence of varying pulse widths indicates encoded data
 *     (pulse-width / pulse-position coding);
 *   - a noise-like intra-burst envelope indicates no coherent
 *     intra-burst structure.
 *
 * The analysis deliberately does NOT reuse the pulse module's decimated
 * envelope: it re-reads a fresh copy of the raw I/Q at the FULL sample
 * rate, mixes it down at the selection centre, removes the residual
 * carrier offset estimated from the cyclic-cumulant carrier, and builds
 * the envelope without decimation or smoothing, so pulse edges stay as
 * sharp as the sample rate allows (resolution = 1/fs).
 */
#ifndef TE15_INTRABURST_H
#define TE15_INTRABURST_H

#include <stdbool.h>

#include "ccc.h"
#include "iq_source.h"
#include "region.h"
#include "sot.h"

typedef struct {
    bool   valid;          /* frame contained usable bursts          */
    int    bursts_examined;/* bursts inspected                       */
    int    sub_pulses;     /* intra-burst segments found (total)     */
    double burst_len_s;    /* mean burst length                      */
    double width_mean_s;   /* mean intra-burst segment width         */
    double width_cv;       /* width variability inside the bursts    */
    double period_mean_s;  /* mean intra-burst segment spacing       */
    double period_cv;      /* spacing variability                    */
    double rise_mean_s;    /* mean 10-90% rise of the sub-pulses     */
    double fall_mean_s;    /* mean 90-10% fall of the sub-pulses     */
    double duty;           /* intra-burst on-time fraction           */
    double env_cv;         /* intra-burst envelope variability       */
    double sample_rate;    /* rate the inspection ran at, Hz         */
    double resolution_s;   /* 1 / sample_rate                        */
    double residual_cfo_hz;/* carrier offset removed via the cumulant
                            * carrier estimate (0 when unavailable)  */
    /* --- context-derived band filter (full-rate slice) ---------------
     * The slice is read over the FULL original band (no frequency
     * extraction); the selection is mixed to DC and a low-pass with a
     * passband that keeps the OOK/ASK edge sidelobes but rejects
     * everything farther away is designed from the wideband spectrum. */
    double filter_pass_hz; /* passband edge (kept), Hz                */
    double filter_stop_hz; /* stopband edge (rejected beyond), Hz     */
    double filter_trans_hz;/* transition width, Hz                    */
    int    filter_taps;    /* FIR length                              */
    double neighbour_hz;   /* nearest out-of-selection emission, Hz   */
    double noise_floor_db; /* wideband noise floor estimate, dB       */
    /* --- synchronised folding ---------------------------------------- */
    double rms_window_s;   /* short-RMS envelope window used          */
    int    bursts_detected;/* burst regions found                     */
    double period_hz;      /* burst period from the envelope spectrum */
    int    folded_bursts;  /* bursts folded into the average          */
    int    fold_coherent;  /* 1 = phase-coherent complex fold used    */
    double coherence;      /* |<x>|/<|x|> across the fold             */

    /* --- sample-level segmentation of the folded average ------------- */
    double on_level;       /* on level (2-component amplitude model)  */
    double off_level;      /* off level                               */
    int    delta_bic;      /* multi-segment vs single-segment BIC     */
    double surrogate_fpr;  /* phase-randomised false-alarm rate       */
    int    detectors_confirming; /* of {levels, hysteresis, change-
                                  * point, HMM, multiscale edges}      */

    /* --- per-sub-pulse characterisation ------------------------------ */
    double snr_db;         /* per-pulse peak/floor SNR                */
    double if_span_hz;     /* intra-pulse frequency span              */
    int    fsk_steps;      /* distinct intra-pulse IF levels          */
    double bandwidth_hz;   /* -3 dB bandwidth of the mean pulse       */
    double matched_filter; /* correlation with the mean pulse         */

    int    uniform;        /* 1 = uniform train (clock-like)         */
    /* intra-burst modulation hypotheses (PPM/PIM/PWM/PFM/Manchester/...)*/
    char   burst_mod[96];      /* e.g. "PPM, 4 equally-spaced slots"     */
    char   burst_mod_ev[192];  /* the evidence behind the hypothesis     */
    /* per-burst classification over the individually isolated bursts */
    int    bursts_classified;  /* how many bursts were classified         */
    char   burst_mod_cand[2][96]; /* top two candidates (per-burst votes)  */
    double burst_mod_support[2];  /* their support fraction (0..1)         */
    char   burst_mod_combo[160];  /* likelihood-based choice: single class
                                   * or a preferred combination            */
    double burst_mod_dll;         /* 2*(LL_combo - LL_best_single)        */
    double burst_mod_threshold;   /* BIC threshold (ln of burst count)    */
    char   burst_mod_choice[192]; /* how the choice was made (evidence)   */
    int    width_levels;       /* discrete width levels (0 = continuous) */
    int    interval_levels;    /* discrete interval levels (0 = cont.)   */
    char   verdict[80];    /* human-readable classification          */
    char   note[1024];
} IntraBurstResult;

void intraburst_clear(IntraBurstResult *r);

/*
 * Inspect the envelope inside the bursts of the analysed selection.
 * `ccc` may be NULL (or invalid) when the cyclic-cumulant stage did not
 * run: the residual carrier offset then stays uncorrected.
 */
bool intraburst_analyze(const IqSource *src, const Region *region,
                        const CccResult *ccc, IntraBurstResult *out);

/*
 * Analyse the SHARED prepared source (sot_prepare): the data is already
 * mixed and band-filtered, so only the residual-CFO de-rotation (from the
 * cyclic-cumulant carrier) is applied locally on a copy.
 */
bool intraburst_analyze_sot(const Sot *sot, const CccResult *ccc,
                            IntraBurstResult *out);

#endif /* TE15_INTRABURST_H */
