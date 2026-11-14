/*
 * pulse.h — pulse-train analysis of a user-selected region.
 *
 * Follows the standard pulse-analysis workflow: envelope extraction
 * (magnitude + smoothing), thresholding with hysteresis, edge
 * detection, false-trigger rejection, pulse grouping, and per-train
 * metrics (width tau, period T, PRF, duty cycle, inter-pulse interval,
 * rise/fall times, peak/RMS amplitude, jitter), plus the frequency
 * comb (line spectrum, fundamental f0 = 1/T, sinc envelope and the
 * 1/tau nulls) and a modulation-style verdict (constant-PRF / PWM /
 * PPM). Selection-scoped.
 */
#ifndef TE15_PULSE_H
#define TE15_PULSE_H

#include <stdbool.h>
#include "iq_source.h"
#include "region.h"

#define PULSE_MAX_PULSES 2048
#define PULSE_FS_TARGET  500000.0 /* envelope processing rate, Hz */
#define PULSE_ENV_FFT    65536

typedef struct {
    double  fs;              /* effective envelope rate, Hz      */
    double *env;             /* smoothed envelope (n samples)    */
    size_t  env_len;
    double  threshold;       /* detection threshold              */
    int     num_pulses;
    double  pulse_start[PULSE_MAX_PULSES]; /* sample indices */
    double  pulse_end[PULSE_MAX_PULSES];
    double  pulse_amp[PULSE_MAX_PULSES];   /* peak amplitude   */

    double  pulse_width_s;   /* mean tau                         */
    double  period_s;        /* mean T (start-to-start)          */
    double  prf_hz;          /* 1/T                              */
    double  duty_cycle;      /* tau / T                          */
    double  ipi_s;           /* mean inter-pulse interval        */
    double  rise_s, fall_s;  /* mean 10%-90% / 90%-10% times     */
    double  von_peak;        /* mean peak amplitude              */
    double  arms;            /* RMS of the selection             */
    double  jitter_s;        /* std of the start-to-start periods */
    double  ipi_std_s;       /* std of the IPIs                  */

    double *line_spectrum;   /* dB |FFT(mean-removed envelope)|  */
    int     line_n;
    double  f0_hz;           /* fundamental of the line spectrum */
    double  sinc_null_hz;    /* first null -> 1/tau estimate     */
    double  tau_from_sinc_s;

    /* --- intra-pulse analysis (variable waveforms / chirps) --- */
    double  chirp_rate_mean;  /* mean linear IF slope, Hz/s        */
    double  chirp_rate_cv;    /* variability across pulses (0..)   */
    double  chirp_bw_mean;    /* mean swept bandwidth, Hz          */
    int     chirp_pulses;     /* pulses with a significant chirp   */
    int     up_chirps;        /* upward sweeps among those         */
    int     down_chirps;      /* downward sweeps among those       */
    double  pulse_freq_cv;    /* cv of the pulse centre freqs      */
    /* --- interval-modulation analysis (PPM / PIM / PWM) --- */
    int     interval_levels;  /* discrete interval clusters (0=none) */
    double  interval_cv;      /* cv of the pulse periods           */
    double  width_cv;         /* cv of the pulse widths            */
    char    mod_detail[96];   /* e.g. "PPM (4 discrete intervals)" */

    char    modulation[64];  /* constant-PRF / PWM / PPM / PIM / chirps */
    char    note[512];

    /* --- credibility of the pulse-train interpretation ------------------
     * A near-continuous envelope, a period jitter comparable to the
     * period, or a flat envelope (no pulsed structure) mean the detected
     * "pulses" are threshold crossings of envelope noise, NOT a train:
     * the PRF is then meaningless and must not be reported. */
    int     credible;        /* 1 = the pulse train is credible        */
    double  env_cv;          /* envelope std-dev / mean                */
    double  jitter_frac;     /* jitter / period                        */
    char    credibility_note[160];
} PulseResult;

bool pulse_analyze(const IqSource *src, const Region *region, PulseResult *out);
void pulse_clear(PulseResult *r);

#endif /* TE15_PULSE_H */
