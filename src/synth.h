/*
 * synth.h — synthetic signal generator driven by the analysis results.
 *
 * Builds a 2-second IQ segment at the recording's centre frequency whose
 * signal matches the last analysis (modulation family, bandwidth, symbol
 * rate, burst rate/duty, intra-burst pulse timing) and whose noise floor
 * reproduces the measured noise environment: the noise PSD is estimated
 * from signal-free regions of the data underlying the visible waterfall
 * (Welch average over the visible window, outside the selection) and
 * coloured noise is generated from it.
 *
 * The output is written in the loaded file's own sample format.
 */
#ifndef TE15_SYNTH_H
#define TE15_SYNTH_H

#include <stdbool.h>
#include <stddef.h>

#include "analysis.h"
#include "intraburst.h"
#include "iq_source.h"
#include "metrics.h"
#include "modconf.h"
#include "pulse.h"
#include "region.h"
#include "signals.h"

typedef enum {
    SYNTH_OOK_BURSTS = 0, /* bursty OOK/pulsed emission (the common case) */
    SYNTH_FSK,
    SYNTH_PSK,
    SYNTH_OFDM,
    SYNTH_CHIRP,
    SYNTH_CW,
    SYNTH_DSSS,   /* direct-sequence: PN chips at the measured chip rate  */
    SYNTH_FHSS,   /* frequency hopping inside the measured bandwidth     */
} SynthFamily;

typedef struct {
    double center_hz;      /* generation centre (the loaded file's)   */
    double sample_rate;    /* the loaded file's sample rate           */
    double duration;       /* 2 s                                     */
    double bw_hz;          /* occupied bandwidth                       */
    double burst_rate_hz;  /* burst/PRF rate                           */
    double duty;           /* burst duty cycle                         */
    double symbol_rate;    /* baud / chip rate where known             */
    double intra_width_s;  /* intra-burst sub-pulse width              */
    double intra_period_s; /* intra-burst sub-pulse period             */
    double snr_db;         /* measured signal-to-noise ratio           */
    int    n_states;       /* PSK/FSK order                            */
    SynthFamily family;
    IqFormat out_format;   /* the loaded file's format                 */
    /* best-matching database system and its canonical parameters (0/empty
     * when no database match was supplied) */
    char   system[48];
    bool   system_pulsed;
    double canonical_bw_hz;
    double canonical_symbol_rate;
    char   description[256]; /* one-line summary shown in the window   */
} SynthSpec;

/* Noise profile estimated from signal-free data. */
typedef struct {
    double *psd;     /* normalised power spectral density, n bins      */
    int     n;       /* number of bins                                 */
    double  sample_rate;
    double  floor_db;
} SynthNoise;

/*
 * `cand` is the modulation consensus from modconf_rank() (may be NULL);
 * `spread`/`spread_n` are the spread-spectrum classification (may be
 * NULL/0) and `chip_rate_hz` the validated chip rate (0 when unknown).
 * Every available analysis contributes: the spread-spectrum family and the
 * chip rate select DSSS/FHSS/chirp generation, the intra-burst modulation
 * and timing shape the burst envelope, and the pulse analysis sets the
 * burst rate/duty.
 */
bool synth_spec_from(const Metrics *m, const Analysis *a,
                     const PulseResult *pulse, const ModCandidate *cand,
                     const IntraBurstResult *ib,
                     const void *spread, size_t spread_n, double chip_rate_hz,
                     double center_hz, double sample_rate, IqFormat out_format,
                     const RankedMatch *matches, size_t n_matches,
                     SynthSpec *out);

bool synth_noise_profile(const IqSource *src, const Region *visible,
                         const Region *selection, SynthNoise *out);
void synth_noise_clear(SynthNoise *n);

/* Generate the segment. Returns malloc'd interleaved float32 I/Q (2*n
 * floats) and the complex sample count. */
bool synth_generate(const SynthSpec *spec, const SynthNoise *noise,
                    float **iq, size_t *n);

/* Write the generated segment in `format` as a gqrx-style raw file. */
bool synth_write(const char *path, const float *iq, size_t n, IqFormat format);

#endif /* TE15_SYNTH_H */
