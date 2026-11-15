/*
 * sot.h — single source of truth for the analysis suite.
 *
 * Before any analysis runs, ONE shared buffer is prepared from the
 * recording:
 *   - the selection's TIME range copied across the FULL original band, at
 *     the original sample rate (no decimation, no frequency extraction);
 *   - mixed down at the selection centre;
 *   - band-filtered with settings derived ONLY from the user's selection
 *     (the selected bandwidth defines the passband, a fixed transition the
 *     stopband, and the in-band PSD the noise floor): no wideband context,
 *     neighbouring emissions or other data outside the selection is
 *     consulted, because the analyses must be prior-agnostic.
 *
 * The buffer is exposed as a pseudo-IqSource so every analysis reads the
 * same data through the existing API. Together with the settings and the
 * results published by analyses that ran earlier, this is the ONLY input
 * every analysis gets: no signal database, known-signal hint or wideband
 * context is passed to them. Each analysis may still decimate
 * for cost, but only by the factor dsp_decimation_for() allows - a fixed
 * low target rate would alias and throw away the source resolution.
 */
#ifndef TE15_SOT_H
#define TE15_SOT_H

#include <stdbool.h>

#include "iq_source.h"
#include "region.h"

typedef struct {
    float   *iq;            /* prepared full-rate complex samples        */
    size_t   n;             /* complex samples                           */
    IqSource view;          /* pseudo-source over `iq`                   */
    Region   region;        /* the selection (original coordinates)      */
    double   mix_hz;        /* frequency shift applied to DC             */
    double   pass_hz;       /* context filter passband edge              */
    double   stop_hz;       /* context filter stopband edge              */
    double   trans_hz;      /* transition width                          */
    double   noise_floor_db;/* wideband noise floor                      */
    /* Uniform real-gain normalisation applied to the raw slice BEFORE any
     * mixing/filtering. A single real scalar cannot change the phase
     * (arg x) or the frequency of any component - it only scales all I/Q
     * magnitudes by the same known factor, so dividing by it recovers the
     * original values exactly. rms_in is the RMS before the gain. */
    double   gain;          /* applied multiplier                        */
    double   gain_db;       /* 20*log10(gain)                            */
    double   rms_in;        /* RMS of the raw slice before normalising   */
    bool     valid;
} Sot;

/*
 * Prepare the shared source for `region`. Returns false (and leaves the
 * structure invalid) when the selection cannot be read. `out` must be
 * zero-initialised by the caller (sot_clear() is called first).
 */
bool sot_prepare(const IqSource *src, const Region *region, Sot *out);

/* Release the prepared buffer. */
void sot_clear(Sot *out);

#endif /* TE15_SOT_H */
