/*
 * analysis.h — region analysis of an IQ recording.
 */
#ifndef TE15_ANALYSIS_H
#define TE15_ANALYSIS_H

#include <stdbool.h>
#include <stddef.h>
#include "iq_source.h"
#include "region.h"

typedef struct {
    double center_freq; /* Hz */
    double bandwidth;   /* Hz */
    double duration;    /* s  */
    double avg_power;   /* dB */
    double max_power;   /* dB */
    double entropy;     /* bits */
    double prf;         /* Hz */

    char signal_type[32];
    char modulation[32];

    /*
     * Spectrogram of the region: `frames` rows (oldest first, i.e. row 0
     * is the earliest time) x `bins` columns (ascending frequency).
     * Values are dB magnitudes. `bin_hz` is the frequency width of one
     * column and `dt` the time step between rows.
     */
    float *image;
    int    frames;
    int    bins;
    double bin_hz;
    double dt;
} Analysis;

/* Free any heap storage held by an analysis. */
void analysis_clear(Analysis *a);

/*
 * Analyse a region of `src` and fill `out`. The previous contents of
 * `out` are released.
 *
 * The selected part of the recording is first copied out of the memory
 * map into a private buffer, and the whole analysis runs on that buffer.
 * Segments longer than TE15_MAX_ANALYSIS_SAMPLES (see region.h) are
 * truncated (the first part is kept) with a warning.
 *
 * Returns false (and logs to stderr) when the region is empty, out of
 * bounds, or too short to analyse.
 */
bool analysis_run(const IqSource *src, const Region *region, Analysis *out);

#endif /* TE15_ANALYSIS_H */
