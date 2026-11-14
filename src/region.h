/*
 * region.h — a rectangular (frequency, time) selection of a recording,
 * in Hz and seconds relative to the recording start.
 */
#ifndef TE15_REGION_H
#define TE15_REGION_H

#include <stdbool.h>

#include "iq_source.h"

/*
 * Hard cap for the number of complex samples an analysis copies out of
 * the recording into its private buffer (~256 MB at 8 bytes/sample).
 */
#define TE15_MAX_ANALYSIS_SAMPLES 32000000

typedef struct Region {
    double freq_min, freq_max; /* Hz      */
    double time_min, time_max; /* seconds */
} Region;

/*
 * Normalise min/max ordering and clamp to the recording bounds
 * (frequency to the sampled span, time to [0, duration]). Returns false
 * when the region is empty, entirely out of bounds, or degenerate.
 */
bool region_normalize(Region *r, const IqSource *src);

#endif /* TE15_REGION_H */
