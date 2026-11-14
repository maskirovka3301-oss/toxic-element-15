/*
 * region.c — selection normalisation (see region.h).
 */
#include "region.h"

bool region_normalize(Region *r, const IqSource *src)
{
    if (r == NULL || src == NULL) return false;

    /* Order min/max. */
    if (r->freq_min > r->freq_max) {
        double t = r->freq_min; r->freq_min = r->freq_max; r->freq_max = t;
    }
    if (r->time_min > r->time_max) {
        double t = r->time_min; r->time_min = r->time_max; r->time_max = t;
    }

    /* Clamp to the recording. */
    double f_lo = src->center_freq - src->sample_rate / 2.0;
    double f_hi = src->center_freq + src->sample_rate / 2.0;
    if (r->freq_min < f_lo) r->freq_min = f_lo;
    if (r->freq_max > f_hi) r->freq_max = f_hi;
    if (r->time_min < 0.0)  r->time_min = 0.0;
    if (r->time_max > src->duration) r->time_max = src->duration;

    /* Degenerate or empty after clamping? */
    if (r->freq_max <= r->freq_min || r->time_max <= r->time_min) {
        return false;
    }
    return true;
}
