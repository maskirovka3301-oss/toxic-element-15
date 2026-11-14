/*
 * colormap.c — viridis colormap for waterfall rendering.
 *
 * The 256-entry LUT is taken verbatim from the generated data file
 * viridis_lut.inc (which lives in src/, next to this file): one
 * "{R, G, B}," line per entry, values 0..255. No colour values are
 * hardcoded in this file.
 */
#include "colormap.h"

#include <math.h>
#include <string.h>

void viridis_build(Colormap *cm)
{
    static const Rgb8 viridis_data[COLORMAP_ENTRIES] = {
#include "viridis_lut.inc"
    };
    memcpy(cm->table, viridis_data, sizeof(cm->table));
}

Rgb8 viridis_lookup(const Colormap *cm, float t)
{
    if (isnan(t)) {
        t = 0.0f;               /* NaN maps to the low end */
    }
    if (t < 0.0f) {
        t = 0.0f;
    } else if (t > 1.0f) {
        t = 1.0f;
    }

    int idx = (int)(t * 255.0f + 0.5f);
    if (idx < 0) {
        idx = 0;
    } else if (idx > COLORMAP_ENTRIES - 1) {
        idx = COLORMAP_ENTRIES - 1;
    }
    return cm->table[idx];
}
