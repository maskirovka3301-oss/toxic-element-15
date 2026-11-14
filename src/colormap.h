/*
 * colormap.h — viridis colormap for waterfall rendering.
 */
#ifndef TE15_COLORMAP_H
#define TE15_COLORMAP_H

#include <stdint.h>

#define COLORMAP_ENTRIES 256

typedef struct {
    uint8_t r, g, b;
} Rgb8;

typedef struct {
    Rgb8 table[COLORMAP_ENTRIES];
} Colormap;

/*
 * Fill `cm` with the viridis colormap (256 entries, piecewise-linear
 * interpolation over the canonical 33 anchor colours).
 */
void viridis_build(Colormap *cm);

/*
 * Look up the colour for t in [0,1]; values outside the range are clamped.
 */
Rgb8 viridis_lookup(const Colormap *cm, float t);

#endif /* TE15_COLORMAP_H */
