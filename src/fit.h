/*
 * fit.h — aspect-ratio-preserving "contain" fit of an image into a box.
 *
 * This is the geometry behind the "Selected Region" preview: the selected
 * waterfall region is scaled so that it fits the box exactly, honouring
 * whether the region is wider than tall (width-constrained) or taller
 * than wide (height-constrained), and centred in the remaining axis.
 */
#ifndef TE15_FIT_H
#define TE15_FIT_H

typedef struct {
    int x, y;  /* top-left corner inside the box */
    int w, h;  /* scaled image size                */
} IntRect;

/*
 * Fit an image of `img_w` x `img_h` pixels into a `box_w` x `box_h`
 * pixel box, preserving aspect ratio and centring the result.
 *
 * Degenerate inputs (zero or negative sizes) yield an empty rect at the
 * centre of the box: {x=box_w/2, y=box_h/2, w=0, h=0}.
 */
IntRect fit_contain(int box_w, int box_h, int img_w, int img_h);

#endif /* TE15_FIT_H */
