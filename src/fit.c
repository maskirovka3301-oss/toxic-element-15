/*
 * fit.c — aspect-preserving contain fit (see fit.h).
 */
#include "fit.h"

#include <math.h>

IntRect fit_contain(int box_w, int box_h, int img_w, int img_h)
{
    IntRect r = {0, 0, 0, 0};

    /* Degenerate inputs: empty rect centred in the box. */
    if (box_w <= 0 || box_h <= 0 || img_w <= 0 || img_h <= 0) {
        r.x = box_w > 0 ? box_w / 2 : 0;
        r.y = box_h > 0 ? box_h / 2 : 0;
        return r;
    }

    /* Scale factor is limited by whichever axis is relatively tighter,
     * so wide images are width-constrained and tall images are
     * height-constrained. */
    double sx = (double)box_w / (double)img_w;
    double sy = (double)box_h / (double)img_h;
    double s  = sx < sy ? sx : sy;

    r.w = (int)lround((double)img_w * s);
    r.h = (int)lround((double)img_h * s);
    if (r.w < 1) r.w = 1;
    if (r.h < 1) r.h = 1;
    if (r.w > box_w) r.w = box_w; /* rounding guard */
    if (r.h > box_h) r.h = box_h;

    r.x = (box_w - r.w) / 2; /* centre the leftover axis */
    r.y = (box_h - r.h) / 2;
    return r;
}
