/*
 * sdist.h — distribution analysis via the empirical characteristic
 * function and the Laplace transform (moment-generating function)
 * (SPTK: the characteristic function; the Laplace transform).
 *
 * The distance between the empirical characteristic function and the
 * Gaussian characteristic function e^{i mu w - sigma^2 w^2/2} gives a
 * Gaussian-likeness verdict; the empirical MGF probes the distribution's
 * tail behaviour. This is a whole-file statistic (long averaging gain).
 */
#ifndef TE15_SDIST_H
#define TE15_SDIST_H

#include <stdbool.h>
#include "iq_source.h"
#include "region.h"

#define SDIST_GRID 64

typedef struct {
    int     grid;
    double *omega;         /* CF grid                       */
    double *cf_re;         /* empirical CF real part        */
    double *cf_im;         /* empirical CF imag part        */
    double *mgf_s;         /* MGF grid (s values)           */
    double *mgf;           /* empirical MGF                 */
    double  gauss_dist;    /* L2 distance to Gaussian CF    */
    bool    gaussian_like;
    size_t  samples;
    char    verdict[128];
    char    note[256];
} SdistResult;

bool sdist_analyze(const IqSource *src, const Region *region, SdistResult *out);
void sdist_clear(SdistResult *r);

#endif /* TE15_SDIST_H */
