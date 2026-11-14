/*
 * cyclic.h — cyclostationary analysis of a selection.
 *
 * Computes the spectral correlation function (SCF) with the FFT
 * Accumulation Method (FAM, Gardner), derives the cyclic domain profile
 * (CDP = max over f of |SCF| per cycle frequency alpha), and extracts
 * the strongest cyclic features — peaks at the symbol/baud/chip rate
 * and its harmonics, which are the fingerprints of modulated signals.
 */
#ifndef TE15_CYCLIC_H
#define TE15_CYCLIC_H

#include <stdbool.h>
#include "iq_source.h"
#include "region.h"

#define CYCLIC_N        1024  /* SCF grid: CYCLIC_N x CYCLIC_N     */
#define CYCLIC_MAX_PEAKS 12   /* reported cyclic features          */
#define CYCLIC_MAX_CF 32      /* coherence-thresholded CF list     */

typedef struct {
    double alpha;     /* cyclic frequency, Hz    */
    double strength;  /* normalized CDP value    */
} CyclicPeak;

typedef struct {
    int    n;         /* grid size (CYCLIC_N)                    */
    double alpha_res; /* Hz per bin                             */
    double fs_eff;    /* effective sample rate after decimation  */

    float  *scf;      /* n*n magnitudes (alpha-major), 0..1      */
    double *cdp;      /* n normalized profile values             */

    CyclicPeak peaks[CYCLIC_MAX_PEAKS];
    int        peak_count;

    /* coherence-thresholded significant cycle frequencies:
     * (alpha Hz, normalised coherence) pairs */
    double cf_list[CYCLIC_MAX_CF][2];
    size_t cf_count;
    double cf_threshold;
    double cf_pfa_threshold; /* P_FA-based FAM coherence threshold */
} CyclicResult;

/*
 * Run a cyclostationary analysis over `region` of `src` and fill `out`.
 * The selection is copied into a private buffer first (like the rest of
 * the analysis suite). Returns false for degenerate regions or too
 * little data. Previous contents of `out` are released.
 *
 * `conjugate` selects the conjugate spectral correlation function
 * (S_x*^alpha), whose denominator uses the PSD mirrored around
 * alpha/2; false = the non-conjugate SCF.
 */
bool cyclic_analyze_ex(const IqSource *src, const Region *region,
                       bool conjugate, CyclicResult *out);
bool cyclic_analyze(const IqSource *src, const Region *region,
                    CyclicResult *out);

/* Free heap storage held by a result. */
void cyclic_clear(CyclicResult *r);

#endif /* TE15_CYCLIC_H */
