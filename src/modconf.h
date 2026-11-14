/*
 * modconf.h — modulation-type confidence scoring.
 *
 * Every modulation type the analysis suite can detect gets a 0..100
 * confidence score assembled from all the evidence: the classifier's
 * decision, the FFT modulation string, the cyclic-cumulant constellation
 * verdict, the higher-order-statistics hint, the pulse-train structure,
 * and the measured feature consistency (duty cycle, crest factor,
 * spectral flatness, FM deviation, FSK levels, chirp trend, DSSS
 * detector agreement).
 */
#ifndef TE15_MODCONF_H
#define TE15_MODCONF_H

#include <stddef.h>

#include "analysis.h"
#include "cpar.h"
#include "ccc.h"
#include "cyclic.h"
#include "fluct.h"
#include "hos.h"
#include "metrics.h"
#include "pulse.h"

typedef struct {
    char name[48];      /* e.g. "OFDM / QAM", "FM", "CW", "DSSS" */
    int  score;         /* 0..100                                 */
    char why[96];       /* brief supporting evidence              */
} ModCandidate;

/*
 * Score all detectable modulation types. Any of the optional result
 * pointers may be NULL (that analysis did not run). Returns the number
 * of candidates written to `out` (sorted by descending score, capped at
 * max_out, a good value is 12).
 */
size_t modconf_rank(const Analysis *a, const Metrics *m,
                    const CyclicResult *cyc, const CccResult *ccc,
                    const HosResult *hos, const PulseResult *pulse,
                    const FluctResult *fluct, const CparResult *cpar,
                    ModCandidate *out, size_t max_out);

#endif /* TE15_MODCONF_H */
