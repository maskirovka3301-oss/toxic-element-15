/*
 * ssclass.h — spread-spectrum presence and type classification.
 *
 * Combines every analysis into scored hypotheses for the spread-spectrum
 * families rather than answering only "DSSS present / not present":
 *
 *   DSSS   direct-sequence: a chip rate from the ACF-fluctuation (Burel)
 *          and/or CPAR detectors, ideally agreeing, plus a flat spectrum
 *          and cyclic features consistent with the chip rate;
 *   FHSS   frequency hopping: bursts whose centre frequency changes from
 *          burst to burst (frequency-agile bursts), low duty;
 *   THSS   time hopping: bursts at irregular intervals (interval
 *          modulation / no stable PRF), low duty;
 *   CSS    chirp spread spectrum: chirped pulses (LoRa-like sweeps),
 *          fixed or variable sweep rate;
 *   FH-DSSS hybrid: both a chip rate and hop evidence.
 *
 * Each hypothesis gets a 0..100 confidence with the supporting evidence
 * and, when applicable, the reason a detector pair disagrees.
 */
#ifndef TE15_SSCLASS_H
#define TE15_SSCLASS_H

#include <stddef.h>

#include "analysis.h"
#include "cpar.h"
#include "cyclic.h"
#include "fluct.h"
#include "metrics.h"
#include "pulse.h"
#include "ssca.h"

typedef struct {
    char name[48];       /* DSSS / FHSS / THSS / CSS / FH-DSSS hybrid   */
    int  score;          /* 0..100                                      */
    char evidence[192];  /* why                                         */
} SpreadCandidate;

/*
 * Score the spread-spectrum families. Any analysis pointer may be NULL
 * (not executed). Returns the number of candidates written to `out`,
 * sorted by descending score (the first entry is the best-supported
 * hypothesis; a low best score means "not spread spectrum").
 */
size_t spread_classify(const Metrics *m, const Analysis *a,
                       const PulseResult *pulse, const FluctResult *fluct,
                       const CparResult *cpar, const CyclicResult *cyclic,
                       const SscaResult *ssca, SpreadCandidate *out,
                       size_t max_out);

#endif /* TE15_SSCLASS_H */
