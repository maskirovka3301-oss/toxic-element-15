/*
 * ccc.h — modulation recognition using cyclic temporal cumulants
 * (Spooner, "Modulation Recognition Using Cyclic Cumulants").
 *
 * The normalised magnitudes of the fourth-order cyclic temporal
 * cumulants C40 (at 4fc), C41 (at 2fc) and C42 (at 2fc) are computed
 * from delay products of the bandpass signal, without carrier removal.
 * The constellation classes have well-separated feature values:
 * BPSK (2,2,2), QPSK (1,0,1), 8PSK (0,0,1), 16QAM (~0.68,0,~0.68),
 * which drives a hierarchical classification verdict.
 */
#ifndef TE15_CCC_H
#define TE15_CCC_H

#include <stdbool.h>
#include "iq_source.h"
#include "region.h"

typedef struct {
    double n40;        /* |C40(4fc)| / P^2   */
    double n41;        /* |C41(2fc)| / P^2   */
    double n42;        /* |C42(2fc)| / P^2   */
    double power;      /* P = <|x|^2>        */
    double carrier_hz; /* fc used (region centre) */
    size_t samples;
    bool   valid;
    char   verdict[128]; /* BPSK-like / QPSK-like / ... */
    char   note[256];
} CccResult;

bool ccc_analyze(const IqSource *src, const Region *region, CccResult *out);
void ccc_clear(CccResult *r);

#endif /* TE15_CCC_H */
