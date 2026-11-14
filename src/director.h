/*
 * director.h — the evidence director.
 *
 * After the DAG has executed, the Director consumes the per-analysis
 * envelopes (AnalysisResult) plus the DAG run status, the program settings
 * and the file/selection context, and produces ONE consolidated view:
 *
 *   - ranked hypotheses (claims) with their weight and evidence,
 *   - the best-supported characterisation sentence and the final verdict,
 *   - the cross-analysis agreement summaries (modulation votes, cycle
 *     frequency cross-check, chip-rate/DSSS consistency, detector presence),
 *   - a provenance list of every DAG node (executed/skipped/failed, why,
 *     how long) and a rationale trail explaining how the result was ranked
 *     and which analyses it discounted.
 *
 * The Reporter renders this struct; nothing downstream needs to know any
 * analysis' private result struct. The Director is GTK-free and can be
 * unit-tested with synthetic envelopes.
 */
#ifndef TE15_DIRECTOR_H
#define TE15_DIRECTOR_H

#include <stdbool.h>
#include <stddef.h>

#include "aresult.h"
#include "dag.h"
#include "modconf.h"
#include "region.h"
#include "ssclass.h"

#define DIR_MAX_CAND     10
#define DIR_MAX_RATIONALE 24
#define DIR_MAX_PROV     (DAG_NODES + 1)

typedef struct {
    const char *file;       /* recording name (basename)                  */
    double      center_hz;
    double      rate_hz;
    double      duration;
    Region      region;     /* analysed selection                          */
    int         fft_size;
    const char *format;     /* recording data format label                 */
    const char *window_name;

    const ArSet        *analyses;
    const DagRun       *dag;
    const ModCandidate *mods;    /* modulation consensus candidates         */
    size_t              n_mods;
    const SpreadCandidate *spread;
    size_t              n_spread;
} DirectorInput;

typedef struct {
    char        name[96];      /* the hypothesis                          */
    double      weight;        /* 0..100                                   */
    char        evidence[192]; /* why it scores this                       */
    const char *source;        /* analysis id that supports it             */
} DirClaim;

typedef struct {
    DirClaim cand[DIR_MAX_CAND];
    size_t   n_cand;

    /* consolidated summaries the report prints */
    char characterisation[512]; /* best-supported characterisation        */
    char verdict[1024];         /* final verdict sentence                 */
    char top_mod[64];           /* strongest modulation candidate         */
    char vote_line[200];        /* 'demod x | cumulants y | ...'          */
    char majority[24];
    int  majority_count;
    char cf_summary[160];       /* cycle-frequency cross-check            */
    char dsss_label[128];       /* chip-rate/DSSS consistency             */
    double bchip, cchip;        /* Burl / CPAR chip rates                 */
    int  presence;              /* detector presence count (0..4)         */

    /* rationale + provenance */
    char rationale[DIR_MAX_RATIONALE][192];
    size_t n_rationale;
    char provenance[DIR_MAX_PROV][192];
    size_t n_provenance;
} Director;

/* Build the consolidated view. `d` is fully overwritten. */
void director_build(Director *d, const DirectorInput *in);

/* Modulation token used by the vote summary (BPSK/QPSK/OFDM/...). */
void director_mod_token(const char *s, char *token, size_t cap);

#endif /* TE15_DIRECTOR_H */
