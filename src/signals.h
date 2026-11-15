/*
 * signals.h — known-signal database and matching (sigidwiki-style).
 */
#ifndef TE15_SIGNALS_H
#define TE15_SIGNALS_H

#include <stddef.h>

#include "modconf.h"
#include "pulse.h"

typedef struct Metrics Metrics; /* metrics.h; forward-declared to avoid a cycle */

/*
 * Modulation families, assigned from the literature (sigidwiki / ITU /
 * common practice) for each database entry. Used by signals_rank to
 * score measured-parameter agreement.
 */
typedef enum {
    SIG_FAM_UNKNOWN = 0,
    SIG_FAM_AM,     /* amplitude-modulated voice (AM, VSB TV)      */
    SIG_FAM_FM,     /* frequency-modulated voice (FM, NFM, WFM)    */
    SIG_FAM_CW,     /* continuous-wave tone / modulated CW (MCW)   */
    SIG_FAM_PSK,    /* phase shift keying                           */
    SIG_FAM_FSK,    /* frequency shift keying (incl. GMSK, AFSK)    */
    SIG_FAM_OOK,    /* on-off keying / amplitude shift keying       */
    SIG_FAM_OFDM,   /* multi-carrier (LTE, WiFi, DVB, ...)          */
    SIG_FAM_CHIRP,  /* linear frequency sweeps (LoRa, radar)        */
    SIG_FAM_SSB,    /* single-sideband voice                        */
} SignalFamily;

typedef struct {
    const char *name;
    const char *band;      /* VLF / LF / MF / HF / VHF / UHF */
    double      freq_min;  /* Hz */
    double      freq_max;  /* Hz */
    double      bw_min;    /* Hz */
    double      bw_max;    /* Hz */
    SignalFamily family;   /* expected modulation family */
} KnownSignal;

typedef struct {
    const KnownSignal *signal;
    int                confidence; /* 0..100 */
} SignalMatch;

typedef struct {
    const KnownSignal *signal;
    int base_confidence; /* 0..120, the original frequency/bandwidth score */
    int agreement;       /* -30..+30 bonus from measured-parameter agreement */
    int total;           /* base + agreement, used for ranking              */
    int confidence_pct;  /* 0..100 normalised confidence                    */
    char why[160];       /* matching basis, e.g. "12.5 kHz channel match; 4FSK family; 4.8 kBd" */
    int  in_band;        /* 1 = the measured centre lies in the system's
                          * licensed frequency range; 0 = out of band     */
    int  valid;          /* 1 = in band AND bandwidth AND modulation
                          * conform to the documented specification       */
    char invalid_reason[96]; /* why the candidate is invalid (empty when valid) */
} RankedMatch;

/*
 * Published characteristic parameters for the major systems, from the
 * standards/ITU/ETSI/sigidwiki references cited per entry. Used by the
 * confidence model (bandwidth/symbol-rate/pulsed matching); entries
 * without a spec use only the frequency/bandwidth/family comparisons.
 */
typedef struct {
    const char *name;      /* must match a signal_db entry name          */
    const char *desc;      /* published characteristic summary           */
    double channel_bw;     /* nominal channel bandwidth, Hz (0 = n/a)    */
    double symbol_rate;    /* typical symbol/chip rate, Hz (0 = n/a)     */
    int    pulsed;         /* 1 = pulsed/bursty emission                 */
    const char *ref;       /* source of the parameters                   */
} SignalSpec;

extern const SignalSpec signal_specs[];
extern const size_t     signal_specs_size;

/* Look up the spec for a system name (NULL when not tabulated). */
const SignalSpec *signal_spec_find(const char *name);

/*
 * Anomaly detection: when the measured signal lies outside the licensed
 * frequency range of every candidate (or inside a band but wildly
 * inconsistent with its published channel width / modulation), the
 * candidates cannot be considered valid. Returns 1 when the signal is an
 * anomaly and writes an explanation comparing the measured centre,
 * bandwidth and modulation family with the nearest candidates' specs
 * into `explain`. Returns 0 when at least one candidate is valid.
 */
int signals_anomaly_check(double center_freq, double bandwidth,
                          const RankedMatch *ids, size_t n,
                          char *explain, size_t explain_len);

extern const KnownSignal signal_db[];
extern const size_t      signal_db_size;

/*
 * Candidate-first identification helpers (used by the Spectrum DAG):
 *   - signals_preselect(): entries that are POSSIBLE given only the measured
 *     centre frequency and the approximate bandwidth (the band must contain
 *     the centre and the bandwidth must be compatible with the documented
 *     channel/occupied width). No modulation evidence is used.
 *   - signals_family_from_mod_name(): map a modulation-candidate name to a
 *     SignalFamily (-1 when it implies none).
 *   - signals_filter_family(): keep the preselected candidates whose
 *     documented modulation family matches the detected one.
 * Both list helpers sort by descending frequency/bandwidth confidence.
 */
size_t signals_preselect(double center_freq, double bandwidth,
                         SignalMatch *out, size_t max_matches);
int    signals_family_from_mod_name(const char *name);
size_t signals_filter_family(const SignalMatch *in, size_t n, int family,
                             SignalMatch *out, size_t max_matches);

/*
 * Score the database against an observed (center frequency, bandwidth)
 * pair and return the best matches, sorted by descending confidence.
 * `out` may be NULL if `max_matches` is 0. Returns the number of matches.
 */
size_t signals_match(double center_freq, double bandwidth,
                     SignalMatch *out, size_t max_matches);

/*
 * Rank the most likely known-signal candidates for the analysed region,
 * combining the database confidence with measured parameters (`m`),
 * using literature-based agreement:
 *   - occupied-bandwidth containment (+/-15)
 *   - modulation-family match (+20) / mismatch (-15)
 *   - family-specific parameter consistency (e.g. Carson's rule for FM,
 *     duty cycle for OOK, crest factor for OFDM, symbol rate tables for
 *     known digital modes): +10 each
 * Fills up to `max_matches` entries, sorted by descending `total`.
 * Returns the number of ranked candidates. `m` may be NULL, in which
 * case the result equals the plain database ranking.
 */
size_t signals_rank(double center_freq, double bandwidth, const Metrics *m,
                    RankedMatch *out, size_t max_matches);

/*
 * Rank known systems combining the database score, the measured metrics
 * AND the modulation-confidence evidence (the top candidates win the
 * family-agreement bonus according to their score weight). Fills
 * confidence_pct with a 0..100 normalised confidence. `mods` may be
 * NULL (equivalent to signals_rank plus percentage normalisation).
 */
size_t signals_rank_conf(double center_freq, double bandwidth,
                         const Metrics *m, const ModCandidate *mods,
                         size_t nmods, const PulseResult *pulse,
                         RankedMatch *out, size_t max_matches);

#endif /* TE15_SIGNALS_H */
