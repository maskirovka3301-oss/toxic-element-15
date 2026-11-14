/*
 * fcdrift.h — device warm-up drift estimation and correction.
 *
 * Frequency counters and their reference oscillators drift while the device
 * warms up: the measured frequency approaches its settled value along a
 * roughly exponential (often double-exponential: fast thermal transient +
 * slow asymptote) curve. Literature:
 *
 *   - OCXO/TCXO warm-up curves are modelled as f(t) = f_inf + A*exp(-t/tau)
 *     or a sum of two exponentials (fast + slow thermal time constants);
 *   - Mitscherlich/asymptotic regression with a "settling time" (the time
 *     to reach 95 % of the final value, t95 = tau*ln(20) ~ 3*tau) is the
 *     usual device specification;
 *   - exponential fits are solved efficiently by variable projection: grid
 *     search over tau with linear least squares for the amplitudes
 *     (Golub & Pereyra);
 *   - robust (Huber/IRLS) regression rejects transient outliers such as
 *     multipath fades; models are selected by BIC;
 *   - across rounds/datasets the time constant is pooled robustly (median
 *     of per-round estimates), which is the "true device temperature drift
 *     and settling time" used to correct every round.
 *
 * Observation times: when a dataset provides per-measurement times (seconds
 * since video start, a future schema addition), they are used; until then
 * the measurement index is the time base ("pseudo-time", uniform sampling),
 * which is the correct proxy when samples are taken at a constant cadence.
 */
#ifndef TE15_FCDRIFT_H
#define TE15_FCDRIFT_H

#include <stdbool.h>
#include <stddef.h>

typedef enum {
    FCD_MODEL_NONE = 0,   /* no significant drift: values unchanged      */
    FCD_MODEL_CONSTANT,   /* flat level (no drift)                       */
    FCD_MODEL_LINEAR,     /* polynomial drift                            */
    FCD_MODEL_EXP,        /* single exponential warm-up                  */
    FCD_MODEL_EXP2        /* fast + slow thermal transients              */
} FcdModel;

typedef struct {
    FcdModel model;
    double   tau1;        /* fast/single time constant (time units)      */
    double   tau2;        /* slow time constant (0 when unused)          */
    double   amp1;        /* drift amplitude of the fast term (Hz)       */
    double   amp2;        /* drift amplitude of the slow term (Hz)       */
    double   level;       /* settled level (Hz)                          */
    double   slope;       /* linear model slope (Hz per time unit)       */
    double   rms_before;  /* residual RMS before correction (Hz)         */
    double   rms_after;   /* residual RMS after correction (Hz)          */
    double   settle_time; /* time to reach 95 % of the settled value     */
    size_t   n;
    bool     corrected;   /* correction was applied                      */
    bool     timestamps;  /* true: times are seconds; false: sample index*/
} FcdResult;

/* Fit the drift model to one round/subject's measurements.
 * `times` may be NULL (measurement index is then used). Returns false when
 * the series is too short or the fit is unusable. */
bool fcd_estimate(const double *values, const double *times, size_t n,
                  FcdResult *out);

/* Write the drift-corrected series (same length/order). */
void fcd_correct(const double *values, const double *times, size_t n,
                 const FcdResult *result, double *corrected);

/* Pooled device parameters across rounds/datasets (robust medians). */
typedef struct {
    double tau_median;    /* pooled time constant of the slow term      */
    double t95_median;    /* pooled 95 % settling time                  */
    double amp_median;    /* typical peak drift amplitude (Hz)          */
    size_t n_rounds;      /* rounds that contributed a drift model      */
    size_t n_corrected;   /* rounds that were actually corrected        */
    bool   timestamps;    /* any round used real timestamps             */
} FcdGlobal;

void fcd_pool(const FcdResult *results, size_t n, FcdGlobal *global);

const char *fcd_model_name(FcdModel model);

/* --- observation-time stub -------------------------------------------- */
/* The current JSON schema has no per-measurement timestamps. These hooks are
 * the extension point for the future "timestamp relative to video start"
 * field: when the schema gains a per-measurement `times` array the loader
 * populates it and these return true, so the drift fitter automatically
 * uses seconds instead of the sample index. */
bool fcd_timestamps_available(void);
/* Returns the per-measurement times of a round in seconds, or NULL. */
const double *fcd_round_times(const void *round, size_t *n_out);

#endif /* TE15_FCDRIFT_H */
