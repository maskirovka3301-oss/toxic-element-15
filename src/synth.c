/*
 * synth.c — synthetic signal generator driven by the analysis results.
 */
#include "synth.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsp.h"
#include "ssclass.h"

#define SYNTH_SECONDS 2.0
#define SYNTH_PSD_N   1024

/* deterministic PRNG so a saved file can be regenerated */
typedef struct {
    unsigned long long s;
} Rng;

static double rnd(Rng *r)
{
    r->s = r->s * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)((r->s >> 11) & 0xFFFFFFFFFFFFULL) / 281474976710656.0;
}

/* --------------- specification from the analysis ---------------------- */

static void pick_family(const char *name, SynthFamily *f, int *states)
{
    *states = 2;
    if (name == NULL || name[0] == '\0') {
        *f = SYNTH_OOK_BURSTS;
        return;
    }
    if (strstr(name, "OFDM") != NULL) {
        *f = SYNTH_OFDM;
    } else if (strstr(name, "PSK") != NULL ||
               strstr(name, "QAM") != NULL) {
        *f = SYNTH_PSK;
        *states = strstr(name, "16") != NULL ? 16
                  : strstr(name, "8") != NULL  ? 8
                  : strstr(name, "QPSK") != NULL ? 4 : 2;
    } else if (strstr(name, "FSK") != NULL || strstr(name, "GFSK") != NULL) {
        *f = SYNTH_FSK;
        *states = strstr(name, "4") != NULL ? 4 : 2;
    } else if (strstr(name, "chirp") != NULL || strstr(name, "CSS") != NULL ||
               strstr(name, "LFM") != NULL) {
        *f = SYNTH_CHIRP;
    } else if (strstr(name, "CW") != NULL) {
        *f = SYNTH_CW;
    } else {
        *f = SYNTH_OOK_BURSTS;
    }
}

/* Map a database modulation family to a synthesis model. */
static bool synth_family_from_db(int family, SynthFamily *f, int *states)
{
    switch (family) {
    case SIG_FAM_OFDM: *f = SYNTH_OFDM; return true;
    case SIG_FAM_FSK: *f = SYNTH_FSK; *states = 2; return true;
    case SIG_FAM_PSK: *f = SYNTH_PSK; *states = 4; return true;
    case SIG_FAM_OOK: *f = SYNTH_OOK_BURSTS; return true;
    case SIG_FAM_CHIRP: *f = SYNTH_CHIRP; return true;
    case SIG_FAM_CW: *f = SYNTH_CW; return true;
    default: return false;
    }
}

bool synth_spec_from(const Metrics *m, const Analysis *a,
                     const PulseResult *pulse, const ModCandidate *cand,
                     const IntraBurstResult *ib,
                     const void *spread, size_t spread_n,
                     double chip_rate_hz, double center_hz,
                     double sample_rate, IqFormat out_format,
                     const RankedMatch *matches, size_t n_matches,
                     SynthSpec *out)
{
    const char *cand_name = cand != NULL ? cand->name : NULL;
    const SpreadCandidate *sc = (const SpreadCandidate *)spread;

    if (out == NULL || sample_rate <= 0.0) return false;
    memset(out, 0, sizeof(*out));
    out->center_hz = center_hz;
    out->sample_rate = sample_rate;
    out->duration = SYNTH_SECONDS;
    out->out_format = out_format;
    out->n_states = 2;
    out->family = SYNTH_OOK_BURSTS;
    pick_family(cand_name, &out->family, &out->n_states);
    /* The spread-spectrum classification is evidence about the STRUCTURE
     * of the emission and overrides the generic modulation guess when it
     * is confident (>= 60 %): a DSSS/FHSS/CSS verdict changes the signal
     * model far more than the modulation label does. */
    if (sc != NULL && spread_n > 0 && sc[0].score >= 60) {
        if (strcmp(sc[0].name, "DSSS") == 0) {
            out->family = SYNTH_DSSS;
        } else if (strcmp(sc[0].name, "FHSS") == 0) {
            out->family = SYNTH_FHSS;
        } else if (strcmp(sc[0].name, "CSS") == 0) {
            out->family = SYNTH_CHIRP;
        } else if (strcmp(sc[0].name, "FH-DSSS hybrid") == 0) {
            out->family = SYNTH_FHSS;
        }
    }
    if (chip_rate_hz > 0.0 && out->family == SYNTH_DSSS) {
        out->symbol_rate = chip_rate_hz;
    } else if (chip_rate_hz > 0.0 && out->family == SYNTH_FHSS) {
        out->symbol_rate = chip_rate_hz;
    }

    out->bw_hz = m != NULL && m->occupied_bw_hz > 0.0
                     ? m->occupied_bw_hz
                     : sample_rate * 0.25;
    if (out->bw_hz > sample_rate) out->bw_hz = sample_rate * 0.8;
    out->snr_db = m != NULL ? m->snr_db : 20.0;
    if (out->snr_db < 5.0) out->snr_db = 5.0;
    if (out->snr_db > 60.0) out->snr_db = 60.0;

    /* the best-matching database system contributes its canonical
     * parameters: the modulation family when the measured consensus is
     * weak, and the published channel bandwidth / symbol rate / pulsed
     * structure. A decisive match (valid, high confidence) makes the
     * simulation follow the system rather than the raw measurement. */
    {
        const RankedMatch *best = NULL;

        for (size_t i = 0; i < n_matches && matches != NULL; i++) {
            if (best == NULL || matches[i].total > best->total) {
                best = &matches[i];
            }
        }
        if (best != NULL && best->signal != NULL) {
            const SignalSpec *sp = signal_spec_find(best->signal->name);
            bool decisive = best->valid && best->total >= 90;
            bool weak_cand = cand == NULL || cand->score < 60;

            snprintf(out->system, sizeof(out->system), "%s",
                     best->signal->name);
            if (sp != NULL) {
                out->canonical_bw_hz = sp->channel_bw;
                out->canonical_symbol_rate = sp->symbol_rate;
                out->system_pulsed = sp->pulsed != 0;
                if (decisive && sp->channel_bw > 0.0 &&
                    out->bw_hz > 0.0) {
                    double ratio = out->bw_hz / sp->channel_bw;

                    if (ratio > 1.0) ratio = 1.0 / ratio;
                    if (ratio > 0.4) {
                        out->bw_hz = sp->channel_bw;
                    }
                }
                if (sp->symbol_rate > 0.0 && sp->pulsed == 0 &&
                    (out->symbol_rate <= 0.0 || decisive ||
                     (m != NULL && m->sym_rate_conf < 0.3))) {
                    out->symbol_rate = sp->symbol_rate;
                }
            }
            if (weak_cand) {
                SynthFamily f;
                int states = out->n_states;

                if (synth_family_from_db((int)best->signal->family, &f,
                                         &states)) {
                    out->family = f;
                    out->n_states = states;
                }
            }
        }
    }

    /* burst structure (pulse analysis) */
    if (pulse != NULL && pulse->num_pulses >= 4) {
        out->burst_rate_hz = pulse->prf_hz > 0.0 ? pulse->prf_hz
                                                 : pulse->f0_hz;
        out->duty = pulse->duty_cycle > 0.0 ? pulse->duty_cycle : 0.3;
        if (out->duty > 0.95) out->duty = 0.5;
    } else if (m != NULL && m->duty_cycle > 0.0 && m->duty_cycle < 0.95) {
        out->duty = m->duty_cycle;
    } else if (out->system_pulsed) {
        out->duty = 0.3;     /* the identified system is bursty */
    } else if (out->system[0] != '\0') {
        out->duty = 1.0;     /* the identified system is continuous */
    } else {
        out->duty = 0.35;
    }
    if (out->burst_rate_hz <= 0.0) {
        out->burst_rate_hz = a != NULL && a->prf > 0.0 ? a->prf : 200.0;
    }
    /* symbol rate: a validated chip rate or the system's published rate
     * wins; otherwise a fraction of the occupied bandwidth by family */
    if (out->symbol_rate <= 0.0) {
        out->symbol_rate = out->bw_hz * (out->family == SYNTH_OFDM ? 1.0
                                                                : 0.5);
    }
    /* intra-burst sub-pulse timing */
    if (ib != NULL && ib->valid) {
        out->intra_width_s = ib->width_mean_s;
        out->intra_period_s = ib->period_mean_s > 0.0 ? ib->period_mean_s
                                                      : ib->width_mean_s * 2.0;
        if (out->intra_width_s <= 0.0) out->intra_width_s = 10e-6;
        if (out->intra_period_s <= out->intra_width_s) {
            out->intra_period_s = out->intra_width_s * 2.0;
        }
        if (ib->burst_mod[0] != '\0') {
            snprintf(out->description + strlen(out->description),
                     sizeof(out->description) - strlen(out->description),
                     "; intra-burst %s", ib->burst_mod);
        }
    } else {
        out->intra_width_s = 10e-6;
        out->intra_period_s = 40e-6;
    }
    if (out->system[0] != '\0') {
        snprintf(out->description, sizeof(out->description),
                 "%s (database match)%s%s at %.3f MHz, BW %.1f kHz, "
                 "burst %.1f Hz, duty %.0f%%",
                 out->system,
                 cand != NULL ? " / " : "",
                 cand != NULL ? cand->name : "", center_hz / 1e6,
                 out->bw_hz / 1e3, out->burst_rate_hz, out->duty * 100.0);
    } else {
        snprintf(out->description, sizeof(out->description),
                 "%s at %.3f MHz, BW %.1f kHz, burst %.1f Hz, duty %.0f%%",
                 cand != NULL ? cand->name : "pulsed emission",
                 center_hz / 1e6, out->bw_hz / 1e3, out->burst_rate_hz,
                 out->duty * 100.0);
    }
    return true;
}

/* --------------- noise profile --------------------------------------- */

void synth_noise_clear(SynthNoise *n)
{
    if (n == NULL) return;
    free(n->psd);
    memset(n, 0, sizeof(*n));
}

/*
 * Estimate the noise PSD from signal-free data of the visible waterfall
 * window: Welch-average the spectra of blocks whose power stays close to
 * the window's median (blocks that contain signal are excluded), so the
 * profile is the actual receiver noise rather than the analysed signal.
 */
bool synth_noise_profile(const IqSource *src, const Region *visible,
                         const Region *selection, SynthNoise *out)
{
    enum { NF = SYNTH_PSD_N };
    Region v = *visible;
    size_t n_blocks = 0;
    double *acc = NULL;
    double *blk_psd = NULL;
    float *seg = NULL;
    size_t s0, n_total;
    double fs;
    bool ok = false;

    (void)selection;
    if (src == NULL || visible == NULL || out == NULL) return false;
    synth_noise_clear(out);
    fs = src->sample_rate;
    s0 = (size_t)(v.time_min * fs);
    n_total = (size_t)((v.time_max - v.time_min) * fs);
    if (n_total < NF * 2) return false;
    acc = calloc(NF, sizeof(double));
    blk_psd = malloc(NF * sizeof(double));
    seg = malloc(2 * NF * sizeof(float));
    if (acc == NULL || blk_psd == NULL || seg == NULL) goto done;

    /* frequency band that holds the signal: skip blocks whose power inside
     * it is much higher than the median block (those contain the signal) */
    {
        size_t blocks = n_total / NF;
        double *pow_db = calloc(blocks, sizeof(double));
        size_t used = 0;
        double med;
        FftPlan *plan = fft_plan_create(NF);

        if (pow_db == NULL || plan == NULL) {
            if (plan != NULL) fft_plan_destroy(plan);
            free(pow_db);
            goto done;
        }
        for (size_t b = 0; b < blocks; b++) {
            size_t off = s0 + b * NF;
            double sum = 0.0;

            if (off + NF > src->total_samples) break;
            iq_source_read_complex(src, off, seg, NF);
            for (size_t i = 0; i < NF; i++) {
                sum += (double)seg[2 * i] * seg[2 * i] +
                       (double)seg[2 * i + 1] * seg[2 * i + 1];
            }
            pow_db[b] = 10.0 * log10(sum / (double)NF + 1e-20);
            n_blocks = b + 1;
        }
        /* median block power */
        {
            double *tmp = malloc(n_blocks * sizeof(double));

            if (tmp != NULL && n_blocks > 0) {
                memcpy(tmp, pow_db, n_blocks * sizeof(double));
                for (size_t i = 0; i + 1 < n_blocks; i++) {
                    for (size_t j = i + 1; j < n_blocks; j++) {
                        if (tmp[j] < tmp[i]) {
                            double t = tmp[i];

                            tmp[i] = tmp[j];
                            tmp[j] = t;
                        }
                    }
                }
                med = tmp[n_blocks / 2];
                free(tmp);
            } else {
                med = 0.0;
            }
        }
        /* average the spectra of quiet blocks */
        for (size_t b = 0; b < n_blocks; b++) {
            size_t off = s0 + b * NF;

            if (pow_db[b] > med + 6.0) continue; /* contains signal */
            if (off + NF > src->total_samples) break;
            iq_source_read_complex(src, off, seg, NF);
            {
                fftwf_complex *xa = fftwf_malloc(sizeof(fftwf_complex) * NF);

                if (xa == NULL) break;
                fft_plan_transform(plan, seg, NULL, xa);
                for (size_t k = 0; k < NF; k++) {
                    double p = (double)xa[k][0] * xa[k][0] +
                               (double)xa[k][1] * xa[k][1];

                    acc[k] += p;
                }
                fftwf_free(xa);
                used++;
            }
        }
        fft_plan_destroy(plan);
        free(pow_db);
        if (used == 0) goto done;
        {
            double sum = 0.0;

            for (size_t k = 0; k < NF; k++) {
                acc[k] /= (double)used;
                sum += acc[k];
            }
            if (sum <= 0.0) goto done;
            for (size_t k = 0; k < NF; k++) acc[k] /= sum; /* normalise */
            out->psd = acc;
            out->n = NF;
            out->sample_rate = fs;
            out->floor_db = 10.0 * log10(sum / (double)NF + 1e-20);
            acc = NULL;
            ok = true;
        }
    }
done:
    free(acc);
    free(blk_psd);
    free(seg);
    return ok;
}

/* --------------- signal generation ----------------------------------- */

/*
 * Shaped white noise from the profile: generate white noise, weight the
 * spectrum by sqrt(PSD) in the frequency domain, inverse transform. The
 * result has the measured noise colour (its autocorrelation), not just the
 * measured power.
 */
static void coloured_noise(const SynthNoise *np, float *out, size_t n, Rng *r)
{
    size_t nf = 1;
    float *work;

    while (nf < n) nf <<= 1;
    work = calloc(2 * nf, sizeof(float));
    if (work == NULL) {
        for (size_t i = 0; i < 2 * n; i++) out[i] = (float)(rnd(r) - 0.5);
        return;
    }
    if (np == NULL || np->psd == NULL || np->n < 8) {
        for (size_t i = 0; i < 2 * n; i++) out[i] = (float)(rnd(r) - 0.5);
        free(work);
        return;
    }
    {
        FftPlan *plan = fft_plan_create((int)nf);
        fftwf_complex *xa = NULL, *xb = NULL;

        if (plan == NULL) {
            for (size_t i = 0; i < 2 * n; i++) out[i] = (float)(rnd(r) - 0.5);
            free(work);
            return;
        }
        xa = fftwf_malloc(sizeof(fftwf_complex) * nf);
        xb = fftwf_malloc(sizeof(fftwf_complex) * nf);
        if (xa == NULL || xb == NULL) {
            if (xa != NULL) fftwf_free(xa);
            if (xb != NULL) fftwf_free(xb);
            fft_plan_destroy(plan);
            for (size_t i = 0; i < 2 * n; i++) out[i] = (float)(rnd(r) - 0.5);
            free(work);
            return;
        }
        for (size_t i = 0; i < nf; i++) {
            work[2 * i] = (float)(rnd(r) - 0.5) * 2.0f;
            work[2 * i + 1] = (float)(rnd(r) - 0.5) * 2.0f;
        }
        fft_plan_transform(plan, work, NULL, xa);
        for (size_t k = 0; k < nf; k++) {
            size_t pk = (size_t)((double)k / (double)nf * (double)np->n);
            double h = sqrt(np->psd[pk < (size_t)np->n ? pk
                                                       : (size_t)np->n - 1] *
                            (double)np->n);

            xb[k][0] = (float)((double)xa[k][0] * h);
            xb[k][1] = (float)((double)xa[k][1] * h);
        }
        /* inverse transform via the conjugate trick */
        for (size_t k = 0; k < nf; k++) xb[k][1] = -xb[k][1];
        fft_plan_transform(plan, (float *)xb, NULL, xa);
        for (size_t i = 0; i < n; i++) {
            out[2 * i] = (float)((double)xa[i][0] / (double)nf);
            out[2 * i + 1] = (float)(-(double)xa[i][1] / (double)nf);
        }
        fftwf_free(xa);
        fftwf_free(xb);
        fft_plan_destroy(plan);
    }
    free(work);
}

/* band-limited pulse shape */
static void add_pulse(float *iq, size_t n, double fs, double t0, double width,
                      double freq_hz, double amp, Rng *r)
{
    size_t a = (size_t)(t0 * fs);
    size_t len = (size_t)(width * fs);
    double edge = 0.15;

    if (a >= n) return;
    if (a + len > n) len = n - a;
    for (size_t k = 0; k < len; k++) {
        double x = (double)k / (double)(len > 0 ? len : 1);
        double env = 1.0;

        if (x < edge) env = 0.5 - 0.5 * cos(M_PI * x / edge);
        else if (x > 1.0 - edge) env = 0.5 - 0.5 * cos(M_PI * (1.0 - x) / edge);
        {
            double ph = 2.0 * M_PI * freq_hz * (double)(a + k) / fs;

            iq[2 * (a + k)] += (float)(amp * env * cos(ph));
            iq[2 * (a + k) + 1] += (float)(amp * env * sin(ph));
        }
    }
    (void)r;
}

bool synth_generate(const SynthSpec *spec, const SynthNoise *noise,
                    float **iq_out, size_t *n_out)
{
    size_t n, total_periods;
    float *iq;
    Rng r;
    double fs = spec->sample_rate;
    double target_burst_rate = spec->burst_rate_hz > 0.0
                                   ? spec->burst_rate_hz
                                   : 200.0;

    if (iq_out == NULL || n_out == NULL || fs <= 0.0) return false;
    n = (size_t)(spec->duration * fs);
    iq = calloc(2 * n, sizeof(float));
    if (iq == NULL) return false;
    r.s = 0xC0FFEE123456789ULL;

    total_periods = (size_t)(spec->duration * target_burst_rate);

    switch (spec->family) {
    case SYNTH_CW:
        for (size_t i = 0; i < n; i++) {
            double ph = 2.0 * M_PI * 0.0;

            iq[2 * i] = (float)(0.5 * cos(ph));
            iq[2 * i + 1] = (float)(0.5 * sin(ph));
        }
        break;
    case SYNTH_PSK: {
        double sps = fs / (spec->symbol_rate > 0.0 ? spec->symbol_rate
                                                   : fs / 8.0);
        size_t sym = (size_t)(sps < 1.0 ? 1.0 : sps);
        int states = spec->n_states >= 2 ? spec->n_states : 2;
        double ph0 = 0.0;

        for (size_t i = 0; i < n; i++) {
            if (i % sym == 0) {
                ph0 = 2.0 * M_PI * (double)(rnd(&r) * (double)states) /
                      (double)states;
            }
            {
                double ph = 2.0 * M_PI * 0.0 * (double)i / fs + ph0;

                iq[2 * i] = (float)(0.6 * cos(ph));
                iq[2 * i + 1] = (float)(0.6 * sin(ph));
            }
        }
        break;
    }
    case SYNTH_FSK: {
        double baud = spec->symbol_rate > 0.0 ? spec->symbol_rate : 1e5;
        double dev = spec->bw_hz * 0.25;
        double sps = fs / baud;
        size_t sym = (size_t)(sps < 1.0 ? 1.0 : sps);
        double f = 0.0, ph = 0.0;

        for (size_t i = 0; i < n; i++) {
            if (i % sym == 0) {
                double s = rnd(&r) > 0.5 ? 1.0 : -1.0;

                f = s * dev;
            }
            ph += 2.0 * M_PI * f / fs;
            iq[2 * i] = (float)(0.6 * cos(ph));
            iq[2 * i + 1] = (float)(0.6 * sin(ph));
        }
        break;
    }
    case SYNTH_OFDM: {
        size_t nfft = 1024;
        size_t cp = nfft / 8;
        size_t nused = (size_t)(nfft * spec->bw_hz / fs);

        if (nused > nfft) nused = nfft;
        if (nused < 8) nused = 8;
        {
            size_t pos = 0;

            while (pos + nfft + cp < n) {
                for (size_t i = 0; i < nfft + cp; i++) {
                    double acc = 0.0;

                    for (size_t k = 1; k <= nused / 2; k++) {
                        double ph = 2.0 * M_PI * (double)k * (double)i /
                                    (double)nfft;
                        double c = rnd(&r) - 0.5;

                        acc += c * cos(ph + M_PI * (rnd(&r) - 0.5) * 0.0);
                    }
                    iq[2 * (pos + i)] = (float)(acc * 0.6);
                }
                pos += nfft + cp;
            }
        }
        break;
    }
    case SYNTH_CHIRP: {
        double tau = 1.0 / (target_burst_rate > 0.0 ? target_burst_rate
                                                    : 1000.0);

        for (size_t p = 0; p < total_periods; p++) {
            double t0 = (double)p / target_burst_rate;
            size_t a = (size_t)(t0 * fs);
            size_t len = (size_t)(tau * fs);

            if (a + len > n) break;
            for (size_t k = 0; k < len; k++) {
                double t = (double)k / fs;
                double ph = M_PI * spec->bw_hz / tau * t * t;

                iq[2 * (a + k)] += (float)(0.6 * cos(ph));
                iq[2 * (a + k) + 1] += (float)(0.6 * sin(ph));
            }
        }
        break;
    }
    case SYNTH_DSSS: {
        /* direct sequence: PN chips at the measured chip rate, BPSK,
         * continuous (the common DSSS case) */
        double chip = spec->symbol_rate > 0.0 ? spec->symbol_rate
                                              : spec->bw_hz * 0.5;
        double sps = fs / chip;
        size_t nsym = (size_t)(sps < 1.0 ? 1.0 : sps);
        unsigned int lfsr = 0xACE1u;
        double ph = 0.0;

        for (size_t i = 0; i < n; i++) {
            if (i % nsym == 0) {
                unsigned int bit = ((lfsr >> 0) ^ (lfsr >> 2) ^
                                    (lfsr >> 3) ^ (lfsr >> 5)) & 1u;

                lfsr = (lfsr >> 1) | (bit << 15);
                ph = (lfsr & 1u) ? 0.0 : M_PI;
            }
            iq[2 * i] = (float)(0.6 * cos(ph));
            iq[2 * i + 1] = (float)(0.6 * sin(ph));
        }
        break;
    }
    case SYNTH_FHSS: {
        /* frequency hopping: dwell on a random channel inside the
         * measured bandwidth at the measured hop rate */
        double hop_rate = target_burst_rate > 0.0 ? target_burst_rate : 200.0;
        double dwell = 1.0 / hop_rate;
        double half = spec->bw_hz * 0.5;
        double ph = 0.0, f = 0.0;

        for (size_t i = 0; i < n; i++) {
            double t = (double)i / fs;

            if (fmod(t, dwell) < 1.0 / fs) {
                f = (rnd(&r) * 2.0 - 1.0) * half;
            }
            ph += 2.0 * M_PI * f / fs;
            iq[2 * i] = (float)(0.6 * cos(ph));
            iq[2 * i + 1] = (float)(0.6 * sin(ph));
        }
        break;
    }
    case SYNTH_OOK_BURSTS:
    default: {
        /* bursts at the measured rate and duty; inside each burst the
         * measured sub-pulse width/period train (the intra-burst
         * structure), or a solid burst when none was resolved */
        double burst_len = spec->duration * spec->duty / total_periods;
        size_t npulses;

        if (burst_len <= 0.0) burst_len = 1.0 / target_burst_rate * 0.3;
        npulses = spec->intra_period_s > 0.0
                      ? (size_t)(burst_len / spec->intra_period_s)
                      : 1;
        if (npulses < 1) npulses = 1;
        for (size_t p = 0; p < total_periods; p++) {
            double t0 = (double)p / target_burst_rate;
            double t = t0;

            if (npulses == 1) {
                add_pulse(iq, n, fs, t, burst_len, 0.0, 0.6, &r);
            } else {
                for (size_t q = 0; q < npulses; q++) {
                    double w = spec->intra_width_s > 0.0
                                   ? spec->intra_width_s
                                   : burst_len / (double)npulses;

                    add_pulse(iq, n, fs, t, w, 0.0, 0.6, &r);
                    t += spec->intra_period_s;
                    if (t >= t0 + burst_len) break;
                }
            }
        }
        break;
    }
    }

    /* scale the signal to the requested SNR against the noise, then add
     * the coloured noise */
    {
        double nsig = 0.0;
        double ntot;

        for (size_t i = 0; i < n; i++) {
            nsig += (double)iq[2 * i] * iq[2 * i] +
                    (double)iq[2 * i + 1] * iq[2 * i + 1];
        }
        ntot = (double)n;
        if (nsig <= 0.0) nsig = 1e-20;
        {
            double target_sig_pow = 1.0;
            double noise_pow = target_sig_pow /
                               pow(10.0, spec->snr_db / 10.0);
            double scale = sqrt(target_sig_pow * ntot / nsig);
            float *noise_buf = malloc(2 * n * sizeof(float));

            if (noise_buf == NULL) {
                free(iq);
                return false;
            }
            coloured_noise(noise, noise_buf, n, &r);
            {
                double npow = 0.0;

                for (size_t i = 0; i < n; i++) {
                    npow += (double)noise_buf[2 * i] * noise_buf[2 * i] +
                            (double)noise_buf[2 * i + 1] *
                                noise_buf[2 * i + 1];
                }
                if (npow <= 0.0) npow = 1e-20;
                {
                    double nscale = sqrt(noise_pow * ntot / npow);

                    for (size_t i = 0; i < n; i++) {
                        iq[2 * i] = (float)(iq[2 * i] * scale +
                                            noise_buf[2 * i] * nscale);
                        iq[2 * i + 1] = (float)(iq[2 * i + 1] * scale +
                                                noise_buf[2 * i + 1] * nscale);
                    }
                }
            }
            free(noise_buf);
        }
    }
    *iq_out = iq;
    *n_out = n;
    return true;
}

bool synth_write(const char *path, const float *iq, size_t n, IqFormat format)
{
    FILE *f;

    if (path == NULL || iq == NULL) return false;
    f = fopen(path, "wb");
    if (f == NULL) return false;
    for (size_t i = 0; i < n; i++) {
        float re = iq[2 * i];
        float im = iq[2 * i + 1];

        switch (format) {
        case IQ_FORMAT_FLOAT32:
            if (fwrite(&re, sizeof(float), 1, f) != 1 ||
                fwrite(&im, sizeof(float), 1, f) != 1) {
                fclose(f);
                return false;
            }
            break;
        case IQ_FORMAT_INT16: {
            int16_t v[2];

            v[0] = (int16_t)(fmax(-32767.0, fmin(32767.0, re * 32767.0f)));
            v[1] = (int16_t)(fmax(-32767.0, fmin(32767.0, im * 32767.0f)));
            if (fwrite(v, sizeof(int16_t), 2, f) != 2) {
                fclose(f);
                return false;
            }
            break;
        }
        case IQ_FORMAT_INT8: {
            int8_t v[2];

            v[0] = (int8_t)fmax(-127.0, fmin(127.0, re * 127.0f));
            v[1] = (int8_t)fmax(-127.0, fmin(127.0, im * 127.0f));
            if (fwrite(v, 1, 2, f) != 2) {
                fclose(f);
                return false;
            }
            break;
        }
        case IQ_FORMAT_UINT8: {
            uint8_t v[2];

            v[0] = (uint8_t)fmax(0.0, fmin(255.0, re * 127.0f + 127.5f));
            v[1] = (uint8_t)fmax(0.0, fmin(255.0, im * 127.0f + 127.5f));
            if (fwrite(v, 1, 2, f) != 2) {
                fclose(f);
                return false;
            }
            break;
        }
        }
    }
    fclose(f);
    return true;
}
