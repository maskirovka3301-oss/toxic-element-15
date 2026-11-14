/*
 * director.c — the evidence director (see director.h).
 *
 * The ranking rules, in priority order (each one appends a rationale line):
 *   1. analyses that were skipped or failed contribute nothing; the
 *      rationale names them and their DAG reason so the reader knows the
 *      claim was discounted, not merely absent;
 *   2. validated cyclostationary / cyclic-cumulant evidence outranks
 *      classifier heuristics (it is measurement, not a prior);
 *   3. validated intra-burst structure outranks the pulse/classic
 *      heuristics (it is measured at the sample level, at full rate);
 *   4. the pulse train ranks above the generic classifier only when it
 *      passed the credibility gate;
 *   5. spread-spectrum families rank by their own confidence and are
 *      appended to the characterisation only when non-null and >= 40%.
 */
#include "director.h"

#include "aconfig.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* envelope access helpers                                             */
/* ------------------------------------------------------------------ */

static double num(const DirectorInput *in, const char *id, const char *key,
                  double dflt)
{
    const AnalysisResult *r = ar_find(in->analyses, id);
    double v;

    if (r == NULL || r->status != AR_EXECUTED) return dflt;
    if (!ar_get_num(r, key, &v)) return dflt;
    return v;
}

static const char *text(const DirectorInput *in, const char *id,
                        const char *key, const char *dflt)
{
    const AnalysisResult *r = ar_find(in->analyses, id);
    const char *s;

    if (r == NULL || r->status != AR_EXECUTED) return dflt;
    if (!ar_get_text(r, key, &s) || s == NULL) return dflt;
    return s;
}

static bool ran(const DirectorInput *in, const char *id)
{
    const AnalysisResult *r = ar_find(in->analyses, id);

    return r != NULL && r->status == AR_EXECUTED;
}

static void rationale(Director *d, const char *fmt, ...)
{
    va_list ap;

    if (d->n_rationale >= DIR_MAX_RATIONALE) return;
    va_start(ap, fmt);
    vsnprintf(d->rationale[d->n_rationale], sizeof(d->rationale[0]), fmt, ap);
    va_end(ap);
    d->n_rationale++;
}

/* ------------------------------------------------------------------ */
/* modulation token (moved from the report builder)                    */
/* ------------------------------------------------------------------ */

void director_mod_token(const char *s, char *token, size_t cap)
{
    token[0] = '\0';
    if (s == NULL || s[0] == '\0') return;
    if (strstr(s, "BPSK") != NULL) snprintf(token, cap, "BPSK");
    else if (strstr(s, "QPSK") != NULL) snprintf(token, cap, "QPSK");
    else if (strstr(s, "MSK") != NULL || strstr(s, "8PSK") != NULL) {
        snprintf(token, cap, "8PSK/MSK");
    } else if (strstr(s, "QAM") != NULL) snprintf(token, cap, "QAM");
    else if (strstr(s, "OFDM") != NULL) snprintf(token, cap, "OFDM");
    else if (strstr(s, "FSK") != NULL) snprintf(token, cap, "FSK");
    else if (strstr(s, "OOK") != NULL) snprintf(token, cap, "OOK");
    else if (strstr(s, "PSK") != NULL) snprintf(token, cap, "PSK");
    else if (strstr(s, "CHIRP") != NULL) snprintf(token, cap, "CHIRP");
    else if (strstr(s, "CW") != NULL) snprintf(token, cap, "CW");
    else if (strstr(s, "AM") != NULL) snprintf(token, cap, "AM");
    else if (strstr(s, "FM") != NULL) snprintf(token, cap, "FM");
    else if (strstr(s, "NOISE") != NULL) snprintf(token, cap, "NOISE");
    else if (strstr(s, "PWM") != NULL) snprintf(token, cap, "PWM");
    else if (strstr(s, "PPM") != NULL) snprintf(token, cap, "PPM");
    else if (strstr(s, "constant") != NULL) snprintf(token, cap, "constant-PRF");
    else if (strstr(s, "burst") != NULL) snprintf(token, cap, "bursty");
    else {
        size_t i = 0;

        while (i + 1 < cap && s[i] != '\0' && s[i] != ' ' && s[i] != '(' &&
               s[i] != '/' && s[i] != ',' && s[i] != ';' && s[i] != ':') {
            token[i] = s[i];
            i++;
        }
        token[i] = '\0';
    }
}

/* ------------------------------------------------------------------ */
/* claim ranking                                                       */
/* ------------------------------------------------------------------ */

static void claim_add(Director *d, const char *name, double weight,
                      const char *source, const char *fmt, ...)
{
    DirClaim *c;
    va_list ap;

    if (d->n_cand >= DIR_MAX_CAND) return;
    c = &d->cand[d->n_cand++];
    memset(c, 0, sizeof(*c));
    snprintf(c->name, sizeof(c->name), "%s", name);
    c->weight = weight;
    c->source = source;
    va_start(ap, fmt);
    vsnprintf(c->evidence, sizeof(c->evidence), fmt, ap);
    va_end(ap);
}


/* ------------------------------------------------------------------ */
/* build                                                               */
/* ------------------------------------------------------------------ */

void director_build(Director *d, const DirectorInput *in)
{
    char vd[24] = "", vc[24] = "", vh[24] = "", vp[24] = "";
    char majority[24] = "";
    int maj = 0;
    double fam_cf = 0.0, ssca_cf = 0.0, det_cf = 0.0;
    const char *cf_verdict = "n/a";
    double bchip, cchip;
    const char *topmod;
    char best[512];

    if (d == NULL || in == NULL) return;
    memset(d, 0, sizeof(*d));

    bchip = num(in, "fluct", "chip_rate_hz", 0.0);
    cchip = num(in, "cpar", "chip_rate_hz", 0.0);
    d->bchip = bchip;
    d->cchip = cchip;

    /* ---- modulation votes (same four-way agreement as before) ---- */
    director_mod_token(text(in, "analysis", "modulation", ""), vd, sizeof(vd));
    director_mod_token(text(in, "ccc", "verdict", ""), vc, sizeof(vc));
    director_mod_token(text(in, "hos", "hint", ""), vh, sizeof(vh));
    director_mod_token(text(in, "pulse", "modulation", ""), vp, sizeof(vp));
    {
        const char *toks[4] = { vd, vc, vh, vp };
        int counts[4] = { 0, 0, 0, 0 };

        for (int i = 0; i < 4; i++) {
            if (toks[i][0] == '\0') continue;
            for (int j = 0; j < 4; j++) {
                if (toks[j][0] != '\0' && strcmp(toks[i], toks[j]) == 0) {
                    counts[i]++;
                }
            }
            if (counts[i] > maj) {
                maj = counts[i];
                snprintf(majority, sizeof(majority), "%s", toks[i]);
            }
        }
        snprintf(d->vote_line, sizeof(d->vote_line),
                 "demod %s | cumulants %s | HOS %s | pulses %s",
                 vd[0] ? vd : "n/a", vc[0] ? vc : "n/a",
                 vh[0] ? vh : "n/a", vp[0] ? vp : "n/a");
        snprintf(d->majority, sizeof(d->majority), "%s", majority);
        d->majority_count = maj;
    }

    /* ---- cycle-frequency cross-check ---- */
    fam_cf = num(in, "cyclic", "strongest_alpha_hz", 0.0);
    ssca_cf = num(in, "ssca", "strongest_cf_hz", 0.0);
    det_cf = num(in, "cycledet", "best_cf_hz", 0.0);
    if (num(in, "cycledet", "detected", 0.0) == 0.0) det_cf = 0.0;
    {
        double vals[3];
        int nv = 0;

        if (fam_cf > 0.0) vals[nv++] = fam_cf;
        if (ssca_cf > 0.0) vals[nv++] = ssca_cf;
        if (det_cf > 0.0) vals[nv++] = det_cf;
        if (nv >= 2) {
            double lo = vals[0], hi = vals[0];

            for (int i = 1; i < nv; i++) {
                if (vals[i] < lo) lo = vals[i];
                if (vals[i] > hi) hi = vals[i];
            }
            cf_verdict = (hi <= lo * 1.1) ? "agree within 10%" : "disagree";
        }
    }
    snprintf(d->cf_summary, sizeof(d->cf_summary), "%s", cf_verdict);

    /* ---- chip-rate / DSSS consistency ---- */
    if (bchip > 0.0 && cchip > 0.0) {
        double ratio = bchip > cchip ? bchip / cchip : cchip / bchip;

        snprintf(d->dsss_label, sizeof(d->dsss_label), "%s",
                 ratio < 1.5 ? "supported" : "not supported");
    } else {
        snprintf(d->dsss_label, sizeof(d->dsss_label),
                 "not supported (only one chip-rate detector produced a "
                 "result)");
    }

    /* ---- detector presence ---- */
    d->presence = (num(in, "cycledet", "detected", 0.0) > 0.0 ? 1 : 0) +
                  (num(in, "hos", "non_gaussian", 0.0) > 0.0 ? 1 : 0) +
                  (num(in, "pulse", "num_pulses", 0.0) >= 5 ? 1 : 0) +
                  ((bchip > 0.0 || cchip > 0.0) ? 1 : 0);

    /* ---- strongest modulation candidate (not the crude vote) ---- */
    {
        const char *nm = in->n_mods > 0 ? in->mods[0].name
                                        : (majority[0] ? majority : "unclassified");
        char topbuf[64];
        size_t l;

        snprintf(topbuf, sizeof(topbuf), "%s", nm);
        l = strlen(topbuf);
        if (l > 9 && strcmp(topbuf + l - 9, " emission") == 0) {
            topbuf[l - 9] = '\0';
        }
        snprintf(d->top_mod, sizeof(d->top_mod), "%s", topbuf);
        topmod = d->top_mod;
    }

    /* ---- best-supported characterisation ---- */
    {
        double pulses = num(in, "pulse", "num_pulses", 0.0);
        double credible = num(in, "pulse", "credible", 0.0);
        double prf = num(in, "pulse", "prf_hz", 0.0);
        double f0 = num(in, "pulse", "f0_hz", 0.0);
        double tau = num(in, "pulse", "pulse_width_s", 0.0);
        double duty = num(in, "pulse", "duty_cycle", 0.0);
        double classic_period = num(in, "classic", "period_hz", 0.0);

        if (ran(in, "pulse") && pulses >= 2.0 && credible > 0.0) {
            snprintf(best, sizeof(best),
                     "bursty emission with a %s signature: %.1f ms bursts at "
                     "%.1f Hz (duty %.0f%%)", topmod, tau * 1e3, prf,
                     duty * 100.0);
            rationale(d, "A2 credible pulse train ranks first (width %.3f ms, "
                      "PRF %.1f Hz)", tau * 1e3, prf);
        } else if (ran(in, "pulse") && pulses >= 2.0 && f0 > 0.0) {
            snprintf(best, sizeof(best),
                     "bursty emission with a %s signature at %.2f Hz "
                     "(envelope line-spectrum fundamental, %.3f ms period); "
                     "threshold-based pulse-train segmentation rejected",
                     topmod, f0, 1e3 / f0);
            rationale(d, "A2 pulse train rejected by the credibility gate: "
                      "the %.2f Hz envelope fundamental is used instead", f0);
        } else if (ran(in, "classic") && classic_period > 0.0) {
            snprintf(best, sizeof(best),
                     "emission with a %s signature and a %.3f Hz periodic "
                     "structure", topmod, classic_period);
            rationale(d, "A1 classic cepstrum/AMDF period %.3f Hz is the only "
                      "periodic evidence", classic_period);
        } else {
            snprintf(best, sizeof(best), "emission with a %s signature",
                     topmod);
            rationale(d, "no time-structure analysis passed; only the "
                      "classifier evidence remains");
        }
        int present = aconfig_get_int("ssclass", "present_score", 40);
        bool append = aconfig_get_bool("ssclass", "append_to_verdict", true);

        if (present < 0) present = 0;
        if (append && in->n_spread > 0 && in->spread[0].score >= present &&
            strcmp(in->spread[0].name, "Not spread spectrum") != 0) {
            size_t l = strlen(best);

            snprintf(best + l, sizeof(best) - l,
                     "; spread-spectrum structure: %s (%d %%)",
                     in->spread[0].name, in->spread[0].score);
            rationale(d, "spread-spectrum classification '%s' at %d%% is "
                      "confident enough to append", in->spread[0].name,
                      in->spread[0].score);
        }
        snprintf(d->characterisation, sizeof(d->characterisation), "%s",
                 best);
    }

    /* ---- final verdict (characterisation + intra-burst clause) ---- */
    {
        double uniform = num(in, "intraburst", "uniform", 0.0);
        const char *ib_verdict = text(in, "intraburst", "verdict", "");
        char ib[160];

        if (ran(in, "intraburst")) {
            if (uniform > 0.0) {
                snprintf(ib, sizeof(ib),
                         "uniform train (clock/preamble-like)");
            } else if (strstr(ib_verdict, "varying") != NULL) {
                snprintf(ib, sizeof(ib), "width-encoded data");
            } else {
                snprintf(ib, sizeof(ib), "%s", ib_verdict);
            }
            snprintf(d->verdict, sizeof(d->verdict),
                     "%s; intra-burst: %s", d->characterisation, ib);
            rationale(d, "A9 intra-burst result ('%s') appended to the "
                      "verdict", ib_verdict);
        } else {
            snprintf(d->verdict, sizeof(d->verdict), "%s",
                     d->characterisation);
        }
    }

    /* ---- ranked claims: the hypotheses the Director considered ---- */
    {
        double w;

        if (in->n_mods > 0) {
            claim_add(d, in->mods[0].name, in->mods[0].score, "modconf",
                      "top modulation consensus candidate (score %d): %s",
                      in->mods[0].score, in->mods[0].why);
        }
        if (in->n_spread > 0) {
            claim_add(d, in->spread[0].name, in->spread[0].score, "ssclass",
                      "spread-spectrum classification: %s",
                      in->spread[0].evidence);
        }
        w = num(in, "pulse", "credible", 0.0) > 0.0 ? 75.0
                                                    : 25.0;
        if (num(in, "pulse", "num_pulses", 0.0) >= 2) {
            claim_add(d, "pulse-train emission", w, "pulse",
                      "A2 %s", num(in, "pulse", "credible", 0.0) > 0.0
                                   ? "pulse train passed the credibility gate"
                                   : "pulse train rejected (threshold "
                                     "crossings, not a train)");
        }
        if (ran(in, "intraburst")) {
            double fpr = num(in, "intraburst", "surrogate_fpr", 1.0);

            claim_add(d, "intra-burst structure",
                      60.0 + (num(in, "intraburst", "delta_bic", 0.0) > 20.0
                                  ? 10.0 : 0.0) - fpr * 20.0,
                      "intraburst", "A9 %s",
                      text(in, "intraburst", "verdict", "no verdict"));
        }
        if (num(in, "ccc", "valid", 0.0) > 0.0) {
            claim_add(d, "cyclic-cumulant modulation", 70.0, "ccc",
                      "A8 %s", text(in, "ccc", "verdict", ""));
        }
    }
    {
        /* sort the claims by descending weight */
        for (size_t i = 0; i + 1 < d->n_cand; i++) {
            for (size_t j = i + 1; j < d->n_cand; j++) {
                if (d->cand[j].weight > d->cand[i].weight) {
                    DirClaim t = d->cand[i];

                    d->cand[i] = d->cand[j];
                    d->cand[j] = t;
                }
            }
        }
        if (d->n_cand > 0) {
            rationale(d, "ranking: %zu hypotheses ordered by weight; '%s' "
                      "leads at %.0f%%", d->n_cand, d->cand[0].name,
                      d->cand[0].weight);
        }
    }

    /* ---- provenance: every DAG node and why it contributed ---- */
    if (in->dag != NULL) {
        for (int n = 0; n < DAG_NODES && d->n_provenance < DIR_MAX_PROV; n++) {
            DagState st = in->dag->state[n];

            if (st == DAG_PENDING) continue;
            snprintf(d->provenance[d->n_provenance],
                     sizeof(d->provenance[0]),
                     "%s: %s (%.2f s)%s%s", dag_node_name(n),
                     dag_state_name(st), in->dag->elapsed[n],
                     st == DAG_SKIPPED || st == DAG_FAILED ? " - " : "",
                     in->dag->skip[n] != NULL ? in->dag->skip[n] : "");
            d->n_provenance++;
            if (st == DAG_SKIPPED && in->dag->skip[n] != NULL) {
                rationale(d, "%s discounted: %s", dag_node_name(n),
                          in->dag->skip[n]);
            }
        }
    }
}
