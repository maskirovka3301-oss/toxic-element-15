/*
 * ssclass.c — spread-spectrum presence and type classification.
 */
#include "ssclass.h"

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    char name[48];
    int  score;
    char evidence[192];
} Cand;

static void cand_add(Cand *c, size_t *n, const char *name, int score,
                     const char *fmt, ...)
{
    va_list ap;
    int idx;

    for (idx = 0; idx < (int)*n; idx++) {
        if (strcmp(c[idx].name, name) == 0) break;
    }
    if (idx == (int)*n) {
        if (*n >= 8) return;
        snprintf(c[idx].name, sizeof(c[idx].name), "%s", name);
        c[idx].score = 0;
        c[idx].evidence[0] = '\0';
        (*n)++;
    }
    c[idx].score += score;
    if (c[idx].score > 100) c[idx].score = 100;
    va_start(ap, fmt);
    {
        size_t l = strlen(c[idx].evidence);
        char piece[160];

        vsnprintf(piece, sizeof(piece), fmt, ap);
        if (l > 0 && l + 2 < sizeof(c[idx].evidence)) {
            snprintf(c[idx].evidence + l, sizeof(c[idx].evidence) - l,
                     "; %s", piece);
        } else if (l == 0) {
            snprintf(c[idx].evidence, sizeof(c[idx].evidence), "%s", piece);
        }
    }
    va_end(ap);
}

size_t spread_classify(const Metrics *m, const Analysis *a,
                       const PulseResult *pulse, const FluctResult *fluct,
                       const CparResult *cpar, const CyclicResult *cyclic,
                       const SscaResult *ssca, SpreadCandidate *out,
                       size_t max_out)
{
    Cand c[8];
    size_t n = 0;
    double bchip = 0.0, cchip = 0.0;
    double flat = (m != NULL) ? m->spectral_flatness : 0.0;
    double duty = (m != NULL) ? m->duty_cycle : 1.0;
    double chip_ratio = 0.0;
    int chip_known = 0;
    int chip_agree = 0;

    if (out == NULL || max_out == 0) return 0;
    memset(c, 0, sizeof(c));

    if (fluct != NULL && fluct->detected && fluct->chip_rate_hz > 0.0) {
        bchip = fluct->chip_rate_hz;
    }
    if (cpar != NULL && cpar->detected && cpar->chip_rate_hz > 0.0) {
        cchip = cpar->chip_rate_hz;
    }
    if (bchip > 0.0 && cchip > 0.0) {
        chip_known = 1;
        chip_ratio = (bchip > cchip) ? bchip / cchip : cchip / bchip;
        chip_agree = (chip_ratio < 1.5);
    }

    /* ---------------- DSSS: chip-rate evidence ----------------
     * The score depends on whether the chip rate is actually supported:
     * a single detector's estimate is suggestive (30), two agreeing
     * detectors are strong (75), and two DISAGREEING detectors leave the
     * chip rate unvalidated (20) - an unvalidated rate must not read as
     * a confident DSSS identification. */
    if (chip_known && chip_agree) {
        cand_add(c, &n, "DSSS", 50,
                 "both detectors agree on the chip rate (%.2fx: Burel "
                 "%.1f Hz, CPAR %.1f Hz)", chip_ratio, bchip, cchip);
        cand_add(c, &n, "DSSS", 25, "chip-rate validation");
    } else if (chip_known) {
        cand_add(c, &n, "DSSS", 20,
                 "chip-rate detectors disagree (%.1fx: Burel %.1f Hz vs "
                 "CPAR %.1f Hz) - the chip rate is unvalidated",
                 chip_ratio, bchip, cchip);
    } else if (bchip > 0.0) {
        cand_add(c, &n, "DSSS", 30,
                 "Burel ACF-fluctuation chip rate %.1f Hz", bchip);
    } else if (cchip > 0.0) {
        cand_add(c, &n, "DSSS", 30, "CPAR code/chip rate %.1f Hz", cchip);
    }
    if (flat > 0.6 && (bchip > 0.0 || cchip > 0.0)) {
        cand_add(c, &n, "DSSS", 10, "flat spectrum (%.2f) consistent with "
                 "spread energy", flat);
    }
    if (cyclic != NULL && cyclic->cf_count > 0 && bchip > 0.0) {
        int consistent = 0;

        for (size_t i = 0; i < cyclic->cf_count; i++) {
            double alpha = cyclic->cf_list[i][0];
            double mult = alpha / bchip;

            while (mult > 2.0) mult /= 2.0;
            while (mult < 0.5 && mult > 0.0) mult *= 2.0;
            if (fabs(mult - 1.0) < 0.10) {
                consistent = 1;
                break;
            }
        }
        if (consistent) {
            cand_add(c, &n, "DSSS", 15,
                     "a cycle frequency is consistent with the chip rate");
        }
    }

    /* ---------------- FHSS: hop evidence ----------------
     * When the pulse-train segmentation was rejected the burst statistics
     * are only indicative: the evidence is kept (frequency-agile bursts
     * are visible in the envelope) but weighted down and labelled. */
    {
        int pulse_ok = (pulse != NULL) &&
                       (pulse->credible || pulse->num_pulses >= 3);
        int weight = (pulse != NULL && pulse->credible) ? 100 : 60;

        if (pulse != NULL && pulse->num_pulses >= 3 &&
            pulse->pulse_freq_cv > 0.20) {
        cand_add(c, &n, "FHSS", (50 * weight) / 100,
                 "burst centre frequency changes between bursts (CV %.0f%%)%s",
                 pulse->pulse_freq_cv * 100.0,
                 (pulse->credible) ? "" : " [pulse-train segmentation "
                 "rejected: indicative only]");
        if (pulse->f0_hz > 0.0) {
            cand_add(c, &n, "FHSS", 15, "hop rate ~%.1f Hz", pulse->f0_hz);
        }
        if (duty > 0.0 && duty < 0.5) {
            cand_add(c, &n, "FHSS", (15 * weight) / 100,
                     "low duty (%.0f%%)", duty * 100.0);
        }
        (void)pulse_ok;
    }
    }

    /* ---------------- THSS: irregular burst timing ---------------- */
    if (pulse != NULL && pulse->num_pulses >= 5 && pulse->credible) {
        if (pulse->interval_cv > 0.30) {
            cand_add(c, &n, "THSS", 45,
                     "burst intervals vary (CV %.0f%%) - no stable PRF",
                     (pulse->interval_cv > 2.0 ? 200.0
                                               : pulse->interval_cv * 100.0));
        } else if (pulse->jitter_frac > 0.60) {
            cand_add(c, &n, "THSS", 30,
                     "burst timing is essentially random (jitter %.0f%%)",
                     pulse->jitter_frac * 100.0);
        }
        if (duty > 0.0 && duty < 0.5) {
            cand_add(c, &n, "THSS", 10, "low duty (%.0f%%)", duty * 100.0);
        }
    }

    /* ---------------- CSS: chirp evidence ---------------- */
    if (pulse != NULL && pulse->chirp_pulses >= 3 &&
        pulse->chirp_pulses * 2 >= pulse->num_pulses) {
        cand_add(c, &n, "CSS", 60,
                 "%d of %d pulses sweep in frequency (%.1f kHz mean sweep)",
                 pulse->chirp_pulses, pulse->num_pulses,
                 pulse->chirp_bw_mean / 1e3);
        if (pulse->chirp_rate_cv > 0.30) {
            cand_add(c, &n, "CSS", 20,
                     "variable sweep rate between pulses (CV %.0f%%) - "
                     "LoRa-like", pulse->chirp_rate_cv * 100.0);
        } else {
            cand_add(c, &n, "CSS", 10, "constant sweep rate");
        }
    } else if (m != NULL && m->chirp_trend > 0.6) {
        cand_add(c, &n, "CSS", 40,
                 "whole-signal IF-vs-time trend %.2f", m->chirp_trend);
    }

    /* ---------------- hybrid: chip rate + hopping ---------------- */
    if (chip_agree && pulse != NULL && pulse->num_pulses >= 3 &&
        pulse->pulse_freq_cv > 0.20) {
        cand_add(c, &n, "FH-DSSS hybrid", 60,
                 "a chip rate (%.1f Hz) AND hopping bursts (freq CV %.0f%%)",
                 bchip > 0.0 ? bchip : cchip, pulse->pulse_freq_cv * 100.0);
    }

    /* ---------------- narrowband / not spread ---------------- */
    {
        int evidence = (bchip > 0.0) + (cchip > 0.0) +
                       (pulse != NULL && pulse->pulse_freq_cv > 0.20) +
                       (pulse != NULL && pulse->interval_cv > 0.30) +
                       (pulse != NULL && pulse->chirp_pulses >= 3);

        cand_add(c, &n, "Not spread spectrum", evidence == 0 ? 55 : 10,
                 evidence == 0
                     ? "no chip-rate, hop, time-hop or chirp evidence"
                     : "only %d family evidences", evidence);
    }
    (void)a;
    (void)ssca;

    /* sort descending */
    for (size_t i = 0; i + 1 < n; i++) {
        for (size_t j = i + 1; j < n; j++) {
            if (c[j].score > c[i].score) {
                Cand t = c[i];

                c[i] = c[j];
                c[j] = t;
            }
        }
    }
    {
        size_t w = n < max_out ? n : max_out;

        for (size_t i = 0; i < w; i++) {
            snprintf(out[i].name, sizeof(out[i].name), "%s", c[i].name);
            out[i].score = c[i].score;
            snprintf(out[i].evidence, sizeof(out[i].evidence), "%s",
                     c[i].evidence);
        }
        return w;
    }
}
