/*
 * modconf.c — modulation confidence scoring (see header).
 */
#include "modconf.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static void add(ModCandidate *c, size_t *n, const char *name, int score,
                const char *why)
{
    if (*n >= 24) return;
    snprintf(c[(*n)].name, sizeof(c[(*n)].name), "%s", name);
    c[(*n)].score = score;
    snprintf(c[(*n)].why, sizeof(c[(*n)].why), "%s", why);
    (*n)++;
}

size_t modconf_rank(const Analysis *a, const Metrics *m,
                    const CyclicResult *cyc, const CccResult *ccc,
                    const HosResult *hos, const PulseResult *pulse,
                    const FluctResult *fluct, const CparResult *cpar,
                    ModCandidate *out, size_t max_out)
{
    ModCandidate c[24];
    size_t n = 0;
    if (out == NULL || max_out == 0) return 0;

    if (m != NULL) {
        double bw = m->occupied_bw_hz > 0.0 ? m->occupied_bw_hz : 1e9;
        double flat = m->spectral_flatness;
        double papr = m->papr_db;
        double amd = m->am_depth;
        double duty = m->duty_cycle;
        double fmd = m->fm_deviation_hz / bw;

        /* CW tone */
        if (m->tone_purity > 0.3 && bw < 3000.0 && m->snr_db > 10.0) {
            add(c, &n, "CW tone",
                (int)(50.0 + 45.0 * m->tone_purity), "tone purity + narrow band");
        }
        /* AM / SSB */
        if (amd > 0.25 && duty > 0.7 && fmd < 0.3) {
            char why[96];
            snprintf(why, sizeof(why), "AM depth %.0f%%, symmetric envelope",
                     amd * 100.0);
            add(c, &n, "AM (voice/SW)",
                (int)(30.0 + 60.0 * amd), why);
        }
        /* FM */
        if (fmd > 0.10 && amd < 0.5) {
            char why[96];
            snprintf(why, sizeof(why), "FM deviation %.1f kHz (%.2f x bw)",
                     m->fm_deviation_hz / 1e3, fmd);
            add(c, &n, "FM (voice)",
                (int)(30.0 + 60.0 * fmin(1.0, fmd * 2.0)), why);
        }
        /* FSK */
        if (m->fsk_levels >= 2) {
            char why[96];
            snprintf(why, sizeof(why), "%d FSK levels", m->fsk_levels);
            add(c, &n, m->fsk_levels >= 4 ? "4-FSK / multilevel FSK" : "2-FSK",
                80, why);
        }
        /* OFDM / QAM (multi-carrier) */
        if (flat > 0.45 && papr > 7.0 && amd < 0.85) {
            char why[96];
            snprintf(why, sizeof(why),
                     "flat spectrum %.2f, crest %.1f dB", flat, papr);
            add(c, &n, "OFDM / QAM (multi-carrier)",
                (int)(30.0 + 45.0 * flat + fmin(20.0, papr)), why);
        }
        /* OOK / pulsed */
        if (duty < 0.92 && amd > 0.6) {
            char why[96];
            snprintf(why, sizeof(why), "duty %.0f%%, envelope depth %.0f%%",
                     duty * 100.0, amd * 100.0);
            add(c, &n, "OOK / pulsed emission",
                (int)(40.0 + 55.0 * amd), why);
        }
        /* chirp (whole-signal IF trend) */
        if (m->chirp_trend > 0.6) {
            char why[96];
            snprintf(why, sizeof(why), "IF vs time trend %.2f", m->chirp_trend);
            add(c, &n, "CHIRP (linear sweep)",
                (int)(40.0 + 50.0 * m->chirp_trend), why);
        }
        /* noise floor */
        if (flat > 0.85 && m->snr_db < 6.0 && amd < 0.35) {
            add(c, &n, "NOISE (no structure)", 70,
                "flat spectrum, low SNR, constant envelope");
        }
    }
    if (a != NULL) {
        if (strstr(a->modulation, "QAM") != NULL ||
            strstr(a->modulation, "OFDM") != NULL) {
            add(c, &n, "OFDM / QAM (multi-carrier)", 60,
                "analysis modulation classifier");
        }
        if (strstr(a->modulation, "FSK") != NULL) {
            add(c, &n, "FSK", 55, "analysis modulation classifier");
        }
        if (strstr(a->modulation, "PSK") != NULL) {
            add(c, &n, "PSK", 55, "analysis modulation classifier");
        }
    }
    if (ccc != NULL && ccc->valid) {
        if (strstr(ccc->verdict, "BPSK") != NULL) {
            add(c, &n, "BPSK", 85, "cyclic cumulants: BPSK-like features");
        } else if (strstr(ccc->verdict, "QPSK") != NULL) {
            add(c, &n, "QPSK", 80, "cyclic cumulants: QPSK-like features");
        } else if (strstr(ccc->verdict, "8PSK") != NULL ||
                   strstr(ccc->verdict, "MSK") != NULL) {
            add(c, &n, "8PSK / MSK", 75,
                "cyclic cumulants: no C40, C42 present");
        } else if (strstr(ccc->verdict, "16QAM") != NULL) {
            add(c, &n, "16QAM", 75, "cyclic cumulants: 16QAM-like features");
        }
    }
    if (hos != NULL && hos->non_gaussian) {
        add(c, &n, hos->hint, 40,
            "non-Gaussian statistics (higher-order cumulants)");
    }
    if (pulse != NULL && pulse->num_pulses >= 2) {
        if (strstr(pulse->modulation, "PPM") != NULL) {
            add(c, &n, "PPM pulse train", 80, "discrete interval levels");
        } else if (strstr(pulse->modulation, "PIM") != NULL) {
            add(c, &n, "PIM pulse train", 75, "interval-modulated bursts");
        } else if (strstr(pulse->modulation, "chirp") != NULL) {
            add(c, &n, "Chirped pulse train", 85,
                "per-pulse IF sweeps (variable waveform)");
        } else if (strstr(pulse->modulation, "PWM") != NULL) {
            add(c, &n, "PWM pulse train", 75, "width-modulated pulses");
        } else if (strstr(pulse->modulation, "constant-PRF") != NULL) {
            add(c, &n, "Pulsed radar / beacon", 70,
                "constant-PRF pulse train");
        } else if (pulse->num_pulses >= 5) {
            add(c, &n, "Bursty pulsed emission", 65, "burst train detected");
        }
    }
    if (cyc != NULL && cyc->peak_count > 0) {
        add(c, &n, "Cyclostationary digital (unknown)", 35,
            "significant cycle frequencies");
    }
    if (fluct != NULL && fluct->detected && cpar != NULL && cpar->detected) {
        double r = fluct->chip_rate_hz > 0.0 && cpar->chip_rate_hz > 0.0
                       ? (cpar->chip_rate_hz > fluct->chip_rate_hz
                              ? cpar->chip_rate_hz / fluct->chip_rate_hz
                              : fluct->chip_rate_hz / cpar->chip_rate_hz)
                       : 1e9;
        if (r < 1.5) {
            char why[96];
            snprintf(why, sizeof(why),
                     "Burel/CPAR chip rates agree (%.0f Hz)", fluct->chip_rate_hz);
            add(c, &n, "DSSS / spread spectrum", 85, why);
        } else {
            add(c, &n, "DSSS / spread spectrum", 45,
                "DSSS detectors disagree on the chip rate");
        }
    }

    if (n == 0) {
        add(c, &n, "Unknown / no strong evidence", 20,
            "insufficient evidence");
    }

    /* merge duplicate names: keep the higher score, average-ish */
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i + 1; j < n; j++) {
            if (strcmp(c[i].name, c[j].name) == 0) {
                if (c[j].score > c[i].score) {
                    c[i].score = (c[i].score + c[j].score) / 2 + 10;
                }
                c[j].name[0] = '\0';
            }
        }
    }
    size_t written = 0;
    for (size_t i = 0; i < n && written < max_out; i++) {
        if (c[i].name[0] == '\0') continue;
        memcpy(out[written].name, c[i].name, sizeof(out[written].name));
        out[written].score = c[i].score > 100 ? 100 : c[i].score;
        memcpy(out[written].why, c[i].why, sizeof(out[written].why));
        written++;
    }
    /* sort descending */
    for (size_t i = 0; i + 1 < written; i++) {
        for (size_t j = i + 1; j < written; j++) {
            if (out[j].score > out[i].score) {
                ModCandidate t = out[i];
                out[i] = out[j];
                out[j] = t;
            }
        }
    }
    return written;
}
