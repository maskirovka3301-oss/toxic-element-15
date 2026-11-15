/*
 * sot.c — the shared, preprocessed analysis source (see header).
 */
#include "sot.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsp.h"

#define SOT_MAX_SAMPLES (8u * 1024u * 1024u) /* complex samples        */
#define SOT_PSD_N       65536
#define SOT_PSD_AVG     8

void sot_clear(Sot *out)
{
    if (out == NULL) return;
    free(out->iq);
    memset(out, 0, sizeof(*out));
}

/*
 * PSD of the prepared selection slice, used ONLY to derive the band filter
 * and the noise floor from data inside the user's selection: the passband is
 * the selected bandwidth itself, the stopband a fixed transition outside it,
 * and the noise floor is the median of the in-band PSD bins. No wideband or
 * neighbouring-emission context is consulted - the analyses are given the
 * selected IQ samples and the selected bandwidth, nothing else.
 */
static bool derive_filter(const float *iq, size_t n, double fs,
                          double sel_bw, double *pass_hz, double *stop_hz,
                          double *floor_db)
{
    float *blk = NULL;
    float *spec = NULL;
    double *psd = NULL;
    double hann[256];
    size_t psd_n = SOT_PSD_N;
    size_t blocks = 0;
    bool ok = false;

    *pass_hz = *stop_hz = 0.0;
    *floor_db = -200.0;
    if (n < psd_n || sel_bw <= 0.0 || fs <= 0.0) return false;
    blk = malloc(2 * psd_n * sizeof(float));
    spec = malloc(psd_n * sizeof(float));
    psd = calloc(psd_n, sizeof(double));
    if (blk == NULL || spec == NULL || psd == NULL) {
        free(blk);
        free(spec);
        free(psd);
        return false;
    }
    for (int w = 0; w < 256; w++) {
        hann[w] = 0.5 - 0.5 * cos(2.0 * M_PI * (double)w / 255.0);
    }
    for (size_t b = 0; b < SOT_PSD_AVG; b++) {
        size_t off = (size_t)((double)b / (double)SOT_PSD_AVG *
                              (double)(n - psd_n));
        FftPlan *plan = fft_plan_create((int)psd_n);
        fftwf_complex *xa = NULL;

        if (plan == NULL) break;
        xa = fftwf_malloc(sizeof(fftwf_complex) * psd_n);
        if (xa == NULL) {
            fft_plan_destroy(plan);
            break;
        }
        for (size_t i = 0; i < psd_n; i++) {
            double wnd = hann[i * 256 / psd_n];

            blk[2 * i] = (float)(iq[2 * (off + i)] * wnd);
            blk[2 * i + 1] = (float)(iq[2 * (off + i) + 1] * wnd);
        }
        fft_plan_transform(plan, blk, NULL, xa);
        for (size_t i = 0; i < psd_n; i++) {
            psd[i] += (double)xa[i][0] * xa[i][0] +
                      (double)xa[i][1] * xa[i][1];
        }
        fftwf_free(xa);
        fft_plan_destroy(plan);
        blocks++;
    }
    free(blk);
    free(spec);
    if (blocks == 0) {
        free(psd);
        return false;
    }
    {
        size_t prof_n = psd_n / 2;
        double bin_hz = fs / (double)psd_n;
        size_t k0 = (size_t)(0.5 * sel_bw / bin_hz);
        double *inband;
        size_t inband_n = 0;

        if (k0 >= prof_n) k0 = prof_n - 1;
        inband = malloc((k0 + 1) * sizeof(*inband));
        if (inband == NULL) {
            free(psd);
            return false;
        }
        /* the in-band PSD bins: the noise floor is their median */
        for (size_t k = 0; k <= k0; k++) inband[inband_n++] = psd[k];
        for (size_t i = 0; i + 1 < inband_n; i++) {
            for (size_t j = i + 1; j < inband_n; j++) {
                if (inband[j] < inband[i]) {
                    double t = inband[i];

                    inband[i] = inband[j];
                    inband[j] = t;
                }
            }
        }
        *floor_db = 10.0 * log10(inband[inband_n / 2] + 1e-30);
        free(inband);
        {
            double pass_edge = 0.5 * sel_bw;
            double stop_edge = pass_edge * 1.35;

            if (stop_edge > 0.49 * fs) stop_edge = 0.49 * fs;
            if (stop_edge <= pass_edge) stop_edge = pass_edge * 1.05;
            *pass_hz = pass_edge;
            *stop_hz = stop_edge;
            ok = true;
        }
    }
    free(psd);
    return ok;
}

bool sot_prepare(const IqSource *src, const Region *region, Sot *out)
{
    Region r;
    double fs, centre, mix_hz;
    size_t s0, n;

    if (out == NULL) return false;
    sot_clear(out);
    if (src == NULL || region == NULL) return false;
    r = *region;
    if (!region_normalize(&r, src)) return false;
    fs = src->sample_rate;
    s0 = (size_t)(r.time_min * fs);
    n = (size_t)((r.time_max - r.time_min) * fs);
    if (s0 >= src->total_samples) return false;
    if (s0 + n > src->total_samples) n = src->total_samples - s0;
    if (n > SOT_MAX_SAMPLES) n = SOT_MAX_SAMPLES;
    if (n < 1024) return false;

    out->iq = malloc(2 * n * sizeof(float));
    if (out->iq == NULL) return false;
    iq_source_read_complex(src, s0, out->iq, n);
    out->n = n;
    out->region = r;

    /* --- 0. normalise the raw slice BEFORE anything else ---------------
     * A single POSITIVE REAL gain is applied to the whole block: it cannot
     * change the phase (arg x is untouched) or the frequency of any
     * component - it scales every I and Q value by the same known factor.
     * Dividing by sot->gain recovers the original samples exactly, and all
     * ratios (SNR, on/off levels, CVs, duty) are invariant. No DC removal,
     * AGC or per-block scaling is done: those would alter phase or the
     * amplitude structure the analyses depend on. */
    {
        double sum = 0.0;
        double peak = 0.0;

        for (size_t i = 0; i < n; i++) {
            double re = out->iq[2 * i], im = out->iq[2 * i + 1];
            double p = re * re + im * im;

            sum += p;
            if (p > peak) peak = p;
        }
        {
            double rms = sqrt(sum / (double)n);
            double peak_abs = sqrt(peak);
            const double target_rms = 0.1;   /* -20 dBFS */
            const double max_peak = 0.98;    /* headroom under full scale */
            double g = 1.0;

            if (rms > 0.0 && isfinite(rms) && isfinite(peak_abs)) {
                g = target_rms / rms;
                if (peak_abs > 0.0 && g * peak_abs > max_peak) {
                    g = max_peak / peak_abs; /* never clip */
                }
            }
            if (!isfinite(g) || g <= 0.0) g = 1.0;
            out->gain = g;
            out->gain_db = 20.0 * log10(g);
            out->rms_in = rms;
            if (g != 1.0) {
                for (size_t i = 0; i < 2 * n; i++) {
                    out->iq[i] = (float)((double)out->iq[i] * g);
                }
            }
        }
    }

    /* mix the selection to DC (phase from the absolute sample index so
     * long slices do not drift) */
    centre = 0.5 * (r.freq_min + r.freq_max);
    mix_hz = centre - src->center_freq;
    out->mix_hz = mix_hz;
    if (mix_hz != 0.0) {
        double dth = -2.0 * M_PI * mix_hz / fs;
        double th = fmod(dth * (double)s0, 2.0 * M_PI);

        for (size_t i = 0; i < n; i++) {
            double c = cos(th), sn = sin(th);
            double re = out->iq[2 * i], im = out->iq[2 * i + 1];

            out->iq[2 * i] = (float)(re * c - im * sn);
            out->iq[2 * i + 1] = (float)(re * sn + im * c);
            th += dth;
        }
    }

    /* context-derived band filter (see derive_filter) */
    {
        double pass_hz = 0.0, stop_hz = 0.0, floor_db = 0.0;

        if (derive_filter(out->iq, n, fs, r.freq_max - r.freq_min, &pass_hz,
                          &stop_hz, &floor_db)) {
            if (dsp_fft_mask_lowpass(out->iq, n, fs, pass_hz, stop_hz)) {
                out->pass_hz = pass_hz;
                out->stop_hz = stop_hz;
                out->trans_hz = stop_hz - pass_hz;
            }
            out->noise_floor_db = floor_db;
        }
    }

    /* expose the buffer as a pseudo-source: the analyses keep their
     * existing (src, region) API and read this data through it */
    memset(&out->view, 0, sizeof(out->view));
    out->view.path = "(prepared selection source)";
    out->view.fd = -1;
    out->view.file_size = 2 * n * sizeof(float);
    out->view.samples = out->iq;
    out->view.raw = (const uint8_t *)out->iq;
    out->view.total_samples = n;
    out->view.bytes_per_sample = 8;
    out->view.sample_rate = fs;
    out->view.center_freq = centre;
    out->view.duration = (double)n / fs;
    out->view.format = IQ_FORMAT_FLOAT32;
    out->valid = true;
    return true;
}
