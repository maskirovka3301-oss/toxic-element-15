/*
 * analysis.c — region spectrogram and statistics (see header).
 */
#include "analysis.h"
#include "debug.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsp.h"
#include "region.h"

/* Analysis spectrogram settings (match the original program's look). */
#define ANALYSIS_FFT_SIZE 2048
#define ANALYSIS_OVERLAP  0.75
#define ANALYSIS_MAX_FRAMES 200

void analysis_clear(Analysis *a)
{
    if (a == NULL) return;
    free(a->image);
    memset(a, 0, sizeof(*a));
}

static void classify(Analysis *a)
{
    if (a->bandwidth < 5000.0) {
        strcpy(a->signal_type, "Narrowband");
    } else if (a->bandwidth < 1000000.0) {
        strcpy(a->signal_type, "Wideband");
    } else {
        strcpy(a->signal_type, "Very Wideband (OFDM)");
    }

    if (a->bandwidth < 3000.0) {
        strcpy(a->modulation, "AM/SSB/CW");
    } else if (a->bandwidth < 15000.0) {
        strcpy(a->modulation, "FM/NFM");
    } else if (a->bandwidth < 500000.0) {
        strcpy(a->modulation, "QAM/OFDM");
    } else {
        strcpy(a->modulation, "Wideband OFDM");
    }
}

bool analysis_run(const IqSource *src, const Region *region, Analysis *out)
{
    if (src == NULL || region == NULL || out == NULL) return false;

    Region r = *region;
    if (!region_normalize(&r, src)) {
        fprintf(stderr, "analysis: empty or out-of-bounds selection\n");
        return false;
    }

    size_t start_sample = (size_t)(r.time_min * src->sample_rate);
    size_t end_sample   = (size_t)(r.time_max * src->sample_rate);
    if (end_sample > src->total_samples) end_sample = src->total_samples;
    if (start_sample >= src->total_samples) {
        fprintf(stderr, "analysis: selection starts past end of recording\n");
        return false;
    }

    size_t segment_len = end_sample - start_sample;
    if (segment_len < (size_t)ANALYSIS_FFT_SIZE) {
        fprintf(stderr, "analysis: selection too short (%zu samples)\n",
                segment_len);
        return false;
    }
    if (segment_len > TE15_MAX_ANALYSIS_SAMPLES) {
        fprintf(stderr,
                "analysis: selection is %.1f s long; truncating to the "
                "first %.1f s (%d M samples)\n",
                (double)segment_len / src->sample_rate,
                (double)TE15_MAX_ANALYSIS_SAMPLES / src->sample_rate,
                TE15_MAX_ANALYSIS_SAMPLES / 1000000);
        segment_len = TE15_MAX_ANALYSIS_SAMPLES;
    }

    /*
     * Copy the selected part of the recording out of the memory map
     * into a private buffer; the whole analysis below runs on this
     * buffer, never touching the file mapping again.
     */
    float *segment = malloc(2 * segment_len * sizeof(float));
    if (segment == NULL) {
        fprintf(stderr, "analysis: out of memory for the segment buffer\n");
        return false;
    }
    iq_source_read_complex(src, start_sample, segment, segment_len);

    analysis_clear(out);
    out->bandwidth   = r.freq_max - r.freq_min;
    out->center_freq = (r.freq_min + r.freq_max) / 2.0;
    out->duration    = r.time_max - r.time_min;

    const int    fft = ANALYSIS_FFT_SIZE;
    size_t       hop = (size_t)((int)(fft * (1.0 - ANALYSIS_OVERLAP)));
    size_t num_frames = (segment_len - (size_t)fft) / hop + 1;
    if (num_frames > ANALYSIS_MAX_FRAMES) {
        /* Do not truncate: adapt the frame hop so the image spans the
         * WHOLE selection (a truncated image showed only the first few
         * milliseconds of long selections). */
        hop = (segment_len - (size_t)fft) / ANALYSIS_MAX_FRAMES;
        if (hop == 0) hop = 1;
        num_frames = (segment_len - (size_t)fft) / hop + 1;
        if (num_frames > ANALYSIS_MAX_FRAMES) num_frames = ANALYSIS_MAX_FRAMES;
    }

    /* Map the selection's frequency span onto FFT bins. */
    const double bin_hz = src->sample_rate / (double)fft;
    const double f_lo = src->center_freq - src->sample_rate / 2.0;
    int64_t bmin = (int64_t)((r.freq_min - f_lo) / bin_hz);
    int64_t bmax = (int64_t)((r.freq_max - f_lo) / bin_hz);
    if (bmin < 0) bmin = 0;
    if (bmax > fft) bmax = fft;
    if (bmax - bmin <= 0) {
        bmin = 0;
        bmax = 1;
    }
    const int bins = (int)(bmax - bmin);

    out->image = malloc(num_frames * (size_t)bins * sizeof(float));
    if (out->image == NULL) {
        fprintf(stderr, "analysis: out of memory for spectrogram\n");
        free(segment);
        return false;
    }
    out->frames = (int)num_frames;
    out->bins   = bins;
    out->bin_hz = bin_hz;
    out->dt     = (double)hop / src->sample_rate;

    FftPlan *plan   = fft_plan_create(fft);
    float   *window = malloc((size_t)fft * sizeof(float));
    if (plan == NULL || window == NULL) {
        fprintf(stderr, "analysis: cannot allocate FFT plan\n");
        fft_plan_destroy(plan);
        free(window);
        free(segment);
        analysis_clear(out);
        return false;
    }
    dsp_hann(window, (size_t)fft);

    float *frame_power = malloc(num_frames * sizeof(float));
    float *spectrum = malloc((size_t)fft * sizeof(float));
    if (frame_power == NULL || spectrum == NULL) {
        fprintf(stderr, "analysis: out of memory for analysis buffers\n");
        fft_plan_destroy(plan);
        free(window);
        free(frame_power);
        free(spectrum);
        free(segment);
        analysis_clear(out);
        return false;
    }

    double   sum = 0.0;
    double   max_val = -1e9;
    for (size_t i = 0; i < num_frames; i++) {
        const float *iq = segment + 2 * (i * (size_t)hop);
        fft_plan_spectrum(plan, iq, window, spectrum);

        float band_sum = 0.0f;
        for (int j = 0; j < bins; j++) {
            /*
             * `spectrum` is already fftshifted (index j' corresponds to
             * RF frequency f_lo + (j' - fft/2) * bin_hz). Bin indices are
             * therefore computed directly from the RF low edge and read
             * without an additional shift.
             */
            int idx = (int)((bmin + j) % fft);
            float val = spectrum[idx];
            out->image[i * (size_t)bins + (size_t)j] = val;
            band_sum += val;
            sum += val;
            if (val > max_val) max_val = val;
        }
        frame_power[i] = band_sum / (float)bins;
    }

    fft_plan_destroy(plan);
    free(window);
    free(spectrum);
    free(segment);

    const size_t cell_count = num_frames * (size_t)bins;
    out->avg_power = sum / (double)cell_count;
    out->max_power = max_val;
    out->entropy   = dsp_entropy(out->image, cell_count);
    out->prf       = dsp_estimate_prf(frame_power, num_frames, out->dt);
    free(frame_power);

    classify(out);

    TE15_LOG("\n========================================\n"
           "ANALYZING SELECTION\n"
           "========================================\n"
           "Frequency: %.3f - %.3f MHz\n"
           "Bandwidth: %.2f kHz\n"
           "Time: %.2f - %.2f s\n"
           "Duration: %.2f s\n"
           "Entropy: %.4f bits\n"
           "PRF: %.2f Hz\n"
           "Signal Type: %s\n"
           "Modulation: %s\n",
           r.freq_min / 1e6, r.freq_max / 1e6, out->bandwidth / 1e3,
           r.time_min, r.time_max, out->duration,
           out->entropy, out->prf, out->signal_type, out->modulation);
    TE15_LOG("========================================\n\n");
    return true;
}
