/*
 * detect.c — automatic signal detection over a spectrogram window.
 *
 * Pipeline:
 *   1. Average the frames into a mean PSD (linear power).
 *   2. Noise floor = median of the mean PSD; threshold = floor + 6 dB.
 *   3. Frequency runs: contiguous above-threshold stretches (gaps up to
 *      2 bins merged) become candidate regions.
 *   4. Region merging, iterated until stable:
 *      a. Proximity: runs separated by a short gap (<= 16 bins) whose
 *         bins never fall below the noise floor belong to the same wide
 *         signal (FM comb spectra have such dips).
 *      b. Time correlation: fragments whose per-frame band power varies
 *         with high correlation (r >= 0.8) and sits within 128 bins are
 *         one signal (pulsed carriers + their side lobes).
 *      Constant-power pairs that are not close in frequency never merge,
 *      so independent narrow signals stay separate.
 *   5. Tighten each region in time: per-frame band power vs its own
 *      floor; the longest on-run (small gaps merged) becomes the time
 *      span; if nothing stands out, the full window.
 */
#include "detect.h"

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#define DETECT_DB_ABOVE       6.0   /* detection threshold over floor */
#define DETECT_KEEP_DB_ABOVE   14.0  /* post-merge minimum peak, dB over floor */
#define DETECT_GAP_BINS       2     /* gap merged inside a run        */
#define PROX_GAP_BINS         16    /* proximity-merge gap            */
#define PROX_FLOOR_RATIO      0.9   /* gap bins must stay >= floor*ratio */
#define CORR_CLOSE_BINS       128   /* correlation-merge span         */
#define CORR_MIN              0.8   /* Pearson correlation threshold  */
#define DETECT_COLLECT        64    /* internal region collection cap */
#define TIME_DB_ABOVE         6.0   /* on/off threshold over floor    */
#define TIME_GAP_FRAMES       3     /* merged gap between on-runs     */

typedef struct {
    int    lo, hi;      /* bin range in the mean PSD */
    double peak;        /* strongest bin (linear)    */
} Run;

static int cmp_run_peak_desc(const void *a, const void *b)
{
    const Run *x = a, *y = b;
    return (x->peak < y->peak) - (x->peak > y->peak);
}

/* Per-frame band power (dB) of a region; also reports the variance. */
static void region_frame_power(const float *db, size_t frames, size_t fft_size,
                               int lo, int hi, double *power, double *var_out)
{
    double mean = 0.0;
    for (size_t f = 0; f < frames; f++) {
        const float *row = db + f * fft_size;
        float best = -1e30f;
        for (int b = lo; b < hi; b++) {
            if (row[b] > best) best = row[b];
        }
        power[f] = best;
        mean += best;
    }
    mean /= (double)frames;
    double var = 0.0;
    for (size_t f = 0; f < frames; f++) {
        double d = power[f] - mean;
        var += d * d;
    }
    var /= (double)frames;
    if (var_out != NULL) *var_out = var;
}

size_t detect_signals(const float *db, size_t frames, size_t fft_size,
                      size_t win_start_frame, size_t frame_step, double fps,
                      double sample_rate, double center_freq,
                      SignalRegion *out, size_t max_regions)
{
    if (out == NULL || max_regions == 0) return 0;
    if (db == NULL || frames == 0 || fft_size < 8 || frame_step == 0 ||
        fps <= 0.0 || sample_rate <= 0.0) {
        return 0;
    }

    /* --- 1. mean PSD (linear power) --- */
    double *psd = calloc(fft_size, sizeof(double));
    if (psd == NULL) return 0;
    for (size_t f = 0; f < frames; f++) {
        const float *row = db + f * fft_size;
        for (size_t b = 0; b < fft_size; b++) {
            psd[b] += pow(10.0, (double)row[b] / 10.0);
        }
    }
    for (size_t b = 0; b < fft_size; b++) psd[b] /= (double)frames;

    /* --- 2. noise floor (median) and threshold --- */
    double *sorted = malloc(fft_size * sizeof(double));
    if (sorted == NULL) {
        free(psd);
        return 0;
    }
    memcpy(sorted, psd, fft_size * sizeof(double));
    for (size_t i = 1; i < fft_size; i++) { /* insertion sort */
        double key = sorted[i];
        size_t j = i;
        while (j > 0 && sorted[j - 1] > key) {
            sorted[j] = sorted[j - 1];
            j--;
        }
        sorted[j] = key;
    }
    double floor_psd = sorted[fft_size / 2];
    free(sorted);
    if (floor_psd <= 0.0) {
        free(psd);
        return 0;
    }
    double threshold = floor_psd * pow(10.0, DETECT_DB_ABOVE / 10.0);
    double gap_floor = floor_psd * PROX_FLOOR_RATIO;

    /* --- 3. frequency runs --- */
    Run *runs = malloc((DETECT_COLLECT + 1) * sizeof(Run));
    if (runs == NULL) {
        free(psd);
        return 0;
    }
    size_t run_count = 0;
    int above = -1;
    for (size_t b = 0; b <= fft_size; b++) {
        bool hot = (b < fft_size) && (psd[b] >= threshold);
        if (hot && above < 0) {
            above = (int)b;
        } else if (!hot && above >= 0) {
            int run_lo = above;
            int run_hi = (int)b;
            if (run_count > 0 &&
                run_lo - runs[run_count - 1].hi <= DETECT_GAP_BINS) {
                runs[run_count - 1].hi = run_hi;
            } else if (run_count < DETECT_COLLECT) {
                runs[run_count].lo = run_lo;
                runs[run_count].hi = run_hi;
                run_count++;
            }
            above = -1;
        }
    }
    for (size_t r = 0; r < run_count; r++) {
        double peak = 0.0;
        for (int b = runs[r].lo; b < runs[r].hi; b++)
            if (psd[b] > peak) peak = psd[b];
        runs[r].peak = peak;
    }

    /* --- 4. iterative merging --- */
    double *pa = malloc(frames * sizeof(double));
    double *pb = malloc(frames * sizeof(double));
    if (pa != NULL && pb != NULL) {
        bool changed = true;
        while (changed && run_count > 1) {
            changed = false;
            for (size_t i = 0; i < run_count; i++) {
                for (size_t j = i + 1; j < run_count; j++) {
                    /* ensure i is left of j for the gap test */
                    const Run *a = &runs[i];
                    const Run *b = &runs[j];
                    if (a->lo > b->lo) { const Run *t = a; a = b; b = t; }
                    int gap = b->lo - a->hi;

                    bool merge = false;

                    /* (a) short gap that never falls below the floor:
                     * part of the same wide signal */
                    if (gap <= PROX_GAP_BINS && gap > 0) {
                        merge = true;
                        for (int k = a->hi; k < b->lo; k++) {
                            if (psd[k] < gap_floor) {
                                merge = false;
                                break;
                            }
                        }
                    }

                    /* (b) temporally correlated fragments */
                    if (!merge && gap <= CORR_CLOSE_BINS) {
                        double var_a = 0.0, var_b = 0.0;
                        region_frame_power(db, frames, fft_size,
                                           a->lo, a->hi, pa, &var_a);
                        region_frame_power(db, frames, fft_size,
                                           b->lo, b->hi, pb, &var_b);
                        if (var_a >= 1e-6 && var_b >= 1e-6) {
                            double ma = 0.0, mb = 0.0;
                            for (size_t f = 0; f < frames; f++) {
                                ma += pa[f];
                                mb += pb[f];
                            }
                            ma /= (double)frames;
                            mb /= (double)frames;
                            double s_ab = 0.0, s_aa = 0.0, s_bb = 0.0;
                            for (size_t f = 0; f < frames; f++) {
                                double da = pa[f] - ma, dbb = pb[f] - mb;
                                s_ab += da * dbb;
                                s_aa += da * da;
                                s_bb += dbb * dbb;
                            }
                            double r = (s_aa > 0.0 && s_bb > 0.0)
                                           ? s_ab / sqrt(s_aa * s_bb)
                                           : 0.0;
                            merge = r >= CORR_MIN;
                        }
                    }

                    if (merge) {
                        if (runs[j].lo < runs[i].lo) runs[i].lo = runs[j].lo;
                        if (runs[j].hi > runs[i].hi) runs[i].hi = runs[j].hi;
                        if (runs[j].peak > runs[i].peak)
                            runs[i].peak = runs[j].peak;
                        runs[j] = runs[run_count - 1]; /* drop j */
                        run_count--;
                        j--;
                        changed = true;
                    }
                }
            }
        }
    }
    free(pa);
    free(pb);

    /* drop marginal remnants (deep-fringe side lobes, noise bumps):
     * a probable signal must clear the floor by KEEP dB */
    {
        double keep_threshold = floor_psd * pow(10.0, DETECT_KEEP_DB_ABOVE / 10.0);
        size_t k = 0;
        for (size_t r = 0; r < run_count; r++) {
            if (runs[r].peak >= keep_threshold) {
                runs[k++] = runs[r];
            }
        }
        run_count = k;
    }

    /* keep only the strongest max_regions, in descending peak order */
    if (run_count > max_regions) {
        qsort(runs, run_count, sizeof(Run), cmp_run_peak_desc);
        run_count = max_regions;
    }

    const double bin_hz = sample_rate / (double)fft_size;
    const double f_lo = center_freq - sample_rate / 2.0;

    size_t out_count = 0;
    for (size_t r = 0; r < run_count && out_count < max_regions; r++) {
        int lo = runs[r].lo, hi = runs[r].hi;

        /* --- 5. tighten in time: per-frame band power --- */
        double *frame_power = malloc(frames * sizeof(double));
        if (frame_power == NULL) break;
        for (size_t f = 0; f < frames; f++) {
            const float *row = db + f * fft_size;
            double best = -1e300;
            for (int b = lo; b < hi; b++) {
                if (row[b] > best) best = row[b];
            }
            frame_power[f] = best;
        }
        double *fp_sorted = malloc(frames * sizeof(double));
        if (fp_sorted == NULL) {
            free(frame_power);
            break;
        }
        memcpy(fp_sorted, frame_power, frames * sizeof(double));
        for (size_t i = 1; i < frames; i++) {
            double key = fp_sorted[i];
            size_t j = i;
            while (j > 0 && fp_sorted[j - 1] > key) {
                fp_sorted[j] = fp_sorted[j - 1];
                j--;
            }
            fp_sorted[j] = key;
        }
        double fp_floor = fp_sorted[frames / 2];
        free(fp_sorted);
        double fp_thresh = fp_floor + TIME_DB_ABOVE;

        /* longest on-run with small gaps merged */
        size_t best_len = 0, best_lo = 0;
        size_t run_lo = 0;
        bool in_run = false;
        size_t last_on = 0;
        for (size_t f = 0; f <= frames; f++) {
            bool on = (f < frames) && (frame_power[f] >= fp_thresh);
            if (on) {
                if (!in_run) run_lo = f;
                in_run = true;
                last_on = f;
            } else if (in_run && (f - last_on) > TIME_GAP_FRAMES) {
                size_t len = last_on - run_lo + 1;
                if (len > best_len) {
                    best_len = len;
                    best_lo = run_lo;
                }
                in_run = false;
            }
        }
        free(frame_power);

        size_t t_lo = 0, t_hi = frames - 1;
        if (best_len > 0 && best_len < frames) {
            t_lo = best_lo;
            t_hi = best_lo + best_len - 1;
            if (t_lo > 0) t_lo--; /* a little headroom around the run */
            if (t_hi + 1 < frames) t_hi++;
        }

        SignalRegion *sr = &out[out_count];
        sr->freq_min = f_lo + (double)lo * bin_hz;
        sr->freq_max = f_lo + (double)hi * bin_hz;
        sr->time_min = (double)(win_start_frame + t_lo * frame_step) / fps;
        sr->time_max = (double)(win_start_frame + (t_hi + 1) * frame_step) / fps;
        sr->peak_db = 10.0 * log10(runs[r].peak);
        out_count++;
    }

    free(psd);
    free(runs);
    return out_count;
}

