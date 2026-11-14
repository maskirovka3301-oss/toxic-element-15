/*
 * detect.h — automatic signal detection over a spectrogram window.
 *
 * Finds probable signals in a dB spectrogram: averages the frames into a
 * PSD, thresholds against the noise floor, grows contiguous
 * above-threshold frequency runs, and tightens each candidate in time
 * using its per-frame band power. The result is a list of clickable
 * regions for the UI.
 */
#ifndef TE15_DETECT_H
#define TE15_DETECT_H

#include <stddef.h>

#define DETECT_MAX_REGIONS 16

typedef struct {
    double freq_min, freq_max; /* absolute RF Hz                  */
    double time_min, time_max; /* seconds from recording start    */
    double peak_db;            /* strongest averaged bin, dB      */
} SignalRegion;

/*
 * Detect up to `max_regions` probable signals in a spectrogram.
 *
 * `db` holds `frames` x `fft_size` dB magnitudes, row-major per frame;
 * frame 0 of the array is absolute frame `win_start_frame`, and
 * consecutive rows are spaced `frame_step` frames apart (>= 1, so a
 * decimated cache window maps to time correctly). The spectrogram
 * spans the full sample rate around `center_freq`.
 *
 * Returns the number of regions written to `out` (regions are sorted by
 * descending peak power and capped at max_regions).
 */
size_t detect_signals(const float *db, size_t frames, size_t fft_size,
                      size_t win_start_frame, size_t frame_step, double fps,
                      double sample_rate, double center_freq,
                      SignalRegion *out, size_t max_regions);

#endif /* TE15_DETECT_H */
