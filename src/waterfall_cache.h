/*
 * waterfall_cache.h — background spectrogram computation with an LRU
 * cache of contiguous frame windows.
 *
 * The cache owns one worker thread. `wc_request` asks for the window
 * containing a given frame; the worker computes it off the UI thread and
 * publishes it atomically. Readers lock a window with `wc_lock`, read the
 * dB data, and unlock it — eviction and publication happen under the same
 * lock, so a locked window's data can never be freed mid-draw.
 */
#ifndef TE15_WATERFALL_CACHE_H
#define TE15_WATERFALL_CACHE_H

#include <stddef.h>
#include "iq_source.h"
#include "dsp.h"

typedef struct {
    int      fft_size;
    double   overlap;     /* 0..1 */
    DspWindow window_type; /* spectrum window              */
} WcSpec;

typedef struct WaterfallCache WaterfallCache;

/*
 * Create a cache over `src`. At most `max_windows` contiguous frame
 * windows are retained; a window holds at most `max_window_frames`
 * spectra. The cache borrows `src`; it must outlive the cache.
 */
WaterfallCache *wc_create(const IqSource *src, const WcSpec *spec,
                          size_t max_windows, size_t max_window_frames);
void wc_destroy(WaterfallCache *wc);

size_t wc_total_frames(const WaterfallCache *wc);
double wc_frames_per_second(const WaterfallCache *wc);

/* The FFT size the cache was created with (rows are this many bins). */
int wc_fft_size(const WaterfallCache *wc);

/*
 * Asynchronously ensure that the window containing `frame` is computed.
 * `want_frames` is a hint for the window span (clamped to the configured
 * maximum and the recording length). Safe to call from any thread;
 * returns immediately.
 */
void wc_request(WaterfallCache *wc, size_t frame, size_t want_frames);

/*
 * Lock the valid window covering `frame` for reading. On success the
 * returned pointer points at win_frames*fft_size dB floats laid out
 * row-major per frame (frame 0 of the window first) and `*win_start` is
 * the absolute index of that first frame. The window's rows are spaced
 * `*win_step` frames apart (>= 1; the window spans
 * [win_start, win_start + win_frames*win_step) in absolute frames), so
 * long visible ranges stay fully covered by decimating the frame rate.
 * Returns NULL when no valid window covers `frame` yet. Pair every
 * successful call with `wc_unlock`. While locked, the window cannot be
 * evicted. The cache supports one concurrent reader at a time (the UI
 * thread); locks must not be nested.
 */
const float *wc_lock(WaterfallCache *wc, size_t frame,
                     size_t *win_start, size_t *win_frames, size_t *win_step);
void wc_unlock(WaterfallCache *wc);

/*
 * Sample hop between consecutive frames (the number of samples one
 * spectrogram row advances the stream by).
 */
size_t wc_hop_samples(const WaterfallCache *wc);

/*
 * Compute ONE frame synchronously into `db_out` (fft_size floats, dB).
 * Used by the streaming renderer to append new pixel rows on the fly,
 * independent of the cached windows. Returns false when `frame` lies
 * beyond the recording or the cache has no plan. Safe to call from the
 * UI thread; it never blocks on the worker's cached windows.
 */
bool wc_row(WaterfallCache *wc, size_t frame, float *db_out);

/*
 * Current loading progress in percent (0..100), or -1 when idle.
 */
int wc_progress(WaterfallCache *wc);

#endif /* TE15_WATERFALL_CACHE_H */
