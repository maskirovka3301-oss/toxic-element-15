/*
 * streambuf.h — forward/backward streaming buffers for the waterfall.
 *
 * These are the only buffers that compute spectrogram rows from the file.
 * The waterfall renderer does not read them directly: it mirrors the render
 * buffer (renderbuf.h), which is filled from these buffers. Two buffers
 * surround the playback position (the last rendered row):
 *
 *   - the FORWARD buffer always strives to be full: it streams rows after
 *     the playback position and, once the end of the file is reached, it
 *     continues from the beginning until it is full again;
 *   - the BACKWARD buffer streams the rows before the playback position; it
 *     is left alone while the playback position is within one buffer size of
 *     the beginning of the file (there is nothing before the start).
 *
 * The buffer size is expressed as a fraction of the sample rate (1/4, 1/6,
 * 1/8, 1/10 or 1/12 of a second worth of samples), i.e. the buffer holds
 * sample_rate * fraction samples, which is exactly
 * frames_per_second * fraction spectrogram rows.
 *
 * Rows outside both buffers (a full redraw of a long view, a jump) are
 * computed on demand by the same buffer module, which is the only place
 * that reads the file.
 */
#ifndef TE15_STREAMBUF_H
#define TE15_STREAMBUF_H

#include <stdbool.h>
#include <stddef.h>

#include "waterfall_cache.h"

typedef struct StreamBuf StreamBuf;

/* Create the buffers for a cache, sized fraction * sample_rate samples each
 * (clamped to a sane minimum and to the memory budget). The forward buffer
 * is filled immediately. */
StreamBuf *sb_create(WaterfallCache *wc, double fraction);
void sb_destroy(StreamBuf *sb);

/* Buffer size as a fraction of the sample rate (e.g. 0.25 for 1/4). */
void   sb_set_fraction(StreamBuf *sb, double fraction);
double sb_fraction(const StreamBuf *sb);

/* Rows the buffers hold on each side (frames_per_second * fraction). */
size_t sb_rows(const StreamBuf *sb);
/* Samples the buffers hold on each side (sample_rate * fraction). */
size_t sb_samples(const StreamBuf *sb);

/* Make `frame` (absolute stream frame of the last rendered row) the centre
 * of the buffers; the worker refills them in the background. */
void sb_anchor(StreamBuf *sb, size_t frame);

/* Number of rows currently present (0 while the first fill is running). */
size_t sb_ready_rows(const StreamBuf *sb);

/* Wait until at least one row is available (used at startup, so the first
 * paint comes from filled buffers). Returns false on timeout. */
bool sb_wait_ready(StreamBuf *sb, int timeout_ms);

/* Copy the row for absolute `frame` (fft_size dB values) into `out`.
 * Rows inside the buffers are served from them; rows outside are computed
 * on demand (still through this module, never by the caller). */
bool sb_row(StreamBuf *sb, size_t frame, float *db_out);

#endif /* TE15_STREAMBUF_H */
