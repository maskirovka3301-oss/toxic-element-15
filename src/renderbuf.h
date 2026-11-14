/*
 * renderbuf.h — render buffer between the streaming buffers and the
 * waterfall renderer.
 *
 * The waterfall renderer must not talk to the forward/backward streaming
 * buffers directly: it mirrors this render buffer instead. The render
 * buffer is filled from the streaming buffers (which are the only place
 * that computes spectrogram rows from the file), and it can mirror the
 * COMPLETE range of rows a redraw needs before any painting happens. That
 * is what makes an FFT-size change work: a new cache means a new streaming
 * buffer and a new render buffer, and the whole surface is then computed
 * from the complete render buffer and copied to the waterfall surface.
 *
 * Row storage is a sliding ring of dB rows plus a per-row presence bitmap,
 * so moving the visible window keeps the overlapping rows (incremental
 * scrolling) and never serves a row that was not filled.
 */
#ifndef TE15_RENDERBUF_H
#define TE15_RENDERBUF_H

#include <stdbool.h>
#include <stddef.h>

#include "streambuf.h"
#include "waterfall_cache.h"

typedef struct RenderBuf RenderBuf;

/* Create a render buffer mirroring `sb` for a cache (the FFT size of the
 * cache defines the row length). */
RenderBuf *rb_create(WaterfallCache *wc, StreamBuf *sb);
void rb_destroy(RenderBuf *rb);

/* Point the render buffer at a new streaming buffer (after the cache or
 * the streaming buffers were rebuilt). */
void rb_set_stream(RenderBuf *rb, StreamBuf *sb);

/* dB bins per row (the cache's FFT size). */
int rb_fft_size(const RenderBuf *rb);

/* Rows in the mirrored window and the frame of its first row. */
size_t rb_first(const RenderBuf *rb);
size_t rb_count(const RenderBuf *rb);

/* Mirror rows [first, first + rows) from the streaming buffers. Rows that
 * are already mirrored are kept; the window slides when needed. */
void rb_fill(RenderBuf *rb, size_t first, size_t rows);

/* Copy the mirrored row for absolute `frame` into `out` (fft_size floats).
 * Rows outside the mirrored window are fetched through the streaming
 * buffers, which remains the render buffer's only data source. */
bool rb_row(RenderBuf *rb, size_t frame, float *db_out);

#endif /* TE15_RENDERBUF_H */
