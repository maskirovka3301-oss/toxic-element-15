/*
 * renderbuf.c — see renderbuf.h.
 */
#include "renderbuf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "debug.h"

/* Memory budget for the mirrored rows (the ring is clamped to it, and rows
 * beyond the window fall back to the streaming buffers). */
#define RB_BUDGET_BYTES (32u * 1024u * 1024u)
#define RB_MIN_ROWS 64u

struct RenderBuf {
    WaterfallCache *wc;
    StreamBuf      *sb;
    int             fft_n;
    float          *rows;       /* cap * fft_n dB values                    */
    unsigned char  *present;    /* cap flags                               */
    size_t          cap;        /* ring capacity in rows                   */
    size_t          start;      /* first absolute frame mirrored           */
    size_t          count;      /* number of mirrored rows                 */
};

static bool rb_grow(RenderBuf *rb, size_t rows)
{
    size_t cap = rb->cap;
    size_t limit;

    if (rows <= cap) return true;
    limit = RB_BUDGET_BYTES / (size_t)(rb->fft_n > 0 ? rb->fft_n : 1) /
            sizeof(float);
    if (limit < RB_MIN_ROWS) limit = RB_MIN_ROWS;
    while (cap < rows && cap < limit) cap <<= 1;
    if (cap < rows) cap = rows;
    if (cap > limit) cap = limit;
    if (cap < rows) return false;   /* the caller falls back per row */
    {
        float *rows_new = malloc(cap * (size_t)rb->fft_n * sizeof(float));
        unsigned char *present_new = calloc(cap, 1);

        if (rows_new == NULL || present_new == NULL) {
            free(rows_new);
            free(present_new);
            return false;
        }
        /* re-index the existing window into the new ring */
        for (size_t i = 0; i < rb->count; i++) {
            size_t frame = rb->start + i;
            size_t old_idx = frame % rb->cap;
            size_t new_idx = frame % cap;

            if (!rb->present[old_idx]) continue;
            memcpy(rows_new + new_idx * (size_t)rb->fft_n,
                   rb->rows + old_idx * (size_t)rb->fft_n,
                   (size_t)rb->fft_n * sizeof(float));
            present_new[new_idx] = 1;
        }
        free(rb->rows);
        free(rb->present);
        rb->rows = rows_new;
        rb->present = present_new;
        rb->cap = cap;
    }
    return true;
}

RenderBuf *rb_create(WaterfallCache *wc, StreamBuf *sb)
{
    RenderBuf *rb;

    if (wc == NULL) return NULL;
    rb = calloc(1, sizeof(*rb));
    if (rb == NULL) return NULL;
    rb->wc = wc;
    rb->sb = sb;
    rb->fft_n = wc_fft_size(wc);
    rb->cap = RB_MIN_ROWS;
    rb->rows = malloc(rb->cap * (size_t)rb->fft_n * sizeof(float));
    rb->present = calloc(rb->cap, 1);
    if (rb->rows == NULL || rb->present == NULL) {
        rb_destroy(rb);
        return NULL;
    }
    return rb;
}

void rb_destroy(RenderBuf *rb)
{
    if (rb == NULL) return;
    free(rb->rows);
    free(rb->present);
    free(rb);
}

void rb_set_stream(RenderBuf *rb, StreamBuf *sb)
{
    if (rb == NULL) return;
    rb->sb = sb;
    /* the cache may have changed: reset the mirror */
    rb->fft_n = rb->wc != NULL ? wc_fft_size(rb->wc) : 0;
    rb->start = 0;
    rb->count = 0;
    memset(rb->present, 0, rb->cap);
}

int rb_fft_size(const RenderBuf *rb)
{
    return rb != NULL ? rb->fft_n : 0;
}

size_t rb_first(const RenderBuf *rb)
{
    return rb != NULL ? rb->start : 0;
}

size_t rb_count(const RenderBuf *rb)
{
    return rb != NULL ? rb->count : 0;
}

void rb_fill(RenderBuf *rb, size_t first, size_t rows)
{
    size_t end = first + rows;

    if (rb == NULL || rb->fft_n <= 0 || rb->sb == NULL || rows == 0) return;
    if (!rb_grow(rb, rows)) {
        /* too large for the budget: keep a window of what fits and let
         * rb_row() fall back for the rest */
        rows = rb->cap;
        end = first + rows;
    }
    /* Drop rows that fall outside the new window before sliding. */
    for (size_t i = 0; i < rb->count; i++) {
        size_t frame = rb->start + i;

        if (frame < first || frame >= end) {
            rb->present[frame % rb->cap] = 0;
        }
    }
    for (size_t frame = first; frame < end; frame++) {
        size_t idx = frame % rb->cap;

        if (rb->present[idx]) continue;
        if (!sb_row(rb->sb, frame, rb->rows + idx * (size_t)rb->fft_n)) {
            continue;
        }
        rb->present[idx] = 1;
    }
    rb->start = first;
    rb->count = rows;
}

bool rb_row(RenderBuf *rb, size_t frame, float *db_out)
{
    if (rb == NULL || db_out == NULL || rb->fft_n <= 0) return false;
    if (rb->count > 0 && frame >= rb->start && frame < rb->start + rb->count) {
        size_t idx = frame % rb->cap;

        if (rb->present[idx]) {
            memcpy(db_out, rb->rows + idx * (size_t)rb->fft_n,
                   (size_t)rb->fft_n * sizeof(float));
            return true;
        }
    }
    if (rb->sb == NULL) return false;
    return sb_row(rb->sb, frame, db_out);
}
