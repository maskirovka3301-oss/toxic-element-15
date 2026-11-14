/*
 * waterfall_cache.c — background spectrogram computation with an LRU
 * cache of contiguous frame windows (see header).
 *
 * Locking discipline:
 *   - `mutex` (cache-wide) is always acquired BEFORE a slot mutex and is
 *     never acquired while a slot mutex is held.
 *   - A slot's data may only be freed while its mutex is held, and
 *     readers hold the slot mutex for the whole time they use the data,
 *     so a locked slot can never be evicted mid-read.
 *   - One concurrent reader is supported (the UI thread).
 */
#include "waterfall_cache.h"

#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "dsp.h"
#include "debug.h"

#define PROGRESS_BATCH 50      /* update progress every N frames */
#define MIN_WINDOW_FRAMES 100  /* never compute fewer frames than this */
#define WAIT_RETRY_MS 100      /* poll interval while slots are busy */

typedef struct {
    size_t      start_frame; /* absolute index of the first frame  */
    size_t      num_frames;  /* frames currently stored            */
    size_t      frame_step;  /* spacing between stored frames      */
    size_t      capacity;    /* allocated frames                   */
    float      *db;          /* capacity * fft_size dB values      */
    bool        valid;
    uint64_t    last_access; /* LRU tick                           */
    pthread_mutex_t lock;
} WcWindow;

struct WaterfallCache {
    const IqSource *src;
    WcSpec          spec;
    size_t          total_frames;
    double          fps;
    size_t          max_window_frames;

    WcWindow       *windows;
    size_t          num_windows;

    pthread_t       thread;
    pthread_mutex_t mutex;
    pthread_cond_t  cond;

    bool     quit;
    bool     req_pending;
    size_t   req_frame;
    size_t   req_want;

    bool     busy;
    int      progress;
    uint64_t tick;

    int      locked_slot; /* index of the slot held by the reader, or -1 */

    /* single-row computation for the streaming renderer (its own plan, so
     * it never contends with the window worker) */
    FftPlan         *row_plan;
    float           *row_window;
    float           *row_iq;
    pthread_mutex_t  row_lock;
};

/* ------------------------------------------------------------------ */
/* internal helpers                                                    */
/* ------------------------------------------------------------------ */

static size_t clamp_size(size_t v, size_t lo, size_t hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

/* Caller holds `mutex`. */
static WcWindow *window_covering_locked(WaterfallCache *wc, size_t frame)
{
    for (size_t i = 0; i < wc->num_windows; i++) {
        WcWindow *w = &wc->windows[i];
        if (w->valid && frame >= w->start_frame &&
            frame < w->start_frame + w->num_frames * w->frame_step) {
            return w;
        }
    }
    return NULL;
}

/* Caller holds `mutex`. Picks a free slot: prefer an invalid one, else
 * the least-recently-used slot that is not currently locked by the
 * reader. Returns NULL when every slot is busy. */
static WcWindow *pick_slot_locked(WaterfallCache *wc)
{
    WcWindow *candidate = NULL;
    uint64_t  oldest = UINT64_MAX;

    for (size_t i = 0; i < wc->num_windows; i++) {
        WcWindow *w = &wc->windows[i];
        if (!w->valid) return w;
        if (pthread_mutex_trylock(&w->lock) == 0) {
            pthread_mutex_unlock(&w->lock); /* probe only */
            if (w->last_access < oldest) {
                oldest = w->last_access;
                candidate = w;
            }
        }
    }
    return candidate;
}

static void free_window_data(WcWindow *w)
{
    /* Caller holds w->lock. */
    free(w->db);
    w->db = NULL;
    w->capacity = 0;
    w->num_frames = 0;
    w->frame_step = 1;
    w->valid = false;
}

static void set_progress(WaterfallCache *wc, int pct)
{
    pthread_mutex_lock(&wc->mutex);
    wc->progress = pct;
    pthread_mutex_unlock(&wc->mutex);
}

/* ------------------------------------------------------------------ */
/* worker                                                              */
/* ------------------------------------------------------------------ */

static void *wc_worker(void *arg)
{
    WaterfallCache *wc = arg;
    const size_t n = (size_t)wc->spec.fft_size;
    size_t hop = (size_t)(n * (1.0 - wc->spec.overlap) + 0.5);
    if (hop < 1) hop = 1;

    FftPlan *plan = fft_plan_create((int)n);
    float   *window = malloc(n * sizeof(float));
    if (plan == NULL || window == NULL) {
        fft_plan_destroy(plan);
        free(window);
        fprintf(stderr, "waterfall_cache: cannot allocate FFT plan\n");
        return NULL;
    }
    dsp_window(wc->spec.window_type, window, n);

    for (;;) {
        pthread_mutex_lock(&wc->mutex);
        while (!wc->req_pending && !wc->quit) {
            pthread_cond_wait(&wc->cond, &wc->mutex);
        }
        if (wc->quit && !wc->req_pending) {
            pthread_mutex_unlock(&wc->mutex);
            break;
        }
        size_t frame = wc->req_frame;
        size_t want  = wc->req_want;
        wc->req_pending = false;
        pthread_mutex_unlock(&wc->mutex);

        /* Skip if the frame is already covered by a valid window. */
        pthread_mutex_lock(&wc->mutex);
        if (window_covering_locked(wc, frame) == NULL) {
            WcWindow *slot = pick_slot_locked(wc);
            if (slot == NULL) {
                /* All slots busy (being read). Retry shortly. */
                wc->req_pending = true;
                pthread_mutex_unlock(&wc->mutex);
                usleep(WAIT_RETRY_MS * 1000);
                continue;
            }
            wc->busy = true;
            wc->progress = 0;

            /* The window must span the requested range: store at most
             * max_window_frames rows, decimated by frame_step so the
             * whole span stays covered (long waterfall lengths). */
            size_t span = clamp_size(want, MIN_WINDOW_FRAMES,
                                     wc->total_frames);
            size_t num = clamp_size(span, 1, wc->max_window_frames);
            size_t step = (span + num - 1) / num; /* ceil */
            /* Bias the window so 1/3 lies before the target frame,
             * aligned to the stride. */
            size_t lead = span / 3;
            size_t start = (frame > lead) ? frame - lead : 0;
            start = (start / step) * step;
            size_t end = start + num * step;
            if (end > wc->total_frames) {
                /* shift back so the window still covers `frame` */
                size_t excess = end - wc->total_frames;
                size_t shift = (excess + step - 1) / step * step;
                start = (start > shift) ? start - shift : 0;
                end = start + num * step;
                if (end > wc->total_frames) end = wc->total_frames;
            }
            size_t frames = num;
            if (frames < 1) frames = 1;

            /* Allocate the new buffer before touching the slot. */
            float *db = malloc(frames * n * sizeof(float));
            if (db == NULL) {
                wc->busy = false;
                wc->progress = -1;
                pthread_mutex_unlock(&wc->mutex);
                fprintf(stderr, "waterfall_cache: out of memory (%zu frames)\n",
                        frames);
                continue;
            }

            pthread_mutex_lock(&slot->lock);
            free_window_data(slot);
            slot->start_frame = start;
            slot->num_frames  = 0;
            slot->frame_step  = step;
            slot->capacity    = frames;
            slot->db          = db;
            pthread_mutex_unlock(&slot->lock);
            pthread_mutex_unlock(&wc->mutex);

            /* Compute (no locks held); samples are converted on the fly
             * through the source accessor so non-float formats work. */
            float *frame_iq = malloc(2 * n * sizeof(float));
            if (frame_iq == NULL) {
                fprintf(stderr, "waterfall_cache: out of memory\n");
                break;
            }
            size_t computed = 0;
            for (size_t i = 0; i < frames; i++) {
                size_t sample_start = (start + i * step) * hop;
                if (sample_start + n > wc->src->total_samples) break;
                iq_source_read_complex(wc->src, sample_start, frame_iq, n);
                fft_plan_spectrum(plan, frame_iq, window,
                                  db + i * n);
                computed++;
                if (computed % PROGRESS_BATCH == 0) {
                    set_progress(wc, (int)(computed * 100 / frames));
                }
            }
            free(frame_iq);

            /* Publish under both locks (main -> slot order, which is
             * safe because nobody acquires `mutex` while holding a
             * slot lock). */
            pthread_mutex_lock(&wc->mutex);
            pthread_mutex_lock(&slot->lock);
            slot->num_frames = computed;
            slot->valid = computed > 0;
            slot->last_access = ++wc->tick;
            pthread_mutex_unlock(&slot->lock);
            wc->busy = false;
            wc->progress = -1;
            pthread_mutex_unlock(&wc->mutex);

            TE15_LOG("waterfall_cache: window frames %zu..%zu loaded (%zu frames)\n",
                   start, start + computed, computed);
        } else {
            pthread_mutex_unlock(&wc->mutex);
        }
    }

    fft_plan_destroy(plan);
    free(window);
    return NULL;
}

/* ------------------------------------------------------------------ */
/* public API                                                          */
/* ------------------------------------------------------------------ */

WaterfallCache *wc_create(const IqSource *src, const WcSpec *spec,
                          size_t max_windows, size_t max_window_frames)
{
    if (src == NULL || spec == NULL || spec->fft_size < 2 ||
        max_windows < 1 || max_window_frames < 1) {
        return NULL;
    }

    WaterfallCache *wc = calloc(1, sizeof(*wc));
    if (wc == NULL) return NULL;
    wc->src = src;
    wc->spec = *spec;
    wc->max_window_frames = max_window_frames;
    wc->num_windows = max_windows;
    wc->locked_slot = -1;
    wc->progress = -1;

    const size_t n = (size_t)spec->fft_size;
    size_t hop = (size_t)(n * (1.0 - spec->overlap) + 0.5);
    if (hop < 1) hop = 1;

    wc->fps = src->sample_rate / (double)hop;
    wc->total_frames =
        (src->total_samples > n) ? (src->total_samples - n) / hop + 1 : 0;

    if (wc->total_frames == 0) {
        fprintf(stderr, "waterfall_cache: recording too short for FFT %d\n",
                spec->fft_size);
        free(wc);
        return NULL;
    }

    wc->windows = calloc(max_windows, sizeof(WcWindow));
    if (wc->windows == NULL) {
        free(wc);
        return NULL;
    }
    for (size_t i = 0; i < max_windows; i++) {
        pthread_mutex_init(&wc->windows[i].lock, NULL);
    }

    pthread_mutex_init(&wc->mutex, NULL);
    pthread_cond_init(&wc->cond, NULL);
    pthread_mutex_init(&wc->row_lock, NULL);
    if (pthread_create(&wc->thread, NULL, wc_worker, wc) != 0) {
        fprintf(stderr, "waterfall_cache: cannot start worker thread\n");
        pthread_cond_destroy(&wc->cond);
        pthread_mutex_destroy(&wc->mutex);
        for (size_t i = 0; i < max_windows; i++) {
            pthread_mutex_destroy(&wc->windows[i].lock);
        }
        free(wc->windows);
        free(wc);
        return NULL;
    }
    return wc;
}

void wc_destroy(WaterfallCache *wc)
{
    if (wc == NULL) return;

    pthread_mutex_lock(&wc->mutex);
    wc->quit = true;
    pthread_cond_signal(&wc->cond);
    pthread_mutex_unlock(&wc->mutex);
    pthread_join(wc->thread, NULL);

    for (size_t i = 0; i < wc->num_windows; i++) {
        WcWindow *w = &wc->windows[i];
        pthread_mutex_lock(&w->lock);
        free_window_data(w);
        pthread_mutex_unlock(&w->lock);
        pthread_mutex_destroy(&w->lock);
    }
    free(wc->windows);
    pthread_mutex_lock(&wc->row_lock);
    fft_plan_destroy(wc->row_plan);
    free(wc->row_window);
    free(wc->row_iq);
    pthread_mutex_unlock(&wc->row_lock);
    pthread_mutex_destroy(&wc->row_lock);
    pthread_cond_destroy(&wc->cond);
    pthread_mutex_destroy(&wc->mutex);
    free(wc);
}

size_t wc_total_frames(const WaterfallCache *wc)
{
    return wc != NULL ? wc->total_frames : 0;
}

double wc_frames_per_second(const WaterfallCache *wc)
{
    return wc != NULL ? wc->fps : 0.0;
}

int wc_fft_size(const WaterfallCache *wc)
{
    return wc != NULL ? wc->spec.fft_size : 0;
}

void wc_request(WaterfallCache *wc, size_t frame, size_t want_frames)
{
    if (wc == NULL || wc->total_frames == 0) return;
    pthread_mutex_lock(&wc->mutex);
    frame = clamp_size(frame, 0, wc->total_frames - 1);
    wc->req_frame = frame;
    wc->req_want  = want_frames;
    wc->req_pending = true;
    pthread_cond_signal(&wc->cond);
    pthread_mutex_unlock(&wc->mutex);
}

const float *wc_lock(WaterfallCache *wc, size_t frame,
                     size_t *win_start, size_t *win_frames, size_t *win_step)
{
    if (wc == NULL || wc->total_frames == 0) return NULL;

    pthread_mutex_lock(&wc->mutex);
    WcWindow *w = window_covering_locked(wc, frame);
    if (w == NULL) {
        pthread_mutex_unlock(&wc->mutex);
        return NULL;
    }
    pthread_mutex_lock(&w->lock); /* held for the whole read */
    wc->locked_slot = (int)(w - wc->windows);
    if (win_start != NULL) *win_start = w->start_frame;
    if (win_frames != NULL) *win_frames = w->num_frames;
    if (win_step != NULL) *win_step = w->frame_step;
    const float *db = w->db;
    pthread_mutex_unlock(&wc->mutex);
    return db;
}

void wc_unlock(WaterfallCache *wc)
{
    if (wc == NULL) return;
    /*
     * Single-reader discipline: `locked_slot` is only ever written by
     * the (single) reader thread, so it can be read here without the
     * cache mutex. Taking `mutex` here while holding a slot lock would
     * deadlock against the worker, which holds `mutex` while waiting
     * for a busy slot.
     */
    int slot = wc->locked_slot;
    wc->locked_slot = -1;
    if (slot >= 0 && (size_t)slot < wc->num_windows) {
        pthread_mutex_unlock(&wc->windows[slot].lock);
    }
}

int wc_progress(WaterfallCache *wc)
{
    if (wc == NULL) return -1;
    pthread_mutex_lock(&wc->mutex);
    int p = wc->busy ? wc->progress : -1;
    pthread_mutex_unlock(&wc->mutex);
    return p;
}

size_t wc_hop_samples(const WaterfallCache *wc)
{
    size_t n, hop;

    if (wc == NULL) return 1;
    n = (size_t)wc->spec.fft_size;
    hop = (size_t)(n * (1.0 - wc->spec.overlap) + 0.5);
    return hop < 1 ? 1 : hop;
}

bool wc_row(WaterfallCache *wc, size_t frame, float *db_out)
{
    size_t n;
    size_t hop;
    bool ok = false;

    if (wc == NULL || db_out == NULL) return false;
    n = (size_t)wc->spec.fft_size;
    hop = wc_hop_samples(wc);
    if (frame >= wc->total_frames) return false;

    pthread_mutex_lock(&wc->row_lock);
    if (wc->row_plan == NULL) {
        wc->row_plan = fft_plan_create((int)n);
        wc->row_window = malloc(n * sizeof(float));
        wc->row_iq = malloc(2 * n * sizeof(float));
        if (wc->row_plan == NULL || wc->row_window == NULL ||
            wc->row_iq == NULL) {
            fft_plan_destroy(wc->row_plan);
            free(wc->row_window);
            free(wc->row_iq);
            wc->row_plan = NULL;
            wc->row_window = NULL;
            wc->row_iq = NULL;
            pthread_mutex_unlock(&wc->row_lock);
            return false;
        }
        dsp_window(wc->spec.window_type, wc->row_window, n);
    }
    iq_source_read_complex(wc->src, frame * hop, wc->row_iq, n);
    fft_plan_spectrum(wc->row_plan, wc->row_iq, wc->row_window, db_out);
    ok = true;
    pthread_mutex_unlock(&wc->row_lock);
    return ok;
}
