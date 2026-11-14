/*
 * streambuf.c — forward/backward streaming buffers (see streambuf.h).
 */
#include "streambuf.h"

#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define SB_MIN_ROWS   8
#define SB_MAX_ROWS   (1024u * 1024u)                  /* hard row cap       */
#define SB_MEM_LIMIT  (256u * 1024u * 1024u)           /* bytes for rows     */
#define SB_ROW_WAIT_MS 250                             /* wait for a refill  */

struct StreamBuf {
    WaterfallCache *wc;
    size_t fft_n;
    size_t total_frames;
    double fps;
    double sample_rate;

    double fraction;
    size_t rows;            /* rows each side (fps * fraction)             */

    float  *ring;           /* (2*rows+1) rows of fft_n floats             */
    unsigned char *present; /* (2*rows+1) 1 when the row was computed      */
    int64_t start;          /* absolute frame of ring[0]                   */
    size_t  count;          /* rows in the window                          */
    size_t  ready;          /* rows actually computed (present)            */

    int64_t anchor;         /* playback frame (the last rendered row)      */
    bool    pending;

    pthread_t       thread;
    pthread_mutex_t lock;
    pthread_cond_t  cond;     /* worker wake-up                            */
    pthread_cond_t  filled;   /* readers waiting for a refill              */
    bool            quit;
};

/* ------------------------------------------------------------------ */
/* sizing                                                              */
/* ------------------------------------------------------------------ */

static size_t clamp_rows(const StreamBuf *sb, size_t rows)
{
    size_t mem_rows;

    if (rows < SB_MIN_ROWS) rows = SB_MIN_ROWS;
    if (rows > SB_MAX_ROWS) rows = SB_MAX_ROWS;
    mem_rows = (size_t)(SB_MEM_LIMIT / (sizeof(float) * sb->fft_n));
    if (mem_rows < 2 * SB_MIN_ROWS + 1) mem_rows = 2 * SB_MIN_ROWS + 1;
    if (2 * rows + 1 > mem_rows) rows = (mem_rows - 1) / 2;
    return rows;
}

static size_t rows_from_fraction(const StreamBuf *sb, double fraction)
{
    double rows;

    if (fraction <= 0.0) fraction = 1.0 / 8.0;
    rows = sb->fps * fraction;
    if (rows < (double)SB_MIN_ROWS) rows = (double)SB_MIN_ROWS;
    return clamp_rows(sb, (size_t)llround(rows));
}

/* ------------------------------------------------------------------ */
/* worker                                                              */
/* ------------------------------------------------------------------ */

/* Compute row `frame` (absolute stream frame) into `dst`. */
static void compute_row(StreamBuf *sb, int64_t frame, float *dst)
{
    size_t f;

    if (sb->total_frames == 0) {
        memset(dst, 0, sb->fft_n * sizeof(float));
        return;
    }
    f = (size_t)(frame % (int64_t)sb->total_frames);
    if (!wc_row(sb->wc, f, dst)) memset(dst, 0, sb->fft_n * sizeof(float));
}

/* The window the buffers should hold for the current anchor:
 * [max(0, anchor - rows), anchor + rows] - the backward part simply does not
 * exist before the start of the file, the forward part wraps past the end. */
static void window_for(const StreamBuf *sb, int64_t anchor, int64_t *out_start,
                       int64_t *out_end)
{
    int64_t start = anchor - (int64_t)sb->rows;

    if (start < 0) start = 0;
    *out_start = start;
    *out_end = anchor + (int64_t)sb->rows;
}

static void *sb_worker(void *arg)
{
    StreamBuf *sb = arg;

    for (;;) {
        int64_t anchor, new_start, new_end, old_start, old_end;
        size_t rows;
        float *tmp = NULL;
        size_t missing = 0;

        pthread_mutex_lock(&sb->lock);
        while (!sb->pending && !sb->quit) {
            pthread_cond_wait(&sb->cond, &sb->lock);
        }
        if (sb->quit) {
            pthread_mutex_unlock(&sb->lock);
            break;
        }
        sb->pending = false;
        anchor = sb->anchor;
        rows = sb->rows;
        old_start = sb->start;
        old_end = sb->start + (int64_t)sb->count;
        pthread_mutex_unlock(&sb->lock);

        if (sb->fft_n == 0) continue;
        window_for(sb, anchor, &new_start, &new_end);

        /* count the rows that are not in the current window yet */
        for (int64_t f = new_start; f <= new_end; f++) {
            if (!(sb->count > 0 && f >= old_start && f < old_end)) missing++;
        }
        if (missing > 0) {
            tmp = malloc(missing * sb->fft_n * sizeof(float));
            if (tmp == NULL) continue;
        }
        {
            size_t slot = 0;

            for (int64_t f = new_start; f <= new_end; f++) {
                bool stop;

                if (sb->count > 0 && f >= old_start && f < old_end) continue;
                compute_row(sb, f, tmp + slot * sb->fft_n);
                slot++;
                /* a newer anchor or shutdown makes this fill obsolete */
                pthread_mutex_lock(&sb->lock);
                stop = sb->quit || (sb->anchor != anchor);
                pthread_mutex_unlock(&sb->lock);
                if (stop) break;
            }
        }

        pthread_mutex_lock(&sb->lock);
        if (sb->quit) {
            pthread_mutex_unlock(&sb->lock);
            free(tmp);
            break;
        }
        if (sb->anchor != anchor || sb->rows != rows) {
            /* obsolete (newer anchor or resized): drop and refill */
            pthread_mutex_unlock(&sb->lock);
            free(tmp);
            continue;
        }
        {
            size_t rows_in_window = (size_t)(new_end - new_start + 1);
            size_t ready = 0;
            /* The window has to be moved inside the same ring buffer. A
             * row must be read BEFORE its slot is overwritten, so the
             * overlapping rows are copied in the direction that never
             * clobbers a source that has not been read yet: ascending when
             * the window slides forward (sources are at higher indices),
             * descending when it slides backwards (sources are at lower
             * indices). Getting this wrong silently serves rows from a
             * shifted frame - the "duplicate rows" seen when scrolling up. */
            bool slide_back = new_start < old_start;

            for (size_t k = 0; k < rows_in_window; k++) {
                size_t idx = slide_back ? rows_in_window - 1 - k : k;
                int64_t f = new_start + (int64_t)idx;
                bool have = false;

                if (sb->count > 0 && f >= old_start && f < old_end) {
                    size_t oi = (size_t)(f - old_start);

                    if (sb->present[oi]) {
                        memcpy(sb->ring + idx * sb->fft_n,
                               sb->ring + oi * sb->fft_n,
                               sb->fft_n * sizeof(float));
                        have = true;
                    }
                }
                if (have) {
                    sb->present[idx] = 1;
                    ready++;
                } else {
                    sb->present[idx] = 0;
                }
            }
            /* then fill the rows that were not in the old window from the
             * freshly computed block (their slots are disjoint) */
            {
                size_t slot = 0;

                for (int64_t f = new_start; f <= new_end; f++) {
                    size_t idx = (size_t)(f - new_start);

                    if (sb->count > 0 && f >= old_start && f < old_end) {
                        continue;
                    }
                    if (tmp != NULL && slot < missing) {
                        memcpy(sb->ring + idx * sb->fft_n,
                               tmp + slot * sb->fft_n,
                               sb->fft_n * sizeof(float));
                        sb->present[idx] = 1;
                        ready++;
                        slot++;
                    } else {
                        sb->present[idx] = 0;
                    }
                }
            }
            sb->start = new_start;
            sb->count = rows_in_window;
            sb->ready = ready;
        }
        pthread_cond_broadcast(&sb->filled);
        pthread_mutex_unlock(&sb->lock);
        free(tmp);
    }
    return NULL;
}

/* ------------------------------------------------------------------ */
/* public API                                                          */
/* ------------------------------------------------------------------ */

StreamBuf *sb_create(WaterfallCache *wc, double fraction)
{
    StreamBuf *sb;

    if (wc == NULL) return NULL;
    sb = calloc(1, sizeof(*sb));
    if (sb == NULL) return NULL;
    sb->wc = wc;
    sb->fft_n = (size_t)wc_fft_size(wc);
    sb->total_frames = wc_total_frames(wc);
    sb->fps = wc_frames_per_second(wc);
    sb->sample_rate = sb->fps > 0.0 && wc_hop_samples(wc) > 0
                          ? sb->fps * (double)wc_hop_samples(wc)
                          : 0.0;
    if (sb->fft_n == 0 || sb->total_frames == 0 || sb->fps <= 0.0) {
        free(sb);
        return NULL;
    }
    sb->fraction = (fraction > 0.0) ? fraction : 1.0 / 8.0;
    sb->rows = rows_from_fraction(sb, sb->fraction);
    sb->ring = malloc((2 * sb->rows + 1) * sb->fft_n * sizeof(float));
    sb->present = calloc(2 * sb->rows + 1, 1);
    if (sb->ring == NULL || sb->present == NULL) {
        free(sb->ring);
        free(sb->present);
        free(sb);
        return NULL;
    }
    sb->start = 0;
    sb->count = 0;
    sb->anchor = 0;
    pthread_mutex_init(&sb->lock, NULL);
    pthread_cond_init(&sb->cond, NULL);
    pthread_cond_init(&sb->filled, NULL);
    if (pthread_create(&sb->thread, NULL, sb_worker, sb) != 0) {
        pthread_mutex_destroy(&sb->lock);
        pthread_cond_destroy(&sb->cond);
        pthread_cond_destroy(&sb->filled);
        free(sb->ring);
        free(sb->present);
        free(sb);
        return NULL;
    }
    /* the buffers are filled first: the worker starts on the first window */
    sb_anchor(sb, 0);
    return sb;
}

void sb_destroy(StreamBuf *sb)
{
    if (sb == NULL) return;
    pthread_mutex_lock(&sb->lock);
    sb->quit = true;
    pthread_cond_signal(&sb->cond);
    pthread_cond_broadcast(&sb->filled);
    pthread_mutex_unlock(&sb->lock);
    pthread_join(sb->thread, NULL);
    pthread_cond_destroy(&sb->filled);
    pthread_cond_destroy(&sb->cond);
    pthread_mutex_destroy(&sb->lock);
    free(sb->ring);
    free(sb->present);
    free(sb);
}

void sb_set_fraction(StreamBuf *sb, double fraction)
{
    size_t rows;
    float *ring;
    unsigned char *present;

    if (sb == NULL) return;
    if (fraction <= 0.0) return;
    rows = rows_from_fraction(sb, fraction);
    ring = realloc(sb->ring, (2 * rows + 1) * sb->fft_n * sizeof(float));
    if (ring == NULL) return;
    present = realloc(sb->present, 2 * rows + 1);
    if (present == NULL) return;
    pthread_mutex_lock(&sb->lock);
    sb->ring = ring;
    sb->present = present;
    sb->fraction = fraction;
    sb->rows = rows;
    sb->count = 0;          /* the layout changed: refill */
    sb->ready = 0;
    memset(sb->present, 0, 2 * rows + 1);
    sb->pending = true;
    pthread_cond_signal(&sb->cond);
    pthread_mutex_unlock(&sb->lock);
}

double sb_fraction(const StreamBuf *sb)
{
    return sb != NULL ? sb->fraction : 0.0;
}

size_t sb_rows(const StreamBuf *sb)
{
    return sb != NULL ? sb->rows : 0;
}

size_t sb_samples(const StreamBuf *sb)
{
    if (sb == NULL || sb->fps <= 0.0) return 0;
    return (size_t)(sb->sample_rate * sb->fraction);
}

void sb_anchor(StreamBuf *sb, size_t frame)
{
    if (sb == NULL) return;
    pthread_mutex_lock(&sb->lock);
    sb->anchor = (int64_t)frame;
    sb->pending = true;
    pthread_cond_signal(&sb->cond);
    pthread_mutex_unlock(&sb->lock);
}

size_t sb_ready_rows(const StreamBuf *sb)
{
    size_t n;

    if (sb == NULL) return 0;
    pthread_mutex_lock((pthread_mutex_t *)&sb->lock);
    n = sb->ready;
    pthread_mutex_unlock((pthread_mutex_t *)&sb->lock);
    return n;
}

bool sb_wait_ready(StreamBuf *sb, int timeout_ms)
{
    struct timespec ts;

    if (sb == NULL) return false;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec += timeout_ms / 1000;
    ts.tv_nsec += (long)(timeout_ms % 1000) * 1000000L;
    if (ts.tv_nsec >= 1000000000L) {
        ts.tv_sec++;
        ts.tv_nsec -= 1000000000L;
    }
    pthread_mutex_lock(&sb->lock);
    while (sb->ready == 0 && !sb->quit) {
        if (pthread_cond_timedwait(&sb->filled, &sb->lock, &ts) != 0) break;
    }
    {
        bool ok = sb->ready > 0;

        pthread_mutex_unlock(&sb->lock);
        return ok;
    }
}

bool sb_row(StreamBuf *sb, size_t frame, float *db_out)
{
    bool hit = false;

    if (sb == NULL || db_out == NULL) return false;
    pthread_mutex_lock(&sb->lock);
    if (sb->count > 0 && (int64_t)frame >= sb->start &&
        (int64_t)frame < sb->start + (int64_t)sb->count) {
        size_t idx = (size_t)((int64_t)frame - sb->start);

        if (sb->present[idx]) {
            memcpy(db_out, sb->ring + idx * sb->fft_n,
                   sb->fft_n * sizeof(float));
            hit = true;
        }
    }
    pthread_mutex_unlock(&sb->lock);
    if (hit) return true;

    /* The row is outside the buffers: it may be just outside because the
     * worker is still filling (wait briefly), otherwise compute it on
     * demand - the waterfall never reads the file itself. */
    {
        int64_t start, end;
        int64_t f = (int64_t)frame;

        window_for(sb, sb->anchor, &start, &end);
        if (f >= start && f <= end) {
            struct timespec ts;

            clock_gettime(CLOCK_REALTIME, &ts);
            ts.tv_sec += SB_ROW_WAIT_MS / 1000;
            ts.tv_nsec += (long)(SB_ROW_WAIT_MS % 1000) * 1000000L;
            if (ts.tv_nsec >= 1000000000L) {
                ts.tv_sec++;
                ts.tv_nsec -= 1000000000L;
            }
            pthread_mutex_lock(&sb->lock);
            for (;;) {
                size_t idx;

                if (sb->count > 0 && (int64_t)frame >= sb->start &&
                    (int64_t)frame < sb->start + (int64_t)sb->count) {
                    idx = (size_t)((int64_t)frame - sb->start);
                    if (sb->present[idx]) {
                        memcpy(db_out, sb->ring + idx * sb->fft_n,
                               sb->fft_n * sizeof(float));
                        hit = true;
                        break;
                    }
                }
                if (pthread_cond_timedwait(&sb->filled, &sb->lock, &ts) != 0) {
                    break;
                }
            }
            pthread_mutex_unlock(&sb->lock);
            if (hit) return true;
        }
    }
    /* on-demand row: still computed by the buffer module, not the caller */
    compute_row(sb, (int64_t)frame, db_out);
    return true;
}
