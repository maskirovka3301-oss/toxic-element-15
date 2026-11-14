/*
 * audio.c — mono audio playback via CoreAudio AudioQueue (see header).
 *
 * The float samples are converted once to 16-bit PCM and kept in a
 * private buffer. The AudioQueue callback copies from that buffer at
 * the shared read position (atomic), so the UI thread can read the
 * position and seek at any time without locks.
 */
#include "audio.h"

#include <AudioToolbox/AudioToolbox.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUM_BUFFERS   3
#define BUFFER_FRAMES 2048 /* small buffers keep the finish flag timely */

struct AudioPlayer {
    int16_t *pcm;
    size_t   frames;
    double   sample_rate;

    AudioQueueRef        queue;
    AudioQueueBufferRef  buffers[NUM_BUFFERS];
    bool                 queue_ready;

    atomic_size_t pos;      /* next frame to emit, advanced by callback */
    atomic_bool   playing;
    atomic_bool   finished;
};

static float clampf(float v, float lo, float hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static void aq_callback(void *user, AudioQueueRef queue, AudioQueueBufferRef buf)
{
    AudioPlayer *p = user;
    size_t pos = atomic_load_explicit(&p->pos, memory_order_relaxed);
    size_t n = BUFFER_FRAMES;
    if (pos + n > p->frames) n = p->frames - pos;

    if (n > 0) {
        memcpy(buf->mAudioData, p->pcm + pos, n * sizeof(int16_t));
    }
    if (n < BUFFER_FRAMES) {
        /* tail silence so the last partial buffer plays out cleanly */
        memset((int16_t *)buf->mAudioData + n, 0,
               (BUFFER_FRAMES - n) * sizeof(int16_t));
    }
    buf->mAudioDataByteSize = BUFFER_FRAMES * sizeof(int16_t);
    atomic_store_explicit(&p->pos, pos + n, memory_order_relaxed);

    if (pos + n >= p->frames) {
        atomic_store(&p->finished, true);
        atomic_store(&p->playing, false);
        AudioQueueStop(queue, false); /* let the last buffer drain */
        return;                        /* do not re-enqueue */
    }
    AudioQueueEnqueueBuffer(queue, buf, 0, NULL);
}

/* Fill and enqueue every buffer from the current position. */
static void prime_all(AudioPlayer *p)
{
    for (int i = 0; i < NUM_BUFFERS; i++) {
        aq_callback(p, p->queue, p->buffers[i]);
    }
}

AudioPlayer *audio_player_create(const float *mono, size_t n, int sample_rate)
{
    if ((mono == NULL && n > 0) || sample_rate <= 0) return NULL;

    AudioPlayer *p = calloc(1, sizeof(*p));
    if (p == NULL) return NULL;
    p->sample_rate = (double)sample_rate;
    p->frames = n;

    if (n > 0) {
        p->pcm = malloc(n * sizeof(int16_t));
        if (p->pcm == NULL) {
            free(p);
            return NULL;
        }
        for (size_t i = 0; i < n; i++) {
            float s = clampf(mono[i], -1.0f, 1.0f);
            p->pcm[i] = (int16_t)(s * 32767.0f);
        }
    }
    return p;
}

void audio_player_destroy(AudioPlayer *p)
{
    if (p == NULL) return;
    if (p->queue_ready) {
        AudioQueueStop(p->queue, true);
        AudioQueueDispose(p->queue, true); /* frees its buffers too */
    }
    free(p->pcm);
    free(p);
}

static bool ensure_queue(AudioPlayer *p)
{
    if (p->queue_ready) return true;

    AudioStreamBasicDescription fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.mSampleRate       = p->sample_rate;
    fmt.mFormatID         = kAudioFormatLinearPCM;
    fmt.mFormatFlags      = kLinearPCMFormatFlagIsSignedInteger |
                            kLinearPCMFormatFlagIsPacked;
    fmt.mBitsPerChannel   = 16;
    fmt.mChannelsPerFrame = 1;
    fmt.mFramesPerPacket  = 1;
    fmt.mBytesPerFrame    = 2;
    fmt.mBytesPerPacket   = 2;

    OSStatus st = AudioQueueNewOutput(&fmt, aq_callback, p,
                                      NULL, NULL, 0, &p->queue);
    if (st != noErr) {
        fprintf(stderr, "audio: cannot open output queue (%d)\n", (int)st);
        return false;
    }
    for (int i = 0; i < NUM_BUFFERS; i++) {
        st = AudioQueueAllocateBuffer(p->queue, BUFFER_FRAMES * sizeof(int16_t),
                                      &p->buffers[i]);
        if (st != noErr) {
            AudioQueueDispose(p->queue, true);
            memset(p->buffers, 0, sizeof(p->buffers));
            fprintf(stderr, "audio: cannot allocate queue buffer (%d)\n",
                    (int)st);
            return false;
        }
    }
    p->queue_ready = true;
    return true;
}

bool audio_player_play(AudioPlayer *p)
{
    if (p == NULL || p->frames == 0) return false;
    if (!ensure_queue(p)) return false;

    if (atomic_load(&p->finished)) {
        atomic_store(&p->pos, 0);
        atomic_store(&p->finished, false);
    }
    AudioQueueStop(p->queue, true); /* synchronous; buffers return */
    prime_all(p);
    OSStatus st = AudioQueueStart(p->queue, NULL);
    if (st != noErr) {
        fprintf(stderr, "audio: cannot start playback (%d)\n", (int)st);
        return false;
    }
    atomic_store(&p->playing, true);
    return true;
}

void audio_player_pause(AudioPlayer *p)
{
    if (p == NULL) return;
    if (p->queue_ready && atomic_load(&p->playing)) {
        AudioQueuePause(p->queue);
    }
    atomic_store(&p->playing, false);
}

void audio_player_seek(AudioPlayer *p, double seconds)
{
    if (p == NULL) return;
    double dur = audio_player_duration(p);
    if (seconds < 0.0) seconds = 0.0;
    if (seconds > dur) seconds = dur;
    size_t target = (size_t)(seconds * p->sample_rate);
    if (target > p->frames) target = p->frames;

    bool was_playing = atomic_load(&p->playing);
    if (p->queue_ready && was_playing) {
        AudioQueueStop(p->queue, true);
    }
    atomic_store(&p->pos, target);
    atomic_store(&p->finished, target >= p->frames);
    if (was_playing && target < p->frames) {
        prime_all(p);
        AudioQueueStart(p->queue, NULL);
    }
}

double audio_player_position(AudioPlayer *p)
{
    if (p == NULL || p->sample_rate <= 0.0) return 0.0;
    return (double)atomic_load(&p->pos) / p->sample_rate;
}

double audio_player_duration(AudioPlayer *p)
{
    if (p == NULL || p->sample_rate <= 0.0) return 0.0;
    return (double)p->frames / p->sample_rate;
}

bool audio_player_is_playing(AudioPlayer *p)
{
    return p != NULL && atomic_load(&p->playing);
}

bool audio_player_finished(AudioPlayer *p)
{
    return p != NULL && atomic_load(&p->finished);
}
