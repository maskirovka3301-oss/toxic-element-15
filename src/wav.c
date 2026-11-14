/*
 * wav.c — minimal mono 16-bit PCM WAV writer (see header).
 */
#include "wav.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    uint8_t  riff[4];   /* "RIFF"            */
    uint32_t chunk_size;
    uint8_t  wave[4];   /* "WAVE"            */
    uint8_t  fmt[4];    /* "fmt "            */
    uint32_t fmt_size;  /* 16                */
    uint16_t audio_fmt; /* 1 = PCM           */
    uint16_t channels;  /* 1                 */
    uint32_t sample_rate;
    uint32_t byte_rate; /* rate * block      */
    uint16_t block_align;
    uint16_t bits;      /* 16                */
    uint8_t  data[4];   /* "data"            */
    uint32_t data_size;
} WavHeader;

static float clampf(float v, float lo, float hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

bool wav_write(const char *path, const float *mono, size_t n, int sample_rate)
{
    if (path == NULL || (mono == NULL && n > 0) || sample_rate <= 0) {
        return false;
    }

    FILE *f = fopen(path, "wb");
    if (f == NULL) return false;

    WavHeader h;
    memset(&h, 0, sizeof(h));
    memcpy(h.riff, "RIFF", 4);
    memcpy(h.wave, "WAVE", 4);
    memcpy(h.fmt, "fmt ", 4);
    h.fmt_size  = 16;
    h.audio_fmt = 1;
    h.channels  = 1;
    h.sample_rate = (uint32_t)sample_rate;
    h.block_align = 2; /* channels * bytes per sample */
    h.byte_rate   = (uint32_t)sample_rate * h.block_align;
    h.bits        = 16;
    memcpy(h.data, "data", 4);
    h.data_size  = (uint32_t)(n * sizeof(int16_t));
    h.chunk_size = 36 + h.data_size;

    bool ok = fwrite(&h, sizeof(h), 1, f) == 1;
    if (ok && n > 0) {
        for (size_t i = 0; i < n && ok; i++) {
            float s = clampf(mono[i], -1.0f, 1.0f);
            int16_t pcm = (int16_t)(s * 32767.0f);
            ok = fwrite(&pcm, sizeof(pcm), 1, f) == 1;
        }
    }
    if (fclose(f) != 0) ok = false;
    return ok;
}
