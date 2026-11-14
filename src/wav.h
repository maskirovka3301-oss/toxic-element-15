/*
 * wav.h — minimal mono 16-bit PCM WAV writer.
 */
#ifndef TE15_WAV_H
#define TE15_WAV_H

#include <stdbool.h>
#include <stddef.h>

/*
 * Write `n` mono float samples (values outside [-1,1] are clamped) as a
 * 16-bit PCM WAV file at `sample_rate` Hz. Returns true on success.
 */
bool wav_write(const char *path, const float *mono, size_t n, int sample_rate);

#endif /* TE15_WAV_H */
