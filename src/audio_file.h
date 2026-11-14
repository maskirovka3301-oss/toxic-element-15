/*
 * audio_file.h — read audio recordings (WAV/AIFF natively, compressed
 * formats through the platform decoder) into interleaved float samples.
 *
 * The waterfall analyser works on complex IQ, but audio files are a valid
 * input too: their spectrum is shown as a real signal (mono -> I with a
 * zero Q, stereo -> L as I and R as Q), centred at rate/2 so the frequency
 * axis covers the audio band 0..rate.
 */
#ifndef TE15_AUDIO_FILE_H
#define TE15_AUDIO_FILE_H

#include <stdbool.h>
#include <stddef.h>

typedef struct {
    double sample_rate; /* from the container header                    */
    int    channels;    /* 1 = mono, 2 = stereo                        */
    int    bits;        /* source bit depth (8/16/24/32/64)            */
    bool   is_float;    /* source samples are IEEE float               */
    bool   is_compressed;/* decoded through the platform (AAC/MP3/...)  */
    char   codec[16];   /* "PCM", "AAC", "MP3", ...                    */
    size_t frames;      /* frames per channel                          */
} AudioInfo;

/*
 * Probe `path`: returns true when the file is an audio recording we can
 * read and fills `info` (frames may be 0 when the size is unknown).
 * `force_bits` (0 = use the header) overrides the sample type for
 * uncompressed formats, mirroring the IQ data-format override.
 */
bool audio_file_probe(const char *path, int force_bits, AudioInfo *info);

/*
 * Decode `path` to interleaved float32 samples (channels * frames floats,
 * values in [-1,1]). Returns a malloc'd buffer in *pcm and the frame count.
 */
bool audio_file_load(const char *path, int force_bits, AudioInfo *info,
                     float **pcm, size_t *frames);

#endif /* TE15_AUDIO_FILE_H */
