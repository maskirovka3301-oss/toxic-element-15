/*
 * iq_source.h — read-only access to gqrx raw IQ recordings.
 *
 * gqrx I/Q recorder files contain interleaved 32-bit float I/Q samples
 * (8 bytes per complex sample). Metadata (centre frequency, sample rate
 * and the recording start wall-clock) is recovered from the conventional
 * filename:
 *
 *     gqrx_YYYYMMDD_HHMMSS_<freq_hz>_<rate_hz>_fc.raw
 */
#ifndef TE15_IQ_SOURCE_H
#define TE15_IQ_SOURCE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <time.h>

/* IQ sample data formats (bytes per complex sample in brackets). */
typedef enum {
    IQ_FORMAT_FLOAT32 = 0, /* interleaved float32 I/Q (gqrx), 8 bytes */
    IQ_FORMAT_INT16,       /* interleaved int16 I/Q, 4 bytes          */
    IQ_FORMAT_INT8,        /* interleaved int8 I/Q, 2 bytes           */
    IQ_FORMAT_UINT8,       /* interleaved uint8 I/Q, 2 bytes          */
} IqFormat;

/* How the recording's metadata was recovered. */
typedef enum {
    IQ_NAME_GQRX = 0,  /* gqrx_<time>_<freq>_<rate>_fc.raw (prefix ok) */
    IQ_NAME_SDRSHARP,  /* SDRSharp_<date>_<time>Z_<freq>Hz_<rate>Hz_IQ.wav */
    IQ_NAME_HDSDR,     /* HDSDR_<date>_<time>Z_<freq>kHz_RF.wav */
    IQ_NAME_SIGMF,     /* <base>.sigmf-data + <base>.sigmf-meta sidecar    */
    IQ_NAME_SDRPP,     /* <base>.raw + <base>.json sidecar (SDR++)         */
    IQ_NAME_GENERIC,   /* frequency/rate tokens found in the name          */
    IQ_NAME_UNKNOWN,   /* nothing could be inferred                        */
} IqNameKind;

typedef struct {
    const char *path;
    int         fd;
    size_t      file_size;
    size_t      data_offset;   /* start of sample data (e.g. WAV header) */
    const float *samples;      /* mmap'd interleaved float32 I/Q;
                                * VALID ONLY when format == FLOAT32  */
    const uint8_t *raw;        /* the memory map itself              */
    size_t      total_samples; /* number of complex samples          */
    size_t      bytes_per_sample;
    double      sample_rate;   /* Hz (effective playback rate)       */
    double      center_freq;   /* Hz */
    double      duration;      /* seconds */
    IqFormat    format;

    /* Recording start wall-clock, parsed from the filename (local time
     * as written by gqrx). Only valid when has_start_time is true. */
    bool   has_start_time;
    time_t start_time;

    /* How the metadata was obtained (diagnostics / UI). */
    IqNameKind name_kind;

    /* Audio recording (WAV/AIFF native, compressed through the platform
     * decoder): the samples are decoded to an owned float buffer and read
     * as complex with mono -> I (Q = 0) and stereo -> I/Q. The sample rate
     * comes from the container and the centre is rate/2 so the frequency
     * axis covers the audio band 0..rate. */
    bool   is_audio;
    int    channels;
    int    audio_bits;
    bool   audio_float;
    char   audio_codec[16];
} IqSource;

/*
 * Parse a gqrx-style filename for start time, frequency and sample rate.
 * Returns true and fills *start_time (may be NULL) / *freq_hz /
 * *rate_hz on success. The path component is stripped before matching.
 * The date/time is interpreted as local wall-clock time.
 */
bool iq_source_parse_filename(const char *path, time_t *start_time,
                              double *freq_hz, double *rate_hz);

/*
 * Open and memory-map an IQ recording. `err` (if non-NULL) receives a
 * human-readable description of any failure. Returns true on success.
 */
bool iq_source_open(IqSource *src, const char *path, char *err, size_t err_len);

/*
 * Infer centre frequency, sample rate and data format from a filename that
 * is not gqrx-style, using the conventions of the common SDR recorders
 * (SDR#, HDSDR, SDR++, SigMF sidecars, generic "<freq>e6_<rate>e6" names)
 * and, when nothing is in the name, from the file size/content (a guessed
 * format is validated against the actual bytes). Missing values are left
 * at 0. Returns the kind of evidence found.
 */
IqNameKind iq_source_guess_metadata(const char *path, double *freq_hz,
                                    double *rate_hz, IqFormat *format);

/*
 * Open a recording with explicit parameters (the manual-entry fallback when
 * neither the gqrx convention nor any other inference applies). The file is
 * still memory-mapped; only the metadata comes from the caller.
 */
bool iq_source_open_custom(IqSource *src, const char *path, double center_hz,
                           double rate_hz, IqFormat format, char *err,
                           size_t err_len);

/* Copy `n` complex samples starting at `offset` into `out` (2*n floats),
 * converting from the source's data format on the fly. */
void iq_source_read_complex(const IqSource *src, size_t offset, float *out, size_t n);

/*
 * Change the data format and the effective sample rate at runtime
 * (the Settings page): both recompute total_samples / duration and are
 * used by every subsequent read. Safe to call while open.
 */
void iq_source_set_format(IqSource *src, IqFormat format);
void iq_source_set_rate(IqSource *src, double rate_hz);

/*
 * Format `offset_s` seconds into the recording as an RFC 3339 / ISO 8601
 * timestamp with a local-time UTC offset, e.g. "2026-03-06T17:44:05+01:00".
 * When the recording start time is unknown, falls back to a relative
 * form such as "T+12.345s". Returns `buf` (always NUL-terminated).
 */
const char *iq_source_format_time(const IqSource *src, double offset_s,
                                  char *buf, size_t buf_len);

/*
 * Re-decode an audio source with a forced source bit depth (0 = keep the
 * container's type). Only meaningful for uncompressed containers; the
 * decoded buffer is replaced in place. Returns true on success.
 */
bool iq_source_reopen_audio(IqSource *src, int force_bits);

/* Unmap and close. Safe to call on a zeroed struct. */
void iq_source_close(IqSource *src);

#endif /* TE15_IQ_SOURCE_H */
