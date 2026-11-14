/*
 * iq_source.c — read-only access to gqrx raw IQ recordings (see header).
 */
#include "iq_source.h"

#include "audio_file.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

bool iq_source_parse_filename(const char *path, time_t *start_time,
                              double *freq_hz, double *rate_hz)
{
    if (path == NULL) return false;

    const char *base = strrchr(path, '/');
    base = (base != NULL) ? base + 1 : path;

    unsigned int date = 0, hms = 0;
    long long    freq = 0, rate = 0;
    char         suffix[16];

    /*
     * Use a scanset for the suffix so it stops at the '.' before the
     * extension. (A plain %s would swallow "fc.raw" on some C
     * libraries, leaving a dirty suffix buffer.)
     */
    /*
     * The gqrx marker may be prefixed by a descriptive label (reference
     * fixtures are named "<signal>_gqrx_..." so the files identify
     * themselves), so locate it instead of requiring it at the start.
     */
    {
        const char *g = strstr(base, "gqrx_");

        if (g == NULL) return false;
        if (sscanf(g, "gqrx_%u_%u_%lld_%lld_%15[^.].raw",
                   &date, &hms, &freq, &rate, suffix) != 5) {
            return false;
        }
    }
    /* Darwin's scanf can be lenient about the trailing literal, so
     * verify the extension explicitly as well. */
    size_t len = strlen(base);
    if (len < 4 || strcmp(base + len - 4, ".raw") != 0) {
        return false;
    }
    /* The conventional gqrx suffix is "fc" (float complex); a label may
     * follow it (fc_<signal name>). */
    if (strncmp(suffix, "fc", 2) != 0) {
        return false;
    }
    if (freq <= 0 || rate <= 0) return false;

    if (start_time != NULL) {
        struct tm tm;
        memset(&tm, 0, sizeof(tm));
        tm.tm_year = (int)(date / 10000) - 1900;
        tm.tm_mon  = (int)((date / 100) % 100) - 1;
        tm.tm_mday = (int)(date % 100);
        tm.tm_hour = (int)(hms / 10000);
        tm.tm_min  = (int)((hms / 100) % 100);
        tm.tm_sec  = (int)(hms % 100);
        tm.tm_isdst = -1; /* resolve DST locally */

        /* Basic sanity on the fields before mktime normalises them. */
        bool sane = tm.tm_year >= 0 && tm.tm_year < 300 &&
                    tm.tm_mon >= 0 && tm.tm_mon < 12 &&
                    tm.tm_mday >= 1 && tm.tm_mday <= 31 &&
                    tm.tm_hour >= 0 && tm.tm_hour < 24 &&
                    tm.tm_min >= 0 && tm.tm_min < 60 &&
                    tm.tm_sec >= 0 && tm.tm_sec < 62;
        time_t t = sane ? mktime(&tm) : (time_t)-1;
        *start_time = (t != (time_t)-1) ? t : (time_t)0;
    }

    *freq_hz = (double)freq;
    *rate_hz = (double)rate;
    return true;
}

static void set_error(char *err, size_t err_len, const char *fmt, const char *arg)
{
    if (err != NULL && err_len > 0) {
        snprintf(err, err_len, fmt, arg);
    }
}

/* ------------------------------------------------------------------ */
/* metadata inference for non-gqrx recordings                          */
/* ------------------------------------------------------------------ */

/* parse a number with an optional SI/engineering suffix at `p` */
static bool parse_scaled(const char *p, double *out)
{
    char *end = NULL;
    double v = strtod(p, &end);

    if (end == p || v <= 0.0) return false;
    {
        char c0 = *end;
        char c1 = end[1];
        double mult = 1.0;

        if (c0 == 'k' || c0 == 'K') mult = 1e3;
        else if (c0 == 'M') mult = 1e6;
        else if (c0 == 'G') mult = 1e9;
        else if (c0 == 'm') mult = 1e-3;
        else if (c0 == 'u' || (unsigned char)c0 == 0xC2) mult = 1e-6;
        if (mult != 1.0) {
            /* only treat as a suffix when followed by a unit letter */
            if (!(c1 == 'H' || c1 == 'h' || c1 == 'z' || c1 == 'Z' ||
                  c1 == 'S' || c1 == 's')) {
                return false;
            }
            v *= mult;
        } else if (c0 == 'e' || c0 == 'E') {
            v = strtod(p, &end); /* exponent form, already handled */
        }
    }
    if (v < 1e2 || v > 1e12) return false;
    *out = v;
    return true;
}

/* scan a filename for <number>[kM]Hz / <number>Hz / <number>S/s tokens */
static bool scan_freq_rate(const char *base, double *freq, double *rate)
{
    const char *p = base;
    bool have_f = false, have_r = false;

    while ((p = strpbrk(p, "0123456789")) != NULL) {
        double v;
        const char *end = p;
        char *num_end = NULL;

        v = strtod(p, &num_end);
        end = num_end;
        if (end == p) {
            p++;
            continue;
        }
        {
            char unit = *end;
            double mult = 1.0;

            if (unit == 'k' || unit == 'K') { mult = 1e3; end++; }
            else if (unit == 'M') { mult = 1e6; end++; }
            else if (unit == 'G') { mult = 1e9; end++; }
            v *= mult;
            if (v >= 1e5 && v <= 1e12) {
                /* classify: rate if followed by S/s, Msps, or 'S' */
                if ((end[0] == 'S' || end[0] == 's') ||
                    (end[0] == 'M' && (end[1] == 's' || end[1] == 'S'))) {
                    if (!have_r) {
                        *rate = v;
                        have_r = true;
                    }
                } else if (!have_f && v >= 1e6) {
                    *freq = v;
                    have_f = true;
                } else if (!have_r) {
                    *rate = v;
                    have_r = true;
                }
            }
        }
        p = end > p ? end : p + 1;
    }
    return have_f && have_r;
}

/* minimal JSON number/string lookup: "key": 1234 */
static bool json_number(const char *json, const char *key, double *out)
{
    char pat[96];
    const char *p;

    snprintf(pat, sizeof(pat), "\"%s\"", key);
    p = json != NULL ? strstr(json, pat) : NULL;
    if (p == NULL) return false;
    p += strlen(pat);
    while (*p == ' ' || *p == ':' || *p == '\t') p++;
    if (strchr("0123456789-", *p) != NULL) {
        *out = strtod(p, NULL);
        return true;
    }
    return false;
}

static bool json_string(const char *json, const char *key, char *out, size_t n)
{
    char pat[96];
    const char *p;

    snprintf(pat, sizeof(pat), "\"%s\"", key);
    p = json != NULL ? strstr(json, pat) : NULL;
    if (p == NULL) return false;
    p += strlen(pat);
    while (*p == ' ' || *p == ':' || *p == '\t') p++;
    if (*p != '"') return false;
    p++;
    {
        size_t i = 0;

        while (*p != '"' && *p != '\0' && i + 1 < n) out[i++] = *p++;
        out[i] = '\0';
        return i > 0;
    }
}

static char *read_sidecar(const char *path, const char *suffix)
{
    size_t len = strlen(path) + strlen(suffix) + 16;
    char *alt = malloc(len);
    FILE *f = NULL;
    char *buf = NULL;
    long sz;

    if (alt == NULL) return NULL;
    /* try path + suffix and path with the extension replaced */
    snprintf(alt, len, "%s%s", path, suffix);
    f = fopen(alt, "rb");
    if (f == NULL) {
        const char *dot = strrchr(path, '.');

        if (dot != NULL) {
            snprintf(alt, len, "%.*s%s", (int)(dot - path), path, suffix);
            f = fopen(alt, "rb");
        }
    }
    free(alt);
    if (f == NULL) return NULL;
    fseek(f, 0, SEEK_END);
    sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0 || sz > 4 * 1024 * 1024) {
        fclose(f);
        return NULL;
    }
    buf = malloc((size_t)sz + 1);
    if (buf == NULL) {
        fclose(f);
        return NULL;
    }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf);
        fclose(f);
        return NULL;
    }
    buf[sz] = '\0';
    fclose(f);
    return buf;
}

/* infer the data format from the extension */
static bool format_from_name(const char *base, IqFormat *out)
{
    const char *dot = strrchr(base, '.');

    if (dot == NULL) return false;
    if (strstr(base, ".cf32") || strstr(base, ".fc32") ||
        strstr(base, ".complex") || strstr(base, ".cfile")) {
        *out = IQ_FORMAT_FLOAT32;
        return true;
    }
    if (strstr(base, ".cs16") || strstr(base, ".sc16")) {
        *out = IQ_FORMAT_INT16;
        return true;
    }
    if (strstr(base, ".cs8") || strstr(base, ".sc8")) {
        *out = IQ_FORMAT_INT8;
        return true;
    }
    if (strstr(base, ".cu8") || strstr(base, ".uc8")) {
        *out = IQ_FORMAT_UINT8;
        return true;
    }
    if (strstr(base, ".sigmf-data")) {
        *out = IQ_FORMAT_FLOAT32; /* refined by the sidecar */
        return true;
    }
    (void)dot;
    return false;
}

/* validate a candidate format against the file's actual bytes */
static bool format_plausible(const uint8_t *raw, size_t bytes, IqFormat fmt)
{
    size_t nsamp = bytes / (fmt == IQ_FORMAT_FLOAT32 ? 8 : 2);

    if (nsamp < 256) return true;
    if (fmt == IQ_FORMAT_FLOAT32) {
        const float *f = (const float *)raw;
        size_t bad = 0, checked = 0;

        for (size_t i = 0; i < 2048 && i < nsamp; i++) {
            for (int k = 0; k < 2; k++) {
                float v = f[2 * i + k];

                if (!isfinite(v) || fabsf(v) > 64.0f) bad++;
                checked++;
            }
        }
        return checked > 0 && (double)bad / (double)checked < 0.02;
    }
    if (fmt == IQ_FORMAT_INT16) {
        return true; /* any bit pattern is a valid int16 stream */
    }
    {
        /* int8 vs uint8: the DC offset tells them apart */
        size_t n = nsamp * 2 < 4096 ? nsamp * 2 : 4096;
        double sum = 0.0;

        for (size_t i = 0; i < n; i++) sum += (double)raw[i];
        sum /= (double)(n == 0 ? 1 : n);
        if (fmt == IQ_FORMAT_UINT8) return sum > 40.0;
        return sum < 60.0;
    }
}

IqNameKind iq_source_guess_metadata(const char *path, double *freq_hz,
                                    double *rate_hz, IqFormat *format)
{
    const char *base = strrchr(path, '/');
    double f = 0.0, r = 0.0;
    IqFormat fmt = IQ_FORMAT_FLOAT32;
    IqNameKind kind = IQ_NAME_UNKNOWN;

    if (freq_hz != NULL) *freq_hz = 0.0;
    if (rate_hz != NULL) *rate_hz = 0.0;
    if (format != NULL) *format = IQ_FORMAT_FLOAT32;
    if (base == NULL) base = path; else base++;

    /* SDR#: SDRSharp_<date>_<time>Z_<freq>Hz_<rate>Hz_IQ.wav */
    if (strstr(base, "SDRSharp") != NULL || strstr(base, "SDR#") != NULL) {
        const char *p = strstr(base, "_IQ");

        if (p != NULL) {
            const char *q = p;

            while (q > base && q[-1] != '_') q--;
            if (!parse_scaled(q, &r)) {
                /* fall back: last two numeric tokens */
            }
        }
        {
            /* tokens ..._<freq>Hz_<rate>Hz_IQ */
            const char *p2 = strstr(base, "Hz_");

            if (p2 != NULL) {
                const char *num = p2;

                while (num > base && (isdigit((unsigned char)num[-1]) ||
                                      num[-1] == '.')) num--;
                {
                    char tmp[64];
                    size_t l = (size_t)(p2 - num);

                    if (l < sizeof(tmp) && l > 0) {
                        memcpy(tmp, num, l);
                        tmp[l] = '\0';
                        f = strtod(tmp, NULL);
                        {
                            const char *q2 = p2 + 3;

                            if (!parse_scaled(q2, &r)) {
                                char tmp2[64];
                                const char *z = q2;
                                size_t l2 = 0;

                                while (isdigit((unsigned char)*z) ||
                                       *z == '.') {
                                    if (l2 + 1 < sizeof(tmp2)) {
                                        tmp2[l2++] = *z;
                                    }
                                    z++;
                                }
                                tmp2[l2] = '\0';
                                r = strtod(tmp2, NULL);
                            }
                        }
                    }
                }
            }
        }
        if (f > 1e5) kind = IQ_NAME_SDRSHARP;
    }
    /* HDSDR: HDSDR_<date>_<time>Z_<freq>kHz_RF.wav */
    else if (strstr(base, "HDSDR") != NULL) {
        const char *p = strstr(base, "kHz");

        if (p != NULL) {
            const char *num = p;

            while (num > base && (isdigit((unsigned char)num[-1]) ||
                                  num[-1] == '.')) num--;
            {
                char tmp[64];
                size_t l = (size_t)(p - num);

                if (l > 0 && l < sizeof(tmp)) {
                    memcpy(tmp, num, l);
                    tmp[l] = '\0';
                    f = strtod(tmp, NULL) * 1e3;
                }
            }
        }
        kind = f > 1e5 ? IQ_NAME_HDSDR : IQ_NAME_UNKNOWN;
    } else {
        /* sidecars: SigMF / SDR++ */
        char *meta = read_sidecar(path, ".sigmf-meta");

        if (meta == NULL) meta = read_sidecar(path, ".json");
        if (meta != NULL) {
            char dtype[64] = "";

            if (!json_number(meta, "core:frequency", &f)) {
                json_number(meta, "frequency", &f);
            }
            if (!json_number(meta, "core:sample_rate", &r)) {
                json_number(meta, "sampleRate", &r);
            }
            json_string(meta, "core:datatype", dtype, sizeof(dtype));
            if (dtype[0] != '\0') {
                if (strstr(dtype, "f32")) fmt = IQ_FORMAT_FLOAT32;
                else if (strstr(dtype, "i16")) fmt = IQ_FORMAT_INT16;
                else if (strstr(dtype, "i8")) fmt = IQ_FORMAT_INT8;
                else if (strstr(dtype, "u8")) fmt = IQ_FORMAT_UINT8;
                if (strstr(dtype, "cf32") || strstr(dtype, "ci16")) {
                    /* complex interleaved - our layout */
                }
            }
            free(meta);
            if (f > 1e5 || r > 1e3) {
                kind = strstr(path, ".sigmf") != NULL ? IQ_NAME_SIGMF
                                                      : IQ_NAME_SDRPP;
            }
        }
        if (kind == IQ_NAME_UNKNOWN) {
            format_from_name(base, &fmt);
            if (scan_freq_rate(base, &f, &r)) {
                kind = IQ_NAME_GENERIC;
            }
        }
    }
    if (format != NULL) *format = fmt;
    if (freq_hz != NULL) *freq_hz = f >= 1e5 ? f : 0.0;
    if (rate_hz != NULL) *rate_hz = r >= 1e3 ? r : 0.0;
    return kind;
}

/*
 * Minimal RIFF/WAVE reader: locate the fmt and data chunks so WAV
 * recordings (SDR#, HDSDR, rtl_sdr and SDR++ can all write WAV) work.
 * Returns true when the file is a WAV and the parameters were found.
 */
static bool parse_wav_header(const uint8_t *raw, size_t size, double *rate,
                             IqFormat *format, size_t *data_offset,
                             size_t *data_size)
{
    if (size < 44 || memcmp(raw, "RIFF", 4) != 0 ||
        memcmp(raw + 8, "WAVE", 4) != 0) {
        return false;
    }
    {
        size_t pos = 12;
        bool have_fmt = false;
        double fs = 0.0;
        int channels = 0, bits = 0, fmttag = 0;
        size_t data_off = 0, data_len = 0;

        while (pos + 8 <= size) {
            uint32_t clen = (uint32_t)raw[pos + 4] | ((uint32_t)raw[pos + 5] << 8) |
                            ((uint32_t)raw[pos + 6] << 16) |
                            ((uint32_t)raw[pos + 7] << 24);
            const uint8_t *body = raw + pos + 8;

            if (memcmp(raw + pos, "fmt ", 4) == 0 && clen >= 16 &&
                pos + 8 + 16 <= size) {
                fmttag = (int)(body[0] | (body[1] << 8));
                channels = (int)(body[2] | (body[3] << 8));
                fs = (double)((uint32_t)body[4] | ((uint32_t)body[5] << 8) |
                              ((uint32_t)body[6] << 16) |
                              ((uint32_t)body[7] << 24));
                bits = (int)(body[14] | (body[15] << 8));
                have_fmt = true;
            } else if (memcmp(raw + pos, "data", 4) == 0) {
                data_off = pos + 8;
                data_len = clen;
                break;
            }
            pos += 8 + clen + (clen & 1);
        }
        if (!have_fmt || data_off == 0 || channels <= 0 || bits <= 0) {
            return false;
        }
        if (data_len == 0 || data_off + data_len > size) {
            data_len = size - data_off;
        }
        /* only stereo complex layouts make sense for IQ */
        if (channels != 2) {
            return false;
        }
        if (fmttag == 3 && bits == 32) {
            *format = IQ_FORMAT_FLOAT32;
        } else if (fmttag == 1 && bits == 16) {
            *format = IQ_FORMAT_INT16;
        } else if (fmttag == 1 && bits == 8) {
            *format = IQ_FORMAT_UINT8;
        } else {
            return false;
        }
        *rate = fs;
        *data_offset = data_off;
        *data_size = data_len;
        return true;
    }
}

/* Sample-data extensions that are never audio containers. */
static bool is_iq_extension(const char *path)
{
    static const char *const exts[] = {
        ".raw", ".iq", ".bin", ".cf32", ".fc32", ".cs16", ".sc16", ".cs8",
        ".sc8", ".cu8", ".uc8", ".complex", ".cfile", ".sigmf-data", NULL
    };
    const char *dot = strrchr(path, '.');

    if (dot == NULL) return false;
    for (int i = 0; exts[i] != NULL; i++) {
        if (strcasecmp(dot, exts[i]) == 0) return true;
    }
    return false;
}

/*
 * Try to open `path` as an audio recording: decode it to an owned float
 * buffer and expose it as a complex source (mono -> I with Q = 0, stereo ->
 * I/Q). Returns true when the file was audio.
 */
static bool open_audio(IqSource *src, const char *path, char *err,
                       size_t err_len)
{
    AudioInfo info;
    float *pcm = NULL;
    size_t frames = 0;

    if (is_iq_extension(path)) return false;
    if (!audio_file_load(path, 0, &info, &pcm, &frames)) return false;
    if (frames == 0 || info.channels <= 0 || info.sample_rate <= 0) {
        free(pcm);
        return false;
    }
    {
        float *iq = malloc(2 * frames * sizeof(float));

        if (iq == NULL) {
            free(pcm);
            set_error(err, err_len, "Out of memory decoding audio: %s", path);
            return false;
        }
        if (info.channels == 1) {
            for (size_t i = 0; i < frames; i++) {
                iq[2 * i] = pcm[i];
                iq[2 * i + 1] = 0.0f;
            }
        } else {
            for (size_t i = 0; i < frames; i++) {
                iq[2 * i] = pcm[2 * i];
                iq[2 * i + 1] = pcm[2 * i + 1];
            }
        }
        free(pcm);
        src->raw = (const uint8_t *)iq;
        src->samples = iq;
        src->is_audio = true;
        src->channels = info.channels;
        src->audio_bits = info.bits;
        src->audio_float = info.is_float;
        snprintf(src->audio_codec, sizeof(src->audio_codec), "%s",
                 info.codec);
        src->total_samples = frames;
        src->bytes_per_sample = 8;
        src->format = IQ_FORMAT_FLOAT32;
        src->sample_rate = info.sample_rate;
        /* the frequency axis covers the audio band 0..rate */
        src->center_freq = info.sample_rate / 2.0;
        src->duration = (double)frames / info.sample_rate;
        src->has_start_time = false;
        src->start_time = 0;
        src->name_kind = IQ_NAME_UNKNOWN;
        src->fd = -1;
        {
            /* keep the path ownership rule from iq_source_open */
            src->path = strdup(path);
            if (src->path == NULL) {
                free(iq);
                set_error(err, err_len, "Out of memory: %s", path);
                return false;
            }
        }
        return true;
    }
}

bool iq_source_open(IqSource *src, const char *path, char *err, size_t err_len)
{
    memset(src, 0, sizeof(*src));
    src->fd = -1;
    /* Own the path: callers (the GTK file chooser) free their string as
     * soon as this returns, and every later report/filename use would
     * then read freed memory. */
    src->path = strdup(path);
    if (src->path == NULL) {
        set_error(err, err_len, "Out of memory opening: %s", path);
        return false;
    }

    double freq = 0.0, rate = 0.0;
    time_t start = 0;
    IqFormat guessed = IQ_FORMAT_FLOAT32;
    IqNameKind kind = IQ_NAME_GQRX;
    bool have_meta = iq_source_parse_filename(path, &start, &freq, &rate);

    if (!have_meta) {
        /* an audio recording carries everything in its container header */
        if (open_audio(src, path, err, err_len)) {
            return true;
        }
        /* not gqrx-style and not audio: try the names/sidecars of other SDR
         * recorders, then infer what we can from the file itself */
        kind = iq_source_guess_metadata(path, &freq, &rate, &guessed);
        if (freq <= 0.0) {
            set_error(err, err_len,
                      "Cannot determine the recording parameters from the "
                      "filename (%s). Enter the centre frequency, sample "
                      "rate and sample format manually.", path);
            return false;
        }
    }

    src->fd = open(path, O_RDONLY);
    if (src->fd < 0) {
        set_error(err, err_len, "Cannot open file: %s", path);
        return false;
    }

    struct stat st;
    if (fstat(src->fd, &st) < 0) {
        set_error(err, err_len, "Cannot stat file: %s", path);
        iq_source_close(src);
        return false;
    }
    src->file_size = (size_t)st.st_size;

    src->raw = mmap(NULL, src->file_size, PROT_READ, MAP_PRIVATE,
                    src->fd, 0);
    if (src->raw == MAP_FAILED) {
        set_error(err, err_len, "Cannot memory-map file: %s", path);
        iq_source_close(src);
        return false;
    }
    src->samples = (const float *)src->raw; /* valid for FLOAT32 */

    /* WAV containers carry their own rate/format (and a data offset) */
    {
        double wav_rate = 0.0;
        IqFormat wav_fmt = IQ_FORMAT_FLOAT32;
        size_t wav_off = 0, wav_len = 0;

        if (parse_wav_header(src->raw, src->file_size, &wav_rate, &wav_fmt,
                             &wav_off, &wav_len)) {
            rate = wav_rate;
            guessed = wav_fmt;
            src->data_offset = wav_off;
            src->file_size = wav_len; /* reads stop at the data chunk */
            src->file_size += wav_off;
            kind = kind == IQ_NAME_GQRX ? IQ_NAME_GENERIC : kind;
        }
        if (rate <= 0.0) {
            set_error(err, err_len,
                      "Cannot determine the sample rate from the filename or "
                      "container (%s). Enter the centre frequency, sample "
                      "rate and sample format manually.", path);
            iq_source_close(src);
            return false;
        }
        if (false) {
            /* validate the inferred format against the bytes; fall back to
             * the most likely alternative when it is implausible */
            if (!format_plausible(src->raw + src->data_offset,
                                  src->file_size - src->data_offset,
                                  guessed)) {
                guessed = IQ_FORMAT_FLOAT32;
                if (!format_plausible(src->raw, src->file_size,
                                      IQ_FORMAT_FLOAT32)) {
                    guessed = IQ_FORMAT_INT16;
                }
            }
        }
    }

    src->center_freq      = freq;
    src->sample_rate      = rate;
    src->has_start_time   = start != 0;
    src->start_time       = start;
    src->name_kind        = kind;
    /* the guessed rate is only a starting point: the caller can refine it */
    iq_source_set_format(src, guessed == IQ_FORMAT_FLOAT32 && have_meta
                                    ? IQ_FORMAT_FLOAT32
                                    : guessed);
    iq_source_set_rate(src, rate);

    if (src->total_samples == 0) {
        set_error(err, err_len, "File contains no samples: %s", path);
        iq_source_close(src);
        return false;
    }
    return true;
}

void iq_source_read_complex(const IqSource *src, size_t offset,
                            float *out, size_t n)
{
    if (src == NULL || src->raw == NULL || out == NULL || n == 0) {
        return;
    }
    size_t avail = src->total_samples - (offset < src->total_samples
                                             ? offset : src->total_samples);
    if (n > avail) n = avail;
    if (n == 0) return;

    const uint8_t *base = src->raw + src->data_offset +
                          offset * src->bytes_per_sample;
    switch (src->format) {
    case IQ_FORMAT_FLOAT32:
        memcpy(out, base, 2 * n * sizeof(float));
        break;
    case IQ_FORMAT_INT16: {
        const int16_t *p = (const int16_t *)base;
        for (size_t i = 0; i < n; i++) {
            out[2 * i]     = (float)p[2 * i] / 32768.0f;
            out[2 * i + 1] = (float)p[2 * i + 1] / 32768.0f;
        }
        break;
    }
    case IQ_FORMAT_INT8:
        for (size_t i = 0; i < n; i++) {
            out[2 * i]     = (float)((const int8_t *)base)[2 * i] / 128.0f;
            out[2 * i + 1] = (float)((const int8_t *)base)[2 * i + 1] / 128.0f;
        }
        break;
    case IQ_FORMAT_UINT8:
        for (size_t i = 0; i < n; i++) {
            out[2 * i]     = ((float)base[2 * i] - 127.5f) / 127.5f;
            out[2 * i + 1] = ((float)base[2 * i + 1] - 127.5f) / 127.5f;
        }
        break;
    }
}

void iq_source_set_format(IqSource *src, IqFormat format)
{
    if (src == NULL) return;
    src->format = format;
    switch (format) {
    case IQ_FORMAT_FLOAT32: src->bytes_per_sample = 8; break;
    case IQ_FORMAT_INT16:   src->bytes_per_sample = 4; break;
    case IQ_FORMAT_INT8:
    case IQ_FORMAT_UINT8:   src->bytes_per_sample = 2; break;
    }
    {
        size_t bytes = src->file_size > src->data_offset
                           ? src->file_size - src->data_offset
                           : 0;

        src->total_samples = bytes / src->bytes_per_sample;
    }
    if (src->sample_rate > 0.0) {
        src->duration = (double)src->total_samples / src->sample_rate;
    }
}

void iq_source_set_rate(IqSource *src, double rate_hz)
{
    if (src == NULL || rate_hz <= 0.0) return;
    src->sample_rate = rate_hz;
    src->duration = (double)src->total_samples / src->sample_rate;
}

const char *iq_source_format_time(const IqSource *src, double offset_s,
                                  char *buf, size_t buf_len)
{
    if (buf == NULL || buf_len == 0) return buf;

    if (src == NULL || !src->has_start_time || src->start_time == 0) {
        snprintf(buf, buf_len, "T+%.3fs", offset_s);
        return buf;
    }

    time_t t = src->start_time + (time_t)offset_s;
    struct tm lt;
    if (localtime_r(&t, &lt) == NULL) {
        snprintf(buf, buf_len, "T+%.3fs", offset_s);
        return buf;
    }

    /* RFC 3339 local time with explicit UTC offset (tm_gmtoff is in
     * seconds east of UTC on BSD/macOS and glibc). */
    long off = lt.tm_gmtoff;
    char sign = (off < 0) ? '-' : '+';
    long aoff = labs(off);
    snprintf(buf, buf_len, "%04d-%02d-%02dT%02d:%02d:%02d%c%02ld:%02ld",
             lt.tm_year + 1900, lt.tm_mon + 1, lt.tm_mday,
             lt.tm_hour, lt.tm_min, lt.tm_sec,
             sign, aoff / 3600, (aoff % 3600) / 60);
    return buf;
}

void iq_source_close(IqSource *src)
{
    if (src == NULL) return;
    if (src->is_audio) {
        free((void *)src->raw); /* decoded float buffer, not a mapping */
    } else if (src->raw != NULL && src->raw != MAP_FAILED) {
        munmap((void *)src->raw, src->file_size);
    }
    if (src->fd >= 0) {
        close(src->fd);
    }
    free((void *)src->path); /* owned copy (see iq_source_open) */
    memset(src, 0, sizeof(*src));
    src->fd = -1;
}

bool iq_source_open_custom(IqSource *src, const char *path, double center_hz,
                           double rate_hz, IqFormat format, char *err,
                           size_t err_len)
{
    struct stat st;

    if (src == NULL || path == NULL || center_hz <= 0.0 || rate_hz <= 0.0) {
        set_error(err, err_len, "Invalid recording parameters: %s", path);
        return false;
    }
    memset(src, 0, sizeof(*src));
    src->fd = -1;
    /* Own the path: callers (the GTK file chooser) free their string as
     * soon as this returns, and every later report/filename use would
     * then read freed memory. */
    src->path = strdup(path);
    if (src->path == NULL) {
        set_error(err, err_len, "Out of memory opening: %s", path);
        return false;
    }
    src->fd = open(path, O_RDONLY);
    if (src->fd < 0) {
        set_error(err, err_len, "Cannot open file: %s", path);
        return false;
    }
    if (fstat(src->fd, &st) < 0 || st.st_size <= 0) {
        set_error(err, err_len, "Cannot stat file: %s", path);
        iq_source_close(src);
        return false;
    }
    src->file_size = (size_t)st.st_size;
    src->raw = mmap(NULL, src->file_size, PROT_READ, MAP_PRIVATE, src->fd, 0);
    if (src->raw == MAP_FAILED) {
        set_error(err, err_len, "Cannot memory-map file: %s", path);
        iq_source_close(src);
        return false;
    }
    src->samples = (const float *)src->raw;
    src->center_freq = center_hz;
    src->sample_rate = rate_hz;
    src->name_kind = IQ_NAME_UNKNOWN;
    {
        double wav_rate = 0.0;
        IqFormat wav_fmt = IQ_FORMAT_FLOAT32;
        size_t wav_off = 0, wav_len = 0;

        if (parse_wav_header(src->raw, src->file_size, &wav_rate, &wav_fmt,
                             &wav_off, &wav_len)) {
            format = wav_fmt;
            src->data_offset = wav_off;
            src->file_size = wav_off + wav_len;
        }
    }
    iq_source_set_format(src, format);
    iq_source_set_rate(src, rate_hz);
    if (src->total_samples == 0) {
        set_error(err, err_len, "File contains no samples: %s", path);
        iq_source_close(src);
        return false;
    }
    return true;
}

bool iq_source_reopen_audio(IqSource *src, int force_bits)
{
    AudioInfo info;
    float *pcm = NULL;
    size_t frames = 0;
    float *iq;

    if (src == NULL || !src->is_audio || src->path == NULL) return false;
    if (force_bits <= 0) return true;
    if (!audio_file_load(src->path, force_bits, &info, &pcm, &frames) ||
        frames == 0 || info.channels <= 0 || info.sample_rate <= 0) {
        free(pcm);
        return false;
    }
    iq = malloc(2 * frames * sizeof(float));
    if (iq == NULL) {
        free(pcm);
        return false;
    }
    if (info.channels == 1) {
        for (size_t i = 0; i < frames; i++) {
            iq[2 * i] = pcm[i];
            iq[2 * i + 1] = 0.0f;
        }
    } else {
        for (size_t i = 0; i < frames; i++) {
            iq[2 * i] = pcm[2 * i];
            iq[2 * i + 1] = pcm[2 * i + 1];
        }
    }
    free(pcm);
    free((void *)src->raw);
    src->raw = (const uint8_t *)iq;
    src->samples = iq;
    src->audio_bits = info.bits;
    src->audio_float = info.is_float;
    src->total_samples = frames;
    src->duration = (double)frames / src->sample_rate;
    return true;
}
