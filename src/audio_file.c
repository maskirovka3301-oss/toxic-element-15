/*
 * audio_file.c — audio container decoding (see header).
 *
 * Uncompressed RIFF/WAVE and AIFF/AIFC are parsed natively so the sample
 * rate, channel count and bit depth always come from the header, even
 * without any platform decoder. Compressed formats (AAC/M4A/MP3/ALAC/
 * OGG/FLAC/OPUS, whatever the platform understands) go through
 * AudioToolbox's ExtAudioFile on macOS; other platforms report failure.
 */
#include "audio_file.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__APPLE__)
#include <AudioToolbox/AudioToolbox.h>
#endif

typedef struct {
    uint32_t clen;
    int32_t  format;    /* WAVE format tag (1 = PCM, 3 = float)        */
    int      channels;
    uint32_t sample_rate;
    int      bits;
    size_t   data_offset;
    size_t   data_size;
    bool     ok;
} WavInfo;

static uint32_t rd_le32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static uint32_t rd_be32(const uint8_t *p)
{
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

static uint32_t rd_le16(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8);
}

/* ---- RIFF/WAVE -------------------------------------------------------- */

static bool wav_parse(FILE *f, size_t size, WavInfo *w)
{
    uint8_t hdr[12];

    memset(w, 0, sizeof(*w));
    if (size < 12) return false;
    if (fread(hdr, 1, 12, f) != 12) return false;
    if (memcmp(hdr, "RIFF", 4) != 0 || memcmp(hdr + 8, "WAVE", 4) != 0) {
        return false;
    }
    {
        size_t pos = 12;
        int fmt_tag = 0;

        while (pos + 8 <= size) {
            uint8_t ch[8];

            if (fread(ch, 1, 8, f) != 8) break;
            {
                uint32_t clen = rd_le32(ch + 4);

                if (memcmp(ch, "fmt ", 4) == 0 && clen >= 16) {
                    uint8_t b[40];

                    if (clen > sizeof(b)) {
                        if (fread(b, 1, sizeof(b), f) != sizeof(b)) break;
                        /* skip the rest of the chunk */
                        if (fseek(f, (long)(clen - sizeof(b)), SEEK_CUR) !=
                            0) {
                            break;
                        }
                    } else if (fread(b, 1, clen, f) != clen) {
                        break;
                    }
                    w->format = (int32_t)rd_le16(b);
                    w->channels = (int)rd_le16(b + 2);
                    w->sample_rate = rd_le32(b + 4);
                    w->bits = (int)rd_le16(b + 14);
                    fmt_tag = w->format;
                    /* WAVE_FORMAT_EXTENSIBLE: the real tag is the first
                     * two bytes of the subformat GUID */
                    if (w->format == 0xFFFE && clen >= 40) {
                        w->format = (int32_t)rd_le16(b + 24);
                    }
                    w->ok = true;
                } else if (memcmp(ch, "data", 4) == 0) {
                    w->data_offset = pos + 8;
                    w->data_size = clen;
                    w->clen = clen;
                    break;
                } else {
                    if (fseek(f, (long)clen + (long)(clen & 1), SEEK_CUR) != 0) {
                        break;
                    }
                }
                pos += 8 + clen + (clen & 1);
            }
        }
        (void)fmt_tag;
    }
    if (!w->ok || w->data_offset == 0 || w->channels <= 0 ||
        w->sample_rate == 0) {
        return false;
    }
    if (w->data_size == 0 || w->data_offset + w->data_size > size) {
        w->data_size = size - w->data_offset;
    }
    return true;
}

/* decode a sample of the given source type to [-1,1] */
static double sample_at(const uint8_t *p, int bits, bool is_float, int idx)
{
    if (is_float) {
        if (bits == 32) {
            float v;

            memcpy(&v, p + 4 * (size_t)idx, 4);
            return (double)v;
        }
        if (bits == 64) {
            double v;

            memcpy(&v, p + 8 * (size_t)idx, 8);
            return v;
        }
        return 0.0;
    }
    switch (bits) {
    case 8:
        return ((double)p[idx] - 128.0) / 128.0;
    case 16: {
        int16_t v;

        memcpy(&v, p + 2 * (size_t)idx, 2);
        return (double)v / 32768.0;
    }
    case 24: {
        const uint8_t *s = p + 3 * (size_t)idx;
        int32_t v = (int32_t)(((uint32_t)s[0] << 8) | ((uint32_t)s[1] << 16) |
                              ((uint32_t)s[2] << 24));

        v >>= 8; /* sign-extend */
        return (double)v / 8388608.0;
    }
    case 32: {
        int32_t v;

        memcpy(&v, p + 4 * (size_t)idx, 4);
        return (double)v / 2147483648.0;
    }
    default:
        return 0.0;
    }
}

/* ---- AIFF/AIFC -------------------------------------------------------- */

static uint32_t aiff_rate(const uint8_t *p)
{
    /* 80-bit IEEE extended -> double (integer part only is enough for
     * audio rates) */
    int exp = (int)(((p[0] & 0x7F) << 8) | p[1]);
    uint32_t mant = ((uint32_t)p[2] << 24) | ((uint32_t)p[3] << 16) |
                    ((uint32_t)p[4] << 8) | (uint32_t)p[5];
    double v = (double)mant * pow(2.0, (double)(exp - 16383 - 31));

    if (v < 1.0 || v > 1e7) return 0;
    return (uint32_t)(v + 0.5);
}

/* ---- platform decoder (macOS) ---------------------------------------- */

#if defined(__APPLE__)
static bool decode_platform(const char *path, AudioInfo *info, float **pcm,
                            size_t *frames)
{
    CFURLRef url = CFURLCreateFromFileSystemRepresentation(
        NULL, (const UInt8 *)path, (CFIndex)strlen(path), false);
    ExtAudioFileRef file = NULL;
    AudioStreamBasicDescription in_fmt;
    AudioStreamBasicDescription out_fmt;
    OSStatus st;
    float *buf = NULL;
    size_t cap = 0, got = 0;

    if (url == NULL) return false;
    st = ExtAudioFileOpenURL(url, &file);
    CFRelease(url);
    if (st != noErr || file == NULL) return false;

    {
        UInt32 sz = sizeof(in_fmt);

        st = ExtAudioFileGetProperty(file,
                                     kExtAudioFileProperty_FileDataFormat,
                                     &sz, &in_fmt);
        if (st != noErr || in_fmt.mSampleRate <= 0) {
            ExtAudioFileDispose(file);
            return false;
        }
    }
    memset(&out_fmt, 0, sizeof(out_fmt));
    out_fmt.mSampleRate = in_fmt.mSampleRate;
    out_fmt.mFormatID = kAudioFormatLinearPCM;
    out_fmt.mFormatFlags = kAudioFormatFlagIsFloat |
                           kAudioFormatFlagIsPacked; /* interleaved */
    out_fmt.mChannelsPerFrame = in_fmt.mChannelsPerFrame > 0
                                    ? in_fmt.mChannelsPerFrame : 2;
    out_fmt.mBitsPerChannel = 32;
    out_fmt.mBytesPerFrame = 4 * out_fmt.mChannelsPerFrame;
    out_fmt.mFramesPerPacket = 1;
    out_fmt.mBytesPerPacket = out_fmt.mBytesPerFrame;
    st = ExtAudioFileSetProperty(file,
                                 kExtAudioFileProperty_ClientDataFormat,
                                 sizeof(out_fmt), &out_fmt);
    if (st != noErr) {
        ExtAudioFileDispose(file);
        return false;
    }
    {
        SInt64 total = 0;
        UInt32 sz = sizeof(total);

        (void)ExtAudioFileGetProperty(file,
                                      kExtAudioFileProperty_FileLengthFrames,
                                      &sz, &total);
        cap = (size_t)(total > 0 ? total : 0) + 4096;
    }
    buf = malloc(cap * out_fmt.mChannelsPerFrame * sizeof(float));
    if (buf == NULL) {
        ExtAudioFileDispose(file);
        return false;
    }
    for (;;) {
        AudioBufferList abl;
        UInt32 want = 65536;

        if (got + want > cap) {
            size_t ncap = cap + want + 4096;
            float *nb = realloc(buf, ncap * out_fmt.mChannelsPerFrame *
                                     sizeof(float));

            if (nb == NULL) break;
            buf = nb;
            cap = ncap;
        }
        abl.mNumberBuffers = 1;
        abl.mBuffers[0].mNumberChannels = out_fmt.mChannelsPerFrame;
        abl.mBuffers[0].mDataByteSize =
            (UInt32)(want * out_fmt.mChannelsPerFrame * sizeof(float));
        abl.mBuffers[0].mData = buf + got * out_fmt.mChannelsPerFrame;
        st = ExtAudioFileRead(file, &want, &abl);
        if (st != noErr || want == 0) break;
        got += want;
    }
    ExtAudioFileDispose(file);
    if (got == 0) {
        free(buf);
        return false;
    }
    info->sample_rate = out_fmt.mSampleRate;
    info->channels = (int)out_fmt.mChannelsPerFrame;
    info->bits = 32;
    info->is_float = true;
    info->is_compressed = true;
    info->frames = got;
    snprintf(info->codec, sizeof(info->codec), "compressed");
    *pcm = buf;
    *frames = got;
    return true;
}
#endif

/* ---- public API ------------------------------------------------------- */

bool audio_file_probe(const char *path, int force_bits, AudioInfo *info)
{
    float *pcm = NULL;
    size_t frames = 0;
    bool ok;

    if (path == NULL || info == NULL) return false;
    ok = audio_file_load(path, force_bits, info, &pcm, &frames);
    free(pcm);
    return ok;
}

bool audio_file_load(const char *path, int force_bits, AudioInfo *info,
                     float **pcm, size_t *frames)
{
    FILE *f;
    size_t size;
    uint8_t hdr[12];

    if (path == NULL || info == NULL || pcm == NULL || frames == NULL) {
        return false;
    }
    memset(info, 0, sizeof(*info));
    *pcm = NULL;
    *frames = 0;
    f = fopen(path, "rb");
    if (f == NULL) return false;
    fseek(f, 0, SEEK_END);
    size = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    if (size < 12 || fread(hdr, 1, 12, f) != 12) {
        fclose(f);
        return false;
    }
    fseek(f, 0, SEEK_SET);

    /* RIFF/WAVE */
    if (memcmp(hdr, "RIFF", 4) == 0 && memcmp(hdr + 8, "WAVE", 4) == 0) {
        WavInfo w;

        if (!wav_parse(f, size, &w)) {
            fclose(f);
            return false;
        }
        fclose(f);
        {
            bool is_float = w.format == 3 || (w.format == 1 && 0);
            int bits = force_bits > 0 ? force_bits : w.bits;
            size_t bytes_per = (size_t)((bits + 7) / 8) * (size_t)w.channels;
            size_t nframes;
            uint8_t *data;
            float *out;
            FILE *g = fopen(path, "rb");

            if (g == NULL) return false;
            if (bytes_per == 0 || w.data_size < bytes_per) {
                /* fall back to the header type when the override would
                 * not fit the payload */
                bits = w.bits;
                bytes_per = (size_t)((bits + 7) / 8) * (size_t)w.channels;
            }
            nframes = w.data_size / bytes_per;
            if (nframes == 0) {
                fclose(g);
                return false;
            }
            data = malloc(nframes * bytes_per);
            out = malloc(nframes * (size_t)w.channels * sizeof(float));
            if (data == NULL || out == NULL) {
                free(data);
                free(out);
                fclose(g);
                return false;
            }
            if (fseek(g, (long)w.data_offset, SEEK_SET) != 0 ||
                fread(data, 1, nframes * bytes_per, g) !=
                    nframes * bytes_per) {
                free(data);
                free(out);
                fclose(g);
                return false;
            }
            fclose(g);
            for (size_t i = 0; i < nframes * (size_t)w.channels; i++) {
                out[i] = (float)sample_at(data, bits, is_float, (int)i);
            }
            free(data);
            info->sample_rate = (double)w.sample_rate;
            info->channels = w.channels;
            info->bits = bits;
            info->is_float = is_float;
            info->is_compressed = false;
            snprintf(info->codec, sizeof(info->codec), "PCM");
            info->frames = nframes;
            *pcm = out;
            *frames = nframes;
            return true;
        }
    }
    /* AIFF/AIFC */
    if (memcmp(hdr, "FORM", 4) == 0 && (memcmp(hdr + 8, "AIFF", 4) == 0 ||
                                        memcmp(hdr + 8, "AIFC", 4) == 0)) {
        fclose(f);
        {
            /* minimal: reuse the WAV reader's structure by scanning chunks */
            FILE *g = fopen(path, "rb");
            uint8_t ch[8];
            int channels = 0, bits = 0;
            uint32_t rate = 0;
            size_t data_off = 0, data_size = 0;
            bool have_comm = false;

            if (g == NULL) return false;
            fseek(g, 12, SEEK_SET);
            {
                size_t pos = 12;

                while (pos + 8 <= size) {
                    if (fread(ch, 1, 8, g) != 8) break;
                    {
                        uint32_t clen = rd_be32(ch + 4);

                        if (memcmp(ch, "COMM", 4) == 0 && clen >= 18) {
                            uint8_t b[32];
                            uint32_t n = clen < sizeof(b) ? clen
                                                          : sizeof(b);

                            if (fread(b, 1, n, g) != n) break;
                            channels = (int)((b[0] << 8) | b[1]);
                            rate = aiff_rate(b + 8);
                            bits = (int)((b[6] << 8) | b[7]);
                            have_comm = true;
                            if (clen > n &&
                                fseek(g, (long)(clen - n), SEEK_CUR) != 0) {
                                break;
                            }
                        } else if (memcmp(ch, "SSND", 4) == 0) {
                            data_off = pos + 8 + 8; /* offset+blocksize */
                            if (data_off > size) break;
                            data_size = size - data_off;
                            break;
                        } else if (fseek(g, (long)clen + (long)(clen & 1),
                                         SEEK_CUR) != 0) {
                            break;
                        }
                        pos += 8 + clen + (clen & 1);
                    }
                }
            }
            if (!have_comm || data_off == 0 || channels <= 0) {
                fclose(g);
                return false;
            }
            {
                int b = force_bits > 0 ? force_bits : bits;
                size_t bytes_per = (size_t)((b + 7) / 8) * (size_t)channels;
                size_t nframes = bytes_per > 0 ? data_size / bytes_per : 0;
                uint8_t *data;
                float *out;

                if (nframes == 0) {
                    fclose(g);
                    return false;
                }
                data = malloc(nframes * bytes_per);
                out = malloc(nframes * (size_t)channels * sizeof(float));
                if (data == NULL || out == NULL) {
                    free(data);
                    free(out);
                    fclose(g);
                    return false;
                }
                if (fseek(g, (long)data_off, SEEK_SET) != 0 ||
                    fread(data, 1, nframes * bytes_per, g) !=
                        nframes * bytes_per) {
                    free(data);
                    free(out);
                    fclose(g);
                    return false;
                }
                fclose(g);
                /* AIFF samples are big-endian: byte-swap into the reader */
                for (size_t i = 0; i < nframes * (size_t)channels; i++) {
                    uint8_t *p = data + i * (size_t)((b + 7) / 8);
                    int nb = (b + 7) / 8;

                    for (int k = 0; k < nb / 2; k++) {
                        uint8_t t = p[k];

                        p[k] = p[nb - 1 - k];
                        p[nb - 1 - k] = t;
                    }
                    out[i] = (float)sample_at(p, b, false, 0);
                }
                free(data);
                info->sample_rate = (double)rate;
                info->channels = channels;
                info->bits = b;
                info->is_float = false;
                info->codec[0] = '\0';
                snprintf(info->codec, sizeof(info->codec), "PCM (AIFF)");
                info->frames = nframes;
                *pcm = out;
                *frames = nframes;
                return true;
            }
        }
    }
    fclose(f);
#if defined(__APPLE__)
    /* compressed formats through the platform decoder */
    return decode_platform(path, info, pcm, frames);
#else
    return false;
#endif
}
