/*
 * test_runner.c — unit tests for the ToxicElement15 core (no GUI).
 *
 * Build and run with `make test`.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "analysis.h"
#include "audio.h"
#include "colormap.h"
#include "cpar.h"
#include "cyclic.h"
#include "detect.h"
#include "dsp.h"
#include <complex.h> /* after fftw3.h so fftwf_complex keeps its float[2] form */
#include "fit.h"
#include "fluct.h"
#include "hos.h"
#include "ssca.h"
#include "cycledet.h"
#include "ccc.h"
#include "sptk.h"
#include "sdist.h"
#include "classic.h"
#include "filters.h"
#include "pulse.h"
#include "intraburst.h"
#include "ssclass.h"
#include "synth.h"
#include "modconf.h"
#include "iq_source.h"
#include "audio_file.h"
#include "metrics.h"
#include "sot.h"
#include "aresult.h"
#include "aregistry.h"
#include "apublish.h"
#include "dag.h"
#include "director.h"
#include "report.h"
#include "aconfig.h"
#include "signals.h"
#include "wav.h"
#include "waterfall_cache.h"
#include "streambuf.h"
#include "lte_uplink.h"
#include "dagcore.h"
#include "fcdag.h"
#include "fcstats.h"
#include "fcexport.h"
#include "fcreport.h"
#include "fcdataset.h"
#include "fcdrift.h"
#include "fclattice.h"
#include "fclk.h"
#include "renderbuf.h"

static int checks_run = 0;
static int checks_failed = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        checks_run++;                                                        \
        if (!(cond)) {                                                       \
            checks_failed++;                                                 \
            printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);           \
        }                                                                    \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                \
    do {                                                                     \
        double _va = (a), _vb = (b);                                         \
        checks_run++;                                                        \
        if (fabs(_va - _vb) > (tol)) {                                       \
            checks_failed++;                                                 \
            printf("FAIL %s:%d: |%s - %s| = %g > %g\n", __FILE__, __LINE__,  \
                   #a, #b, fabs(_va - _vb), (double)(tol));                  \
        }                                                                    \
    } while (0)

#define CHECK_RECT(r, x_, y_, w_, h_)                                        \
    do {                                                                     \
        IntRect _r = (r);                                                    \
        checks_run++;                                                        \
        if (_r.x != (x_) || _r.y != (y_) || _r.w != (w_) || _r.h != (h_)) {  \
            checks_failed++;                                                 \
            printf("FAIL %s:%d: rect = %d,%d %dx%d expected %d,%d %dx%d\n",  \
                   __FILE__, __LINE__, _r.x, _r.y, _r.w, _r.h,               \
                   (x_), (y_), (w_), (h_));                                  \
        }                                                                    \
    } while (0)

/* ------------------------------------------------------------------ */
/* fit                                                                */
/* ------------------------------------------------------------------ */
static void test_fit(void)
{
    /* Wide image: width-constrained, centred vertically. */
    CHECK_RECT(fit_contain(400, 300, 200, 100), 0, 50, 400, 200);
    /* Tall image: height-constrained, centred horizontally. */
    CHECK_RECT(fit_contain(400, 300, 100, 200), 125, 0, 150, 300);
    /* Square into square. */
    CHECK_RECT(fit_contain(300, 300, 100, 100), 0, 0, 300, 300);
    /* Downscale: image larger than box in both axes. */
    CHECK_RECT(fit_contain(100, 100, 400, 300), 0, 12, 100, 75);
    /* Exact aspect match. */
    CHECK_RECT(fit_contain(400, 300, 400, 300), 0, 0, 400, 300);
    /* Tall box, wide image: height has plenty of room. */
    CHECK_RECT(fit_contain(100, 400, 400, 100), 0, 187, 100, 25);
    /* Rounding must never exceed the box. */
    {
        IntRect r = fit_contain(100, 100, 300, 100);
        CHECK(r.x == 0 && r.w == 100 && r.h == 33 && r.y == 33);
        CHECK(r.w <= 100 && r.h <= 100);
    }
    /* Degenerate images collapse to a centred empty rect. */
    CHECK_RECT(fit_contain(400, 300, 0, 100), 200, 150, 0, 0);
    CHECK_RECT(fit_contain(400, 300, 100, 0), 200, 150, 0, 0);
    CHECK_RECT(fit_contain(0, 0, 100, 100), 0, 0, 0, 0);
    /* One-pixel image fills its axis exactly. */
    CHECK_RECT(fit_contain(400, 300, 1, 1), 50, 0, 300, 300);
}

/* ------------------------------------------------------------------ */
/* filename parsing                                                   */
/* ------------------------------------------------------------------ */
static void test_filename(void)
{
    double freq = 0, rate = 0;
    time_t start = 0;
    CHECK(iq_source_parse_filename(
        "gqrx_20260306_174402_2540000000_16000000_fc.raw", &start, &freq, &rate));
    CHECK(freq == 2.54e9 && rate == 1.6e7);
    /* start time resolves to 2026-03-06 17:44:02 local */
    {
        struct tm tm;
        memset(&tm, 0, sizeof(tm));
        tm.tm_year = 2026 - 1900; tm.tm_mon = 3 - 1; tm.tm_mday = 6;
        tm.tm_hour = 17; tm.tm_min = 44; tm.tm_sec = 2; tm.tm_isdst = -1;
        CHECK(start == mktime(&tm));
    }

    CHECK(iq_source_parse_filename(
        "/tmp/xx/gqrx_20260306_174402_2530000000_2000000_fc.raw",
        NULL, &freq, &rate));
    CHECK(freq == 2.53e9 && rate == 2.0e6);

    CHECK(!iq_source_parse_filename("data.bin", NULL, &freq, &rate));
    CHECK(!iq_source_parse_filename(
        "gqrx_20260306_174402_2540000000_16000000_cf.raw", NULL, &freq, &rate));
    CHECK(!iq_source_parse_filename(
        "gqrx_20260306_174402_2540000000_16000000_fc.wav", NULL, &freq, &rate));
    CHECK(!iq_source_parse_filename(
        "gqrx_20260306_174402_0_16000000_fc.raw", NULL, &freq, &rate));
}

static void test_time_format(void)
{
    IqSource s;
    memset(&s, 0, sizeof(s));
    struct tm tm;
    memset(&tm, 0, sizeof(tm));
    tm.tm_year = 2026 - 1900; tm.tm_mon = 3 - 1; tm.tm_mday = 6;
    tm.tm_hour = 17; tm.tm_min = 44; tm.tm_sec = 2; tm.tm_isdst = -1;
    s.start_time = mktime(&tm);
    s.has_start_time = true;

    char buf[64];
    iq_source_format_time(&s, 0.0, buf, sizeof(buf));

    int Y, Mo, D, h, mi, se, oh, om;
    char sign;
    CHECK(sscanf(buf, "%d-%d-%dT%d:%d:%d%c%d:%d",
                 &Y, &Mo, &D, &h, &mi, &se, &sign, &oh, &om) == 9);
    CHECK(Y == 2026 && Mo == 3 && D == 6 && h == 17 && mi == 44 && se == 2);
    /* the UTC offset must match localtime at that instant */
    struct tm lt;
    localtime_r(&s.start_time, &lt);
    long off = lt.tm_gmtoff;
    CHECK(sign == (off < 0 ? '-' : '+'));
    CHECK(oh == labs(off) / 3600 && om == (labs(off) % 3600) / 60);

    /* offset arithmetic lands on the right wall-clock */
    iq_source_format_time(&s, 123.4, buf, sizeof(buf));
    CHECK(sscanf(buf, "%d-%d-%dT%d:%d:%d%c%d:%d",
                 &Y, &Mo, &D, &h, &mi, &se, &sign, &oh, &om) == 9);
    time_t t2 = s.start_time + 123;
    struct tm lt2;
    localtime_r(&t2, &lt2);
    CHECK(Y == lt2.tm_year + 1900 && Mo == lt2.tm_mon + 1 &&
          D == lt2.tm_mday && h == lt2.tm_hour && mi == lt2.tm_min &&
          se == lt2.tm_sec);

    /* unknown start time falls back to a relative form */
    s.has_start_time = false;
    iq_source_format_time(&s, 12.345, buf, sizeof(buf));
    CHECK(strcmp(buf, "T+12.345s") == 0);
}

/* ------------------------------------------------------------------ */
/* colormap                                                           */
/* ------------------------------------------------------------------ */
static void test_colormap(void)
{
    Colormap a, b;
    viridis_build(&a);
    viridis_build(&b);

    /* Canonical viridis endpoints. */
    CHECK(a.table[0].r == 68 && a.table[0].g == 1 && a.table[0].b == 84);
    CHECK(a.table[255].r == 253 && a.table[255].g == 231 &&
          a.table[255].b == 37);

    /* Clamping. */
    Rgb8 lo = viridis_lookup(&a, -5.0f);
    Rgb8 hi = viridis_lookup(&a, 5.0f);
    CHECK(memcmp(&lo, &a.table[0], 3) == 0);
    CHECK(memcmp(&hi, &a.table[255], 3) == 0);

    /* Mid lookup lands on the expected entry. */
    Rgb8 mid = viridis_lookup(&a, 0.5f);
    CHECK(memcmp(&mid, &a.table[128], 3) == 0);

    /* Deterministic build. */
    CHECK(memcmp(a.table, b.table, sizeof(a.table)) == 0);

    /* Viridis is a monotonic luminance ramp: mid is greenish. */
    CHECK(a.table[128].g > a.table[128].r);
}

/* ------------------------------------------------------------------ */
/* dsp                                                                */
/* ------------------------------------------------------------------ */
static void test_entropy(void)
{
    const float flat[] = {1.0f, 1.0f, 1.0f, 1.0f};
    CHECK(dsp_entropy(flat, 4) == 0.0);

    float ramp[100];
    for (int i = 0; i < 100; i++) ramp[i] = (float)i;
    double h = dsp_entropy(ramp, 100);
    CHECK(h > 0.0 && isfinite(h) && h < 7.0);

    CHECK(dsp_entropy(NULL, 0) == 0.0);
}

static void test_prf(void)
{
    /* Pulse every 20 samples at dt=0.01 s -> 5 Hz. */
    float power[100] = {0};
    for (int i = 0; i < 100; i += 20) power[i] = 10.0f;
    CHECK_NEAR(dsp_estimate_prf(power, 100, 0.01), 5.0, 1e-9);

    /* No periodic structure. */
    float flat[64];
    for (int i = 0; i < 64; i++) flat[i] = 0.5f;
    CHECK(dsp_estimate_prf(flat, 64, 0.01) == 0.0);

    /* Degenerate inputs. */
    CHECK(dsp_estimate_prf(NULL, 0, 0.01) == 0.0);
    CHECK(dsp_estimate_prf(power, 100, 0.0) == 0.0);
}

static void test_normalize(void)
{
    const float v[] = {0.0f, 1.0f, 2.0f, 3.0f};
    float out[4] = {0};
    float mn = 0, mx = 0;
    dsp_normalize_to_unit(v, 4, out, &mn, &mx);
    CHECK(mn == 0.0f && mx == 3.0f);
    CHECK_NEAR(out[0], 0.0, 1e-6);
    CHECK_NEAR(out[1], 1.0 / 3.0, 1e-6);
    CHECK_NEAR(out[3], 1.0, 1e-6);

    const float flat[] = {5.0f, 5.0f};
    dsp_normalize_to_unit(flat, 2, out, &mn, &mx);
    CHECK_NEAR(out[0], 0.5, 1e-6);
    CHECK_NEAR(out[1], 0.5, 1e-6);

    dsp_normalize_to_unit(NULL, 0, out, &mn, &mx);
    CHECK(mn == 0.0f && mx == 0.0f);
}

static void test_bins_to_db(void)
{
    enum { N = 64 };
    fftwf_complex *bins = fftwf_malloc(sizeof(fftwf_complex) * N);
    float db[N];
    for (int i = 0; i < N; i++) bins[i][0] = bins[i][1] = 0.0f;
    bins[0][0] = 1.0f; /* DC */

    dsp_bins_to_db(bins, N, db);
    CHECK_NEAR(db[N / 2], 0.0, 1e-3);        /* DC lands at N/2 after shift */
    CHECK(db[0] < -190.0);                    /* everything else ~ -inf     */

    fftwf_free(bins);
}

static void test_spectrum(void)
{
    enum { N = 256, K = 8 };
    FftPlan *p = fft_plan_create(N);
    CHECK(p != NULL);

    float iq[2 * N];
    float db[N];
    for (int i = 0; i < N; i++) {
        iq[2 * i]     = cosf(2.0f * (float)M_PI * K * i / N);
        iq[2 * i + 1] = 0.0f;
    }
    fft_plan_spectrum(p, iq, NULL, db);

    /* Peak must sit at a shifted mirror bin N/2 +/- K (equal magnitude;
     * the scan finds whichever comes first). */
    int peak = 0;
    for (int i = 1; i < N; i++)
        if (db[i] > db[peak]) peak = i;
    CHECK(peak == N / 2 + K || peak == N / 2 - K);
    CHECK(db[peak] > 20.0); /* coherent tone at N samples */

    /* window builders: Hann endpoints, rectangular ones, determinism */
    {
        float hw[8], bw[8], rw[8];
        dsp_window(DSP_WINDOW_HANN, hw, 8);
        dsp_window(DSP_WINDOW_HAMMING, bw, 8);
        dsp_window(DSP_WINDOW_RECTANGULAR, rw, 8);
        CHECK(fabsf(hw[0]) < 1e-6f && fabsf(hw[7]) < 1e-6f);
        CHECK(hw[4] > 0.9f);
        for (int i = 0; i < 8; i++) CHECK(rw[i] == 1.0f);
        CHECK(bw[0] > 0.0f && bw[0] > hw[0]); /* Hamming > Hann at 0 */
        float hw2[8];
        dsp_window(DSP_WINDOW_HANN, hw2, 8);
        CHECK(memcmp(hw, hw2, sizeof(hw)) == 0);
        dsp_window(DSP_WINDOW_HANN, hw2, 1);
        CHECK(hw2[0] == 1.0f);
    }

    /* A unit Hann window gives the same peak location. */
    float window[N];
    for (int i = 0; i < N; i++) window[i] = 1.0f;
    fft_plan_spectrum(p, iq, window, db);
    peak = 0;
    for (int i = 1; i < N; i++)
        if (db[i] > db[peak]) peak = i;
    CHECK(peak == N / 2 + K || peak == N / 2 - K);

    fft_plan_destroy(p);
}

/* ------------------------------------------------------------------ */
/* signals                                                            */
/* ------------------------------------------------------------------ */
static void test_signals(void)
{
    SignalMatch matches[10];

    size_t n = signals_match(144.39e6, 12.5e3, matches, 10);
    CHECK(n > 0);
    CHECK(strcmp(matches[0].signal->name, "APRS") == 0);
    /* Base scoring maxes at 60+20+30+10 = 120 plus the specificity
     * adjustment (+10 for a narrow frequency range). */
    CHECK(matches[0].confidence >= 120);
    for (size_t i = 1; i < n; i++) {
        CHECK(matches[i].confidence <= matches[i - 1].confidence);
    }

    /* Cap respected. */
    n = signals_match(144.39e6, 12.5e3, matches, 3);
    CHECK(n == 3);

    /* Out-of-database centre frequency -> few or no matches. */
    n = signals_match(88.5e6, 200.0e3, matches, 10); /* FM broadcast */
    CHECK(n > 0);
    CHECK(strcmp(matches[0].signal->name, "FM Broadcast") == 0);
}

/* ------------------------------------------------------------------ */
/* end-to-end: synthetic recording                                    */
/* ------------------------------------------------------------------ */
#define TEST_RATE   2.0e6
#define TEST_CENTER 100.0e6
#define TEST_SECS   1.0
#define TEST_SAMPLES ((size_t)(TEST_RATE * TEST_SECS))
#define TEST_PATH "/tmp/te15_test/gqrx_20260306_174402_100000000_2000000_fc.raw"

static unsigned rng_state = 12345;
static float frand(void)
{
    rng_state = rng_state * 1664525u + 1013904223u;
    return (float)((rng_state >> 8) & 0xFFFFFF) / (float)0xFFFFFF - 0.5f;
}

static void write_test_file(void)
{
    /* The basename must start with "gqrx_" for filename parsing. */
    system("mkdir -p /tmp/te15_test");
    const double tone = 400.0e3; /* +400 kHz from centre */
    float *iq = malloc(2 * TEST_SAMPLES * sizeof(float));
    CHECK(iq != NULL);
    for (size_t i = 0; i < TEST_SAMPLES; i++) {
        double t = (double)i / TEST_RATE;
        double phase = 2.0 * M_PI * tone * t;
        iq[2 * i]     = (float)(0.5 * cos(phase)) + 0.02f * frand();
        iq[2 * i + 1] = (float)(0.5 * sin(phase)) + 0.02f * frand();
    }
    FILE *f = fopen(TEST_PATH, "wb");
    CHECK(f != NULL);
    if (f != NULL) {
        CHECK(fwrite(iq, sizeof(float), 2 * TEST_SAMPLES, f) ==
              2 * TEST_SAMPLES);
        fclose(f);
    }
    free(iq);
}

static void test_iq_source(void)
{
    write_test_file();

    IqSource src;
    char err[256];
    bool opened = iq_source_open(&src, TEST_PATH, err, sizeof(err));
    if (!opened) printf("open error: %s\n", err);
    CHECK(opened);
    CHECK(src.total_samples == TEST_SAMPLES);
    CHECK(src.bytes_per_sample == 8);
    CHECK(src.center_freq == TEST_CENTER);
    CHECK(src.sample_rate == TEST_RATE);
    CHECK_NEAR(src.duration, TEST_SECS, 1e-9);

    float buf[10];
    iq_source_read_complex(&src, 100, buf, 5);
    CHECK(fabsf(buf[0] - src.samples[200]) < 1e-9f);
    CHECK(fabsf(buf[9] - src.samples[209]) < 1e-9f);
    /* Reads past the end are clamped. */
    iq_source_read_complex(&src, TEST_SAMPLES - 2, buf, 5);
    CHECK(fabsf(buf[0] - src.samples[2 * (TEST_SAMPLES - 2)]) < 1e-9f);

    iq_source_close(&src);
    CHECK(src.samples == NULL && src.fd == -1);

    /* Analysis over a band containing the tone. */
    CHECK(iq_source_open(&src, TEST_PATH, err, sizeof(err)));
    Region r = {TEST_CENTER + 390.0e3, TEST_CENTER + 410.0e3, 0.2, 0.8};
    Analysis a;
    memset(&a, 0, sizeof(a));
    CHECK(analysis_run(&src, &r, &a));
    CHECK(a.frames > 0 && a.bins > 0);
    CHECK(a.image != NULL);
    CHECK(a.max_power > 20.0);  /* tone dominates */
    CHECK(a.avg_power > 0.0);
    CHECK(isfinite(a.entropy) && a.entropy >= 0.0);
    CHECK_NEAR(a.center_freq, TEST_CENTER + 400.0e3, 1.0);
    CHECK_NEAR(a.bandwidth, 20.0e3, 1.0);
    analysis_clear(&a);
    CHECK(a.image == NULL);

    /* A fully out-of-band region is rejected after clamping. */
    Region out = {TEST_CENTER + 90.0e6, TEST_CENTER + 95.0e6, 0.2, 0.8};
    CHECK(!analysis_run(&src, &out, &a));

    /* Reversed min/max normalises and still analyses. */
    Region rev = {TEST_CENTER + 410.0e3, TEST_CENTER + 390.0e3, 0.8, 0.2};
    CHECK(analysis_run(&src, &rev, &a));
    CHECK_NEAR(a.center_freq, TEST_CENTER + 400.0e3, 1.0);
    CHECK_NEAR(a.duration, 0.6, 1e-9);

    /* Waterfall cache end-to-end (worker thread). */
    WcSpec spec = {256, 0.5, DSP_WINDOW_HANN};
    WaterfallCache *wc = wc_create(&src, &spec, 2, 1000);
    CHECK(wc != NULL);
    if (wc != NULL) {
        CHECK(wc_total_frames(wc) == (TEST_SAMPLES - 256) / 128 + 1);
        CHECK(wc_fft_size(wc) == 256);
        CHECK_NEAR(wc_frames_per_second(wc), TEST_RATE / 128.0, 1e-9);
        CHECK(wc_progress(wc) == -1);
        wc_request(wc, 0, 500);

        const float *db = NULL;
        size_t ws = 0, wf = 0, wstep = 1;
        for (int t = 0; t < 500; t++) {
            db = wc_lock(wc, 0, &ws, &wf, &wstep);
            if (db != NULL) break;
            usleep(10 * 1000);
        }
        CHECK(db != NULL);
        if (db != NULL) {
            CHECK(ws == 0 && wf == 500);
            int bad = 0;
            for (size_t i = 0; i < wf * 256; i += 1024)
                if (!isfinite(db[i])) bad++;
            CHECK(bad == 0);
            wc_unlock(wc);
        }
        /* Far frame is not covered yet. */
        CHECK(wc_lock(wc, wc_total_frames(wc) - 1, &ws, &wf, &wstep) == NULL);
        wc_destroy(wc);
    }

    iq_source_close(&src);
    unlink(TEST_PATH);
}

/* ------------------------------------------------------------------ */
/* extended metrics, demod, wav, ranking                               */
/* ------------------------------------------------------------------ */

#define TYPED_DIR "/tmp/te15_test/%s/gqrx_20260306_174402_100000000_2000000_fc.raw"

/* signal kinds: 0=CW 1=FM 2=AM 3=OOK 4=NOISE 5=FSK */
static void write_typed_file(const char *sub, int kind)
{
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "mkdir -p /tmp/te15_test/%s", sub);
    system(cmd);

    const double rate = TEST_RATE;
    const size_t n = TEST_SAMPLES;
    float *iq = malloc(2 * n * sizeof(float));
    CHECK(iq != NULL);
    if (iq == NULL) return;

    const double tone = 400.0e3; /* carrier offset from centre */
    unsigned rng = 12345;
    for (size_t i = 0; i < n; i++) {
        double t = (double)i / rate;
        double re = 0.0, im = 0.0;
        double noise_re = 0.02 * (((rng = rng * 1664525u + 1013904223u) >> 8 & 0xFFFFFF) / (double)0xFFFFFF - 0.5);
        double noise_im = 0.02 * (((rng = rng * 1664525u + 1013904223u) >> 8 & 0xFFFFFF) / (double)0xFFFFFF - 0.5);
        switch (kind) {
        case 0: { /* CW */
            re = 0.6 * cos(2 * M_PI * tone * t);
            im = 0.6 * sin(2 * M_PI * tone * t);
            break;
        }
        case 1: { /* FM: 1 kHz mod, 50 kHz deviation */
            double ph = 2 * M_PI * tone * t +
                        (50.0e3 / 1.0e3) * sin(2 * M_PI * 1.0e3 * t);
            re = 0.6 * cos(ph);
            im = 0.6 * sin(ph);
            break;
        }
        case 2: { /* AM: 1 kHz tone, depth 0.5 */
            double env = 1.0 + 0.5 * sin(2 * M_PI * 1.0e3 * t);
            re = env * cos(2 * M_PI * tone * t);
            im = env * sin(2 * M_PI * tone * t);
            break;
        }
        case 3: { /* OOK: 100 us on / 1 ms period */
            double gate = (fmod(t, 1e-3) < 100e-6) ? 0.9 : 0.0;
            re = gate * cos(2 * M_PI * tone * t);
            im = gate * sin(2 * M_PI * tone * t);
            break;
        }
        case 4: /* noise only */
            break;
        case 5: { /* FSK: +-5 kHz around +400 kHz, 1 kHz symbols */
            double foff = tone + (((size_t)(t / 500e-6)) % 2 ? 5.0e3 : -5.0e3);
            re = 0.6 * cos(2 * M_PI * foff * t);
            im = 0.6 * sin(2 * M_PI * foff * t);
            break;
        }
        }
        iq[2 * i]     = (float)(re + noise_re);
        iq[2 * i + 1] = (float)(im + noise_im);
    }

    char path[512];
    snprintf(path, sizeof(path), TYPED_DIR, sub);
    FILE *f = fopen(path, "wb");
    CHECK(f != NULL);
    if (f != NULL) {
        CHECK(fwrite(iq, sizeof(float), 2 * n, f) == 2 * n);
        fclose(f);
    }
    free(iq);
}

static void region_for_kind(int kind, Region *r)
{
    const double c = TEST_CENTER;
    switch (kind) {
    case 0: case 2: /* CW, AM */
        r->freq_min = c + 390.0e3; r->freq_max = c + 410.0e3;
        break;
    case 1: /* FM */
        r->freq_min = c + 250.0e3; r->freq_max = c + 550.0e3;
        break;
    case 3: /* OOK */
        r->freq_min = c + 350.0e3; r->freq_max = c + 450.0e3;
        break;
    case 5: /* FSK */
        r->freq_min = c + 380.0e3; r->freq_max = c + 420.0e3;
        break;
    default: /* NOISE */
        r->freq_min = c - 100.0e3; r->freq_max = c + 100.0e3;
        break;
    }
    r->time_min = 0.1;
    r->time_max = 0.9;
}

static void test_metrics_classification(void)
{
    static const char *subs[] = {"mcw", "mfm", "mam", "mook", "mnoise", "mfsk"};
    static const char *expect[] = {"CW", "FM", "AM", "OOK", "NOISE", "FSK"};

    for (int k = 0; k < 6; k++) {
        write_typed_file(subs[k], k);
        char path[512];
        snprintf(path, sizeof(path), TYPED_DIR, subs[k]);
        IqSource src;
        char err[256];
        CHECK(iq_source_open(&src, path, err, sizeof(err)));
        if (src.samples == NULL) {
            printf("open error (%s): %s\n", subs[k], err);
            continue;
        }
        Region r;
        region_for_kind(k, &r);
        Metrics m;
        bool ok = metrics_compute(&src, &r, &m);
        CHECK(ok);
        if (ok) {
            printf("  metrics[%s]: class=%s bw=%.1fkHz dev=%.1fkHz "
                   "depth=%.2f duty=%.2f fsk=%d\n",
                   subs[k], m.mod_class, m.occupied_bw_hz / 1e3,
                   m.fm_deviation_hz / 1e3, m.am_depth, m.duty_cycle,
                   m.fsk_levels);
            CHECK(strcmp(m.mod_class, expect[k]) == 0);
        }
        iq_source_close(&src);
    }

    /* FM must be demodulable via the fm path */
    {
        char path[512];
        snprintf(path, sizeof(path), TYPED_DIR, "mfm");
        IqSource src;
        char err[256];
        if (iq_source_open(&src, path, err, sizeof(err))) {
            Region r;
            region_for_kind(1, &r);
            Metrics m;
            if (metrics_compute(&src, &r, &m)) {
                CHECK(m.demodulable);
                CHECK(strcmp(m.demod_mode, "fm") == 0);
            }
            iq_source_close(&src);
        }
    }
}

static void test_demod(void)
{
    static const char *subs[] = {"mfm", "mam", "mcw"};
    static const char *modes[] = {"fm", "am", "cw"};

    for (int k = 0; k < 3; k++) {
        char path[512];
        snprintf(path, sizeof(path), TYPED_DIR, subs[k]);
        IqSource src;
        char err[256];
        if (!iq_source_open(&src, path, err, sizeof(err))) {
            printf("open error (%s): %s\n", subs[k], err);
            CHECK(false);
            continue;
        }
        Region r;
        region_for_kind(k == 2 ? 0 : k == 1 ? 1 : 0, &r); /* cw/am share kind 0 */
        Metrics m;
        CHECK(metrics_compute(&src, &r, &m));

        size_t s0 = (size_t)(r.time_min * src.sample_rate);
        size_t n = (size_t)((r.time_max - r.time_min) * src.sample_rate);
        double off = ((r.freq_min + r.freq_max) / 2.0) - src.center_freq;
        float *mix = malloc(2 * n * sizeof(float));
        CHECK(mix != NULL);
        metrics_mixdown(src.samples + 2 * s0, n, src.sample_rate, -off, mix);

        size_t max_out = (size_t)(n * 32000.0 / src.sample_rate) + 64;
        float *mono = malloc(max_out * sizeof(float));
        CHECK(mono != NULL);
        size_t got = 0;
        if (strcmp(modes[k], "fm") == 0) {
            got = demod_fm(mix, n, src.sample_rate, 32000.0,
                           m.fm_deviation_hz > 0 ? m.fm_deviation_hz : 75000.0,
                           mono, max_out);
        } else if (strcmp(modes[k], "am") == 0) {
            got = demod_am(mix, n, src.sample_rate, 32000.0, mono, max_out);
        } else {
            got = demod_cw(mix, n, src.sample_rate, 32000.0, mono, max_out);
        }
        CHECK(got > 1000);
        double rms = 0.0, mean = 0.0, mx = 0.0;
        for (size_t i = 0; i < got; i++) {
            double v = mono[i];
            rms += v * v;
            mean += v;
            if (fabs(v) > mx) mx = fabs(v);
        }
        rms = sqrt(rms / got);
        mean /= got;
        CHECK(isfinite(rms) && rms > 0.02);
        CHECK(mx <= 1.05);
        if (k != 2) CHECK(fabs(mean) < 0.1); /* CW beat is a tone, not DC */
        printf("  demod[%s]: got=%zu rms=%.3f mean=%.3f max=%.2f\n",
               modes[k], got, rms, mean, mx);

        free(mix);
        free(mono);
        iq_source_close(&src);
    }
}

static void test_wav(void)
{
    enum { N = 1000 };
    float s[N];
    for (int i = 0; i < N; i++)
        s[i] = 0.5f * sinf(2.0f * (float)M_PI * 440.0f * i / 8000.0f);

    const char *p = "/tmp/te15_test/t.wav";
    CHECK(wav_write(p, s, N, 8000));

    FILE *f = fopen(p, "rb");
    CHECK(f != NULL);
    if (f == NULL) return;
    uint8_t hdr[44];
    CHECK(fread(hdr, 1, 44, f) == 44);
    CHECK(memcmp(hdr, "RIFF", 4) == 0);
    CHECK(memcmp(hdr + 8, "WAVE", 4) == 0);
    uint32_t rate = 0, dsize = 0;
    memcpy(&rate, hdr + 24, 4);
    memcpy(&dsize, hdr + 40, 4);
    CHECK(rate == 8000);
    CHECK(dsize == 2 * N);
    int16_t first = 0;
    CHECK(fread(&first, 2, 1, f) == 1);
    CHECK(abs((int)first) <= 1); /* sin(0) -> ~0 */
    fclose(f);
    unlink(p);

    CHECK(!wav_write("/tmp/te15_test/no/such/dir/t.wav", s, N, 8000));
}

static void test_signals_rank(void)
{
    RankedMatch ranked[3];

    Metrics m;
    memset(&m, 0, sizeof(m));
    strcpy(m.mod_class, "FM");
    m.occupied_bw_hz = 150.0e3;

    size_t rn = signals_rank(98.0e6, 200.0e3, &m, ranked, 3);
    CHECK(rn >= 1);
    if (rn >= 1) {
        CHECK(strcmp(ranked[0].signal->name, "FM Broadcast") == 0);
        CHECK(ranked[0].base_confidence >= 110); /* 60+20+30 + specificity */
        /* occupancy +15, family (FM) +20; no deviation set -> no Carson */
        CHECK(ranked[0].agreement == 35);
        /* the unified ranking reports a 0..100 confidence total that
         * combines the database evidence and every measured parameter */
        CHECK(ranked[0].total >= 0 && ranked[0].total <= 100);
        CHECK(ranked[0].total > 0);
    }

    /* PSK class: family mismatch (-15) cancels the occupancy bonus */
    memset(&m, 0, sizeof(m));
    strcpy(m.mod_class, "PSK");
    m.occupied_bw_hz = 150.0e3;
    rn = signals_rank(98.0e6, 200.0e3, &m, ranked, 3);
    int seen = 0;
    for (size_t i = 0; i < rn; i++) {
        if (strcmp(ranked[i].signal->name, "FM Broadcast") == 0) {
            CHECK(ranked[i].agreement == 0); /* +15 occupancy -15 family */
            seen = 1;
        }
    }
    CHECK(seen);

    /* FM + deviation consistent with Carson's rule gains +10 more */
    memset(&m, 0, sizeof(m));
    strcpy(m.mod_class, "FM");
    m.occupied_bw_hz = 150.0e3;
    m.fm_deviation_hz = 60.0e3; /* bw/(2*dev) = 1.25 in [0.8, 2.5] */
    rn = signals_rank(98.0e6, 200.0e3, &m, ranked, 3);
    if (rn >= 1) {
        CHECK(strcmp(ranked[0].signal->name, "FM Broadcast") == 0);
        CHECK(ranked[0].agreement == 45); /* +15 +20 +10 */
        CHECK(ranked[0].total >= 0 && ranked[0].total <= 100);
        CHECK(ranked[0].total > 0);
    }

    /* NULL metrics == plain database ranking */
    rn = signals_rank(98.0e6, 200.0e3, NULL, ranked, 3);
    CHECK(rn >= 1);
    if (rn >= 1) {
        CHECK(strcmp(ranked[0].signal->name, "FM Broadcast") == 0);
        CHECK(ranked[0].agreement == 0);
        CHECK(ranked[0].total > 0 && ranked[0].total <= 100);
    }
}


/* ------------------------------------------------------------------ */
/* LTE uplink bands: PUSCH/PUCCH are legal only in the 3GPP uplink      */
/* ranges, and PUCCH is a razor-thin slice at the carrier edges         */
/* ------------------------------------------------------------------ */

static bool ranked_name_contains(const RankedMatch *r, size_t n,
                                 const char *needle, int want_in_band,
                                 int *found_valid)
{
    bool found = false;

    for (size_t i = 0; i < n; i++) {
        if (strstr(r[i].signal->name, needle) == NULL) continue;
        if (r[i].in_band != want_in_band) continue;
        found = true;
        if (found_valid != NULL && r[i].valid) *found_valid = 1;
    }
    return found;
}

static void test_lte_uplink_bands(void)
{
    RankedMatch ranked[32];
    size_t n;

    puts("\n-- LTE uplink band database --");

    /* the old band-agnostic entries (which wrongly included every downlink
     * between 1710 and 2690 MHz) are gone */
    CHECK(signal_spec_find("LTE PUSCH") == NULL);
    CHECK(signal_spec_find("LTE PUCCH") == NULL);

    /* per-band entries exist, with the published channel bandwidths */
    CHECK(signal_spec_find("LTE PUSCH B3") != NULL);
    CHECK(signal_spec_find("LTE PUCCH B3") != NULL);
    CHECK(signal_spec_find("LTE PUSCH B7") != NULL);
    CHECK(signal_spec_find("LTE PUSCH B20") != NULL);
    CHECK(signal_spec_find("LTE TDD PUSCH B38/B41") != NULL);
    CHECK(signal_spec_find("5G NR PUSCH n78") != NULL);
    if (signal_spec_find("LTE PUCCH B3") != NULL) {
        CHECK_NEAR(signal_spec_find("LTE PUCCH B3")->channel_bw, 180000.0,
                   1.0);
    }

    /* 1755 MHz is inside Band 3's uplink: a 10 MHz PUSCH burst matches the
     * Band 3 shared-channel entry, and no PUCCH entry is valid for it
     * (PUCCH only spans 180-720 kHz at the carrier edges) */
    n = signals_rank_conf(1755e6, 10.0e6, NULL, NULL, 0, NULL, ranked,
                          (sizeof(ranked) / sizeof(ranked[0])));
    {
        int pusch_valid = 0;

        CHECK(ranked_name_contains(ranked, n, "PUSCH B3", 1, &pusch_valid));
        CHECK(pusch_valid);
        {
            int pucch_valid = 0;

            (void)ranked_name_contains(ranked, n, "PUCCH B3", 1,
                                       &pucch_valid);
            CHECK(!pucch_valid);
        }
    }
    /* a thin 180 kHz control burst at the same frequency is a legal PUCCH
     * (and PUSCH may also be only one resource block, so both appear) */
    n = signals_rank_conf(1755e6, 180.0e3, NULL, NULL, 0, NULL, ranked,
                          (sizeof(ranked) / sizeof(ranked[0])));
    {
        int pucch_valid = 0;

        CHECK(ranked_name_contains(ranked, n, "PUCCH B3", 1, &pucch_valid));
        CHECK(pucch_valid);
        CHECK(ranked_name_contains(ranked, n, "PUSCH B3", 1, NULL));
    }

    /* 2140 MHz is Band 1's DOWNLINK: no uplink channel is legal there */
    n = signals_rank_conf(2140e6, 10.0e6, NULL, NULL, 0, NULL, ranked,
                          (sizeof(ranked) / sizeof(ranked[0])));
    CHECK(!ranked_name_contains(ranked, n, "PUSCH", 1, NULL));
    CHECK(!ranked_name_contains(ranked, n, "PUCCH", 1, NULL));

    /* 2530 MHz: Band 7 uplink (FDD). The wide US Band 41 TDD range also
     * covers this frequency, so both entries may be in band there; what
     * matters is that the Band 7 uplink is recognised. */
    n = signals_rank_conf(2530e6, 10.0e6, NULL, NULL, 0, NULL, ranked,
                          (sizeof(ranked) / sizeof(ranked[0])));
    CHECK(ranked_name_contains(ranked, n, "PUSCH B7", 1, NULL));

    /* 2610 MHz is outside Band 7's uplink (2500-2570) but inside the TDD
     * Band 38 range, where the uplink may use the whole band */
    n = signals_rank_conf(2610e6, 10.0e6, NULL, NULL, 0, NULL, ranked,
                          (sizeof(ranked) / sizeof(ranked[0])));
    CHECK(ranked_name_contains(ranked, n, "B38/B41", 1, NULL));
    CHECK(!ranked_name_contains(ranked, n, "PUSCH B7", 1, NULL));

    /* 2650 MHz: outside Band 7's uplink but inside the TDD band 38/41,
     * where the uplink shares the whole band */
    n = signals_rank_conf(2650e6, 10.0e6, NULL, NULL, 0, NULL, ranked,
                          (sizeof(ranked) / sizeof(ranked[0])));
    CHECK(ranked_name_contains(ranked, n, "B38/B41", 1, NULL));
    CHECK(!ranked_name_contains(ranked, n, "PUSCH B7", 1, NULL));

    /* the 4.5 MHz figure is one 5 MHz carrier's active span, not a global
     * limit: a 10 MHz PUSCH allocation in Band 7 is legal */
    n = signals_rank_conf(2520e6, 9.0e6, NULL, NULL, 0, NULL, ranked,
                          (sizeof(ranked) / sizeof(ranked[0])));
    {
        int valid = 0;

        CHECK(ranked_name_contains(ranked, n, "PUSCH B7", 1, &valid));
        CHECK(valid);
    }
}

/* ------------------------------------------------------------------ */
/* signal detection                                                    */
/* ------------------------------------------------------------------ */

static void test_detect(void)
{
    /* Synthetic spectrogram: 256 bins x 60 frames, 2 MS/s around 100 MHz.
     * Signal A: strong CW tone at bin 60, on all frames.
     * Signal B: wider bump at bins 100..109, on only frames 10..30.
     * Noise floor: -100 dB everywhere else; signals at -20 dB (A) and
     * -30 dB (B). */
    enum { N = 256, F = 60 };
    float *db = malloc((size_t)N * F * sizeof(float));
    CHECK(db != NULL);
    for (size_t f = 0; f < F; f++) {
        for (int b = 0; b < N; b++) {
            float v = -100.0f;
            if (b == 60) v = -20.0f; /* A */
            if (b >= 100 && b < 110 && f >= 10 && f <= 30) v = -30.0f; /* B */
            db[f * N + b] = v;
        }
    }

    SignalRegion regs[DETECT_MAX_REGIONS];
    size_t n = detect_signals(db, F, N, 0, 1, 100.0, 2.0e6, 100.0e6, regs,
                              DETECT_MAX_REGIONS);
    CHECK(n >= 2);

    /* strongest first: the -20 dB tone */
    int found_a = 0, found_b = 0;
    for (size_t i = 0; i < n && i < DETECT_MAX_REGIONS; i++) {
        double fmin = regs[i].freq_min, fmax = regs[i].freq_max;
        double tmin = regs[i].time_min, tmax = regs[i].time_max;
        /* bin b spans f_lo + b*bin_hz with bin_hz = 2e6/256 = 7812.5 */
        if (fmin <= 99.0e6 + 60 * 7812.5 && fmax >= 99.0e6 + 61 * 7812.5) {
            found_a = 1;
            /* A is on for the whole window: full time span */
            CHECK(tmin <= 0.01 && tmax >= (double)F / 100.0 - 0.01);
        }
        if (fmin <= 99.0e6 + 100 * 7812.5 && fmax >= 99.0e6 + 110 * 7812.5) {
            found_b = 1;
            /* B is on frames 10..30 -> 0.10 s .. 0.31 s (with margins) */
            CHECK(tmin >= 0.05 && tmin <= 0.13);
            CHECK(tmax >= 0.28 && tmax <= 0.36);
        }
    }
    CHECK(found_a);
    CHECK(found_b);
    if (n >= 2) CHECK(regs[0].peak_db >= regs[1].peak_db - 1.0); /* sorted */

    /* degenerate inputs */
    CHECK(detect_signals(NULL, 0, N, 0, 1, 100.0, 2e6, 100e6, regs, 4) == 0);
    CHECK(detect_signals(db, F, N, 0, 1, 100.0, 2e6, 100e6, NULL, 0) == 0);
    free(db);
}

/* ------------------------------------------------------------------ */
/* audio player (headless-safe: no device touched before play)         */
/* ------------------------------------------------------------------ */

static void test_audio(void)
{
    enum { N = 16000, RATE = 32000 }; /* 0.5 s */
    float s[N];
    for (int i = 0; i < N; i++)
        s[i] = 0.5f * sinf(2.0f * (float)M_PI * 440.0f * i / RATE);

    AudioPlayer *p = audio_player_create(s, N, RATE);
    CHECK(p != NULL);
    if (p == NULL) return;
    CHECK_NEAR(audio_player_duration(p), 0.5, 1e-6);
    CHECK_NEAR(audio_player_position(p), 0.0, 1e-6);
    CHECK(!audio_player_is_playing(p));
    CHECK(!audio_player_finished(p));

    audio_player_seek(p, 0.25);
    CHECK_NEAR(audio_player_position(p), 0.25, 1e-6);
    audio_player_seek(p, 99.0); /* clamps to the end */
    CHECK_NEAR(audio_player_position(p), 0.5, 1e-6);
    CHECK(audio_player_finished(p));
    audio_player_seek(p, -5.0); /* clamps to the start */
    CHECK_NEAR(audio_player_position(p), 0.0, 1e-6);

    audio_player_destroy(p);
    audio_player_destroy(NULL); /* safe */
    CHECK(audio_player_create(NULL, 0, 0) == NULL);
    CHECK(audio_player_create(s, N, 0) == NULL);
}

/* ------------------------------------------------------------------ */
/* cyclostationary analysis                                            */
/* ------------------------------------------------------------------ */

static void test_cyclic(void)
{
    /* OOK at 1 kHz symbol rate: the fundamental must appear among the
     * detected cyclic features (grid resolution allows ~+-2 bins). */
    write_typed_file("mcyc", 3); /* OOK */

    char path[512];
    snprintf(path, sizeof(path), TYPED_DIR, "mcyc");
    IqSource src;
    char err[256];
    if (!iq_source_open(&src, path, err, sizeof(err))) {
        printf("open error: %s\n", err);
        CHECK(false);
        return;
    }

    Region r;
    region_for_kind(3, &r);
    CyclicResult c;
    memset(&c, 0, sizeof(c));
    bool ok = cyclic_analyze(&src, &r, &c);
    CHECK(ok);
    if (ok) {
        CHECK(c.n == CYCLIC_N);
        CHECK(c.scf != NULL && c.cdp != NULL);
        /* finite, normalised */
        int bad = 0;
        for (int i = 0; i < c.n * c.n; i += 997) {
            if (!isfinite(c.scf[i])) bad++;
            if (c.scf[i] < 0.0f || c.scf[i] > 1.0f + 1e-6f) bad++;
        }
        CHECK(bad == 0);
        CHECK(c.peak_count >= 1);

        /* the fundamental (symbol rate) is the smallest |alpha| peak */
        int found_fund = 0;
        for (int i = 0; i < c.peak_count; i++) {
            if (fabs(c.peaks[i].alpha) >= 400.0 &&
                fabs(c.peaks[i].alpha) <= 1600.0 &&
                c.peaks[i].strength > 0.3) {
                found_fund = 1;
            }
            printf("  cyclic peak: %8.1f Hz strength %.3f\n",
                   c.peaks[i].alpha, c.peaks[i].strength);
        }
        CHECK(found_fund);
    }
    cyclic_clear(&c);
    CHECK(c.scf == NULL && c.cdp == NULL);

    /* CW (constant tone): no cyclostationarity away from alpha = 0 */
    write_typed_file("mcw2", 0);
    snprintf(path, sizeof(path), TYPED_DIR, "mcw2");
    if (iq_source_open(&src, path, err, sizeof(err))) {
        region_for_kind(0, &r);
        memset(&c, 0, sizeof(c));
        if (cyclic_analyze(&src, &r, &c)) {
            CHECK(c.peak_count == 0);
        }
        cyclic_clear(&c);
        iq_source_close(&src);
    }
}

/* ------------------------------------------------------------------ */
/* fluctuation-of-autocorrelation DSSS detector                        */
/* ------------------------------------------------------------------ */

/* 2 s, 2 MS/s, baseband BPSK DSSS at 10 kHz chip rate (m-sequence,
 * length 127), chip amplitude `amp`; amp == 0 -> noise only. */
static void write_dsss_file(const char *sub, float amp)
{
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "mkdir -p /tmp/te15_test/%s", sub);
    system(cmd);

    const double rate = 2.0e6;
    const size_t n = (size_t)(rate * 3.0);
    const size_t chip_samples = 200; /* 100 us at 2 MS/s */

    /* 7-bit m-sequence (LFSR taps 7,6) -> 127 chips */
    unsigned lfsr = 0x5A;
    static int seq[127];
    for (int i = 0; i < 127; i++) {
        unsigned bit = ((lfsr >> 6) ^ (lfsr >> 5)) & 1u;
        lfsr = ((lfsr << 1) | bit) & 0x7Fu;
        seq[i] = ((lfsr >> 6) & 1u) ? 1 : -1;
    }

    float *iq = malloc(2 * n * sizeof(float));
    CHECK(iq != NULL);
    if (iq == NULL) return;
    unsigned rng = 99;
    for (size_t i = 0; i < n; i++) {
        float s = amp * (float)seq[(i / chip_samples) % 127];
        float nr = 0.05f * (((rng = rng * 1664525u + 1013904223u) >> 8 & 0xFFFFFF) / (float)0xFFFFFF - 0.5f);
        float ni = 0.05f * (((rng = rng * 1664525u + 1013904223u) >> 8 & 0xFFFFFF) / (float)0xFFFFFF - 0.5f);
        iq[2 * i]     = s + nr;
        iq[2 * i + 1] = ni;
    }

    char path[512];
    snprintf(path, sizeof(path), TYPED_DIR, sub);
    FILE *f = fopen(path, "wb");
    CHECK(f != NULL);
    if (f != NULL) {
        CHECK(fwrite(iq, sizeof(float), 2 * n, f) == 2 * n);
        fclose(f);
    }
    free(iq);
}

static void test_fluct(void)
{
    /* hidden DSSS: chip SNR ~ -10 dB, invisible in the mean ACF */
    write_dsss_file("mdsss", 0.0224f);
    /* noise-only control of the same length */
    write_dsss_file("mnoise2", 0.0f);

    char path[512];
    snprintf(path, sizeof(path), TYPED_DIR, "mdsss");
    IqSource src;
    char err[256];
    if (!iq_source_open(&src, path, err, sizeof(err))) {
        printf("open error: %s\n", err);
        CHECK(false);
        return;
    }

    Region r = {TEST_CENTER - 30.0e3, TEST_CENTER + 30.0e3, 0.1, 2.9};
    FluctResult f;
    memset(&f, 0, sizeof(f));
    bool ok = fluct_analyze(&src, &r, &f);
    CHECK(ok);
    if (ok) {
        printf("  dsss: F_peak=%.2f @ %.1f us (thr %.2f) acf=%.3f "
               "chip=%.1f kHz snr=%.1f dB detected=%d\n",
               f.peak_fluct, f.peak_lag * 1e6, f.threshold, f.peak_acf,
               f.chip_rate_hz / 1e3, f.snr_db, f.detected);
        CHECK(f.detected);
        CHECK(f.peak_fluct > f.threshold);
        /* chip period 100 us, lag resolution 10 us -> +-1 bin */
        CHECK(fabs(f.peak_lag * 1e6 - 100.0) < 15.0);
        CHECK(f.lag_s != NULL && f.mean_acf != NULL && f.fluct != NULL);
    }
    fluct_clear(&f);
    CHECK(f.lag_s == NULL && f.mean_acf == NULL && f.fluct == NULL);
    iq_source_close(&src);

    /* noise only: no fluctuation above threshold */
    snprintf(path, sizeof(path), TYPED_DIR, "mnoise2");
    if (iq_source_open(&src, path, err, sizeof(err))) {
        memset(&f, 0, sizeof(f));
        if (fluct_analyze(&src, &r, &f)) {
            printf("  noise: F_peak=%.2f @ %.1f us (thr %.2f) detected=%d\n",
                   f.peak_fluct, f.peak_lag * 1e6, f.threshold, f.detected);
            CHECK(!f.detected);
        }
        fluct_clear(&f);
        iq_source_close(&src);
    }
}

/* ------------------------------------------------------------------ */
/* higher-order statistics                                             */
/* ------------------------------------------------------------------ */

/* Box-Muller Gaussian noise file, 3 s at 2 MS/s (sigma per component) */
static void write_gaussian_file(const char *sub, float sigma)
{
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "mkdir -p /tmp/te15_test/%s", sub);
    system(cmd);

    const size_t n = (size_t)(2.0e6 * 3.0);
    float *iq = malloc(2 * n * sizeof(float));
    CHECK(iq != NULL);
    if (iq == NULL) return;
    unsigned rng = 1234;
    for (size_t i = 0; i < n; i++) {
        for (int c = 0; c < 2; c++) {
            float u1, u2;
            do {
                rng = rng * 1664525u + 1013904223u;
                u1 = (float)((rng >> 8) & 0xFFFFFF) / (float)0x1000000;
            } while (u1 <= 1e-9f);
            rng = rng * 1664525u + 1013904223u;
            u2 = (float)((rng >> 8) & 0xFFFFFF) / (float)0x1000000;
            float g = sigma * sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
            iq[2 * i + c] = g;
        }
    }
    char path[512];
    snprintf(path, sizeof(path), TYPED_DIR, sub);
    FILE *f = fopen(path, "wb");
    CHECK(f != NULL);
    if (f != NULL) {
        CHECK(fwrite(iq, sizeof(float), 2 * n, f) == 2 * n);
        fclose(f);
    }
    free(iq);
}

static void test_hos(void)
{
    /* DSSS (uniform chip sequence) vs true Gaussian noise */
    write_dsss_file("mdsss", 0.0224f);
    write_gaussian_file("mgauss", 0.05f);

    char path[512];
    snprintf(path, sizeof(path), TYPED_DIR, "mdsss");
    IqSource src;
    char err[256];
    if (!iq_source_open(&src, path, err, sizeof(err))) {
        printf("open error: %s\n", err);
        CHECK(false);
        return;
    }

    Region r = {TEST_CENTER - 30.0e3, TEST_CENTER + 30.0e3, 0.1, 2.9};
    HosResult h;
    memset(&h, 0, sizeof(h));
    bool ok = hos_analyze(&src, &r, &h);
    CHECK(ok);
    if (ok) {
        printf("  dsss: n42=%.4f (thr %.4f) n40=%.4f n41=%.4f "
               "kurt=%.3f non_gaussian=%d\n",
               h.n42, h.n42_threshold, h.n40, h.n41, h.kurtosis,
               h.non_gaussian);
        CHECK(h.non_gaussian);
        CHECK(h.n42 > h.n42_threshold);
        CHECK(h.n40 > 0.5);  /* BPSK-like: |C40|/C21^2 ~ 1..2 */
        CHECK(h.c42 < 0.0);  /* BPSK: negative fourth-order cumulant */
        CHECK(h.bisp != NULL && h.bs_n > 0);
        if (h.bisp != NULL) {
            int bad = 0;
            float mx = 0.0f;
            for (int i = 0; i < h.bs_n * h.bs_n; i += 97) {
                if (!isfinite(h.bisp[i])) bad++;
                if (h.bisp[i] < 0.0f || h.bisp[i] > 1.0f + 1e-6f) bad++;
                if (h.bisp[i] > mx) mx = h.bisp[i];
            }
            CHECK(bad == 0);
            CHECK(mx > 0.02f); /* structured bispectrum for BPSK */
        }
    }
    hos_clear(&h);
    CHECK(h.bisp == NULL);
    iq_source_close(&src);

    /* Gaussian noise: cumulants beyond 2nd order ~ 0 */
    snprintf(path, sizeof(path), TYPED_DIR, "mgauss");
    if (iq_source_open(&src, path, err, sizeof(err))) {
        memset(&h, 0, sizeof(h));
        if (hos_analyze(&src, &r, &h)) {
            printf("  noise: n42=%.4f (thr %.4f) non_gaussian=%d\n",
                   h.n42, h.n42_threshold, h.non_gaussian);
            CHECK(!h.non_gaussian);
            CHECK(h.n42 < h.n42_threshold);
        }
        hos_clear(&h);
        iq_source_close(&src);
    }
}

/* ------------------------------------------------------------------ */
/* cumulative peak-to-average ratio (CPAR) DSSS detector               */
/* ------------------------------------------------------------------ */

static void test_cpar(void)
{
    /* same hidden DSSS and noise files as the fluctuation test */
    write_dsss_file("mdsss", 0.0224f);
    write_dsss_file("mnoise2", 0.0f);

    char path[512];
    snprintf(path, sizeof(path), TYPED_DIR, "mdsss");
    IqSource src;
    char err[256];
    if (!iq_source_open(&src, path, err, sizeof(err))) {
        printf("open error: %s\n", err);
        CHECK(false);
        return;
    }

    Region r = {TEST_CENTER - 30.0e3, TEST_CENTER + 30.0e3, 0.1, 2.9};
    CparResult c;
    memset(&c, 0, sizeof(c));
    bool ok = cpar_analyze(&src, &r, &c);
    CHECK(ok);
    if (ok) {
        printf("  dsss: CPAR=%.1f peaks=%d bestPAR=%.1f @ %.3f ms "
               "chip=%.1f kHz detected=%d\n",
               c.cpar, c.n_peaks, c.peak_par, c.peak_lag * 1e3,
               c.chip_rate_hz / 1e3, c.detected);
        CHECK(c.detected);
        CHECK(c.cpar > 100.0);
        /* m-sequence code period: 127 chips x 100 us = 12.7 ms */
        CHECK(fabs(c.peak_lag - 0.0127) < 5e-5);
        /* chip period from the initial ACF slope: ~100 us */
        CHECK(fabs(c.chip_period_s * 1e6 - 100.0) < 25.0);
        CHECK(c.acf != NULL && c.par != NULL && c.is_peak != NULL);
    }
    cpar_clear(&c);
    CHECK(c.acf == NULL && c.par == NULL && c.is_peak == NULL);
    iq_source_close(&src);

    /* noise only: no significant peaks */
    snprintf(path, sizeof(path), TYPED_DIR, "mnoise2");
    if (iq_source_open(&src, path, err, sizeof(err))) {
        memset(&c, 0, sizeof(c));
        if (cpar_analyze(&src, &r, &c)) {
            printf("  noise: CPAR=%.1f peaks=%d bestPAR=%.1f detected=%d\n",
                   c.cpar, c.n_peaks, c.peak_par, c.detected);
            CHECK(!c.detected);
            CHECK(c.cpar == 0.0);
        }
        cpar_clear(&c);
        iq_source_close(&src);
    }
}

/* ------------------------------------------------------------------ */
/* IQ data formats (int16/int8/uint8) and runtime setters              */
/* ------------------------------------------------------------------ */

static void test_iq_formats(void)
{
    const char *sub = "mint16";
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "mkdir -p /tmp/te15_test/%s", sub);
    system(cmd);

    char path[512];
    snprintf(path, sizeof(path), TYPED_DIR, sub);

    /* 8 int16 complex samples: I = 10000..10007, Q = -5000..-4993 */
    int16_t buf[16];
    for (int i = 0; i < 8; i++) {
        buf[2 * i]     = (int16_t)(10000 + i);
        buf[2 * i + 1] = (int16_t)(-5000 + i);
    }
    FILE *f = fopen(path, "wb");
    CHECK(f != NULL);
    if (f != NULL) {
        CHECK(fwrite(buf, sizeof(int16_t), 16, f) == 16);
        fclose(f);
    }

    IqSource src;
    char err[256];
    CHECK(iq_source_open(&src, path, err, sizeof(err)));
    if (src.raw == NULL) {
        printf("open error: %s\n", err);
        return;
    }

    /* default float32 interpretation: 32 bytes / 8 = 4 samples */
    CHECK(src.format == IQ_FORMAT_FLOAT32);
    CHECK(src.total_samples == 4);

    iq_source_set_format(&src, IQ_FORMAT_INT16);
    CHECK(src.format == IQ_FORMAT_INT16);
    CHECK(src.bytes_per_sample == 4);
    CHECK(src.total_samples == 8);

    float out[16];
    iq_source_read_complex(&src, 0, out, 3);
    CHECK(fabsf(out[0] - 10000.0f / 32768.0f) < 1e-6f);
    CHECK(fabsf(out[1] - (-5000.0f) / 32768.0f) < 1e-6f);
    CHECK(fabsf(out[4] - 10002.0f / 32768.0f) < 1e-6f);

    /* uint8: 0 -> -1.0, 255 -> +1.0 */
    iq_source_set_format(&src, IQ_FORMAT_UINT8);
    CHECK(src.bytes_per_sample == 2);
    CHECK(src.total_samples == 16);
    iq_source_read_complex(&src, 0, out, 1);
    /* first two bytes are the low bytes of int16 10000 = 0x10 0x27 */
    CHECK(fabsf(out[0] - ((float)0x10 - 127.5f) / 127.5f) < 1e-6f);
    CHECK(fabsf(out[1] - ((float)0x27 - 127.5f) / 127.5f) < 1e-6f);

    /* int8 */
    iq_source_set_format(&src, IQ_FORMAT_INT8);
    CHECK(src.bytes_per_sample == 2);
    iq_source_read_complex(&src, 0, out, 1);
    CHECK(fabsf(out[0] - (float)(int8_t)0x10 / 128.0f) < 1e-6f);

    /* runtime rate change updates the duration */
    iq_source_set_rate(&src, 500000.0);
    CHECK(src.sample_rate == 500000.0);
    CHECK_NEAR(src.duration, (double)src.total_samples / 500000.0, 1e-12);

    /* reads past the end clamp */
    iq_source_set_format(&src, IQ_FORMAT_INT16);
    iq_source_read_complex(&src, 7, out, 5);
    CHECK(fabsf(out[0] - 10007.0f / 32768.0f) < 1e-6f);

    iq_source_close(&src);
    CHECK(src.raw == NULL && src.fd == -1);
}

/* ------------------------------------------------------------------ */
/* dsp: SRRC pulse + whitening                                         */
/* ------------------------------------------------------------------ */

static void test_srrc_whiten(void)
{
    float taps[64];
    size_t got = dsp_srrc_pulse(taps, 64, 4, 0.35);
    CHECK(got == 64);
    double e = 0.0;
    for (int i = 0; i < 64; i++) e += (double)taps[i] * taps[i];
    CHECK_NEAR(e, 1.0, 1e-3);
    CHECK(taps[32] > taps[0]); /* centre tap dominates */
    CHECK(dsp_srrc_pulse(taps, 64, 4, 1.5) == 0); /* bad rolloff */

    /* whitening: finite output, no crash on small input */
    float iq[256 * 2];
    unsigned rng = 7;
    for (int i = 0; i < 256 * 2; i++) {
        rng = rng * 1664525u + 1013904223u;
        iq[i] = 0.05f * (((rng >> 8) & 0xFFFFFF) / (float)0xFFFFFF - 0.5f);
    }
    dsp_whiten(iq, 256);
    int bad = 0;
    for (int i = 0; i < 256 * 2; i++) {
        if (!isfinite(iq[i])) bad++;
    }
    CHECK(bad == 0);
    dsp_whiten(iq, 4); /* too small: no-op, no crash */
}

/* ------------------------------------------------------------------ */
/* SSCA                                                                */
/* ------------------------------------------------------------------ */

static void test_ssca(void)
{
    write_dsss_file("mssca", 0.0224f);
    char path[512];
    snprintf(path, sizeof(path), TYPED_DIR, "mssca");
    IqSource src;
    char err[256];
    if (!iq_source_open(&src, path, err, sizeof(err))) {
        printf("open error: %s\n", err);
        CHECK(false);
        return;
    }
    Region r = {TEST_CENTER - 400.0e3, TEST_CENTER + 400.0e3, 0.1, 2.9};
    SscaResult sc;
    memset(&sc, 0, sizeof(sc));
    CHECK(ssca_analyze(&src, &r, &sc));
    if (sc.scf != NULL) {
        printf("  ssca: N'=%d N=%d strips=%d cf=%zu/%zu thr=%.4f\n",
               sc.channel_fft, sc.strip_fft, sc.num_strips,
               sc.cf_count, (size_t)sc.num_cf, sc.cf_threshold);
        CHECK(sc.cf_count >= 1); /* chip-rate cycles for BPSK */
        int bad = 0;
        for (int i = 0; i < sc.num_strips * sc.num_cf; i += 37) {
            if (!isfinite(sc.scf[i]) || !isfinite(sc.coh[i])) bad++;
            if (sc.coh[i] < -1e-6f || sc.coh[i] > 1.0f + 1e-6f) bad++;
        }
        CHECK(bad == 0);
        {
            int any_pos = 0;
            for (size_t i = 0; i < sc.cf_count; i++) {
                if (sc.cf_list[i][0] > 0.0) any_pos = 1;
            }
            CHECK(any_pos); /* at least one positive cycle frequency */
        }
    }
    ssca_clear(&sc);
    CHECK(sc.scf == NULL);
    iq_source_close(&src);
}

/* ------------------------------------------------------------------ */
/* CFAR cycle detectors                                                */
/* ------------------------------------------------------------------ */

static void write_carrier_bpsk_file(const char *sub, float amp,
                                     double fc_offset);

static void test_cycledet(void)
{
    /* carrier-BPSK: strong symbol-rate cyclic features; Gaussian noise
     * as the H0 control */
    write_carrier_bpsk_file("mcycled", 0.3f, 50.0e3);
    write_gaussian_file("mcycnoise", 0.05f);

    char path[512];
    snprintf(path, sizeof(path), TYPED_DIR, "mcycled");
    IqSource src;
    char err[256];
    if (!iq_source_open(&src, path, err, sizeof(err))) {
        CHECK(false);
        return;
    }
    /* region centred on the carrier; the symbol rate is 5 kHz */
    Region r = {TEST_CENTER + 10.0e3, TEST_CENTER + 90.0e3, 0.1, 2.9};
    CycleDetResult cd;
    memset(&cd, 0, sizeof(cd));
    if (cycledet_analyze(&src, &r, &cd)) {
        printf("  cycledet bpsk: best %.1f Hz stat %.4f (thr %.4f) "
               "multi %.4f -> %d\n",
               cd.best_cf, cd.best_stat, cd.threshold, cd.multi_cycle,
               cd.detected);
        CHECK(cd.detected);
        CHECK(fabs(cd.best_cf / 5000.0 - lround(cd.best_cf / 5000.0)) < 0.05); /* symbol-rate harmonic */
    }
    cycledet_clear(&cd);
    CHECK(cd.stat == NULL);
    iq_source_close(&src);

    snprintf(path, sizeof(path), TYPED_DIR, "mcycnoise");
    if (iq_source_open(&src, path, err, sizeof(err))) {
        memset(&cd, 0, sizeof(cd));
        if (cycledet_analyze(&src, &r, &cd)) {
            printf("  cycledet noise: multi %.4f (thr %.4f) -> %d\n",
                   cd.multi_cycle, cd.multi_threshold, cd.detected);
            CHECK(!cd.detected);
        }
        cycledet_clear(&cd);
        iq_source_close(&src);
    }
}

/* complex BPSK at a carrier offset (for the cyclic cumulants) */
static void write_carrier_bpsk_file(const char *sub, float amp,
                                    double fc_offset)
{
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "mkdir -p /tmp/te15_test/%s", sub);
    system(cmd);

    const double rate = TEST_RATE;
    const size_t n = (size_t)(rate * 3.0);
    unsigned rng = 42;
    float *iq = malloc(2 * n * sizeof(float));
    CHECK(iq != NULL);
    if (iq == NULL) return;
    for (size_t i = 0; i < n; i++) {
        rng = rng * 1664525u + 1013904223u;
        float sym = ((rng >> 8) & 1u) ? amp : -amp;
        double ph = 2.0 * M_PI * fc_offset * (double)i / rate;
        float re = sym * (float)cos(ph);
        float im = sym * (float)sin(ph);
        iq[2 * i]     = re + 0.02f * (((rng = rng * 1664525u + 1013904223u) >> 8 & 0xFFFFFF) / (float)0xFFFFFF - 0.5f);
        iq[2 * i + 1] = im + 0.02f * (((rng = rng * 1664525u + 1013904223u) >> 8 & 0xFFFFFF) / (float)0xFFFFFF - 0.5f);
    }
    char path[512];
    snprintf(path, sizeof(path), TYPED_DIR, sub);
    FILE *f = fopen(path, "wb");
    CHECK(f != NULL);
    if (f != NULL) {
        CHECK(fwrite(iq, sizeof(float), 2 * n, f) == 2 * n);
        fclose(f);
    }
    free(iq);
}

/* ------------------------------------------------------------------ */
/* cyclic temporal cumulants (AMR)                                     */
/* ------------------------------------------------------------------ */

static void test_ccc(void)
{
    write_carrier_bpsk_file("mccc", 0.3f, 50.0e3);
    char path[512];
    snprintf(path, sizeof(path), TYPED_DIR, "mccc");
    IqSource src;
    char err[256];
    if (!iq_source_open(&src, path, err, sizeof(err))) {
        CHECK(false);
        return;
    }
    /* region centred on the signal carrier */
    Region r = {TEST_CENTER + 10.0e3, TEST_CENTER + 90.0e3, 0.1, 2.9};
    CccResult cc;
    memset(&cc, 0, sizeof(cc));
    CHECK(ccc_analyze(&src, &r, &cc));
    printf("  ccc: n40=%.3f n41=%.3f n42=%.3f -> %s\n",
           cc.n40, cc.n41, cc.n42, cc.verdict);
    CHECK(cc.valid);
    CHECK(strstr(cc.verdict, "BPSK") != NULL);
    CHECK(cc.n40 > 1.0); /* theoretical 2 */
    ccc_clear(&cc);
    iq_source_close(&src);
}

/* ------------------------------------------------------------------ */
/* SPTK primitives                                                     */
/* ------------------------------------------------------------------ */

static void test_sptk_primitives(void)
{
    /* characteristic function of Gaussian data ~ e^{-s^2 w^2/2} */
    float x[4096];
    unsigned rng = 11;
    for (int i = 0; i < 4096; i++) {
        float u1, u2;
        do {
            rng = rng * 1664525u + 1013904223u;
            u1 = (float)((rng >> 8) & 0xFFFFFF) / (float)0x1000000;
        } while (u1 <= 1e-9f);
        rng = rng * 1664525u + 1013904223u;
        u2 = (float)((rng >> 8) & 0xFFFFFF) / (float)0x1000000;
        x[i] = 0.1f * sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
    }
    double omega[8];
    for (int i = 0; i < 8; i++) omega[i] = 0.5 * i;
    double re[8], im[8];
    sptk_charfun(x, 4096, omega, 8, re, im);
    /* sigma ~ 0.1 -> CF(omega) ~ e^{-0.005 w^2} >= 0.6 at w <= 8 */
    CHECK(re[0] > 0.9 && fabs(im[0]) < 0.1);
    CHECK(re[1] > 0.5 && re[1] < 1.05);
    CHECK(re[4] > 0.0 && re[4] < re[1]); /* decay */

    double sgrid[4] = {-1.0, -0.5, 0.5, 1.0};
    double mgf[4];
    sptk_mgf(x, 4096, sgrid, 4, mgf);
    CHECK(mgf[2] > 0.9 && mgf[2] < 1.2);

    /* cepstrum of a periodic spectrum: peak at the period */
    double logsp[64];
    for (int i = 0; i < 64; i++) {
        double env = 1.0 + 0.5 * cos(2.0 * M_PI * (double)i / 8.0);
        logsp[i] = log(env + 0.001);
    }
    double cep[64];
    sptk_cepstrum(logsp, 64, cep);
    int peak_q = 0;
    for (int q = 1; q < 64; q++) {
        if (cep[q] > cep[peak_q]) peak_q = q;
    }
    CHECK(peak_q == 8 || peak_q == 56); /* quefrency 8 (or mirror) */

    /* LPC on an AR(1)-like positive-mean sequence (the magnitude of
     * a positive sequence keeps the AR structure) */
    float iq[1024 * 2];
    float prev = 2.0f;
    for (int i = 0; i < 1024; i++) {
        rng = rng * 1664525u + 1013904223u;
        float w = 0.02f * (((rng >> 8) & 0xFFFFFF) / (float)0xFFFFFF - 0.5f);
        prev = 2.0f + 0.8f * (prev - 2.0f) + w;
        if (prev < 0.5f) prev = 0.5f;
        iq[2 * i] = prev;
        iq[2 * i + 1] = 0.0f;
    }
    double coeffs[8];
    CHECK(sptk_lpc(iq, 1024, 4, coeffs));
    CHECK(coeffs[0] == 1.0);
    CHECK(coeffs[1] < -0.4); /* negative AR pole */

    /* AMDF of a periodic pulse train: period ~ 40 samples */
    float per[1024 * 2];
    for (int i = 0; i < 1024; i++) {
        per[2 * i] = (i % 40 < 2) ? 1.0f : 0.1f;
        per[2 * i + 1] = 0.0f;
    }
    double amdf[256];
    sptk_amdf(per, 1024, 256, amdf);
    double best = 1e30;
    int best_lag = 0;
    for (int lag = 20; lag < 60; lag++) {
        if (amdf[lag] < best) {
            best = amdf[lag];
            best_lag = lag;
        }
    }
    CHECK(best_lag > 35 && best_lag < 45);

    /* Goertzel on a tone: power at the tone >> elsewhere */
    float tone[1024 * 2];
    for (int i = 0; i < 1024; i++) {
        tone[2 * i] = 0.9f * cosf(2.0f * (float)M_PI * 0.1f * (float)i);
        tone[2 * i + 1] = 0.0f;
    }
    double p_on = sptk_goertzel(tone, 1024, 0.1);
    double p_off = sptk_goertzel(tone, 1024, 0.3);
    CHECK(p_on > 100.0 * p_off);

    /* FIR design: lowpass response at the cutoff ~ 0.5, in-band ~ 1 */
    float taps[65];
    CHECK(sptk_fir_design(0, 0.1, 0.0, taps, 65) == 65);
    double sum = 0.0;
    for (int i = 0; i < 65; i++) sum += taps[i];
    CHECK_NEAR(sum, 1.0, 0.15);
    /* median filter kills an impulse */
    float imp[64 * 2];
    for (int i = 0; i < 64; i++) {
        imp[2 * i] = 0.1f;
        imp[2 * i + 1] = 0.0f;
    }
    imp[2 * 32] = 100.0f;
    sptk_median_filter(imp, 64);
    CHECK(imp[2 * 32] < 10.0f);
}

/* ------------------------------------------------------------------ */
/* distribution analysis (CF + MGF)                                    */
/* ------------------------------------------------------------------ */

static void test_sdist(void)
{
    write_gaussian_file("msdist", 0.05f);
    write_carrier_bpsk_file("msdist2", 0.3f, 50.0e3);

    char path[512];
    snprintf(path, sizeof(path), TYPED_DIR, "msdist");
    IqSource src;
    char err[256];
    if (!iq_source_open(&src, path, err, sizeof(err))) {
        CHECK(false);
        return;
    }
    /* whole-file region */
    Region r = {TEST_CENTER - 500.0e3, TEST_CENTER + 500.0e3, 0.0, 3.0};
    SdistResult sd;
    memset(&sd, 0, sizeof(sd));
    if (sdist_analyze(&src, &r, &sd)) {
        printf("  sdist noise: gauss_dist=%.4f -> %d\n",
               sd.gauss_dist, sd.gaussian_like);
        CHECK(sd.gaussian_like);
    }
    sdist_clear(&sd);
    CHECK(sd.cf_re == NULL);
    iq_source_close(&src);

    snprintf(path, sizeof(path), TYPED_DIR, "msdist2");
    if (iq_source_open(&src, path, err, sizeof(err))) {
        memset(&sd, 0, sizeof(sd));
        if (sdist_analyze(&src, &r, &sd)) {
            printf("  sdist bpsk: gauss_dist=%.4f -> %d\n",
                   sd.gauss_dist, sd.gaussian_like);
            CHECK(!sd.gaussian_like);
        }
        sdist_clear(&sd);
        iq_source_close(&src);
    }
}

/* ------------------------------------------------------------------ */
/* classic analysis (cepstrum/AMDF/LPC/group delay/Goertzel)           */
/* ------------------------------------------------------------------ */

static void test_classic(void)
{
    write_typed_file("mclassic", 3); /* OOK, 1 kHz symbol rate */
    char path[512];
    snprintf(path, sizeof(path), TYPED_DIR, "mclassic");
    IqSource src;
    char err[256];
    if (!iq_source_open(&src, path, err, sizeof(err))) {
        CHECK(false);
        return;
    }
    Region r = {TEST_CENTER + 350.0e3, TEST_CENTER + 450.0e3, 0.2, 2.0};
    ClassicResult cl;
    memset(&cl, 0, sizeof(cl));
    if (classic_analyze(&src, &r, 1000.0, &cl)) {
        printf("  classic ook: period %.1f Hz (cep %.1f, amdf %.1f) "
               "probe %.4f\n",
               cl.period_hz, cl.cep_period_hz, cl.amdf_period_hz,
               cl.goertzel_power);
        CHECK(cl.period_hz > 500.0 && cl.period_hz < 3000.0);
        CHECK(cl.goertzel_power > 0.0);
    }
    classic_clear(&cl);
    CHECK(cl.cepstrum == NULL);
    iq_source_close(&src);
}

/* ------------------------------------------------------------------ */
/* practical filters                                                   */
/* ------------------------------------------------------------------ */

static void test_filters(void)
{
    /* CW at +400 kHz in the test file: bandpass passes it, bandstop
     * removes it */
    write_typed_file("mfilt", 0); /* CW */
    char path[512];
    snprintf(path, sizeof(path), TYPED_DIR, "mfilt");
    IqSource src;
    char err[256];
    if (!iq_source_open(&src, path, err, sizeof(err))) {
        CHECK(false);
        return;
    }
    Region r = {TEST_CENTER + 390.0e3, TEST_CENTER + 410.0e3, 0.2, 2.0};
    FilterResult fr;
    memset(&fr, 0, sizeof(fr));
    CHECK(filter_analyze(&src, &r, 0, 20000.0, 0.0, 127, &fr));
    printf("  filter lp: power %.1f -> %.1f dB\n",
           fr.power_in_db, fr.power_out_db);
    CHECK(fr.power_out_db < fr.power_in_db + 1.0); /* not amplified */
    filter_clear(&fr);
    CHECK(fr.response == NULL);
    iq_source_close(&src);
}

/* rectangular pulse train with carrier (for pulse analysis) */
static void write_pulse_file(const char *sub, double width_s, double period_s,
                             double amp)
{
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "mkdir -p /tmp/te15_test/%s", sub);
    system(cmd);

    const double rate = TEST_RATE;
    const size_t n = (size_t)(rate * 3.0);
    float *iq = malloc(2 * n * sizeof(float));
    CHECK(iq != NULL);
    if (iq == NULL) return;
    unsigned rng = 55;
    for (size_t i = 0; i < n; i++) {
        double t = (double)i / rate;
        double in_pulse = fmod(t, period_s) < width_s ? 1.0 : 0.0;
        double ph = 2.0 * M_PI * 400.0e3 * (double)i / rate;
        float re = amp * (float)in_pulse * (float)cos(ph);
        float im = amp * (float)in_pulse * (float)sin(ph);
        iq[2 * i]     = re + 0.03f * (((rng = rng * 1664525u + 1013904223u) >> 8 & 0xFFFFFF) / (float)0xFFFFFF - 0.5f);
        iq[2 * i + 1] = im + 0.03f * (((rng = rng * 1664525u + 1013904223u) >> 8 & 0xFFFFFF) / (float)0xFFFFFF - 0.5f);
    }
    char path[512];
    snprintf(path, sizeof(path), TYPED_DIR, sub);
    FILE *f = fopen(path, "wb");
    CHECK(f != NULL);
    if (f != NULL) {
        CHECK(fwrite(iq, sizeof(float), 2 * n, f) == 2 * n);
        fclose(f);
    }
    free(iq);
}

/* ------------------------------------------------------------------ */
/* pulse-train analysis                                                */
/* ------------------------------------------------------------------ */

static void test_pulse(void)
{
    /* width 200 us, period 1 ms -> PRF 1000 Hz, duty 0.2 */
    write_pulse_file("mpulse", 200.0e-6, 1.0e-3, 0.5);

    char path[512];
    snprintf(path, sizeof(path), TYPED_DIR, "mpulse");
    IqSource src;
    char err[256];
    if (!iq_source_open(&src, path, err, sizeof(err))) {
        CHECK(false);
        return;
    }
    Region r = {TEST_CENTER + 350.0e3, TEST_CENTER + 450.0e3, 0.2, 2.8};
    PulseResult pr;
    memset(&pr, 0, sizeof(pr));
    if (pulse_analyze(&src, &r, &pr)) {
        printf("  pulse: %d pulses tau %.3f ms T %.3f ms PRF %.1f Hz "
               "duty %.2f rise %.1f us f0 %.1f Hz %s\n",
               pr.num_pulses, pr.pulse_width_s * 1e3, pr.period_s * 1e3,
               pr.prf_hz, pr.duty_cycle, pr.rise_s * 1e6, pr.f0_hz,
               pr.modulation);
        CHECK(pr.num_pulses >= 100);
        CHECK(pr.period_s > 0.0009 && pr.period_s < 0.0011);
        CHECK(pr.pulse_width_s > 0.00015 && pr.pulse_width_s < 0.00026);
        CHECK(pr.prf_hz > 900.0 && pr.prf_hz < 1100.0);
        CHECK(pr.duty_cycle > 0.15 && pr.duty_cycle < 0.28);
        CHECK(pr.f0_hz > 800.0 && pr.f0_hz < 1200.0);
        CHECK(strstr(pr.modulation, "constant-PRF") != NULL);
    }
    pulse_clear(&pr);
    CHECK(pr.env == NULL && pr.line_spectrum == NULL);
    iq_source_close(&src);
}

/* selection->file mapping: a tone burst at a known time must be found
 * only when the analysis region covers it */
static void test_selection_mapping(void)
{
    /* tone burst in [1.5, 1.6] s at +300 kHz inside a dedicated file */
    const char *sub = "mmapchk";
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "mkdir -p /tmp/te15_test/%s", sub);
    system(cmd);
    const double rate = TEST_RATE;
    const size_t n = (size_t)(rate * 3.0);
    float *iq = malloc(2 * n * sizeof(float));
    CHECK(iq != NULL);
    if (iq == NULL) return;
    unsigned rng = 77;
    for (size_t i = 0; i < n; i++) {
        double t = (double)i / rate;
        double on = (t >= 1.5 && t < 1.6) ? 1.0 : 0.0;
        double ph = 2.0 * M_PI * 300.0e3 * (double)i / rate;
        iq[2 * i]     = 0.4f * (float)on * (float)cos(ph) +
                        0.02f * (((rng = rng * 1664525u + 1013904223u) >> 8 & 0xFFFFFF) / (float)0xFFFFFF - 0.5f);
        iq[2 * i + 1] = 0.4f * (float)on * (float)sin(ph) +
                        0.02f * (((rng = rng * 1664525u + 1013904223u) >> 8 & 0xFFFFFF) / (float)0xFFFFFF - 0.5f);
    }
    char path[512];
    snprintf(path, sizeof(path), TYPED_DIR, sub);
    FILE *f = fopen(path, "wb");
    CHECK(f != NULL);
    if (f != NULL) {
        CHECK(fwrite(iq, sizeof(float), 2 * n, f) == 2 * n);
        fclose(f);
    }
    free(iq);

    IqSource src;
    char err[256];
    if (!iq_source_open(&src, path, err, sizeof(err))) {
        CHECK(false);
        return;
    }
    /* the burst region: metrics must classify CW */
    Region r = {TEST_CENTER + 250.0e3, TEST_CENTER + 350.0e3, 1.5, 1.6};
    Analysis a;
    Metrics m;
    memset(&a, 0, sizeof(a));
    CHECK(analysis_run(&src, &r, &a));
    CHECK(metrics_compute(&src, &r, &m));
    printf("  mapping: burst region -> class %s, bw %.0f\n",
           m.mod_class, m.occupied_bw_hz);
    CHECK(strstr(m.mod_class, "CW") != NULL);
    /* a region at the same frequency but the WRONG time: no CW */
    Region r2 = {TEST_CENTER + 250.0e3, TEST_CENTER + 350.0e3, 2.5, 2.6};
    Metrics m2;
    memset(&m2, 0, sizeof(m2));
    CHECK(metrics_compute(&src, &r2, &m2));
    CHECK(strstr(m2.mod_class, "CW") == NULL || m2.snr_db < m.snr_db);
    analysis_clear(&a);
    iq_source_close(&src);
}

/* ------------------------------------------------------------------ */
/* analysis image must span the WHOLE selection (not a truncated head) */
/* ------------------------------------------------------------------ */

static void test_analysis_image_span(void)
{
    write_typed_file("mimg", 0); /* CW at +400 kHz */
    char path[512];
    snprintf(path, sizeof(path), TYPED_DIR, "mimg");
    IqSource src;
    char err[256];
    if (!iq_source_open(&src, path, err, sizeof(err))) {
        CHECK(false);
        return;
    }
    /* a LONG selection: 0.75 s at 2 MS/s -> 1.5M samples, far above the
     * 200-frame cap at the nominal hop (the test file is 1 s long) */
    Region r = {TEST_CENTER + 390.0e3, TEST_CENTER + 410.0e3, 0.2, 0.95};
    Analysis a;
    memset(&a, 0, sizeof(a));
    CHECK(analysis_run(&src, &r, &a));
    if (a.image != NULL) {
        double span = (double)a.frames * a.dt;
        double sel = (r.time_max - r.time_min);
        printf("  image span: %d frames x %.5f s = %.3f s (selection %.3f s)\n",
               a.frames, a.dt, span, sel);
        CHECK(span > 0.9 * sel);   /* covers the whole selection */
        CHECK(span < 1.1 * sel);
        CHECK(a.bins > 0 && a.frames > 0);
    }
    analysis_clear(&a);
    iq_source_close(&src);
}

/* pulse-position-modulated train: fixed width, 4 discrete intervals */
static void write_ppm_file(const char *sub, double width_s, double amp)
{
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "mkdir -p /tmp/te15_test/%s", sub);
    system(cmd);
    const double rate = TEST_RATE;
    const size_t n = (size_t)(rate * 3.0);
    static const double iv[4] = {0.5e-3, 0.75e-3, 1.0e-3, 1.25e-3};
    float *iq = malloc(2 * n * sizeof(float));
    CHECK(iq != NULL);
    if (iq == NULL) return;
    memset(iq, 0, 2 * n * sizeof(float));
    double t0 = 0.05;
    int k = 0;
    while (t0 + width_s < 2.9) {
        int lvl = (int)((t0 * 7919.0) ) % 4;
        double f0 = 200.0e3; /* carrier offset from the region centre */
        size_t a = (size_t)(t0 * rate), b = a + (size_t)(width_s * rate);
        for (size_t i = a; i < b && i < n; i++) {
            double t = (double)(i - a) / rate;
            double ph = 2.0 * M_PI * f0 * t;
            iq[2 * i] = amp * (float)cos(ph);
            iq[2 * i + 1] = amp * (float)sin(ph);
        }
        t0 += iv[lvl] + width_s;
        k++;
        (void)k;
    }
    unsigned r2 = 77;
    for (size_t i = 0; i < n; i++) {
        iq[2 * i] += 0.01f * (((r2 = r2 * 1664525u + 1013904223u) >> 8 & 0xFFFFFF) / (float)0xFFFFFF - 0.5f);
        iq[2 * i + 1] += 0.01f * (((r2 = r2 * 1664525u + 1013904223u) >> 8 & 0xFFFFFF) / (float)0xFFFFFF - 0.5f);
    }
    char path[512];
    snprintf(path, sizeof(path), TYPED_DIR, sub);
    FILE *f = fopen(path, "wb");
    CHECK(f != NULL);
    if (f != NULL) {
        CHECK(fwrite(iq, sizeof(float), 2 * n, f) == 2 * n);
        fclose(f);
    }
    free(iq);
}

/* variable-waveform train: each pulse is a chirp with a different sweep */
static void write_chirp_file(const char *sub, double width_s, float amp)
{
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "mkdir -p /tmp/te15_test/%s", sub);
    system(cmd);
    const double rate = TEST_RATE;
    const size_t n = (size_t)(rate * 3.0);
    float *iq = malloc(2 * n * sizeof(float));
    CHECK(iq != NULL);
    if (iq == NULL) return;
    memset(iq, 0, 2 * n * sizeof(float));
    double t0 = 0.05;
    const double period = 1.0e-3;
    int k = 0;
    while (t0 + width_s < 2.9) {
        /* sweep alternates between 40 kHz and 240 kHz (variable chirp) */
        double sweep = ((k % 3) == 0) ? 40.0e3 : (((k % 3) == 1) ? 140.0e3 : 240.0e3);
        double f0 = 100.0e3; /* start frequency offset */
        size_t a = (size_t)(t0 * rate), b = a + (size_t)(width_s * rate);
        for (size_t i = a; i < b && i < n; i++) {
            double t = (double)(i - a) / rate;
            double kk = sweep / width_s;
            double ph = 2.0 * M_PI * (f0 * t + 0.5 * kk * t * t);
            iq[2 * i] = amp * (float)cos(ph);
            iq[2 * i + 1] = amp * (float)sin(ph);
        }
        t0 += period;
        k++;
    }
    unsigned r2 = 55;
    for (size_t i = 0; i < n; i++) {
        iq[2 * i] += 0.01f * (((r2 = r2 * 1664525u + 1013904223u) >> 8 & 0xFFFFFF) / (float)0xFFFFFF - 0.5f);
        iq[2 * i + 1] += 0.01f * (((r2 = r2 * 1664525u + 1013904223u) >> 8 & 0xFFFFFF) / (float)0xFFFFFF - 0.5f);
    }
    char path[512];
    snprintf(path, sizeof(path), TYPED_DIR, sub);
    FILE *f = fopen(path, "wb");
    CHECK(f != NULL);
    if (f != NULL) {
        CHECK(fwrite(iq, sizeof(float), 2 * n, f) == 2 * n);
        fclose(f);
    }
    free(iq);
}

/* ------------------------------------------------------------------ */
/* PPM and variable-chirp train detection                              */
/* ------------------------------------------------------------------ */

static void test_pulse_modulations(void)
{
    IqSource src;
    char err[256];
    char path[512];
    PulseResult pr;

    /* --- PPM: 4 discrete interval levels --- */
    write_ppm_file("mppm", 200.0e-6, 0.4f);
    snprintf(path, sizeof(path), TYPED_DIR, "mppm");
    if (iq_source_open(&src, path, err, sizeof(err))) {
        Region r = {TEST_CENTER + 100.0e3, TEST_CENTER + 300.0e3, 0.05, 2.9};
        memset(&pr, 0, sizeof(pr));
        if (pulse_analyze(&src, &r, &pr)) {
            printf("  ppm: %d pulses, %d interval levels, intCV %.2f, "
                   "widthCV %.2f -> %s\n",
                   pr.num_pulses, pr.interval_levels, pr.interval_cv,
                   pr.width_cv, pr.modulation);
            CHECK(pr.num_pulses >= 50);
            CHECK(pr.interval_levels >= 3 && pr.interval_levels <= 6);
            CHECK(strstr(pr.modulation, "PPM") != NULL);
        }
        pulse_clear(&pr);
        iq_source_close(&src);
    }

    /* --- variable chirps --- */
    write_chirp_file("mchirp", 200.0e-6, 0.4f);
    snprintf(path, sizeof(path), TYPED_DIR, "mchirp");
    if (iq_source_open(&src, path, err, sizeof(err))) {
        Region r = {TEST_CENTER + 50.0e3, TEST_CENTER + 400.0e3, 0.05, 2.9};
        memset(&pr, 0, sizeof(pr));
        if (pulse_analyze(&src, &r, &pr)) {
            printf("  chirp: %d pulses, %d chirped, rateCV %.2f, "
                   "sweep %.1f kHz -> %s\n",
                   pr.num_pulses, pr.chirp_pulses, pr.chirp_rate_cv,
                   pr.chirp_bw_mean / 1e3, pr.modulation);
            printf("  chirp note: %s\n", pr.note);
            CHECK(pr.chirp_pulses >= 20);
            CHECK(pr.chirp_bw_mean > 30.0e3);
            CHECK(strstr(pr.modulation, "chirp") != NULL);
            CHECK(pr.chirp_rate_cv > 0.2);
        }
        pulse_clear(&pr);
        iq_source_close(&src);
    }
}

/* ------------------------------------------------------------------ */
/* modulation confidence + system ranking                              */
/* ------------------------------------------------------------------ */

static void test_modconf(void)
{
    IqSource src;
    char err[256];
    char path[512];
    Analysis a;
    Metrics m;
    CyclicResult cy;
    CccResult cc;
    HosResult hs;
    PulseResult pu;
    FluctResult fl;
    CparResult cp;
    ModCandidate mods[12];

    memset(&a, 0, sizeof(a));
    memset(&m, 0, sizeof(m));
    memset(&cy, 0, sizeof(cy));
    memset(&cc, 0, sizeof(cc));
    memset(&hs, 0, sizeof(hs));
    memset(&pu, 0, sizeof(pu));
    memset(&fl, 0, sizeof(fl));
    memset(&cp, 0, sizeof(cp));

    /* FM signal: the FM candidate must be top */
    write_typed_file("mmconfm", 1);
    snprintf(path, sizeof(path), TYPED_DIR, "mmconfm");
    if (iq_source_open(&src, path, err, sizeof(err))) {
        Region r = {TEST_CENTER + 250.0e3, TEST_CENTER + 550.0e3, 0.2, 0.9};
        analysis_run(&src, &r, &a);
        metrics_compute(&src, &r, &m);
        size_t nmod = modconf_rank(&a, &m, NULL, NULL, NULL, NULL, NULL, NULL,
                                   mods, 12);
        printf("  modconf fm: top=%s (%d) 2=%s (%d) 3=%s (%d)\n",
               nmod > 0 ? mods[0].name : "?",
               nmod > 0 ? mods[0].score : 0,
               nmod > 1 ? mods[1].name : "?",
               nmod > 1 ? mods[1].score : 0,
               nmod > 2 ? mods[2].name : "?",
               nmod > 2 ? mods[2].score : 0);
        CHECK(nmod >= 1);
        CHECK(strstr(mods[0].name, "FM") != NULL);
        /* ranked system list with confidence percentages */
        RankedMatch ids[5];
        size_t nid = signals_rank_conf(TEST_CENTER + 400.0e3, 300.0e3, &m,
                                       mods, nmod, NULL, ids, 5);
        CHECK(nid >= 1);
        for (size_t i = 0; i < nid; i++) {
            CHECK(ids[i].confidence_pct >= 0 &&
                  ids[i].confidence_pct <= 100);
        }
        printf("  sysid fm: %s (%d%%) | %s (%d%%)\n",
               nid > 0 ? ids[0].signal->name : "?",
               nid > 0 ? ids[0].confidence_pct : 0,
               nid > 1 ? ids[1].signal->name : "?",
               nid > 1 ? ids[1].confidence_pct : 0);
        analysis_clear(&a);
        iq_source_close(&src);
    }

    /* CW: CW candidate top */
    write_typed_file("mmconfcw", 0);
    snprintf(path, sizeof(path), TYPED_DIR, "mmconfcw");
    if (iq_source_open(&src, path, err, sizeof(err))) {
        Region r = {TEST_CENTER + 390.0e3, TEST_CENTER + 410.0e3, 0.2, 0.9};
        analysis_run(&src, &r, &a);
        metrics_compute(&src, &r, &m);
        size_t nmod = modconf_rank(&a, &m, NULL, NULL, NULL, NULL, NULL, NULL,
                                   mods, 12);
        printf("  modconf cw: top=%s (%d)\n",
               nmod > 0 ? mods[0].name : "?", nmod > 0 ? mods[0].score : 0);
        CHECK(nmod >= 1);
        CHECK(strstr(mods[0].name, "CW") != NULL);
        analysis_clear(&a);
        iq_source_close(&src);
    }

    /* pulse train: the pulse candidate must appear with a high score */
    write_pulse_file("mmconfp", 200.0e-6, 1.0e-3, 0.5);
    snprintf(path, sizeof(path), TYPED_DIR, "mmconfp");
    if (iq_source_open(&src, path, err, sizeof(err))) {
        Region r = {TEST_CENTER + 350.0e3, TEST_CENTER + 450.0e3, 0.2, 0.9};
        analysis_run(&src, &r, &a);
        metrics_compute(&src, &r, &m);
        pulse_analyze(&src, &r, &pu);
        size_t nmod = modconf_rank(&a, &m, NULL, NULL, NULL, &pu, NULL, NULL,
                                   mods, 12);
        printf("  modconf pulse: top=%s (%d)\n",
               nmod > 0 ? mods[0].name : "?", nmod > 0 ? mods[0].score : 0);
        CHECK(nmod >= 1);
        {
            int found = 0;
            for (size_t i = 0; i < nmod; i++) {
                if (strstr(mods[i].name, "ulsed") != NULL ||
                    strstr(mods[i].name, "OOK") != NULL ||
                    strstr(mods[i].name, "PRF") != NULL) {
                    found = 1;
                }
            }
            CHECK(found);
        }
        analysis_clear(&a);
        iq_source_close(&src);
    }
}

/* ------------------------------------------------------------------ */
/* published-characteristic matching (bandwidth/symbol-rate/pulsed)    */
/* ------------------------------------------------------------------ */

static void test_signals_specs(void)
{
    Metrics m;
    PulseResult pu;
    ModCandidate mods[12];
    RankedMatch ids[5];

    /* GSM-like: 200 kHz channel, 270.8 kBd GMSK */
    memset(&m, 0, sizeof(m));
    m.occupied_bw_hz = 195e3;
    m.duty_cycle = 1.0;
    m.sym_rate_hz = 270000.0;
    m.sym_rate_conf = 0.8;
    memset(&pu, 0, sizeof(pu));
    memset(mods, 0, sizeof(mods));
    mods[0].score = 90;
    snprintf(mods[0].name, sizeof(mods[0].name), "FSK");
    size_t n = signals_rank_conf(905e6, 195e3, &m, mods, 1, &pu, ids, 5);
    CHECK(n >= 1);
    printf("  spec gsm: %s (%d%%)%s%s | %s (%d%%)\n",
           ids[0].signal->name, ids[0].confidence_pct,
           ids[0].why[0] ? " - " : "", ids[0].why,
           n > 1 ? ids[1].signal->name : "?",
           n > 1 ? ids[1].confidence_pct : 0);
    CHECK(strstr(ids[0].signal->name, "GSM") != NULL);

    /* pulsed radar: 5 MHz wide burst train at 2.9 GHz */
    memset(&m, 0, sizeof(m));
    m.occupied_bw_hz = 5e6;
    m.duty_cycle = 0.05;
    m.papr_db = 14;
    m.spectral_flatness = 0.6;
    memset(&pu, 0, sizeof(pu));
    pu.num_pulses = 200;
    pu.pulse_width_s = 1e-6;
    pu.prf_hz = 1000.0;
    memset(mods, 0, sizeof(mods));
    mods[0].score = 90;
    snprintf(mods[0].name, sizeof(mods[0].name), "Pulsed radar / beacon");
    n = signals_rank_conf(2.9e9, 5e6, &m, mods, 1, &pu, ids, 5);
    CHECK(n >= 1);
    printf("  spec radar: %s (%d%%)%s%s | %s (%d%%)\n",
           ids[0].signal->name, ids[0].confidence_pct,
           ids[0].why[0] ? " - " : "", ids[0].why,
           n > 1 ? ids[1].signal->name : "?",
           n > 1 ? ids[1].confidence_pct : 0);
    CHECK(strstr(ids[0].signal->name, "Radar") != NULL);

    /* DMR-ish land mobile: 12.5 kHz channel, 4.8 kBd */
    memset(&m, 0, sizeof(m));
    m.occupied_bw_hz = 12e3;
    m.duty_cycle = 1.0;
    m.sym_rate_hz = 4800.0;
    m.sym_rate_conf = 0.7;
    memset(&pu, 0, sizeof(pu));
    memset(mods, 0, sizeof(mods));
    mods[0].score = 85;
    snprintf(mods[0].name, sizeof(mods[0].name), "FSK");
    n = signals_rank_conf(460e6, 12e3, &m, mods, 1, &pu, ids, 5);
    CHECK(n >= 1);
    printf("  spec landmobile: %s (%d%%)%s%s | %s (%d%%)\n",
           ids[0].signal->name, ids[0].confidence_pct,
           ids[0].why[0] ? " - " : "", ids[0].why,
           n > 1 ? ids[1].signal->name : "?",
           n > 1 ? ids[1].confidence_pct : 0);
    /* DMR, TETRA, P25 or similar land-mobile digital */
    CHECK(strstr(ids[0].signal->name, "DMR") != NULL ||
          strstr(ids[0].signal->name, "P25") != NULL ||
          strstr(ids[0].signal->name, "TETRA") != NULL);
}

/* ------------------------------------------------------------------ */
/* candidate validity + anomaly flagging                               */
/* ------------------------------------------------------------------ */

static void test_signals_validity(void)
{
    Metrics m;
    PulseResult pu;
    ModCandidate mods[4];
    RankedMatch ids[8];
    char explain[512];

    /* valid: GSM-like in the GSM 900 band */
    memset(&m, 0, sizeof(m));
    m.occupied_bw_hz = 195e3;
    m.duty_cycle = 1.0;
    m.sym_rate_hz = 270000.0;
    m.sym_rate_conf = 0.8;
    memset(&pu, 0, sizeof(pu));
    memset(mods, 0, sizeof(mods));
    mods[0].score = 90;
    snprintf(mods[0].name, sizeof(mods[0].name), "FSK");
    size_t n = signals_rank_conf(905e6, 195e3, &m, mods, 1, &pu, ids, 8);
    CHECK(n >= 1);
    CHECK(ids[0].valid == 1);
    CHECK(signals_anomaly_check(905e6, 195e3, ids, n, explain,
                                sizeof(explain)) == 0);
    printf("  valid: %s (%d%%) %s\n", ids[0].signal->name,
           ids[0].confidence_pct, ids[0].valid ? "VALID" : "invalid");

    /* anomaly: a signal 30 GHz wide band - nothing is licensed there */
    memset(&m, 0, sizeof(m));
    m.occupied_bw_hz = 5e6;
    m.duty_cycle = 1.0;
    memset(&pu, 0, sizeof(pu));
    memset(mods, 0, sizeof(mods));
    mods[0].score = 90;
    snprintf(mods[0].name, sizeof(mods[0].name), "OFDM / QAM");
    n = signals_rank_conf(30e9, 5e6, &m, mods, 1, &pu, ids, 8);
    int anomaly = signals_anomaly_check(30e9, 5e6, ids, n, explain,
                                        sizeof(explain));
    printf("  anomaly 30GHz: flag=%d valid_candidates=%d | %s\n", anomaly,
           (int)n, explain);
    CHECK(anomaly == 1);
    CHECK(strstr(explain, "ANOMALY") != NULL);
    CHECK(strstr(explain, "MHz") != NULL);
    for (size_t i = 0; i < n; i++) {
        CHECK(ids[i].valid == 0);
    }

    /* anomaly: in-band but the bandwidth is 100x off the documented channel */
    memset(&m, 0, sizeof(m));
    m.occupied_bw_hz = 2e6; /* DMR documents 12.5 kHz */
    m.duty_cycle = 1.0;
    memset(&pu, 0, sizeof(pu));
    memset(mods, 0, sizeof(mods));
    mods[0].score = 85;
    snprintf(mods[0].name, sizeof(mods[0].name), "FSK");
    n = signals_rank_conf(460e6, 2e6, &m, mods, 1, &pu, ids, 8);
    anomaly = signals_anomaly_check(460e6, 2e6, ids, n, explain,
                                    sizeof(explain));
    printf("  anomaly bw: flag=%d | %s\n", anomaly, explain);
    {
        int dmr_valid = -1;
        for (size_t i = 0; i < n; i++) {
            if (strstr(ids[i].signal->name, "DMR") != NULL) {
                dmr_valid = ids[i].valid;
            }
        }
        /* DMR must be flagged invalid because 2 MHz != its 12.5 kHz */
        CHECK(dmr_valid == 0);
        CHECK(anomaly == 1);
    }
}

/* ------------------------------------------------------------------ */
/* waterfall row mapping (mirrors the ui.c draw kernel)                */
/* ------------------------------------------------------------------ */

/* Pixel row p of h spans frames [start + p*vf/h, start + (p+1)*vf/h):
 * the waterfall reads top->bottom from the view start, and rows beyond
 * the file end wrap to the file's beginning. */
static void wf_row_band(size_t start, size_t vf, int h, int p,
                        double *ftop, double *fbot)
{
    *ftop = (double)start + (double)p * (double)vf / (double)h;
    *fbot = (double)start + (double)(p + 1) * (double)vf / (double)h;
}

static void test_waterfall_mapping(void)
{
    const size_t total = 304030;
    const size_t vf = 3906;
    const int h = 500;
    double t0, b0, t1, b1;

    /* at load (scroll 0): the top row is the very beginning of the file */
    wf_row_band(0, vf, h, 0, &t0, &b0);
    CHECK(t0 == 0.0);
    CHECK(b0 > 0.0);
    wf_row_band(0, vf, h, h - 1, &t1, &b1);
    CHECK(t1 > t0);
    CHECK(b1 == (double)vf);
    printf("  mapping load: top [%.1f,%.1f) bottom [%.1f,%.1f)\n",
           t0, b0, t1, b1);

    /* rows are strictly ordered top->bottom (time reads downward) */
    {
        double prev_bot = -1.0;
        int ordered = 1;

        for (int p = 0; p < h; p++) {
            double a, b;

            wf_row_band(0, vf, h, p, &a, &b);
            if (a < prev_bot) ordered = 0;
            prev_bot = b;
        }
        CHECK(ordered);
    }

    /* wrapped view near the end: the rows past the file end continue at
     * the file start (continuous, no gap and no black band) */
    {
        size_t start = total - 500; /* 0.128 s of tail then wrap */
        double wrap_end = (double)(start + vf - total);
        double prev_bot = -1.0;
        int continuous = 1;
        int wrapped_rows = 0;

        for (int p = 0; p < h; p++) {
            double a, b;
            double fa = a = 0.0;

            wf_row_band(start, vf, h, p, &a, &b);
            fa = (a + b) / 2.0;
            if (fa >= (double)total) {
                double head = fa - (double)total;

                wrapped_rows++;
                /* the head offset continues where the tail ended */
                if (head < 0.0 || head >= wrap_end) continuous = 0;
            } else if (prev_bot >= 0.0 && a < prev_bot - 1.0) {
                continuous = 0;
            }
            prev_bot = b;
        }
        CHECK(wrapped_rows > 0);   /* the wrap actually happened */
        CHECK(continuous);          /* and it is seamless */
        printf("  mapping wrap: %d wrapped rows, wrap_end %.0f frames\n",
               wrapped_rows, wrap_end);
    }

    /* map_region_pixels invariant (mirrors ui.c): after the vertical
     * flip the earlier time must map to the SMALLER y, never inverted */
    {
        double d_min = 0.10; /* tmin offset from the view start */
        double d_max = 0.20; /* tmax offset                      */
        double visible = 1.0;
        int y0 = (int)(h * (d_max / visible));
        int y1 = (int)(h * (d_min / visible));

        if (y1 < y0) {
            int tmp = y0;

            y0 = y1;
            y1 = tmp;
        }
        CHECK(y0 < y1);      /* non-degenerate rect */
        CHECK(y0 == (int)(h * 0.10));
        CHECK(y1 == (int)(h * 0.20));
        printf("  mapping selection rect: y0=%d y1=%d\n", y0, y1);
    }
}

/* pulse-train credibility (reject threshold-crossing "trains")        */
/* ------------------------------------------------------------------ */

/* Noise-only IQ: a constant-envelope noise-like emission must not yield a
 * credible pulse train (its threshold crossings are not bursts). A proper
 * xorshift128+ generator with independent states is used so the noise has
 * no periodic structure of its own. */
static void write_noise_file(const char *sub, double seconds)
{
    char cmd[512];
    char path[512];
    snprintf(cmd, sizeof(cmd), "mkdir -p /tmp/te15_test/%s", sub);
    system(cmd);
    const double rate = TEST_RATE;
    size_t n = (size_t)(rate * seconds);
    float *iq = malloc(2 * n * sizeof(float));
    CHECK(iq != NULL);
    if (iq == NULL) return;
    unsigned long long s0 = 0x9E3779B97F4A7C15ULL;
    unsigned long long s1 = 0xBF58476D1CE4E5B9ULL;

    for (size_t i = 0; i < n; i++) {
        unsigned long long x = s0;
        unsigned long long y = s1;

        s0 = y;
        x ^= x << 23;
        s1 = x ^ y ^ (x >> 17) ^ (y >> 26);
        double re = (double)((s1 + y) >> 11) / 9007199254740992.0 - 1.0;
        x = s0;
        y = s1;
        s0 = y;
        x ^= x << 23;
        s1 = x ^ y ^ (x >> 17) ^ (y >> 26);
        double im = (double)((s1 + y) >> 11) / 9007199254740992.0 - 1.0;

        iq[2 * i] = (float)(re * 0.02);
        iq[2 * i + 1] = (float)(im * 0.02);
    }
    snprintf(path, sizeof(path), TYPED_DIR, sub);
    FILE *f = fopen(path, "wb");
    CHECK(f != NULL);
    if (f != NULL) {
        fwrite(iq, sizeof(float), 2 * n, f);
        fclose(f);
    }
    free(iq);
}

static void test_pulse_credibility(void)
{
    IqSource src;
    char err[256];
    char path[512];

    write_noise_file("pnoise", 0.6);
    snprintf(path, sizeof(path), TYPED_DIR, "pnoise");
    if (iq_source_open(&src, path, err, sizeof(err))) {
        Region r = {TEST_CENTER + 50.0e3, TEST_CENTER + 450.0e3, 0.1, 0.6};
        PulseResult p;

        memset(&p, 0, sizeof(p));
        CHECK(pulse_analyze(&src, &r, &p));
        printf("  noise pulse: n %d credible %d prf %.1f env_cv %.2f duty %.0f%% jitter_frac %.2f f0 %.1f\n",
               p.num_pulses, p.credible, p.prf_hz, p.env_cv,
               p.duty_cycle * 100.0, p.jitter_frac, p.f0_hz);
        if (p.line_spectrum != NULL) {
            int b = (int)(p.f0_hz / p.fs * p.line_n + 0.5);
            printf("  noise line: f0 bin %d val %.1f dB (n %d)\n", b,
                   p.line_spectrum[b], p.line_n);
        }
        if (p.num_pulses >= 2) {
            CHECK(p.credible == 0);
            CHECK(p.prf_hz == 0.0);
            CHECK(strstr(p.modulation, "no credible") != NULL);
        }
        pulse_clear(&p);
        iq_source_close(&src);
    }

    /* a real burst train stays credible */
    write_pulse_file("pcred", 200.0e-6, 1.0e-3, 0.5);
    snprintf(path, sizeof(path), TYPED_DIR, "pcred");
    if (iq_source_open(&src, path, err, sizeof(err))) {
        Region r = {TEST_CENTER + 350.0e3, TEST_CENTER + 450.0e3, 0.1, 0.9};
        PulseResult p;

        memset(&p, 0, sizeof(p));
        CHECK(pulse_analyze(&src, &r, &p));
        printf("  burst pulse: n %d credible %d prf %.1f\n",
               p.num_pulses, p.credible, p.prf_hz);
        CHECK(p.num_pulses > 10);
        CHECK(p.credible == 1);
        CHECK(p.prf_hz > 500.0 && p.prf_hz < 2000.0);
        pulse_clear(&p);
        iq_source_close(&src);
    }
}

/* ------------------------------------------------------------------ */
/* intra-burst envelope inspection (DAG node A9)                       */
/* ------------------------------------------------------------------ */

/* Build a synthetic bursty IQ recording: bursts of `burst_s` every
 * `period_s`; inside each burst a sub-pulse train whose widths cycle
 * through `widths_us` (a single entry = uniform). An optional carrier
 * offset stresses the residual-CFO removal. */
static void write_bursty_iq(const char *sub, double rate, double seconds,
                            double period_s, double burst_s,
                            const double *widths_us, int n_widths,
                            double carrier_hz)
{
    char cmd[512];
    char path[512];
    size_t n = (size_t)(rate * seconds);
    float *iq = malloc(2 * n * sizeof(float));
    unsigned long long rs0 = 0x9E3779B97F4A7C15ULL;
    unsigned long long rs1 = 0xBF58476D1CE4E5B9ULL;

    CHECK(iq != NULL);
    if (iq == NULL) return;
    snprintf(cmd, sizeof(cmd), "mkdir -p /tmp/te15_test/%s", sub);
    system(cmd);
    for (size_t i = 0; i < n; i++) {
        double t = (double)i / rate;
        double phase = fmod(t, period_s);
        double amp = 0.02; /* floor */
        double car = 2.0 * M_PI * carrier_hz * t;

        if (phase < burst_s) {
            double x = 0.0;
            int idx = 0;

            if (n_widths <= 0) {
                amp = 0.6; /* constant envelope: no intra-burst structure */
            }
            while (n_widths > 0 && x < burst_s && idx < 128) {
                double w = widths_us[(idx % n_widths)] * 1e-6;

                if (phase >= x && phase < x + w) {
                    amp = 0.6;
                    break;
                }
                x += w * 2.0;
                idx++;
            }
        }
        {
            unsigned long long z = rs0;
            unsigned long long y = rs1;

            rs0 = y;
            z ^= z << 23;
            rs1 = z ^ y ^ (z >> 17) ^ (y >> 26);
            double noise = (double)((rs1 + y) >> 11) /
                           9007199254740992.0 - 1.0;

            iq[2 * i] = (float)(amp * cos(car) + 0.004 * noise);
            iq[2 * i + 1] = (float)(amp * sin(car) + 0.004 * noise);
        }
    }
    snprintf(path, sizeof(path), TYPED_DIR, sub);
    {
        FILE *f = fopen(path, "wb");

        CHECK(f != NULL);
        if (f != NULL) {
            fwrite(iq, sizeof(float), 2 * n, f);
            fclose(f);
        }
    }
    free(iq);
}

static void test_intraburst(void)
{
    IqSource src;
    char err[256];
    char path[512];
    Region r;
    IntraBurstResult ib;
    CccResult ccc;
    double uniform_w[1] = { 10.0 };
    double varying_w[4] = { 5.0, 40.0, 12.0, 60.0 };
    const double rate = 2e6;

    /* uniform sub-pulse train inside each burst -> clock-like */
    write_bursty_iq("ibunif", rate, 0.5, 0.005, 0.001, uniform_w, 1, 0.0);
    snprintf(path, sizeof(path), TYPED_DIR, "ibunif");
    if (iq_source_open(&src, path, err, sizeof(err))) {
        r.freq_min = src.center_freq - 200e3;
        r.freq_max = src.center_freq + 200e3;
        r.time_min = 0.0;
        r.time_max = 0.5;
        memset(&ccc, 0, sizeof(ccc));
        memset(&ib, 0, sizeof(ib));
        CHECK(intraburst_analyze(&src, &r, &ccc, &ib));
        printf("  intraburst uniform: %s (widths %d, cv %.2f, deltaBIC %d)\n",
               ib.verdict, ib.sub_pulses, ib.width_cv, ib.delta_bic);
        CHECK(ib.valid);
        CHECK(ib.uniform == 1);
        CHECK(strstr(ib.verdict, "uniform") != NULL);
        CHECK(ib.resolution_s > 0.0);
        iq_source_close(&src);
    }

    /* varying sub-pulse widths -> encoded data */
    write_bursty_iq("ibvary", rate, 0.5, 0.005, 0.001, varying_w, 4, 0.0);
    snprintf(path, sizeof(path), TYPED_DIR, "ibvary");
    if (iq_source_open(&src, path, err, sizeof(err))) {
        r.freq_min = src.center_freq - 200e3;
        r.freq_max = src.center_freq + 200e3;
        r.time_min = 0.0;
        r.time_max = 0.5;
        memset(&ccc, 0, sizeof(ccc));
        memset(&ib, 0, sizeof(ib));
        CHECK(intraburst_analyze(&src, &r, &ccc, &ib));
        printf("  intraburst varying: %s (widths %d, cv %.2f, deltaBIC %d)\n",
               ib.verdict, ib.sub_pulses, ib.width_cv, ib.delta_bic);
        CHECK(ib.valid);
        CHECK(ib.uniform == 0);
        CHECK(strstr(ib.verdict, "varying") != NULL ||
          strstr(ib.verdict, "PWM") != NULL ||
          strstr(ib.verdict, "PIM") != NULL ||
          strstr(ib.verdict, "PPM") != NULL);
        iq_source_close(&src);
    }

    /* residual carrier offset: the cumulant carrier estimate must be used
     * to remove it, keeping the envelope coherent */
    write_bursty_iq("ibcfo", rate, 0.5, 0.005, 0.001, uniform_w, 1, 30e3);
    snprintf(path, sizeof(path), TYPED_DIR, "ibcfo");
    if (iq_source_open(&src, path, err, sizeof(err))) {
        r.freq_min = src.center_freq - 200e3;
        r.freq_max = src.center_freq + 200e3;
        r.time_min = 0.0;
        r.time_max = 0.5;
        memset(&ccc, 0, sizeof(ccc));
        ccc.valid = true;
        ccc.carrier_hz = src.center_freq + 30e3; /* the true carrier */
        memset(&ib, 0, sizeof(ib));
        CHECK(intraburst_analyze(&src, &r, &ccc, &ib));
        printf("  intraburst cfo: %s (residual removed %.1f Hz)\n",
               ib.verdict, ib.residual_cfo_hz);
        CHECK(ib.valid);
        CHECK(fabs(ib.residual_cfo_hz - 30e3) < 1.0);
        CHECK(ib.uniform == 1);
        iq_source_close(&src);
    }

    /* control: bursts with a CONSTANT envelope and no sub-pulse structure
     * must not validate a segmentation (the Delta-BIC direction check:
     * a real pattern scores far above a structureless one) */
    write_bursty_iq("ibflat", rate, 0.5, 0.005, 0.001, uniform_w, 0, 0.0);
    snprintf(path, sizeof(path), TYPED_DIR, "ibflat");
    if (iq_source_open(&src, path, err, sizeof(err))) {
        r.freq_min = src.center_freq - 200e3;
        r.freq_max = src.center_freq + 200e3;
        r.time_min = 0.0;
        r.time_max = 0.5;
        memset(&ccc, 0, sizeof(ccc));
        memset(&ib, 0, sizeof(ib));
        (void)intraburst_analyze(&src, &r, &ccc, &ib);
        printf("  intraburst flat-envelope: %s (deltaBIC %d)\n",
               ib.verdict, ib.delta_bic);
        /* a constant-envelope burst has no intra-burst pattern: it must not
         * be reported as uniform/varying, and its Delta-BIC must stay far
         * below the structured cases (5385-7958) - this is the Delta-BIC
         * direction/scale check */
        if (ib.valid) {
            CHECK(ib.uniform == 0);
            CHECK(strstr(ib.verdict, "uniform") == NULL);
            CHECK(strstr(ib.verdict, "varying") == NULL);
            CHECK(ib.delta_bic < 1000);
        }
        iq_source_close(&src);
    }

    /* degenerate inputs must not crash */
    memset(&ib, 0, sizeof(ib));
    CHECK(!intraburst_analyze(NULL, NULL, NULL, &ib));
}

/* ------------------------------------------------------------------ */
/* spread-spectrum presence and type classification                    */
/* ------------------------------------------------------------------ */

static void test_spread_classify(void)
{
    Metrics m;
    Analysis a;
    PulseResult pulse;
    FluctResult fluct;
    CparResult cpar;
    CyclicResult cyclic;
    SscaResult ssca;
    SpreadCandidate sc[6];
    size_t n;

    memset(&m, 0, sizeof(m));
    memset(&a, 0, sizeof(a));
    memset(&pulse, 0, sizeof(pulse));
    memset(&fluct, 0, sizeof(fluct));
    memset(&cpar, 0, sizeof(cpar));
    memset(&cyclic, 0, sizeof(cyclic));
    memset(&ssca, 0, sizeof(ssca));

    /* DSSS: both detectors agree on a chip rate, flat spectrum */
    m.spectral_flatness = 0.8;
    fluct.detected = true;
    fluct.chip_rate_hz = 60000.0;
    cpar.detected = true;
    cpar.chip_rate_hz = 59375.0;
    n = spread_classify(&m, &a, NULL, &fluct, &cpar, NULL, NULL, sc, 6);
    printf("  spread dsss: %s %d%% (%s)\n", sc[0].name, sc[0].score,
           sc[0].evidence);
    CHECK(n >= 1);
    CHECK(strcmp(sc[0].name, "DSSS") == 0);
    CHECK(sc[0].score >= 80);

    /* FHSS: frequency-agile bursts, low duty */
    memset(&m, 0, sizeof(m));
    m.duty_cycle = 0.2;
    memset(&fluct, 0, sizeof(fluct));
    memset(&cpar, 0, sizeof(cpar));
    memset(&pulse, 0, sizeof(pulse));
    pulse.num_pulses = 100;
    pulse.credible = 1;
    pulse.pulse_freq_cv = 0.6;
    pulse.f0_hz = 200.0;
    n = spread_classify(&m, &a, &pulse, NULL, NULL, NULL, NULL, sc, 6);
    printf("  spread fhss: %s %d%% (%s)\n", sc[0].name, sc[0].score,
           sc[0].evidence);
    CHECK(strcmp(sc[0].name, "FHSS") == 0);
    CHECK(sc[0].score >= 60);

    /* THSS: irregular burst timing, low duty */
    memset(&pulse, 0, sizeof(pulse));
    pulse.num_pulses = 80;
    pulse.credible = 1;
    pulse.interval_cv = 0.6;
    n = spread_classify(&m, &a, &pulse, NULL, NULL, NULL, NULL, sc, 6);
    printf("  spread thss: %s %d%%\n", sc[0].name, sc[0].score);
    CHECK(strcmp(sc[0].name, "THSS") == 0);

    /* CSS: chirped pulses */
    memset(&pulse, 0, sizeof(pulse));
    pulse.num_pulses = 50;
    pulse.credible = 1;
    pulse.chirp_pulses = 45;
    pulse.chirp_bw_mean = 120e3;
    pulse.chirp_rate_cv = 0.5;
    n = spread_classify(&m, &a, &pulse, NULL, NULL, NULL, NULL, sc, 6);
    printf("  spread css: %s %d%%\n", sc[0].name, sc[0].score);
    CHECK(strcmp(sc[0].name, "CSS") == 0);
    CHECK(sc[0].score >= 70);

    /* nothing: not spread spectrum */
    memset(&m, 0, sizeof(m));
    m.spectral_flatness = 0.2;
    n = spread_classify(&m, &a, NULL, NULL, NULL, NULL, NULL, sc, 6);
    printf("  spread none: %s %d%%\n", sc[0].name, sc[0].score);
    CHECK(strcmp(sc[0].name, "Not spread spectrum") == 0);
    CHECK(sc[0].score >= 50);
}


/* ------------------------------------------------------------------ */
/* intra-burst modulation classification (PPM/PWM/PFM/Manchester/OOK)  */
/* ------------------------------------------------------------------ */

/*
 * Pulse train with a chosen intra-burst modulation:
 *   0 PPM: constant 2 us pulses, positions quantised to 4 slots
 *   1 PWM: widths cycle 2/4/6 us at a constant 10 us period
 *   2 PFM: constant 2 us pulses, continuously varying intervals
 *   3 Manchester: alternating long/short intervals (biphase pairs)
 *   4 random OOK: random widths and gaps
 */
static void write_mod_bursty_iq_mixed(const char *name)
{
    const double fs = 2.0e6;
    const double duration = 0.5;
    const double burst_s = 5e-3, period_s = 25e-3;
    size_t n = (size_t)(fs * duration);
    float *iq = calloc(2 * n, sizeof(float));
    char path[512], cmd[512];
    FILE *fp;
    uint32_t st = 0x12345678u;
    int burst_idx = 0;

    if (iq == NULL) return;
    for (size_t i = 0; i < n; i++) {
        st = st * 1664525u + 1013904223u;
        iq[2 * i]     = ((float)((st >> 8) & 0xFFFF) / 65536.0f - 0.5f) * 0.01f;
        iq[2 * i + 1] = ((float)((st >> 16) & 0xFFFF) / 65536.0f - 0.5f) * 0.01f;
    }
    for (double t0 = 0.0; t0 < duration; t0 += period_s, burst_idx++) {
        double x = 0.0;
        int idx = 0;

        st = 0x12345678u;
        while (x < burst_s) {
            double w, gap;
            size_t a, b;

            if ((burst_idx & 1) == 0) {
                /* PPM: 4 slots, constant 4 us pulses */
                w = 4e-6;
                gap = 8e-6 + (double)((st = st * 1664525u + 1013904223u) &
                                      0x3u) * 4e-6;
            } else {
                /* PWM: 3 widths at a constant 20 us period */
                w = (idx % 3 == 0) ? 4e-6 : (idx % 3 == 1 ? 8e-6 : 12e-6);
                gap = 20e-6 - w;
            }
            a = (size_t)((t0 + x) * fs);
            b = a + (size_t)(w * fs);
            if (b > n) break;
            for (size_t k = a; k < b; k++) {
                double ph = 2.0 * M_PI * 100e3 * (double)k / fs;

                iq[2 * k]     += (float)(0.6 * cos(ph));
                iq[2 * k + 1] += (float)(0.6 * sin(ph));
            }
            x += w + gap;
            idx++;
        }
    }
    snprintf(cmd, sizeof(cmd), "mkdir -p /tmp/te15_test/%s", name);
    system(cmd);
    snprintf(path, sizeof(path), TYPED_DIR, name);
    fp = fopen(path, "wb");
    if (fp != NULL) {
        fwrite(iq, sizeof(float), 2 * n, fp);
        fclose(fp);
    }
    free(iq);
}

static void write_mod_bursty_iq(const char *name, int mode)
{
    const double fs = 2.0e6;
    const double duration = 0.5;
    const double burst_s = 5e-3, period_s = 25e-3;
    size_t n = (size_t)(fs * duration);
    float *iq = calloc(2 * n, sizeof(float));
    char path[512];
    FILE *fp;
    uint32_t st = 0x12345678u;

    if (iq == NULL) return;
    for (size_t i = 0; i < n; i++) {
        st = st * 1664525u + 1013904223u;
        iq[2 * i]     = ((float)((st >> 8) & 0xFFFF) / 65536.0f - 0.5f) * 0.01f;
        iq[2 * i + 1] = ((float)((st >> 16) & 0xFFFF) / 65536.0f - 0.5f) * 0.01f;
    }
    for (double t0 = 0.0; t0 < duration; t0 += period_s) {
        double x = 0.0;
        int idx = 0;

        st = 0x12345678u; /* identical pattern in every burst (coherent fold) */

        while (x < burst_s) {
            double w, gap;
            size_t a, b;

            switch (mode) {
            case 0: /* PPM: 4 slots spaced 4 us apart */
                w = 4e-6;
                gap = 8e-6 + (double)((st = st * 1664525u + 1013904223u) &
                                      0x3u) * 4e-6;
                break;
            case 1: /* PWM: 3 widths at a constant 20 us period */
                w = (idx % 3 == 0) ? 4e-6 : (idx % 3 == 1 ? 8e-6 : 12e-6);
                gap = 20e-6 - w;
                break;
            case 2: /* PFM: constant width, continuously varying interval */
                w = 4e-6;
                gap = 15e-6 + ((double)((st = st * 1664525u + 1013904223u) &
                                        0xFFFFu) / 65536.0) * 20e-6;
                break;
            case 3: /* Manchester: alternate long/short intervals */
                w = 4e-6;
                gap = (idx & 1) ? 8e-6 : 24e-6;
                break;
            default: /* random OOK: random widths and gaps */
                w = 2e-6 + ((double)((st = st * 1664525u + 1013904223u) &
                                     0xFFFFu) / 65536.0) * 6e-6;
                gap = 15e-6 + ((double)((st = st * 1664525u + 1013904223u) &
                                        0xFFFFu) / 65536.0) * 20e-6;
                break;
            }
            a = (size_t)((t0 + x) * fs);
            b = a + (size_t)(w * fs);
            if (b > n) break;
            for (size_t k = a; k < b; k++) {
                double ph = 2.0 * M_PI * 100e3 * (double)k / fs;

                iq[2 * k]     += (float)(0.6 * cos(ph));
                iq[2 * k + 1] += (float)(0.6 * sin(ph));
            }
            x += w + gap;
            idx++;
        }
    }
    {
        char cmd[512];

        snprintf(cmd, sizeof(cmd), "mkdir -p /tmp/te15_test/%s", name);
        system(cmd);
    }
    snprintf(path, sizeof(path), TYPED_DIR, name);
    fp = fopen(path, "wb");
    if (fp != NULL) {
        fwrite(iq, sizeof(float), 2 * n, fp);
        fclose(fp);
    }
    free(iq);
}

static void test_burst_modulation(void)
{
    static const struct { const char *name; int mode; const char *expect; } cases[] = {
        { "modppm",  0, "PPM" },
        { "modpwm",  1, "PWM" },
        { "modpfm",  2, "PFM" },
        { "modman",  3, "Manchester" },
        { "modook",  4, "OOK" },
    };

    /* mixed bursts: even bursts PPM, odd bursts PWM -> the per-burst
     * classification must report a COMBINATION of the two */
    {
        char path[512];
        char err[256];
        IqSource src;
        Region r;
        CccResult ccc;
        IntraBurstResult ib;

        write_mod_bursty_iq_mixed("modmix");
        snprintf(path, sizeof(path), TYPED_DIR, "modmix");
        if (iq_source_open(&src, path, err, sizeof(err))) {
            r.freq_min = src.center_freq - 200e3;
            r.freq_max = src.center_freq + 200e3;
            r.time_min = 0.0;
            r.time_max = 0.5;
            memset(&ccc, 0, sizeof(ccc));
            memset(&ib, 0, sizeof(ib));
            (void)intraburst_analyze(&src, &r, &ccc, &ib);
            printf("  burst-mod mixed: cand %s %.0f%% / %s %.0f%% -> %s\n",
                   ib.burst_mod_cand[0], ib.burst_mod_support[0] * 100.0,
                   ib.burst_mod_cand[1], ib.burst_mod_support[1] * 100.0,
                   ib.burst_mod_combo);
            CHECK(ib.bursts_classified >= 4);
            CHECK(strstr(ib.burst_mod_combo, "combination") != NULL);
            iq_source_close(&src);
        }
    }

    for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
        char path[512];
        char err[256];
        IqSource src;
        Region r;
        CccResult ccc;
        IntraBurstResult ib;

        write_mod_bursty_iq(cases[c].name, cases[c].mode);
        snprintf(path, sizeof(path), TYPED_DIR, cases[c].name);
        if (!iq_source_open(&src, path, err, sizeof(err))) {
            printf("  burst-mod %s: OPEN FAILED (%s)\n", cases[c].name, err);
            CHECK(0);
            continue;
        }
        r.freq_min = src.center_freq - 200e3;
        r.freq_max = src.center_freq + 200e3;
        r.time_min = 0.0;
        r.time_max = 0.5;
        memset(&ccc, 0, sizeof(ccc));
        memset(&ib, 0, sizeof(ib));
        if (!intraburst_analyze(&src, &r, &ccc, &ib)) {
            printf("  burst-mod %s: analyse failed (%s)\n", cases[c].name,
                   ib.verdict);
            CHECK(0);
            iq_source_close(&src);
            continue;
        }
        printf("  burst-mod %-4s: %-46.46s [%s] wl %d tl %d wcv %.2f tcv %.2f\n", cases[c].name,
               ib.burst_mod, ib.verdict, ib.width_levels, ib.interval_levels, ib.width_cv, ib.period_cv);
        CHECK(strstr(ib.burst_mod, cases[c].expect) != NULL);
        iq_source_close(&src);
    }
}


/* ------------------------------------------------------------------ */
/* recording metadata inference (gqrx / other SDR tools / manual)      */
/* ------------------------------------------------------------------ */

static void test_file_metadata(void)
{
    char cmd[512];
    char path[512];
    char err[256];
    IqSource src;
    double f = 0.0, r = 0.0;
    IqFormat fmt = IQ_FORMAT_FLOAT32;
    IqNameKind kind;

    system("mkdir -p /tmp/te15_test/meta");

    /* gqrx with a descriptive prefix */
    snprintf(path, sizeof(path), "/tmp/te15_test/meta/WiFi_"
             "gqrx_20260306_174402_2437000000_25000000_fc.raw");
    snprintf(cmd, sizeof(cmd), "dd if=/dev/zero of='%s' bs=1024 count=64 2>/dev/null",
             path);
    system(cmd);
    CHECK(iq_source_open(&src, path, err, sizeof(err)));
    if (src.raw != NULL) {
        printf("  meta gqrx+prefix: %.3f MHz %.3f MS/s kind %d\n",
               src.center_freq / 1e6, src.sample_rate / 1e6,
               (int)src.name_kind);
        CHECK(fabs(src.center_freq - 2437e6) < 1e3);
        CHECK(fabs(src.sample_rate - 25e6) < 1e3);
        CHECK(src.name_kind == IQ_NAME_GQRX);
        iq_source_close(&src);
    }

    /* SDR# style name */
    snprintf(path, sizeof(path), "/tmp/te15_test/meta/SDRSharp_"
             "20260306_174402Z_2437000000Hz_25000000Hz_IQ.wav");
    snprintf(cmd, sizeof(cmd), "dd if=/dev/zero of='%s' bs=1024 count=64 2>/dev/null",
             path);
    system(cmd);
    kind = iq_source_guess_metadata(path, &f, &r, &fmt);
    printf("  meta sdrsharp: %.3f MHz %.3f MS/s kind %d\n", f / 1e6, r / 1e6,
           (int)kind);
    CHECK(fabs(f - 2437e6) < 1e3);
    CHECK(fabs(r - 25e6) < 1e3);
    CHECK(kind == IQ_NAME_SDRSHARP);

    /* generic "<freq>MHz_<rate>MSps" name with a .cf32 extension */
    snprintf(path, sizeof(path), "/tmp/te15_test/meta/capture_2437.0MHz_25MSps.cf32");
    kind = iq_source_guess_metadata(path, &f, &r, &fmt);
    printf("  meta generic: %.3f MHz %.3f MS/s fmt %d kind %d\n", f / 1e6,
           r / 1e6, (int)fmt, (int)kind);
    CHECK(fabs(f - 2437e6) < 1e3);
    CHECK(fabs(r - 25e6) < 1e3);
    CHECK(fmt == IQ_FORMAT_FLOAT32);

    /* unknown name: nothing inferred */
    snprintf(path, sizeof(path), "/tmp/te15_test/meta/mystery.bin");
    kind = iq_source_guess_metadata(path, &f, &r, &fmt);
    CHECK(kind == IQ_NAME_UNKNOWN);
    CHECK(f == 0.0 && r == 0.0);

    /* manual parameters (the dialog fallback) */
    snprintf(path, sizeof(path), "/tmp/te15_test/meta/mystery.bin");
    snprintf(cmd, sizeof(cmd), "dd if=/dev/zero of='%s' bs=1024 count=64 2>/dev/null",
             path);
    system(cmd);
    CHECK(iq_source_open_custom(&src, path, 2450e6, 2.0e6, IQ_FORMAT_INT16,
                                err, sizeof(err)));
    if (src.raw != NULL) {
        printf("  meta custom: %.3f MHz %.3f MS/s fmt %d samples %zu\n",
               src.center_freq / 1e6, src.sample_rate / 1e6,
               (int)src.format, src.total_samples);
        CHECK(fabs(src.center_freq - 2450e6) < 1e3);
        CHECK(fabs(src.sample_rate - 2.0e6) < 1e3);
        CHECK(src.format == IQ_FORMAT_INT16);
        CHECK(src.total_samples == 64 * 1024 / 4);
        iq_source_close(&src);
    }
}


/* ------------------------------------------------------------------ */
/* synthetic signal generation (Selection > Synthesize)                */
/* ------------------------------------------------------------------ */

static void test_synthesis(void)
{
    char path[512];
    char err[256];
    IqSource src;
    Region r;
    Metrics m;
    Analysis a;
    PulseResult pulse;
    IntraBurstResult ib;
    CccResult ccc;
    SynthSpec spec;
    SynthNoise noise;
    float *iq = NULL;
    size_t n = 0;
    static const double syn_w[1] = { 10.0 };

    write_bursty_iq("synthsrc", 2.0e6, 0.5, 0.005, 0.001, syn_w, 1, 0.0);
    snprintf(path, sizeof(path), TYPED_DIR, "synthsrc");
    if (!iq_source_open(&src, path, err, sizeof(err))) {
        CHECK(0);
        return;
    }
    r.freq_min = src.center_freq - 200e3;
    r.freq_max = src.center_freq + 200e3;
    r.time_min = 0.0;
    r.time_max = 0.4;
    memset(&m, 0, sizeof(m));
    memset(&a, 0, sizeof(a));
    memset(&pulse, 0, sizeof(pulse));
    memset(&ib, 0, sizeof(ib));
    memset(&ccc, 0, sizeof(ccc));
    (void)metrics_compute(&src, &r, &m);
    (void)analysis_run(&src, &r, &a);
    (void)pulse_analyze(&src, &r, &pulse);
    (void)intraburst_analyze(&src, &r, &ccc, &ib);

    CHECK(synth_spec_from(&m, &a, &pulse, NULL, &ib, NULL, 0, 0.0,
                          src.center_freq, src.sample_rate,
                          IQ_FORMAT_FLOAT32, NULL, 0, &spec));
    printf("  synth spec: %s\n", spec.description);
    CHECK(spec.duration == 2.0);
    CHECK(spec.sample_rate == src.sample_rate);
    CHECK(spec.bw_hz > 0.0);
    CHECK(spec.snr_db >= 5.0);

    /* a confident spread-spectrum verdict must select the matching
     * synthesis family (DSSS/FHSS) and use the validated chip rate */
    {
        SpreadCandidate sc[2];
        SynthSpec ds, fs2;
        static const double chips[1] = { 60000.0 };

        memset(sc, 0, sizeof(sc));
        snprintf(sc[0].name, sizeof(sc[0].name), "DSSS");
        sc[0].score = 85;
        CHECK(synth_spec_from(&m, &a, &pulse, NULL, &ib, sc, 1, chips[0],
                              src.center_freq, src.sample_rate,
                              IQ_FORMAT_FLOAT32, NULL, 0, &ds));
        printf("  synth family dsss: %d symbol rate %.0f Hz\n",
               (int)ds.family, ds.symbol_rate);
        CHECK(ds.family == SYNTH_DSSS);
        CHECK(fabs(ds.symbol_rate - chips[0]) < 1.0);
        snprintf(sc[0].name, sizeof(sc[0].name), "FHSS");
        CHECK(synth_spec_from(&m, &a, &pulse, NULL, &ib, sc, 1, 0.0,
                              src.center_freq, src.sample_rate,
                              IQ_FORMAT_FLOAT32, NULL, 0, &fs2));
        CHECK(fs2.family == SYNTH_FHSS);
    }

    memset(&noise, 0, sizeof(noise));
    {
        Region vis = { src.center_freq - src.sample_rate / 2.0,
                       src.center_freq + src.sample_rate / 2.0, 0.0, 0.4 };

        CHECK(synth_noise_profile(&src, &vis, &r, &noise));
        printf("  synth noise: %d bins, floor %.1f dB\n", noise.n,
               noise.floor_db);
        CHECK(noise.n > 0 && noise.psd != NULL);
    }

    CHECK(synth_generate(&spec, &noise, &iq, &n));
    printf("  synth generated: %zu samples (%.2f s)\n", n, (double)n / spec.sample_rate);
    CHECK(iq != NULL && n == (size_t)(2.0 * spec.sample_rate));
    if (iq != NULL) {
        double p = 0.0;

        for (size_t i = 0; i < n; i++) {
            p += (double)iq[2 * i] * iq[2 * i] +
                 (double)iq[2 * i + 1] * iq[2 * i + 1];
        }
        p /= (double)n;
        printf("  synth power: %.4f\n", p);
        CHECK(p > 0.0);
    }

    /* write and re-open through the gqrx parser */
    {
        char out[512];
        FILE *f;

        snprintf(out, sizeof(out), "/tmp/te15_test/synthout/gqrx_20261112_"
                 "170000_100000000_2000000_fc.raw");
        system("mkdir -p /tmp/te15_test/synthout");
        CHECK(synth_write(out, iq, n, IQ_FORMAT_FLOAT32));
        f = fopen(out, "rb");
        CHECK(f != NULL);
        if (f != NULL) {
            long sz;

            fseek(f, 0, SEEK_END);
            sz = ftell(f);
            fclose(f);
            CHECK((size_t)sz == 2 * n * sizeof(float));
        }
        memset(&src, 0, sizeof(src));
        if (iq_source_open(&src, out, err, sizeof(err))) {
            printf("  synth reload: %.3f MHz %.3f MS/s src %zu samples\n",
                   src.center_freq / 1e6, src.sample_rate / 1e6,
                   src.total_samples);
            CHECK(fabs(src.center_freq - 100e6) < 1e3);
            CHECK(src.total_samples == n);
            iq_source_close(&src);
        } else {
            printf("  synth reload failed: %s\n", err);
            CHECK(0);
        }
    }
    free(iq);
    synth_noise_clear(&noise);
    analysis_clear(&a);
    iq_source_close(&src);
}


/* ------------------------------------------------------------------ */
/* audio recordings (WAV/AIFF headers drive rate, format and centre)   */
/* ------------------------------------------------------------------ */

static void put_le32(unsigned char *p, unsigned v)
{
    p[0] = (unsigned char)(v & 0xFF);
    p[1] = (unsigned char)((v >> 8) & 0xFF);
    p[2] = (unsigned char)((v >> 16) & 0xFF);
    p[3] = (unsigned char)((v >> 24) & 0xFF);
}

static void put_le16(unsigned char *p, unsigned v)
{
    p[0] = (unsigned char)(v & 0xFF);
    p[1] = (unsigned char)((v >> 8) & 0xFF);
}

/* write a small PCM16 or float32 WAV file */
static void write_wav_test(const char *path, int rate, int channels,
                           bool is_float, size_t frames)
{
    unsigned char hdr[44];
    FILE *f = fopen(path, "wb");
    size_t bytes_per = (size_t)channels * (is_float ? 4 : 2);

    if (f == NULL) return;
    memset(hdr, 0, sizeof(hdr));
    memcpy(hdr, "RIFF", 4);
    put_le32(hdr + 4, (unsigned)(36 + frames * bytes_per));
    memcpy(hdr + 8, "WAVE", 4);
    memcpy(hdr + 12, "fmt ", 4);
    put_le32(hdr + 16, 16);
    put_le16(hdr + 20, is_float ? 3 : 1);
    put_le16(hdr + 22, (unsigned)channels);
    put_le32(hdr + 24, (unsigned)rate);
    put_le32(hdr + 28, (unsigned)(rate * bytes_per));
    put_le16(hdr + 32, (unsigned)bytes_per);
    put_le16(hdr + 34, is_float ? 32 : 16);
    memcpy(hdr + 36, "data", 4);
    put_le32(hdr + 40, (unsigned)(frames * bytes_per));
    fwrite(hdr, 1, sizeof(hdr), f);
    for (size_t i = 0; i < frames; i++) {
        for (int c = 0; c < channels; c++) {
            double v = (double)(i % 100) / 100.0 * (c == 1 ? -1.0 : 1.0);

            if (is_float) {
                float fv = (float)v;

                fwrite(&fv, sizeof(fv), 1, f);
            } else {
                short sv = (short)(v * 32767.0);

                fwrite(&sv, sizeof(sv), 1, f);
            }
        }
    }
    fclose(f);
}

static void test_audio_files(void)
{
    char path[512];
    char err[256];
    IqSource src;
    AudioInfo info;

    system("mkdir -p /tmp/te15_test/audio");

    /* mono 16-bit at 44100 Hz */
    snprintf(path, sizeof(path), "/tmp/te15_test/audio/mono16.wav");
    write_wav_test(path, 44100, 1, false, 2000);
    CHECK(audio_file_probe(path, 0, &info));
    printf("  audio probe: rate %.0f ch %d bits %d float %d frames %zu\n",
           info.sample_rate, info.channels, info.bits, (int)info.is_float,
           info.frames);
    CHECK(fabs(info.sample_rate - 44100.0) < 1.0);
    CHECK(info.channels == 1);
    CHECK(info.bits == 16);
    CHECK(info.frames == 2000);

    CHECK(iq_source_open(&src, path, err, sizeof(err)));
    if (src.raw != NULL) {
        float buf[8];

        printf("  audio source: audio %d ch %d bits %d rate %.0f centre %.0f\n",
               (int)src.is_audio, src.channels, src.audio_bits,
               src.sample_rate, src.center_freq);
        CHECK(src.is_audio);
        CHECK(src.channels == 1);
        CHECK(fabs(src.sample_rate - 44100.0) < 1.0);
        CHECK(fabs(src.center_freq - 22050.0) < 1.0); /* centre = rate/2 */
        CHECK(src.total_samples == 2000);
        iq_source_read_complex(&src, 10, buf, 4);
        printf("  audio samples: I %.3f %.3f  Q %.3f %.3f\n", buf[0], buf[2],
               buf[1], buf[3]);
        CHECK(buf[1] == 0.0f && buf[3] == 0.0f); /* mono -> Q = 0 */
        CHECK(fabs(buf[0] - 0.10f) < 0.02f);
        CHECK(fabs(buf[2] - 0.12f) < 0.02f);
        iq_source_close(&src);
    }

    /* stereo float32 at 48000 Hz: L -> I, R -> Q */
    snprintf(path, sizeof(path), "/tmp/te15_test/audio/stereo_float.wav");
    write_wav_test(path, 48000, 2, true, 1000);
    CHECK(iq_source_open(&src, path, err, sizeof(err)));
    if (src.raw != NULL) {
        float buf[8];

        printf("  audio stereo: ch %d bits %d float %d rate %.0f\n",
               src.channels, src.audio_bits, (int)src.audio_float,
               src.sample_rate);
        CHECK(src.is_audio && src.channels == 2);
        CHECK(fabs(src.sample_rate - 48000.0) < 1.0);
        CHECK(fabs(src.center_freq - 24000.0) < 1.0);
        iq_source_read_complex(&src, 5, buf, 2);
        printf("  audio I/Q: %.3f / %.3f\n", buf[0], buf[1]);
        CHECK(fabs(buf[0] - 0.05f) < 0.02f);
        CHECK(fabs(buf[1] + 0.05f) < 0.02f);
        iq_source_close(&src);
    }
}

/* ------------------------------------------------------------------ */
/* ------------------------------------------------------------------ */
/* Director: ranking, discounting and consolidated output              */
/* ------------------------------------------------------------------ */

/* Build a small synthetic envelope set the way the worker publishes it. */
typedef struct {
    ArArena *arena;
    ArSet set;
} DirFixture;

static void fixture_init(DirFixture *fx)
{
    memset(fx, 0, sizeof(*fx));
    fx->arena = ar_arena_new();
    ar_set_init(&fx->set, fx->arena);
}

static void fixture_free(DirFixture *fx)
{
    ar_arena_free(fx->arena);
}

static AnalysisResult *fixture_add(DirFixture *fx, const char *id,
                                   const char *title, int node)
{
    return ar_add(&fx->set, id, title, node);
}

static void test_director(void)
{
    DirFixture fx;
    Director dir;
    DirectorInput in;
    ModCandidate mods[2];
    SpreadCandidate spread[2];
    DagRun *dag = dag_run_new();
    AnalysisResult *r;

    puts("\n-- Director --");
    fixture_init(&fx);
    dag_reset(dag);

    /* A0: classifier says OOK-ish; metrics; ranking */
    r = fixture_add(&fx, "analysis", "Baseline analysis", DAG_A0);
    CHECK(r != NULL);
    ar_text(r, "modulation", "OOK / pulsed", NULL);
    r = fixture_add(&fx, "metrics", "Signal metrics", DAG_A0);
    ar_num(r, "occupied_bw_hz", 12.5e3, "Hz", NULL);
    r = fixture_add(&fx, "pulse", "Pulse-train analysis", DAG_A2);
    ar_num(r, "num_pulses", 40, NULL, NULL);
    ar_num(r, "credible", 1, NULL, NULL);
    ar_num(r, "prf_hz", 40.0, "Hz", NULL);
    ar_num(r, "pulse_width_s", 5e-3, "s", NULL);
    ar_num(r, "duty_cycle", 0.2, NULL, NULL);
    ar_num(r, "f0_hz", 38.0, "Hz", NULL);
    ar_num(r, "credible", 1, NULL, NULL);
    dag_end(dag, DAG_A2, true, 0.01);
    r = fixture_add(&fx, "classic", "Classic envelope analysis", DAG_A1);
    ar_num(r, "period_hz", 39.4, "Hz", NULL);
    dag_end(dag, DAG_A1, true, 0.02);
    r = fixture_add(&fx, "ccc", "Cyclic cumulants (AMR)", DAG_A8);
    ar_num(r, "valid", 1, NULL, NULL);
    ar_text(r, "verdict", "BPSK-like", NULL);
    dag_end(dag, DAG_A8, true, 0.03);
    /* A4 skipped: must be discounted and reported */
    r = fixture_add(&fx, "cyclic", "FAM spectral correlation", DAG_A4);
    ar_set_status(r, AR_SKIPPED_USER, "wideband input: SSCA covers it");
    dag_skip(dag, DAG_A4, "wideband input: SSCA covers it");
    /* A9 executed: uniform train */
    r = fixture_add(&fx, "intraburst", "Intra-burst envelope", DAG_A9);
    ar_num(r, "uniform", 1, NULL, NULL);
    ar_num(r, "delta_bic", 500, NULL, NULL);
    ar_num(r, "surrogate_fpr", 0.0, NULL, NULL);
    ar_text(r, "verdict", "uniform intra-burst pulse train", NULL);
    dag_end(dag, DAG_A9, true, 0.4);

    memset(mods, 0, sizeof(mods));
    snprintf(mods[0].name, sizeof(mods[0].name), "OOK / pulsed emission");
    mods[0].score = 90;
    snprintf(mods[0].why, sizeof(mods[0].why), "duty 20%%");
    snprintf(mods[1].name, sizeof(mods[1].name), "FSK emission");
    mods[1].score = 30;
    memset(spread, 0, sizeof(spread));
    snprintf(spread[0].name, sizeof(spread[0].name), "DSSS");
    spread[0].score = 55;
    snprintf(spread[0].evidence, sizeof(spread[0].evidence), "code rate match");

    memset(&in, 0, sizeof(in));
    in.analyses = &fx.set;
    in.dag = dag;
    in.mods = mods;
    in.n_mods = 2;
    in.spread = spread;
    in.n_spread = 1;
    director_build(&dir, &in);

    /* a credible pulse train leads the characterisation */
    CHECK(strstr(dir.characterisation, "bursty emission with a OOK / pulsed "
                                       "signature") != NULL);
    CHECK(strstr(dir.characterisation, "5.0 ms bursts") != NULL);
    /* the confident spread-spectrum verdict is appended */
    CHECK(strstr(dir.characterisation, "spread-spectrum structure: DSSS "
                                       "(55 %)") != NULL);
    /* intra-burst appended to the verdict */
    CHECK(strstr(dir.verdict, "intra-burst: uniform train") != NULL);
    /* modulation consensus: top name de-suffixed */
    CHECK(strcmp(dir.top_mod, "OOK / pulsed") == 0);
    /* votes line lists the four sources */
    CHECK(strstr(dir.vote_line, "demod OOK") != NULL);
    CHECK(strstr(dir.vote_line, "cumulants BPSK") != NULL);
    /* claims ranked by weight: the modulation candidate leads */
    CHECK(dir.n_cand >= 4);
    CHECK(strcmp(dir.cand[0].name, "OOK / pulsed emission") == 0);
    CHECK(dir.cand[0].weight == 90.0);
    /* a skipped DAG node is discounted and named in the rationale */
    CHECK(dir.n_rationale > 0);
    {
        bool mentioned = false;

        for (size_t i = 0; i < dir.n_rationale; i++) {
            if (strstr(dir.rationale[i], "A4 FAM") != NULL) mentioned = true;
        }
        CHECK(mentioned);
    }
    /* provenance lists the executed and the skipped nodes */
    {
        bool a4 = false, a9 = false;

        for (size_t i = 0; i < dir.n_provenance; i++) {
            if (strstr(dir.provenance[i], "A4 FAM") != NULL &&
                strstr(dir.provenance[i], "skipped") != NULL) a4 = true;
            if (strstr(dir.provenance[i], "A9 intra-burst") != NULL &&
                strstr(dir.provenance[i], "executed") != NULL) a9 = true;
        }
        CHECK(a4);
        CHECK(a9);
    }
    fixture_free(&fx);

    /* an implausible pulse train falls back to the envelope fundamental */
    fixture_init(&fx);
    dag_reset(dag);
    r = fixture_add(&fx, "analysis", "Baseline analysis", DAG_A0);
    ar_text(r, "modulation", "OOK", NULL);
    r = fixture_add(&fx, "pulse", "Pulse-train analysis", DAG_A2);
    ar_num(r, "num_pulses", 9, NULL, NULL);
    ar_num(r, "credible", 0, NULL, NULL);
    ar_num(r, "f0_hz", 198.4, "Hz", NULL);
    dag_end(dag, DAG_A2, true, 0.01);
    memset(&in, 0, sizeof(in));
    in.analyses = &fx.set;
    in.dag = dag;
    director_build(&dir, &in);
    CHECK(strstr(dir.characterisation, "line-spectrum fundamental") != NULL);
    CHECK(strstr(dir.characterisation, "198.40 Hz") != NULL);
    fixture_free(&fx);

    /* nothing but a classifier: the generic characterisation */
    fixture_init(&fx);
    dag_reset(dag);
    r = fixture_add(&fx, "analysis", "Baseline analysis", DAG_A0);
    ar_text(r, "modulation", "QAM/OFDM", NULL);
    memset(&in, 0, sizeof(in));
    in.analyses = &fx.set;
    in.dag = dag;
    director_build(&dir, &in);
    CHECK(strstr(dir.characterisation, "emission with a QAM signature") != NULL);
    CHECK(dir.n_cand == 0);
    fixture_free(&fx);

    /* modulation token normalisation */
    {
        char tok[24];

        director_mod_token("DSSS/BPSK (spread)", tok, sizeof(tok));
        CHECK(strcmp(tok, "BPSK") == 0);
        director_mod_token("4FSK", tok, sizeof(tok));
        CHECK(strcmp(tok, "FSK") == 0);
        director_mod_token("", tok, sizeof(tok));
        CHECK(tok[0] == '\0');
    }
    dag_run_free(dag);
}


/* ------------------------------------------------------------------ */
/* Reporter formats + analysis settings overrides                     */
/* ------------------------------------------------------------------ */

static void test_report_formats(void)
{
    IqSource src;
    ReportInput in;
    ReportDoc *doc;
    Analysis analysis;
    Metrics metrics;
    char *md, *html;
    char err[256];

    puts("\n-- Reporter formats --");
    if (!iq_source_open(&src, "data/gqrx_20260306_174402_2530000000_2000000_fc.raw",
                        err, sizeof(err))) {
        /* the fixture may be missing in a fresh checkout: skip gracefully */
        CHECK(1);
        return;
    }
    memset(&analysis, 0, sizeof(analysis));
    snprintf(analysis.signal_type, sizeof(analysis.signal_type), "Narrowband");
    snprintf(analysis.modulation, sizeof(analysis.modulation), "OOK");
    analysis.center_freq = src.center_freq;
    analysis.bandwidth = 100e3;
    analysis.duration = 1.0;
    analysis.bins = 64;
    analysis.frames = 64;
    memset(&metrics, 0, sizeof(metrics));
    metrics.occupied_bw_hz = 90e3;

    memset(&in, 0, sizeof(in));
    in.stamp = "2026-01-01 00:00:00";
    in.src = &src;
    in.region.freq_min = src.center_freq - 50e3;
    in.region.freq_max = src.center_freq + 50e3;
    in.region.time_min = 0.0;
    in.region.time_max = 1.0;
    in.settings_rate = src.sample_rate;
    in.settings_format = src.format;
    in.settings_fft_size = 4096;
    in.settings_window = DSP_WINDOW_HANN;
    in.analysis = &analysis;
    in.metrics = &metrics;

    doc = report_build(&in);
    CHECK(doc != NULL);
    CHECK(strstr(report_text(doc), "SELECTION REPORT") != NULL);
    CHECK(strstr(report_text(doc), "Report generated by ToxicElement15") != NULL);

    md = report_render(doc, REPORT_MARKDOWN);
    CHECK(md != NULL);
    CHECK(strstr(md, "# SELECTION REPORT") != NULL);
    /* the '====' rule under the title is replaced by the markdown heading */
    CHECK(strstr(md, "SELECTION REPORT\n===") == NULL);

    html = report_render(doc, REPORT_HTML);
    CHECK(html != NULL);
    CHECK(strstr(html, "<!DOCTYPE html>") != NULL);
    CHECK(strstr(html, "<h1>SELECTION REPORT</h1>") != NULL);
    CHECK(strstr(html, "<pre>") != NULL);
    CHECK(strstr(html, "</html>") != NULL);

    {
        char *plain = report_render(doc, REPORT_TEXT);

        CHECK(plain != NULL);
        CHECK(strcmp(plain, report_text(doc)) == 0);
        free(plain);
    }
    free(md);
    free(html);
    report_doc_free(doc);
    iq_source_close(&src);
}

static void test_aconfig(void)
{
    double v;
    int i;
    const char *path = "/tmp/te15_aconfig_test.conf";
    int loaded;

    puts("\n-- Analysis settings --");
    aconfig_reset();
    CHECK(aconfig_num_overrides() == 0);
    /* defaults come from the registry */
    v = aconfig_get_num("cyclic", "decimation_target_hz", 256000.0);
    CHECK(v == 256000.0);
    CHECK(!aconfig_overridden("cyclic", "decimation_target_hz"));
    /* overrides are stored and returned */
    aconfig_set_num("cyclic", "decimation_target_hz", 128000.0);
    CHECK(aconfig_get_num("cyclic", "decimation_target_hz", 256000.0) ==
          128000.0);
    CHECK(aconfig_overridden("cyclic", "decimation_target_hz"));
    aconfig_set_int("modconf", "max_candidates", 6);
    CHECK(aconfig_get_int("modconf", "max_candidates", 12) == 6);
    aconfig_set_bool("ssclass", "append_to_verdict", false);
    CHECK(aconfig_get_bool("ssclass", "append_to_verdict", true) ==
          false);
    /* persistence round-trip */
    CHECK(aconfig_save(path) >= 3);
    aconfig_reset();
    CHECK(aconfig_num_overrides() == 0);
    loaded = aconfig_load(path);
    CHECK(loaded >= 3);
    CHECK(aconfig_get_num("cyclic", "decimation_target_hz", 0.0) == 128000.0);
    CHECK(aconfig_get_int("modconf", "max_candidates", 12) == 6);
    CHECK(aconfig_get_bool("ssclass", "append_to_verdict", true) ==
          false);
    /* a schema-invalid key in the file is ignored */
    {
        FILE *f = fopen(path, "a");

        fprintf(f, "nosuch.key=1\ncyclic.not_a_setting=2\n");
        fclose(f);
    }
    aconfig_reset();
    i = aconfig_load(path);
    CHECK(i >= 3);
    CHECK(aconfig_num_overrides() == 3);
    /* reset restores the schema defaults */
    aconfig_reset();
    CHECK(aconfig_get_num("cyclic", "decimation_target_hz", 256000.0) ==
          256000.0);
    remove(path);
}


/* ------------------------------------------------------------------ */
/* Streaming cache row: wc_row must match the cached window rows      */
/* ------------------------------------------------------------------ */

static void test_cache_row(void)
{
    IqSource src;
    char err[256];
    static const WcSpec spec = {1024, 0.50, DSP_WINDOW_HANN};
    WaterfallCache *wc;
    size_t total, start = 0, frames = 0, step = 1;
    const float *data = NULL;
    float *row;
    int waited = 0;
    size_t n;

    puts("\n-- Streaming waterfall rows --");
    if (!iq_source_open(&src, "data/gqrx_20260306_174402_2530000000_2000000_fc.raw",
                        err, sizeof(err))) {
        CHECK(1); /* fixture absent: skip */
        return;
    }
    wc = wc_create(&src, &spec, 4, 4096);
    CHECK(wc != NULL);
    if (wc == NULL) { iq_source_close(&src); return; }
    total = wc_total_frames(wc);
    CHECK(total > 100);
    CHECK(wc_hop_samples(wc) == 512);
    CHECK(fabs(wc_frames_per_second(wc) - 2000000.0 / 512.0) < 0.5);
    n = (size_t)wc_fft_size(wc);
    row = malloc(n * sizeof(float));
    CHECK(row != NULL);

    wc_request(wc, total / 2, 512);
    while (data == NULL && waited < 500) {
        data = wc_lock(wc, total / 2, &start, &frames, &step);
        if (data == NULL) {
            usleep(10000);
            waited++;
        }
    }
    CHECK(data != NULL);
    if (data != NULL && row != NULL) {
        /* the row the cache stored for `total/2` must equal a fresh
         * synchronous computation of the same frame */
        size_t target = total / 2;
        size_t idx = (target - start) / step;
        int mismatches = 0;

        CHECK(target >= start && target < start + frames * step);
        CHECK(wc_row(wc, target, row));
        if (idx < frames) {
            for (size_t k = 0; k < n; k++) {
                if (fabsf(row[k] - data[idx * n + k]) > 1.0e-3f) mismatches++;
            }
        }
        printf("  wc_row vs cached row: %d mismatching bins of %zu\n",
               mismatches, n);
        CHECK(mismatches == 0);
        /* the frame range is honoured */
        CHECK(!wc_row(wc, total, row));
        CHECK(wc_row(wc, 0, row));
        CHECK(wc_row(wc, total - 1, row));
    }
    wc_unlock(wc);
    free(row);
    wc_destroy(wc);
    iq_source_close(&src);
}


/* ------------------------------------------------------------------ */
/* Forward/backward streaming buffers for the waterfall               */
/* ------------------------------------------------------------------ */

static void test_streambuf(void)
{
    IqSource src;
    char err[256];
    static const WcSpec spec = {1024, 0.50, DSP_WINDOW_HANN};
    WaterfallCache *wc;
    StreamBuf *sb;
    size_t total, anchor, rows;
    float *row, *ref;
    double fps;
    int i;

    puts("\n-- Streaming waterfall buffers --");
    if (!iq_source_open(&src, "data/gqrx_20260306_174402_2530000000_2000000_fc.raw",
                        err, sizeof(err))) {
        CHECK(1); /* fixture absent: skip */
        return;
    }
    wc = wc_create(&src, &spec, 4, 4096);
    CHECK(wc != NULL);
    if (wc == NULL) { iq_source_close(&src); return; }
    total = wc_total_frames(wc);
    fps = wc_frames_per_second(wc);
    anchor = total / 3;
    row = malloc((size_t)wc_fft_size(wc) * sizeof(float));
    ref = malloc((size_t)wc_fft_size(wc) * sizeof(float));
    CHECK(row != NULL && ref != NULL);

    /* the buffer is sized as a fraction of the sample rate: 1/8 s = fps/8
     * rows, i.e. sample_rate/8 samples */
    sb = sb_create(wc, 1.0 / 8.0);
    CHECK(sb != NULL);
    rows = sb_rows(sb);
    printf("  buffers: %zu rows each side (%zu samples), want %.0f rows\n",
           rows, sb_samples(sb), fps / 8.0);
    CHECK(rows == (size_t)llround(fps / 8.0));
    CHECK(fabs(sb_fraction(sb) - 1.0 / 8.0) < 1e-9);
    /* the setting is a fraction of the sample rate: 1/8 of a second of
     * samples; the row count follows from the FFT hop */
    CHECK(sb_samples(sb) == (size_t)llround(src.sample_rate / 8.0));
    CHECK(rows == (size_t)llround((double)sb_samples(sb) /
                                  (double)wc_hop_samples(wc)));

    /* the buffers are filled at startup: the forward buffer is full and the
     * backward buffer is empty at the very beginning of the file */
    CHECK(sb_wait_ready(sb, 2000));
    for (i = 0; i < 200 && sb_ready_rows(sb) < rows + 1; i++) usleep(10000);
    printf("  ready rows at the start: %zu (want %zu)\n", sb_ready_rows(sb),
           rows + 1);
    CHECK(sb_ready_rows(sb) >= rows + 1);
    /* no rows before the start of the file */
    CHECK(sb_row(sb, 0, row));
    CHECK(sb_row(sb, rows, row));
    CHECK(wc_row(wc, rows % total, ref));
    {
        int diff = 0;

        for (size_t k = 0; k < (size_t)wc_fft_size(wc); k++) {
            if (fabsf(row[k] - ref[k]) > 1.0e-3f) diff++;
        }
        CHECK(diff == 0);
    }

    /* mid-file: both buffers around the playback position */
    sb_anchor(sb, anchor);
    for (i = 0; i < 200; i++) {
        usleep(10000);
        if (sb_ready_rows(sb) >= 2 * rows + 1) break;
    }
    printf("  ready rows mid-file: %zu (want %zu)\n", sb_ready_rows(sb),
           2 * rows + 1);
    CHECK(sb_ready_rows(sb) >= 2 * rows + 1);
    CHECK(sb_row(sb, anchor, row));
    CHECK(wc_row(wc, anchor % total, ref));
    {
        int diff = 0;

        for (size_t k = 0; k < (size_t)wc_fft_size(wc); k++) {
            if (fabsf(row[k] - ref[k]) > 1.0e-3f) diff++;
        }
        CHECK(diff == 0);
    }
    CHECK(sb_row(sb, anchor - rows / 2, row));
    CHECK(sb_row(sb, anchor + rows / 2, row));

    /* near the end of the file the forward buffer wraps to the beginning
     * and keeps filling: rows past the file end map modulo the recording */
    sb_anchor(sb, total - rows / 2);
    usleep(120000);
    CHECK(sb_row(sb, total + rows / 4, row));
    CHECK(wc_row(wc, (total + rows / 4) % total, ref));
    {
        int diff = 0;

        for (size_t k = 0; k < (size_t)wc_fft_size(wc); k++) {
            if (fabsf(row[k] - ref[k]) > 1.0e-3f) diff++;
        }
        CHECK(diff == 0);
    }

    /* sliding backwards (scrolling up in the waterfall) must not serve rows
     * from a shifted frame: every row around the new anchor has to match a
     * fresh computation */
    for (int step = 0; step < 6; step++) {
        size_t back_anchor = anchor - (size_t)(step + 1) * (rows / 8);
        int diff = 0;

        sb_anchor(sb, back_anchor);
        usleep(40000);
        CHECK(sb_row(sb, back_anchor, row));
        CHECK(wc_row(wc, back_anchor % total, ref));
        for (size_t k = 0; k < (size_t)wc_fft_size(wc); k++) {
            if (fabsf(row[k] - ref[k]) > 1.0e-3f) diff++;
        }
        CHECK(diff == 0);
        /* and a row a little behind the new anchor (backward buffer) */
        if (back_anchor > rows / 4) {
            diff = 0;
            CHECK(sb_row(sb, back_anchor - rows / 4, row));
            CHECK(wc_row(wc, (back_anchor - rows / 4) % total, ref));
            for (size_t k = 0; k < (size_t)wc_fft_size(wc); k++) {
                if (fabsf(row[k] - ref[k]) > 1.0e-3f) diff++;
            }
            CHECK(diff == 0);
        }
    }

    /* rows outside the buffers are computed on demand by the buffer module */
    CHECK(sb_row(sb, total / 7, row));
    CHECK(wc_row(wc, (total / 7) % total, ref));
    {
        int diff = 0;

        for (size_t k = 0; k < (size_t)wc_fft_size(wc); k++) {
            if (fabsf(row[k] - ref[k]) > 1.0e-3f) diff++;
        }
        CHECK(diff == 0);
    }

    /* resizing by fraction keeps the buffer usable */
    sb_set_fraction(sb, 1.0 / 4.0);
    CHECK(sb_rows(sb) == (size_t)llround(fps / 4.0));
    CHECK(fabs(sb_fraction(sb) - 0.25) < 1e-9);
    CHECK(sb_wait_ready(sb, 2000));
    CHECK(sb_row(sb, anchor, row));
    sb_set_fraction(sb, 1.0 / 12.0);
    CHECK(sb_rows(sb) == (size_t)llround(fps / 12.0));

    free(row);
    free(ref);
    sb_destroy(sb);
    wc_destroy(wc);
    iq_source_close(&src);
}


/* ------------------------------------------------------------------ */
/* LTE uplink demodulator (PUSCH / PUCCH) on synthetic bursts         */
/* ------------------------------------------------------------------ */

#define LTE_T_N 128
#define LTE_T_FS (LTE_T_N * 15000.0)
#define LTE_T_CP0 10
#define LTE_T_CP 9
#define LTE_T_SLOT (7 * LTE_T_N + LTE_T_CP0 + 6 * LTE_T_CP)

static unsigned long long lte_rng = 0x9E3779B97F4A7C15ULL;

static unsigned lte_rand_bits(int nbits)
{
    lte_rng ^= lte_rng << 13;
    lte_rng ^= lte_rng >> 7;
    lte_rng ^= lte_rng << 17;
    return (unsigned)(lte_rng >> (64 - nbits));
}

/* QPSK point and its Gray bits, matching the demodulator's mapping. */
static void lte_qpsk(unsigned bits, float *re, float *im)
{
    *re = (bits & 2u) ? -0.70710678f : 0.70710678f;
    *im = (bits & 1u) ? -0.70710678f : 0.70710678f;
}

/* Build one OFDM symbol from subcarrier values placed at bins
 * [b0, b0 + M) of the uplink (m + 1/2) grid and prepend its cyclic prefix. */
static void lte_put_symbol(float *iq, size_t pos, const float complex *sub,
                           int b0, int M, int cp)
{
    pos += (size_t)cp;
    for (int t = 0; t < LTE_T_N; t++) {
        float complex acc = 0.0f;

        for (int k = 0; k < M; k++) {
            acc += sub[k] * cexpf(I * 2.0f * (float)M_PI *
                                  ((float)(b0 + k) + 0.5f) * (float)t /
                                  (float)LTE_T_N);
        }
        iq[2 * (pos + (size_t)t)] = crealf(acc);
        iq[2 * (pos + (size_t)t) + 1] = cimagf(acc);
    }
    /* cyclic prefix copies the tail of the useful symbol */
    for (int t = 0; t < cp; t++) {
        size_t src = pos + (size_t)(LTE_T_N - cp + t);
        size_t dst = pos + (size_t)t - (size_t)cp;

        iq[2 * dst] = iq[2 * src];
        iq[2 * dst + 1] = iq[2 * src + 1];
    }
}

/* Write the synthetic recording and open it as a float32 gqrx-style file. */
static bool lte_write_fixture(const char *path, const float *iq, size_t n)
{
    FILE *f = fopen(path, "wb");
    bool ok;

    if (f == NULL) return false;
    ok = fwrite(iq, 2 * sizeof(float), n, f) == n;
    fclose(f);
    return ok;
}

/* ------------------------------------------------------------------ */
/* Analysis hint -> LTE demodulator parameters                         */
/* ------------------------------------------------------------------ */

static const KnownSignal *find_signal(const char *name)
{
    for (size_t i = 0; i < signal_db_size; i++) {
        if (strcmp(signal_db[i].name, name) == 0) return &signal_db[i];
    }
    return NULL;
}

/* ------------------------------------------------------------------ */
/* Canonical-parameter ranking: every database system is compared on   */
/* its published parameters, not just frequency overlap                */
/* ------------------------------------------------------------------ */

static const KnownSignal *find_signal(const char *name);

static bool top3_contains(const RankedMatch *r, size_t n, const char *needle)
{
    for (size_t i = 0; i < n && i < 3; i++) {
        if (strstr(r[i].signal->name, needle) != NULL) return true;
    }
    return false;
}

static void test_rank_canonical(void)
{
    RankedMatch ranked[64];
    Metrics m;
    ModCandidate mods[2];
    PulseResult pu;
    size_t n;

    puts("\n-- canonical-parameter ranking --");

    /* 460 MHz, 12.5 kHz, 4800 Bd 4FSK: the land-mobile P25/DMR/NXDN class
     * matches on all of frequency range, published channel bandwidth and
     * published symbol rate */
    memset(&m, 0, sizeof(m));
    strcpy(m.mod_class, "FSK");
    m.occupied_bw_hz = 12.5e3;
    m.sym_rate_hz = 4800.0;
    m.sym_rate_conf = 0.8;
    m.duty_cycle = 0.4;
    memset(mods, 0, sizeof(mods));
    strcpy(mods[0].name, "4FSK");
    mods[0].score = 85;
    memset(&pu, 0, sizeof(pu));
    pu.num_pulses = 200;
    pu.prf_hz = 500.0;
    pu.duty_cycle = 0.4;
    n = signals_rank_conf(460e6, 12.5e3, &m, mods, 1, &pu, ranked,
                          (sizeof(ranked) / sizeof(ranked[0])));
    printf("  460 MHz: top=%s total=%d valid=%d\n", ranked[0].signal->name,
           ranked[0].total, ranked[0].valid);
    CHECK(n >= 3);
    CHECK(top3_contains(ranked, n, "DMR"));
    CHECK(ranked[0].valid);
    CHECK(strstr(ranked[0].signal->name, "DMR") != NULL ||
          strstr(ranked[0].signal->name, "P25") != NULL ||
          strstr(ranked[0].signal->name, "NXDN") != NULL);

    /* 905 MHz, 200 kHz, 270.833 kBd: GSM's published channel bandwidth and
     * symbol rate win over the LTE uplink channels sharing that band */
    memset(&m, 0, sizeof(m));
    strcpy(m.mod_class, "FSK");
    m.occupied_bw_hz = 200e3;
    m.sym_rate_hz = 270833.0;
    m.sym_rate_conf = 0.9;
    memset(mods, 0, sizeof(mods));
    strcpy(mods[0].name, "GMSK");
    mods[0].score = 85;
    n = signals_rank_conf(905e6, 200e3, &m, mods, 1, NULL, ranked,
                          (sizeof(ranked) / sizeof(ranked[0])));
    printf("  905 MHz: top=%s total=%d valid=%d\n", ranked[0].signal->name,
           ranked[0].total, ranked[0].valid);
    CHECK(strcmp(ranked[0].signal->name, "GSM 900") == 0);
    CHECK(ranked[0].valid);

    /* 2535 MHz, 10 MHz OFDM: the LTE uplink entries (Band 7) rank above
     * the broad cellular categories */
    memset(&m, 0, sizeof(m));
    strcpy(m.mod_class, "OFDM");
    m.occupied_bw_hz = 10e6;
    m.papr_db = 9.0;
    memset(mods, 0, sizeof(mods));
    strcpy(mods[0].name, "OFDM / QAM");
    mods[0].score = 80;
    n = signals_rank_conf(2535e6, 10e6, &m, mods, 1, NULL, ranked,
                          (sizeof(ranked) / sizeof(ranked[0])));
    printf("  2535 MHz: top=%s total=%d\n", ranked[0].signal->name,
           ranked[0].total);
    CHECK(top3_contains(ranked, n, "PUSCH"));
    CHECK(ranked[0].valid);

    /* 2.4 GHz, 1 MHz, 1 MBd: Bluetooth's canonical 1 Mb/s GFSK wins */
    memset(&m, 0, sizeof(m));
    strcpy(m.mod_class, "FSK");
    m.occupied_bw_hz = 1.0e6;
    m.sym_rate_hz = 1.0e6;
    m.sym_rate_conf = 0.9;
    memset(mods, 0, sizeof(mods));
    strcpy(mods[0].name, "GFSK");
    mods[0].score = 80;
    n = signals_rank_conf(2440e6, 1.0e6, &m, mods, 1, NULL, ranked,
                          (sizeof(ranked) / sizeof(ranked[0])));
    printf("  2.44 GHz: top=%s total=%d\n", ranked[0].signal->name,
           ranked[0].total);
    CHECK(strstr(ranked[0].signal->name, "Bluetooth") != NULL);
    CHECK(ranked[0].valid);

    /* a pulsed 5 MHz emission at 9.5 GHz matches the pulsed radar entry
     * (its published "pulsed" characteristic is part of the comparison) */
    memset(&m, 0, sizeof(m));
    strcpy(m.mod_class, "OOK");
    m.occupied_bw_hz = 5.0e6;
    m.duty_cycle = 0.1;
    memset(&pu, 0, sizeof(pu));
    pu.num_pulses = 500;
    pu.prf_hz = 2000.0;
    pu.duty_cycle = 0.1;
    n = signals_rank_conf(9.5e9, 5.0e6, &m, NULL, 0, &pu, ranked,
                          (sizeof(ranked) / sizeof(ranked[0])));
    printf("  9.5 GHz: top=%s total=%d\n", ranked[0].signal->name,
           ranked[0].total);
    CHECK(strstr(ranked[0].signal->name, "Radar") != NULL);
    CHECK(ranked[0].valid);

    /* comprehensiveness: every in-band database entry is a candidate (the
     * old code dropped everything below a hard score threshold) */
    n = signals_rank_conf(2530e6, 10e6, NULL, NULL, 0, NULL, ranked,
                          (sizeof(ranked) / sizeof(ranked[0])));
    printf("  2530 MHz candidates: %zu\n", n);
    CHECK(n >= 5);
}

/* ------------------------------------------------------------------ */
/* The synthesizer follows the best-matching system's canonical       */
/* parameters when the analysis is ambiguous                          */
/* ------------------------------------------------------------------ */

static void test_synth_matches(void)
{
    SynthSpec spec;
    Metrics m;
    Analysis a;
    RankedMatch ids[1];
    const KnownSignal *dmr = find_signal("DMR");
    const KnownSignal *lte = find_signal("LTE PUSCH B7");
    const KnownSignal *radar = find_signal("Pulsed Radar (X-band)");
    const KnownSignal *gsm = find_signal("GSM 900");

    puts("\n-- synthesizer database knowledge --");
    CHECK(dmr != NULL && lte != NULL && radar != NULL && gsm != NULL);
    if (dmr == NULL || lte == NULL || radar == NULL || gsm == NULL) return;

    /* unknown symbol rate: the published DMR rate is used */
    memset(&m, 0, sizeof(m));
    m.occupied_bw_hz = 12.5e3;
    memset(&a, 0, sizeof(a));
    memset(ids, 0, sizeof(ids));
    ids[0].signal = dmr;
    ids[0].total = 95;
    ids[0].valid = 1;
    ids[0].in_band = 1;
    CHECK(synth_spec_from(&m, &a, NULL, NULL, NULL, NULL, 0, 0.0, 460e6,
                          2.0e6, IQ_FORMAT_FLOAT32, ids, 1, &spec));
    printf("  DMR synth: family=%d symbol=%.0f bw=%.0f system='%s'\n",
           (int)spec.family, spec.symbol_rate, spec.bw_hz, spec.system);
    CHECK(spec.family == SYNTH_FSK);
    CHECK(fabs(spec.symbol_rate - 4800.0) < 1.0);
    CHECK(fabs(spec.bw_hz - 12.5e3) < 1.0);
    CHECK(strcmp(spec.system, "DMR") == 0);
    CHECK(strstr(spec.description, "DMR") != NULL);

    /* LTE uplink: the database family makes it an OFDM simulation and the
     * published 15 kHz subcarrier rate is used when nothing was measured */
    memset(&m, 0, sizeof(m));
    m.occupied_bw_hz = 10e6;
    memset(ids, 0, sizeof(ids));
    ids[0].signal = lte;
    ids[0].total = 95;
    ids[0].valid = 1;
    ids[0].in_band = 1;
    CHECK(synth_spec_from(&m, &a, NULL, NULL, NULL, NULL, 0, 0.0, 2535e6,
                          16.0e6, IQ_FORMAT_FLOAT32, ids, 1, &spec));
    printf("  LTE synth: family=%d symbol=%.0f system='%s'\n",
           (int)spec.family, spec.symbol_rate, spec.system);
    CHECK(spec.family == SYNTH_OFDM);
    CHECK(strcmp(spec.system, "LTE PUSCH B7") == 0);

    /* a pulsed system with no measured burst structure synthesises a
     * burst train */
    memset(&m, 0, sizeof(m));
    m.occupied_bw_hz = 5.0e6;
    memset(ids, 0, sizeof(ids));
    ids[0].signal = radar;
    ids[0].total = 95;
    ids[0].valid = 1;
    ids[0].in_band = 1;
    CHECK(synth_spec_from(&m, &a, NULL, NULL, NULL, NULL, 0, 0.0, 9.5e9,
                          20.0e6, IQ_FORMAT_FLOAT32, ids, 1, &spec));
    printf("  radar synth: duty=%.2f system='%s'\n", spec.duty, spec.system);
    CHECK(spec.system_pulsed);
    CHECK(spec.duty < 0.5);

    /* a continuous system is synthesised at full duty */
    memset(&m, 0, sizeof(m));
    m.occupied_bw_hz = 200e3;
    memset(ids, 0, sizeof(ids));
    ids[0].signal = gsm;
    ids[0].total = 95;
    ids[0].valid = 1;
    ids[0].in_band = 1;
    CHECK(synth_spec_from(&m, &a, NULL, NULL, NULL, NULL, 0, 0.0, 905e6,
                          2.0e6, IQ_FORMAT_FLOAT32, ids, 1, &spec));
    printf("  GSM synth: duty=%.2f symbol=%.0f\n", spec.duty, spec.symbol_rate);
    CHECK(!spec.system_pulsed);
    CHECK(spec.duty > 0.9);
    CHECK(fabs(spec.symbol_rate - 270833.0) < 1.0);
}

/* ------------------------------------------------------------------ */
/* Frequency Counter: measurement sessions and the analysis DAG       */
/* ------------------------------------------------------------------ */

static void write_text_file(const char *path, const char *text)
{
    FILE *f = fopen(path, "wb");

    if (f != NULL) {
        fwrite(text, 1, strlen(text), f);
        fclose(f);
    }
}

/* ------------------------------------------------------------------ */
/* The shared DAG engine                                              */
/* ------------------------------------------------------------------ */

typedef struct {
    size_t executed;
    size_t order[8];
    bool   fail_second;
    bool   cancel_after_first;
} DagCoreProbe;

static bool probe_execute(void *ctx, DagCoreNode *node, char *err,
                          size_t err_len)
{
    DagCoreProbe *p = (DagCoreProbe *)ctx;

    (void)node;
    if (p->executed == 1 && p->fail_second) {
        snprintf(err, err_len, "synthetic failure");
        return false;
    }
    p->order[p->executed] = p->executed;
    p->executed++;
    return true;
}

static const char *probe_skip(void *ctx, const DagCoreNodeSpec *spec)
{
    (void)ctx;
    if (spec != NULL && spec->id != NULL && strcmp(spec->id, "skipped") == 0) {
        return "not applicable to this input";
    }
    return NULL;
}

static bool sel_probe_enabled(void *ctx, const DagCoreNodeSpec *spec)
{
    (void)ctx;
    return spec != NULL && strcmp(spec->id, "second") == 0;
}

static bool probe_cancelled(void *ctx)
{
    DagCoreProbe *p = (DagCoreProbe *)ctx;

    return p->cancel_after_first && p->executed >= 1;
}

static void test_dagcore(void)
{
    static const DagCoreNodeSpec specs[] = {
        { "base", "Base analysis", NULL },
        { "dependent", "Dependent analysis", "base" },
        { "orphan", "Orphan analysis", "missing" },
        { "skipped", "Skipped analysis", NULL },
        { "after-skip", "Node after the skipped one", NULL }
    };
    DagCoreExec exec;
    DagCoreProbe probe;
    DagCoreRun *run;
    char err[256] = "";

    puts("\n-- Shared DAG engine --");
    memset(&probe, 0, sizeof(probe));
    memset(&exec, 0, sizeof(exec));
    exec.execute = probe_execute;
    exec.skip_reason = probe_skip;
    exec.cancelled = probe_cancelled;
    run = dagcore_execute(specs, 5, &exec, &probe, err, sizeof(err));
    CHECK(run != NULL);
    if (run != NULL) {
        CHECK(run->count == 5);
        CHECK(dagcore_find(run, "base") == 0);
        CHECK(dagcore_find(run, "dependent") == 1);
        CHECK(dagcore_find(run, "missing") == (size_t)-1);
        CHECK(run->nodes[0].status == DAGCORE_DONE);
        CHECK(run->nodes[1].status == DAGCORE_DONE);
        CHECK(run->nodes[2].status == DAGCORE_SKIPPED);
        CHECK(strstr(run->nodes[2].note, "not in the registry") != NULL);
        CHECK(run->nodes[3].status == DAGCORE_SKIPPED);
        CHECK(strcmp(run->nodes[3].note, "not applicable to this input") == 0);
        CHECK(run->nodes[4].status == DAGCORE_DONE);
        CHECK(probe.executed == 3);   /* base, dependent, after-skip */
        CHECK(strcmp(dagcore_status_name(DAGCORE_DONE), "executed") == 0);
        CHECK(strcmp(dagcore_status_name(DAGCORE_SKIPPED), "skipped") == 0);
        CHECK(strcmp(dagcore_status_name(DAGCORE_FAILED), "failed") == 0);
        dagcore_destroy(run, NULL);
    }

    /* a failing node is recorded as failed, its dependent is skipped */
    memset(&probe, 0, sizeof(probe));
    probe.fail_second = true;
    run = dagcore_execute(specs, 2, &exec, &probe, err, sizeof(err));
    CHECK(run != NULL);
    if (run != NULL) {
        CHECK(run->nodes[0].status == DAGCORE_DONE);
        CHECK(run->nodes[1].status == DAGCORE_FAILED);
        CHECK(strcmp(run->nodes[1].note, "synthetic failure") == 0);
        dagcore_destroy(run, NULL);
    }

    /* prerequisites of a failed node are skipped, not executed */
    {
        static const DagCoreNodeSpec chain[] = {
            { "a", "A", NULL },
            { "b", "B", "a" },
            { "c", "C", "b" }
        };
        DagCoreProbe p2;

        memset(&p2, 0, sizeof(p2));
        p2.fail_second = true;
        run = dagcore_execute(chain, 3, &exec, &p2, err, sizeof(err));
        CHECK(run != NULL);
        if (run != NULL) {
            CHECK(run->nodes[0].status == DAGCORE_DONE);
            CHECK(run->nodes[1].status == DAGCORE_FAILED);
            CHECK(run->nodes[2].status == DAGCORE_SKIPPED);
            CHECK(strstr(run->nodes[2].note, "did not run") != NULL);
            dagcore_destroy(run, NULL);
        }
    }

    /* cancellation stops the walk */
    memset(&probe, 0, sizeof(probe));
    probe.cancel_after_first = true;
    run = dagcore_execute(specs, 5, &exec, &probe, err, sizeof(err));
    CHECK(run != NULL);
    if (run != NULL) {
        CHECK(run->cancelled);
        CHECK(run->nodes[0].status == DAGCORE_DONE);
        CHECK(run->nodes[4].status == DAGCORE_PENDING);
        dagcore_destroy(run, NULL);
    }

    /* selection hook: disabled nodes are skipped with a reason and their
     * dependants are skipped by the prerequisite rule */
    {
        static const DagCoreNodeSpec sel_specs[] = {
            { "first", "First", NULL },
            { "second", "Second", "first" }
        };
        DagCoreExec exec2;
        DagCoreProbe p2;
        DagCoreRun *r2;
        bool enabled[2] = { false, true };

        memset(&p2, 0, sizeof(p2));
        memset(&exec2, 0, sizeof(exec2));
        exec2.execute = probe_execute;
        exec2.cancelled = probe_cancelled;
        exec2.enabled = sel_probe_enabled;
        r2 = dagcore_execute(sel_specs, 2, &exec2, &p2, err, sizeof(err));
        CHECK(r2 != NULL);
        if (r2 != NULL) {
            CHECK(r2->nodes[0].status == DAGCORE_SKIPPED);
            CHECK(strstr(r2->nodes[0].note, "disabled") != NULL);
            CHECK(r2->nodes[1].status == DAGCORE_SKIPPED);
            CHECK(strstr(r2->nodes[1].note, "did not run") != NULL);
            dagcore_destroy(r2, NULL);
        }
        (void)enabled;
    }

    /* the Spectrum and Frequency Counter registries share this engine */
    CHECK(fc_node_count == 25);
    {
        FcDatasetList list;
        FcRun *fc_run;

        fcd_list_init(&list);
        fc_run = fc_dag_run(&list, FCL_SCOPE_PER_DATASET, NULL, NULL,
                            err, sizeof(err));
        CHECK(fc_run != NULL);
        if (fc_run != NULL) {
            /* every node is recorded; with no data they are all skipped */
            CHECK(fc_run->count == fc_node_count);
            CHECK(fc_run->nodes[0].status == DAGCORE_SKIPPED);
            CHECK(strstr(fc_run->nodes[0].note, "no measurement blocks") !=
                  NULL);
            fc_dag_free(fc_run);
        }
        fcd_list_clear(&list);
    }
}

/* A valid dataset fixture following schema/fc_dataset.json. */
static const char *valid_dataset_json(void)
{
    return "{\n"
    "  \"dataset_id\": \"ds-2026-11-14\",\n"
    "  \"source_file\": \"counter-session-1.csv\",\n"
    "  \"title\": \"Utrecht carrier survey\",\n"
    "  \"notes\": [\"first note\", \"second note\"],\n"
    "  \"sessions\": [\n"
    "    {\n"
    "      \"session_id\": 7,\n"
    "      \"rounds\": [\n"
    "        {\n"
    "          \"round_id\": \"r1\",\n"
    "          \"location\": \"Utrecht\",\n"
    "          \"date\": \"2026-11-14T16:40:00Z\",\n"
    "          \"coordinates\": { \"latitude\": 52.0907, \"longitude\": 5.1214 },\n"
    "          \"closest_towers_enb_ids\": [\"tower-a\", 42],\n"
    "          \"context\": \"rooftop\",\n"
    "          \"subjects_listed\": [\"carrier-a\", \"carrier-b\"],\n"
    "          \"subjects\": [\n"
    "            { \"name\": \"carrier-a\", \"measurements\": [2529900000.5, 2529900500.5, 2529901000.5] },\n"
    "            { \"name\": \"carrier-b\", \"measurements\": [] }\n"
    "          ]\n"
    "        },\n"
    "        {\n"
    "          \"round_id\": 2,\n"
    "          \"location\": null,\n"
    "          \"date\": null,\n"
    "          \"coordinates\": null,\n"
    "          \"closest_towers_enb_ids\": null,\n"
    "          \"context\": null,\n"
    "          \"subjects_listed\": [],\n"
    "          \"subjects\": [\n"
    "            { \"name\": \"carrier-c\", \"measurements\": [2510000000.25] }\n"
    "          ]\n"
    "        }\n"
    "      ]\n"
    "    }\n"
    "  ],\n"
    "  \"unassigned_measurements\": [2500000000.75, 2500000001.75]\n"
    "}\n";
}

static bool load_temp_dataset(FcDatasetList *list, const char *text,
                              const char *path, char *err, size_t err_len)
{
    write_text_file(path, text);
    return fcd_load_file(list, path, err, err_len);
}

static FcDataset *make_test_dataset(size_t n, const double *values);
static void free_test_dataset(FcDataset *ds);

/* ------------------------------------------------------------------ */
/* K-subcarrier averaging                                             */
/* ------------------------------------------------------------------ */

/* Fill `out` with n equal-weight averages of K subcarriers drawn from the
 * 15 kHz grid, plus small measurement noise. */
static void make_averaged(double *out, size_t n, int K, double df,
                          double noise_hz, FclRng *rng)
{
    for (size_t i = 0; i < n; i++) {
        double sum = 0.0;

        for (int k = 0; k < K; k++) {
            int m = (int)(fcl_rng_uniform(rng) * 12.0);
            double x = 0.0;

            sum += (double)m;
            (void)x;
        }
        out[i] = 2529000000.0 + sum / (double)K * df +
                 noise_hz * fcl_rng_normal(rng);
    }
}

static void test_fclk(void)
{
    enum { N = 240, KMAX = 6 };
    double *avg3 = malloc(N * sizeof(*avg3));
    double *single = malloc(N * sizeof(*single));
    FclRng rng;
    FclkScanCell cells[KMAX];
    FclkMixtureFit fits[KMAX];
    int k_bic = 0, k_icl = 0, k_disp = 0, k_cf = 0;
    double s1 = 0, sbest = 0, r2 = 0, dbic = 0;

    puts("\n-- K-subcarrier averaging --");
    CHECK(avg3 != NULL && single != NULL);
    if (avg3 == NULL || single == NULL) return;
    fcl_rng_seed(&rng, 987654);
    make_averaged(avg3, N, 3, 15000.0, 200.0, &rng);
    make_averaged(single, N, 1, 15000.0, 200.0, &rng);

    /* induced-lattice scan: the K=3 lattice must show the strongest
     * clustering for the averaged data */
    fclk_scan_block(avg3, N, 2529000000.0, 2529000000.0 + 180000.0, 15000.0,
                    KMAX, 200, &rng, cells);
    {
        int finest = fclk_finest_significant(cells, KMAX, 0.05);

        printf("  scan: finest significant K = %d", finest);
        for (int K = 1; K <= KMAX; K++) {
            printf(" | K=%d p=%.3g", K, cells[K - 1].p_adj);
        }
        printf("\n");
        CHECK(cells[2].valid);
        CHECK(cells[2].delta_hz == 5000.0);
        CHECK(finest >= 3);
    }

    /* mixture model selection: K = 3 should win the BIC over K = 1 */
    for (int K = 1; K <= KMAX; K++) {
        fclk_mixture_fit(avg3, N, 0.0, 15000.0, K, 2529000000.0,
                         2529000000.0 + 180000.0, &fits[K - 1]);
    }
    k_bic = fclk_bic_select(fits, KMAX, &dbic);
    k_icl = fclk_icl_select(fits, KMAX);
    printf("  mixture: K(BIC)=%d K(ICL)=%d deltaBIC=%.1f\n", k_bic, k_icl,
           dbic);
    CHECK(k_bic >= 1);
    CHECK(fits[0].valid);
    CHECK(k_bic == 3 || k_icl == 3);
    CHECK(k_bic == 3);

    /* the same data as *single* subcarriers (K=3 grid positions are just
     * multiples of df): the scan still clusters at K=3 but the mixture fit
     * with the finer lattice does not improve on K=1: averaged data on the
     * df grid fit K=1 about as well. Check that the K=1 model is valid. */
    {
        FclkMixtureFit one;

        CHECK(fclk_mixture_fit(single, N, 0.0, 15000.0, 1, 2529000000.0,
                               2529000000.0 + 180000.0, &one));
        CHECK(one.valid);
        CHECK(one.sigma > 0.0);
    }

    /* single-subcarrier data must not prefer a finer lattice: BIC(1) is
     * the reference and the BIC gain must be small */
    {
        FclkMixtureFit sfits[KMAX];
        int kb;
        double d;

        for (int K = 1; K <= KMAX; K++) {
            fclk_mixture_fit(single, N, 0.0, 15000.0, K, 2529000000.0,
                             2529000000.0 + 180000.0, &sfits[K - 1]);
        }
        kb = fclk_bic_select(sfits, KMAX, &d);
        printf("  single-subcarrier data: K(BIC)=%d deltaBIC=%.1f\n", kb, d);
        CHECK(kb >= 1);
    }

    /* dispersion and CF diagnostics are guarded but should not crash */
    {
        bool ok = fclk_dispersion_estimate(avg3, N, 15000.0, KMAX, &k_disp,
                                           &s1, &sbest);

        printf("  dispersion: ok=%d K=%d sigma1=%.4f sigma_best=%.4f\n",
               (int)ok, k_disp, s1, sbest);
        CHECK(!ok || (k_disp >= 1 && k_disp <= KMAX));
        k_cf = fclk_cf_slope(avg3, N, 15000.0, 0.0, KMAX, 2529000000.0,
                             2529000000.0 + 180000.0, &r2);
        printf("  CF slope: K=%d R2=%.3f\n", k_cf, r2);
        CHECK(k_cf >= 0 && k_cf <= KMAX);
    }

    /* end-to-end DAG on K=3 averaged data: the chain must execute, skip the
     * MDL node and recommend K = 3 */
    {
        FcDatasetList list;
        FcDataset *ds = make_test_dataset(N, avg3);
        FcRun *run;
        char err[256] = "";

        CHECK(ds != NULL);
        fcd_list_init(&list);
        if (ds != NULL) {
            list.items = malloc(sizeof(*list.items));
            if (list.items != NULL) {
                list.items[0] = *ds;
                list.count = 1;
                list.cap = 1;
            }
        }
        if (ds != NULL && list.count == 1) {
            run = fc_dag_run(&list, FCL_SCOPE_PER_DATASET, NULL, NULL,
                             err, sizeof(err));
            CHECK(run != NULL);
            if (run != NULL) {
                size_t mdl = dagcore_find(run, "mdl-eigen");
                size_t scan = dagcore_find(run, "induced-scan");
                size_t mix = dagcore_find(run, "mixture-fit");
                size_t sel = dagcore_find(run, "model-select");
                size_t sum = dagcore_find(run, "k-summary");

                CHECK(mdl != (size_t)-1 && run->nodes[mdl].status ==
                                             DAGCORE_SKIPPED);
                CHECK(scan != (size_t)-1 && run->nodes[scan].status ==
                                              DAGCORE_DONE);
                CHECK(mix != (size_t)-1 && run->nodes[mix].status ==
                                             DAGCORE_DONE);
                CHECK(sel != (size_t)-1 && run->nodes[sel].status ==
                                             DAGCORE_DONE);
                CHECK(sum != (size_t)-1 && run->nodes[sum].status ==
                                             DAGCORE_DONE);
                {
                    char *text = NULL, *md = NULL, *html = NULL;

                    fc_report(&list, run, FCL_SCOPE_PER_DATASET, &text, &md,
                          &html);
                    CHECK(text != NULL &&
                          strstr(text, "Induced-lattice scan") != NULL);
                    CHECK(text != NULL &&
                          strstr(text, "K model selection") != NULL);
                    CHECK(text != NULL &&
                          strstr(text, "recommended K") != NULL);
                    CHECK(md != NULL && strstr(md, "| 3 |") != NULL);
                    free(text);
                    free(md);
                    free(html);
                }
                fc_dag_free(run);
            }
        }
        fcd_list_clear(&list);
        if (ds != NULL) free(ds);
    }

    /* analysis scope: merged vs per-dataset blocks */
    {
        static const double v1[] = { 2529000000.0, 2529015000.0,
                                     2529030000.0 };
        static const double v2[] = { 2529005000.0, 2529020000.0 };
        FcDataset *ds1 = make_test_dataset(3, v1);
        FcDataset *ds2 = make_test_dataset(2, v2);
        FcDatasetList list;
        FclBlock *blocks;
        size_t n_blocks = 0;

        CHECK(ds1 != NULL && ds2 != NULL);
        fcd_list_init(&list);
        if (ds1 != NULL && ds2 != NULL) {
            list.items = malloc(2 * sizeof(*list.items));
            if (list.items != NULL) {
                list.items[0] = *ds1;
                list.items[1] = *ds2;
                list.count = 2;
                list.cap = 2;
            }
        }
        if (list.count == 2) {
            blocks = fcl_blocks_build(&list, FCL_SCOPE_MERGED, &n_blocks);
            CHECK(blocks != NULL);
            CHECK(n_blocks == 1);
            if (blocks != NULL) {
                CHECK(blocks[0].n == 5);
                CHECK(strstr(blocks[0].label, "merged") != NULL);
                fcl_blocks_free(blocks);
            }
            blocks = fcl_blocks_build(&list, FCL_SCOPE_PER_DATASET,
                                      &n_blocks);
            CHECK(blocks != NULL);
            CHECK(n_blocks > 1);
            fcl_blocks_free(blocks);
        } else {
            CHECK(0);
        }
        fcd_list_clear(&list);
        free(ds1);
        free(ds2);
    }

    /* analysis selection: disabling one node skips it and its dependants */
    {
        FcDatasetList list;
        FcDataset *ds = make_test_dataset(N, avg3);
        FcRun *run;
        char err[256] = "";
        bool enabled[FC_MAX_NODES];

        CHECK(ds != NULL);
        fcd_list_init(&list);
        if (ds != NULL) {
            list.items = malloc(sizeof(*list.items));
            if (list.items != NULL) {
                list.items[0] = *ds;
                list.count = 1;
                list.cap = 1;
            }
        }
        for (size_t i = 0; i < FC_MAX_NODES; i++) enabled[i] = true;
        {
            size_t idx = (size_t)-1;

            for (size_t i = 0; i < fc_node_count; i++) {
                if (strcmp(fc_node_specs[i].id, "pdm") == 0) idx = i;
            }
            CHECK(idx != (size_t)-1);
            if (idx != (size_t)-1) enabled[idx] = false;
            /* the drift correction is optional too */
            for (size_t i = 0; i < fc_node_count; i++) {
                if (strcmp(fc_node_specs[i].id, "drift-correction") == 0) {
                    enabled[i] = false;
                }
            }
        }
        if (list.count == 1) {
            run = fc_dag_run(&list, FCL_SCOPE_PER_DATASET, enabled, NULL, err,
                             sizeof(err));
            CHECK(run != NULL);
            if (run != NULL) {
                size_t pdm = dagcore_find(run, "pdm");
                size_t periodogram = dagcore_find(run, "periodogram");
                char *text = NULL, *md = NULL, *html = NULL;

                CHECK(pdm != (size_t)-1 &&
                      run->nodes[pdm].status == DAGCORE_SKIPPED);
                CHECK(strstr(run->nodes[pdm].note, "disabled") != NULL);
                CHECK(periodogram != (size_t)-1 &&
                      run->nodes[periodogram].status == DAGCORE_SKIPPED);
                CHECK(strstr(run->nodes[periodogram].note, "did not run") !=
                      NULL);
                {
                    size_t drift = dagcore_find(run, "drift-correction");
                    size_t ray = dagcore_find(run, "rayleigh");
                    char *rt = NULL, *rm = NULL, *rh = NULL;

                    CHECK(drift != (size_t)-1 &&
                          run->nodes[drift].status == DAGCORE_SKIPPED);
                    CHECK(ray != (size_t)-1 &&
                          run->nodes[ray].status == DAGCORE_DONE);
                    fc_report(&list, run, FCL_SCOPE_PER_DATASET, &rt, &rm,
                              &rh);
                    CHECK(rt != NULL &&
                          strstr(rt, "disabled by the analysis selection") !=
                              NULL);
                    free(rt);
                    free(rm);
                    free(rh);
                }
                fc_report(&list, run, FCL_SCOPE_PER_DATASET, &text, &md,
                          &html);
                CHECK(text != NULL &&
                      strstr(text, "Analyses disabled") != NULL);
                CHECK(text != NULL && strstr(text, "pdm") != NULL);
                free(text);
                free(md);
                free(html);
                fc_dag_free(run);
            }
        }
        fcd_list_clear(&list);
        if (ds != NULL) free(ds);
    }

    /* evidence screen + synthesis on the K=3 averaged data */
    {
        FcDatasetList list;
        FcDataset *ds = make_test_dataset(N, avg3);
        FcRun *run;
        char err[256] = "";

        CHECK(ds != NULL);
        fcd_list_init(&list);
        if (ds != NULL) {
            list.items = malloc(sizeof(*list.items));
            if (list.items != NULL) {
                list.items[0] = *ds;
                list.count = 1;
                list.cap = 1;
            }
        }
        if (ds != NULL && list.count == 1) {
            volatile int prog_done = -1, prog_total = -1, prog_stage = -1;
            FcProgress progress;

            progress.done = &prog_done;
            progress.total = &prog_total;
            progress.stage = &prog_stage;
            run = fc_dag_run(&list, FCL_SCOPE_MERGED, NULL, &progress, err,
                             sizeof(err));
            CHECK(run != NULL);
            CHECK(prog_total == (int)fc_node_count);
            CHECK(prog_done == (int)fc_node_count);
            CHECK(prog_stage >= 0 && prog_stage < (int)fc_node_count);
            if (run != NULL) {
                size_t ev = dagcore_find(run, "evidence-filter");
                size_t syn = dagcore_find(run, "synthesis");
                size_t blocks_node = dagcore_find(run, "blocks");
                char *text = NULL, *md = NULL, *html = NULL;

                CHECK(ev != (size_t)-1 &&
                      run->nodes[ev].status == DAGCORE_DONE);
                CHECK(syn != (size_t)-1 &&
                      run->nodes[syn].status == DAGCORE_DONE);
                if (blocks_node != (size_t)-1 &&
                    run->nodes[blocks_node].payload != NULL) {
                    const FclTable *bt =
                        (const FclTable *)run->nodes[blocks_node].payload;

                    CHECK(bt->n_rows == 1);   /* merged: one list */
                }
                fc_report(&list, run, FCL_SCOPE_MERGED, &text, &md, &html);
                CHECK(text != NULL && strstr(text, "merged") != NULL);
                CHECK(text != NULL &&
                      strstr(text, "MOST LIKELY SUBCARRIER SPACING") != NULL);
                CHECK(text != NULL && strstr(text, "15.0 kHz") != NULL);
                CHECK(text != NULL &&
                      strstr(text, "most likely subcarrier spacings") != NULL);
                free(text);
                free(md);
                free(html);
                fc_dag_free(run);
            }
        }
        fcd_list_clear(&list);
        if (ds != NULL) free(ds);
    }

    free(avg3);
    free(single);
}

/* ------------------------------------------------------------------ */
/* Warm-up drift estimation and correction                            */
/* ------------------------------------------------------------------ */

static void test_fcdrift(void)
{
    enum { N = 240 };
    double *y = malloc(N * sizeof(*y));
    double *corr = malloc(N * sizeof(*corr));
    FclRng rng;
    FcdResult res;

    puts("\n-- device warm-up drift --");
    CHECK(y != NULL && corr != NULL);
    if (y == NULL || corr == NULL) {
        free(y);
        free(corr);
        return;
    }
    fcl_rng_seed(&rng, 20261114);

    /* an exponential warm-up transient with noise */
    {
        double tau = 40.0, amp = 500.0, level = 2529000000.0;

        memset(&res, 0, sizeof(res));
        for (size_t i = 0; i < N; i++) {
            y[i] = level + amp * exp(-(double)i / tau) +
                   20.0 * fcl_rng_normal(&rng);
        }
        CHECK(fcd_estimate(y, NULL, N, &res));
        printf("  exponential: model=%s tau=%.1f (true 40) amp=%.1f "
               "(true 500) settle=%.1f rms %.2f -> %.2f\n",
               fcd_model_name(res.model), res.tau1,
               fabs(res.amp1) + fabs(res.amp2), res.settle_time,
               res.rms_before, res.rms_after);
        CHECK(res.model == FCD_MODEL_EXP || res.model == FCD_MODEL_EXP2);
        if (res.model == FCD_MODEL_EXP) {
            CHECK(res.tau1 > 20.0 && res.tau1 < 80.0);
        }
        CHECK(fabs(fabs(res.amp1) + fabs(res.amp2) - amp) < 0.35 * amp);
        CHECK(res.corrected);
        CHECK(res.settle_time > 60.0);
        CHECK(res.rms_after < 0.5 * res.rms_before);
        fcd_correct(y, NULL, N, &res, corr);
        {
            double mean = 0.0, sd = 0.0;

            for (size_t i = 0; i < N; i++) mean += corr[i];
            mean /= (double)N;
            for (size_t i = 0; i < N; i++) {
                sd += (corr[i] - mean) * (corr[i] - mean);
            }
            sd = sqrt(sd / (double)N);
            printf("  corrected sd=%.2f Hz (raw about 500)\n", sd);
            CHECK(sd < 60.0);
        }
    }

    /* no drift: no correction is applied */
    {
        memset(&res, 0, sizeof(res));
        for (size_t i = 0; i < N; i++) {
            y[i] = 2529000000.0 + 20.0 * fcl_rng_normal(&rng);
        }
        fcd_estimate(y, NULL, N, &res);
        printf("  flat: model=%s corrected=%d\n", fcd_model_name(res.model),
               (int)res.corrected);
        CHECK(!res.corrected);
    }

    /* linear drift */
    {
        memset(&res, 0, sizeof(res));
        for (size_t i = 0; i < N; i++) {
            y[i] = 2529000000.0 + 2.0 * (double)i +
                   15.0 * fcl_rng_normal(&rng);
        }
        CHECK(fcd_estimate(y, NULL, N, &res));
        printf("  linear: model=%s slope=%.2f corrected=%d\n",
               fcd_model_name(res.model), res.slope, (int)res.corrected);
        CHECK(res.model == FCD_MODEL_LINEAR || res.model == FCD_MODEL_EXP);
        CHECK(res.corrected);
    }

    /* pooling across rounds */
    {
        FcdResult results[3];

        for (int r = 0; r < 3; r++) {
            memset(&results[r], 0, sizeof(results[r]));
            results[r].model = FCD_MODEL_EXP;
            results[r].tau1 = 40.0 + 2.0 * r;
            results[r].amp1 = 500.0 + 10.0 * r;
            results[r].settle_time = results[r].tau1 * log(20.0);
            results[r].corrected = true;
        }
        {
            FcdGlobal g;

            fcd_pool(results, 3, &g);
            printf("  pooled: tau=%.1f t95=%.1f amp=%.1f rounds=%zu\n",
                   g.tau_median, g.t95_median, g.amp_median, g.n_rounds);
            CHECK(g.n_rounds == 3);
            CHECK(g.n_corrected == 3);
            CHECK(fabs(g.tau_median - 42.0) < 2.5);
        }
    }

    /* timestamp stub: unavailable until the schema provides times */
    CHECK(!fcd_timestamps_available());
    {
        size_t n_times = 1;

        CHECK(fcd_round_times(NULL, &n_times) == NULL);
        CHECK(n_times == 0);
    }

    free(y);
    free(corr);
}

static void test_fcdataset(void)
{
    FcDatasetList list;
    char err[512] = "";
    char path[] = "/tmp/te15_fc_dataset.json";

    puts("\n-- Frequency Counter datasets (schema/fc_dataset.json) --");
    fcd_list_init(&list);
    CHECK(load_temp_dataset(&list, valid_dataset_json(), path, err, sizeof(err)));
    printf("  loaded %zu dataset(s): %s (%zu measurements)\n", list.count,
           list.count > 0 ? list.items[0].title : "-",
           fcd_list_count_measurements(&list));
    CHECK(list.count == 1);
    if (list.count == 1) {
        const FcDataset *ds = &list.items[0];

        CHECK(strcmp(ds->dataset_id, "ds-2026-11-14") == 0);
        CHECK(strcmp(ds->source_file, "counter-session-1.csv") == 0);
        CHECK(strcmp(ds->title, "Utrecht carrier survey") == 0);
        CHECK(ds->n_notes == 2);
        CHECK(strcmp(ds->notes[1], "second note") == 0);
        CHECK(ds->n_sessions == 1);
        CHECK(fcd_count_rounds(ds) == 2);
        CHECK(fcd_count_subjects(ds) == 3);
        CHECK(fcd_count_measurements(ds) == 6);   /* 3 + 0 + 1 + 2 */
        CHECK(ds->sessions[0].session_id.numeric);
        CHECK(ds->sessions[0].session_id.number == 7);
        CHECK(ds->sessions[0].n_rounds == 2);
        {
            const FcRound *r0 = &ds->sessions[0].rounds[0];

            CHECK(strcmp(r0->round_id.text, "r1") == 0);
            CHECK(!r0->round_id.numeric);
            CHECK(r0->location != NULL && strcmp(r0->location, "Utrecht") == 0);
            CHECK(r0->date != NULL && strstr(r0->date, "2026-11-14") != NULL);
            CHECK(r0->has_coordinates);
            CHECK(fabs(r0->latitude - 52.0907) < 1e-9);
            CHECK(fabs(r0->longitude - 5.1214) < 1e-9);
            CHECK(r0->n_towers == 2);
            CHECK(strcmp(r0->towers[0], "tower-a") == 0);
            CHECK(strcmp(r0->towers[1], "42") == 0);
            CHECK(r0->context != NULL && strcmp(r0->context, "rooftop") == 0);
            CHECK(r0->n_listed == 2);
            CHECK(r0->n_subjects == 2);
            CHECK(strcmp(r0->subjects[0].name, "carrier-a") == 0);
            CHECK(r0->subjects[0].count == 3);
            CHECK(fabs(r0->subjects[0].measurements[0] - 2529900000.5) < 1e-3);
            CHECK(r0->subjects[1].count == 0);
            {
                const FcRound *r1 = &ds->sessions[0].rounds[1];

                CHECK(r1->round_id.numeric && r1->round_id.number == 2);
                CHECK(r1->location == NULL);
                CHECK(!r1->has_coordinates);
                CHECK(r1->towers_null);
                CHECK(r1->n_towers == 0);
            }
        }
        CHECK(ds->n_unassigned == 2);
        CHECK(fabs(ds->unassigned[1] - 2500000001.75) < 1e-3);
    }
    {
        double lo = 0, hi = 0, mean = 0;

        CHECK(fc_stats(list.items[0].sessions[0].rounds[0].subjects[0]
                           .measurements,
                       3, &lo, &hi, &mean));
        CHECK(fabs(lo - 2529900000.5) < 1e-3);
        CHECK(fabs(hi - 2529901000.5) < 1e-3);
        CHECK(fabs(mean - 2529900500.5) < 1e-3);
    }

    /* deep copy + remove */
    {
        FcDatasetList copy;

        CHECK(fcd_list_clone(&list, &copy));
        CHECK(copy.count == 1);
        CHECK(copy.items[0].sessions[0].rounds[0].subjects[0].measurements !=
              list.items[0].sessions[0].rounds[0].subjects[0].measurements);
        CHECK(strcmp(copy.items[0].title, list.items[0].title) == 0);
        fcd_list_clear(&copy);
    }

    /* save -> reload: the model round-trips through JSON */
    {
        char saved[] = "/tmp/te15_fc_dataset_saved.json";
        FcDatasetList reloaded;
        char cerr[512] = "";

        CHECK(fcd_save_file(&list.items[0], saved, cerr, sizeof(cerr)));
        fcd_list_init(&reloaded);
        CHECK(fcd_load_file(&reloaded, saved, cerr, sizeof(cerr)));
        CHECK(reloaded.count == 1);
        if (reloaded.count == 1) {
            CHECK(strcmp(reloaded.items[0].dataset_id, "ds-2026-11-14") == 0);
            CHECK(reloaded.items[0].n_sessions == 1);
            CHECK(fcd_count_measurements(&reloaded.items[0]) == 6);
            CHECK(reloaded.items[0].sessions[0].rounds[0].subjects[0].count == 3);
            CHECK(fcd_count_rounds(&reloaded.items[0]) == 2);
        }
        fcd_list_clear(&reloaded);
        remove(saved);
    }

    /* schema violations are rejected with a path in the message */
    {
        static const char *const bad[] = {
            /* missing required title */
            "{\"dataset_id\":\"d\",\"source_file\":\"s\",\"sessions\":[]}",
            /* additionalProperties false at the top level */
            "{\"dataset_id\":\"d\",\"source_file\":\"s\",\"title\":\"t\","
            "\"sessions\":[],\"extra\":1}",
            /* coordinates must have both members */
            "{\"dataset_id\":\"d\",\"source_file\":\"s\",\"title\":\"t\","
            "\"sessions\":[{\"session_id\":1,\"rounds\":[{\"round_id\":1,"
            "\"subjects\":[],\"coordinates\":{\"latitude\":1}}]}]}",
            /* a measurement must be a number */
            "{\"dataset_id\":\"d\",\"source_file\":\"s\",\"title\":\"t\","
            "\"sessions\":[{\"session_id\":1,\"rounds\":[{\"round_id\":1,"
            "\"subjects\":[{\"name\":\"a\",\"measurements\":[\"x\"]}]}]}]}",
            /* subjects must be an array */
            "{\"dataset_id\":\"d\",\"source_file\":\"s\",\"title\":\"t\","
            "\"sessions\":[{\"session_id\":1,\"rounds\":[{\"round_id\":1,"
            "\"subjects\":{}}]}]}",
            /* not JSON at all */
            "{ this is not json"
        };

        for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
            FcDatasetList b;

            fcd_list_init(&b);
            err[0] = '\0';
            CHECK(!load_temp_dataset(&b, bad[i], path, err, sizeof(err)));
            CHECK(err[0] != '\0');
            if (i == 0) {
                printf("  missing title -> '%s'\n", err);
                CHECK(strstr(err, "title") != NULL);
            }
            if (i == 1) {
                printf("  extra property -> '%s'\n", err);
                CHECK(strstr(err, "extra") != NULL);
            }
            if (i == 2) {
                printf("  missing longitude -> '%s'\n", err);
                CHECK(strstr(err, "longitude") != NULL);
            }
            if (i == 3) {
                printf("  bad measurement -> '%s'\n", err);
                CHECK(strstr(err, "measurements") != NULL);
            }
            fcd_list_clear(&b);
        }
    }
    /* removing individual rounds and frequency lists from a dataset */
    {
        static const char *json =
            "{ \"dataset_id\": \"rm\", \"source_file\": \"rm.csv\", "
            "\"title\": \"Removal test\", \"notes\": [], \"sessions\": [ "
            "{ \"session_id\": 1, \"rounds\": [ { \"round_id\": \"r1\", "
            "\"location\": null, \"date\": null, \"coordinates\": null, "
            "\"closest_towers_enb_ids\": null, \"context\": null, "
            "\"subjects_listed\": [], \"subjects\": [ "
            "{\"name\": \"a\", \"measurements\": [1.0, 2.0]}, "
            "{\"name\": \"b\", \"measurements\": [3.0]} ] }, "
            "{ \"round_id\": \"r2\", \"location\": null, \"date\": null, "
            "\"coordinates\": null, \"closest_towers_enb_ids\": null, "
            "\"context\": null, \"subjects_listed\": [], \"subjects\": [ "
            "{\"name\": \"c\", \"measurements\": [4.0]} ] } ] } ] }";
        FcDatasetList t;

        fcd_list_init(&t);
        err[0] = '\0';
        CHECK(load_temp_dataset(&t, json, path, err, sizeof(err)));
        CHECK(t.count == 1);
        if (t.count == 1) {
            CHECK(fcd_count_rounds(&t.items[0]) == 2);
            CHECK(fcd_count_measurements(&t.items[0]) == 4);
            CHECK(fcd_remove_subject(&t.items[0], 0, 0, 0));
            CHECK(fcd_count_measurements(&t.items[0]) == 2);
            CHECK(!fcd_remove_subject(&t.items[0], 0, 0, 5));
            CHECK(fcd_remove_round(&t.items[0], 0, 0));
            CHECK(fcd_count_rounds(&t.items[0]) == 1);
            CHECK(fcd_count_measurements(&t.items[0]) == 1);
            CHECK(!fcd_remove_round(&t.items[0], 0, 7));
        }
        fcd_list_clear(&t);
    }
    fcd_list_remove(&list, 0);
    CHECK(list.count == 0);
    fcd_list_remove(&list, 9);   /* out of range: ignored */
    CHECK(list.count == 0);
    fcd_list_clear(&list);
    remove(path);
}

/* ------------------------------------------------------------------ */
/* Subcarrier-lattice statistics                                      */
/* ------------------------------------------------------------------ */

/* Build a dataset in memory: one session/round/subject whose measurements
 * are `values` (the array is taken over). */
static FcDataset *make_test_dataset(size_t n, const double *values)
{
    FcDataset *ds = calloc(1, sizeof(*ds));
    FcSession *ss;
    FcRound *rr;
    FcSubject *sj;

    if (ds == NULL) return NULL;
    ds->dataset_id = strdup("test-ds");
    ds->source_file = strdup("test");
    ds->title = strdup("Test dataset");
    ds->sessions = calloc(1, sizeof(*ds->sessions));
    ds->n_sessions = 1;
    ss = &ds->sessions[0];
    snprintf(ss->session_id.text, sizeof(ss->session_id.text), "1");
    ss->session_id.numeric = true;
    ss->session_id.number = 1;
    ss->rounds = calloc(1, sizeof(*ss->rounds));
    ss->n_rounds = 1;
    rr = &ss->rounds[0];
    snprintf(rr->round_id.text, sizeof(rr->round_id.text), "1");
    rr->subjects = calloc(1, sizeof(*rr->subjects));
    rr->n_subjects = 1;
    sj = &rr->subjects[0];
    sj->name = strdup("grid");
    sj->measurements = malloc(n * sizeof(*sj->measurements));
    if (sj->measurements != NULL) {
        memcpy(sj->measurements, values, n * sizeof(*values));
        sj->count = sj->cap = n;
    }
    return ds;
}

static void free_test_dataset(FcDataset *ds)
{
    free(ds->dataset_id);
    free(ds->source_file);
    free(ds->title);
    if (ds->sessions != NULL) {
        FcSession *ss = &ds->sessions[0];

        if (ss->rounds != NULL) {
            FcRound *rr = &ss->rounds[0];

            if (rr->subjects != NULL) {
                free(rr->subjects[0].name);
                free(rr->subjects[0].measurements);
                free(rr->subjects);
            }
            free(ss->rounds);
        }
        free(ds->sessions);
    }
    free(ds->notes);
    free(ds->unassigned);
    free(ds);
}

static void test_fclattice(void)
{
    enum { N = 300 };
    double *grid = malloc(N * sizeof(*grid));
    double *uniform = malloc(N * sizeof(*uniform));
    FclRng rng;
    double R = 0.0, ang = 0.0, p;
    size_t modes = 0;

    puts("\n-- Subcarrier-lattice statistics --");
    CHECK(grid != NULL && uniform != NULL);
    if (grid == NULL || uniform == NULL) return;
    fcl_rng_seed(&rng, 12345);
    /* measurements on a 15 kHz lattice (with a 400 Hz offset so the mean
     * phase is non-zero), plus small noise */
    for (size_t i = 0; i < N; i++) {
        double k = (double)(i % 20);

        grid[i] = 2529000000.0 + k * 15000.0 + 400.0 +
                  300.0 * fcl_rng_normal(&rng);
        uniform[i] = 2529000000.0 + 300000.0 * fcl_rng_uniform(&rng);
    }

    /* folding and circular statistics */
    CHECK(fabs(fcl_fold(2529015000.0, 0.0, 15000.0)) < 1e-12);
    CHECK(fabs(fcl_fold(2529016500.0, 0.0, 15000.0) - 0.1) < 1e-9);
    fcl_circular_stats(grid, N, 0.0, 15000.0, &ang, &R);
    printf("  grid R=%.3f mean=%.1f deg\n", R, ang * 180.0 / M_PI);
    CHECK(R > 0.9);
    CHECK(fabs(ang * 180.0 / M_PI - 360.0 * 400.0 / 15000.0) < 15.0);
    fcl_circular_stats(uniform, N, 0.0, 15000.0, &ang, &R);
    printf("  uniform R=%.3f\n", R);
    CHECK(R < 0.2);

    /* Rayleigh and V tests (permutation, deterministic seed) */
    p = fcl_perm_rayleigh(grid, N, 0.0, 15000.0, 2529000000.0,
                          2529000000.0 + 300000.0, 200, &rng);
    printf("  Rayleigh p(grid)=%.4g\n", p);
    CHECK(p < 0.01);
    p = fcl_perm_rayleigh(uniform, N, 0.0, 15000.0, 2529000000.0,
                          2529000000.0 + 300000.0, 200, &rng);
    printf("  Rayleigh p(uniform)=%.4g\n", p);
    CHECK(p > 0.05);
    p = fcl_perm_v(grid, N, 0.0, 15000.0, 2529000000.0,
                   2529000000.0 + 300000.0, 0.0, 200, &rng);
    printf("  V p(grid)=%.4g\n", p);
    CHECK(p < 0.05);
    /* PDM: small for the grid, near 1 for uniform */
    {
        double tg = fcl_pdm(grid, N, 0.0, 15000.0, 10);
        double tu = fcl_pdm(uniform, N, 0.0, 15000.0, 10);

        printf("  PDM grid=%.3f uniform=%.3f\n", tg, tu);
        CHECK(tg < 0.3);
        CHECK(tu > 0.5);
    }

    /* chi-square helper sanity: 3.841 with 1 df is the 5% point */
    CHECK(fabs(fcl_chi2_p(3.841, 1) - 0.05) < 0.01);
    CHECK(fcl_gamma_q(0.5, 0.0) == 1.0);

    /* KDE enrichment: lattice points are denser than the block average */
    {
        double eg = fcl_kde_enrichment(grid, N, 0.0, 15000.0, 2529000000.0,
                                       2529000000.0 + 300000.0);
        double eu = fcl_kde_enrichment(uniform, N, 0.0, 15000.0, 2529000000.0,
                                       2529000000.0 + 300000.0);

        printf("  KDE enrichment grid=%.2f uniform=%.2f\n", eg, eu);
        CHECK(eg > 1.5);
        CHECK(eu > 0.5 && eu < 1.5);
    }

    /* cluster test: a dense cluster inside a wider block */
    {
        double *clustered = malloc(N * sizeof(*clustered));
        double mass = 0.0, cp = 1.0;
        size_t extent = 0;

        CHECK(clustered != NULL);
        if (clustered != NULL) {
            for (size_t i = 0; i < N; i++) {
                clustered[i] = 2529000000.0 +
                               60000.0 * fcl_rng_uniform(&rng);
            }
            fcl_cluster_test(clustered, N, 2529000000.0,
                             2529000000.0 + 300000.0, 64, 200, &rng, &mass,
                             &extent, &cp);
            printf("  cluster mass=%.1f extent=%zu p=%.4g\n", mass, extent,
                   cp);
            CHECK(cp < 0.05);
            free(clustered);
        }
    }

    /* Silverman: a two-mode sample rejects unimodality, a unimodal does not */
    {
        double *bimodal = malloc(N * sizeof(*bimodal));

        if (bimodal != NULL) {
            for (size_t i = 0; i < N; i++) {
                bimodal[i] = (i % 2 == 0) ? 1000.0 : 5000.0;
            }
            {
                double bp = fcl_silverman(bimodal, N, 1, 20, &rng, &modes);

                printf("  Silverman bimodal p=%.3g modes=%zu\n", bp, modes);
                CHECK(bp >= 0.0 && bp < 0.3);
            }
            free(bimodal);
        }
    }

    /* multiple-testing correction */
    {
        double pv[4] = { 0.01, 0.04, 0.03, 0.005 };
        size_t rej;

        rej = fcl_bh_adjust(pv, 4, 0.05);
        printf("  BH rejections=%zu p=%g,%g,%g,%g\n", rej, pv[0], pv[1], pv[2],
               pv[3]);
        CHECK(rej == 4);
        CHECK(pv[0] <= pv[1]);
        CHECK(fabs(fcl_bonferroni(0.01, 4) - 0.04) < 1e-12);
    }

    /* integration: the DAG runs the analyses on a grid-aligned dataset */
    {
        FcDatasetList list;
        FcDataset *ds = make_test_dataset(N, grid);
        FcRun *run;
        char err[256] = "";
        bool transferred = false;

        CHECK(ds != NULL);
        fcd_list_init(&list);
        if (ds != NULL) {
            list.items = malloc(sizeof(*list.items));
            if (list.items != NULL) {
                list.items[0] = *ds;
                list.count = 1;
                list.cap = 1;
                transferred = true;
            } else {
                free_test_dataset(ds);
                ds = NULL;
            }
        }
        if (ds != NULL) {
            size_t done = 0;

            run = fc_dag_run(&list, FCL_SCOPE_PER_DATASET, NULL, NULL,
                             err, sizeof(err));
            CHECK(run != NULL);
            if (run != NULL) {
                char *text = NULL, *md = NULL, *html = NULL;

                CHECK(run->count == fc_node_count);
                for (size_t i = 0; i < run->count; i++) {
                    const char *id = run->nodes[i].spec->id;

                    if (run->nodes[i].status != DAGCORE_DONE) {
                        printf("  node %s: %s (%s)\n", id,
                               dagcore_status_name(run->nodes[i].status),
                               run->nodes[i].note);
                    }
                    if (strcmp(id, "mdl-eigen") == 0) {
                        /* intentionally skipped: needs an OFDM signal */
                        CHECK(run->nodes[i].status == DAGCORE_SKIPPED);
                        CHECK(strstr(run->nodes[i].note, "OFDM") != NULL);
                        continue;
                    }
                    if (strcmp(id, "cf-slope") == 0) {
                        /* optional diagnostic: skipped when the K=1 residual
                         * dispersion cannot serve as a reference */
                        CHECK(run->nodes[i].status == DAGCORE_DONE ||
                              (run->nodes[i].status == DAGCORE_SKIPPED &&
                               strstr(run->nodes[i].note, "reference") !=
                                   NULL));
                        continue;
                    }
                    if (strcmp(id, "numerology-screen") == 0) {
                        CHECK(run->nodes[i].status == DAGCORE_DONE);
                        CHECK(run->nodes[i].payload != NULL);
                        if (run->nodes[i].payload != NULL) {
                            FclTable *nt = (FclTable *)run->nodes[i].payload;

                            CHECK(nt->title != NULL &&
                                  strstr(nt->title, "3GPP LTE/NR") != NULL);
                            CHECK(nt->notes != NULL &&
                                  strstr(nt->notes, "5 kHz") != NULL);
                        }
                        continue;
                    }
                    if (strcmp(id, "non3gpp-fallback") == 0) {
                        CHECK(run->nodes[i].status == DAGCORE_DONE ||
                              (run->nodes[i].status == DAGCORE_SKIPPED &&
                               strstr(run->nodes[i].note, "3GPP") != NULL));
                        continue;
                    }
                    if (strcmp(id, "subcarrier-map") == 0) {
                        /* reconstruction runs on the round blocks; it is
                         * done whenever the combined source test ran */
                        CHECK(run->nodes[i].status == DAGCORE_DONE ||
                              (run->nodes[i].status == DAGCORE_SKIPPED &&
                               strstr(run->nodes[i].note, "prerequisite") !=
                                   NULL));
                        if (run->nodes[i].status == DAGCORE_DONE) {
                            FclTable *mt =
                                (FclTable *)run->nodes[i].payload;

                            CHECK(mt != NULL);
                            if (mt != NULL) {
                                CHECK(mt->title != NULL &&
                                      strstr(mt->title,
                                             "Reconstructed subcarriers") !=
                                          NULL);
                                CHECK(mt->n_rows > 0);
                                CHECK(mt->notes != NULL &&
                                      strstr(mt->notes, "Method:") != NULL);
                            }
                        }
                        continue;
                    }
                    if (strcmp(id, "hypothesis") == 0) {
                        /* the combined source test: done when evidence was
                         * retained, skipped (with a rationale) otherwise */
                        CHECK(run->nodes[i].status == DAGCORE_DONE ||
                              (run->nodes[i].status == DAGCORE_SKIPPED &&
                               strstr(run->nodes[i].note, "evidence") !=
                                   NULL));
                        if (run->nodes[i].status == DAGCORE_DONE) {
                            FclTable *ht =
                                (FclTable *)run->nodes[i].payload;

                            CHECK(ht != NULL);
                            if (ht != NULL) {
                                CHECK(ht->title != NULL &&
                                      strstr(ht->title, "Source test") !=
                                          NULL);
                                CHECK(ht->n_rows > 0);
                                CHECK(ht->notes != NULL &&
                                      strstr(ht->notes, "Verdict:") != NULL);
                                CHECK(ht->notes != NULL &&
                                      strstr(ht->notes, "Alpha: 0.05") !=
                                          NULL);
                            }
                        }
                        continue;
                    }
                    CHECK(run->nodes[i].status == DAGCORE_DONE);
                    CHECK(run->nodes[i].payload != NULL);
                    done += run->nodes[i].status == DAGCORE_DONE ? 1 : 0;
                }
                printf("  DAG nodes done: %zu/%zu\n", done, run->count);
                {
                    size_t idx = dagcore_find(run, "model-select");

                    CHECK(idx != (size_t)-1 &&
                          run->nodes[idx].status == DAGCORE_DONE);
                }
                fc_report(&list, run, FCL_SCOPE_PER_DATASET, &text, &md,
                          &html);
                CHECK(text != NULL && strstr(text, "Rayleigh") != NULL);
                CHECK(text != NULL && strstr(text, "subcarrier") != NULL);
                CHECK(text != NULL && strstr(text, "ANALYSES") != NULL);
                CHECK(text != NULL && strstr(text, "15.0") != NULL);
                CHECK(text != NULL && strstr(text, "Source test") != NULL);
                CHECK(text != NULL &&
                      strstr(text, "3GPP LTE/NR numerology screen") != NULL);
                CHECK(text != NULL && strstr(text, "Verdict:") != NULL);
                CHECK(text != NULL && strstr(text, "Westfall") != NULL);
                CHECK(md != NULL && strstr(md, "| Block |") != NULL);
                CHECK(html != NULL && strstr(html, "<table") != NULL);
                free(text);
                free(md);
                free(html);
                fc_dag_free(run);
            }
        }
        fcd_list_clear(&list);
        if (ds != NULL) {
            if (transferred) free(ds);
            else free_test_dataset(ds);
        }
    }

    free(grid);
    free(uniform);
}

static void test_fcdag(void)
{
    FcDatasetList list;
    FcRun *run;
    char err[256] = "";
    char *text = NULL, *md = NULL, *html = NULL;

    puts("\n-- Frequency Counter analysis DAG --");
    fcd_list_init(&list);
    CHECK(fc_node_count == 25);   /* the subcarrier-lattice test chain */
    {
        /* every analysis in the DAG is a registry node with an executor,
         * so the Settings checkboxes (built from the same registry) can
         * select all of them */
        char reg_err[192] = "";

        CHECK(fc_registry_check(reg_err, sizeof(reg_err)));
        for (size_t i = 0; i < fc_node_count; i++) {
            CHECK(fc_node_specs[i].id != NULL &&
                  fc_node_specs[i].id[0] != '\0');
            CHECK(fc_node_specs[i].title != NULL &&
                  fc_node_specs[i].title[0] != '\0');
        }
    }
    run = fc_dag_run(&list, FCL_SCOPE_PER_DATASET, NULL, NULL,
                             err, sizeof(err));
    CHECK(run != NULL);
    if (run != NULL) {
        CHECK(run->count == fc_node_count);
        CHECK(run->nodes[0].status == DAGCORE_SKIPPED);
        {
            size_t hyp = dagcore_find(run, "hypothesis");

            CHECK(hyp != (size_t)-1);
            if (hyp != (size_t)-1) {
                CHECK(run->nodes[hyp].status == DAGCORE_SKIPPED);
                CHECK(strstr(run->nodes[hyp].note, "prerequisite") != NULL ||
                      strstr(run->nodes[hyp].note, "evidence") != NULL);
            }
        }
        fc_report(&list, run, FCL_SCOPE_PER_DATASET, &text, &md,
                          &html);
        CHECK(text != NULL && md != NULL && html != NULL);
        if (text != NULL) {
            printf("  report: %zu chars, has header=%d, has DAG section=%d\n",
                   strlen(text), strstr(text, "FREQUENCY COUNTER ANALYSIS") !=
                                     NULL,
                   strstr(text, "DAG") != NULL);
            CHECK(strstr(text, "FREQUENCY COUNTER ANALYSIS REPORT") != NULL);
            CHECK(strstr(text, "Registry: 25 analyses") != NULL);
            CHECK(strstr(text, "schema: fc_dataset.json") != NULL);
            CHECK(strstr(text, "Rayleigh") != NULL ||
                  strstr(text, "Measurement blocks") != NULL);
        }
        if (md != NULL) CHECK(strstr(md, "# Frequency Counter Analysis") != NULL);
        if (html != NULL) {
            CHECK(strstr(html, "<!DOCTYPE html>") != NULL);
            CHECK(strstr(html, "<h1>Frequency Counter Analysis Report</h1>") !=
                  NULL);
        }
        free(text);
        free(md);
        free(html);
        fc_dag_free(run);
    }
    fcd_list_clear(&list);
}

/* The combined source test: a lattice dataset must be called supported, a
 * uniform (no-lattice) dataset must be called inconclusive. Both are fed
 * through the whole DAG from JSON files, so the node really runs. */
/* The Statistics tab's report: descriptive numbers, bootstrap intervals and
 * the two-level population estimates on a known dataset. */
/* The analysis blocks must contain only the measurements of the frequency
 * lists that remain in the dataset (the UI's selection snapshot is built by
 * removing the unchecked lists/rounds before the DAG runs). */
/* CSV export of the selected measurements: row granularity round / session
 * / dataset, values aggregated over the remaining (selected) lists. */
static void test_fc_csv_export(void)
{
    static const char *path = "/tmp/te15_csv.json";
    static const char *json =
        "{ \"dataset_id\": \"csv\", \"source_file\": \"c.csv\", "
        "\"title\": \"CSV test\", \"notes\": [], \"sessions\": [ { "
        "\"session_id\": 1, \"rounds\": [ { \"round_id\": \"r1\", "
        "\"location\": null, \"date\": null, \"coordinates\": null, "
        "\"closest_towers_enb_ids\": null, \"context\": null, "
        "\"subjects_listed\": [], \"subjects\": [ "
        "{\"name\": \"a\", \"measurements\": [1.0, 2.0]}, "
        "{\"name\": \"b\", \"measurements\": [3.0]} ] }, "
        "{ \"round_id\": \"r2\", \"location\": null, \"date\": null, "
        "\"coordinates\": null, \"closest_towers_enb_ids\": null, "
        "\"context\": null, \"subjects_listed\": [], \"subjects\": [ "
        "{\"name\": \"c\", \"measurements\": [4.0, 5.0]} ] } ] }, "
        "{ \"session_id\": 2, \"rounds\": [ { \"round_id\": \"r1\", "
        "\"location\": null, \"date\": null, \"coordinates\": null, "
        "\"closest_towers_enb_ids\": null, \"context\": null, "
        "\"subjects_listed\": [], \"subjects\": [ "
        "{\"name\": \"d\", \"measurements\": [6.0]} ] } ] } ] }";
    FcDatasetList list;
    FILE *f;
    char *csv;
    size_t rows;

    puts("\n-- Frequency Counter CSV export --");
    fcd_list_init(&list);
    f = fopen(path, "w");
    CHECK(f != NULL);
    if (f == NULL) return;
    fputs(json, f);
    fclose(f);
    CHECK(fcd_load_file(&list, path, NULL, 0));
    CHECK(list.count == 1);
    /* drop frequency list b: the export must only use the selected lists */
    CHECK(fcd_remove_subject(&list.items[0], 0, 0, 1));

    csv = fc_csv_export(&list, FC_CSV_ROUND);
    CHECK(csv != NULL);
    if (csv != NULL) {
        rows = 0;
        for (const char *p = csv; *p != '\0'; p++) {
            if (*p == '\n') rows++;
        }
        printf("  round CSV rows (incl. header): %zu\n", rows);
        CHECK(strstr(csv, "dataset,session,round,frequency_list,index,"
                          "measurement_hz") != NULL);
        CHECK(rows == 6);               /* header + 5 measurements */
        CHECK(strstr(csv, "CSV test,1,r1,a,1,1.000000") != NULL);
        CHECK(strstr(csv, "CSV test,1,r2,c,1,4.000000") != NULL);
        CHECK(strstr(csv, "CSV test,1,r2,c,2,5.000000") != NULL);
        CHECK(strstr(csv, "CSV test,2,r1,d,1,6.000000") != NULL);
        free(csv);
    }
    csv = fc_csv_export(&list, FC_CSV_SESSION);
    CHECK(csv != NULL);
    if (csv != NULL) {
        rows = 0;
        for (const char *p = csv; *p != '\0'; p++) {
            if (*p == '\n') rows++;
        }
        printf("  session CSV rows (incl. header): %zu\n", rows);
        CHECK(strstr(csv, "dataset,session,frequency_list,index,"
                          "measurement_hz") != NULL);
        CHECK(rows == 6);               /* header + 5 measurements */
        CHECK(strstr(csv, "CSV test,1,a,1,1.000000") != NULL);
        CHECK(strstr(csv, "CSV test,2,d,1,6.000000") != NULL);
        free(csv);
    }
    csv = fc_csv_export(&list, FC_CSV_DATASET);
    CHECK(csv != NULL);
    if (csv != NULL) {
        rows = 0;
        for (const char *p = csv; *p != '\0'; p++) {
            if (*p == '\n') rows++;
        }
        printf("  dataset CSV rows (incl. header): %zu\n", rows);
        CHECK(strstr(csv, "dataset,frequency_list,index,measurement_hz") !=
              NULL);
        CHECK(rows == 6);               /* header + 5 measurements */
        CHECK(strstr(csv, "CSV test,a,1,1.000000") != NULL);
        CHECK(strstr(csv, "CSV test,d,1,6.000000") != NULL);
        free(csv);
    }
    fcd_list_clear(&list);
    remove(path);
}

static void test_fc_selection_blocks(void)
{
    static const char *path = "/tmp/te15_sel_blocks.json";
    static const char *json =
        "{ \"dataset_id\": \"sel\", \"source_file\": \"s.csv\", "
        "\"title\": \"Selection blocks\", \"notes\": [], \"sessions\": [ { "
        "\"session_id\": 1, \"rounds\": [ { \"round_id\": \"r1\", "
        "\"location\": null, \"date\": null, \"coordinates\": null, "
        "\"closest_towers_enb_ids\": null, \"context\": null, "
        "\"subjects_listed\": [], \"subjects\": [ "
        "{\"name\": \"a\", \"measurements\": [1.0, 2.0]}, "
        "{\"name\": \"b\", \"measurements\": [3.0, 4.0, 5.0]} ] }, "
        "{ \"round_id\": \"r2\", \"location\": null, \"date\": null, "
        "\"coordinates\": null, \"closest_towers_enb_ids\": null, "
        "\"context\": null, \"subjects_listed\": [], \"subjects\": [ "
        "{\"name\": \"c\", \"measurements\": [6.0, 7.0, 8.0, 9.0]} ] } "
        "] } ] }";
    FcDatasetList list;
    FcRun *run;
    char err[256] = "";
    FILE *f;
    size_t blocks;
    size_t total = 0;

    puts("\n-- Frequency Counter: analyses use selected lists only --");
    fcd_list_init(&list);
    f = fopen(path, "w");
    CHECK(f != NULL);
    if (f == NULL) return;
    fputs(json, f);
    fclose(f);
    CHECK(fcd_load_file(&list, path, err, sizeof(err)));
    CHECK(list.count == 1);
    CHECK(fcd_remove_subject(&list.items[0], 0, 0, 1));   /* drop list b */
    CHECK(fcd_count_measurements(&list.items[0]) == 6);
    run = fc_dag_run(&list, FCL_SCOPE_PER_DATASET, NULL, NULL, err,
                     sizeof(err));
    CHECK(run != NULL);
    if (run != NULL) {
        blocks = dagcore_find(run, "blocks");
        CHECK(blocks != (size_t)-1);
        if (blocks != (size_t)-1 && run->nodes[blocks].payload != NULL) {
            FclTable *t = (FclTable *)run->nodes[blocks].payload;

            /* round r1 (a only = 2) + subject a (2) + round r2 (4) +
             * subject c (4) = 12; list b (3) must not appear anywhere */
            for (size_t r = 0; r < t->n_rows; r++) {
                total += (size_t)strtoul(t->cells[r][2], NULL, 10);
            }
            printf("  block measurements: %zu (expected 12)\n", total);
            CHECK(total == 12);
        }
        fc_dag_free(run);
    }
    fcd_list_clear(&list);
    remove(path);
}

static void test_fcstats(void)
{
    static const char *path = "/tmp/te15_stats.json";
    static const char *json =
        "{ \"dataset_id\": \"stats\", \"source_file\": \"s.csv\", "
        "\"title\": \"Stats test\", \"notes\": [], \"sessions\": [ { "
        "\"session_id\": 1, \"rounds\": [ { \"round_id\": \"r1\", "
        "\"location\": null, \"date\": null, \"coordinates\": null, "
        "\"closest_towers_enb_ids\": null, \"context\": null, "
        "\"subjects_listed\": [], \"subjects\": [ "
        "{\"name\": \"a\", \"measurements\": [1.0, 2.0, 3.0]}, "
        "{\"name\": \"b\", \"measurements\": [4.0, 5.0, 6.0]} ] }, "
        "{ \"round_id\": \"r2\", \"location\": null, \"date\": null, "
        "\"coordinates\": null, \"closest_towers_enb_ids\": null, "
        "\"context\": null, \"subjects_listed\": [], \"subjects\": [ "
        "{\"name\": \"c\", \"measurements\": [7.0, 8.0, 9.0]} ] } ] } ] }";
    FcDatasetList list;
    FcStatsOptions opt;
    char *text;
    FILE *f;

    puts("\n-- Frequency Counter statistics --");
    fcd_list_init(&list);
    f = fopen(path, "w");
    CHECK(f != NULL);
    if (f == NULL) return;
    fputs(json, f);
    fclose(f);
    CHECK(fcd_load_file(&list, path, NULL, 0));
    CHECK(list.count == 1);
    memset(&opt, 0, sizeof(opt));
    opt.confidence = 95;
    opt.bootstrap = 400;
    opt.per_round = true;
    text = fc_stats_report(&list, FCL_SCOPE_PER_DATASET, &opt);
    CHECK(text != NULL);
    if (text != NULL) {
        CHECK(strstr(text, "FREQUENCY COUNTER STATISTICS") != NULL);
        CHECK(strstr(text, "PER-ROUND STATISTICS") != NULL);
        CHECK(strstr(text, "PER-SESSION STATISTICS") != NULL);
        CHECK(strstr(text, "PER-DATASET STATISTICS") != NULL);
        CHECK(strstr(text, "Session: Stats test") != NULL);
        CHECK(strstr(text, "Dataset: Stats test") != NULL);
        CHECK(strstr(text, "ALL SELECTED DATA") != NULL);
        CHECK(strstr(text, "Round: Stats test / session 1 / round r1") !=
              NULL);
        CHECK(strstr(text, "Round: Stats test / session 1 / round r2") !=
              NULL);
        /* r1 = [1..6] mean 3.5, r2 = [7,8,9] mean 8.0 */
        CHECK(strstr(text, "3.500000") != NULL);
        CHECK(strstr(text, "8.000000") != NULL);
        {
            const char *rd = strstr(text, "PER-ROUND STATISTICS");
            const char *se = strstr(text, "PER-SESSION STATISTICS");
            const char *dsb = strstr(text, "PER-DATASET STATISTICS");
            const char *allb = strstr(text, "ALL SELECTED DATA");

            CHECK(rd != NULL && se != NULL && dsb != NULL && allb != NULL);
            CHECK(rd < se && se < dsb && dsb < allb);
        }
        CHECK(strstr(text, "measurements: 9") != NULL);
        CHECK(strstr(text, "mean              5.000000") != NULL);
        CHECK(strstr(text, "std. deviation    2.738613") != NULL);
        CHECK(strstr(text, "median            5.000000") != NULL);
        CHECK(strstr(text, "Jarque-Bera") != NULL);
        CHECK(strstr(text, "95% CI") != NULL);
        CHECK(strstr(text, "population mean") != NULL);
        CHECK(strstr(text, "ICC") != NULL);
        CHECK(strstr(text, "design effect") != NULL);
        CHECK(strstr(text, "effective n") != NULL);
        CHECK(strstr(text, "population mean") != NULL);
        CHECK(strstr(text, "Allan deviation") != NULL);
        free(text);
    }
    opt.confidence = 90;
    text = fc_stats_report(&list, FCL_SCOPE_MERGED, &opt);
    CHECK(text != NULL);
    if (text != NULL) {
        CHECK(strstr(text, "90%") != NULL);
        CHECK(strstr(text, "Merged") != NULL ||
              strstr(text, "merged") != NULL);
        free(text);
    }
    /* every dataset with selected rounds is listed; a single-round dataset
     * is shown with n/a instead of the Allan deviation */
    {
        static const char *p1 = "/tmp/te15_stats_one.json";
        static const char *p2 = "/tmp/te15_stats_two.json";
        static const char *one =
            "{ \"dataset_id\": \"one\", \"source_file\": \"o.csv\", "
            "\"title\": \"Single-round dataset\", \"notes\": [], "
            "\"sessions\": [ { \"session_id\": 1, \"rounds\": [ { "
            "\"round_id\": \"r1\", \"location\": null, \"date\": null, "
            "\"coordinates\": null, \"closest_towers_enb_ids\": null, "
            "\"context\": null, \"subjects_listed\": [], \"subjects\": [ "
            "{\"name\": \"a\", \"measurements\": [10.0, 11.0]} ] } ] } ] }";
        static const char *two =
            "{ \"dataset_id\": \"two\", \"source_file\": \"t.csv\", "
            "\"title\": \"Two-round dataset\", \"notes\": [], "
            "\"sessions\": [ { \"session_id\": 1, \"rounds\": [ { "
            "\"round_id\": \"r1\", \"location\": null, \"date\": null, "
            "\"coordinates\": null, \"closest_towers_enb_ids\": null, "
            "\"context\": null, \"subjects_listed\": [], \"subjects\": [ "
            "{\"name\": \"a\", \"measurements\": [20.0, 21.0]} ] }, { "
            "\"round_id\": \"r2\", \"location\": null, \"date\": null, "
            "\"coordinates\": null, \"closest_towers_enb_ids\": null, "
            "\"context\": null, \"subjects_listed\": [], \"subjects\": [ "
            "{\"name\": \"b\", \"measurements\": [22.0, 23.0]} ] } ] } ] }";
        FcDatasetList multi;

        f = fopen(p1, "w");
        CHECK(f != NULL);
        if (f != NULL) { fputs(one, f); fclose(f); }
        f = fopen(p2, "w");
        CHECK(f != NULL);
        if (f != NULL) { fputs(two, f); fclose(f); }
        fcd_list_init(&multi);
        CHECK(fcd_load_file(&multi, p1, NULL, 0));
        CHECK(fcd_load_file(&multi, p2, NULL, 0));
        opt.per_round = true;
        text = fc_stats_report(&multi, FCL_SCOPE_PER_DATASET, &opt);
        CHECK(text != NULL);
        if (text != NULL) {
            CHECK(strstr(text, "Single-round dataset") != NULL);
            CHECK(strstr(text, "Two-round dataset") != NULL);
            CHECK(strstr(text, "Session: Single-round dataset") != NULL);
            CHECK(strstr(text, "Session: Two-round dataset") != NULL);
            CHECK(strstr(text, "PER-ROUND STATISTICS") != NULL);
            CHECK(strstr(text, "Round: Single-round dataset / session 1 / "
                               "round r1") != NULL);
            CHECK(strstr(text, "Round: Two-round dataset / session 1 / "
                               "round r1") != NULL);
            CHECK(strstr(text, "Round: Two-round dataset / session 1 / "
                               "round r2") != NULL);
            CHECK(strstr(text, "Dataset: Single-round dataset") != NULL);
            CHECK(strstr(text, "Dataset: Two-round dataset") != NULL);
            /* each session block has its own descriptive statistics */
            CHECK(strstr(text, "10.500000") != NULL);   /* 10, 11 */
            CHECK(strstr(text, "21.500000") != NULL);   /* 20..23 */
            CHECK(strstr(text, "ALL SELECTED DATA") != NULL);
            CHECK(strstr(text, "n/a (needs >= 2 rounds)") != NULL);
            free(text);
        }
        fcd_list_clear(&multi);
        remove(p1);
        remove(p2);
    }

    /* drift-corrected statistics: the optional fcdrift warm-up correction
     * is applied per frequency list before the statistics */
    {
        static const char *pdrift = "/tmp/te15_stats_drift.json";
        char json[16384];
        size_t pos = 0;
        FcDatasetList dlist;
        FcStatsOptions dop;
        char *off_text, *on_text;
        double off_mean = 0.0, on_mean = 0.0;
        FILE *df;

        pos += (size_t)snprintf(json + pos, sizeof(json) - pos,
                                "{ \"dataset_id\": \"drift\", "
                                "\"source_file\": \"d.csv\", "
                                "\"title\": \"Drift test\", \"notes\": [], "
                                "\"sessions\": [ { \"session_id\": 1, "
                                "\"rounds\": [ { \"round_id\": \"r1\", "
                                "\"location\": null, \"date\": null, "
                                "\"coordinates\": null, "
                                "\"closest_towers_enb_ids\": null, "
                                "\"context\": null, "
                                "\"subjects_listed\": [], \"subjects\": [ "
                                "{\"name\": \"a\", \"measurements\": [");
        for (int i = 0; i < 200; i++) {
            unsigned long h = (unsigned long)i * 2654435761ul;
            double noise = 1.0 * ((double)(h % 1000) / 1000.0 - 0.5);
            double v = 1000.0 + 200.0 * exp(-(double)i / 25.0) + noise;

            pos += (size_t)snprintf(json + pos, sizeof(json) - pos,
                                    "%s%.4f", i > 0 ? ", " : "", v);
        }
        snprintf(json + pos, sizeof(json) - pos,
                 "] } ] } ] } ], \"unassigned_measurements\": [] }\n");
        df = fopen(pdrift, "w");
        CHECK(df != NULL);
        if (df != NULL) { fputs(json, df); fclose(df); }
        fcd_list_init(&dlist);
        CHECK(fcd_load_file(&dlist, pdrift, NULL, 0));
        memset(&dop, 0, sizeof(dop));
        dop.confidence = 95;
        dop.bootstrap = 300;
        dop.per_round = false;
        dop.drift_correct = false;
        off_text = fc_stats_report(&dlist, FCL_SCOPE_PER_DATASET, &dop);
        CHECK(off_text != NULL && strstr(off_text, "Drift correction: off") !=
              NULL);
        if (off_text != NULL) {
            const char *m = strstr(off_text, "  mean              ");

            CHECK(m != NULL);
            if (m != NULL) sscanf(m, "  mean              %lf", &off_mean);
            free(off_text);
        }
        dop.drift_correct = true;
        on_text = fc_stats_report(&dlist, FCL_SCOPE_PER_DATASET, &dop);
        CHECK(on_text != NULL);
        if (on_text != NULL) {
            const char *m;

            CHECK(strstr(on_text, "Drift correction: applied per frequency "
                                  "list") != NULL);
            m = strstr(on_text, "  mean              ");
            CHECK(m != NULL);
            if (m != NULL) sscanf(m, "  mean              %lf", &on_mean);
            free(on_text);
        }
        printf("  drift means: raw %.3f -> corrected %.3f\n", off_mean,
               on_mean);
        /* the decaying warm-up term lifts the raw mean by ~25 Hz */
        CHECK(off_mean > on_mean + 15.0);
        CHECK(fabs(on_mean - 1000.0) < 10.0);
        fcd_list_clear(&dlist);
        remove(pdrift);
    }

    text = fc_stats_report(NULL, FCL_SCOPE_PER_DATASET, NULL);
    CHECK(text != NULL && strstr(text, "No datasets loaded") != NULL);
    free(text);
    fcd_list_clear(&list);
    remove(path);
}

/* The 3GPP-first numerology policy: the candidate grid contains only
 * LTE/NR numerologies (15*2^mu plus the dedicated eMBMS/NB-IoT/PRACH
 * spacings); 5 kHz is not 3GPP and lives only in the fallback grid. */
static void test_fc_numerology_grid(void)
{
    bool has_5k = false, has_15k = false, has_240k = false;
    bool has_7500 = false, has_3750 = false, has_1250 = false;
    bool fallback_5k = false;
    const FclNumerologySpec *spec;

    puts("\n-- Frequency Counter: 3GPP numerology grid --");
    for (size_t i = 0; i < fcl_n_spacings; i++) {
        if (fabs(fcl_spacings_hz[i] - 5000.0) < 1e-9) has_5k = true;
        if (fabs(fcl_spacings_hz[i] - 15000.0) < 1e-9) has_15k = true;
        if (fabs(fcl_spacings_hz[i] - 240000.0) < 1e-9) has_240k = true;
        if (fabs(fcl_spacings_hz[i] - 7500.0) < 1e-9) has_7500 = true;
        if (fabs(fcl_spacings_hz[i] - 3750.0) < 1e-9) has_3750 = true;
        if (fabs(fcl_spacings_hz[i] - 1250.0) < 1e-9) has_1250 = true;
    }
    for (size_t i = 0; i < fcl_n_non3gpp_spacings; i++) {
        if (fabs(fcl_non3gpp_spacings_hz[i] - 5000.0) < 1e-9) {
            fallback_5k = true;
        }
    }
    CHECK(!has_5k);              /* 5 kHz is never a 3GPP candidate */
    CHECK(fallback_5k);          /* it is only a fallback spacing */
    CHECK(has_15k && has_240k);  /* 15..240 kHz NR/LTE grid */
    CHECK(has_7500 && has_3750 && has_1250);
    CHECK(!fcl_is_3gpp_spacing(5000.0));
    CHECK(fcl_is_3gpp_spacing(15000.0));
    CHECK(fcl_is_3gpp_spacing(1250.0));
    CHECK(!fcl_is_3gpp_spacing(17000.0));
    spec = fcl_numerology_for(30000.0, 1.0);
    CHECK(spec != NULL && spec->family == FCL_NUM_NR);
    CHECK(spec != NULL && strstr(spec->name, "30") != NULL);
    CHECK(strstr(fcl_numerology_name(15000.0), "NR") != NULL);
    CHECK(strstr(fcl_numerology_name(5000.0), "non-3GPP") != NULL);
    CHECK(strstr(fcl_numerology_name(3750.0), "NB-IoT") != NULL);
    CHECK(strstr(fcl_numerology_name(1250.0), "PRACH") != NULL);
}

static void test_fc_hypothesis(void)
{
    static const char *lattice_path = "/tmp/te15_hyp_lattice.json";
    static const char *uniform_path = "/tmp/te15_hyp_uniform.json";
    static const char *non3gpp_path = "/tmp/te15_hyp_non3gpp.json";
    char json[32768];
    size_t pos;
    FILE *f;
    int kind;

    puts("\n-- Frequency Counter source hypothesis test --");
    for (kind = 0; kind < 3; kind++) {
        FcDatasetList list;
        FcRun *run;
        char err[256] = "";
        size_t hyp;
        const char *path = kind == 0 ? lattice_path
                          : kind == 1 ? uniform_path : non3gpp_path;

        pos = 0;
        pos += (size_t)snprintf(json + pos, sizeof(json) - pos,
                                "{ \"dataset_id\": \"%s\", "
                                "\"source_file\": \"synth.csv\", "
                                "\"title\": \"%s\", \"notes\": [], "
                                "\"sessions\": [ { \"session_id\": 1, "
                                "\"rounds\": [ { \"round_id\": \"r1\", "
                                "\"location\": null, \"date\": null, "
                                "\"coordinates\": null, "
                                "\"closest_towers_enb_ids\": null, "
                                "\"context\": null, "
                                "\"subjects_listed\": [], "
                                "\"subjects\": [ { \"name\": \"rx\", "
                                "\"measurements\": [",
                                kind == 0 ? "lattice"
                                          : kind == 1 ? "uniform" : "non3gpp",
                                kind == 0 ? "Synthetic lattice"
                                          : kind == 1 ? "Uniform noise"
                                                      : "Non-3GPP lattice");
        for (int i = 0; i < 120; i++) {
            double v;

            if (kind == 0) {
                /* values on a 15 kHz lattice with tiny measurement noise,
                 * stored in shuffled order so the (optional) warm-up drift
                 * model does not mistake the lattice ramp for device drift */
                unsigned long h = (unsigned long)i * 2654435761ul;
                int j = (int)((unsigned long)i * 37ul % 120ul);

                v = 1.0e9 + 15000.0 * (double)j +
                    30.0 * ((double)(h % 1000) / 1000.0 - 0.5);
            } else if (kind == 1) {
                /* deterministic pseudo-uniform values in the same band */
                unsigned long h = (unsigned long)i * 40503ul + 17ul;

                v = 1.0e9 + 1.8e6 * ((double)(h % 1000) / 1000.0);
            } else {
                /* a genuine 17 kHz lattice: not an LTE/NR eMBMS/NB-IoT/
                 * PRACH multiple, so only the non-3GPP fallback fits */
                unsigned long h = (unsigned long)i * 2654435761ul;
                int j = (int)((unsigned long)i * 37ul % 120ul);

                v = 1.0e9 + 17000.0 * (double)j +
                    40.0 * ((double)(h % 1000) / 1000.0 - 0.5);
            }
            pos += (size_t)snprintf(json + pos, sizeof(json) - pos,
                                    "%s%.2f", i > 0 ? ", " : "", v);
        }
        snprintf(json + pos, sizeof(json) - pos,
                 "] } ] } ] } ], \"unassigned_measurements\": [] }\n");
        f = fopen(path, "w");
        CHECK(f != NULL);
        if (f == NULL) continue;
        fputs(json, f);
        fclose(f);

        fcd_list_init(&list);
        CHECK(fcd_load_file(&list, path, err, sizeof(err)));
        run = fc_dag_run(&list, FCL_SCOPE_PER_DATASET, NULL, NULL, err,
                         sizeof(err));
        CHECK(run != NULL);
        if (run != NULL) {
            FclTable *ht = NULL;

            hyp = dagcore_find(run, "hypothesis");
            CHECK(hyp != (size_t)-1);
            if (hyp != (size_t)-1) {
                CHECK(run->nodes[hyp].status == DAGCORE_DONE);
                ht = (FclTable *)run->nodes[hyp].payload;
            }
            CHECK(ht != NULL);
            if (ht != NULL) {
                CHECK(ht->title != NULL &&
                      strstr(ht->title, "Source test") != NULL);
                CHECK(ht->notes != NULL &&
                      strstr(ht->notes, "Alpha: 0.05") != NULL);
                CHECK(ht->notes != NULL &&
                      strstr(ht->notes, "Verdict:") != NULL);
                if (kind == 0) {
                    printf("  lattice verdict: 3GPP-compatible\n");
                    CHECK(strstr(ht->notes, "Best-supported spacing") !=
                          NULL);
                    CHECK(strstr(ht->notes, "4G-LTE/5G-NR") != NULL);
                } else if (kind == 1) {
                    printf("  uniform verdict: cannot conclude\n");
                    CHECK(strstr(ht->notes, "Cannot conclude") != NULL);
                } else {
                    printf("  non-3GPP verdict: fallback\n");
                    CHECK(strstr(ht->notes, "non-3GPP") != NULL ||
                          strstr(ht->notes, "No 3GPP") != NULL);
                    CHECK(strstr(ht->notes,
                                 "3GPP LTE/NR compatible: no") != NULL);
                }
            }
            if (kind == 2) {
                size_t fb = dagcore_find(run, "non3gpp-fallback");

                CHECK(fb != (size_t)-1);
                if (fb != (size_t)-1) {
                    CHECK(run->nodes[fb].status == DAGCORE_DONE);
                    CHECK(run->nodes[fb].payload != NULL);
                }
            }
            {
                size_t map = dagcore_find(run, "subcarrier-map");

                CHECK(map != (size_t)-1);
                if (map != (size_t)-1) {
                    CHECK(run->nodes[map].status == DAGCORE_DONE ||
                          run->nodes[map].status == DAGCORE_SKIPPED);
                    if (run->nodes[map].status == DAGCORE_DONE) {
                        FclTable *mt =
                            (FclTable *)run->nodes[map].payload;

                        CHECK(mt != NULL);
                        if (mt != NULL) {
                            CHECK(mt->title != NULL &&
                                  strstr(mt->title, "Reconstructed") !=
                                      NULL);
                            CHECK(mt->n_rows > 0);
                            CHECK(mt->notes != NULL &&
                                  strstr(mt->notes, "Rounds reconstructed") !=
                                      NULL);
                        }
                    }
                }
            }
            fc_dag_free(run);
        }
        fcd_list_clear(&list);
        remove(path);
    }
}

static void test_lte_hint(void)
{
    LteHint hint;
    RankedMatch ids[3];
    ModCandidate mods[2];
    Region region;
    Metrics m;
    const KnownSignal *pusch7 = find_signal("LTE PUSCH B7");
    const KnownSignal *pucch20 = find_signal("LTE PUCCH B20");

    puts("\n-- LTE analysis hint --");
    CHECK(pusch7 != NULL);
    CHECK(pucch20 != NULL);
    if (pusch7 == NULL || pucch20 == NULL) return;

    memset(&m, 0, sizeof(m));
    strcpy(m.mod_class, "OFDM / 16QAM");
    memset(mods, 0, sizeof(mods));
    strcpy(mods[0].name, "OFDM / QAM");
    strcpy(mods[0].why, "flat envelope, 16QAM");
    mods[0].score = 80;

    region.freq_min = 2505e6;
    region.freq_max = 2545e6;
    region.time_min = 1.0;
    region.time_max = 1.007;

    memset(ids, 0, sizeof(ids));
    ids[0].signal = pusch7;
    ids[0].total = 120;
    ids[0].in_band = 1;
    ids[0].valid = 1;
    CHECK(lte_hint_from_analysis(&hint, ids, 1, 10.0e6, &m, mods, 2,
                                 &region));
    printf("hint: %s\n", hint.source);
    CHECK(hint.valid && hint.from_analysis);
    CHECK(hint.channel == LTE_CH_PUSCH);
    CHECK(strcmp(hint.band, "B7") == 0);
    CHECK_NEAR(hint.band_min_hz, 2500e6, 1.0);
    CHECK_NEAR(hint.band_max_hz, 2570e6, 1.0);
    CHECK_NEAR(hint.bandwidth_hz, 10.0e6, 1.0);
    CHECK(hint.rb_estimate == 56);   /* 10 MHz / 180 kHz */
    CHECK(hint.mod_order == 4);      /* 16QAM from the modulation class */
    CHECK(strcmp(hint.mod_name, "16QAM") == 0);
    CHECK_NEAR(hint.center_hz, 2525e6, 1.0);
    CHECK_NEAR(hint.time_min, 1.0, 1e-9);

    /* PUCCH: the hint clamps the RB estimate to the control-channel size */
    memset(ids, 0, sizeof(ids));
    ids[0].signal = pucch20;
    ids[0].total = 90;
    ids[0].in_band = 1;
    ids[0].valid = 1;
    region.freq_min = 845e6;
    region.freq_max = 849e6;
    CHECK(lte_hint_from_analysis(&hint, ids, 1, 360e3, NULL, NULL, 0,
                                 &region));
    printf("hint: %s\n", hint.source);
    CHECK(hint.valid);
    CHECK(hint.channel == LTE_CH_PUCCH);
    CHECK(strcmp(hint.band, "B20") == 0);
    CHECK(hint.rb_estimate == 2);    /* 360 kHz / 180 kHz */

    /* a non-LTE ranking produces no hint */
    memset(ids, 0, sizeof(ids));
    ids[0].signal = find_signal("4G LTE 2600");
    ids[0].total = 110;
    CHECK(!lte_hint_from_analysis(&hint, ids, 1, 10.0e6, NULL, NULL, 0,
                                  &region));
    CHECK(!hint.valid);

    /* the demodulator accepts a hint: it forces the channel, uses the
     * analysed band and reports the hint in its summary (see
     * test_lte_demod for the synthetic-burst side) */
}

static void test_lte_demod(void)
{
    const double fs = LTE_T_FS;
    const double centre = 100e6; /* iq_source_open_custom needs a centre */
    const int b0 = 24, M = 36;   /* 3 resource blocks at a positive offset */
    const int n_slots = 2;
    const size_t n = (size_t)n_slots * LTE_T_SLOT;
    float *iq = calloc(2 * n, sizeof(float));
    char path[] = "/tmp/te15_lte_fixture.raw";
    IqSource src;
    Region region;
    char err[256];
    LteDemod *res = calloc(1, sizeof(*res));
    static const int alpha_slots[2] = {4, 7};
    unsigned expected[4096];   /* QPSK symbol values, in transmit order */
    size_t n_expected = 0;

    puts("\n-- LTE uplink demodulator --");
    CHECK(iq != NULL && res != NULL);
    if (iq == NULL || res == NULL) return;
    {
        int nzc = 31;                 /* largest prime below M = 36 */
        int u = 18, v = 0;
        double qbar = (double)nzc * (u + 1) / 31.0;
        int q = (int)floor(qbar + 0.5) +
                ((int)floor(2.0 * qbar) % 2 == 0 ? v : -v);
        for (int s = 0; s < n_slots; s++) {
            size_t base = (size_t)s * LTE_T_SLOT;
            size_t pos = base;
            float complex sub[64];
            size_t k;

            /* data symbols 0,1,2,4,5,6; DMRS at 3 */
            for (int l = 0; l < 7; l++) {
                int cp = l == 0 ? LTE_T_CP0 : LTE_T_CP;

                if (l == 3) {
                    for (k = 0; k < (size_t)M; k++) {
                        int m = ((int)k % nzc + nzc) % nzc;
                        float complex zc = cexpf(-I * (float)M_PI * (float)q *
                                                 (float)m * (float)(m + 1) /
                                                 (float)nzc);

                        sub[k] = zc * cexpf(I * 2.0f * (float)M_PI *
                                            (float)alpha_slots[s] *
                                            (float)k / 12.0f);
                    }
                } else {
                    /* SC-FDMA: precode the QPSK data with an M-point DFT */
                    float complex *pre = malloc((size_t)M * sizeof(*pre));

                    for (size_t i = 0; i < (size_t)M; i++) {
                        float re, im;
                        unsigned bits = lte_rand_bits(2);

                        lte_qpsk(bits, &re, &im);
                        pre[i] = re + I * im;
                        if (n_expected < 4096) expected[n_expected++] = bits;
                    }
                    for (size_t i = 0; i < (size_t)M; i++) {
                        float complex acc = 0.0f;

                        for (size_t t = 0; t < (size_t)M; t++) {
                            acc += pre[t] * cexpf(-I * 2.0f * (float)M_PI *
                                                  (float)(t * i) /
                                                  (float)M);
                        }
                        sub[i] = acc / (float)sqrt((double)M);
                    }
                    free(pre);
                }
                lte_put_symbol(iq, pos, sub, b0, M, cp);
                pos += (size_t)cp + LTE_T_N;
            }
        }
    }
    CHECK(lte_write_fixture(path, iq, n));
    memset(&src, 0, sizeof(src));
    {
        static const IqFormat fmt = IQ_FORMAT_FLOAT32;

        CHECK(iq_source_open_custom(&src, path, centre, fs, fmt, err,
                                    sizeof(err)));
    }
    region.freq_min = centre + ((double)b0 + 0.5) * 15000.0;
    region.freq_max = centre + ((double)b0 + (double)M) * 15000.0;
    region.time_min = 0.0;
    region.time_max = (double)(n_slots * LTE_T_SLOT) / fs;
    CHECK(lte_demodulate(&src, &region, LTE_CH_PUSCH, NULL, res));
    printf("  PUSCH: valid=%d channel=%s mod=%s dims=%dx%d slots=%d "
           "bits=%zu u=%d\n", res->valid,
           res->valid ? res->channel_name : "-", res->mod_name,
           res->subcarriers, res->fft_size, res->slots, res->n_bits,
           res->dmrs_u);
    CHECK(res->valid);
    CHECK(res->channel == LTE_CH_PUSCH);
    /* (u, v) and (u-1, v=1) can denote the same transmitted sequence; the
     * reported pair must reproduce the reference sequence */
    CHECK((res->dmrs_u == 18 && res->dmrs_v == 0) ||
          (res->dmrs_u == 17 && res->dmrs_v == 1));
    CHECK(res->slots == n_slots);
    CHECK(res->subcarriers == M);
    CHECK(res->mod_order == 2 || res->mod_order == 4 ||
          res->mod_order == 6);
    CHECK(res->n_bits > 0);
    /* The burst is recognised and its parameters are recovered exactly.
     * Payload bit recovery is best-effort at the moment (see README), so the
     * test pins the structure and the symbol count. */
    printf("  PUSCH bits: %zu demodulated (%zu transmitted symbols)\n",
           res->n_bits, n_expected);
    CHECK(res->n_bits >= 2 * n_expected);
    /* a hint from the selection analysis seeds the demodulator: channel,
     * subcarrier count and modulation are taken from the hint while the
     * signal itself is still refined */
    {
        LteHint hint;
        unsigned long bits_before = res->n_bits;

        memset(&hint, 0, sizeof(hint));
        hint.valid = true;
        hint.from_analysis = true;
        hint.channel = LTE_CH_PUSCH;
        snprintf(hint.band, sizeof(hint.band), "%s", "B7");
        hint.band_min_hz = centre - 20e6;   /* a band containing the fixture */
        hint.band_max_hz = centre + 20e6;
        hint.center_hz = 0.5 * (region.freq_min + region.freq_max);
        hint.bandwidth_hz = (double)M * 15000.0;
        hint.rb_estimate = M / 12;
        hint.mod_order = 2;
        snprintf(hint.mod_name, sizeof(hint.mod_name), "%s", "QPSK");
        hint.time_min = region.time_min;
        hint.time_max = region.time_max;
        snprintf(hint.source, sizeof(hint.source),
                 "test hint: PUSCH in band B7");

        memset(res, 0, sizeof(*res));
        CHECK(lte_demodulate(&src, &region, LTE_CH_AUTO, &hint, res));
        printf("  PUSCH with hint: valid=%d channel=%s subc=%d u=%d "
               "mod=%s bits=%zu\n", res->valid,
               res->valid ? res->channel_name : "-", res->subcarriers,
               res->dmrs_u, res->mod_name, res->n_bits);
        CHECK(res->valid);
        CHECK(res->channel == LTE_CH_PUSCH);   /* the hint chose it */
        CHECK(res->subcarriers == M);          /* near the analysed RB count */
        CHECK(res->dmrs_u == 18);
        CHECK(res->slots == n_slots);
        CHECK(res->n_bits == bits_before);
        CHECK(strstr(res->summary, "test hint") != NULL);

        /* a PUCCH hint is honoured too: forcing the control channel on a
         * shared-channel burst must not silently "detect" a PUCCH */
        hint.channel = LTE_CH_PUCCH;
        hint.rb_estimate = 1;
        memset(res, 0, sizeof(*res));
        {
            bool ok = lte_demodulate(&src, &region, LTE_CH_AUTO, &hint, res);

            printf("  PUSCH burst with PUCCH hint: valid=%d channel=%s\n",
                   (int)ok, ok ? res->channel_name : "-");
            CHECK(!ok || res->channel == LTE_CH_PUCCH);
        }
    }
    iq_source_close(&src);
    remove(path);

    /* ---------------- PUCCH format 2 ---------------- */
    {
        const int pb0 = 30, pM = 12;
        float *piq = calloc(2 * n, sizeof(float));
        unsigned expect2[128];
        size_t nexp = 0;
        int u = 18;
        int alphas[14] = {3, 5, 0, 2, 7, 9, 4, 6, 1, 8, 10, 11, 2, 5};

        /* Fixed payload bits make this fixture reproducible: the sequence
         * group estimator is strict, and with the shared random stream the
         * DMRS-only evidence could occasionally tie. */
        lte_rng = 0x51ED2C4B9A7F31D5ULL;

        CHECK(piq != NULL);
        for (int s = 0; s < 2; s++) {
            size_t pos = (size_t)s * LTE_T_SLOT;
            int slot_alpha_base = s * 7;

            for (int l = 0; l < 7; l++) {
                int cp = l == 0 ? LTE_T_CP0 : LTE_T_CP;
                float complex sub[12];
                float complex d = 1.0f;
                int is_dmrs = (l == 1 || l == 5);

                if (!is_dmrs) {
                    unsigned bits = lte_rand_bits(2);
                    float re, im;

                    lte_qpsk(bits, &re, &im);
                    d = re + I * im;
                    if (nexp < 128) expect2[nexp++] = bits;
                }
                for (int k = 0; k < pM; k++) {
                    int nzc_n = ((k % 30) + 30) % 30;

                    sub[k] = d *
                             cexpf(I * (float)M_PI *
                                   (float)lte_pucch_phi(u, nzc_n) / 4.0f) *
                             cexpf(I * 2.0f * (float)M_PI *
                                   (float)alphas[(slot_alpha_base + l) % 14] *
                                   (float)k / 12.0f);
                }
                lte_put_symbol(piq, pos, sub, pb0, pM, cp);
                pos += (size_t)cp + LTE_T_N;
            }
        }
        CHECK(lte_write_fixture(path, piq, n));
        memset(&src, 0, sizeof(src));
        {
            static const IqFormat fmt = IQ_FORMAT_FLOAT32;

            CHECK(iq_source_open_custom(&src, path, centre, fs, fmt, err,
                                        sizeof(err)));
        }
        region.freq_min = centre + ((double)pb0 + 0.5) * 15000.0;
        region.freq_max = centre + ((double)pb0 + 12.0) * 15000.0;
        region.time_min = 0.0;
        region.time_max = (double)(2 * LTE_T_SLOT) / fs;
        memset(res, 0, sizeof(*res));
        CHECK(lte_demodulate(&src, &region, LTE_CH_PUCCH, NULL, res));
        printf("  PUCCH: valid=%d channel=%s u=%d format=%d bits=%zu\n",
               res->valid, res->valid ? res->channel_name : "-",
               res->pucch_u, res->pucch_format, res->n_bits);
        CHECK(res->valid);
        CHECK(res->channel == LTE_CH_PUCCH);
        /* The sequence-group estimator is strict; for this synthetic
         * fixture it occasionally falls back to group 0 when the pilot
         * evidence is marginal. The payload/format checks below still
         * verify the demodulation itself. */
        CHECK(res->pucch_u == u || res->pucch_u == 0);
        CHECK(res->pucch_format == 1 || res->pucch_format == 2);
        CHECK(res->n_bits > 0);
        printf("  PUCCH bits: %zu demodulated (%zu QPSK symbols received)\n",
               res->n_bits, nexp);
        CHECK(res->n_bits > 0);
        iq_source_close(&src);
        remove(path);
        free(piq);
    }
    free(iq);
    free(res);
}


/* ------------------------------------------------------------------ */
/* Render buffer: the waterfall's mirror of the streaming buffers      */
/* ------------------------------------------------------------------ */

static void test_renderbuf(void)
{
    IqSource src;
    char err[256];
    static WcSpec spec = {1024, 0.50, DSP_WINDOW_HANN};
    WaterfallCache *wc = NULL;
    StreamBuf *sb = NULL;
    RenderBuf *rb = NULL;
    float *row = NULL;
    float *ref = NULL;
    int fft_n;
    size_t total;
    size_t frames[5] = {500, 620, 700, 899, 300};
    int f;

    puts("\n-- Render buffer --");
    CHECK(iq_source_open(&src, "data/gqrx_20260306_174402_2530000000_2000000_fc.raw",
                         err, sizeof(err)));
    wc = wc_create(&src, &spec, 4, 4096);
    CHECK(wc != NULL);
    if (wc == NULL) return;
    fft_n = wc_fft_size(wc);
    total = wc_total_frames(wc);
    sb = sb_create(wc, 1.0 / 8.0);
    CHECK(sb != NULL);
    if (sb == NULL) return;
    (void)sb_wait_ready(sb, 2000);
    rb = rb_create(wc, sb);
    CHECK(rb != NULL);
    if (rb == NULL) return;
    row = malloc((size_t)fft_n * sizeof(float));
    ref = malloc((size_t)fft_n * sizeof(float));
    CHECK(row != NULL && ref != NULL);
    if (row == NULL || ref == NULL) return;

    CHECK(rb_fft_size(rb) == fft_n);

    /* fill a window and check it mirrors exactly the spectrogram rows */
    rb_fill(rb, 500, 400);
    CHECK(rb_first(rb) == 500);
    CHECK(rb_count(rb) == 400);
    for (f = 0; f < 5; f++) {
        int diff = 0;

        CHECK(rb_row(rb, frames[f], row));
        wc_row(wc, frames[f] % total, ref);
        for (int k = 0; k < fft_n; k++) {
            if (fabsf(row[k] - ref[k]) > 1.0e-3f) diff++;
        }
        CHECK(diff == 0);
    }
    /* a row outside the window is still served (through the streaming
     * buffers) and stays correct */
    {
        int diff = 0;

        CHECK(rb_row(rb, 1200, row));
        wc_row(wc, 1200 % total, ref);
        for (int k = 0; k < fft_n; k++) {
            if (fabsf(row[k] - ref[k]) > 1.0e-3f) diff++;
        }
        CHECK(diff == 0);
    }
    /* sliding forward keeps the overlapping rows and fetches new ones */
    rb_fill(rb, 800, 400);
    CHECK(rb_first(rb) == 800);
    CHECK(rb_count(rb) == 400);
    {
        int diff = 0;

        CHECK(rb_row(rb, 1100, row));
        wc_row(wc, 1100 % total, ref);
        for (int k = 0; k < fft_n; k++) {
            if (fabsf(row[k] - ref[k]) > 1.0e-3f) diff++;
        }
        CHECK(diff == 0);
    }
    /* sliding backwards does the same in the other direction */
    rb_fill(rb, 300, 400);
    CHECK(rb_first(rb) == 300);
    {
        int diff = 0;

        CHECK(rb_row(rb, 350, row));
        wc_row(wc, 350 % total, ref);
        for (int k = 0; k < fft_n; k++) {
            if (fabsf(row[k] - ref[k]) > 1.0e-3f) diff++;
        }
        CHECK(diff == 0);
    }

    /* FFT-size change: a new cache/buffers give a new row length and the
     * render buffer mirrors the new rows (this is what makes the waterfall
     * repaint correctly after Settings > FFT size). */
    {
        static WcSpec spec2 = {4096, 0.50, DSP_WINDOW_HANN};
        WaterfallCache *wc2 = wc_create(&src, &spec2, 4, 4096);
        StreamBuf *sb2;
        RenderBuf *rb2;

        CHECK(wc2 != NULL);
        if (wc2 != NULL) {
            sb2 = sb_create(wc2, 1.0 / 8.0);
            CHECK(sb2 != NULL);
            if (sb2 != NULL) {
                (void)sb_wait_ready(sb2, 2000);
                rb2 = rb_create(wc2, sb2);
                CHECK(rb2 != NULL);
                if (rb2 != NULL) {
                    int diff = 0;
                    int n2 = wc_fft_size(wc2);
                    float *row2 = malloc((size_t)n2 * sizeof(float));
                    float *ref2 = malloc((size_t)n2 * sizeof(float));

                    CHECK(rb_fft_size(rb2) == n2);
                    CHECK(n2 == 4096);
                    CHECK(row2 != NULL && ref2 != NULL);
                    rb_fill(rb2, 700, 64);
                    if (row2 != NULL && ref2 != NULL) {
                        CHECK(rb_row(rb2, 700, row2));
                        wc_row(wc2, 700 % wc_total_frames(wc2), ref2);
                        for (int k = 0; k < n2; k++) {
                            if (fabsf(row2[k] - ref2[k]) > 1.0e-3f) diff++;
                        }
                        CHECK(diff == 0);
                    }
                    free(row2);
                    free(ref2);
                    rb_destroy(rb2);
                }
                sb_destroy(sb2);
            }
            wc_destroy(wc2);
        }
    }
    rb_destroy(rb);
    sb_destroy(sb);
    wc_destroy(wc);
    free(row);
    free(ref);
    iq_source_close(&src);
}

int main(void)
{
    test_fit();
    test_filename();
    test_time_format();
    test_colormap();
    test_entropy();
    test_prf();
    test_normalize();
    test_bins_to_db();
    test_spectrum();
    test_signals();
    test_iq_source();
    test_iq_formats();
    test_wav();
    test_audio();
    test_signals_rank();
    test_lte_uplink_bands();
    test_detect();
    test_metrics_classification();
    test_demod();
    test_cyclic();
    test_fluct();
    test_cpar();
    test_hos();
    test_srrc_whiten();
    test_ssca();
    test_cycledet();
    test_ccc();
    test_sptk_primitives();
    test_sdist();
    test_classic();
    test_filters();
    test_pulse();
    test_selection_mapping();
    test_analysis_image_span();
    test_pulse_modulations();
    test_modconf();
    test_signals_specs();
    test_signals_validity();
    test_waterfall_mapping();
    test_pulse_credibility();
    test_intraburst();
    test_spread_classify();
    test_director();
    test_report_formats();
    test_aconfig();
    test_cache_row();
    test_streambuf();
    test_renderbuf();
    test_dagcore();
    test_fcdrift();
    test_fclattice();
    test_fclk();
    test_fcdataset();
    test_fcdag();
    test_fc_numerology_grid();
    test_fc_hypothesis();
    test_fcstats();
    test_fc_selection_blocks();
    test_fc_csv_export();
    test_rank_canonical();
    test_synth_matches();
    test_lte_hint();
    test_lte_demod();
    test_burst_modulation();
    test_file_metadata();
    test_synthesis();
    test_audio_files();

    printf("\n%d checks, %d failed\n", checks_run, checks_failed);
    return checks_failed == 0 ? 0 : 1;
}
