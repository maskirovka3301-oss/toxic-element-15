/*
 * main.c — entry point for the ToxicElement15 SDR waterfall analyzer.
 *
 * Usage:
 *   te15 [options] <gqrx_raw_file>
 *
 * Options:
 *   --rate HZ    override the sample rate parsed from the filename
 *   --freq HZ    override the centre frequency parsed from the filename
 *   --check      headless end-to-end verification (no GUI)
 *   -h, --help   show usage
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "analysis.h"
#include "cyclic.h"
#include "detect.h"
#include "cpar.h"
#include "fit.h"
#include "fluct.h"
#include "hos.h"
#include "ssca.h"
#include "cycledet.h"
#include "ccc.h"
#include "iq_source.h"
#include "metrics.h"
#include "signals.h"
#include "ui.h"
#include "wav.h"
#include "waterfall_cache.h"

#define FFT_SIZE          8192
#define FFT_OVERLAP       0.50
#define MAX_WINDOWS       4
#define MAX_WINDOW_FRAMES 4096

static void usage(const char *prog)
{
    printf("Usage: %s [options] <gqrx_raw_file>\n"
           "\n"
           "  --rate HZ   override sample rate from filename\n"
           "  --freq HZ   override centre frequency from filename\n"
           "  --check     headless end-to-end verification (no GUI)\n"
           "  -h, --help  show this help\n"
           "\n"
           "The file must be a gqrx I/Q recording named like\n"
           "gqrx_YYYYMMDD_HHMMSS_<freq_hz>_<rate_hz>_fc.raw\n",
           prog);
}

/*
 * Headless verification: load the recording, compute the first cache
 * window through the real worker thread, run a region analysis, and
 * print the results. Exercises everything except the GTK widgets.
 */
static int run_check(IqSource *src)
{
    WcSpec spec = {FFT_SIZE, FFT_OVERLAP, DSP_WINDOW_HANN};
    WaterfallCache *wc =
        wc_create(src, &spec, MAX_WINDOWS, MAX_WINDOW_FRAMES);
    if (wc == NULL) {
        fprintf(stderr, "check: cannot create waterfall cache\n");
        return 1;
    }

    const double fps = wc_frames_per_second(wc);
    printf("Recording: %.2f s, %.0f fps, %zu frames\n",
           src->duration, fps, wc_total_frames(wc));
    wc_request(wc, 0, 2 * (size_t)(1.0 * fps) + 1);

    const float *db = NULL;
    size_t win_start = 0, win_frames = 0, win_step = 1;
    for (int tries = 0; tries < 300; tries++) { /* up to 30 s */
        db = wc_lock(wc, 0, &win_start, &win_frames, &win_step);
        if (db != NULL) break;
        usleep(100 * 1000);
    }
    if (db == NULL) {
        fprintf(stderr, "check: first cache window never became ready\n");
        wc_destroy(wc);
        return 1;
    }
    printf("First window: frames %zu..%zu (%.2f s of data)\n",
           win_start, win_start + win_frames,
           win_frames / fps);
    /* Finite-sample spot check: data must be finite. */
    int fft_n = wc_fft_size(wc);
    int bad = 0;
    for (size_t i = 0; i < win_frames * (size_t)fft_n; i += 4096) {
        if (!isfinite(db[i])) bad++;
    }
    printf("Finite-sample spot check: %s\n", bad == 0 ? "ok" : "FAILED");

    /* Automatic signal detection over the first window. */
    SignalRegion regs[DETECT_MAX_REGIONS];
    size_t dn = detect_signals(db, win_frames, fft_n, win_start, win_step,
                               fps, src->sample_rate, src->center_freq,
                               regs, DETECT_MAX_REGIONS);
    printf("Detected %zu probable signals in the first window:\n", dn);
    for (size_t i = 0; i < dn && i < 16; i++) {
        printf("  %zu. %.3f-%.3f MHz  %.2f-%.2f s  peak %.1f dB\n", i + 1,
               regs[i].freq_min / 1e6, regs[i].freq_max / 1e6,
               regs[i].time_min, regs[i].time_max, regs[i].peak_db);
    }
    wc_unlock(wc);

    /* Region analysis across the centre of the band. */
    Region r = {
        src->center_freq - 1.0e6, src->center_freq + 1.0e6,
        0.25, 1.25,
    };
    Analysis a;
    memset(&a, 0, sizeof(a));
    bool ok = analysis_run(src, &r, &a);
    if (!ok) {
        fprintf(stderr, "check: analysis_run failed\n");
        wc_destroy(wc);
        return 1;
    }
    printf("Analysis image: %d frames x %d bins (%.1f Hz/bin, %.3f ms/frame)\n",
           a.frames, a.bins, a.bin_hz, a.dt * 1e3);

    /* Extended metrics + ranked candidates + demodulation pipeline. */
    Metrics m;
    if (metrics_compute(src, &r, &m)) {
        printf("\nExtended measurements:\n"
               "  occupied bw: %.1f kHz (%.3f..%.3f MHz)\n"
               "  flatness: %.3f  symmetry: %+.2f  purity: %.2f  snr: %.1f dB\n"
               "  am depth: %.2f  papr: %.1f dB  duty: %.2f  fm dev: %.1f kHz\n"
               "  fsk levels: %d  sym rate: %.1f Hz (conf %.0f%%)\n"
               "  class: %s — %s (demodulable=%d mode=%s)\n",
               m.occupied_bw_hz / 1e3, m.occupied_lo_hz / 1e6,
               m.occupied_hi_hz / 1e6, m.spectral_flatness, m.band_symmetry,
               m.tone_purity, m.snr_db, m.am_depth, m.papr_db, m.duty_cycle,
               m.fm_deviation_hz / 1e3, m.fsk_levels, m.sym_rate_hz,
               m.sym_rate_conf * 100.0, m.mod_class, m.demod_note,
               m.demodulable, m.demod_mode);

        RankedMatch ranked[3];
        size_t rn = signals_rank(a.center_freq, a.bandwidth, &m, ranked, 3);
        printf("Ranked candidates (%zu):\n", rn);
        for (size_t i = 0; i < rn; i++) {
            printf("  %zu. %-28s [%s] base %d agree %+d total %d\n",
                   i + 1, ranked[i].signal->name, ranked[i].signal->band,
                   ranked[i].base_confidence, ranked[i].agreement,
                   ranked[i].total);
        }

        if (m.demodulable) {
            size_t s0 = (size_t)(r.time_min * src->sample_rate);
            size_t n = (size_t)((r.time_max - r.time_min) * src->sample_rate);
            if (n > 32000000) n = 32000000;
            float *mix = malloc(2 * n * sizeof(float));
            size_t max_out = (size_t)(n * 32000.0 / src->sample_rate) + 64;
            float *mono = malloc(max_out * sizeof(float));
            if (mix != NULL && mono != NULL) {
                double off = a.center_freq - src->center_freq;
                iq_source_read_complex(src, s0, mix, n);
                metrics_mixdown(mix, n, src->sample_rate, -off, mix);
                size_t got = 0;
                if (strcmp(m.demod_mode, "fm") == 0) {
                    got = demod_fm(mix, n, src->sample_rate, 32000.0,
                                   m.fm_deviation_hz > 0 ? m.fm_deviation_hz
                                                         : 75000.0,
                                   mono, max_out);
                } else if (strcmp(m.demod_mode, "am") == 0) {
                    got = demod_am(mix, n, src->sample_rate, 32000.0,
                                   mono, max_out);
                } else if (strcmp(m.demod_mode, "cw") == 0) {
                    got = demod_cw(mix, n, src->sample_rate, 32000.0,
                                   mono, max_out);
                }
                if (got > 0) {
                    bool wok = wav_write("/tmp/te15_check_demod.wav", mono,
                                         got, 32000);
                    printf("Demod check: %zu samples -> %s\n", got,
                           wok ? "/tmp/te15_check_demod.wav"
                               : "wav write FAILED");
                }
            }
            free(mix);
            free(mono);
        }
    } else {
        printf("Extended measurements: FAILED\n");
    }
    analysis_clear(&a);

    /* Cyclostationary analysis over the same region. */
    CyclicResult cyc;
    memset(&cyc, 0, sizeof(cyc));
    if (cyclic_analyze(src, &r, &cyc)) {
        printf("Cyclostationary: %d cyclic features (res %.1f Hz)\n",
               cyc.peak_count, cyc.alpha_res);
        for (int i = 0; i < cyc.peak_count && i < 6; i++) {
            printf("  %d. %+.1f Hz (strength %.3f)\n", i + 1,
                   cyc.peaks[i].alpha, cyc.peaks[i].strength);
        }
    } else {
        printf("Cyclostationary: FAILED\n");
    }
    cyclic_clear(&cyc);

    /* Fluctuation-of-autocorrelation DSSS detector over the region. */
    FluctResult fl;
    memset(&fl, 0, sizeof(fl));
    if (fluct_analyze(src, &r, &fl)) {
        printf("Fluctuation: %s (F_peak %.1f @ %.1f us, thr %.1f)\n",
               fl.detected ? "DETECTED" : "none", fl.peak_fluct,
               fl.peak_lag * 1e6, fl.threshold);
    } else {
        printf("Fluctuation: FAILED\n");
    }
    fluct_clear(&fl);

    /* Cumulative peak-to-average ratio over the region. */
    CparResult cp;
    memset(&cp, 0, sizeof(cp));
    if (cpar_analyze(src, &r, &cp)) {
        printf("CPAR: %s (CPAR %.1f, best PAR %.1f @ %.3f ms, "
               "chip ~%.1f kHz)\n",
               cp.detected ? "DETECTED" : "none", cp.cpar, cp.peak_par,
               cp.peak_lag * 1e3, cp.chip_rate_hz / 1e3);
    } else {
        printf("CPAR: FAILED\n");
    }
    cpar_clear(&cp);

    /* Higher-order statistics over the region. */
    HosResult hs;
    memset(&hs, 0, sizeof(hs));
    if (hos_analyze(src, &r, &hs)) {
        printf("HOS: %s (|C42|/C21^2 = %.4f vs thr %.4f, "
               "|C40|/C21^2 = %.4f)\n",
               hs.non_gaussian ? "NON-GAUSSIAN" : "Gaussian-consistent",
               hs.n42, hs.n42_threshold, hs.n40);
    } else {
        printf("HOS: FAILED\n");
    }
    hos_clear(&hs);

    /* SSCA, cycle detectors and cyclic cumulants over the region. */
    SscaResult ss;
    memset(&ss, 0, sizeof(ss));
    if (ssca_analyze(src, &r, &ss)) {
        printf("SSCA: %zu significant cycle frequencies (thr %.4f)\n",
               ss.cf_count, ss.cf_threshold);
    } else {
        printf("SSCA: FAILED\n");
    }
    ssca_clear(&ss);

    CycleDetResult cd;
    memset(&cd, 0, sizeof(cd));
    if (cycledet_analyze(src, &r, &cd)) {
        printf("CYCLE DETECTOR: %s (multi-cycle %.4f vs thr %.4f, "
               "best CF %.1f Hz)\n",
               cd.detected ? "SIGNAL PRESENT" : "no signal",
               cd.multi_cycle, cd.multi_threshold, cd.best_cf);
    } else {
        printf("CYCLE DETECTOR: FAILED\n");
    }
    cycledet_clear(&cd);

    CccResult cc;
    memset(&cc, 0, sizeof(cc));
    if (ccc_analyze(src, &r, &cc)) {
        printf("CYCLIC CUMULANTS: %s (|C40|=%.2f |C41|=%.2f |C42|=%.2f)\n",
               cc.verdict, cc.n40, cc.n41, cc.n42);
    } else {
        printf("CYCLIC CUMULANTS: FAILED\n");
    }
    ccc_clear(&cc);

    /* Fit-rule spot checks (also covered by unit tests). */
    IntRect wide = fit_contain(400, 300, 200, 100);
    IntRect tall = fit_contain(400, 300, 100, 200);
    printf("fit_contain spot checks: wide=%dx%d@%d,%d tall=%dx%d@%d,%d\n",
           wide.w, wide.h, wide.x, wide.y, tall.w, tall.h, tall.x, tall.y);

    wc_destroy(wc);
    printf("check: PASS\n");
    return 0;
}

/*
 * File > Open integration: swap in a new recording, rebuild the cache,
 * and reset the UI. Returns 0 on success.
 */
/*
 * File > Open: open a recording using the metadata recovered from the
 * filename/container. Returns -2 when that metadata is missing (the UI
 * then asks the user for the parameters) and -1 on other failures.
 */
static int app_swap_file(Ui *ui, IqSource *ns, const char *path,
                         bool apply_settings);

static int on_open_file(Ui *ui, const char *path)
{
    IqSource *ns = calloc(1, sizeof(*ns));
    if (ns == NULL) return -1;
    char err[512];
    if (!iq_source_open(ns, path, err, sizeof(err))) {
        fprintf(stderr, "te15: %s\n", err);
        free(ns);
        return -2;
    }
    return app_swap_file(ui, ns, path, true);
}

/*
 * Manual-entry fallback (the parameter dialog): open the recording with
 * the centre frequency / sample rate / format the user supplied.
 */
static int on_open_file_custom(Ui *ui, const char *path, double center_hz,
                               double rate_hz, IqFormat format)
{
    IqSource *ns = calloc(1, sizeof(*ns));
    char err[512];

    if (ns == NULL) return -1;
    if (!iq_source_open_custom(ns, path, center_hz, rate_hz, format, err,
                               sizeof(err))) {
        fprintf(stderr, "te15: %s\n", err);
        free(ns);
        return -1;
    }
    return app_swap_file(ui, ns, path, false);
}

/* Swap a freshly opened source into the UI (shared by both open paths). */
static int app_swap_file(Ui *ui, IqSource *ns, const char *path, bool apply_settings)
{
    if (ns == NULL || ui == NULL) return -1;
    /* A newly opened recording keeps the parameters recovered from its own
     * header/filename: they are what the Settings page shows (preselected)
     * afterwards, so an audio file is not reinterpreted with the previous
     * file's sample format. */
    (void)apply_settings;

    WcSpec spec = {ui->settings_fft_size, FFT_OVERLAP, ui->settings_window};
    WaterfallCache *nwc =
        wc_create(ns, &spec, MAX_WINDOWS, MAX_WINDOW_FRAMES);
    if (nwc == NULL) {
        fprintf(stderr, "te15: cannot create waterfall cache for %s\n", path);
        iq_source_close(ns);
        free(ns);
        return -1;
    }

    /* Re-point the UI first, then release the old resources. */
    IqSource *os = ui->src;
    WaterfallCache *owc = ui->cache;
    ui->src = ns;
    ui->cache = nwc;
    ui_reset_source(ui);

    wc_destroy(owc);
    iq_source_close(os);
    free(os);

    printf("Opened: %s (%.2f s at %.3f MS/s)\n",
           ns->path, ns->duration, ns->sample_rate / 1e6);
    return 0;
}

/*
 * Settings integration: apply rate/format/centre to the open source,
 * rebuild the waterfall cache, and reset the view. Returns 0 on success.
 */
static int on_apply_settings(Ui *ui, double rate_hz, IqFormat format,
                             double center_hz, int fft_size,
                             DspWindow window_type)
{
    if (ui == NULL || rate_hz <= 0.0 || fft_size < 2) return -1;
    IqSource *s = ui->src;
    if (center_hz > 0.0) {
        s->center_freq = center_hz;
    }
    iq_source_set_format(s, format);
    iq_source_set_rate(s, rate_hz);

    WcSpec spec = {fft_size, FFT_OVERLAP, window_type};
    WaterfallCache *nwc =
        wc_create(s, &spec, MAX_WINDOWS, MAX_WINDOW_FRAMES);
    if (nwc == NULL) {
        fprintf(stderr, "te15: cannot rebuild waterfall cache\n");
        return -1;
    }
    WaterfallCache *owc = ui->cache;
    ui->cache = nwc;
    ui_reset_source(ui);
    wc_destroy(owc);

    printf("Settings applied: %.3f MS/s, format %d, centre %.3f MHz\n",
           s->sample_rate / 1e6, (int)s->format, s->center_freq / 1e6);
    return 0;
}

int main(int argc, char **argv)
{
    const char *path = NULL;
    double rate_override = 0.0, freq_override = 0.0;
    bool do_check = false;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (strcmp(a, "-h") == 0 || strcmp(a, "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else if (strcmp(a, "--check") == 0) {
            do_check = true;
        } else if (strcmp(a, "--rate") == 0 && i + 1 < argc) {
            rate_override = atof(argv[++i]);
        } else if (strcmp(a, "--freq") == 0 && i + 1 < argc) {
            freq_override = atof(argv[++i]);
        } else if (a[0] == '-') {
            fprintf(stderr, "Unknown option: %s\n", a);
            usage(argv[0]);
            return 2;
        } else if (path == NULL) {
            path = a;
        } else {
            fprintf(stderr, "Unexpected argument: %s\n", a);
            usage(argv[0]);
            return 2;
        }
    }

    if (path == NULL) {
        usage(argv[0]);
        return 2;
    }

    /* The source is heap-owned: File > Open replaces it at runtime. */
    IqSource *src = calloc(1, sizeof(*src));
    if (src == NULL) {
        fprintf(stderr, "te15: out of memory\n");
        return 1;
    }
    char err[512];
    if (!iq_source_open(src, path, err, sizeof(err))) {
        fprintf(stderr, "te15: %s\n", err);
        free(src);
        return 1;
    }
    /* Settings: the file's own parameters (header/filename) are the
     * defaults; the CLI overrides win when given. */
    double settings_rate =
        rate_override > 0.0 ? rate_override : src->sample_rate;
    double settings_center =
        freq_override > 0.0 ? freq_override : src->center_freq;
    IqFormat settings_format = src->is_audio ? IQ_FORMAT_FLOAT32 : src->format;
    if (rate_override > 0.0) {
        iq_source_set_rate(src, settings_rate);
    }
    if (freq_override > 0.0) {
        src->center_freq = settings_center;
    }
    printf("File: %s\n"
           "Samples: %zu\n"
           "Center: %.3f MHz\n"
           "Rate: %.3f MS/s\n"
           "Duration: %.2f s\n",
           src->path, src->total_samples,
           src->center_freq / 1e6, src->sample_rate / 1e6, src->duration);

    if (do_check) {
        int rc = run_check(src);
        iq_source_close(src);
        free(src);
        return rc;
    }

    WcSpec spec = {FFT_SIZE, FFT_OVERLAP, DSP_WINDOW_HANN};
    WaterfallCache *wc =
        wc_create(src, &spec, MAX_WINDOWS, MAX_WINDOW_FRAMES);
    if (wc == NULL) {
        fprintf(stderr, "te15: cannot create waterfall cache\n");
        iq_source_close(src);
        free(src);
        return 1;
    }

    Ui ui;
    ui_init(&ui, src, wc);
    ui.settings_rate = settings_rate;
    ui.settings_format = settings_format;
    ui.settings_center = settings_center;
    ui.settings_fft_size = FFT_SIZE;
    ui.settings_window = DSP_WINDOW_HANN;
    ui.open_file_cb = on_open_file;
    ui.open_file_custom_cb = on_open_file_custom;
    ui.apply_settings_cb = on_apply_settings;
    wc_request(wc, 0, 2 * (size_t)(ui.visible_seconds *
                                   wc_frames_per_second(wc)) + 1);
    int rc = ui_run(&ui);
    ui_cleanup(&ui);

    wc_destroy(wc);
    iq_source_close(src);
    free(src);
    return rc;
}
