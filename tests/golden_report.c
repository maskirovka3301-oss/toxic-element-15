/*
 * tests/golden_report.c — golden-report regression test.
 *
 * Generates a deterministic synthetic IQ fixture, runs the real analysis
 * worker on it, and compares the produced report text with the checked-in
 * golden file (the 'Generated:' timestamp line is ignored). Any change to
 * the report must be intentional and accompanied by a regenerated golden
 * file (run with --update).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ui.c"

static void make_fixture(const char *dir)
{
    const double rate = 2.0e6, seconds = 0.5, period_s = 0.025;
    const double burst_s = 0.005;
    size_t n = (size_t)(rate * seconds);
    float *iq = calloc(2 * n, sizeof(float));
    unsigned long long r = 0x1234567890ABCDEFULL;
    char path[512], cmd[512];
    FILE *f;

    for (size_t i = 0; i < n; i++) {
        r ^= r << 13;
        r ^= r >> 7;
        r ^= r << 17;
        iq[2 * i] = (float)(((r >> 8) & 0xFFFF) / 65536.0 - 0.5) * 0.01f;
        iq[2 * i + 1] = (float)(((r >> 24) & 0xFFFF) / 65536.0 - 0.5) * 0.01f;
    }
    for (double t0 = 0.0; t0 < seconds; t0 += period_s) {
        double x = 0.0;

        while (x < burst_s) {
            size_t a = (size_t)((t0 + x) * rate);
            size_t b = a + (size_t)(10e-6 * rate);

            for (size_t k = a; k < b && k < n; k++) {
                double ph = 2.0 * M_PI * 50e3 * (double)k / rate;

                iq[2 * k] += (float)(0.6 * cos(ph));
                iq[2 * k + 1] += (float)(0.6 * sin(ph));
            }
            x += 20e-6;
        }
    }
    snprintf(cmd, sizeof(cmd), "mkdir -p %s", dir);
    if (system(cmd) != 0) { /* best effort */ }
    snprintf(path, sizeof(path),
             "%s/gqrx_20261112_190000_100000000_2000000_fc.raw", dir);
    f = fopen(path, "wb");
    if (f != NULL) {
        fwrite(iq, sizeof(float), 2 * n, f);
        fclose(f);
    }
    free(iq);
}

/* Normalise the volatile parts of the report so the comparison is stable:
 * the 'Generated:' timestamp line and the DAG stage elapsed times (which
 * depend on machine load). Everything else must match byte for byte. */
static char *strip_timestamp(const char *text)
{
    char *out = malloc(strlen(text) + 1);
    char *dst = out;
    const char *p = text;

    if (out == NULL) return NULL;
    while (*p != '\0') {
        const char *eol = strchr(p, '\n');
        size_t len = eol != NULL ? (size_t)(eol - p) + 1 : strlen(p);

        if (len < 10 || strncmp(p, "Generated:", 10) != 0) {
            memcpy(dst, p, len);
            dst += len;
        }
        p += len;
    }
    *dst = '\0';
    /* normalise "(1.23 s)" stage timings to "(0.00 s)" */
    {
        char *p = out;

        while ((p = strstr(p, " s)")) != NULL) {
            char *open = p;

            /* walk back to the '(' */
            while (open > out && open[-1] != '(' && open[-1] != '\n' &&
                   p - open < 20) {
                open--;
            }
            if (open > out && open[-1] == '(' && open < p) {
                /* only digits and dots between '(' and ' s)' */
                bool numeric = open < p;

                for (char *q = open; q < p; q++) {
                    if (!(g_ascii_isdigit(*q) || *q == '.')) numeric = false;
                }
                if (numeric) {
                    memmove(open, "0.00 s)", 7);
                    p = open + 7;
                    continue;
                }
            }
            p += 3;
        }
    }
    return out;
}

static char *run_report(void)
{
    const char *dir = "/tmp/te15_golden";
    char path[512];
    IqSource s;
    char err[256];
    Ui ui;
    ReportJob *job;
    char *text;

    make_fixture(dir);
    snprintf(path, sizeof(path),
             "%s/gqrx_20261112_190000_100000000_2000000_fc.raw", dir);
    if (!iq_source_open(&s, path, err, sizeof(err))) {
        fprintf(stderr, "cannot open fixture: %s\n", err);
        return NULL;
    }
    memset(&ui, 0, sizeof(ui));
    ui.src = &s;
    job = g_new0(ReportJob, 1);
    job->ui = &ui;
    job->region.freq_min = 100e6 - 100e3;
    job->region.freq_max = 100e6 + 100e3;
    job->region.time_min = 0.0;
    job->region.time_max = 0.4;
    job->total = REPORT_TOTAL_MAX;
    job->settings_rate = s.sample_rate;
    job->settings_format = s.format;
    job->settings_center = s.center_freq;
    job->settings_fft_size = 32768;
    job->settings_window = DSP_WINDOW_RECTANGULAR;
    job->update_only = true;
    report_worker(job);
    text = job->report != NULL ? strip_timestamp(job->report) : NULL;
    g_free(job->report);
    g_free(job);
    iq_source_close(&s);
    return text;
}

int main(int argc, char **argv)
{
    const char *golden = argc > 1 ? argv[1] : "tests/golden/report_synth.txt";
    bool update = (argc > 2 && strcmp(argv[2], "--update") == 0);
    char *text = run_report();
    FILE *f;

    if (text == NULL) {
        fprintf(stderr, "golden: the worker produced no report\n");
        return 1;
    }
    if (update) {
        f = fopen(golden, "wb");
        if (f == NULL) {
            fprintf(stderr, "golden: cannot write %s\n", golden);
            free(text);
            return 1;
        }
        fwrite(text, 1, strlen(text), f);
        fclose(f);
        printf("golden: wrote %s (%zu bytes)\n", golden, strlen(text));
        free(text);
        return 0;
    }
    {
        long size;
        char *want;

        f = fopen(golden, "rb");
        if (f == NULL) {
            fprintf(stderr, "golden: missing %s (run --update once)\n",
                    golden);
            free(text);
            return 1;
        }
        fseek(f, 0, SEEK_END);
        size = ftell(f);
        fseek(f, 0, SEEK_SET);
        want = malloc((size_t)size + 1);
        if (want == NULL || fread(want, 1, (size_t)size, f) != (size_t)size) {
            fprintf(stderr, "golden: cannot read %s\n", golden);
            fclose(f);
            free(want);
            free(text);
            return 1;
        }
        want[size] = '\0';
        fclose(f);
        if (strcmp(want, text) != 0) {
            size_t i = 0;

            while (want[i] != '\0' && text[i] != '\0' && want[i] == text[i]) {
                i++;
            }
            fprintf(stderr,
                    "golden: report changed at offset %zu\n  expected: %.120s\n"
                    "  got:      %.120s\n",
                    i, want + i, text + i);
            free(want);
            free(text);
            return 1;
        }
        printf("golden: report text matches %s (%zu bytes)\n", golden,
               strlen(text));
        free(want);
        free(text);
    }
    return 0;
}
