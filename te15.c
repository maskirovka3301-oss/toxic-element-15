/*
 * SDR IQ Waterfall Analyzer - COMPLETELY FIXED VERSION
 * 
 * Compile: gcc -o sdr_waterfall sdr_waterfall.c \
 *          $(pkg-config --cflags --libs gtk+-3.0) \
 *          -I/opt/homebrew/include -L/opt/homebrew/lib \
 *          -lfftw3f -lpthread -lm -O3
 */

#include <gtk/gtk.h>
#include <fftw3.h>
#include <cairo.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <complex.h>

#define FFT_SIZE 8192
#define FFT_OVERLAP 0.50
#define SCROLL_SECONDS 1.0
#define CACHE_SECONDS 3.0
#define MAX_CACHE_FRAMES 2000
#define PRECACHE_WINDOWS 2
#define MAX_CACHE_WINDOWS 4
#define LOAD_BATCH_SIZE 50

#define DEFAULT_DB_MIN -100.0
#define DEFAULT_DB_MAX -20.0
#define MIN_VISIBLE_SECONDS 0.1
#define MAX_VISIBLE_SECONDS 10.0
#define DEFAULT_VISIBLE_SECONDS 1.0

typedef struct { float r, g, b; } Color;
static Color viridis_colormap[256];

static void init_viridis_colormap(void) {
    for (int i = 0; i < 256; i++) {
        float t = i / 255.0f;
        if (t < 0.25f) {
            float x = t * 4.0f;
            viridis_colormap[i].r = 0.267f + (0.118f - 0.267f) * x;
            viridis_colormap[i].g = 0.005f + (0.384f - 0.005f) * x;
            viridis_colormap[i].b = 0.329f + (0.468f - 0.329f) * x;
        } else if (t < 0.5f) {
            float x = (t - 0.25f) * 4.0f;
            viridis_colormap[i].r = 0.118f + (0.194f - 0.118f) * x;
            viridis_colormap[i].g = 0.384f + (0.718f - 0.384f) * x;
            viridis_colormap[i].b = 0.468f + (0.482f - 0.468f) * x;
        } else if (t < 0.75f) {
            float x = (t - 0.5f) * 4.0f;
            viridis_colormap[i].r = 0.194f + (0.478f - 0.194f) * x;
            viridis_colormap[i].g = 0.718f + (0.878f - 0.718f) * x;
            viridis_colormap[i].b = 0.482f + (0.314f - 0.482f) * x;
        } else {
            float x = (t - 0.75f) * 4.0f;
            viridis_colormap[i].r = 0.478f + (0.993f - 0.478f) * x;
            viridis_colormap[i].g = 0.878f + (0.988f - 0.878f) * x;
            viridis_colormap[i].b = 0.314f + (0.090f - 0.314f) * x;
        }
    }
}

typedef struct {
    int start_frame;
    int end_frame;
    float *data;
    int valid;
    int last_access;
    int ref_count;
    pthread_mutex_t mutex;
} CacheWindow;

typedef struct {
    double freq_min;
    double freq_max;
    double time_min;
    double time_max;
    int active;
} Selection;

typedef struct {
    double freq_min;
    double freq_max;
    double time_min;
    double time_max;
    double bandwidth;
    double center_freq;
    double duration;
    double avg_power;
    double max_power;
    double entropy;
    double prf;
    char signal_type[64];
    char modulation[64];
    int match_count;
    char matches[10][64];
    int match_confidence[10];
    float *waterfall_image;
    int image_frames;
    int image_bins;
} AnalysisResults;

typedef struct {
    char *filepath;
    float complex *iq_data;
    size_t total_samples;
    double sample_rate;
    double center_freq;
    double total_duration;
    int fd;
    size_t file_size;
    
    CacheWindow cache_windows[MAX_CACHE_WINDOWS];
    int num_cache_windows;
    int cache_frames;
    int fft_size;
    double fft_overlap;
    double visible_seconds;
    double scroll_seconds;
    double scroll_position;
    int visible_frames;
    int total_frames;
    double frames_per_second;
    double time_per_frame;
    
    int cache_start_frame;
    int cache_end_frame;
    
    double db_min;
    double db_max;
    int auto_scale;
    
    Selection selection;
    AnalysisResults analysis;
    
    cairo_surface_t *waterfall_surface;
    
    GtkWidget *window;
    GtkWidget *waterfall_drawing;
    GtkWidget *info_label;
    GtkWidget *zoom_label;
    GtkWidget *db_min_slider;
    GtkWidget *db_max_slider;
    GtkWidget *auto_scale_check;
    GtkWidget *analyze_btn;
    GtkWidget *screenshot_btn;
    GtkWidget *popup_window;
    
    pthread_mutex_t cache_mutex;
    pthread_t load_thread;
    int loading;
    int load_start_frame;
    int load_end_frame;
    int load_progress;
    int quit_thread;
    int needs_redraw;
    
    int is_scrolling;
    int last_cached_frame;
    
} SDRAnalyzer;

// Forward Declarations
static void* cache_load_thread(void *arg);
static void manage_cache_windows(SDRAnalyzer *a, int target_frame);
static int find_or_create_cache_window(SDRAnalyzer *a, int start_frame);
static void update_zoom_display(SDRAnalyzer *a);
static CacheWindow* find_cache_window(SDRAnalyzer *a, int frame_idx);
static void analyze_selection(SDRAnalyzer *a);
static gboolean on_scroll_timeout(gpointer user_data);
static void ensure_cache(SDRAnalyzer *a, int frame_idx);
static void init_cache_windows(SDRAnalyzer *a);
static void free_cache_window(CacheWindow *cw);
static void request_cache_load(SDRAnalyzer *a, int start_frame);
static int parse_filename(SDRAnalyzer *a);
static int load_iq_file(SDRAnalyzer *a);
static gboolean ui_update_timer(gpointer user_data);
static void on_db_min_changed(GtkRange *range, gpointer user_data);
static void on_db_max_changed(GtkRange *range, gpointer user_data);
static void on_auto_scale_toggled(GtkToggleButton *toggle, gpointer user_data);
static void on_zoom_in(GtkButton *button, gpointer user_data);
static void on_zoom_out(GtkButton *button, gpointer user_data);
static void on_scroll(SDRAnalyzer *a, double delta);
static gboolean on_key_press(GtkWidget *widget, GdkEventKey *event, gpointer user_data);
static gboolean on_waterfall_button_press(GtkWidget *widget, GdkEventButton *event, gpointer user_data);
static gboolean on_waterfall_button_release(GtkWidget *widget, GdkEventButton *event, gpointer user_data);
static gboolean draw_waterfall(GtkWidget *widget, cairo_t *cr, gpointer user_data);
static void on_analyze_button(GtkButton *button, gpointer user_data);
static void on_screenshot_button(GtkButton *button, gpointer user_data);
static gboolean draw_analysis_popup(GtkWidget *widget, cairo_t *cr, gpointer user_data);
static double calculate_entropy(float *data, int size);
static double calculate_prf(float *data, int frames, double time_per_frame);
static void match_known_signals(SDRAnalyzer *a, double center_freq, double bandwidth);

static double calculate_entropy(float *data, int size) {
    if (size <= 0 || data == NULL) return 0.0;
    double sum = 0.0, min_val = data[0], max_val = data[0];
    for (int i = 0; i < size; i++) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }
    double range = max_val - min_val + 1e-10;
    double *norm = malloc(size * sizeof(double));
    if (!norm) return 0.0;
    for (int i = 0; i < size; i++) {
        norm[i] = (data[i] - min_val) / range;
        sum += norm[i];
    }
    double entropy = 0.0;
    for (int i = 0; i < size; i++) {
        double p = norm[i] / (sum + 1e-10);
        if (p > 1e-10) entropy -= p * log2(p);
    }
    free(norm);
    return entropy;
}

static double calculate_prf(float *data, int frames, double time_per_frame) {
    if (frames <= 0 || data == NULL || time_per_frame <= 0) return 0.0;
    int num_bins = frames > 0 ? (int)(sqrt(frames)) : 10;
    if (num_bins < 10) num_bins = 10;
    int actual_frames = frames / num_bins;
    if (actual_frames < 3) return 0.0;
    double *time_series = malloc(actual_frames * sizeof(double));
    if (!time_series) return 0.0;
    for (int i = 0; i < actual_frames; i++) {
        double sum = 0.0;
        for (int j = 0; j < num_bins; j++) {
            int idx = i * num_bins + j;
            if (idx < frames) sum += data[idx];
        }
        time_series[i] = sum / num_bins;
    }
    double sum = 0.0;
    for (int i = 0; i < actual_frames; i++) sum += time_series[i];
    double mean = sum / actual_frames;
    double variance = 0.0;
    for (int i = 0; i < actual_frames; i++) {
        double diff = time_series[i] - mean;
        variance += diff * diff;
    }
    variance /= actual_frames;
    double std = sqrt(variance);
    double threshold = mean + 1.0 * std;
    int *peaks = malloc(actual_frames * sizeof(int));
    if (!peaks) { free(time_series); return 0.0; }
    int peak_count = 0, above = 0, min_gap = 2;
    for (int i = 0; i < actual_frames; i++) {
        if (time_series[i] > threshold && !above) {
            if (peak_count == 0 || (i - peaks[peak_count-1]) >= min_gap) {
                peaks[peak_count++] = i;
                above = 1;
            }
        } else if (time_series[i] < mean) {
            above = 0;
        }
    }
    double prf = 0.0;
    if (peak_count >= 2) {
        double interval_sum = 0.0;
        int interval_count = 0;
        for (int i = 1; i < peak_count; i++) {
            double interval = (peaks[i] - peaks[i-1]) * time_per_frame * num_bins;
            if (interval > 0.001 && interval < 10.0) {
                interval_sum += interval;
                interval_count++;
            }
        }
        if (interval_count > 0) {
            double avg_interval = interval_sum / interval_count;
            if (avg_interval > 0) prf = 1.0 / avg_interval;
        }
    }
    free(time_series);
    free(peaks);
    return prf;
}

static void match_known_signals(SDRAnalyzer *a, double center_freq, double bandwidth) {
    a->analysis.match_count = 0;
    struct { const char *name; double freq_min; double freq_max; double bw_min; double bw_max; } signals[] = {
        {"5G NR Downlink", 2300e6, 2690e6, 5e6, 100e6}, {"5G NR Uplink", 2300e6, 2690e6, 5e6, 100e6},
        {"4G LTE Downlink", 2300e6, 2690e6, 1.4e6, 20e6}, {"4G LTE Uplink", 2300e6, 2690e6, 1.4e6, 20e6},
        {"WiFi 802.11n/g", 2400e6, 2483.5e6, 20e6, 40e6}, {"WiFi 802.11ac/ax", 2400e6, 2483.5e6, 20e6, 160e6},
        {"Logitech Wireless", 2400e6, 2483.5e6, 500e3, 2e6}, {"Zigbee", 2400e6, 2483.5e6, 2e6, 2e6},
        {"Bluetooth Classic", 2400e6, 2483.5e6, 1e6, 3e6}, {"Bluetooth LE", 2400e6, 2483.5e6, 1e6, 2e6},
    };
    for (int i = 0; i < 10 && a->analysis.match_count < 10; i++) {
        int confidence = 0;
        if (center_freq >= signals[i].freq_min && center_freq <= signals[i].freq_max) confidence += 70;
        if (bandwidth >= signals[i].bw_min && bandwidth <= signals[i].bw_max) confidence += 30;
        if (confidence >= 50) {
            strcpy(a->analysis.matches[a->analysis.match_count], signals[i].name);
            a->analysis.match_confidence[a->analysis.match_count] = confidence;
            a->analysis.match_count++;
        }
    }
}

static gboolean on_scroll_timeout(gpointer user_data) {
    SDRAnalyzer *a = (SDRAnalyzer *)user_data;
    pthread_mutex_lock(&a->cache_mutex);
    a->is_scrolling = 0;
    pthread_mutex_unlock(&a->cache_mutex);
    int current_frame = (int)(a->scroll_position * a->frames_per_second);
    ensure_cache(a, current_frame);
    return G_SOURCE_REMOVE;
}

static void init_cache_windows(SDRAnalyzer *a) {
    a->num_cache_windows = 0;
    for (int i = 0; i < MAX_CACHE_WINDOWS; i++) {
        a->cache_windows[i].valid = 0;
        a->cache_windows[i].data = NULL;
        a->cache_windows[i].ref_count = 0;
        a->cache_windows[i].last_access = 0;
        pthread_mutex_init(&a->cache_windows[i].mutex, NULL);
    }
    a->selection.active = 0;
    a->is_scrolling = 0;
    a->last_cached_frame = -1;
}

static void free_cache_window(CacheWindow *cw) {
    pthread_mutex_lock(&cw->mutex);
    if (cw->data) { free(cw->data); cw->data = NULL; }
    cw->valid = 0;
    pthread_mutex_unlock(&cw->mutex);
}

static CacheWindow* find_cache_window(SDRAnalyzer *a, int frame_idx) {
    for (int i = 0; i < a->num_cache_windows; i++) {
        pthread_mutex_lock(&a->cache_windows[i].mutex);
        if (a->cache_windows[i].valid && frame_idx >= a->cache_windows[i].start_frame && frame_idx < a->cache_windows[i].end_frame) {
            pthread_mutex_unlock(&a->cache_windows[i].mutex);
            return &a->cache_windows[i];
        }
        pthread_mutex_unlock(&a->cache_windows[i].mutex);
    }
    return NULL;
}

static void manage_cache_windows(SDRAnalyzer *a, int target_frame) {
    pthread_mutex_lock(&a->cache_mutex);
    int frames_per_window = a->cache_frames;
    int max_distance = frames_per_window * 2;
    for (int i = a->num_cache_windows - 1; i >= 0; i--) {
        pthread_mutex_lock(&a->cache_windows[i].mutex);
        if (!a->cache_windows[i].valid || a->cache_windows[i].ref_count > 0) {
            pthread_mutex_unlock(&a->cache_windows[i].mutex);
            continue;
        }
        int mid_frame = (a->cache_windows[i].start_frame + a->cache_windows[i].end_frame) / 2;
        int dist = abs(mid_frame - target_frame);
        if (dist > max_distance && a->num_cache_windows > PRECACHE_WINDOWS) {
            if (a->cache_windows[i].data) { free(a->cache_windows[i].data); a->cache_windows[i].data = NULL; }
            a->cache_windows[i].valid = 0;
            pthread_mutex_unlock(&a->cache_windows[i].mutex);
            for (int j = i; j < a->num_cache_windows - 1; j++) a->cache_windows[j] = a->cache_windows[j + 1];
            a->num_cache_windows--;
            continue;
        }
        pthread_mutex_unlock(&a->cache_windows[i].mutex);
    }
    pthread_mutex_unlock(&a->cache_mutex);
}

static int find_or_create_cache_window(SDRAnalyzer *a, int start_frame) {
    pthread_mutex_lock(&a->cache_mutex);
    for (int i = 0; i < a->num_cache_windows; i++) {
        pthread_mutex_lock(&a->cache_windows[i].mutex);
        if (a->cache_windows[i].valid && a->cache_windows[i].start_frame == start_frame) {
            a->cache_windows[i].last_access = time(NULL);
            pthread_mutex_unlock(&a->cache_windows[i].mutex);
            pthread_mutex_unlock(&a->cache_mutex);
            return i;
        }
        pthread_mutex_unlock(&a->cache_windows[i].mutex);
    }
    if (a->num_cache_windows >= MAX_CACHE_WINDOWS) {
        int lru_idx = 0;
        time_t lru_time = a->cache_windows[0].last_access;
        for (int i = 1; i < a->num_cache_windows; i++) {
            pthread_mutex_lock(&a->cache_windows[i].mutex);
            if (a->cache_windows[i].ref_count == 0 && a->cache_windows[i].last_access < lru_time) {
                lru_time = a->cache_windows[i].last_access;
                lru_idx = i;
            }
            pthread_mutex_unlock(&a->cache_windows[i].mutex);
        }
        pthread_mutex_lock(&a->cache_windows[lru_idx].mutex);
        if (a->cache_windows[lru_idx].data) { free(a->cache_windows[lru_idx].data); a->cache_windows[lru_idx].data = NULL; }
        a->cache_windows[lru_idx].valid = 0;
        pthread_mutex_unlock(&a->cache_windows[lru_idx].mutex);
        pthread_mutex_unlock(&a->cache_mutex);
        return lru_idx;
    }
    int idx = a->num_cache_windows++;
    pthread_mutex_unlock(&a->cache_mutex);
    pthread_mutex_lock(&a->cache_windows[idx].mutex);
    a->cache_windows[idx].valid = 0;
    a->cache_windows[idx].data = NULL;
    a->cache_windows[idx].ref_count = 0;
    pthread_mutex_unlock(&a->cache_windows[idx].mutex);
    return idx;
}

static void* cache_load_thread(void *arg) {
    SDRAnalyzer *a = (SDRAnalyzer *)arg;
    fftwf_complex *in = fftwf_malloc(sizeof(fftwf_complex) * a->fft_size);
    fftwf_complex *out = fftwf_malloc(sizeof(fftwf_complex) * a->fft_size);
    fftwf_plan plan = fftwf_plan_dft_1d(a->fft_size, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    float *window = malloc(a->fft_size * sizeof(float));
    for (int i = 0; i < a->fft_size; i++) window[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (a->fft_size - 1)));
    int hop_size = (int)(a->fft_size * (1.0 - a->fft_overlap));
    while (!a->quit_thread) {
        pthread_mutex_lock(&a->cache_mutex);
        while (!a->loading && !a->quit_thread) { pthread_mutex_unlock(&a->cache_mutex); usleep(10000); pthread_mutex_lock(&a->cache_mutex); }
        if (a->quit_thread) { pthread_mutex_unlock(&a->cache_mutex); break; }
        int start_frame = a->load_start_frame;
        int end_frame = a->load_end_frame;
        int max_frames = end_frame - start_frame;
        if (max_frames > MAX_CACHE_FRAMES) end_frame = start_frame + MAX_CACHE_FRAMES;
        pthread_mutex_unlock(&a->cache_mutex);
        printf("Loading cache: frames %d to %d...\n", start_frame, end_frame);
        int cw_idx = find_or_create_cache_window(a, start_frame);
        pthread_mutex_lock(&a->cache_windows[cw_idx].mutex);
        CacheWindow *cw = &a->cache_windows[cw_idx];
        cw->start_frame = start_frame;
        cw->end_frame = end_frame;
        cw->ref_count = 1;
        pthread_mutex_unlock(&a->cache_windows[cw_idx].mutex);
        int num_frames = end_frame - start_frame;
        size_t data_size = (size_t)num_frames * a->fft_size * sizeof(float);
        cw->data = malloc(data_size);
        if (!cw->data) { pthread_mutex_lock(&a->cache_windows[cw_idx].mutex); cw->valid = 0; cw->ref_count = 0; pthread_mutex_unlock(&a->cache_windows[cw_idx].mutex); continue; }
        int frames_loaded = 0;
        for (int i = start_frame; i < end_frame; i++) {
            if (a->quit_thread) break;
            int cache_idx = i - start_frame;
            size_t sample_start = (size_t)i * hop_size;
            size_t sample_end = sample_start + a->fft_size;
            if (sample_end > a->total_samples) break;
            for (int j = 0; j < a->fft_size; j++) {
                float complex sample = a->iq_data[sample_start + j];
                in[j][0] = crealf(sample) * window[j];
                in[j][1] = cimagf(sample) * window[j];
            }
            fftwf_execute(plan);
            for (int j = 0; j < a->fft_size; j++) {
                int shifted_idx = (j + a->fft_size / 2) % a->fft_size;
                double mag = sqrt(out[shifted_idx][0] * out[shifted_idx][0] + out[shifted_idx][1] * out[shifted_idx][1]);
                cw->data[cache_idx * a->fft_size + j] = 20.0 * log10(mag + 1e-10);
            }
            frames_loaded++;
            if (frames_loaded % LOAD_BATCH_SIZE == 0) { pthread_mutex_lock(&a->cache_mutex); a->load_progress = (frames_loaded * 100) / num_frames; a->needs_redraw = 1; pthread_mutex_unlock(&a->cache_mutex); }
        }
        pthread_mutex_lock(&a->cache_windows[cw_idx].mutex);
        cw->valid = 1; cw->last_access = time(NULL);
        pthread_mutex_unlock(&a->cache_windows[cw_idx].mutex);
        pthread_mutex_lock(&a->cache_mutex);
        a->cache_start_frame = start_frame;
        a->cache_end_frame = end_frame;
        a->loading = 0;
        a->load_progress = 100;
        a->needs_redraw = 1;
        a->last_cached_frame = start_frame;
        pthread_mutex_unlock(&a->cache_mutex);
        pthread_mutex_lock(&a->cache_windows[cw_idx].mutex);
        cw->ref_count = 0;
        pthread_mutex_unlock(&a->cache_windows[cw_idx].mutex);
        printf("Cache loaded (%d frames)\n", frames_loaded);
    }
    fftwf_destroy_plan(plan);
    fftwf_free(in); fftwf_free(out); free(window);
    return NULL;
}

static int parse_filename(SDRAnalyzer *a) {
    char *basename = strrchr(a->filepath, '/');
    basename = basename ? basename + 1 : a->filepath;
    unsigned int date, time;
    long long freq, rate;
    char type[16];
    if (sscanf(basename, "gqrx_%u_%u_%lld_%lld_%15s.raw", &date, &time, &freq, &rate, type) == 5) {
        a->center_freq = (double)freq;
        a->sample_rate = (double)rate;
        printf("File: %s\nCenter Frequency: %.2f MHz\nSample Rate: %.2f MS/s\n", basename, a->center_freq/1e6, a->sample_rate/1e6);
        return 1;
    }
    return 0;
}

static int load_iq_file(SDRAnalyzer *a) {
    printf("Loading IQ file...\n");
    a->fd = open(a->filepath, O_RDONLY);
    if (a->fd < 0) { perror("Failed to open file"); return -1; }
    struct stat st;
    if (fstat(a->fd, &st) < 0) { perror("Failed to stat file"); close(a->fd); return -1; }
    a->file_size = st.st_size;
    printf("File size: %.1f MB\n", a->file_size / 1e6);
    a->iq_data = mmap(NULL, a->file_size, PROT_READ, MAP_PRIVATE, a->fd, 0);
    if (a->iq_data == MAP_FAILED) { perror("Failed to mmap file"); close(a->fd); return -1; }
    if (a->file_size % 8 == 0) a->total_samples = a->file_size / 8;
    else a->total_samples = a->file_size / 16;
    a->total_duration = a->total_samples / a->sample_rate;
    printf("Memory-mapped %zu samples\nTotal duration: %.2f seconds\n", a->total_samples, a->total_duration);
    return 0;
}

static void request_cache_load(SDRAnalyzer *a, int start_frame) {
    pthread_mutex_lock(&a->cache_mutex);
    if (a->loading) { pthread_mutex_unlock(&a->cache_mutex); return; }
    if (a->last_cached_frame >= 0) { int diff = abs(start_frame - a->last_cached_frame); if (diff < 100) { pthread_mutex_unlock(&a->cache_mutex); return; } }
    CacheWindow *existing = find_cache_window(a, start_frame);
    if (existing && existing->valid) { pthread_mutex_unlock(&a->cache_mutex); return; }
    int end_frame = start_frame + a->cache_frames;
    if (end_frame > a->total_frames) end_frame = a->total_frames;
    if (end_frame - start_frame > MAX_CACHE_FRAMES) end_frame = start_frame + MAX_CACHE_FRAMES;
    a->load_start_frame = start_frame;
    a->load_end_frame = end_frame;
    a->loading = 1;
    a->load_progress = 0;
    a->needs_redraw = 1;
    pthread_mutex_unlock(&a->cache_mutex);
}

static void ensure_cache(SDRAnalyzer *a, int frame_idx) {
    pthread_mutex_lock(&a->cache_mutex);
    if (a->is_scrolling || a->loading) { pthread_mutex_unlock(&a->cache_mutex); return; }
    pthread_mutex_unlock(&a->cache_mutex);
    CacheWindow *existing = find_cache_window(a, frame_idx);
    if (existing && existing->valid) return;
    pthread_mutex_lock(&a->cache_mutex);
    if (a->last_cached_frame >= 0) { int diff = abs(frame_idx - a->last_cached_frame); if (diff < a->cache_frames / 2) { pthread_mutex_unlock(&a->cache_mutex); return; } }
    pthread_mutex_unlock(&a->cache_mutex);
    int new_start = frame_idx - a->cache_frames / 3;
    if (new_start < 0) new_start = 0;
    manage_cache_windows(a, frame_idx);
    request_cache_load(a, new_start);
}

static void update_zoom_display(SDRAnalyzer *a) {
    if (a->zoom_label) { char zoom_text[64]; snprintf(zoom_text, sizeof(zoom_text), "Zoom: %.1fs", a->visible_seconds); gtk_label_set_text(GTK_LABEL(a->zoom_label), zoom_text); }
}

static gboolean ui_update_timer(gpointer user_data) {
    SDRAnalyzer *a = (SDRAnalyzer *)user_data;
    pthread_mutex_lock(&a->cache_mutex);
    int needs_redraw = a->needs_redraw, loading = a->loading, progress = a->load_progress, is_scrolling = a->is_scrolling;
    pthread_mutex_unlock(&a->cache_mutex);
    if (needs_redraw && !is_scrolling) {
        if (a->waterfall_drawing) gtk_widget_queue_draw(a->waterfall_drawing);
        if (loading) { char loading_text[64]; snprintf(loading_text, sizeof(loading_text), "Loading... %d%%", progress); gtk_label_set_text(GTK_LABEL(a->info_label), loading_text); }
        else { char info[512]; snprintf(info, sizeof(info), "Pos: %.2f/%.1f s | Zoom: %.1fs | FFT: %d", a->scroll_position, a->total_duration, a->visible_seconds, a->fft_size); gtk_label_set_text(GTK_LABEL(a->info_label), info); }
        pthread_mutex_lock(&a->cache_mutex);
        a->needs_redraw = 0;
        pthread_mutex_unlock(&a->cache_mutex);
    }
    return G_SOURCE_CONTINUE;
}

static void on_db_min_changed(GtkRange *range, gpointer user_data) {
    SDRAnalyzer *a = (SDRAnalyzer *)user_data;
    a->db_min = gtk_range_get_value(range);
    if (a->db_min >= a->db_max) { a->db_max = a->db_min + 1.0; gtk_range_set_value(GTK_RANGE(a->db_max_slider), a->db_max); }
    a->auto_scale = 0;
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(a->auto_scale_check), FALSE);
    if (a->waterfall_drawing) gtk_widget_queue_draw(a->waterfall_drawing);
}

static void on_db_max_changed(GtkRange *range, gpointer user_data) {
    SDRAnalyzer *a = (SDRAnalyzer *)user_data;
    a->db_max = gtk_range_get_value(range);
    if (a->db_max <= a->db_min) { a->db_min = a->db_max - 1.0; gtk_range_set_value(GTK_RANGE(a->db_min_slider), a->db_min); }
    a->auto_scale = 0;
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(a->auto_scale_check), FALSE);
    if (a->waterfall_drawing) gtk_widget_queue_draw(a->waterfall_drawing);
}

static void on_auto_scale_toggled(GtkToggleButton *toggle, gpointer user_data) {
    SDRAnalyzer *a = (SDRAnalyzer *)user_data;
    a->auto_scale = gtk_toggle_button_get_active(toggle);
    gtk_widget_set_sensitive(a->db_min_slider, !a->auto_scale);
    gtk_widget_set_sensitive(a->db_max_slider, !a->auto_scale);
    if (a->waterfall_drawing) gtk_widget_queue_draw(a->waterfall_drawing);
}

static void on_zoom_in(GtkButton *button, gpointer user_data) {
    SDRAnalyzer *a = (SDRAnalyzer *)user_data;
    double new_visible = a->visible_seconds / 2.0;
    if (new_visible < MIN_VISIBLE_SECONDS) new_visible = MIN_VISIBLE_SECONDS;
    if (new_visible > a->total_duration) new_visible = a->total_duration;
    if (new_visible != a->visible_seconds) {
        a->visible_seconds = new_visible;
        a->visible_frames = (int)(a->visible_seconds * a->frames_per_second);
        a->cache_frames = (int)(a->visible_seconds * a->frames_per_second * 2);
        if (a->cache_frames > MAX_CACHE_FRAMES) a->cache_frames = MAX_CACHE_FRAMES;
        if (a->cache_frames < 100) a->cache_frames = 100;
        update_zoom_display(a);
        int current_frame = (int)(a->scroll_position * a->frames_per_second);
        pthread_mutex_lock(&a->cache_mutex);
        a->last_cached_frame = -1;
        pthread_mutex_unlock(&a->cache_mutex);
        request_cache_load(a, current_frame);
        if (a->waterfall_drawing) gtk_widget_queue_draw(a->waterfall_drawing);
        printf("Zoomed in to %.1f seconds\n", a->visible_seconds);
    }
}

static void on_zoom_out(GtkButton *button, gpointer user_data) {
    SDRAnalyzer *a = (SDRAnalyzer *)user_data;
    double new_visible = a->visible_seconds * 2.0;
    if (new_visible > MAX_VISIBLE_SECONDS) new_visible = MAX_VISIBLE_SECONDS;
    if (new_visible > a->total_duration) new_visible = a->total_duration;
    if (new_visible != a->visible_seconds) {
        a->visible_seconds = new_visible;
        a->visible_frames = (int)(a->visible_seconds * a->frames_per_second);
        a->cache_frames = (int)(a->visible_seconds * a->frames_per_second * 2);
        if (a->cache_frames > MAX_CACHE_FRAMES) a->cache_frames = MAX_CACHE_FRAMES;
        if (a->cache_frames < 100) a->cache_frames = 100;
        update_zoom_display(a);
        int current_frame = (int)(a->scroll_position * a->frames_per_second);
        pthread_mutex_lock(&a->cache_mutex);
        a->last_cached_frame = -1;
        pthread_mutex_unlock(&a->cache_mutex);
        request_cache_load(a, current_frame);
        if (a->waterfall_drawing) gtk_widget_queue_draw(a->waterfall_drawing);
        printf("Zoomed out to %.1f seconds\n", a->visible_seconds);
    }
}

static void on_scroll(SDRAnalyzer *a, double delta) {
    double new_pos = a->scroll_position + delta;
    double max_scroll = a->total_duration - a->visible_seconds;
    if (max_scroll < 0) max_scroll = 0;
    if (new_pos < 0) new_pos = 0;
    if (new_pos > max_scroll) new_pos = max_scroll;
    if (new_pos != a->scroll_position) {
        a->scroll_position = new_pos;
        if (a->waterfall_drawing) gtk_widget_queue_draw(a->waterfall_drawing);
        g_timeout_add(300, on_scroll_timeout, a);
    }
}

static gboolean on_key_press(GtkWidget *widget, GdkEventKey *event, gpointer user_data) {
    SDRAnalyzer *a = (SDRAnalyzer *)user_data;
    switch (event->keyval) {
        case GDK_KEY_Up: on_scroll(a, -SCROLL_SECONDS); break;
        case GDK_KEY_Down: on_scroll(a, SCROLL_SECONDS); break;
        case GDK_KEY_plus: case GDK_KEY_equal: on_zoom_in(NULL, a); break;
        case GDK_KEY_minus: on_zoom_out(NULL, a); break;
        case GDK_KEY_a: case GDK_KEY_A: if (a->selection.active) analyze_selection(a); break;
        case GDK_KEY_q: case GDK_KEY_Q: gtk_main_quit(); break;
    }
    return TRUE;
}

static gboolean on_waterfall_button_press(GtkWidget *widget, GdkEventButton *event, gpointer user_data) {
    SDRAnalyzer *a = (SDRAnalyzer *)user_data;
    if (event->button == 1) {
        int width = gtk_widget_get_allocated_width(widget);
        int height = gtk_widget_get_allocated_height(widget);
        double x = event->x, y = event->y;
        double freq = a->center_freq - a->sample_rate/2 + (x / width) * a->sample_rate;
        double time = a->scroll_position + (1.0 - y/height) * a->visible_seconds;
        a->selection.freq_min = freq; a->selection.freq_max = freq;
        a->selection.time_min = time; a->selection.time_max = time;
        a->selection.active = 1;
        gtk_widget_queue_draw(widget);
    }
    return TRUE;
}

static gboolean on_waterfall_button_release(GtkWidget *widget, GdkEventButton *event, gpointer user_data) {
    SDRAnalyzer *a = (SDRAnalyzer *)user_data;
    if (event->button == 1 && a->selection.active) {
        int width = gtk_widget_get_allocated_width(widget);
        int height = gtk_widget_get_allocated_height(widget);
        double x = event->x, y = event->y;
        double freq = a->center_freq - a->sample_rate/2 + (x / width) * a->sample_rate;
        double time = a->scroll_position + (1.0 - y/height) * a->visible_seconds;
        if (freq < a->selection.freq_min) a->selection.freq_min = freq; else a->selection.freq_max = freq;
        if (time < a->selection.time_min) a->selection.time_min = time; else a->selection.time_max = time;
        printf("Selection: %.3f-%.3f MHz, %.2f-%.2f s\n", a->selection.freq_min/1e6, a->selection.freq_max/1e6, a->selection.time_min, a->selection.time_max);
        gtk_widget_queue_draw(widget);
    }
    return TRUE;
}

static void on_analyze_button(GtkButton *button, gpointer user_data) {
    SDRAnalyzer *a = (SDRAnalyzer *)user_data;
    if (a->selection.active) analyze_selection(a);
    else printf("No selection made. Click and drag on waterfall to select a region.\n");
}

static void on_screenshot_button(GtkButton *button, gpointer user_data) {
    SDRAnalyzer *a = (SDRAnalyzer *)user_data;
    time_t now = time(NULL);
    char filename[256];
    strftime(filename, sizeof(filename), "sdr_waterfall_%Y%m%d_%H%M%S.png", localtime(&now));
    if (a->waterfall_surface) { cairo_surface_write_to_png(a->waterfall_surface, filename); printf("Screenshot saved: %s\n", filename); }
}

static gboolean draw_analysis_popup(GtkWidget *widget, cairo_t *cr, gpointer user_data) {
    SDRAnalyzer *a = (SDRAnalyzer *)user_data;
    int width = gtk_widget_get_allocated_width(widget);
    int height = gtk_widget_get_allocated_height(widget);
    cairo_set_source_rgb(cr, 0.1, 0.1, 0.15);
    cairo_paint(cr);
    if (!a->analysis.waterfall_image || a->analysis.image_frames <= 0) {
        cairo_set_source_rgb(cr, 1.0, 1.0, 1.0);
        cairo_select_font_face(cr, "Sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_NORMAL);
        cairo_set_font_size(cr, 16);
        cairo_move_to(cr, width/2 - 100, height/2);
        cairo_show_text(cr, "No analysis data");
        return FALSE;
    }
    float min_val = 1000.0, max_val = -1000.0;
    for (int i = 0; i < a->analysis.image_frames * a->analysis.image_bins; i++) {
        if (a->analysis.waterfall_image[i] < min_val) min_val = a->analysis.waterfall_image[i];
        if (a->analysis.waterfall_image[i] > max_val) max_val = a->analysis.waterfall_image[i];
    }
    float range = max_val - min_val;
    if (range < 1.0) range = 1.0;
    int img_width = width * 2/3, img_height = height;
    for (int i = 0; i < a->analysis.image_frames && i < img_height; i++) {
        int y = img_height - 1 - i;
        for (int j = 0; j < a->analysis.image_bins && j < img_width; j++) {
            float val = a->analysis.waterfall_image[i * a->analysis.image_bins + j];
            double normalized = (val - min_val) / range;
            if (normalized < 0.0) normalized = 0.0;
            if (normalized > 1.0) normalized = 1.0;
            int color_idx = (int)(normalized * 255.0);
            if (color_idx < 0) color_idx = 0;
            if (color_idx > 255) color_idx = 255;
            Color c = viridis_colormap[color_idx];
            cairo_set_source_rgb(cr, c.r, c.g, c.b);
            cairo_rectangle(cr, j, y, 1, 1);
            cairo_fill(cr);
        }
    }
    return FALSE;
}

static void analyze_selection(SDRAnalyzer *a) {
    if (!a->selection.active) { printf("No selection made\n"); return; }
    printf("\n========================================\nANALYZING SELECTION\n========================================\n");
    printf("Frequency: %.3f - %.3f MHz\nBandwidth: %.2f kHz\nTime: %.2f - %.2f s\nDuration: %.2f s\n",
           a->selection.freq_min/1e6, a->selection.freq_max/1e6, (a->selection.freq_max - a->selection.freq_min)/1e3,
           a->selection.time_min, a->selection.time_max, a->selection.time_max - a->selection.time_min);
    size_t start_sample = (size_t)(a->selection.time_min * a->sample_rate);
    size_t end_sample = (size_t)(a->selection.time_max * a->sample_rate);
    if (start_sample >= a->total_samples) { printf("Error: Selection out of bounds\n"); return; }
    if (end_sample > a->total_samples) end_sample = a->total_samples;
    size_t segment_len = end_sample - start_sample;
    if (segment_len < 8192) { printf("Error: Selection too short\n"); return; }
    float complex *segment = malloc(segment_len * sizeof(float complex));
    if (!segment) return;
    memcpy(segment, &a->iq_data[start_sample], segment_len * sizeof(float complex));
    int fft_size = 2048, hop_size = (int)(fft_size * 0.75);
    int num_frames = (segment_len - fft_size) / hop_size + 1;
    if (num_frames > 200) num_frames = 200;
    double freq_bin_width = a->sample_rate / fft_size;
    int freq_bin_min = (int)((a->selection.freq_min - (a->center_freq - a->sample_rate/2)) / freq_bin_width);
    int freq_bin_max = (int)((a->selection.freq_max - (a->center_freq - a->sample_rate/2)) / freq_bin_width);
    if (freq_bin_min < 0) freq_bin_min = 0;
    if (freq_bin_max > fft_size) freq_bin_max = fft_size;
    int analysis_bins = freq_bin_max - freq_bin_min;
    if (analysis_bins <= 0) analysis_bins = 1;
    a->analysis.freq_min = a->selection.freq_min;
    a->analysis.freq_max = a->selection.freq_max;
    a->analysis.time_min = a->selection.time_min;
    a->analysis.time_max = a->selection.time_max;
    a->analysis.bandwidth = a->selection.freq_max - a->selection.freq_min;
    a->analysis.center_freq = (a->selection.freq_min + a->selection.freq_max) / 2.0;
    a->analysis.duration = a->selection.time_max - a->selection.time_min;
    if (a->analysis.waterfall_image) free(a->analysis.waterfall_image);
    a->analysis.waterfall_image = malloc(num_frames * analysis_bins * sizeof(float));
    a->analysis.image_frames = num_frames;
    a->analysis.image_bins = analysis_bins;
    fftwf_complex *in = fftwf_malloc(sizeof(fftwf_complex) * fft_size);
    fftwf_complex *out = fftwf_malloc(sizeof(fftwf_complex) * fft_size);
    fftwf_plan plan = fftwf_plan_dft_1d(fft_size, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    float *window = malloc(fft_size * sizeof(float));
    for (int i = 0; i < fft_size; i++) window[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (fft_size - 1)));
    float sum = 0.0, max_val = -1000.0;
    for (int i = 0; i < num_frames; i++) {
        size_t idx = i * hop_size;
        for (int j = 0; j < fft_size && idx + j < segment_len; j++) {
            in[j][0] = crealf(segment[idx + j]) * window[j];
            in[j][1] = cimagf(segment[idx + j]) * window[j];
        }
        fftwf_execute(plan);
        for (int j = 0; j < analysis_bins; j++) {
            int shifted_idx = (freq_bin_min + j + fft_size / 2) % fft_size;
            double mag = sqrt(out[shifted_idx][0] * out[shifted_idx][0] + out[shifted_idx][1] * out[shifted_idx][1]);
            float val = 20.0 * log10(mag + 1e-10);
            a->analysis.waterfall_image[i * analysis_bins + j] = val;
            sum += val;
            if (val > max_val) max_val = val;
        }
    }
    fftwf_destroy_plan(plan);
    fftwf_free(in); fftwf_free(out); free(window);
    free(segment);
    a->analysis.avg_power = sum / (num_frames * analysis_bins);
    a->analysis.max_power = max_val;
    a->analysis.entropy = calculate_entropy(a->analysis.waterfall_image, num_frames * analysis_bins);
    double time_per_frame = hop_size / a->sample_rate;
    a->analysis.prf = calculate_prf(a->analysis.waterfall_image, num_frames, time_per_frame);
    if (a->analysis.bandwidth < 5000) strcpy(a->analysis.signal_type, "Narrowband");
    else if (a->analysis.bandwidth < 1000000) strcpy(a->analysis.signal_type, "Wideband");
    else strcpy(a->analysis.signal_type, "Very Wideband (OFDM)");
    if (a->analysis.bandwidth < 3000) strcpy(a->analysis.modulation, "AM/SSB/CW");
    else if (a->analysis.bandwidth < 15000) strcpy(a->analysis.modulation, "FM/NFM");
    else if (a->analysis.bandwidth < 500000) strcpy(a->analysis.modulation, "QAM/OFDM");
    else strcpy(a->analysis.modulation, "Wideband OFDM");
    match_known_signals(a, a->analysis.center_freq, a->analysis.bandwidth);
    printf("Entropy: %.4f bits\nPRF: %.2f Hz\nSignal Type: %s\nModulation: %s\n", a->analysis.entropy, a->analysis.prf, a->analysis.signal_type, a->analysis.modulation);
    if (a->analysis.match_count > 0) {
        printf("\nKnown Signal Matches:\n");
        for (int i = 0; i < a->analysis.match_count && i < 5; i++) printf("  %d. %s (%d%%)\n", i+1, a->analysis.matches[i], a->analysis.match_confidence[i]);
    }
    printf("========================================\n\n");
    if (a->popup_window) gtk_widget_destroy(a->popup_window);
    a->popup_window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_title(GTK_WINDOW(a->popup_window), "Signal Analysis Results");
    gtk_window_set_default_size(GTK_WINDOW(a->popup_window), 1000, 700);
    gtk_window_set_transient_for(GTK_WINDOW(a->popup_window), GTK_WINDOW(a->window));
    GtkWidget *vbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
    gtk_container_add(GTK_CONTAINER(a->popup_window), vbox);
    gtk_widget_set_margin_start(vbox, 10);
    gtk_widget_set_margin_end(vbox, 10);
    gtk_widget_set_margin_top(vbox, 10);
    gtk_widget_set_margin_bottom(vbox, 10);
    GtkWidget *image_frame = gtk_frame_new("Selected Region");
    GtkWidget *image_drawing = gtk_drawing_area_new();
    gtk_widget_set_size_request(image_drawing, 600, 400);
    g_signal_connect(image_drawing, "draw", G_CALLBACK(draw_analysis_popup), a);
    gtk_container_add(GTK_CONTAINER(image_frame), image_drawing);
    gtk_box_pack_start(GTK_BOX(vbox), image_frame, TRUE, TRUE, 0);
    GtkWidget *stats_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    gtk_widget_set_size_request(stats_box, 350, -1);
    char stats_text[2048];
    snprintf(stats_text, sizeof(stats_text), "<b>Selection Information</b>\nFrequency: %.3f - %.3f MHz\nBandwidth: %.2f kHz\nCenter Frequency: %.6f MHz\nTime: %.2f - %.2f s\nDuration: %.2f seconds\n\n<b>Signal Statistics</b>\nEntropy: %.4f bits\nPulse Repetition Frequency: %.2f Hz\nAverage Power: %.2f dB\nPeak Power: %.2f dB\n\n<b>Signal Characterization</b>\nSignal Type: %s\nModulation: %s\n\n<b>Known Signal Matches</b>\n",
        a->analysis.freq_min/1e6, a->analysis.freq_max/1e6, a->analysis.bandwidth/1e3, a->analysis.center_freq/1e6,
        a->analysis.time_min, a->analysis.time_max, a->analysis.duration, a->analysis.entropy, a->analysis.prf,
        a->analysis.avg_power, a->analysis.max_power, a->analysis.signal_type, a->analysis.modulation);
    GtkWidget *stats_label = gtk_label_new(NULL);
    gtk_label_set_markup(GTK_LABEL(stats_label), stats_text);
    gtk_label_set_xalign(GTK_LABEL(stats_label), 0.0);
    gtk_box_pack_start(GTK_BOX(stats_box), stats_label, FALSE, FALSE, 0);
    for (int i = 0; i < a->analysis.match_count && i < 5; i++) {
        char match_text[128];
        snprintf(match_text, sizeof(match_text), "%d. %s (%d%%)", i+1, a->analysis.matches[i], a->analysis.match_confidence[i]);
        GtkWidget *match_label = gtk_label_new(match_text);
        gtk_label_set_xalign(GTK_LABEL(match_label), 0.0);
        gtk_box_pack_start(GTK_BOX(stats_box), match_label, FALSE, FALSE, 0);
    }
    gtk_box_pack_start(GTK_BOX(vbox), stats_box, FALSE, FALSE, 0);
    GtkWidget *close_btn = gtk_button_new_with_label("Close");
    g_signal_connect_swapped(close_btn, "clicked", G_CALLBACK(gtk_widget_destroy), a->popup_window);
    gtk_box_pack_start(GTK_BOX(stats_box), close_btn, FALSE, FALSE, 10);
    gtk_widget_show_all(a->popup_window);
}

static gboolean draw_waterfall(GtkWidget *widget, cairo_t *cr, gpointer user_data) {
    SDRAnalyzer *a = (SDRAnalyzer *)user_data;
    
    int width = gtk_widget_get_allocated_width(widget);
    int height = gtk_widget_get_allocated_height(widget);
    
    if (width <= 0 || height <= 0) return FALSE;
    
    if (a->waterfall_surface) {
        cairo_surface_destroy(a->waterfall_surface);
    }
    a->waterfall_surface = cairo_image_surface_create(CAIRO_FORMAT_RGB24, width, height);
    cairo_t *img_cr = cairo_create(a->waterfall_surface);
    
    cairo_set_source_rgb(img_cr, 0.0, 0.0, 0.0);
    cairo_paint(img_cr);
    
    // Calculate which frames to display based on zoom level (visible_seconds)
    int start_frame = (int)(a->scroll_position * a->frames_per_second);
    int end_frame = (int)((a->scroll_position + a->visible_seconds) * a->frames_per_second);
    if (end_frame > a->total_frames) end_frame = a->total_frames;
    
    // Ensure we have enough frames for the visible time
    int frames_to_show = (int)(a->visible_seconds * a->frames_per_second);
    if (end_frame - start_frame < frames_to_show) {
        start_frame = a->total_frames - frames_to_show;
        if (start_frame < 0) start_frame = 0;
        end_frame = start_frame + frames_to_show;
        if (end_frame > a->total_frames) end_frame = a->total_frames;
    }
    
    CacheWindow *cw = find_cache_window(a, start_frame);
    if (!cw || !cw->valid || !cw->data) {
        cairo_destroy(img_cr);
        cairo_set_source_surface(cr, a->waterfall_surface, 0, 0);
        cairo_paint(cr);
        return FALSE;
    }
    
    pthread_mutex_lock(&cw->mutex);
    int cache_start = cw->start_frame;
    int cache_end = cw->end_frame;
    pthread_mutex_unlock(&cw->mutex);
    
    int vis_start = start_frame > cache_start ? start_frame : cache_start;
    int vis_end = end_frame < cache_end ? end_frame : cache_end;
    
    if (vis_end <= vis_start) {
        cairo_destroy(img_cr);
        cairo_set_source_surface(cr, a->waterfall_surface, 0, 0);
        cairo_paint(cr);
        return FALSE;
    }
    
    // Calculate dB range
    float display_min = 1000.0, display_max = -1000.0;
    
    if (a->auto_scale) {
        pthread_mutex_lock(&cw->mutex);
        for (int i = vis_start; i < vis_end; i++) {
            int cache_idx = i - cache_start;
            for (int j = 0; j < a->fft_size; j++) {
                float val = cw->data[cache_idx * a->fft_size + j];
                if (val < display_min) display_min = val;
                if (val > display_max) display_max = val;
            }
        }
        pthread_mutex_unlock(&cw->mutex);
    } else {
        display_min = a->db_min;
        display_max = a->db_max;
    }
    
    float range = display_max - display_min;
    if (range < 1.0) range = 1.0;
    
    int actual_frames = vis_end - vis_start;
    
    // FIXED: Calculate frame height to fill the display
    double frame_height = (double)height / actual_frames;
    if (frame_height < 1.0) frame_height = 1.0;
    
    // Draw each frame
    for (int i = 0; i < actual_frames; i++) {
        int frame_idx = vis_start + i;
        int cache_idx = frame_idx - cache_start;
        
        // Calculate Y position (newest at top, oldest at bottom)
        int y_start = (int)((actual_frames - i - 1) * frame_height);
        int y_end = (int)((actual_frames - i) * frame_height);
        if (y_end <= y_start) y_end = y_start + 1;
        if (y_end > height) y_end = height;
        if (y_start < 0) y_start = 0;
        
        pthread_mutex_lock(&cw->mutex);
        float *frame_data = &cw->data[cache_idx * a->fft_size];
        pthread_mutex_unlock(&cw->mutex);
        
        // Draw each frequency bin
        for (int j = 0; j < a->fft_size; j++) {
            float val = frame_data[j];
            double normalized = (val - display_min) / range;
            if (normalized < 0.0) normalized = 0.0;
            if (normalized > 1.0) normalized = 1.0;
            
            int color_idx = (int)(normalized * 255.0);
            if (color_idx < 0) color_idx = 0;
            if (color_idx > 255) color_idx = 255;
            
            Color c = viridis_colormap[color_idx];
            
            int x_start = (int)((double)j * width / a->fft_size);
            int x_end = (int)((double)(j + 1) * width / a->fft_size);
            if (x_end <= x_start) x_end = x_start + 1;
            
            cairo_set_source_rgb(img_cr, c.r, c.g, c.b);
            cairo_rectangle(img_cr, x_start, y_start, x_end - x_start, y_end - y_start);
            cairo_fill(img_cr);
        }
    }
    
    // Draw selection box
    if (a->selection.active) {
        double view_start_time = a->scroll_position;
        double view_end_time = a->scroll_position + a->visible_seconds;
        
        if (a->selection.time_max > view_start_time && 
            a->selection.time_min < view_end_time) {
            
            int sel_y_start = (int)((view_end_time - fmax(a->selection.time_max, view_start_time)) / 
                                    (view_end_time - view_start_time) * height);
            int sel_y_end = (int)((view_end_time - fmin(a->selection.time_min, view_end_time)) / 
                                  (view_end_time - view_start_time) * height);
            
            int sel_x_start = (int)((a->selection.freq_min - (a->center_freq - a->sample_rate/2)) / 
                                    a->sample_rate * width);
            int sel_x_end = (int)((a->selection.freq_max - (a->center_freq - a->sample_rate/2)) / 
                                  a->sample_rate * width);
            
            if (sel_y_start < 0) sel_y_start = 0;
            if (sel_y_end > height) sel_y_end = height;
            if (sel_x_start < 0) sel_x_start = 0;
            if (sel_x_end > width) sel_x_end = width;
            
            if (sel_y_end > sel_y_start && sel_x_end > sel_x_start) {
                cairo_set_source_rgba(img_cr, 1.0, 0.0, 0.0, 0.3);
                cairo_rectangle(img_cr, sel_x_start, sel_y_start, 
                               sel_x_end - sel_x_start, sel_y_end - sel_y_start);
                cairo_fill(img_cr);
                
                cairo_set_source_rgb(img_cr, 1.0, 0.0, 0.0);
                cairo_set_line_width(img_cr, 2.0);
                cairo_rectangle(img_cr, sel_x_start, sel_y_start, 
                               sel_x_end - sel_x_start, sel_y_end - sel_y_start);
                cairo_stroke(img_cr);
            }
        }
    }
    
    cairo_destroy(img_cr);
    cairo_set_source_surface(cr, a->waterfall_surface, 0, 0);
    cairo_paint(cr);
    
    return FALSE;
}

int main(int argc, char *argv[]) {
    if (argc < 2) { printf("Usage: %s <gqrx_raw_file>\n", argv[0]); return 1; }
    init_viridis_colormap();
    SDRAnalyzer analyzer = {0};
    analyzer.filepath = argv[1];
    analyzer.fft_size = FFT_SIZE;
    analyzer.fft_overlap = FFT_OVERLAP;
    analyzer.visible_seconds = DEFAULT_VISIBLE_SECONDS;
    analyzer.scroll_seconds = SCROLL_SECONDS;
    analyzer.cache_frames = (int)(CACHE_SECONDS * 256);
    if (analyzer.cache_frames > MAX_CACHE_FRAMES) analyzer.cache_frames = MAX_CACHE_FRAMES;
    analyzer.db_min = DEFAULT_DB_MIN;
    analyzer.db_max = DEFAULT_DB_MAX;
    analyzer.auto_scale = 1;
    pthread_mutex_init(&analyzer.cache_mutex, NULL);
    init_cache_windows(&analyzer);
    if (!parse_filename(&analyzer)) return 1;
    if (load_iq_file(&analyzer) < 0) return 1;
    int hop_size = (int)(analyzer.fft_size * (1.0 - analyzer.fft_overlap));
    analyzer.frames_per_second = analyzer.sample_rate / hop_size;
    analyzer.total_frames = (int)(analyzer.total_duration * analyzer.frames_per_second);
    analyzer.time_per_frame = 1.0 / analyzer.frames_per_second;
    analyzer.visible_frames = (int)(analyzer.visible_seconds * analyzer.frames_per_second);
    if (analyzer.visible_seconds > analyzer.total_duration) analyzer.visible_seconds = analyzer.total_duration;
    if (analyzer.visible_seconds > MAX_VISIBLE_SECONDS) analyzer.visible_seconds = MAX_VISIBLE_SECONDS;
    printf("\n========================================\nSDR Waterfall with Zoom Controls\n========================================\n");
    printf("FFT Size: %d bins\nDefault Zoom: %.1f seconds\nControls:\n  Up/Down    : Scroll 1 second\n  +/-        : Zoom in/out\n  Mouse drag : Select region\n  A          : Analyze selection\n========================================\n\n");
    pthread_create(&analyzer.load_thread, NULL, cache_load_thread, &analyzer);
    request_cache_load(&analyzer, 0);
    gtk_init(&argc, &argv);
    analyzer.window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_title(GTK_WINDOW(analyzer.window), "SDR Waterfall (with Zoom)");
    gtk_window_set_default_size(GTK_WINDOW(analyzer.window), 1400, 900);
    g_signal_connect(analyzer.window, "destroy", G_CALLBACK(gtk_main_quit), NULL);
    g_signal_connect(analyzer.window, "key-press-event", G_CALLBACK(on_key_press), &analyzer);
    GtkWidget *vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    gtk_container_add(GTK_CONTAINER(analyzer.window), vbox);
    analyzer.waterfall_drawing = gtk_drawing_area_new();
    gtk_widget_set_size_request(analyzer.waterfall_drawing, -1, 600);
    gtk_widget_set_hexpand(analyzer.waterfall_drawing, TRUE);
    gtk_widget_set_vexpand(analyzer.waterfall_drawing, TRUE);
    g_signal_connect(analyzer.waterfall_drawing, "draw", G_CALLBACK(draw_waterfall), &analyzer);
    g_signal_connect(analyzer.waterfall_drawing, "button-press-event", G_CALLBACK(on_waterfall_button_press), &analyzer);
    g_signal_connect(analyzer.waterfall_drawing, "button-release-event", G_CALLBACK(on_waterfall_button_release), &analyzer);
    gtk_widget_add_events(analyzer.waterfall_drawing, GDK_BUTTON_PRESS_MASK | GDK_BUTTON_RELEASE_MASK);
    gtk_box_pack_start(GTK_BOX(vbox), analyzer.waterfall_drawing, TRUE, TRUE, 0);
    GtkWidget *control_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    gtk_widget_set_margin_start(control_box, 10);
    gtk_widget_set_margin_end(control_box, 10);
    gtk_widget_set_margin_top(control_box, 5);
    gtk_widget_set_margin_bottom(control_box, 5);
    analyzer.analyze_btn = gtk_button_new_with_label("Analyze (A)");
    g_signal_connect(analyzer.analyze_btn, "clicked", G_CALLBACK(on_analyze_button), &analyzer);
    gtk_box_pack_start(GTK_BOX(control_box), analyzer.analyze_btn, FALSE, FALSE, 5);
    analyzer.screenshot_btn = gtk_button_new_with_label("Screenshot (S)");
    g_signal_connect(analyzer.screenshot_btn, "clicked", G_CALLBACK(on_screenshot_button), &analyzer);
    gtk_box_pack_start(GTK_BOX(control_box), analyzer.screenshot_btn, FALSE, FALSE, 5);
    GtkWidget *zoom_in_btn = gtk_button_new_with_label("Zoom In (+)");
    g_signal_connect(zoom_in_btn, "clicked", G_CALLBACK(on_zoom_in), &analyzer);
    gtk_box_pack_start(GTK_BOX(control_box), zoom_in_btn, FALSE, FALSE, 5);
    GtkWidget *zoom_out_btn = gtk_button_new_with_label("Zoom Out (-)");
    g_signal_connect(zoom_out_btn, "clicked", G_CALLBACK(on_zoom_out), &analyzer);
    gtk_box_pack_start(GTK_BOX(control_box), zoom_out_btn, FALSE, FALSE, 5);
    analyzer.zoom_label = gtk_label_new("");
    update_zoom_display(&analyzer);
    gtk_box_pack_start(GTK_BOX(control_box), analyzer.zoom_label, FALSE, FALSE, 10);
    GtkWidget *db_min_label = gtk_label_new("Min dB:");
    gtk_box_pack_start(GTK_BOX(control_box), db_min_label, FALSE, FALSE, 5);
    analyzer.db_min_slider = gtk_scale_new_with_range(GTK_ORIENTATION_HORIZONTAL, -150.0, 0.0, 1.0);
    gtk_range_set_value(GTK_RANGE(analyzer.db_min_slider), DEFAULT_DB_MIN);
    gtk_widget_set_size_request(analyzer.db_min_slider, 120, -1);
    g_signal_connect(analyzer.db_min_slider, "value-changed", G_CALLBACK(on_db_min_changed), &analyzer);
    gtk_box_pack_start(GTK_BOX(control_box), analyzer.db_min_slider, FALSE, FALSE, 5);
    GtkWidget *db_max_label = gtk_label_new("Max dB:");
    gtk_box_pack_start(GTK_BOX(control_box), db_max_label, FALSE, FALSE, 5);
    analyzer.db_max_slider = gtk_scale_new_with_range(GTK_ORIENTATION_HORIZONTAL, -150.0, 0.0, 1.0);
    gtk_range_set_value(GTK_RANGE(analyzer.db_max_slider), DEFAULT_DB_MAX);
    gtk_widget_set_size_request(analyzer.db_max_slider, 120, -1);
    g_signal_connect(analyzer.db_max_slider, "value-changed", G_CALLBACK(on_db_max_changed), &analyzer);
    gtk_box_pack_start(GTK_BOX(control_box), analyzer.db_max_slider, FALSE, FALSE, 5);
    analyzer.auto_scale_check = gtk_check_button_new_with_label("Auto");
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(analyzer.auto_scale_check), TRUE);
    g_signal_connect(analyzer.auto_scale_check, "toggled", G_CALLBACK(on_auto_scale_toggled), &analyzer);
    gtk_box_pack_start(GTK_BOX(control_box), analyzer.auto_scale_check, FALSE, FALSE, 5);
    gtk_box_pack_start(GTK_BOX(vbox), control_box, FALSE, FALSE, 0);
    analyzer.info_label = gtk_label_new("Loading...");
    gtk_widget_set_halign(analyzer.info_label, GTK_ALIGN_START);
    gtk_widget_set_margin_start(analyzer.info_label, 10);
    gtk_widget_set_margin_bottom(analyzer.info_label, 5);
    gtk_box_pack_start(GTK_BOX(vbox), analyzer.info_label, FALSE, FALSE, 0);
    printf("Showing all widgets...\n");
    gtk_widget_show_all(analyzer.window);
    printf("Widgets shown\n");
    gtk_widget_set_sensitive(analyzer.db_min_slider, FALSE);
    gtk_widget_set_sensitive(analyzer.db_max_slider, FALSE);
    g_timeout_add(50, ui_update_timer, &analyzer);
    printf("Ready!\n");
    gtk_main();
    analyzer.quit_thread = 1;
    pthread_join(analyzer.load_thread, NULL);
    munmap(analyzer.iq_data, analyzer.file_size);
    close(analyzer.fd);
    for (int i = 0; i < analyzer.num_cache_windows; i++) { free_cache_window(&analyzer.cache_windows[i]); pthread_mutex_destroy(&analyzer.cache_windows[i].mutex); }
    if (analyzer.waterfall_surface) cairo_surface_destroy(analyzer.waterfall_surface);
    if (analyzer.analysis.waterfall_image) free(analyzer.analysis.waterfall_image);
    pthread_mutex_destroy(&analyzer.cache_mutex);
    return 0;
}