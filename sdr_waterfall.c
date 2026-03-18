/*
 * SDR IQ Waterfall Analyzer - OpenGL + Smart Caching
 * 
 * Features:
 * - OpenGL hardware-accelerated rendering
 * - Multi-window pre-caching
 * - Automatic memory unloading
 * - GPU color mapping
 * 
 * Compile: gcc -o sdr_waterfall sdr_waterfall.c \
 *          $(pkg-config --cflags --libs gtk+-3.0 gl) \
 *          -I/opt/homebrew/include -L/opt/homebrew/lib \
 *          -lfftw3f -lGL -lpthread -lm -O3 -ffast-math
 */

#include <gtk/gtk.h>
#include <gtk/gtkglarea.h>
#include <GL/gl.h>
#include <fftw3.h>
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

// ============================================================================
// Configuration
// ============================================================================
#define FFT_SIZE 8192
#define FFT_OVERLAP 0.50
#define VISIBLE_SECONDS 5.0
#define SCROLL_SECONDS 0.1
#define CACHE_SECONDS 6.0
#define PRECACHE_WINDOWS 3  // Number of cache windows to keep loaded
#define MAX_CACHE_WINDOWS 5 // Maximum cache windows in memory
#define LOAD_BATCH_SIZE 100

#define DEFAULT_DB_MIN -100.0
#define DEFAULT_DB_MAX -20.0

// ============================================================================
// Viridis Colormap
// ============================================================================
typedef struct {
    float r, g, b;
} Color;

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

// ============================================================================
// Cache Window Structure
// ============================================================================
typedef struct {
    int start_frame;
    int end_frame;
    float *data;
    int valid;
    int last_access;
    int ref_count;
} CacheWindow;

// ============================================================================
// Data Structures
// ============================================================================
typedef struct {
    char name[64];
    double freq_min;
    double freq_max;
    double typical_bw;
    double bw_min;
    double bw_max;
    char use[128];
} KnownSignal;

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
    
    // Cache management
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
    
    double db_min;
    double db_max;
    int auto_scale;
    
    double sel_freq_min;
    double sel_freq_max;
    double sel_time_min;
    double sel_time_max;
    int has_selection;
    
    AnalysisResults analysis;
    float *analysis_image;
    int analysis_frames;
    int analysis_bins;
    
    // OpenGL
    GLuint texture_id;
    GLuint shader_program;
    GLuint vao;
    GLuint vbo;
    int gl_initialized;
    int texture_width;
    int texture_height;
    
    // GUI
    GtkWidget *window;
    GtkWidget *gl_area;
    GtkWidget *spectrum_drawing;
    GtkWidget *info_label;
    GtkWidget *popup_window;
    GtkWidget *db_min_slider;
    GtkWidget *db_max_slider;
    GtkWidget *auto_scale_check;
    
    // Threading
    pthread_mutex_t cache_mutex;
    pthread_t load_thread;
    int loading;
    int load_start_frame;
    int load_end_frame;
    int load_progress;
    int quit_thread;
    int needs_redraw;
    int current_cache_window;
    
} SDRAnalyzer;

// ============================================================================
// Forward Declarations
// ============================================================================
static void show_analysis_popup(SDRAnalyzer *a);
static void on_db_min_changed(GtkRange *range, gpointer user_data);
static void on_db_max_changed(GtkRange *range, gpointer user_data);
static void on_auto_scale_toggled(GtkCheckButton *check, gpointer user_data);
static void* cache_load_thread(void *arg);
static void manage_cache_windows(SDRAnalyzer *a, int target_frame);
static int find_or_create_cache_window(SDRAnalyzer *a, int start_frame);

// ============================================================================
// Known Signal Database
// ============================================================================
static KnownSignal known_signals_db[] = {
    {"5G NR Downlink", 2300e6, 2690e6, 20e6, 5e6, 100e6, "5G cellular downlink"},
    {"5G NR Uplink", 2300e6, 2690e6, 20e6, 5e6, 100e6, "5G cellular uplink"},
    {"4G LTE Downlink", 2300e6, 2690e6, 20e6, 1.4e6, 20e6, "4G cellular downlink"},
    {"4G LTE Uplink", 2300e6, 2690e6, 20e6, 1.4e6, 20e6, "4G cellular uplink"},
    {"WiFi 802.11n/g", 2400e6, 2483.5e6, 20e6, 20e6, 40e6, "Wireless LAN"},
    {"WiFi 802.11ac/ax", 2400e6, 2483.5e6, 80e6, 20e6, 160e6, "Wireless LAN high speed"},
    {"Logitech Wireless", 2400e6, 2483.5e6, 1e6, 500e3, 2e6, "Peripheral wireless"},
    {"Zigbee", 2400e6, 2483.5e6, 2e6, 2e6, 2e6, "IoT home automation"},
    {"Bluetooth Classic", 2400e6, 2483.5e6, 1e6, 1e6, 3e6, "Short-range audio"},
    {"Bluetooth LE", 2400e6, 2483.5e6, 2e6, 1e6, 2e6, "IoT wearables"},
};

static const int num_known_signals = sizeof(known_signals_db) / sizeof(KnownSignal);

// ============================================================================
// Cache Management
// ============================================================================
static void init_cache_windows(SDRAnalyzer *a) {
    a->num_cache_windows = 0;
    for (int i = 0; i < MAX_CACHE_WINDOWS; i++) {
        a->cache_windows[i].valid = 0;
        a->cache_windows[i].data = NULL;
        a->cache_windows[i].ref_count = 0;
        a->cache_windows[i].last_access = 0;
    }
}

static void free_cache_window(CacheWindow *cw) {
    if (cw->data) {
        free(cw->data);
        cw->data = NULL;
    }
    cw->valid = 0;
    cw->start_frame = 0;
    cw->end_frame = 0;
    cw->ref_count = 0;
}

static void manage_cache_windows(SDRAnalyzer *a, int target_frame) {
    pthread_mutex_lock(&a->cache_mutex);
    
    // Find cache window closest to target
    int closest_idx = -1;
    int closest_dist = INT_MAX;
    
    for (int i = 0; i < a->num_cache_windows; i++) {
        if (!a->cache_windows[i].valid) continue;
        
        int mid_frame = (a->cache_windows[i].start_frame + a->cache_windows[i].end_frame) / 2;
        int dist = abs(mid_frame - target_frame);
        
        if (dist < closest_dist) {
            closest_dist = dist;
            closest_idx = i;
        }
    }
    
    // Unload windows that are too far away
    int frames_per_window = a->cache_frames;
    int max_distance = frames_per_window * 2; // Keep windows within 2x cache size
    
    for (int i = a->num_cache_windows - 1; i >= 0; i--) {
        if (!a->cache_windows[i].valid) continue;
        if (a->cache_windows[i].ref_count > 0) continue; // In use
        
        int mid_frame = (a->cache_windows[i].start_frame + a->cache_windows[i].end_frame) / 2;
        int dist = abs(mid_frame - target_frame);
        
        if (dist > max_distance && a->num_cache_windows > PRECACHE_WINDOWS) {
            printf("  Unloading cache window %d-%d (distance: %d)\n", 
                   a->cache_windows[i].start_frame, 
                   a->cache_windows[i].end_frame,
                   dist);
            free_cache_window(&a->cache_windows[i]);
            
            // Remove from array
            for (int j = i; j < a->num_cache_windows - 1; j++) {
                a->cache_windows[j] = a->cache_windows[j + 1];
            }
            a->num_cache_windows--;
        }
    }
    
    pthread_mutex_unlock(&a->cache_mutex);
}

static int find_or_create_cache_window(SDRAnalyzer *a, int start_frame) {
    // Check if we already have this window
    for (int i = 0; i < a->num_cache_windows; i++) {
        if (a->cache_windows[i].valid && 
            a->cache_windows[i].start_frame == start_frame) {
            a->cache_windows[i].last_access = time(NULL);
            return i;
        }
    }
    
    // Need to create new window
    if (a->num_cache_windows >= MAX_CACHE_WINDOWS) {
        // Find least recently used
        int lru_idx = 0;
        time_t lru_time = a->cache_windows[0].last_access;
        
        for (int i = 1; i < a->num_cache_windows; i++) {
            if (a->cache_windows[i].ref_count == 0 && 
                a->cache_windows[i].last_access < lru_time) {
                lru_time = a->cache_windows[i].last_access;
                lru_idx = i;
            }
        }
        
        free_cache_window(&a->cache_windows[lru_idx]);
        return lru_idx;
    }
    
    // Use next slot
    int idx = a->num_cache_windows++;
    a->cache_windows[idx].valid = 0;
    a->cache_windows[idx].data = NULL;
    a->cache_windows[idx].ref_count = 0;
    
    return idx;
}

// ============================================================================
// Background Cache Loading Thread
// ============================================================================
static void* cache_load_thread(void *arg) {
    SDRAnalyzer *a = (SDRAnalyzer *)arg;
    
    fftwf_complex *in = fftwf_malloc(sizeof(fftwf_complex) * a->fft_size);
    fftwf_complex *out = fftwf_malloc(sizeof(fftwf_complex) * a->fft_size);
    fftwf_plan plan = fftwf_plan_dft_1d(a->fft_size, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    
    float *window = malloc(a->fft_size * sizeof(float));
    for (int i = 0; i < a->fft_size; i++) {
        window[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (a->fft_size - 1)));
    }
    
    int hop_size = (int)(a->fft_size * (1.0 - a->fft_overlap));
    
    while (!a->quit_thread) {
        pthread_mutex_lock(&a->cache_mutex);
        while (!a->loading && !a->quit_thread) {
            pthread_mutex_unlock(&a->cache_mutex);
            usleep(10000);
            pthread_mutex_lock(&a->cache_mutex);
        }
        
        if (a->quit_thread) {
            pthread_mutex_unlock(&a->cache_mutex);
            break;
        }
        
        int start_frame = a->load_start_frame;
        int end_frame = a->load_end_frame;
        pthread_mutex_unlock(&a->cache_mutex);
        
        printf("  Loading cache: frames %d to %d...\n", start_frame, end_frame);
        
        // Find or create cache window
        int cw_idx = find_or_create_cache_window(a, start_frame);
        
        pthread_mutex_lock(&a->cache_mutex);
        CacheWindow *cw = &a->cache_windows[cw_idx];
        cw->start_frame = start_frame;
        cw->end_frame = end_frame;
        cw->ref_count = 1;
        pthread_mutex_unlock(&a->cache_mutex);
        
        // Allocate memory
        int num_frames = end_frame - start_frame;
        size_t data_size = num_frames * a->fft_size * sizeof(float);
        cw->data = malloc(data_size);
        
        if (!cw->data) {
            printf("  Error: Failed to allocate cache memory\n");
            pthread_mutex_lock(&a->cache_mutex);
            cw->valid = 0;
            cw->ref_count = 0;
            pthread_mutex_unlock(&a->cache_mutex);
            continue;
        }
        
        int total_frames = num_frames;
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
                double mag = sqrt(out[shifted_idx][0] * out[shifted_idx][0] + 
                                 out[shifted_idx][1] * out[shifted_idx][1]);
                cw->data[cache_idx * a->fft_size + j] = 20.0 * log10(mag + 1e-10);
            }
            
            frames_loaded++;
            
            if (frames_loaded % LOAD_BATCH_SIZE == 0) {
                pthread_mutex_lock(&a->cache_mutex);
                a->load_progress = (frames_loaded * 100) / total_frames;
                a->needs_redraw = 1;
                pthread_mutex_unlock(&a->cache_mutex);
            }
        }
        
        pthread_mutex_lock(&a->cache_mutex);
        cw->valid = 1;
        cw->last_access = time(NULL);
        a->loading = 0;
        a->load_progress = 100;
        a->needs_redraw = 1;
        cw->ref_count = 0;
        pthread_mutex_unlock(&a->cache_mutex);
        
        printf("  Cache loaded (%d frames, %.1f MB).\n", 
               frames_loaded, frames_loaded * a->fft_size * sizeof(float) / 1e6);
    }
    
    fftwf_destroy_plan(plan);
    fftwf_free(in);
    fftwf_free(out);
    free(window);
    
    return NULL;
}

// ============================================================================
// OpenGL Setup
// ============================================================================
static const char *vertex_shader_source =
    "#version 330 core\n"
    "layout (location = 0) in vec2 aPos;\n"
    "layout (location = 1) in vec2 aTexCoord;\n"
    "out vec2 TexCoord;\n"
    "void main() {\n"
    "    gl_Position = vec4(aPos, 0.0, 1.0);\n"
    "    TexCoord = aTexCoord;\n"
    "}\n";

static const char *fragment_shader_source =
    "#version 330 core\n"
    "in vec2 TexCoord;\n"
    "out vec4 FragColor;\n"
    "uniform sampler2D texture1;\n"
    "uniform float db_min;\n"
    "uniform float db_max;\n"
    "\n"
    "vec3 viridis(float t) {\n"
    "    t = clamp(t, 0.0, 1.0);\n"
    "    if (t < 0.25) {\n"
    "        float x = t * 4.0;\n"
    "        return vec3(0.267 + (0.118 - 0.267) * x,\n"
    "                   0.005 + (0.384 - 0.005) * x,\n"
    "                   0.329 + (0.468 - 0.329) * x);\n"
    "    } else if (t < 0.5) {\n"
    "        float x = (t - 0.25) * 4.0;\n"
    "        return vec3(0.118 + (0.194 - 0.118) * x,\n"
    "                   0.384 + (0.718 - 0.384) * x,\n"
    "                   0.468 + (0.482 - 0.468) * x);\n"
    "    } else if (t < 0.75) {\n"
    "        float x = (t - 0.5) * 4.0;\n"
    "        return vec3(0.194 + (0.478 - 0.194) * x,\n"
    "                   0.718 + (0.878 - 0.718) * x,\n"
    "                   0.482 + (0.314 - 0.482) * x);\n"
    "    } else {\n"
    "        float x = (t - 0.75) * 4.0;\n"
    "        return vec3(0.478 + (0.993 - 0.478) * x,\n"
    "                   0.878 + (0.988 - 0.878) * x,\n"
    "                   0.314 + (0.090 - 0.314) * x);\n"
    "    }\n"
    "}\n"
    "\n"
    "void main() {\n"
    "    float val = texture(texture1, TexCoord).r;\n"
    "    float normalized = (val - db_min) / (db_max - db_min);\n"
    "    normalized = clamp(normalized, 0.0, 1.0);\n"
    "    vec3 color = viridis(normalized);\n"
    "    FragColor = vec4(color, 1.0);\n"
    "}\n";

static GLuint compile_shader(GLenum type, const char *source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
    
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        printf("Shader compilation error: %s\n", infoLog);
    }
    
    return shader;
}

static void init_opengl(SDRAnalyzer *a) {
    // Create shader program
    GLuint vertex_shader = compile_shader(GL_VERTEX_SHADER, vertex_shader_source);
    GLuint fragment_shader = compile_shader(GL_FRAGMENT_SHADER, fragment_shader_source);
    
    a->shader_program = glCreateProgram();
    glAttachShader(a->shader_program, vertex_shader);
    glAttachShader(a->shader_program, fragment_shader);
    glLinkProgram(a->shader_program);
    
    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);
    
    // Create VAO and VBO
    glGenVertexArrays(1, &a->vao);
    glGenBuffers(1, &a->vbo);
    
    glBindVertexArray(a->vao);
    glBindBuffer(GL_ARRAY_BUFFER, a->vbo);
    
    // Vertex data (quad covering the screen)
    float vertices[] = {
        // positions        // texture coords
        -1.0f,  1.0f, 0.0f, 1.0f,
        -1.0f, -1.0f, 0.0f, 0.0f,
         1.0f, -1.0f, 1.0f, 0.0f,
        
        -1.0f,  1.0f, 0.0f, 1.0f,
         1.0f, -1.0f, 1.0f, 0.0f,
         1.0f,  1.0f, 1.0f, 1.0f
    };
    
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    
    // Position attribute
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    // Texture coord attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    glBindVertexArray(0);
    
    // Create texture
    glGenTextures(1, &a->texture_id);
    glBindTexture(GL_TEXTURE_2D, a->texture_id);
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    
    a->gl_initialized = 1;
    printf("OpenGL initialized successfully\n");
}

// ============================================================================
// Utility Functions
// ============================================================================
static double calculate_entropy(float *data, int size) {
    if (size <= 0 || data == NULL) return 0.0;
    
    double sum = 0.0;
    double min_val = data[0];
    double max_val = data[0];
    
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
        if (p > 1e-10) {
            entropy -= p * log2(p);
        }
    }
    
    free(norm);
    return entropy;
}

static double calculate_prf(float *data, int frames, double time_per_frame) {
    if (frames <= 0 || data == NULL || time_per_frame <= 0) return 0.0;
    
    double *time_series = malloc(frames * sizeof(double));
    if (!time_series) return 0.0;
    
    double sum = 0.0;
    for (int i = 0; i < frames; i++) {
        time_series[i] = 0.0;
        int count = 0;
        for (int j = 0; j < 100 && j < frames; j++) {
            time_series[i] += data[i * frames + j];
            count++;
        }
        if (count > 0) time_series[i] /= count;
        sum += time_series[i];
    }
    
    double mean = sum / frames;
    double variance = 0.0;
    for (int i = 0; i < frames; i++) {
        double diff = time_series[i] - mean;
        variance += diff * diff;
    }
    variance /= frames;
    double std = sqrt(variance);
    double threshold = mean + 1.5 * std;
    
    int *peaks = malloc(frames * sizeof(int));
    if (!peaks) {
        free(time_series);
        return 0.0;
    }
    
    int peak_count = 0;
    int above = 0;
    
    for (int i = 0; i < frames; i++) {
        if (time_series[i] > threshold && !above) {
            peaks[peak_count++] = i;
            above = 1;
        } else if (time_series[i] <= threshold) {
            above = 0;
        }
    }
    
    double prf = 0.0;
    if (peak_count >= 2) {
        double interval_sum = 0.0;
        for (int i = 1; i < peak_count; i++) {
            interval_sum += (peaks[i] - peaks[i-1]) * time_per_frame;
        }
        double avg_interval = interval_sum / (peak_count - 1);
        if (avg_interval > 0) {
            prf = 1.0 / avg_interval;
        }
    }
    
    free(time_series);
    free(peaks);
    return prf;
}

static void match_known_signals(SDRAnalyzer *a, double center_freq, double bandwidth) {
    a->analysis.match_count = 0;
    
    for (int i = 0; i < num_known_signals && a->analysis.match_count < 10; i++) {
        KnownSignal *sig = &known_signals_db[i];
        int confidence = 0;
        
        if (center_freq >= sig->freq_min && center_freq <= sig->freq_max) {
            confidence += 70;
        }
        if (bandwidth >= sig->bw_min && bandwidth <= sig->bw_max) {
            confidence += 30;
        }
        
        if (confidence >= 50) {
            strcpy(a->analysis.matches[a->analysis.match_count], sig->name);
            a->analysis.match_confidence[a->analysis.match_count] = confidence;
            a->analysis.match_count++;
        }
    }
}

// ============================================================================
// File Loading
// ============================================================================
static int parse_filename(SDRAnalyzer *a) {
    char *basename = strrchr(a->filepath, '/');
    basename = basename ? basename + 1 : a->filepath;
    
    unsigned int date, time;
    long long freq, rate;
    char type[16];
    
    if (sscanf(basename, "gqrx_%u_%u_%lld_%lld_%15s.raw", 
               &date, &time, &freq, &rate, type) == 5) {
        a->center_freq = (double)freq;
        a->sample_rate = (double)rate;
        printf("File: %s\n", basename);
        printf("Date: %u, Time: %u\n", date, time);
        printf("Center Frequency: %.2f MHz\n", a->center_freq / 1e6);
        printf("Sample Rate: %.2f MS/s\n", a->sample_rate / 1e6);
        return 1;
    }
    
    printf("Warning: Could not parse filename\n");
    return 0;
}

static int load_iq_file(SDRAnalyzer *a) {
    printf("Loading IQ file...\n");
    
    a->fd = open(a->filepath, O_RDONLY);
    if (a->fd < 0) {
        perror("Failed to open file");
        return -1;
    }
    
    struct stat st;
    if (fstat(a->fd, &st) < 0) {
        perror("Failed to stat file");
        close(a->fd);
        return -1;
    }
    
    a->file_size = st.st_size;
    printf("File size: %.1f MB\n", a->file_size / 1e6);
    
    a->iq_data = mmap(NULL, a->file_size, PROT_READ, MAP_PRIVATE, a->fd, 0);
    if (a->iq_data == MAP_FAILED) {
        perror("Failed to mmap file");
        close(a->fd);
        return -1;
    }
    
    if (a->file_size % 8 == 0) {
        a->total_samples = a->file_size / 8;
    } else {
        a->total_samples = a->file_size / 16;
    }
    
    a->total_duration = a->total_samples / a->sample_rate;
    
    printf("Memory-mapped %zu samples (%.1f MB)\n", a->total_samples, a->file_size / 1e6);
    printf("Total duration: %.2f seconds\n", a->total_duration);
    
    return 0;
}

// ============================================================================
// Waterfall Cache Loading
// ============================================================================
static void request_cache_load(SDRAnalyzer *a, int start_frame) {
    pthread_mutex_lock(&a->cache_mutex);
    
    if (a->loading) {
        pthread_mutex_unlock(&a->cache_mutex);
        return;
    }
    
    int end_frame = start_frame + a->cache_frames;
    if (end_frame > a->total_frames) {
        end_frame = a->total_frames;
    }
    
    a->load_start_frame = start_frame;
    a->load_end_frame = end_frame;
    a->loading = 1;
    a->load_progress = 0;
    a->needs_redraw = 1;
    
    pthread_mutex_unlock(&a->cache_mutex);
}

static void ensure_cache(SDRAnalyzer *a, int frame_idx) {
    pthread_mutex_lock(&a->cache_mutex);
    
    int found = 0;
    for (int i = 0; i < a->num_cache_windows; i++) {
        if (a->cache_windows[i].valid &&
            frame_idx >= a->cache_windows[i].start_frame &&
            frame_idx < a->cache_windows[i].end_frame) {
            found = 1;
            break;
        }
    }
    
    pthread_mutex_unlock(&a->cache_mutex);
    
    if (!found) {
        int new_start = frame_idx - a->cache_frames / 3;
        if (new_start < 0) new_start = 0;
        
        manage_cache_windows(a, frame_idx);
        request_cache_load(a, new_start);
    }
}

// ============================================================================
// Timer callback for UI updates
// ============================================================================
static gboolean ui_update_timer(gpointer user_data) {
    SDRAnalyzer *a = (SDRAnalyzer *)user_data;
    
    pthread_mutex_lock(&a->cache_mutex);
    int needs_redraw = a->needs_redraw;
    int loading = a->loading;
    int progress = a->load_progress;
    pthread_mutex_unlock(&a->cache_mutex);
    
    if (needs_redraw) {
        if (a->gl_area) {
            gtk_gl_area_queue_render(GTK_GL_AREA(a->gl_area));
        }
        if (a->spectrum_drawing) {
            gtk_widget_queue_draw(a->spectrum_drawing);
        }
        
        if (loading) {
            char loading_text[64];
            snprintf(loading_text, sizeof(loading_text), "Loading... %d%%", progress);
            gtk_label_set_text(GTK_LABEL(a->info_label), loading_text);
        }
        
        pthread_mutex_lock(&a->cache_mutex);
        a->needs_redraw = 0;
        pthread_mutex_unlock(&a->cache_mutex);
    }
    
    return G_SOURCE_CONTINUE;
}

// ============================================================================
// Analysis
// ============================================================================
static void analyze_selection(SDRAnalyzer *a) {
    if (!a->has_selection) {
        printf("No selection made.\n");
        return;
    }
    
    if (a->sel_time_min > a->sel_time_max) {
        double tmp = a->sel_time_min;
        a->sel_time_min = a->sel_time_max;
        a->sel_time_max = tmp;
    }
    
    if (a->sel_freq_min > a->sel_freq_max) {
        double tmp = a->sel_freq_min;
        a->sel_freq_min = a->sel_freq_max;
        a->sel_freq_max = tmp;
    }
    
    printf("\nAnalyzing...\n");
    
    size_t start_sample = (size_t)(a->sel_time_min * a->sample_rate);
    size_t end_sample = (size_t)(a->sel_time_max * a->sample_rate);
    
    if (start_sample >= a->total_samples || end_sample <= start_sample) {
        printf("  Invalid selection\n");
        return;
    }
    if (end_sample > a->total_samples) end_sample = a->total_samples;
    
    size_t segment_len = end_sample - start_sample;
    
    float complex *segment = malloc(segment_len * sizeof(float complex));
    if (!segment) return;
    
    memcpy(segment, &a->iq_data[start_sample], segment_len * sizeof(float complex));
    
    int fft_size = 8192;
    int hop_size = (int)(fft_size * 0.25);
    int num_frames = (segment_len - fft_size) / hop_size + 1;
    
    if (num_frames <= 0 || num_frames > 500) {
        free(segment);
        return;
    }
    
    double freq_bin_width = a->sample_rate / fft_size;
    double center_freq_edge = a->center_freq - a->sample_rate/2;
    int freq_bin_min = (int)((a->sel_freq_min - center_freq_edge) / freq_bin_width);
    int freq_bin_max = (int)((a->sel_freq_max - center_freq_edge) / freq_bin_width);
    
    if (freq_bin_min < 0) freq_bin_min = 0;
    if (freq_bin_max > fft_size) freq_bin_max = fft_size;
    if (freq_bin_max <= freq_bin_min) {
        free(segment);
        return;
    }
    
    int analysis_bins = freq_bin_max - freq_bin_min;
    
    if (a->analysis_image) free(a->analysis_image);
    a->analysis_image = malloc(num_frames * analysis_bins * sizeof(float));
    a->analysis_frames = num_frames;
    a->analysis_bins = analysis_bins;
    
    fftwf_complex *in = fftwf_malloc(sizeof(fftwf_complex) * fft_size);
    fftwf_complex *out = fftwf_malloc(sizeof(fftwf_complex) * fft_size);
    fftwf_plan plan = fftwf_plan_dft_1d(fft_size, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    
    float *window = malloc(fft_size * sizeof(float));
    for (int i = 0; i < fft_size; i++) {
        window[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (fft_size - 1)));
    }
    
    for (int i = 0; i < num_frames; i++) {
        size_t idx = i * hop_size;
        for (int j = 0; j < fft_size && idx + j < segment_len; j++) {
            in[j][0] = crealf(segment[idx + j]) * window[j];
            in[j][1] = cimagf(segment[idx + j]) * window[j];
        }
        
        fftwf_execute(plan);
        
        for (int j = 0; j < analysis_bins; j++) {
            int shifted_idx = (freq_bin_min + j + fft_size / 2) % fft_size;
            double mag = sqrt(out[shifted_idx][0] * out[shifted_idx][0] + 
                             out[shifted_idx][1] * out[shifted_idx][1]);
            a->analysis_image[i * analysis_bins + j] = 20.0 * log10(mag + 1e-10);
        }
    }
    
    fftwf_destroy_plan(plan);
    fftwf_free(in);
    fftwf_free(out);
    free(window);
    free(segment);
    
    a->analysis.bandwidth = a->sel_freq_max - a->sel_freq_min;
    a->analysis.center_freq = (a->sel_freq_min + a->sel_freq_max) / 2.0;
    a->analysis.duration = a->sel_time_max - a->sel_time_min;
    
    float sum = 0.0, max_val = -1000.0;
    for (int i = 0; i < num_frames * analysis_bins; i++) {
        sum += a->analysis_image[i];
        if (a->analysis_image[i] > max_val) max_val = a->analysis_image[i];
    }
    a->analysis.avg_power = sum / (num_frames * analysis_bins);
    a->analysis.max_power = max_val;
    
    a->analysis.entropy = calculate_entropy(a->analysis_image, num_frames * analysis_bins);
    a->analysis.prf = calculate_prf(a->analysis_image, num_frames, hop_size / a->sample_rate);
    
    strcpy(a->analysis.signal_type, a->analysis.bandwidth < 1000000 ? "Wideband" : "Very Wideband");
    strcpy(a->analysis.modulation, "OFDM");
    
    match_known_signals(a, a->analysis.center_freq, a->analysis.bandwidth);
    
    printf("Analysis complete.\n");
    show_analysis_popup(a);
}

// ============================================================================
// GTK Callbacks
// ============================================================================
static void on_db_min_changed(GtkRange *range, gpointer user_data) {
    SDRAnalyzer *a = (SDRAnalyzer *)user_data;
    a->db_min = gtk_range_get_value(range);
    if (a->db_min >= a->db_max) {
        a->db_max = a->db_min + 1.0;
        gtk_range_set_value(GTK_RANGE(a->db_max_slider), a->db_max);
    }
    a->auto_scale = 0;
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(a->auto_scale_check), FALSE);
    if (a->gl_area) gtk_gl_area_queue_render(GTK_GL_AREA(a->gl_area));
}

static void on_db_max_changed(GtkRange *range, gpointer user_data) {
    SDRAnalyzer *a = (SDRAnalyzer *)user_data;
    a->db_max = gtk_range_get_value(range);
    if (a->db_max <= a->db_min) {
        a->db_min = a->db_max - 1.0;
        gtk_range_set_value(GTK_RANGE(a->db_min_slider), a->db_min);
    }
    a->auto_scale = 0;
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(a->auto_scale_check), FALSE);
    if (a->gl_area) gtk_gl_area_queue_render(GTK_GL_AREA(a->gl_area));
}

static void on_auto_scale_toggled(GtkCheckButton *check, gpointer user_data) {
    SDRAnalyzer *a = (SDRAnalyzer *)user_data;
    a->auto_scale = gtk_toggle_button_get_active(check);
    gtk_widget_set_sensitive(a->db_min_slider, !a->auto_scale);
    gtk_widget_set_sensitive(a->db_max_slider, !a->auto_scale);
    if (a->gl_area) gtk_gl_area_queue_render(GTK_GL_AREA(a->gl_area));
}

static void show_analysis_popup(SDRAnalyzer *a) {
    // Simplified popup - implement as needed
    printf("Analysis popup would show here\n");
}

static void on_scroll(SDRAnalyzer *a, double delta) {
    double new_pos = a->scroll_position + delta;
    double max_scroll = a->total_duration - a->visible_seconds;
    if (max_scroll < 0) max_scroll = 0;
    if (new_pos < 0) new_pos = 0;
    if (new_pos > max_scroll) new_pos = max_scroll;
    
    if (new_pos != a->scroll_position) {
        a->scroll_position = new_pos;
        
        // Pre-cache adjacent windows
        int current_frame = (int)(a->scroll_position * a->frames_per_second);
        for (int i = -1; i <= PRECACHE_WINDOWS; i++) {
            int preload_frame = current_frame + i * a->cache_frames;
            if (preload_frame >= 0 && preload_frame < a->total_frames) {
                ensure_cache(a, preload_frame);
            }
        }
        
        if (a->gl_area) gtk_gl_area_queue_render(GTK_GL_AREA(a->gl_area));
        
        char info[512];
        snprintf(info, sizeof(info),
            "Pos: %.2f/%.1f s | FFT: %d | Windows: %d",
            a->scroll_position, a->total_duration, a->fft_size, a->num_cache_windows);
        gtk_label_set_text(GTK_LABEL(a->info_label), info);
    }
}

static gboolean on_key_press(GtkWidget *widget, GdkEventKey *event, gpointer user_data) {
    SDRAnalyzer *a = (SDRAnalyzer *)user_data;
    
    switch (event->keyval) {
        case GDK_KEY_Up:
            on_scroll(a, -a->scroll_seconds);
            break;
        case GDK_KEY_Down:
            on_scroll(a, a->scroll_seconds);
            break;
        case GDK_KEY_q:
        case GDK_KEY_Q:
            gtk_main_quit();
            break;
    }
    return TRUE;
}

// ============================================================================
// OpenGL Render Callback
// ============================================================================
static gboolean on_gl_area_render(GtkGLArea *area, GdkGLContext *context, gpointer user_data) {
    SDRAnalyzer *a = (SDRAnalyzer *)user_data;
    
    if (!a->gl_initialized) {
        init_opengl(a);
    }
    
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    
    // Find cache window for current position
    int current_frame = (int)(a->scroll_position * a->frames_per_second);
    int end_frame = (int)((a->scroll_position + a->visible_seconds) * a->frames_per_second);
    
    pthread_mutex_lock(&a->cache_mutex);
    
    CacheWindow *active_window = NULL;
    for (int i = 0; i < a->num_cache_windows; i++) {
        if (a->cache_windows[i].valid &&
            current_frame >= a->cache_windows[i].start_frame &&
            current_frame < a->cache_windows[i].end_frame) {
            active_window = &a->cache_windows[i];
            active_window->ref_count++;
            break;
        }
    }
    
    pthread_mutex_unlock(&a->cache_mutex);
    
    if (!active_window || !active_window->valid) {
        return TRUE;
    }
    
    // Update texture
    glBindTexture(GL_TEXTURE_2D, a->texture_id);
    
    int frames_to_show = end_frame - current_frame;
    if (frames_to_show <= 0) frames_to_show = 1;
    
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, 
                 a->fft_size, frames_to_show, 0,
                 GL_RED, GL_FLOAT, 
                 &active_window->data[(current_frame - active_window->start_frame) * a->fft_size]);
    
    // Render
    glUseProgram(a->shader_program);
    
    GLint db_min_loc = glGetUniformLocation(a->shader_program, "db_min");
    GLint db_max_loc = glGetUniformLocation(a->shader_program, "db_max");
    
    glUniform1f(db_min_loc, a->db_min);
    glUniform1f(db_max_loc, a->db_max);
    
    glBindVertexArray(a->vao);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    
    pthread_mutex_lock(&a->cache_mutex);
    active_window->ref_count--;
    pthread_mutex_unlock(&a->cache_mutex);
    
    return TRUE;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <gqrx_raw_file>\n", argv[0]);
        return 1;
    }
    
    init_viridis_colormap();
    
    SDRAnalyzer analyzer = {0};
    analyzer.filepath = argv[1];
    analyzer.fft_size = FFT_SIZE;
    analyzer.fft_overlap = FFT_OVERLAP;
    analyzer.visible_seconds = VISIBLE_SECONDS;
    analyzer.scroll_seconds = SCROLL_SECONDS;
    analyzer.cache_frames = (int)(CACHE_SECONDS * 256);
    
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
    analyzer.cache_frames = (int)(CACHE_SECONDS * analyzer.frames_per_second);
    
    printf("\n========================================\n");
    printf("OpenGL SDR Waterfall Analyzer\n");
    printf("========================================\n");
    printf("FFT Size: %d bins\n", analyzer.fft_size);
    printf("Cache Windows: %d (max %d)\n", PRECACHE_WINDOWS, MAX_CACHE_WINDOWS);
    printf("Cache Size: %.1f MB per window\n", 
           analyzer.cache_frames * analyzer.fft_size * sizeof(float) / 1e6);
    printf("========================================\n\n");
    
    pthread_create(&analyzer.load_thread, NULL, cache_load_thread, &analyzer);
    request_cache_load(&analyzer, 0);
    
    gtk_init(&argc, &argv);
    
    analyzer.window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_title(GTK_WINDOW(analyzer.window), "SDR Waterfall (OpenGL)");
    gtk_window_set_default_size(GTK_WINDOW(analyzer.window), 1400, 900);
    g_signal_connect(analyzer.window, "destroy", G_CALLBACK(gtk_main_quit), NULL);
    g_signal_connect(analyzer.window, "key-press-event", G_CALLBACK(on_key_press), &analyzer);
    
    GtkWidget *vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    gtk_container_add(GTK_CONTAINER(analyzer.window), vbox);
    
    // OpenGL Area
    analyzer.gl_area = gtk_gl_area_new();
    gtk_widget_set_size_request(analyzer.gl_area, -1, 600);
    g_signal_connect(analyzer.gl_area, "render", G_CALLBACK(on_gl_area_render), &analyzer);
    gtk_box_pack_start(GTK_BOX(vbox), analyzer.gl_area, TRUE, TRUE, 0);
    
    // Spectrum display (simplified)
    analyzer.spectrum_drawing = gtk_drawing_area_new();
    gtk_widget_set_size_request(analyzer.spectrum_drawing, -1, 150);
    gtk_box_pack_start(GTK_BOX(vbox), analyzer.spectrum_drawing, FALSE, FALSE, 0);
    
    // Controls
    GtkWidget *control_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
    gtk_widget_set_margin_start(control_box, 10);
    gtk_widget_set_margin_end(control_box, 10);
    gtk_widget_set_margin_top(control_box, 5);
    gtk_widget_set_margin_bottom(control_box, 5);
    
    GtkWidget *db_min_label = gtk_label_new("Min dB:");
    gtk_box_pack_start(GTK_BOX(control_box), db_min_label, FALSE, FALSE, 0);
    
    analyzer.db_min_slider = gtk_scale_new_with_range(GTK_ORIENTATION_HORIZONTAL, -150.0, 0.0, 1.0);
    gtk_range_set_value(GTK_RANGE(analyzer.db_min_slider), DEFAULT_DB_MIN);
    gtk_widget_set_size_request(analyzer.db_min_slider, 200, -1);
    g_signal_connect(analyzer.db_min_slider, "value-changed", G_CALLBACK(on_db_min_changed), &analyzer);
    gtk_box_pack_start(GTK_BOX(control_box), analyzer.db_min_slider, FALSE, FALSE, 0);
    
    GtkWidget *db_max_label = gtk_label_new("Max dB:");
    gtk_box_pack_start(GTK_BOX(control_box), db_max_label, FALSE, FALSE, 0);
    
    analyzer.db_max_slider = gtk_scale_new_with_range(GTK_ORIENTATION_HORIZONTAL, -150.0, 0.0, 1.0);
    gtk_range_set_value(GTK_RANGE(analyzer.db_max_slider), DEFAULT_DB_MAX);
    gtk_widget_set_size_request(analyzer.db_max_slider, 200, -1);
    g_signal_connect(analyzer.db_max_slider, "value-changed", G_CALLBACK(on_db_max_changed), &analyzer);
    gtk_box_pack_start(GTK_BOX(control_box), analyzer.db_max_slider, FALSE, FALSE, 0);
    
    analyzer.auto_scale_check = gtk_check_button_new_with_label("Auto-Scale");
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(analyzer.auto_scale_check), TRUE);
    g_signal_connect(analyzer.auto_scale_check, "toggled", G_CALLBACK(on_auto_scale_toggled), &analyzer);
    gtk_box_pack_start(GTK_BOX(control_box), analyzer.auto_scale_check, FALSE, FALSE, 0);
    
    gtk_box_pack_start(GTK_BOX(vbox), control_box, FALSE, FALSE, 0);
    
    analyzer.info_label = gtk_label_new("Loading...");
    gtk_widget_set_halign(analyzer.info_label, GTK_ALIGN_START);
    gtk_widget_set_margin_start(analyzer.info_label, 10);
    gtk_box_pack_start(GTK_BOX(vbox), analyzer.info_label, FALSE, FALSE, 0);
    
    gtk_widget_show_all(analyzer.window);
    
    gtk_widget_set_sensitive(analyzer.db_min_slider, FALSE);
    gtk_widget_set_sensitive(analyzer.db_max_slider, FALSE);
    
    g_timeout_add(50, ui_update_timer, &analyzer);  // 20 Hz update
    
    printf("Ready! Use Up/Down arrows to scroll.\n");
    
    gtk_main();
    
    analyzer.quit_thread = 1;
    pthread_join(analyzer.load_thread, NULL);
    
    munmap(analyzer.iq_data, analyzer.file_size);
    close(analyzer.fd);
    
    for (int i = 0; i < analyzer.num_cache_windows; i++) {
        free_cache_window(&analyzer.cache_windows[i]);
    }
    
    pthread_mutex_destroy(&analyzer.cache_mutex);
    
    return 0;
}