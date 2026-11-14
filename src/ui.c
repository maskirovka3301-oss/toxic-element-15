/*
 * ui.c — GTK3 user interface for the SDR waterfall analyzer.
 *
 * Owns the widget tree, the scroll/zoom view model, drag-selection, the
 * signal-analysis popup window, and the offscreen render surfaces.
 *
 * Threading rule: everything in this file runs on the GTK main thread.
 * The only communication with the cache worker goes through the
 * WaterfallCache API — wc_progress() for the loading indicator and
 * wc_lock()/wc_unlock() around the data reads inside the draw handler.
 * Cache internals are never touched.
 */

#include <float.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
#include <time.h>

#include "ui.h"
#include "intraburst.h"
#include "ssclass.h"
#include "sot.h"
#include "aresult.h"
#include "aregistry.h"
#include "apublish.h"
#include "dag.h"
#include "director.h"
#include "report.h"
#include "renderbuf.h"
#include "streambuf.h"
#include "lte_uplink.h"
#include "aconfig.h"
#include "debug.h"
#include "about_image.h"
#include "fcdag.h"
#include "fcreport.h"
#include "fcdataset.h"
#include "fit.h"
#include "wav.h"

/* --- constants ---------------------------------------------------------- */

#define FFT_SIZE 8192 /* display value only; must match main.c's cache spec */

#define SCROLL_SECONDS          1.0
#define DEFAULT_VISIBLE_SECONDS 1.0
#define UI_TIMER_MS            50
/* waterfall playback-position line */
#define PLAYHEAD_R             1.0
#define PLAYHEAD_G             0.15
#define PLAYHEAD_B             0.12
#define PLAYHEAD_WIDTH         2.0
/* transient notices in the bottom-right status label disappear after this
 * many seconds (the routine filename/progress status then returns) */
#define INFO_MESSAGE_TIMEOUT_S 5
#define DEFAULT_DB_MIN       -100.0
#define DEFAULT_DB_MAX        -20.0

/* zoom and cache requests */
/* Waterfall scrolling: a two-finger swipe scrolls through time. GTK
 * reports touchpads as smooth scrolls in "scroll units"; a fixed number of
 * pixels per unit keeps the feel consistent across devices. */
#define WF_SCROLL_PIXELS_PER_UNIT  40.0
#define WF_SCROLL_NOTCH_PIXELS     60.0
#define CACHE_WINDOW_FACTOR 2

/* main window */
#define WINDOW_TITLE       "Toxic Element 15"
#define WINDOW_WIDTH       1280
#define WINDOW_HEIGHT      800
#define WINDOW_MIN_WIDTH   1024
#define WINDOW_MIN_HEIGHT  768

/* window size persistence */
#define WINDOW_CONF_DIR    "te15"
#define WINDOW_CONF_FILE   "window.conf"
#define WINDOW_CONF_GROUP  "window"
#define WINDOW_CONF_WIDTH  "width"
#define WINDOW_CONF_HEIGHT "height"
#define DRAWING_MIN_HEIGHT 600
#define CONTROL_MARGIN_X   10
#define CONTROL_MARGIN_Y   5
#define CONTROL_SPACING    5

/* dB sliders */
#define DB_SCALE_MIN  -150.0
#define DB_SCALE_MAX     0.0
#define DB_SCALE_STEP    1.0
#define DB_SCALE_WIDTH 120

/* labels */
#define INFO_LOADING      "Loading..."
#define INFO_PROGRESS_FMT "Loading... %d%%"
#define LABEL_BUFFER_SIZE 128
#define RANGE_LABEL_SIZE  128

/* waterfall drawing */
#define FONT_FAMILY       "Sans"
#define FONT_SIZE_LOADING 18.0
#define FONT_SIZE_NO_DATA 16.0
#define DB_RANGE_MIN_SPAN 1.0
/* absolute dB domain of the waterfall colour scheme (one LUT entry per
 * 1/255 of this span); the dB sliders select which window of it maps to the
 * full viridis gradient inside an in-memory copy of the scheme */
#define WF_LUT_DB_MIN (-180.0)
#define WF_LUT_DB_MAX   20.0
#define HZ_PER_KHZ        1e3
#define HZ_PER_MHZ        1e6
#define SEL_FILL_R        1.0
#define SEL_FILL_G        0.0
#define SEL_FILL_B        0.0
#define SEL_FILL_A        0.3
#define SEL_EDGE_PX       6   /* edge-grab tolerance, pixels */
#define SEL_MOVE_PX       3   /* min motion for a new selection */
#define SEL_LINE_R        1.0
/* full-height (whole-file) selection mode: blue rectangle */
#define SEL_FULL_LINE_R  0.25f
/* loop-marker row (one pixel row of complex zeros) */
#define LOOP_ROW_R 0.90
#define LOOP_ROW_G 0.10
#define LOOP_ROW_B 0.10
#define SEL_FULL_LINE_G  0.55f
#define SEL_FULL_LINE_B  1.00f
#define SEL_FULL_FILL_A  0.10f
#define SEL_LINE_G        0.0
#define SEL_LINE_B        0.0
#define SEL_LINE_WIDTH    2.0

/* analysis popup */
#define POPUP_TITLE          "Selection Details"
#define POPUP_WIDTH          1000
#define POPUP_HEIGHT         700
#define POPUP_SPACING        10
#define POPUP_DRAWING_WIDTH  600
#define POPUP_DRAWING_HEIGHT 400
#define STATS_BOX_WIDTH      350
#define POPUP_CLOSE_LABEL    "Close"
#define POPUP_BORDER_R       0.4
#define POPUP_BORDER_G       0.4
#define POPUP_BORDER_B       0.5
#define POPUP_BORDER_WIDTH   1.0

/* screenshot and stats text */
#define SCREENSHOT_PREFIX   "sdr_waterfall_%Y%m%d_%H%M%S.png"
#define FILENAME_BUFFER_SIZE 256
#define STATS_BUFFER_SIZE    4096

/* detected-signal overlay */
#define DETECT_FILL_R     1.0
#define DETECT_FILL_G     0.84
#define DETECT_FILL_B     0.0
#define DETECT_FILL_A     0.18
#define DETECT_LINE_R     1.0
#define DETECT_LINE_G     0.84
#define DETECT_LINE_B     0.0
#define DETECT_LINE_WIDTH 2.0
/* Selected detected region: red edges instead of yellow, slightly thicker. */
#define DET_SEL_R         0.95f
#define DET_SEL_G         0.15f
#define DET_SEL_B         0.15f
#define DET_SEL_LINE_WIDTH 3.0

/* timeline scrolling, frequency zoom and peak pane */
#define MIN_FREQ_BW            1000.0
#define USEC_PER_SEC           1e6
#define FULL_ZOOM_EPSILON      1.0
#define PEAK_UPDATE_INTERVAL_US 200000
/* pixel rows averaged into the peak strip (it shows the mean spectrum of
 * the visible waterfall rows, from the render buffer) */
#define PEAK_MEAN_ROWS 128

/* peak pane (spectrum strip between waterfall and toolbar) */
#define PEAK_PANE_HEIGHT 150
#define PEAK_BG_R        0.05
#define PEAK_BG_G        0.05
#define PEAK_BG_B        0.06
#define PEAK_HANDLE_GRAB 5.0
#define PEAK_HANDLE_WIDTH 4.0
#define PEAK_NO_DATA     "no spectrum"
/* left dB axis of the peak strip: tick labels are computed from the allowed
 * dB range (the same DB_SCALE_MIN..DB_SCALE_MAX the sliders use) */
#define PEAK_DB_AXIS_W   42
#define PEAK_DB_TICK_LEN 4.0
#define PEAK_SHADE_R     0.16
#define PEAK_SHADE_G     0.38
#define PEAK_SHADE_B     0.95
#define PEAK_SHADE_A     0.5
#define PEAK_FILL_R      0.2
#define PEAK_FILL_G      0.7
#define PEAK_FILL_B      0.3
#define PEAK_FILL_A      0.9
#define PEAK_RECT_R      0.0
#define PEAK_RECT_G      1.0
#define PEAK_RECT_B      0.0
#define PEAK_RECT_A      0.15
#define PEAK_LINE_R      0.0
#define PEAK_LINE_G      0.9
#define PEAK_LINE_B      0.0
#define PEAK_LINE_WIDTH  1.5

/* extended analysis popup */
#define STATS_SCROLLER_WIDTH 360
#define MOD_LABEL_SIZE       256
#define RANKED_LABEL_SIZE    1024

/* demodulation */
#define DEMOD_OUT_RATE     32000.0
#define DEMOD_MAX_SAMPLES  32000000 /* 32M complex samples ~= 256 MB mix buffer */
#define DEMOD_FALLBACK_DEV 75000.0
#define DEMOD_PAD_SAMPLES  64
#define DEMOD_FILENAME_SIZE 1024
#define DEMOD_STATUS_SIZE   1280
#define RAW_SUFFIX_LEN      4

/* audio player */
#define PLAYER_TITLE         "Demod Player"
#define PLAYER_WIDTH         420
#define PLAYER_HEIGHT        140
#define PLAYER_TIMER_MS      100
#define PLAYER_SCALE_STEP    0.1
#define PLAYER_TIME_BUFFER   16
#define PLAYER_TIME_SEP      4
#define PLAYER_TIME_INITIAL  "0:00.0 / 0:00.0"
#define PLAY_BTN_LABEL       "Play"
#define PAUSE_BTN_LABEL      "Pause"

/* menus and dialogs */
#define MENU_FILE_LABEL     "File"
/* The whole main view lives in a tabbed pane under this tab (room for
 * further views later). */
#define MAIN_TAB_LABEL      "Spectrum"
#define FREQ_COUNTER_TAB_LABEL "Frequency Counter"
#define MENU_OPEN_LABEL     "Open…"
#define MENU_ABOUT_LABEL    "About"
#define MENU_QUIT_LABEL     "Quit"
#define MENU_SAVE_SEL_LABEL "Save Selection…"

/* Save Selection: chunked raw-IQ writer */
#define SAVE_SEL_TITLE       "Save Selection"
#define SAVE_SEL_ACCEPT      "Save"
#define SAVE_SEL_CANCEL      "Cancel"
#define SAVE_SEL_FILTER_NAME "Raw IQ recordings (*.raw)"
#define SAVE_SEL_CHUNK       262144
#define MENU_ANALYZE_LABEL  "Analyze"
#define MENU_DETAILS_LABEL  "Selection Details"
#define MENU_CYCLIC_LABEL   "Cyclostationary Analysis"
#define MENU_FLUCT_LABEL    "Fluctuations of Autocorrelation Estimators"
#define MENU_CPAR_LABEL     "Cumulative Peak-to-Average Ratio"
#define MENU_HOS_LABEL      "Higher-Order Statistics"
#define OPEN_DIALOG_TITLE   "Open recording"
#define OPEN_BTN_LABEL      "Open"
#define CANCEL_BTN_LABEL    "Cancel"
#define SETTINGS_TITLE      "Settings"
#define SETTINGS_WIDTH      360
#define SETTINGS_HEIGHT     200
#define SETTINGS_TEXT       "Configuration options are coming soon."
#define FILTER_IQ_LABEL     "IQ recordings"
#define FILTER_IQ_PATTERN   "*.raw"
#define FILTER_ALL_LABEL    "All files"
#define FILTER_ALL_PATTERN  "*"
#define SAVE_DIALOG_TITLE   "Save demodulated audio"
#define SAVE_BTN_LABEL      "Save"

/* Selection page buttons */
#define ANALYZE_BTN_LABEL      "Analyze"
#define SYNTH_BTN_LABEL        "Synthesize"
#define SAVE_REPORT_BTN_LABEL  "Save"

/* cyclostationary window */
#define CYCLIC_TITLE          "Cyclostationary Analysis"
#define CYCLIC_WIDTH          960
#define CYCLIC_HEIGHT         640
#define CYCLIC_SPACING        8
#define CYCLIC_MARGIN         10
#define CYCLIC_CDP_HEIGHT     280
#define CYCLIC_FRAME_LABEL    "Spectral Correlation Function"
#define CYCLIC_NO_DATA        "No cyclic data"
#define CYCLIC_LABEL_SIZE     4096
#define CYCLIC_FONT_SIZE      14.0
#define CYCLIC_TICK_FONT_SIZE 10.0
#define CYCLIC_TICK_MARGIN    6
#define CYCLIC_TICK_TEXT      "1.0"
#define CYCLIC_BG_R           0.06
#define CYCLIC_BG_G           0.06
#define CYCLIC_BG_B           0.08
#define CYCLIC_CDP_Y_MAX      1.1
#define CYCLIC_LINE_WIDTH     1.5
#define CYCLIC_AXIS_GREY      0.5
#define CYCLIC_PEAK_RADIUS    3.5
#define CYCLIC_PEAK_LINE      2.0

/* fluctuation analysis window */
#define FLUCT_TITLE        "Fluctuations of Autocorrelation Estimators"
#define FLUCT_WIDTH        900
#define FLUCT_HEIGHT       640
#define FLUCT_SPACING      8
#define FLUCT_MARGIN       10
#define FLUCT_PLOT_HEIGHT  220
#define FLUCT_ACF_FRAME    "Mean Autocorrelation (|R(tau)| / sigma^2)"
#define FLUCT_F_FRAME      "Fluctuation F(tau) = T * Var[R(tau)] / sigma^4"
#define FLUCT_NO_DATA      "No fluctuation data"
#define FLUCT_LABEL_SIZE   1024
#define FLUCT_FONT_SIZE    14.0
#define FLUCT_LINE_WIDTH   1.5
#define FLUCT_Y_SCALE      1.1
#define FLUCT_ACF_FLOOR    0.001
#define FLUCT_F_FLOOR      2.2
#define FLUCT_F_UNITY      1.0
#define FLUCT_PEAK_RADIUS  4.0
#define FLUCT_AXIS_GREY    0.5
#define FLUCT_THRESH_RED   0.9
#define FLUCT_ZERO_GREY    0.8
#define FLUCT_BG_R         0.06
#define FLUCT_BG_G         0.06
#define FLUCT_BG_B         0.08

/* CPAR analysis window */
#define CPAR_TITLE         "Cumulative Peak-to-Average Ratio"
#define CPAR_WIDTH         900
#define CPAR_HEIGHT        640
#define CPAR_SPACING       8
#define CPAR_MARGIN        10
#define CPAR_ACF_HEIGHT    240
#define CPAR_PAR_HEIGHT    200
#define CPAR_ACF_FRAME     "Autocorrelation (normalised)"
#define CPAR_PAR_FRAME     "Peak-to-Average Ratio per lag"
#define CPAR_NO_DATA       "No CPAR data"
#define CPAR_LABEL_SIZE    1024
#define CPAR_FONT_SIZE     14.0
#define CPAR_LINE_WIDTH    1.5
#define CPAR_ACF_Y_MAX     1.05
#define CPAR_PEAK_RADIUS   4.0
#define CPAR_GREY_RADIUS   2.5
#define CPAR_TICK_LEN      6.0
#define CPAR_Y_SCALE       1.1
#define CPAR_THRESH_FACTOR 1.2
#define CPAR_PAR_FLOOR     1.0
#define CPAR_STEM_R        0.4
#define CPAR_STEM_G        0.8
#define CPAR_STEM_B        0.9
#define CPAR_GREY_DOT      0.6
#define CPAR_THRESH_RED    0.9
#define CPAR_BG_R          0.06
#define CPAR_BG_G          0.06
#define CPAR_BG_B          0.08

/* higher-order statistics window */
#define HOS_TITLE      "Higher-Order Statistics"
#define HOS_WIDTH      940
#define HOS_HEIGHT     680
#define HOS_SPACING    10
#define HOS_MARGIN     10
#define HOS_BISP_SIZE  480
#define HOS_RIGHT_WIDTH 340
#define HOS_BISP_FRAME "Bispectrum (bicoherence)"
#define HOS_NO_DATA    "No bispectrum data"
#define HOS_LABEL_SIZE 2048
#define HOS_FONT_SIZE  14.0
#define HOS_BG_R       0.06
#define HOS_BG_G       0.06
#define HOS_BG_B       0.08

/* cycle detection + cyclic cumulants pages */
#define CYCLEDET_NO_DATA   "No cycle-detection data"
#define CCC_FRAME_LABEL    "Cyclic Temporal Cumulants (modulation recognition)"
#define CCC_PLACEHOLDER    "Press Run Analysis to compute the cyclic cumulants."
#define CYCLEDET_LABEL_SIZE 1024
#define CYCLEDET_PLOT_H    180

/* distribution / classic / filter pages */
#define SDIST_PLOT_H       160
#define CLASSIC_PLOT_H     160
#define FILTER_PLOT_H      160
#define CLASSIC_ENTRY_WIDTH 90
#define FILTER_DEFAULT_TAPS "127"
#define SDIST_NO_DATA      "No distribution data"
#define CLASSIC_NO_DATA    "No classic-analysis data"
#define FILTER_NO_DATA     "No filter data"
#define FILTER_FINDINGS_SIZE 512
#define PULSE_PLOT_H       160
#define PULSE_NO_DATA      "No data"
#define PULSE_THIN_TICKS   200
/* The envelope/line-spectrum plots are decimated to at most this many
 * polyline points: pulse.c reports the envelope at the full sample rate
 * (millions of points), and replotting every sample on each redraw
 * freezes the UI. */
#define PULSE_PLOT_MAX_POINTS 2048

/* docked analysis pane */
/* The tabbed dock pane starts at a third of the main window width and is
 * freely resizable with the divider; it can also be detached into its own
 * window and re-attached. */
#define DOCK_DEFAULT_FRACTION  (1.0 / 3.0)
#define PAGE_MARGIN        8
#define PAGE_SPACING       6
#define SETTINGS_TAB_LABEL "Settings"
#define DETAILS_TAB_LABEL  "Selection"
#define CYCLIC_TAB_LABEL   "Spectral Correlation"
#define CYCLEDET_TAB_LABEL "Cycle Detection"
#define SS_TAB_LABEL       "Spread-Spectrum Detection"
#define FLUCT_TAB_LABEL    "Fluctuations"
#define CPAR_TAB_LABEL     "CPAR"
#define SDIST_TAB_LABEL    "Distribution Analysis"
#define CLASSIC_TAB_LABEL  "Classic Analysis"
#define FILTER_TAB_LABEL   "Filters"
#define PULSE_TAB_LABEL    "Pulse Trains"
#define HOS_TAB_LABEL      "HOS"
#define RUN_BTN_LABEL      "Run Analysis"
#define RUN_PLACEHOLDER    "Press Run to analyse the selection — or the visible view when nothing is selected."
#define HOS_PAGE_BISP_HEIGHT 340
#define DETAILS_TEXT_SIZE  8192
#define DETAILS_NO_SELECTION "No signal selected.\nClick a highlighted rectangle or drag on the waterfall to select a signal."
#define MIN_RECT_HALF      3

/* settings page */
#define SETTINGS_FRAME_LABEL    "Playback Settings"
#define SETTINGS_HINT           "Changes apply immediately to the loaded file."
#define SETTINGS_CUSTOM_LABEL   "Custom.."
#define SETTINGS_ENTRY_WIDTH    90
#define SETTINGS_TEXT_SIZE      64
#define SETTINGS_GRID_ROW_SPACE 4
#define SETTINGS_GRID_COL_SPACE 8
#define SETTINGS_RATE_PRESETS   8
#define SETTINGS_FFT_PRESETS    6
#define SETTINGS_MIN_FFT        64

/* waterfall length */
#define WF_FRAME_LABEL      "Waterfall"
#define WF_LEN_CUSTOM_LABEL "Custom.."
#define WF_LEN_ENTRY_WIDTH  70
#define WF_LEN_MIN          0.1
#define WF_LEN_PRESETS      7
/* cursor frequency read-out: red text, a few pixels below the top edge */
#define CURSOR_FREQ_MARGIN     14.0
#define CURSOR_FREQ_FONT_SIZE  11.0
#define CURSOR_FREQ_LINE_HEIGHT 22.0   /* gap between the two tags */

/* Sample-rate presets offered by the Settings page (Hz, in order). */
#define SETTINGS_RATE_PRESETS_IQ    8
#define SETTINGS_RATE_PRESETS_AUDIO 11
#define SETTINGS_RATE_MAX           12
#define SETTINGS_FORMAT_MAX         8

static const double settings_rate_presets[SETTINGS_RATE_PRESETS_IQ] = {
    0.25e6, 0.5e6, 1e6, 2e6, 4e6, 8e6, 16e6, 32e6
};

static const char *settings_rate_labels[SETTINGS_RATE_PRESETS_IQ] = {
    "0.25 Msps", "0.5 Msps", "1 Msps", "2 Msps", "4 Msps", "8 Msps",
    "16 Msps", "32 Msps"
};

/* audio sample rates offered when the open file is an audio recording */
static const double settings_rate_presets_audio[SETTINGS_RATE_PRESETS_AUDIO] = {
    8000.0, 11025.0, 16000.0, 22050.0, 32000.0, 44100.0, 48000.0, 88200.0,
    96000.0, 176400.0, 192000.0
};
static const char *settings_rate_labels_audio[SETTINGS_RATE_PRESETS_AUDIO] = {
    "8 kHz", "11.025 kHz", "16 kHz", "22.05 kHz", "32 kHz", "44.1 kHz",
    "48 kHz", "88.2 kHz", "96 kHz", "176.4 kHz", "192 kHz"
};

static const char *settings_format_labels[4] = {
    "Float32 I/Q (gqrx)", "Int16 I/Q", "Int8 I/Q", "Uint8 I/Q"
};

/* audio "data format" entries (source encodings) and their bit depths */
static const char *settings_format_labels_audio[6] = {
    "PCM 8-bit", "PCM 16-bit", "PCM 24-bit", "PCM 32-bit", "Float32",
    "Float64"
};
static const int settings_format_bits_audio[6] = { 8, 16, 24, 32, 32, 64 };

/* The tables currently shown in the Settings combos (rebuilt per source).
 * They are initialised with the IQ defaults so the widgets can be built
 * before a source is configured. */
static double settings_cur_rate_hz[SETTINGS_RATE_MAX] = {
    0.25e6, 0.5e6, 1e6, 2e6, 4e6, 8e6, 16e6, 32e6
};
static const char *settings_cur_rate_labels[SETTINGS_RATE_MAX] = {
    "0.25 Msps", "0.5 Msps", "1 Msps", "2 Msps", "4 Msps", "8 Msps",
    "16 Msps", "32 Msps", NULL, NULL, NULL, NULL
};
static int settings_cur_rate_count = SETTINGS_RATE_PRESETS_IQ;
static const char *settings_cur_format_labels[SETTINGS_FORMAT_MAX] = {
    "Float32 I/Q (gqrx)", "Int16 I/Q", "Int8 I/Q", "Uint8 I/Q"
};
static int settings_cur_format_count = 4;
static bool settings_cur_format_audio = false;

/* Waterfall FFT-size presets (in order) and window-type labels. */
static const int settings_fft_presets[SETTINGS_FFT_PRESETS] = {
    1024, 2048, 4096, 8192, 16384, 32768
};

static const char *settings_fft_labels[SETTINGS_FFT_PRESETS] = {
    "1024", "2048", "4096", "8192", "16384", "32768"
};

static const char *settings_window_labels[4] = {
    "Hann", "Hamming", "Blackman", "Rectangular"
};

/* Estimator selectors for the Spectral Correlation and Spread-Spectrum
 * Detection pages. */


static const char *filter_kind_labels[4] = {
    "Lowpass", "Highpass", "Bandpass", "Bandstop"
};

/* Guards the Settings-page handlers during programmatic initialisation. */
static bool settings_initialising = false;

/* Waterfall-length presets (seconds, in order). */
static const double wf_len_presets[WF_LEN_PRESETS] = {
    1.0, 5.0, 10.0, 15.0, 30.0, 60.0, 120.0
};

static const char *wf_len_labels[WF_LEN_PRESETS] = {
    "1 s", "5 s", "10 s", "15 s", "30 s", "60 s", "120 s"
};

/* Waterfall-length widgets (file-scope: ui.h has no fields for them). */
static GtkWidget *wf_len_combo = NULL;
static GtkWidget *wf_len_entry = NULL;

/* Region behind the live Selection page (the waterfall selection, or the
 * peak-pane frequency window over the visible time range). */
static Region last_details_region;

/* Spread-Spectrum Detection widget groups (shown/hidden per estimator). */

/* Player refresh timer id (file-scope so reopening the player does not
 * stack one timer per open). */
static guint player_timer_id = 0;

/* Last time update_peak_spectrum ran; throttles the playback tick. */
static gint64 last_peak_update = 0;

/* Allocated length of ui->peak_spectrum (follows the cache's FFT size). */
static int peak_alloc_bins = 0;

/* forward declarations: helpers defined below their first call sites. */
static void update_details_page(Ui *ui);
static int selection_hit(const Ui *ui, double x, double y, int w, int h, int *inside);
static void on_settings_filter_changed(GtkWidget *widget, gpointer user_data);
static void update_details_for_region(Ui *ui, const Region *region_in);
static bool dock_has_target(const Ui *ui);
static bool dock_target_region(Ui *ui, Region *r);
static char *utf8_sanitize(const char *in);
static bool selection_contains_px(Ui *ui, double x, double y, int w, int h);
static void wrap_scroll(Ui *ui);
static size_t wf_zero_row_frames(const Ui *ui, int h);
static double wf_stream_period_time(const Ui *ui);
static void update_playhead(Ui *ui);
static void draw_playhead(Ui *ui, cairo_t *cr, int w, int h);
static void toggle_full_height_selection(Ui *ui);
static void clear_selection_state(Ui *ui);
static void set_info(Ui *ui, const char *fmt, ...);
static void update_info_label(Ui *ui);
static void wf_update_from_renderbuf(Ui *ui, int w, int h);
static void lte_hint_apply(Ui *ui, const RankedMatch *ids, size_t n,
                           double measured_bw_hz, const Metrics *m,
                           const ModCandidate *mods, size_t nmods);
static void update_sel_bw_label(Ui *ui);
static void update_range_label(Ui *ui);
static void update_peak_spectrum(Ui *ui);
static void wf_len_init_widgets(Ui *ui);
static bool map_region_pixels(Ui *ui, double fmin, double fmax,
                              double tmin, double tmax, int w, int h,
                              int *x0, int *y0, int *x1, int *y1);

/* --- pixel <-> data-space helpers --------------------------------------- */

/* Frequency (Hz) of a horizontal pixel position (within the visible
 * frequency window). */
/* Human-readable frequency: audio uses Hz/kHz, RF recordings MHz. */
static void format_freq_hz(const Ui *ui, double hz, char *buf, size_t n)
{
    bool audio = (ui != NULL && ui->src != NULL && ui->src->is_audio);

    if (audio) {
        if (fabs(hz) < 1000.0) {
            snprintf(buf, n, "%.1f Hz", hz);
        } else {
            snprintf(buf, n, "%.2f kHz", hz / 1e3);
        }
    } else {
        snprintf(buf, n, "%.3f MHz", hz / HZ_PER_MHZ);
    }
}

/* Audio recordings are displayed like a spectrogram: a logarithmic
 * frequency axis over the audible band (20 Hz .. Nyquist) instead of the
 * linear RF axis. The helpers below own that mapping so clicks, overlays
 * and the drawing itself stay consistent. */
static bool audio_view(const Ui *ui)
{
    return ui->src != NULL && ui->src->is_audio;
}

/* Absolute frequency (Hz) of the left and right edge of the view. */
static double view_lo_hz(const Ui *ui)
{
    double lo = ui->src->center_freq - ui->src->sample_rate / 2.0 +
                ui->freq_offset;

    if (audio_view(ui)) {
        double nyq = ui->src->sample_rate / 2.0;

        if (lo < 20.0) lo = 20.0;
        if (lo > nyq * 0.9) lo = nyq * 0.9;
    }
    return lo;
}

static double view_hi_hz(const Ui *ui)
{
    return view_lo_hz(ui) + ui->visible_bw;
}

/* Pixel column for a frequency in the current view. */
static double x_from_freq(const Ui *ui, double f, int width)
{
    double lo = view_lo_hz(ui);
    double hi = view_hi_hz(ui);

    if (width <= 0 || hi <= lo) return 0.0;
    if (!audio_view(ui)) {
        return (f - lo) / (hi - lo) * (double)width;
    }
    if (f <= lo) return 0.0;
    if (f >= hi) return (double)width;
    return log(f / lo) / log(hi / lo) * (double)width;
}

/* Frequency of a pixel column in the current view. */
static double freq_from_x(const Ui *ui, double x, int width)
{
    double lo = view_lo_hz(ui);
    double hi = view_hi_hz(ui);

    if (width <= 0 || hi <= lo) return lo;
    if (!audio_view(ui)) {
        return lo + (x / (double)width) * (hi - lo);
    }
    if (x <= 0.0) return lo;
    if (x >= (double)width) return hi;
    return lo * pow(hi / lo, x / (double)width);
}

static double freq_at_x(const Ui *ui, double x, int width)
{
    return freq_from_x(ui, x, width);
}

/* Time (seconds from recording start) of a vertical pixel position.
 * The waterfall reads chronologically top->bottom: the TOP row is the
 * oldest time in the window (the view start), the BOTTOM row the newest,
 * so a freshly loaded file starts at the top with its first samples.
 * Wrap-aware: with the looping playhead the view may straddle the file
 * end, so the unwrapped position is reduced modulo the duration. */
static double time_at_y(const Ui *ui, double y, int height)
{
    double t;
    double dur = ui->src->duration;

    if (dur <= 0.0 || height <= 0) {
        return 0.0;
    }
    /* Map WITHOUT wrapping: the waterfall can wrap the file end back to its
     * beginning, and a wrapped time makes a drag rectangle jump to the
     * other side of the cursor (and change height) when the button is
     * released. Positions past the file end clamp to the end instead, so a
     * selection only ever spans real data. */
    t = ui->scroll_position +
        (y / (double)height) * ui->visible_seconds;
    if (t < 0.0) t = 0.0;
    if (t > dur) t = dur;
    return t;
}


/* --- view model helpers -------------------------------------------------- */

/* Update the bottom status readout with the visible window and the stream
 * position. The view loops, so when it crosses the end of the file the
 * numbers are split into the tail and the wrapped head (the loop point)
 * instead of running past the recording. */
static void update_range_label(Ui *ui)
{
    char text[RANGE_LABEL_SIZE];
    double dur;
    double rate;
    double start;
    double end;
    size_t s0;

    if (ui->range_label == NULL || ui->src == NULL) return;
    wrap_scroll(ui);
    dur = ui->src->duration;
    rate = ui->src->sample_rate;
    start = ui->scroll_position;
    end = start + ui->visible_seconds;
    s0 = (size_t)(start * rate);
    if (dur > 0.0 && start >= dur) {
        /* the stream is on the one-row silence marker: the position is
         * about to reset to the beginning of the file */
        snprintf(text, sizeof(text),
                 "loop marker (%.0f zero samples) at %.3f s -> 0.000 s | "
                 "samples %zu (loop)",
                 rate * ui->visible_seconds /
                     (double)(ui->wf_h > 0 ? ui->wf_h : 480),
                 dur, (size_t)(ui->src->total_samples));
    } else if (dur > 0.0 && end > dur) {
        size_t s_end = (size_t)(dur * rate);
        size_t s_head = (size_t)((end - dur) * rate);

        snprintf(text, sizeof(text),
                 "%.3f s - %.3f s + 0.000 - %.3f s | samples %zu - %zu + "
                 "0 - %zu (loop)",
                 start, dur, end - dur, s0, s_end, s_head);
    } else {
        size_t s1 = (size_t)(end * rate);

        snprintf(text, sizeof(text), "%.3f s - %.3f s | samples %zu - %zu",
                 start, end, s0, s1);
    }
    gtk_label_set_text(GTK_LABEL(ui->range_label), text);
}

/*
 * The loop marker: after the last sample of the recording the stream holds
 * exactly ONE pixel row worth of silence (complex zeros, i.e. zero
 * magnitude) before the playback position resets to the beginning. The
 * row's sample count follows the current settings: one pixel row spans
 * visible_seconds / height of the time axis, so it takes
 * sample_rate * visible_seconds / height samples, which is
 * frames_per_second * visible_seconds / height stream frames.
 */
static size_t wf_zero_row_frames(const Ui *ui, int h)
{
    double fps;
    size_t total;
    size_t n;

    if (ui == NULL || ui->cache == NULL) return 0;
    fps = wc_frames_per_second(ui->cache);
    total = wc_total_frames(ui->cache);
    if (fps <= 0.0 || h <= 0 || total == 0 || ui->visible_seconds <= 0.0) {
        return 0;
    }
    n = (size_t)llround(fps * ui->visible_seconds / (double)h);
    return n < 1 ? 1 : n;
}

/* The stream period in frames: the recording plus the zero row. */
static size_t wf_stream_period_frames(const Ui *ui, int h)
{
    return wc_total_frames(ui->cache) + wf_zero_row_frames(ui, h);
}

/* The stream period in seconds (the playback position wraps here). */
static double wf_stream_period_time(const Ui *ui)
{
    double fps;
    int h;

    if (ui == NULL) return 0.0;
    fps = ui->cache != NULL ? wc_frames_per_second(ui->cache) : 0.0;
    if (fps <= 0.0) return ui->src != NULL ? ui->src->duration : 0.0;
    h = ui->wf_h > 0 ? ui->wf_h : 480;
    return (double)wf_stream_period_frames(ui, h) / fps;
}

/*
 * The stream only ever advances: moving forward past the end of the file
 * streams the one-row zero marker and then continues from the beginning
 * (the playback position - the BOTTOM row of the waterfall - resets), while
 * moving BACKWARDS stops at the start of the file (the view is clamped at
 * zero and nothing is prepended from the end).
 *
 * The playback position is the last rendered row, i.e. the bottom edge of
 * the view: the stream resets only once that row reaches the end of the
 * file, so the marker row is appended after the very last renderable row
 * and never appears a screen early.
 */
static void wrap_scroll_dir(Ui *ui, int dir)
{
    double period = wf_stream_period_time(ui);
    double top;

    if (period <= 0.0) {
        ui->scroll_position = 0.0;
        return;
    }
    top = ui->scroll_position;
    if (dir < 0) {
        /* backwards: the view never goes before the start of the file */
        if (top < 0.0) top = 0.0;
        if (top >= period) top = fmod(top, period);
        ui->scroll_position = top;
        return;
    }
    if (dir > 0 && top + ui->visible_seconds >= period) {
        /* the last rendered row reached the end: the playback position
         * resets and the view continues wrapped over the loop seam */
        double bottom = fmod(top + ui->visible_seconds, period);

        top = bottom - ui->visible_seconds;
        if (top < 0.0) top += period;
    } else {
        if (top < 0.0) top = 0.0;
        if (top >= period) top = fmod(top, period);
    }
    ui->scroll_position = top;
}

static void wrap_scroll(Ui *ui)
{
    wrap_scroll_dir(ui, 0);
}

/* Ask the cache worker for a window covering twice the visible range. */
static void request_current_window(Ui *ui)
{
    double fps = wc_frames_per_second(ui->cache);
    size_t current_frame = (size_t)(ui->scroll_position * fps);
    size_t visible_frames = (size_t)(ui->visible_seconds * fps);

    if (visible_frames == 0) {
        visible_frames = 1;
    }
    wc_request(ui->cache, current_frame, CACHE_WINDOW_FACTOR * visible_frames + 1);
}

static void scroll_by(Ui *ui, double delta)
{
    double new_pos = ui->scroll_position + delta;

    if (new_pos == ui->scroll_position) {
        return;
    }
    ui->scroll_position = new_pos;
    wrap_scroll_dir(ui, delta > 0.0 ? 1 : -1);
    request_current_window(ui);
    update_range_label(ui);
    update_info_label(ui);
    update_peak_spectrum(ui);
    gtk_widget_queue_draw(ui->drawing);
    gtk_widget_queue_draw(ui->peak_pane);
}


/* --- screenshot ---------------------------------------------------------- */

static void take_screenshot(Ui *ui)
{
    if (ui->waterfall_surface == NULL) {
        return;
    }
    char filename[FILENAME_BUFFER_SIZE];
    time_t now = time(NULL);
    struct tm *lt = localtime(&now);

    if (lt == NULL) {
        return;
    }
    strftime(filename, sizeof(filename), SCREENSHOT_PREFIX, lt);
    if (cairo_surface_write_to_png(ui->waterfall_surface, filename) == CAIRO_STATUS_SUCCESS) {
        TE15_LOG("Screenshot saved: %s\n", filename);
    } else {
        fprintf(stderr, "Failed to write screenshot: %s\n", filename);
    }
}

/* --- drag selection ------------------------------------------------------ */

/* Turn the anchor/live-endpoint pair into a normalised, clamped Region. */
static void finalize_selection(Ui *ui)
{
    double fmin = MIN(ui->drag_anchor_freq, ui->drag_freq);
    double fmax = MAX(ui->drag_anchor_freq, ui->drag_freq);
    double tmin = MIN(ui->drag_anchor_time, ui->drag_time);
    double tmax = MAX(ui->drag_anchor_time, ui->drag_time);
    double freq_lo = ui->src->center_freq - ui->src->sample_rate / 2.0;
    double freq_hi = ui->src->center_freq + ui->src->sample_rate / 2.0;

    if (fmin < freq_lo) {
        fmin = freq_lo;
    }
    if (fmax > freq_hi) {
        fmax = freq_hi;
    }
    if (tmin < 0.0) {
        tmin = 0.0;
    }
    if (tmax > ui->src->duration) {
        tmax = ui->src->duration;
    }
    ui->selection.freq_min = fmin;
    ui->selection.freq_max = fmax;
    ui->selection.time_min = tmin;
    ui->selection.time_max = tmax;
    ui->has_selection = (fmax > fmin) && (tmax > tmin);
}

static gboolean on_button_press(GtkWidget *widget, GdkEventButton *event, gpointer user_data)
{
    Ui *ui = (Ui *)user_data;
    int width = gtk_widget_get_allocated_width(widget);
    int height = gtk_widget_get_allocated_height(widget);
    double freq;
    double time;

    if (event->button != 1) {
        return FALSE;
    }
    if (width <= 0 || height <= 0) {
        return TRUE;
    }
    freq = freq_at_x(ui, event->x, width);
    time = time_at_y(ui, event->y, height);

    /* remember where the press landed: a CLICK (no drag) inside the
     * selection toggles the whole-file (blue) mode on release */
    ui->sel_press_x = event->x;
    ui->sel_press_y = event->y;
    ui->sel_press_inside = selection_contains_px(ui, event->x, event->y,
                                                 width, height);

    /* Shift+click on the waterfall drives the selection:
     *   - on an edge of the current rect: resize that edge
     *   - inside the rect: keep it; a drag replaces it
     *   - outside the rect: remove the rect, then start a new one
     * Plain drag: pan the view. */
    if ((event->state & GDK_SHIFT_MASK) != 0) {
        int inside = 0;
        int hit = 0;

        if (ui->sel_full_height) {
            /* drawing a new rectangle exits the whole-file mode */
            ui->sel_full_height = false;
        }
        ui->sel_resizing = false;
        ui->drag_anchor_freq = freq;
        ui->drag_anchor_time = time;
        ui->drag_freq = freq;
        ui->drag_time = time;
        if (ui->has_selection) {
            hit = selection_hit(ui, event->x, event->y, width, height,
                                &inside);
        }
        if (hit != 0 && !inside) {
            /* grab one or more edges of the existing rectangle */
            ui->sel_resizing = true;
            ui->sel_resize_edges = hit;
            ui->dragging = true;
            gtk_widget_queue_draw(widget);
            return TRUE;
        }
        if (inside) {
            /* a click inside keeps the selection; dragging replaces it */
            ui->sel_press_inside = true;
        } else if (ui->has_selection) {
            /* Shift+click outside the rectangle removes it */
            clear_selection_state(ui);
            update_details_page(ui);
            update_info_label(ui);
            gtk_widget_queue_draw(widget);
        }
        ui->dragging = true;
    } else {
        ui->panning = true;
        ui->drag_pan_x = event->x;
        ui->drag_pan_y = event->y;
    }
    gtk_widget_queue_draw(widget);
    return TRUE;
}

/*
 * Stationary-hover hint: when the pointer rests over the waterfall for a
 * second a "Shift+Click to select" tag is drawn INSIDE the waterfall, in
 * the same style as the cursor frequency readout (rounded dark backing,
 * bold 11 px text) but in white. Any pointer movement removes it again.
 */
#define HOVER_HINT_TEXT      "Shift+Click to select"
#define HOVER_HINT_DELAY_MS  1000

static void hover_hint_destroy(Ui *ui)
{
    if (ui->hover_hint_visible) {
        ui->hover_hint_visible = false;
        if (ui->drawing != NULL) gtk_widget_queue_draw(ui->drawing);
    }
}

static void hover_hint_show(Ui *ui)
{
    if (ui->drawing == NULL) return;
    ui->hover_hint_visible = true;
    gtk_widget_queue_draw(ui->drawing);
}

/* 2 s stationary callback: show the hint */
/* True when the pointer is currently over the waterfall widget (not just
 * somewhere in the main window): used to confine the hover hint. */
static bool pointer_over_waterfall(Ui *ui)
{
    GdkDisplay *display;
    GdkSeat *seat;
    GdkDevice *pointer;
    GdkWindow *win;
    GdkWindow *under = NULL;
    gint px = 0, py = 0;

    if (ui == NULL || ui->drawing == NULL) return false;
    win = gtk_widget_get_window(ui->drawing);
    if (win == NULL) return false;
    display = gtk_widget_get_display(ui->drawing);
    seat = gdk_display_get_default_seat(display);
    pointer = seat != NULL ? gdk_seat_get_pointer(seat) : NULL;
    if (pointer == NULL) return false;
    under = gdk_device_get_window_at_position(pointer, &px, &py);
    if (under == NULL) return false;
    /* the pointer may be over a child window of the drawing area */
    {
        GdkWindow *w = under;

        while (w != NULL) {
            if (w == win) return true;
            w = gdk_window_get_parent(w);
        }
    }
    return false;
}

static gboolean on_hover_timeout(gpointer data)
{
    Ui *ui = (Ui *)data;

    ui->hover_timeout = 0;
    if (ui->drawing != NULL && gtk_widget_get_visible(ui->drawing) &&
        !ui->dragging && !ui->panning && pointer_over_waterfall(ui)) {
        hover_hint_show(ui);
    }
    return G_SOURCE_REMOVE;
}

/* pointer left the waterfall: cancel the delay and fade the hint out */
static gboolean on_drawing_leave(GtkWidget *widget, GdkEventCrossing *event,
                                 gpointer user_data)
{
    Ui *ui = (Ui *)user_data;

    (void)widget;
    (void)event;
    if (ui->hover_timeout != 0) {
        g_source_remove(ui->hover_timeout);
        ui->hover_timeout = 0;
    }
    /* mouse left the waterfall: the hint and the frequency read-out must
     * disappear at once */
    hover_hint_destroy(ui);
    if (ui->cursor_active) {
        ui->cursor_active = false;
        gtk_widget_queue_draw(widget);
    }
    return FALSE;
}

/* Apply a vertical pan drag (dy pixels over `height`) and clamp the view
 * so it never scrolls past the last spectrogram row. */
static void pan_by_pixels(Ui *ui, double dy, int height)
{
    double step;

    if (height <= 0) return;
    step = dy * (ui->visible_seconds / (double)height);
    ui->scroll_position -= step;
    /* dragging up (dy < 0) advances the stream, dragging down goes back */
    wrap_scroll_dir(ui, step < 0.0 ? 1 : -1);
}

static gboolean on_motion_notify(GtkWidget *widget, GdkEventMotion *event, gpointer user_data)
{
    Ui *ui = (Ui *)user_data;
    int width = gtk_widget_get_allocated_width(widget);
    int height = gtk_widget_get_allocated_height(widget);

    /* track the cursor for the red frequency read-out at the top edge */
    if (width > 0) {
        bool redraw = !ui->cursor_active || fabs(ui->cursor_x - event->x) >= 0.5;
        double f = freq_at_x(ui, event->x, width);

        ui->cursor_active = true;
        ui->cursor_x = event->x;
        if (!redraw && fabs(f - ui->cursor_freq_hz) > 0.5) redraw = true;
        ui->cursor_freq_hz = f;
        if (redraw) gtk_widget_queue_draw(widget);
    }
    /* any pointer movement removes the hint immediately and restarts the
     * 1 s idle delay */
    if (!ui->panning && !ui->dragging) {
        hover_hint_destroy(ui);
        if (ui->hover_timeout != 0) {
            g_source_remove(ui->hover_timeout);
        }
        ui->hover_timeout = g_timeout_add(HOVER_HINT_DELAY_MS,
                                          on_hover_timeout, ui);
    }
    if (ui->panning) {
        double dx = event->x - ui->drag_pan_x;
        double dy = event->y - ui->drag_pan_y;
        double sr = ui->src->sample_rate;
        double fps;

        ui->drag_pan_x = event->x;
        ui->drag_pan_y = event->y;
        if (width > 0 && height > 0) {
            /* Vertical drag scrolls the stream slowly (one visible-window
             * height per drag of the waterfall height); the two-finger
             * swipe is the fast way to scroll. Horizontal drag pans the
             * band: drag right moves toward lower frequencies. */
            pan_by_pixels(ui, dy, height);
            ui->freq_offset -= dx * (ui->visible_bw / width);
            if (ui->freq_offset < 0.0) {
                ui->freq_offset = 0.0;
            }
            if (ui->freq_offset > sr - ui->visible_bw) {
                ui->freq_offset = sr - ui->visible_bw;
            }
        }
        if (ui->playing) {
            /* a long backward drag must not cause a forward jump on
             * release: restart the playback tick from the dragged pos */
            ui->last_play_tick = g_get_monotonic_time();
        }
        fps = wc_frames_per_second(ui->cache);
        wc_request(ui->cache, (size_t)(ui->scroll_position * fps),
                   2 * (size_t)(ui->visible_seconds * fps) + 1);
        update_range_label(ui);
        update_peak_spectrum(ui);
        gtk_widget_queue_draw(ui->drawing);
        gtk_widget_queue_draw(ui->peak_pane);
        return TRUE;
    }
    if (!ui->dragging) {
        return FALSE;
    }
    if (width <= 0 || height <= 0) {
        return TRUE;
    }
    if (ui->sel_resizing) {
        double f = freq_at_x(ui, event->x, width);
        double t = time_at_y(ui, event->y, height);
        double f_lo = ui->src->center_freq - ui->src->sample_rate / 2.0
                    + ui->freq_offset;
        double f_hi = f_lo + ui->visible_bw;
        double dur = ui->src->duration;
        double min_d = 1e-6;

        if (ui->sel_resize_edges & SEL_EDGE_LEFT) {
            ui->selection.freq_min = MIN(MAX(f, f_lo), f_hi - min_d);
        }
        if (ui->sel_resize_edges & SEL_EDGE_RIGHT) {
            ui->selection.freq_max = MAX(MIN(f, f_hi), f_lo + min_d);
        }
        if (ui->sel_resize_edges & SEL_EDGE_TOP) {
            ui->selection.time_min = MIN(MAX(t, 0.0), dur - min_d);
        }
        if (ui->sel_resize_edges & SEL_EDGE_BOTTOM) {
            ui->selection.time_max = MAX(MIN(t, dur), min_d);
        }
        /* dragging an edge past the opposite one flips the rectangle */
        if (ui->selection.freq_min > ui->selection.freq_max) {
            double tmp = ui->selection.freq_min;

            ui->selection.freq_min = ui->selection.freq_max;
            ui->selection.freq_max = tmp;
        }
        if (ui->selection.time_min > ui->selection.time_max) {
            double tmp = ui->selection.time_min;

            ui->selection.time_min = ui->selection.time_max;
            ui->selection.time_max = tmp;
        }
        ui->has_selection = (ui->selection.freq_max > ui->selection.freq_min) &&
                            (ui->selection.time_max > ui->selection.time_min);
        update_sel_bw_label(ui);
        gtk_widget_queue_draw(widget);
        return TRUE;
    }
    ui->drag_freq = freq_at_x(ui, event->x, width);
    ui->drag_time = time_at_y(ui, event->y, height);
    update_sel_bw_label(ui); /* live width of the rectangle being drawn */
    gtk_widget_queue_draw(widget);
    return TRUE;
}

static gboolean on_button_release(GtkWidget *widget, GdkEventButton *event, gpointer user_data)
{
    Ui *ui = (Ui *)user_data;
    int width = gtk_widget_get_allocated_width(widget);
    int height = gtk_widget_get_allocated_height(widget);

    /* track the cursor for the red frequency read-out at the top edge */
    if (width > 0) {
        bool redraw = !ui->cursor_active || fabs(ui->cursor_x - event->x) >= 0.5;
        double f = freq_at_x(ui, event->x, width);

        ui->cursor_active = true;
        ui->cursor_x = event->x;
        if (!redraw && fabs(f - ui->cursor_freq_hz) > 0.5) redraw = true;
        ui->cursor_freq_hz = f;
        if (redraw) gtk_widget_queue_draw(widget);
    }
    /* any pointer movement removes the hint immediately and restarts the
     * 1 s idle delay */
    if (!ui->panning && !ui->dragging) {
        hover_hint_destroy(ui);
        if (ui->hover_timeout != 0) {
            g_source_remove(ui->hover_timeout);
        }
        ui->hover_timeout = g_timeout_add(HOVER_HINT_DELAY_MS,
                                          on_hover_timeout, ui);
    }
    /* A CLICK inside the selection (press and release without dragging)
     * toggles the whole-file (blue) mode. This runs before the panning
     * early-return so it also works for a plain click. */
    if (event->button == 1 && width > 0 && height > 0 &&
        ui->sel_press_inside && !ui->sel_resizing &&
        ui->has_selection &&
        fabs(event->x - ui->sel_press_x) < SEL_MOVE_PX &&
        fabs(event->y - ui->sel_press_y) < SEL_MOVE_PX) {
        /* ignore the second click of a double click */
        if (g_get_monotonic_time() - ui->sel_toggle_us > 250000) {
            ui->sel_toggle_us = g_get_monotonic_time();
            toggle_full_height_selection(ui);
        }
        ui->dragging = false;
        ui->panning = false;
        ui->sel_press_inside = false;
        return TRUE;
    }
    /* A CLICK on empty waterfall space (outside the rectangle, without
     * dragging) removes the selection. This runs before the panning
     * early-return so a plain click works; a drag still pans. */
    if (event->button == 1 && width > 0 && height > 0 &&
        ui->has_selection && !ui->sel_press_inside && !ui->sel_resizing &&
        fabs(event->x - ui->sel_press_x) < SEL_MOVE_PX &&
        fabs(event->y - ui->sel_press_y) < SEL_MOVE_PX) {
        clear_selection_state(ui);
        ui->dragging = false;
        ui->panning = false;
        update_details_page(ui);
        update_info_label(ui);
        set_info(ui, "Selection removed");
        gtk_widget_queue_draw(widget);
        return TRUE;
    }
    if (ui->panning) {
        ui->panning = false;
        return TRUE;
    }
    if (event->button != 1 || !ui->dragging) {
        return FALSE;
    }
    if (ui->sel_resizing) {
        ui->sel_resizing = false;
        ui->dragging = false;
        update_details_page(ui);
        update_info_label(ui);
        gtk_widget_queue_draw(widget);
        return TRUE;
    }
    if (width > 0 && height > 0) {
        ui->drag_freq = freq_at_x(ui, event->x, width);
        ui->drag_time = time_at_y(ui, event->y, height);
    }
    if (ui->sel_press_inside &&
        fabs(event->x - ui->sel_press_x) < SEL_MOVE_PX &&
        fabs(event->y - ui->sel_press_y) < SEL_MOVE_PX) {
        /* a click inside the rectangle: keep it unchanged */
        ui->dragging = false;
        return TRUE;
    }
    finalize_selection(ui);
    ui->dragging = false;
    ui->sel_press_inside = false;
    update_details_page(ui);
    update_info_label(ui);
    gtk_widget_queue_draw(widget);
    return TRUE;
}

/* --- analysis markup helpers -------------------------------------------- */

/*
 * Append formatted text to a NUL-terminated buffer at `used`; returns the
 * new used length (== size once full, after which appends are no-ops).
 */
static size_t stats_append(char *buf, size_t size, size_t used, const char *fmt, ...)
{
    va_list ap;
    int n;

    if (used >= size) {
        return used;
    }
    va_start(ap, fmt);
    n = vsnprintf(buf + used, size - used, fmt, ap);
    va_end(ap);
    if (n < 0) {
        return used;
    }
    used += (size_t)n;
    if (used >= size) {
        used = size;
    }
    return used;
}

/* Guarded demod-status write (the status label is not on the Selection page). */
static void set_demod_status(Ui *ui, const char *msg)
{
    (void)ui;
    if (ui->details_status != NULL) {
        gtk_label_set_text(GTK_LABEL(ui->details_status), msg);
    }
}

/* Release the cached demod buffer (new analysis, reset, or cleanup). */
static void clear_demod(Ui *ui)
{
    free(ui->demod_audio);
    ui->demod_audio = NULL;
    ui->demod_len = 0;
    ui->demod_rate = 0;
    ui->demod_default_name[0] = '\0';
}

/*
 * Compute the demodulated mono audio for the current selection once and
 * cache it in ui->demod_audio (32 kHz float mono, length demod_len),
 * together with the default WAV filename. Returns true when the buffer
 * is ready (already cached, or freshly computed); on failure frees
 * everything and reports the reason through demod_status.
 */
static bool compute_demod(Ui *ui)
{
    const double rate = ui->src->sample_rate;
    const char *mode;
    size_t s0;
    size_t n;
    size_t max_out;
    size_t got = 0;
    double offset;
    float *mix;
    float *mono;
    const char *base;
    char stem[DEMOD_FILENAME_SIZE];
    size_t stem_len;
    char msg[DEMOD_STATUS_SIZE];

    if (ui->demod_audio != NULL) {
        return true; /* already computed; survives until reset/re-analysis */
    }
    if (!ui->metrics.demodulable) {
        set_demod_status(ui, "No audio demodulator available for this signal class.");
        return false;
    }
    s0 = (size_t)(last_details_region.time_min * rate);
    n = (size_t)((last_details_region.time_max - last_details_region.time_min) * rate);
    /* hard cap in samples: 32M complex samples ~= 256 MB mix buffer */
    n = MIN(n, (size_t)DEMOD_MAX_SAMPLES);
    if (s0 >= ui->src->total_samples) {
        set_demod_status(ui, "Selection out of bounds.");
        return false;
    }
    if (s0 + n > ui->src->total_samples) {
        n = ui->src->total_samples - s0;
    }
    if (n == 0) {
        set_demod_status(ui, "Selection too short.");
        return false;
    }

    mix = malloc(2 * n * sizeof(float));
    if (mix == NULL) {
        set_demod_status(ui, "Out of memory.");
        return false;
    }
    offset = ui->analysis.center_freq - ui->src->center_freq;
    /* format-aware read (INT8/INT16/UINT8/FLOAT32), then mixdown in place */
    iq_source_read_complex(ui->src, s0, mix, n);
    metrics_mixdown(mix, n, rate, -offset, mix);

    max_out = (size_t)(n * DEMOD_OUT_RATE / rate) + DEMOD_PAD_SAMPLES;
    mono = malloc(max_out * sizeof(float));
    if (mono == NULL) {
        free(mix);
        set_demod_status(ui, "Out of memory.");
        return false;
    }

    mode = ui->metrics.demod_mode;
    if (strcmp(mode, "fm") == 0) {
        double dev = (ui->metrics.fm_deviation_hz > 0.0)
                         ? ui->metrics.fm_deviation_hz
                         : DEMOD_FALLBACK_DEV;
        got = demod_fm(mix, n, rate, DEMOD_OUT_RATE, dev, mono, max_out);
    } else if (strcmp(mode, "am") == 0) {
        got = demod_am(mix, n, rate, DEMOD_OUT_RATE, mono, max_out);
    } else if (strcmp(mode, "cw") == 0) {
        got = demod_cw(mix, n, rate, DEMOD_OUT_RATE, mono, max_out);
    } else {
        set_demod_status(ui, "Unknown demod mode.");
        free(mix);
        free(mono);
        return false;
    }

    ui->demod_audio = mono;
    ui->demod_len = got;
    ui->demod_rate = (int)DEMOD_OUT_RATE;

    /* default filename: demod_<recording>_<freq>MHz_<t0>s_<mode>.wav */
    base = strrchr(ui->src->path, '/');
    base = (base != NULL) ? base + 1 : ui->src->path;
    snprintf(stem, sizeof(stem), "%s", base);
    stem_len = strlen(stem);
    if (stem_len >= RAW_SUFFIX_LEN &&
        strcmp(stem + stem_len - RAW_SUFFIX_LEN, ".raw") == 0) {
        stem[stem_len - RAW_SUFFIX_LEN] = '\0';
    }
    snprintf(ui->demod_default_name, sizeof(ui->demod_default_name),
             "demod_%s_%.3fMHz_%.2fs_%s.wav",
             stem, ui->analysis.center_freq / HZ_PER_MHZ,
             last_details_region.time_min, mode);

    snprintf(msg, sizeof(msg), "Demodulated: %.1f s audio (%s mode)",
             got / DEMOD_OUT_RATE, mode);
    set_demod_status(ui, msg);

    free(mix);
    return true;
}

/* Format seconds as m:ss.t (e.g. "0:07.3"). */
static void format_time(double s, char *buf, size_t n)
{
    int minutes;
    double secs;

    if (s < 0.0) {
        s = 0.0;
    }
    minutes = (int)(s / 60.0);
    secs = s - minutes * 60.0;
    snprintf(buf, n, "%d:%04.1f", minutes, secs);
}

static void update_player_time_label(Ui *ui)
{
    char cur[PLAYER_TIME_BUFFER];
    char total[PLAYER_TIME_BUFFER];
    char text[2 * PLAYER_TIME_BUFFER + PLAYER_TIME_SEP];

    if (ui->player == NULL || ui->player_time_label == NULL) {
        return;
    }
    format_time(audio_player_position(ui->player), cur, sizeof(cur));
    format_time(audio_player_duration(ui->player), total, sizeof(total));
    snprintf(text, sizeof(text), "%s / %s", cur, total);
    gtk_label_set_text(GTK_LABEL(ui->player_time_label), text);
}

static void toggle_playback(Ui *ui)
{
    if (ui->player == NULL) {
        return;
    }
    if (audio_player_is_playing(ui->player)) {
        audio_player_pause(ui->player);
        gtk_button_set_label(GTK_BUTTON(ui->player_play_btn), PLAY_BTN_LABEL);
    } else if (!audio_player_play(ui->player)) {
        set_demod_status(ui, "Audio output unavailable");
    } else {
        gtk_button_set_label(GTK_BUTTON(ui->player_play_btn), PAUSE_BTN_LABEL);
    }
}

static void on_player_scale_changed(GtkRange *range, gpointer user_data)
{
    Ui *ui = (Ui *)user_data;

    if (ui->player_syncing) {
        return;
    }
    if (ui->player != NULL) {
        audio_player_seek(ui->player, gtk_range_get_value(range));
        update_player_time_label(ui);
    }
}

static void on_player_play_clicked(GtkButton *button, gpointer user_data)
{
    (void)button;
    toggle_playback((Ui *)user_data);
}

static void on_player_destroy(GtkWidget *widget, gpointer user_data)
{
    Ui *ui = (Ui *)user_data;

    if (ui->player_window == widget) {
        ui->player_window = NULL;
    }
    if (ui->player != NULL) {
        audio_player_destroy(ui->player);
        ui->player = NULL;
    }
    ui->player_scale = NULL;
    ui->player_play_btn = NULL;
    ui->player_time_label = NULL;
}

static gboolean player_timer(gpointer user_data)
{
    Ui *ui = (Ui *)user_data;

    if (ui->player == NULL || ui->player_window == NULL) {
        player_timer_id = 0;
        return G_SOURCE_REMOVE;
    }
    if (audio_player_is_playing(ui->player)) {
        ui->player_syncing = true;
        gtk_range_set_value(GTK_RANGE(ui->player_scale),
                            audio_player_position(ui->player));
        ui->player_syncing = false;
        update_player_time_label(ui);
        if (audio_player_finished(ui->player)) {
            gtk_button_set_label(GTK_BUTTON(ui->player_play_btn), PLAY_BTN_LABEL);
        }
    }
    return G_SOURCE_CONTINUE;
}

/* Open (or replace) the demod audio player window. */
static void open_demod_player(Ui *ui)
{
    GtkWidget *vbox;
    GtkWidget *hbox;
    GtkWidget *close_btn;
    double duration;

    if (ui->player_window != NULL) {
        gtk_widget_destroy(ui->player_window); /* handler frees the player */
    }
    ui->player = audio_player_create(ui->demod_audio, ui->demod_len, ui->demod_rate);
    if (ui->player == NULL) {
        set_demod_status(ui, "Out of memory");
        return;
    }
    ui->player_window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_title(GTK_WINDOW(ui->player_window), PLAYER_TITLE);
    gtk_window_set_default_size(GTK_WINDOW(ui->player_window), PLAYER_WIDTH, PLAYER_HEIGHT);
    gtk_window_set_transient_for(GTK_WINDOW(ui->player_window),
                                 GTK_WINDOW(ui->window));
    g_signal_connect(ui->player_window, "destroy", G_CALLBACK(on_player_destroy), ui);

    vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, POPUP_SPACING);
    gtk_widget_set_margin_start(vbox, POPUP_SPACING);
    gtk_widget_set_margin_end(vbox, POPUP_SPACING);
    gtk_widget_set_margin_top(vbox, POPUP_SPACING);
    gtk_widget_set_margin_bottom(vbox, POPUP_SPACING);
    gtk_container_add(GTK_CONTAINER(ui->player_window), vbox);

    duration = audio_player_duration(ui->player);
    if (duration < PLAYER_SCALE_STEP) {
        duration = PLAYER_SCALE_STEP; /* avoid a degenerate scale range */
    }
    ui->player_scale = gtk_scale_new_with_range(GTK_ORIENTATION_HORIZONTAL,
                                                0.0, duration, PLAYER_SCALE_STEP);
    gtk_scale_set_draw_value(GTK_SCALE(ui->player_scale), FALSE);
    g_signal_connect(ui->player_scale, "value-changed", G_CALLBACK(on_player_scale_changed), ui);
    gtk_box_pack_start(GTK_BOX(vbox), ui->player_scale, FALSE, FALSE, 0);

    hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, POPUP_SPACING);
    ui->player_play_btn = gtk_button_new_with_label(PLAY_BTN_LABEL);
    g_signal_connect(ui->player_play_btn, "clicked", G_CALLBACK(on_player_play_clicked), ui);
    gtk_box_pack_start(GTK_BOX(hbox), ui->player_play_btn, FALSE, FALSE, 0);
    ui->player_time_label = gtk_label_new(PLAYER_TIME_INITIAL);
    gtk_box_pack_start(GTK_BOX(hbox), ui->player_time_label, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(vbox), hbox, FALSE, FALSE, 0);

    close_btn = gtk_button_new_with_label(POPUP_CLOSE_LABEL);
    g_signal_connect_swapped(close_btn, "clicked", G_CALLBACK(gtk_widget_destroy),
                             ui->player_window);
    gtk_box_pack_start(GTK_BOX(vbox), close_btn, FALSE, FALSE, 0);

    if (player_timer_id != 0) {
        g_source_remove(player_timer_id);
    }
    player_timer_id = g_timeout_add(PLAYER_TIMER_MS, player_timer, ui);

    update_player_time_label(ui);
    gtk_widget_show_all(ui->player_window);
}

/* Play/Save Demod are no longer part of the Selection page; the handlers
 * (and the audio player they drive) are kept for a future demod page. */
static void G_GNUC_UNUSED on_play_demod_clicked(GtkButton *button, gpointer user_data)
{
    Ui *ui = (Ui *)user_data;

    (void)button;
    if (!compute_demod(ui)) {
        return;
    }
    open_demod_player(ui);
}

static void G_GNUC_UNUSED on_save_demod_clicked(GtkButton *button, gpointer user_data)
{
    Ui *ui = (Ui *)user_data;
    GtkWidget *dialog;
    gint response;

    (void)button;
    if (!compute_demod(ui)) {
        return;
    }
    dialog = gtk_file_chooser_dialog_new(SAVE_DIALOG_TITLE,
                                         GTK_WINDOW(ui->window),
                                         GTK_FILE_CHOOSER_ACTION_SAVE,
                                         CANCEL_BTN_LABEL, GTK_RESPONSE_CANCEL,
                                         SAVE_BTN_LABEL, GTK_RESPONSE_ACCEPT,
                                         NULL);
    gtk_file_chooser_set_current_name(GTK_FILE_CHOOSER(dialog), ui->demod_default_name);
    response = gtk_dialog_run(GTK_DIALOG(dialog));
    if (response == GTK_RESPONSE_ACCEPT) {
        char *path = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dialog));

        if (path != NULL) {
            char msg[DEMOD_STATUS_SIZE];

            if (wav_write(path, ui->demod_audio, ui->demod_len, ui->demod_rate)) {
                snprintf(msg, sizeof(msg), "Saved: %s", path);
                TE15_LOG("%s\n", msg);
            } else {
                snprintf(msg, sizeof(msg), "Failed to write %s", path);
                TE15_LOG("%s\n", msg);
            }
            set_demod_status(ui, msg);
            g_free(path);
        }
    }
    gtk_widget_destroy(dialog);
}

/*
 * Build the Selection Details markup (all sections) into `buf`:
 * Selection Information, Signal Statistics, Signal Characterization,
 * Extended Measurements, Detected Modulation Class and Most Likely
 * Matches — the same content the old popup showed.
 */
static void build_details_markup(Ui *ui, const Region *r, char *buf, size_t size)
{
    const Analysis *a = &ui->analysis;
    const Metrics *m = &ui->metrics;
    bool have_metrics = (m->occupied_bw_hz > 0.0) || (m->mod_class[0] != '\0');
    size_t used = 0;

    {
        char flo[40], fhi[40], fcen[40], fbw[40];

        format_freq_hz(ui, r->freq_min, flo, sizeof(flo));
        format_freq_hz(ui, r->freq_max, fhi, sizeof(fhi));
        format_freq_hz(ui, a->center_freq, fcen, sizeof(fcen));
        if (ui->src != NULL && ui->src->is_audio) {
            snprintf(fbw, sizeof(fbw), "%.2f Hz",
                     a->bandwidth);
            if (a->bandwidth >= 1000.0) {
                snprintf(fbw, sizeof(fbw), "%.2f kHz",
                         a->bandwidth / 1e3);
            }
        } else {
            snprintf(fbw, sizeof(fbw), "%.2f kHz",
                     a->bandwidth / HZ_PER_KHZ);
        }
    used = stats_append(buf, size, used,
        "<b>Selection Information</b>\n"
        "Frequency: %s - %s\n"
        "Bandwidth: %s\n"
        "Center Frequency: %s\n"
        "Time: %.2f - %.2f s\n"
        "Duration: %.2f seconds\n"
        "\n"
        "<b>Signal Statistics</b>\n"
        "Entropy: %.4f bits\n"
        "Pulse Repetition Frequency: %.2f Hz\n"
        "Average Power: %.2f dB\n"
        "Peak Power: %.2f dB\n"
        "\n"
        "<b>Signal Characterization</b>\n"
        "Signal Type: %s\n"
        "Modulation: %s",
        flo, fhi, fbw, fcen,
        r->time_min,
        r->time_max,
        a->duration,
        a->entropy,
        a->prf,
        a->avg_power,
        a->max_power,
        a->signal_type,
        a->modulation);
    }
    if (have_metrics) {
        used = stats_append(buf, size, used,
            "\n<b>Extended Measurements</b>\n"
            "Occupied Bandwidth: %.1f kHz\n"
            "SNR: %.1f dB\n"
            "Crest Factor (PAPR): %.1f dB\n"
            "Spectral Flatness: %.2f\n"
            "AM Depth: %.0f %%\n"
            "FM Deviation: %.1f kHz\n"
            "Symbol/Envelope Rate: %.1f Hz (conf %.0f %%)\n"
            "Duty Cycle: %.1f %%\n"
            "FSK Levels: %d",
            m->occupied_bw_hz / HZ_PER_KHZ,
            m->snr_db,
            m->papr_db,
            m->spectral_flatness,
            m->am_depth * 100.0,
            m->fm_deviation_hz / HZ_PER_KHZ,
            m->sym_rate_hz,
            m->sym_rate_conf * 100.0,
            m->duty_cycle * 100.0,
            m->fsk_levels);
    } else {
        used = stats_append(buf, size, used,
            "\n<b>Extended Measurements</b>\n"
            "Occupied Bandwidth: n/a\n"
            "SNR: n/a\n"
            "Crest Factor (PAPR): n/a\n"
            "Spectral Flatness: n/a\n"
            "AM Depth: n/a\n"
            "FM Deviation: n/a\n"
            "Symbol/Envelope Rate: n/a\n"
            "Duty Cycle: n/a\n"
            "FSK Levels: n/a");
    }
    used = stats_append(buf, size, used, "\n<b>Most Likely Matches</b>");
    if (ui->ranked_count == 0) {
        used = stats_append(buf, size, used, "\nNo confident matches");
    } else {
        for (size_t i = 0; i < ui->ranked_count; i++) {
            used = stats_append(buf, size, used,
                                "\n%d. %s [%s]\nbase %d, agreement %+d, total %d",
                                (int)(i + 1),
                                ui->ranked[i].signal->name,
                                ui->ranked[i].signal->band,
                                ui->ranked[i].base_confidence,
                                ui->ranked[i].agreement,
                                ui->ranked[i].total);
        }
    }
}

/*
 * Refresh the analysis for the current selection. The Selection page no
 * longer displays the live stats markup/preview (it only shows the
 * parameter-estimation report), but running the analysis here keeps the
 * reference parameters for Analyze up to date. Every widget update is
 * NULL-guarded because those widgets are not created.
 */
/* True when the Selection page is the notebook's current page. The live
 * analysis only runs while that tab is visible. */
static bool demod_tab_active(const Ui *ui)
{
    GtkWidget *scroller;
    gint cur;

    if (ui->details_page == NULL || ui->dock_pane == NULL) {
        return false;
    }
    scroller = gtk_widget_get_parent(ui->details_page);
    if (scroller == NULL) {
        return false;
    }
    cur = gtk_notebook_get_current_page(GTK_NOTEBOOK(ui->dock_pane));
    if (cur < 0) {
        return false;
    }
    return gtk_notebook_get_nth_page(GTK_NOTEBOOK(ui->dock_pane), cur) == scroller;
}

/* Deferred live-Selection refresh: set when the selection/zoom changed
 * while the Selection tab was inactive; consumed when it becomes current. */
static bool details_page_dirty = false;

static void on_dock_switch(GtkNotebook *notebook, GtkWidget *page,
                           guint page_num, gpointer user_data)
{
    Ui *ui = (Ui *)user_data;

    (void)notebook;
    (void)page;
    (void)page_num;
    /* the Selection tab is the only live one: catch it up when it becomes
     * current; every other tab analyses only via its Run button */
    if (details_page_dirty && demod_tab_active(ui)) {
        update_details_page(ui);
    }
}

static void update_details_page(Ui *ui)
{
    Region r;

    if (!demod_tab_active(ui)) {
        /* not the current tab: defer the re-analysis until it is shown;
         * other tabs never re-analyse on their own */
        details_page_dirty = true;
        return;
    }
    details_page_dirty = false;
    if (!dock_has_target(ui)) {
        /* The live-stats widgets are no longer displayed on the
         * Selection page (it only shows the analysis report), but the
         * analysis state is still reset so Analyze never uses a stale
         * reference. */
        if (ui->details_label != NULL) {
            gtk_label_set_text(GTK_LABEL(ui->details_label),
                               DETAILS_NO_SELECTION);
        }
        if (ui->details_status != NULL) {
            gtk_label_set_text(GTK_LABEL(ui->details_status), "");
        }
        return;
    }
    if (!dock_target_region(ui, &r)) {
        return;
    }
    update_details_for_region(ui, &r);
}

/*
 * Run the full analysis flow for an explicit region. The Selection page
 * itself only displays the analysis report (the old stats markup /
 * preview widgets are gone), but the analysis/metrics/ranking computed
 * here feed the Analyze job.
 * The region is normalised defensively.
 */
static void update_details_for_region(Ui *ui, const Region *region_in)
{
    Region region = *region_in;
    char text[DETAILS_TEXT_SIZE];
    bool metrics_ok;

    /* a newly analysed selection invalidates the red-edge marker of the
     * previously clicked detection */
    if (!region_normalize(&region, ui->src) ||
        !analysis_run(ui->src, &region, &ui->analysis)) {
        /* analysis_run logs its own failure */
        if (ui->details_label != NULL) {
            gtk_label_set_text(GTK_LABEL(ui->details_label),
                               "Analysis failed for this selection.");
        }
        if (ui->details_status != NULL) {
            gtk_label_set_text(GTK_LABEL(ui->details_status), "");
        }
        return;
    }
    clear_demod(ui); /* the cached demod buffer belongs to the old analysis */
    metrics_ok = metrics_compute(ui->src, &region, &ui->metrics);
    if (!metrics_ok) {
        memset(&ui->metrics, 0, sizeof(ui->metrics));
    }
    ui->ranked_count = signals_rank(ui->analysis.center_freq,
                                    ui->analysis.bandwidth,
                                    metrics_ok ? &ui->metrics : NULL,
                                    ui->ranked,
                                    sizeof(ui->ranked) / sizeof(ui->ranked[0]));
    /* a synchronous analysis also seeds the LTE demodulator */
    lte_hint_apply(ui, ui->ranked, ui->ranked_count, ui->analysis.bandwidth,
                   metrics_ok ? &ui->metrics : NULL, NULL, 0);

    last_details_region = region;
    if (ui->details_label != NULL) {
        build_details_markup(ui, &region, text, sizeof(text));
        gtk_label_set_markup(GTK_LABEL(ui->details_label), text);
    }
    if (ui->details_status != NULL) {
        gtk_label_set_text(GTK_LABEL(ui->details_status), "");
    }
}

#define REPORT_TOTAL        11
/* upper bound of progress steps: the DAG phase counts every executed
 * analysis and the report blocks advance once each */
#define REPORT_TOTAL_MAX    24
#define REPORT_SAVE_TITLE   "Save analysis report"
#define REPORT_FILTER_NAME  "Text files (*.txt)"
#define REPORT_FILTER_GLOB  "*.txt"
#define REPORT_SAVE_LABEL   "Save"
#define REPORT_CANCEL_LABEL "Cancel"
#define DEMOD_UPDATE_LABEL  ANALYZE_BTN_LABEL
/* A "stall" now means the worker thread stopped bumping its heartbeat
 * (a genuinely hung analysis, not merely a slow one - a single stage on a
 * wide selection can take longer than 20 s). A slow stage never cancels
 * the job; only the absolute hard limit does, and even then a partial
 * report is produced. */
#define JOB_STALL_US        (90 * G_USEC_PER_SEC)
#define JOB_HARD_LIMIT_US   (10 * 60 * G_USEC_PER_SEC)

/* --- async Analyze worker (thread + idle delivery) ----------------------- */

typedef struct {
    Ui *ui;                 /* worker reads ONLY ui->src (thread-safe mmap) */
    Region region;          /* the selection, normalised                  */
    volatile int progress;  /* analyses finished so far                   */
    volatile int total;     /* REPORT_TOTAL_MAX upper bound               */
    char *report;           /* plain-text report (heap)                   */
    char *report_md;        /* Markdown presentation of the same document */
    char *report_html;      /* HTML presentation of the same document     */
    volatile int cancelled; /* set by the main thread on reset/cleanup    */
    volatile int done;      /* worker finished; deliver queued            */
    /* UI settings captured on the MAIN thread (the worker must not touch
     * any Ui field besides src). */
    double settings_rate;
    IqFormat settings_format;
    double settings_center;
    int settings_fft_size;
    DspWindow settings_window;
    /* Filters page settings, likewise captured on the main thread. */
    int filter_kind;
    double filter_f1;
    double filter_f2;
    int filter_taps;
    bool filter_valid;
    /* Selection-page "Analyze" mode: the worker fills the page's report
     * view and ranking instead of opening a window. The worker stores the
     * modulation/ID ranking in job-local arrays. */
    bool update_only;
    ModCandidate job_mods[12];
    size_t job_mod_count;
    RankedMatch job_ids[5];
    size_t job_ids_count;
    /* baseline analysis measurements, used to seed the LTE demodulator */
    double measured_bw_hz;
    double measured_center_hz;
    char best_notes[1024];
    int anomaly;
    char anomaly_explain[512];
    /* surrogate-signal spec for Selection > Synthesize */
    SynthSpec synth;
    bool synth_valid;
    /* DAG execution diagnostics: the stage currently running (so a stall
     * names it) and a heartbeat bumped by a helper thread, which lets the
     * watchdog distinguish "slow stage" from "truly stuck worker". */
    const char *volatile stage;
    volatile int heartbeat;
    gint64 started_us;
    int stages_done;
} ReportJob;

static ReportJob *current_report_job = NULL;

/* Stall watchdog state (main thread). */
static gint64 job_last_progress_us = 0;
static int job_last_progress_val = -1;
static int job_last_heartbeat = -1;
static gint64 job_last_heartbeat_us = 0;
static int job_stall_reported = 0;
static const char *job_stage_shown = NULL; /* label shows this stage */

/* Selection page Save button (the report text itself lives in
 * ui->report_text). */
static GtkWidget *report_save_btn = NULL;

static gboolean report_deliver(gpointer data);
static void report_show_selected_format(Ui *ui);
static gboolean job_discard(gpointer data);







static void job_advance(ReportJob *job, int *done, const char *next_stage,
                        gint64 *stage_start);

/* Report progress hooks: the Reporter reports its stages through the
 * worker's usual job_advance/cancelled mechanism. */
static void report_hook_advance(void *ctx, const char *label)
{
    ReportJob *job = (ReportJob *)ctx;
    static gint64 hook_stage_start;

    job_advance(job, &job->stages_done, label, &hook_stage_start);
}

static bool report_hook_cancelled(void *ctx)
{
    ReportJob *job = (ReportJob *)ctx;

    return job->cancelled != 0;
}

/* ------------------------------------------------------------------ */
/* Phase 1: analysis envelopes (result structs -> AnalysisResult)      */
/* ------------------------------------------------------------------ */

/* Begin a DAG node: mark it running, start its timer and add its
 * envelope (with the registry's default settings). */
static AnalysisResult *node_begin(ArSet *set, DagRun *dag, const char *id,
                                  int node, gint64 *t0)
{
    const ArAnalysisDesc *desc = aregistry_find(id);
    AnalysisResult *ar;

    dag_begin(dag, node);
    *t0 = g_get_monotonic_time();
    ar = ar_add(set, id, desc != NULL ? desc->title : id, node);
    if (ar != NULL) aregistry_apply_settings(ar);
    return ar;
}

/* End a DAG node: executed / failed, or skipped with a reason. */
static void node_end(AnalysisResult *ar, DagRun *dag, int node, gint64 t0,
                     bool ok, const char *skip_why)
{
    double elapsed = skip_why != NULL
                         ? 0.0
                         : (double)(g_get_monotonic_time() - t0) / 1e6;

    if (ar == NULL) return;
    ar->elapsed_s = elapsed;
    if (skip_why != NULL) {
        ar_set_status(ar, AR_SKIPPED_USER, skip_why);
        dag_skip(dag, node, skip_why);
    } else if (ok) {
        ar_set_status(ar, AR_EXECUTED, NULL);
        dag_end(dag, node, true, elapsed);
    } else {
        ar_set_status(ar, AR_FAILED, NULL);
        dag_end(dag, node, false, elapsed);
    }
}

/* Record a node that the DAG decided not to run. */
static void node_skip(ArSet *set, DagRun *dag, const char *id, int node,
                      const char *why)
{
    const ArAnalysisDesc *desc = aregistry_find(id);
    AnalysisResult *ar = ar_add(set, id, desc != NULL ? desc->title : id,
                                node);

    if (ar != NULL) node_end(ar, dag, node, g_get_monotonic_time(), false, why);
}

/* ------------------------------------------------------------------ */
/* DAG execution diagnostics                                           */
/* ------------------------------------------------------------------ */

/* advance the stage counter and log the stage with its wall time */
static void job_advance(ReportJob *job, int *done, const char *next_stage,
                        gint64 *stage_start)
{
    gint64 now = g_get_monotonic_time();

    g_atomic_int_inc(&job->heartbeat); /* keep the watchdog alive */
    job->stages_done = *done;
    if (te15_verbose()) fprintf(stderr, "te15: DAG stage %d/%d '%s' completed in %.2f s\n",
            *done + 1, REPORT_TOTAL_MAX,
            next_stage != NULL ? next_stage : "?", 
            stage_start != NULL ? (double)(now - *stage_start) / 1e6 : 0.0);
    (*done)++;
    job->stage = next_stage;
    if (stage_start != NULL) *stage_start = now;
    g_atomic_int_set(&job->progress, *done);
}

/* Heartbeat helper thread: bumps job->heartbeat every 250 ms while the
 * worker is alive. The progress counter alone cannot tell "a long stage"
 * from "a hung worker"; the heartbeat can, because a stalled worker thread
 * stops bumping even if the process is responsive. */
typedef struct {
    ReportJob *job;
    volatile int stop;
} HeartbeatArgs;

static gpointer heartbeat_thread(gpointer data)
{
    HeartbeatArgs *hb = (HeartbeatArgs *)data;

    while (!hb->stop) {
        g_atomic_int_inc(&hb->job->heartbeat);
        g_usleep(250000);
    }
    return NULL;
}

static gpointer report_worker(gpointer data)
{
    ReportJob *job = (ReportJob *)data;
    const IqSource *src = job->ui->src;
    int done = 0;

    /* job-local result structs — ui-> result structs are never touched */
    Analysis analysis;
    Metrics metrics;
    RankedMatch ranked[3];
    CyclicResult cyclic;
    SscaResult ssca;
    CycleDetResult cycledet;
    FluctResult fluct;
    CparResult cpar;
    HosResult hos;
    CccResult ccc;
    ClassicResult classic;
    PulseResult pulse;
    IntraBurstResult intraburst;
    SpreadCandidate spread[6];
    size_t spread_n = 0;
    FilterResult filt;

    /* per-analysis success flags (for the consensus summary) */
    bool ok_fam = false;
    bool ok_ssca = false;
    bool ok_cycledet = false;
    bool ok_fluct = false;
    bool ok_cpar = false;
    bool ok_hos = false;
    bool ok_ccc = false;
    bool ok_classic = false;
    bool ok_pulse = false;

    /* DAG execution diagnostics: a heartbeat thread lets the watchdog tell
     * a slow stage from a hung worker, and `stage` names what is running */
    HeartbeatArgs hb;
    GThread *hb_thread = NULL;
    gint64 stage_start = g_get_monotonic_time();

    /* Phase 1 foundation: one envelope per analysis (arena-owned) and the
     * per-run DAG status. The Director reads these after the DAG. */
    ArArena *arena = ar_arena_new();
    ArSet envelopes;
    DagRun *dag = dag_run_new();
    gint64 node_t0 = g_get_monotonic_time();

    ar_set_init(&envelopes, arena);
    dag_reset(dag);

    hb.job = job;
    hb.stop = 0;
    job->started_us = stage_start;
    job->stage = "preparing the common source";
    job->heartbeat = 0;
    hb_thread = g_thread_new("analysis-heartbeat", heartbeat_thread, &hb);

    /* DAG execution state: which nodes ran and why the others did not */
    bool ok_a0 = false;
    bool ran_a1 = false, ran_a2 = false, ran_a3 = false;
    bool ran_a4 = false, ran_a5 = false, ran_a6 = false;
    bool ran_a7 = false, ran_a8 = false, ran_a9 = false;
    bool ok_intraburst = false;
    const char *skip_a4 = "";
    const char *skip_a5 = "";
    const char *skip_a6 = "";
    const char *skip_a7 = "";
    const char *skip_a8 = "";
    const char *skip_a9 = "";
    /* DAG decision outcomes (recorded for the report) */
    bool d4_have_cf = false, d5_no_cycles = false, d6_wideband = false;
    bool validated_cycles = false;

    memset(&analysis, 0, sizeof(analysis));
    memset(&metrics, 0, sizeof(metrics));
    memset(ranked, 0, sizeof(ranked));
    memset(&cyclic, 0, sizeof(cyclic));
    memset(&ssca, 0, sizeof(ssca));
    memset(&cycledet, 0, sizeof(cycledet));
    memset(&fluct, 0, sizeof(fluct));
    memset(&cpar, 0, sizeof(cpar));
    memset(&hos, 0, sizeof(hos));
    memset(&ccc, 0, sizeof(ccc));
    memset(&classic, 0, sizeof(classic));
    memset(&pulse, 0, sizeof(pulse));
    memset(&intraburst, 0, sizeof(intraburst));
    memset(spread, 0, sizeof(spread));
    memset(&filt, 0, sizeof(filt));

    /* --- common prepared source (single source of truth) -------------
     * One full-band slice of the selection's time range is prepared once:
     * mixed at the selection centre and band-filtered from the wideband
     * context. Every analysis below reads this buffer (via a
     * pseudo-source), so no analysis re-extracts or re-filters data. */
    Sot sot;
    const IqSource *s = src;

    memset(&sot, 0, sizeof(sot));
    {
        AnalysisResult *ar = node_begin(&envelopes, dag, "sot", DAG_SOT,
                                        &node_t0);
        bool ok = sot_prepare(src, &job->region, &sot) != 0;

        if (ok && ar != NULL) {
            apub_sot(ar, sot.pass_hz, sot.stop_hz, sot.gain,
                     sot.noise_floor_db, sot.n, sot.view.sample_rate);
            apub_set_confidence(ar, 1.0);
        }
        node_end(ar, dag, DAG_SOT, node_t0, ok,
                 ok ? NULL : "could not prepare the common analysis source");
        if (ok) {
            s = &sot.view;
        }
    }
    /* the prepared source's time axis starts at zero: analyses get the
     * selection rebased onto it (the frequency axis is unchanged) */
    Region sreg = job->region;
    if (sot.valid) {
        sreg.time_min = job->region.time_min - sot.region.time_min;
        sreg.time_max = job->region.time_max - sot.region.time_min;
        if (sreg.time_min < 0.0) sreg.time_min = 0.0;
        if (sreg.time_max > sot.view.duration) {
            sreg.time_max = sot.view.duration;
        }
    }

    /* A0 baseline context: analysis + metrics + system ranking (computed
     * here, reported in the SIGNAL CHARACTERISATION synthesis) */
    {
        bool ok = region_normalize(&job->region, src) &&
                  analysis_run(s, &sreg, &analysis);

        ok_a0 = ok;

        if (ok) {
            bool mok = metrics_compute(s, &sreg, &metrics);

            if (mok) {
                (void)signals_rank(analysis.center_freq, analysis.bandwidth,
                                   &metrics, ranked,
                                   sizeof(ranked) / sizeof(ranked[0]));
            }
        }
        {
            AnalysisResult *ar;

            ar = ar_add(&envelopes, "analysis", "Baseline analysis", DAG_A0);
            if (ar != NULL) {
                aregistry_apply_settings(ar);
                apub_analysis(ar, &analysis);
                apub_set_confidence(ar, ok ? 1.0 : 0.0);
                ar_set_status(ar, ok ? AR_EXECUTED : AR_FAILED,
                              ok ? NULL : "baseline analysis failed");
            }
            ar = ar_add(&envelopes, "metrics", "Signal metrics", DAG_A0);
            if (ar != NULL) {
                aregistry_apply_settings(ar);
                apub_metrics(ar, &metrics);
                apub_set_confidence(ar, 1.0);
            }
            dag_end(dag, DAG_A0, ok, 0.0);
        }
        job_advance(job, &done, "baseline analysis (analysis + metrics + ranking)", &stage_start);
    }
    if (job->cancelled) {
        goto cancelled_out;
    }

    /* --- DAG EXECUTION PHASE ---------------------------------------------
     * The analyses run per the user-specified DAG:
     *   A0 -> {A1, A2, A3, D6} -> (A4 | A5) -> A6 -> (A8 | A7)
     * A0 (the block above) supplies the baseline context; the cheap
     * exploratory probes A1/A2/A3 run next; their outcomes decide whether
     * the cycle-frequency search (A4 FAM or A5 SSCA for wideband inputs),
     * the detector validation stage (A6), the cyclic-cumulant stage (A8)
     * and the ACF-fluctuation diagnostic (A7) are warranted. Every node
     * not executed records its rationale for the report. */
    ran_a1 = true;
    {
        AnalysisResult *ar = node_begin(&envelopes, dag, "classic", DAG_A1,
                                        &node_t0);

        ok_classic = classic_analyze(s, &sreg, 0.0, &classic);
        if (ar != NULL) {
            apub_classic(ar, &classic);
            apub_set_confidence(ar, (ok_classic && classic.period_hz > 0.0)
                                        ? 1.0 : 0.3);
        }
        node_end(ar, dag, DAG_A1, node_t0, ok_classic, NULL);
    }
    job_advance(job, &done, "classic analysis", &stage_start);
    if (job->cancelled) {
        goto cancelled_out;
    }
    ran_a2 = true;
    {
        AnalysisResult *ar = node_begin(&envelopes, dag, "pulse", DAG_A2,
                                        &node_t0);

        ok_pulse = pulse_analyze(s, &sreg, &pulse);
        if (ar != NULL) {
            apub_pulse(ar, &pulse);
            apub_set_confidence(ar, ok_pulse
                                        ? (pulse.credible ? 1.0 : 0.35) : 0.0);
        }
        node_end(ar, dag, DAG_A2, node_t0, ok_pulse, NULL);
    }
    job_advance(job, &done, "pulse-train analysis", &stage_start);
    if (job->cancelled) {
        goto cancelled_out;
    }
    ran_a3 = true;
    {
        AnalysisResult *ar = node_begin(&envelopes, dag, "hos", DAG_A3,
                                        &node_t0);

        ok_hos = hos_analyze(s, &sreg, &hos);
        if (ar != NULL) {
            apub_hos(ar, &hos);
            apub_set_confidence(ar, ok_hos ? 1.0 : 0.0);
        }
        node_end(ar, dag, DAG_A3, node_t0, ok_hos, NULL);
    }
    job_advance(job, &done, "higher-order statistics", &stage_start);
    if (job->cancelled) {
        goto cancelled_out;
    }

    /* decisions after A0..A3 */
    d6_wideband = (metrics.spectral_flatness > 0.45 &&
                   metrics.papr_db > 7.0) ||
                  (strstr(analysis.modulation, "QAM") != NULL) ||
                  (strstr(analysis.modulation, "OFDM") != NULL) ||
                  (metrics.occupied_bw_hz > 0.2 * src->sample_rate);

    if (d6_wideband) {
        /* D6 yes: SSCA is the wideband cycle-frequency search */
        ran_a5 = true;
        {
            AnalysisResult *ar = node_begin(&envelopes, dag, "ssca", DAG_A5,
                                            &node_t0);

            ok_ssca = ssca_analyze(s, &sreg, &ssca);
            if (ar != NULL) {
                apub_ssca(ar, &ssca);
                apub_set_confidence(ar, ok_ssca
                                            ? (ssca.cf_count > 0 ? 1.0 : 0.4)
                                            : 0.0);
            }
            node_end(ar, dag, DAG_A5, node_t0, ok_ssca, NULL);
        }
        job_advance(job, &done, "SSCA", &stage_start);
        if (job->cancelled) {
            goto cancelled_out;
        }
        d4_have_cf = ok_ssca && (ssca.cf_count > 0);
        skip_a4 = "wideband/OFDM-like input: the heavier SSCA search covers "
                  "the cycle frequencies, so the conventional FAM estimate "
                  "was not needed";
        node_skip(&envelopes, dag, "cyclic", DAG_A4, skip_a4);
    } else {
        /* D6 no: conventional cyclostationary analysis (A4) */
        ran_a4 = true;
        {
            AnalysisResult *ar = node_begin(&envelopes, dag, "cyclic", DAG_A4,
                                            &node_t0);

            ok_fam = cyclic_analyze_ex(s, &sreg, false, &cyclic);
            if (ar != NULL) {
                apub_cyclic(ar, &cyclic);
                apub_set_confidence(ar, ok_fam
                                            ? (cyclic.peak_count > 0 ? 1.0
                                                                     : 0.4)
                                            : 0.0);
            }
            node_end(ar, dag, DAG_A4, node_t0, ok_fam, NULL);
        }
        job_advance(job, &done, "FAM", &stage_start);
        if (job->cancelled) {
            goto cancelled_out;
        }
        d4_have_cf = ok_fam && cyclic.peak_count > 0;
        if (!d4_have_cf) {
            /* D4 no -> D5 yes: no significant cycles in FAM, the signal
             * may be wideband/noise-like: run the SSCA search */
            ran_a5 = true;
            {
                AnalysisResult *ar = node_begin(&envelopes, dag, "ssca",
                                                DAG_A5, &node_t0);

                ok_ssca = ssca_analyze(s, &sreg, &ssca);
                if (ar != NULL) {
                    apub_ssca(ar, &ssca);
                    apub_set_confidence(ar, ok_ssca
                                                ? (ssca.cf_count > 0 ? 1.0
                                                                     : 0.4)
                                                : 0.0);
                }
                node_end(ar, dag, DAG_A5, node_t0, ok_ssca, NULL);
            }
            job_advance(job, &done, "SSCA", &stage_start);
            if (job->cancelled) {
                goto cancelled_out;
            }
            d4_have_cf = d4_have_cf ||
                         (ok_ssca && (ssca.cf_count > 0));
            d5_no_cycles = !d4_have_cf;
            skip_a5 = "";
        } else {
            skip_a5 = "the conventional cyclostationary analysis already "
                      "produced candidate cycle frequencies; the heavier "
                      "SSCA search was not required";
            node_skip(&envelopes, dag, "ssca", DAG_A5, skip_a5);
        }
    }
    if (d4_have_cf) {
        /* A6: validation by the CFAR / whitened / multi-cycle detectors */
        ran_a6 = true;
        {
            AnalysisResult *ar = node_begin(&envelopes, dag, "cycledet",
                                            DAG_A6, &node_t0);

            ok_cycledet = cycledet_analyze(s, &sreg, &cycledet);
            if (ar != NULL) {
                apub_cycledet(ar, &cycledet);
                apub_set_confidence(ar, ok_cycledet
                                            ? (cycledet.detected ? 1.0 : 0.4)
                                            : 0.0);
            }
            node_end(ar, dag, DAG_A6, node_t0, ok_cycledet, NULL);
        }
        {
            AnalysisResult *ar = node_begin(&envelopes, dag, "cpar", DAG_A6,
                                            &node_t0);

            ok_cpar = cpar_analyze(s, &sreg, &cpar);
            if (ar != NULL) {
                apub_cpar(ar, &cpar);
                apub_set_confidence(ar, ok_cpar
                                            ? (cpar.detected ? 1.0 : 0.4)
                                            : 0.0);
            }
            node_end(ar, dag, DAG_A6, node_t0, ok_cpar, NULL);
        }
        job_advance(job, &done, "CPAR", &stage_start);
        if (job->cancelled) {
            goto cancelled_out;
        }
        /* Cycle validation comes from the CFAR/multi-cycle detectors
         * only: CPAR is a DSSS chip-rate detector, not a cycle-frequency
         * validator (it fired on this burst emission while the cycle
         * detector found nothing, which previously read as "validated
         * cycles: yes"). */
        validated_cycles = ok_cycledet && cycledet.detected;
    } else {
        skip_a6 = "no candidate cycle frequencies: validating detectors "
                  "without candidates would only invite false positives";
        node_skip(&envelopes, dag, "cycledet", DAG_A6, skip_a6);
        node_skip(&envelopes, dag, "cpar", DAG_A6, skip_a6);
    }
    if (d4_have_cf) {
        /* A8: cyclic cumulants / AMR at the candidate cycle frequencies */
        ran_a8 = true;
        {
            AnalysisResult *ar = node_begin(&envelopes, dag, "ccc", DAG_A8,
                                            &node_t0);

            ok_ccc = ccc_analyze(s, &sreg, &ccc);
            if (ar != NULL) {
                apub_ccc(ar, &ccc);
                apub_set_confidence(ar, ok_ccc ? (ccc.valid ? 1.0 : 0.4) : 0.0);
            }
            node_end(ar, dag, DAG_A8, node_t0, ok_ccc, NULL);
        }
        job_advance(job, &done, "cyclic cumulants", &stage_start);
        if (job->cancelled) {
            goto cancelled_out;
        }
    } else {
        skip_a8 = "cyclic cumulants need a candidate carrier/cycle "
                  "frequency, and none was found";
        node_skip(&envelopes, dag, "ccc", DAG_A8, skip_a8);
    }
    /* A9: intra-burst time-domain envelope inspection - only for a
     * bursty/pulsed signature (a burst to zoom into). Runs after the
     * cyclic-cumulant stage so the cumulant carrier estimate can remove
     * any residual carrier offset. */
    {
        bool bursty = (ok_pulse && pulse.num_pulses >= 2) ||
                      (metrics.am_depth > 0.5 && metrics.duty_cycle > 0.0 &&
                       metrics.duty_cycle < 0.95);

        if (bursty) {
            ran_a9 = true;
            {
                AnalysisResult *ar = node_begin(&envelopes, dag, "intraburst",
                                                DAG_A9, &node_t0);

                ok_intraburst = intraburst_analyze_sot(
                    &sot, (ran_a8 && ok_ccc) ? &ccc : NULL, &intraburst);
                if (ar != NULL) {
                    apub_intraburst(ar, &intraburst);
                    apub_set_confidence(ar, ok_intraburst
                                                ? (intraburst.surrogate_fpr < 0.5
                                                       ? 0.9 : 0.45)
                                                : 0.0);
                }
                node_end(ar, dag, DAG_A9, node_t0, ok_intraburst, NULL);
            }
            job_advance(job, &done, "intra-burst analysis", &stage_start);
            if (job->cancelled) {
                goto cancelled_out;
            }
        } else {
            skip_a9 = "the signal is not bursty/pulsed, so there is no "
                      "burst to zoom into";
            node_skip(&envelopes, dag, "intraburst", DAG_A9, skip_a9);
        }
    }

    if (!validated_cycles || d5_no_cycles) {
        /* A7: the ACF-fluctuation diagnostic for noise-like / hidden
         * spread-spectrum signals runs when the detectors failed */
        ran_a7 = true;
        {
            AnalysisResult *ar = node_begin(&envelopes, dag, "fluct", DAG_A7,
                                            &node_t0);

            ok_fluct = fluct_analyze(s, &sreg, &fluct);
            if (ar != NULL) {
                apub_fluct(ar, &fluct);
                apub_set_confidence(ar, ok_fluct
                                            ? (fluct.detected ? 1.0 : 0.4)
                                            : 0.0);
            }
            node_end(ar, dag, DAG_A7, node_t0, ok_fluct, NULL);
        }
        job_advance(job, &done, "autocorrelation fluctuation (Burel)", &stage_start);
        if (job->cancelled) {
            goto cancelled_out;
        }
    } else {
        skip_a7 = "validated cyclic features are present, so the "
                  "spread-spectrum/noise diagnostic was unnecessary";
        node_skip(&envelopes, dag, "fluct", DAG_A7, skip_a7);
    }
    if (!skip_a5[0] && !ran_a5) {
        skip_a5 = "not selected by the DAG";
    }
    if (!skip_a4[0] && !ran_a4) {
        skip_a4 = "not selected by the DAG";
    }

    /* (The execution plan is not printed: each analysis section below
     * already states EXECUTED/SKIPPED with its rationale.) */


    job_advance(job, &done, "report: synthesis", &stage_start);

    /* the modulation/ID ranking + anomaly check feed both job kinds */
    {
        int mc = aconfig_get_int("modconf", "max_candidates", 12);

        if (mc < 1) mc = 1;
        if (mc > 12) mc = 12;
        job->job_mod_count = modconf_rank(&analysis, &metrics, &cyclic, &ccc,
                                          &hos, &pulse, &fluct, &cpar,
                                          job->job_mods, (size_t)mc);
    }
    {
        int rk = aconfig_get_int("signals", "rank_max", 5);

        if (rk < 1) rk = 1;
        if (rk > 5) rk = 5;
        job->job_ids_count = signals_rank_conf(analysis.center_freq,
                                               analysis.bandwidth, &metrics,
                                               job->job_mods,
                                               job->job_mod_count, &pulse,
                                               job->job_ids, (size_t)rk);
        job->measured_bw_hz = analysis.bandwidth;
        job->measured_center_hz = analysis.center_freq;
    }
    job->anomaly = signals_anomaly_check(analysis.center_freq,
                                         analysis.bandwidth,
                                         job->job_ids, job->job_ids_count,
                                         job->anomaly_explain,
                                         sizeof(job->anomaly_explain));
    {
        AnalysisResult *ar = ar_add(&envelopes, "modconf",
                                    "Modulation consensus", DAG_MODCONF);

        if (ar != NULL) {
            aregistry_apply_settings(ar);
            apub_modconf(ar, job->job_mods, job->job_mod_count);
            apub_set_confidence(ar, job->job_mod_count > 0
                                        ? job->job_mods[0].score / 100.0 : 0.0);
            dag_end(dag, DAG_MODCONF, true, 0.0);
        }
        ar = ar_add(&envelopes, "signals", "Signal database match", DAG_A0);
        if (ar != NULL) {
            aregistry_apply_settings(ar);
            apub_signals(ar, job->job_ids, job->job_ids_count, job->anomaly,
                         job->anomaly_explain);
            apub_set_confidence(ar, job->job_ids_count > 0
                                        ? job->job_ids[0].confidence_pct /
                                              100.0 : 0.0);
        }
    }
    if (job->update_only) {
        const char *mode = (metrics.demod_mode[0] != '\0')
                               ? metrics.demod_mode
                               : "digital/unknown";
        double bw = (metrics.occupied_bw_hz > 0.0)
                        ? metrics.occupied_bw_hz
                        : analysis.bandwidth;
        double burst = (pulse.num_pulses >= 5) ? pulse.prf_hz : 0.0;

        snprintf(job->best_notes, sizeof(job->best_notes),
                 "mode %s | bandwidth %.1f kHz | symbol rate %.0f Hz | "
                 "burst rate %.0f Hz | centre %.3f MHz",
                 mode, bw / 1e3, metrics.sym_rate_hz, burst,
                 analysis.center_freq / 1e6);
    }

    /* --- spread-spectrum classification (all analyses) ---------------- */
    spread_n = spread_classify(&metrics, &analysis,
                               (ran_a2 && ok_pulse) ? &pulse : NULL,
                               (ran_a7 && ok_fluct) ? &fluct : NULL,
                               (ran_a6 && ok_cpar) ? &cpar : NULL,
                               (ran_a4 && ok_fam) ? &cyclic : NULL,
                               (ran_a5 && ok_ssca) ? &ssca : NULL,
                               spread, 6);
    {
        AnalysisResult *ar = ar_add(&envelopes, "ssclass",
                                    "Spread-spectrum classification",
                                    DAG_SSCLASS);

        if (ar != NULL) {
            aregistry_apply_settings(ar);
            apub_ssclass(ar, spread, spread_n);
            apub_set_confidence(ar, spread_n > 0
                                        ? spread[0].score / 100.0 : 0.0);
            dag_end(dag, DAG_SSCLASS, true, 0.0);
        }
    }

    /* keep the analysis results for the Synthesis window */
    job->synth_valid = synth_spec_from(
        &metrics, &analysis, (ran_a2 && ok_pulse) ? &pulse : NULL,
        job->job_mod_count > 0 ? &job->job_mods[0] : NULL,
        (ran_a9 && ok_intraburst) ? &intraburst : NULL, spread, spread_n,
        (ran_a7 && ok_fluct && fluct.detected) ? fluct.chip_rate_hz
                                             : ((ran_a6 && ok_cpar &&
                                                 cpar.detected)
                                                    ? cpar.chip_rate_hz
                                                    : 0.0),
        sot.valid ? sot.view.center_freq : src->center_freq,
        sot.valid ? sot.view.sample_rate : src->sample_rate,
        src->format, job->job_ids, job->job_ids_count, &job->synth);

    /* --- REPORTER: build the document once, render every format ------
     * The Reporter owns the layout and the output formats; the worker only
     * supplies the consolidated data (analyses, Director inputs, DAG state)
     * and its progress/cancellation callbacks. */
    {
        ReportInput in;
        ReportDoc *doc;
        GDateTime *now = g_date_time_new_now_local();
        gchar *stamp = g_date_time_format(now, "%Y-%m-%d %H:%M:%S");

        memset(&in, 0, sizeof(in));
        in.stamp = stamp;
        in.src = src;
        in.region = job->region;
        in.settings_rate = job->settings_rate;
        in.settings_format = job->settings_format;
        in.settings_center = job->settings_center;
        in.settings_fft_size = job->settings_fft_size;
        in.settings_window = job->settings_window;
        in.sot = sot.valid ? &sot : NULL;
        in.sot_failed = !sot.valid;
        in.a0_ok = ok_a0;
        in.analysis = &analysis;
        in.metrics = &metrics;
        in.classic = &classic;
        in.pulse = &pulse;
        in.hos = &hos;
        in.cyclic = &cyclic;
        in.ssca = &ssca;
        in.cycledet = &cycledet;
        in.cpar = &cpar;
        in.fluct = &fluct;
        in.ccc = &ccc;
        in.intraburst = &intraburst;
        in.mods = job->job_mods;
        in.n_mods = job->job_mod_count;
        in.ids = job->job_ids;
        in.n_ids = job->job_ids_count;
        in.spread = spread;
        in.n_spread = spread_n;
        in.anomaly = job->anomaly != 0;
        in.anomaly_explain = job->anomaly_explain;
        in.envelopes = &envelopes;
        in.dag = dag;
        in.ran = (typeof(in.ran)){ ran_a1, ran_a2, ran_a3, ran_a4, ran_a5,
                                   ran_a6, ran_a7, ran_a8, ran_a9 };
        in.ok = (typeof(in.ok)){ ok_a0, ok_classic, ok_pulse, ok_hos,
                                 ok_fam, ok_ssca, ok_cycledet, ok_cpar,
                                 ok_fluct, ok_ccc, ok_intraburst };
        in.skip_a4 = skip_a4;
        in.skip_a5 = skip_a5;
        in.skip_a6 = skip_a6;
        in.skip_a7 = skip_a7;
        in.skip_a8 = skip_a8;
        in.skip_a9 = skip_a9;
        in.d6_wideband = d6_wideband;
        in.d4_have_cf = d4_have_cf;
        in.d5_no_cycles = d5_no_cycles;
        in.validated_cycles = validated_cycles;
        in.cancelled = job->cancelled != 0;
        in.stages_done = job->stages_done;
        in.stage_name = job->stage;
        in.elapsed_s = job->started_us > 0
                           ? (double)(g_get_monotonic_time() - job->started_us) /
                                 1e6
                           : 0.0;
        in.hooks.ctx = job;
        in.hooks.advance = report_hook_advance;
        in.hooks.cancelled = report_hook_cancelled;
        doc = report_build(&in);
        if (doc != NULL) {
            job->report = report_render(doc, REPORT_TEXT);
            job->report_md = report_render(doc, REPORT_MARKDOWN);
            job->report_html = report_render(doc, REPORT_HTML);
            report_doc_free(doc);
        }
        g_date_time_unref(now);
        g_free(stamp);
    }

    /* release the job-local result storage */
    cyclic_clear(&cyclic);
    ssca_clear(&ssca);
    cycledet_clear(&cycledet);
    fluct_clear(&fluct);
    cpar_clear(&cpar);
    hos_clear(&hos);
    ccc_clear(&ccc);
    classic_clear(&classic);
    pulse_clear(&pulse);
    filter_clear(&filt);

    sot_clear(&sot);
    hb.stop = 1;
    if (hb_thread != NULL) g_thread_join(hb_thread);
    ar_arena_free(arena);

    /* Defensive: a GTK text buffer refuses invalid UTF-8 and would leave
     * the page blank, so validate the assembled report here and log the
     * exact location of any bad byte (the deliver path sanitises it, but
     * the log pinpoints the producer that needs fixing). */
    if (job->report != NULL) {
        const char *bad = NULL;

        if (!g_utf8_validate(job->report, -1, &bad)) {
            size_t off = (size_t)(bad - job->report);
            size_t from = off > 120 ? off - 120 : 0;
            size_t len = strlen(job->report);

            fprintf(stderr,
                    "te15: report is not valid UTF-8 (first bad byte at "
                    "offset %zu of %zu): ...%.*s...\n", off, len, 200,
                    job->report + from);
        }
    }
    /* The DAG is complete even when some analyses were skipped or
     * excluded: mark every slot done so the progress bar fills to 100% at
     * the end of the run instead of stopping at the number of executed
     * stages. */
    job->stages_done = REPORT_TOTAL_MAX;
    g_atomic_int_set(&job->progress, REPORT_TOTAL_MAX);
    g_atomic_int_set(&job->total, REPORT_TOTAL_MAX);
    g_atomic_int_set(&job->done, 1);
    dag_run_free(dag);
    /* the worker never frees the job: hand it back via an idle callback */
    g_idle_add(job->cancelled ? job_discard : report_deliver, job);
    return NULL;

cancelled_out:
    hb.stop = 1;
    if (hb_thread != NULL) g_thread_join(hb_thread);
    ar_arena_free(arena);
    sot_clear(&sot);
    cyclic_clear(&cyclic);
    ssca_clear(&ssca);
    cycledet_clear(&cycledet);
    fluct_clear(&fluct);
    cpar_clear(&cpar);
    hos_clear(&hos);
    ccc_clear(&ccc);
    classic_clear(&classic);
    pulse_clear(&pulse);
    filter_clear(&filt);
    /* A cancellation must never mean "no output": report what completed.
     * The header and every finished stage are already in `out`, so append
     * a banner and deliver the partial report like a normal one. */
    job->report = g_strdup("ANALYSIS CANCELLED BEFORE THE REPORT WAS "
                           "ASSEMBLED\n");
    g_atomic_int_set(&job->total, REPORT_TOTAL_MAX);
    g_atomic_int_set(&job->done, 1);
    g_idle_add(report_deliver, job);
    return NULL;
}


/* Main-thread cleanup for cancelled jobs. Tolerates current_report_job
 * already being NULL (watchdog / reset paths). The worker never frees
 * the job itself. */
static gboolean job_discard(gpointer data)
{
    ReportJob *job = (ReportJob *)data;
    Ui *ui = job->ui;

    if (current_report_job == job) {
        current_report_job = NULL;
    }
    job_stage_shown = NULL;
    if (ui->analysis_status != NULL) {
        gtk_label_set_text(GTK_LABEL(ui->analysis_status), "");
    }
    g_free(job->report);
    g_free(job->report_md);
    g_free(job->report_html);
    g_free(job);
    if (ui->report_progress != NULL) {
        gtk_progress_bar_set_fraction(GTK_PROGRESS_BAR(ui->report_progress), 0.0);
        gtk_widget_set_visible(ui->report_progress, FALSE);
    }
    if (ui->demod_update_btn != NULL) {
        gtk_widget_set_sensitive(ui->demod_update_btn, TRUE);
    }
    return G_SOURCE_REMOVE;
}

static gboolean report_deliver(gpointer data)
{
    ReportJob *job = (ReportJob *)data;
    Ui *ui = job->ui;

    if (current_report_job == job) {
        current_report_job = NULL;
    }
    /* the DAG finished: clear the "Analyzing: ..." label */
    job_stage_shown = NULL;
    if (ui->analysis_status != NULL) {
        gtk_label_set_text(GTK_LABEL(ui->analysis_status), "");
    }
    /* the idle callbacks own the UI cleanup */
    if (ui->report_progress != NULL) {
        gtk_progress_bar_set_fraction(GTK_PROGRESS_BAR(ui->report_progress), 1.0);
        gtk_widget_set_visible(ui->report_progress, FALSE);
    }
    if (ui->demod_update_btn != NULL) {
        gtk_widget_set_sensitive(ui->demod_update_btn, TRUE);
    }
    if (ui->window == NULL ||
        (job->report == NULL && job->report_md == NULL &&
         job->report_html == NULL)) {
        g_free(job->report);
        g_free(job->report_md);
        g_free(job->report_html);
        g_free(job);
        return G_SOURCE_REMOVE;
    }
    if (job->cancelled) {
        /* partial report: deliver it (the banner explains the truncation),
         * then let the normal path store and display it */
        fprintf(stderr, "te15: delivering the partial report after "
                        "cancellation at stage '%s'\n",
                job->stage != NULL ? job->stage : "unknown");
    }
    if (job->update_only) {
        /* the analysis delivered in the page: ranking + anomaly notes */
        ui->demod_mod_count = job->job_mod_count;
        memcpy(ui->demod_mods, job->job_mods, sizeof(job->job_mods));
        ui->demod_id_count = job->job_ids_count;
        memcpy(ui->demod_ids, job->job_ids, sizeof(job->job_ids));
        memcpy(ui->demod_best_notes, job->best_notes, sizeof(job->best_notes));
        ui->demod_anomaly = job->anomaly;
        memcpy(ui->demod_anomaly_explain, job->anomaly_explain,
               sizeof(job->anomaly_explain));
        /* hand the LTE uplink parameters to the demodulator */
        if (ui->src != NULL) {
            lte_hint_apply(ui, ui->demod_ids, ui->demod_id_count,
                           job->measured_bw_hz, NULL, ui->demod_mods,
                           ui->demod_mod_count);
        }
    }
    /* store the rendered copies (all three formats are kept so switching
     * the format needs no re-analysis) and display the selected one. */
    g_free(ui->report_text);
    g_free(ui->report_markdown);
    g_free(ui->report_html);
    ui->report_text = job->report != NULL ? utf8_sanitize(job->report) : NULL;
    ui->report_markdown = job->report_md != NULL
                              ? utf8_sanitize(job->report_md) : NULL;
    ui->report_html = job->report_html != NULL
                          ? utf8_sanitize(job->report_html) : NULL;
    g_free(job->report);
    g_free(job->report_md);
    g_free(job->report_html);
    job->report = NULL;
    job->report_md = NULL;
    job->report_html = NULL;
    report_show_selected_format(ui);
    if (report_save_btn != NULL) {
        gtk_widget_set_sensitive(report_save_btn, TRUE);
    }
    ui->last_synth_valid = job->synth_valid;
    if (job->synth_valid) {
        ui->last_synth = job->synth;
    }
    if (ui->synth_btn != NULL) {
        gtk_widget_set_sensitive(ui->synth_btn, job->synth_valid);
    }
    g_free(job);
    return G_SOURCE_REMOVE;
}

/* ------------------------------------------------------------------ */
/* Selection > Synthesize: surrogate signal from the last analysis     */
/* ------------------------------------------------------------------ */

typedef struct {
    Ui      *ui;
    float   *iq;        /* generated interleaved I/Q                */
    size_t   n;         /* complex samples                          */
    double   fs;
    double   center;
    IqFormat format;
    char     label[512];
    GtkWidget *drawing;
    GtkWidget *info;
    float   *image;     /* dB spectrogram, rows x bins              */
    int      rows;
    int      bins;
} SynthWin;

static void synthwin_compute_image(SynthWin *w)
{
    enum { NFFT = 1024 };
    int rows = 300;
    int bins = NFFT / 2;
    float *img;
    float *seg;
    FftPlan *plan;
    double peak = -200.0, lo;

    if (w->iq == NULL || w->n < NFFT || w->rows > 0) return;
    img = malloc((size_t)rows * bins * sizeof(float));
    seg = malloc(2u * NFFT * sizeof(float)); /* NFFT complex samples */
    plan = fft_plan_create(NFFT);
    if (img == NULL || seg == NULL || plan == NULL) {
        free(img);
        free(seg);
        if (plan != NULL) fft_plan_destroy(plan);
        return;
    }
    /* start from a neutral level so a partial fill cannot show garbage */
    for (size_t i = 0; i < (size_t)rows * bins; i++) img[i] = -200.0f;
    for (int r = 0; r < rows; r++) {
        size_t off = (size_t)((double)r / (double)rows *
                              (double)(w->n > NFFT ? w->n - NFFT : 0));
        double *db = malloc(bins * sizeof(double));

        if (db == NULL) break;
        memcpy(seg, w->iq + 2 * off, 2u * NFFT * sizeof(float));
        for (int i = 0; i < NFFT; i++) {
            double win = 0.5 - 0.5 * cos(2.0 * M_PI * (double)i /
                                         (double)(NFFT - 1));

            seg[2 * i] *= (float)win;
            seg[2 * i + 1] *= (float)win;
        }
        {
            fftwf_complex *xa = fftwf_malloc(sizeof(fftwf_complex) * NFFT);

            if (xa == NULL) {
                free(db);
                break;
            }
            fft_plan_transform(plan, seg, NULL, xa);
            for (int k = 0; k < bins; k++) {
                /* centred display: -fs/2 .. +fs/2 */
                int idx = (k < bins / 2) ? (k + bins) : (k - bins / 2);
                double p = (double)xa[idx][0] * xa[idx][0] +
                           (double)xa[idx][1] * xa[idx][1];

                db[k] = 10.0 * log10(p + 1e-30);
                if (db[k] > peak) peak = db[k];
            }
            fftwf_free(xa);
        }
        for (int k = 0; k < bins; k++) {
            img[(size_t)r * bins + k] = (float)db[k];
        }
        free(db);
    }
    fft_plan_destroy(plan);
    free(seg);
    lo = peak - 80.0;
    if (lo < -200.0) lo = -200.0;
    {
        float span = (float)(peak - lo + 1e-9);

        for (size_t i = 0; i < (size_t)rows * bins; i++) {
            float v = (img[i] - (float)lo) / span;

            if (v < 0.0f) v = 0.0f;
            if (v > 1.0f) v = 1.0f;
            img[i] = v;
        }
    }
    w->image = img;
    w->rows = rows;
    w->bins = bins;
}

static gboolean synthwin_draw(GtkWidget *widget, cairo_t *cr,
                              gpointer user_data)
{
    SynthWin *w = (SynthWin *)user_data;
    int width = gtk_widget_get_allocated_width(widget);
    int height = gtk_widget_get_allocated_height(widget);

    cairo_set_source_rgb(cr, 0.06, 0.06, 0.08);
    cairo_paint(cr);
    if (w->image == NULL || w->rows < 1 || w->bins < 1 || width < 2 ||
        height < 2) {
        return FALSE;
    }
    {
        int cols = width;
        int rows_px = height;
        double xstep = (double)w->bins / (double)cols;
        double ystep = (double)w->rows / (double)rows_px;

        for (int x = 0; x < cols; x++) {
            for (int y = 0; y < rows_px; y++) {
                int bx = (int)((double)x * xstep);
                int by = (int)((double)y * ystep);
                float v;

                if (bx >= w->bins) bx = w->bins - 1;
                if (by >= w->rows) by = w->rows - 1;
                v = w->image[(size_t)by * w->bins + bx];
                {
                    Rgb8 c = viridis_lookup(&w->ui->colormap, v);

                    cairo_set_source_rgb(cr, c.r / 255.0, c.g / 255.0,
                                         c.b / 255.0);
                }
                cairo_rectangle(cr, x, y, 1.5, 1.5);
                cairo_fill(cr);
            }
        }
    }
    return FALSE;
}

/* the window is closed: free the generated data */
static void synthwin_destroy(GtkWidget *widget, gpointer user_data)
{
    SynthWin *w = (SynthWin *)user_data;

    (void)widget;
    free(w->iq);
    free(w->image);
    free(w);
}

static void synthwin_save_dialog(SynthWin *w);

static void on_synth_save(GtkButton *btn, gpointer user_data)
{
    (void)btn;
    synthwin_save_dialog((SynthWin *)user_data);
}

static void synthwin_save_dialog(SynthWin *w)
{
    GtkWidget *dialog = gtk_file_chooser_dialog_new(
        "Save synthetic recording", GTK_WINDOW(gtk_widget_get_toplevel(
                                      w->drawing)),
        GTK_FILE_CHOOSER_ACTION_SAVE, CANCEL_BTN_LABEL, GTK_RESPONSE_CANCEL,
        SAVE_BTN_LABEL, GTK_RESPONSE_ACCEPT, NULL);
    char name[256];
    time_t now = time(NULL);
    struct tm tmv;

    localtime_r(&now, &tmv);
    snprintf(name, sizeof(name), "gqrx_%04d%02d%02d_%02d%02d%02d_%ld_%ld_fc.raw",
             tmv.tm_year + 1900, tmv.tm_mon + 1, tmv.tm_mday, tmv.tm_hour,
             tmv.tm_min, tmv.tm_sec, (long)llround(w->center),
             (long)llround(w->fs));
    gtk_file_chooser_set_current_name(GTK_FILE_CHOOSER(dialog), name);
    if (gtk_dialog_run(GTK_DIALOG(dialog)) == GTK_RESPONSE_ACCEPT) {
        char *path = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dialog));

        if (path != NULL) {
            if (synth_write(path, w->iq, w->n, w->format)) {
                char msg[600];

                snprintf(msg, sizeof(msg), "Saved %s (%.2f s, %s)",
                         path, (double)w->n / w->fs,
                         w->format == IQ_FORMAT_FLOAT32 ? "float32" :
                         w->format == IQ_FORMAT_INT16 ? "int16" :
                         w->format == IQ_FORMAT_INT8 ? "int8" : "uint8");
                gtk_label_set_text(GTK_LABEL(w->info), msg);
            } else {
                gtk_label_set_text(GTK_LABEL(w->info), "Could not write file");
            }
            g_free(path);
        }
    }
    gtk_widget_destroy(dialog);
}

static void on_synthesize(GtkButton *btn, gpointer user_data)
{
    Ui *ui = (Ui *)user_data;
    SynthWin *w;
    Region visible;
    SynthNoise noise;

    (void)btn;
    if (!ui->last_synth_valid) {
        set_info(ui, "Run Analyze first: the synthesis uses its results");
        return;
    }
    memset(&noise, 0, sizeof(noise));
    visible.freq_min = ui->src->center_freq - ui->src->sample_rate / 2.0;
    visible.freq_max = ui->src->center_freq + ui->src->sample_rate / 2.0;
    visible.time_min = ui->scroll_position;
    visible.time_max = ui->scroll_position + ui->visible_seconds;
    /* the noise profile comes from the data behind the visible waterfall */
    (void)synth_noise_profile(ui->src, &visible, &ui->selection, &noise);

    w = calloc(1, sizeof(*w));
    if (w == NULL) {
        synth_noise_clear(&noise);
        return;
    }
    w->ui = ui;
    w->fs = ui->last_synth.sample_rate;
    w->center = ui->last_synth.center_hz;
    w->format = ui->last_synth.out_format;
    if (!synth_generate(&ui->last_synth, noise.n > 0 ? &noise : NULL, &w->iq,
                        &w->n)) {
        synth_noise_clear(&noise);
        free(w);
        set_info(ui, "Synthesis failed");
        return;
    }
    synth_noise_clear(&noise);

    /* window */
    {
        GtkWidget *win = gtk_window_new(GTK_WINDOW_TOPLEVEL);
        GtkWidget *vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 6);
        GtkWidget *header = gtk_label_new(NULL);
        GtkWidget *info = gtk_label_new("");
        GtkWidget *hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
        GtkWidget *save = gtk_button_new_with_label(SAVE_BTN_LABEL);
        char head[900];
        GtkWidget *drawing = gtk_drawing_area_new();

        snprintf(head, sizeof(head),
                 "Synthetic signal at %.6f MHz | %s | %.2f s at %.3f MS/s | "
                 "%s | SNR %.1f dB\nSignal: %s",
                 w->center / 1e6,
                 report_format_label(w->format), (double)w->n / w->fs,
                 w->fs / 1e6, report_format_label(w->format),
                 ui->last_synth.snr_db, ui->last_synth.description);
        gtk_label_set_text(GTK_LABEL(header), head);
        gtk_label_set_xalign(GTK_LABEL(header), 0.0);
        gtk_label_set_line_wrap(GTK_LABEL(header), TRUE);
        gtk_box_pack_start(GTK_BOX(vbox), header, FALSE, FALSE, 4);
        gtk_widget_set_size_request(drawing, 900, 360);
        gtk_widget_set_events(drawing, 0); /* non-interactive waterfall */
        gtk_box_pack_start(GTK_BOX(vbox), drawing, TRUE, TRUE, 0);
        gtk_label_set_xalign(GTK_LABEL(info), 0.0);
        gtk_box_pack_start(GTK_BOX(vbox), info, FALSE, FALSE, 2);
        gtk_box_pack_end(GTK_BOX(hbox), save, FALSE, FALSE, 0);
        gtk_box_pack_start(GTK_BOX(vbox), hbox, FALSE, FALSE, 2);
        gtk_container_set_border_width(GTK_CONTAINER(vbox), 8);
        gtk_container_add(GTK_CONTAINER(win), vbox);

        w->drawing = drawing;
        w->info = info;
        g_signal_connect(drawing, "draw", G_CALLBACK(synthwin_draw), w);
        g_signal_connect(save, "clicked", G_CALLBACK(on_synth_save), w);
        g_signal_connect(win, "destroy", G_CALLBACK(synthwin_destroy), w);
        gtk_window_set_title(GTK_WINDOW(win), "Synthetic signal");
        gtk_window_set_default_size(GTK_WINDOW(win), 940, 520);
        gtk_widget_show_all(win);
        synthwin_compute_image(w);
        gtk_widget_queue_draw(drawing);
    }
}

/* Selection page "Save": write the stored report text to a .txt file. */
/* Render the generated HTML into the report text view. Only the subset the
 * Reporter emits is handled (h1/h2/pre/b/br/li), so headings stand out and
 * the monospace body keeps the report's alignment. */
static void report_html_to_buffer(GtkTextBuffer *buf, const char *html)
{
    GtkTextIter it;
    const char *p = html;
    char tag[16];
    const char *stack[16];
    int depth = 0;

    gtk_text_buffer_set_text(buf, "", -1);
    gtk_text_buffer_get_end_iter(buf, &it);
    gtk_text_buffer_create_tag(buf, "h1", "weight", PANGO_WEIGHT_BOLD,
                               "scale", 1.5, "pixels-above-lines", 10,
                               "pixels-below-lines", 4, NULL);
    gtk_text_buffer_create_tag(buf, "h2", "weight", PANGO_WEIGHT_BOLD,
                               "scale", 1.12, "pixels-above-lines", 14,
                               "pixels-below-lines", 4, NULL);
    gtk_text_buffer_create_tag(buf, "pre", "family", "monospace",
                               "scale", 0.88, NULL);
    gtk_text_buffer_create_tag(buf, "b", "weight", PANGO_WEIGHT_BOLD, NULL);

    while (*p != '\0') {
        if (*p == '<') {
            const char *close = strchr(p, '>');
            size_t n;

            if (close == NULL) break;
            n = (size_t)(close - p - 1);
            if (n >= sizeof(tag)) n = sizeof(tag) - 1;
            memcpy(tag, p + 1, n);
            tag[n] = '\0';
            if (tag[0] == '/') {
                const char *name = tag + 1;

                if (depth > 0 && strcmp(stack[depth - 1], name) == 0) {
                    depth--;
                }
                /* a block ends with a line break */
                if (strcmp(name, "pre") == 0 || strcmp(name, "h1") == 0 ||
                    strcmp(name, "h2") == 0) {
                    gtk_text_buffer_insert(buf, &it, "\n", -1);
                }
            } else if (g_str_has_prefix(tag, "h1") ||
                       g_str_has_prefix(tag, "h2") ||
                       g_str_has_prefix(tag, "pre") ||
                       g_str_has_prefix(tag, "b") ||
                       g_str_has_prefix(tag, "li")) {
                const char *name = g_str_has_prefix(tag, "h1") ? "h1"
                                 : g_str_has_prefix(tag, "h2") ? "h2"
                                 : g_str_has_prefix(tag, "pre") ? "pre"
                                 : g_str_has_prefix(tag, "li") ? "b" : "b";

                if (depth < (int)G_N_ELEMENTS(stack) &&
                    tag[0] != 'l') { /* <li> is self-contained here */
                    stack[depth++] = name;
                }
                if (g_str_has_prefix(tag, "li")) {
                    gtk_text_buffer_insert(buf, &it,
                                           "  \xE2\x80\xA2 ", -1);
                }
            } else if (g_str_has_prefix(tag, "br")) {
                gtk_text_buffer_insert(buf, &it, "\n", -1);
            }
            p = close + 1;
            continue;
        }
        {
            const char *next = p;
            const char *active = depth > 0 ? stack[depth - 1] : NULL;
            char rep[4];
            const char *out = p;
            size_t out_len;

            while (*next != '\0' && *next != '<' && *next != '&') next++;
            out_len = (size_t)(next - p);
            if (*next == '&') {
                const char *semi = strchr(next, ';');

                if (semi != NULL) {
                    size_t n = (size_t)(semi - next - 1);

                    if (n == 4 && strncmp(next + 1, "amp;", 4) == 0) {
                        rep[0] = '&';
                        rep[1] = '\0';
                        out = rep;
                        out_len = 1;
                    } else if (n == 3 && strncmp(next + 1, "lt;", 3) == 0) {
                        rep[0] = '<';
                        rep[1] = '\0';
                        out = rep;
                        out_len = 1;
                    } else if (n == 3 && strncmp(next + 1, "gt;", 3) == 0) {
                        rep[0] = '>';
                        rep[1] = '\0';
                        out = rep;
                        out_len = 1;
                    } else if (n == 5 && strncmp(next + 1, "quot;", 5) == 0) {
                        rep[0] = '"';
                        rep[1] = '\0';
                        out = rep;
                        out_len = 1;
                    }
                    next = semi + 1;
                }
            }
            if (out_len > 0) {
                if (active != NULL) {
                    gtk_text_buffer_insert_with_tags_by_name(buf, &it, out,
                                                             (gint)out_len,
                                                             active, NULL);
                } else {
                    gtk_text_buffer_insert(buf, &it, out, (gint)out_len);
                }
            }
            p = next;
        }
    }
}

/* Show the report in the format selected in the Selection tab. */
static void report_show_selected_format(Ui *ui)
{
    const char *text = NULL;
    GtkTextBuffer *buf;

    if (ui->report_view == NULL) return;
    buf = gtk_text_view_get_buffer(GTK_TEXT_VIEW(ui->report_view));
    switch ((ReportFormat)ui->report_format) {
    case REPORT_MARKDOWN: text = ui->report_markdown; break;
    case REPORT_HTML:     text = ui->report_html; break;
    case REPORT_TEXT:
    default:              text = ui->report_text; break;
    }
    if (text == NULL) text = "";
    if ((ReportFormat)ui->report_format == REPORT_HTML && text[0] != '\0') {
        report_html_to_buffer(buf, text);
    } else {
        gtk_text_buffer_set_text(buf, text, -1);
    }
}

static void on_report_format_changed(GtkComboBox *combo, gpointer user_data)
{
    Ui *ui = (Ui *)user_data;
    int idx = gtk_combo_box_get_active(combo);

    ui->report_format = (idx >= 0 && idx <= REPORT_HTML) ? idx : REPORT_TEXT;
    report_show_selected_format(ui);
}

/* Settings > Waterfall: buffer size as a fraction of the sample rate. */
static void on_buffer_size_changed(GtkComboBox *combo, gpointer user_data)
{
    Ui *ui = (Ui *)user_data;
    static const double buf_fracs[] = { 1.0 / 4.0, 1.0 / 6.0, 1.0 / 8.0,
                                        1.0 / 10.0, 1.0 / 12.0 };
    static const char *const buf_names[] = { "1/4", "1/6", "1/8", "1/10",
                                             "1/12" };
    int idx = gtk_combo_box_get_active(combo);

    if (idx < 0 || idx > 4) return;
    ui->stream_buffer_fraction = buf_fracs[idx];
    if (ui->streambuf != NULL) {
        sb_set_fraction(ui->streambuf, ui->stream_buffer_fraction);
        sb_wait_ready(ui->streambuf, 500);
    }
    set_info(ui, "Waterfall buffers: %s of the sample rate (%zu samples, "
                 "%zu rows each side).", buf_names[idx],
             ui->streambuf != NULL ? sb_samples(ui->streambuf) : 0,
             ui->streambuf != NULL ? sb_rows(ui->streambuf) : 0);
}

static void on_save_report(GtkButton *button, gpointer user_data)
{
    Ui *ui = (Ui *)user_data;
    GtkWidget *dialog;
    GtkFileFilter *filter;
    GError *err = NULL;
    char *path = NULL;
    GDateTime *now;
    gchar *stamp;
    char *filename;

    (void)button;
    if (ui->report_text == NULL || ui->report_text[0] == '\0') {
        set_info(ui, "Press Analyze first.");
        return;
    }
    dialog = gtk_file_chooser_dialog_new(
        REPORT_SAVE_TITLE,
        NULL,
        GTK_FILE_CHOOSER_ACTION_SAVE,
        REPORT_CANCEL_LABEL, GTK_RESPONSE_CANCEL,
        REPORT_SAVE_LABEL, GTK_RESPONSE_ACCEPT,
        NULL);
    filter = gtk_file_filter_new();
    if ((ReportFormat)ui->report_format == REPORT_MARKDOWN) {
        gtk_file_filter_set_name(filter, "Markdown report (*.md)");
        gtk_file_filter_add_pattern(filter, "*.md");
    } else if ((ReportFormat)ui->report_format == REPORT_HTML) {
        gtk_file_filter_set_name(filter, "HTML report (*.html)");
        gtk_file_filter_add_pattern(filter, "*.html");
    } else {
        gtk_file_filter_set_name(filter, REPORT_FILTER_NAME);
        gtk_file_filter_add_pattern(filter, REPORT_FILTER_GLOB);
    }
    gtk_file_chooser_add_filter(GTK_FILE_CHOOSER(dialog), filter);
    now = g_date_time_new_now_local();
    stamp = g_date_time_format(now, "%Y%m%d_%H%M%S");
    g_date_time_unref(now);
    filename = g_strdup_printf("selection_report_%s%s", stamp,
                               report_format_suffix(
                                   (ReportFormat)ui->report_format));
    g_free(stamp);
    gtk_file_chooser_set_current_name(GTK_FILE_CHOOSER(dialog), filename);
    g_free(filename);
    if (gtk_dialog_run(GTK_DIALOG(dialog)) == GTK_RESPONSE_ACCEPT) {
        path = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dialog));
    }
    gtk_widget_destroy(dialog);
    if (path == NULL) {
        return; /* cancelled */
    }
    /* append the format's extension when missing (case-insensitive) */
    {
        const char *suffix = report_format_suffix(
            (ReportFormat)ui->report_format);
        size_t len = strlen(path);
        size_t slen = strlen(suffix);

        if (len < slen ||
            g_ascii_strcasecmp(path + len - slen, suffix) != 0) {
            char *with_ext = g_strdup_printf("%s%s", path, suffix);

            g_free(path);
            path = with_ext;
        }
    }
    {
        const char *content = ui->report_text;

        if ((ReportFormat)ui->report_format == REPORT_MARKDOWN) {
            content = ui->report_markdown;
        } else if ((ReportFormat)ui->report_format == REPORT_HTML) {
            content = ui->report_html;
        }
        if (content == NULL) content = "";
        if (!g_file_set_contents(path, content, -1, &err)) {
            set_info(ui, "Cannot save the report: %s.",
                     err != NULL ? err->message : "unknown error");
            if (err != NULL) {
                g_error_free(err);
                err = NULL;
            }
        } else {
            set_info(ui, "Report saved to %s", path);
        }
    }
    g_free(path);
}

/* Fill a report/update job from the current UI state (main thread only).
 * Returns false when no valid target region exists. */
static bool report_job_prepare(ReportJob *job, Ui *ui)
{
    job->ui = ui;
    if (!dock_target_region(ui, &job->region)) {
        return false;
    }
    job->total = REPORT_TOTAL_MAX;
    /* capture UI settings on the main thread (the worker must not touch
     * any other Ui field) */
    job->settings_rate = ui->settings_rate;
    job->settings_format = ui->settings_format;
    job->settings_center = ui->settings_center;
    job->settings_fft_size = ui->settings_fft_size;
    job->settings_window = ui->settings_window;
    return true;
}

/* Selection page "Analyze": the full async analysis chain, delivered
 * into the page (report text view + best-parameters label). */
static void on_demod_update(GtkButton *button, gpointer user_data)
{
    Ui *ui = (Ui *)user_data;
    ReportJob *job;

    (void)button;
    if (ui->src == NULL) {
        return;
    }
    if (!dock_has_target(ui)) {
        set_info(ui, "Select a signal first (Shift+drag on the waterfall, or set a frequency range in the peak pane).");
        return;
    }
    if (current_report_job != NULL) {
        return; /* one job at a time */
    }
    job = g_new0(ReportJob, 1);
    if (!report_job_prepare(job, ui)) {
        g_free(job);
        set_info(ui, "Invalid selection region.");
        return;
    }
    job->update_only = true;
    /* clear the report box for the new run: it stays visible but shows
     * only the results of the analysis in flight */
    g_free(ui->report_text);
    g_free(ui->report_markdown);
    g_free(ui->report_html);
    ui->report_text = NULL;
    ui->report_markdown = NULL;
    ui->report_html = NULL;
    if (ui->analysis_status != NULL) {
        gtk_label_set_text(GTK_LABEL(ui->analysis_status), "");
    }
    if (ui->report_view != NULL) {
        GtkTextBuffer *buf =
            gtk_text_view_get_buffer(GTK_TEXT_VIEW(ui->report_view));

        gtk_text_buffer_set_text(buf, "", -1);
    }
    if (report_save_btn != NULL) {
        gtk_widget_set_sensitive(report_save_btn, FALSE);
    }
    set_info(ui, "Analyzing the selection...");
    if (ui->analysis_status != NULL) {
        gtk_label_set_text(GTK_LABEL(ui->analysis_status),
                           "Analyzing: preparing the common source");
    }
    job_last_progress_val = 0;
    job_last_progress_us = g_get_monotonic_time();
    job_last_heartbeat = -1;
    job_last_heartbeat_us = g_get_monotonic_time();
    job_stall_reported = 0;
    job_stage_shown = NULL;
    current_report_job = job;
    gtk_widget_set_sensitive(ui->demod_update_btn, FALSE);
    g_thread_new("selection-update", report_worker, job);
}

/*
 * GTK text widgets require valid UTF-8: gtk_text_buffer_set_text() and
 * gtk_label_set_text() abort with
 *   gtk_text_buffer_emit_insert: assertion 'g_utf8_validate (text, ...)'
 * when a report string contains an invalid byte sequence (a truncated
 * multi-byte character, an unterminated analysis buffer or a filename with
 * non-UTF-8 bytes). Replace every invalid byte with '?' so the analysis
 * output is always displayable; the count/offset is logged so the producer
 * can be tracked down.
 */
static size_t utf8_seq_len(const unsigned char *p)
{
    if (p[0] < 0x80) return 1;
    if (p[0] >= 0xC2 && p[0] <= 0xDF) {
        return (p[1] >= 0x80 && p[1] <= 0xBF) ? 2 : 0;
    }
    if (p[0] == 0xE0) {
        return (p[1] >= 0xA0 && p[1] <= 0xBF && p[2] >= 0x80 &&
                p[2] <= 0xBF) ? 3 : 0;
    }
    if ((p[0] >= 0xE1 && p[0] <= 0xEC) || (p[0] >= 0xEE && p[0] <= 0xEF)) {
        return (p[1] >= 0x80 && p[1] <= 0xBF && p[2] >= 0x80 &&
                p[2] <= 0xBF) ? 3 : 0;
    }
    if (p[0] == 0xED) {
        return (p[1] >= 0x80 && p[1] <= 0x9F && p[2] >= 0x80 &&
                p[2] <= 0xBF) ? 3 : 0;
    }
    if (p[0] == 0xF0) {
        return (p[1] >= 0x90 && p[1] <= 0xBF && p[2] >= 0x80 &&
                p[2] <= 0xBF && p[3] >= 0x80 && p[3] <= 0xBF) ? 4 : 0;
    }
    if (p[0] >= 0xF1 && p[0] <= 0xF3) {
        return (p[1] >= 0x80 && p[1] <= 0xBF && p[2] >= 0x80 &&
                p[2] <= 0xBF && p[3] >= 0x80 && p[3] <= 0xBF) ? 4 : 0;
    }
    if (p[0] == 0xF4) {
        return (p[1] >= 0x80 && p[1] <= 0x8F && p[2] >= 0x80 &&
                p[2] <= 0xBF && p[3] >= 0x80 && p[3] <= 0xBF) ? 4 : 0;
    }
    return 0;
}

/*
 * Returns a heap copy of `in` with every invalid UTF-8 byte replaced by
 * '?'. Valid text is copied unchanged (so ordinary reports are untouched).
 */
static char *utf8_sanitize(const char *in)
{
    const unsigned char *p;
    size_t n, bad = 0, first_bad = (size_t)-1;
    char *out;
    size_t o = 0;

    if (in == NULL) return g_strdup("");
    n = strlen(in);
    /* fast path: validate first, return an unmodified copy when clean */
    {
        size_t i = 0;
        bool clean = true;

        while (i < n) {
            size_t l = utf8_seq_len((const unsigned char *)in + i);

            if (l == 0 || i + l > n) {
                clean = false;
                break;
            }
            i += l;
        }
        if (clean) return g_strdup(in);
    }
    out = malloc(n + 1);
    if (out == NULL) return g_strdup("");
    p = (const unsigned char *)in;
    {
        size_t i = 0;

        while (i < n) {
            size_t l = utf8_seq_len(p + i);

            if (l == 0 || i + l > n) {
                if (first_bad == (size_t)-1) first_bad = i;
                bad++;
                out[o++] = '?';
                i++;
            } else {
                memcpy(out + o, in + i, l);
                o += l;
                i += l;
            }
        }
    }
    out[o] = '\0';
    if (bad > 0) {
        fprintf(stderr, "te15: report text had %zu invalid UTF-8 byte(s) "
                        "(first at offset %zu) - replaced with '?'\n",
                bad, first_bad);
    }
    return out;
}

/* Set a GtkLabel from analysis text, sanitising invalid UTF-8 first (GTK
 * aborts on invalid sequences, which would blank the page or the status
 * line). */
/* Set the info label to a one-off message (status/notice takeover). */
static void set_info(Ui *ui, const char *fmt, ...)
{
    va_list ap;
    char text[LABEL_BUFFER_SIZE];

    va_start(ap, fmt);
    vsnprintf(text, sizeof(text), fmt, ap);
    va_end(ap);
    {
        char *safe = utf8_sanitize(text);

        gtk_label_set_text(GTK_LABEL(ui->info_label), safe);
        g_free(safe);
        ui->info_expiry_us =
            g_get_monotonic_time() + (gint64)INFO_MESSAGE_TIMEOUT_S *
                                         G_USEC_PER_SEC;
    }
}

/* --- signal handlers ----------------------------------------------------- */

static gboolean on_key_press(GtkWidget *widget, GdkEventKey *event, gpointer user_data)
{
    Ui *ui = (Ui *)user_data;

    (void)widget;
    switch (event->keyval) {
        case GDK_KEY_Up:
            scroll_by(ui, -SCROLL_SECONDS);
            return TRUE;
        case GDK_KEY_Down:
            scroll_by(ui, SCROLL_SECONDS);
            return TRUE;
        case GDK_KEY_s:
        case GDK_KEY_S:
            take_screenshot(ui);
            return TRUE;
        case GDK_KEY_q:
        case GDK_KEY_Q:
            gtk_main_quit();
            return TRUE;
        default:
            break;
    }
    return FALSE;
}

/* Map one raw dB value to a viridis colour through the committed dynamic
 * range: the colour scheme is used as-is, and the render buffer values are
 * never modified — the range normalisation happens here, per pixel. */
static Rgb8 wf_db_colour(const Ui *ui, float db)
{
    float lo = (float)ui->wf_range_lo;
    float hi = (float)ui->wf_range_hi;
    float span = hi - lo;

    if (span < (float)DB_RANGE_MIN_SPAN) span = (float)DB_RANGE_MIN_SPAN;
    return viridis_lookup(&ui->colormap, (db - lo) / span);
}

/* Commit the slider window as the waterfall dynamic range and repaint.
 * Called only when a slider is released, so dragging does not change the
 * waterfall until the user lets go (the peak shading tracks live). */
static void apply_db_range(Ui *ui)
{
    double lo, hi;

    if (ui == NULL || ui->auto_scale) return;   /* auto manages the window */
    lo = MIN(ui->db_min, ui->db_max);
    hi = MAX(ui->db_min, ui->db_max);
    if (hi - lo < DB_RANGE_MIN_SPAN) hi = lo + DB_RANGE_MIN_SPAN;
    if (fabs(lo - ui->wf_range_lo) < 1e-9 &&
        fabs(hi - ui->wf_range_hi) < 1e-9) {
        return;
    }
    ui->wf_range_lo = lo;
    ui->wf_range_hi = hi;
    ui->wf_valid = false;   /* full repaint with the committed range */
    gtk_widget_queue_draw(ui->drawing);
}

static gboolean on_db_range_released(GtkWidget *widget, GdkEvent *event,
                                     gpointer user_data)
{
    (void)widget;
    (void)event;
    apply_db_range((Ui *)user_data);
    return FALSE;
}

/* Move the other dB slider so the dynamic-range span is preserved (only
 * used while "Lock" is checked). */
static void db_lock_follow(GtkRange *other, double value,
                           GCallback handler, gpointer ui)
{
    g_signal_handlers_block_by_func(other, handler, ui);
    gtk_range_set_value(other, value);
    g_signal_handlers_unblock_by_func(other, handler, ui);
}

/* Sliders are independent unless "Lock" is checked: an unlocked slider
 * changes only its own value (the lower bound may even cross the upper one,
 * the renderer then simply maps the ordered range). With Lock the two move
 * together keeping the current span. */
static void on_db_min_changed(GtkRange *range, gpointer user_data);
static void on_db_max_changed(GtkRange *range, gpointer user_data);

static void on_db_min_changed(GtkRange *range, gpointer user_data)
{
    Ui *ui = (Ui *)user_data;
    double value = gtk_range_get_value(range);

    if (ui->db_locked) {
        double span = ui->db_max - ui->db_min;

        if (span < DB_SCALE_STEP) span = DB_SCALE_STEP;
        if (value > DB_SCALE_MAX - span) {
            value = DB_SCALE_MAX - span;
            db_lock_follow(range, value, G_CALLBACK(on_db_min_changed), ui);
        }
        ui->db_min = value;
        ui->db_max = value + span;
        if (ui->db_max_scale != NULL) {
            db_lock_follow(GTK_RANGE(ui->db_max_scale), ui->db_max,
                           G_CALLBACK(on_db_max_changed), ui);
        }
    } else {
        ui->db_min = value;      /* independent: no effect on db_max */
    }
    if (ui->db_min < DB_SCALE_MIN) ui->db_min = DB_SCALE_MIN;
    if (ui->db_min > DB_SCALE_MAX) ui->db_min = DB_SCALE_MAX;
    ui->auto_scale = false;
    /* the peak view's out-of-range shading tracks the sliders live; the
     * waterfall colour scheme still applies only on release */
    gtk_widget_queue_draw(ui->peak_pane);
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(ui->auto_check), FALSE);
    /* not committed yet: the waterfall keeps the old colour scheme until
     * the slider is released (db_range_committed) */
}

static void on_db_max_changed(GtkRange *range, gpointer user_data)
{
    Ui *ui = (Ui *)user_data;
    double value = gtk_range_get_value(range);

    if (ui->db_locked) {
        double span = ui->db_max - ui->db_min;

        if (span < DB_SCALE_STEP) span = DB_SCALE_STEP;
        if (value < DB_SCALE_MIN + span) {
            value = DB_SCALE_MIN + span;
            db_lock_follow(range, value, G_CALLBACK(on_db_max_changed), ui);
        }
        ui->db_max = value;
        ui->db_min = value - span;
        if (ui->db_min_scale != NULL) {
            db_lock_follow(GTK_RANGE(ui->db_min_scale), ui->db_min,
                           G_CALLBACK(on_db_min_changed), ui);
        }
    } else {
        ui->db_max = value;      /* independent: no effect on db_min */
    }
    if (ui->db_max < DB_SCALE_MIN) ui->db_max = DB_SCALE_MIN;
    if (ui->db_max > DB_SCALE_MAX) ui->db_max = DB_SCALE_MAX;
    ui->auto_scale = false;
    gtk_widget_queue_draw(ui->peak_pane);
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(ui->auto_check), FALSE);
}

static void on_db_lock_toggled(GtkToggleButton *toggle, gpointer user_data)
{
    Ui *ui = (Ui *)user_data;

    ui->db_locked = gtk_toggle_button_get_active(toggle);
    if (ui->db_locked) {
        /* locking normalises the span so the pair stays a valid window */
        if (ui->db_max - ui->db_min < DB_SCALE_STEP) {
            ui->db_max = ui->db_min + DB_SCALE_STEP;
            if (ui->db_max > DB_SCALE_MAX) {
                ui->db_max = DB_SCALE_MAX;
                ui->db_min = DB_SCALE_MAX - DB_SCALE_STEP;
            }
            if (ui->db_min_scale != NULL) {
                db_lock_follow(GTK_RANGE(ui->db_min_scale), ui->db_min,
                               G_CALLBACK(on_db_min_changed), ui);
            }
            if (ui->db_max_scale != NULL) {
                db_lock_follow(GTK_RANGE(ui->db_max_scale), ui->db_max,
                               G_CALLBACK(on_db_max_changed), ui);
            }
        }
    }
}

static void on_auto_toggled(GtkToggleButton *toggle, gpointer user_data)
{
    Ui *ui = (Ui *)user_data;

    ui->auto_scale = gtk_toggle_button_get_active(toggle);
    gtk_widget_set_sensitive(ui->db_min_scale, !ui->auto_scale);
    gtk_widget_set_sensitive(ui->db_max_scale, !ui->auto_scale);
    if (!ui->auto_scale) {
        /* leaving auto: the sliders' current window becomes the committed
         * one (built into the colour-scheme copy) */
        ui->wf_range_lo = MIN(ui->db_min, ui->db_max);
        ui->wf_range_hi = MAX(ui->db_min, ui->db_max);
    }
    ui->wf_valid = false;
    gtk_widget_queue_draw(ui->drawing);
}

/* --- menu handlers ------------------------------------------------------- */

/*
 * Parameter dialog for recordings whose metadata cannot be recovered:
 * the user supplies centre frequency, sample rate and sample format.
 */
static void on_manual_open_ok(GtkButton *btn, gpointer user_data)
{
    Ui *ui = (Ui *)user_data;
    GtkWidget **w = g_object_get_data(G_OBJECT(btn), "te15-widgets");
    const char *path = g_object_get_data(G_OBJECT(btn), "te15-path");
    double mhz = g_ascii_strtod(gtk_entry_get_text(GTK_ENTRY(w[0])), NULL);
    double msps = g_ascii_strtod(gtk_entry_get_text(GTK_ENTRY(w[1])), NULL);
    int fmt = gtk_combo_box_get_active(GTK_COMBO_BOX(w[2]));

    (void)btn;
    if (mhz <= 0.0 || msps <= 0.0) {
        set_info(ui, "Enter the centre frequency and sample rate");
        return;
    }
    if (ui->open_file_custom_cb != NULL &&
        ui->open_file_custom_cb(ui, path, mhz * 1e6, msps * 1e6,
                                (IqFormat)fmt) == 0) {
        set_info(ui, "Loaded %s (%s)", path, "manual parameters");
    } else {
        set_info(ui, "Failed to load %s", path);
    }
    gtk_widget_destroy(gtk_widget_get_toplevel(GTK_WIDGET(btn)));
}

static void ui_manual_open_dialog(Ui *ui, const char *path)
{
    GtkWidget *dialog;
    GtkWidget *grid;
    GtkWidget *label;
    GtkWidget *entry_f;
    GtkWidget *entry_r;
    GtkWidget *combo;
    GtkWidget *ok;
    double f = 0.0, r = 0.0;
    IqFormat fmt = IQ_FORMAT_FLOAT32;

    (void)iq_source_guess_metadata(path, &f, &r, &fmt);

    dialog = gtk_dialog_new_with_buttons("Recording parameters",
                                         GTK_WINDOW(ui->window),
                                         GTK_DIALOG_MODAL,
                                         "Cancel", GTK_RESPONSE_CANCEL,
                                         "Open", GTK_RESPONSE_ACCEPT, NULL);
    gtk_window_set_default_size(GTK_WINDOW(dialog), 420, -1);
    grid = gtk_grid_new();
    gtk_grid_set_row_spacing(GTK_GRID(grid), 6);
    gtk_grid_set_column_spacing(GTK_GRID(grid), 8);
    gtk_widget_set_margin_start(grid, 12);
    gtk_widget_set_margin_end(grid, 12);
    gtk_widget_set_margin_top(grid, 12);
    gtk_widget_set_margin_bottom(grid, 12);
    gtk_container_add(GTK_CONTAINER(gtk_dialog_get_content_area(
                         GTK_DIALOG(dialog))), grid);

    label = gtk_label_new("The filename does not contain the recording "
                          "parameters. Enter them:");
    gtk_label_set_xalign(GTK_LABEL(label), 0.0);
    gtk_grid_attach(GTK_GRID(grid), label, 0, 0, 2, 1);
    {
        char *base = g_path_get_basename(path);

        label = gtk_label_new(base);
        g_free(base);
    }
    gtk_label_set_ellipsize(GTK_LABEL(label), PANGO_ELLIPSIZE_MIDDLE);
    gtk_label_set_xalign(GTK_LABEL(label), 0.0);
    gtk_grid_attach(GTK_GRID(grid), label, 0, 1, 2, 1);

    label = gtk_label_new("Centre frequency (MHz)");
    gtk_label_set_xalign(GTK_LABEL(label), 0.0);
    gtk_grid_attach(GTK_GRID(grid), label, 0, 2, 1, 1);
    entry_f = gtk_entry_new();
    gtk_entry_set_text(GTK_ENTRY(entry_f),
                       f > 0.0 ? g_strdup_printf("%.3f", f / 1e6) : "");
    gtk_grid_attach(GTK_GRID(grid), entry_f, 1, 2, 1, 1);

    label = gtk_label_new("Sample rate (Msamples/s)");
    gtk_label_set_xalign(GTK_LABEL(label), 0.0);
    gtk_grid_attach(GTK_GRID(grid), label, 0, 3, 1, 1);
    entry_r = gtk_entry_new();
    gtk_entry_set_text(GTK_ENTRY(entry_r),
                       r > 0.0 ? g_strdup_printf("%.3f", r / 1e6) : "");
    gtk_grid_attach(GTK_GRID(grid), entry_r, 1, 3, 1, 1);

    label = gtk_label_new("Data format");
    gtk_label_set_xalign(GTK_LABEL(label), 0.0);
    gtk_grid_attach(GTK_GRID(grid), label, 0, 4, 1, 1);
    combo = gtk_combo_box_text_new();
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(combo),
                                   "float32 I/Q (8 B/sample)");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(combo),
                                   "int16 I/Q (4 B/sample)");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(combo),
                                   "int8 I/Q (2 B/sample)");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(combo),
                                   "uint8 I/Q (2 B/sample)");
    gtk_combo_box_set_active(GTK_COMBO_BOX(combo), (int)fmt);
    gtk_grid_attach(GTK_GRID(grid), combo, 1, 4, 1, 1);

    gtk_widget_show_all(dialog);
    ok = gtk_dialog_get_widget_for_response(GTK_DIALOG(dialog),
                                            GTK_RESPONSE_ACCEPT);
    {
        GtkWidget *w[3];
        static GtkWidget *store[3];

        w[0] = entry_f;
        w[1] = entry_r;
        w[2] = combo;
        store[0] = w[0];
        store[1] = w[1];
        store[2] = w[2];
        g_object_set_data(G_OBJECT(ok), "te15-widgets", store);
        g_object_set_data_full(G_OBJECT(ok), "te15-path", g_strdup(path),
                               g_free);
    }
    g_signal_connect(ok, "clicked", G_CALLBACK(on_manual_open_ok), ui);
    gtk_dialog_run(GTK_DIALOG(dialog));
    gtk_widget_destroy(dialog);
}

static void on_open_menu(GtkMenuItem *item, gpointer user_data)
{
    Ui *ui = (Ui *)user_data;
    GtkWidget *dialog;
    GtkFileFilter *filter;
    gint response;

    (void)item;
    dialog = gtk_file_chooser_dialog_new(OPEN_DIALOG_TITLE,
                                         GTK_WINDOW(ui->window),
                                         GTK_FILE_CHOOSER_ACTION_OPEN,
                                         CANCEL_BTN_LABEL, GTK_RESPONSE_CANCEL,
                                         OPEN_BTN_LABEL, GTK_RESPONSE_ACCEPT,
                                         NULL);
    filter = gtk_file_filter_new();
    gtk_file_filter_set_name(filter, FILTER_IQ_LABEL);
    gtk_file_filter_add_pattern(filter, FILTER_IQ_PATTERN);
    gtk_file_chooser_add_filter(GTK_FILE_CHOOSER(dialog), filter);
    filter = gtk_file_filter_new();
    gtk_file_filter_set_name(filter, FILTER_ALL_LABEL);
    gtk_file_filter_add_pattern(filter, FILTER_ALL_PATTERN);
    gtk_file_chooser_add_filter(GTK_FILE_CHOOSER(dialog), filter);

    response = gtk_dialog_run(GTK_DIALOG(dialog));
    if (response == GTK_RESPONSE_ACCEPT) {
        char *path = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dialog));

        if (path != NULL) {
            if (ui->open_file_cb != NULL && ui->open_file_cb(ui, path) == 0) {
                set_info(ui, "Loaded %s", path);
            } else {
                /* The parameters were not recoverable from the file:
                 * ask for them, prefilled with whatever could be inferred. */
                ui_manual_open_dialog(ui, path);
            }
            g_free(path);
        }
    }
    gtk_widget_destroy(dialog);
}

/* --- About screen ------------------------------------------------------- */

#define ABOUT_TITLE   "Toxic Element 15"
#define ABOUT_TAGLINE "RF Signals Intelligence"
#define ABOUT_AUTHOR  "Daniel R. Azulay | Radio Defense"
#define ABOUT_YEAR    "2026"
#define ABOUT_WINDOW_W 900
#define ABOUT_WINDOW_H 604

/* pale green text with a black outline */
#define ABOUT_TEXT_R 0.72
#define ABOUT_TEXT_G 0.96
#define ABOUT_TEXT_B 0.76

static GtkWidget *about_window = NULL;
static cairo_surface_t *about_image = NULL;

/* Decode the embedded About artwork (an image in the executable; nothing is
 * read from disk at run time). The surface is cached. */
static cairo_surface_t *about_image_load(void)
{
    GdkPixbufLoader *loader;
    GdkPixbuf *pixbuf = NULL;
    GError *err = NULL;

    if (about_image != NULL) return about_image;
    loader = gdk_pixbuf_loader_new();
    if (loader == NULL) return NULL;
    if (!gdk_pixbuf_loader_write(loader, te15_about_image, te15_about_image_len,
                                 &err) ||
        !gdk_pixbuf_loader_close(loader, &err)) {
        if (err != NULL) {
            g_error_free(err);
            err = NULL;
        }
        g_object_unref(loader);
        return NULL;
    }
    pixbuf = gdk_pixbuf_loader_get_pixbuf(loader);
    if (pixbuf != NULL) g_object_ref(pixbuf);
    g_object_unref(loader);
    if (pixbuf == NULL) return NULL;
    about_image = cairo_image_surface_create(CAIRO_FORMAT_RGB24,
                                             gdk_pixbuf_get_width(pixbuf),
                                             gdk_pixbuf_get_height(pixbuf));
    if (cairo_surface_status(about_image) == CAIRO_STATUS_SUCCESS) {
        cairo_t *cr = cairo_create(about_image);

        gdk_cairo_set_source_pixbuf(cr, pixbuf, 0.0, 0.0);
        cairo_paint(cr);
        cairo_destroy(cr);
    } else {
        cairo_surface_destroy(about_image);
        about_image = NULL;
    }
    g_object_unref(pixbuf);
    return about_image;
}

/* One line of the overlay: large bold pale-green text with a black
 * border (stroked outline), centred on cx at the given baseline. */
static void about_draw_text(cairo_t *cr, const char *text, double cx,
                            double baseline, double size)
{
    cairo_text_extents_t ext;
    double x;

    cairo_select_font_face(cr, FONT_FAMILY, CAIRO_FONT_SLANT_NORMAL,
                           CAIRO_FONT_WEIGHT_BOLD);
    cairo_set_font_size(cr, size);
    cairo_text_extents(cr, text, &ext);
    x = cx - (ext.width / 2.0 + ext.x_bearing);
    cairo_move_to(cr, x, baseline);
    cairo_text_path(cr, text);
    cairo_set_line_join(cr, CAIRO_LINE_JOIN_ROUND);
    cairo_set_line_width(cr, size * 0.14);
    cairo_set_source_rgb(cr, 0.0, 0.0, 0.0);
    cairo_stroke_preserve(cr);
    cairo_set_source_rgb(cr, ABOUT_TEXT_R, ABOUT_TEXT_G, ABOUT_TEXT_B);
    cairo_fill(cr);
}

/* Draw the whole About screen at an arbitrary size (the draw callback and
 * the preview renderer both use this). */
static void about_render(cairo_t *cr, int w, int h)
{
    cairo_surface_t *img = about_image_load();

    if (w <= 0 || h <= 0) return;
    /* dark fallback background */
    cairo_set_source_rgb(cr, 0.05, 0.06, 0.05);
    cairo_paint(cr);
    if (img != NULL) {
        int iw = cairo_image_surface_get_width(img);
        int ih = cairo_image_surface_get_height(img);

        if (iw > 0 && ih > 0) {
            /* fill the window: cover, centred (no distortion) */
            double scale = MAX((double)w / (double)iw,
                               (double)h / (double)ih);

            cairo_save(cr);
            cairo_translate(cr, ((double)w - (double)iw * scale) / 2.0,
                            ((double)h - (double)ih * scale) / 2.0);
            cairo_scale(cr, scale, scale);
            cairo_set_source_surface(cr, img, 0.0, 0.0);
            cairo_paint(cr);
            cairo_restore(cr);
        }
    }
    about_draw_text(cr, ABOUT_TITLE, w / 2.0, h * 0.46, h * 0.085);
    /* the tagline sits between the title and the author line, with a size
     * between theirs */
    about_draw_text(cr, ABOUT_TAGLINE, w / 2.0, h * 0.545, h * 0.048);
    about_draw_text(cr, ABOUT_AUTHOR, w / 2.0, h * 0.62, h * 0.034);
    about_draw_text(cr, ABOUT_YEAR, w / 2.0, h * 0.70, h * 0.034);
}

static gboolean about_draw(GtkWidget *widget, cairo_t *cr, gpointer user_data)
{
    (void)user_data;
    about_render(cr, gtk_widget_get_allocated_width(widget),
                 gtk_widget_get_allocated_height(widget));
    return FALSE;
}

/* Close the About screen (explicit user action only: Escape, Return or the
 * window's close button - clicking the artwork must not dismiss it). */
static gboolean about_close(GtkWidget *widget, GdkEvent *event,
                            gpointer user_data)
{
    (void)widget;
    (void)event;
    (void)user_data;
    if (about_window != NULL) {
        GtkWidget *win = about_window;

        about_window = NULL;
        gtk_widget_destroy(win);
    }
    return TRUE;
}

/* Keys while the About screen has the focus: only Escape/Return close it. */
static gboolean about_key_press(GtkWidget *widget, GdkEventKey *event,
                                gpointer user_data)
{
    (void)widget;
    if (about_window == NULL) return FALSE;
    if (event->keyval == GDK_KEY_Escape || event->keyval == GDK_KEY_Return ||
        event->keyval == GDK_KEY_KP_Enter) {
        about_close(NULL, NULL, user_data);
        return TRUE;
    }
    /* every other key is ignored: the screen stays up */
    return TRUE;
}

/* the user clicked the artwork: keep the About screen open */
static gboolean about_click_keep(GtkWidget *widget, GdkEvent *event,
                                 gpointer user_data)
{
    (void)widget;
    (void)event;
    (void)user_data;
    if (about_window != NULL) gtk_window_present(GTK_WINDOW(about_window));
    return TRUE;
}

/* the window is gone: just forget it (the artwork stays cached) */
static void about_on_destroy(GtkWidget *widget, gpointer user_data)
{
    (void)widget;
    (void)user_data;
    about_window = NULL;
}

/* File > About: a full-window artwork with the version overlay. Click or
 * Escape closes it. */
static void on_about_menu(GtkMenuItem *item, gpointer user_data)
{
    Ui *ui = (Ui *)user_data;
    GtkWidget *drawing;

    (void)item;
    if (about_window != NULL) {
        gtk_window_present(GTK_WINDOW(about_window));
        return;
    }
    about_window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_title(GTK_WINDOW(about_window), "About " ABOUT_TITLE);
    gtk_window_set_default_size(GTK_WINDOW(about_window), ABOUT_WINDOW_W,
                                ABOUT_WINDOW_H);
    gtk_window_set_resizable(GTK_WINDOW(about_window), TRUE);
    gtk_window_set_modal(GTK_WINDOW(about_window), TRUE);
    /* the About screen sits in the middle of the screen */
    gtk_window_set_position(GTK_WINDOW(about_window), GTK_WIN_POS_CENTER);
    if (ui != NULL && ui->window != NULL) {
        gtk_window_set_transient_for(GTK_WINDOW(about_window),
                                     GTK_WINDOW(ui->window));
    }
    drawing = gtk_drawing_area_new();
    gtk_widget_set_hexpand(drawing, TRUE);
    gtk_widget_set_vexpand(drawing, TRUE);
    gtk_widget_set_can_focus(drawing, TRUE);
    gtk_widget_add_events(drawing, GDK_BUTTON_PRESS_MASK);
    g_signal_connect(drawing, "draw", G_CALLBACK(about_draw), ui);
    /* a click only re-presents the window: it must stay open until the
     * user explicitly closes it */
    g_signal_connect(drawing, "button-press-event",
                     G_CALLBACK(about_click_keep), ui);
    g_signal_connect(about_window, "key-press-event",
                     G_CALLBACK(about_key_press), ui);
    g_signal_connect(about_window, "destroy",
                     G_CALLBACK(about_on_destroy), ui);
    gtk_container_add(GTK_CONTAINER(about_window), drawing);
    gtk_widget_show_all(about_window);
    /* the About screen owns the focus while it is up */
    gtk_window_set_focus(GTK_WINDOW(about_window), drawing);
    gtk_widget_grab_focus(drawing);
    gtk_window_present(GTK_WINDOW(about_window));
}

/* File > Quit: destroy the main window (the existing quit path). */
static void on_quit_menu(GtkMenuItem *item, gpointer user_data)
{
    Ui *ui = (Ui *)user_data;

    (void)item;
    if (ui->window != NULL) {
        gtk_widget_destroy(ui->window);
    }
}

/* Write one interleaved IQ chunk in the Settings sample format. */
static bool save_sel_write_chunk(FILE *fp, const float *iq, size_t n,
                                 IqFormat fmt)
{
    const size_t k = 2 * n;
    size_t written;
    size_t i;

    switch (fmt) {
    case IQ_FORMAT_FLOAT32:
        written = fwrite(iq, sizeof(float), k, fp);
        return written == k;
    case IQ_FORMAT_INT16: {
        int16_t *p = malloc(k * sizeof(int16_t));

        if (p == NULL) {
            return false;
        }
        for (i = 0; i < k; i++) {
            float v = iq[i];

            if (v > 1.0f) {
                v = 1.0f;
            }
            if (v < -1.0f) {
                v = -1.0f;
            }
            p[i] = (int16_t)(v * 32767.0f);
        }
        written = fwrite(p, sizeof(int16_t), k, fp);
        free(p);
        return written == k;
    }
    case IQ_FORMAT_INT8: {
        int8_t *p = malloc(k);

        if (p == NULL) {
            return false;
        }
        for (i = 0; i < k; i++) {
            float v = iq[i];

            if (v > 1.0f) {
                v = 1.0f;
            }
            if (v < -1.0f) {
                v = -1.0f;
            }
            p[i] = (int8_t)(v * 127.0f);
        }
        written = fwrite(p, 1, k, fp);
        free(p);
        return written == k;
    }
    case IQ_FORMAT_UINT8: {
        uint8_t *p = malloc(k);

        if (p == NULL) {
            return false;
        }
        for (i = 0; i < k; i++) {
            float v = iq[i];

            if (v > 1.0f) {
                v = 1.0f;
            }
            if (v < -1.0f) {
                v = -1.0f;
            }
            p[i] = (uint8_t)((v + 1.0f) * 127.5f);
        }
        written = fwrite(p, 1, k, fp);
        free(p);
        return written == k;
    }
    }
    return false;
}

/*
 * File > Save Selection… — write the raw IQ of the selection's time
 * range to a .raw file in the sample format selected on the Settings
 * page. The default file name follows the gqrx convention: a timestamp
 * plus the centre frequency and bandwidth of the selection.
 */
static void on_save_selection_menu(GtkMenuItem *item, gpointer user_data)
{
    Ui *ui = (Ui *)user_data;
    Region r;
    double rate;
    double sel_rate;
    double f_off;
    size_t D;
    size_t s0;
    size_t n;
    size_t off;
    size_t remaining;
    size_t saved_out = 0; /* complex samples written */
    float *buf;
    GtkWidget *dialog;
    GtkFileFilter *filter;
    GDateTime *now;
    gchar *stamp;
    char *filename;
    char *path = NULL;
    FILE *fp;
    bool ok = true;

    (void)item;
    if (ui->src == NULL) {
        return;
    }
    if (!dock_has_target(ui)) {
        set_info(ui, "Select a signal first (Shift+drag on the waterfall, or set a frequency range in the peak pane).");
        return;
    }
    if (!dock_target_region(ui, &r)) {
        set_info(ui, "Invalid selection region.");
        return;
    }
    rate = ui->src->sample_rate;
    s0 = (size_t)(r.time_min * rate);
    n = (size_t)((r.time_max - r.time_min) * rate);
    if (s0 >= ui->src->total_samples) {
        set_info(ui, "Selection out of bounds.");
        return;
    }
    if (s0 + n > ui->src->total_samples) {
        n = ui->src->total_samples - s0;
    }
    if (n == 0) {
        set_info(ui, "Selection too short.");
        return;
    }

    /* The saved file holds ONLY the signal inside the drawn rectangle:
     * the selected band is mixed down to zero offset and decimated to a
     * rate just above the selected bandwidth. The gqrx-style name must
     * therefore describe the NEW centre frequency and rate, otherwise
     * reopening the file would analyse the wrong band. */
    {
        double sel_bw = r.freq_max - r.freq_min;
        double sel_centre = 0.5 * (r.freq_min + r.freq_max);

        rate = ui->src->sample_rate;
        D = (size_t)(rate / (1.15 * (sel_bw > 0.0 ? sel_bw : rate)));
        if (D < 1) {
            D = 1;
        }
        sel_rate = rate / (double)D;
        f_off = sel_centre - ui->src->center_freq;
        now = g_date_time_new_now_local();
        stamp = g_date_time_format(now, "%Y%m%d_%H%M%S");
        g_date_time_unref(now);
        filename = g_strdup_printf("gqrx_%s_%.0f_%.0f_fc.raw", stamp,
                                   sel_centre, sel_rate);
        g_free(stamp);
    }

    dialog = gtk_file_chooser_dialog_new(
        SAVE_SEL_TITLE, GTK_WINDOW(ui->window), GTK_FILE_CHOOSER_ACTION_SAVE,
        SAVE_SEL_CANCEL, GTK_RESPONSE_CANCEL,
        SAVE_SEL_ACCEPT, GTK_RESPONSE_ACCEPT,
        NULL);
    filter = gtk_file_filter_new();
    gtk_file_filter_set_name(filter, SAVE_SEL_FILTER_NAME);
    gtk_file_filter_add_pattern(filter, "*.raw");
    gtk_file_chooser_add_filter(GTK_FILE_CHOOSER(dialog), filter);
    gtk_file_chooser_set_current_name(GTK_FILE_CHOOSER(dialog), filename);
    g_free(filename);
    if (gtk_dialog_run(GTK_DIALOG(dialog)) == GTK_RESPONSE_ACCEPT) {
        path = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dialog));
    }
    gtk_widget_destroy(dialog);
    if (path == NULL) {
        return; /* cancelled */
    }
    /* append ".raw" when missing (case-insensitive) */
    {
        size_t len = strlen(path);

        if (len < 4 || g_ascii_strcasecmp(path + len - 4, ".raw") != 0) {
            char *with_ext = g_strdup_printf("%s.raw", path);

            g_free(path);
            path = with_ext;
        }
    }

    fp = fopen(path, "wb");
    if (fp == NULL) {
        set_info(ui, "Cannot save the selection: could not open %s.", path);
        g_free(path);
        return;
    }
    buf = malloc(2 * (SAVE_SEL_CHUNK + 128) * sizeof(float));
    if (buf == NULL) {
        fclose(fp);
        remove(path);
        set_info(ui, "Cannot save the selection: out of memory.");
        g_free(path);
        return;
    }
    /* Extract the selected band: per chunk, mix the samples down by the
     * band's offset from the file centre, then decimate with the
     * anti-alias FIR. A 64-sample overlap carries the FIR lookahead
     * across chunk boundaries so the filtered output is seamless. */
    off = s0;
    remaining = n;
    {
        const size_t overlap = 64; /* dsp_fir_decimate's filter length */
        size_t hop = SAVE_SEL_CHUNK;
        float *dec = malloc(2 * (hop / D + 160) * sizeof(float));
        double dth = -2.0 * M_PI * f_off / rate;

        if (dec == NULL) {
            ok = false;
        }
        while (ok && remaining > 0) {
            size_t chunk = (remaining > hop) ? hop : remaining;
            size_t rd = chunk + overlap;
            size_t keep;
            size_t got;

            if (off + rd > ui->src->total_samples) {
                rd = ui->src->total_samples - off; /* clamp the lookahead */
            }
            iq_source_read_complex(ui->src, off, buf, rd);
            if (f_off != 0.0) {
                double th = fmod(dth * (double)off, 2.0 * M_PI);
                size_t i;

                for (i = 0; i < rd; i++) {
                    double c = cos(th), sn = sin(th);
                    double re = buf[2 * i], im = buf[2 * i + 1];

                    buf[2 * i] = (float)(re * c - im * sn);
                    buf[2 * i + 1] = (float)(re * sn + im * c);
                    th += dth;
                }
            }
            got = dsp_fir_decimate(buf, rd, D, dec);
            keep = (chunk + D - 1) / D;
            if (keep > got) {
                keep = got;
            }
            /* Always float32: the gqrx "_fc" filename convention means
             * float complex, so the saved band reopens correctly. */
            ok = save_sel_write_chunk(fp, dec, keep, IQ_FORMAT_FLOAT32);
            if (ok) {
                saved_out += keep;
            }
            off += chunk;
            remaining -= chunk;
        }
        free(dec);
    }
    free(buf);
    fclose(fp);
    if (!ok) {
        remove(path);
        set_info(ui, "Cannot save the selection: write failed.");
        g_free(path);
        return;
    }
    set_info(ui,
             "Selection saved to %s: %.1f kHz band at centre %.6f MHz, "
             "%.1f kHz rate, %zu samples (Float32 I/Q).", path,
             (r.freq_max - r.freq_min) / 1e3,
             0.5 * (r.freq_min + r.freq_max) / 1e6, sel_rate / 1e3,
             saved_out);
    g_free(path);
}

/* --- cyclostationary analysis window -------------------------------------- */

/* Centered white label on a dark drawing. */
/* Cache the SCF magnitudes (alpha-major) as a 1:1 RGB24 surface,
 * normalised over the whole grid. */
/* Cache the SSCA coherence grid (strip x CF, 0..1) as a 1:1 surface. */
/*
 * draw_cyclic_scf — render the spectral correlation function. With the
 * SSCA estimator selected the cached coherence grid is shown instead of
 * the FAM SCF; both are blitted with an aspect-preserving contain fit
 * and a negative y-scale, so row 0 sits at the BOTTOM.
 */
/*
 * draw_cyclic_cdp — plot the cyclic domain profile. Alpha spans
 * -fs_eff/2..+fs_eff/2 across the width; the y axis maps 0..1.1 with 0
 * at the BOTTOM. A dashed grey line marks alpha = 0, the profile is a
 * white polyline, and the detected peaks are red dots with crosses.
 */
/* --- fluctuation analysis page -------------------------------------------- */

/*
 * draw_fluct_acf — plot the mean autocorrelation |Rbar(tau)|/sigma^2.
 * Lags map linearly across the width; the y axis scales to
 * 1.1 * max(|mean_acf|) with 0 at the BOTTOM, and a light grey baseline
 * marks the zero line.
 */
/*
 * draw_fluct_f — plot the fluctuation F(tau). The y axis scales to
 * 1.1 * max(F, threshold) with 0 at the BOTTOM. Dashed lines mark F=1.0
 * (grey) and F=threshold (red); the peak is a red dot with a vertical
 * dashed red line at peak_lag.
 */
/* Open (or replace) the fluctuation analysis window. */
/* --- CPAR analysis page --------------------------------------------------- */

/*
 * draw_cpar_acf — plot the normalised autocorrelation. Lags map
 * linearly across the width; the y axis spans [0, 1.0] (scaled to 1.05)
 * with 0 at the BOTTOM. Candidate peaks are dots: grey for
 * non-significant ones (par <= threshold), red with a short vertical
 * tick for significant ones (par > threshold).
 */
/*
 * draw_cpar_par — plot the per-lag peak-to-average ratio. The y axis
 * spans max(1.1 * max(par), 1.2 * threshold) with 0 at the BOTTOM; a
 * dashed red line marks the significance threshold. Vertical stems rise
 * from zero to the PAR value at each candidate peak lag — cyan for
 * insignificant peaks, red for significant ones — over a light grey
 * baseline.
 */
/* --- higher-order statistics page ----------------------------------------- */

/* Cache the bispectrum magnitudes as a 1:1 RGB24 surface, normalised
 * over the whole grid. Cell (f1, f2) maps to pixel (x=f2, y=f1). */
/*
 * draw_hos_bisp — render the bispectrum image. The cached bs_n×bs_n
 * surface is blitted with an aspect-preserving contain fit and a
 * negative y-scale, so row 0 sits at the BOTTOM (same pattern as the
 * cyclic SCF draw).
 */
/* Open (or replace) the higher-order statistics window. */
/* --- dock-page Run handlers ----------------------------------------------- */

/*
 * The per-technique Run target: the current selection. A selection made
 * in the peak pane (the zoom window, narrower than the full band) also
 * counts: the analyses then run over that frequency range across the
 * visible time window. Callers gate on dock_has_target first and show
 * the "select a signal" feedback otherwise.
 */
static bool peak_band_limited(const Ui *ui)
{
    return ui->freq_offset > FULL_ZOOM_EPSILON ||
           ui->visible_bw < ui->src->sample_rate - FULL_ZOOM_EPSILON;
}

static bool dock_has_target(const Ui *ui)
{
    return ui->has_selection || peak_band_limited(ui);
}

static bool dock_target_region(Ui *ui, Region *r)
{
    if (ui->has_selection) {
        *r = ui->selection;
        return region_normalize(r, ui->src);
    }
    if (peak_band_limited(ui)) {
        /* peak-pane frequency window over the visible time range */
        r->freq_min = ui->src->center_freq - ui->src->sample_rate / 2.0 + ui->freq_offset;
        r->freq_max = r->freq_min + ui->visible_bw;
        r->time_min = ui->scroll_position;
        r->time_max = ui->scroll_position + ui->visible_seconds;
        return region_normalize(r, ui->src);
    }
    return false;
}

/* Feedback shown by selection-scoped techniques when no target exists. */
/* --- Cycle Detection page (CFAR multi-cycle detector) --------------------- */

/*
 * draw_cycledet_cdp — the detector statistic per candidate cycle
 * frequency, with the red dashed threshold line. X auto-scales to the
 * largest candidate CF, y to max(stat, threshold), 0 at the bottom.
 */
/* Shared Run for the Cycle Detection page: dispatches on ss_estimator
 * (0 Fluctuations, 1 CPAR, 2 CFAR cycle detector). */
/* --- Distribution Analysis (whole-file, peak-band frequency) --------------- */

/* Draw a min/max-normalised curve (0..1) into [0,w]x[0,h], 0 at bottom. */
/* |CF| vs omega (blue) + Gaussian reference exp(-w^2/2) (green dashed). */
/* MGF vs s (auto-scaled linear). */
/* --- Classic Analysis (selection-scoped) ---------------------------------- */

/* --- Pulse Trains (selection-scoped) -------------------------------------- */

/* Reset all Run-driven pages to their placeholder state. */
static void reset_analysis_pages(Ui *ui)
{
    /* Spectral Correlation (FAM + SSCA) */
    ui->has_cyclic = false;
    ui->has_ssca = false;
    cyclic_clear(&ui->cyclic);
    ssca_clear(&ui->ssca);
    if (ui->cyclic_surface != NULL) {
        cairo_surface_destroy(ui->cyclic_surface);
        ui->cyclic_surface = NULL;
    }
    if (ui->ssca_surface != NULL) {
        cairo_surface_destroy(ui->ssca_surface);
        ui->ssca_surface = NULL;
    }
    gtk_label_set_text(GTK_LABEL(ui->cyclic_label), RUN_PLACEHOLDER);
    gtk_widget_queue_draw(ui->cyclic_scf_drawing);
    gtk_widget_queue_draw(ui->cyclic_cdp_drawing);

    /* Cycle Detection */
    ui->has_cycledet = false;
    cycledet_clear(&ui->cycledet);
    gtk_label_set_text(GTK_LABEL(ui->cycledet_label), RUN_PLACEHOLDER);
    gtk_widget_queue_draw(ui->cycledet_cdp_drawing);

    /* Spread-Spectrum Detection (Fluctuations + CPAR) */
    ui->has_fluct = false;
    fluct_clear(&ui->fluct);
    gtk_label_set_text(GTK_LABEL(ui->fluct_label), RUN_PLACEHOLDER);
    gtk_widget_queue_draw(ui->fluct_acf_drawing);
    gtk_widget_queue_draw(ui->fluct_f_drawing);

    ui->has_cpar = false;
    cpar_clear(&ui->cpar);
    gtk_label_set_text(GTK_LABEL(ui->cpar_label), RUN_PLACEHOLDER);
    gtk_widget_queue_draw(ui->cpar_acf_drawing);
    gtk_widget_queue_draw(ui->cpar_par_drawing);

    /* Higher-Order Statistics (cumulants + cyclic cumulants) */
    ui->has_hos = false;
    hos_clear(&ui->hos);
    if (ui->hos_surface != NULL) {
        cairo_surface_destroy(ui->hos_surface);
        ui->hos_surface = NULL;
    }
    gtk_label_set_text(GTK_LABEL(ui->hos_label), RUN_PLACEHOLDER);
    gtk_widget_queue_draw(ui->hos_bisp_drawing);

    ui->has_ccc = false;
    ccc_clear(&ui->ccc);
    gtk_label_set_text(GTK_LABEL(ui->ccc_label), CCC_PLACEHOLDER);
}

static void on_screenshot_clicked(GtkButton *button, gpointer user_data)
{
    (void)button;
    take_screenshot((Ui *)user_data);
}

/* --- wheel zoom and playback handlers ------------------------------------- */

/*
 * Wheel zoom on the waterfall drawing: EVERY scroll direction zooms the
 * FREQUENCY axis centred at the pointer (vertical zooming is gone).
 */
/* Two-finger swipe up/down scrolls the waterfall through time; the scroll
 * gesture no longer zooms the frequency window (the peak view's rectangle is
 * driven by its own drag gestures only). Swipe up advances to newer data,
 * swipe down goes back; the pan helper keeps the forward wrap and the
 * backward clamp identical to the playback timer. */
static gboolean on_drawing_scroll(GtkWidget *widget, GdkEventScroll *event, gpointer user_data)
{
    Ui *ui = (Ui *)user_data;
    int height = gtk_widget_get_allocated_height(widget);
    double dy_px;

    if (height <= 0) {
        return FALSE;
    }
    if (event->direction == GDK_SCROLL_SMOOTH) {
        if (event->delta_y == 0.0) {
            return FALSE;
        }
        dy_px = event->delta_y * WF_SCROLL_PIXELS_PER_UNIT;
    } else if (event->direction == GDK_SCROLL_UP) {
        dy_px = -WF_SCROLL_NOTCH_PIXELS;
    } else if (event->direction == GDK_SCROLL_DOWN) {
        dy_px = WF_SCROLL_NOTCH_PIXELS;
    } else {
        return FALSE;   /* horizontal-only gestures do not scroll time */
    }
    /* positive delta = scroll down = newer rows: dragging up advances */
    pan_by_pixels(ui, -dy_px, height);
    update_range_label(ui);
    update_peak_spectrum(ui);
    gtk_widget_queue_draw(ui->drawing);
    gtk_widget_queue_draw(ui->peak_pane);
    return TRUE;
}

/* Play/Pause toggle for the waterfall playback (looping scroll). */
/* Force a full redraw at the (new) playback position. */
static void playback_jump(Ui *ui)
{
    /* the playback position changed: rebuild the surface from the render
     * buffer with the current dynamic-range mapping */
    if (ui->drawing != NULL) {
        int w = gtk_widget_get_allocated_width(ui->drawing);
        int h = gtk_widget_get_allocated_height(ui->drawing);

        if (w > 0 && h > 0) wf_update_from_renderbuf(ui, w, h);
    }
    update_range_label(ui);
    update_peak_spectrum(ui);
    gtk_widget_queue_draw(ui->drawing);
    gtk_widget_queue_draw(ui->peak_pane);
}

static void on_play_clicked(GtkButton *button, gpointer user_data)
{
    Ui *ui = (Ui *)user_data;

    (void)button;
    if (ui->playing) {
        ui->playing = false;
        gtk_button_set_label(GTK_BUTTON(ui->play_btn), PLAY_BTN_LABEL);
    } else {
        ui->playing = true;
        update_playhead(ui);
        ui->last_play_tick = g_get_monotonic_time();
        gtk_button_set_label(GTK_BUTTON(ui->play_btn), PAUSE_BTN_LABEL);
    }
}

/* Rewind: step the stream back by one visible window (clamped at zero). */
static void on_rewind_clicked(GtkButton *button, gpointer user_data)
{
    Ui *ui = (Ui *)user_data;

    (void)button;
    ui->scroll_position -= ui->visible_seconds;
    if (ui->scroll_position < 0.0) {
        ui->scroll_position = 0.0;
    }
    playback_jump(ui);
}

/* --- window size persistence ---------------------------------------------- */

/* Last size written to the config (avoids redundant saves on every
 * size-allocate). */
static int last_saved_w = 0;
static int last_saved_h = 0;

static char *window_conf_path(void)
{
    return g_build_filename(g_get_user_config_dir(), WINDOW_CONF_DIR,
                            WINDOW_CONF_FILE, NULL);
}

/* Persist the window size (atomically via GKeyFile) unless it is below
 * the enforced minimum. */
static void save_window_config(int w, int h)
{
    GKeyFile *kf;
    GError *err = NULL;
    char *dir;
    char *path;

    if (w < WINDOW_MIN_WIDTH || h < WINDOW_MIN_HEIGHT) {
        return; /* never persist sizes below the minimum */
    }
    if (w == last_saved_w && h == last_saved_h) {
        return; /* nothing changed */
    }
    last_saved_w = w;
    last_saved_h = h;
    kf = g_key_file_new();
    g_key_file_set_integer(kf, WINDOW_CONF_GROUP, WINDOW_CONF_WIDTH, w);
    g_key_file_set_integer(kf, WINDOW_CONF_GROUP, WINDOW_CONF_HEIGHT, h);
    dir = g_build_filename(g_get_user_config_dir(), WINDOW_CONF_DIR, NULL);
    path = window_conf_path();
    g_mkdir_with_parents(dir, 0755);
    if (!g_key_file_save_to_file(kf, path, &err) && err != NULL) {
        g_clear_error(&err); /* best effort */
    }
    g_free(dir);
    g_free(path);
    g_key_file_free(kf);
}

/* Restore a previously saved window size (must be >= the minimum). */
static void load_window_config(Ui *ui)
{
    GKeyFile *kf = g_key_file_new();
    GError *err = NULL;
    char *path = window_conf_path();
    int w = 0;
    int h = 0;

    if (g_key_file_load_from_file(kf, path, G_KEY_FILE_NONE, &err)) {
        w = g_key_file_get_integer(kf, WINDOW_CONF_GROUP, WINDOW_CONF_WIDTH, NULL);
        h = g_key_file_get_integer(kf, WINDOW_CONF_GROUP, WINDOW_CONF_HEIGHT, NULL);
    } else if (err != NULL) {
        g_clear_error(&err); /* a missing config file is fine */
    }
    if (w >= WINDOW_MIN_WIDTH && h >= WINDOW_MIN_HEIGHT) {
        gtk_window_set_default_size(GTK_WINDOW(ui->window), w, h);
        last_saved_w = w;
        last_saved_h = h;
    }
    g_key_file_free(kf);
    g_free(path);
}

static void on_window_size_allocate(GtkWidget *widget, GdkRectangle *allocation,
                                    gpointer user_data)
{
    (void)widget;
    (void)user_data;
    save_window_config(allocation->width, allocation->height);
}

static void on_window_size_save(GtkWidget *widget, gpointer user_data)
{
    int w = 0;
    int h = 0;

    (void)user_data;
    gtk_window_get_size(GTK_WINDOW(widget), &w, &h);
    save_window_config(w, h);
}

static void on_window_destroy(GtkWidget *widget, gpointer user_data)
{
    Ui *ui = (Ui *)user_data;

    if (ui->window == widget) {
        ui->window = NULL;
    }
    ui->drawing = NULL;
    ui->info_label = NULL;
    ui->range_label = NULL;
    ui->db_min_scale = NULL;
    ui->db_max_scale = NULL;
    ui->auto_check = NULL;
    ui->db_lock_check = NULL;
    ui->play_btn = NULL;
    ui->sel_bw_label = NULL;
    ui->zoom_bw_label = NULL;
    ui->demod_kind_combo = NULL;
    ui->demod_result_label = NULL;
    ui->demod_const_drawing = NULL;
    ui->demod_run_btn = NULL;
    ui->peak_pane = NULL;
}

/* --- info label and refresh timer ---------------------------------------- */

/* Human-readable bandwidth for the toolbar label. */
static void format_bandwidth(const Ui *ui, double hz, char *buf, size_t n)
{
    if (ui != NULL && ui->src != NULL && ui->src->is_audio) {
        if (fabs(hz) < 1000.0) {
            snprintf(buf, n, "%.1f Hz", hz);
        } else {
            snprintf(buf, n, "%.2f kHz", hz / 1e3);
        }
    } else if (fabs(hz) >= HZ_PER_MHZ) {
        snprintf(buf, n, "%.6f MHz", hz / HZ_PER_MHZ);
    } else {
        snprintf(buf, n, "%.1f kHz", hz / 1e3);
    }
}

/* Show the selected bandwidth in the bottom toolbar (the drag's live width
 * while a rectangle is being drawn). */
/* "zoomed bandwidth": the bandwidth currently shown in the peak view (the
 * green zoom window over the full-band spectrum). */
static void update_zoom_bw_label(Ui *ui)
{
    char text[96];
    char bw[48];

    if (ui == NULL || ui->zoom_bw_label == NULL) return;
    format_bandwidth(ui, ui->visible_bw, bw, sizeof(bw));
    snprintf(text, sizeof(text), "zoomed bandwidth: %s", bw);
    gtk_label_set_text(GTK_LABEL(ui->zoom_bw_label), text);
}

static void update_sel_bw_label(Ui *ui)
{
    char text[96];
    char bw[48];
    double hz = 0.0;
    bool have = false;

    if (ui == NULL || ui->sel_bw_label == NULL) return;
    update_zoom_bw_label(ui);
    if (ui->dragging) {
        hz = fabs(ui->drag_freq - ui->drag_anchor_freq);
        have = hz > 0.0;
    } else if (ui->has_selection) {
        hz = ui->selection.freq_max - ui->selection.freq_min;
        have = hz > 0.0;
    }
    if (have) {
        format_bandwidth(ui, hz, bw, sizeof(bw));
        snprintf(text, sizeof(text), "selected bandwidth: %s", bw);
    } else {
        snprintf(text, sizeof(text), "selected bandwidth: \xE2\x80\x94");
    }
    gtk_label_set_text(GTK_LABEL(ui->sel_bw_label), text);
}

/* Expire a transient bottom-right message; returns true when the message
 * is still live (the routine status must then wait). */
static bool info_message_active(Ui *ui)
{
    if (ui->info_expiry_us == 0) return false;
    if (g_get_monotonic_time() < ui->info_expiry_us) return true;
    ui->info_expiry_us = 0;
    gtk_label_set_text(GTK_LABEL(ui->info_label), "");
    return false;
}

static void update_info_label(Ui *ui)
{
    char text[LABEL_BUFFER_SIZE];

    if (ui->info_label == NULL) return;
    if (info_message_active(ui)) return;   /* transient notice takes over */
    if (ui->last_progress >= 0) {
        snprintf(text, sizeof(text), INFO_PROGRESS_FMT, ui->last_progress);
    } else {
        /* loaded: show the recording's filename */
        const char *base = strrchr(ui->src->path, '/');
        base = (base != NULL) ? base + 1 : ui->src->path;
        snprintf(text, sizeof(text), "%s", base);
    }
    gtk_label_set_text(GTK_LABEL(ui->info_label), text);
    update_sel_bw_label(ui);
}

/* Async analysis job: a deep copy of the datasets is analysed in a worker
 * thread (the engine only reads it) and the report is delivered back. */
typedef struct {
    Ui            *ui;
    FclScope       scope;
    bool           enabled[FC_MAX_NODES];
    int            n_enabled_nodes;
    FcDatasetList  snapshot;
    volatile int   cancelled;
    char          *text;
    char          *markdown;
    char          *html;
    char           err[256];
    size_t         nodes;
    double         elapsed_s;
    /* worker progress (read by the UI timer) */
    volatile int   prog_done;
    volatile int   prog_total;
    volatile int   prog_stage;
} FcJob;

/* stage index last rendered into the Frequency Counter status label */
static int fc_stage_shown = -1;

static gboolean on_timer(gpointer user_data)
{
    Ui *ui = (Ui *)user_data;
    int progress = wc_progress(ui->cache);

    /* poll the frequency-counter DAG (READ ONLY; the idle callback owns the
     * cleanup). fc_dag_run publishes the current node index and the number
     * of completed analyses through the volatile counters. */
    if (ui->fc_job != NULL) {
        FcJob *fjob = (FcJob *)ui->fc_job;
        int done = fjob->prog_done;
        int total = fjob->prog_total;
        int stage = fjob->prog_stage;

        if (total <= 0) total = (int)fc_node_count;
        if (done < 0) done = 0;
        if (done > total) done = total;
        if (ui->fc_progress != NULL) {
            char text[96];

            gtk_progress_bar_set_fraction(
                GTK_PROGRESS_BAR(ui->fc_progress),
                total > 0 ? (double)done / (double)total : 0.0);
            snprintf(text, sizeof(text), "%d / %d analyses", done, total);
            gtk_progress_bar_set_text(GTK_PROGRESS_BAR(ui->fc_progress), text);
            gtk_widget_set_visible(ui->fc_progress, TRUE);
        }
        if (stage >= 0 && stage < (int)fc_node_count &&
            stage != fc_stage_shown && ui->fc_status_label != NULL) {
            char text[256];

            fc_stage_shown = stage;
            snprintf(text, sizeof(text), "Analysing %d/%d: %s…", stage + 1,
                     total, fc_node_specs[stage].title);
            gtk_label_set_text(GTK_LABEL(ui->fc_status_label), text);
        }
    }

    /* a transient bottom-right message expires after 5 seconds and the
     * routine status (filename/progress) comes back */
    if (ui->info_expiry_us != 0 && !info_message_active(ui)) {
        update_info_label(ui);
    }

    /* poll the async Analyze job (READ ONLY — the idle
     * callbacks own the cleanup) */
    if (current_report_job != NULL) {
        ReportJob *job = current_report_job;
        int p = g_atomic_int_get(&job->progress);

        if (!gtk_widget_get_visible(ui->report_progress)) {
            gtk_widget_set_visible(ui->report_progress, TRUE);
        }
        /* Selection tab, right of Save: name the analysis currently
         * running; it is cleared when the job finishes (deliver/discard) */
        if (p < REPORT_TOTAL_MAX && job->stage != NULL &&
            ui->analysis_status != NULL) {
            if (job_stage_shown != job->stage) {
                job_stage_shown = job->stage;
                {
                    char text[160];

                    snprintf(text, sizeof(text), "Analyzing: %s",
                             job->stage);
                    gtk_label_set_text(GTK_LABEL(ui->analysis_status), text);
                }
            }
        }
        gtk_progress_bar_set_fraction(GTK_PROGRESS_BAR(ui->report_progress),
                                      (job->total > 0) ? p / (double)job->total : 0.0);
        /* watchdog: cancel the job when it stalls */
        if (p != job_last_progress_val) {
            job_last_progress_val = p;
            job_last_progress_us = g_get_monotonic_time();
        }
        {
            gint64 now = g_get_monotonic_time();
            int hb = g_atomic_int_get(&job->heartbeat);

            /* a heartbeat bump means the worker thread is alive: a slow
             * stage is NOT a stall and must not cancel the DAG */
            if (hb != job_last_heartbeat) {
                job_last_heartbeat = hb;
                job_last_heartbeat_us = now;
                if (job_stall_reported) {
                    job_stall_reported = 0;
                    gtk_widget_set_visible(ui->report_progress, TRUE);
                    set_info(ui, "Analysis resumed (%s)", job->stage);
                }
            }
            if (!g_atomic_int_get(&job->done) &&
                now - job_last_heartbeat_us > JOB_STALL_US) {
                if (!job_stall_reported) {
                    job_stall_reported = 1;
                    fprintf(stderr,
                            "te15: analysis has not heart-beaten for %.0f s "
                            "in stage '%s' - still waiting (the watchdog no "
                            "longer cancels slow stages)\n",
                            (double)(now - job_last_heartbeat_us) / 1e6,
                            job->stage != NULL ? job->stage : "unknown");
                    set_info(ui, "Analysis is slow in stage '%s' (%.0f s) - "
                                 "still running", job->stage);
                }
            }
            /* absolute limit: report whatever completed instead of
             * silently dropping the whole analysis */
            if (!g_atomic_int_get(&job->done) && job->started_us > 0 &&
                now - job->started_us > JOB_HARD_LIMIT_US) {
                fprintf(stderr,
                        "te15: analysis exceeded the %.0f min limit in stage "
                        "'%s' - cancelling and reporting partial results\n",
                        (double)JOB_HARD_LIMIT_US / 60.0 / 1e6,
                        job->stage != NULL ? job->stage : "unknown");
                job->cancelled = 1;
                set_info(ui, "Analysis exceeded the time limit - reporting "
                             "partial results");
                current_report_job = NULL;
                job_last_progress_val = -1;
            }
        }
    }

    if (progress != ui->last_progress) {
        ui->last_progress = progress;
        update_info_label(ui);
        update_range_label(ui);
        update_peak_spectrum(ui);
        gtk_widget_queue_draw(ui->drawing);
        gtk_widget_queue_draw(ui->peak_pane);
    }
    if (ui->playing) {
        /* advance the looping playback by the elapsed wall time */
        gint64 now = g_get_monotonic_time();
        double dt = (now - ui->last_play_tick) / USEC_PER_SEC;

        ui->last_play_tick = now;
        ui->scroll_position += dt;
        /* looping playback: once the last rendered row reaches the end of
         * the file the one-row zero marker is streamed, the playback
         * position resets and the beginning follows seamlessly */
        wrap_scroll_dir(ui, 1);
        update_playhead(ui);
        {
            double fps = wc_frames_per_second(ui->cache);

            wc_request(ui->cache, (size_t)(ui->scroll_position * fps),
                       2 * (size_t)(ui->visible_seconds * fps) + 1);
            /* prefetch the head window so the wrap renders immediately */
            if (ui->scroll_position >
                ui->src->duration - 2 * ui->visible_seconds) {
                wc_request(ui->cache, 0,
                           2 * (size_t)(ui->visible_seconds * fps) + 1);
            }
        }
        update_range_label(ui);
        /* the spectrum strip does not need 20 Hz refresh */
        if (now - last_peak_update >= PEAK_UPDATE_INTERVAL_US) {
            last_peak_update = now;
            update_peak_spectrum(ui);
            gtk_widget_queue_draw(ui->peak_pane);
        }
        gtk_widget_queue_draw(ui->drawing);
    }
    return G_SOURCE_CONTINUE;
}

/* --- waterfall rendering -------------------------------------------------- */

static void rebuild_waterfall_surface(Ui *ui, int w, int h)
{
    if (ui->waterfall_surface != NULL) {
        cairo_surface_destroy(ui->waterfall_surface);
    }
    ui->waterfall_surface = cairo_image_surface_create(CAIRO_FORMAT_RGB24, w, h);
    cairo_t *cr = cairo_create(ui->waterfall_surface);

    cairo_set_source_rgb(cr, 0.0, 0.0, 0.0); /* black background */
    cairo_paint(cr);
    cairo_destroy(cr);
}

static void draw_loading_text(cairo_t *cr, int w, int h)
{
    cairo_set_source_rgb(cr, 1.0, 1.0, 1.0);
    cairo_select_font_face(cr, FONT_FAMILY, CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_NORMAL);
    cairo_set_font_size(cr, FONT_SIZE_LOADING);
    cairo_text_extents_t extents;

    cairo_text_extents(cr, INFO_LOADING, &extents);
    cairo_move_to(cr, (w - extents.width) / 2.0,
                  h / 2.0 - extents.y_bearing - extents.height / 2.0);
    cairo_show_text(cr, INFO_LOADING);
}

static void compose_surface(cairo_t *cr, cairo_surface_t *surface)
{
    cairo_set_source_surface(cr, surface, 0, 0);
    cairo_paint(cr);
}

/*
 * Map a (freq, time) rectangle to widget pixels in the current view:
 * frequency maps linearly to the horizontal axis, time to the vertical
 * axis with the earliest time at the bottom. Returns false when the
 * rectangle lies entirely outside the visible view; otherwise the
 * clipped pixel rect is written to x0/y0/x1/y1.
 */
static bool map_region_pixels(Ui *ui, double fmin, double fmax, double tmin, double tmax,
                              int w, int h, int *x0, int *y0, int *x1, int *y1)
{
    double view_start = ui->scroll_position;
    double freq_lo;
    double freq_span;
    double dur;
    double d_max;
    double d_min;

    if (w <= 0 || h <= 0) {
        return false;
    }
    dur = ui->src->duration;
    if (dur <= 0.0) {
        return false;
    }
    freq_lo = ui->src->center_freq - ui->src->sample_rate / 2.0 + ui->freq_offset;
    freq_span = ui->visible_bw;

    (void)freq_lo;
    (void)freq_span;
    *x0 = (int)x_from_freq(ui, fmin, w);
    *x1 = (int)x_from_freq(ui, fmax, w);

    /* wrap-aware time mapping: d = wrapped offset of t from the view
     * start; visible iff d < visible_seconds */
    d_max = fmod(tmax - view_start, dur);
    if (d_max < 0.0) {
        d_max += dur;
    }
    d_min = fmod(tmin - view_start, dur);
    if (d_min < 0.0) {
        d_min += dur;
    }
    *y0 = (int)(h * (d_max / ui->visible_seconds));
    *y1 = (int)(h * (d_min / ui->visible_seconds));

    /* clip to the widget */
    if (*x0 < 0) {
        *x0 = 0;
    }
    if (*x1 > w) {
        *x1 = w;
    }
    if (*y0 < 0) {
        *y0 = 0;
    }
    if (*y1 > h) {
        *y1 = h;
    }
    /* the time axis reads downward, so the earlier time maps to the
     * SMALLER y: swap if the caller's tmin/tmax produced an inverted
     * rect (otherwise the rectangle would be dropped as degenerate) */
    if (*y1 < *y0) {
        int tmp = *y0;

        *y0 = *y1;
        *y1 = tmp;
    }
    return (*x1 > *x0) && (*y1 > *y0);
}

/* Hit-test the current selection: returns a mask of the edges grabbed
 * (bitwise OR of SEL_EDGE_*), SEL_EDGE_INSIDE when the point is well
 * inside the rectangle, or 0 when the point misses it. */
static int selection_hit(const Ui *ui, double x, double y, int w, int h,
                         int *inside)
{
    int x0, y0, x1, y1;
    int edges = 0;

    *inside = 0;
    if (!ui->has_selection) {
        return 0;
    }
    if (!map_region_pixels((Ui *)ui, ui->selection.freq_min,
                           ui->selection.freq_max, ui->selection.time_min,
                           ui->selection.time_max, w, h, &x0, &y0, &x1, &y1)) {
        return 0;
    }
    if (x < x0 - SEL_EDGE_PX || x > x1 + SEL_EDGE_PX ||
        y < y0 - SEL_EDGE_PX || y > y1 + SEL_EDGE_PX) {
        return 0;
    }
    if (fabs(x - x0) <= SEL_EDGE_PX) edges |= SEL_EDGE_LEFT;
    if (fabs(x - x1) <= SEL_EDGE_PX) edges |= SEL_EDGE_RIGHT;
    if (fabs(y - y0) <= SEL_EDGE_PX) edges |= SEL_EDGE_TOP;
    if (fabs(y - y1) <= SEL_EDGE_PX) edges |= SEL_EDGE_BOTTOM;
    if (edges == 0 && x > x0 && x < x1 && y > y0 && y < y1) {
        edges = SEL_EDGE_INSIDE;
    }
    return edges;
}

/* Remove the selection rectangle (and leave the whole-file mode). */
static void clear_selection_state(Ui *ui)
{
    ui->has_selection = false;
    ui->sel_full_height = false;
    ui->sel_press_inside = false;
    memset(&ui->sel_saved, 0, sizeof(ui->sel_saved));
}

/*
 * True when a widget pixel lies on the current selection rectangle.
 * In the full-height (whole-file) mode the rectangle has no height on the
 * time axis (its range is the whole file, which maps to a degenerate pixel
 * band), so only its horizontal span is tested there.
 */
static bool selection_contains_px(Ui *ui, double x, double y, int w,
                                  int h)
{
    double x0, x1;

    if (!ui->has_selection) return false;
    x0 = x_from_freq(ui, ui->selection.freq_min, w);
    x1 = x_from_freq(ui, ui->selection.freq_max, w);
    if (x1 < x0) {
        double t = x0;

        x0 = x1;
        x1 = t;
    }
    if (x < x0 - 3.0 || x > x1 + 3.0) return false;
    if (ui->sel_full_height) return true; /* covers the whole height */
    {
        int px0, py0, px1, py1;

        if (!map_region_pixels(ui, ui->selection.freq_min,
                               ui->selection.freq_max,
                               ui->selection.time_min,
                               ui->selection.time_max, w, h, &px0, &py0,
                               &px1, &py1)) {
            return false;
        }
        return y >= py0 - 2 && y <= py1 + 2;
    }
}

/* State-only toggle of the whole-file (blue) selection mode: the
 * rectangle keeps its frequency span but covers the entire recording;
 * toggling again restores the previously drawn time range. The caller
 * refreshes the UI (details page, labels, redraw). */
static void toggle_full_height_selection_state(Ui *ui)
{
    if (ui->src == NULL || !ui->has_selection) return;
    if (!ui->sel_full_height) {
        ui->sel_saved = ui->selection;
        ui->sel_full_height = true;
        ui->selection.time_min = 0.0;
        ui->selection.time_max = ui->src->duration;
    } else {
        ui->sel_full_height = false;
        ui->selection = ui->sel_saved;
    }
    ui->dragging = false;
    ui->sel_press_inside = false;
}

static void toggle_full_height_selection(Ui *ui)
{
    if (ui->src == NULL || !ui->has_selection) return;
    toggle_full_height_selection_state(ui);
    update_details_page(ui);
    update_info_label(ui);
    set_info(ui, "%s: %.6f-%.6f MHz, %.3f-%.3f s",
             ui->sel_full_height
                 ? "Full-height selection (whole file)"
                 : "Selection",
             ui->selection.freq_min / HZ_PER_MHZ,
             ui->selection.freq_max / HZ_PER_MHZ,
             ui->selection.time_min, ui->selection.time_max);
    if (ui->drawing != NULL) {
        gtk_widget_queue_draw(ui->drawing);
    }
}

/* Paint the selection rectangle, or the live drag rectangle. */
static void draw_selection_overlay(Ui *ui, cairo_t *cr, int w, int h)
{
    double fmin, fmax, tmin, tmax;
    int x0, y0, x1, y1;

    if (ui->dragging && !ui->sel_resizing) {
        fmin = MIN(ui->drag_anchor_freq, ui->drag_freq);
        fmax = MAX(ui->drag_anchor_freq, ui->drag_freq);
        tmin = MIN(ui->drag_anchor_time, ui->drag_time);
        tmax = MAX(ui->drag_anchor_time, ui->drag_time);
    } else if (ui->has_selection) {
        /* the whole-file mode has no height on the time axis: its time
         * range is the entire recording, which maps to a degenerate
         * band - draw it as a full-height column instead */
        if (ui->sel_full_height) {
            int xa = (int)x_from_freq(ui, ui->selection.freq_min, w);
            int xb = (int)x_from_freq(ui, ui->selection.freq_max, w);

            if (xb < xa) {
                int t = xa;

                xa = xb;
                xb = t;
            }
            if (xb <= xa) xb = xa + 1;
            if (xb > w) xb = w;
            if (xa < 0) xa = 0;
            cairo_set_source_rgba(cr, SEL_FULL_LINE_R, SEL_FULL_LINE_G,
                                  SEL_FULL_LINE_B, SEL_FULL_FILL_A);
            cairo_rectangle(cr, xa, 0, xb - xa, h);
            cairo_fill(cr);
            cairo_set_source_rgb(cr, SEL_FULL_LINE_R, SEL_FULL_LINE_G,
                                 SEL_FULL_LINE_B);
            cairo_set_line_width(cr, SEL_LINE_WIDTH);
            cairo_rectangle(cr, xa, 0, xb - xa, h);
            cairo_stroke(cr);
            return;
        }
        fmin = ui->selection.freq_min;
        fmax = ui->selection.freq_max;
        tmin = ui->selection.time_min;
        tmax = ui->selection.time_max;
    } else {
        return;
    }
    if (!map_region_pixels(ui, fmin, fmax, tmin, tmax, w, h,
                           &x0, &y0, &x1, &y1)) {
        return;
    }
    if (ui->sel_full_height && !ui->dragging) {
        /* whole-file mode: the rectangle keeps its frequency span but is
         * drawn from the top to the bottom of the waterfall, in blue */
        cairo_set_source_rgba(cr, SEL_FULL_LINE_R, SEL_FULL_LINE_G,
                              SEL_FULL_LINE_B, SEL_FULL_FILL_A);
        cairo_rectangle(cr, x0, 0, x1 - x0, h);
        cairo_fill(cr);
        cairo_set_source_rgb(cr, SEL_FULL_LINE_R, SEL_FULL_LINE_G,
                             SEL_FULL_LINE_B);
        cairo_set_line_width(cr, SEL_LINE_WIDTH);
        cairo_rectangle(cr, x0, 0, x1 - x0, h);
        cairo_stroke(cr);
    } else {
        cairo_set_source_rgba(cr, SEL_FILL_R, SEL_FILL_G, SEL_FILL_B,
                              SEL_FILL_A);
        cairo_rectangle(cr, x0, y0, x1 - x0, y1 - y0);
        cairo_fill(cr);
        cairo_set_source_rgb(cr, SEL_LINE_R, SEL_LINE_G, SEL_LINE_B);
        cairo_set_line_width(cr, SEL_LINE_WIDTH);
        cairo_rectangle(cr, x0, y0, x1 - x0, y1 - y0);
        cairo_stroke(cr);
    }
}

/*
 * draw_waterfall — repaint the main waterfall view.
 *
 * Pipeline per draw pass:
 *   1. Rebuild the offscreen RGB24 surface at the widget's size (kept
 *      between draws so screenshots can be written from it).
 *   2. Lock the cache window covering the visible frame range and paint
 *      the waterfall — ALL cache reads happen while the lock is held,
 *      and the lock is released immediately after the pixel loop.
 *      Rows are decimated to the pixel height and columns to the pixel
 *      width, so each pass costs O(w*h) instead of O(frames*bins).
 *   3. Paint the selection / live-drag rectangle on top (after unlock).
 *   4. Compose the finished surface onto the widget's cairo context.
 */

/* Paint the visible frequency bins of one spectrum row onto pixel row p. */
static void paint_row_columns(cairo_t *cr, const float *row, int p, int w,
                              int b0, int b1, int nbins, int col_step,
                              const Ui *ui)
{
    for (int j = b0; j < b1; j += col_step) {
        Rgb8 c = wf_db_colour(ui, row[j]);
        int j_end = j + col_step;
        int x0 = (j - b0) * w / nbins;
        int x1;

        if (j_end > b1) {
            j_end = b1;
        }
        x1 = (j_end - b0) * w / nbins;
        if (x1 > w) {
            x1 = w;
        }
        if (x1 <= x0) {
            x1 = x0 + 1;
        }
        cairo_set_source_rgb(cr, c.r / 255.0, c.g / 255.0, c.b / 255.0);
        cairo_rectangle(cr, x0, p, x1 - x0, 1);
        cairo_fill(cr);
    }
}

/*
 * Audio spectrogram painting: one column per pixel, each column samples
 * the LOG-spaced frequency from the view and maps it to the nearest FFT
 * bin (taking the strongest bin in the column's bin span so narrow tones
 * do not fall between columns).
 */
static void paint_row_log(cairo_t *cr, const float *row, int p, int w,
                          int h, int fft_n, double bin_hz, const Ui *ui)
{
    double lo = view_lo_hz(ui);
    double hi = view_hi_hz(ui);
    (void)h;
    if (hi <= lo) return;
    for (int x = 0; x < w; x++) {
        double f0 = lo * pow(hi / lo, (double)x / (double)w);
        double f1 = lo * pow(hi / lo, (double)(x + 1) / (double)w);
        double off0 = f0 - (ui->src->center_freq - ui->src->sample_rate / 2.0);
        double off1 = f1 - (ui->src->center_freq - ui->src->sample_rate / 2.0);
        int k0 = (int)floor(off0 / bin_hz);
        int k1 = (int)ceil(off1 / bin_hz);
        int best = k0;
        float bv;

        if (k0 < 0) k0 = 0;
        if (k1 <= k0) k1 = k0 + 1;
        if (k1 > fft_n) k1 = fft_n;
        if (k0 >= fft_n) continue;
        bv = row[k0];
        for (int k = k0; k < k1; k++) {
            if (row[k] > bv) {
                bv = row[k];
                best = k;
            }
        }
        {
            Rgb8 c = wf_db_colour(ui, bv);

            cairo_set_source_rgb(cr, c.r / 255.0, c.g / 255.0, c.b / 255.0);
            cairo_rectangle(cr, x, p, 1, 1);
            cairo_fill(cr);
        }
        (void)best;
    }
}


/*
 * draw_waterfall — repaint the main waterfall view.
 *
 * Pipeline per draw pass:
 *   1. Rebuild the offscreen RGB24 surface at the widget's size (kept
 *      between draws so screenshots can be written from it).
 *   2. Auto-scale scan (tail window + head window when wrapping), then
 *      two paint passes: the tail segment [start, total) via the window
 *      covering start_frame, and the head segment [0, wrap_end) via a
 *      window locked at frame 0. Locks never nest.
 *   3. Paint the selection / live-drag rectangle on top (after unlock).
 *   4. Compose the finished surface onto the widget's cairo context.
 */
/* ------------------------------------------------------------------ */
/* streaming waterfall renderer                                        */
/*                                                                     */
/* The waterfall is a stream: the file's samples are transformed into  */
/* rows on the fly and the newest rows are appended at the bottom      */
/* while older rows slide up. The visible surface is kept between      */
/* frames, so playing (or scrolling down) shifts it by exactly the     */
/* number of rows that elapsed and paints only the newly exposed ones  */
/* - the waterfall never clears, jumps or reverses. Scrolling past the */
/* end wraps to the file's beginning (rows keep being appended);       */
/* scrolling backwards stops at the start of the file.                 */
/* ------------------------------------------------------------------ */

/* Signature of everything that changes the pixel mapping or colouring. */
static uint64_t wf_signature(const Ui *ui, int w, int h)
{
    uint64_t s = 1469598103934665603ULL;
    double fps = wc_frames_per_second(ui->cache);

#define WFS_MIX(v) do { s ^= (uint64_t)(v); s *= 1099511628211ULL; } while (0)
    WFS_MIX((uint64_t)w);
    WFS_MIX((uint64_t)h);
    WFS_MIX((uint64_t)wc_fft_size(ui->cache));
    WFS_MIX((uint64_t)wc_total_frames(ui->cache));
    WFS_MIX((uint64_t)llround(fps));
    WFS_MIX((uint64_t)llround(ui->visible_seconds * 1e6));
    WFS_MIX((uint64_t)llround(ui->visible_bw));
    WFS_MIX((uint64_t)llround(ui->freq_offset));
    WFS_MIX((uint64_t)llround(ui->wf_range_lo * 100.0));
    WFS_MIX((uint64_t)llround(ui->wf_range_hi * 100.0));
    WFS_MIX(ui->auto_scale ? 1u : 0u);
    WFS_MIX(audio_view(ui) ? 1u : 0u);
    WFS_MIX((uint64_t)llround(ui->src != NULL ? ui->src->sample_rate : 0.0));
    return s;
#undef WFS_MIX
}

/* Frequency (and audio log-axis) window for the current view. */
static void wf_freq_window(const Ui *ui, int w, int fft_n, double *bin_hz,
                           int *b0, int *b1, int *nbins, int *col_step)
{
    double hz = ui->src->sample_rate / fft_n;
    int lo = (int)floor(ui->freq_offset / hz);
    int hi = (int)ceil((ui->freq_offset + ui->visible_bw) / hz);

    if (lo < 0) lo = 0;
    if (hi > fft_n) hi = fft_n;
    if (hi <= lo) hi = lo + 1;
    *bin_hz = hz;
    *b0 = lo;
    *b1 = hi;
    *nbins = hi - lo;
    *col_step = (int)ceil((double)(hi - lo) / (double)(w > 0 ? w : 1));
    if (*col_step < 1) *col_step = 1;
}

/* Absolute stream frame pixel row `p` shows (may pass the file end; the
 * buffer maps it modulo the recording length so the stream loops). */
static bool wf_row_frame(const Ui *ui, int h, int p, size_t *out)
{
    double fps = wc_frames_per_second(ui->cache);
    size_t total = wc_total_frames(ui->cache);
    double t;

    if (fps <= 0.0 || total == 0 || h <= 0) return false;
    t = ui->wf_top_time + ((double)p + 0.5) * ui->visible_seconds / (double)h;
    if (t < 0.0) t = 0.0;
    *out = (size_t)floor(t * fps);
    return true;
}

/* Fetch one row from the render buffer. The waterfall never talks to the
 * forward/backward streaming buffers: it mirrors the render buffer, which
 * in turn is filled from the streaming buffers. */
static bool wf_row_get(Ui *ui, size_t frame, float *db_out)
{
    if (ui->renderbuf != NULL) return rb_row(ui->renderbuf, frame, db_out);
    if (ui->streambuf == NULL) return false;
    return sb_row(ui->streambuf, frame, db_out);
}

/* Mirror the rows the next paint/range scan needs into the render buffer
 * (contiguous segments: the stream wraps at most once per view). */
static void wf_fill_rows(Ui *ui, int h, int y0, int nr)
{
    size_t total = wc_total_frames(ui->cache);
    size_t period = wf_stream_period_frames(ui, h);
    size_t seg_first = 0, seg_count = 0;
    bool have = false;

    if (ui->renderbuf == NULL || total == 0) return;
    if (y0 < 0) y0 = 0;
    for (int p = y0; p < y0 + nr; p++) {
        size_t frame;
        size_t start, count;

        if (!wf_row_frame(ui, h, p, &frame)) break;
        if (period > total) {
            size_t k = frame % period;

            if (k >= total) {          /* the red loop-marker row */
                if (have) rb_fill(ui->renderbuf, seg_first, seg_count);
                have = false;
                continue;
            }
            frame = k;
        } else {
            frame %= total;
        }
        /* one contiguous frame run per segment */
        if (have && frame == seg_first + seg_count) {
            seg_count++;
            continue;
        }
        if (have) rb_fill(ui->renderbuf, seg_first, seg_count);
        seg_first = frame;
        seg_count = 1;
        have = true;
        (void)start;
        (void)count;
    }
    if (have) rb_fill(ui->renderbuf, seg_first, seg_count);
}

/* Ensure the single-row buffer fits the cache's FFT size. */
static bool wf_row_buffer(Ui *ui, int fft_n)
{
    if (ui->wf_rowbuf != NULL && ui->wf_rowbuf_bins >= fft_n) return true;
    free(ui->wf_rowbuf);
    ui->wf_rowbuf = malloc((size_t)fft_n * sizeof(float));
    ui->wf_rowbuf_bins = ui->wf_rowbuf != NULL ? fft_n : 0;
    return ui->wf_rowbuf != NULL;
}

/* Paint the loop-marker pixel row: the stream holds one row of complex
 * zeros here (zero magnitude), drawn red so the loop boundary is visible. */
static void paint_loop_row(cairo_t *cr, int p, int w)
{
    cairo_set_source_rgb(cr, LOOP_ROW_R, LOOP_ROW_G, LOOP_ROW_B);
    cairo_rectangle(cr, 0, p, w, 1);
    cairo_fill(cr);
}

/* Paint pixel rows [y0, y0 + nr) using the current window and colour scale. */
static void wf_paint_rows(Ui *ui, int w, int h, int y0, int nr)
{
    int fft_n = wc_fft_size(ui->cache);
    int b0, b1, nbins, col_step;
    double bin_hz;
    float range;
    cairo_t *cr;

    if (fft_n <= 0 || !wf_row_buffer(ui, fft_n)) return;
    wf_freq_window(ui, w, fft_n, &bin_hz, &b0, &b1, &nbins, &col_step);
    range = ui->wf_display_max - ui->wf_display_min;
    if (range < DB_RANGE_MIN_SPAN) range = DB_RANGE_MIN_SPAN;
    if (y0 < 0) y0 = 0;
    if (y0 + nr > h) nr = h - y0;
    wf_fill_rows(ui, h, y0, nr);   /* compute the rows from the render buffer */
    {
        size_t total = wc_total_frames(ui->cache);
        size_t period = wf_stream_period_frames(ui, h);

        cr = cairo_create(ui->waterfall_surface);
        for (int p = y0; p < y0 + nr; p++) {
            size_t frame;

            if (!wf_row_frame(ui, h, p, &frame)) break;
            if (period > total) {
                size_t k = frame % period;

                if (k >= total) {
                    /* the one-pixel-row silence at the loop point */
                    paint_loop_row(cr, p, w);
                    continue;
                }
                frame = k;
            } else {
                frame %= total;
            }
            if (!wf_row_get(ui, frame, ui->wf_rowbuf)) continue;
            if (audio_view(ui)) {
            paint_row_log(cr, ui->wf_rowbuf, p, w, h, fft_n, bin_hz,
                              ui);
        } else {
                paint_row_columns(cr, ui->wf_rowbuf, p, w, b0, b1, nbins,
                                  col_step, ui);
            }
        }
        cairo_destroy(cr);
    }
    /* keep the buffer centred on the last row painted */
    if (ui->streambuf != NULL && y0 + nr > 0) {
        size_t last;

        if (wf_row_frame(ui, h, y0 + nr - 1, &last)) {
            sb_anchor(ui->streambuf, last);
        }
    }
}

/* Colour scale for a fresh surface: auto-scale over sampled visible rows,
 * otherwise the fixed slider range. Kept stable between redraws so the
 * appended rows match the ones already on screen. */
static void wf_pick_range(Ui *ui, int w, int h)
{
    int fft_n = wc_fft_size(ui->cache);
    int b0, b1, nbins, col_step;
    double bin_hz;
    double lo = FLT_MAX, hi = -FLT_MAX;

    if (!ui->auto_scale || fft_n <= 0 || !wf_row_buffer(ui, fft_n)) {
        /* the committed window comes from the sliders (applied on release,
         * never while dragging): the in-memory colour scheme is built from
         * it and the renderer only looks values up in that copy */
        float dlo = (float)MIN(ui->wf_range_lo, ui->wf_range_hi);
        float dhi = (float)MAX(ui->wf_range_lo, ui->wf_range_hi);

        if (dhi - dlo < DB_RANGE_MIN_SPAN) {
            dhi = dlo + (float)DB_RANGE_MIN_SPAN;
        }
        ui->wf_display_min = dlo;
        ui->wf_display_max = dhi;
        return;
    }
    wf_freq_window(ui, w, fft_n, &bin_hz, &b0, &b1, &nbins, &col_step);
    wf_fill_rows(ui, h, 0, h);
    {
        int step = h > 96 ? h / 96 : 1;

        size_t total = wc_total_frames(ui->cache);
        size_t period = wf_stream_period_frames(ui, h);

        for (int p = 0; p < h; p += step) {
            size_t frame;

            if (!wf_row_frame(ui, h, p, &frame)) continue;
            if (period > total) {
                size_t k = frame % period;

                if (k >= total) continue; /* the silence marker */
                frame = k;
            } else {
                frame %= total;
            }
            if (!wf_row_get(ui, frame, ui->wf_rowbuf)) continue;
            for (int j = b0; j < b1; j += col_step) {
                if (ui->wf_rowbuf[j] < lo) lo = ui->wf_rowbuf[j];
                if (ui->wf_rowbuf[j] > hi) hi = ui->wf_rowbuf[j];
            }
        }
    }
    if (lo == FLT_MAX || hi == -FLT_MAX || hi - lo < DB_RANGE_MIN_SPAN) {
        lo = ui->db_min;
        hi = ui->db_max;
    }
    ui->wf_display_min = (float)lo;
    ui->wf_display_max = (float)hi;
    ui->wf_range_lo = lo;
    ui->wf_range_hi = hi;
}

/* The playback position in stream time: the leading (last rendered) edge of
 * the stream. It advances with playback and stays put when the user scrolls
 * or pauses, so the waterfall can draw a red line where playback is. */
static void update_playhead(Ui *ui)
{
    double period = wf_stream_period_time(ui);

    if (period <= 0.0) {
        ui->playhead_valid = false;
        return;
    }
    ui->playhead_time = fmod(ui->scroll_position + ui->visible_seconds,
                             period);
    if (ui->playhead_time < 0.0) ui->playhead_time += period;
    ui->playhead_valid = true;
}

/* Draw the red playback-position line over the waterfall. The line is only
 * drawn when the playback position falls inside the visible time span (it
 * can leave the view when the user scrolls back while paused). */
static void draw_playhead(Ui *ui, cairo_t *cr, int w, int h)
{
    double period, delta, yv;

    if (!ui->playhead_valid || w <= 0 || h <= 0 ||
        ui->visible_seconds <= 0.0) {
        return;
    }
    period = wf_stream_period_time(ui);
    delta = ui->playhead_time - ui->wf_top_time;
    if (period > 0.0) {
        while (delta < 0.0) delta += period;
        while (delta >= period) delta -= period;
    }
    if (delta < 0.0 || delta > ui->visible_seconds) return;
    yv = delta / ui->visible_seconds * (double)h;
    if (yv < 0.0) yv = 0.0;
    if (yv > (double)h - 1.0) yv = (double)h - 1.0;
    cairo_save(cr);
    cairo_set_source_rgba(cr, PLAYHEAD_R, PLAYHEAD_G, PLAYHEAD_B, 0.95);
    cairo_set_line_width(cr, PLAYHEAD_WIDTH);
    cairo_move_to(cr, 0.0, yv + 0.5);
    cairo_line_to(cr, (double)w, yv + 0.5);
    cairo_stroke(cr);
    /* a small marker at the right edge makes the line easy to spot */
    cairo_move_to(cr, (double)w, yv + 0.5);
    cairo_line_to(cr, (double)w - 10.0, yv - 5.0);
    cairo_line_to(cr, (double)w - 10.0, yv + 6.0);
    cairo_close_path(cr);
    cairo_fill(cr);
    cairo_restore(cr);
}

/* How many pixel rows the kept surface has to move to follow the stream:
 * positive = the stream advanced (new rows at the bottom), negative = the
 * user scrolled back (new rows at the top). A forward wrap past the end of
 * the file appears as a large backward jump on the time axis and is folded
 * back into the stream here. */
static double wf_pending_rows(const Ui *ui, int h)
{
    double period = wf_stream_period_time(ui);
    double period_px = (period > 0.0 && ui->visible_seconds > 0.0)
                           ? period / ui->visible_seconds * (double)h : 0.0;
    double dp;

    if (ui->visible_seconds <= 0.0 || h <= 0) return 0.0;
    dp = (ui->scroll_position - ui->wf_top_time) / ui->visible_seconds *
         (double)h;
    while (dp < -0.5 * (double)h && period_px > 0.0) dp += period_px;
    return dp;
}

/*
 * Update the whole waterfall from the render buffer: mirror every row the
 * visible view needs (the render buffer is filled from the streaming
 * buffers, so this is the only place the renderer reads spectrogram rows
 * from), derive the colour range from the two dynamic-range sliders and
 * paint all rows with that mapping. Used for every full repaint: scrolling,
 * a change of the playback position and a change of the dynamic range all
 * go through here, so the colour transformation is always applied
 * mathematically to the rows held in the render buffer.
 */
static void wf_update_from_renderbuf(Ui *ui, int w, int h)
{
    if (w <= 0 || h <= 0) return;
    if (wc_fft_size(ui->cache) <= 0 || wc_total_frames(ui->cache) == 0) {
        ui->wf_valid = false;
        return;
    }
    rebuild_waterfall_surface(ui, w, h);
    ui->wf_top_time = ui->scroll_position;
    wf_pick_range(ui, w, h);        /* the sliders' mapping, mathematically */
    wf_fill_rows(ui, h, 0, h);      /* complete visible range from the rb    */
    wf_paint_rows(ui, w, h, 0, h);
    ui->wf_w = w;
    ui->wf_h = h;
    ui->wf_sig = wf_signature(ui, w, h);
    ui->wf_valid = true;
}

/* Redraw the whole surface at the current view position. */
static void wf_full_redraw(Ui *ui, int w, int h)
{
    wf_update_from_renderbuf(ui, w, h);
    gtk_widget_queue_draw(ui->drawing);
}

/* Shift the kept surface by `d` rows and paint the newly exposed ones.
 * d > 0: the stream advanced, rows enter at the bottom; d < 0: the user
 * scrolled back, rows enter at the top. */
static void wf_shift(Ui *ui, int w, int h, int d)
{
    cairo_t *cr;

    if (d == 0) return;
    if (ui->wf_scratch != NULL &&
        (cairo_image_surface_get_width(ui->wf_scratch) != w ||
         cairo_image_surface_get_height(ui->wf_scratch) != h)) {
        cairo_surface_destroy(ui->wf_scratch);
        ui->wf_scratch = NULL;
    }
    if (ui->wf_scratch == NULL) {
        ui->wf_scratch = cairo_image_surface_create(CAIRO_FORMAT_RGB24, w, h);
    }
    /* scratch = old surface shifted by -d (content moves up for d > 0) */
    cr = cairo_create(ui->wf_scratch);
    cairo_set_source_surface(cr, ui->waterfall_surface, 0.0, -(double)d);
    cairo_paint(cr);
    cairo_destroy(cr);
    /* main = scratch */
    cr = cairo_create(ui->waterfall_surface);
    cairo_set_operator(cr, CAIRO_OPERATOR_SOURCE);
    cairo_set_source_surface(cr, ui->wf_scratch, 0.0, 0.0);
    cairo_paint(cr);
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);
    cairo_destroy(cr);

    ui->wf_top_time += (double)d * ui->visible_seconds / (double)h;
    if (d > 0) {
        wf_paint_rows(ui, w, h, h - d, d);
    } else {
        wf_paint_rows(ui, w, h, 0, -d);
    }
}

/* In-widget tag: bold 11 px text on a translucent dark rounded backing,
 * centred on `x` and clamped inside the widget width. `y` is the text
 * baseline. Used by the cursor frequency read-out (red) and by the
 * stationary-hover hint (white) so both look the same. */
static void draw_overlay_tag(cairo_t *cr, const char *text, double x,
                             double y, int w, double r, double g, double b)
{
    cairo_text_extents_t ext;
    double tx;

    cairo_select_font_face(cr, FONT_FAMILY, CAIRO_FONT_SLANT_NORMAL,
                           CAIRO_FONT_WEIGHT_BOLD);
    cairo_set_font_size(cr, CURSOR_FREQ_FONT_SIZE);
    cairo_text_extents(cr, text, &ext);
    tx = x - ext.width / 2.0;
    if (tx < 4.0) tx = 4.0;
    if (tx + ext.width > w - 4.0) tx = w - 4.0 - ext.width;
    {
        double bx = tx - 4.0;
        double by = y - ext.height - 1.0;
        double bw = ext.width + 8.0;
        double bh = ext.height + 5.0;
        double rad = 3.0;

        cairo_new_path(cr);
        cairo_move_to(cr, bx + rad, by);
        cairo_line_to(cr, bx + bw - rad, by);
        cairo_arc(cr, bx + bw - rad, by + rad, rad, -M_PI / 2.0, 0.0);
        cairo_line_to(cr, bx + bw, by + bh - rad);
        cairo_arc(cr, bx + bw - rad, by + bh - rad, rad, 0.0, M_PI / 2.0);
        cairo_line_to(cr, bx + rad, by + bh);
        cairo_arc(cr, bx + rad, by + bh - rad, rad, M_PI / 2.0, M_PI);
        cairo_line_to(cr, bx, by + rad);
        cairo_arc(cr, bx + rad, by + rad, rad, M_PI, 3.0 * M_PI / 2.0);
        cairo_close_path(cr);
        cairo_set_source_rgba(cr, 0.0, 0.0, 0.0, 0.65);
        cairo_fill(cr);
    }
    cairo_set_source_rgb(cr, r, g, b);
    cairo_move_to(cr, tx, y);
    cairo_show_text(cr, text);
}

/* Red frequency read-out painted at the top of the waterfall, centred on
 * the cursor's x position (it disappears when the pointer leaves). */
static void draw_cursor_frequency(const Ui *ui, cairo_t *cr, int w)
{
    char text[64];

    if (!ui->cursor_active || ui->src == NULL) return;
    format_freq_hz(ui, ui->cursor_freq_hz, text, sizeof(text));
    draw_overlay_tag(cr, text, ui->cursor_x, CURSOR_FREQ_MARGIN, w,
                     1.0, 0.15, 0.15);
}

/* White "Shift+Click to select" tag: the same in-widget style as the
 * frequency read-out, just below it. */
static void draw_hover_hint(const Ui *ui, cairo_t *cr, int w)
{
    if (!ui->hover_hint_visible) return;
    draw_overlay_tag(cr, HOVER_HINT_TEXT,
                     ui->cursor_active ? ui->cursor_x : w / 2.0,
                     CURSOR_FREQ_MARGIN + CURSOR_FREQ_LINE_HEIGHT, w,
                     1.0, 1.0, 1.0);
}

static gboolean draw_waterfall(GtkWidget *widget, cairo_t *cr, gpointer user_data)
{
    Ui *ui = (Ui *)user_data;
    int w = gtk_widget_get_allocated_width(widget);
    int h = gtk_widget_get_allocated_height(widget);

    if (w <= 0 || h <= 0) {
        return FALSE;
    }
    /* the stream only advances: forward wraps at the file end, backwards
     * stops at the beginning of the file */
    wrap_scroll(ui);

    if (wc_fft_size(ui->cache) <= 0 || wc_total_frames(ui->cache) == 0 ||
        ui->visible_seconds <= 0.0) {
        draw_loading_text(cr, w, h);
        return FALSE;
    }

    {
        uint64_t sig = wf_signature(ui, w, h);
        bool need_full = !ui->wf_valid || ui->wf_w != w || ui->wf_h != h ||
                         ui->wf_sig != sig;
        double dp = wf_pending_rows(ui, h);

        if (!need_full && fabs(dp) >= 0.5) {
            int d = (int)llround(dp);

            if (d != 0 && abs(d) < h) {
                /* wf_shift() moves the surface AND advances wf_top_time by
                 * exactly the same number of rows, so the kept and the
                 * newly painted rows stay on one consistent frame grid.
                 * Re-anchoring wf_top_time to scroll_position here would
                 * make the assumed top differ from the top the kept rows
                 * were painted for by up to half a row: neighbouring rows
                 * then map to the same frame (duplicates) or skip one. */
                wf_shift(ui, w, h, d);
            } else if (d != 0) {
                need_full = true;
            }
        }
        if (need_full) {
            wf_full_redraw(ui, w, h);
        }
    }
    if (!ui->wf_valid) {
        draw_loading_text(cr, w, h);
        return FALSE;
    }
    compose_surface(cr, ui->waterfall_surface);
    /* overlays are painted on the widget, never into the kept surface, and
     * use the same view model as the selection code */
    draw_selection_overlay(ui, cr, w, h);
    draw_playhead(ui, cr, w, h);
    draw_cursor_frequency(ui, cr, w);
    draw_hover_hint(ui, cr, w);
    return FALSE;
}

/* --- peak pane (full-band spectrum strip) -------------------------------- */

/*
 * Compute the mean dB spectrum over the visible frames (wc_fft_size
 * bins), clipped to the locked cache window's overlap like
 * draw_waterfall. The buffer follows the cache's FFT size; ui_cleanup
 * frees it. Sets peak_valid = false when no data is available.
 */
static void update_peak_spectrum_cached(Ui *ui)
{
    double fps;
    size_t total_frames;
    size_t start_frame;
    size_t visible_frames;
    size_t win_start = 0;
    size_t win_frames = 0;
    size_t win_step = 1;
    size_t i0;
    size_t i1;
    const float *data;
    float inv;
    int fft_n = wc_fft_size(ui->cache);

    if (fft_n <= 0) {
        ui->peak_valid = false;
        return;
    }
    if (ui->peak_spectrum == NULL || peak_alloc_bins != fft_n) {
        free(ui->peak_spectrum);
        ui->peak_spectrum = malloc((size_t)fft_n * sizeof(float));
        if (ui->peak_spectrum == NULL) {
            peak_alloc_bins = 0;
            ui->peak_valid = false;
            return;
        }
        peak_alloc_bins = fft_n;
    }
    fps = wc_frames_per_second(ui->cache);
    total_frames = wc_total_frames(ui->cache);
    start_frame = (size_t)(ui->scroll_position * fps);
    visible_frames = (size_t)(ui->visible_seconds * fps);
    if (visible_frames == 0) {
        visible_frames = 1;
    }
    if (start_frame + visible_frames > total_frames) {
        start_frame = (total_frames >= visible_frames)
                          ? total_frames - visible_frames
                          : 0;
    }
    data = wc_lock(ui->cache, start_frame, &win_start, &win_frames, &win_step);
    if (data == NULL) {
        ui->peak_valid = false;
        return;
    }
    /* stride-aware visible row range, like draw_waterfall */
    if (start_frame > win_start) {
        i0 = (start_frame - win_start + win_step - 1) / win_step;
    } else {
        i0 = 0;
    }
    if (start_frame + visible_frames <= win_start) {
        i1 = 0;
    } else {
        i1 = (start_frame + visible_frames - win_start + win_step - 1) / win_step;
        if (i1 > win_frames) {
            i1 = win_frames;
        }
    }
    if (i1 <= i0) {
        wc_unlock(ui->cache);
        ui->peak_valid = false;
        return;
    }
    memset(ui->peak_spectrum, 0, (size_t)fft_n * sizeof(float));
    for (size_t i = i0; i < i1; i++) {
        const float *row = data + i * fft_n;

        for (int b = 0; b < fft_n; b++) {
            ui->peak_spectrum[b] += row[b];
        }
    }
    wc_unlock(ui->cache);
    inv = 1.0f / (float)(i1 - i0);
    for (int b = 0; b < fft_n; b++) {
        ui->peak_spectrum[b] *= inv;
    }
    ui->peak_valid = true;
}

/* The peak strip shows the mean spectrum of the waterfall pixels that are
 * currently on screen, fetched from the RENDER BUFFER (the same rows the
 * renderer displays, via wf_row_frame()/rb_row()), so it matches the visible
 * data and averages over the view instead of showing one noisy row. Up to
 * PEAK_MEAN_ROWS evenly spaced pixel rows are averaged. */
static void update_peak_spectrum(Ui *ui)
{
    int fft_n = wc_fft_size(ui->cache);
    int h = ui->wf_h > 0 ? ui->wf_h : 480;
    float *row;
    int sampled = 0;
    int step;

    if (fft_n <= 0 || wc_total_frames(ui->cache) == 0 || h <= 0 ||
        ui->src == NULL) {
        ui->peak_valid = false;
        return;
    }
    if (ui->peak_spectrum == NULL || peak_alloc_bins != fft_n) {
        free(ui->peak_spectrum);
        ui->peak_spectrum = malloc((size_t)fft_n * sizeof(float));
        if (ui->peak_spectrum == NULL) {
            peak_alloc_bins = 0;
            ui->peak_valid = false;
            return;
        }
        peak_alloc_bins = fft_n;
    }
    if (ui->renderbuf == NULL) {
        update_peak_spectrum_cached(ui);
        return;
    }
    row = malloc((size_t)fft_n * sizeof(float));
    if (row == NULL) {
        update_peak_spectrum_cached(ui);
        return;
    }
    step = h > PEAK_MEAN_ROWS ? (int)((double)h / PEAK_MEAN_ROWS) : 1;
    if (step < 1) step = 1;
    memset(ui->peak_spectrum, 0, (size_t)fft_n * sizeof(float));
    for (int p = 0; p < h; p += step) {
        size_t frame;

        if (!wf_row_frame(ui, h, p, &frame)) continue;
        if (!rb_row(ui->renderbuf, frame, row)) continue;
        for (int b = 0; b < fft_n; b++) ui->peak_spectrum[b] += row[b];
        sampled++;
    }
    free(row);
    if (sampled == 0) {
        update_peak_spectrum_cached(ui);
        return;
    }
    {
        float inv = 1.0f / (float)sampled;

        for (int b = 0; b < fft_n; b++) ui->peak_spectrum[b] *= inv;
    }
    ui->peak_valid = true;
}

/*
 * draw_peak_pane — the full-band mean spectrum with the green zoom
 * rectangle (the horizontal zoom window from the view model) and 4px
 * handle ticks at both edges.
 */
/* plot area of the peak strip (everything right of the dB axis) */
static double peak_plot_w(int w)
{
    return (double)(w - PEAK_DB_AXIS_W);
}

static double peak_y_for_db(int h, double db)
{
    double span = DB_SCALE_MAX - DB_SCALE_MIN;

    if (db < DB_SCALE_MIN) db = DB_SCALE_MIN;
    if (db > DB_SCALE_MAX) db = DB_SCALE_MAX;
    return 1.0 + (DB_SCALE_MAX - db) / span * ((double)h - 2.0);
}

/* A "nice" tick step for the allowed dB range (5..8 ticks). */
static double peak_tick_step(void)
{
    static const double steps[] = { 5.0, 10.0, 20.0, 25.0, 50.0 };
    double span = DB_SCALE_MAX - DB_SCALE_MIN;

    for (size_t i = 0; i < sizeof(steps) / sizeof(steps[0]); i++) {
        if (span / steps[i] <= 8.0) return steps[i];
    }
    return 50.0;
}

static gboolean draw_peak_pane(GtkWidget *widget, cairo_t *cr, gpointer user_data)
{
    Ui *ui = (Ui *)user_data;
    int w = gtk_widget_get_allocated_width(widget);
    int h = gtk_widget_get_allocated_height(widget);
    double sr;
    double x0;
    double x1;

    if (w <= 0 || h <= 0) {
        return FALSE;
    }
    cairo_set_source_rgb(cr, PEAK_BG_R, PEAK_BG_G, PEAK_BG_B);
    cairo_paint(cr);

    if (!ui->peak_valid || ui->peak_spectrum == NULL || peak_alloc_bins < 2) {
        cairo_set_source_rgb(cr, 0.4, 0.4, 0.4);
        cairo_select_font_face(cr, FONT_FAMILY, CAIRO_FONT_SLANT_NORMAL,
                               CAIRO_FONT_WEIGHT_NORMAL);
        cairo_set_font_size(cr, 9.0);
        cairo_move_to(cr, 4, h / 2.0 + 3);
        cairo_show_text(cr, PEAK_NO_DATA);
        return FALSE;
    }
    {
        double pw = peak_plot_w(w);
        /* The shading follows the sliders live (every value change), unlike
         * the waterfall colour scheme which commits on release. Auto mode
         * shades the auto window. */
        double sel_lo = MIN(ui->auto_scale ? ui->wf_range_lo : ui->db_min,
                            ui->auto_scale ? ui->wf_range_hi : ui->db_max);
        double sel_hi = MAX(ui->auto_scale ? ui->wf_range_lo : ui->db_min,
                            ui->auto_scale ? ui->wf_range_hi : ui->db_max);
        double step = peak_tick_step();
        double y_top, y_bot;

        if (pw <= 2.0) pw = 2.0;
        if (sel_lo < DB_SCALE_MIN) sel_lo = DB_SCALE_MIN;
        if (sel_hi > DB_SCALE_MAX) sel_hi = DB_SCALE_MAX;
        y_top = peak_y_for_db(h, sel_hi);   /* top of the selected range */
        y_bot = peak_y_for_db(h, sel_lo);   /* bottom of it              */

        /* the spectrum, mapped onto the allowed dB axis (values outside it
         * are clamped to the axis ends, like the colour scheme does) */
        cairo_set_source_rgba(cr, PEAK_FILL_R, PEAK_FILL_G, PEAK_FILL_B,
                              PEAK_FILL_A);
        cairo_move_to(cr, PEAK_DB_AXIS_W, (double)h);
        for (int b = 0; b < peak_alloc_bins; b++) {
            double x = PEAK_DB_AXIS_W +
                       (double)b / (double)(peak_alloc_bins - 1) * pw;
            double y = peak_y_for_db(h, ui->peak_spectrum[b]);

            cairo_line_to(cr, x, y);
        }
        cairo_line_to(cr, (double)w, (double)h);
        cairo_close_path(cr);
        cairo_fill(cr);

        /* shade the areas outside the selected min..max range (50 %
         * opaque blue), before the curve so it stays readable */
        cairo_set_source_rgba(cr, PEAK_SHADE_R, PEAK_SHADE_G, PEAK_SHADE_B,
                              PEAK_SHADE_A);
        if (y_top > 1.0) {
            cairo_rectangle(cr, 0.0, 1.0, (double)w, y_top - 1.0);
            cairo_fill(cr);
        }
        if (y_bot < (double)h - 1.0) {
            cairo_rectangle(cr, 0.0, y_bot, (double)w,
                            (double)h - 1.0 - y_bot);
            cairo_fill(cr);
        }


        /* dB ticks on the left, calculated from the allowed dB range */
        cairo_set_source_rgb(cr, 0.62, 0.66, 0.62);
        cairo_set_line_width(cr, 1.0);
        cairo_select_font_face(cr, FONT_FAMILY, CAIRO_FONT_SLANT_NORMAL,
                               CAIRO_FONT_WEIGHT_NORMAL);
        cairo_set_font_size(cr, 8.0);
        for (double db = DB_SCALE_MIN; db <= DB_SCALE_MAX + 1e-9;
             db += step) {
            double y = peak_y_for_db(h, db);
            char label[24];

            cairo_move_to(cr, (double)PEAK_DB_AXIS_W - PEAK_DB_TICK_LEN, y);
            cairo_line_to(cr, (double)PEAK_DB_AXIS_W, y);
            cairo_stroke(cr);
            snprintf(label, sizeof(label), "%.0f", db);
            {
                cairo_text_extents_t ext;

                cairo_text_extents(cr, label, &ext);
                cairo_move_to(cr,
                              (double)PEAK_DB_AXIS_W - PEAK_DB_TICK_LEN - 4.0 -
                                  ext.width,
                              y + ext.height / 2.0);
                cairo_show_text(cr, label);
            }
        }
        cairo_set_source_rgb(cr, 0.35, 0.38, 0.35);
        cairo_move_to(cr, (double)PEAK_DB_AXIS_W + 0.5, 1.0);
        cairo_line_to(cr, (double)PEAK_DB_AXIS_W + 0.5, (double)h - 1.0);
        cairo_stroke(cr);

        /* green zoom-window rectangle, clipped to the plot area */
        sr = ui->src->sample_rate;
        x0 = PEAK_DB_AXIS_W + (ui->freq_offset / sr) * pw;
        x1 = PEAK_DB_AXIS_W + ((ui->freq_offset + ui->visible_bw) / sr) * pw;
        if (x0 < PEAK_DB_AXIS_W) x0 = PEAK_DB_AXIS_W;
        if (x1 > (double)w) x1 = (double)w;
        cairo_set_source_rgba(cr, PEAK_RECT_R, PEAK_RECT_G, PEAK_RECT_B,
                              PEAK_RECT_A);
        cairo_rectangle(cr, x0, 0, x1 - x0, h);
        cairo_fill(cr);
        cairo_set_source_rgb(cr, PEAK_LINE_R, PEAK_LINE_G, PEAK_LINE_B);
        cairo_set_line_width(cr, PEAK_LINE_WIDTH);
        cairo_rectangle(cr, x0, 0, x1 - x0, h);
        cairo_stroke(cr);

        /* 4px-wide vertical handle ticks at both edges */
        cairo_set_line_width(cr, PEAK_HANDLE_WIDTH);
        cairo_move_to(cr, x0, 0);
        cairo_line_to(cr, x0, h);
        cairo_move_to(cr, x1, 0);
        cairo_line_to(cr, x1, h);
        cairo_stroke(cr);

        /* The areas outside the selected min..max dB range are shaded
         * 50 % opaque blue over the finished plot; the plot area only, so
         * the dB axis and its labels stay readable. */
        cairo_set_source_rgba(cr, PEAK_SHADE_R, PEAK_SHADE_G, PEAK_SHADE_B,
                              PEAK_SHADE_A);
        if (y_top > 1.0) {
            cairo_rectangle(cr, PEAK_DB_AXIS_W, 1.0, pw, y_top - 1.0);
            cairo_fill(cr);
        }
        if (y_bot < (double)h - 1.0) {
            cairo_rectangle(cr, PEAK_DB_AXIS_W, y_bot, pw,
                            (double)h - 1.0 - y_bot);
            cairo_fill(cr);
        }
    }
    return FALSE;
}

/* Press on the peak pane: grab an edge, pan the window, or nothing. */
static gboolean on_peak_press(GtkWidget *widget, GdkEventButton *event, gpointer user_data)
{
    Ui *ui = (Ui *)user_data;
    int width = gtk_widget_get_allocated_width(widget);
    double sr;
    double x0;
    double x1;

    if (event->button != 1) {
        return FALSE;
    }
    if (event->type == GDK_2BUTTON_PRESS) {
        /* double-click: reset to the full band */
        ui->visible_bw = ui->src->sample_rate;
        ui->freq_offset = 0.0;
        update_sel_bw_label(ui);
        gtk_widget_queue_draw(ui->peak_pane);
        gtk_widget_queue_draw(ui->drawing);
        if (!ui->has_selection) {
            update_details_page(ui);
        }
        return TRUE;
    }
    /* at full size, clicking anywhere starts drawing a new rectangle */
    if (ui->freq_offset <= FULL_ZOOM_EPSILON &&
        ui->visible_bw >= ui->src->sample_rate - FULL_ZOOM_EPSILON) {
        ui->peak_handle = 3; /* drawing mode */
        ui->drag_pan_x = event->x;
        return TRUE;
    }
    if (width <= PEAK_DB_AXIS_W + 2) {
        return TRUE;
    }
    if (event->x < (double)PEAK_DB_AXIS_W) {
        return TRUE;   /* clicks on the dB axis do not zoom */
    }
    sr = ui->src->sample_rate;
    {
        double pw = peak_plot_w(width);

        x0 = PEAK_DB_AXIS_W + (ui->freq_offset / sr) * pw;
        x1 = PEAK_DB_AXIS_W + ((ui->freq_offset + ui->visible_bw) / sr) * pw;
    }
    if (fabs(event->x - x0) <= PEAK_HANDLE_GRAB) {
        ui->peak_handle = -1; /* left edge */
    } else if (fabs(event->x - x1) <= PEAK_HANDLE_GRAB) {
        ui->peak_handle = 1; /* right edge */
    } else if (event->x >= x0 && event->x <= x1) {
        ui->peak_handle = 2; /* pan the window */
        ui->drag_pan_x = event->x;
    } else {
        /* outside the rectangle: start drawing a new one */
        ui->peak_handle = 3;
        ui->drag_pan_x = event->x;
    }
    return TRUE;
}

/* Drag the grabbed handle/edge; the waterfall follows live. */
static gboolean on_peak_motion(GtkWidget *widget, GdkEventMotion *event, gpointer user_data)
{
    Ui *ui = (Ui *)user_data;
    int width = gtk_widget_get_allocated_width(widget);
    double sr;
    double f_lo;
    double f_hi;

    if (ui->peak_handle == 0 || width <= PEAK_DB_AXIS_W + 2) {
        return FALSE;
    }
    if ((event->state & GDK_BUTTON1_MASK) == 0) {
        return FALSE;
    }
    sr = ui->src->sample_rate;
    f_lo = ui->src->center_freq - sr / 2.0;
    f_hi = ui->src->center_freq + sr / 2.0;
    if (ui->peak_handle == 3) {
        /* drawing mode: the new rectangle spans [anchor, pointer] */
        double pw = peak_plot_w(width);
        double x_a = MIN(ui->drag_pan_x, event->x) - PEAK_DB_AXIS_W;
        double x_b = MAX(ui->drag_pan_x, event->x) - PEAK_DB_AXIS_W;
        double new_lo, new_hi, new_bw;

        if (x_a < 0.0) x_a = 0.0;
        if (x_b > pw) x_b = pw;
        new_lo = x_a / pw * sr;
        new_hi = x_b / pw * sr;
        new_bw = new_hi - new_lo;

        if (new_bw >= MIN_FREQ_BW) {
            ui->freq_offset = new_lo;
            ui->visible_bw = new_bw;
            if (ui->freq_offset < 0.0) {
                ui->freq_offset = 0.0;
            }
            if (ui->freq_offset > sr - ui->visible_bw) {
                ui->freq_offset = sr - ui->visible_bw;
            }
        }
        /* else: a click without a real drag leaves the zoom unchanged */
        update_sel_bw_label(ui);
        gtk_widget_queue_draw(ui->peak_pane);
        gtk_widget_queue_draw(ui->drawing);
        return TRUE;
    }
    if (ui->peak_handle == -1) {
        /* left edge: keep the right edge fixed */
        double ratio = (event->x - PEAK_DB_AXIS_W) / peak_plot_w(width);
        double f;

        if (ratio < 0.0) ratio = 0.0;
        if (ratio > 1.0) ratio = 1.0;
        f = f_lo + ratio * sr;
        double hi_abs = f_lo + ui->freq_offset + ui->visible_bw;

        if (f > hi_abs - MIN_FREQ_BW) {
            f = hi_abs - MIN_FREQ_BW;
        }
        if (f < f_lo) {
            f = f_lo;
        }
        ui->visible_bw = hi_abs - f;
        ui->freq_offset = f - f_lo;
    } else if (ui->peak_handle == 1) {
        /* right edge: keep the left edge fixed */
        double ratio = (event->x - PEAK_DB_AXIS_W) / peak_plot_w(width);
        double f;

        if (ratio < 0.0) ratio = 0.0;
        if (ratio > 1.0) ratio = 1.0;
        f = f_lo + ratio * sr;
        double lo_abs = f_lo + ui->freq_offset;

        if (f < lo_abs + MIN_FREQ_BW) {
            f = lo_abs + MIN_FREQ_BW;
        }
        if (f > f_hi) {
            f = f_hi;
        }
        ui->visible_bw = f - lo_abs;
    } else {
        /* pan both edges */
        double dx = (event->x - ui->drag_pan_x) * sr /
                    peak_plot_w(width);

        ui->drag_pan_x = event->x;
        ui->freq_offset -= dx;
        if (ui->freq_offset < 0.0) {
            ui->freq_offset = 0.0;
        }
        if (ui->freq_offset > sr - ui->visible_bw) {
            ui->freq_offset = sr - ui->visible_bw;
        }
    }
    update_sel_bw_label(ui);
    gtk_widget_queue_draw(ui->peak_pane);
    gtk_widget_queue_draw(ui->drawing);
    return TRUE;
}

static gboolean on_peak_release(GtkWidget *widget, GdkEventButton *event, gpointer user_data)
{
    Ui *ui = (Ui *)user_data;

    (void)widget;
    (void)event;
    ui->peak_handle = 0;
    /* the peak-pane window is final: refresh the live Selection tab when
     * the waterfall selection is not overriding it */
    if (!ui->has_selection) {
        update_details_page(ui);
    }
    return TRUE;
}

/* --- public API ---------------------------------------------------------- */

void ui_init(Ui *ui, IqSource *src, WaterfallCache *cache)
{
    memset(ui, 0, sizeof(*ui));
    ui->visible_seconds = DEFAULT_VISIBLE_SECONDS;
    ui->visible_bw = src->sample_rate; /* full band by default */
    ui->freq_offset = 0.0;
    ui->db_min = DEFAULT_DB_MIN;
    ui->db_max = DEFAULT_DB_MAX;
    ui->auto_scale = true;
    /* the waterfall colour scheme starts from the default slider window;
     * the committed window is what the renderer uses (sliders commit on
     * release) */
    ui->wf_range_lo = MIN(ui->db_min, ui->db_max);
    ui->wf_range_hi = MAX(ui->db_min, ui->db_max);
    ui->has_selection = false;
    ui->dragging = false;
    report_save_btn = NULL;
    ui->has_cyclic = false;
    ui->has_ssca = false;
    ui->has_cycledet = false;
    ui->has_fluct = false;
    ui->has_cpar = false;
    ui->has_hos = false;
    ui->has_ccc = false;
    ui->cyclic_estimator = 0; /* FAM */
    ui->ss_estimator = 0;      /* Fluctuations */
    ui->last_progress = -1;
    ui->stream_buffer_fraction = 1.0 / 8.0;
    fcd_list_init(&ui->fc_datasets);
    for (size_t i = 0; i < FC_MAX_NODES; i++) ui->fc_node_enabled[i] = true;
    ui->fc_node_check_count = 0;
    ui->streambuf = sb_create(cache, ui->stream_buffer_fraction);
    if (ui->streambuf != NULL) sb_wait_ready(ui->streambuf, 2000);
    ui->renderbuf = rb_create(cache, ui->streambuf);
    ui->aconfig_path = g_build_filename(g_get_user_config_dir(),
                                        "ToxicElement15", "analysis.conf",
                                        NULL);
    aconfig_load(ui->aconfig_path);
    ui->report_format = REPORT_TEXT;
    ui->settings_rate = src->sample_rate;
    ui->settings_format = src->format;
    ui->settings_center = 0.0; /* keep the filename centre */
    ui->settings_fft_size = FFT_SIZE;
    ui->settings_window = DSP_WINDOW_HANN;
    /* demodulation filter defaults (playback-only control) */
    ui->demod_filter_enabled = false;
    ui->demod_filter_kind = 2; /* band-pass */
    ui->demod_filter_f1_hz = 0.0;
    ui->demod_filter_f2_hz = 0.0;
    ui->demod_filter_taps = 127;
    ui->src = src;
    ui->cache = cache;
    viridis_build(&ui->colormap);
}

/* --- Settings page -------------------------------------------------------- */

/* Demodulation filter control (Settings page). Playback-only: the
 * filter config is stored for the (future) player path and is never
 * consulted by any analysis. */
static void on_settings_filter_changed(GtkWidget *widget, gpointer user_data)
{
    Ui *ui = (Ui *)user_data;
    const char *text;
    int kind;
    int taps;

    (void)widget;
    if (ui->settings_filter_enable == NULL ||
        ui->settings_filter_kind == NULL) {
        return; /* construction in progress */
    }
    kind = gtk_combo_box_get_active(GTK_COMBO_BOX(ui->settings_filter_kind));
    if (kind < 0 || kind > 3) {
        kind = 0;
    }
    ui->demod_filter_kind = kind;
    ui->demod_filter_enabled = gtk_toggle_button_get_active(
        GTK_TOGGLE_BUTTON(ui->settings_filter_enable));
    text = gtk_entry_get_text(GTK_ENTRY(ui->settings_filter_f1));
    ui->demod_filter_f1_hz = (text != NULL && text[0] != '\0')
                                 ? g_ascii_strtod(text, NULL) : 0.0;
    text = gtk_entry_get_text(GTK_ENTRY(ui->settings_filter_f2));
    ui->demod_filter_f2_hz = (text != NULL && text[0] != '\0')
                                 ? g_ascii_strtod(text, NULL) : 0.0;
    text = gtk_entry_get_text(GTK_ENTRY(ui->settings_filter_taps));
    taps = (text != NULL && text[0] != '\0') ? atoi(text) : 0;
    ui->demod_filter_taps = (taps >= 3) ? taps : 127;
    gtk_widget_set_sensitive(ui->settings_filter_f1,
                             ui->demod_filter_enabled);
    gtk_widget_set_sensitive(ui->settings_filter_f2,
                             ui->demod_filter_enabled && kind >= 2);
    gtk_widget_set_sensitive(ui->settings_filter_taps,
                             ui->demod_filter_enabled);
    set_info(ui, ui->demod_filter_enabled
                     ? "Demodulation filter enabled (%s) for playback."
                     : "Demodulation filter disabled.",
             filter_kind_labels[kind]);
}

/*
 * Apply the widget state to the view model and to main.c. Returns
 * without doing anything while the widgets are being initialised.
 */
static void apply_settings_from_widgets(Ui *ui)
{
    double rate;
    IqFormat format;
    double center_hz;
    int fft_size;
    DspWindow window_type;
    int idx;
    const char *text;
    char *end = NULL;

    if (settings_initialising) {
        return;
    }
    /* sample rate: preset or parsed custom entry (Hz) */
    idx = gtk_combo_box_get_active(GTK_COMBO_BOX(ui->settings_rate_combo));
    if (idx >= 0 && idx < settings_cur_rate_count) {
        rate = settings_cur_rate_hz[idx];
    } else {
        text = gtk_entry_get_text(GTK_ENTRY(ui->settings_custom_entry));
        rate = g_ascii_strtod(text, &end);
        if (end == text || rate <= 0.0) {
            set_info(ui, "Invalid custom sample rate.");
            return;
        }
    }
    /* data format */
    idx = gtk_combo_box_get_active(GTK_COMBO_BOX(ui->settings_format_combo));
    if (settings_cur_format_audio) {
        format = IQ_FORMAT_FLOAT32; /* audio is decoded to float */
        if (idx >= 0 && idx < settings_cur_format_count &&
            ui->src != NULL && ui->src->is_audio) {
            int bits = settings_format_bits_audio[idx];

            if (bits != ui->src->audio_bits ||
                (idx >= 4) != ui->src->audio_float) {
                (void)iq_source_reopen_audio(ui->src, bits);
            }
        }
    } else {
        if (idx < 0 || idx > (int)IQ_FORMAT_UINT8) {
            idx = (int)IQ_FORMAT_FLOAT32;
        }
        format = (IqFormat)idx;
    }
    /* centre frequency (MHz; empty or 0 keeps the filename centre) */
    text = gtk_entry_get_text(GTK_ENTRY(ui->settings_center_entry));
    center_hz = 0.0;
    if (text != NULL && text[0] != '\0') {
        center_hz = g_ascii_strtod(text, NULL) * HZ_PER_MHZ;
    }
    /* waterfall FFT size (clamped to >= 64) */
    idx = gtk_combo_box_get_active(GTK_COMBO_BOX(ui->settings_fft_combo));
    if (idx >= 0 && idx < SETTINGS_FFT_PRESETS) {
        fft_size = settings_fft_presets[idx];
    } else {
        fft_size = FFT_SIZE; /* default */
    }
    if (fft_size < SETTINGS_MIN_FFT) {
        fft_size = SETTINGS_MIN_FFT;
    }
    /* waterfall window type */
    idx = gtk_combo_box_get_active(GTK_COMBO_BOX(ui->settings_window_combo));
    if (idx < 0 || idx > (int)DSP_WINDOW_RECTANGULAR) {
        idx = (int)DSP_WINDOW_HANN;
    }
    window_type = (DspWindow)idx;

    ui->settings_rate = rate;
    ui->settings_format = format;
    ui->settings_center = center_hz;
    ui->settings_fft_size = fft_size;
    ui->settings_window = window_type;
    if (ui->apply_settings_cb != NULL &&
        ui->apply_settings_cb(ui, rate, format, center_hz, fft_size,
                              window_type) != 0) {
        set_info(ui, "Failed to apply settings.");
        return;
    }
    /* the rate change may have changed the duration: re-clamp the
     * waterfall length and refresh its widgets */
    if (ui->src->duration > 0.0 &&
        ui->visible_seconds > ui->src->duration) {
        ui->visible_seconds = ui->src->duration;
    }
    if (ui->visible_seconds < WF_LEN_MIN) {
        ui->visible_seconds = WF_LEN_MIN;
    }
    if (ui->src->duration > 0.0 &&
        ui->visible_seconds > ui->src->duration) {
        ui->visible_seconds = ui->src->duration; /* duration wins */
    }
    wf_len_init_widgets(ui);
    wrap_scroll(ui);
}

static void on_rate_combo_changed(GtkComboBox *combo, gpointer user_data)
{
    Ui *ui = (Ui *)user_data;
    int idx = gtk_combo_box_get_active(combo);
    char text[SETTINGS_TEXT_SIZE];

    if (settings_initialising) {
        return;
    }
    if (idx >= 0 && idx < settings_cur_rate_count) {
        gtk_widget_set_sensitive(ui->settings_custom_entry, FALSE);
        snprintf(text, sizeof(text), "%.0f", settings_cur_rate_hz[idx]);
        gtk_entry_set_text(GTK_ENTRY(ui->settings_custom_entry), text);
    } else {
        /* Custom..: the entry becomes editable, showing the old rate */
        gtk_widget_set_sensitive(ui->settings_custom_entry, TRUE);
    }
    apply_settings_from_widgets(ui);
}

static void on_format_combo_changed(GtkComboBox *combo, gpointer user_data)
{
    Ui *ui = (Ui *)user_data;

    (void)combo;
    if (settings_initialising) {
        return;
    }
    apply_settings_from_widgets(ui);
}

static void on_fft_combo_changed(GtkComboBox *combo, gpointer user_data)
{
    Ui *ui = (Ui *)user_data;

    (void)combo;
    if (settings_initialising) {
        return;
    }
    apply_settings_from_widgets(ui);
}

static void on_window_combo_changed(GtkComboBox *combo, gpointer user_data)
{
    Ui *ui = (Ui *)user_data;

    (void)combo;
    if (settings_initialising) {
        return;
    }
    apply_settings_from_widgets(ui);
}

static void on_center_entry_activate(GtkEntry *entry, gpointer user_data)
{
    Ui *ui = (Ui *)user_data;

    (void)entry;
    if (settings_initialising) {
        return;
    }
    apply_settings_from_widgets(ui);
}

static gboolean on_center_entry_focus_out(GtkWidget *widget, GdkEventFocus *event, gpointer user_data)
{
    Ui *ui = (Ui *)user_data;

    (void)widget;
    (void)event;
    if (settings_initialising) {
        return FALSE;
    }
    apply_settings_from_widgets(ui);
    return FALSE;
}

/* Apply the waterfall-length widgets to the view model (no-op while
 * the Settings widgets are being initialised). */
static void apply_waterfall_length(Ui *ui)
{
    double len;
    double max_len;
    int idx;
    const char *text;
    char *end = NULL;
    double fps;

    if (settings_initialising) {
        return;
    }
    idx = gtk_combo_box_get_active(GTK_COMBO_BOX(wf_len_combo));
    if (idx >= 0 && idx < WF_LEN_PRESETS) {
        len = wf_len_presets[idx];
    } else {
        text = gtk_entry_get_text(GTK_ENTRY(wf_len_entry));
        len = g_ascii_strtod(text, &end);
        if (end == text || len <= 0.0) {
            set_info(ui, "Invalid waterfall length.");
            return;
        }
    }
    max_len = ui->src->duration;
    if (max_len > 0.0 && len > max_len) {
        len = max_len;
    }
    if (len < WF_LEN_MIN) {
        len = WF_LEN_MIN;
    }
    /* a file shorter than WF_LEN_MIN keeps its own duration: the window
     * must never be longer than the data or the view wraps and selections
     * become ambiguous */
    if (max_len > 0.0 && len > max_len) {
        len = max_len;
    }
    ui->visible_seconds = len;
    wrap_scroll(ui);
    fps = wc_frames_per_second(ui->cache);
    wc_request(ui->cache, (size_t)(ui->scroll_position * fps),
               2 * (size_t)(ui->visible_seconds * fps) + 1);
    update_range_label(ui);
    update_peak_spectrum(ui);
    gtk_widget_queue_draw(ui->drawing);
    gtk_widget_queue_draw(ui->peak_pane);
}

static void on_wf_len_combo_changed(GtkComboBox *combo, gpointer user_data)
{
    Ui *ui = (Ui *)user_data;
    int idx = gtk_combo_box_get_active(combo);
    char text[SETTINGS_TEXT_SIZE];

    if (settings_initialising) {
        return;
    }
    if (idx >= 0 && idx < WF_LEN_PRESETS) {
        gtk_widget_set_sensitive(wf_len_entry, FALSE);
        snprintf(text, sizeof(text), "%.1f", wf_len_presets[idx]);
        gtk_entry_set_text(GTK_ENTRY(wf_len_entry), text);
    } else {
        gtk_widget_set_sensitive(wf_len_entry, TRUE);
    }
    apply_waterfall_length(ui);
}

static void on_wf_len_entry_activate(GtkEntry *entry, gpointer user_data)
{
    (void)entry;
    apply_waterfall_length((Ui *)user_data);
}

static gboolean on_wf_len_entry_focus_out(GtkWidget *widget, GdkEventFocus *event, gpointer user_data)
{
    (void)widget;
    (void)event;
    apply_waterfall_length((Ui *)user_data);
    return FALSE;
}

/* Initialise the waterfall-length widgets from the view model (no apply). */
static void wf_len_init_widgets(Ui *ui)
{
    int preset = -1;
    char text[SETTINGS_TEXT_SIZE];

    settings_initialising = true;
    for (int i = 0; i < WF_LEN_PRESETS; i++) {
        if (fabs(ui->visible_seconds - wf_len_presets[i]) <= 0.05) {
            preset = i;
            break;
        }
    }
    if (preset >= 0) {
        gtk_combo_box_set_active(GTK_COMBO_BOX(wf_len_combo), preset);
        gtk_widget_set_sensitive(wf_len_entry, FALSE);
    } else {
        gtk_combo_box_set_active(GTK_COMBO_BOX(wf_len_combo), WF_LEN_PRESETS);
        gtk_widget_set_sensitive(wf_len_entry, TRUE);
    }
    /* the entry text always shows the effective length */
    snprintf(text, sizeof(text), "%.1f", ui->visible_seconds);
    gtk_entry_set_text(GTK_ENTRY(wf_len_entry), text);
    settings_initialising = false;
}

/*
 * Fill the Settings preset tables and combos for the kind of source that is
 * open: IQ recordings offer the SDR sample rates and I/Q formats, audio
 * recordings offer the audio rates and source encodings. The values
 * detected from the file (rate, format, centre) are preselected.
 */
static void ui_configure_source_settings(Ui *ui)
{
    bool audio = (ui->src != NULL && ui->src->is_audio);
    int rate_idx = -1, fmt_idx = -1;

    settings_cur_format_audio = audio;
    if (audio) {
        settings_cur_rate_count = SETTINGS_RATE_PRESETS_AUDIO;
        for (int i = 0; i < SETTINGS_RATE_PRESETS_AUDIO; i++) {
            settings_cur_rate_hz[i] = settings_rate_presets_audio[i];
            settings_cur_rate_labels[i] = settings_rate_labels_audio[i];
        }
        settings_cur_format_count = 6;
        for (int i = 0; i < 6; i++) {
            settings_cur_format_labels[i] = settings_format_labels_audio[i];
        }
    } else {
        settings_cur_rate_count = SETTINGS_RATE_PRESETS_IQ;
        for (int i = 0; i < SETTINGS_RATE_PRESETS_IQ; i++) {
            settings_cur_rate_hz[i] = settings_rate_presets[i];
            settings_cur_rate_labels[i] = settings_rate_labels[i];
        }
        settings_cur_format_count = 4;
        for (int i = 0; i < 4; i++) {
            settings_cur_format_labels[i] = settings_format_labels[i];
        }
    }
    if (ui->src == NULL) return;

    /* pick the entries matching the file's detected values */
    for (int i = 0; i < settings_cur_rate_count; i++) {
        if (fabs(ui->src->sample_rate - settings_cur_rate_hz[i]) <= 1.0) {
            rate_idx = i;
            break;
        }
    }
    if (audio) {
        for (int i = 0; i < settings_cur_format_count; i++) {
            bool is_float_entry = i >= 4;

            if (settings_format_bits_audio[i] == ui->src->audio_bits &&
                is_float_entry == ui->src->audio_float) {
                fmt_idx = i;
                break;
            }
        }
        ui->settings_format = IQ_FORMAT_FLOAT32;
    } else {
        fmt_idx = (int)ui->src->format;
    }
    ui->settings_rate = ui->src->sample_rate;
    ui->settings_center = ui->src->center_freq;

    /* rebuild the combos (sensitive: no apply while doing it) */
    settings_initialising = true;
    gtk_combo_box_text_remove_all(GTK_COMBO_BOX_TEXT(ui->settings_rate_combo));
    for (int i = 0; i < settings_cur_rate_count; i++) {
        gtk_combo_box_text_append_text(
            GTK_COMBO_BOX_TEXT(ui->settings_rate_combo),
            settings_cur_rate_labels[i]);
    }
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(ui->settings_rate_combo),
                                   SETTINGS_CUSTOM_LABEL);
    gtk_combo_box_text_remove_all(GTK_COMBO_BOX_TEXT(ui->settings_format_combo));
    for (int i = 0; i < settings_cur_format_count; i++) {
        gtk_combo_box_text_append_text(
            GTK_COMBO_BOX_TEXT(ui->settings_format_combo),
            settings_cur_format_labels[i]);
    }
    if (rate_idx >= 0) {
        gtk_combo_box_set_active(GTK_COMBO_BOX(ui->settings_rate_combo),
                                 rate_idx);
        gtk_widget_set_sensitive(ui->settings_custom_entry, FALSE);
        {
            char text[SETTINGS_TEXT_SIZE];

            snprintf(text, sizeof(text), "%.0f", ui->settings_rate);
            gtk_entry_set_text(GTK_ENTRY(ui->settings_custom_entry), text);
        }
    } else {
        gtk_combo_box_set_active(GTK_COMBO_BOX(ui->settings_rate_combo),
                                 settings_cur_rate_count); /* Custom.. */
        gtk_widget_set_sensitive(ui->settings_custom_entry, TRUE);
        {
            char text[SETTINGS_TEXT_SIZE];

            snprintf(text, sizeof(text), "%.0f", ui->settings_rate);
            gtk_entry_set_text(GTK_ENTRY(ui->settings_custom_entry), text);
        }
    }
    if (fmt_idx >= 0 && fmt_idx < settings_cur_format_count) {
        gtk_combo_box_set_active(GTK_COMBO_BOX(ui->settings_format_combo),
                                 fmt_idx);
    }
    {
        char text[64];

        snprintf(text, sizeof(text), "%.6f",
                 ui->settings_center / HZ_PER_MHZ);
        gtk_entry_set_text(GTK_ENTRY(ui->settings_center_entry), text);
    }
    settings_initialising = false;
}

/* Initialise the Settings widgets from the stored model (no apply). */
static void settings_init_widgets(Ui *ui)
{
    int preset = -1;
    char text[SETTINGS_TEXT_SIZE];

    settings_initialising = true;
    for (int i = 0; i < settings_cur_rate_count; i++) {
        if (fabs(ui->settings_rate - settings_cur_rate_hz[i]) <= 1.0) {
            preset = i;
            break;
        }
    }
    if (preset >= 0) {
        gtk_combo_box_set_active(GTK_COMBO_BOX(ui->settings_rate_combo), preset);
        gtk_widget_set_sensitive(ui->settings_custom_entry, FALSE);
    } else {
        gtk_combo_box_set_active(GTK_COMBO_BOX(ui->settings_rate_combo),
                                 settings_cur_rate_count);
        gtk_widget_set_sensitive(ui->settings_custom_entry, TRUE);
    }
    /* the entry text always shows the effective rate */
    snprintf(text, sizeof(text), "%.0f", ui->settings_rate);
    gtk_entry_set_text(GTK_ENTRY(ui->settings_custom_entry), text);
    gtk_combo_box_set_active(GTK_COMBO_BOX(ui->settings_format_combo),
                             (int)ui->settings_format);
    snprintf(text, sizeof(text), "%.6f", ui->settings_center / HZ_PER_MHZ);
    gtk_entry_set_text(GTK_ENTRY(ui->settings_center_entry), text);
    /* FFT size: match the stored value, fallback 8192 */
    {
        int fft_idx = -1;

        for (int i = 0; i < SETTINGS_FFT_PRESETS; i++) {
            if (ui->settings_fft_size == settings_fft_presets[i]) {
                fft_idx = i;
                break;
            }
        }
        gtk_combo_box_set_active(GTK_COMBO_BOX(ui->settings_fft_combo),
                                 fft_idx >= 0 ? fft_idx : 3 /* "8192" */);
    }
    gtk_combo_box_set_active(GTK_COMBO_BOX(ui->settings_window_combo),
                             (int)ui->settings_window);
    settings_initialising = false;
}

/*
 * Build the docked analysis notebook (right side of the HPaned) with its
 * pages: Settings, the live Selection page, and the Run-button driven
 * analysis pages.
 */

/* Wrap a dock page in a scroller so tall pages scroll instead of
 * growing the main window. */
static void dock_add_page(Ui *ui, GtkWidget *page, const char *label)
{
    GtkWidget *scroller = gtk_scrolled_window_new(NULL, NULL);

    gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(scroller),
                                   GTK_POLICY_NEVER, GTK_POLICY_AUTOMATIC);
    gtk_container_add(GTK_CONTAINER(scroller), page);
    gtk_notebook_append_page(GTK_NOTEBOOK(ui->dock_pane), scroller,
                             gtk_label_new(label));
}

/* One row of the analysis settings editor. */
typedef struct {
    Ui *ui;
    char analysis[32];
    char key[48];
    const ArSettingDesc *schema;
    GtkWidget *control;
    GtkWidget *marker;
} SettingRow;

static void analysis_settings_save(Ui *ui)
{
    if (ui->aconfig_path == NULL) return;
    {
        char *dir = g_path_get_dirname(ui->aconfig_path);

        g_mkdir_with_parents(dir, 0755);
        g_free(dir);
    }
    aconfig_save(ui->aconfig_path);
}

static void setting_row_update_marker(SettingRow *row)
{
    bool over = aconfig_overridden(row->analysis, row->key);

    if (row->marker != NULL) {
        gtk_widget_set_visible(row->marker, over);
        gtk_label_set_text(GTK_LABEL(row->marker),
                           over ? "modified" : "");
    }
}

static void on_setting_num_changed(GtkSpinButton *spin, gpointer user_data)
{
    SettingRow *row = (SettingRow *)user_data;

    aconfig_set_num(row->analysis, row->key,
                    gtk_spin_button_get_value(spin));
    analysis_settings_save(row->ui);
    setting_row_update_marker(row);
    set_info(row->ui, "Analysis setting %s.%s applies to the next analysis.",
             row->analysis, row->key);
}

static void on_setting_bool_toggled(GtkToggleButton *btn, gpointer user_data)
{
    SettingRow *row = (SettingRow *)user_data;

    aconfig_set_bool(row->analysis, row->key,
                     gtk_toggle_button_get_active(btn));
    analysis_settings_save(row->ui);
    setting_row_update_marker(row);
    set_info(row->ui, "Analysis setting %s.%s applies to the next analysis.",
             row->analysis, row->key);
}

static void on_setting_text_changed(GtkEntry *entry, gpointer user_data)
{
    SettingRow *row = (SettingRow *)user_data;

    aconfig_set_text(row->analysis, row->key,
                     gtk_entry_get_text(entry));
    analysis_settings_save(row->ui);
    setting_row_update_marker(row);
}

static void analysis_settings_add_rows(Ui *ui, GtkWidget *box);

static void on_analysis_settings_reset(GtkButton *button, gpointer user_data)
{
    Ui *ui = (Ui *)user_data;

    (void)button;
    aconfig_reset();
    analysis_settings_save(ui);
    if (ui->analysis_settings_box != NULL) {
        /* rebuild the editor from the schema defaults */
        GList *children = gtk_container_get_children(
            GTK_CONTAINER(ui->analysis_settings_box));

        for (GList *l = children; l != NULL; l = l->next) {
            gtk_widget_destroy(GTK_WIDGET(l->data));
        }
        g_list_free(children);
        analysis_settings_add_rows(ui, ui->analysis_settings_box);
    }
    set_info(ui, "Analysis settings reset to their defaults.");
}

/* Build one frame per analysis with its settings widgets. */
static void analysis_settings_add_rows(Ui *ui, GtkWidget *box)
{
    size_t n = aregistry_count();

    for (size_t i = 0; i < n; i++) {
        const ArAnalysisDesc *desc = aregistry_at(i);
        GtkWidget *frame;
        GtkWidget *grid;

        if (desc == NULL || desc->n_settings == 0) continue;
        frame = gtk_frame_new(desc->title);
        grid = gtk_grid_new();
        gtk_grid_set_row_spacing(GTK_GRID(grid), 4);
        gtk_grid_set_column_spacing(GTK_GRID(grid), 8);
        gtk_widget_set_margin_start(grid, 8);
        gtk_widget_set_margin_end(grid, 8);
        gtk_widget_set_margin_top(grid, 6);
        gtk_widget_set_margin_bottom(grid, 6);
        gtk_container_add(GTK_CONTAINER(frame), grid);
        gtk_box_pack_start(GTK_BOX(box), frame, FALSE, FALSE, 0);

        for (size_t j = 0; j < desc->n_settings; j++) {
            const ArSettingDesc *set = &desc->settings[j];
            SettingRow *row = g_new0(SettingRow, 1);
            GtkWidget *label;
            char keytext[128];

            row->ui = ui;
            row->schema = set;
            snprintf(row->analysis, sizeof(row->analysis), "%s", desc->id);
            snprintf(row->key, sizeof(row->key), "%s", set->key);
            snprintf(keytext, sizeof(keytext), "%s:", set->key);
            label = gtk_label_new(keytext);
            gtk_widget_set_halign(label, GTK_ALIGN_START);
            gtk_widget_set_tooltip_text(label, set->desc);
            gtk_grid_attach(GTK_GRID(grid), label, 0, (int)j, 1, 1);

            switch (set->kind) {
            case AR_SET_NUMBER: {
                double v = aconfig_get_num(desc->id, set->key, set->number);
                GtkAdjustment *adj;
                double mn = set->min != 0.0 ? set->min : 0.0;
                double mx = set->max != 0.0 ? set->max : 1.0e9;
                double step = set->step != 0.0 ? set->step : 1.0;

                adj = gtk_adjustment_new(v, mn, mx, step, step * 10.0, 0.0);
                row->control = gtk_spin_button_new(adj,
                                                   step < 1.0 ? 3 : 0, 0);
                gtk_spin_button_set_numeric(GTK_SPIN_BUTTON(row->control),
                                            TRUE);
                g_signal_connect(row->control, "value-changed",
                                 G_CALLBACK(on_setting_num_changed), row);
                break;
            }
            case AR_SET_INT: {
                int v = aconfig_get_int(desc->id, set->key, set->integer);
                double mn = set->min != 0.0 ? set->min : 0.0;
                double mx = set->max != 0.0 ? set->max : 1.0e9;
                double step = set->step != 0.0 ? set->step : 1.0;
                GtkAdjustment *adj = gtk_adjustment_new(v, mn, mx, step,
                                                        step * 10.0, 0.0);

                row->control = gtk_spin_button_new(adj, step, 0);
                g_signal_connect(row->control, "value-changed",
                                 G_CALLBACK(on_setting_num_changed), row);
                break;
            }
            case AR_SET_BOOL: {
                bool v = aconfig_get_bool(desc->id, set->key, set->boolean);

                row->control = gtk_check_button_new();
                gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(row->control),
                                             v);
                g_signal_connect(row->control, "toggled",
                                 G_CALLBACK(on_setting_bool_toggled), row);
                break;
            }
            case AR_SET_TEXT: {
                const char *v = aconfig_get_text(desc->id, set->key,
                                                 set->text);

                row->control = gtk_entry_new();
                gtk_entry_set_text(GTK_ENTRY(row->control), v != NULL ? v : "");
                g_signal_connect(row->control, "changed",
                                 G_CALLBACK(on_setting_text_changed), row);
                break;
            }
            }
            gtk_grid_attach(GTK_GRID(grid), row->control, 1, (int)j, 1, 1);
            row->marker = gtk_label_new("");
            gtk_widget_set_halign(row->marker, GTK_ALIGN_START);
            gtk_grid_attach(GTK_GRID(grid), row->marker, 2, (int)j, 1, 1);
            setting_row_update_marker(row);
        }
    }
}

/* The "Analysis" section of the Settings page: per-analysis settings,
 * persisted to the user's config directory and applied to the next run. */
static GtkWidget *build_analysis_settings_frame(Ui *ui)
{
    GtkWidget *frame = gtk_frame_new("Analysis");
    GtkWidget *box = gtk_box_new(GTK_ORIENTATION_VERTICAL, PAGE_SPACING);
    GtkWidget *hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
    GtkWidget *title = gtk_label_new(
        "Per-analysis settings, applied to the next analysis run. The "
        "defaults come from the analysis registry.");
    GtkWidget *reset = gtk_button_new_with_label("Reset all to defaults");

    gtk_widget_set_margin_start(box, PAGE_MARGIN);
    gtk_widget_set_margin_end(box, PAGE_MARGIN);
    gtk_widget_set_margin_top(box, PAGE_MARGIN);
    gtk_widget_set_margin_bottom(box, PAGE_MARGIN);
    gtk_label_set_xalign(GTK_LABEL(title), 0.0);
    gtk_label_set_line_wrap(GTK_LABEL(title), TRUE);
    gtk_box_pack_start(GTK_BOX(box), title, FALSE, FALSE, 0);
    g_signal_connect(reset, "clicked",
                     G_CALLBACK(on_analysis_settings_reset), ui);
    gtk_box_pack_start(GTK_BOX(hbox), reset, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(box), hbox, FALSE, FALSE, 0);

    ui->analysis_settings_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 6);
    gtk_box_pack_start(GTK_BOX(box), ui->analysis_settings_box, FALSE, FALSE, 0);
    analysis_settings_add_rows(ui, ui->analysis_settings_box);
    gtk_container_add(GTK_CONTAINER(frame), box);
    return frame;
}

/* ------------------------------------------------------------------ */
/* Demodulate tab: LTE uplink PUSCH / PUCCH                            */
/* ------------------------------------------------------------------ */

/* Parameters guessed by the Selection analysis: build the demodulation
 * hint from the ranked database matches and update the LTE tab label. */
static void update_lte_hint_label(Ui *ui)
{
    char text[640];

    if (ui->demod_hint_label == NULL) return;
    if (!ui->lte_hint.valid) {
        snprintf(text, sizeof(text),
                 "No LTE uplink hint yet: run Analyze on a selection that "
                 "matches PUSCH/PUCCH and the demodulator will use the "
                 "analysed band, bandwidth and modulation.");
    } else {
        snprintf(text, sizeof(text), "Analysis hint: %s",
                 ui->lte_hint.source);
    }
    gtk_label_set_text(GTK_LABEL(ui->demod_hint_label), text);
}

static void lte_hint_apply(Ui *ui, const RankedMatch *ids, size_t n,
                           double measured_bw_hz, const Metrics *m,
                           const ModCandidate *mods, size_t nmods)
{
    Region region;

    if (ui->has_selection) {
        region = ui->selection;
    } else {
        double lo = ui->src->center_freq - ui->src->sample_rate / 2.0 +
                    ui->freq_offset;

        region.freq_min = lo;
        region.freq_max = lo + ui->visible_bw;
        region.time_min = ui->scroll_position;
        region.time_max = ui->scroll_position + ui->visible_seconds;
    }
    lte_hint_from_analysis(&ui->lte_hint, ids, n, measured_bw_hz, m, mods,
                           nmods, &region);
    update_lte_hint_label(ui);
}

static LteDemod lte_last;        /* last demodulation (32 KB steady state) */static LteDemod lte_last;        /* last demodulation (32 KB steady state) */
static bool lte_last_valid;
static bool lte_last_ran;        /* a demodulation has been attempted */

static void update_lte_result_label(Ui *ui)
{
    char text[1400];

    if (ui->demod_result_label == NULL) return;
    if (!lte_last_valid && !lte_last_ran) {
        snprintf(text, sizeof(text),
                 "Run a demodulation on a selection containing an LTE "
                 "uplink burst (PUSCH or PUCCH).");
    } else if (!lte_last_valid) {
        snprintf(text, sizeof(text), "%s", lte_last.summary);
    } else {
        snprintf(text, sizeof(text), "%s\n\nFirst bits: ", lte_last.summary);
        {
            size_t used = strlen(text);
            size_t show = lte_last.n_bits < 48 ? lte_last.n_bits : 48;

            for (size_t i = 0; i < show && used + 2 < sizeof(text); i++) {
                text[used++] = ((lte_last.bits[i >> 3] >> (7 - (i & 7))) & 1)
                                   ? '1' : '0';
            }
            text[used] = '\0';
            if (lte_last.n_bits > show && used + 20 < sizeof(text)) {
                snprintf(text + used, sizeof(text) - used, " ... (%zu bits)",
                         lte_last.n_bits);
            }
        }
    }
    gtk_label_set_text(GTK_LABEL(ui->demod_result_label), text);
}

static gboolean draw_lte_constellation(GtkWidget *widget, cairo_t *cr,
                                       gpointer user_data)
{
    int w = gtk_widget_get_allocated_width(widget);
    int h = gtk_widget_get_allocated_height(widget);
    double cx = w / 2.0, cy = h / 2.0;
    double r = (w < h ? w : h) / 2.0 - 12.0;

    cairo_set_source_rgb(cr, 0.07, 0.07, 0.09);
    cairo_paint(cr);
    if (r < 10.0) return FALSE;
    cairo_set_line_width(cr, 1.0);
    cairo_set_source_rgba(cr, 0.5, 0.5, 0.6, 0.5);
    cairo_move_to(cr, cx - r, cy);
    cairo_line_to(cr, cx + r, cy);
    cairo_move_to(cr, cx, cy - r);
    cairo_line_to(cr, cx, cy + r);
    cairo_stroke(cr);
    if (lte_last_valid && lte_last.n_points > 0) {
        cairo_set_source_rgba(cr, 0.3, 0.9, 0.5, 0.85);
        for (int i = 0; i < lte_last.n_points; i++) {
            double x = cx + (double)lte_last.pt_re[i] * r * 0.9;
            double y = cy - (double)lte_last.pt_im[i] * r * 0.9;

            cairo_arc(cr, x, y, 1.6, 0.0, 2.0 * M_PI);
            cairo_fill(cr);
        }
    } else {
        cairo_set_source_rgb(cr, 0.6, 0.6, 0.65);
        cairo_select_font_face(cr, FONT_FAMILY, CAIRO_FONT_SLANT_NORMAL,
                               CAIRO_FONT_WEIGHT_NORMAL);
        cairo_set_font_size(cr, 12.0);
        cairo_move_to(cr, 10.0, h / 2.0);
        cairo_show_text(cr, "constellation");
    }
    (void)user_data;
    return FALSE;
}

static void on_lte_demodulate(GtkButton *button, gpointer user_data)
{
    Ui *ui = (Ui *)user_data;
    Region region;
    LteChannel want = LTE_CH_AUTO;

    (void)button;
    if (ui->has_selection) {
        region.freq_min = ui->selection.freq_min;
        region.freq_max = ui->selection.freq_max;
        region.time_min = ui->selection.time_min;
        region.time_max = ui->selection.time_max;
    } else {
        double lo = ui->src->center_freq - ui->src->sample_rate / 2.0 +
                    ui->freq_offset;

        region.freq_min = lo;
        region.freq_max = lo + ui->visible_bw;
        region.time_min = ui->scroll_position;
        region.time_max = ui->scroll_position + ui->visible_seconds;
    }
    if (ui->demod_kind_combo != NULL) {
        int idx = gtk_combo_box_get_active(GTK_COMBO_BOX(ui->demod_kind_combo));

        if (idx == 1) want = LTE_CH_PUSCH;
        else if (idx == 2) want = LTE_CH_PUCCH;
    }
    set_info(ui, "Demodulating the selection as an LTE uplink burst ...");
    {
        const LteHint *hint = ui->lte_hint.valid ? &ui->lte_hint : NULL;

        lte_last_valid = lte_demodulate(ui->src, &region, want, hint,
                                        &lte_last);
    }
    lte_last_ran = true;
    update_lte_result_label(ui);
    if (ui->demod_const_drawing != NULL) {
        gtk_widget_queue_draw(ui->demod_const_drawing);
    }
    if (lte_last_valid) {
        set_info(ui, "%s demodulated: %zu bits", lte_last.channel_name,
                 lte_last.n_bits);
    } else {
        set_info(ui, "No LTE uplink burst matched the selection.");
    }
}

static GtkWidget *build_demodulate_page(Ui *ui)
{
    GtkWidget *page = gtk_box_new(GTK_ORIENTATION_VERTICAL, PAGE_SPACING);
    GtkWidget *hbox = gtk_flow_box_new();
    GtkWidget *notice = gtk_label_new(
        "LTE uplink demodulator: PUSCH (SC-FDMA shared channel) and PUCCH "
        "(control channel, formats 1/1a/1b and 2/2a/2b). Select a burst in "
        "the waterfall (or use the visible view) and demodulate it.");
    GtkWidget *scroll = gtk_scrolled_window_new(NULL, NULL);

    gtk_widget_set_margin_start(page, PAGE_MARGIN);
    gtk_widget_set_margin_end(page, PAGE_MARGIN);
    gtk_widget_set_margin_top(page, PAGE_MARGIN);
    gtk_widget_set_margin_bottom(page, PAGE_MARGIN);
    gtk_label_set_xalign(GTK_LABEL(notice), 0.0);
    gtk_label_set_line_wrap(GTK_LABEL(notice), TRUE);
    gtk_box_pack_start(GTK_BOX(page), notice, FALSE, FALSE, 0);
    ui->demod_hint_label = gtk_label_new(NULL);
    gtk_label_set_xalign(GTK_LABEL(ui->demod_hint_label), 0.0);
    gtk_label_set_line_wrap(GTK_LABEL(ui->demod_hint_label), TRUE);
    gtk_label_set_selectable(GTK_LABEL(ui->demod_hint_label), TRUE);
    gtk_box_pack_start(GTK_BOX(page), ui->demod_hint_label, FALSE, FALSE, 0);
    gtk_flow_box_set_selection_mode(GTK_FLOW_BOX(hbox), GTK_SELECTION_NONE);
    gtk_flow_box_set_column_spacing(GTK_FLOW_BOX(hbox), PAGE_SPACING);
    gtk_flow_box_set_row_spacing(GTK_FLOW_BOX(hbox), PAGE_SPACING);

    ui->demod_kind_combo = gtk_combo_box_text_new();
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(ui->demod_kind_combo),
                                   "Auto");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(ui->demod_kind_combo),
                                   "PUSCH");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(ui->demod_kind_combo),
                                   "PUCCH");
    gtk_widget_set_tooltip_text(ui->demod_kind_combo,
                                "Auto: detect the channel; PUSCH: shared "
                                "channel; PUCCH: control channel");
    gtk_combo_box_set_active(GTK_COMBO_BOX(ui->demod_kind_combo), 0);
    gtk_flow_box_insert(GTK_FLOW_BOX(hbox), ui->demod_kind_combo, -1);
    ui->demod_run_btn = gtk_button_new_with_label("Demodulate");
    g_signal_connect(ui->demod_run_btn, "clicked",
                     G_CALLBACK(on_lte_demodulate), ui);
    gtk_flow_box_insert(GTK_FLOW_BOX(hbox), ui->demod_run_btn, -1);
    gtk_box_pack_start(GTK_BOX(page), hbox, FALSE, FALSE, 0);

    ui->demod_const_drawing = gtk_drawing_area_new();
    gtk_widget_set_size_request(ui->demod_const_drawing, -1, 220);
    g_signal_connect(ui->demod_const_drawing, "draw",
                     G_CALLBACK(draw_lte_constellation), ui);
    gtk_box_pack_start(GTK_BOX(page), ui->demod_const_drawing, FALSE, FALSE, 0);

    scroll = gtk_scrolled_window_new(NULL, NULL);
    gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(scroll),
                                   GTK_POLICY_NEVER, GTK_POLICY_AUTOMATIC);
    gtk_widget_set_vexpand(scroll, TRUE);
    ui->demod_result_label = gtk_label_new(NULL);
    gtk_label_set_xalign(GTK_LABEL(ui->demod_result_label), 0.0);
    gtk_label_set_line_wrap(GTK_LABEL(ui->demod_result_label), TRUE);
    gtk_label_set_selectable(GTK_LABEL(ui->demod_result_label), TRUE);
    gtk_container_add(GTK_CONTAINER(scroll), ui->demod_result_label);
    gtk_box_pack_start(GTK_BOX(page), scroll, TRUE, TRUE, 0);
    lte_last_valid = false;
    lte_last_ran = false;
    update_lte_result_label(ui);
    update_lte_hint_label(ui);
    return page;
}

/* --- Frequency Counter tab (schema/fc_dataset.json) -------------------- */

enum {
    FC_T_ITEM = 0,      /* tree label                                   */
    FC_T_ID,            /* schema id / name                            */
    FC_T_DETAILS,       /* short schema summary                        */
    FC_T_MEAS,          /* measurement count                           */
    FC_T_DATASET,       /* dataset index (-1: not a dataset node)      */
    FC_T_SESSION,       /* session index inside the dataset            */
    FC_T_ROUND,         /* round index inside the session              */
    FC_T_SUBJECT,       /* subject index inside the round              */
    FC_T_KIND,          /* FcNodeKind                                  */
    FC_T_COUNT
};

typedef enum {
    FC_KIND_DATASET = 0,
    FC_KIND_SESSION,
    FC_KIND_ROUND,
    FC_KIND_SUBJECT,
    FC_KIND_UNASSIGNED
} FcNodeKind;



static gboolean fc_deliver(gpointer data);
static gpointer fc_worker(gpointer data);

/* Case-insensitive substring test. */
static bool fc_icontains(const char *haystack, const char *needle)
{
    char *low_h, *low_n;
    bool found;

    if (needle == NULL || needle[0] == '\0') return true;
    if (haystack == NULL) return false;
    low_h = g_utf8_strdown(haystack, -1);
    low_n = g_utf8_strdown(needle, -1);
    found = low_h != NULL && low_n != NULL && strstr(low_h, low_n) != NULL;
    g_free(low_h);
    g_free(low_n);
    return found;
}

/* ---- schema-aware details text ---------------------------------------- */

static void details_add(GString *s, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));

static void details_add(GString *s, const char *fmt, ...)
{
    va_list ap;

    va_start(ap, fmt);
    g_string_append_vprintf(s, fmt, ap);
    va_end(ap);
}

static void details_measurements(GString *s, const char *label,
                                 const double *values, size_t n)
{
    double lo = 0.0, hi = 0.0, mean = 0.0;

    if (fc_stats(values, n, &lo, &hi, &mean)) {
        details_add(s, "%-20s %zu measurements, min %.3f Hz, max %.3f Hz, "
                       "mean %.3f Hz\n", label, n, lo, hi, mean);
        details_add(s, "%-20s first values:", "");
        for (size_t i = 0; i < n && i < 8; i++) {
            details_add(s, " %.3f", values[i]);
        }
        if (n > 8) details_add(s, " …");
        details_add(s, "\n");
    } else {
        details_add(s, "%-20s no measurements\n", label);
    }
}

static GString *fc_details_for(const FcDatasetList *list, FcNodeKind kind,
                               int di, int si, int ri, int ji)
{
    GString *s = g_string_new("");

    if (list == NULL || di < 0 || (size_t)di >= list->count) {
        g_string_append(s, "No selection.\n");
        return s;
    }
    {
        const FcDataset *ds = &list->items[di];

        switch (kind) {
        case FC_KIND_DATASET:
            details_add(s, "DATASET\n");
            details_add(s, "%-20s %s\n", "dataset_id:", ds->dataset_id);
            details_add(s, "%-20s %s\n", "source_file:", ds->source_file);
            details_add(s, "%-20s %s\n", "title:", ds->title);
            details_add(s, "%-20s %zu\n", "notes:", ds->n_notes);
            for (size_t i = 0; i < ds->n_notes; i++) {
                details_add(s, "  note %zu: %s\n", i + 1, ds->notes[i]);
            }
            details_add(s, "%-20s %zu\n", "sessions:", ds->n_sessions);
            details_add(s, "%-20s %zu\n", "rounds:", fcd_count_rounds(ds));
            details_add(s, "%-20s %zu\n", "subjects:", fcd_count_subjects(ds));
            details_add(s, "%-20s %zu\n", "measurements:",
                        fcd_count_measurements(ds));
            if (ds->n_unassigned > 0) {
                details_add(s, "\nUNASSIGNED MEASUREMENTS\n");
                details_measurements(s, "unassigned:", ds->unassigned,
                                     ds->n_unassigned);
            }
            details_add(s, "\nSESSIONS\n");
            for (size_t i = 0; i < ds->n_sessions; i++) {
                details_add(s, "  session %s — %zu rounds\n",
                            ds->sessions[i].session_id.text,
                            ds->sessions[i].n_rounds);
            }
            break;
        case FC_KIND_SESSION:
            if (si < 0 || (size_t)si >= ds->n_sessions) {
                g_string_append(s, "No session.\n");
                break;
            }
            {
                const FcSession *ss = &ds->sessions[si];

                details_add(s, "SESSION\n");
                details_add(s, "%-20s %s\n", "session_id:", ss->session_id.text);
                details_add(s, "%-20s %zu\n", "rounds:", ss->n_rounds);
                for (size_t r = 0; r < ss->n_rounds; r++) {
                    const FcRound *rr = &ss->rounds[r];

                    details_add(s, "  round %s — %zu subjects, %zu listed\n",
                                rr->round_id.text, rr->n_subjects,
                                rr->n_listed);
                }
            }
            break;
        case FC_KIND_ROUND:
            if (si < 0 || (size_t)si >= ds->n_sessions ||
                ri < 0 || (size_t)ri >= ds->sessions[si].n_rounds) {
                g_string_append(s, "No round.\n");
                break;
            }
            {
                const FcRound *rr = &ds->sessions[si].rounds[ri];

                details_add(s, "ROUND\n");
                details_add(s, "%-20s %s\n", "round_id:", rr->round_id.text);
                details_add(s, "%-20s %s\n", "location:",
                            rr->location != NULL ? rr->location : "(none)");
                details_add(s, "%-20s %s\n", "date:",
                            rr->date != NULL ? rr->date : "(none)");
                if (rr->has_coordinates) {
                    details_add(s, "%-20s %.6f, %.6f\n", "coordinates:",
                                rr->latitude, rr->longitude);
                } else {
                    details_add(s, "%-20s %s\n", "coordinates:", "(none)");
                }
                details_add(s, "%-20s ", "closest towers:");
                if (rr->towers_null) details_add(s, "(null)\n");
                else if (rr->n_towers == 0) details_add(s, "(none)\n");
                else {
                    for (size_t i = 0; i < rr->n_towers; i++) {
                        details_add(s, "%s%s", i > 0 ? ", " : "",
                                    rr->towers[i]);
                    }
                    details_add(s, "\n");
                }
                details_add(s, "%-20s %s\n", "context:",
                            rr->context != NULL ? rr->context : "(none)");
                details_add(s, "%-20s %zu", "subjects listed:", rr->n_listed);
                for (size_t i = 0; i < rr->n_listed; i++) {
                    details_add(s, "%s%s", i > 0 ? ", " : " ",
                                rr->subjects_listed[i]);
                }
                details_add(s, "\n\nSUBJECTS\n");
                for (size_t j = 0; j < rr->n_subjects; j++) {
                    const FcSubject *sj = &rr->subjects[j];
                    GString *one = g_string_new("");

                    details_measurements(one, sj->name, sj->measurements,
                                         sj->count);
                    details_add(s, "%s", one->str);
                    g_string_free(one, TRUE);
                }
            }
            break;
        case FC_KIND_SUBJECT:
            if (si < 0 || (size_t)si >= ds->n_sessions ||
                ri < 0 || (size_t)ri >= ds->sessions[si].n_rounds ||
                ji < 0 ||
                (size_t)ji >= ds->sessions[si].rounds[ri].n_subjects) {
                g_string_append(s, "No subject.\n");
                break;
            }
            {
                const FcSubject *sj =
                    &ds->sessions[si].rounds[ri].subjects[ji];

                details_add(s, "SUBJECT\n");
                details_add(s, "%-20s %s\n", "name:", sj->name);
                details_measurements(s, "measurements:", sj->measurements,
                                     sj->count);
            }
            break;
        case FC_KIND_UNASSIGNED:
            details_add(s, "UNASSIGNED MEASUREMENTS\n");
            details_measurements(s, "unassigned:", ds->unassigned,
                                 ds->n_unassigned);
            break;
        }
    }
    return s;
}

/* ---- tree building ---------------------------------------------------- */

static bool fc_dataset_matches(const FcDataset *ds, const char *filter)
{
    if (filter == NULL || filter[0] == '\0') return true;
    if (fc_icontains(ds->dataset_id, filter) ||
        fc_icontains(ds->source_file, filter) ||
        fc_icontains(ds->title, filter)) {
        return true;
    }
    for (size_t i = 0; i < ds->n_notes; i++) {
        if (fc_icontains(ds->notes[i], filter)) return true;
    }
    return false;
}

static bool fc_session_matches(const FcDataset *ds, size_t si,
                               const char *filter)
{
    if (filter == NULL || filter[0] == '\0') return true;
    if (fc_icontains(ds->sessions[si].session_id.text, filter)) return true;
    for (size_t r = 0; r < ds->sessions[si].n_rounds; r++) {
        const FcRound *rr = &ds->sessions[si].rounds[r];

        if (fc_icontains(rr->round_id.text, filter) ||
            fc_icontains(rr->location, filter) || fc_icontains(rr->date, filter) ||
            fc_icontains(rr->context, filter)) {
            return true;
        }
        for (size_t t = 0; t < rr->n_towers; t++) {
            if (fc_icontains(rr->towers[t], filter)) return true;
        }
        for (size_t l = 0; l < rr->n_listed; l++) {
            if (fc_icontains(rr->subjects_listed[l], filter)) return true;
        }
        for (size_t j = 0; j < rr->n_subjects; j++) {
            if (fc_icontains(rr->subjects[j].name, filter)) return true;
        }
    }
    return false;
}

static bool fc_round_matches(const FcRound *rr, const char *filter)
{
    if (filter == NULL || filter[0] == '\0') return true;
    if (fc_icontains(rr->round_id.text, filter) ||
        fc_icontains(rr->location, filter) || fc_icontains(rr->date, filter) ||
        fc_icontains(rr->context, filter)) {
        return true;
    }
    for (size_t t = 0; t < rr->n_towers; t++) {
        if (fc_icontains(rr->towers[t], filter)) return true;
    }
    for (size_t l = 0; l < rr->n_listed; l++) {
        if (fc_icontains(rr->subjects_listed[l], filter)) return true;
    }
    for (size_t j = 0; j < rr->n_subjects; j++) {
        if (fc_icontains(rr->subjects[j].name, filter)) return true;
    }
    return false;
}

static bool fc_subject_matches(const FcSubject *sj, const char *filter)
{
    return filter == NULL || filter[0] == '\0' ||
           fc_icontains(sj->name, filter);
}

static void fc_tree_append(Ui *ui, GtkTreeIter *it, GtkTreeIter *parent,
                           const char *item, const char *id,
                           const char *details, size_t measurements,
                           int kind, int di, int si, int ri, int ji)
{
    gtk_tree_store_append(ui->fc_model, it, parent);
    gtk_tree_store_set(ui->fc_model, it, FC_T_ITEM, item, FC_T_ID, id,
                       FC_T_DETAILS, details, FC_T_MEAS,
                       (int)(measurements > 999999 ? 999999 : measurements),
                       FC_T_KIND, kind, FC_T_DATASET, di, FC_T_SESSION, si,
                       FC_T_ROUND, ri, FC_T_SUBJECT, ji, -1);
}

static void fc_tree_build(Ui *ui, const char *filter)
{
    gtk_tree_store_clear(ui->fc_model);
    for (size_t d = 0; d < ui->fc_datasets.count; d++) {
        const FcDataset *ds = &ui->fc_datasets.items[d];
        GtkTreeIter dit;
        char buf[512];
        bool include = fc_dataset_matches(ds, filter);

        for (size_t s = 0; s < ds->n_sessions && !include; s++) {
            include = fc_session_matches(ds, s, filter);
        }
        if (!include) continue;
        snprintf(buf, sizeof(buf), "dataset: %s", ds->title);
        {
            char details[256];

            snprintf(details, sizeof(details), "%zu sessions · %zu rounds · "
                                               "%zu subjects",
                     ds->n_sessions, fcd_count_rounds(ds),
                     fcd_count_subjects(ds));
            fc_tree_append(ui, &dit, NULL, buf, ds->dataset_id, details,
                           fcd_count_measurements(ds), FC_KIND_DATASET,
                           (int)d, -1, -1, -1);
        }
        for (size_t s = 0; s < ds->n_sessions; s++) {
            const FcSession *ss = &ds->sessions[s];
            GtkTreeIter sit;
            char sid[512];
            char sdetails[256];
            bool s_include = fc_session_matches(ds, s, filter);

            if (!s_include) continue;
            snprintf(sid, sizeof(sid), "session %s", ss->session_id.text);
            snprintf(sdetails, sizeof(sdetails), "%zu rounds", ss->n_rounds);
            fc_tree_append(ui, &sit, &dit, sid, ss->session_id.text,
                           sdetails, 0, FC_KIND_SESSION, (int)d, (int)s,
                           -1, -1);
            for (size_t r = 0; r < ss->n_rounds; r++) {
                const FcRound *rr = &ss->rounds[r];
                GtkTreeIter rit;
                char rid[512];
                char rdetails[512];
                bool r_include =
                    filter == NULL || filter[0] == '\0' ||
                    fc_round_matches(rr, filter) || !s_include;

                if (!r_include) continue;
                snprintf(rid, sizeof(rid), "round %s", rr->round_id.text);
                snprintf(rdetails, sizeof(rdetails), "%s%s%s · %zu subjects",
                         rr->has_coordinates ? "coords · " : "",
                         rr->date != NULL ? rr->date : "no date",
                         rr->location != NULL ? "" : "",
                         rr->n_subjects);
                fc_tree_append(ui, &rit, &sit, rid, rr->round_id.text,
                               rdetails, 0, FC_KIND_ROUND, (int)d, (int)s,
                               (int)r, -1);
                for (size_t j = 0; j < rr->n_subjects; j++) {
                    const FcSubject *sj = &rr->subjects[j];
                    GtkTreeIter jit;
                    char jdetails[256];

                    if (!(filter == NULL || filter[0] == '\0' ||
                          fc_subject_matches(sj, filter) || !r_include)) {
                        continue;
                    }
                    snprintf(jdetails, sizeof(jdetails),
                             "subject · %zu measurements", sj->count);
                    fc_tree_append(ui, &jit, &rit, sj->name, sj->name,
                                   jdetails, sj->count, FC_KIND_SUBJECT,
                                   (int)d, (int)s, (int)r, (int)j);
                }
            }
        }
        if (ds->n_unassigned > 0) {
            GtkTreeIter uit;

            fc_tree_append(ui, &uit, &dit, "unassigned measurements", "",
                           "not tied to a session", ds->n_unassigned,
                           FC_KIND_UNASSIGNED, (int)d, -1, -1, -1);
        }
    }
}

static void fc_refresh_status(Ui *ui)
{
    char text[256];

    if (ui->fc_datasets.count == 0) {
        gtk_label_set_text(GTK_LABEL(ui->fc_status_label),
                           "No datasets loaded.");
        return;
    }
    snprintf(text, sizeof(text),
             "%zu dataset%s · %zu sessions · %zu rounds · %zu subjects · "
             "%zu measurements", ui->fc_datasets.count,
             ui->fc_datasets.count == 1 ? "" : "s",
             fcd_list_count_sessions(&ui->fc_datasets),
             fcd_list_count_rounds(&ui->fc_datasets),
             fcd_list_count_subjects(&ui->fc_datasets),
             fcd_list_count_measurements(&ui->fc_datasets));
    gtk_label_set_text(GTK_LABEL(ui->fc_status_label), text);
}

static void fc_rebuild(Ui *ui, bool expand)
{
    const char *filter = ui->fc_search_entry != NULL
                             ? gtk_entry_get_text(GTK_ENTRY(ui->fc_search_entry))
                             : "";
    int di = -1;

    /* remember the selected dataset so the tree can be re-focused */
    {
        GtkTreeSelection *sel = gtk_tree_view_get_selection(
            GTK_TREE_VIEW(ui->fc_view));
        GtkTreeModel *model;
        GtkTreeIter it;

        if (gtk_tree_selection_get_selected(sel, &model, &it)) {
            gtk_tree_model_get(model, &it, FC_T_DATASET, &di, -1);
        }
    }
    fc_tree_build(ui, filter);
    if (expand || (filter != NULL && filter[0] != '\0')) {
        gtk_tree_view_expand_all(GTK_TREE_VIEW(ui->fc_view));
    } else {
        gtk_tree_view_collapse_all(GTK_TREE_VIEW(ui->fc_view));
    }
    if (di >= 0) {
        GtkTreeIter root;

        if (gtk_tree_model_get_iter_first(GTK_TREE_MODEL(ui->fc_model),
                                          &root)) {
            int guard = 0;

            do {
                int row_di = -1;

                gtk_tree_model_get(GTK_TREE_MODEL(ui->fc_model), &root,
                                   FC_T_DATASET, &row_di, -1);
                if (row_di == di) {
                    GtkTreeSelection *sel = gtk_tree_view_get_selection(
                        GTK_TREE_VIEW(ui->fc_view));

                    gtk_tree_selection_select_iter(sel, &root);
                    break;
                }
            } while (++guard < 10000 &&
                     gtk_tree_model_iter_next(GTK_TREE_MODEL(ui->fc_model),
                                              &root));
        }
    }
    fc_refresh_status(ui);
    if (ui->fc_save_btn != NULL) {
        gtk_widget_set_sensitive(ui->fc_save_btn, ui->fc_datasets.count > 0);
    }
}

/* Selected node: kind and indices. */
static bool fc_selection(Ui *ui, int *kind, int *di, int *si, int *ri, int *ji)
{
    GtkTreeSelection *sel;
    GtkTreeModel *model;
    GtkTreeIter it;
    int k = -1;

    if (ui->fc_view == NULL) return false;
    sel = gtk_tree_view_get_selection(GTK_TREE_VIEW(ui->fc_view));
    if (!gtk_tree_selection_get_selected(sel, &model, &it)) return false;
    *di = -1;
    *si = -1;
    *ri = -1;
    *ji = -1;
    gtk_tree_model_get(model, &it, FC_T_KIND, &k, FC_T_DATASET, di,
                       FC_T_SESSION, si, FC_T_ROUND, ri, FC_T_SUBJECT, ji, -1);
    if (k < 0) return false;
    *kind = k;
    return true;
}

static void fc_show_details(Ui *ui)
{
    GString *text;
    int kind = FC_KIND_DATASET, di = -1, si = -1, ri = -1, ji = -1;

    if (ui->fc_details_view == NULL) return;
    if (!fc_selection(ui, &kind, &di, &si, &ri, &ji)) {
        di = -1;
        kind = FC_KIND_DATASET;
    }
    text = fc_details_for(&ui->fc_datasets, (FcNodeKind)kind, di, si, ri, ji);
    gtk_text_buffer_set_text(
        gtk_text_view_get_buffer(GTK_TEXT_VIEW(ui->fc_details_view)),
        text->str, -1);
    g_string_free(text, TRUE);
}

static void on_fc_tree_selection_changed(GtkTreeSelection *sel,
                                         gpointer user_data)
{
    (void)sel;
    fc_show_details((Ui *)user_data);
}

/* ---- loading / saving / removing -------------------------------------- */

static void fc_load_paths(Ui *ui, char *const *paths, size_t n_paths)
{
    size_t loaded = 0, first = ui->fc_datasets.count;
    char last_err[256] = "";

    for (size_t i = 0; i < n_paths; i++) {
        char err[512] = "";

        if (fcd_load_file(&ui->fc_datasets, paths[i], err, sizeof(err))) {
            loaded++;
        } else {
            snprintf(last_err, sizeof(last_err), "%s", err);
        }
    }
    fc_rebuild(ui, false);
    if (loaded > 0) {
        GtkTreePath *path = gtk_tree_path_new_from_indices((int)first, -1);

        gtk_tree_view_set_cursor(GTK_TREE_VIEW(ui->fc_view), path, NULL, FALSE);
        gtk_tree_path_free(path);
    }
    if (ui->fc_status_label != NULL) {
        char text[512];

        if (loaded > 0 && last_err[0] == '\0') {
            snprintf(text, sizeof(text), "Loaded %zu dataset%s (%zu "
                                         "measurements).", loaded,
                     loaded == 1 ? "" : "s",
                     fcd_list_count_measurements(&ui->fc_datasets));
        } else if (loaded > 0) {
            snprintf(text, sizeof(text), "Loaded %zu dataset%s; %s", loaded,
                     loaded == 1 ? "" : "s", last_err);
        } else {
            snprintf(text, sizeof(text), "%s",
                     last_err[0] != '\0' ? last_err : "No files loaded.");
        }
        gtk_label_set_text(GTK_LABEL(ui->fc_status_label), text);
    }
}

static void on_fc_load_clicked(GtkButton *button, gpointer user_data)
{
    Ui *ui = (Ui *)user_data;
    GtkWidget *dialog;
    GtkFileFilter *filter;
    int response;

    (void)button;
    dialog = gtk_file_chooser_dialog_new(
        "Load measurement dataset", GTK_WINDOW(ui->window),
        GTK_FILE_CHOOSER_ACTION_OPEN, "Cancel", GTK_RESPONSE_CANCEL, "Open",
        GTK_RESPONSE_ACCEPT, NULL);
    gtk_file_chooser_set_select_multiple(GTK_FILE_CHOOSER(dialog), TRUE);
    filter = gtk_file_filter_new();
    gtk_file_filter_set_name(filter, "Measurement datasets (JSON)");
    gtk_file_filter_add_pattern(filter, "*.json");
    gtk_file_chooser_add_filter(GTK_FILE_CHOOSER(dialog), filter);
    filter = gtk_file_filter_new();
    gtk_file_filter_set_name(filter, "All files");
    gtk_file_filter_add_pattern(filter, "*");
    gtk_file_chooser_add_filter(GTK_FILE_CHOOSER(dialog), filter);
    response = gtk_dialog_run(GTK_DIALOG(dialog));
    if (response == GTK_RESPONSE_ACCEPT) {
        GSList *names = gtk_file_chooser_get_filenames(
            GTK_FILE_CHOOSER(dialog));
        size_t n = g_slist_length(names);
        char **paths = n > 0 ? calloc(n, sizeof(*paths)) : NULL;

        if (paths != NULL) {
            size_t i = 0;

            for (GSList *l = names; l != NULL; l = l->next) {
                paths[i++] = (char *)l->data;
            }
            fc_load_paths(ui, paths, n);
            free(paths);
        }
        for (GSList *l = names; l != NULL; l = l->next) g_free(l->data);
        g_slist_free(names);
    }
    gtk_widget_destroy(dialog);
}

static char *fc_suggested_filename(const FcDataset *ds)
{
    const char *base = ds->dataset_id != NULL && ds->dataset_id[0] != '\0'
                           ? ds->dataset_id
                           : (ds->title != NULL ? ds->title : "dataset");
    GString *name = g_string_new("");
    char *out;

    for (const char *p = base; *p != '\0'; p++) {
        if (g_ascii_isalnum(*p) || *p == '-' || *p == '_') {
            g_string_append_c(name, *p);
        } else {
            g_string_append_c(name, '_');
        }
    }
    g_string_append(name, ".json");
    out = g_strdup(name->str);
    g_string_free(name, TRUE);
    return out;
}

static void on_fc_save_clicked(GtkButton *button, gpointer user_data)
{
    Ui *ui = (Ui *)user_data;
    int kind, di = -1, si = -1, ri = -1, ji = -1;
    GtkWidget *dialog;
    char *suggested;

    (void)button;
    if (!fc_selection(ui, &kind, &di, &si, &ri, &ji) || di < 0) {
        if (ui->fc_datasets.count == 0) return;
        di = 0;
    }
    suggested = fc_suggested_filename(&ui->fc_datasets.items[di]);
    dialog = gtk_file_chooser_dialog_new(
        "Save measurement dataset", GTK_WINDOW(ui->window),
        GTK_FILE_CHOOSER_ACTION_SAVE, "Cancel", GTK_RESPONSE_CANCEL, "Save",
        GTK_RESPONSE_ACCEPT, NULL);
    gtk_file_chooser_set_current_name(GTK_FILE_CHOOSER(dialog), suggested);
    g_free(suggested);
    if (gtk_dialog_run(GTK_DIALOG(dialog)) == GTK_RESPONSE_ACCEPT) {
        char *filename = gtk_file_chooser_get_filename(
            GTK_FILE_CHOOSER(dialog));
        char err[512] = "";

        if (filename != NULL &&
            fcd_save_file(&ui->fc_datasets.items[di], filename, err,
                          sizeof(err))) {
            char msg[1024];

            snprintf(msg, sizeof(msg), "Saved dataset: %s", filename);
            gtk_label_set_text(GTK_LABEL(ui->fc_status_label), msg);
        } else {
            char msg[1024];

            snprintf(msg, sizeof(msg), "%s",
                     err[0] != '\0' ? err : "Cannot save the dataset.");
            gtk_label_set_text(GTK_LABEL(ui->fc_status_label), msg);
        }
        g_free(filename);
    }
    gtk_widget_destroy(dialog);
}

static void on_fc_remove_clicked(GtkButton *button, gpointer user_data)
{
    Ui *ui = (Ui *)user_data;
    int kind, di = -1, si = -1, ri = -1, ji = -1;

    (void)button;
    if (!fc_selection(ui, &kind, &di, &si, &ri, &ji) || di < 0) return;
    fcd_list_remove(&ui->fc_datasets, (size_t)di);
    fc_rebuild(ui, false);
    fc_show_details(ui);
    if (ui->fc_status_label != NULL) {
        char msg[256];

        snprintf(msg, sizeof(msg), "Dataset removed (%zu left).",
                 ui->fc_datasets.count);
        gtk_label_set_text(GTK_LABEL(ui->fc_status_label), msg);
    }
}

static void on_fc_remove_all_clicked(GtkButton *button, gpointer user_data)
{
    Ui *ui = (Ui *)user_data;

    (void)button;
    fcd_list_clear(&ui->fc_datasets);
    fc_rebuild(ui, false);
    fc_show_details(ui);
    gtk_label_set_text(GTK_LABEL(ui->fc_status_label), "All datasets removed.");
}

static void on_fc_search_changed(GtkEditable *editable, gpointer user_data)
{
    (void)editable;
    fc_rebuild((Ui *)user_data, true);
}

static void on_fc_expand_clicked(GtkButton *button, gpointer user_data)
{
    (void)button;
    gtk_tree_view_expand_all(GTK_TREE_VIEW(((Ui *)user_data)->fc_view));
}

static void on_fc_collapse_clicked(GtkButton *button, gpointer user_data)
{
    (void)button;
    gtk_tree_view_collapse_all(GTK_TREE_VIEW(((Ui *)user_data)->fc_view));
}

/* ---- Analysis pane ---------------------------------------------------- */

static void fc_show_report(Ui *ui)
{
    const char *text = NULL;
    GtkTextBuffer *buf;

    if (ui->fc_report_view == NULL) return;
    switch (ui->fc_report_format) {
    case 1: text = ui->fc_report_markdown; break;
    case 2: text = ui->fc_report_html; break;
    default: text = ui->fc_report_text; break;
    }
    buf = gtk_text_view_get_buffer(GTK_TEXT_VIEW(ui->fc_report_view));
    gtk_text_buffer_set_text(buf, text != NULL ? text : "", -1);
}

static gboolean fc_deliver(gpointer data)
{
    FcJob *job = (FcJob *)data;
    Ui *ui;

    if (job == NULL) return G_SOURCE_REMOVE;
    ui = job->ui;
    if (ui != NULL && ui->fc_job == job) ui->fc_job = NULL;
    if (ui == NULL || job->cancelled) {
        free(job->text);
        free(job->markdown);
        free(job->html);
        fcd_list_clear(&job->snapshot);
        free(job);
        return G_SOURCE_REMOVE;
    }
    if (job->err[0] != '\0') {
        if (ui->fc_status_label != NULL) {
            gtk_label_set_text(GTK_LABEL(ui->fc_status_label), job->err);
        }
    } else {
        free(ui->fc_report_text);
        free(ui->fc_report_markdown);
        free(ui->fc_report_html);
        ui->fc_report_text = job->text;
        ui->fc_report_markdown = job->markdown;
        ui->fc_report_html = job->html;
        job->text = job->markdown = job->html = NULL;
        fc_show_report(ui);
        if (ui->fc_status_label != NULL) {
            char text[256];

            snprintf(text, sizeof(text),
                     "Analysis complete: %zu DAG node(s), %.3f s.",
                     job->nodes, job->elapsed_s);
            gtk_label_set_text(GTK_LABEL(ui->fc_status_label), text);
        }
        if (ui->fc_report_save_btn != NULL) {
            gtk_widget_set_sensitive(ui->fc_report_save_btn, TRUE);
        }
    }
    if (ui->fc_progress != NULL) {
        gtk_progress_bar_set_fraction(GTK_PROGRESS_BAR(ui->fc_progress), 1.0);
        gtk_progress_bar_set_text(GTK_PROGRESS_BAR(ui->fc_progress), "100 %");
        gtk_widget_set_visible(ui->fc_progress, FALSE);
    }
    fc_stage_shown = -1;
    if (ui->fc_analyze_btn != NULL) {
        gtk_widget_set_sensitive(ui->fc_analyze_btn, TRUE);
    }
    free(job->text);
    free(job->markdown);
    free(job->html);
    fcd_list_clear(&job->snapshot);
    free(job);
    return G_SOURCE_REMOVE;
}

static gpointer fc_worker(gpointer data)
{
    FcJob *job = (FcJob *)data;
    char err[256] = "";
    FcProgress progress;
    FcRun *run;

    progress.done = &job->prog_done;
    progress.total = &job->prog_total;
    progress.stage = &job->prog_stage;
    run = fc_dag_run(&job->snapshot, job->scope, job->enabled, &progress, err,
                     sizeof(err));

    if (run == NULL) {
        snprintf(job->err, sizeof(job->err), "%s",
                 err[0] != '\0' ? err : "analysis failed");
    } else {
        job->nodes = run->count;
        job->elapsed_s = run->elapsed_s;
        fc_report(&job->snapshot, run, job->scope, &job->text,
                  &job->markdown, &job->html);
        fc_dag_free(run);
    }
    g_idle_add(fc_deliver, job);
    return NULL;
}

static void on_fc_analyze_clicked(GtkButton *button, gpointer user_data)
{
    Ui *ui = (Ui *)user_data;
    FcJob *job;

    (void)button;
    if (ui->fc_job != NULL) return;
    job = calloc(1, sizeof(*job));
    if (job == NULL) {
        gtk_label_set_text(GTK_LABEL(ui->fc_status_label),
                           "Out of memory starting the analysis.");
        return;
    }
    job->ui = ui;
    job->scope = (FclScope)ui->fc_scope;
    job->n_enabled_nodes = fc_node_count;
    {
        int n = ui->fc_node_check_count;

        if (n > (int)fc_node_count) n = (int)fc_node_count;
        for (int i = 0; i < n; i++) {
            job->enabled[i] = ui->fc_node_enabled[i];
        }
        job->n_enabled_nodes = n;
    }
    if (!fcd_list_clone(&ui->fc_datasets, &job->snapshot)) {
        fcd_list_clear(&job->snapshot);
        free(job);
        gtk_label_set_text(GTK_LABEL(ui->fc_status_label),
                           "Out of memory copying the datasets.");
        return;
    }
    ui->fc_job = job;
    job->prog_done = 0;
    job->prog_total = (int)fc_node_count;
    job->prog_stage = -1;
    fc_stage_shown = -1;
    {
        char text[256];

        snprintf(text, sizeof(text),
                 "Running the analysis DAG over %zu dataset%s (%zu "
                 "measurements), scope: %s…", ui->fc_datasets.count,
                 ui->fc_datasets.count == 1 ? "" : "s",
                 fcd_list_count_measurements(&ui->fc_datasets),
                 fcl_scope_name((FclScope)ui->fc_scope));
        gtk_label_set_text(GTK_LABEL(ui->fc_status_label), text);
    }
    if (ui->fc_progress != NULL) {
        gtk_progress_bar_set_fraction(GTK_PROGRESS_BAR(ui->fc_progress), 0.0);
        gtk_progress_bar_set_text(GTK_PROGRESS_BAR(ui->fc_progress), "0 %");
        gtk_widget_set_visible(ui->fc_progress, TRUE);
    }
    gtk_widget_set_sensitive(ui->fc_analyze_btn, FALSE);
    if (g_thread_try_new("fc-analysis", fc_worker, job, NULL) == NULL) {
        ui->fc_job = NULL;
        fcd_list_clear(&job->snapshot);
        free(job);
        gtk_label_set_text(GTK_LABEL(ui->fc_status_label),
                           "Cannot start the analysis thread.");
        if (ui->fc_progress != NULL) {
            gtk_widget_set_visible(ui->fc_progress, FALSE);
        }
        gtk_widget_set_sensitive(ui->fc_analyze_btn, TRUE);
    }
}

static void on_fc_report_format_changed(GtkComboBox *combo, gpointer user_data)
{
    Ui *ui = (Ui *)user_data;
    int idx = gtk_combo_box_get_active(combo);

    ui->fc_report_format = (idx >= 0 && idx <= 2) ? idx : 0;
    fc_show_report(ui);
}

static void on_fc_save_report(GtkButton *button, gpointer user_data)
{
    Ui *ui = (Ui *)user_data;
    GtkWidget *dialog;
    const char *text;

    (void)button;
    switch (ui->fc_report_format) {
    case 1: text = ui->fc_report_markdown; break;
    case 2: text = ui->fc_report_html; break;
    default: text = ui->fc_report_text; break;
    }
    if (text == NULL) return;
    dialog = gtk_file_chooser_dialog_new(
        "Save frequency-counter report", GTK_WINDOW(ui->window),
        GTK_FILE_CHOOSER_ACTION_SAVE, "Cancel", GTK_RESPONSE_CANCEL, "Save",
        GTK_RESPONSE_ACCEPT, NULL);
    gtk_file_chooser_set_current_name(GTK_FILE_CHOOSER(dialog),
                                      "fc_report.txt");
    if (gtk_dialog_run(GTK_DIALOG(dialog)) == GTK_RESPONSE_ACCEPT) {
        char *filename = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(
            dialog));
        FILE *f = filename != NULL ? fopen(filename, "wb") : NULL;

        if (f != NULL) {
            fwrite(text, 1, strlen(text), f);
            fclose(f);
        }
        g_free(filename);
    }
    gtk_widget_destroy(dialog);
}

/* ---- the page --------------------------------------------------------- */

static GtkWidget *build_fc_left_panel(Ui *ui)
{
    GtkWidget *left = gtk_box_new(GTK_ORIENTATION_VERTICAL, PAGE_SPACING);
    GtkWidget *toolbar = gtk_flow_box_new();
    GtkWidget *explore = gtk_flow_box_new();
    GtkWidget *scroll = gtk_scrolled_window_new(NULL, NULL);
    GtkWidget *details_scroll = gtk_scrolled_window_new(NULL, NULL);
    GtkWidget *details_frame = gtk_frame_new("Dataset details");

    gtk_widget_set_margin_start(left, PAGE_MARGIN);
    gtk_widget_set_margin_end(left, PAGE_MARGIN);
    gtk_widget_set_margin_top(left, PAGE_MARGIN);
    gtk_widget_set_margin_bottom(left, PAGE_MARGIN);

    /* load / save / remove */
    gtk_flow_box_set_selection_mode(GTK_FLOW_BOX(toolbar), GTK_SELECTION_NONE);
    gtk_flow_box_set_column_spacing(GTK_FLOW_BOX(toolbar), PAGE_SPACING);
    {
        GtkWidget *load_btn = gtk_button_new_with_label("Load dataset…");
        GtkWidget *save_btn = gtk_button_new_with_label("Save dataset…");
        GtkWidget *rm = gtk_button_new_with_label("Remove dataset");
        GtkWidget *rm_all = gtk_button_new_with_label("Remove all");

        gtk_widget_set_tooltip_text(load_btn,
                                    "Load one or more measurement datasets "
                                    "(schema/fc_dataset.json)");
        gtk_widget_set_tooltip_text(save_btn,
                                    "Write the selected dataset back to JSON");
        gtk_widget_set_tooltip_text(rm,
                                    "Remove the dataset containing the "
                                    "selection");
        g_signal_connect(load_btn, "clicked",
                         G_CALLBACK(on_fc_load_clicked), ui);
        g_signal_connect(save_btn, "clicked",
                         G_CALLBACK(on_fc_save_clicked), ui);
        g_signal_connect(rm, "clicked", G_CALLBACK(on_fc_remove_clicked), ui);
        g_signal_connect(rm_all, "clicked",
                         G_CALLBACK(on_fc_remove_all_clicked), ui);
        ui->fc_save_btn = save_btn;
        ui->fc_remove_btn = rm;
        ui->fc_remove_all_btn = rm_all;
        gtk_widget_set_sensitive(save_btn, FALSE);
        gtk_widget_set_sensitive(rm, FALSE);
        gtk_flow_box_insert(GTK_FLOW_BOX(toolbar), load_btn, -1);
        gtk_flow_box_insert(GTK_FLOW_BOX(toolbar), save_btn, -1);
        gtk_flow_box_insert(GTK_FLOW_BOX(toolbar), rm, -1);
        gtk_flow_box_insert(GTK_FLOW_BOX(toolbar), rm_all, -1);
    }
    gtk_box_pack_start(GTK_BOX(left), toolbar, FALSE, FALSE, 0);

    /* explore: filter + expand/collapse */
    gtk_flow_box_set_selection_mode(GTK_FLOW_BOX(explore),
                                    GTK_SELECTION_NONE);
    gtk_flow_box_set_column_spacing(GTK_FLOW_BOX(explore), PAGE_SPACING);
    {
        GtkWidget *expand = gtk_button_new_with_label("Expand all");
        GtkWidget *collapse = gtk_button_new_with_label("Collapse all");

        ui->fc_search_entry = gtk_entry_new();
        gtk_entry_set_placeholder_text(GTK_ENTRY(ui->fc_search_entry),
                                       "Filter datasets, rounds, subjects…");
        gtk_widget_set_hexpand(ui->fc_search_entry, TRUE);
        g_signal_connect(ui->fc_search_entry, "changed",
                         G_CALLBACK(on_fc_search_changed), ui);
        g_signal_connect(expand, "clicked",
                         G_CALLBACK(on_fc_expand_clicked), ui);
        g_signal_connect(collapse, "clicked",
                         G_CALLBACK(on_fc_collapse_clicked), ui);
        ui->fc_expand_btn = expand;
        ui->fc_collapse_btn = collapse;
        gtk_flow_box_insert(GTK_FLOW_BOX(explore), ui->fc_search_entry, -1);
        gtk_flow_box_insert(GTK_FLOW_BOX(explore), expand, -1);
        gtk_flow_box_insert(GTK_FLOW_BOX(explore), collapse, -1);
    }
    gtk_box_pack_start(GTK_BOX(left), explore, FALSE, FALSE, 0);

    /* the dataset / session / round / subject tree */
    ui->fc_model = gtk_tree_store_new(FC_T_COUNT, G_TYPE_STRING,
                                      G_TYPE_STRING, G_TYPE_STRING,
                                      G_TYPE_INT, G_TYPE_INT, G_TYPE_INT,
                                      G_TYPE_INT, G_TYPE_INT, G_TYPE_INT);
    ui->fc_view = gtk_tree_view_new_with_model(GTK_TREE_MODEL(ui->fc_model));
    gtk_tree_view_set_headers_visible(GTK_TREE_VIEW(ui->fc_view), TRUE);
    {
        static const char *titles[4] = { "Item", "ID / name", "Details",
                                         "Meas." };

        for (int i = 0; i < 4; i++) {
            GtkCellRenderer *r = gtk_cell_renderer_text_new();
            GtkTreeViewColumn *c = gtk_tree_view_column_new_with_attributes(
                titles[i], r, "text", i, NULL);

            if (i == 2) {
                g_object_set(r, "ellipsize", PANGO_ELLIPSIZE_END, NULL);
                gtk_tree_view_column_set_expand(c, TRUE);
            }
            gtk_tree_view_column_set_resizable(c, TRUE);
            gtk_tree_view_append_column(GTK_TREE_VIEW(ui->fc_view), c);
        }
    }
    gtk_widget_set_size_request(ui->fc_view, -1, 220);
    gtk_container_add(GTK_CONTAINER(scroll), ui->fc_view);
    gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(scroll),
                                   GTK_POLICY_AUTOMATIC, GTK_POLICY_AUTOMATIC);
    gtk_box_pack_start(GTK_BOX(left), scroll, TRUE, TRUE, 0);
    g_signal_connect(gtk_tree_view_get_selection(GTK_TREE_VIEW(ui->fc_view)),
                     "changed", G_CALLBACK(on_fc_tree_selection_changed), ui);

    /* schema details of the selected node */
    ui->fc_details_view = gtk_text_view_new();
    gtk_text_view_set_editable(GTK_TEXT_VIEW(ui->fc_details_view), FALSE);
    gtk_text_view_set_cursor_visible(GTK_TEXT_VIEW(ui->fc_details_view),
                                     FALSE);
    gtk_text_view_set_wrap_mode(GTK_TEXT_VIEW(ui->fc_details_view),
                                GTK_WRAP_WORD_CHAR);
    gtk_text_view_set_monospace(GTK_TEXT_VIEW(ui->fc_details_view), TRUE);
    gtk_container_add(GTK_CONTAINER(details_scroll), ui->fc_details_view);
    gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(details_scroll),
                                   GTK_POLICY_AUTOMATIC, GTK_POLICY_AUTOMATIC);
    gtk_widget_set_size_request(details_scroll, -1, 180);
    gtk_container_add(GTK_CONTAINER(details_frame), details_scroll);
    gtk_box_pack_start(GTK_BOX(left), details_frame, TRUE, TRUE, 0);

    /* totals */
    ui->fc_status_label = gtk_label_new("No datasets loaded.");
    gtk_label_set_xalign(GTK_LABEL(ui->fc_status_label), 0.0);
    gtk_label_set_line_wrap(GTK_LABEL(ui->fc_status_label), TRUE);
    gtk_label_set_ellipsize(GTK_LABEL(ui->fc_status_label),
                            PANGO_ELLIPSIZE_END);
    gtk_box_pack_start(GTK_BOX(left), ui->fc_status_label, FALSE, FALSE, 0);
    return left;
}

/* ---- Settings: analysis scope ---------------------------------------- */

static void on_fc_scope_toggled(GtkToggleButton *button, gpointer user_data)
{
    Ui *ui = (Ui *)user_data;

    if (!gtk_toggle_button_get_active(button)) return;
    ui->fc_scope = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(button),
                                                     "fc-scope"));
    if (ui->fc_status_label != NULL) {
        char text[256];

        snprintf(text, sizeof(text), "Analysis scope: %s.",
                 fcl_scope_name((FclScope)ui->fc_scope));
        gtk_label_set_text(GTK_LABEL(ui->fc_status_label), text);
    }
}

static void on_fc_node_check_toggled(GtkToggleButton *button,
                                     gpointer user_data)
{
    Ui *ui = (Ui *)user_data;
    int idx = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(button),
                                                "fc-node-index"));

    if (idx < 0 || idx >= (int)FC_MAX_NODES) return;
    ui->fc_node_enabled[idx] = gtk_toggle_button_get_active(button);
}

static void fc_set_all_node_checks(Ui *ui, bool active)
{
    for (int i = 0; i < ui->fc_node_check_count && i < (int)FC_MAX_NODES;
         i++) {
        ui->fc_node_enabled[i] = active;
        if (ui->fc_node_checks[i] != NULL) {
            gtk_toggle_button_set_active(
                GTK_TOGGLE_BUTTON(ui->fc_node_checks[i]), active);
        }
    }
}

static void on_fc_check_all_clicked(GtkButton *button, gpointer user_data)
{
    (void)button;
    fc_set_all_node_checks((Ui *)user_data, true);
}

static void on_fc_uncheck_all_clicked(GtkButton *button, gpointer user_data)
{
    (void)button;
    fc_set_all_node_checks((Ui *)user_data, false);
}

static GtkWidget *build_fc_settings_page(Ui *ui)
{
    GtkWidget *page = gtk_box_new(GTK_ORIENTATION_VERTICAL, PAGE_SPACING);
    GtkWidget *frame = gtk_frame_new("Analysis scope");
    GtkWidget *radio_per;
    GtkWidget *radio_merged;
    GtkWidget *grid;

    gtk_widget_set_margin_start(page, PAGE_MARGIN * 2);
    gtk_widget_set_margin_end(page, PAGE_MARGIN * 2);
    gtk_widget_set_margin_top(page, PAGE_MARGIN * 2);
    gtk_widget_set_margin_bottom(page, PAGE_MARGIN * 2);
    grid = gtk_grid_new();
    gtk_grid_set_row_spacing(GTK_GRID(grid), 6);
    gtk_widget_set_margin_start(grid, PAGE_MARGIN);
    gtk_widget_set_margin_end(grid, PAGE_MARGIN);
    gtk_widget_set_margin_top(grid, PAGE_MARGIN);
    gtk_widget_set_margin_bottom(grid, PAGE_MARGIN);
    radio_per = gtk_radio_button_new_with_label(
        NULL, "Analyse every dataset separately");
    radio_merged = gtk_radio_button_new_with_label_from_widget(
        GTK_RADIO_BUTTON(radio_per),
        "Merge all datasets into one measurement list");
    g_object_set_data(G_OBJECT(radio_per), "fc-scope",
                      GINT_TO_POINTER(FCL_SCOPE_PER_DATASET));
    g_object_set_data(G_OBJECT(radio_merged), "fc-scope",
                      GINT_TO_POINTER(FCL_SCOPE_MERGED));
    g_signal_connect(radio_per, "toggled",
                     G_CALLBACK(on_fc_scope_toggled), ui);
    g_signal_connect(radio_merged, "toggled",
                     G_CALLBACK(on_fc_scope_toggled), ui);
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(radio_per),
                                 ui->fc_scope == FCL_SCOPE_PER_DATASET);
    gtk_grid_attach(GTK_GRID(grid), radio_per, 0, 0, 1, 1);
    gtk_grid_attach(GTK_GRID(grid), radio_merged, 0, 1, 1, 1);
    {
        GtkWidget *hint = gtk_label_new(
            "Per dataset: each session round (all subjects combined) and each "
            "subject is analysed separately. Merged: all measurements of all "
            "loaded datasets form one list and all analyses run once on that "
            "list.");
        GtkWidget *hint2 = gtk_label_new(
            "The choice applies to the next Analyze run; the report states "
            "which scope was used.");

        gtk_label_set_xalign(GTK_LABEL(hint), 0.0);
        gtk_label_set_line_wrap(GTK_LABEL(hint), TRUE);
        gtk_label_set_xalign(GTK_LABEL(hint2), 0.0);
        gtk_label_set_line_wrap(GTK_LABEL(hint2), TRUE);
        gtk_grid_attach(GTK_GRID(grid), hint, 0, 2, 1, 1);
        gtk_grid_attach(GTK_GRID(grid), hint2, 0, 3, 1, 1);
    }
    gtk_container_add(GTK_CONTAINER(frame), grid);
    gtk_box_pack_start(GTK_BOX(page), frame, FALSE, FALSE, 0);

    /* which analyses to run (all checked by default) */
    {
        GtkWidget *aframe = gtk_frame_new("Analyses to run");
        GtkWidget *scroll = gtk_scrolled_window_new(NULL, NULL);
        GtkWidget *abox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 2);
        GtkWidget *all_btn = gtk_button_new_with_label("Check all");
        GtkWidget *none_btn = gtk_button_new_with_label("Uncheck all");
        GtkWidget *buttons = gtk_flow_box_new();

        gtk_flow_box_set_selection_mode(GTK_FLOW_BOX(buttons),
                                        GTK_SELECTION_NONE);
        gtk_flow_box_set_column_spacing(GTK_FLOW_BOX(buttons), PAGE_SPACING);
        gtk_flow_box_insert(GTK_FLOW_BOX(buttons), all_btn, -1);
        gtk_flow_box_insert(GTK_FLOW_BOX(buttons), none_btn, -1);
        gtk_box_pack_start(GTK_BOX(abox), buttons, FALSE, FALSE, 2);
        for (size_t i = 0; i < fc_node_count && i < FC_MAX_NODES; i++) {
            char label[160];
            GtkWidget *cb;

            snprintf(label, sizeof(label), "%s (%s)", fc_node_specs[i].title,
                     fc_node_specs[i].id);
            cb = gtk_check_button_new_with_label(label);
            g_object_set_data(G_OBJECT(cb), "fc-node-index",
                              GINT_TO_POINTER((int)i));
            gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(cb),
                                         ui->fc_node_enabled[i]);
            g_signal_connect(cb, "toggled",
                             G_CALLBACK(on_fc_node_check_toggled), ui);
            ui->fc_node_checks[i] = cb;
            gtk_box_pack_start(GTK_BOX(abox), cb, FALSE, FALSE, 0);
            ui->fc_node_check_count = (int)i + 1;
        }
        g_signal_connect(all_btn, "clicked",
                         G_CALLBACK(on_fc_check_all_clicked), ui);
        g_signal_connect(none_btn, "clicked",
                         G_CALLBACK(on_fc_uncheck_all_clicked), ui);
        gtk_container_add(GTK_CONTAINER(scroll), abox);
        gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(scroll),
                                       GTK_POLICY_NEVER, GTK_POLICY_AUTOMATIC);
        gtk_widget_set_size_request(scroll, -1, 220);
        gtk_container_add(GTK_CONTAINER(aframe), scroll);
        gtk_box_pack_start(GTK_BOX(page), aframe, TRUE, TRUE, 0);
    }
    ui->fc_scope_radio_per = radio_per;
    ui->fc_scope_radio_merged = radio_merged;
    return page;
}

static GtkWidget *build_fc_analysis_page(Ui *ui)
{
    GtkWidget *page = gtk_box_new(GTK_ORIENTATION_VERTICAL, PAGE_SPACING);
    GtkWidget *toolbar = gtk_flow_box_new();
    GtkWidget *scroll = gtk_scrolled_window_new(NULL, NULL);

    gtk_widget_set_margin_start(page, PAGE_MARGIN);
    gtk_widget_set_margin_end(page, PAGE_MARGIN);
    gtk_widget_set_margin_top(page, PAGE_MARGIN);
    gtk_widget_set_margin_bottom(page, PAGE_MARGIN);
    gtk_flow_box_set_selection_mode(GTK_FLOW_BOX(toolbar), GTK_SELECTION_NONE);
    gtk_flow_box_set_column_spacing(GTK_FLOW_BOX(toolbar), PAGE_SPACING);
    ui->fc_analyze_btn = gtk_button_new_with_label("Analyze");
    gtk_widget_set_tooltip_text(ui->fc_analyze_btn,
                                "Run the frequency-counter analysis DAG "
                                "over the loaded datasets");
    g_signal_connect(ui->fc_analyze_btn, "clicked",
                     G_CALLBACK(on_fc_analyze_clicked), ui);
    gtk_flow_box_insert(GTK_FLOW_BOX(toolbar), ui->fc_analyze_btn, -1);
    ui->fc_report_format_combo = gtk_combo_box_text_new();
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(
                                       ui->fc_report_format_combo), "Text");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(
                                       ui->fc_report_format_combo), "Markdown");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(
                                       ui->fc_report_format_combo), "HTML");
    gtk_combo_box_set_active(GTK_COMBO_BOX(ui->fc_report_format_combo), 0);
    g_signal_connect(ui->fc_report_format_combo, "changed",
                     G_CALLBACK(on_fc_report_format_changed), ui);
    gtk_flow_box_insert(GTK_FLOW_BOX(toolbar), ui->fc_report_format_combo, -1);
    ui->fc_report_save_btn = gtk_button_new_with_label("Save report");
    gtk_widget_set_sensitive(ui->fc_report_save_btn, FALSE);
    g_signal_connect(ui->fc_report_save_btn, "clicked",
                     G_CALLBACK(on_fc_save_report), ui);
    gtk_flow_box_insert(GTK_FLOW_BOX(toolbar), ui->fc_report_save_btn, -1);
    gtk_box_pack_start(GTK_BOX(page), toolbar, FALSE, FALSE, 0);

    ui->fc_progress = gtk_progress_bar_new();
    gtk_progress_bar_set_pulse_step(GTK_PROGRESS_BAR(ui->fc_progress), 0.1);
    gtk_progress_bar_set_show_text(GTK_PROGRESS_BAR(ui->fc_progress), TRUE);
    gtk_widget_set_visible(ui->fc_progress, FALSE);
    gtk_box_pack_start(GTK_BOX(page), ui->fc_progress, FALSE, FALSE, 0);

    ui->fc_report_view = gtk_text_view_new();
    gtk_text_view_set_editable(GTK_TEXT_VIEW(ui->fc_report_view), FALSE);
    gtk_text_view_set_cursor_visible(GTK_TEXT_VIEW(ui->fc_report_view), FALSE);
    gtk_text_view_set_wrap_mode(GTK_TEXT_VIEW(ui->fc_report_view),
                                GTK_WRAP_WORD_CHAR);
    gtk_text_view_set_monospace(GTK_TEXT_VIEW(ui->fc_report_view), TRUE);
    gtk_container_add(GTK_CONTAINER(scroll), ui->fc_report_view);
    gtk_box_pack_start(GTK_BOX(page), scroll, TRUE, TRUE, 0);
    return page;
}

static GtkWidget *build_freq_counter_page(Ui *ui)
{
    GtkWidget *paned = gtk_paned_new(GTK_ORIENTATION_HORIZONTAL);
    GtkWidget *left = build_fc_left_panel(ui);
    GtkWidget *right = gtk_notebook_new();
    GtkWidget *analysis_page;
    GtkWidget *settings_page;

    fcd_list_init(&ui->fc_datasets);
    for (size_t i = 0; i < FC_MAX_NODES; i++) ui->fc_node_enabled[i] = true;
    ui->fc_node_check_count = 0;
    analysis_page = build_fc_analysis_page(ui);
    gtk_notebook_append_page(GTK_NOTEBOOK(right), analysis_page,
                             gtk_label_new("Analysis"));
    settings_page = build_fc_settings_page(ui);
    gtk_notebook_append_page(GTK_NOTEBOOK(right), settings_page,
                             gtk_label_new("Settings"));
    gtk_paned_pack1(GTK_PANED(paned), left, TRUE, FALSE);
    gtk_paned_pack2(GTK_PANED(paned), right, FALSE, FALSE);
    gtk_paned_set_position(GTK_PANED(paned), 620);
    fc_rebuild(ui, false);
    fc_show_details(ui);
    return paned;
}

/* --- dock pane: labels, floating window, initial width ------------------ */

/* Dock-page labels wrap so the pane can be narrowed to whatever width the
 * user drags it to while every word stays visible. */
static void dock_wrap_labels(GtkWidget *w)
{
    if (w == NULL) return;
    if (GTK_IS_LABEL(w)) {
        gtk_label_set_line_wrap(GTK_LABEL(w), TRUE);
        gtk_label_set_line_wrap_mode(GTK_LABEL(w), PANGO_WRAP_WORD_CHAR);
        return;
    }
    if (GTK_IS_CONTAINER(w)) {
        GList *kids = gtk_container_get_children(GTK_CONTAINER(w));

        for (GList *l = kids; l != NULL; l = l->next) {
            dock_wrap_labels(GTK_WIDGET(l->data));
        }
        g_list_free(kids);
    }
}

/* Default width: a third of the main window (set once, when the paned is
 * first allocated); afterwards the divider stays where the user put it. */
static void dock_initial_position(Ui *ui)
{
    int total;

    if (ui->dock_paned == NULL) return;
    total = gtk_widget_get_allocated_width(ui->dock_paned);
    if (total <= 2) return;
    if (ui->dock_pos_initialised) return;
    ui->dock_pos_initialised = true;
    gtk_paned_set_position(GTK_PANED(ui->dock_paned),
                           (int)((double)total * (1.0 - DOCK_DEFAULT_FRACTION)));
}

static void on_dock_paned_allocate(GtkWidget *widget, GtkAllocation *alloc,
                                   gpointer user_data)
{
    Ui *ui = (Ui *)user_data;

    (void)widget;
    (void)alloc;
    dock_initial_position(ui);
}

static void build_dock_pane(Ui *ui)
{
    GtkWidget *frame;
    GtkWidget *hbox;

    ui->dock_pane = gtk_notebook_new();
    gtk_notebook_set_tab_pos(GTK_NOTEBOOK(ui->dock_pane), GTK_POS_TOP);
    /* no width request: the dock starts at a third of the window and the
     * divider (or the detached window) sets its width from then on */
    gtk_widget_set_size_request(ui->dock_pane, -1, -1);

    /* --- 1. Settings page (inserted before Demod) --- */
    ui->settings_page = gtk_box_new(GTK_ORIENTATION_VERTICAL, PAGE_MARGIN);
    gtk_widget_set_margin_start(ui->settings_page, PAGE_MARGIN);
    gtk_widget_set_margin_end(ui->settings_page, PAGE_MARGIN);
    gtk_widget_set_margin_top(ui->settings_page, PAGE_MARGIN);
    gtk_widget_set_margin_bottom(ui->settings_page, PAGE_MARGIN);

    frame = gtk_frame_new(SETTINGS_FRAME_LABEL);
    GtkWidget *grid = gtk_grid_new();
    gtk_grid_set_row_spacing(GTK_GRID(grid), SETTINGS_GRID_ROW_SPACE);
    gtk_grid_set_column_spacing(GTK_GRID(grid), SETTINGS_GRID_COL_SPACE);
    gtk_widget_set_margin_start(grid, PAGE_MARGIN);
    gtk_widget_set_margin_end(grid, PAGE_MARGIN);
    gtk_widget_set_margin_top(grid, PAGE_MARGIN);
    gtk_widget_set_margin_bottom(grid, PAGE_MARGIN);
    gtk_container_add(GTK_CONTAINER(frame), grid);
    gtk_box_pack_start(GTK_BOX(ui->settings_page), frame, FALSE, FALSE, 0);

    /* row 1: sample rate */
    GtkWidget *rate_label = gtk_label_new("Sample rate:");
    gtk_widget_set_halign(rate_label, GTK_ALIGN_START);
    gtk_grid_attach(GTK_GRID(grid), rate_label, 0, 0, 1, 1);
    GtkWidget *rate_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 4);
    ui->settings_rate_combo = gtk_combo_box_text_new();
    for (int i = 0; i < settings_cur_rate_count; i++) {
        gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(ui->settings_rate_combo),
                                       settings_cur_rate_labels[i]);
    }
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(ui->settings_rate_combo),
                                   SETTINGS_CUSTOM_LABEL);
    gtk_box_pack_start(GTK_BOX(rate_box), ui->settings_rate_combo, FALSE, FALSE, 0);
    ui->settings_custom_entry = gtk_entry_new();
    gtk_widget_set_size_request(ui->settings_custom_entry, SETTINGS_ENTRY_WIDTH, -1);
    gtk_entry_set_placeholder_text(GTK_ENTRY(ui->settings_custom_entry), "Hz");
    gtk_box_pack_start(GTK_BOX(rate_box), ui->settings_custom_entry, FALSE, FALSE, 0);
    gtk_grid_attach(GTK_GRID(grid), rate_box, 1, 0, 1, 1);

    /* row 2: data format */
    GtkWidget *format_label = gtk_label_new("Data format:");
    gtk_widget_set_halign(format_label, GTK_ALIGN_START);
    gtk_grid_attach(GTK_GRID(grid), format_label, 0, 1, 1, 1);
    ui->settings_format_combo = gtk_combo_box_text_new();
    for (int i = 0; i < settings_cur_format_count; i++) {
        gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(ui->settings_format_combo),
                                       settings_cur_format_labels[i]);
    }
    gtk_grid_attach(GTK_GRID(grid), ui->settings_format_combo, 1, 1, 1, 1);

    /* row 3: centre frequency */
    GtkWidget *center_label = gtk_label_new("Center frequency (MHz):");
    gtk_widget_set_halign(center_label, GTK_ALIGN_START);
    gtk_grid_attach(GTK_GRID(grid), center_label, 0, 2, 1, 1);
    ui->settings_center_entry = gtk_entry_new();
    gtk_widget_set_size_request(ui->settings_center_entry, SETTINGS_ENTRY_WIDTH, -1);
    gtk_grid_attach(GTK_GRID(grid), ui->settings_center_entry, 1, 2, 1, 1);

    GtkWidget *hint_label = gtk_label_new(SETTINGS_HINT);
    gtk_label_set_line_wrap(GTK_LABEL(hint_label), TRUE);
    gtk_label_set_xalign(GTK_LABEL(hint_label), 0.0);
    gtk_box_pack_start(GTK_BOX(ui->settings_page), hint_label, FALSE, FALSE, 0);

    /* --- Waterfall frame: FFT size, window and visible length --- */
    frame = gtk_frame_new(WF_FRAME_LABEL);
    GtkWidget *wf_grid = gtk_grid_new();
    gtk_grid_set_row_spacing(GTK_GRID(wf_grid), SETTINGS_GRID_ROW_SPACE);
    gtk_grid_set_column_spacing(GTK_GRID(wf_grid), SETTINGS_GRID_COL_SPACE);
    gtk_widget_set_margin_start(wf_grid, PAGE_MARGIN);
    gtk_widget_set_margin_end(wf_grid, PAGE_MARGIN);
    gtk_widget_set_margin_top(wf_grid, PAGE_MARGIN);
    gtk_widget_set_margin_bottom(wf_grid, PAGE_MARGIN);
    gtk_container_add(GTK_CONTAINER(frame), wf_grid);

    GtkWidget *fft_label = gtk_label_new("FFT size:");
    gtk_widget_set_halign(fft_label, GTK_ALIGN_START);
    gtk_grid_attach(GTK_GRID(wf_grid), fft_label, 0, 0, 1, 1);
    ui->settings_fft_combo = gtk_combo_box_text_new();
    for (int i = 0; i < SETTINGS_FFT_PRESETS; i++) {
        gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(ui->settings_fft_combo),
                                       settings_fft_labels[i]);
    }
    gtk_grid_attach(GTK_GRID(wf_grid), ui->settings_fft_combo, 1, 0, 1, 1);

    GtkWidget *window_label = gtk_label_new("Window:");
    gtk_widget_set_halign(window_label, GTK_ALIGN_START);
    gtk_grid_attach(GTK_GRID(wf_grid), window_label, 0, 1, 1, 1);
    ui->settings_window_combo = gtk_combo_box_text_new();
    for (int i = 0; i < 4; i++) {
        gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(ui->settings_window_combo),
                                       settings_window_labels[i]);
    }
    gtk_grid_attach(GTK_GRID(wf_grid), ui->settings_window_combo, 1, 1, 1, 1);

    /* forward/backward streaming buffers, sized as a fraction of the
     * sample rate (1/4, 1/6, 1/8, 1/10, 1/12 of a second of samples) */
    GtkWidget *buf_label = gtk_label_new("Buffer size (samples):");
    {
        GtkWidget *buf_combo = gtk_combo_box_text_new();
        static const double buf_fracs[] = { 1.0 / 4.0, 1.0 / 6.0, 1.0 / 8.0,
                                            1.0 / 10.0, 1.0 / 12.0 };
        static const char *const buf_names[] = { "1/4", "1/6", "1/8", "1/10",
                                                 "1/12" };
        int active = 2;

        gtk_widget_set_halign(buf_label, GTK_ALIGN_START);
        gtk_grid_attach(GTK_GRID(wf_grid), buf_label, 0, 3, 1, 1);
        for (int i = 0; i < 5; i++) {
            gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(buf_combo),
                                           buf_names[i]);
            if (fabs(ui->stream_buffer_fraction - buf_fracs[i]) < 1e-9) {
                active = i;
            }
        }
        gtk_combo_box_set_active(GTK_COMBO_BOX(buf_combo), active);
        gtk_widget_set_tooltip_text(buf_combo,
                                    "Samples streamed into each buffer: a "
                                    "fraction of the sample rate (1/4 = a "
                                    "quarter second of samples)");
        g_signal_connect(buf_combo, "changed",
                         G_CALLBACK(on_buffer_size_changed), ui);
        gtk_grid_attach(GTK_GRID(wf_grid), buf_combo, 1, 3, 1, 1);
        buf_combo = NULL;
    }

    GtkWidget *len_label = gtk_label_new("Length:");
    gtk_widget_set_halign(len_label, GTK_ALIGN_START);
    gtk_grid_attach(GTK_GRID(wf_grid), len_label, 0, 2, 1, 1);
    GtkWidget *len_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 4);
    wf_len_combo = gtk_combo_box_text_new();
    for (int i = 0; i < WF_LEN_PRESETS; i++) {
        gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(wf_len_combo),
                                       wf_len_labels[i]);
    }
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(wf_len_combo),
                                   WF_LEN_CUSTOM_LABEL);
    gtk_box_pack_start(GTK_BOX(len_box), wf_len_combo, FALSE, FALSE, 0);
    wf_len_entry = gtk_entry_new();
    gtk_widget_set_size_request(wf_len_entry, WF_LEN_ENTRY_WIDTH, -1);
    gtk_entry_set_placeholder_text(GTK_ENTRY(wf_len_entry), "seconds");
    gtk_box_pack_start(GTK_BOX(len_box), wf_len_entry, FALSE, FALSE, 0);
    gtk_grid_attach(GTK_GRID(wf_grid), len_box, 1, 2, 1, 1);
    gtk_box_pack_start(GTK_BOX(ui->settings_page), frame, FALSE, FALSE, 0);

    g_signal_connect(ui->settings_rate_combo, "changed", G_CALLBACK(on_rate_combo_changed), ui);
    g_signal_connect(ui->settings_format_combo, "changed", G_CALLBACK(on_format_combo_changed), ui);
    g_signal_connect(ui->settings_center_entry, "activate", G_CALLBACK(on_center_entry_activate), ui);
    g_signal_connect(ui->settings_center_entry, "focus-out-event", G_CALLBACK(on_center_entry_focus_out), ui);
    g_signal_connect(ui->settings_fft_combo, "changed", G_CALLBACK(on_fft_combo_changed), ui);
    g_signal_connect(ui->settings_window_combo, "changed", G_CALLBACK(on_window_combo_changed), ui);
    /* the initial file decides which presets and values the Settings page
     * shows (IQ vs audio) */
    ui_configure_source_settings(ui);
    settings_init_widgets(ui);

    g_signal_connect(wf_len_combo, "changed", G_CALLBACK(on_wf_len_combo_changed), ui);
    g_signal_connect(wf_len_entry, "activate", G_CALLBACK(on_wf_len_entry_activate), ui);
    g_signal_connect(wf_len_entry, "focus-out-event", G_CALLBACK(on_wf_len_entry_focus_out), ui);
    wf_len_init_widgets(ui);

    /* --- Demodulation filter (playback-only control) ---
     * If enabled, this filter is applied to the samples inside the
     * bandwidth selected in the peak view (the demodulated band) when
     * the file is played. It is deliberately NOT connected to the
     * analyses: its output never feeds the analytical DAG. The playback
     * path itself is not implemented yet; the control only records the
     * configuration. */
    frame = gtk_frame_new("Demodulation Filter");
    {
        GtkWidget *fbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, PAGE_SPACING);

        gtk_widget_set_margin_start(fbox, PAGE_MARGIN);
        gtk_widget_set_margin_end(fbox, PAGE_MARGIN);
        gtk_widget_set_margin_top(fbox, PAGE_MARGIN);
        gtk_widget_set_margin_bottom(fbox, PAGE_MARGIN);
        ui->settings_filter_enable =
            gtk_check_button_new_with_label("Enable filter when playing (applied to the peak-view bandwidth)");
        g_signal_connect(ui->settings_filter_enable, "toggled",
                         G_CALLBACK(on_settings_filter_changed), ui);
        gtk_box_pack_start(GTK_BOX(fbox), ui->settings_filter_enable,
                           FALSE, FALSE, 0);

        hbox = gtk_flow_box_new();
        gtk_flow_box_set_selection_mode(GTK_FLOW_BOX(hbox), GTK_SELECTION_NONE);
        gtk_flow_box_set_column_spacing(GTK_FLOW_BOX(hbox), PAGE_SPACING);
        gtk_flow_box_set_row_spacing(GTK_FLOW_BOX(hbox), PAGE_SPACING);
        ui->settings_filter_kind = gtk_combo_box_text_new();
        for (int i = 0; i < 4; i++) {
            gtk_combo_box_text_append_text(
                GTK_COMBO_BOX_TEXT(ui->settings_filter_kind),
                filter_kind_labels[i]);
        }
        gtk_combo_box_set_active(GTK_COMBO_BOX(ui->settings_filter_kind), 0);
        g_signal_connect(ui->settings_filter_kind, "changed",
                         G_CALLBACK(on_settings_filter_changed), ui);
        gtk_flow_box_insert(GTK_FLOW_BOX(hbox), ui->settings_filter_kind,
                            -1);
        ui->settings_filter_f1 = gtk_entry_new();
        gtk_entry_set_placeholder_text(GTK_ENTRY(ui->settings_filter_f1),
                                       "cutoff Hz");
        gtk_widget_set_size_request(ui->settings_filter_f1, CLASSIC_ENTRY_WIDTH, -1);
        g_signal_connect(ui->settings_filter_f1, "changed",
                         G_CALLBACK(on_settings_filter_changed), ui);
        gtk_flow_box_insert(GTK_FLOW_BOX(hbox), ui->settings_filter_f1, -1);
        ui->settings_filter_f2 = gtk_entry_new();
        gtk_entry_set_placeholder_text(GTK_ENTRY(ui->settings_filter_f2),
                                       "upper cutoff Hz (BP/BS)");
        gtk_widget_set_size_request(ui->settings_filter_f2, CLASSIC_ENTRY_WIDTH, -1);
        g_signal_connect(ui->settings_filter_f2, "changed",
                         G_CALLBACK(on_settings_filter_changed), ui);
        gtk_flow_box_insert(GTK_FLOW_BOX(hbox), ui->settings_filter_f2, -1);
        ui->settings_filter_taps = gtk_entry_new();
        gtk_entry_set_placeholder_text(GTK_ENTRY(ui->settings_filter_taps),
                                       "taps");
        gtk_widget_set_size_request(ui->settings_filter_taps, CLASSIC_ENTRY_WIDTH, -1);
        g_signal_connect(ui->settings_filter_taps, "changed",
                         G_CALLBACK(on_settings_filter_changed), ui);
        gtk_flow_box_insert(GTK_FLOW_BOX(hbox), ui->settings_filter_taps, -1);
        gtk_box_pack_start(GTK_BOX(fbox), hbox, FALSE, FALSE, 0);

        {
            GtkWidget *hint = gtk_label_new(
                "Playback use only: the filter is applied to the samples "
                "inside the peak-view bandwidth (the demodulated band). It "
                "does not affect any analysis.");
            gtk_label_set_xalign(GTK_LABEL(hint), 0.0);
            gtk_label_set_line_wrap(GTK_LABEL(hint), TRUE);
            gtk_box_pack_start(GTK_BOX(fbox), hint, FALSE, FALSE, 0);
        }
        gtk_container_add(GTK_CONTAINER(frame), fbox);
        gtk_box_pack_start(GTK_BOX(ui->settings_page), frame, FALSE, FALSE, 0);
    }

    /* --- Analysis settings: a section at the end of the Settings page --- */
    gtk_box_pack_start(GTK_BOX(ui->settings_page),
                       build_analysis_settings_frame(ui), FALSE, FALSE, 0);


    /* --- 2. Selection page (parameter-estimation report) --- */
    ui->details_page = gtk_box_new(GTK_ORIENTATION_VERTICAL, PAGE_SPACING);
    gtk_widget_set_margin_start(ui->details_page, PAGE_MARGIN);
    gtk_widget_set_margin_end(ui->details_page, PAGE_MARGIN);
    gtk_widget_set_margin_top(ui->details_page, PAGE_MARGIN);
    gtk_widget_set_margin_bottom(ui->details_page, PAGE_MARGIN);

    /* top row: Analyze (async full analysis), Synthesize, Save report.
     * A flow box wraps the controls instead of forcing the pane wide. */
    hbox = gtk_flow_box_new();
    gtk_flow_box_set_selection_mode(GTK_FLOW_BOX(hbox), GTK_SELECTION_NONE);
    gtk_flow_box_set_column_spacing(GTK_FLOW_BOX(hbox), PAGE_SPACING);
    gtk_flow_box_set_row_spacing(GTK_FLOW_BOX(hbox), PAGE_SPACING);
    ui->demod_update_btn = gtk_button_new_with_label(DEMOD_UPDATE_LABEL);
    g_signal_connect(ui->demod_update_btn, "clicked", G_CALLBACK(on_demod_update), ui);
    gtk_flow_box_insert(GTK_FLOW_BOX(hbox), ui->demod_update_btn, -1);
    ui->synth_btn = gtk_button_new_with_label(SYNTH_BTN_LABEL);
    gtk_widget_set_sensitive(ui->synth_btn, FALSE);
    gtk_widget_set_tooltip_text(ui->synth_btn,
                                "Generate a synthetic 2 s signal from the "
                                "last analysis result");
    g_signal_connect(ui->synth_btn, "clicked", G_CALLBACK(on_synthesize), ui);
    gtk_flow_box_insert(GTK_FLOW_BOX(hbox), ui->synth_btn, -1);
    report_save_btn = gtk_button_new_with_label(SAVE_REPORT_BTN_LABEL);
    g_signal_connect(report_save_btn, "clicked", G_CALLBACK(on_save_report), ui);
    /* insensitive until an analysis report has been stored */
    gtk_widget_set_sensitive(report_save_btn, FALSE);
    gtk_flow_box_insert(GTK_FLOW_BOX(hbox), report_save_btn, -1);
    /* report output format: text, Markdown or HTML (all three are rendered
     * by the Reporter after every analysis, so switching is instant) */
    ui->report_format_combo = gtk_combo_box_text_new();
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(ui->report_format_combo),
                                   "Text");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(ui->report_format_combo),
                                   "Markdown");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(ui->report_format_combo),
                                   "HTML");
    gtk_combo_box_set_active(GTK_COMBO_BOX(ui->report_format_combo), 0);
    gtk_widget_set_tooltip_text(ui->report_format_combo,
                                "Report output format (Save uses the same "
                                "format)");
    g_signal_connect(ui->report_format_combo, "changed",
                     G_CALLBACK(on_report_format_changed), ui);
    gtk_flow_box_insert(GTK_FLOW_BOX(hbox), ui->report_format_combo, -1);
    /* running-analysis indicator, to the right of Save: shows which DAG
     * stage is executing and is cleared when the job finishes */
    ui->analysis_status = gtk_label_new("");
    gtk_label_set_xalign(GTK_LABEL(ui->analysis_status), 0.0);
    gtk_label_set_ellipsize(GTK_LABEL(ui->analysis_status),
                            PANGO_ELLIPSIZE_END);
    gtk_widget_set_hexpand(ui->analysis_status, TRUE);
    gtk_widget_set_margin_start(ui->analysis_status, 8);
    gtk_flow_box_insert(GTK_FLOW_BOX(hbox), ui->analysis_status, -1);
    gtk_box_pack_start(GTK_BOX(ui->details_page), hbox, FALSE, FALSE, 0);

    /* progress bar of the async Analyze job, hidden while idle */
    ui->report_progress = gtk_progress_bar_new();
    gtk_widget_set_visible(ui->report_progress, FALSE);
    gtk_box_pack_start(GTK_BOX(ui->details_page), ui->report_progress,
                       FALSE, FALSE, 0);

    /* full parameter-estimation report: scrollable, read-only, monospace */
    ui->report_scroll = gtk_scrolled_window_new(NULL, NULL);
    gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(ui->report_scroll),
                                   GTK_POLICY_AUTOMATIC, GTK_POLICY_AUTOMATIC);
    gtk_widget_set_hexpand(ui->report_scroll, TRUE);
    gtk_widget_set_vexpand(ui->report_scroll, TRUE);
    ui->report_view = gtk_text_view_new();
    gtk_text_view_set_editable(GTK_TEXT_VIEW(ui->report_view), FALSE);
    gtk_text_view_set_cursor_visible(GTK_TEXT_VIEW(ui->report_view), FALSE);
    gtk_text_view_set_wrap_mode(GTK_TEXT_VIEW(ui->report_view), GTK_WRAP_WORD);
    gtk_text_view_set_monospace(GTK_TEXT_VIEW(ui->report_view), TRUE);
    gtk_widget_set_margin_start(ui->report_view, PAGE_MARGIN);
    gtk_widget_set_margin_end(ui->report_view, PAGE_MARGIN);
    gtk_widget_set_margin_top(ui->report_view, PAGE_MARGIN);
    gtk_widget_set_margin_bottom(ui->report_view, PAGE_MARGIN);
    gtk_container_add(GTK_CONTAINER(ui->report_scroll), ui->report_view);
    gtk_box_pack_start(GTK_BOX(ui->details_page), ui->report_scroll,
                       TRUE, TRUE, 0);

    /* The old live-stats widgets (details_scroll/details_label/
     * details_status) are NOT part of the Selection page: it only shows
     * the parameter-estimation output. They stay NULL and every update
     * site guards on that. */
    ui->details_scroll = NULL;
    ui->details_label = NULL;
    ui->details_status = NULL;

    dock_add_page(ui, ui->details_page, DETAILS_TAB_LABEL);

    /* --- 3. LTE page (uplink PUSCH / PUCCH demodulation) --- */
    dock_add_page(ui, build_demodulate_page(ui), "LTE");

    /* --- 4. Settings page (last: configuration is rarely changed) --- */
    dock_add_page(ui, ui->settings_page, SETTINGS_TAB_LABEL);

    /* let every label on the dock pages wrap so the pane can shrink to the
     * width its controls really need */
    {
        int pages = gtk_notebook_get_n_pages(GTK_NOTEBOOK(ui->dock_pane));

        for (int i = 0; i < pages; i++) {
            dock_wrap_labels(gtk_notebook_get_nth_page(
                GTK_NOTEBOOK(ui->dock_pane), i));
        }
    }

    /* the Selection tab refreshes lazily when it becomes the current page */
    g_signal_connect(ui->dock_pane, "switch-page", G_CALLBACK(on_dock_switch), ui);
}

void ui_cleanup(Ui *ui)
{
    if (ui->hover_timeout != 0) {
        g_source_remove(ui->hover_timeout);
        ui->hover_timeout = 0;
    }
    hover_hint_destroy(ui);
    if (ui->player_window != NULL) {
        gtk_widget_destroy(ui->player_window); /* handler frees the player */
    }
    /* cancel a running Analyze job (the worker frees the job) */
    if (current_report_job != NULL) {
        current_report_job->cancelled = 1;
        current_report_job = NULL;
    }
    if (ui->streambuf != NULL) {
        sb_destroy(ui->streambuf);
        ui->streambuf = NULL;
    }
    if (ui->renderbuf != NULL) {
        rb_destroy(ui->renderbuf);
        ui->renderbuf = NULL;
    }
    if (ui->fc_job != NULL) {
        ((FcJob *)ui->fc_job)->cancelled = 1;
        ui->fc_job = NULL;
    }
    fcd_list_clear(&ui->fc_datasets);
    free(ui->fc_report_text);
    free(ui->fc_report_markdown);
    free(ui->fc_report_html);
    ui->fc_report_text = NULL;
    ui->fc_report_markdown = NULL;
    ui->fc_report_html = NULL;
    if (about_window != NULL) {
        gtk_widget_destroy(about_window);
        about_window = NULL;
    }
    if (about_image != NULL) {
        cairo_surface_destroy(about_image);
        about_image = NULL;
    }
    ui->wf_valid = false;
    clear_demod(ui);
    g_free(ui->report_text);
    g_free(ui->report_markdown);
    g_free(ui->report_html);
    ui->report_text = NULL;
    ui->report_markdown = NULL;
    ui->report_html = NULL;
    g_free(ui->aconfig_path);
    ui->aconfig_path = NULL;
    if (ui->streambuf != NULL) {
        sb_destroy(ui->streambuf);
        ui->streambuf = NULL;
    }
    if (ui->wf_scratch != NULL) {
        cairo_surface_destroy(ui->wf_scratch);
        ui->wf_scratch = NULL;
    }
    free(ui->wf_rowbuf);
    ui->wf_rowbuf = NULL;
    ui->wf_rowbuf_bins = 0;
    ui->wf_valid = false;
    cyclic_clear(&ui->cyclic);
    ssca_clear(&ui->ssca);
    cycledet_clear(&ui->cycledet);
    fluct_clear(&ui->fluct);
    cpar_clear(&ui->cpar);
    hos_clear(&ui->hos);
    ccc_clear(&ui->ccc);
    sdist_clear(&ui->sdist);
    classic_clear(&ui->classic);
    filter_clear(&ui->filt);
    pulse_clear(&ui->pulse);
    if (ui->popup_surface != NULL) {
        cairo_surface_destroy(ui->popup_surface);
        ui->popup_surface = NULL;
    }
    ui->popup_surface_bins = 0;
    ui->popup_surface_frames = 0;
    if (ui->cyclic_surface != NULL) {
        cairo_surface_destroy(ui->cyclic_surface);
        ui->cyclic_surface = NULL;
    }
    if (ui->ssca_surface != NULL) {
        cairo_surface_destroy(ui->ssca_surface);
        ui->ssca_surface = NULL;
    }
    if (ui->hos_surface != NULL) {
        cairo_surface_destroy(ui->hos_surface);
        ui->hos_surface = NULL;
    }
    if (ui->waterfall_surface != NULL) {
        cairo_surface_destroy(ui->waterfall_surface);
        ui->waterfall_surface = NULL;
    }
    free(ui->peak_spectrum);
    ui->peak_spectrum = NULL;
    peak_alloc_bins = 0;
    ui->peak_valid = false;
    if (ui->window != NULL) {
        gtk_widget_destroy(ui->window);
    }
    analysis_clear(&ui->analysis);

    /* widgets died with their windows; drop the dangling pointers */
    ui->window = NULL;
    ui->drawing = NULL;
    ui->info_label = NULL;
    ui->range_label = NULL;
    ui->db_min_scale = NULL;
    ui->db_max_scale = NULL;
    ui->auto_check = NULL;
    ui->db_lock_check = NULL;
    ui->play_btn = NULL;
    ui->sel_bw_label = NULL;
    ui->zoom_bw_label = NULL;
    ui->demod_kind_combo = NULL;
    ui->demod_result_label = NULL;
    ui->demod_const_drawing = NULL;
    ui->demod_run_btn = NULL;
    ui->peak_pane = NULL;
    ui->playing = false;
    ui->panning = false;
    ui->peak_handle = 0;
    ui->player = NULL;
    ui->player_window = NULL;
    ui->player_scale = NULL;
    ui->player_play_btn = NULL;
    ui->player_time_label = NULL;
    ui->dock_pane = NULL;
    ui->settings_page = NULL;
    ui->settings_rate_combo = NULL;
    ui->settings_custom_entry = NULL;
    ui->settings_format_combo = NULL;
    ui->settings_center_entry = NULL;
    ui->settings_fft_combo = NULL;
    ui->settings_window_combo = NULL;
    ui->details_page = NULL;
    ui->details_scroll = NULL;
    ui->details_label = NULL;
    ui->details_status = NULL;
    ui->cyclic_page = NULL;
    ui->cyclic_run_btn = NULL;
    ui->cyclic_est_combo = NULL;
    ui->cyclic_conj_check = NULL;
    ui->cyclic_scf_drawing = NULL;
    ui->cyclic_cdp_drawing = NULL;
    ui->cyclic_label = NULL;
    ui->has_cyclic = false;
    ui->ssca_surface = NULL;
    ui->has_ssca = false;
    ui->cycledet_page = NULL;
    ui->cycledet_run_btn = NULL;
    ui->cycledet_cdp_drawing = NULL;
    ui->cycledet_label = NULL;
    ui->has_cycledet = false;
    ui->ss_est_combo = NULL;
    ui->fluct_run_btn = NULL;
    ui->fluct_acf_drawing = NULL;
    ui->fluct_f_drawing = NULL;
    ui->fluct_label = NULL;
    ui->has_fluct = false;
    ui->cpar_acf_drawing = NULL;
    ui->cpar_par_drawing = NULL;
    ui->cpar_label = NULL;
    ui->has_cpar = false;
    ui->hos_page = NULL;
    ui->hos_run_btn = NULL;
    ui->hos_bisp_drawing = NULL;
    ui->hos_label = NULL;
    ui->has_hos = false;
    ui->ccc_label = NULL;
    ui->report_progress = NULL;
    ui->report_view = NULL;
    ui->report_scroll = NULL;
    report_save_btn = NULL;
    ui->demod_update_btn = NULL;
    ui->synth_btn = NULL;
    ui->analysis_status = NULL;
    ui->has_ccc = false;
    ui->sdist_page = NULL;
    ui->sdist_run_btn = NULL;
    ui->sdist_cf_drawing = NULL;
    ui->sdist_mgf_drawing = NULL;
    ui->sdist_label = NULL;
    ui->has_sdist = false;
    ui->classic_page = NULL;
    ui->classic_run_btn = NULL;
    ui->classic_probe_entry = NULL;
    ui->classic_cep_drawing = NULL;
    ui->classic_amdf_drawing = NULL;
    ui->classic_label = NULL;
    ui->has_classic = false;
    ui->filter_page = NULL;
    ui->filter_kind_combo = NULL;
    ui->filter_f1_entry = NULL;
    ui->filter_f2_entry = NULL;
    ui->filter_taps_entry = NULL;
    ui->filter_run_btn = NULL;
    ui->filter_resp_drawing = NULL;
    ui->filter_label = NULL;
    ui->has_filter = false;
    ui->pulse_page = NULL;
    ui->pulse_run_btn = NULL;
    ui->pulse_env_drawing = NULL;
    ui->pulse_spec_drawing = NULL;
    ui->pulse_label = NULL;
    ui->has_pulse = false;
}

void ui_reset_source(Ui *ui)
{
    /* stop playback and close the demod player + drop its buffer */
    ui->playing = false;
    if (ui->play_btn != NULL) {
        gtk_button_set_label(GTK_BUTTON(ui->play_btn), PLAY_BTN_LABEL);
    }
    if (ui->player_window != NULL) {
        gtk_widget_destroy(ui->player_window); /* handler frees the player */
    }
    clear_demod(ui);

    /* cancel a running Analyze job (worker frees it) */
    if (current_report_job != NULL) {
        current_report_job->cancelled = 1;
        current_report_job = NULL;
    }
    if (ui->report_progress != NULL) {
        gtk_widget_set_visible(ui->report_progress, FALSE);
    }
    if (ui->demod_update_btn != NULL) {
        gtk_widget_set_sensitive(ui->demod_update_btn, TRUE);
    }
    /* drop the stored analysis report and its text view */
    g_free(ui->report_text);
    ui->report_text = NULL;
    if (ui->analysis_status != NULL) {
        gtk_label_set_text(GTK_LABEL(ui->analysis_status), "");
    }
    if (ui->report_view != NULL) {
        GtkTextBuffer *buf = gtk_text_view_get_buffer(GTK_TEXT_VIEW(ui->report_view));

        gtk_text_buffer_set_text(buf, "", -1);
    }
    if (report_save_btn != NULL) {
        gtk_widget_set_sensitive(report_save_btn, FALSE);
    }
    lte_hint_clear(&ui->lte_hint);
    update_lte_hint_label(ui);
    /* reset the Analyze-derived ranking on the Selection page */
    ui->demod_mod_count = 0;
    ui->demod_id_count = 0;
    ui->demod_best_notes[0] = '\0';
    ui->demod_anomaly = 0;
    ui->demod_anomaly_explain[0] = '\0';

    analysis_clear(&ui->analysis);
    ui->has_selection = false;
    ui->sel_full_height = false;
    memset(&ui->sel_saved, 0, sizeof(ui->sel_saved));
    ui->dragging = false;
    ui->panning = false;
    /* the Settings presets depend on the kind of source (IQ or audio):
     * rebuild them and preselect the values detected from the file */
    if (ui->settings_rate_combo != NULL && ui->settings_format_combo != NULL) {
        ui_configure_source_settings(ui);
    }
    /* a window longer than the file would wrap the waterfall and make
     * drag selections ambiguous (and a very short file cannot fill the
     * height at all): the duration wins over every preset */
    if (ui->src != NULL && ui->src->duration > 0.0 &&
        ui->visible_seconds > ui->src->duration) {
        ui->visible_seconds = ui->src->duration;
        ui->scroll_position = 0.0;
        if (ui->visible_seconds < WF_LEN_MIN) {
            ui->visible_seconds = ui->src->duration;
        }
    }

    /* reset the dock pages: live details + the Run-driven pages */
    update_details_page(ui);
    reset_analysis_pages(ui);
    /* reset the per-tab Run-All findings labels */

    /* reset the three newer pages */
    ui->has_sdist = false;
    sdist_clear(&ui->sdist);
    gtk_label_set_text(GTK_LABEL(ui->sdist_label), RUN_PLACEHOLDER);
    gtk_widget_queue_draw(ui->sdist_cf_drawing);
    gtk_widget_queue_draw(ui->sdist_mgf_drawing);

    ui->has_classic = false;
    classic_clear(&ui->classic);
    gtk_entry_set_text(GTK_ENTRY(ui->classic_probe_entry), "0");
    gtk_label_set_text(GTK_LABEL(ui->classic_label), RUN_PLACEHOLDER);
    gtk_widget_queue_draw(ui->classic_cep_drawing);
    gtk_widget_queue_draw(ui->classic_amdf_drawing);


    ui->has_pulse = false;
    pulse_clear(&ui->pulse);
    gtk_label_set_text(GTK_LABEL(ui->pulse_label), RUN_PLACEHOLDER);
    gtk_widget_queue_draw(ui->pulse_env_drawing);
    gtk_widget_queue_draw(ui->pulse_spec_drawing);

    ui->scroll_position = 0.0;
    ui->visible_seconds = DEFAULT_VISIBLE_SECONDS; /* default span */
    /* never show a window longer than the recording: that wraps the view
     * and makes drag selections ambiguous */
    if (ui->src != NULL && ui->src->duration > 0.0 &&
        ui->visible_seconds > ui->src->duration) {
        ui->visible_seconds = ui->src->duration;
    }
    wf_len_init_widgets(ui); /* refresh the length combo/entry (guarded) */
    ui->freq_offset = 0.0;
    ui->visible_bw = ui->src->sample_rate;
    update_sel_bw_label(ui);   /* the zoom read-out starts at the full band */
    /* The cache may have been rebuilt with a different FFT size (Settings >
     * Waterfall > FFT size): the streaming buffers and the render buffer
     * must be rebuilt for it, and the surface must be computed again from
     * the complete render buffer instead of reusing rows from the old
     * cache. */
    if (ui->streambuf != NULL) {
        sb_destroy(ui->streambuf);
        ui->streambuf = NULL;
    }
    if (ui->renderbuf != NULL) {
        rb_destroy(ui->renderbuf);
        ui->renderbuf = NULL;
    }
    if (ui->cache != NULL) {
        ui->streambuf = sb_create(ui->cache, ui->stream_buffer_fraction);
        if (ui->streambuf != NULL) sb_wait_ready(ui->streambuf, 500);
        ui->renderbuf = rb_create(ui->cache, ui->streambuf);
    }
    ui->wf_valid = false;
    {
        double fps = wc_frames_per_second(ui->cache);

        wc_request(ui->cache, 0, 2 * (size_t)(ui->visible_seconds * fps) + 1);
    }
    update_range_label(ui);
    update_peak_spectrum(ui);
    ui->last_progress = -1; /* show the position text, not a stale % */
    update_info_label(ui);
    gtk_widget_queue_draw(ui->drawing);
    gtk_widget_queue_draw(ui->peak_pane);
}

/*
 * Build the main window and all of its widgets (without entering the main
 * loop). ui_run() calls this after gtk_init; harnesses can call it directly
 * to exercise the real widget tree.
 */
void ui_build_window(Ui *ui)
{
    /* --- main window --- */
    ui->window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_title(GTK_WINDOW(ui->window), WINDOW_TITLE);
    gtk_window_set_default_size(GTK_WINDOW(ui->window), WINDOW_WIDTH, WINDOW_HEIGHT);
    gtk_widget_set_size_request(ui->window, WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT);
    load_window_config(ui); /* restore the last user-chosen size */
    g_signal_connect(ui->window, "destroy", G_CALLBACK(gtk_main_quit), NULL);
    g_signal_connect(ui->window, "destroy", G_CALLBACK(on_window_size_save), ui);
    g_signal_connect(ui->window, "destroy", G_CALLBACK(on_window_destroy), ui);
    g_signal_connect(ui->window, "size-allocate", G_CALLBACK(on_window_size_allocate), ui);
    g_signal_connect(ui->window, "key-press-event", G_CALLBACK(on_key_press), ui);

    GtkWidget *vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    gtk_container_add(GTK_CONTAINER(ui->window), vbox);

    /* --- menu bar (first child of the main vbox; File only) --- */
    GtkWidget *menubar = gtk_menu_bar_new();
    GtkWidget *file_item = gtk_menu_item_new_with_label(MENU_FILE_LABEL);
    GtkWidget *file_menu = gtk_menu_new();
    GtkWidget *open_item = gtk_menu_item_new_with_label(MENU_OPEN_LABEL);

    gtk_menu_item_set_submenu(GTK_MENU_ITEM(file_item), file_menu);
    g_signal_connect(open_item, "activate", G_CALLBACK(on_open_menu), ui);
    gtk_menu_shell_append(GTK_MENU_SHELL(file_menu), open_item);
    GtkWidget *save_sel_item = gtk_menu_item_new_with_label(MENU_SAVE_SEL_LABEL);
    g_signal_connect(save_sel_item, "activate", G_CALLBACK(on_save_selection_menu), ui);
    gtk_menu_shell_append(GTK_MENU_SHELL(file_menu), save_sel_item);
    GtkWidget *quit_sep = gtk_separator_menu_item_new();
    gtk_menu_shell_append(GTK_MENU_SHELL(file_menu), quit_sep);
    GtkWidget *about_item = gtk_menu_item_new_with_label(MENU_ABOUT_LABEL);
    g_signal_connect(about_item, "activate", G_CALLBACK(on_about_menu), ui);
    gtk_menu_shell_append(GTK_MENU_SHELL(file_menu), about_item);
    gtk_menu_shell_append(GTK_MENU_SHELL(file_menu),
                          gtk_separator_menu_item_new());
    GtkWidget *quit_item = gtk_menu_item_new_with_label(MENU_QUIT_LABEL);
    g_signal_connect(quit_item, "activate", G_CALLBACK(on_quit_menu), ui);
    gtk_menu_shell_append(GTK_MENU_SHELL(file_menu), quit_item);
    gtk_menu_shell_append(GTK_MENU_SHELL(menubar), file_item);

    gtk_box_pack_start(GTK_BOX(vbox), menubar, FALSE, FALSE, 0);

    /* --- the main view: everything below the menu bar lives in the
     * "Spectrum" tab of a notebook (a pane that can host further views) --- */
    ui->main_notebook = gtk_notebook_new();
    gtk_notebook_set_tab_pos(GTK_NOTEBOOK(ui->main_notebook), GTK_POS_TOP);
    ui->spectrum_page = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    gtk_notebook_append_page(GTK_NOTEBOOK(ui->main_notebook),
                             ui->spectrum_page,
                             gtk_label_new(MAIN_TAB_LABEL));
    /* next to it: the frequency counter */
    ui->freq_counter_page = build_freq_counter_page(ui);
    gtk_notebook_append_page(GTK_NOTEBOOK(ui->main_notebook),
                             ui->freq_counter_page,
                             gtk_label_new(FREQ_COUNTER_TAB_LABEL));
    gtk_box_pack_start(GTK_BOX(vbox), ui->main_notebook, TRUE, TRUE, 0);

    /* --- waterfall drawing area + peak pane --- */
    /* --- left pane: vertical paned [waterfall | peak pane] --- */
    GtkWidget *vpaned = gtk_paned_new(GTK_ORIENTATION_VERTICAL);

    ui->drawing = gtk_drawing_area_new();
    gtk_widget_set_size_request(ui->drawing, -1, DRAWING_MIN_HEIGHT);
    gtk_widget_set_hexpand(ui->drawing, TRUE);
    gtk_widget_set_vexpand(ui->drawing, TRUE);
    gtk_widget_add_events(ui->drawing,
                          GDK_BUTTON_PRESS_MASK | GDK_BUTTON_RELEASE_MASK |
                          GDK_POINTER_MOTION_MASK | GDK_BUTTON_MOTION_MASK |
                          GDK_SCROLL_MASK | GDK_SMOOTH_SCROLL_MASK);
    g_signal_connect(ui->drawing, "draw", G_CALLBACK(draw_waterfall), ui);
    g_signal_connect(ui->drawing, "button-press-event", G_CALLBACK(on_button_press), ui);
    g_signal_connect(ui->drawing, "motion-notify-event", G_CALLBACK(on_motion_notify), ui);
    g_signal_connect(ui->drawing, "leave-notify-event",
                     G_CALLBACK(on_drawing_leave), ui);

    g_signal_connect(ui->drawing, "button-release-event", G_CALLBACK(on_button_release), ui);
    g_signal_connect(ui->drawing, "scroll-event", G_CALLBACK(on_drawing_scroll), ui);

    ui->peak_pane = gtk_drawing_area_new();
    gtk_widget_set_size_request(ui->peak_pane, -1, PEAK_PANE_HEIGHT);
    gtk_widget_set_hexpand(ui->peak_pane, TRUE);
    gtk_widget_add_events(ui->peak_pane,
                          GDK_BUTTON_PRESS_MASK | GDK_BUTTON_RELEASE_MASK |
                          GDK_POINTER_MOTION_MASK | GDK_BUTTON_MOTION_MASK);
    g_signal_connect(ui->peak_pane, "draw", G_CALLBACK(draw_peak_pane), ui);
    g_signal_connect(ui->peak_pane, "button-press-event", G_CALLBACK(on_peak_press), ui);
    g_signal_connect(ui->peak_pane, "motion-notify-event", G_CALLBACK(on_peak_motion), ui);
    g_signal_connect(ui->peak_pane, "button-release-event", G_CALLBACK(on_peak_release), ui);
    gtk_paned_pack1(GTK_PANED(vpaned), ui->drawing, TRUE, FALSE);
    gtk_paned_pack2(GTK_PANED(vpaned), ui->peak_pane, FALSE, FALSE);

    /* --- right: docked analysis notebook --- */
    build_dock_pane(ui);

    /* --- main content: HPaned [waterfall view | dock pane] --- */
    /* dock header (float/attach button) + the tabbed notebook; the whole
     * box moves between the paned and a floating window */
    ui->dock_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    {
        GtkWidget *header = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 4);
        GtkWidget *title = gtk_label_new("Analysis");
        GtkWidget *spacer = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);

        gtk_widget_set_halign(title, GTK_ALIGN_START);
        gtk_box_pack_start(GTK_BOX(header), title, FALSE, FALSE, 4);
        gtk_widget_set_hexpand(spacer, TRUE);
        gtk_box_pack_start(GTK_BOX(header), spacer, TRUE, TRUE, 0);

        gtk_box_pack_start(GTK_BOX(ui->dock_box), header, FALSE, FALSE, 0);
    }
    gtk_box_pack_start(GTK_BOX(ui->dock_box), ui->dock_pane, TRUE, TRUE, 0);

    GtkWidget *paned = gtk_paned_new(GTK_ORIENTATION_HORIZONTAL);
    gtk_paned_pack1(GTK_PANED(paned), vpaned, TRUE, FALSE);
    gtk_paned_pack2(GTK_PANED(paned), ui->dock_box, FALSE, FALSE);
    ui->dock_paned = paned;
    g_signal_connect(paned, "size-allocate",
                     G_CALLBACK(on_dock_paned_allocate), ui);
    gtk_box_pack_start(GTK_BOX(ui->spectrum_page), paned, TRUE, TRUE, 0);

    /* --- horizontal control bar --- */
    GtkWidget *control_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, CONTROL_SPACING);
    gtk_widget_set_margin_start(control_box, CONTROL_MARGIN_X);
    gtk_widget_set_margin_end(control_box, CONTROL_MARGIN_X);
    gtk_widget_set_margin_top(control_box, CONTROL_MARGIN_Y);
    gtk_widget_set_margin_bottom(control_box, CONTROL_MARGIN_Y);

    GtkWidget *screenshot_btn = gtk_button_new_with_label("Screenshot (S)");
    g_signal_connect(screenshot_btn, "clicked", G_CALLBACK(on_screenshot_clicked), ui);
    gtk_box_pack_start(GTK_BOX(control_box), screenshot_btn, FALSE, FALSE, CONTROL_SPACING);

    ui->play_btn = gtk_button_new_with_label(PLAY_BTN_LABEL);
    g_signal_connect(ui->play_btn, "clicked", G_CALLBACK(on_play_clicked), ui);
    gtk_box_pack_start(GTK_BOX(control_box), ui->play_btn, FALSE, FALSE, CONTROL_SPACING);
    {
        GtkWidget *rewind_btn = gtk_button_new_with_label("Rewind");

        gtk_widget_set_tooltip_text(rewind_btn,
                                    "Step the stream back one visible window");
        g_signal_connect(rewind_btn, "clicked",
                         G_CALLBACK(on_rewind_clicked), ui);
        gtk_box_pack_start(GTK_BOX(control_box), rewind_btn, FALSE, FALSE,
                           CONTROL_SPACING);
    }
    /* selected bandwidth, centred in the toolbar */
    /* the two bandwidth read-outs: selection above, peak-view zoom below */
    {
        GtkWidget *bw_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);

        gtk_widget_set_halign(bw_box, GTK_ALIGN_CENTER);
        gtk_widget_set_hexpand(bw_box, TRUE);
        ui->sel_bw_label = gtk_label_new("selected bandwidth: \xE2\x80\x94");
        gtk_widget_set_halign(ui->sel_bw_label, GTK_ALIGN_CENTER);
        gtk_widget_set_hexpand(ui->sel_bw_label, TRUE);
        gtk_label_set_ellipsize(GTK_LABEL(ui->sel_bw_label),
                                PANGO_ELLIPSIZE_END);
        ui->zoom_bw_label = gtk_label_new("zoomed bandwidth: \xE2\x80\x94");
        gtk_widget_set_halign(ui->zoom_bw_label, GTK_ALIGN_CENTER);
        gtk_widget_set_hexpand(ui->zoom_bw_label, TRUE);
        gtk_label_set_ellipsize(GTK_LABEL(ui->zoom_bw_label),
                                PANGO_ELLIPSIZE_END);
        gtk_box_pack_start(GTK_BOX(bw_box), ui->sel_bw_label, FALSE, FALSE, 0);
        gtk_box_pack_start(GTK_BOX(bw_box), ui->zoom_bw_label, FALSE, FALSE,
                           0);
        gtk_box_pack_start(GTK_BOX(control_box), bw_box, TRUE, TRUE, 0);
        update_sel_bw_label(ui);   /* start with the current zoom window */
    }

    GtkWidget *db_min_label = gtk_label_new("Min dB:");
    gtk_box_pack_start(GTK_BOX(control_box), db_min_label, FALSE, FALSE, CONTROL_SPACING);
    ui->db_min_scale = gtk_scale_new_with_range(GTK_ORIENTATION_HORIZONTAL,
                                                DB_SCALE_MIN, DB_SCALE_MAX, DB_SCALE_STEP);
    gtk_range_set_value(GTK_RANGE(ui->db_min_scale), ui->db_min);
    gtk_widget_set_size_request(ui->db_min_scale, DB_SCALE_WIDTH, -1);
    g_signal_connect(ui->db_min_scale, "value-changed", G_CALLBACK(on_db_min_changed), ui);
    g_signal_connect(ui->db_min_scale, "button-release-event",
                     G_CALLBACK(on_db_range_released), ui);
    g_signal_connect(ui->db_min_scale, "key-release-event",
                     G_CALLBACK(on_db_range_released), ui);
    gtk_box_pack_start(GTK_BOX(control_box), ui->db_min_scale, FALSE, FALSE, CONTROL_SPACING);

    GtkWidget *db_max_label = gtk_label_new("Max dB:");
    gtk_box_pack_start(GTK_BOX(control_box), db_max_label, FALSE, FALSE, CONTROL_SPACING);
    ui->db_max_scale = gtk_scale_new_with_range(GTK_ORIENTATION_HORIZONTAL,
                                                DB_SCALE_MIN, DB_SCALE_MAX, DB_SCALE_STEP);
    gtk_range_set_value(GTK_RANGE(ui->db_max_scale), ui->db_max);
    gtk_widget_set_size_request(ui->db_max_scale, DB_SCALE_WIDTH, -1);
    g_signal_connect(ui->db_max_scale, "value-changed", G_CALLBACK(on_db_max_changed), ui);
    g_signal_connect(ui->db_max_scale, "button-release-event",
                     G_CALLBACK(on_db_range_released), ui);
    g_signal_connect(ui->db_max_scale, "key-release-event",
                     G_CALLBACK(on_db_range_released), ui);
    gtk_box_pack_start(GTK_BOX(control_box), ui->db_max_scale, FALSE, FALSE, CONTROL_SPACING);

    ui->db_lock_check = gtk_check_button_new_with_label("Lock");
    gtk_widget_set_tooltip_text(ui->db_lock_check,
                                "Move both dynamic-range bounds together "
                                "(keeps the span); off: the sliders are "
                                "independent");
    g_signal_connect(ui->db_lock_check, "toggled",
                     G_CALLBACK(on_db_lock_toggled), ui);
    gtk_box_pack_start(GTK_BOX(control_box), ui->db_lock_check, FALSE, FALSE,
                       CONTROL_SPACING);

    ui->auto_check = gtk_check_button_new_with_label("Auto");
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(ui->auto_check), TRUE);
    g_signal_connect(ui->auto_check, "toggled", G_CALLBACK(on_auto_toggled), ui);
    gtk_box_pack_start(GTK_BOX(control_box), ui->auto_check, FALSE, FALSE, CONTROL_SPACING);

    gtk_box_pack_start(GTK_BOX(ui->spectrum_page), control_box, FALSE,
                       FALSE, 0);

    /* dB sliders start disabled because auto-scale is on */
    gtk_widget_set_sensitive(ui->db_min_scale, !ui->auto_scale);
    gtk_widget_set_sensitive(ui->db_max_scale, !ui->auto_scale);

    /* --- bottom status bar: visible range left, filename/status right --- */
    GtkWidget *status_bar = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, CONTROL_SPACING);
    gtk_widget_set_margin_start(status_bar, CONTROL_MARGIN_X);
    gtk_widget_set_margin_end(status_bar, CONTROL_MARGIN_X);
    gtk_widget_set_margin_top(status_bar, CONTROL_MARGIN_Y);
    gtk_widget_set_margin_bottom(status_bar, CONTROL_MARGIN_Y);
    ui->range_label = gtk_label_new("");
    gtk_widget_set_halign(ui->range_label, GTK_ALIGN_START);
    gtk_label_set_ellipsize(GTK_LABEL(ui->range_label), PANGO_ELLIPSIZE_MIDDLE);
    gtk_box_pack_start(GTK_BOX(status_bar), ui->range_label, TRUE, TRUE, 0);
    ui->info_label = gtk_label_new(INFO_LOADING);
    gtk_widget_set_halign(ui->info_label, GTK_ALIGN_END);
    gtk_label_set_ellipsize(GTK_LABEL(ui->info_label), PANGO_ELLIPSIZE_MIDDLE);
    gtk_box_pack_start(GTK_BOX(status_bar), ui->info_label, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(ui->spectrum_page), status_bar, FALSE,
                       FALSE, 0);

    /* --- startup request, refresh timer, main loop --- */
    request_current_window(ui);
    g_timeout_add(UI_TIMER_MS, on_timer, ui);
    gtk_widget_show_all(ui->window);
    /* the dock pane's default width (a third of the window) is applied on
     * the paned's first allocation (dock_initial_position) */
    /* peak pane ≈ 2/3 as tall as the waterfall: position = 3/5 of the
     * left column's total height (fall back to the drawing's request) */
    {
        int vpaned_h = gtk_widget_get_allocated_height(vpaned);

        if (vpaned_h > 1) {
            gtk_paned_set_position(GTK_PANED(vpaned), vpaned_h * 3 / 5);
        } else {
            gtk_paned_set_position(GTK_PANED(vpaned), DRAWING_MIN_HEIGHT);
        }
    }
    update_range_label(ui);
    update_peak_spectrum(ui);
    gtk_widget_queue_draw(ui->drawing);
    gtk_widget_queue_draw(ui->peak_pane);
}

int ui_run(Ui *ui)
{
    /*
     * Initialise GTK here so the UI module owns the toolkit lifecycle;
     * building widgets before gtk_init crashes some backends (quartz).
     */
    if (!gtk_init_check(NULL, NULL)) {
        fprintf(stderr, "ui: cannot initialise GTK (no display?)\n");
        return 1;
    }
    ui_build_window(ui);
    gtk_main();
    return 0;
}
