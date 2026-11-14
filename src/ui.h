/*
 * ui.h — GTK3 user interface for the SDR waterfall analyzer.
 *
 * Owns all widget state, the scroll/zoom model, the drag-selection state,
 * the docked analysis pane (right of the waterfall) and the offscreen
 * render surfaces used for screenshots.
 *
 * Threading rule: every field of `Ui` is touched exclusively from the
 * GTK main thread. The waterfall cache is accessed through its
 * thread-safe API only.
 */
#ifndef TE15_UI_H
#define TE15_UI_H

#include <stdbool.h>
#include <gtk/gtk.h>
#include <cairo.h>

#include "iq_source.h"
#include "waterfall_cache.h"
#include "fcdag.h"
#include "fcdataset.h"
#include "lte_uplink.h"
#include "renderbuf.h"
#include "analysis.h"
#include "audio.h"
#include "colormap.h"
#include "cpar.h"
#include "cyclic.h"
#include "detect.h"
#include "fluct.h"
#include "hos.h"
#include "metrics.h"
#include "signals.h"
#include "synth.h"
#include "ssca.h"
#include "cycledet.h"
#include "ccc.h"
#include "sdist.h"
#include "classic.h"
#include "filters.h"
#include "pulse.h"
#include "modconf.h"
#include "signals.h"
#include "synth.h"

/* Edge masks for Shift+drag selection resizing. */
enum {
    SEL_EDGE_NONE = 0,
    SEL_EDGE_LEFT = 1,
    SEL_EDGE_RIGHT = 2,
    SEL_EDGE_TOP = 4,
    SEL_EDGE_BOTTOM = 8,
    SEL_EDGE_INSIDE = 16
};

typedef struct StreamBuf StreamBuf;

typedef struct Ui {
    /* --- view model (main thread only) --- */
    double visible_seconds;  /* vertical (time) zoom: visible span   */
    double scroll_position;  /* start of the visible time window     */
    double visible_bw;       /* horizontal (freq) zoom: visible span */
    double freq_offset;      /* low edge of the visible freq window,
                              * Hz above (center_freq - sr/2)        */
    double db_min, db_max;
    bool   auto_scale;

    /* --- selection & analysis --- */
    Region selection;      /* normalised selection rect  */
    bool   has_selection;  /* true after a valid drag    */
    /* full-height (blue) selection mode: the rectangle keeps its frequency
     * span but covers the whole file time range; double-clicking inside it
     * toggles the mode, `sel_saved` remembers the normal dimensions */
    bool   sel_full_height;
    Region sel_saved;
    gint64 sel_toggle_us;   /* last full-height toggle (double-click guard) */
    bool   dragging;       /* button held during drag    */
    double drag_anchor_freq, drag_anchor_time; /* press point */
    double drag_freq, drag_time;               /* live endpoint */
    bool   sel_resizing;   /* Shift+drag is resizing an edge */
    int    sel_resize_edges;  /* SEL_EDGE_* mask being dragged */
    bool   sel_press_inside;  /* press landed inside the rect   */
    double sel_press_x, sel_press_y; /* press position, px     */
    Analysis analysis;     /* result of the last analysis */
    Metrics  metrics;      /* extended feature suite      */
    RankedMatch ranked[3]; /* top-3 most likely signals   */
    size_t  ranked_count;

    /* --- Selection page analysis report --- */
    char      *report_text;    /* plain-text report                      */
    char      *report_markdown;/* Markdown presentation                  */
    char      *report_html;    /* HTML presentation                      */
    int        report_format;  /* selected ReportFormat                  */
    char      *aconfig_path;   /* persisted analysis settings            */
    GtkWidget *analysis_settings_box; /* container of the settings editor */
    GtkWidget *report_format_combo; /* Text / Markdown / HTML selector  */
    GtkWidget *report_view;    /* GtkTextView showing report_text       */
    GtkWidget *report_scroll;  /* scrolled window around report_view    */

    /* --- navigation & playback --- */
    bool    playing;         /* waterfall playback running (loops) */
    bool    playhead_valid;  /* the playback position is known          */
    double  playhead_time;   /* stream time of the playback position    */
    GtkWidget *play_btn;     /* Play/Pause toggle in the control bar */
    int64_t last_play_tick;  /* g_get_monotonic_time of last advance */
    double drag_pan_x, drag_pan_y; /* waterfall pan-drag anchor */
    bool   panning;          /* plain drag = pan (Shift+drag = select) */

    /* --- peak pane (spectrum strip between waterfall and toolbar) --- */
    GtkWidget *peak_pane;    /* drawing area: full-band PSD + green zoom rect */
    float     *peak_spectrum;/* FFT_SIZE mean dB over the visible window */
    bool       peak_valid;
    double     peak_db_lo, peak_db_hi; /* data-driven dB axis of the strip */
    int        peak_handle;  /* 0 none, -1 left edge, +1 right edge,
                              * 2 middle (pan) while dragging          */

    /* --- demodulated audio (computed on demand, kept for play/save) --- */
    float       *demod_audio;   /* mono float at demod_rate */
    size_t       demod_len;
    int          demod_rate;
    char         demod_default_name[1024]; /* suggested WAV filename */

    /* --- audio player (CoreAudio) --- */
    AudioPlayer *player;          /* non-NULL while a player window exists */
    GtkWidget   *player_window;
    GtkWidget   *player_scale;    /* seek bar */
    GtkWidget   *player_play_btn; /* Play/Pause toggle */
    GtkWidget   *player_time_label;
    bool         player_syncing;  /* guard for programmatic scale updates */

    /* --- Settings page (before Demod) --- */
    GtkWidget *settings_page;
    /* Demodulation filter (playback-only, never used by the analyses) */
    GtkWidget *settings_filter_enable; /* checkbox: enable the filter */
    GtkWidget *settings_filter_kind;   /* combo: LP / HP / BP / BS    */
    GtkWidget *settings_filter_f1;     /* cutoff entry, Hz            */
    GtkWidget *settings_filter_f2;     /* upper cutoff entry, Hz      */
    GtkWidget *settings_filter_taps;   /* tap-count entry             */
    bool   demod_filter_enabled;
    int    demod_filter_kind;
    double demod_filter_f1_hz;
    double demod_filter_f2_hz;
    int    demod_filter_taps;
    GtkWidget *settings_rate_combo;   /* preset rates + Custom..    */
    GtkWidget *settings_custom_entry; /* Hz text, active for Custom */
    GtkWidget *settings_format_combo; /* data format dropdown        */
    GtkWidget *settings_center_entry; /* centre frequency, MHz       */
    GtkWidget *settings_fft_combo;    /* waterfall FFT size          */
    GtkWidget *settings_window_combo; /* waterfall FFT window        */
    double     settings_rate;         /* current effective rate, Hz  */
    IqFormat   settings_format;       /* current data format         */
    double     settings_center;       /* current centre, Hz          */
    int        settings_fft_size;     /* current waterfall FFT size  */
    DspWindow  settings_window;       /* current waterfall window    */
    /*
     * Called whenever a Settings control changes; main.c applies the
     * values to the source, rebuilds the waterfall cache and resets the
     * view. Returns 0 on success.
     */
    int (*apply_settings_cb)(struct Ui *ui, double rate_hz, IqFormat format,
                             double center_hz, int fft_size,
                             DspWindow window_type);

    /* --- docked analysis pane (GtkNotebook, right of the waterfall) --- */
    GtkWidget *dock_pane;       /* tabbed analysis pane (right of the
                                 * waterfall) */
    GtkWidget *dock_box;        /* dock header + notebook                */
    GtkWidget *dock_paned;      /* the HPaned holding the waterfall + dock */
    bool       dock_pos_initialised; /* default width already applied   */

    /* Demod page (live: follows the current selection) */
    GtkWidget *details_page;
    GtkWidget *details_preview;   /* contain-fit spectrogram preview */
    GtkWidget *details_scroll;    /* scrolled stats area             */
    GtkWidget *details_label;     /* stats markup (inside the scroll)*/
    GtkWidget *details_play_btn;  /* Play Demod                      */
    GtkWidget *details_save_btn;  /* Save Demod                      */
    GtkWidget *details_status;    /* demod status line               */

    /* Spectral Correlation page (FAM + SSCA estimators) */
    CyclicResult cyclic;
    bool         has_cyclic;
    bool         cyclic_conjugate;  /* conjugate SCF selected      */
    int          cyclic_estimator;  /* 0 FAM, 1 SSCA               */
    GtkWidget   *cyclic_page;
    GtkWidget   *cyclic_run_btn;
    GtkWidget   *cyclic_est_combo;  /* FAM / SSCA selector         */
    GtkWidget   *cyclic_conj_check; /* Conjugate checkbox          */
    GtkWidget   *cyclic_scf_drawing;
    GtkWidget   *cyclic_cdp_drawing;
    GtkWidget   *cyclic_label;
    cairo_surface_t *cyclic_surface;
    SscaResult   ssca;
    bool         has_ssca;
    cairo_surface_t *ssca_surface;

    /* Cycle Detection page (CFAR multi-cycle detector + Burel + CPAR) */
    CycleDetResult cycledet;
    bool           has_cycledet;
    GtkWidget     *cycledet_page;
    GtkWidget     *cycledet_run_btn;
    GtkWidget     *cycledet_cdp_drawing; /* statistic per candidate CF */
    GtkWidget     *cycledet_label;
    int          ss_estimator;  /* 0 Fluctuations, 1 CPAR,
                                 * 2 CFAR cycle detector      */
    GtkWidget   *ss_est_combo;
    FluctResult fluct;
    bool        has_fluct;
    GtkWidget  *fluct_run_btn;
    GtkWidget  *fluct_acf_drawing;
    GtkWidget  *fluct_f_drawing;
    GtkWidget  *fluct_label;

    CparResult  cpar;
    bool        has_cpar;
    GtkWidget  *cpar_acf_drawing;
    GtkWidget  *cpar_par_drawing;
    GtkWidget  *cpar_label;

    /* Higher-Order Statistics page (cumulants + cyclic cumulants) */
    HosResult   hos;
    bool        has_hos;
    GtkWidget  *hos_page;
    GtkWidget  *hos_run_btn;
    GtkWidget  *hos_bisp_drawing;
    GtkWidget  *hos_label;
    cairo_surface_t *hos_surface;
    CccResult   ccc;               /* cyclic temporal cumulants */
    bool        has_ccc;
    GtkWidget  *ccc_label;         /* AMR features + verdict    */

    /* --- Distribution Analysis page (whole-file scope) --- */
    SdistResult sdist;
    bool       has_sdist;
    GtkWidget *sdist_page;
    GtkWidget *sdist_run_btn;
    GtkWidget *sdist_cf_drawing;   /* |CF| vs omega (+ Gaussian) */
    GtkWidget *sdist_mgf_drawing;  /* MGF vs s                    */
    GtkWidget *sdist_label;

    /* --- Classic Analysis page (selection scope) --- */
    ClassicResult classic;
    bool       has_classic;
    GtkWidget *classic_page;
    GtkWidget *classic_run_btn;
    GtkWidget *classic_probe_entry; /* Goertzel probe Hz */
    GtkWidget *classic_cep_drawing; /* cepstrum plot */
    GtkWidget *classic_amdf_drawing;/* AMDF plot */
    GtkWidget *classic_label;

    /* --- Filters page (selection scope) --- */
    FilterResult filt;
    bool       has_filter;
    GtkWidget *filter_page;
    GtkWidget *filter_kind_combo;  /* LP/HP/BP/BS */
    GtkWidget *filter_f1_entry;    /* Hz */
    GtkWidget *filter_f2_entry;    /* Hz (BP/BS) */
    GtkWidget *filter_taps_entry;  /* odd number */
    GtkWidget *filter_run_btn;
    GtkWidget *filter_resp_drawing; /* |H| + PSD before/after */
    GtkWidget *filter_label;

    /* --- Pulse Trains page (selection scope) --- */
    PulseResult pulse;
    bool       has_pulse;
    GtkWidget *pulse_page;
    GtkWidget *pulse_run_btn;
    GtkWidget *pulse_env_drawing;   /* envelope + threshold + pulses */
    GtkWidget *pulse_spec_drawing;  /* line spectrum + f0 marker     */
    GtkWidget *pulse_label;

    /* --- Selection Report / Update (Demod tab) --- */
    GtkWidget *demod_report_btn; /* runs all selection analyses async */
    GtkWidget *demod_update_btn; /* async re-analysis -> best parameters */
    GtkWidget *synth_btn;        /* Selection > Synthesize (after Analyze) */
    GtkWidget *analysis_status;  /* Selection tab: running analysis label  */
    GtkWidget *report_progress;  /* progress bar in the Demod page    */
    GtkWidget *demod_best_label; /* best parameters + top mods/system */
    ModCandidate demod_mods[12];
    size_t       demod_mod_count;
    RankedMatch  demod_ids[5];
    size_t       demod_id_count;
    char         demod_best_notes[1024];
    int          demod_anomaly;
    char         demod_anomaly_explain[512];

    /* --- last analysis -> synthetic signal (Selection > Synthesize) ---
     * Filled from the completed Analyze job; the Synthesis window builds
     * a 2 s surrogate from it plus the measured noise profile. */
    SynthSpec  last_synth;
    bool       last_synth_valid;

    /* --- main.c integration --- */
    /*
     * Called when the user picks a recording via File > Open. The owner
     * (main.c) must open the new source, rebuild the waterfall cache,
     * assign ui->src/ui->cache and call ui_reset_source(). Return 0 on
     * success, non-zero on failure.
     */
    int (*open_file_cb)(struct Ui *ui, const char *path);
    /*
     * Manual-entry fallback: called when the recording parameters could
     * not be recovered from the filename/container. The owner opens the
     * file with the values entered in the dialog (File > Open > choose
     * file, then the parameter dialog). Return 0 on success.
     */
    int (*open_file_custom_cb)(struct Ui *ui, const char *path,
                               double center_hz, double rate_hz,
                               IqFormat format);

    /* --- borrowed resources (owned by main.c) --- */
    IqSource       *src;
    WaterfallCache *cache;
    Colormap        colormap;

    /* --- widgets --- */
    GtkWidget *window;
    GtkWidget *main_notebook;   /* main view tabs (currently "Spectrum") */
    GtkWidget *spectrum_page;   /* the page holding the waterfall + dock */
    GtkWidget *freq_counter_page;   /* second main tab                  */
    /* --- Frequency Counter: measurement sessions + analysis --- */
    FcDatasetList fc_datasets;       /* loaded measurement datasets     */
    GtkWidget  *fc_view;             /* dataset/session tree (GtkTreeView) */
    GtkTreeStore *fc_model;          /* its model                       */
    GtkWidget  *fc_details_view;     /* schema details of the selection */
    GtkWidget  *fc_search_entry;     /* filter datasets/sessions/...    */
    GtkWidget  *fc_save_btn;
    GtkWidget  *fc_remove_btn;
    GtkWidget  *fc_remove_all_btn;
    GtkWidget  *fc_expand_btn;
    GtkWidget  *fc_collapse_btn;
    GtkWidget  *fc_analyze_btn;      /* Analysis pane controls          */
    GtkWidget  *fc_progress;
    GtkWidget  *fc_status_label;
    GtkWidget  *fc_report_view;      /* GtkTextView with the report     */
    GtkWidget  *fc_report_format_combo;
    GtkWidget  *fc_report_save_btn;
    char       *fc_report_text;      /* rendered report, all formats    */
    char       *fc_report_markdown;
    char       *fc_report_html;
    int         fc_report_format;
    int         fc_scope;            /* FclScope: per dataset or merged  */
    GtkWidget  *fc_scope_radio_per;   /* Settings: analysis scope         */
    GtkWidget  *fc_scope_radio_merged;
    bool        fc_node_enabled[32];  /* Settings: analyses to run        */
    int         fc_node_check_count;  /* number of checkboxes             */
    GtkWidget  *fc_node_checks[32];   /* the checkbox widgets             */
    void       *fc_job;              /* running FcJob (async analysis)  */
    GtkWidget *drawing;
    /* stationary-hover hint ("Shift+Click to select"): drawn inside the
     * waterfall with the same in-widget tag style as the frequency readout */
    guint      hover_timeout;   /* 1 s stationary delay                  */
    bool       hover_hint_visible;
    GtkWidget *info_label;   /* bottom-right: filename / transient notice */
    /* committed (released) dynamic range: raw dB values are normalised
     * through it when the waterfall looks up the viridis colour scheme */
    double     wf_range_lo, wf_range_hi;
    gint64     info_expiry_us;   /* transient message deadline (0 = none) */
    GtkWidget *range_label;  /* bottom-right: visible time span and
                              * first/last sample index           */
    GtkWidget *sel_bw_label;   /* "selected bandwidth: ..." in the toolbar */
    GtkWidget *zoom_bw_label;  /* "zoomed bandwidth: ..." below it (peak view) */
    /* cursor frequency read-out at the top of the waterfall */
    bool   cursor_active;
    double cursor_x;
    double cursor_freq_hz;
    /* --- LTE tab: parameters guessed by the Selection analysis --- */
    LteHint    lte_hint;
    /* --- Demodulate tab --- */
    GtkWidget *demod_kind_combo;   /* Auto / PUSCH / PUCCH             */
    GtkWidget *demod_hint_label;   /* parameters guessed by Analyze    */
    GtkWidget *demod_result_label; /* summary of the last demodulation */
    GtkWidget *demod_const_drawing;/* constellation plot               */
    GtkWidget *demod_run_btn;
    GtkWidget *db_min_scale;
    GtkWidget *db_max_scale;
    GtkWidget *auto_check;
    GtkWidget *db_lock_check;   /* "Lock": the dB sliders move together */
    bool       db_locked;

    /* --- streaming waterfall render state --- */
    cairo_surface_t *waterfall_surface; /* last rendered waterfall     */
    cairo_surface_t *wf_scratch;        /* blit scratch for the shift  */
    float  *wf_rowbuf;                  /* one FFT row of dB values    */
    int     wf_rowbuf_bins;             /* allocated bins of wf_rowbuf */
    bool    wf_valid;                   /* surface content is current  */
    int     wf_w, wf_h;                 /* surface size when rendered  */
    uint64_t wf_sig;                    /* mapping/colour signature    */
    double  wf_top_time;                /* time of the top pixel row   */
    float   wf_display_min;             /* colour scale of the surface */
    float   wf_display_max;
    StreamBuf *streambuf;
    RenderBuf *renderbuf;               /* forward/backward buffers    */
    double  stream_buffer_fraction;     /* buffer size as a fraction of
                                         * the sample rate (1/4 .. 1/12) */

    cairo_surface_t *popup_surface;     /* details preview image RGB24 */
    int popup_surface_bins, popup_surface_frames; /* dims of the cache */

    int last_progress; /* last observed load progress (-1 when idle) */
} Ui;

/* Fill defaults (view model, colormap) and borrow src/cache. */
void ui_init(Ui *ui, IqSource *src, WaterfallCache *cache);

/* Build the main window and widgets (called by ui_run; exposed so tests can
 * exercise the real widget tree without entering the main loop). */
void ui_build_window(Ui *ui);

/* Destroy widgets and offscreen surfaces. */
void ui_cleanup(Ui *ui);

/*
 * Reset the UI after the recording changed (File > Open): clears the
 * selection, detected rectangles, analysis, demod buffer and player,
 * returns to the start of the new recording, resyncs the scrollbar and
 * requests the first cache window. Call after assigning ui->src and
 * ui->cache (the previous ones must already be destroyed by the caller).
 */
void ui_reset_source(Ui *ui);

/* Build the window and run the GTK main loop; returns the exit code. */
int ui_run(Ui *ui);

#endif /* TE15_UI_H */
