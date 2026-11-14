/*
 * report.h — the Reporter.
 *
 * The Reporter owns the document model and every output format. It receives
 * the consolidated analysis data (the envelopes, the Director's view and the
 * raw per-analysis results the sections print), assembles one document and
 * renders it as plain text, Markdown or HTML.
 *
 * The plain-text rendering is the historical report and must stay
 * byte-identical (locked by the golden-report test); Markdown and HTML are
 * presentations of the same document.
 */
#ifndef TE15_REPORT_H
#define TE15_REPORT_H

#include <stdbool.h>
#include <stddef.h>

#include "analysis.h"
#include "aresult.h"
#include "ccc.h"
#include "classic.h"
#include "cpar.h"
#include "cyclic.h"
#include "cycledet.h"
#include "dag.h"
#include "dsp.h"
#include "director.h"
#include "fluct.h"
#include "hos.h"
#include "intraburst.h"
#include "iq_source.h"
#include "metrics.h"
#include "modconf.h"
#include "pulse.h"
#include "region.h"
#include "signals.h"
#include "sot.h"
#include "ssca.h"
#include "ssclass.h"

#define REPORT_TOTAL_MAX 24

/* Output formats. */
typedef enum {
    REPORT_TEXT = 0,
    REPORT_MARKDOWN,
    REPORT_HTML
} ReportFormat;

const char *report_format_name(ReportFormat fmt);   /* "text"/"markdown"/"html" */
const char *report_format_suffix(ReportFormat fmt); /* ".txt"/".md"/".html"      */

/* Progress/cancellation callbacks: the section formatters report their
 * stages like the worker used to, so the progress DAG is unchanged. */
typedef struct {
    void *ctx;
    void (*advance)(void *ctx, const char *label);
    bool (*cancelled)(void *ctx);
} ReportHooks;

/* Everything the document needs. All pointers are borrowed for the duration
 * of report_build(); structs are shallow-copied (the formatters only read
 * scalars and arrays owned by the caller). */
typedef struct {
    /* context */
    const char *stamp;        /* "YYYY-MM-DD HH:MM:SS"                   */
    const IqSource *src;
    Region  region;
    double  settings_rate;
    IqFormat settings_format;
    double  settings_center;
    int     settings_fft_size;
    DspWindow settings_window;

    /* common source of truth + flags */
    const Sot *sot;
    bool a0_ok;
    bool sot_failed;

    /* per-analysis results */
    const Analysis        *analysis;
    const Metrics         *metrics;
    const ClassicResult   *classic;
    const PulseResult     *pulse;
    const HosResult       *hos;
    const CyclicResult    *cyclic;
    const SscaResult      *ssca;
    const CycleDetResult  *cycledet;
    const CparResult      *cpar;
    const FluctResult     *fluct;
    const CccResult       *ccc;
    const IntraBurstResult *intraburst;

    /* ranking / spread classification / envelopes for the Director */
    const ModCandidate    *mods;
    size_t                 n_mods;
    const RankedMatch     *ids;
    size_t                 n_ids;
    const SpreadCandidate *spread;
    size_t                 n_spread;
    bool                   anomaly;
    const char            *anomaly_explain;
    const ArSet           *envelopes;
    const DagRun          *dag;

    /* DAG flags */
    struct { bool a1, a2, a3, a4, a5, a6, a7, a8, a9; } ran;
    struct { bool a0, a1, a2, a3, a4, a5, a6a, a6b, a7, a8, a9; } ok;
    const char *skip_a4, *skip_a5, *skip_a6, *skip_a7, *skip_a8, *skip_a9;
    bool d6_wideband, d4_have_cf, d5_no_cycles, validated_cycles;

    /* cancellation bookkeeping */
    bool        cancelled;
    int         stages_done;
    const char *stage_name;
    double      elapsed_s;

    ReportHooks hooks;
} ReportInput;

struct ReportDoc;
typedef struct ReportDoc ReportDoc;

/* Build the document once (runs the progress hooks), render formats from it. */
ReportDoc  *report_build(ReportInput *in);
const char *report_text(const ReportDoc *doc);
char       *report_render(const ReportDoc *doc, ReportFormat fmt);
void        report_doc_free(ReportDoc *doc);

/* Labels shared with the UI. */
const char *report_format_label(IqFormat f);
const char *report_window_name(DspWindow w);

#endif /* TE15_REPORT_H */
