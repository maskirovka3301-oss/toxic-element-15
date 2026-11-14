/*
 * apublish.h — publishers that copy an analysis' private result struct into
 * the common AnalysisResult envelope.
 *
 * The analysis modules stay untouched: these adapters are the single place
 * that knows both representations, so the envelope (and the Director / the
 * Reporter downstream) never depends on a module's internals.
 */
#ifndef TE15_APUBLISH_H
#define TE15_APUBLISH_H

#include "aresult.h"
#include "analysis.h"
#include "ccc.h"
#include "classic.h"
#include "cpar.h"
#include "cyclic.h"
#include "cycledet.h"
#include "fluct.h"
#include "hos.h"
#include "intraburst.h"
#include "metrics.h"
#include "modconf.h"
#include "pulse.h"
#include "signals.h"
#include "ssca.h"
#include "ssclass.h"

void apub_analysis(AnalysisResult *r, const Analysis *a);
void apub_metrics(AnalysisResult *r, const Metrics *m);
void apub_classic(AnalysisResult *r, const ClassicResult *c);
void apub_pulse(AnalysisResult *r, const PulseResult *p);
void apub_hos(AnalysisResult *r, const HosResult *h);
void apub_cyclic(AnalysisResult *r, const CyclicResult *c);
void apub_ssca(AnalysisResult *r, const SscaResult *s);
void apub_cycledet(AnalysisResult *r, const CycleDetResult *d);
void apub_fluct(AnalysisResult *r, const FluctResult *f);
void apub_cpar(AnalysisResult *r, const CparResult *c);
void apub_ccc(AnalysisResult *r, const CccResult *c);
void apub_intraburst(AnalysisResult *r, const IntraBurstResult *i);
void apub_modconf(AnalysisResult *r, const ModCandidate *cand, size_t n);
void apub_ssclass(AnalysisResult *r, const SpreadCandidate *cand, size_t n);
void apub_signals(AnalysisResult *r, const RankedMatch *m, size_t n,
                  bool anomaly, const char *anomaly_note);
void apub_sot(AnalysisResult *r, double pass_hz, double stop_hz, double gain,
              double noise_floor_db, size_t samples, double rate_hz);

/* Confidence helpers the worker/Director share: publish the analysis'
 * self-assessed confidence (0..1) so ranking never invents it. */
void apub_set_confidence(AnalysisResult *r, double confidence);

#endif /* TE15_APUBLISH_H */
