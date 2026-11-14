/*
 * aregistry.h — declarative description of every analysis in the DAG.
 *
 * Each entry names the analysis (stable id + human title), its DAG node and
 * documents:
 *   - the parameters it publishes (key, unit, description), which the
 *     Reporter uses for labels/units and the Director for ranking;
 *   - its tunable settings (key, type, default, range), which a future
 *     settings UI can edit and the analyses can read as overrides.
 *
 * The registry is static data: adding an analysis means adding one entry.
 */
#ifndef TE15_AREGISTRY_H
#define TE15_AREGISTRY_H

#include <stdbool.h>
#include <stddef.h>

#include "aresult.h"
#include "dag.h"

#if 0
typedef struct { const char *key; const char *unit; const char *desc; } ArParamDesc;
#endif

typedef struct {
    const char *key;
    const char *unit;   /* NULL for dimensionless / text                    */
    const char *desc;
} ArParamDesc;

typedef struct {
    const char *key;
    ArSetKind   kind;
    double      number;  /* defaults for NUMBER / INT                        */
    int         integer;
    bool        boolean;
    const char *text;
    const char *desc;
    /* widget range/step for the settings UI (0 = unbounded) */
    double      min, max, step;
} ArSettingDesc;

typedef struct {
    const char       *id;      /* "classic", "intraburst", ...            */
    const char       *title;   /* human title                             */
    int               dag_node;
    const ArParamDesc   *params;
    size_t             n_params;
    const ArSettingDesc *settings;
    size_t             n_settings;
} ArAnalysisDesc;

size_t aregistry_count(void);
const ArAnalysisDesc *aregistry_at(size_t i);
const ArAnalysisDesc *aregistry_find(const char *id);

/* Apply the schema's default settings to an envelope. */
void aregistry_apply_settings(AnalysisResult *r);

#endif /* TE15_AREGISTRY_H */
