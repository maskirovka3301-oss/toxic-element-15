/*
 * aconfig.c — per-analysis settings overrides (see aconfig.h).
 */
#include "aconfig.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aregistry.h"

#define ACONF_MAX 128
#define ACONF_KEY  96

typedef struct {
    char analysis[ACONF_KEY];
    char key[ACONF_KEY];
    ArSetKind kind;
    double number;
    int integer;
    bool boolean;
    char text[128];
} AConf;

static AConf overrides[ACONF_MAX];
static size_t n_overrides;

static AConf *find(const char *analysis, const char *key)
{
    if (analysis == NULL || key == NULL) return NULL;
    for (size_t i = 0; i < n_overrides; i++) {
        if (strcmp(overrides[i].analysis, analysis) == 0 &&
            strcmp(overrides[i].key, key) == 0) {
            return &overrides[i];
        }
    }
    return NULL;
}

static AConf *slot(const char *analysis, const char *key, ArSetKind kind)
{
    AConf *c = find(analysis, key);

    if (c != NULL) {
        c->kind = kind;
        return c;
    }
    if (n_overrides >= ACONF_MAX) return NULL;
    c = &overrides[n_overrides++];
    memset(c, 0, sizeof(*c));
    snprintf(c->analysis, sizeof(c->analysis), "%s", analysis);
    snprintf(c->key, sizeof(c->key), "%s", key);
    c->kind = kind;
    return c;
}

void aconfig_reset(void)
{
    memset(overrides, 0, sizeof(overrides));
    n_overrides = 0;
}

void aconfig_set_num(const char *analysis, const char *key, double value)
{
    AConf *c = slot(analysis, key, AR_SET_NUMBER);

    if (c != NULL) c->number = value;
}

void aconfig_set_int(const char *analysis, const char *key, int value)
{
    AConf *c = slot(analysis, key, AR_SET_INT);

    if (c != NULL) c->integer = value;
}

void aconfig_set_bool(const char *analysis, const char *key, bool value)
{
    AConf *c = slot(analysis, key, AR_SET_BOOL);

    if (c != NULL) c->boolean = value;
}

void aconfig_set_text(const char *analysis, const char *key, const char *value)
{
    AConf *c = slot(analysis, key, AR_SET_TEXT);

    if (c != NULL) snprintf(c->text, sizeof(c->text), "%s",
                            value != NULL ? value : "");
}

bool aconfig_overridden(const char *analysis, const char *key)
{
    return find(analysis, key) != NULL;
}

double aconfig_get_num(const char *analysis, const char *key, double dflt)
{
    const AConf *c = find(analysis, key);

    return (c != NULL && c->kind == AR_SET_NUMBER) ? c->number : dflt;
}

int aconfig_get_int(const char *analysis, const char *key, int dflt)
{
    const AConf *c = find(analysis, key);

    return (c != NULL && c->kind == AR_SET_INT) ? c->integer : dflt;
}

bool aconfig_get_bool(const char *analysis, const char *key, bool dflt)
{
    const AConf *c = find(analysis, key);

    return (c != NULL && c->kind == AR_SET_BOOL) ? c->boolean : dflt;
}

const char *aconfig_get_text(const char *analysis, const char *key,
                             const char *dflt)
{
    const AConf *c = find(analysis, key);

    return (c != NULL && c->kind == AR_SET_TEXT && c->text[0] != '\0')
               ? c->text : dflt;
}

size_t aconfig_num_overrides(void)
{
    return n_overrides;
}

int aconfig_save(const char *path)
{
    FILE *f = fopen(path, "w");
    int n = 0;

    if (f == NULL) return -1;
    fprintf(f, "# ToxicElement15 analysis settings (analysis.key=value)\n");
    for (size_t i = 0; i < n_overrides; i++) {
        const AConf *c = &overrides[i];

        switch (c->kind) {
        case AR_SET_NUMBER: fprintf(f, "%s.%s=%.10g\n", c->analysis, c->key,
                                    c->number); break;
        case AR_SET_INT:    fprintf(f, "%s.%s=%d\n", c->analysis, c->key,
                                    c->integer); break;
        case AR_SET_BOOL:   fprintf(f, "%s.%s=%d\n", c->analysis, c->key,
                                    c->boolean ? 1 : 0); break;
        case AR_SET_TEXT:   fprintf(f, "%s.%s=%s\n", c->analysis, c->key,
                                    c->text); break;
        }
        n++;
    }
    fclose(f);
    return n;
}

int aconfig_load(const char *path)
{
    FILE *f = fopen(path, "r");
    char line[512];
    int n = 0;

    if (f == NULL) return 0;
    aconfig_reset();
    while (fgets(line, sizeof(line), f) != NULL) {
        char *eq = strchr(line, '=');
        char *dot = strchr(line, '.');
        char *nl;
        const ArAnalysisDesc *desc;
        const ArSettingDesc *set = NULL;

        if (line[0] == '#' || eq == NULL || dot == NULL || dot > eq) continue;
        nl = strpbrk(line, "\r\n");
        if (nl != NULL) *nl = '\0';
        *dot = '\0';
        *eq = '\0';
        desc = aregistry_find(line);
        if (desc == NULL) continue;
        for (size_t i = 0; i < desc->n_settings; i++) {
            if (strcmp(desc->settings[i].key, dot + 1) == 0) {
                set = &desc->settings[i];
                break;
            }
        }
        if (set == NULL) continue;
        switch (set->kind) {
        case AR_SET_NUMBER:
            aconfig_set_num(line, set->key, strtod(eq + 1, NULL));
            break;
        case AR_SET_INT:
            aconfig_set_int(line, set->key, (int)strtod(eq + 1, NULL));
            break;
        case AR_SET_BOOL:
            aconfig_set_bool(line, set->key,
                             strtod(eq + 1, NULL) != 0.0);
            break;
        case AR_SET_TEXT:
            aconfig_set_text(line, set->key, eq + 1);
            break;
        }
        n++;
    }
    fclose(f);
    return n;
}
