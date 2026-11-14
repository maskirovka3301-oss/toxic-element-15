/*
 * aconfig.h — runtime overrides of the per-analysis settings.
 *
 * The analysis registry (aregistry.h) declares each analysis' tunable
 * settings with their schema defaults; this module stores the user's
 * overrides, persists them (one "analysis.key=value" line per setting) and
 * answers typed queries. Analyses read their settings here, so a change
 * applies to the next analysis run without touching any algorithm.
 */
#ifndef TE15_ACONFIG_H
#define TE15_ACONFIG_H

#include <stdbool.h>
#include <stddef.h>

/* Number of overrides the table can hold (the registry's total). */
void aconfig_reset(void);

/* Load/save a simple key=value configuration file. Missing file or parse
 * errors leave the defaults in place (returns the number of settings read
 * or written, or -1 when the file cannot be opened for writing). */
int  aconfig_load(const char *path);
int  aconfig_save(const char *path);

/* Override one setting. Values are validated against the schema by the
 * caller (the settings page clamps to the widget range). */
void aconfig_set_num(const char *analysis, const char *key, double value);
void aconfig_set_int(const char *analysis, const char *key, int value);
void aconfig_set_bool(const char *analysis, const char *key, bool value);
void aconfig_set_text(const char *analysis, const char *key, const char *value);

/* Is the setting overridden (non-default)? */
bool aconfig_overridden(const char *analysis, const char *key);

/* Typed queries; the schema's default is used when there is no override. */
double aconfig_get_num(const char *analysis, const char *key, double dflt);
int    aconfig_get_int(const char *analysis, const char *key, int dflt);
bool   aconfig_get_bool(const char *analysis, const char *key, bool dflt);
const char *aconfig_get_text(const char *analysis, const char *key,
                             const char *dflt);

/* Number of overrides currently set (for the UI's "modified" hint). */
size_t aconfig_num_overrides(void);

#endif /* TE15_ACONFIG_H */
