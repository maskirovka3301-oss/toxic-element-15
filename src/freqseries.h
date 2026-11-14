/*
 * freqseries.h — frequency-measurement sessions for the Frequency Counter.
 *
 * A session is one text file with one frequency measurement per line (Hz,
 * as produced by a frequency counter). Each session carries the user
 * metadata shown in the session list: a label, a location (latitude and
 * longitude), a date/time and free-form notes.
 */
#ifndef TE15_FREQSERIES_H
#define TE15_FREQSERIES_H

#include <stdbool.h>
#include <stddef.h>

typedef struct {
    char   *label;       /* display name (defaults to the file stem)      */
    char   *path;        /* the source file                               */
    char   *datetime;    /* free-form date/time shown in the list         */
    char   *notes;       /* free-form notes                               */
    double  lat;         /* latitude  (valid when has_location)           */
    double  lon;         /* longitude                                     */
    bool    has_location;
    double *freqs;       /* measurements in Hz, in file order             */
    size_t  count;
    size_t  cap;
} FreqSession;

typedef struct {
    FreqSession *items;
    size_t count;
    size_t cap;
} FreqSeriesList;

void fsl_init(FreqSeriesList *list);
void fsl_clear(FreqSeriesList *list);

/* Load one measurement file (one frequency in Hz per line; empty lines and
 * lines starting with '#' are ignored, a trailing '#' comment is allowed).
 * The new session is appended to the list. Returns false with `err` filled
 * when the file cannot be read or contains no valid measurement. */
bool fsl_load_file(FreqSeriesList *list, const char *path, char *err,
                   size_t err_len);

/* Deep copy (used to hand a stable snapshot to the analysis worker). */
bool fsl_clone(const FreqSeriesList *src, FreqSeriesList *dst);

/* Remove one session (clamped: out-of-range indices are ignored). */
void fsl_remove(FreqSeriesList *list, size_t index);

/* Set session metadata (NULL/empty clears the field). */
void fsl_set_label(FreqSession *s, const char *label);
void fsl_set_datetime(FreqSession *s, const char *datetime);
void fsl_set_notes(FreqSession *s, const char *notes);
void fsl_set_location(FreqSession *s, bool has_location, double lat,
                      double lon);

/* Total number of measurements over all sessions. */
size_t fsl_total_points(const FreqSeriesList *list);

#endif /* TE15_FREQSERIES_H */
