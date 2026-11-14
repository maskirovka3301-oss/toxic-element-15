/*
 * fcdataset.h — measurement-dataset model for the Frequency Counter tab.
 *
 * The file format is the JSON schema in schema/fc_dataset.json:
 *
 *   dataset          { dataset_id, source_file, title, notes[],
 *                      sessions[], unassigned_measurements[] }
 *   session          { session_id (string|integer), rounds[] }
 *   round            { round_id (string|integer), location, date,
 *                      coordinates {latitude, longitude}, closest_towers_enb_ids,
 *                      context, subjects_listed[], subjects[] }
 *   subject          { name, measurements[] }        (numbers in Hz)
 *
 * The model is loaded and saved through json-c, with validation errors
 * reported against the JSON path of the offending value.
 */
#ifndef TE15_FCDATASET_H
#define TE15_FCDATASET_H

#include <stdbool.h>
#include <stddef.h>

/* An id is either a string or an integer in the schema. */
typedef struct {
    char       text[64];   /* rendered form (always valid)          */
    bool       numeric;    /* true when the source was an integer   */
    long long  number;     /* valid when numeric                    */
} FcId;

typedef struct {
    char   *name;
    double *measurements;  /* Hz, in file order                     */
    size_t  count;
    size_t  cap;
} FcSubject;

typedef struct {
    FcId        round_id;
    char       *location;        /* NULL when absent/null            */
    char       *date;            /* NULL when absent/null            */
    bool        has_coordinates; /* coordinates != null               */
    double      latitude;
    double      longitude;
    bool        towers_null;     /* closest_towers_enb_ids was null   */
    char      **towers;          /* rendered tower ids                */
    size_t      n_towers;
    char       *context;         /* NULL when absent/null            */
    char      **subjects_listed;
    size_t      n_listed;
    FcSubject  *subjects;
    size_t      n_subjects;
} FcRound;

typedef struct {
    FcId     session_id;
    FcRound *rounds;
    size_t   n_rounds;
} FcSession;

typedef struct {
    char       *dataset_id;
    char       *source_file;
    char       *title;
    char      **notes;
    size_t      n_notes;
    FcSession  *sessions;
    size_t      n_sessions;
    double     *unassigned;      /* Hz                               */
    size_t      n_unassigned;
} FcDataset;

typedef struct {
    FcDataset *items;
    size_t     count;
    size_t     cap;
} FcDatasetList;

void fcd_list_init(FcDatasetList *list);
void fcd_list_clear(FcDatasetList *list);

/* Load a dataset JSON file and append it to the list. Returns false with a
 * schema-validation message in `err` (including the JSON path). */
bool fcd_load_file(FcDatasetList *list, const char *path, char *err,
                   size_t err_len);

/* Write one dataset as JSON conforming to schema/fc_dataset.json. */
bool fcd_save_file(const FcDataset *dataset, const char *path, char *err,
                   size_t err_len);

/* Deep copy (worker snapshots). */
bool fcd_list_clone(const FcDatasetList *src, FcDatasetList *dst);

void fcd_list_remove(FcDatasetList *list, size_t index);

/* Totals over one dataset / the whole list. */
size_t fcd_count_rounds(const FcDataset *dataset);
size_t fcd_count_subjects(const FcDataset *dataset);
size_t fcd_count_measurements(const FcDataset *dataset);
size_t fcd_list_count_sessions(const FcDatasetList *list);
size_t fcd_list_count_rounds(const FcDatasetList *list);
size_t fcd_list_count_subjects(const FcDatasetList *list);
size_t fcd_list_count_measurements(const FcDatasetList *list);

/* Measurement statistics (Hz); returns false when there are no values. */
bool fc_stats(const double *values, size_t n, double *min_hz, double *max_hz,
              double *mean_hz);

#endif /* TE15_FCDATASET_H */
