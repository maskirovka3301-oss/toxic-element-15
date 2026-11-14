/*
 * debug.h — console verbosity control.
 *
 * The analysis pipeline used to print its intermediate sections, and the
 * waterfall cache printed every loaded window, on every run. That is noise
 * in normal use, so all informational output goes through TE15_LOG(), which
 * is silent unless the TE15_DEBUG environment variable is set:
 *
 *     TE15_DEBUG=1 ./build/te15 recording.raw
 *
 * Genuine errors keep using fprintf(stderr, ...) and are never silenced.
 */
#ifndef TE15_DEBUG_H
#define TE15_DEBUG_H

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

static inline bool te15_verbose(void)
{
    static int cached = -1;

    if (cached < 0) cached = (getenv("TE15_DEBUG") != NULL) ? 1 : 0;
    return cached != 0;
}

/* Informational console output, shown only with TE15_DEBUG set. */
#define TE15_LOG(...)                          \
    do {                                       \
        if (te15_verbose()) printf(__VA_ARGS__); \
    } while (0)

#endif /* TE15_DEBUG_H */
