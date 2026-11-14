/*
 * about_image.h — the About-screen artwork, embedded in the executable.
 *
 * The image bytes are generated at build time from media/about.gif by
 * tools/bin2c.c (see the Makefile), so the About screen never reads the
 * image from disk at run time.
 */
#ifndef TE15_ABOUT_IMAGE_H
#define TE15_ABOUT_IMAGE_H

#include <stddef.h>

extern const unsigned char te15_about_image[];
extern const size_t te15_about_image_len;

#endif /* TE15_ABOUT_IMAGE_H */
