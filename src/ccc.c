/*
 * ccc.c — modulation recognition using cyclic temporal cumulants
 * (see header). Feature estimators (Dobre / Spooner), normalised by
 * P^2 with P = <|x|^2>:
 *   C40(4fc) ~ M40(4fc) - 3 M20(2fc)^2
 *   C41(2fc) ~ M41(2fc) - 3 M20(2fc) M21(0)
 *   C42(2fc) ~ M42(2fc) - |M20(2fc)|^2 - 2 M21(0)^2
 * where Mpq(alpha) = < x^p (x*)^(q-p) e^{-i 2 pi alpha n} > over the
 * samples. Hierarchical verdicts follow the constellation separations.
 */
#include "ccc.h"
#include "debug.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "metrics.h"
#include "region.h"

void ccc_clear(CccResult *r)
{
    if (r != NULL) memset(r, 0, sizeof(*r));
}

bool ccc_analyze(const IqSource *src, const Region *region, CccResult *out)
{
    ccc_clear(out);
    if (src == NULL || region == NULL) return false;

    Region r = *region;
    if (!region_normalize(&r, src)) return false;

    size_t s0 = (size_t)(r.time_min * src->sample_rate);
    size_t s1 = (size_t)(r.time_max * src->sample_rate);
    if (s1 > src->total_samples) s1 = src->total_samples;
    if (s0 >= s1) return false;
    size_t n = s1 - s0;
    if (n < 2048) return false;
    if (n > TE15_MAX_ANALYSIS_SAMPLES) n = TE15_MAX_ANALYSIS_SAMPLES;

    float *seg = malloc(2 * n * sizeof(float));
    if (seg == NULL) return false;
    iq_source_read_complex(src, s0, seg, n);
    /* keep the carrier: the cyclic features are evaluated AT 2fc/4fc,
     * so no mixdown is applied (that is the point of cyclic cumulants) */

    const double fc = (r.freq_min + r.freq_max) / 2.0;
    const double a2 = 2.0 * fc / src->sample_rate;  /* 2fc, cycles/sample */
    const double a4 = 4.0 * fc / src->sample_rate;  /* 4fc, cycles/sample */

    double M21_0 = 0.0;          /* <|x|^2> */
    double M20_2r = 0.0, M20_2i = 0.0; /* M20 at 2fc */
    double M40_4r = 0.0, M40_4i = 0.0; /* M40 at 4fc */
    double M41_2r = 0.0, M41_2i = 0.0; /* M41 at 2fc */
    double M42_2r = 0.0, M42_2i = 0.0; /* M42 at 2fc */
    for (size_t i = 0; i < n; i++) {
        double x = seg[2 * i], y = seg[2 * i + 1];
        double ab2 = x * x + y * y;
        double ph2 = 2.0 * M_PI * a2 * (double)i;
        double ph4 = 2.0 * M_PI * a4 * (double)i;
        double c2 = cos(ph2), s2 = sin(ph2);
        double c4 = cos(ph4), s4 = sin(ph4);

        double x2r = x * x - y * y, x2i = 2.0 * x * y;
        double x4r = x2r * x2r - x2i * x2i, x4i = 2.0 * x2r * x2i;
        /* x^3 x* = x^2 |x|^2 */
        double x3c_r = x2r * ab2, x3c_i = -x2i * ab2;
        /* x^2 (x*)^2 = |x|^4 */
        double m42 = ab2 * ab2;

        M21_0 += ab2;
        M20_2r += x2r * c2 + x2i * s2;
        M20_2i += x2i * c2 - x2r * s2;
        M40_4r += x4r * c4 + x4i * s4;
        M40_4i += x4i * c4 - x4r * s4;
        M41_2r += x3c_r * c2 + x3c_i * s2;
        M41_2i += x3c_i * c2 - x3c_r * s2;
        M42_2r += m42 * c2;
        M42_2i += -m42 * s2;
    }
    M21_0 /= (double)n;
    M20_2r /= (double)n; M20_2i /= (double)n;
    M40_4r /= (double)n; M40_4i /= (double)n;
    M41_2r /= (double)n; M41_2i /= (double)n;
    M42_2r /= (double)n; M42_2i /= (double)n;

    /* moment -> cyclic cumulant */
    double C40r = M40_4r - 3.0 * (M20_2r * M20_2r - M20_2i * M20_2i);
    double C40i = M40_4i - 3.0 * (2.0 * M20_2r * M20_2i);
    double C41r = M41_2r - 3.0 * (M20_2r * M21_0);
    double C41i = M41_2i - 3.0 * (M20_2i * M21_0);
    double C42r = M42_2r - (M20_2r * M20_2r + M20_2i * M20_2i) -
                  2.0 * M21_0 * M21_0;
    double C42i = M42_2i;

    double P2 = M21_0 * M21_0;
    out->power = M21_0;
    out->carrier_hz = fc;
    out->samples = n;
    out->valid = true;
    if (P2 > 0.0) {
        out->n40 = sqrt(C40r * C40r + C40i * C40i) / P2;
        out->n41 = sqrt(C41r * C41r + C41i * C41i) / P2;
        out->n42 = sqrt(C42r * C42r + C42i * C42i) / P2;
    }

    /* hierarchical verdict (tolerances per Dobre/Spooner tables) */
    if (out->n40 > 1.5 && out->n41 > 1.5 && out->n42 > 1.5) {
        snprintf(out->verdict, sizeof(out->verdict), "BPSK-like");
    } else if (out->n40 > 0.4 && out->n40 <= 1.5 && out->n41 < 0.6 &&
               out->n42 > 0.4) {
        snprintf(out->verdict, sizeof(out->verdict), "QPSK-like");
    } else if (out->n40 < 0.4 && out->n41 < 0.4 && out->n42 > 1.5) {
        /* C42 = 2 P^2 with C40/C41 absent is the signature of a REAL
         * (one-dimensional) amplitude modulation with a large amplitude
         * variance (E|x|^4 = 4 (E|x|^2)^2, i.e. a ~25% duty burst train):
         * OOK/ASK, not a circular PSK/QAM constellation (those give
         * C42 = 0). */
        snprintf(out->verdict, sizeof(out->verdict),
                 "OOK / ASK-like burst emission (C42 = 2 P^2, C40/C41 "
                 "absent)");
    } else if (out->n40 < 0.4 && out->n41 < 0.4 && out->n42 > 0.4) {
        snprintf(out->verdict, sizeof(out->verdict),
                 "circular PSK/QAM-like (C40 = C41 = 0, weak C42)");
    } else if (out->n40 > 0.4 && out->n40 <= 1.5 && out->n41 < 0.6 &&
               out->n42 < 1.2) {
        snprintf(out->verdict, sizeof(out->verdict), "16QAM-like");
    } else {
        snprintf(out->verdict, sizeof(out->verdict),
                 "ambiguous (features %.2f/%.2f/%.2f)",
                 out->n40, out->n41, out->n42);
    }
    snprintf(out->note, sizeof(out->note),
             "Cyclic temporal cumulants (fc %.3f MHz, N %zu): "
             "|C40(4fc)|/P^2 = %.3f, |C41(2fc)|/P^2 = %.3f, "
             "|C42(2fc)|/P^2 = %.3f -> %s.",
             fc / 1e6, n, out->n40, out->n41, out->n42, out->verdict);

    TE15_LOG("\nCYCLIC CUMULANTS (AMR)\nfc = %.3f MHz, N = %zu\n"
           "|C40(4fc)|/P^2 = %.3f\n|C41(2fc)|/P^2 = %.3f\n"
           "|C42(2fc)|/P^2 = %.3f\n%s\n",
           fc / 1e6, n, out->n40, out->n41, out->n42, out->verdict);
    free(seg);
    return true;
}
