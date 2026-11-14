/*
 * ssca.c — Strip Spectral Correlation Analyzer (Spooner / Roberts et al.).
 *
 * Exhaustive blind cycle-frequency estimation. The selection is copied
 * and mixdowned to the region centre, then channelized by short Hann-
 * windowed FFTs of N' = 64 points hopped by L = 16. Each channelizer
 * demodulate X_T(r, f_k) is multiplied by conj(x(rL)) (the signal at the
 * same time origin) and the P-length product is Fourier transformed with
 * a long "strip" FFT of length P (one hop sequence per strip). Each
 * strip yields point estimates of the spectral correlation along a line
 * parallel to alpha = -2f, at cycle frequencies
 *
 *     alpha = k/N' + q/(P*L)   (cycles/sample),  k in [0, N'/2), q in [0, P).
 *
 * Normalising by the (coarsely indexed) power spectrum gives the
 * spectral coherence; the cyclic domain profile (max coherence per q)
 * is thresholded at the CFAR floor max(sqrt(-2 ln(Pfa)/P), 0.03) and its
 * local maxima become the significant cycle-frequency list.
 */
#include "ssca.h"
#include "debug.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsp.h"
#include "metrics.h"

#define SSCA_CHANNEL_FFT 64                 /* N' channelizer FFT size   */
#define SSCA_HOP         (SSCA_CHANNEL_FFT / 4) /* L = N'/4 hop          */
#define SSCA_STRIP_FFT   65536               /* nominal strip FFT length
                                             * (the actual strip FFT is
                                             * the P-length hop sequence
                                             * itself: alpha = k/N' +
                                             * q/(P*L)) */
#define SSCA_PFA         1e-4               /* CFAR false-alarm floor    */

/* qsort comparator for detected CDP peaks: descending coherence,
 * ascending cycle frequency on ties. */
static int ssca_peak_cmp(const void *a, const void *b)
{
    const double *pa = (const double *)a;
    const double *pb = (const double *)b;
    if (pb[1] > pa[1]) return 1;
    if (pb[1] < pa[1]) return -1;
    if (pa[0] < pb[0]) return -1;
    if (pa[0] > pb[0]) return 1;
    return 0;
}

void ssca_clear(SscaResult *r)
{
    if (r == NULL) return;
    free(r->scf);
    free(r->coh);
    free(r->cdp);
    memset(r, 0, sizeof(*r));
}

bool ssca_analyze(const IqSource *src, const Region *region, SscaResult *out)
{
    ssca_clear(out);
    if (src == NULL || region == NULL) return false;

    Region r = *region;
    if (!region_normalize(&r, src)) return false;

    size_t s0 = (size_t)(r.time_min * src->sample_rate);
    size_t s1 = (size_t)(r.time_max * src->sample_rate);
    if (s1 > src->total_samples) s1 = src->total_samples;
    if (s0 >= s1) return false;
    size_t n_work = s1 - s0;
    if (n_work > TE15_MAX_ANALYSIS_SAMPLES) n_work = TE15_MAX_ANALYSIS_SAMPLES;

    /* private copy of the selected segment, mixdowned to the centre */
    float *seg = malloc(2 * n_work * sizeof(float));
    if (seg == NULL) return false;
    iq_source_read_complex(src, s0, seg, n_work);
    double offset = ((r.freq_min + r.freq_max) / 2.0) - src->center_freq;
    metrics_mixdown(seg, n_work, src->sample_rate, -offset, seg);

    const int np = SSCA_CHANNEL_FFT;      /* N' */
    const int hop = SSCA_HOP;             /* L  */
    const int num_strips = np / 2;        /* channelizer bins kept: 32 */
    if (n_work < (size_t)np) {
        free(seg);
        return false;
    }
    /* demodulate-sequence length (channelizer hops). The strip FFT
     * length must stay within what FFTW can plan efficiently, so long
     * selections use only the first SSCA_STRIP_FFT hops (the cycle-
     * frequency resolution is fs/65536, ample for symbol rates). */
    size_t P = (n_work - (size_t)np) / (size_t)hop + 1;
    if (P < 4) {
        free(seg);
        return false;
    }
    size_t P_used = P;
    if (P_used > SSCA_STRIP_FFT) P_used = SSCA_STRIP_FFT;
    const int strip_n = (int)P_used;

    /* Hann window for the channelizer */
    float *win = malloc((size_t)np * sizeof(float));
    if (win == NULL) {
        free(seg);
        return false;
    }
    dsp_hann(win, (size_t)np);

    /* Channelizer: X_T(r, f_k) for r in [0,P), k in [0,num_strips).
     * Per-strip power sums are accumulated for the coherence
     * normalisation (unnormalised; the 1/P cancels in the ratio). */
    fftwf_complex *XT =
        fftwf_malloc(P_used * (size_t)num_strips * sizeof(fftwf_complex));
    fftwf_complex *row = fftwf_malloc((size_t)np * sizeof(fftwf_complex));
    FftPlan *plan64 = fft_plan_create(np);
    if (XT == NULL || row == NULL || plan64 == NULL) {
        fftwf_free(XT);
        fftwf_free(row);
        fft_plan_destroy(plan64);
        free(win);
        free(seg);
        return false;
    }

    double psd[SSCA_CHANNEL_FFT / 2];
    memset(psd, 0, sizeof(psd));
    double wsum2 = 0.0;                 /* window power gain (for the
                                         * unbiased PSD estimate) */
    for (int i = 0; i < np; i++) {
        wsum2 += (double)win[i] * (double)win[i];
    }
    for (size_t rr = 0; rr < P_used; rr++) {
        fft_plan_transform(plan64, seg + 2 * rr * (size_t)hop, win, row);
        for (int k = 0; k < num_strips; k++) {
            double re = (double)row[k][0];
            double im = (double)row[k][1];
            XT[rr * (size_t)num_strips + (size_t)k][0] = row[k][0];
            XT[rr * (size_t)num_strips + (size_t)k][1] = row[k][1];
            psd[k] += re * re + im * im;
        }
    }
    fft_plan_destroy(plan64);
    fftwf_free(row);
    free(win);

    /* Strip FFTs: FFT length = P, so num_cf = P and the P bins map to
     * alpha = k/N' + q/(P*L) (the sanctioned "simplest" option). */
    float *scf = malloc((size_t)num_strips * strip_n * sizeof(float));
    float *coh = malloc((size_t)num_strips * strip_n * sizeof(float));
    double *cdp = calloc((size_t)strip_n, sizeof(double));
    int *argk = malloc((size_t)strip_n * sizeof(int));
    float *prod = malloc(2 * strip_n * sizeof(float));
    fftwf_complex *S = fftwf_malloc((size_t)strip_n * sizeof(fftwf_complex));
    FftPlan *planP = fft_plan_create(strip_n);
    if (scf == NULL || coh == NULL || cdp == NULL || argk == NULL ||
        prod == NULL || S == NULL || planP == NULL) {
        free(scf);
        free(coh);
        free(cdp);
        free(argk);
        free(prod);
        fftwf_free(S);
        fft_plan_destroy(planP);
        fftwf_free(XT);
        free(seg);
        return false;
    }

    for (int k = 0; k < num_strips; k++) {
        const double f_k = (double)k / (2.0 * (double)np); /* k/(2N') */
        const double k_over_np = (double)k / (double)np;   /* k/N'  */

        for (size_t rr = 0; rr < P_used; rr++) {
            const fftwf_complex *xt =
                &XT[rr * (size_t)num_strips + (size_t)k];
            float xi = seg[2 * rr * (size_t)hop];
            float xq = seg[2 * rr * (size_t)hop + 1];
            /* X_T(r, f_k) * conj(x(rL)) */
            prod[2 * rr]     = (*xt)[0] * xi + (*xt)[1] * xq;
            prod[2 * rr + 1] = (*xt)[1] * xi - (*xt)[0] * xq;
        }

        fft_plan_transform(planP, prod, NULL, S);

        const double p_inv = 1.0 / (double)strip_n;
        for (size_t q = 0; q < (size_t)strip_n; q++) {
            double mre = (double)S[q][0];
            double mim = (double)S[q][1];
            double mag = sqrt(mre * mre + mim * mim);
            scf[(size_t)k * strip_n + q] = (float)(mag * p_inv);

            /* Coherence via Spooner's coarse-PSD normalisation:
             * alpha = k/N' + q/(P*L), f = k/(2N') - alpha/2, and PSD(f')
             * is the strip PSD at the strip nearest to f'. The q term is
             * divided by the hop L: the product sequence lives at the
             * hop rate, so one strip-FFT bin spans 1/(P*L) cycles per
             * sample. */
            double alpha = k_over_np + (double)q * p_inv / (double)hop;
            double f = f_k - 0.5 * alpha;
            double f1 = f + 0.5 * alpha; /* = k/(2N') */
            double f2 = f - 0.5 * alpha; /* = -k/(2N') - q/(P*L) */
            int k1 = (int)lround(f1 * 2.0 * (double)np);
            int k2 = (int)lround(f2 * 2.0 * (double)np);
            if (k1 < 0) {
                k1 = 0;
            } else if (k1 >= num_strips) {
                k1 = num_strips - 1;
            }
            if (k2 < 0) {
                k2 = 0;
            } else if (k2 >= num_strips) {
                k2 = num_strips - 1;
            }
            double denom = sqrt(psd[k1] * psd[k2]) * p_inv / wsum2;
            double c = denom > 0.0 ? mag * p_inv / denom : 0.0;
            if (c > 1.0) c = 1.0;
            coh[(size_t)k * strip_n + q] = (float)c;
            if (c > cdp[q]) {
                cdp[q] = c;
                argk[q] = k;
            }
        }
    }

    fft_plan_destroy(planP);
    fftwf_free(S);
    free(prod);
    fftwf_free(XT);
    free(seg);

    out->channel_fft = np;
    out->strip_fft = strip_n;
    out->num_strips = num_strips;
    out->num_cf = strip_n;
    out->scf = scf;
    out->coh = coh;
    out->cdp = cdp;
    out->cf_threshold = sqrt(-2.0 * log(SSCA_PFA) / (double)P_used);
    if (out->cf_threshold < 0.03) out->cf_threshold = 0.03;

    /* CDP peaks: local maxima above the statistical floor, separated by
     * at least 2 bins; alpha reported at the argmax strip k. The
     * survivors are sorted by descending coherence before capping at
     * SSCA_MAX_CF, so weak noise crossings cannot crowd out the strong
     * features (the CDP is scanned in q order, and a pure q-order cap
     * drops genuine late-q features such as a pulse-train PRF). */
    {
        double (*found)[2] = malloc(((size_t)strip_n / 2 + 2) * sizeof(*found));
        if (found == NULL) {
            out->cf_count = 0;
        } else {
            size_t count = 0;
            int last_q = -3;
            /* Skip the strip-FFT DC and Nyquist neighbourhoods: the
             * q ~ 0 bins carry the demodulate product's DC (cycle
             * leakage, coherence ~1 regardless of signal) and q near
             * strip_n aliases the same wall - they produced spurious
             * "significant" cycle frequencies. */
            for (size_t q = 3; q + 3 < (size_t)strip_n; q++) {
                if (cdp[q] <= out->cf_threshold) continue;
                if (q > 0 && cdp[q] <= cdp[q - 1]) continue;
                if (q + 1 < (size_t)strip_n && cdp[q] < cdp[q + 1]) continue;
                if ((int)q - last_q < 2) continue;
                /* Report the CARRIER-INDEPENDENT modulation rate
                 * beta = q/(P*L) (cycles/sample), not the full
                 * alpha = 2f_k + beta: the k term is the conjugate
                 * carrier offset (alpha = 2f features) of this
                 * channel, so mixing it in made the reported "cycle
                 * frequency" shift by k*fs/N' (250 kHz combs) and hide
                 * the modulation rate. The alpha = 2f conjugate
                 * features are characterised by the k axis and the
                 * note states so. */
                {
                    /* signed beta: q near P is the negative image of the
                     * same feature (beta is ambiguous modulo fs/L) */
                    double beta = (double)q / (double)strip_n /
                                  (double)hop * src->sample_rate;

                    if (q > (size_t)strip_n / 2) {
                        beta -= src->sample_rate / (double)hop;
                    }
                    /* dedupe +/- images of the same modulation rate */
                    {
                        bool dup = false;
                        double bin = src->sample_rate /
                                     ((double)strip_n * (double)hop);

                        for (size_t fidx = 0; fidx < count; fidx++) {
                            if (fabs(fabs(found[fidx][0]) - fabs(beta)) <=
                                bin) {
                                dup = true;
                                break;
                            }
                        }
                        if (dup) {
                            continue;
                        }
                    }
                    found[count][0] = beta;
                }
                found[count][1] = cdp[q];
                last_q = (int)q;
                count++;
            }
            qsort(found, count, sizeof(*found), ssca_peak_cmp);
            size_t keep = count < SSCA_MAX_CF ? count : SSCA_MAX_CF;
            for (size_t i = 0; i < keep; i++) {
                out->cf_list[i][0] = found[i][0];
                out->cf_list[i][1] = found[i][1];
            }
            out->cf_count = keep;
            free(found);
        }
    }
    free(argk);

    snprintf(out->note, sizeof(out->note),
             "SSCA: N'=64 N=%d strips=32 | %zu significant modulation-rate "
             "cycle frequencies (carrier-independent, beta = q/(P*L); "
             "conjugate alpha=2f features live on the k axis)",
             (int)P, out->cf_count);

    TE15_LOG("\nSSCA (STRIP SPECTRAL CORRELATION) ANALYSIS\n"
           "N'=%d L=%d strip FFT N=%d strips=%d | threshold %.3f\n",
           np, hop, (int)P, num_strips, out->cf_threshold);
    if (out->cf_count > 0) {
        TE15_LOG("Significant cycle frequencies:\n");
        for (size_t i = 0; i < out->cf_count; i++) {
            TE15_LOG("  %2zu. %12.1f Hz (coherence %.3f)\n", i + 1,
                   out->cf_list[i][0], out->cf_list[i][1]);
        }
    } else {
        TE15_LOG("No significant cycle frequencies detected\n");
    }

    return true;
}
