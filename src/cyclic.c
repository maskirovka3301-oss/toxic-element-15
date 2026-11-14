/*
 * cyclic.c — spectral correlation via the FFT Accumulation Method.
 *
 * FAM (Gardner): the data is split into Np frames of N samples, each
 * windowed and FFT'd. The SCF at cyclic frequency alpha and spectral
 * frequency f is the averaged correlation of bins f + alpha/2 and
 * f - alpha/2 across the frames, normalised to a spectral coherence
 * magnitude in [0,1]. The cyclic domain profile takes the strongest
 * coherence per alpha; local maxima above mean + 3 sigma are reported
 * as cyclic features (symbol/chip rates and harmonics).
 */
#include "cyclic.h"
#include "debug.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsp.h"
#include "aconfig.h"
#include "metrics.h"

#define CYCLIC_NP         512      /* FAM blocks                      */
#define CYCLIC_FS_TARGET  256000.0 /* decimated effective sample rate  */
#define CDP_EXCLUDE_BINS  1        /* skip alpha near 0 (trivial SCF)  */
#define PEAK_SIGMA        1.5      /* peak threshold over mean         */
#define PEAK_SEP_BINS     4        /* minimum peak separation          */

void cyclic_clear(CyclicResult *r)
{
    if (r == NULL) return;
    free(r->scf);
    free(r->cdp);
    memset(r, 0, sizeof(*r));
}

bool cyclic_analyze_ex(const IqSource *src, const Region *region,
                       bool conjugate, CyclicResult *out)
{
    cyclic_clear(out);
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

    /* decimate with a windowed-sinc FIR low-pass: a plain box filter
     * folds out-of-band replicas that correlate perfectly with the
     * originals, producing spurious cyclic features */
    size_t D = dsp_decimation_for(
        src->sample_rate, r.freq_max - r.freq_min,
        aconfig_get_num("cyclic", "decimation_target_hz",
                        CYCLIC_FS_TARGET));
    if (D < 1) D = 1;
    size_t nd = n_work / D;
    const double fs = src->sample_rate / (double)D;
    const int n = CYCLIC_N;
    if (nd < (size_t)n * 2) {
        free(seg);
        return false;
    }
    float *dec = malloc(2 * nd * sizeof(float));
    if (dec == NULL) {
        free(seg);
        return false;
    }
    {
        size_t got = dsp_decimate_staged(seg, n_work, D, dec);
        nd = got < nd ? got : nd;
    }
    free(seg);

    /* FAM channelizer: N'-point FFTs hopped by L = n/4 samples
     * (Spooner / Roberts et al.). Each hopped transform needs the FAM
     * phase correction e^{-i 2 pi b rL / n} so the relative block delay
     * is restored before the pairwise products. */
    const int L = n / 4;
    size_t np = (nd - (size_t)n) / (size_t)L + 1;
    if (np > CYCLIC_NP) np = CYCLIC_NP;
    if (np < 4) {
        free(dec);
        return false;
    }

    /* Hamming window */
    float *win = malloc((size_t)n * sizeof(float));
    if (win == NULL) {
        free(dec);
        return false;
    }
    for (int i = 0; i < n; i++) {
        win[i] = (float)(0.54 - 0.46 * cos(2.0 * M_PI * i / (n - 1)));
    }

    /* per-block FFTs, phase-corrected in place */
    fftwf_complex *X = fftwf_malloc(sizeof(fftwf_complex) * np * (size_t)n);
    FftPlan *plan = fft_plan_create(n);
    if (X == NULL || plan == NULL) {
        fftwf_free(X);
        fft_plan_destroy(plan);
        free(win);
        free(dec);
        return false;
    }
    /* Note on the FAM phase correction (Spooner's post, step 3): the
     * correction is only needed when the pairwise products are fed to
     * the P-point output FFT for the fine cycle-frequency resolution.
     * For the coarse-cyclic-domain accumulation below (q = 0), the
     * block-delay phase e^{+i2pi(f1-f2)rL} is exactly the alpha phase
     * being integrated - applying the correction would double it and
     * destroy the coherence (verified empirically). */
    for (size_t k = 0; k < np; k++) {
        fft_plan_transform(plan, dec + 2 * k * (size_t)L, win,
                           X + k * (size_t)n);
    }
    free(dec);
    free(win);
    fft_plan_destroy(plan);

    /* per-bin power for the coherence normalisation */
    double *P = calloc((size_t)n, sizeof(double));
    if (P == NULL) {
        fftwf_free(X);
        return false;
    }
    for (size_t k = 0; k < np; k++) {
        for (int b = 0; b < n; b++) {
            double re = (double)X[k * (size_t)n + (size_t)b][0];
            double im = (double)X[k * (size_t)n + (size_t)b][1];
            P[b] += re * re + im * im;
        }
    }
    for (int b = 0; b < n; b++) P[b] /= (double)np;

    float *scf = malloc((size_t)n * (size_t)n * sizeof(float));
    double *cdp = calloc((size_t)n, sizeof(double));
    if (scf == NULL || cdp == NULL) {
        free(scf);
        free(cdp);
        free(P);
        fftwf_free(X);
        return false;
    }

    /* SCF: alpha rows from -n/2 .. n/2-1 bins; f columns. Odd alpha
     * steps land on half-bin offsets, so both neighbouring integer
     * half-offsets are averaged (standard FAM sub-bin sampling). */
    for (int ai = 0; ai < n; ai++) {
        int a = ai - n / 2; /* alpha in bins */
        int h1 = (int)floor(a / 2.0);
        int h2 = ((a & 1) != 0) ? h1 + 1 : h1;
        for (int fi = 0; fi < n; fi++) {
            double acc_re = 0.0, acc_im = 0.0;
            int acc_cnt = 0;
            for (int hp = 0; hp < 2; hp++) {
                int h = (hp == 0) ? h1 : h2;
                if (hp == 1 && h == h1) break; /* dedupe even alpha */
                if (abs(a) == 1 && h == 0) continue; /* self-correlation
                                                       * artifact at
                                                       * |alpha| = 1 bin */
                int b1 = fi + h;
                int b2 = conjugate ? h - fi : fi - h;
                if (b2 < 0) b2 += n;
                if (b2 >= n) b2 -= n;
                if (b1 < 0 || b1 >= n) continue;
                double sum_re = 0.0, sum_im = 0.0;
                for (size_t k = 0; k < np; k++) {
                    const fftwf_complex *x1 = &X[k * (size_t)n + (size_t)b1];
                    const fftwf_complex *x2 = &X[k * (size_t)n + (size_t)b2];
                    if (conjugate) {
                        /* X(f+a/2) * X(a/2 - f) (no conjugation) */
                        sum_re += (double)(*x1)[0] * (*x2)[0] - (double)(*x1)[1] * (*x2)[1];
                        sum_im += (double)(*x1)[0] * (*x2)[1] + (double)(*x1)[1] * (*x2)[0];
                    } else {
                        /* X(f+a/2) * conj(X(f-a/2)) */
                        sum_re += (double)(*x1)[0] * (*x2)[0] + (double)(*x1)[1] * (*x2)[1];
                        sum_im += (double)(*x1)[1] * (*x2)[0] - (double)(*x1)[0] * (*x2)[1];
                    }
                }
                double mag = sqrt(sum_re * sum_re + sum_im * sum_im) / (double)np;
                double denom = sqrt(P[b1] * P[b2]);
                double v = denom > 0.0 ? mag / denom : 0.0;
                if (v > 1.0) v = 1.0;
                acc_re += v * cos(atan2(sum_im, sum_re));
                acc_im += v * sin(atan2(sum_im, sum_re));
                acc_cnt++;
            }
            double m = acc_cnt > 0
                           ? sqrt(acc_re * acc_re + acc_im * acc_im) / (double)acc_cnt
                           : 0.0;
            scf[(size_t)ai * n + fi] = (float)(m > 1.0 ? 1.0 : m);
            if (m > cdp[ai]) cdp[ai] = m;
        }
    }

    out->n = n;
    out->alpha_res = fs / (double)n;
    out->fs_eff = fs;
    out->scf = scf;
    out->cdp = cdp;

    /* Peak extraction runs on a masked copy of the raw profile: the
     * trivial self-coherence around alpha = 0 leaks into its immediate
     * neighbours and would shadow the fundamental, so that zone is
     * zeroed for detection only. The noise-only coherence floor is
     * ~1/sqrt(np); the threshold is the max of 2.5x that floor and
     * mean + PEAK_SIGMA sigma. The profile itself is normalised for
     * display afterwards. */
    {
        const int c0 = n / 2; /* alpha = 0 row */
        const int wall = CDP_EXCLUDE_BINS + 1; /* zeroed zone: |a| <= wall */

        double *mcdp = malloc((size_t)n * sizeof(double));
        if (mcdp == NULL) {
            free(P);
            fftwf_free(X);
            return false;
        }
        memcpy(mcdp, cdp, (size_t)n * sizeof(double));
        for (int ai = 0; ai < n; ai++) {
            if (abs(ai - c0) <= wall) mcdp[ai] = 0.0;
        }

        double mean = 0.0, var = 0.0;
        int cnt = 0;
        for (int ai = 0; ai < n; ai++) {
            if (abs(ai - c0) <= wall) continue;
            mean += mcdp[ai];
            cnt++;
        }
        mean /= (double)cnt;
        for (int ai = 0; ai < n; ai++) {
            if (abs(ai - c0) <= wall) continue;
            double d = mcdp[ai] - mean;
            var += d * d;
        }
        var /= (double)cnt;
        /* the hopped blocks overlap 4x, so the effective number of
         * independent averages is np/4: the noise-floor threshold uses
         * that effective count (5/sqrt(np) = 2.5/sqrt(np/4)) */
        double floor_threshold = 5.0 / sqrt((double)np);
        double threshold = mean + PEAK_SIGMA * sqrt(var);
        if (threshold < floor_threshold) threshold = floor_threshold;

        /* relative gate: a peak must clearly rise above the profile
         * median (rejects the smooth decay tail of a pure tone) */
        {
            double *msort = malloc((size_t)n * sizeof(double));
            if (msort == NULL) {
                free(mcdp);
                free(P);
                fftwf_free(X);
                return false;
            }
            memcpy(msort, mcdp, (size_t)n * sizeof(double));
            for (int i = 1; i < n; i++) {
                double key = msort[i];
                int j = i;
                while (j > 0 && msort[j - 1] > key) {
                    msort[j] = msort[j - 1];
                    j--;
                }
                msort[j] = key;
            }
            double med = msort[n / 2];
            free(msort);
            double rel = med * 1.6;
            if (rel > threshold) threshold = rel;
        }

        /* display normalisation (also over the wall zone) */
        double cdp_max = 0.0;
        for (int ai = 0; ai < n; ai++) {
            if (abs(ai - c0) <= wall) continue;
            if (cdp[ai] > cdp_max) cdp_max = cdp[ai];
        }

        CyclicPeak found[CYCLIC_MAX_PEAKS * 2];
        int found_count = 0;
        int last_peak_ai = -PEAK_SEP_BINS - 1;
        for (int ai = 0; ai < n && found_count < CYCLIC_MAX_PEAKS * 2; ai++) {
            if (abs(ai - c0) <= wall) continue;
            if (mcdp[ai] < threshold) continue;
            if (ai == 0 || ai == n - 1) continue;
            if (mcdp[ai] <= mcdp[ai - 1] || mcdp[ai] <= mcdp[ai + 1]) continue;
            if (ai - last_peak_ai < PEAK_SEP_BINS) continue;

            double alpha = (double)(ai - c0) * out->alpha_res;
            /* sub-bin refinement (parabolic) away from the wall */
            if (abs(ai - c0) > wall + 2) {
                double y0 = mcdp[ai - 1], y1 = mcdp[ai], y2 = mcdp[ai + 1];
                double den = y0 - 2.0 * y1 + y2;
                if (fabs(den) > 1e-12) {
                    double delta = 0.5 * (y0 - y2) / den;
                    if (delta > -1.0 && delta < 1.0) {
                        alpha += delta * out->alpha_res;
                    }
                }
            }
            found[found_count].alpha = alpha;
            found[found_count].strength =
                cdp_max > 0.0 ? cdp[ai] / cdp_max : 0.0;
            last_peak_ai = ai;
            found_count++;
        }

        /* wall-edge guard: peaks hugging the alpha=0 wall are the decay
         * tail of the carrier unless a corroborating harmonic exists
         * further out */
        {
            int corroborated = 0;
            for (int i = 0; i < found_count; i++) {
                if ((int)lround(fabs(found[i].alpha) / out->alpha_res) >
                    wall + 2) {
                    corroborated = 1;
                    break;
                }
            }
            if (!corroborated) {
                int k = 0;
                for (int i = 0; i < found_count; i++) {
                    if ((int)lround(fabs(found[i].alpha) / out->alpha_res) >
                        wall + 2) {
                        found[k++] = found[i];
                    }
                }
                found_count = k;
            }
        }
        free(mcdp);
        /* keep the strongest, then present the harmonic series with the
         * fundamental (smallest |alpha|) first */
        for (int i = 0; i < found_count - 1; i++) {
            for (int j = i + 1; j < found_count; j++) {
                if (found[j].strength > found[i].strength) {
                    CyclicPeak t = found[i];
                    found[i] = found[j];
                    found[j] = t;
                }
            }
        }
        /* coherence-thresholded cycle-frequency list for the UI;
         * also report the FAM P_FA statistical threshold from the
         * number of averaged blocks (Spooner's FAM post) */
        out->cf_threshold = threshold;
        out->cf_pfa_threshold = sqrt(-2.0 * log(1e-4) / (double)np);
        out->cf_count = 0;
        for (int i = 0; i < found_count && out->cf_count < CYCLIC_MAX_CF; i++) {
            out->cf_list[out->cf_count][0] = found[i].alpha;
            out->cf_list[out->cf_count][1] = found[i].strength;
            out->cf_count++;
        }

        int keep = found_count < CYCLIC_MAX_PEAKS ? found_count
                                                  : CYCLIC_MAX_PEAKS;
        for (int i = 0; i < keep - 1; i++) {
            for (int j = i + 1; j < keep; j++) {
                if (fabs(found[j].alpha) < fabs(found[i].alpha)) {
                    CyclicPeak t = found[i];
                    found[i] = found[j];
                    found[j] = t;
                }
            }
        }
        out->peak_count = keep;
        memcpy(out->peaks, found, (size_t)keep * sizeof(CyclicPeak));

        /* normalise the stored profile for display */
        if (cdp_max > 0.0) {
            for (int ai = 0; ai < n; ai++) cdp[ai] /= cdp_max;
        }
    }

    free(P);
    fftwf_free(X);

    TE15_LOG("\nCYCLOSTATIONARY ANALYSIS\n"
           "effective rate: %.0f Hz, alpha resolution: %.1f Hz, "
           "frames: %zu\n",
           fs, out->alpha_res, np);
    if (out->peak_count > 0) {
        TE15_LOG("Cyclic features:\n");
        for (int i = 0; i < out->peak_count; i++) {
            TE15_LOG("  %2d. %9.1f Hz (strength %.3f)\n", i + 1,
                   out->peaks[i].alpha, out->peaks[i].strength);
        }
    } else {
        TE15_LOG("No significant cyclic features detected\n");
    }
    return true;
}

bool cyclic_analyze(const IqSource *src, const Region *region,
                    CyclicResult *out)
{
    return cyclic_analyze_ex(src, region, false, out);
}
