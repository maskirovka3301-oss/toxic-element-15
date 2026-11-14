/*
 * lte_uplink.h — LTE uplink (PUSCH / PUCCH) demodulator.
 *
 * Both channels are OFDM with a 15 kHz subcarrier spacing; the uplink lays
 * its subcarriers at (m + 1/2) * 15 kHz (no subcarrier at DC), which is
 * handled per symbol (phase reset at the useful-symbol start).
 *
 *   PUSCH  physical uplink shared channel: SC-FDMA (DFT-precoded OFDM) data
 *          symbols with an OFDMA Zadoff-Chu DMRS in the middle of every
 *          slot (3GPP TS 36.211 §5.3–5.5; see D. Estévez, "Demodulation of
 *          the LTE uplink", 2022).
 *   PUCCH  physical uplink control channel: one resource block (12
 *          subcarriers) hopped between the two slots of a subframe, formats
 *          1/1a/1b (BPSK control symbol spread with an orthogonal cover)
 *          and 2/2a/2b (QPSK CQI symbols) on a computer-optimised QPSK base
 *          sequence (TS 36.211 §5.4, Table 5.5.1.2-1; see D. Estévez,
 *          "Demodulation of LTE PUCCH", 2022).
 */
#ifndef TE15_LTE_UPLINK_H
#define TE15_LTE_UPLINK_H

#include <stdbool.h>
#include <stddef.h>

#include <stdint.h>

#include "iq_source.h"
#include "metrics.h"
#include "modconf.h"
#include "region.h"
#include "signals.h"

#define LTE_MAX_BITS   32768   /* demodulated payload bits (capped)      */
#define LTE_MAX_POINTS 1024    /* constellation points kept for plotting */

typedef enum {
    LTE_CH_AUTO = 0,
    LTE_CH_PUSCH,
    LTE_CH_PUCCH
} LteChannel;

/* Parameters guessed by the Selection analysis: when a PUSCH/PUCCH match
 * comes out of the DAG/Reporter path, the LTE demodulator uses these to
 * pick the channel and to narrow the search (it stays free to refine them
 * from the signal itself). */
typedef struct {
    bool        valid;
    bool        from_analysis;
    LteChannel  channel;          /* PUSCH / PUCCH                       */
    char        band[24];         /* "B7", "B38/B41", "n78", ...         */
    double      band_min_hz;      /* legal uplink range of that band     */
    double      band_max_hz;
    double      center_hz;        /* measured centre / occupied span     */
    double      bandwidth_hz;
    int         rb_estimate;      /* occupied bandwidth / 180 kHz        */
    int         mod_order;        /* 0 unknown, else 1/2/4/6             */
    char        mod_name[12];
    double      time_min, time_max;
    char        source[128];      /* human description of the analysis   */
} LteHint;

/* Reset/fill a hint. `ids` are the ranked database matches from an Analyse
 * run, `measured_bw_hz` the measured occupied bandwidth and `mods` the
 * ranked modulation candidates. Returns true when an LTE uplink match was
 * found (hint->valid). */
void lte_hint_clear(LteHint *hint);
bool lte_hint_from_analysis(LteHint *hint, const RankedMatch *ids, size_t n,
                            double measured_bw_hz, const Metrics *m,
                            const ModCandidate *mods, size_t nmods,
                            const Region *region);

typedef struct {
    bool        valid;
    LteChannel  channel;
    const char *channel_name;   /* "PUSCH" / "PUCCH"                     */
    double      center_hz;      /* absolute RF centre of the allocation  */
    double      bw_hz;          /* occupied bandwidth                    */
    double      scs_hz;         /* subcarrier spacing (15 kHz nominal)   */
    double      cfo_hz;         /* carrier frequency offset estimate     */
    double      snr_db;
    double      evm_pct;
    int         fft_size;       /* OFDM FFT size used                    */
    int         cp_first;       /* first symbol of a slot (longer CP)    */
    int         cp_normal;
    int         slots;          /* slots demodulated                     */
    int         subcarriers;    /* occupied subcarriers (PUSCH) / 12     */
    int         rb_count;
    int         mod_order;      /* 1 = BPSK, 2 = QPSK, 4 = 16QAM, 6 = 64QAM */
    char        mod_name[12];
    int         dmrs_u;         /* PUSCH DMRS sequence group             */
    int         dmrs_v;
    int         dmrs_alpha;     /* 0..11 cyclic shift index              */
    int         pucch_format;   /* 1, 1a, 1b, 2, 2a or 2b                */
    int         pucch_u;        /* base sequence group index             */
    int         pucch_alpha[4]; /* per-slot cyclic shift index           */
    int         control_bits;   /* decoded PUCCH control bits            */
    char        control_text[96]; /* "ACK (1)" / "NACK (0)" / "SR" ...   */
    size_t      n_bits;
    unsigned char bits[(LTE_MAX_BITS + 7) / 8];
    int         n_points;
    float       pt_re[LTE_MAX_POINTS];
    float       pt_im[LTE_MAX_POINTS];
    char        summary[768];
} LteDemod;

/* Demodulate the selection (or, when the selection is unusable, the whole
 * visible band is not tried: `region` is the analysis selection). `want`
 * selects a channel or lets the demodulator decide. */
/* Demodulate the selection. `want` selects a channel (LTE_CH_AUTO lets the
 * hint or the signal decide) and `hint` may be NULL; a valid hint narrows
 * the frequency search to the band's uplink range / measured allocation,
 * forces the channel and seeds the modulation and subcarrier estimate. */
bool lte_demodulate(const IqSource *src, const Region *region, LteChannel want,
                    const LteHint *hint, LteDemod *out);

const char *lte_channel_name(LteChannel ch);

/* PUCCH base-sequence value phi_u(n) (TS 36.211 Table 5.5.1.2-1) and the
 * sequence element r_u(n) = exp(i phi_u(n) pi / 4). Exposed so callers and
 * tests can build reference waveforms. */
int lte_pucch_phi(int u, int n);
float lte_pucch_r_re(int u, int n);
float lte_pucch_r_im(int u, int n);

#endif /* TE15_LTE_UPLINK_H */
