/*
 * signals.c — known-signal database (sigidwiki-style) and confidence-based
 * matching against an observed (center frequency, bandwidth) pair.
 *
 * The database is ported verbatim from the reference single-file
 * implementation (match_known_signals() in te15.c), preserving every
 * entry's name, frequency range, bandwidth range and band, in the same
 * order, plus a literature-based modulation family per entry.
 */
#include "signals.h"
#include "pulse.h"
#include "metrics.h" /* Metrics definition (tagged struct, C11-compatible) */

#include <ctype.h>
#include <math.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

/* strcasestr (case-insensitive substring search) lives in <strings.h> on
 * the BSDs (macOS) and glibc; elsewhere we fall back to a manual scan. */
#if defined(__APPLE__) || defined(__GLIBC__) || defined(__FreeBSD__) || \
    defined(__OpenBSD__) || defined(__NetBSD__)
#include <strings.h>
#endif

const KnownSignal signal_db[] = {
/* VLF (3-30 kHz) */
{"VLF Navigation Beacon", "VLF", 3e3, 30e3, 10, 500, SIG_FAM_CW},
{"VLF Time Signal", "VLF", 3e3, 30e3, 10, 100, SIG_FAM_UNKNOWN},
{"Submarine Communication", "VLF", 3e3, 30e3, 50, 1000, SIG_FAM_FM},
/* LF (30-300 kHz) */
{"Longwave Broadcast", "LF", 30e3, 300e3, 5e3, 10e3, SIG_FAM_AM},
{"LORAN-C", "LF", 90e3, 110e3, 1e3, 5e3, SIG_FAM_OOK},
{"Non-Directional Beacon (NDB)", "LF", 190e3, 535e3, 500, 2e3, SIG_FAM_CW},
{"Time Signal (WWVB)", "LF", 60e3, 60e3, 100, 500, SIG_FAM_PSK},
/* MF (300-3000 kHz) */
{"AM Broadcast", "MF", 530e3, 1700e3, 5e3, 10e3, SIG_FAM_AM},
{"Maritime Mobile", "MF", 1.6e6, 4e6, 2e3, 6e3, SIG_FAM_SSB},
{"Aviation NDB", "MF", 190e3, 535e3, 500, 2e3, SIG_FAM_CW},
{"Amateur Radio 160m", "MF", 1.8e6, 2e6, 2e3, 6e3, SIG_FAM_UNKNOWN},
{"Shortwave Broadcast", "MF", 3e6, 30e6, 3e3, 10e3, SIG_FAM_AM},
/* HF (3-30 MHz) */
{"Amateur Radio 80m", "HF", 3.5e6, 4e6, 2e3, 6e3, SIG_FAM_UNKNOWN},
{"Amateur Radio 40m", "HF", 7e6, 7.3e6, 2e3, 6e3, SIG_FAM_UNKNOWN},
{"Amateur Radio 20m", "HF", 14e6, 14.35e6, 2e3, 6e3, SIG_FAM_UNKNOWN},
{"Amateur Radio 15m", "HF", 21e6, 21.45e6, 2e3, 6e3, SIG_FAM_UNKNOWN},
{"Amateur Radio 10m", "HF", 28e6, 29.7e6, 2e3, 6e3, SIG_FAM_UNKNOWN},
{"CB Radio", "HF", 26.965e6, 27.405e6, 5e3, 15e3, SIG_FAM_SSB},
{"Shortwave Broadcast", "HF", 3e6, 30e6, 3e3, 10e3, SIG_FAM_AM},
{"Maritime HF", "HF", 2e6, 26e6, 2e3, 6e3, SIG_FAM_SSB},
{"Aviation HF", "HF", 3e6, 30e6, 2e3, 6e3, SIG_FAM_SSB},
{"Military HF", "HF", 2e6, 30e6, 2e3, 12e3, SIG_FAM_SSB},
{"Number Station", "HF", 3e6, 30e6, 2e3, 6e3, SIG_FAM_UNKNOWN},
/* VHF (30-300 MHz) */
{"FM Broadcast", "VHF", 88e6, 108e6, 100e3, 200e3, SIG_FAM_FM},
{"VHF TV", "VHF", 54e6, 216e6, 6e6, 6e6, SIG_FAM_AM},
{"Airband (VHF)", "VHF", 118e6, 137e6, 8.33e3, 25e3, SIG_FAM_AM},
{"Amateur Radio 6m", "VHF", 50e6, 54e6, 2e3, 6e3, SIG_FAM_UNKNOWN},
{"Amateur Radio 2m", "VHF", 144e6, 148e6, 5e3, 25e3, SIG_FAM_UNKNOWN},
{"Marine VHF", "VHF", 156e6, 174e6, 12.5e3, 25e3, SIG_FAM_FM},
{"Public Safety VHF", "VHF", 136e6, 174e6, 12.5e3, 25e3, SIG_FAM_FM},
{"Business Radio VHF", "VHF", 150e6, 174e6, 12.5e3, 25e3, SIG_FAM_FM},
{"Weather Radio", "VHF", 162.4e6, 162.55e6, 5e3, 5e3, SIG_FAM_FM},
{"NOAA Weather", "VHF", 162.4e6, 162.55e6, 5e3, 5e3, SIG_FAM_FM},
/* UHF (300-3000 MHz) */
{"UHF TV", "UHF", 470e6, 890e6, 6e6, 6e6, SIG_FAM_AM},
{"Amateur Radio 70cm", "UHF", 420e6, 450e6, 5e3, 25e3, SIG_FAM_UNKNOWN},
{"Amateur Radio 23cm", "UHF", 1240e6, 1300e6, 5e3, 25e3, SIG_FAM_UNKNOWN},
{"GSM 850", "UHF", 824e6, 849e6, 200e3, 200e3, SIG_FAM_UNKNOWN},
{"GSM 900", "UHF", 890e6, 915e6, 200e3, 200e3, SIG_FAM_UNKNOWN},
{"GSM 1800", "UHF", 1710e6, 1785e6, 200e3, 200e3, SIG_FAM_UNKNOWN},
{"GSM 1900", "UHF", 1850e6, 1990e6, 200e3, 200e3, SIG_FAM_UNKNOWN},
{"3G UMTS 850", "UHF", 824e6, 849e6, 5e6, 5e6, SIG_FAM_OFDM},
{"3G UMTS 900", "UHF", 880e6, 915e6, 5e6, 5e6, SIG_FAM_OFDM},
{"3G UMTS 2100", "UHF", 1920e6, 2170e6, 5e6, 5e6, SIG_FAM_OFDM},
{"4G LTE 700", "UHF", 700e6, 800e6, 1.4e6, 20e6, SIG_FAM_OFDM},
{"4G LTE 850", "UHF", 824e6, 849e6, 1.4e6, 20e6, SIG_FAM_OFDM},
{"4G LTE 900", "UHF", 880e6, 915e6, 1.4e6, 20e6, SIG_FAM_OFDM},
{"4G LTE 1800", "UHF", 1710e6, 1785e6, 1.4e6, 20e6, SIG_FAM_OFDM},
{"4G LTE 2100", "UHF", 1920e6, 2170e6, 1.4e6, 20e6, SIG_FAM_OFDM},
{"4G LTE 2600", "UHF", 2500e6, 2700e6, 1.4e6, 20e6, SIG_FAM_OFDM},
/* LTE/5G uplink channels: legal ONLY inside the 3GPP uplink range of a
 * band (never the paired downlink of an FDD band). Inside one operator's
 * carrier the active span is 90 % of the nominal width (5/10/15/20 MHz ->
 * 4.5/9.0/13.5/18.0 MHz, 5 % guard band at each edge); PUSCH fills the
 * middle of that span while PUCCH occupies only 1-4 resource blocks
 * (180-720 kHz) at each of its two outer edges. */
{"LTE PUSCH B1", "UHF", 1920e6, 1980e6, 180e3, 18e6, SIG_FAM_OFDM},
{"LTE PUCCH B1", "UHF", 1920e6, 1980e6, 180e3, 720e3, SIG_FAM_OFDM},
{"LTE PUSCH B3", "UHF", 1710e6, 1785e6, 180e3, 18e6, SIG_FAM_OFDM},
{"LTE PUCCH B3", "UHF", 1710e6, 1785e6, 180e3, 720e3, SIG_FAM_OFDM},
{"LTE PUSCH B7", "UHF", 2500e6, 2570e6, 180e3, 18e6, SIG_FAM_OFDM},
{"LTE PUCCH B7", "UHF", 2500e6, 2570e6, 180e3, 720e3, SIG_FAM_OFDM},
{"LTE PUSCH B8", "UHF", 880e6, 915e6, 180e3, 18e6, SIG_FAM_OFDM},
{"LTE PUCCH B8", "UHF", 880e6, 915e6, 180e3, 720e3, SIG_FAM_OFDM},
{"LTE PUSCH B20", "UHF", 832e6, 862e6, 180e3, 18e6, SIG_FAM_OFDM},
{"LTE PUCCH B20", "UHF", 832e6, 862e6, 180e3, 720e3, SIG_FAM_OFDM},
{"LTE PUSCH B28", "UHF", 703e6, 733e6, 180e3, 18e6, SIG_FAM_OFDM},
{"LTE PUCCH B28", "UHF", 703e6, 733e6, 180e3, 720e3, SIG_FAM_OFDM},
{"LTE TDD PUSCH B38/B41", "UHF", 2496e6, 2690e6, 180e3, 18e6,
 SIG_FAM_OFDM},
{"LTE TDD PUCCH B38/B41", "UHF", 2496e6, 2690e6, 180e3, 720e3,
 SIG_FAM_OFDM},
{"LTE TDD PUSCH B40", "UHF", 2.3e9, 2.4e9, 180e3, 18e6, SIG_FAM_OFDM},
{"LTE TDD PUCCH B40", "UHF", 2.3e9, 2.4e9, 180e3, 720e3, SIG_FAM_OFDM},
{"5G NR PUSCH n78", "UHF", 3300e6, 3800e6, 180e3, 18e6, SIG_FAM_OFDM},
{"5G NR PUCCH n78", "UHF", 3300e6, 3800e6, 180e3, 720e3, SIG_FAM_OFDM},
{"5G NR Sub-6", "UHF", 600e6, 6000e6, 5e6, 100e6, SIG_FAM_OFDM},
{"WiFi 2.4 GHz", "UHF", 2400e6, 2483.5e6, 20e6, 40e6, SIG_FAM_OFDM},
{"WiFi 5 GHz", "UHF", 5150e6, 5850e6, 20e6, 160e6, SIG_FAM_OFDM},
{"Bluetooth Classic", "UHF", 2400e6, 2483.5e6, 1e6, 3e6, SIG_FAM_FSK},
{"Bluetooth LE", "UHF", 2400e6, 2483.5e6, 1e6, 2e6, SIG_FAM_FSK},
{"Zigbee", "UHF", 2400e6, 2483.5e6, 2e6, 2e6, SIG_FAM_PSK},
{"GPS L1", "UHF", 1575.42e6, 1575.42e6, 1e6, 2e6, SIG_FAM_PSK},
{"GPS L2", "UHF", 1227.6e6, 1227.6e6, 1e6, 2e6, SIG_FAM_PSK},
{"GLONASS L1", "UHF", 1602e6, 1602e6, 1e6, 2e6, SIG_FAM_PSK},
{"Galileo E1", "UHF", 1575.42e6, 1575.42e6, 1e6, 4e6, SIG_FAM_PSK},
{"Satellite Phone", "UHF", 1.5e9, 2.5e9, 5e3, 25e3, SIG_FAM_UNKNOWN},
{"Public Safety UHF", "UHF", 450e6, 512e6, 12.5e3, 25e3, SIG_FAM_FM},
{"Business Radio UHF", "UHF", 450e6, 512e6, 12.5e3, 25e3, SIG_FAM_FM},
{"FRS/GMRS", "UHF", 462e6, 467e6, 12.5e3, 25e3, SIG_FAM_FM},
{"PMR446", "UHF", 446e6, 446.2e6, 12.5e3, 12.5e3, SIG_FAM_FM},
{"Wireless Microphone", "UHF", 470e6, 900e6, 100e3, 200e3, SIG_FAM_FM},
{"RFID UHF", "UHF", 860e6, 960e6, 200e3, 500e3, SIG_FAM_OOK},
{"LoRa", "UHF", 433e6, 915e6, 125e3, 500e3, SIG_FAM_CHIRP},
{"Pager", "UHF", 138e6, 470e6, 2e3, 8e3, SIG_FAM_FSK},
{"POCSAG", "UHF", 138e6, 470e6, 2e3, 8e3, SIG_FAM_FSK},
{"APRS", "UHF", 144.39e6, 144.39e6, 5e3, 20e3, SIG_FAM_FSK},
{"DMR", "UHF", 400e6, 470e6, 12.5e3, 12.5e3, SIG_FAM_FSK},
{"D-Star", "UHF", 144e6, 440e6, 5e3, 25e3, SIG_FAM_FSK},
{"System Fusion", "UHF", 144e6, 440e6, 5e3, 25e3, SIG_FAM_FSK},
{"TETRA", "UHF", 380e6, 430e6, 25e3, 25e3, SIG_FAM_PSK},
{"P25", "UHF", 136e6, 960e6, 12.5e3, 12.5e3, SIG_FAM_FSK},
{"M17", "UHF", 144e6, 440e6, 5e3, 25e3, SIG_FAM_FSK},
{"FreeDV", "UHF", 7e6, 30e6, 1e3, 3e3, SIG_FAM_OFDM},
{"WSPR", "UHF", 1.8e6, 28e6, 6e3, 6e3, SIG_FAM_FSK},
{"FT8", "UHF", 1.8e6, 50e6, 50, 50, SIG_FAM_FSK},
{"JS8Call", "UHF", 1.8e6, 50e6, 50, 50, SIG_FAM_FSK},
{"Olivia", "UHF", 1.8e6, 30e6, 500, 2e3, SIG_FAM_FSK},
{"RTTY", "UHF", 1.8e6, 30e6, 170, 850, SIG_FAM_FSK},
{"PSK31", "UHF", 1.8e6, 30e6, 31, 31, SIG_FAM_PSK},
{"JT65", "UHF", 1.8e6, 50e6, 100, 200, SIG_FAM_FSK},
{"Packet Radio", "UHF", 28e6, 1296e6, 5e3, 25e3, SIG_FAM_FSK},
{"EchoLink", "UHF", 28e6, 1296e6, 5e3, 25e3, SIG_FAM_UNKNOWN},
{"IRLP", "UHF", 28e6, 1296e6, 5e3, 25e3, SIG_FAM_UNKNOWN},
{"YSF", "UHF", 144e6, 440e6, 5e3, 25e3, SIG_FAM_FSK},
{"NXDN", "UHF", 144e6, 960e6, 12.5e3, 12.5e3, SIG_FAM_FSK},
{"P25 Phase 2", "UHF", 136e6, 960e6, 12.5e3, 12.5e3, SIG_FAM_PSK},
{"ProVoice", "UHF", 136e6, 960e6, 12.5e3, 12.5e3, SIG_FAM_FM},
{"MPT-1327", "UHF", 136e6, 960e6, 12.5e3, 25e3, SIG_FAM_FSK},
{"EDACS", "UHF", 136e6, 960e6, 12.5e3, 25e3, SIG_FAM_FSK},
{"SmartNet", "UHF", 136e6, 960e6, 12.5e3, 25e3, SIG_FAM_FSK},
{"SmartZone", "UHF", 136e6, 960e6, 12.5e3, 25e3, SIG_FAM_FSK},
{"ASTRO 25", "UHF", 136e6, 960e6, 12.5e3, 12.5e3, SIG_FAM_UNKNOWN},
{"iDEN", "UHF", 800e6, 900e6, 25e3, 25e3, SIG_FAM_UNKNOWN},
{"Nextel", "UHF", 800e6, 900e6, 25e3, 25e3, SIG_FAM_UNKNOWN},
{"CDMA", "UHF", 800e6, 1900e6, 1.25e6, 1.25e6, SIG_FAM_OFDM},
{"EV-DO", "UHF", 800e6, 1900e6, 1.25e6, 1.25e6, SIG_FAM_OFDM},
{"WiMAX", "UHF", 2.3e9, 3.5e9, 5e6, 10e6, SIG_FAM_OFDM},
{"Pulsed Radar (X-band)", "UHF", 8e9, 12e9, 1e6, 100e6, SIG_FAM_OOK},
{"Weather Radar (S-band)", "UHF", 2.7e9, 3.0e9, 1e6, 10e6, SIG_FAM_OOK},
{"ATC Radar (L-band)", "UHF", 1.0e9, 1.1e9, 1e6, 10e6, SIG_FAM_OOK},
{"IFF / Secondary Radar", "UHF", 1.03e9, 1.09e9, 1e6, 10e6, SIG_FAM_OOK},
{"Inmarsat", "UHF", 1.5e9, 1.6e9, 5e3, 1e6, SIG_FAM_PSK},
{"Iridium", "UHF", 1.6e9, 1.63e9, 30e3, 100e3, SIG_FAM_PSK},
{"DVB-T (DTT)", "UHF", 470e6, 862e6, 6e6, 8e6, SIG_FAM_OFDM},
{"DAB / DAB+", "VHF", 174e6, 240e6, 1.5e6, 1.7e6, SIG_FAM_OFDM},
{"DVB-S / DVB-S2", "UHF", 10.7e9, 12.75e9, 20e6, 50e6, SIG_FAM_PSK},
{"Military Spread Spectrum", "UHF", 225e6, 400e6, 1e6, 50e6, SIG_FAM_PSK},
{"Meteor Scatter / MSK", "VHF", 50e6, 54e6, 1e3, 20e3, SIG_FAM_FSK},
{"LTE-M", "UHF", 700e6, 2100e6, 1.4e6, 1.4e6, SIG_FAM_OFDM},
{"NB-IoT", "UHF", 700e6, 2100e6, 180e3, 180e3, SIG_FAM_OFDM},
{"Sigfox", "UHF", 868e6, 915e6, 100, 600, SIG_FAM_FSK},
{"LoRaWAN", "UHF", 433e6, 915e6, 125e3, 500e3, SIG_FAM_CHIRP},
{"Z-Wave", "UHF", 868e6, 915e6, 20e3, 100e3, SIG_FAM_FSK},
{"Thread", "UHF", 2400e6, 2483.5e6, 2e6, 2e6, SIG_FAM_PSK},
{"Matter", "UHF", 2400e6, 2483.5e6, 2e6, 2e6, SIG_FAM_PSK},
{"WirelessHART", "UHF", 2400e6, 2483.5e6, 2e6, 2e6, SIG_FAM_PSK},
{"ISA100.11a", "UHF", 2400e6, 2483.5e6, 2e6, 2e6, SIG_FAM_PSK},
{"EnOcean", "UHF", 868e6, 315e6, 125e3, 125e3, SIG_FAM_OOK},
{"MiWi", "UHF", 2400e6, 2483.5e6, 2e6, 2e6, SIG_FAM_PSK},
{"ANT+", "UHF", 2400e6, 2483.5e6, 1e6, 1e6, SIG_FAM_FSK},
{"NFC", "UHF", 13.56e6, 13.56e6, 100e3, 848e3, SIG_FAM_OOK},
{"RFID HF", "UHF", 13.56e6, 13.56e6, 100e3, 848e3, SIG_FAM_OOK},
{"RFID LF", "UHF", 125e3, 134e3, 1e3, 10e3, SIG_FAM_OOK},
{"Inductive Loop", "UHF", 125e3, 134e3, 1e3, 10e3, SIG_FAM_UNKNOWN},
{"Wireless Charging", "UHF", 100e3, 300e3, 10e3, 100e3, SIG_FAM_CW},
{"Qi Wireless", "UHF", 100e3, 300e3, 10e3, 100e3, SIG_FAM_CW},
{"PMA Wireless", "UHF", 275e3, 375e3, 10e3, 100e3, SIG_FAM_CW},
{"A4WP", "UHF", 6.78e6, 6.78e6, 100e3, 500e3, SIG_FAM_CW},
{"AirFuel", "UHF", 6.78e6, 6.78e6, 100e3, 500e3, SIG_FAM_CW},
};

const size_t signal_db_size = sizeof(signal_db) / sizeof(signal_db[0]);

/* Base evidence carried by the database entry alone: frequency
 * containment, the documented occupied-bandwidth range and the band
 * specificity. Every database entry is scored with this, so the ranking can
 * compare ALL systems on their canonical parameters. */
static int base_confidence(const KnownSignal *s, double center_freq,
                           double bandwidth) {
    int confidence = 0;

    /* Frequency match: +60 in range, +20 within 1% of range centre. */
    if (center_freq >= s->freq_min && center_freq <= s->freq_max) {
        confidence += 60;
        double center = (s->freq_min + s->freq_max) / 2.0;
        double freq_diff = fabs(center_freq - center);
        if (freq_diff < center * 0.01) {
            confidence += 20;
        }
    }

    /* Bandwidth match: +30 in range, +10 within 10% of bw centre. */
    if (bandwidth >= s->bw_min && bandwidth <= s->bw_max) {
        confidence += 30;
        double bw_center = (s->bw_min + s->bw_max) / 2.0;
        double bw_diff = fabs(bandwidth - bw_center);
        if (bw_diff < bw_center * 0.1) {
            confidence += 10;
        }
    }

    /* Specificity: channelized systems (a narrow frequency range)
     * are more meaningful identifications than broad categories that
     * span decades of spectrum (e.g. "Packet Radio" 28-1296 MHz). */
    {
        double lo = s->freq_min > 0.0 ? s->freq_min : 1.0;
        double span = log10(s->freq_max / lo);
        if (span < 0.3) {
            confidence += 10;   /* less than 2x span: very specific */
        } else if (span > 1.0) {
            confidence -= 10;   /* more than 10x span: broad category */
        }
    }
    return confidence;
}

size_t signals_match(double center_freq, double bandwidth,
                     SignalMatch *out, size_t max_matches) {
    size_t count = 0;

    for (size_t i = 0; i < signal_db_size; i++) {
        if (count >= max_matches) {
            break;
        }

        const KnownSignal *s = &signal_db[i];
        int confidence = base_confidence(s, center_freq, bandwidth);

        /* Only collect matches that score high enough. */
        if (confidence < 50) {
            continue;
        }

        /* Deduplicate by identical signal name. */
        int already_added = 0;
        for (size_t j = 0; j < count; j++) {
            if (strcmp(out[j].signal->name, s->name) == 0) {
                already_added = 1;
                break;
            }
        }
        if (already_added) {
            continue;
        }

        out[count].signal = s;
        out[count].confidence = confidence;
        count++;
    }

    /* Stable insertion sort by descending confidence. */
    for (size_t i = 1; i < count; i++) {
        SignalMatch key = out[i];
        size_t j = i;
        while (j > 0 && out[j - 1].confidence < key.confidence) {
            out[j] = out[j - 1];
            j--;
        }
        out[j] = key;
    }

    return count;
}
/* ---------------------------------------------------------------------
 * signals_rank — combine the database confidence with measured
 * parameters: occupied-bandwidth containment, modulation-family
 * agreement, per-family consistency rules and literature symbol-rate
 * tables.
 * ------------------------------------------------------------------ */

#define TE15_ARR_LEN(a) (sizeof(a) / sizeof((a)[0]))

/* Case-insensitive substring test. Uses strcasestr where available;
 * defining TE15_MANUAL_CI forces the portable fallback (for testing). */
static int ci_contains(const char *haystack, const char *needle) {
    if (*needle == '\0') {
        return 1;
    }
#if defined(TE15_MANUAL_CI) ||                                     \
    !(defined(__APPLE__) || defined(__GLIBC__) ||                  \
      defined(__FreeBSD__) || defined(__OpenBSD__) ||              \
      defined(__NetBSD__))
    for (const char *h = haystack; *h != '\0'; h++) {
        const char *a = h;
        const char *b = needle;
        while (*a != '\0' && *b != '\0' &&
               tolower((unsigned char)*a) == tolower((unsigned char)*b)) {
            a++;
            b++;
        }
        if (*b == '\0') {
            return 1;
        }
    }
    return 0;
#else
    return strcasestr(haystack, needle) != NULL;
#endif
}

/* Case-insensitive string equality (mod_class is canonical, but be kind). */
static int ci_streq(const char *a, const char *b) {
    while (*a != '\0' && *b != '\0') {
        if (tolower((unsigned char)*a) != tolower((unsigned char)*b)) {
            return 0;
        }
        a++;
        b++;
    }
    return *a == *b;
}

/* Occupancy agreement: +15 when the candidate bandwidth range contains
 * the measured occupied bandwidth, -15 when they are disjoint, else 0. */
static int occupancy_agreement(const KnownSignal *s, double occupied_bw) {
    if (!(occupied_bw > 0.0)) {
        return 0;               /* unknown (or NaN) occupancy */
    }
    if (occupied_bw >= s->bw_min && occupied_bw <= s->bw_max) {
        return 15;
    }
    if (occupied_bw < s->bw_min || occupied_bw > s->bw_max) {
        return -15;
    }
    return 0;
}

/* Map a measured modulation class to a SignalFamily; -1 when the class
 * carries no family signal (NOISE) or is unknown. */
static int measured_family_from_class(const char *mod_class) {
    if (ci_streq(mod_class, "CW"))    return (int)SIG_FAM_CW;
    if (ci_streq(mod_class, "AM"))    return (int)SIG_FAM_AM;
    if (ci_streq(mod_class, "FM"))    return (int)SIG_FAM_FM;
    if (ci_streq(mod_class, "FSK"))   return (int)SIG_FAM_FSK;
    if (ci_streq(mod_class, "PSK"))   return (int)SIG_FAM_PSK;
    if (ci_streq(mod_class, "OOK"))   return (int)SIG_FAM_OOK;
    if (ci_streq(mod_class, "OFDM"))  return (int)SIG_FAM_OFDM;
    if (ci_streq(mod_class, "CHIRP")) return (int)SIG_FAM_CHIRP;
    return -1;
}

/* Family-specific parameter-consistency bonus (+10). The CANDIDATE's
 * family selects the rule, not the measured one. */
static int family_consistency_agreement(const KnownSignal *s,
                                        const Metrics *m) {
    switch (s->family) {
    case SIG_FAM_FM:
        /* Carson's rule: BW ~ 2*(dev + f_m) -> ratio in [0.8, 2.5]. */
        if (m->fm_deviation_hz > 0.0 && m->occupied_bw_hz > 0.0) {
            double ratio = m->occupied_bw_hz / (2.0 * m->fm_deviation_hz);
            if (ratio >= 0.8 && ratio <= 2.5) {
                return 10;
            }
        }
        return 0;
    case SIG_FAM_CW:
        return (m->sym_rate_conf < 0.3) ? 10 : 0;
    case SIG_FAM_OOK:
        return (m->duty_cycle > 0.0 && m->duty_cycle < 0.9) ? 10 : 0;
    case SIG_FAM_OFDM:
        return (m->papr_db > 7.0) ? 10 : 0;
    case SIG_FAM_FSK:
        return (m->fsk_levels >= 2) ? 10 : 0;
    case SIG_FAM_PSK:
        return (m->papr_db < 3.5 && m->am_depth < 0.2) ? 10 : 0;
    case SIG_FAM_CHIRP:
        return (m->chirp_trend > 0.5) ? 10 : 0;
    default:
        return 0;               /* SIG_FAM_UNKNOWN / SIG_FAM_SSB */
    }
}

/* Literature symbol-rate ranges for known digital modes: +10 per row
 * whose keyword appears in the candidate name and whose range contains
 * the measured symbol rate (each row fires at most once per candidate). */
static const struct {
    const char *keyword;
    double      lo;
    double      hi;
} te15_sym_rate_table[] = {
    {"FT8", 5.6, 6.9},
    {"WSPR", 1.3, 1.6},
    {"JT65", 2.4, 3.0},
    {"PSK31", 28.0, 34.5},
    {"RTTY", 40.0, 50.0},
    {"POCSAG", 400.0, 2600.0},
    {"APRS", 1100.0, 10000.0},
    {"DMR", 4300.0, 5300.0},
    {"D-Star", 4300.0, 5300.0},
    {"P25", 4300.0, 10500.0},
    {"NXDN", 4300.0, 5300.0},
    {"TETRA", 16000.0, 20000.0},
    {"GSM", 240000.0, 300000.0},
    {"Bluetooth", 900000.0, 1100000.0},
    {"Zigbee", 56000.0, 69000.0},
    {"Olivia", 28.0, 34.5},
    {"Sigfox", 100.0, 600.0},
    {"M17", 4300.0, 5300.0},
    {"YSF", 4300.0, 5300.0},
    {"System Fusion", 4300.0, 5300.0},
};

static int symbol_rate_agreement(const char *name, const Metrics *m) {
    if (!(m->sym_rate_conf > 0.2)) {
        return 0;
    }
    int bonus = 0;
    for (size_t i = 0; i < TE15_ARR_LEN(te15_sym_rate_table); i++) {
        if (ci_contains(name, te15_sym_rate_table[i].keyword) &&
            m->sym_rate_hz >= te15_sym_rate_table[i].lo &&
            m->sym_rate_hz <= te15_sym_rate_table[i].hi) {
            bonus += 10;
        }
    }
    return bonus;
}

static const char *mod_family_name(int fam);
static int modconf_family(const char *name);

/* Unified ranking core: every database entry whose licensed range contains
 * the measured centre is compared on ALL of its canonical parameters
 * (frequency containment, documented bandwidth range, published channel
 * bandwidth, published symbol rate, modulation family, pulsed structure and
 * the legacy family/symbol-rate evidence). `mods`/`pulse` may be NULL when
 * that analysis did not run; no candidate is dropped before the comparison.
 */
static size_t rank_core(double center_freq, double bandwidth,
                        const Metrics *m, const ModCandidate *mods,
                        size_t nmods, const PulseResult *pulse,
                        RankedMatch *out, size_t max_matches)
{
    RankedMatch *pool;
    size_t np = 0;
    int fam_score[9];
    int decisive_fam = -1;
    int decisive_score = 0;
    double meas_bw;
    double meas_sym;
    double sym_conf;
    double meas_duty;
    int pulsed_seen;

    pool = calloc(signal_db_size, sizeof(*pool));
    if (pool == NULL || max_matches == 0) {
        free(pool);
        return 0;
    }
    /* every database entry is a candidate: the measured centre is scored
     * as one canonical parameter among many, so entries that match the
     * bandwidth/symbol rate but sit outside their licensed range are still
     * reported (as invalid, with that reason) instead of disappearing */
    for (size_t i = 0; i < signal_db_size; i++) {
        const KnownSignal *s = &signal_db[i];

        pool[np].signal = s;
        pool[np].base_confidence = base_confidence(s, center_freq, bandwidth);
        np++;
    }

    /* best modulation candidate per family (strongest evidence) */
    for (int i = 0; i < 9; i++) fam_score[i] = 0;
    for (size_t i = 0; i < nmods && i < 16 && mods != NULL; i++) {
        int fam = modconf_family(mods[i].name);
        if (fam >= 0 && mods[i].score > fam_score[fam]) {
            fam_score[fam] = mods[i].score;
        }
    }
    for (int f = 0; f < 9; f++) {
        if (fam_score[f] > decisive_score) {
            decisive_score = fam_score[f];
            decisive_fam = f;
        }
    }

    meas_bw = (m != NULL && m->occupied_bw_hz > 0.0) ? m->occupied_bw_hz
                                                     : bandwidth;
    meas_sym = m != NULL ? m->sym_rate_hz : 0.0;
    sym_conf = m != NULL ? m->sym_rate_conf : 0.0;
    meas_duty = m != NULL ? m->duty_cycle : 1.0;
    pulsed_seen = pulse != NULL && pulse->num_pulses >= 5;

    for (size_t i = 0; i < np; i++) {
        const KnownSignal *sig = pool[i].signal;
        const SignalSpec *sp = signal_spec_find(sig->name);
        char why[160];
        int conf;
        int bonus = 0;
        int fam = (int)sig->family;

        why[0] = '\0';
        pool[i].in_band = (center_freq >= sig->freq_min &&
                           center_freq <= sig->freq_max) ? 1 : 0;
        pool[i].invalid_reason[0] = '\0';
        pool[i].valid = 1;
        pool[i].agreement = 0;

        /* the published-spec evidence dominates: the frequency/band overlap
         * alone cannot discriminate systems sharing a band */
        conf = (int)(0.45 * (double)pool[i].base_confidence);

        /* 1. published channel bandwidth vs the measured occupied band */
        if (sp != NULL && sp->channel_bw > 0.0 && meas_bw > 0.0) {
            double ratio = meas_bw / sp->channel_bw;

            if (ratio > 1.0) ratio = 1.0 / ratio;
            if (ratio > 0.8) {
                bonus += 25;
                snprintf(why, sizeof(why), "bandwidth matches %.3g kHz channel",
                         sp->channel_bw / 1e3);
            } else if (ratio > 0.5) {
                bonus += 12;
                snprintf(why, sizeof(why), "bandwidth near %.3g kHz channel",
                         sp->channel_bw / 1e3);
            } else {
                bonus -= 10;
            }
        }

        /* 2. published symbol/chip rate (needs a confident measurement) */
        if (sp != NULL && sp->symbol_rate > 0.0 && meas_sym > 0.0 &&
            sym_conf > 0.25) {
            double ratio = meas_sym / sp->symbol_rate;

            if (ratio > 1.0) ratio = 1.0 / ratio;
            if (ratio > 0.7) {
                size_t l = strlen(why);

                bonus += 20;
                snprintf(why + l, sizeof(why) - l,
                         "%ssymbol rate %.3g kBd matches", l ? "; " : "",
                         sp->symbol_rate / 1e3);
            } else if (ratio > 0.4) {
                bonus += 8;
            } else {
                bonus -= 8;
            }
        }

        /* 3. modulation-family evidence from the candidate scores */
        if (fam >= 0 && fam < 9) {
            if (fam_score[fam] > 0) {
                size_t l = strlen(why);

                bonus += (fam_score[fam] * 25) / 100;
                snprintf(why + l, sizeof(why) - l, "%s%s-family evidence %d%%",
                         l ? "; " : "", mod_family_name(fam), fam_score[fam]);
            } else {
                bonus -= 8;
            }
        }

        /* 4. structural agreement: pulsed systems expect a burst train,
         * continuous systems a full-duty emission */
        if (sp != NULL) {
            if (sp->pulsed && pulsed_seen) {
                size_t l = strlen(why);

                bonus += 15;
                snprintf(why + l, sizeof(why) - l, "%spulsed train matches",
                         l ? "; " : "");
            } else if (!sp->pulsed && !pulsed_seen && meas_duty > 0.8) {
                bonus += 5;
            } else if (sp->pulsed && !pulsed_seen && pulse != NULL) {
                bonus -= 15;
            } else if (!sp->pulsed && pulsed_seen) {
                bonus -= 15;
            }
        }

        /* 5. the legacy family/occupancy evidence (from the measured
         * modulation class, when one was supplied) contributes a little */
        if (m != NULL) {
            int agreement = occupancy_agreement(sig, meas_bw);
            int mf = measured_family_from_class(m->mod_class);

            pool[i].agreement = agreement;
            if (m->mod_class[0] != '\0' && mf >= 0) {
                if ((int)sig->family == mf) {
                    agreement += 20;
                    agreement += family_consistency_agreement(sig, m);
                } else if (sig->family != SIG_FAM_UNKNOWN) {
                    agreement -= 15;
                }
                agreement += symbol_rate_agreement(sig->name, m);
            }
            if (agreement < -45) agreement = -45;
            if (agreement > 45) agreement = 45;
            pool[i].agreement = agreement;
            conf += agreement / 2;
        }
        conf += bonus;
        if (conf < 0) conf = 0;
        if (conf > 100) conf = 100;

        /* validity: the documented characteristics must conform */
        {
            int bw_ok = 1;
            int fam_ok = 1;

            if (meas_bw > 0.0) {
                if (sp != NULL && sp->channel_bw > 0.0) {
                    double r = meas_bw / sp->channel_bw;

                    if (r > 1.0) r = 1.0 / r;
                    bw_ok = (r > 0.45) ? 1 : 0;
                } else {
                    bw_ok = (meas_bw >= sig->bw_min / 3.0 &&
                             meas_bw <= sig->bw_max * 3.0) ? 1 : 0;
                }
            }
            if (sig->family != SIG_FAM_UNKNOWN && decisive_fam >= 0 &&
                decisive_score >= 60) {
                fam_ok = ((int)sig->family == decisive_fam) ? 1 : 0;
            }
            pool[i].valid = bw_ok && fam_ok && pool[i].in_band;
            if (!pool[i].valid) {
                size_t l = 0;

                if (!bw_ok) {
                    l += snprintf(pool[i].invalid_reason + l,
                                  sizeof(pool[i].invalid_reason) - l,
                                  "bandwidth %.1f kHz does not match %.1f kHz",
                                  meas_bw / 1e3,
                                  sp != NULL && sp->channel_bw > 0.0
                                      ? sp->channel_bw / 1e3
                                      : sig->bw_min / 1e3);
                }
                if (!fam_ok) {
                    snprintf(pool[i].invalid_reason + l,
                             sizeof(pool[i].invalid_reason) - l,
                             "%smodulation (%s expected) does not conform",
                             l ? "; " : "", mod_family_name((int)sig->family));
                }
            }
            if (!pool[i].in_band) {
                size_t l = strlen(why);

                snprintf(why + l, sizeof(why) - l, "%soutside the licensed band",
                         l ? "; " : "");
                if (pool[i].invalid_reason[0] == '\0') {
                    snprintf(pool[i].invalid_reason,
                             sizeof(pool[i].invalid_reason),
                             "outside its %.0f-%.0f MHz allocation",
                             sig->freq_min / 1e6, sig->freq_max / 1e6);
                }
            }
            if (!pool[i].valid && conf > 25) {
                conf = 25;   /* invalid identifications stay low */
            }
        }
        pool[i].total = conf;
        pool[i].confidence_pct = conf;
        snprintf(pool[i].why, sizeof(pool[i].why), "%s", why);
    }

    /* stable sort by descending total (valid candidates first) */
    for (size_t i = 1; i < np; i++) {
        RankedMatch key = pool[i];
        size_t j = i;

        while (j > 0 && (pool[j - 1].total < key.total ||
                         (pool[j - 1].total == key.total &&
                          !pool[j - 1].valid && key.valid))) {
            pool[j] = pool[j - 1];
            j--;
        }
        pool[j] = key;
    }
    {
        size_t n_out = np < max_matches ? np : max_matches;

        if (out != NULL) {
            for (size_t i = 0; i < n_out; i++) out[i] = pool[i];
        }
        free(pool);
        return n_out;
    }
}

size_t signals_rank(double center_freq, double bandwidth, const Metrics *m,
                    RankedMatch *out, size_t max_matches) {
    return rank_core(center_freq, bandwidth, m, NULL, 0, NULL, out,
                     max_matches);
}

/* Map a modulation-candidate name to a SignalFamily for the ranking
 * bonus; -1 when the candidate does not imply a family. */
static const char *mod_family_name(int fam)
{
    static const char *names[9] = {"unknown", "AM", "FM", "CW", "PSK",
                                   "FSK", "OOK", "OFDM", "CHIRP"};
    return (fam >= 0 && fam < 9) ? names[fam] : "unknown";
}

static int modconf_family(const char *name)
{
    if (name == NULL) return -1;
    if (strstr(name, "OFDM") != NULL || strstr(name, "QAM") != NULL ||
        strstr(name, "multi-carrier") != NULL) {
        return SIG_FAM_OFDM;
    }
    if (strstr(name, "FM") != NULL) return SIG_FAM_FM;
    if (strstr(name, "AM") != NULL) return SIG_FAM_AM;
    if (strstr(name, "SSB") != NULL) return SIG_FAM_SSB;
    if (strstr(name, "CW") != NULL || strstr(name, "tone") != NULL) {
        return SIG_FAM_CW;
    }
    if (strstr(name, "PSK") != NULL || strstr(name, "BPSK") != NULL ||
        strstr(name, "QPSK") != NULL || strstr(name, "8PSK") != NULL ||
        strstr(name, "DSSS") != NULL || strstr(name, "spread") != NULL) {
        return SIG_FAM_PSK;
    }
    if (strstr(name, "FSK") != NULL) return SIG_FAM_FSK;
    if (strstr(name, "OOK") != NULL || strstr(name, "pulsed") != NULL ||
        strstr(name, "PPM") != NULL || strstr(name, "PIM") != NULL ||
        strstr(name, "PWM") != NULL || strstr(name, "radar") != NULL) {
        return SIG_FAM_OOK;
    }
    if (strstr(name, "chirp") != NULL || strstr(name, "CHIRP") != NULL) {
        return SIG_FAM_CHIRP;
    }
    return -1;
}

size_t signals_rank_conf(double center_freq, double bandwidth,
                         const Metrics *m, const ModCandidate *mods,
                         size_t nmods, const PulseResult *pulse,
                         RankedMatch *out, size_t max_matches)
{
    return rank_core(center_freq, bandwidth, m, mods, nmods, pulse, out,
                     max_matches);
}

/*
 * Published characteristic parameters for the major systems. Sources:
 *  - ETSI TS 102 361 (DMR), TIA-102 (P25), ETSI EN 300 392 (TETRA),
 *    ETSI TS 102 490 (dPMR), Kenwood/Icom NXDN specs: 12.5/6.25 kHz
 *    channels, 4FSK, symbol rates per the CEPT land-mobile tables and
 *    the GopherTrunk protocol landscape summary
 *  - 3GPP TS 36.101/38.101 (LTE 1.4-20 MHz, 5G NR 5-100 MHz)
 *  - IEEE 802.11 (WiFi 20/40/80/160 MHz), 802.15.1 (Bluetooth GFSK,
 *    1 MHz / 2 MHz BLE channels), 802.15.4 (Zigbee OQPSK, 2 MHz,
 *    250 kb/s)
 *  - Semtech LoRa datasheets (125/250/500 kHz chirp BW)
 *  - ETSI EN 300 401 (DAB 1.536 MHz OFDM), EN 300 744 (DVB-T 6/7/8 MHz),
 *    EN 302 307 (DVB-S2 20-45 MHz QPSK/8PSK transponders)
 *  - ICD-GPS-200 (GPS C/A BPSK 1.023 Mcps, 2.046 MHz main lobe),
 *    GLONASS FDMA 511 kHz
 *  - ITU-R M.1177 / radar handbooks (pulsed radars: 0.1-100 us pulses,
 *    PRF 100-10 kHz, S/X-band)
 */
const SignalSpec signal_specs[] = {
/* land-mobile digital voice (ETSI/TIA) */
{"P25", "12.5 kHz channel, C4FM 4FSK, 4800 Bd", 12500, 4800, 0, "TIA-102"},
{"DMR", "12.5 kHz, 2-slot TDMA 4FSK, 4800 Bd", 12500, 4800, 0, "ETSI TS 102 361"},
{"dPMR", "6.25 kHz, 4FSK, 2400 Bd", 6250, 2400, 0, "ETSI TS 102 490"},
{"NXDN", "6.25/12.5 kHz, 4FSK, 2400/4800 Bd", 12500, 4800, 0, "NXDN Forum"},
{"TETRA", "25 kHz, 4-slot TDMA pi/4-DQPSK, 18 kBd", 25000, 18000, 0, "ETSI EN 300 392"},
/* cellular */
{"GSM 850", "200 kHz, GMSK, 270.833 kBd", 200000, 270833, 0, "3GPP TS 45.001"},
{"GSM 900", "200 kHz, GMSK, 270.833 kBd", 200000, 270833, 0, "3GPP TS 45.001"},
{"GSM 1800", "200 kHz, GMSK, 270.833 kBd", 200000, 270833, 0, "3GPP TS 45.001"},
{"GSM 1900", "200 kHz, GMSK, 270.833 kBd", 200000, 270833, 0, "3GPP TS 45.001"},
{"3G UMTS 2100", "5 MHz WCDMA, QPSK, 3.84 Mcps", 5000000, 3840000, 0, "3GPP TS 25.101"},
{"4G LTE 700", "1.4-20 MHz OFDM, 15 kHz subcarriers", 10000000, 0, 0, "3GPP TS 36.101"},
{"4G LTE 850", "1.4-20 MHz OFDM, 15 kHz subcarriers", 10000000, 0, 0, "3GPP TS 36.101"},
{"4G LTE 1800", "1.4-20 MHz OFDM, 15 kHz subcarriers", 10000000, 0, 0, "3GPP TS 36.101"},
{"4G LTE 2100", "1.4-20 MHz OFDM, 15 kHz subcarriers", 10000000, 0, 0, "3GPP TS 36.101"},
{"4G LTE 2600", "1.4-20 MHz OFDM, 15 kHz subcarriers", 10000000, 0, 0, "3GPP TS 36.101"},
{"LTE PUSCH B1", "uplink shared channel, SC-FDMA, fills the carrier interior (1-100 RB, 15 kHz subcarriers, Zadoff-Chu DMRS, QPSK/16QAM/64QAM); Band 1 uplink 1920-1980 MHz. Active span = 90% of the nominal carrier (5/10/15/20 MHz -> 4.5/9.0/13.5/18.0 MHz) with a 5% guard band per edge", 0, 15000, 1, "3GPP TS 36.101 band table, TS 36.211 5.3-5.5"},
{"LTE PUCCH B1", "uplink control channel, 1-4 RB (180-720 kHz) on each outer edge of the carrier's active span, 15 kHz subcarriers, formats 1/1a/1b/2/2a/2b; Band 1 uplink 1920-1980 MHz. Active span = 90% of the nominal carrier (5/10/15/20 MHz -> 4.5/9.0/13.5/18.0 MHz) with a 5% guard band per edge", 180000, 15000, 1, "3GPP TS 36.101 band table, TS 36.211 5.3-5.5"},
{"LTE PUSCH B3", "uplink shared channel, SC-FDMA, fills the carrier interior (1-100 RB, 15 kHz subcarriers, Zadoff-Chu DMRS, QPSK/16QAM/64QAM); Band 3 uplink 1710-1785 MHz. Active span = 90% of the nominal carrier (5/10/15/20 MHz -> 4.5/9.0/13.5/18.0 MHz) with a 5% guard band per edge", 0, 15000, 1, "3GPP TS 36.101 band table, TS 36.211 5.3-5.5"},
{"LTE PUCCH B3", "uplink control channel, 1-4 RB (180-720 kHz) on each outer edge of the carrier's active span, 15 kHz subcarriers, formats 1/1a/1b/2/2a/2b; Band 3 uplink 1710-1785 MHz. Active span = 90% of the nominal carrier (5/10/15/20 MHz -> 4.5/9.0/13.5/18.0 MHz) with a 5% guard band per edge", 180000, 15000, 1, "3GPP TS 36.101 band table, TS 36.211 5.3-5.5"},
{"LTE PUSCH B7", "uplink shared channel, SC-FDMA, fills the carrier interior (1-100 RB, 15 kHz subcarriers, Zadoff-Chu DMRS, QPSK/16QAM/64QAM); Band 7 uplink 2500-2570 MHz. Active span = 90% of the nominal carrier (5/10/15/20 MHz -> 4.5/9.0/13.5/18.0 MHz) with a 5% guard band per edge", 0, 15000, 1, "3GPP TS 36.101 band table, TS 36.211 5.3-5.5"},
{"LTE PUCCH B7", "uplink control channel, 1-4 RB (180-720 kHz) on each outer edge of the carrier's active span, 15 kHz subcarriers, formats 1/1a/1b/2/2a/2b; Band 7 uplink 2500-2570 MHz. Active span = 90% of the nominal carrier (5/10/15/20 MHz -> 4.5/9.0/13.5/18.0 MHz) with a 5% guard band per edge", 180000, 15000, 1, "3GPP TS 36.101 band table, TS 36.211 5.3-5.5"},
{"LTE PUSCH B8", "uplink shared channel, SC-FDMA, fills the carrier interior (1-100 RB, 15 kHz subcarriers, Zadoff-Chu DMRS, QPSK/16QAM/64QAM); Band 8 uplink 880-915 MHz. Active span = 90% of the nominal carrier (5/10/15/20 MHz -> 4.5/9.0/13.5/18.0 MHz) with a 5% guard band per edge", 0, 15000, 1, "3GPP TS 36.101 band table, TS 36.211 5.3-5.5"},
{"LTE PUCCH B8", "uplink control channel, 1-4 RB (180-720 kHz) on each outer edge of the carrier's active span, 15 kHz subcarriers, formats 1/1a/1b/2/2a/2b; Band 8 uplink 880-915 MHz. Active span = 90% of the nominal carrier (5/10/15/20 MHz -> 4.5/9.0/13.5/18.0 MHz) with a 5% guard band per edge", 180000, 15000, 1, "3GPP TS 36.101 band table, TS 36.211 5.3-5.5"},
{"LTE PUSCH B20", "uplink shared channel, SC-FDMA, fills the carrier interior (1-100 RB, 15 kHz subcarriers, Zadoff-Chu DMRS, QPSK/16QAM/64QAM); Band 20 uplink 832-862 MHz. Active span = 90% of the nominal carrier (5/10/15/20 MHz -> 4.5/9.0/13.5/18.0 MHz) with a 5% guard band per edge", 0, 15000, 1, "3GPP TS 36.101 band table, TS 36.211 5.3-5.5"},
{"LTE PUCCH B20", "uplink control channel, 1-4 RB (180-720 kHz) on each outer edge of the carrier's active span, 15 kHz subcarriers, formats 1/1a/1b/2/2a/2b; Band 20 uplink 832-862 MHz. Active span = 90% of the nominal carrier (5/10/15/20 MHz -> 4.5/9.0/13.5/18.0 MHz) with a 5% guard band per edge", 180000, 15000, 1, "3GPP TS 36.101 band table, TS 36.211 5.3-5.5"},
{"LTE PUSCH B28", "uplink shared channel, SC-FDMA, fills the carrier interior (1-100 RB, 15 kHz subcarriers, Zadoff-Chu DMRS, QPSK/16QAM/64QAM); Band 28 uplink 703-733 MHz. Active span = 90% of the nominal carrier (5/10/15/20 MHz -> 4.5/9.0/13.5/18.0 MHz) with a 5% guard band per edge", 0, 15000, 1, "3GPP TS 36.101 band table, TS 36.211 5.3-5.5"},
{"LTE PUCCH B28", "uplink control channel, 1-4 RB (180-720 kHz) on each outer edge of the carrier's active span, 15 kHz subcarriers, formats 1/1a/1b/2/2a/2b; Band 28 uplink 703-733 MHz. Active span = 90% of the nominal carrier (5/10/15/20 MHz -> 4.5/9.0/13.5/18.0 MHz) with a 5% guard band per edge", 180000, 15000, 1, "3GPP TS 36.101 band table, TS 36.211 5.3-5.5"},
{"LTE TDD PUSCH B38/B41", "uplink shared channel, SC-FDMA, fills the carrier interior (1-100 RB, 15 kHz subcarriers, Zadoff-Chu DMRS, QPSK/16QAM/64QAM); TDD bands 38/41 (full band shared with the downlink, uplink in designated slots). Active span = 90% of the nominal carrier (5/10/15/20 MHz -> 4.5/9.0/13.5/18.0 MHz) with a 5% guard band per edge", 0, 15000, 1, "3GPP TS 36.101 band table, TS 36.211 5.3-5.5"},
{"LTE TDD PUCCH B38/B41", "uplink control channel, 1-4 RB (180-720 kHz) on each outer edge of the carrier's active span, 15 kHz subcarriers, formats 1/1a/1b/2/2a/2b; TDD bands 38/41 (full band shared with the downlink, uplink in designated slots). Active span = 90% of the nominal carrier (5/10/15/20 MHz -> 4.5/9.0/13.5/18.0 MHz) with a 5% guard band per edge", 180000, 15000, 1, "3GPP TS 36.101 band table, TS 36.211 5.3-5.5"},
{"LTE TDD PUSCH B40", "uplink shared channel, SC-FDMA, fills the carrier interior (1-100 RB, 15 kHz subcarriers, Zadoff-Chu DMRS, QPSK/16QAM/64QAM); TDD band 40 (full band shared). Active span = 90% of the nominal carrier (5/10/15/20 MHz -> 4.5/9.0/13.5/18.0 MHz) with a 5% guard band per edge", 0, 15000, 1, "3GPP TS 36.101 band table, TS 36.211 5.3-5.5"},
{"LTE TDD PUCCH B40", "uplink control channel, 1-4 RB (180-720 kHz) on each outer edge of the carrier's active span, 15 kHz subcarriers, formats 1/1a/1b/2/2a/2b; TDD band 40 (full band shared). Active span = 90% of the nominal carrier (5/10/15/20 MHz -> 4.5/9.0/13.5/18.0 MHz) with a 5% guard band per edge", 180000, 15000, 1, "3GPP TS 36.101 band table, TS 36.211 5.3-5.5"},
{"5G NR PUSCH n78", "uplink shared channel, SC-FDMA, fills the carrier interior (1-100 RB, 15 kHz subcarriers, Zadoff-Chu DMRS, QPSK/16QAM/64QAM); 5G NR TDD n78 3300-3800 MHz (numerology-independent PUSCH; this demodulator is the LTE version). Active span = 90% of the nominal carrier (5/10/15/20 MHz -> 4.5/9.0/13.5/18.0 MHz) with a 5% guard band per edge", 0, 15000, 1, "3GPP TS 36.101 band table, TS 36.211 5.3-5.5"},
{"5G NR PUCCH n78", "uplink control channel, 1-4 RB (180-720 kHz) on each outer edge of the carrier's active span, 15 kHz subcarriers, formats 1/1a/1b/2/2a/2b; 5G NR TDD n78 3300-3800 MHz (PUCCH formats 0-4; this demodulator is the LTE version). Active span = 90% of the nominal carrier (5/10/15/20 MHz -> 4.5/9.0/13.5/18.0 MHz) with a 5% guard band per edge", 180000, 15000, 1, "3GPP TS 36.101 band table, TS 36.211 5.3-5.5"},
{"5G NR Sub-6", "5-100 MHz OFDM, 15-60 kHz subcarriers", 20000000, 0, 0, "3GPP TS 38.101"},
/* short range data */
{"WiFi 2.4 GHz", "20/40 MHz OFDM 802.11", 20000000, 0, 0, "IEEE 802.11"},
{"WiFi 5 GHz", "20-160 MHz OFDM 802.11", 80000000, 0, 0, "IEEE 802.11"},
{"Bluetooth Classic", "1 MHz GFSK, 1 Mb/s", 1000000, 1000000, 0, "IEEE 802.15.1"},
{"Bluetooth LE", "2 MHz GFSK, 1 Mb/s", 2000000, 1000000, 0, "IEEE 802.15.1"},
{"Zigbee", "2 MHz, OQPSK 250 kb/s", 2000000, 250000, 0, "IEEE 802.15.4"},
{"LoRa", "125/250/500 kHz chirp (SF7-12)", 125000, 0, 0, "Semtech SX1276"},
/* broadcast */
{"DVB-T (DTT)", "6/7/8 MHz OFDM (2k/8k)", 8000000, 0, 0, "ETSI EN 300 744"},
{"DAB / DAB+", "1.536 MHz OFDM", 1536000, 0, 0, "ETSI EN 300 401"},
{"DVB-S / DVB-S2", "20-45 MHz QPSK/8PSK transponder", 36000000, 0, 0, "ETSI EN 302 307"},
{"AM Broadcast", "10 kHz channel spacing, AM DSB", 10000, 0, 0, "ITU-R BS.560"},
{"FM Broadcast", "200 kHz channel, 75 kHz deviation", 200000, 0, 0, "ITU-R BS.450"},
{"Airband (VHF)", "8.33/25 kHz channel, AM", 25000, 0, 0, "ICAO Annex 10"},
{"Marine VHF", "25 kHz channel, FM", 25000, 0, 0, "ITU-R M.1084"},
{"Weather Radio", "25 kHz channel, FM", 25000, 0, 0, "NOAA/NWS"},
{"CB Radio", "10 kHz channel, AM/SSB/FM", 10000, 0, 0, "47 CFR 95"},
{"LORAN-C", "100 kHz pulse groups, 100 us pulses", 20000, 0, 1, "ITU-R M.589"},
{"Time Signal (WWVB)", "60 kHz, AM/PWM 1 bit/s", 1000, 0, 0, "NIST WWVB"},
/* radionavigation / satellite */
{"GPS L1", "BPSK 1.023 Mcps, 2.046 MHz main lobe", 2046000, 1023000, 0, "ICD-GPS-200"},
{"GPS L2", "BPSK 1.023 Mcps", 2046000, 1023000, 0, "ICD-GPS-200"},
{"GLONASS L1", "FDMA 511 kHz, 0.511 Mcps", 511000, 511000, 0, "GLONASS ICD"},
{"Galileo E1", "BOC(1,1), 4.092 Mcps subcarrier", 4092000, 4092000, 0, "Galileo OS SIS ICD"},
{"Inmarsat", "L-band, QPSK/16QAM bursts", 40000, 0, 1, "Inmarsat SDM"},
{"Iridium", "TDMA bursts 8.28 ms, 41.7 kBd QPSK", 40000, 41700, 1, "Iridium air interface"},
/* radar */
{"Pulsed Radar (X-band)", "0.1-100 us pulses, PRF 0.1-10 kHz, X-band", 10000000, 0, 1, "ITU-R M.1177"},
{"Weather Radar (S-band)", "0.5-4 us pulses, PRF 200-2000 Hz, S-band", 5000000, 0, 1, "ITU-R M.1849"},
{"ATC Radar (L-band)", "0.8-1 us pulses, PRF 300-1200 Hz, L-band", 3000000, 0, 1, "ICAO Annex 10"},
{"IFF / Secondary Radar", "1030/1090 MHz pulse pairs", 6000000, 0, 1, "ICAO Annex 10"},
};
const size_t signal_specs_size = sizeof(signal_specs) / sizeof(signal_specs[0]);

const SignalSpec *signal_spec_find(const char *name)
{
    if (name == NULL) return NULL;
    for (size_t i = 0; i < signal_specs_size; i++) {
        if (strcmp(signal_specs[i].name, name) == 0) {
            return &signal_specs[i];
        }
    }
    return NULL;
}

int signals_anomaly_check(double center_freq, double bandwidth,
                          const RankedMatch *ids, size_t n,
                          char *explain, size_t explain_len)
{
    if (explain != NULL && explain_len > 0) explain[0] = '\0';
    if (ids == NULL || n == 0) {
        if (explain != NULL && explain_len > 0) {
            snprintf(explain, explain_len,
                     "ANOMALY: no known system candidate could be evaluated "
                     "for %.3f MHz / %.1f kHz.", center_freq / 1e6,
                     bandwidth / 1e3);
        }
        return 1;
    }
    for (size_t i = 0; i < n; i++) {
        if (ids[i].valid) return 0; /* at least one valid identification */
    }
    if (explain == NULL || explain_len == 0) return 1;

    size_t used = 0;
    used += snprintf(explain, explain_len,
                     "ANOMALY: no candidate conforms to its documented "
                     "specification. Measured %.3f MHz, %.1f kHz. ",
                     center_freq / 1e6, bandwidth / 1e3);
    for (size_t i = 0; i < n && i < 3; i++) {
        const KnownSignal *sig = ids[i].signal;
        used += snprintf(explain + used, explain_len - used,
                         "%s%s: ", i ? "; " : "", sig->name);
        if (used >= explain_len) break;
        if (!ids[i].in_band) {
            double outside = center_freq < sig->freq_min
                                 ? sig->freq_min - center_freq
                                 : center_freq - sig->freq_max;
            used += snprintf(explain + used, explain_len - used,
                             "%.4g MHz outside its %.4g-%.4g MHz allocation",
                             outside / 1e6, sig->freq_min / 1e6,
                             sig->freq_max / 1e6);
            if (used >= explain_len) break;
        }
        if (ids[i].invalid_reason[0] != '\0') {
            used += snprintf(explain + used, explain_len - used,
                             "%s%s", ids[i].in_band ? "" : "; ",
                             ids[i].invalid_reason);
            if (used >= explain_len) break;
        }
        if (ids[i].in_band && ids[i].invalid_reason[0] == '\0') {
            used += snprintf(explain + used, explain_len - used,
                             "documented bandwidth %.3g kHz",
                             (signal_spec_find(sig->name) != NULL &&
                              signal_spec_find(sig->name)->channel_bw > 0)
                                 ? signal_spec_find(sig->name)->channel_bw / 1e3
                                 : sig->bw_min / 1e3);
        }
    }
    return 1;
}
