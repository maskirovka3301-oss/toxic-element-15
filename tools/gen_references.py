#!/usr/bin/env python3
"""
gen_references.py — generate gqrx-style synthetic IQ reference files.

One 2-second surrogate per signal type of the known-system database that
operates between 2 and 3 GHz. Each signal is synthesised from its
published structure (channel bandwidth, symbol/chip rate, modulation,
frame/burst timing) so the analysis suite can be exercised against
ground truth.

Files are written as `references/gqrx_<date>_<time>_<freq>_<rate>_fc.raw`
(interleaved float32 complex, as produced by gqrx). They are large, so
they are NOT committed (see references/.gitignore); regenerate with

    python3 tools/gen_references.py
"""
import os
import sys

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "references")
SECONDS = 2.0
rng = np.random.default_rng(20261112)


# --------------------------------------------------------------------------
# building blocks
# --------------------------------------------------------------------------
def qpsk(n):
    return (rng.integers(0, 2, n) * 2 - 1) + 1j * (rng.integers(0, 2, n) * 2 - 1)


def qam16(n):
    levels = np.array([-3, -1, 1, 3]) / np.sqrt(10.0)
    return rng.choice(levels, n) + 1j * rng.choice(levels, n)


def rrc(beta, sps, span=8):
    t = np.arange(-span * sps, span * sps + 1) / sps
    h = np.zeros_like(t, dtype=float)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            h[i] = 1.0 - beta + 4.0 * beta / np.pi
        elif abs(abs(4.0 * beta * ti) - 1.0) < 1e-9:
            h[i] = (beta / np.sqrt(2.0)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) +
                (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            h[i] = (np.sin(np.pi * ti * (1 - beta)) +
                    4 * beta * ti * np.cos(np.pi * ti * (1 + beta))) / (
                np.pi * ti * (1 - (4 * beta * ti) ** 2))
    return h / np.sqrt(np.sum(h ** 2))


def shape(symbols, sps, beta=0.35):
    up = np.zeros(len(symbols) * sps, dtype=complex)
    up[::sps] = symbols
    h = rrc(beta, sps)
    return np.convolve(up, h, mode="same")


def ofdm_frame(n_fft, n_used, cp, qam16_flag=False):
    sym = qam16(n_used) if qam16_flag else qpsk(n_used)
    spec = np.zeros(n_fft, dtype=complex)
    idx = np.concatenate([np.arange(1, n_used // 2 + 1),
                          np.arange(-n_used // 2, 0)])
    spec[idx] = sym
    body = np.fft.ifft(np.fft.ifftshift(spec)) * np.sqrt(n_fft)
    return np.concatenate([body[-cp:], body])


def ofdm_stream(fs, n_fft, n_used, cp_len, seconds, burst_period=None,
                burst_len=None, power=0.5):
    """Continuous or bursty OFDM (burst_period None = continuous)."""
    n = int(fs * seconds)
    spf = n_fft + cp_len
    frames = max(1, int(np.ceil(n / spf)))
    sig = np.zeros(n, dtype=complex)
    pos = 0
    for _ in range(frames):
        f = ofdm_frame(n_fft, n_used, cp_len,
                       qam16_flag=(rng.random() < 0.5))
        if pos + len(f) > n:
            break
        sig[pos:pos + len(f)] = f
        pos += len(f)
    sig *= np.sqrt(power / (np.mean(np.abs(sig) ** 2) + 1e-30))
    if burst_period is not None:
        env = np.zeros(n)
        blen = int(burst_len * fs)
        bper = int(burst_period * fs)
        for start in range(0, n, bper):
            env[start:start + blen] = 1.0
        sig *= env
    return sig


def fsk(symbols, sps, mod_index=0.5):
    """CPFSK/GFSK with a Gaussian pulse (BT ~0.5)."""
    n = len(symbols) * sps
    idx = np.repeat(symbols, sps)
    # Gaussian filter on the frequency pulses
    bt = 0.5
    t = np.arange(-2 * sps, 2 * sps + 1) / sps
    g = np.exp(-2.0 * np.pi ** 2 * bt ** 2 * t ** 2)
    g /= g.sum()
    fm = np.convolve(np.repeat(symbols.astype(float), sps), g, mode="same")
    fm *= mod_index * sps / sps
    phase = 2.0 * np.pi * np.cumsum(fm) / sps
    return np.exp(1j * phase[:len(idx)])


def band_limit(sig, fs, bw, transition=1.15):
    """Transmitter-like spectral mask: raised-cosine edges at bw/2 .. bw/2*transition."""
    n = len(sig)
    nf = 1 << int(np.ceil(np.log2(n)))
    spec = np.fft.fft(sig, nf)
    f = np.abs(np.fft.fftfreq(nf, 1.0 / fs))
    half = bw / 2.0
    edge = half * transition
    h = np.ones(nf)
    h[f >= edge] = 0.0
    band = (f > half) & (f < edge)
    h[band] = 0.5 * (1.0 + np.cos(np.pi * (f[band] - half) / (edge - half)))
    return np.fft.ifft(spec * h)[:n]


def awgn(sig, snr_db):
    p = np.mean(np.abs(sig) ** 2) + 1e-30
    npow = p / (10.0 ** (snr_db / 10.0))
    return sig + (rng.normal(0, np.sqrt(npow / 2), len(sig)) +
                  1j * rng.normal(0, np.sqrt(npow / 2), len(sig)))


def slug(name):
    """Filesystem/label-safe signal name: '4G LTE 2600 (10 MHz, Band 7)'
    -> '4G_LTE_2600_10MHz_Band7'."""
    out = []
    for ch in name:
        if ch.isalnum():
            out.append(ch)
        elif ch in " +-_/":
            out.append("_")
    txt = "".join(out)
    while "__" in txt:
        txt = txt.replace("__", "_")
    return txt.strip("_")


def write_gqrx(name, sig, fs, freq_hz, mtime="20261112_160000"):
    n = len(sig)
    data = np.empty(2 * n, dtype=np.float32)
    data[0::2] = sig.real
    data[1::2] = sig.imag
    # the signal name prefixes the file; the gqrx marker and the usual
    # <time>_<freq>_<rate>_fc fields follow so the app parses it
    fname = (f"{slug(name)}_gqrx_{mtime}_{int(round(freq_hz))}_"
             f"{int(round(fs))}_fc.raw")
    path = os.path.join(OUT, fname)
    data.tofile(path)
    print(f"  {os.path.basename(path):70s} {n/fs:5.2f} s "
          f"{fs/1e6:6.2f} MS/s {os.path.getsize(path)/1e6:7.1f} MB  {name}")


# --------------------------------------------------------------------------
# signals (2-3 GHz database entries)
# --------------------------------------------------------------------------
def gen_umts_2100():
    """3GPP TS 25.101: 5 MHz WCDMA, 3.84 Mcps, QPSK data on OVSF codes."""
    fs = 7.68e6
    sps = 2
    chip_rate = 3.84e6
    nchips = int(SECONDS * chip_rate)
    data = (rng.integers(0, 2, nchips) * 2 - 1) + \
           1j * (rng.integers(0, 2, nchips) * 2 - 1)
    # OVSF spreading (SF=8, code 1) + scrambling
    sf = 8
    code = np.array([1, -1, 1, -1, 1, -1, 1, -1])
    spread = np.repeat(data[::sf], sf) * np.tile(code, len(data) // sf)
    scrambler = (rng.integers(0, 2, len(spread)) * 2 - 1) * \
                (1 + 1j) / np.sqrt(2.0)
    chips = spread * scrambler
    sig = shape(chips, sps, beta=0.22)
    sig = band_limit(sig, fs, 5e6)
    sig = awgn(sig, 25.0)
    write_gqrx("3G UMTS 2100 (WCDMA 5 MHz, 3.84 Mcps)", sig, fs, 2140e6)


def gen_lte(center_hz, n_used, fs, name, channel_bw):
    """3GPP TS 36.104: OFDM, 15 kHz subcarriers, 1 ms subframes, CP."""
    n_fft = 1024 if n_used <= 600 else 2048
    scs = 15e3
    cp = int(round(4.7e-6 * fs))
    sig = ofdm_stream(fs, n_fft, n_used, cp, SECONDS)
    sig = band_limit(sig, fs, 1.05 * channel_bw)
    # PSS/SSS every 5 ms: a short sync burst boosts a spectral line
    fs_n = len(sig)
    for t in np.arange(0.0, SECONDS, 0.005):
        i = int(t * fs)
        seg = ofdm_frame(n_fft, n_used, cp)
        sig[i:i + min(len(seg), fs_n - i)] = seg[:max(0, fs_n - i)]
    sig = awgn(sig, 22.0)
    write_gqrx(name, sig, fs, center_hz)


def gen_5g_nr():
    """3GPP TS 38.104: 30 kHz SCS, 20 MHz channel, cyclic prefix."""
    fs = 30.72e6
    n_fft, n_used = 1024, 600
    cp = int(round(2.3e-6 * fs))
    sig = ofdm_stream(fs, n_fft, n_used, cp, SECONDS)
    sig = band_limit(sig, fs, 18e6)
    sig = awgn(sig, 22.0)
    write_gqrx("5G NR Sub-6 (30 kHz SCS, 20 MHz)", sig, fs, 2900e6)


def gen_wifi():
    """IEEE 802.11g: 20 MHz OFDM, 312.5 kHz subcarriers, 800 ns GI,
    frame bursts (preamble + data) every few ms."""
    fs = 25e6
    n_fft, n_used = 64, 52
    cp = int(round(0.8e-6 * fs))
    sig = ofdm_stream(fs, n_fft, n_used, cp, SECONDS, burst_period=0.008,
                      burst_len=0.0012)
    sig = band_limit(sig, fs, 18e6)
    sig = awgn(sig, 24.0)
    write_gqrx("WiFi 2.4 GHz (802.11g 20 MHz OFDM)", sig, fs, 2437e6)


def gen_bt_classic():
    """IEEE 802.15.1: GFSK 1 Mbaud, 1 MHz channels, 625 us slots, hopping
    among the channels inside the captured window."""
    fs = 8e6
    sps = 8
    n = int(fs * SECONDS)
    sig = np.zeros(n, dtype=complex)
    t = 0.0
    hop_channels = np.array([2402, 2404, 2406, 2408, 2410, 2412, 2414]) * 1e6
    center = 2408e6
    slot = 625e-6
    while t < SECONDS:
        nslots = int(rng.integers(1, 6))
        dur = nslots * slot
        nsym = int(dur * 1e6)
        bits = (rng.integers(0, 2, nsym) * 2 - 1).astype(float)
        pkt = fsk(bits, sps, mod_index=0.5)
        off = hop_channels[rng.integers(0, len(hop_channels))] - center
        tt = np.arange(len(pkt)) / fs
        pkt = pkt * np.exp(1j * 2 * np.pi * off * tt)
        i0 = int(t * fs)
        ln = min(len(pkt), n - i0)
        if ln > 0:
            sig[i0:i0 + ln] += pkt[:ln] * 0.4
        t += dur
    sig = band_limit(sig, fs, 1.47e6)
    sig = awgn(sig, 22.0)
    write_gqrx("Bluetooth Classic (GFSK 1 Mbaud, hopping)", sig, fs, center)


def gen_ble():
    """Bluetooth LE: GFSK 1 Mbaud, 2 MHz channels, 1.5 ms advertising
    packets on the advertising channels."""
    fs = 4e6
    sps = 4
    n = int(fs * SECONDS)
    sig = np.zeros(n, dtype=complex)
    t = 0.0
    while t < SECONDS:
        nsym = int(1.5e-3 * 1e6)
        bits = (rng.integers(0, 2, nsym) * 2 - 1).astype(float)
        # advertising preamble is an alternating pattern
        bits[:16] = np.tile([1.0, -1.0], 8)
        pkt = fsk(bits, sps, mod_index=0.5)
        i0 = int(t * fs)
        ln = min(len(pkt), n - i0)
        if ln > 0:
            sig[i0:i0 + ln] += pkt[:ln] * 0.4
        t += 0.0015 + rng.random() * 0.002
    sig = band_limit(sig, fs, 2.2e6)
    sig = awgn(sig, 22.0)
    write_gqrx("Bluetooth LE (GFSK 1 Mbaud, advertising)", sig, fs, 2426e6)


def gen_802154(name, center_hz, slotting=None, duty_gap=0.0):
    """IEEE 802.15.4 (Zigbee/Thread/Matter/WirelessHART/ISA100.11a/MiWi):
    OQPSK 2 Mcps with 32-chip DSSS (250 kb/s), 2 MHz channel."""
    fs = 4e6
    sps = 2
    chip_rate = 2e6
    n = int(fs * SECONDS)
    sig = np.zeros(n, dtype=complex)
    # 802.15.4 chip sequences (first 4 of the standard set)
    chip_seqs = np.array([
        [1, -1, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1, 1, 1, 0, 1, 1, 0, -1, -1,
         0, 0, -1, -1, 0, 0, -1, -1, 1, 1, 0, 1],
        [1, -1, 1, 1, 0, 0, -1, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, -1,
         1, 1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, -1, -1, 1, 1, 0, 0, -1, -1, 1, 1, 0, 1, 0, 0, 1, 1,
         -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0],
        [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1,
         -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]])
    t = 0.0
    slot = slotting if slotting else 0.004
    while t < SECONDS:
        nframe_syms = int(0.0004 * 2e6 / 32)   # ~400 us frames
        syms = rng.integers(0, 4, max(1, nframe_syms))
        chips = np.concatenate([chip_seqs[s] for s in syms]).astype(float)
        pkt = shape(chips + 1j * 0.0, sps, beta=0.5)
        i0 = int(t * fs)
        ln = min(len(pkt), n - i0)
        if ln > 0:
            sig[i0:i0 + ln] += pkt[:ln] * 0.4
        t += slot + rng.random() * slot
    sig = band_limit(sig, fs, 2.2e6)
    sig = awgn(sig, 22.0)
    write_gqrx(name, sig, fs, center_hz)


def gen_satphone():
    """Satellite phone: 25 kHz GMSK voice bursts (bursty TDMA)."""
    fs = 200e3
    sps = 4
    n = int(fs * SECONDS)
    sig = np.zeros(n, dtype=complex)
    t = 0.0
    while t < SECONDS:
        nsym = int(40e-3 * 25e3)          # 40 ms burst at 25 kbaud
        bits = (rng.integers(0, 2, nsym) * 2 - 1).astype(float)
        pkt = fsk(bits, sps, mod_index=0.5)
        i0 = int(t * fs)
        ln = min(len(pkt), n - i0)
        if ln > 0:
            sig[i0:i0 + ln] += pkt[:ln] * 0.4
        t += 0.08
    sig = band_limit(sig, fs, 40e3)
    sig = awgn(sig, 20.0)
    write_gqrx("Satellite Phone (25 kHz GMSK bursts)", sig, fs, 2200e6)


def gen_wimax():
    """IEEE 802.16: OFDM, 10.94 kHz subcarrier spacing, 10 MHz profile."""
    fs = 12e6
    n_fft, n_used = 1024, 840
    cp = int(round(1 / 8 * 1024 / fs * fs / 1.0))  # 1/8 CP
    cp = int(1024 / 8)
    sig = ofdm_stream(fs, n_fft, n_used, cp, SECONDS)
    sig = band_limit(sig, fs, 10e6)
    sig = awgn(sig, 22.0)
    write_gqrx("WiMAX (OFDM 10.94 kHz, 10 MHz)", sig, fs, 2550e6)


def gen_weather_radar():
    """ITU-R M.1849: S-band weather radar, pulsed linear-FM chirp,
    5 MHz sweep, ~1 us pulse, 1 kHz PRF, antenna scan amplitude."""
    fs = 20e6
    n = int(fs * SECONDS)
    sig = np.zeros(n, dtype=complex)
    prf, tau = 1000.0, 1e-6
    bw = 5e6
    for k in range(int(SECONDS * prf)):
        t0 = k / prf
        i0 = int(t0 * fs)
        ln = int(tau * fs)
        if i0 + ln >= n:
            break
        tt = np.arange(ln) / fs
        chirp = np.exp(1j * np.pi * bw / tau * tt ** 2)
        # antenna scan: slow amplitude modulation (~0.5 Hz scan)
        scan = 0.5 + 0.5 * np.cos(2 * np.pi * 0.5 * t0)
        sig[i0:i0 + ln] += chirp * scan * 0.5
    sig = band_limit(sig, fs, 6e6)
    sig = awgn(sig, 20.0)
    write_gqrx("Weather Radar (S-band chirp pulsed)", sig, fs, 2850e6)


def gen_lte_m():
    """3GPP Rel-13 LTE-M: 1.4 MHz LTE channel."""
    fs = 2.4e6
    sig = ofdm_stream(fs, 128, 72, int(9.8e-6 * fs), SECONDS)
    sig = band_limit(sig, fs, 1.47e6)
    sig = awgn(sig, 20.0)
    write_gqrx("LTE-M (1.4 MHz Cat-M1)", sig, fs, 2050e6)


def gen_nb_iot():
    """3GPP Rel-13 NB-IoT: 180 kHz (12 x 15 kHz) in-band."""
    fs = 300e3
    sig = ofdm_stream(fs, 16, 12, int(5.2e-6 * fs), SECONDS)
    sig = band_limit(sig, fs, 180e3)
    sig = awgn(sig, 18.0)
    write_gqrx("NB-IoT (180 kHz, 15 kHz SCS)", sig, fs, 2060e6)


def gen_ant_plus():
    """ANT+ (Dynastream): GFSK 1 Mbaud, 1 MHz channels, 4 Hz-ish TDMA
    slots of 8 messages, adaptive channel selection."""
    fs = 4e6
    sps = 4
    n = int(fs * SECONDS)
    sig = np.zeros(n, dtype=complex)
    t = 0.0
    while t < SECONDS:
        nsym = int(0.00016 * 1e6)        # ~160 us ANT message burst
        bits = (rng.integers(0, 2, nsym) * 2 - 1).astype(float)
        pkt = fsk(bits, sps, mod_index=0.5)
        i0 = int(t * fs)
        ln = min(len(pkt), n - i0)
        if ln > 0:
            sig[i0:i0 + ln] += pkt[:ln] * 0.4
        t += 0.004                        # 4 Hz message rate
    sig = band_limit(sig, fs, 1.47e6)
    sig = awgn(sig, 22.0)
    write_gqrx("ANT+ (GFSK 1 Mbaud TDMA)", sig, fs, 2457e6)


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"writing to {OUT}")
    gen_umts_2100()
    gen_lte(2140e6, 600, 15.36e6, "4G LTE 2100 (10 MHz, Band 1)", 10e6)
    gen_lte(2620e6, 600, 15.36e6, "4G LTE 2600 (10 MHz, Band 7)", 10e6)
    gen_5g_nr()
    gen_wifi()
    gen_bt_classic()
    gen_ble()
    gen_802154("Zigbee (802.15.4 2 MHz)", 2425e6, slotting=0.004)
    gen_802154("Thread (802.15.4 mesh)", 2405e6, slotting=0.010)
    gen_802154("Matter (802.15.4)", 2410e6, slotting=0.008)
    gen_802154("WirelessHART (802.15.4 TDMA)", 2415e6, slotting=0.001)
    gen_802154("ISA100.11a (802.15.4 TDMA hop)", 2420e6, slotting=0.002)
    gen_802154("MiWi (proprietary 802.15.4)", 2430e6, slotting=0.006)
    gen_satphone()
    gen_wimax()
    gen_weather_radar()
    gen_lte_m()
    gen_nb_iot()
    gen_ant_plus()
    print("done")


if __name__ == "__main__":
    sys.exit(main())
