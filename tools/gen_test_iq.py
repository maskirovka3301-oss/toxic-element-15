#!/usr/bin/env python3
"""Generate a synthetic gqrx-style IQ recording for manual testing.

Signals (centre 2.53 GHz, rate 2 MS/s):
  - CW tone at centre + 500 kHz
  - narrowband FM at centre - 300 kHz (50 kHz deviation, 1 kHz modulation)
  - pulse train at centre + 800 kHz (100 us on, 1 ms period)
  - low noise floor

Usage: python3 tools/gen_test_iq.py [out.raw]
"""
import sys

import numpy as np

RATE = 2_000_000.0
CENTER = 2_530_000_000.0
DURATION = 3.0
OUT = "data/gqrx_20260306_174402_2530000000_2000000_fc.raw"


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else OUT
    n = int(RATE * DURATION)
    t = np.arange(n, dtype=np.float64) / RATE

    sig = np.zeros(n, dtype=np.complex128)

    # CW tone: +500 kHz.
    sig += 0.5 * np.exp(2j * np.pi * 500_000.0 * t)

    # Narrowband FM: carrier -300 kHz, 50 kHz deviation, 1 kHz tone.
    phase = 2 * np.pi * (-300_000.0) * t + (50_000.0 / 1_000.0) * np.sin(
        2 * np.pi * 1_000.0 * t
    )
    sig += 0.4 * np.exp(1j * phase)

    # Pulse train: +800 kHz, 100 us on / 1 ms period.
    # (Kept below the 1 MHz Nyquist limit of the 2 MS/s stream.)
    gate = ((t % 1e-3) < 100e-6).astype(np.float64)
    sig += 0.7 * gate * np.exp(2j * np.pi * 800_000.0 * t)

    # Noise floor.
    rng = np.random.default_rng(42)
    sig += 0.05 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))

    iq = np.empty(2 * n, dtype=np.float32)
    iq[0::2] = sig.real.astype(np.float32)
    iq[1::2] = sig.imag.astype(np.float32)

    iq.tofile(out)
    mib = iq.nbytes / (1024 * 1024)
    print(
        f"wrote {out}: {DURATION:.1f}s at {RATE / 1e6:.0f} MS/s, "
        f"{mib:.1f} MiB"
    )


if __name__ == "__main__":
    main()
