#!/bin/bash

# macOS Apple Silicon paths
HOMEBREW_PREFIX="/opt/homebrew"

# Check if Intel Mac (older Homebrew location)
if [ "$(uname -m)" = "x86_64" ]; then
    HOMEBREW_PREFIX="/usr/local"
fi

echo "Building SDR Waterfall Analyzer..."
echo "Homebrew prefix: $HOMEBREW_PREFIX"

gcc -o sdr_waterfall sdr_waterfall.c \
    $(pkg-config --cflags --libs gtk+-3.0) \
    -I/opt/homebrew/include \
    -L/opt/homebrew/lib \
    -lfftw3f -lpthread -lm -O3

if [ $? -eq 0 ]; then
    echo "✓ Build successful!"
    echo "Run with: ./sdr_waterfall <gqrx_raw_file>"
else
    echo "✗ Build failed!"
    exit 1
fi
