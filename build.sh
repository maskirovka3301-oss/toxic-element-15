#!/bin/bash

# macOS Apple Silicon paths
HOMEBREW_PREFIX="/opt/homebrew"

# Check if Intel Mac (older Homebrew location)
if [ "$(uname -m)" = "x86_64" ]; then
    HOMEBREW_PREFIX="/usr/local"
fi

echo "Building SDR Waterfall Analyzer..."
echo "Homebrew prefix: $HOMEBREW_PREFIX"

gcc -o te15 te15.c \
    $(pkg-config --cflags --libs gtk+-3.0) \
    -I${HOMEBREW_PREFIX}/include \
    -L${HOMEBREW_PREFIX}/lib \
    -lfftw3f -lGL -lpthread -lm -O3 \
    -Wno-deprecated-declarations

if [ $? -eq 0 ]; then
    echo "✓ Build successful!"
    echo "Run with: ./te15 <gqrx_raw_file>"
else
    echo "✗ Build failed!"
    exit 1
fi
