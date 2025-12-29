#!/bin/bash
#
# Build faster-whisper-node for macOS
#
# This script downloads prebuilt CTranslate2 libraries from PyPI and builds the Rust bindings.
# No need to build CTranslate2 from source!
#
set -e

cd "$(dirname "$0")/.."

# 1. Download prebuilt CTranslate2 libraries
echo "📦 Downloading prebuilt CTranslate2 libraries..."
./scripts/download-prebuilt.sh

# 2. Build Rust bindings
echo ""
echo "🔨 Building Rust bindings..."
export LIBRARY_PATH="$PWD/lib_build/lib:$LIBRARY_PATH"
export CMAKE_LIBRARY_PATH="$PWD/lib_build/lib:$CMAKE_LIBRARY_PATH"
npm run build:rust

# 3. Copy libraries to output directory
echo ""
echo "📁 Copying libraries..."
npm run postbuild

echo ""
echo "✅ Build complete!"
echo "   Test with: npm test"
