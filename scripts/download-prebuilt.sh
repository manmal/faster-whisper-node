#!/bin/bash
#
# Downloads prebuilt CTranslate2 binaries from PyPI wheels (no Python required)
#
# Usage:
#   ./scripts/download-prebuilt.sh [platform]
#
# Platforms: linux-x64, linux-arm64, macos-arm64, macos-x64, win-x64
#
set -e

CTRANSLATE2_VERSION="4.6.2"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LIB_DIR="$PROJECT_DIR/lib_build/lib"

# Detect platform if not specified
if [ -z "$1" ]; then
    case "$(uname -s)-$(uname -m)" in
        Linux-x86_64)   PLATFORM="linux-x64" ;;
        Linux-aarch64)  PLATFORM="linux-arm64" ;;
        Darwin-arm64)   PLATFORM="macos-arm64" ;;
        Darwin-x86_64)  PLATFORM="macos-x64" ;;
        MINGW*|MSYS*|CYGWIN*) PLATFORM="win-x64" ;;
        *)
            echo "❌ Unknown platform: $(uname -s)-$(uname -m)"
            echo "   Supported: linux-x64, linux-arm64, macos-arm64, macos-x64, win-x64"
            exit 1
            ;;
    esac
else
    PLATFORM="$1"
fi

# Map platform to PyPI wheel URL
case "$PLATFORM" in
    linux-x64)
        WHEEL_URL="https://files.pythonhosted.org/packages/f1/68/c9264cc4779af85e9eec3c7252f34543b0558ae4ea7df717de7be0cdb1bd/ctranslate2-${CTRANSLATE2_VERSION}-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"
        LIB_DIR_IN_WHEEL="ctranslate2.libs"
        ;;
    linux-arm64)
        WHEEL_URL="https://files.pythonhosted.org/packages/26/5a/e8623d7424af6e5c3103dcf1b14a1cb38641e948a7e6c6d8ae9f1ed04445/ctranslate2-${CTRANSLATE2_VERSION}-cp310-cp310-manylinux2014_aarch64.manylinux_2_17_aarch64.whl"
        LIB_DIR_IN_WHEEL="ctranslate2.libs"
        ;;
    macos-arm64)
        WHEEL_URL="https://files.pythonhosted.org/packages/e9/83/88b5d923d16f308e7986eb6d815ca9fe204df78873960f7ae3c685093705/ctranslate2-${CTRANSLATE2_VERSION}-cp310-cp310-macosx_11_0_arm64.whl"
        LIB_DIR_IN_WHEEL="ctranslate2/.dylibs"
        ;;
    macos-x64)
        WHEEL_URL="https://files.pythonhosted.org/packages/34/58/454ecc2406828acd923444c339998322465eab79118c08abff8a3549f935/ctranslate2-${CTRANSLATE2_VERSION}-cp310-cp310-macosx_11_0_x86_64.whl"
        LIB_DIR_IN_WHEEL="ctranslate2/.dylibs"
        ;;
    win-x64)
        WHEEL_URL="https://files.pythonhosted.org/packages/20/dc/ec159aa02f5d6651aa09d665bb106d4b9163efc74562f2a87783ae72c999/ctranslate2-${CTRANSLATE2_VERSION}-cp310-cp310-win_amd64.whl"
        LIB_DIR_IN_WHEEL="ctranslate2"
        ;;
    *)
        echo "❌ Unknown platform: $PLATFORM"
        echo "   Supported: linux-x64, linux-arm64, macos-arm64, macos-x64, win-x64"
        exit 1
        ;;
esac

echo "📦 Downloading CTranslate2 ${CTRANSLATE2_VERSION} for ${PLATFORM}..."

# Create temp directory
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Download wheel
WHEEL_FILE="$TEMP_DIR/ctranslate2.whl"
curl -sL "$WHEEL_URL" -o "$WHEEL_FILE"
echo "   Downloaded wheel ($(du -h "$WHEEL_FILE" | cut -f1))"

# Extract
unzip -q "$WHEEL_FILE" -d "$TEMP_DIR/extracted"
echo "   Extracted wheel contents"

# Create lib directory
mkdir -p "$LIB_DIR"

# Copy libraries based on platform
case "$PLATFORM" in
    win-x64)
        # Windows: DLLs are in ctranslate2/
        cp "$TEMP_DIR/extracted/$LIB_DIR_IN_WHEEL/"*.dll "$LIB_DIR/"
        echo "   Copied DLLs to $LIB_DIR"
        ;;
    macos-*)
        # macOS: dylib is in ctranslate2/.dylibs/
        MAIN_LIB=$(find "$TEMP_DIR/extracted/$LIB_DIR_IN_WHEEL" -name "libctranslate2*.dylib" | head -1)
        if [ -n "$MAIN_LIB" ]; then
            cp "$MAIN_LIB" "$LIB_DIR/libctranslate2.${CTRANSLATE2_VERSION}.dylib"
            cd "$LIB_DIR"
            
            # Fix the install name so it can be found via @rpath or @loader_path
            install_name_tool -id "@rpath/libctranslate2.dylib" "libctranslate2.${CTRANSLATE2_VERSION}.dylib"
            
            # Re-sign the library (code signature invalidated by install_name_tool)
            codesign --force -s - "libctranslate2.${CTRANSLATE2_VERSION}.dylib" 2>/dev/null || true
            
            ln -sf "libctranslate2.${CTRANSLATE2_VERSION}.dylib" "libctranslate2.4.dylib"
            ln -sf "libctranslate2.4.dylib" "libctranslate2.dylib"
            cd - > /dev/null
        fi
        echo "   Copied and fixed dylib install name in $LIB_DIR"
        ;;
    linux-*)
        # Linux: .so files are in ctranslate2.libs/
        MAIN_LIB=$(find "$TEMP_DIR/extracted/$LIB_DIR_IN_WHEEL" -name "libctranslate2*.so.*" | head -1)
        if [ -n "$MAIN_LIB" ]; then
            cp "$MAIN_LIB" "$LIB_DIR/libctranslate2.so.${CTRANSLATE2_VERSION}"
            cd "$LIB_DIR"
            ln -sf "libctranslate2.so.${CTRANSLATE2_VERSION}" "libctranslate2.so.4"
            ln -sf "libctranslate2.so.4" "libctranslate2.so"
            cd - > /dev/null
        fi
        
        # Copy OpenMP runtime if present
        GOMP_LIB=$(find "$TEMP_DIR/extracted/$LIB_DIR_IN_WHEEL" -name "libgomp*.so.*" | head -1)
        if [ -n "$GOMP_LIB" ]; then
            cp "$GOMP_LIB" "$LIB_DIR/libgomp.so.1"
        fi
        echo "   Copied shared libraries to $LIB_DIR"
        ;;
esac

# Copy headers from CTranslate2 source (if available)
if [ -d "$PROJECT_DIR/CTranslate2/include" ]; then
    mkdir -p "$PROJECT_DIR/lib_build/include"
    cp -r "$PROJECT_DIR/CTranslate2/include/"* "$PROJECT_DIR/lib_build/include/"
    echo "   Copied headers from CTranslate2 source"
fi

echo ""
echo "✅ CTranslate2 ${CTRANSLATE2_VERSION} ready for ${PLATFORM}"
ls -la "$LIB_DIR"
