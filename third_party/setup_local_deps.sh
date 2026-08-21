#!/bin/bash
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
THIRD_PARTY="$ROOT/third_party"

QDK_DIR="$THIRD_PARTY/qdk"
QUALTRAN_DIR="$THIRD_PARTY/Qualtran"

QDK_REPO="https://github.com/KevinLi053/qdk.git"
QDK_COMMIT="65e105a1da3b8b32513d111e707ad2e9045f26f0"

QUALTRAN_REPO="https://github.com/KevinLi053/Qualtran.git"
QUALTRAN_COMMIT="61e9261ae36ab75ee5cfe528b622609d165af974"

echo "Setting up local QDK and Qualtran..."

# Check virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "ERROR: Please activate your virtual environment first."
    exit 1
fi

# Clone QDK
if [ ! -d "$QDK_DIR" ]; then
    git clone "$QDK_REPO" "$QDK_DIR"
fi

cd "$QDK_DIR"
git checkout "$QDK_COMMIT"

# Clone Qualtran
if [ ! -d "$QUALTRAN_DIR" ]; then
    git clone "$QUALTRAN_REPO" "$QUALTRAN_DIR"
fi

cd "$QUALTRAN_DIR"
git checkout "$QUALTRAN_COMMIT"

# Install/build Qualtran
python -m pip install -e "$QUALTRAN_DIR"

# Build and install modified QDK
cd "$QDK_DIR"
maturin develop \
    --manifest-path source/qdk_package/Cargo.toml

# Install this project in editable mode
pip install -e "$ROOT"

echo "Done!"