#!/bin/bash
# ============================================================================
# Talon V2 - Build and Run Script for Kaggle (P100 GPU)
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "============================================================"
echo "  TALON V2 - Building for Kaggle (P100 GPU)"
echo "============================================================"
echo ""

# Create build directory
mkdir -p build
cd build

# Configure with Ninja for faster builds
echo "[INFO] Configuring CMake..."
if command -v ninja &> /dev/null; then
    cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCHITECTURES=60
else
    cmake .. -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
fi

# Build
echo "[INFO] Building TalonV2Bot..."
cmake --build . --config Release -- -j$(nproc)

echo "[INFO] Build successful!"

# Go back to project root
cd ..

# Setup collision_meshes if not already done
if [ ! -d "build/collision_meshes/soccar" ]; then
    echo "[INFO] Setting up collision_meshes..."
    mkdir -p build/collision_meshes
    if [ -d "collision_meshes" ]; then
        cp -r collision_meshes/* build/collision_meshes/
    elif [ -d "../GigaLearnCPP-Leak/collision_meshes" ]; then
        cp -r ../GigaLearnCPP-Leak/collision_meshes/* build/collision_meshes/
    fi
fi

# Create checkpoints directory
mkdir -p checkpoints

echo ""
echo "============================================================"
echo "  Starting Training on P100 GPU..."
echo "============================================================"
echo ""

# Run the bot - uses GPU_CUDA by default for P100
./build/TalonV2Bot

echo ""
echo "[INFO] Training session ended."