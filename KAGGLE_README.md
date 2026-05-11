# Talon V2 - Kaggle Setup Guide

## Files to Upload to Kaggle

Upload these folders to your Kaggle working directory:

1. **`Talos V2/`** - The entire bot folder including:
   - `src/ExampleMain.cpp`
   - `CMakeLists.txt`
   - `collision_meshes/`
   - `kaggle_train.sh`
   - `kaggle_auto_resume.sh`

2. **`GigaLearnCPP-Leak/`** - Required dependency
   - Must be in same parent folder as Talos V2

## Directory Structure on Kaggle

```
/kaggle/working/
├── Talos V2/
│   ├── src/ExampleMain.cpp
│   ├── CMakeLists.txt
│   ├── kaggle_train.sh
│   ├── kaggle_auto_resume.sh
│   ├── collision_meshes/
│   └── KAGGLE_README.md
├── GigaLearnCPP-Leak/
│   ├── GigaLearnCPP/
│   ├── RLGymCPP/
│   ├── RLBotCPP/
│   ├── libtorch/
│   └── collision_meshes/
└── checkpoints/       (auto-created)
```

## Running on Kaggle

### Option 1: Using the auto-resume script (RECOMMENDED)
This handles the 12-hour session limit automatically:

```bash
cd /kaggle/working/Talos\ V2
chmod +x kaggle_auto_resume.sh
./kaggle_auto_resume.sh --bot ./build/TalonV2Bot
```

### Option 2: Using the simple build script
```bash
cd /kaggle/working/Talos\ V2
chmod +x kaggle_train.sh
./kaggle_train.sh
```

### Option 3: Manual build
```bash
cd /kaggle/working/Talos\ V2
mkdir -p build
cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release

# Copy collision_meshes
mkdir -p collision_meshes
cp -r ../collision_meshes/* collision_meshes/

# Run
cd ..
./build/TalonV2Bot
```

## Command Line Options

- `--cpu` - Force CPU mode (if CUDA not available)
- `--no-cuda` - Use AUTO device detection instead of GPU_CUDA
- `--mesh-path <path>` - Override collision_meshes path

Example:
```bash
./build/TalonV2Bot --mesh-path "../GigaLearnCPP-Leak/collision_meshes"
```

## 12-Hour Session Limit Workaround

The `kaggle_auto_resume.sh` script handles Kaggle's 12-hour session limit:

### How it works:
1. Starts training and runs in background
2. Monitors elapsed time every 30 seconds
3. At ~11 hours 40 minutes, gracefully stops the bot
4. On restart, automatically loads from latest checkpoint
5. Continues until training completes (48B steps)

### Key features:
- Built-in checkpoint system saves every 5M steps
- Auto-detects latest checkpoint on restart
- Logs progress every 5 minutes
- Handles graceful shutdown properly

### Customization:
```bash
# Custom runtime limit (default: 11h40m = 42000 seconds)
./kaggle_auto_resume.sh --bot ./build/TalonV2Bot --time 36000

# Custom bot path
./kaggle_auto_resume.sh --bot /custom/path/TalonV2Bot

# Disable auto-resume (single session only)
# Just run the bot directly without the script
```

## Expected Behavior on Kaggle

- **Uses GPU_CUDA** (Kaggle has CUDA available)
- **Network:** 1024x4 LEAKY_RELU + LayerNorm (~12M parameters)
- **Checkpoints:** Saves to `checkpoints/` every 5M steps
- **Phase 1:** Kickoff Mastery (0-1B steps)
- **Auto-resume:** Handles session limits automatically

## Training Progress

The bot progresses through 10 phases:
- Phase 1 (0-1B): Kickoff Mastery
- Phase 2 (1B-3B): Ball Control
- Phase 3 (3B-7B): Basic Gameplay
- Phase 4 (7B-16B): Aerial Introduction
- Phase 5 (16B-26B): Aerial Mastery
- Phase 6 (26B-32B): Game Sense Introduction
- Phase 7 (32B-40B): Game Sense Mastery
- Phase 8 (40B-44B): Mechanical Ceiling
- Phase 9 (44B-47B): Full Games
- Phase 10 (47B-48B): Optimization

## Memory Requirements

- **VRAM:** ~2-4GB (12M parameters, 64 game environments)
- **RAM:** ~4-8GB

## Notes

- The bot auto-detects phase from checkpoint on restart
- Use `--cpu` only if GPU not available
- Checkpoints folder should be preserved between sessions