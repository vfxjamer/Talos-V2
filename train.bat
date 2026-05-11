@echo off
REM ============================================================================
REM Talon V2 - Build and Run Script (Windows)
REM ============================================================================

echo.
echo ============================================================
echo  TALON V2 - Building and Running
echo ============================================================
echo.

REM Check if build directory exists
if not exist build (
    echo [INFO] Creating build directory...
    mkdir build
)

REM Navigate to build directory
cd /d "%~dp0build"

REM Configure with CMake
echo [INFO] Running CMake configuration...
cmake .. -G "Visual Studio 17 2022" -A x64
if errorlevel 1 (
    echo [ERROR] CMake configuration failed!
    pause
    exit /b 1
)

REM Build the project
echo [INFO] Building TalonV2Bot...
cmake --build . --config Release
if errorlevel 1 (
    echo [ERROR] Build failed!
    pause
    exit /b 1
)

echo [INFO] Build successful!

REM Copy collision_meshes to bin directory if needed
if not exist "build\bin\Release\collision_meshes" (
    if exist "collision_meshes" (
        echo [INFO] Copying collision_meshes to build directory...
        xcopy /E /I /Y "collision_meshes" "build\bin\Release\collision_meshes"
    )
)

REM Go back to project root
cd /d "%~dp0"

REM Check if executable exists
if not exist "build\bin\Release\TalonV2Bot.exe" (
    echo [ERROR] Executable not found!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Starting Training...
echo ============================================================
echo.

REM Run the bot
"build\bin\Release\TalonV2Bot.exe"

echo.
echo [INFO] Training session ended.
pause