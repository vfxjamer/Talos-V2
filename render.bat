@echo off
REM ============================================
REM Talon V2 Render Launcher
REM ============================================

set PYTHONPATH=%~dp0build

cd /d "%~dp0"
build\bin\Release\TalonV2Bot.exe --render

pause