@echo off
REM hashmap_builder.bat — One-click word frequency hashmap builder (Windows)
REM
REM What this script does:
REM   1. Builds the C++ project (if not already built)
REM   2. Ingests all local text files in datasets\
REM   3. Streams the FineWeb dataset from HuggingFace
REM   4. Saves the combined word frequency hashmap to word_freqs.ddfreq
REM
REM Usage:
REM   scripts\hashmap_builder.bat

setlocal enabledelayedexpansion

REM Resolve project root (script lives in scripts\)
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "PROJECT_ROOT=%%~fI"

set "BUILD_DIR=%PROJECT_ROOT%\build"
set "DATASETS_DIR=%PROJECT_ROOT%\datasets"
set "OUTPUT_FILE=%PROJECT_ROOT%\word_freqs.ddfreq"
set "BUILD_FREQS=%BUILD_DIR%\Release\build_freqs.exe"
set "BUILD_FREQS_DBG=%BUILD_DIR%\Debug\build_freqs.exe"

echo ==========================================
echo   ddtokens — Hashmap Builder
echo ==========================================
echo.

REM --- Step 1: Build the C++ project ---
echo [1/3] Building C++ project...
cmake -B "%BUILD_DIR%" -S "%PROJECT_ROOT%" > nul 2>&1
cmake --build "%BUILD_DIR%" --target build_freqs --config Release > nul 2>&1

REM Find the executable (Release or Debug)
if exist "%BUILD_FREQS%" (
    set "EXE=%BUILD_FREQS%"
) else if exist "%BUILD_FREQS_DBG%" (
    set "EXE=%BUILD_FREQS_DBG%"
) else (
    REM Try build dir root (single-config generators like Ninja/MinGW)
    set "EXE=%BUILD_DIR%\build_freqs.exe"
)

if not exist "!EXE!" (
    echo   ERROR: build_freqs.exe not found. Check CMake build output.
    exit /b 1
)

echo   √ build_freqs compiled
echo.

REM --- Step 2: Collect all local dataset files ---
set "FILE_COUNT=0"
set "LOCAL_FILES="

if exist "%DATASETS_DIR%" (
    for %%F in ("%DATASETS_DIR%\*.txt") do (
        set /a FILE_COUNT+=1
        set "LOCAL_FILES=!LOCAL_FILES! "%%F""
    )
)

if %FILE_COUNT% gtr 0 (
    echo [2/3] Found %FILE_COUNT% local dataset file(s):
    for %%F in ("%DATASETS_DIR%\*.txt") do (
        echo   * %%~nxF
    )
) else (
    echo [2/3] No local .txt files found in datasets\
)
echo.

REM --- Step 3: Stream FineWeb + ingest local files ---
echo [3/3] Building hashmap (local files + FineWeb stream)...
echo   Output: %OUTPUT_FILE%
echo.

REM Pipe FineWeb through stdin, pass local files as args, "-" reads the pipe.
python "%SCRIPT_DIR%stream_fineweb.py" | "!EXE!" "%OUTPUT_FILE%" !LOCAL_FILES! -

echo.
echo ==========================================
echo   √ Hashmap saved to: %OUTPUT_FILE%
echo.
echo   Next step — run BPE training:
echo   build\Release\train_bpe.exe ^<num_merges^> %OUTPUT_FILE%
echo ==========================================

endlocal
