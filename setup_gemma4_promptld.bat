@echo off
title Gemma4 PromptLD — Auto Setup
color 0A
setlocal enabledelayedexpansion

echo.
echo ╔══════════════════════════════════════════════════════╗
echo ║       Gemma4 PromptLD — Auto Setup                  ║
echo ║       by Brojachoeman                               ║
echo ╚══════════════════════════════════════════════════════╝
echo.

:: ── CONFIG ───────────────────────────────────────────────
set LLAMA_DIR=C:\llama
set MODELS_DIR=C:\models
set LLAMA_EXE=%LLAMA_DIR%\llama-server.exe

:: llama.cpp b9222 — CUDA 13.1 (RTX 5090 / latest drivers)
set LLAMA_VER=b9222
set LLAMA_ZIP=%LLAMA_DIR%\llama_main.zip
set CUDART_ZIP=%LLAMA_DIR%\llama_cudart.zip
set LLAMA_URL=https://github.com/ggml-org/llama.cpp/releases/download/b9222/llama-b9222-bin-win-cuda-13.1-x64.zip
set CUDART_URL=https://github.com/ggml-org/llama.cpp/releases/download/b9222/cudart-llama-bin-win-cuda-13.1-x64.zip

:: Model URLs
set GGUF_URL=https://huggingface.co/nohurry/gemma-4-26B-A4B-it-heretic-GUFF/resolve/main/gemma-4-26b-a4b-it-heretic.q4_k_m.gguf
set GGUF_FILE=%MODELS_DIR%\gemma-4-26b-a4b-it-heretic.q4_k_m.gguf
set MMPROJ_URL=https://huggingface.co/nohurry/gemma-4-26B-A4B-it-heretic-GUFF/resolve/main/gemma-4-26B-A4B-it-heretic-mmproj.bf16.gguf
:: MUST start with "mmproj" — llama-server scans for this prefix since b9196
set MMPROJ_FILE=%MODELS_DIR%\mmproj-gemma-4-26B-A4B-it-heretic-bf16.gguf
:: ─────────────────────────────────────────────────────────

echo [STEP 1/3] Checking llama-server...
echo.

:: Check if already in PATH
where llama-server >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ llama-server found in PATH — skipping install.
    goto :check_gguf
)

:: Check C:\llama
if exist "%LLAMA_EXE%" (
    echo ✅ llama-server found at %LLAMA_EXE% — skipping install.
    goto :check_gguf
)

:: Not found — download both zips
echo ⚠  llama-server not found. Downloading %LLAMA_VER% to %LLAMA_DIR%...
echo.

if not exist "%LLAMA_DIR%" mkdir "%LLAMA_DIR%"

echo [1/2] Downloading llama binaries...
curl -L --progress-bar -o "%LLAMA_ZIP%" "%LLAMA_URL%"
if %errorlevel% neq 0 ( echo ❌ Download failed. & pause & exit /b 1 )

echo.
echo [2/2] Downloading CUDA 13.1 runtime DLLs...
curl -L --progress-bar -o "%CUDART_ZIP%" "%CUDART_URL%"
if %errorlevel% neq 0 ( echo ❌ CUDA DLL download failed. & pause & exit /b 1 )

echo.
echo Extracting llama binaries...
powershell -NoProfile -Command "Expand-Archive -Path '%LLAMA_ZIP%' -DestinationPath '%LLAMA_DIR%' -Force"
if %errorlevel% neq 0 ( echo ❌ Extraction failed. & pause & exit /b 1 )

echo Extracting CUDA DLLs...
powershell -NoProfile -Command "Expand-Archive -Path '%CUDART_ZIP%' -DestinationPath '%LLAMA_DIR%' -Force"
if %errorlevel% neq 0 ( echo ❌ CUDA DLL extraction failed. & pause & exit /b 1 )

:: Flatten any subfolders
for /d %%D in ("%LLAMA_DIR%\*") do (
    echo Flattening %%D...
    move "%%D\*" "%LLAMA_DIR%\" >nul 2>&1
    rmdir "%%D" >nul 2>&1
)

del "%LLAMA_ZIP%" >nul 2>&1
del "%CUDART_ZIP%" >nul 2>&1

if exist "%LLAMA_EXE%" (
    echo ✅ llama-server %LLAMA_VER% installed at %LLAMA_EXE%
) else (
    echo ❌ llama-server.exe not found after extraction.
    echo    Check %LLAMA_DIR% manually.
    pause & exit /b 1
)

:check_gguf
echo.
echo [STEP 2/3] Checking GGUF model...
echo.

if not exist "%MODELS_DIR%" mkdir "%MODELS_DIR%"

if exist "%GGUF_FILE%" (
    echo ✅ Model GGUF already present — skipping.
    goto :check_mmproj
)

echo ⚠  Model not found in %MODELS_DIR%
echo.
echo Which quant do you want to download?
echo.
echo   [1] Q4_K_M  — ~15.7 GB  recommended, best quality/size balance
echo   [2] Q4_K_S  — ~15.9 GB  slightly smaller, similar quality
echo   [3] Skip    — I'll place the GGUF manually
echo.
set /p QUANT_CHOICE="Enter choice (1/2/3): "

if "%QUANT_CHOICE%" == "1" (
    set GGUF_URL=https://huggingface.co/nohurry/gemma-4-26B-A4B-it-heretic-GUFF/resolve/main/gemma-4-26b-a4b-it-heretic.q4_k_m.gguf
    set GGUF_FILE=%MODELS_DIR%\gemma-4-26b-a4b-it-heretic.q4_k_m.gguf
    echo Downloading Q4_K_M — ~15.7GB, grab a brew...
    goto :do_gguf_download
)
if "%QUANT_CHOICE%" == "2" (
    set GGUF_URL=https://huggingface.co/nohurry/gemma-4-26B-A4B-it-heretic-GUFF/resolve/main/gemma-4-26b-a4b-it-heretic.q4_k_s.gguf
    set GGUF_FILE=%MODELS_DIR%\gemma-4-26b-a4b-it-heretic.q4_k_s.gguf
    echo Downloading Q4_K_S — ~15.9GB, grab a brew...
    goto :do_gguf_download
)
echo Skipped. Place your GGUF in: %MODELS_DIR%
goto :check_mmproj

:do_gguf_download
echo.
curl -L --progress-bar -o "!GGUF_FILE!" "!GGUF_URL!"
if %errorlevel% neq 0 ( echo ❌ GGUF download failed. & pause & exit /b 1 )
echo ✅ Model downloaded.

:check_mmproj
echo.
if exist "%MMPROJ_FILE%" (
    echo ✅ mmproj already present — skipping.
    goto :check_python
)

echo ⚠  mmproj not found. Downloading (~1.2GB, enables image input)...
echo.
curl -L --progress-bar -o "%MMPROJ_FILE%" "%MMPROJ_URL%"
if %errorlevel% neq 0 (
    echo ⚠  mmproj download failed. Vision will not work but text prompting still will.
    echo    Grab manually from HuggingFace and save as:
    echo    %MMPROJ_FILE%
) else (
    echo ✅ mmproj downloaded — filename starts with "mmproj" for auto-detection.
)

:check_python
echo.
echo [STEP 3/3] Checking Python dependencies...
echo.

pip show requests >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing requests...
    pip install requests
) else (
    echo ✅ requests already installed.
)

:: Done
echo.
echo ╔══════════════════════════════════════════════════════╗
echo ║   ✅ Setup Complete!                                 ║
echo ╠══════════════════════════════════════════════════════╣
echo ║                                                      ║
echo ║   llama-server : C:\llama\llama-server.exe           ║
echo ║   Version      : b9222  (CUDA 13.1)                 ║
echo ║   Models folder: C:\models                          ║
echo ║   Vision       : mmproj-gemma-4-26B-...-bf16.gguf   ║
echo ║                                                      ║
echo ║   Next steps:                                        ║
echo ║   1. Restart ComfyUI                                 ║
echo ║   2. Add the Gemma4 Prompt Engineer node             ║
echo ║   3. Hit PREVIEW — node handles the rest             ║
echo ║                                                      ║
echo ╚══════════════════════════════════════════════════════╝
echo.
pause
