@echo off
title Gemma4 PromptLD — Setup
color 0A
setlocal enabledelayedexpansion

echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║        Gemma4 PromptLD — Auto Setup                     ║
echo  ║        by Brojachoeman / LoRa-Daddy                     ║
echo  ╚══════════════════════════════════════════════════════════╝
echo.

:: ── CONFIG ────────────────────────────────────────────────────────────────
set LLAMA_DIR=C:\llama
set MODELS_DIR=C:\models
set LLAMA_EXE=%LLAMA_DIR%\llama-server.exe
set LLAMA_BUILD=b9009
set CUDA_VER=13.1
set LLAMA_BIN_URL=https://github.com/ggml-org/llama.cpp/releases/download/%LLAMA_BUILD%/llama-%LLAMA_BUILD%-bin-win-cuda-%CUDA_VER%-x64.zip
set LLAMA_CUDART_URL=https://github.com/ggml-org/llama.cpp/releases/download/%LLAMA_BUILD%/cudart-llama-bin-win-cuda-%CUDA_VER%-x64.zip
set LLAMA_BIN_ZIP=%LLAMA_DIR%\llama_bin.zip
set LLAMA_CUDART_ZIP=%LLAMA_DIR%\llama_cudart.zip
set HF_REPO=https://huggingface.co/nohurry/gemma-4-26B-A4B-it-heretic-GUFF/resolve/main
set MMPROJ_FILE=%MODELS_DIR%\gemma-4-26B-A4B-it-heretic-mmproj.f16.gguf
set MMPROJ_URL=%HF_REPO%/gemma-4-26B-A4B-it-heretic-mmproj.f16.gguf
:: ──────────────────────────────────────────────────────────────────────────


:: ════════════════════════════════════════════════════════════════
:: STEP 1 — Detect VRAM and recommend model
:: ════════════════════════════════════════════════════════════════
echo  [1/4] Detecting GPU VRAM...
echo.

set VRAM_INT=0

:: Try nvidia-smi first — most accurate
nvidia-smi >nul 2>&1
if %errorlevel% == 0 (
    for /f "skip=1 tokens=*" %%S in ('nvidia-smi --query-gpu^=memory.total --format^=csv^,noheader^,nounits 2^>nul') do (
        set VRAM_MiB=%%S
        set /a VRAM_INT=!VRAM_MiB! / 1024
        goto :vram_done
    )
)

:: Fallback — WMI via PowerShell
for /f "tokens=*" %%V in ('powershell -NoProfile -Command "try { [math]::Round((Get-WmiObject Win32_VideoController | Where-Object { $_.Name -like '*NVIDIA*' } | Select-Object -First 1).AdapterRAM / 1GB) } catch { 0 }" 2^>nul') do (
    set VRAM_INT=%%V
)

:vram_done
if "%VRAM_INT%"=="0" (
    echo  ⚠  Could not detect VRAM automatically.
    set VRAM_INT=0
)
echo  GPU VRAM detected: ~%VRAM_INT% GB
echo.

:: ── Model recommendation ──────────────────────────────────────
:: Quant sizes + headroom for KV cache:
:: q3_k_s  ~12GB  → 14GB+ needed
:: q4_k_s  ~16GB  → 18GB+ needed
:: q4_k_m  ~17GB  → 19GB+ needed
:: q5_k_m  ~21GB  → 23GB+ needed  ← 24GB sweet spot
:: q8_0    ~29GB  → 32GB+ needed

set RECOMMENDED_QUANT=q4_k_m

if %VRAM_INT% GEQ 32 (
    set RECOMMENDED_QUANT=q8_0
    set REC_SIZE=~29 GB
    set REC_REASON=32GB VRAM - maximum quality
)
if %VRAM_INT% GEQ 23 if %VRAM_INT% LSS 32 (
    set RECOMMENDED_QUANT=q5_k_m
    set REC_SIZE=~21 GB
    set REC_REASON=24GB VRAM sweet spot
)
if %VRAM_INT% GEQ 19 if %VRAM_INT% LSS 23 (
    set RECOMMENDED_QUANT=q4_k_m
    set REC_SIZE=~17 GB
    set REC_REASON=20-22GB - excellent balance
)
if %VRAM_INT% GEQ 18 if %VRAM_INT% LSS 19 (
    set RECOMMENDED_QUANT=q4_k_s
    set REC_SIZE=~16 GB
    set REC_REASON=18-19GB - compact q4
)
if %VRAM_INT% LSS 18 if %VRAM_INT% GTR 0 (
    set RECOMMENDED_QUANT=q3_k_s
    set REC_SIZE=~12 GB
    set REC_REASON=16GB or less - smallest option
)
if %VRAM_INT%==0 (
    set REC_SIZE=~17 GB
    set REC_REASON=VRAM unknown - safe default
)

echo  Recommendation for your GPU (%VRAM_INT%GB VRAM^):
echo    %RECOMMENDED_QUANT% — %REC_SIZE% — %REC_REASON%
echo.
echo  All options:
echo    [1] q3_k_s  ~12 GB  (minimum 14GB VRAM)
echo    [2] q4_k_s  ~16 GB  (minimum 18GB VRAM)
echo    [3] q4_k_m  ~17 GB  (minimum 19GB VRAM)
echo    [4] q5_k_m  ~21 GB  (minimum 23GB VRAM)  ^<-- 24GB sweet spot
echo    [5] q8_0    ~29 GB  (minimum 32GB VRAM)
echo.
set /p MODEL_CHOICE="  Enter number or press Enter to use recommended [%RECOMMENDED_QUANT%]: "

if "%MODEL_CHOICE%"=="" set CHOSEN_QUANT=%RECOMMENDED_QUANT%
if "%MODEL_CHOICE%"=="1" set CHOSEN_QUANT=q3_k_s
if "%MODEL_CHOICE%"=="2" set CHOSEN_QUANT=q4_k_s
if "%MODEL_CHOICE%"=="3" set CHOSEN_QUANT=q4_k_m
if "%MODEL_CHOICE%"=="4" set CHOSEN_QUANT=q5_k_m
if "%MODEL_CHOICE%"=="5" set CHOSEN_QUANT=q8_0
if not defined CHOSEN_QUANT set CHOSEN_QUANT=%RECOMMENDED_QUANT%

set GGUF_FILENAME=gemma-4-26b-a4b-it-heretic.%CHOSEN_QUANT%.gguf
set GGUF_FILE=%MODELS_DIR%\%GGUF_FILENAME%
set GGUF_URL=%HF_REPO%/%GGUF_FILENAME%

echo.
echo  Downloading: %CHOSEN_QUANT% — %GGUF_FILENAME%
echo.


:: ════════════════════════════════════════════════════════════════
:: STEP 2 — llama-server
:: ════════════════════════════════════════════════════════════════
echo  [2/4] Checking llama-server...
echo.

if exist "%LLAMA_EXE%" (
    echo  ✅ llama-server.exe already present — skipping.
    goto :check_model
)

where llama-server >nul 2>&1
if %errorlevel% == 0 (
    echo  ✅ llama-server found in PATH — skipping.
    goto :check_model
)

echo  ⬇  Downloading llama.cpp %LLAMA_BUILD% cu%CUDA_VER% (RTX 30/40/50 series)...
echo.

if not exist "%LLAMA_DIR%" mkdir "%LLAMA_DIR%"

echo  Downloading main binaries...
curl -L --progress-bar --retry 3 --retry-delay 5 -o "%LLAMA_BIN_ZIP%" "%LLAMA_BIN_URL%"
if %errorlevel% neq 0 (
    echo.
    echo  ❌ Failed. Manual link:
    echo     %LLAMA_BIN_URL%
    pause & exit /b 1
)

echo.
echo  Downloading CUDA runtime (separate package required since b9008^)...
curl -L --progress-bar --retry 3 --retry-delay 5 -o "%LLAMA_CUDART_ZIP%" "%LLAMA_CUDART_URL%"
if %errorlevel% neq 0 (
    echo.
    echo  ❌ Failed. Manual link:
    echo     %LLAMA_CUDART_URL%
    pause & exit /b 1
)

echo.
echo  Extracting...
powershell -NoProfile -Command "Expand-Archive -Path '%LLAMA_BIN_ZIP%' -DestinationPath '%LLAMA_DIR%\_bin_tmp' -Force"
if %errorlevel% neq 0 ( echo  ❌ Extraction failed. & pause & exit /b 1 )

powershell -NoProfile -Command "Expand-Archive -Path '%LLAMA_CUDART_ZIP%' -DestinationPath '%LLAMA_DIR%\_cudart_tmp' -Force"
if %errorlevel% neq 0 ( echo  ❌ CUDA runtime extraction failed. & pause & exit /b 1 )

echo  Flattening files to %LLAMA_DIR%...
for /r "%LLAMA_DIR%\_bin_tmp" %%F in (*) do move /y "%%F" "%LLAMA_DIR%\" >nul 2>&1
for /r "%LLAMA_DIR%\_cudart_tmp" %%F in (*) do move /y "%%F" "%LLAMA_DIR%\" >nul 2>&1

rmdir /s /q "%LLAMA_DIR%\_bin_tmp"    >nul 2>&1
rmdir /s /q "%LLAMA_DIR%\_cudart_tmp" >nul 2>&1
del "%LLAMA_BIN_ZIP%"    >nul 2>&1
del "%LLAMA_CUDART_ZIP%" >nul 2>&1

if exist "%LLAMA_EXE%" (
    echo  ✅ llama-server installed at %LLAMA_EXE%
) else (
    echo.
    echo  ❌ llama-server.exe not found after extraction.
    echo     Check %LLAMA_DIR% manually.
    echo     Bin link   : %LLAMA_BIN_URL%
    echo     CUDA link  : %LLAMA_CUDART_URL%
    pause & exit /b 1
)


:: ════════════════════════════════════════════════════════════════
:: STEP 3 — Model GGUF
:: ════════════════════════════════════════════════════════════════
:check_model
echo.
echo  [3/4] Checking model GGUF (%CHOSEN_QUANT%)...
echo.

if not exist "%MODELS_DIR%" mkdir "%MODELS_DIR%"

if exist "%GGUF_FILE%" (
    echo  ✅ %GGUF_FILENAME% already present — skipping.
    goto :check_mmproj
)

echo  ⬇  Downloading %GGUF_FILENAME%...
echo     This is large — grab a brew.
echo.
curl -L --progress-bar --retry 3 --retry-delay 10 -o "%GGUF_FILE%" "%GGUF_URL%"
if %errorlevel% neq 0 (
    echo.
    echo  ❌ Download failed.
    echo     %GGUF_URL%
    echo     Download manually and place in %MODELS_DIR%
    pause & exit /b 1
)
echo  ✅ Model downloaded.


:: ════════════════════════════════════════════════════════════════
:: STEP 4 — mmproj (vision)
:: ════════════════════════════════════════════════════════════════
:check_mmproj
echo.
echo  [4/4] Checking mmproj (image input)...
echo.

if exist "%MMPROJ_FILE%" (
    echo  ✅ mmproj already present — vision enabled.
    goto :done
)

echo  ⬇  Downloading mmproj f16 — ~1.2GB...
curl -L --progress-bar --retry 3 --retry-delay 5 -o "%MMPROJ_FILE%" "%MMPROJ_URL%"
if %errorlevel% neq 0 (
    echo.
    echo  ⚠  mmproj download failed — text prompting still works, vision won't.
    echo     Manual: %MMPROJ_URL%
    goto :done
)
echo  ✅ mmproj downloaded — vision enabled.


:: ════════════════════════════════════════════════════════════════
:: DONE
:: ════════════════════════════════════════════════════════════════
:done
echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║   ✅ Setup Complete                                     ║
echo  ╠══════════════════════════════════════════════════════════╣
echo  ║                                                         ║
echo  ║   llama-server : C:\llama\llama-server.exe              ║
echo  ║   Model        : %GGUF_FILENAME%
echo  ║   mmproj       : gemma-4-26B-A4B-it-heretic-mmproj.f16 ║
echo  ║   CUDA build   : %LLAMA_BUILD% cu%CUDA_VER% (RTX 30/40/50 series)    ║
echo  ║                                                         ║
echo  ║   Next steps:                                           ║
echo  ║   1. Restart ComfyUI                                    ║
echo  ║   2. Add the Gemma4 PromptLD node                       ║
echo  ║   3. Hit PREVIEW — node handles the rest                ║
echo  ║                                                         ║
echo  ╚══════════════════════════════════════════════════════════╝
echo.
pause
