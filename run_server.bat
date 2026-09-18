@echo off
REM Voice Cloning System - watchdog server launcher.
REM Restarts uvicorn automatically if the process dies (low-RAM machines).
title Voice Cloning System

set PATH=D:\SOMU\voice-cloning-system\.venv\Lib\site-packages\torchcodec;%PATH%
set VC_HF_CACHE_DIR=D:\SOMU\voice-cloning-system\data\models\hf
set VC_QUANTIZED=1
set VC_MAX_LOADED_MODELS=1
set VC_NFE_STEPS=16
set VC_PRELOAD=1
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
REM fail fast on HuggingFace metadata checks (otherwise anonymous rate limits
REM can stall local cache hits for minutes on the first request after startup)
set HF_HUB_ETAG_TIMEOUT=5

:loop
echo [%date% %time%] starting server...
cd /d D:\SOMU\voice-cloning-system
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 >> server.log 2>&1
echo [%date% %time%] server exited - restarting in 5s
timeout /t 5 /nobreak >nul
goto loop
