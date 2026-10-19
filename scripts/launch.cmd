@echo off
setlocal
cd /d "%~dp0.."
set "TOKENIZER_LAB_OFFLINE=1"
set "HF_HUB_OFFLINE=1"
set "HF_DATASETS_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "PYTHONNOUSERSITE=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "TOKENIZERS_PARALLELISM=false"
set "PYTHONPATH=%CD%\src;%CD%\vendor"
if not exist "runtime\python.exe" (
  echo The bundled Python runtime is missing. Extract the complete Windows offline kit.
  pause
  exit /b 1
)
"runtime\python.exe" -m tokenizer_lab.cli launch %*
if errorlevel 1 pause
