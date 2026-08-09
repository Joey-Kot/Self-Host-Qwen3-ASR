@echo off
cd /d "%~dp0"
set HF_HOME=%CD%\.cache\huggingface
set TRANSFORMERS_CACHE=%CD%\.cache\huggingface\transformers
set TORCH_HOME=%CD%\.cache\torch
set MODELSCOPE_CACHE=%CD%\.cache\modelscope

rem Model loading policy:
rem 1) ASR_MODEL, if explicitly set, wins.
rem 2) Otherwise server.py loads .\models\Qwen3-ASR-1.7B when it is complete.
rem 3) If the local model is missing/incomplete, it falls back to Qwen/Qwen3-ASR-1.7B.
if not defined ASR_LOCAL_MODEL_DIR set ASR_LOCAL_MODEL_DIR=%CD%\models\Qwen3-ASR-1.7B
if not defined ASR_LOCAL_ALIGNER_DIR set ASR_LOCAL_ALIGNER_DIR=%CD%\models\Qwen3-ForcedAligner-0.6B

rem enable_lid controls whether language metadata is returned.
rem Leave ASR_DEFAULT_LANGUAGE empty to preserve Qwen auto-detection for recognition.
rem Set it to Chinese/English/etc. only if you want LID-off requests to force a language.
if not defined ASR_MAX_INFERENCE_BATCH_SIZE set ASR_MAX_INFERENCE_BATCH_SIZE=8
if not defined ASR_DTYPE set ASR_DTYPE=float16
rem Set to 0 only when timestamp/SRT output is intentionally unavailable.
if not defined ASR_ENABLE_TIMESTAMPS set ASR_ENABLE_TIMESTAMPS=1

"%CD%\.venv\Scripts\python.exe" -m uvicorn server:app --host 127.0.0.1 --port 8000 --workers 1
pause
