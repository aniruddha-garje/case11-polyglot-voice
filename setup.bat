@echo off
REM ============================================================
REM Case 11: Polyglot Voice Companion — Windows Setup Script
REM Run from the project root: setup.bat
REM Requires: Python 3.11+, pip, internet connection
REM ============================================================

echo.
echo ============================================================
echo  Polyglot Voice Companion - Windows Setup
echo ============================================================
echo.

REM --- Check Python version ---
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.11+ from python.org
    exit /b 1
)
echo [1/7] Python found:
python --version

REM --- Create virtual environment ---
echo.
echo [2/7] Creating virtual environment...
if exist venv (
    echo       venv already exists, skipping creation.
) else (
    python -m venv venv
    echo       venv created.
)

REM --- Activate venv ---
call venv\Scripts\activate.bat

REM --- Upgrade pip ---
echo.
echo [3/7] Upgrading pip...
python -m pip install --upgrade pip --quiet

REM --- Install dependencies ---
echo.
echo [4/7] Installing Python dependencies (this may take 5-15 minutes)...
echo       Note: TTS (Coqui XTTS-v2) is large. Grab a coffee.
pip install -r requirements.txt
echo.
echo [INFO] pip install finished. Any errors above are listed per-package.

REM --- Copy .env ---
echo.
echo [5/7] Setting up environment variables...
if not exist .env (
    copy .env.example .env
    echo       .env created from template. Edit it if needed.
) else (
    echo       .env already exists, skipping.
)

REM --- Download Piper voices ---
echo.
echo [6/7] Downloading Piper voices...
if not exist voices mkdir voices

REM English voice
if not exist "voices\en_US-lessac-medium.onnx" (
    echo       Downloading English voice (en_US-lessac-medium)...
    curl -L -o "voices\en_US-lessac-medium.onnx" ^
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
    curl -L -o "voices\en_US-lessac-medium.onnx.json" ^
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
) else (
    echo       English voice already downloaded.
)

REM Spanish voice
if not exist "voices\es_ES-carlfm-x_low.onnx" (
    echo       Downloading Spanish voice (es_ES-carlfm-x_low)...
    curl -L -o "voices\es_ES-carlfm-x_low.onnx" ^
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/carlfm/x_low/es_ES-carlfm-x_low.onnx"
    curl -L -o "voices\es_ES-carlfm-x_low.onnx.json" ^
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/carlfm/x_low/es_ES-carlfm-x_low.onnx.json"
) else (
    echo       Spanish voice already downloaded.
)

REM --- Download turn detector model ---
echo.
echo [7/8] Downloading turn detector model (lk_end_of_utterance_multilingual)...
echo       This downloads ~50MB from HuggingFace into the local cache.
python agent.py download-files
echo.

REM --- Remind about Ollama ---
echo.
echo [8/8] Ollama setup reminder...
echo       Make sure Ollama is installed and the model is pulled:
echo.
echo         ollama pull qwen2.5:1.5b
echo         ollama serve   (in a separate terminal)
echo.

echo ============================================================
echo  Setup complete!
echo ============================================================
echo.
echo  Next steps:
echo    1. Start Ollama:      ollama serve
echo    2. Pull the model:    ollama pull qwen2.5:1.5b
echo    3. Activate venv:     venv\Scripts\activate
echo    4. Run the agent:     python agent.py console
echo.
pause
