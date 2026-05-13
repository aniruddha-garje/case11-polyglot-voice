#!/usr/bin/env bash
# ============================================================
# Case 11: Polyglot Voice Companion — Linux/Mac Setup Script
# Run from the project root: bash setup.sh
# Requires: Python 3.11+, pip, curl, internet connection
# ============================================================

set -e  # Exit on first error

echo ""
echo "============================================================"
echo " Polyglot Voice Companion - Linux/Mac Setup"
echo "============================================================"
echo ""

# --- Check Python version ---
echo "[1/7] Checking Python..."
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] python3 not found. Install Python 3.11+."
    exit 1
fi
python3 --version

# --- Create virtual environment ---
echo ""
echo "[2/7] Creating virtual environment..."
if [ -d "venv" ]; then
    echo "      venv already exists, skipping."
else
    python3 -m venv venv
    echo "      venv created."
fi

# --- Activate ---
source venv/bin/activate

# --- Upgrade pip ---
echo ""
echo "[3/7] Upgrading pip..."
pip install --upgrade pip --quiet

# --- System dependencies (Linux only) ---
echo ""
echo "[4/7] Checking system dependencies..."
if command -v apt-get &>/dev/null; then
    echo "      Installing system packages via apt..."
    sudo apt-get update -qq
    sudo apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 portaudio19-dev curl
elif command -v brew &>/dev/null; then
    echo "      Installing system packages via brew..."
    brew install ffmpeg libsndfile portaudio
fi

# --- Install Python dependencies ---
echo ""
echo "[5/7] Installing Python dependencies (may take 5-15 minutes)..."
pip install -r requirements.txt

# --- Copy .env ---
echo ""
echo "[6/7] Setting up environment variables..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "      .env created from template."
else
    echo "      .env already exists, skipping."
fi

# --- Download Piper voices ---
echo ""
echo "[7/7] Downloading Piper voices..."
mkdir -p voices

BASE_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main"

if [ ! -f "voices/en_US-lessac-medium.onnx" ]; then
    echo "      Downloading English voice..."
    curl -L -o "voices/en_US-lessac-medium.onnx" \
        "${BASE_URL}/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
    curl -L -o "voices/en_US-lessac-medium.onnx.json" \
        "${BASE_URL}/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
else
    echo "      English voice already downloaded."
fi

if [ ! -f "voices/es_ES-carlfm-x_low.onnx" ]; then
    echo "      Downloading Spanish voice..."
    curl -L -o "voices/es_ES-carlfm-x_low.onnx" \
        "${BASE_URL}/es/es_ES/carlfm/x_low/es_ES-carlfm-x_low.onnx"
    curl -L -o "voices/es_ES-carlfm-x_low.onnx.json" \
        "${BASE_URL}/es/es_ES/carlfm/x_low/es_ES-carlfm-x_low.onnx.json"
else
    echo "      Spanish voice already downloaded."
fi

echo ""
echo "============================================================"
echo " Setup complete!"
echo "============================================================"
echo ""
echo " Next steps:"
echo "   1. Start Ollama:    ollama serve"
echo "   2. Pull the model:  ollama pull qwen2.5:1.5b"
echo "   3. Activate venv:   source venv/bin/activate"
echo "   4. Run the agent:   python agent.py console"
echo ""
