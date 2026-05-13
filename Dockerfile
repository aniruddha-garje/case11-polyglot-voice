FROM python:3.11-slim

WORKDIR /app

# System dependencies for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    portaudio19-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Download Piper voices at build time
RUN mkdir -p voices && \
    curl -L -o voices/en_US-lessac-medium.onnx \
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx" && \
    curl -L -o voices/en_US-lessac-medium.onnx.json \
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json" && \
    curl -L -o voices/es_ES-carlfm-x_low.onnx \
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/carlfm/x_low/es_ES-carlfm-x_low.onnx" && \
    curl -L -o voices/es_ES-carlfm-x_low.onnx.json \
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/carlfm/x_low/es_ES-carlfm-x_low.onnx.json"

# Note: For live mic demo, run natively (not in Docker).
# Docker is for reproducibility verification only.
# The container cannot access the host microphone without --device flags,
# and audio device passthrough is OS-dependent.
#
# To test with pre-recorded audio:
#   docker build -t polyglot-voice .
#   docker run --rm polyglot-voice python agent.py dev --audio-file test_audio.wav
#
# For live mic on Linux (may require pulseaudio passthrough):
#   docker run --rm --device /dev/snd polyglot-voice python agent.py console

CMD ["python", "agent.py", "console"]
