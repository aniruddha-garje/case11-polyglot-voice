"""
Multilingual TTS plugin — routes to the right TTS backend per language.

livekit-agents 1.5.x API:
  - Inherits from tts.TTS
  - Must implement: synthesize(text, *, conn_options) -> ChunkedStream
  - ChunkedStream subclass must implement: async _run(output_emitter)
  - output_emitter.initialize() → push(pcm_bytes) → flush()

Language routing:
    en → Piper (en_US-lessac-medium)   ~150-250ms
    es → Piper (es_ES-carlfm-x_low)    ~150-250ms
    hi → pyttsx3 Windows SAPI          ~100ms (lower quality)
"""

import asyncio
import logging
import os
import time
from typing import Optional

import numpy as np
from livekit.agents import tts, utils
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

logger = logging.getLogger("multilingual_tts")

# LiveKit expects 24kHz PCM from TTS
OUTPUT_SAMPLE_RATE = 24000
PIPER_SAMPLE_RATE = 22050   # Piper outputs at 22050 Hz


class MultilingualTTS(tts.TTS):
    """
    Language-aware TTS router.
    Language is set via set_language() after each STT result.
    """

    def __init__(
        self,
        *,
        language_router=None,
        latency_tracker=None,
        voices_dir: str = "voices",
    ):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=OUTPUT_SAMPLE_RATE,
            num_channels=1,
        )
        self._language_router = language_router
        self._latency_tracker = latency_tracker
        self._voices_dir = voices_dir
        self._current_lang = "en"

        self._piper_en = None
        self._piper_es = None
        self._pyttsx3_engine = None

        self._load_backends()

    def _load_backends(self):
        """Load TTS backends at init. Failures are logged, not raised."""
        # --- Piper EN ---
        try:
            from piper.voice import PiperVoice
            en_path = os.path.join(self._voices_dir, "en_US-lessac-medium.onnx")
            if os.path.exists(en_path):
                self._piper_en = PiperVoice.load(en_path)
                logger.info("[TTS] Piper EN loaded.")
            else:
                logger.warning(f"[TTS] Piper EN not found at '{en_path}'. Run: curl commands in README.")
        except Exception as e:
            logger.warning(f"[TTS] Piper EN failed to load: {e}")

        # --- Piper ES ---
        try:
            from piper.voice import PiperVoice
            es_path = os.path.join(self._voices_dir, "es_ES-carlfm-x_low.onnx")
            if os.path.exists(es_path):
                self._piper_es = PiperVoice.load(es_path)
                logger.info("[TTS] Piper ES loaded.")
            else:
                logger.warning(f"[TTS] Piper ES not found at '{es_path}'.")
        except Exception as e:
            logger.warning(f"[TTS] Piper ES failed to load: {e}")

        # --- pyttsx3 for Hindi (Windows SAPI fallback) ---
        try:
            import pyttsx3
            self._pyttsx3_engine = pyttsx3.init()
            logger.info("[TTS] pyttsx3 (Windows SAPI) loaded for Hindi.")
        except Exception as e:
            logger.warning(f"[TTS] pyttsx3 failed to init: {e}. Hindi TTS unavailable.")

    def set_language(self, lang: str):
        """Called by the agent after each STT event to switch TTS language."""
        if lang != self._current_lang:
            logger.info(f"[TTS] Language switch: {self._current_lang} → {lang}")
        self._current_lang = lang

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "MultilingualChunkedStream":
        return MultilingualChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
            lang=self._current_lang,
            latency_tracker=self._latency_tracker,
        )

    def synthesize_piper(self, text: str, lang: str) -> bytes:
        """Synchronous Piper synthesis. Returns raw int16 PCM bytes at PIPER_SAMPLE_RATE."""
        voice = self._piper_en if lang == "en" else self._piper_es
        if voice is None:
            raise RuntimeError(f"Piper voice for '{lang}' not available.")

        pcm_parts = []
        for chunk in voice.synthesize(text):
            # chunk.audio_float_array is float32 [-1, 1] at chunk.sample_rate
            pcm = (chunk.audio_float_array * 32767).astype(np.int16)
            pcm_parts.append(pcm.tobytes())
        return b"".join(pcm_parts)

    def synthesize_hindi(self, text: str) -> tuple[bytes, int]:
        """
        Synchronous Hindi synthesis via pyttsx3 (Windows SAPI).
        Returns (pcm_bytes, sample_rate).
        """
        import tempfile
        import soundfile as sf

        if self._pyttsx3_engine is None:
            raise RuntimeError("pyttsx3 not available for Hindi TTS.")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            self._pyttsx3_engine.save_to_file(text, tmp_path)
            self._pyttsx3_engine.runAndWait()
            audio, sr = sf.read(tmp_path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            pcm = (audio * 32767).astype(np.int16).tobytes()
            return pcm, sr
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


class MultilingualChunkedStream(tts.ChunkedStream):
    """ChunkedStream implementation for MultilingualTTS."""

    def __init__(self, *, tts: MultilingualTTS, input_text: str,
                 conn_options: APIConnectOptions, lang: str, latency_tracker=None):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._lang = lang
        self._latency_tracker = latency_tracker
        self._multilingual_tts = tts  # typed reference for mypy

    async def _run(self, output_emitter) -> None:
        """Synthesize and push audio to LiveKit's output emitter."""
        lang = self._lang
        text = self._input_text
        t0 = time.perf_counter_ns()

        if self._latency_tracker:
            self._latency_tracker.start("tts")

        logger.info(f"[TTS] Synthesizing [{lang}]: '{text[:60]}{'...' if len(text)>60 else ''}'")

        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=OUTPUT_SAMPLE_RATE,
            num_channels=1,
            mime_type="audio/pcm",
        )

        loop = asyncio.get_event_loop()

        try:
            if lang in ("en", "es"):
                # Piper: synchronous, run in thread to avoid blocking event loop
                pcm_bytes = await loop.run_in_executor(
                    None, self._multilingual_tts.synthesize_piper, text, lang
                )
                # Piper outputs at 22050Hz; resample to 24000Hz for LiveKit
                pcm_bytes = self._resample_pcm(pcm_bytes, PIPER_SAMPLE_RATE, OUTPUT_SAMPLE_RATE)
            else:
                # Hindi: pyttsx3 via Windows SAPI
                pcm_bytes, sr = await loop.run_in_executor(
                    None, self._multilingual_tts.synthesize_hindi, text
                )
                if sr != OUTPUT_SAMPLE_RATE:
                    pcm_bytes = self._resample_pcm(pcm_bytes, sr, OUTPUT_SAMPLE_RATE)

            output_emitter.push(pcm_bytes)

        except Exception as e:
            logger.error(f"[TTS] Synthesis failed for lang={lang}: {e}")
            # Push 500ms of silence so pipeline doesn't stall
            silence = bytes(OUTPUT_SAMPLE_RATE * 2 // 2)  # 500ms, 2 bytes/sample
            output_emitter.push(silence)

        output_emitter.flush()

        elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000
        if self._latency_tracker:
            self._latency_tracker.end("tts")
        logger.info(f"[TTS] Done [{lang}] | {elapsed_ms:.0f}ms")

    def _resample_pcm(self, pcm_bytes: bytes, from_rate: int, to_rate: int) -> bytes:
        """Resample int16 PCM bytes from one sample rate to another."""
        if from_rate == to_rate:
            return pcm_bytes
        from math import gcd
        from scipy.signal import resample_poly
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        g = gcd(from_rate, to_rate)
        resampled = resample_poly(audio, to_rate // g, from_rate // g)
        return (resampled * 32767).astype(np.int16).tobytes()
