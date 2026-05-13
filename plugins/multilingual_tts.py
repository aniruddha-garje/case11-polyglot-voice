"""
Multilingual TTS plugin — routes to the right TTS backend per language.

Language routing:
    en → Piper (en_US-lessac-medium)   ~150-250ms on CPU
    es → Piper (es_ES-carlfm-x_low)    ~150-250ms on CPU
    hi → XTTS-v2 (Coqui TTS)          ~4000-8000ms on CPU (ecosystem limitation)

Inherits from livekit.agents.tts.TTS (non-streaming mode).
"""

import io
import logging
import os
import time
from typing import Optional

import numpy as np
from livekit.agents import tts, utils

logger = logging.getLogger("multilingual_tts")

# Sample rate Piper outputs (22050 Hz), resampled to 24000 for XTTS
PIPER_SAMPLE_RATE = 22050
XTTS_SAMPLE_RATE = 24000
LIVEKIT_SAMPLE_RATE = 24000  # LiveKit expects 24kHz PCM


class MultilingualTTS(tts.TTS):
    """
    Language-aware TTS that routes synthesis to the appropriate backend.
    Language is set externally by the agent after STT detects the user's language.
    """

    def __init__(
        self,
        *,
        language_router=None,
        latency_tracker=None,
        voices_dir: str = "voices",
        hindi_backend: Optional[str] = None,
    ):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=LIVEKIT_SAMPLE_RATE,
            num_channels=1,
        )
        self._language_router = language_router
        self._latency_tracker = latency_tracker
        self._voices_dir = voices_dir
        self._current_lang = "en"

        # Resolve Hindi backend from env or constructor arg
        self._hindi_backend = hindi_backend or os.getenv("TTS_HINDI_BACKEND", "xtts-v2")

        # Lazy-load backends to avoid import errors if a package isn't installed
        self._piper_en = None
        self._piper_es = None
        self._xtts = None
        self._pyttsx3_engine = None

        self._init_backends()

    def _init_backends(self):
        """Initialize TTS backends. Log clearly if a backend fails."""
        # --- Piper (EN) ---
        try:
            from piper.voice import PiperVoice
            en_model = os.path.join(self._voices_dir, "en_US-lessac-medium.onnx")
            if os.path.exists(en_model):
                self._piper_en = PiperVoice.load(en_model)
                logger.info("[TTS] Piper EN loaded successfully.")
            else:
                logger.warning(f"[TTS] Piper EN voice not found at {en_model}. Run setup.bat to download.")
        except ImportError:
            logger.warning("[TTS] piper-tts not installed. EN TTS unavailable.")

        # --- Piper (ES) ---
        try:
            from piper.voice import PiperVoice
            es_model = os.path.join(self._voices_dir, "es_ES-carlfm-x_low.onnx")
            if os.path.exists(es_model):
                self._piper_es = PiperVoice.load(es_model)
                logger.info("[TTS] Piper ES loaded successfully.")
            else:
                logger.warning(f"[TTS] Piper ES voice not found at {es_model}. Run setup.bat to download.")
        except ImportError:
            logger.warning("[TTS] piper-tts not installed. ES TTS unavailable.")

        # --- Hindi TTS backend ---
        if self._hindi_backend == "xtts-v2":
            try:
                from TTS.api import TTS as CoquiTTS
                self._xtts = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
                logger.info("[TTS] Coqui XTTS-v2 loaded for Hindi.")
            except ImportError:
                logger.warning("[TTS] TTS (Coqui) not installed. Falling back to pyttsx3 for Hindi.")
                self._init_pyttsx3()
            except Exception as e:
                logger.warning(f"[TTS] XTTS-v2 init failed ({e}). Falling back to pyttsx3.")
                self._init_pyttsx3()
        else:
            self._init_pyttsx3()

    def _init_pyttsx3(self):
        """Fallback: Windows SAPI TTS via pyttsx3."""
        try:
            import pyttsx3
            self._pyttsx3_engine = pyttsx3.init()
            logger.info("[TTS] pyttsx3 (Windows SAPI) initialized as Hindi fallback.")
        except Exception as e:
            logger.error(f"[TTS] pyttsx3 init failed: {e}. Hindi TTS will be unavailable.")

    def set_language(self, lang: str):
        """Called by the agent after each STT result to switch the TTS voice."""
        if lang != self._current_lang:
            logger.info(f"[TTS] Language switch: {self._current_lang} → {lang}")
        self._current_lang = lang

    async def _synthesize_impl(self, text: str) -> tts.SynthesizedAudio:
        """
        Synthesize text to audio using the appropriate backend for the current language.
        Returns a SynthesizedAudio containing raw PCM frames.
        """
        lang = self._current_lang
        t_start = time.perf_counter_ns()
        if self._latency_tracker:
            self._latency_tracker.start("tts")

        logger.info(f"[TTS] Synthesizing in '{lang}' | Text: '{text[:60]}{'...' if len(text) > 60 else ''}'")

        try:
            if lang == "hi":
                audio_data, sample_rate = await self._synthesize_hindi(text)
            elif lang == "es":
                audio_data, sample_rate = await self._synthesize_piper(text, "es")
            else:
                # Default to English for unknown languages
                audio_data, sample_rate = await self._synthesize_piper(text, "en")
        except Exception as e:
            logger.error(f"[TTS] Synthesis failed for lang={lang}: {e}. Returning silence.")
            audio_data = np.zeros(LIVEKIT_SAMPLE_RATE, dtype=np.float32)
            sample_rate = LIVEKIT_SAMPLE_RATE

        elapsed_ms = (time.perf_counter_ns() - t_start) / 1_000_000
        if self._latency_tracker:
            self._latency_tracker.end("tts")
        logger.info(f"[TTS] Synthesis complete | Latency: {elapsed_ms:.1f}ms")

        # Resample to LiveKit's expected sample rate if needed
        if sample_rate != LIVEKIT_SAMPLE_RATE:
            audio_data = self._resample(audio_data, sample_rate, LIVEKIT_SAMPLE_RATE)

        # Convert float32 → int16 PCM
        pcm = (audio_data * 32767).astype(np.int16).tobytes()

        return tts.SynthesizedAudio(
            request_id=utils.shortuuid(),
            frame=agents.AudioFrame(
                data=pcm,
                sample_rate=LIVEKIT_SAMPLE_RATE,
                num_channels=1,
                samples_per_channel=len(audio_data),
            ),
        )

    async def _synthesize_piper(self, text: str, lang: str):
        """Synthesize using Piper. Returns (numpy_float32_array, sample_rate)."""
        import asyncio
        voice = self._piper_en if lang == "en" else self._piper_es
        if voice is None:
            raise RuntimeError(f"Piper voice for '{lang}' not loaded.")

        # Piper is synchronous; run in executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        audio_bytes = await loop.run_in_executor(None, self._piper_synthesize_sync, voice, text)
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return audio_data, PIPER_SAMPLE_RATE

    def _piper_synthesize_sync(self, voice, text: str) -> bytes:
        """Synchronous Piper synthesis — called from executor."""
        buf = io.BytesIO()
        # Piper writes 16-bit PCM to a file-like object
        voice.synthesize(text, buf, sentence_silence=0.0)
        return buf.getvalue()

    async def _synthesize_hindi(self, text: str):
        """Synthesize Hindi using XTTS-v2 or pyttsx3 fallback."""
        import asyncio

        if self._xtts is not None:
            loop = asyncio.get_event_loop()
            audio_data = await loop.run_in_executor(None, self._xtts_synthesize_sync, text)
            return audio_data, XTTS_SAMPLE_RATE
        elif self._pyttsx3_engine is not None:
            loop = asyncio.get_event_loop()
            audio_data = await loop.run_in_executor(None, self._pyttsx3_synthesize_sync, text)
            return audio_data, LIVEKIT_SAMPLE_RATE
        else:
            raise RuntimeError("No Hindi TTS backend available.")

    def _xtts_synthesize_sync(self, text: str) -> np.ndarray:
        """Synchronous XTTS-v2 synthesis — called from executor."""
        # XTTS-v2 returns a list of samples at 24kHz
        wav = self._xtts.tts(text=text, language="hi")
        return np.array(wav, dtype=np.float32)

    def _pyttsx3_synthesize_sync(self, text: str) -> np.ndarray:
        """
        Synchronous pyttsx3 synthesis — saves to temp WAV then reads back.
        This is a last-resort fallback with limited quality.
        """
        import tempfile
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        self._pyttsx3_engine.save_to_file(text, tmp_path)
        self._pyttsx3_engine.runAndWait()

        audio_data, sr = sf.read(tmp_path, dtype="float32")
        os.unlink(tmp_path)

        # Convert stereo to mono if needed
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
        return audio_data

    def _resample(self, audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
        """Simple linear resampling using scipy."""
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(from_rate, to_rate)
        return resample_poly(audio, to_rate // g, from_rate // g).astype(np.float32)
