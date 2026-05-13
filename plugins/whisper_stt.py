"""
Custom STT plugin: faster-whisper with per-utterance language detection.

Inherits from livekit.agents.stt.STT (non-streaming mode).
Extracts language + language_probability from each Whisper transcription.
Stores detected language in the shared LanguageRouter.
"""

import logging
import time
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel
from livekit import agents
from livekit.agents import stt, utils

logger = logging.getLogger("whisper_stt")


class WhisperSTT(stt.STT):
    """
    faster-whisper backed STT plugin for LiveKit Agents.

    Non-streaming: waits for a full VAD-segmented utterance, then transcribes.
    Returns the transcript AND detected language as part of the SpeechEvent.
    """

    def __init__(
        self,
        *,
        model_size: str = "tiny",
        device: str = "cpu",
        compute_type: str = "int8",
        language_router=None,   # LanguageRouter instance (injected from agent.py)
        latency_tracker=None,   # LatencyTracker instance (injected from agent.py)
        beam_size: int = 1,     # beam_size=1 is fastest; increase for accuracy
        best_of: int = 1,
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
            )
        )
        logger.info(f"[STT] Loading faster-whisper model: {model_size} on {device} ({compute_type})")
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self._beam_size = beam_size
        self._best_of = best_of
        self._language_router = language_router
        self._latency_tracker = latency_tracker

        # Last detected language — also readable directly if needed
        self.detected_language: str = "en"
        self.detected_language_prob: float = 1.0

        logger.info("[STT] faster-whisper model loaded.")

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: Optional[str] = None,
    ) -> stt.SpeechEvent:
        """
        Called by LiveKit Agents for each VAD-segmented audio chunk.
        Converts buffer → numpy float32 array → transcribes with Whisper.
        """
        t_start = time.perf_counter_ns()
        if self._latency_tracker:
            self._latency_tracker.start("stt")

        # Convert LiveKit AudioBuffer to numpy float32 (required by faster-whisper)
        audio_data = np.frombuffer(buffer.data, dtype=np.int16).astype(np.float32) / 32768.0

        # Transcribe — beam_size=1/best_of=1 is greedy, fastest on CPU
        segments, info = self._model.transcribe(
            audio_data,
            beam_size=self._beam_size,
            best_of=self._best_of,
            language=language,  # None = auto-detect (what we want)
            vad_filter=False,   # VAD is handled by LiveKit/Silero upstream
        )

        text = " ".join(seg.text.strip() for seg in segments).strip()
        detected_lang = info.language
        lang_prob = info.language_probability

        elapsed_ms = (time.perf_counter_ns() - t_start) / 1_000_000
        if self._latency_tracker:
            self._latency_tracker.end("stt")

        logger.info(
            f"[STT] Detected language: {detected_lang} (prob: {lang_prob:.2f}) | "
            f"Text: '{text}' | Latency: {elapsed_ms:.1f}ms"
        )

        # Update shared language state
        self.detected_language = detected_lang
        self.detected_language_prob = lang_prob
        if self._language_router:
            self._language_router.update_language(detected_lang, lang_prob)

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(
                    text=text,
                    language=detected_lang,
                    confidence=lang_prob,
                )
            ],
        )
