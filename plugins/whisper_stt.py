"""
Custom STT plugin: faster-whisper with per-utterance language detection.

livekit-agents 1.5.x API:
  - Inherits from stt.STT
  - Must implement: _recognize_impl(buffer, *, language, conn_options) -> SpeechEvent
  - AudioBuffer = list[rtc.AudioFrame] | rtc.AudioFrame
"""

import logging
import time
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel
from livekit import rtc
from livekit.agents import stt, utils
from livekit.agents.types import NOT_GIVEN, APIConnectOptions, NotGivenOr

logger = logging.getLogger("whisper_stt")

# Whisper requires 16kHz mono float32
WHISPER_SAMPLE_RATE = 16000


class WhisperSTT(stt.STT):
    """
    faster-whisper STT plugin for livekit-agents 1.5.x.

    Processes full VAD-segmented utterances (non-streaming).
    Extracts language + probability from each transcription.
    """

    def __init__(
        self,
        *,
        model_size: str = "tiny",
        device: str = "cpu",
        compute_type: str = "int8",
        language_router=None,
        latency_tracker=None,
        beam_size: int = 1,
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
            )
        )
        logger.info(f"[STT] Loading faster-whisper '{model_size}' on {device} ({compute_type})...")
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self._beam_size = beam_size
        self._language_router = language_router
        self._latency_tracker = latency_tracker

        self.detected_language: str = "en"
        self.detected_language_prob: float = 1.0
        logger.info("[STT] faster-whisper ready.")

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        """
        Transcribe a complete utterance from a VAD-segmented AudioBuffer.
        Converts LiveKit AudioFrame(s) → 16kHz float32 → faster-whisper.
        """
        if self._latency_tracker:
            self._latency_tracker.start("stt")
        t0 = time.perf_counter_ns()

        # Merge frames if buffer is a list, then convert to float32 numpy
        audio_array = self._buffer_to_float32(buffer)

        # Transcribe — beam_size=1 is greedy/fastest on CPU
        lang_hint = language if language is not NOT_GIVEN else None
        segments, info = self._model.transcribe(
            audio_array,
            beam_size=self._beam_size,
            language=lang_hint,   # None = auto-detect
            vad_filter=False,     # VAD already handled upstream by Silero
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        detected_lang = info.language
        lang_prob = info.language_probability

        elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000
        if self._latency_tracker:
            self._latency_tracker.end("stt")

        logger.info(
            f"[STT] lang={detected_lang} (p={lang_prob:.2f}) | "
            f"text='{text}' | {elapsed_ms:.0f}ms"
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

    def _buffer_to_float32(self, buffer: utils.AudioBuffer) -> np.ndarray:
        """
        Convert a LiveKit AudioBuffer to a 16kHz mono float32 numpy array.
        Whisper requires exactly 16000 Hz single-channel float32.
        """
        # Normalize to list of frames
        if isinstance(buffer, rtc.AudioFrame):
            frames = [buffer]
        else:
            frames = list(buffer)

        if not frames:
            return np.zeros(1600, dtype=np.float32)  # 100ms of silence

        # Merge into one frame (handles multiple chunks from VAD)
        merged: rtc.AudioFrame = rtc.combine_audio_frames(frames)

        # Convert int16 PCM bytes → float32 [-1, 1]
        audio = np.frombuffer(merged.data, dtype=np.int16).astype(np.float32) / 32768.0

        # Convert stereo → mono by averaging channels
        if merged.num_channels > 1:
            audio = audio.reshape(-1, merged.num_channels).mean(axis=1)

        # Resample to 16kHz if needed
        if merged.sample_rate != WHISPER_SAMPLE_RATE:
            audio = self._resample(audio, merged.sample_rate, WHISPER_SAMPLE_RATE)

        return audio

    def _resample(self, audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(from_rate, to_rate)
        return resample_poly(audio, to_rate // g, from_rate // g).astype(np.float32)
