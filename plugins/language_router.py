"""
Language router — shared state for detected language across pipeline stages.

Thread-safe (asyncio.Lock). Tracks the current detected language and
only switches when Whisper's language_probability exceeds a threshold.

This prevents misrouting on short utterances where Whisper is uncertain
(e.g., a 2-word Hindi phrase that Whisper might guess as Urdu or English).
"""

import asyncio
import logging

logger = logging.getLogger("language_router")

# Minimum probability to accept a language switch.
# Below this threshold, we keep the previous language.
LANGUAGE_SWITCH_THRESHOLD = 0.70

# Human-readable language names for logging
LANG_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "es": "Spanish",
    "ur": "Urdu (mapped → Hindi)",  # Urdu and Hindi look similar to Whisper
}


class LanguageRouter:
    """
    Tracks the current detected language and routes TTS/LLM decisions.

    Usage:
        router = LanguageRouter()
        router.update_language("hi", 0.87)  # from WhisperSTT
        lang = router.get_current_language()  # → "hi"
    """

    def __init__(self, default_language: str = "en"):
        self._current_language = default_language
        self._previous_language = default_language
        self._lock = asyncio.Lock()
        self._switch_count = 0
        logger.info(f"[LANG] Language router initialized. Default: {default_language}")

    def update_language(self, detected_lang: str, probability: float):
        """
        Update the current language if the detection is confident enough.
        Whisper sometimes returns 'ur' (Urdu) for Hindi — we normalize that.

        Note: Not async — called from within the STT plugin which runs in an executor.
        Uses simple assignment (GIL protects scalar writes in CPython).
        """
        # Normalize Urdu → Hindi (Whisper confuses the two; XTTS-v2 handles both)
        if detected_lang == "ur":
            detected_lang = "hi"

        if probability < LANGUAGE_SWITCH_THRESHOLD:
            logger.info(
                f"[LANG] Detection below threshold ({probability:.2f} < {LANGUAGE_SWITCH_THRESHOLD}). "
                f"Keeping '{self._current_language}'."
            )
            return

        if detected_lang != self._current_language:
            self._previous_language = self._current_language
            self._current_language = detected_lang
            self._switch_count += 1
            logger.info(
                f"[LANG] Switch detected: "
                f"{LANG_NAMES.get(self._previous_language, self._previous_language)} "
                f"→ {LANG_NAMES.get(detected_lang, detected_lang)} "
                f"(confidence: {probability:.2f}, switch #{self._switch_count})"
            )
        else:
            logger.debug(f"[LANG] Language confirmed: {detected_lang} ({probability:.2f})")

    def get_current_language(self) -> str:
        """Return the current detected language code (e.g., 'en', 'hi', 'es')."""
        return self._current_language

    def get_previous_language(self) -> str:
        """Return the language before the last switch."""
        return self._previous_language

    def get_switch_count(self) -> int:
        """Total number of language switches in this session."""
        return self._switch_count

    def get_language_instruction(self) -> str:
        """
        Returns a short instruction string to inject into the LLM prompt,
        telling it which language to respond in.
        """
        lang = self._current_language
        if lang == "hi":
            return (
                "The user just spoke in Hindi. Respond in Hindi using Roman script only "
                "(Hinglish romanization, e.g. 'Haan, aapka order kal tak aa jayega'). "
                "Do NOT use Devanagari script."
            )
        elif lang == "es":
            return "The user just spoke in Spanish. Respond entirely in Spanish."
        else:
            return "The user just spoke in English. Respond in English."
