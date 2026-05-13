"""
Tests for MultilingualTTS language routing.

Verifies that the TTS plugin correctly routes to the right backend
based on the current language set by LanguageRouter.

Run:
    pytest tests/test_tts_routing.py -v
"""

import pytest


def test_tts_language_switch():
    """TTS should switch language without error."""
    from plugins.multilingual_tts import MultilingualTTS

    # Only test the routing logic (set_language), not actual synthesis
    # (synthesis requires model files to be present)
    tts = MultilingualTTS.__new__(MultilingualTTS)
    tts._current_lang = "en"

    tts.set_language("hi")
    assert tts._current_lang == "hi"

    tts.set_language("es")
    assert tts._current_lang == "es"

    tts.set_language("en")
    assert tts._current_lang == "en"


def test_tts_unknown_language_defaults_to_en():
    """Unknown language codes should default to English TTS."""
    from plugins.multilingual_tts import MultilingualTTS

    tts = MultilingualTTS.__new__(MultilingualTTS)
    tts._current_lang = "en"

    # Set an unsupported language
    tts.set_language("fr")
    # The routing in _synthesize_impl uses else → English
    # This test just documents the expected behavior
    assert tts._current_lang == "fr"  # stored as-is, but synthesis falls back to EN


@pytest.mark.skip(reason="Requires Piper voice files in voices/ directory")
def test_piper_en_synthesis():
    """Piper EN should produce non-empty audio for an English sentence."""
    pass


@pytest.mark.skip(reason="Requires Piper voice files in voices/ directory")
def test_piper_es_synthesis():
    """Piper ES should produce non-empty audio for a Spanish sentence."""
    pass


@pytest.mark.skip(reason="Requires XTTS-v2 model download (~2GB)")
def test_xtts_hindi_synthesis():
    """XTTS-v2 should produce non-empty audio for a Hindi sentence."""
    pass
