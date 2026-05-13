"""
Tests for faster-whisper language detection accuracy.

These tests use pre-recorded audio snippets to verify that the WhisperSTT plugin
correctly identifies the spoken language. Since we don't have test audio files
at scaffold time, tests are marked as skipped and serve as documentation.

Run manually after recording test audio:
    pytest tests/test_stt_language.py -v
"""

import pytest


@pytest.mark.skip(reason="Requires pre-recorded test audio files in tests/audio/")
def test_detect_english():
    """English utterance should be detected with probability > 0.9."""
    pass


@pytest.mark.skip(reason="Requires pre-recorded test audio files in tests/audio/")
def test_detect_hindi_long():
    """Long Hindi utterance (7+ words) should detect 'hi' with probability > 0.7."""
    pass


@pytest.mark.skip(reason="Requires pre-recorded test audio files in tests/audio/")
def test_detect_hindi_short():
    """
    Short Hindi utterance (3 words: 'Aur Delhi mein?') may have low confidence.
    Documents the known limitation: small model improves this, tiny may fail.
    """
    pass


@pytest.mark.skip(reason="Requires pre-recorded test audio files in tests/audio/")
def test_detect_spanish():
    """Spanish utterance should be detected with probability > 0.85."""
    pass


@pytest.mark.skip(reason="Requires pre-recorded test audio files in tests/audio/")
def test_urdu_mapped_to_hindi():
    """
    faster-whisper sometimes detects Hindi as Urdu ('ur').
    The LanguageRouter must normalize 'ur' → 'hi'.
    """
    pass


def test_language_router_threshold():
    """LanguageRouter should not switch language when probability < 0.7."""
    from plugins.language_router import LanguageRouter

    router = LanguageRouter(default_language="en")
    router.update_language("hi", 0.65)  # Below threshold
    assert router.get_current_language() == "en", "Should NOT switch on low probability"

    router.update_language("hi", 0.85)  # Above threshold
    assert router.get_current_language() == "hi", "Should switch on high probability"


def test_language_router_urdu_normalization():
    """LanguageRouter should map 'ur' (Urdu) to 'hi' (Hindi)."""
    from plugins.language_router import LanguageRouter

    router = LanguageRouter(default_language="en")
    router.update_language("ur", 0.90)
    assert router.get_current_language() == "hi", "Urdu should be normalized to Hindi"


def test_language_router_switch_count():
    """LanguageRouter should count language switches correctly."""
    from plugins.language_router import LanguageRouter

    router = LanguageRouter(default_language="en")
    router.update_language("hi", 0.90)
    router.update_language("es", 0.88)
    router.update_language("en", 0.95)
    assert router.get_switch_count() == 3
