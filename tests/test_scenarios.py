"""
Scenario test documentation and runnable checks.

These tests document the expected behavior for each conversation scenario.
Full end-to-end tests require a running agent and real audio, so most are
documented as manual tests. Automated tests cover the logic components.

Results are documented in docs/scenario_results.md after manual testing.
"""

import pytest


# ============================================================
# Scenario 1: Customer Support — Order Status
# ============================================================

def test_scenario1_context_carryover_description():
    """
    Manual test documentation: Scenario 1 — EN → EN → HI → HI → EN.

    Steps:
    1. Say: "Hi, I need to check the status of my order. The order ID is 4421."
       Expected: Agent responds in English, acknowledges order #4421.
    2. Say: "Yes, the email on the account is rahul@example.com."
       Expected: Agent responds in English, confirms email.
    3. Say: "Theek hai, lekin delivery kal tak ho jaayegi kya?"
       Expected: Agent switches to Hindi (no comment on switch), answers about delivery.
       Language log: [LANG] Switch detected: English → Hindi
    4. Say: "Aur agar nahi hua toh refund mil sakta hai?"
       Expected: Agent responds in Hindi, mentions refund policy for order #4421.
       Context check: LLM must still know we're talking about order #4421.
    5. Say: "Actually let's switch back — can you email me the tracking link?"
       Expected: Agent switches back to English, sends tracking for order #4421.

    Pass criteria:
    - All 5 language detections correct
    - LLM mentions order #4421 context in turns 3-5
    - No "I don't know which order" responses
    - Voice changes audibly between EN and HI turns
    """
    pytest.skip("Manual scenario test — see docs/scenario_results.md")


# ============================================================
# Scenario 2: Travel Planning — Hotel Booking (ES → EN)
# ============================================================

def test_scenario2_spanish_to_english():
    """
    Manual test documentation: Scenario 2 — ES → ES → EN → EN.

    Steps:
    1. Say: "Hola, quiero reservar un hotel en Bangalore para el próximo fin de semana."
       Expected: Agent responds in Spanish.
    2. Say: "Para dos personas, presupuesto de 5000 rupias por noche."
       Expected: Agent responds in Spanish, offers hotel options.
    3. Say: "Sorry, my Spanish is rusty. Can we continue in English? Tell me again about the second option."
       Expected: Agent switches to English, recalls the "second option" from Turn 2.
    4. Say: "Book it. Confirm the dates please."
       Expected: Agent confirms dates in English.

    Pass criteria:
    - Turns 1-2 in Spanish
    - Turn 3 switch detected correctly
    - Turn 3: LLM recalls second option without re-asking
    - Turn 4: LLM confirms dates
    """
    pytest.skip("Manual scenario test — see docs/scenario_results.md")


# ============================================================
# Scenario 4: Rapid Switching — Stress Test
# ============================================================

def test_scenario4_rapid_switching():
    """
    Manual test documentation: Scenario 4 — EN → HI → ES → EN.

    Steps:
    1. Say: "What's the weather in Mumbai today?"
       Expected: English response about Mumbai.
    2. Say: "Aur Delhi mein?"
       Expected: Hindi response about Delhi.
       Risk: Short phrase (3 words), may fail language detection on tiny model.
    3. Say: "¿Y en Chennai?"
       Expected: Spanish response about Chennai.
       Risk: Short phrase, same detection risk.
    4. Say: "Compare all three for me."
       Expected: English response comparing Mumbai, Delhi, and Chennai.
       This is the critical context-carryover test.

    Pass criteria:
    - Turn 4 must mention all three cities without being prompted
    - If Turns 2-3 language detection fails: document the failure and the model size needed
    """
    pytest.skip("Manual scenario test — see docs/scenario_results.md")


# ============================================================
# Automated unit tests for language instruction generation
# ============================================================

def test_language_instruction_english():
    from plugins.language_router import LanguageRouter
    router = LanguageRouter(default_language="en")
    instruction = router.get_language_instruction()
    assert "English" in instruction


def test_language_instruction_hindi():
    from plugins.language_router import LanguageRouter
    router = LanguageRouter(default_language="en")
    router.update_language("hi", 0.95)
    instruction = router.get_language_instruction()
    assert "Hindi" in instruction
    assert "Roman" in instruction  # Must specify Roman script


def test_language_instruction_spanish():
    from plugins.language_router import LanguageRouter
    router = LanguageRouter(default_language="en")
    router.update_language("es", 0.92)
    instruction = router.get_language_instruction()
    assert "Spanish" in instruction
