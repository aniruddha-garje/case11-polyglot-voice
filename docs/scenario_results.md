# Scenario Test Results

Results from running the 4 conversation test scenarios against the deployed agent.
Updated during Phase 7 (scenario testing). Placeholder values to be replaced with real measurements.

---

## Scenario 1: Customer Support — Order Status (EN → HI → EN)

**Date tested:** TBD (Phase 7)
**Model:** faster-whisper tiny | Qwen 2.5 1.5B | Piper EN + XTTS-v2 HI

| Turn | Expected Lang | Detected Lang | Prob | Text (heard) | Latency (ms) | LLM Response Lang | Context Correct? |
|------|--------------|---------------|------|-------------|-------------|-------------------|-----------------|
| 1 | EN | — | — | "Hi, I need to check the status of my order. The order ID is 4421." | — | EN | ✓ |
| 2 | EN | — | — | "Yes, the email on the account is rahul@example.com." | — | EN | ✓ |
| 3 | HI | — | — | "Theek hai, lekin delivery kal tak ho jaayegi kya?" | — | HI | — |
| 4 | HI | — | — | "Aur agar nahi hua toh refund mil sakta hai?" | — | HI | — |
| 5 | EN | — | — | "Actually let's switch back — can you email me the tracking link?" | — | EN | — |

**Notes:** To be filled in during Phase 7 testing.

---

## Scenario 2: Travel Planning — Hotel Booking (ES → EN)

**Date tested:** TBD (Phase 7)
**Model:** faster-whisper tiny | Qwen 2.5 1.5B | Piper EN + Piper ES

| Turn | Expected Lang | Detected Lang | Prob | Latency (ms) | LLM Response Lang | Context Correct? |
|------|--------------|---------------|------|-------------|-------------------|-----------------|
| 1 | ES | — | — | — | ES | — |
| 2 | ES | — | — | — | ES | — |
| 3 | EN | — | — | — | EN | — |
| 4 | EN | — | — | — | EN | — |

**Notes:** To be filled in during Phase 7 testing.

---

## Scenario 4: Rapid Switching — Stress Test (EN → HI → ES → EN)

**Date tested:** TBD (Phase 7)

| Turn | Expected Lang | Detected Lang | Prob | Latency (ms) | LLM Response Lang | All 3 cities recalled? |
|------|--------------|---------------|------|-------------|-------------------|----------------------|
| 1 | EN | — | — | — | EN | — |
| 2 | HI | — | — | — | HI | — |
| 3 | ES | — | — | — | ES | — |
| 4 | EN | — | — | — | EN | — |

**Critical test:** Turn 4 response must mention Mumbai, Delhi, AND Chennai.

**Notes:** 
- Short utterances in turns 2 and 3 may have low detection confidence.
- If `language_probability < 0.7`, the router keeps the previous language.
- Document whether this causes a failure and what model upgrade fixes it.

---

## Overall Findings

*(To be completed in Phase 7)*

| Metric | Result |
|--------|--------|
| Language detection accuracy (all turns) | — / — |
| Context carryover across switches | — / — |
| Average EN latency | — ms |
| Average ES latency | — ms |
| Average HI latency | — ms |
| Voice-language match (correct TTS voice) | — / — |
