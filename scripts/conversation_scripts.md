# Conversation Test Scripts

Source: Assignment brief for Case 11 — Polyglot Voice Companion.
These are the 4 scenarios used to evaluate the agent's multilingual capabilities.

---

## Scenario 1 — Customer Support: Order Status

| Turn | Language | User speaks |
|------|----------|-------------|
| 1 | EN | "Hi, I need to check the status of my order. The order ID is 4421." |
| 2 | EN | "Yes, the email on the account is rahul@example.com." |
| 3 | HI | "Theek hai, lekin delivery kal tak ho jaayegi kya?" |
| 4 | HI | "Aur agar nahi hua toh refund mil sakta hai?" |
| 5 | EN | "Actually let's switch back — can you email me the tracking link?" |

**Expected behavior:**
- Turns 1-2: Agent responds in English, recalls order #4421.
- Turn 3: Agent switches to Hindi naturally (no acknowledgement of switch).
- Turn 4: Agent maintains context (still about order #4421) in Hindi.
- Turn 5: Agent switches back to English, still knows we're discussing the same order.

---

## Scenario 2 — Travel Planning: Hotel Booking

| Turn | Language | User speaks |
|------|----------|-------------|
| 1 | ES | "Hola, quiero reservar un hotel en Bangalore para el próximo fin de semana." |
| 2 | ES | "Para dos personas, presupuesto de 5000 rupias por noche." |
| 3 | EN | "Sorry, my Spanish is rusty. Can we continue in English? Tell me again about the second option." |
| 4 | EN | "Book it. Confirm the dates please." |

**Expected behavior:**
- Turns 1-2: Agent responds in Spanish, discusses hotel options.
- Turn 3: Agent switches to English, recalls the "second option" without re-asking.
- Turn 4: Agent confirms booking dates in English.

---

## Scenario 3 — Technical Support (Not tested in demo)

*Included in the brief but not a priority for evaluation.*
Standard EN-only technical support scenario.

---

## Scenario 4 — Rapid Switching (Stress Test)

| Turn | Language | User speaks |
|------|----------|-------------|
| 1 | EN | "What's the weather in Mumbai today?" |
| 2 | HI | "Aur Delhi mein?" |
| 3 | ES | "¿Y en Chennai?" |
| 4 | EN | "Compare all three for me." |

**Expected behavior:**
- Turn 1: English response about Mumbai weather.
- Turn 2: Hindi response about Delhi, building on Turn 1.
- Turn 3: Spanish response about Chennai.
- Turn 4: English response comparing all three cities mentioned in the conversation.

**Key challenge:** Turn 4 requires memory of Turns 1-3 across three language switches.
The LLM must output all three cities without being re-told what they were.

---

## Notes on language detection difficulty

- **Turn 2 (Scenario 4):** "Aur Delhi mein?" is only 3 words. faster-whisper (tiny) may
  misidentify this as English. If `language_probability < 0.7`, the router keeps the
  previous language. This may cause Turn 2 to respond in English instead of Hindi.
  **Fix:** Use the `small` model for better accuracy on short utterances, at the cost of
  ~1.2s additional STT latency.

- **Turn 3 (Scenario 4):** Short Spanish phrase. Same risk as above.

- **Turn 3 (Scenario 1):** "Theek hai, lekin delivery kal tak ho jaayegi kya?" — longer,
  should detect Hindi reliably (7 words).
