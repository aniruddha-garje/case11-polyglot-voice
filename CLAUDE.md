# CLAUDE.md — Case 11: Polyglot Voice Companion

## PROJECT OVERVIEW

Build a real-time multilingual voice agent that:
1. Listens via microphone (Silero VAD → faster-whisper STT)
2. Detects spoken language per utterance
3. Generates response via local LLM (Ollama + Qwen 2.5)
4. Speaks response via TTS in the detected language
5. Maintains conversation context across language switches
6. Supports English, Hindi, and Spanish

**HARD CONSTRAINT:** Every model and library must be open-source. No OpenAI, Anthropic, Google Cloud, Azure, or any managed inference API. All inference runs locally.

**Target latency:** <1.2s end-of-speech to start-of-agent-speech. (Stretch: <800ms.)
**Hardware:** Intel i5 + Intel Iris Xe (integrated GPU, no CUDA). 16GB RAM. Windows 11.
**Framework:** LiveKit Agents SDK (Apache 2.0) in `console` mode for local mic I/O.

---

## TECH STACK (LOCKED — DO NOT CHANGE)

| Component       | Tool                                      | Plugin / Integration                          |
|-----------------|-------------------------------------------|-----------------------------------------------|
| Orchestration   | LiveKit Agents SDK ~1.0 (`console` mode)  | `livekit-agents`                              |
| VAD             | Silero VAD                                | `livekit-plugins-silero`                      |
| STT             | faster-whisper (tiny or small model)      | Custom STT plugin wrapping faster-whisper     |
| Language ID     | faster-whisper native `.language` field   | Extracted in custom STT plugin                |
| LLM             | Ollama + Qwen 2.5 1.5B (or 3B if fast)   | `livekit-plugins-openai` with `.with_ollama()`|
| TTS (English)   | Piper TTS                                 | Custom TTS plugin or community plugin         |
| TTS (Spanish)   | Piper TTS (es_ES voice)                   | Same custom TTS plugin                        |
| TTS (Hindi)     | Coqui XTTS-v2 OR gTTS-cached fallback    | Custom TTS plugin, language-routed            |
| Turn detection  | LiveKit turn detector                     | `livekit-plugins-turn-detector`               |

### Model selection rationale (put this in DECISIONS.md)
- **Qwen 2.5 over Llama 3.2:** Better multilingual (especially Hindi/Chinese), similar parameter count.
- **1.5B over 3B:** CPU-only hardware, need to keep LLM TTFT under 2s. Test 3B first; if >5s TTFT, drop to 1.5B.
- **faster-whisper over whisper.cpp:** Native Python integration, returns language + probability per segment.
- **Piper over XTTS-v2 for EN/ES:** ~200ms on CPU vs ~5-10s for XTTS-v2. Piper has good EN/ES voices.
- **XTTS-v2 for Hindi fallback:** Only open-source TTS with decent Hindi. Acknowledge latency in budget table.
- **LiveKit Agents over raw Python:** Apache 2.0 framework, not a managed API. Gives us streaming TTS + barge-in for free (both are stretch goals in the brief). All inference stays local.

---

## PROJECT STRUCTURE

```
case11-polyglot-voice/
├── CLAUDE.md                    # This file (Claude Code instructions)
├── README.md                    # Entry point for evaluators
├── DECISIONS.md                 # Trade-offs, assumptions, rejections
├── requirements.txt             # Python dependencies
├── setup.bat                    # Windows one-command setup
├── setup.sh                     # Linux/Mac setup (for Docker)
├── Dockerfile                   # Reproducible environment
├── .env.example                 # Environment variable template
├── .gitignore                   # Python + env exclusions
│
├── agent.py                     # Main entry point — LiveKit Agent
├── plugins/
│   ├── __init__.py
│   ├── whisper_stt.py           # Custom STT: faster-whisper + language detection
│   ├── multilingual_tts.py      # Custom TTS: routes to Piper/XTTS by language
│   └── language_router.py       # Language detection + routing logic
├── prompts/
│   └── system_prompt.py         # Multilingual system prompt template
├── utils/
│   ├── __init__.py
│   └── latency.py               # Latency measurement decorators/logging
│
├── tests/
│   ├── test_stt_language.py     # Test language detection accuracy
│   ├── test_tts_routing.py      # Test TTS language routing
│   └── test_scenarios.py        # Test against provided conversation scripts
│
├── docs/
│   ├── architecture.md          # Pipeline diagram (Mermaid)
│   ├── latency_budget.md        # Measured latency table (dual: CPU + projected GPU)
│   └── scenario_results.md      # Results from testing scripts 1-4
│
├── scripts/
│   └── conversation_scripts.md  # Copy of provided test scripts
│
└── deck/
    └── (slides go here)
```

---

## CRITICAL IMPLEMENTATION DETAILS

### Custom STT Plugin (whisper_stt.py)
LiveKit's built-in OpenAI STT plugin calls the OpenAI Whisper API (paid, non-open-source).
We MUST write a custom STT class that:
1. Inherits from `livekit.agents.stt.STT`
2. Uses `faster_whisper.WhisperModel` locally
3. Extracts `language` and `language_probability` from each transcription
4. Stores the detected language in a shared state accessible by the TTS router
5. Uses Silero VAD segments (provided by LiveKit) to know when a full utterance is ready

Key code pattern:
```python
from faster_whisper import WhisperModel
from livekit.agents import stt

class WhisperSTT(stt.STT):
    def __init__(self, *, model_size: str = "tiny", device: str = "cpu"):
        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))
        self._model = WhisperModel(model_size, device=device, compute_type="int8")
        self.detected_language = "en"  # default

    async def _recognize_impl(self, buffer, *, language=None):
        # Convert audio buffer to numpy, run whisper
        segments, info = self._model.transcribe(audio_array, beam_size=1, best_of=1)
        self.detected_language = info.language  # "en", "hi", "es"
        text = " ".join([s.text for s in segments])
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(text=text, language=info.language)]
        )
```

### Multilingual TTS Plugin (multilingual_tts.py)
Must route to different TTS backends based on detected language:
- `en` → Piper (en_US voice, ~200ms)
- `es` → Piper (es_ES voice, ~200ms)
- `hi` → XTTS-v2 (slow but only option) OR pre-cached common phrases

Key code pattern:
```python
from livekit.agents import tts

class MultilingualTTS(tts.TTS):
    def __init__(self):
        super().__init__(capabilities=tts.TTSCapabilities(streaming=False))
        self._piper_en = ...  # initialize Piper with English voice
        self._piper_es = ...  # initialize Piper with Spanish voice
        self._xtts_hi = ...   # initialize XTTS-v2 for Hindi (or fallback)

    def set_language(self, lang: str):
        self._current_lang = lang

    async def _synthesize_impl(self, text):
        if self._current_lang == "hi":
            return self._xtts_hi.synthesize(text)
        elif self._current_lang == "es":
            return self._piper_es.synthesize(text)
        else:
            return self._piper_en.synthesize(text)
```

### Language-Aware System Prompt (system_prompt.py)
```python
SYSTEM_PROMPT = """You are a helpful multilingual voice assistant for customer support.

CRITICAL RULES:
1. ALWAYS respond in the SAME language the user just spoke.
2. If the user speaks Hindi, respond in romanized Hindi (e.g., "Haan, aapka order kal tak aa jayega").
   Do NOT use Devanagari script — use Roman letters for TTS compatibility.
3. If the user speaks Spanish, respond entirely in Spanish.
4. If the user switches language mid-conversation, switch with them immediately.
5. NEVER acknowledge or comment on the language switch. Just switch naturally.
6. Maintain FULL conversation context across language switches.
   If the user asked about order #4421 in English, and then asks about delivery in Hindi,
   you still know they're asking about order #4421.
7. Keep responses concise: 1-3 sentences maximum. This is voice, not text.
8. Never use markdown, bullet points, or special characters in responses.
"""
```

### Latency Measurement (latency.py)
```python
import time
import logging

logger = logging.getLogger("latency")

class LatencyTracker:
    def __init__(self):
        self.stages = {}

    def start(self, stage: str):
        self.stages[stage] = {"start": time.perf_counter_ns()}

    def end(self, stage: str):
        if stage in self.stages:
            elapsed_ms = (time.perf_counter_ns() - self.stages[stage]["start"]) / 1_000_000
            self.stages[stage]["elapsed_ms"] = elapsed_ms
            logger.info(f"[LATENCY] {stage}: {elapsed_ms:.1f}ms")
            return elapsed_ms

    def total(self):
        return sum(s.get("elapsed_ms", 0) for s in self.stages.values())

    def report(self):
        lines = ["Stage | Measured (ms)"]
        lines.append("------|-------------")
        for stage, data in self.stages.items():
            lines.append(f"{stage} | {data.get('elapsed_ms', 'N/A'):.1f}")
        lines.append(f"TOTAL | {self.total():.1f}")
        return "\n".join(lines)
```

### Event Hook: Connecting STT Language Detection to TTS Routing
In `agent.py`, after STT returns a result, intercept the detected language and pass it to the TTS:
```python
@session.on("user_speech_committed")
async def on_speech(event):
    detected_lang = whisper_stt.detected_language
    multilingual_tts.set_language(detected_lang)
    # Also inject language hint into the next LLM turn
```
The exact event name may vary by LiveKit Agents version — check docs.

---

## CONVERSATION TEST SCRIPTS (from the brief)

### Scenario 1 — Customer Support: Order Status
Turn 1 (EN): "Hi, I need to check the status of my order. The order ID is 4421."
Turn 2 (EN): "Yes, the email on the account is rahul@example.com."
Turn 3 (HI): "Theek hai, lekin delivery kal tak ho jaayegi kya?"
Turn 4 (HI): "Aur agar nahi hua toh refund mil sakta hai?"
Turn 5 (EN): "Actually let's switch back — can you email me the tracking link?"

### Scenario 2 — Travel Planning: Hotel Booking
Turn 1 (ES): "Hola, quiero reservar un hotel en Bangalore para el próximo fin de semana."
Turn 2 (ES): "Para dos personas, presupuesto de 5000 rupias por noche."
Turn 3 (EN): "Sorry, my Spanish is rusty. Can we continue in English? Tell me again about the second option."
Turn 4 (EN): "Book it. Confirm the dates please."

### Scenario 4 — Rapid Switching (Stress Test)
Turn 1 (EN): "What's the weather in Mumbai today?"
Turn 2 (HI): "Aur Delhi mein?"
Turn 3 (ES): "¿Y en Chennai?"
Turn 4 (EN): "Compare all three for me."

---

## GIT COMMIT PLAN

After completing each phase, commit with the EXACT message shown:

1. `init: project scaffold, README, DECISIONS.md, requirements.txt`
2. `feat: basic agent with Ollama LLM via LiveKit console mode`
3. `feat: faster-whisper custom STT plugin with language detection`
4. `feat: multilingual system prompt and language-tagged conversation memory`
5. `feat: Piper TTS integration for English and Spanish`
6. `feat: multilingual TTS routing — language detection drives voice selection`
7. `feat: latency instrumentation across all pipeline stages`
8. `test: scenario 1-4 results documented with latency measurements`
9. `infra: Dockerfile and setup automation scripts`
10. `docs: architecture diagram (Mermaid) and latency budget table`
11. `docs: DECISIONS.md finalized with all trade-offs and rejections`
12. `docs: 5-slide presentation deck`
13. `final: README polish, submission-ready`

IMPORTANT: `git add -A && git commit -m "<message>"` after each phase. Do NOT batch commits.

---

## DOCKERFILE

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# System dependencies for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Note: For live mic demo, run natively (not in Docker).
# Docker is for reproducibility verification.
# To test with pre-recorded audio: python agent.py test --audio-file test_audio.wav
CMD ["python", "agent.py", "console"]
```

---

## WHAT THE EVALUATORS ARE LOOKING FOR

From the brief: "The differentiators: (1) the latency budget table — does the candidate know where every millisecond goes? — and (2) language-switch handling — does the agent stay coherent across a switch?"

Evaluation checklist:
- [ ] Did the agent detect the language switch correctly?
- [ ] Was the context (order ID, hotel preference, prior question) carried across the switch?
- [ ] What was the latency from end-of-user-speech to start-of-agent-speech?
- [ ] Did the TTS voice match the language naturally?
- [ ] How did the agent handle the code-switching utterance?
- [ ] Did the agent ever "forget" prior context after a language switch?

---

## STRETCH GOALS WE GET FREE FROM LIVEKIT AGENTS

- ✅ Streaming TTS (start audio while LLM is still generating) — built into AgentSession
- ✅ Interruption handling (barge-in) — built into VoicePipelineAgent
- 🔨 Code-switching within a single utterance — needs custom work
- 🔨 Auto-degrade to smaller model when latency exceeded — needs custom work

---

## KNOWN LIMITATIONS TO DOCUMENT HONESTLY

1. **Latency on CPU:** Cannot hit 1.2s target on Intel Iris Xe. Document measured latency and provide projected GPU latency.
2. **Hindi TTS quality:** Piper has no Hindi voice. XTTS-v2 works but is slow on CPU. This is an ecosystem limitation, not an architecture limitation.
3. **Short Hindi utterances:** faster-whisper may misidentify language on very short phrases (e.g., "Aur Delhi mein?" — only 4 words). Document detection accuracy per scenario.

---

## ENVIRONMENT VARIABLES (.env)

```
# No API keys needed — everything runs locally
OLLAMA_BASE_URL=http://localhost:11434/v1
WHISPER_MODEL_SIZE=tiny
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
PIPER_EN_VOICE=en_US-lessac-medium
PIPER_ES_VOICE=es_ES-carlfm-x_low
TTS_HINDI_BACKEND=xtts-v2
LOG_LEVEL=INFO
```
