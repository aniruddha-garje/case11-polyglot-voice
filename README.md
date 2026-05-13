# Case 11: Polyglot Voice Companion

A real-time multilingual voice agent that speaks English, Hindi, and Spanish — and switches between them mid-conversation without losing context.

**Demo video:** [Link to be added after recording]

---

## What it does

1. Listens via microphone (Silero VAD detects end-of-speech)
2. Transcribes speech and detects language per utterance (faster-whisper)
3. Routes to the correct language persona in the LLM (Ollama + Qwen 2.5)
4. Speaks the response via the matching TTS voice (Piper for EN/ES, XTTS-v2 for HI)
5. Maintains full conversation context across language switches

**Supported languages:** English (`en`), Hindi (`hi`), Spanish (`es`)

---

## How to run locally

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) installed and running
- Windows 11 (Linux/Mac supported via `setup.sh`)

### Setup

```bash
# 1. Clone the repo
git clone <repo-url>
cd case11-polyglot-voice

# 2. Run setup (creates venv, installs deps, downloads Piper voices)
setup.bat          # Windows
# bash setup.sh    # Linux/Mac

# 3. Start Ollama in a separate terminal
ollama serve
ollama pull qwen2.5:1.5b

# 4. Activate venv and run
venv\Scripts\activate
python agent.py console
```

### Environment variables

Copy `.env.example` to `.env`. No API keys needed — everything runs locally.

---

## Stack

| Component | Tool | Why |
|-----------|------|-----|
| Orchestration | LiveKit Agents SDK ~1.0 (console mode) | Apache 2.0, streaming TTS + barge-in built in |
| VAD | Silero VAD | Lightweight, accurate end-of-speech detection |
| STT | faster-whisper (tiny/small) | Native Python, returns language per segment |
| Language ID | faster-whisper `.language` field | Free from the STT step, no extra model |
| LLM | Ollama + Qwen 2.5 1.5B | Best multilingual quality at 1.5B param count |
| TTS (EN/ES) | Piper TTS | ~200ms on CPU, natural-sounding voices |
| TTS (HI) | Coqui XTTS-v2 | Only open-source option with Hindi support |
| Turn detection | LiveKit turn detector | Reduces false end-of-turn triggers |

**Hard constraint:** All inference is local. No OpenAI, Google, Azure, or any managed API.

---

## Measured latency (Intel i5, Iris Xe, CPU-only)

| Stage | Measured | Target |
|-------|----------|--------|
| VAD | ~30ms | — |
| STT (tiny) | ~600–900ms | — |
| LLM TTFT | ~800–1500ms | — |
| TTS EN (Piper) | ~150–250ms | — |
| TTS HI (XTTS) | ~4000–8000ms | — |
| **Total (EN/ES)** | **~1.6–2.7s** | **<1.2s** |

The 1.2s target is achievable on GPU (RTX 3060: projected ~700ms). See `docs/latency_budget.md` for the full table.

---

## What's not done / known limitations

1. **Latency target not met on CPU.** The 1.2s target requires GPU. See `docs/latency_budget.md` for the analysis and the GPU projection.
2. **Hindi TTS is slow.** XTTS-v2 is the only open-source option and takes 4–8s on CPU. Piper has no Hindi voice model.
3. **Short Hindi utterances may misdetect language.** faster-whisper needs at least 3–5 words for reliable language ID. Documented in `docs/scenario_results.md`.
4. **No streaming STT.** faster-whisper processes full utterances, not token streams. LiveKit's VAD handles the end-of-speech boundary.
5. **Docker mic passthrough requires extra setup.** Docker is for reproducibility verification; live mic demo runs natively.

---

## What I'd do in production

- **GPU inference:** Move Whisper and XTTS-v2 to GPU. Estimated 4–6x speedup.
- **Streaming STT:** Integrate Whisper streaming for sub-200ms TTFT.
- **Hindi TTS:** Either fine-tune a Piper voice on Hindi data or use a cloud TTS with a local cache for common phrases.
- **Language detection confidence:** If `language_probability < 0.5`, prompt the user to repeat rather than misroute.
- **Monitoring:** Add Prometheus metrics for per-turn latency, language distribution, and error rates.

---

## Project structure

```
agent.py                 # Main entry point
plugins/
  whisper_stt.py         # Custom STT: faster-whisper + language detection
  multilingual_tts.py    # Custom TTS: routes to Piper/XTTS by language
  language_router.py     # Language detection + routing state
prompts/
  system_prompt.py       # Multilingual system prompt
utils/
  latency.py             # Latency measurement
docs/
  architecture.md        # Mermaid pipeline diagram
  latency_budget.md      # Measured + projected latency table
  scenario_results.md    # Test scenario results
scripts/
  conversation_scripts.md # The 4 test scenarios
tests/                   # Test files
```

---

*Built for a college assignment (Case 11). All models are open-source, all inference is local.*
