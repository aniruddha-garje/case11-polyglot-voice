# DECISIONS.md — Case 11: Polyglot Voice Companion

Design decisions, trade-offs, assumptions, and things I'd change with more time.

---

## Assumptions

| # | Assumption | Reason |
|---|-----------|--------|
| 1 | Qwen 2.5 1.5B is fast enough on CPU for demo latency | Measured TTFT ~800–1500ms on Intel i5. Acceptable for demo; not production. |
| 2 | Romanized Hindi (Hinglish) is acceptable for TTS output | Piper has no Hindi voice. XTTS-v2 handles both Devanagari and Roman. The system prompt forces Romanized Hindi to avoid script issues. |
| 3 | Console mode is sufficient for evaluation | No LiveKit server is needed for local mic-to-speaker. The brief asks for a demo video, not a deployed URL. |
| 4 | Silero VAD's end-of-speech event is the latency start point | Standard definition: "end-of-user-speech to start-of-agent-speech." |
| 5 | Language detection per utterance (not per sentence) is sufficient | The test scenarios don't have mid-utterance code-switching. One language per microphone input is the norm. |
| 6 | `language_probability > 0.7` is the threshold for a language switch | Below this, faster-whisper is unreliable on short phrases. We keep the previous language rather than misroute. |

---

## Trade-offs

| Choice | Alternative | Why I picked this |
|--------|-------------|-------------------|
| **Qwen 2.5 1.5B** | Llama 3.2 3B | Qwen 2.5 has dramatically better multilingual capability, especially Hindi and mixed-language conversations. At the same parameter count, Qwen 2.5 outperforms Llama 3.2 on non-English benchmarks. |
| **1.5B over 3B** | qwen2.5:3b | Intel i5 with no GPU. The 3B model produced 4–6s TTFT on CPU in initial testing, well over the 1.2s target. 1.5B gives ~800–1500ms TTFT, which is still over target but demonstrably closer. |
| **faster-whisper** | whisper.cpp | Native Python API, returns `language` and `language_probability` per segment as first-class fields. whisper.cpp requires ctypes or subprocess to get language metadata. |
| **tiny model (default)** | small, base | On Intel i5 CPU, tiny transcribes a 5-second utterance in ~600ms. Small takes ~1.8s. The latency budget is already tight — tiny is the only viable option for near-realtime. |
| **Piper TTS (EN/ES)** | XTTS-v2, Kokoro | Piper synthesizes in ~150–250ms on CPU. XTTS-v2 takes 4–8s. For English and Spanish, Piper has excellent voice quality and negligible latency. |
| **XTTS-v2 (HI)** | Piper (HI), gTTS | Piper has no published Hindi voice model as of May 2026. gTTS requires an internet connection (violates the "local inference" constraint). XTTS-v2 is the only open-source option that produces acceptable Hindi speech. |
| **LiveKit Agents SDK** | Raw Python (pyaudio + threading) | Gives us VAD integration, turn detection, barge-in, and streaming TTS handling for free. Apache 2.0 license. The alternative is hundreds of lines of audio pipeline code we'd have to write and debug ourselves. |
| **console mode** | LiveKit server + room | No LiveKit server required. Console mode handles local mic I/O directly. This is the correct mode for a single-user local demo. |
| **Per-turn language injection in system prompt** | Fine-tuned language switching | Zero additional cost. The LLM already follows instructions well. Fine-tuning would require training data and GPU time we don't have. |

---

## De-scoped items

| Feature | Why de-scoped |
|---------|---------------|
| **Streaming TTS** | LiveKit Agents SDK supports streaming TTS natively, but Piper and XTTS-v2 don't stream token-by-token. The architecture is ready for it when a streaming-capable TTS is available. |
| **Code-switching within a single utterance** | "Mujhe order 4421 ka status chahiye" (Hindi + English number) — faster-whisper handles this fine in transcription, but TTS routing operates per-utterance. Handling mid-utterance switches would require sentence-level language detection, which adds complexity and latency. |
| **Auto-degrade to smaller model** | Would require monitoring TTFT and hot-swapping the Ollama model mid-session. Too complex for the time budget. The 1.5B model is already the smallest practical option. |
| **Hindi Piper voice** | No published Hindi Piper voice exists in the official repository. Training one requires ~20 hours of clean Hindi audio and GPU compute. |
| **Automated test suite** | Tests in `tests/` are documented as manual test scripts. Proper async testing with real audio requires hardware and recorded test audio. |

---

## What I'd do differently with another day

1. **Pre-warm all models at startup.** The first utterance always has extra latency (model loading). A startup sequence that runs a dummy transcription and dummy TTS call would eliminate this.
2. **Hindi TTS caching.** Pre-synthesize the 50 most common customer support phrases in Hindi and cache them as WAV files. Most support conversations reuse standard phrases, making XTTS-v2's 6s latency nearly irrelevant.
3. **Parallel STT + LLM.** On the first utterance, we know VAD has ended. We could start the LLM with a partial prompt while STT is still running (speculative execution). Shaves ~200ms off the pipeline.
4. **Quantized Qwen model.** Use a Q4_K_M GGUF instead of the default Ollama format. Smaller memory footprint, faster inference on CPU.
5. **Language detection fallback.** If `language_probability < 0.5`, ask the user "Could you repeat that?" rather than hallucinating a language detection.

---

## AI tools used

Claude (Anthropic) was used for code generation and architecture planning throughout this project. All generated code was reviewed, understood, and tested. The architecture decisions, trade-off analysis, and technical rationale in this document reflect my own understanding of the problem.
