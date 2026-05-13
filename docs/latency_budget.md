# Latency Budget

End-of-user-speech → start-of-agent-speech.
Target: < 1200ms. Stretch target: < 800ms.

## Measured: Intel i5-1135G7, Intel Iris Xe (iGPU), 16GB RAM, Windows 11

*(Numbers below are from Phase 7 scenario testing. Updated with real measurements.)*

| Stage | Tool | Measured EN/ES (ms) | Measured HI (ms) | Notes |
|-------|------|---------------------|-------------------|-------|
| VAD (end-of-speech) | Silero VAD | ~30 | ~30 | Negligible |
| STT transcription | faster-whisper tiny, CPU | ~600–900 | ~600–900 | Depends on utterance length |
| Language detection | Part of STT | ~0 (included above) | ~0 | No separate model |
| LLM TTFT | Ollama Qwen 2.5 1.5B | ~800–1500 | ~800–1500 | First token latency |
| TTS synthesis | Piper (EN/ES) | ~150–250 | — | Fast neural TTS |
| TTS synthesis | XTTS-v2 (HI) | — | ~4000–8000 | CPU-only bottleneck |
| **TOTAL (EN/ES)** | | **~1600–2700** | | **Over target** |
| **TOTAL (HI)** | | | **~5400–10400** | **Significantly over target** |

## Projected: NVIDIA RTX 3060, 12GB VRAM

| Stage | Tool | Projected EN/ES (ms) | Projected HI (ms) | Optimization |
|-------|------|----------------------|-------------------|--------------|
| VAD | Silero VAD | ~30 | ~30 | Same (already fast) |
| STT | faster-whisper small, CUDA | ~80–120 | ~80–120 | CUDA: 5-8x faster than CPU |
| Language detection | Included in STT | ~0 | ~0 | — |
| LLM TTFT | Ollama Qwen 2.5 1.5B, GPU | ~150–300 | ~150–300 | GPU: ~5x faster |
| TTS | Piper (EN/ES) | ~50–100 | — | Piper on CPU is already fast |
| TTS | XTTS-v2, GPU | — | ~400–800 | CUDA: ~10x faster |
| **TOTAL (EN/ES)** | | **~310–550** | | **Under 800ms stretch target** |
| **TOTAL (HI)** | | | **~630–1250** | **Near or under 1200ms target** |

## Analysis

### Why we miss the target on CPU

1. **STT is the biggest CPU bottleneck.** faster-whisper tiny on CPU takes 600-900ms. On GPU
   with CUDA, this drops to 80-120ms with the `small` model (better accuracy too).

2. **XTTS-v2 is a fundamental ecosystem limitation.** There is no fast, open-source Hindi TTS
   that runs on CPU in under 1 second. The architecture is ready for GPU acceleration — it's
   not a design flaw, it's a hardware constraint.

3. **Qwen 2.5 1.5B TTFT on CPU.** 800-1500ms for the first token is acceptable for a 1.5B model
   without GPU. The 3B variant takes 2-4x longer and was eliminated in early testing.

### Optimization opportunities (with current hardware)

1. **Use `tiny.en` instead of `tiny` for English-only turns.** The English-specific model is
   slightly faster and more accurate for English. Language-switch to `tiny` for Hindi/Spanish.

2. **Pre-warm all models at startup.** First inference always incurs a JIT/loading overhead.
   Running a dummy utterance during startup eliminates this for real turns.

3. **Hindi phrase caching.** Pre-synthesize the 50 most common support phrases in Hindi
   with XTTS-v2 and cache as WAV files. Most support conversations are formulaic.

4. **Parallel TTS + LLM streaming.** Start TTS synthesis on the first completed sentence
   while LLM is still generating the rest. LiveKit Agents SDK supports this natively.

## Notes

- All CPU measurements are approximate; they vary ±20% based on system load.
- GPU projections are based on published benchmarks for faster-whisper and XTTS-v2 on RTX 3060.
- "TTFT" = Time to First Token. This is when the user first hears the agent start speaking.
