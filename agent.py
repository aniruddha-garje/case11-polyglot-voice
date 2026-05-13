"""
Case 11: Polyglot Voice Companion
Main agent entry point — livekit-agents 1.5.x console mode.

livekit-agents 1.5.x API pattern:
    AgentSession(vad, stt, llm, tts) → session.start(Agent(instructions=...), room=ctx.room)

Pipeline:
    Microphone → Silero VAD → WhisperSTT (faster-whisper + language detection)
    → LanguageRouter → Ollama LLM (Qwen 2.5 1.5B) → MultilingualTTS → Speaker

Run:
    python agent.py console
"""

import logging
import os

from dotenv import load_dotenv
from livekit.agents import Agent, AgentSession, JobContext, TurnHandlingOptions, WorkerOptions, cli, llm
from livekit.plugins import openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from plugins.language_router import LanguageRouter
from plugins.multilingual_tts import MultilingualTTS
from plugins.whisper_stt import WhisperSTT
from prompts.system_prompt import SYSTEM_PROMPT
from utils.latency import LatencyTracker

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("agent")


async def entrypoint(ctx: JobContext):
    """Main agent entrypoint — called once per console session."""

    logger.info("=" * 60)
    logger.info("  Polyglot Voice Companion starting up...")
    logger.info("=" * 60)

    # Shared state — persists across turns
    language_router = LanguageRouter()
    latency_tracker = LatencyTracker()

    # --- VAD: Silero at 8000 Hz to reduce CPU load ---
    # Default 16000 Hz causes "inference slower than realtime" on Intel i5.
    # 8000 Hz halves the computation with negligible accuracy loss for VAD.
    vad = silero.VAD.load(
        sample_rate=8000,
        min_silence_duration=0.8,       # wait 800ms of silence before end-of-turn
        min_speech_duration=0.1,        # ignore very short sounds (clicks, noise)
        activation_threshold=0.6,       # slightly higher threshold = less false positives
    )

    # --- STT: faster-whisper with language detection ---
    whisper_stt = WhisperSTT(
        model_size=os.getenv("WHISPER_MODEL_SIZE", "tiny"),
        device=os.getenv("WHISPER_DEVICE", "cpu"),
        compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
        language_router=language_router,
        latency_tracker=latency_tracker,
    )

    # --- LLM: Ollama via OpenAI-compatible endpoint ---
    ollama_llm = openai.LLM.with_ollama(
        model="qwen2.5:1.5b",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    )

    # --- TTS: Multilingual router (Piper EN/ES, pyttsx3 HI) ---
    multilingual_tts = MultilingualTTS(
        language_router=language_router,
        latency_tracker=latency_tracker,
        voices_dir="voices",
    )

    # --- AgentSession: wires all components together ---
    session = AgentSession(
        vad=vad,
        stt=whisper_stt,
        llm=ollama_llm,
        tts=multilingual_tts,
        turn_handling=TurnHandlingOptions(
            # MultilingualModel: ML-based end-of-utterance detection (EN/HI/ES).
            # Runs the lk_end_of_utterance_multilingual inference locally.
            # Much more accurate than pure silence-based VAD for turn detection.
            turn_detection=MultilingualModel(),
            endpointing={
                "min_delay": 0.5,   # ML model handles most cases; this is a safety floor
                "max_delay": 6.0,   # don't wait more than 6s for a turn to end
            },
            interruption={
                "enabled": True,
                "min_duration": 1.0,    # need at least 1s of speech to count as interruption
                "min_words": 2,         # need at least 2 words (reduces noise/cough triggers)
                "false_interruption_timeout": 3.0,
            },
        ),
    )

    # --- Hook: after STT transcription, sync language to TTS ---
    @session.on("user_input_transcribed")
    def on_transcribed(event):
        """
        Called after each STT result. At this point, language_router already
        has the updated language from WhisperSTT. We sync TTS here too.
        """
        detected_lang = language_router.get_current_language()
        multilingual_tts.set_language(detected_lang)

        logger.info(f"[AGENT] Turn language: {detected_lang} | transcript: '{event.transcript}'")

        # Print latency snapshot for the completed turn
        report = latency_tracker.report()
        if report:
            logger.info(report)
            latency_tracker.reset()

    # --- Agent: defines persona and instructions ---
    agent = Agent(instructions=SYSTEM_PROMPT)

    logger.info("[AGENT] Connecting and starting session...")
    await ctx.connect()
    await session.start(agent, room=ctx.room)

    # Greet the user on session start
    await session.generate_reply(
        instructions="Greet the user briefly. Tell them they can speak in English, Hindi, or Spanish."
    )

    logger.info("[AGENT] Ready. Speak into your microphone.")


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(entrypoint_fnc=entrypoint)
    )
