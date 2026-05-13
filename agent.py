"""
Case 11: Polyglot Voice Companion
Main agent entry point — LiveKit Agents console mode.

Pipeline:
    Microphone → Silero VAD → faster-whisper STT (+ language detection)
    → Language Router → Ollama LLM (Qwen 2.5) → Multilingual TTS → Speaker

Run:
    python agent.py console
"""

import logging
import os

from dotenv import load_dotenv
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.voice import AgentSession, VoicePipelineAgent
from livekit.plugins import openai, silero, turn_detector

from plugins.language_router import LanguageRouter
from plugins.multilingual_tts import MultilingualTTS
from plugins.whisper_stt import WhisperSTT
from prompts.system_prompt import SYSTEM_PROMPT
from utils.latency import LatencyTracker

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("agent")


async def entrypoint(ctx: JobContext):
    """Main agent entrypoint called by LiveKit Workers."""

    logger.info("Starting Polyglot Voice Companion...")

    # Shared state: language router tracks detected language across turns
    language_router = LanguageRouter()
    latency_tracker = LatencyTracker()

    # --- STT: faster-whisper with language detection ---
    whisper_stt = WhisperSTT(
        model_size=os.getenv("WHISPER_MODEL_SIZE", "tiny"),
        device=os.getenv("WHISPER_DEVICE", "cpu"),
        compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
        language_router=language_router,
        latency_tracker=latency_tracker,
    )

    # --- LLM: Ollama via OpenAI-compatible API ---
    ollama_llm = openai.LLM.with_ollama(
        model="qwen2.5:1.5b",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    )

    # --- TTS: Multilingual router (Piper EN/ES, XTTS-v2 HI) ---
    multilingual_tts = MultilingualTTS(
        language_router=language_router,
        latency_tracker=latency_tracker,
    )

    # --- VAD: Silero ---
    vad = silero.VAD.load()

    # --- Turn detection ---
    turn_det = turn_detector.EOUModel()

    # --- Build the agent session ---
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=SYSTEM_PROMPT,
    )

    agent = VoicePipelineAgent(
        vad=vad,
        stt=whisper_stt,
        llm=ollama_llm,
        tts=multilingual_tts,
        turn_detector=turn_det,
        chat_ctx=initial_ctx,
    )

    # Hook: after each user utterance, sync language to TTS
    @agent.on("user_speech_committed")
    def on_user_speech_committed(msg: llm.ChatMessage):
        detected_lang = language_router.get_current_language()
        multilingual_tts.set_language(detected_lang)
        logger.info(f"[AGENT] Language for this turn: {detected_lang}")

        # Log a latency summary after each completed turn
        report = latency_tracker.report()
        if report:
            logger.info(f"\n{report}")
            latency_tracker.reset()

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    agent.start(ctx.room)

    logger.info("Agent ready. Speak into your microphone.")
    await agent.say("Hello! I'm your multilingual assistant. You can speak in English, Hindi, or Spanish.", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
    )
