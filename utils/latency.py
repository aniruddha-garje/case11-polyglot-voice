"""
Latency measurement utilities for the Polyglot Voice pipeline.

Tracks time for each stage: VAD → STT → LLM → TTS.
Produces a formatted report after each turn.

Usage:
    tracker = LatencyTracker()
    tracker.start("stt")
    ... do work ...
    tracker.end("stt")
    print(tracker.report())
    tracker.reset()
"""

import logging
import time

logger = logging.getLogger("latency")

# Canonical stage order for the report table
STAGE_ORDER = ["vad", "stt", "lang_detect", "llm_ttft", "llm_total", "tts", "total"]


class LatencyTracker:
    """
    Per-turn latency tracker. Call start(stage)/end(stage) around each pipeline step.
    After the turn, call report() then reset().
    """

    def __init__(self):
        self.stages: dict = {}
        self._turn_count = 0

    def start(self, stage: str):
        """Record the start time for a pipeline stage."""
        self.stages[stage] = {"start": time.perf_counter_ns(), "elapsed_ms": None}

    def end(self, stage: str) -> float:
        """Record the end time and compute elapsed ms. Returns elapsed ms."""
        if stage not in self.stages or self.stages[stage]["start"] is None:
            logger.warning(f"[LATENCY] end() called for '{stage}' but start() was never called.")
            return 0.0
        elapsed_ms = (time.perf_counter_ns() - self.stages[stage]["start"]) / 1_000_000
        self.stages[stage]["elapsed_ms"] = elapsed_ms
        logger.debug(f"[LATENCY] {stage}: {elapsed_ms:.1f}ms")
        return elapsed_ms

    def record(self, stage: str, elapsed_ms: float):
        """Directly record a latency value (use when you already have timing data)."""
        self.stages[stage] = {"start": None, "elapsed_ms": elapsed_ms}

    def total(self) -> float:
        """Sum of all measured stages (excluding 'total' itself to avoid double-counting)."""
        return sum(
            s.get("elapsed_ms", 0) or 0
            for k, s in self.stages.items()
            if k not in ("total", "llm_total")  # llm_total includes llm_ttft
        )

    def report(self) -> str:
        """
        Return a formatted latency table. Example output:
            [TURN 3] VAD: 32ms | STT: 720ms | LLM-TTFT: 1150ms | TTS: 220ms | TOTAL: 2122ms
        """
        if not self.stages:
            return ""

        self._turn_count += 1

        parts = []
        label_map = {
            "vad": "VAD",
            "stt": "STT",
            "lang_detect": "LangDetect",
            "llm_ttft": "LLM-TTFT",
            "llm_total": "LLM-Total",
            "tts": "TTS",
        }

        for stage in STAGE_ORDER:
            if stage == "total":
                continue
            if stage in self.stages and self.stages[stage].get("elapsed_ms") is not None:
                label = label_map.get(stage, stage.upper())
                ms = self.stages[stage]["elapsed_ms"]
                parts.append(f"{label}: {ms:.0f}ms")

        total_ms = self.total()
        parts.append(f"TOTAL: {total_ms:.0f}ms")

        line = " | ".join(parts)
        full_report = f"[TURN {self._turn_count}] {line}"
        logger.info(full_report)
        return full_report

    def report_table(self) -> str:
        """
        Return a multi-line markdown table for documentation use.
        Used when populating docs/latency_budget.md.
        """
        lines = ["Stage | Measured (ms)"]
        lines.append("------|-------------")
        for stage in STAGE_ORDER:
            if stage == "total":
                continue
            if stage in self.stages and self.stages[stage].get("elapsed_ms") is not None:
                ms = self.stages[stage]["elapsed_ms"]
                lines.append(f"{stage} | {ms:.1f}")
        lines.append(f"**TOTAL** | **{self.total():.1f}**")
        return "\n".join(lines)

    def reset(self):
        """Clear stage data after a turn. Call after report()."""
        self.stages = {}
