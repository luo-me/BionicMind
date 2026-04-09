from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass
class PerceptionResult:
    text: str
    source: str = "user"
    metadata: dict[str, Any] | None = None

    def is_spontaneous(self) -> bool:
        return self.source == "drive"


class PerceptionEncoder:
    def __init__(self):
        self._last_perception_time: float = 0.0
        self._perception_count: int = 0

    def encode_user_input(self, user_input: str) -> PerceptionResult:
        self._perception_count += 1
        logger.debug(f"Encoding user input #{self._perception_count}")
        return PerceptionResult(
            text=user_input,
            source="user",
            metadata={"perception_id": self._perception_count},
        )

    def encode_drive_signal(self, drive_prompt: str) -> PerceptionResult:
        self._perception_count += 1
        logger.debug(f"Encoding drive signal: {drive_prompt[:50]}...")
        return PerceptionResult(
            text=drive_prompt,
            source="drive",
            metadata={"perception_id": self._perception_count},
        )

    def encode_feedback(self, feedback_text: str, feedback_type: str = "implicit") -> PerceptionResult:
        self._perception_count += 1
        return PerceptionResult(
            text=feedback_text,
            source="feedback",
            metadata={"perception_id": self._perception_count, "feedback_type": feedback_type},
        )

    def encode_system_event(self, event_text: str) -> PerceptionResult:
        self._perception_count += 1
        return PerceptionResult(
            text=event_text,
            source="system",
            metadata={"perception_id": self._perception_count},
        )

    def compute_novelty(self, perception: str, recent_perceptions: list[str]) -> float:
        if not recent_perceptions:
            return 1.0
        max_sim = 0.0
        for prev in recent_perceptions[-10:]:
            sim = self._simple_similarity(perception, prev)
            max_sim = max(max_sim, sim)
        return 1.0 - max_sim

    @staticmethod
    def _simple_similarity(a: str, b: str) -> float:
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        word_sim = len(intersection) / len(union)
        chars_a = set(a.lower())
        chars_b = set(b.lower())
        if chars_a and chars_b:
            char_sim = len(chars_a & chars_b) / len(chars_a | chars_b)
        else:
            char_sim = 0.0
        return 0.7 * word_sim + 0.3 * char_sim
