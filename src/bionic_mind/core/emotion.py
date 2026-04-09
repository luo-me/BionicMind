from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from loguru import logger


class EmotionMode(str, Enum):
    FOCUSED = "focused"
    EXPLORATORY = "exploratory"
    SATISFIED = "satisfied"
    DEPRESSED = "depressed"
    NEUTRAL = "neutral"


@dataclass
class EmotionState:
    valence: float = 0.0
    arousal: float = 0.3

    @property
    def mode(self) -> EmotionMode:
        if self.arousal > 0.7 and self.valence < -0.3:
            return EmotionMode.FOCUSED
        elif self.arousal > 0.7 and self.valence > 0.3:
            return EmotionMode.EXPLORATORY
        elif self.arousal < 0.3 and self.valence > 0.3:
            return EmotionMode.SATISFIED
        elif self.arousal < 0.3 and self.valence < -0.3:
            return EmotionMode.DEPRESSED
        else:
            return EmotionMode.NEUTRAL

    def describe(self) -> str:
        mode_names = {
            EmotionMode.FOCUSED: "焦虑/警觉",
            EmotionMode.EXPLORATORY: "兴奋/好奇",
            EmotionMode.SATISFIED: "满足/安全",
            EmotionMode.DEPRESSED: "低落/无聊",
            EmotionMode.NEUTRAL: "平静",
        }
        return f"{mode_names[self.mode]}(效价{self.valence:.2f}, 唤醒{self.arousal:.2f})"

    def to_dict(self) -> dict[str, float]:
        return {"valence": self.valence, "arousal": self.arousal}


@dataclass
class EmotionModulation:
    mode: EmotionMode
    temperature_offset: float = 0.0
    search_breadth: str = "normal"
    creativity_boost: float = 0.0
    focus_narrowing: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "temperature_offset": self.temperature_offset,
            "search_breadth": self.search_breadth,
            "creativity_boost": self.creativity_boost,
            "focus_narrowing": self.focus_narrowing,
        }


class EmotionSystem:
    def __init__(
        self,
        valence_weights: dict[str, float] | None = None,
        arousal_weights: dict[str, float] | None = None,
        history_size: int = 100,
        smoothing_factor: float = 0.3,
    ):
        self.valence_weights = valence_weights or {
            "prediction_error": -0.30,
            "goal_progress": 0.20,
            "consistency": 0.20,
            "social_feedback": 0.20,
            "novelty": 0.10,
        }
        self.arousal_weights = arousal_weights or {
            "prediction_error": 0.30,
            "goal_progress_rate": 0.20,
            "valence_magnitude": 0.20,
            "novelty": 0.30,
        }
        self.smoothing_factor = smoothing_factor
        self.state = EmotionState()
        self.valence_history: deque[float] = deque(maxlen=history_size)
        self.arousal_history: deque[float] = deque(maxlen=history_size)

    def evaluate(
        self,
        prediction_error: float = 0.0,
        goal_progress: float = 0.0,
        consistency: float = 0.5,
        social_feedback: float = 0.0,
        novelty: float = 0.3,
        goal_progress_rate: float = 0.0,
    ) -> EmotionState:
        wv = self.valence_weights
        wa = self.arousal_weights

        novelty_valence = max(0.0, 0.5 - abs(novelty - 0.5)) * 2.0

        raw_valence = (
            wv["prediction_error"] * min(prediction_error, 1.0)
            + wv["goal_progress"] * goal_progress
            + wv["consistency"] * (consistency - 0.5) * 2
            + wv["social_feedback"] * social_feedback
            + wv["novelty"] * novelty_valence
        )

        raw_arousal = (
            wa["prediction_error"] * min(prediction_error, 1.0)
            + wa["goal_progress_rate"] * abs(goal_progress_rate)
            + wa["valence_magnitude"] * abs(raw_valence)
            + wa["novelty"] * novelty
        )

        new_valence = float(np.clip(raw_valence, -1, 1))
        new_arousal = float(np.clip(raw_arousal, 0, 1))

        alpha = self.smoothing_factor
        self.state.valence = alpha * new_valence + (1 - alpha) * self.state.valence
        self.state.arousal = alpha * new_arousal + (1 - alpha) * self.state.arousal

        self.valence_history.append(self.state.valence)
        self.arousal_history.append(self.state.arousal)

        logger.debug(
            f"Emotion evaluated: {self.state.describe()} "
            f"(raw_v={raw_valence:.3f}, raw_a={raw_arousal:.3f})"
        )
        return self.state

    def get_modulation(self) -> EmotionModulation:
        mode = self.state.mode
        if mode == EmotionMode.FOCUSED:
            return EmotionModulation(
                mode=mode,
                temperature_offset=-0.3,
                search_breadth="narrow",
                creativity_boost=0.0,
                focus_narrowing=0.5,
            )
        elif mode == EmotionMode.EXPLORATORY:
            return EmotionModulation(
                mode=mode,
                temperature_offset=0.2,
                search_breadth="wide",
                creativity_boost=0.4,
                focus_narrowing=0.0,
            )
        elif mode == EmotionMode.SATISFIED:
            return EmotionModulation(
                mode=mode,
                temperature_offset=-0.1,
                search_breadth="minimal",
                creativity_boost=0.0,
                focus_narrowing=0.1,
            )
        elif mode == EmotionMode.DEPRESSED:
            return EmotionModulation(
                mode=mode,
                temperature_offset=0.1,
                search_breadth="normal",
                creativity_boost=0.1,
                focus_narrowing=0.0,
            )
        else:
            return EmotionModulation(
                mode=mode,
                temperature_offset=0.0,
                search_breadth="normal",
                creativity_boost=0.0,
                focus_narrowing=0.0,
            )

    def get_trend(self) -> dict[str, str]:
        if len(self.valence_history) < 5:
            return {"valence_trend": "stable", "arousal_trend": "stable"}

        recent_v = list(self.valence_history)[-5:]
        recent_a = list(self.arousal_history)[-5:]

        v_diff = recent_v[-1] - recent_v[0]
        a_diff = recent_a[-1] - recent_a[0]

        def trend(diff: float) -> str:
            if diff > 0.1:
                return "rising"
            elif diff < -0.1:
                return "falling"
            return "stable"

        return {
            "valence_trend": trend(v_diff),
            "arousal_trend": trend(a_diff),
        }

    def reset(self) -> None:
        self.state = EmotionState()
        self.valence_history.clear()
        self.arousal_history.clear()

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.to_dict(),
            "mode": self.state.mode.value,
            "modulation": self.get_modulation().to_dict(),
            "trend": self.get_trend(),
        }
