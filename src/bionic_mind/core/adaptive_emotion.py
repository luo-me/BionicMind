from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger


@dataclass
class EmotionWeights:
    prediction_error: float = -0.30
    goal_progress: float = 0.20
    consistency: float = 0.20
    social_feedback: float = 0.20
    novelty: float = 0.10

    arousal_prediction_error: float = 0.30
    arousal_goal_progress_rate: float = 0.20
    arousal_valence_magnitude: float = 0.20
    arousal_novelty: float = 0.30

    def to_list(self) -> list[float]:
        return [
            self.prediction_error, self.goal_progress, self.consistency,
            self.social_feedback, self.novelty,
            self.arousal_prediction_error, self.arousal_goal_progress_rate,
            self.arousal_valence_magnitude, self.arousal_novelty,
        ]

    @classmethod
    def from_list(cls, values: list[float]) -> EmotionWeights:
        return cls(
            prediction_error=values[0],
            goal_progress=values[1],
            consistency=values[2],
            social_feedback=values[3],
            novelty=values[4],
            arousal_prediction_error=values[5],
            arousal_goal_progress_rate=values[6],
            arousal_valence_magnitude=values[7],
            arousal_novelty=values[8],
        )


class AdaptiveEmotionLearner:
    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        weight_bounds: tuple[float, float] = (-1.0, 1.0),
    ):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_bounds = weight_bounds
        self.weights = EmotionWeights()
        self._velocity = [0.0] * 9
        self._update_count = 0

    def update_weights(
        self,
        inputs: dict[str, float],
        valence_target: float,
        arousal_target: float,
    ) -> dict[str, Any]:
        current_v = self._compute_valence(inputs)
        current_a = self._compute_arousal(inputs)

        valence_error = valence_target - current_v
        arousal_error = arousal_target - current_a

        w = self.weights.to_list()
        input_order = [
            inputs.get("prediction_error", 0),
            inputs.get("goal_progress", 0),
            inputs.get("consistency", 0.5),
            inputs.get("social_feedback", 0),
            inputs.get("novelty", 0.3),
            inputs.get("prediction_error", 0),
            inputs.get("goal_progress_rate", 0),
            abs(valence_target),
            inputs.get("novelty", 0.3),
        ]

        for i in range(5):
            gradient = valence_error * input_order[i]
            self._velocity[i] = self.momentum * self._velocity[i] + self.learning_rate * gradient
            w[i] = np.clip(w[i] + self._velocity[i], *self.weight_bounds)

        for i in range(5, 9):
            gradient = arousal_error * input_order[i]
            self._velocity[i] = self.momentum * self._velocity[i] + self.learning_rate * gradient
            w[i] = np.clip(w[i] + self._velocity[i], *self.weight_bounds)

        self.weights = EmotionWeights.from_list(w)
        self._update_count += 1

        logger.debug(
            f"Emotion weights updated: v_err={valence_error:.3f}, a_err={arousal_error:.3f}, "
            f"count={self._update_count}"
        )

        return {
            "valence_error": valence_error,
            "arousal_error": arousal_error,
            "update_count": self._update_count,
        }

    def _compute_valence(self, inputs: dict[str, float]) -> float:
        w = self.weights
        return (
            w.prediction_error * inputs.get("prediction_error", 0)
            + w.goal_progress * inputs.get("goal_progress", 0)
            + w.consistency * (inputs.get("consistency", 0.5) - 0.5) * 2
            + w.social_feedback * inputs.get("social_feedback", 0)
            + w.novelty * max(0, 0.5 - abs(inputs.get("novelty", 0.3) - 0.5)) * 2
        )

    def _compute_arousal(self, inputs: dict[str, float]) -> float:
        w = self.weights
        return (
            w.arousal_prediction_error * inputs.get("prediction_error", 0)
            + w.arousal_goal_progress_rate * abs(inputs.get("goal_progress_rate", 0))
            + w.arousal_valence_magnitude * abs(self._compute_valence(inputs))
            + w.arousal_novelty * inputs.get("novelty", 0.3)
        )

    def get_weights(self) -> EmotionWeights:
        return self.weights

    def get_stats(self) -> dict[str, Any]:
        return {
            "update_count": self._update_count,
            "weights": self.weights.to_list(),
            "velocity": self._velocity,
        }
