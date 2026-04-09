from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class MetaAction:
    action_type: str
    target: str
    value: float
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "target": self.target,
            "value": self.value,
            "reason": self.reason,
        }


class MetaActionSystem:
    def __init__(
        self,
        learning_rate_range: tuple[float, float] = (0.01, 0.5),
        drive_weight_range: tuple[float, float] = (0.1, 2.0),
        emotion_sensitivity_range: tuple[float, float] = (0.1, 2.0),
        noise_range: tuple[float, float] = (0.0, 0.5),
        cooldown_cycles: int = 10,
    ):
        self.learning_rate_range = learning_rate_range
        self.drive_weight_range = drive_weight_range
        self.emotion_sensitivity_range = emotion_sensitivity_range
        self.noise_range = noise_range
        self.cooldown_cycles = cooldown_cycles

        self.current_learning_rate = 0.1
        self.current_drive_weights: dict[str, float] = {
            "curiosity": 1.0,
            "consistency": 1.0,
            "social_connection": 1.0,
            "energy_efficiency": 1.0,
            "self_preservation": 1.0,
        }
        self.current_emotion_sensitivity = 1.0
        self.current_noise_level = 0.1

        self._action_history: list[MetaAction] = []
        self._cycles_since_last_action = 0
        self._total_meta_actions = 0

    def evaluate_and_propose(
        self,
        emotion_state: dict[str, float],
        drive_state: dict[str, float],
        performance_metrics: dict[str, float],
    ) -> list[MetaAction]:
        self._cycles_since_last_action += 1
        if self._cycles_since_last_action < self.cooldown_cycles:
            return []

        actions: list[MetaAction] = []

        avg_error = performance_metrics.get("avg_prediction_error", 0.5)
        if avg_error > 0.7:
            new_lr = min(self.current_learning_rate * 1.2, self.learning_rate_range[1])
            actions.append(MetaAction(
                action_type="adjust_learning_rate",
                target="learning_rate",
                value=new_lr,
                reason=f"High prediction error ({avg_error:.2f}), increasing learning rate",
            ))
        elif avg_error < 0.2:
            new_lr = max(self.current_learning_rate * 0.8, self.learning_rate_range[0])
            actions.append(MetaAction(
                action_type="adjust_learning_rate",
                target="learning_rate",
                value=new_lr,
                reason=f"Low prediction error ({avg_error:.2f}), decreasing learning rate",
            ))

        arousal = emotion_state.get("arousal", 0.3)
        valence = emotion_state.get("valence", 0.0)
        if arousal > 0.8 and valence < -0.5:
            new_sens = max(self.current_emotion_sensitivity * 0.9, self.emotion_sensitivity_range[0])
            actions.append(MetaAction(
                action_type="adjust_emotion_sensitivity",
                target="emotion_sensitivity",
                value=new_sens,
                reason="Prolonged high-arousal negative state, reducing sensitivity",
            ))
        elif arousal < 0.2 and abs(valence) < 0.2:
            new_sens = min(self.current_emotion_sensitivity * 1.1, self.emotion_sensitivity_range[1])
            actions.append(MetaAction(
                action_type="adjust_emotion_sensitivity",
                target="emotion_sensitivity",
                value=new_sens,
                reason="Prolonged flat emotional state, increasing sensitivity",
            ))

        dominant_drive = max(drive_state, key=drive_state.get)
        dominant_value = drive_state[dominant_drive]
        if dominant_value > 0.9:
            for drive_name, weight in self.current_drive_weights.items():
                if drive_name == dominant_drive:
                    new_weight = max(weight * 0.9, self.drive_weight_range[0])
                    actions.append(MetaAction(
                        action_type="adjust_drive_weight",
                        target=drive_name,
                        value=new_weight,
                        reason=f"Drive {dominant_drive} too dominant ({dominant_value:.2f}), reducing weight",
                    ))
                    break

        creativity_need = performance_metrics.get("creativity_need", 0.0)
        if creativity_need > 0.7:
            new_noise = min(self.current_noise_level + 0.05, self.noise_range[1])
            actions.append(MetaAction(
                action_type="adjust_noise",
                target="noise_level",
                value=new_noise,
                reason="High creativity need, increasing exploration noise",
            ))

        return actions

    def apply(self, actions: list[MetaAction]) -> dict[str, Any]:
        applied = {}
        for action in actions:
            if action.action_type == "adjust_learning_rate":
                self.current_learning_rate = action.value
                applied["learning_rate"] = action.value
            elif action.action_type == "adjust_drive_weight":
                self.current_drive_weights[action.target] = action.value
                applied[f"drive_weight_{action.target}"] = action.value
            elif action.action_type == "adjust_emotion_sensitivity":
                self.current_emotion_sensitivity = action.value
                applied["emotion_sensitivity"] = action.value
            elif action.action_type == "adjust_noise":
                self.current_noise_level = action.value
                applied["noise_level"] = action.value

            self._action_history.append(action)
            self._total_meta_actions += 1

        if applied:
            self._cycles_since_last_action = 0
            logger.info(f"Meta-actions applied: {applied}")

        return applied

    def get_state(self) -> dict[str, Any]:
        return {
            "learning_rate": self.current_learning_rate,
            "drive_weights": self.current_drive_weights,
            "emotion_sensitivity": self.current_emotion_sensitivity,
            "noise_level": self.current_noise_level,
            "total_meta_actions": self._total_meta_actions,
            "cycles_since_last_action": self._cycles_since_last_action,
        }
