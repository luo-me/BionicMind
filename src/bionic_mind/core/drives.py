from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from loguru import logger


@dataclass
class DriveState:
    curiosity: float = 0.5
    consistency: float = 0.5
    social_connection: float = 0.5
    energy_efficiency: float = 0.5
    self_preservation: float = 0.5

    def to_dict(self) -> dict[str, float]:
        return {
            "curiosity": self.curiosity,
            "consistency": self.consistency,
            "social_connection": self.social_connection,
            "energy_efficiency": self.energy_efficiency,
            "self_preservation": self.self_preservation,
        }

    def dominant(self) -> tuple[str, float]:
        drives = self.to_dict()
        name = max(drives, key=drives.get)
        return name, drives[name]

    def total_intensity(self) -> float:
        values = list(self.to_dict().values())
        return float(np.mean(values))


class DriveSystem:
    def __init__(
        self,
        curiosity_baseline: float = 0.3,
        social_feedback_timeout_hours: float = 24.0,
        spontaneous_action_threshold: float = 0.7,
        smoothing_factor: float = 0.2,
    ):
        self.curiosity_baseline = curiosity_baseline
        self.social_feedback_timeout_hours = social_feedback_timeout_hours
        self.spontaneous_action_threshold = spontaneous_action_threshold
        self.smoothing_factor = smoothing_factor
        self.drives = DriveState()
        self.last_social_feedback_time = datetime.now()
        self._drive_history: list[dict[str, float]] = []

    def update(
        self,
        prediction_error: float = 0.0,
        consistency: float = 0.5,
        social_feedback: float = 0.0,
        resource_usage: float = 0.3,
        core_threat_level: float = 0.0,
    ) -> DriveState:
        new_curiosity = np.clip(
            self.curiosity_baseline + 0.7 * prediction_error, 0, 1
        )
        new_consistency = np.clip(1.0 - consistency, 0, 1)

        hours_since = (
            (datetime.now() - self.last_social_feedback_time).total_seconds() / 3600
        )
        social_deprivation = min(hours_since / self.social_feedback_timeout_hours, 1.0)
        new_social = np.clip(0.2 + 0.8 * social_deprivation, 0, 1)

        if social_feedback > 0:
            self.last_social_feedback_time = datetime.now()
            new_social = np.clip(0.1 + 0.1 * (1.0 - social_deprivation), 0, 1)

        new_energy = np.clip(resource_usage, 0, 1)
        new_preservation = np.clip(core_threat_level, 0, 1)

        alpha = self.smoothing_factor
        self.drives.curiosity = float(alpha * new_curiosity + (1 - alpha) * self.drives.curiosity)
        self.drives.consistency = float(alpha * new_consistency + (1 - alpha) * self.drives.consistency)
        self.drives.social_connection = float(alpha * new_social + (1 - alpha) * self.drives.social_connection)
        self.drives.energy_efficiency = float(alpha * new_energy + (1 - alpha) * self.drives.energy_efficiency)
        self.drives.self_preservation = float(alpha * new_preservation + (1 - alpha) * self.drives.self_preservation)

        self._drive_history.append(self.drives.to_dict())
        if len(self._drive_history) > 200:
            self._drive_history.pop(0)

        logger.debug(
            f"Drives updated: curiosity={self.drives.curiosity:.2f}, "
            f"consistency={self.drives.consistency:.2f}, "
            f"social={self.drives.social_connection:.2f}"
        )
        return self.drives

    def should_act_spontaneously(self) -> bool:
        _, intensity = self.drives.dominant()
        return intensity > self.spontaneous_action_threshold

    def get_spontaneous_prompt(self) -> str | None:
        if not self.should_act_spontaneously():
            return None
        dominant, intensity = self.drives.dominant()
        prompts = {
            "curiosity": f"我的好奇心驱力很强({intensity:.1f})，我想主动探索和学习一些新东西。",
            "consistency": f"我的内部一致性驱力很强({intensity:.1f})，我需要梳理和整合我已有的知识。",
            "social_connection": f"我的社交连接驱力很强({intensity:.1f})，我想主动与用户互动。",
            "energy_efficiency": f"我的能量效率驱力很强({intensity:.1f})，我需要优化我的资源使用方式。",
            "self_preservation": f"我的自我保存驱力很强({intensity:.1f})，我需要检查和维护我的核心状态。",
        }
        return prompts.get(dominant, None)

    def get_drive_gradient(self) -> dict[str, float]:
        if len(self._drive_history) < 2:
            return {k: 0.0 for k in self.drives.to_dict()}
        prev = self._drive_history[-2]
        curr = self._drive_history[-1]
        return {k: curr[k] - prev[k] for k in curr}

    def decay_curiosity(self, amount: float = 0.05) -> None:
        self.drives.curiosity = max(0.0, self.drives.curiosity - amount)

    def reset(self) -> None:
        self.drives = DriveState()
        self.last_social_feedback_time = datetime.now()
        self._drive_history.clear()

    def to_dict(self) -> dict[str, Any]:
        dominant_name, dominant_value = self.drives.dominant()
        return {
            "drives": self.drives.to_dict(),
            "dominant": {"name": dominant_name, "value": dominant_value},
            "should_act_spontaneously": self.should_act_spontaneously(),
            "gradient": self.get_drive_gradient(),
        }
