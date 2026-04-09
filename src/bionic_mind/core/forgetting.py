from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger

from bionic_mind.llm.base import LLMProvider


class AbstractionForgetter:
    def __init__(
        self,
        llm: LLMProvider,
        detail_threshold: int = 500,
        abstraction_prompt_max: int = 1000,
    ):
        self.llm = llm
        self.detail_threshold = detail_threshold
        self.abstraction_prompt_max = abstraction_prompt_max

    async def should_abstract(self, content: str, abstraction_level: float) -> bool:
        return len(content) > self.detail_threshold and abstraction_level < 0.5

    async def abstract(self, content: str, current_abstraction: float) -> tuple[str, float]:
        if len(content) <= self.detail_threshold:
            return content, current_abstraction

        prompt = f"""请将以下详细记忆压缩为简洁的摘要，保留核心信息和情绪色彩，丢弃具体细节。

原始记忆:
{content[:self.abstraction_prompt_max]}

请生成一个简洁的摘要(不超过100字):"""

        try:
            response = await self.llm.chat(
                messages=[
                    {"role": "system", "content": "你是一个记忆压缩器。保留核心信息，丢弃细节。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=200,
            )
            new_abstraction = min(1.0, current_abstraction + 0.3)
            logger.info(f"Memory abstracted: {len(content)} chars -> {len(response.content)} chars, abstraction={new_abstraction:.1f}")
            return response.content, new_abstraction
        except Exception as e:
            logger.error(f"Abstraction failed: {e}")
            return content, current_abstraction


class CompetitiveForgetter:
    def __init__(
        self,
        resource_limit: int = 1000,
        suppression_threshold: float = 0.1,
    ):
        self.resource_limit = resource_limit
        self.suppression_threshold = suppression_threshold

    def select_for_suppression(
        self,
        memory_stats: list[dict[str, Any]],
        current_load: int,
    ) -> list[str]:
        if current_load <= self.resource_limit:
            return []

        num_to_suppress = int((current_load - self.resource_limit) * 0.1) + 1
        candidates = []

        for stat in memory_stats:
            score = (
                stat.get("call_frequency", 0.5) * 0.3
                + stat.get("time_decay", 0.5) * 0.2
                + stat.get("emotional_arousal", 0.3) * 0.4
                + stat.get("connection_density", 0.3) * 0.1
            )
            if score < self.suppression_threshold + 0.3:
                candidates.append((stat.get("id", ""), score))

        candidates.sort(key=lambda x: x[1])
        suppressed = [c[0] for c in candidates[:num_to_suppress]]
        logger.info(f"Competitive forgetting: {len(suppressed)} memories suppressed")
        return suppressed

    def promote_creativity(
        self,
        dominant_memory_ids: list[str],
        all_memory_ids: list[str],
        suppression_ratio: float = 0.2,
    ) -> list[str]:
        if not dominant_memory_ids or len(all_memory_ids) <= 5:
            return []

        num_to_suppress = max(1, int(len(dominant_memory_ids) * suppression_ratio))
        suppressed = dominant_memory_ids[:num_to_suppress]

        non_dominant = [m for m in all_memory_ids if m not in set(dominant_memory_ids)]
        promoted = non_dominant[:num_to_suppress]

        logger.info(f"Creativity promotion: suppressed {len(suppressed)} dominant, promoted {len(promoted)} non-dominant")
        return suppressed
