from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from bionic_mind.llm.base import LLMProvider, LLMResponse


@dataclass
class SimulationResult:
    action: str
    predicted_outcome: str
    confidence: float
    pros: list[str] = field(default_factory=list)
    cons: list[str] = field(default_factory=list)
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "predicted_outcome": self.predicted_outcome,
            "confidence": self.confidence,
            "pros": self.pros,
            "cons": self.cons,
            "score": self.score,
        }


class CounterfactualSimulator:
    def __init__(
        self,
        llm: LLMProvider,
        max_candidates: int = 3,
        simulation_temperature: float = 0.5,
    ):
        self.llm = llm
        self.max_candidates = max_candidates
        self.simulation_temperature = simulation_temperature

    async def generate_candidates(self, perception: str, context: str) -> list[str]:
        prompt = f"""基于以下情境，生成{self.max_candidates}种不同的行动方案。每种方案用一行表示，格式为"方案N: 描述"。

情境: {perception}

上下文:
{context[:500]}

请生成{self.max_candidates}种不同的行动方案:"""

        try:
            response = await self.llm.chat(
                messages=[
                    {"role": "system", "content": "你是一个行动方案生成器。只生成方案，不做评价。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.8,
                max_tokens=500,
            )
            candidates = []
            for line in response.content.split("\n"):
                line = line.strip()
                if line and ("方案" in line or ":" in line or "-" in line):
                    clean = line.lstrip("0123456789.-方案: ")
                    if clean:
                        candidates.append(clean)
            return candidates[:self.max_candidates]
        except Exception as e:
            logger.error(f"Failed to generate candidates: {e}")
            return []

    async def simulate_action(self, perception: str, action: str, context: str) -> SimulationResult:
        prompt = f"""请评估以下行动方案的预期结果。

情境: {perception}
行动方案: {action}

上下文:
{context[:500]}

请按以下格式回答:
预期结果: (简述预期会发生什么)
信心: (0-1的数字)
优点: (用逗号分隔)
缺点: (用逗号分隔)"""

        try:
            response = await self.llm.chat(
                messages=[
                    {"role": "system", "content": "你是一个行动方案评估器。客观评估每个方案的利弊。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.simulation_temperature,
                max_tokens=300,
            )
            return self._parse_simulation(action, response.content)
        except Exception as e:
            logger.error(f"Simulation failed for action '{action}': {e}")
            return SimulationResult(action=action, predicted_outcome="模拟失败", confidence=0.0)

    async def simulate_all(
        self,
        perception: str,
        context: str,
        candidates: list[str] | None = None,
    ) -> list[SimulationResult]:
        if candidates is None:
            candidates = await self.generate_candidates(perception, context)
        if not candidates:
            return []

        results = []
        for action in candidates:
            result = await self.simulate_action(perception, action, context)
            results.append(result)

        for r in results:
            r.score = r.confidence * 0.6 + len(r.pros) * 0.05 - len(r.cons) * 0.03

        results.sort(key=lambda x: x.score, reverse=True)
        logger.info(f"Simulated {len(results)} actions, best: {results[0].action[:30]} (score={results[0].score:.2f})")
        return results

    def _parse_simulation(self, action: str, response: str) -> SimulationResult:
        outcome = ""
        confidence = 0.5
        pros: list[str] = []
        cons: list[str] = []

        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("预期结果"):
                outcome = line.split(":", 1)[-1].strip() if ":" in line else line
            elif line.startswith("信心"):
                try:
                    conf_str = line.split(":", 1)[-1].strip() if ":" in line else "0.5"
                    confidence = float("".join(c for c in conf_str if c.isdigit() or c == ".") or "0.5")
                    confidence = max(0.0, min(1.0, confidence))
                except ValueError:
                    confidence = 0.5
            elif line.startswith("优点"):
                content = line.split(":", 1)[-1].strip() if ":" in line else ""
                pros = [p.strip() for p in content.split(",") if p.strip()]
            elif line.startswith("缺点"):
                content = line.split(":", 1)[-1].strip() if ":" in line else ""
                cons = [c.strip() for c in content.split(",") if c.strip()]

        return SimulationResult(
            action=action,
            predicted_outcome=outcome,
            confidence=confidence,
            pros=pros,
            cons=cons,
        )
