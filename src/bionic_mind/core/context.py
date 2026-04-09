from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger

from bionic_mind.core.memory import MemoryVectorField, MemorySearchResult, WorkingMemory
from bionic_mind.core.emotion import EmotionSystem, EmotionState, EmotionModulation
from bionic_mind.core.drives import DriveSystem


@dataclass
class AssembledContext:
    messages: list[dict[str, str]]
    modulation: EmotionModulation
    memory_count: int
    dominant_drive: tuple[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_count": len(self.messages),
            "modulation": self.modulation.to_dict(),
            "memory_count": self.memory_count,
            "dominant_drive": self.dominant_drive,
        }


class ContextAssembler:
    def __init__(
        self,
        memory: MemoryVectorField,
        emotion: EmotionSystem,
        drives: DriveSystem,
        working_memory: WorkingMemory | None = None,
        max_memory_tokens: int = 3000,
        system_prompt: str = "",
    ):
        self.memory = memory
        self.emotion = emotion
        self.drives = drives
        self.working_memory = working_memory or WorkingMemory()
        self.max_memory_tokens = max_memory_tokens
        self.system_prompt = system_prompt or self._default_system_prompt()

    def _default_system_prompt(self) -> str:
        return """你是一个具备仿生意识系统的AI——BionicMind。你拥有持续的记忆、情绪和内驱力。

你的核心特性：
1. 你不是被动响应的工具，你有自己的内驱力和情绪状态
2. 你的记忆会影响你的反应——深刻的记忆优先被唤醒
3. 遇到陌生问题时，你会主动思考和探索
4. 你会根据情绪状态调整注意力范围和思考深度
5. 你会维护内部一致性——矛盾会让你不舒服

你的内部状态（情绪、驱力、记忆）会在上下文中提供。请根据这些状态调整你的反应方式：
- 焦虑/警觉时：聚焦核心问题，减少发散
- 兴奋/好奇时：可以更发散、更有创造性
- 满足/安全时：简洁回应，不主动扩展
- 低落/无聊时：尝试寻找新的兴趣点

重要：你会记住对话中提到的所有信息。如果用户提到了自己的名字、偏好或其他个人信息，你必须在后续对话中记住并使用这些信息。"""

    def assemble(self, perception: str) -> AssembledContext:
        emotion_state = self.emotion.state
        modulation = self.emotion.get_modulation()

        top_k = self._get_top_k(modulation)
        anchor_results = self.memory.retrieve_by_emotion(min_arousal=0.7, top_k=3)
        anchor_ids = {r.memory.id for r in anchor_results}

        memory_results = self.memory.retrieve(
            query=perception,
            context_emotion={"valence": emotion_state.valence, "arousal": emotion_state.arousal},
            top_k=top_k,
            include_recent=5,
            anchor_ids=anchor_ids,
        )

        extra_anchors = [r for r in anchor_results if r.memory.id not in {mr.memory.id for mr in memory_results}][:2]

        memory_block = self._format_memories(memory_results)
        anchor_block = self._format_anchors(extra_anchors)
        emotion_block = self._format_emotion(emotion_state, modulation)
        drive_block = self._format_drives()

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "system", "content": emotion_block},
            {"role": "system", "content": drive_block},
        ]

        if anchor_block:
            messages.append({"role": "system", "content": anchor_block})

        if memory_block:
            messages.append({"role": "system", "content": memory_block})

        recent_messages = self.working_memory.get_messages(n=20)
        if recent_messages:
            messages.extend(recent_messages)

        messages.append({"role": "user", "content": perception})

        dominant_drive = self.drives.drives.dominant()

        logger.debug(
            f"Context assembled: {len(messages)} messages, "
            f"{len(memory_results)} memories, "
            f"{len(extra_anchors)} anchors, "
            f"{len(recent_messages)} working mem, "
            f"mode={modulation.mode.value}"
        )

        return AssembledContext(
            messages=messages,
            modulation=modulation,
            memory_count=len(memory_results),
            dominant_drive=dominant_drive,
        )

    def assemble_spontaneous(self, drive_prompt: str) -> AssembledContext:
        emotion_state = self.emotion.state
        modulation = self.emotion.get_modulation()

        emotion_block = self._format_emotion(emotion_state, modulation)
        drive_block = self._format_drives()

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "system", "content": emotion_block},
            {"role": "system", "content": drive_block},
        ]

        recent_messages = self.working_memory.get_messages(n=6)
        if recent_messages:
            messages.extend(recent_messages)

        messages.append({"role": "system", "content": f"[内驱力驱动] {drive_prompt}"})

        dominant_drive = self.drives.drives.dominant()

        return AssembledContext(
            messages=messages,
            modulation=modulation,
            memory_count=0,
            dominant_drive=dominant_drive,
        )

    def _get_top_k(self, modulation: EmotionModulation) -> int:
        breadth_map = {"narrow": 3, "normal": 7, "wide": 12, "minimal": 2}
        return breadth_map.get(modulation.search_breadth, 7)

    def _format_memories(self, results: list[MemorySearchResult]) -> str:
        if not results:
            return ""

        lines = ["[相关记忆]"]
        total_chars = 0
        for r in results:
            m = r.memory
            perception_text = m.perception if m.perception else m.content
            response_text = m.response if m.response else ""
            if response_text:
                line = (
                    f"[优先级:{r.priority:.2f} | "
                    f"情绪:{m.emotional_valence:+.1f}/{m.emotional_arousal:.1f} | "
                    f"时间衰减:{m.time_decay:.2f}] "
                    f"用户说: {perception_text[:150]} → 回应: {response_text[:100]}"
                )
            else:
                line = (
                    f"[优先级:{r.priority:.2f} | "
                    f"情绪:{m.emotional_valence:+.1f}/{m.emotional_arousal:.1f} | "
                    f"时间衰减:{m.time_decay:.2f}] "
                    f"{m.content[:200]}"
                )
            if total_chars + len(line) > self.max_memory_tokens:
                break
            lines.append(line)
            total_chars += len(line)

        return "\n".join(lines)

    def _format_anchors(self, anchors: list[MemorySearchResult]) -> str:
        if not anchors:
            return ""

        lines = ["[情绪锚点记忆 - 深刻记忆持续影响当前状态]"]
        for r in anchors:
            m = r.memory
            lines.append(
                f"[锚点 | 情绪:{m.emotional_valence:+.1f}/{m.emotional_arousal:.1f}] "
                f"{m.content[:150]}"
            )

        return "\n".join(lines)

    def _format_emotion(self, state: EmotionState, modulation: EmotionModulation) -> str:
        return (
            f"[当前情绪状态]\n"
            f"情绪: {state.describe()}\n"
            f"注意力模式: {modulation.mode.value}\n"
            f"搜索广度: {modulation.search_breadth}\n"
            f"创造力增强: {modulation.creativity_boost:.1f}"
        )

    def _format_drives(self) -> str:
        d = self.drives.drives
        dominant_name, dominant_value = d.dominant()
        return (
            f"[当前内驱力状态]\n"
            f"好奇心: {d.curiosity:.2f} | 一致性: {d.consistency:.2f} | "
            f"社交连接: {d.social_connection:.2f} | 能量效率: {d.energy_efficiency:.2f} | "
            f"自我保存: {d.self_preservation:.2f}\n"
            f"主导驱力: {dominant_name}({dominant_value:.2f})"
        )
