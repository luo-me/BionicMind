from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import yaml
from loguru import logger

from bionic_mind.core.memory import MemoryNode, MemoryVectorField, WorkingMemory
from bionic_mind.core.emotion import EmotionSystem, EmotionState
from bionic_mind.core.drives import DriveSystem, DriveState
from bionic_mind.core.context import ContextAssembler, AssembledContext
from bionic_mind.core.perception import PerceptionEncoder, PerceptionResult
from bionic_mind.core.hebbian import HebbianNetwork, EdgeType
from bionic_mind.core.world_model import WorldModel
from bionic_mind.core.adaptive_emotion import AdaptiveEmotionLearner
from bionic_mind.core.counterfactual import CounterfactualSimulator
from bionic_mind.core.forgetting import AbstractionForgetter, CompetitiveForgetter
from bionic_mind.core.meta_action import MetaActionSystem
from bionic_mind.llm.base import LLMProvider
from bionic_mind.llm.openai_provider import OpenAIProvider
from bionic_mind.llm.ollama_provider import OllamaProvider


@dataclass
class CycleResult:
    cycle_id: str = ""
    perception: str = ""
    perception_source: str = ""
    output: str = ""
    emotion: EmotionState = field(default_factory=EmotionState)
    drives: DriveState = field(default_factory=DriveState)
    modulation: dict[str, Any] = field(default_factory=dict)
    memory_written: str = ""
    tokens_used: int = 0
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "perception": self.perception[:100],
            "perception_source": self.perception_source,
            "output": self.output[:200],
            "emotion": self.emotion.to_dict(),
            "drives": self.drives.to_dict(),
            "modulation": self.modulation,
            "tokens_used": self.tokens_used,
            "timestamp": self.timestamp,
        }


class BionicMind:
    DECAY_INTERVAL = 20
    FORGETTING_INTERVAL = 50

    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.llm: LLMProvider = self._create_llm_provider()
        self.memory = MemoryVectorField(
            persist_dir=self.config.get("memory", {}).get("persist_dir", "./memory_db"),
            decay_lambda=self.config.get("memory", {}).get("decay_lambda", 0.01),
            priority_weights=self.config.get("memory", {}).get("priority_weights"),
        )
        self.emotion = EmotionSystem(
            valence_weights=self.config.get("emotion", {}).get("weights"),
            arousal_weights=self.config.get("emotion", {}).get("arousal_weights"),
        )
        self.drives = DriveSystem(
            curiosity_baseline=self.config.get("drives", {}).get("curiosity_baseline", 0.3),
            social_feedback_timeout_hours=self.config.get("drives", {}).get("social_feedback_timeout_hours", 24),
            spontaneous_action_threshold=self.config.get("drives", {}).get("spontaneous_action_threshold", 0.7),
        )
        self.perception_encoder = PerceptionEncoder()
        self.working_memory = WorkingMemory(
            max_turns=self.config.get("memory", {}).get("working_memory_turns", 30),
        )
        self.context_assembler = ContextAssembler(
            memory=self.memory,
            emotion=self.emotion,
            drives=self.drives,
            working_memory=self.working_memory,
        )
        self.running = False
        self._cycle_count = 0
        self._recent_perceptions: list[str] = []
        self._total_tokens = 0
        self._last_prediction: str = ""
        self._prediction_errors: list[float] = []

        self.hebbian = HebbianNetwork(
            persist_path=self.config.get("memory", {}).get("persist_dir", "./memory_db") + "/hebbian_graph.json"
        )
        self.world_model = WorldModel()
        self.emotion_learner = AdaptiveEmotionLearner()
        self.counterfactual = CounterfactualSimulator(self.llm)
        self.abstraction_forgetter = AbstractionForgetter(self.llm)
        self.competitive_forgetter = CompetitiveForgetter()
        self.meta_action_system = MetaActionSystem()
        self._activated_memory_ids: list[str] = []

        logger.info(f"BionicMind initialized with {self.llm.get_model_name()}")

    @staticmethod
    def _load_config(path: str) -> dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning(f"Config file not found: {path}, using defaults")
            return {}
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}

    def _create_llm_provider(self) -> LLMProvider:
        llm_config = self.config.get("llm", {})
        provider = llm_config.get("provider", "openai")

        if provider == "ollama":
            return OllamaProvider(
                model=llm_config.get("model", "qwen2.5:7b"),
                base_url=llm_config.get("base_url", "http://localhost:11434"),
            )
        else:
            return OpenAIProvider(
                model=llm_config.get("model", "gpt-4o-mini"),
                api_key=llm_config.get("api_key"),
                base_url=llm_config.get("base_url"),
            )

    async def perceive(self, user_input: str | None = None) -> PerceptionResult | None:
        if user_input:
            return self.perception_encoder.encode_user_input(user_input)

        if self.drives.should_act_spontaneously():
            drive_prompt = self.drives.get_spontaneous_prompt()
            if drive_prompt:
                return self.perception_encoder.encode_drive_signal(drive_prompt)

        return None

    async def think(self, perception: PerceptionResult) -> tuple[str, AssembledContext]:
        if perception.is_spontaneous():
            context = self.context_assembler.assemble_spontaneous(perception.text)
        else:
            context = self.context_assembler.assemble(perception.text)

        base_temp = self.config.get("llm", {}).get("temperature", 0.7)
        temp_offset = context.modulation.temperature_offset
        temperature = max(0.1, min(1.5, base_temp + temp_offset))

        max_tokens = self.config.get("llm", {}).get("max_tokens", 2048)

        response = await self.llm.chat(
            messages=context.messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        self._total_tokens += response.total_tokens

        logger.info(
            f"Think cycle: tokens={response.total_tokens}, "
            f"temp={temperature:.2f}, mode={context.modulation.mode.value}"
        )

        return response.content, context

    async def act(self, output: str) -> dict[str, str]:
        return {"type": "text", "content": output}

    async def learn(
        self,
        perception: PerceptionResult,
        output: str,
        context: AssembledContext,
        feedback: dict[str, float] | None = None,
    ) -> None:
        feedback = feedback or {}

        novelty = self.perception_encoder.compute_novelty(
            perception.text, self._recent_perceptions
        )
        self._recent_perceptions.append(perception.text)
        if len(self._recent_perceptions) > 50:
            self._recent_perceptions.pop(0)

        prediction_error = self._estimate_prediction_error(perception.text, output)

        goal_progress = feedback.get("goal_progress", 0.3)
        consistency = self._estimate_consistency(output)
        social_feedback = feedback.get("social_feedback", 0.0)
        core_threat = feedback.get("core_threat", 0.0)

        self.emotion.evaluate(
            prediction_error=prediction_error,
            goal_progress=goal_progress,
            consistency=consistency,
            social_feedback=social_feedback,
            novelty=novelty,
        )

        self.drives.update(
            prediction_error=prediction_error,
            consistency=consistency,
            social_feedback=social_feedback,
            resource_usage=min(self._total_tokens / 100000, 1.0),
            core_threat_level=core_threat,
        )

        if prediction_error > 0.5:
            self.drives.decay_curiosity(0.02)

        self.working_memory.add(
            role="user",
            content=perception.text,
            valence=0.0,
            arousal=0.3,
        )
        self.working_memory.add(
            role="assistant",
            content=output,
            valence=self.emotion.state.valence,
            arousal=self.emotion.state.arousal,
        )

        memory_content = perception.text
        memory = MemoryNode(
            id=f"mem_{uuid.uuid4().hex[:12]}",
            content=memory_content,
            perception=perception.text,
            response=output,
            emotional_valence=self.emotion.state.valence,
            emotional_arousal=self.emotion.state.arousal,
            memory_type="episodic",
        )
        self.memory.write(memory)

        activated_ids = [memory.id]
        for r in self.memory.retrieve(perception.text, top_k=5, include_recent=3):
            self.memory.update_access(r.memory.id)
            activated_ids.append(r.memory.id)
        self.hebbian.co_activate(activated_ids, EdgeType.SEMANTIC)

        spreading = self.hebbian.get_spreading_activation(
            seed_ids=activated_ids[:3],
            activation_energy=0.8,
            decay_factor=0.5,
            max_steps=2,
        )
        for related_id, energy in list(spreading.items())[:3]:
            self.memory.update_access(related_id)

        context_hash = f"ctx_{self._cycle_count}"
        self.world_model.predict(perception.text, context_hash)
        self.world_model.update(context_hash, output)

        if social_feedback != 0:
            self.emotion_learner.update_weights(
                inputs={
                    "prediction_error": prediction_error,
                    "goal_progress": goal_progress,
                    "consistency": consistency,
                    "social_feedback": social_feedback,
                    "novelty": novelty,
                },
                valence_target=social_feedback,
                arousal_target=abs(social_feedback) * 0.8,
            )

        meta_actions = self.meta_action_system.evaluate_and_propose(
            emotion_state=self.emotion.state.to_dict(),
            drive_state=self.drives.drives.to_dict(),
            performance_metrics={
                "avg_prediction_error": self.world_model.get_avg_error(),
                "creativity_need": 1.0 - self.emotion.state.arousal if self.emotion.state.arousal < 0.3 else 0.0,
            },
        )
        if meta_actions:
            self._apply_meta_actions(meta_actions)

        self._last_prediction = output[:200]
        self._prediction_errors.append(prediction_error)
        if len(self._prediction_errors) > 100:
            self._prediction_errors.pop(0)

        if self._cycle_count % self.DECAY_INTERVAL == 0:
            self.memory.decay_all()

        if self._cycle_count % self.FORGETTING_INTERVAL == 0:
            await self._run_forgetting()

        if self._cycle_count % 10 == 0:
            self.hebbian.save()

        logger.debug(
            f"Learn: emotion={self.emotion.state.describe()}, "
            f"novelty={novelty:.2f}, pred_error={prediction_error:.2f}, "
            f"hebbian_edges={self.hebbian.graph.number_of_edges()}, "
            f"working_mem={len(self.working_memory)}"
        )

    def _apply_meta_actions(self, actions: list) -> None:
        for action in actions:
            if action.action_type == "adjust_learning_rate":
                self.emotion.smoothing_factor = max(
                    0.05, min(1.0, 1.0 - action.value)
                )
            elif action.action_type == "adjust_emotion_sensitivity":
                delta = action.value - 1.0
                self.emotion.valence_weights = {
                    k: max(0.05, min(2.0, v + delta * 0.1))
                    for k, v in self.emotion.valence_weights.items()
                }
                self.emotion.arousal_weights = {
                    k: max(0.05, min(2.0, v + delta * 0.1))
                    for k, v in self.emotion.arousal_weights.items()
                }
            elif action.action_type == "adjust_drive_weight":
                drive_name = action.target
                if drive_name in self.drives.drives.to_dict():
                    self.drives.smoothing_factor = max(
                        0.05, min(0.8, self.drives.smoothing_factor * action.value)
                    )
            elif action.action_type == "adjust_noise":
                pass

        self.meta_action_system.apply(actions)

    async def _run_forgetting(self) -> None:
        stats = self.memory.get_stats()
        total = stats.get("total", 0)

        if total > 100:
            all_data = self.memory.collection.get(include=["metadatas"])
            memory_stats = []
            for i, mid in enumerate(all_data["ids"]):
                meta = all_data["metadatas"][i]
                memory_stats.append({
                    "id": mid,
                    "call_frequency": meta.get("call_frequency", 0.0),
                    "time_decay": meta.get("time_decay", 1.0),
                    "emotional_arousal": meta.get("emotional_arousal", 0.0),
                    "connection_density": meta.get("connection_density", 0.0),
                })

            suppressed = self.competitive_forgetter.select_for_suppression(
                memory_stats=memory_stats,
                current_load=total,
            )
            for sid in suppressed:
                self.memory.delete(sid)
                logger.info(f"Competitive forgetting: suppressed {sid}")

            if self.emotion.state.mode.value in ("exploratory", "depressed"):
                high_freq_ids = sorted(
                    memory_stats,
                    key=lambda x: x.get("call_frequency", 0.0),
                    reverse=True,
                )[:5]
                dominant_ids = [s["id"] for s in high_freq_ids]
                all_ids = [s["id"] for s in memory_stats]
                creativity_suppressed = self.competitive_forgetter.promote_creativity(
                    dominant_memory_ids=dominant_ids,
                    all_memory_ids=all_ids,
                    suppression_ratio=0.2,
                )
                for sid in creativity_suppressed:
                    self.memory.update_access(sid)
                    logger.info(f"Creativity promotion: boosted {sid}")

        all_data = self.memory.collection.get(
            include=["documents", "metadatas"],
        )
        if all_data["ids"]:
            candidates = []
            for i, mid in enumerate(all_data["ids"]):
                meta = all_data["metadatas"][i]
                candidates.append((mid, all_data["documents"][i], meta))

            candidates.sort(key=lambda x: x[2].get("time_decay", 1.0))

            for mid, content, meta in candidates[:5]:
                abstraction = meta.get("abstraction", 0.0)
                if await self.abstraction_forgetter.should_abstract(content, abstraction):
                    new_content, new_abstraction = await self.abstraction_forgetter.abstract(
                        content, abstraction
                    )
                    self.memory.update_abstraction(mid, new_abstraction)
                    self.memory.collection.update(
                        ids=[mid],
                        documents=[new_content],
                        metadatas=[{**meta, "abstraction": new_abstraction}],
                    )
                    logger.info(f"Abstraction forgetting: abstracted {mid}")

    def _estimate_prediction_error(self, perception: str, output: str) -> float:
        if not self._last_prediction:
            return 0.5

        if self._prediction_errors:
            recent_avg = sum(self._prediction_errors[-5:]) / len(self._prediction_errors[-5:])
        else:
            recent_avg = 0.5

        last_words = set(self._last_prediction.split())
        curr_words = set(output.split())
        if last_words and curr_words:
            word_overlap = len(last_words & curr_words) / max(len(last_words | curr_words), 1)
        else:
            word_overlap = 0.0

        char_overlap = len(set(self._last_prediction) & set(output))
        char_total = max(len(set(self._last_prediction) | set(output)), 1)
        char_sim = char_overlap / char_total

        similarity = 0.6 * word_overlap + 0.4 * char_sim
        current_error = 1.0 - similarity

        smoothed = 0.7 * current_error + 0.3 * recent_avg
        return float(np.clip(smoothed, 0.0, 1.0))

    def _estimate_consistency(self, output: str) -> float:
        anchors = self.memory.get_emotional_anchors(min_arousal=0.6)
        if not anchors:
            return 0.5

        output_words = set(output.lower().split())
        max_sim = 0.0
        for anchor in anchors[:3]:
            anchor_words = set(anchor.content[:200].lower().split())
            if output_words and anchor_words:
                overlap = len(output_words & anchor_words)
                total = len(output_words | anchor_words)
                sim = overlap / total
                max_sim = max(max_sim, sim)

        return 0.3 * max_sim + 0.7 * 0.5

    async def run_cycle(
        self,
        user_input: str | None = None,
        feedback: dict[str, float] | None = None,
    ) -> CycleResult | None:
        perception = await self.perceive(user_input)
        if perception is None:
            return None

        self._cycle_count += 1
        output, context = await self.think(perception)
        action = await self.act(output)
        await self.learn(perception, output, context, feedback)

        return CycleResult(
            cycle_id=f"cycle_{self._cycle_count}",
            perception=perception.text,
            perception_source=perception.source,
            output=output,
            emotion=EmotionState(
                valence=self.emotion.state.valence,
                arousal=self.emotion.state.arousal,
            ),
            drives=DriveState(
                curiosity=self.drives.drives.curiosity,
                consistency=self.drives.drives.consistency,
                social_connection=self.drives.drives.social_connection,
                energy_efficiency=self.drives.drives.energy_efficiency,
                self_preservation=self.drives.drives.self_preservation,
            ),
            modulation=context.modulation.to_dict(),
            memory_written="",
            tokens_used=self._total_tokens,
            timestamp=datetime.now().isoformat(),
        )

    async def run_interactive(self) -> None:
        self.running = True
        logger.info("BionicMind v1.0 启动。输入 'quit' 退出，'status' 查看内部状态，'feedback <positive|negative>' 提供反馈。")

        while self.running:
            try:
                loop = asyncio.get_event_loop()
                user_input = await loop.run_in_executor(None, lambda: input("\n你: "))
            except (EOFError, KeyboardInterrupt):
                break

            user_input = user_input.strip()
            if not user_input:
                continue

            if user_input.lower() == "quit":
                self.running = False
                break

            if user_input.lower() == "status":
                self._print_status()
                continue

            if user_input.lower() == "memorystats":
                stats = self.memory.get_stats()
                print(f"\n=== 记忆统计 ===")
                for k, v in stats.items():
                    print(f"  {k}: {v}")
                continue

            if user_input.lower().startswith("feedback"):
                parts = user_input.split()
                if len(parts) >= 2:
                    fb_type = parts[1].lower()
                    fb_value = 0.5 if fb_type == "positive" else -0.5
                    self.drives.update(social_feedback=fb_value)
                    print(f"  [反馈已记录: {fb_type}]")
                continue

            result = await self.run_cycle(user_input)
            if result:
                print(f"\nAI: {result.output}")
                mode = result.emotion.mode.value
                print(f"  [{mode} | 效价{result.emotion.valence:+.2f} | 唤醒{result.emotion.arousal:.2f}]")

        logger.info("BionicMind v1.0 关闭。")

    def _print_status(self) -> None:
        e = self.emotion.state
        d = self.drives.drives
        dominant_name, dominant_value = d.dominant()
        modulation = self.emotion.get_modulation()
        stats = self.memory.get_stats()

        print(f"\n{'='*50}")
        print(f"  BionicMind 内部状态")
        print(f"{'='*50}")
        print(f"  情绪: {e.describe()}")
        print(f"  注意力模式: {modulation.mode.value}")
        print(f"  搜索广度: {modulation.search_breadth}")
        print(f"  创造力增强: {modulation.creativity_boost:.1f}")
        print(f"  ---")
        print(f"  好奇心: {d.curiosity:.2f}")
        print(f"  一致性: {d.consistency:.2f}")
        print(f"  社交连接: {d.social_connection:.2f}")
        print(f"  能量效率: {d.energy_efficiency:.2f}")
        print(f"  自我保存: {d.self_preservation:.2f}")
        print(f"  主导驱力: {dominant_name} ({dominant_value:.2f})")
        print(f"  ---")
        print(f"  记忆总数: {stats.get('total', 0)}")
        print(f"  工作记忆: {len(self.working_memory)} 条")
        print(f"  高唤醒记忆: {stats.get('high_arousal_count', 0)}")
        print(f"  Hebbian关联: {self.hebbian.graph.number_of_edges()}")
        print(f"  世界模型准确率: {self.world_model.get_accuracy():.1%}")
        print(f"  总token消耗: {self._total_tokens}")
        print(f"  思考周期: {self._cycle_count}")
        print(f"{'='*50}")

    def get_full_state(self) -> dict[str, Any]:
        return {
            "emotion": self.emotion.to_dict(),
            "drives": self.drives.to_dict(),
            "memory_stats": self.memory.get_stats(),
            "working_memory_size": len(self.working_memory),
            "hebbian_stats": self.hebbian.get_stats(),
            "world_model_stats": self.world_model.get_stats(),
            "meta_action_state": self.meta_action_system.get_state(),
            "cycle_count": self._cycle_count,
            "total_tokens": self._total_tokens,
        }

    async def shutdown(self) -> None:
        self.running = False
        self.memory.decay_all()
        self.hebbian.save()
        logger.info("BionicMind shutdown complete")
