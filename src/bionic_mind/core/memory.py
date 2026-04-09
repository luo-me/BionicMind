from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import chromadb
import numpy as np
from loguru import logger


@dataclass
class MemoryNode:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    content: str = ""
    perception: str = ""
    response: str = ""
    time_decay: float = 1.0
    abstraction: float = 0.0
    emotional_valence: float = 0.0
    emotional_arousal: float = 0.0
    call_frequency: float = 0.0
    connection_density: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    access_count: int = 0
    memory_type: str = "episodic"

    def to_metadata(self) -> dict[str, Any]:
        return {
            "time_decay": self.time_decay,
            "abstraction": self.abstraction,
            "emotional_valence": self.emotional_valence,
            "emotional_arousal": self.emotional_arousal,
            "call_frequency": self.call_frequency,
            "connection_density": self.connection_density,
            "created_at": self.created_at,
            "access_count": self.access_count,
            "memory_type": self.memory_type,
            "perception": self.perception,
            "response": self.response,
        }

    @classmethod
    def from_metadata(cls, mid: str, content: str, meta: dict[str, Any]) -> MemoryNode:
        return cls(
            id=mid,
            content=content,
            perception=meta.get("perception", ""),
            response=meta.get("response", ""),
            time_decay=meta.get("time_decay", 1.0),
            abstraction=meta.get("abstraction", 0.0),
            emotional_valence=meta.get("emotional_valence", 0.0),
            emotional_arousal=meta.get("emotional_arousal", 0.0),
            call_frequency=meta.get("call_frequency", 0.0),
            connection_density=meta.get("connection_density", 0.0),
            created_at=meta.get("created_at", datetime.now().isoformat()),
            access_count=meta.get("access_count", 0),
            memory_type=meta.get("memory_type", "episodic"),
        )


@dataclass
class MemorySearchResult:
    memory: MemoryNode
    priority: float
    semantic_similarity: float

    def __repr__(self) -> str:
        return (
            f"MemorySearchResult(id={self.memory.id}, "
            f"priority={self.priority:.3f}, "
            f"sim={self.semantic_similarity:.3f}, "
            f"arousal={self.memory.emotional_arousal:.2f}, "
            f"decay={self.memory.time_decay:.2f})"
        )


@dataclass
class WorkingMemoryEntry:
    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    emotional_valence: float = 0.0
    emotional_arousal: float = 0.3

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "emotional_valence": self.emotional_valence,
            "emotional_arousal": self.emotional_arousal,
        }


class WorkingMemory:
    def __init__(self, max_turns: int = 20):
        self.max_turns = max_turns
        self._entries: deque[WorkingMemoryEntry] = deque(maxlen=max_turns * 2)

    def add(self, role: str, content: str, valence: float = 0.0, arousal: float = 0.3) -> None:
        self._entries.append(WorkingMemoryEntry(
            role=role,
            content=content,
            emotional_valence=valence,
            emotional_arousal=arousal,
        ))

    def get_recent(self, n: int | None = None) -> list[WorkingMemoryEntry]:
        if n is None:
            return list(self._entries)
        return list(self._entries)[-n:]

    def get_messages(self, n: int | None = None) -> list[dict[str, str]]:
        entries = self.get_recent(n)
        return [{"role": e.role, "content": e.content} for e in entries]

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)

    def is_empty(self) -> bool:
        return len(self._entries) == 0


class MemoryVectorField:
    def __init__(
        self,
        persist_dir: str = "./memory_db",
        decay_lambda: float = 0.01,
        priority_weights: dict[str, float] | None = None,
    ):
        self.decay_lambda = decay_lambda
        self.priority_weights = priority_weights or {
            "semantic_similarity": 0.20,
            "time_decay": 0.25,
            "emotional_arousal": 0.15,
            "call_frequency": 0.10,
            "connection_density": 0.05,
            "emotional_resonance": 0.10,
            "abstraction": 0.05,
            "recency_boost": 0.10,
        }
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="bionic_memories",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"MemoryVectorField initialized, {self.collection.count()} memories loaded")

    def write(self, memory: MemoryNode) -> str:
        try:
            self.collection.upsert(
                ids=[memory.id],
                documents=[memory.content],
                metadatas=[memory.to_metadata()],
            )
            logger.debug(f"Memory written: {memory.id} (arousal={memory.emotional_arousal:.2f})")
            return memory.id
        except Exception as e:
            logger.error(f"Failed to write memory {memory.id}: {e}")
            raise

    def write_batch(self, memories: list[MemoryNode]) -> list[str]:
        ids = []
        docs = []
        metas = []
        for m in memories:
            ids.append(m.id)
            docs.append(m.content)
            metas.append(m.to_metadata())
        try:
            self.collection.upsert(ids=ids, documents=docs, metadatas=metas)
            logger.info(f"Batch written: {len(memories)} memories")
            return ids
        except Exception as e:
            logger.error(f"Failed to batch write: {e}")
            raise

    def retrieve(
        self,
        query: str,
        context_emotion: dict[str, float] | None = None,
        top_k: int = 10,
        min_priority: float = 0.0,
        include_recent: int = 5,
        anchor_ids: set[str] | None = None,
    ) -> list[MemorySearchResult]:
        context_emotion = context_emotion or {"valence": 0.0, "arousal": 0.3}
        total = self.collection.count()
        if total == 0:
            return []

        n_query = min(top_k * 5, total)
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=max(n_query, 1),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error(f"Memory retrieval failed: {e}")
            return []

        if not results["documents"] or not results["documents"][0]:
            return []

        seen_ids: set[str] = set()
        scored: list[MemorySearchResult] = []
        w = self.priority_weights
        now = datetime.now()
        anchor_ids = anchor_ids or set()

        for i, doc in enumerate(results["documents"][0]):
            mid = results["ids"][0][i]
            meta = results["metadatas"][0][i]
            semantic_sim = 1.0 - results["distances"][0][i]

            emotional_resonance = (
                abs(meta.get("emotional_valence", 0.0) - context_emotion.get("valence", 0.0))
                * meta.get("emotional_arousal", 0.0)
            )

            created_at_str = meta.get("created_at", now.isoformat())
            try:
                created_at = datetime.fromisoformat(created_at_str)
                hours_elapsed = (now - created_at).total_seconds() / 3600
            except (ValueError, TypeError):
                hours_elapsed = 0.0

            recency_boost = float(np.exp(-0.05 * hours_elapsed))

            abstraction = meta.get("abstraction", 0.0)
            abstraction_score = 1.0 - abstraction

            anchor_boost = 0.05 if mid in anchor_ids else 0.0

            priority = (
                w["semantic_similarity"] * semantic_sim
                + w["time_decay"] * meta.get("time_decay", 0.0)
                + w["emotional_arousal"] * meta.get("emotional_arousal", 0.0)
                + w["call_frequency"] * meta.get("call_frequency", 0.0)
                + w["connection_density"] * meta.get("connection_density", 0.0)
                + w["emotional_resonance"] * emotional_resonance
                + w.get("abstraction", 0.05) * abstraction_score
                + w.get("recency_boost", 0.10) * recency_boost
                + anchor_boost
            )

            if priority >= min_priority:
                memory = MemoryNode.from_metadata(mid, doc, meta)
                scored.append(MemorySearchResult(
                    memory=memory,
                    priority=priority,
                    semantic_similarity=semantic_sim,
                ))
                seen_ids.add(mid)

        if include_recent > 0:
            recent = self.retrieve_recent(include_recent)
            for r in recent:
                if r.memory.id not in seen_ids:
                    scored.append(r)
                    seen_ids.add(r.memory.id)

        scored.sort(key=lambda x: x.priority, reverse=True)
        return scored[:top_k]

    def retrieve_recent(self, n: int = 5) -> list[MemorySearchResult]:
        try:
            all_data = self.collection.get(include=["documents", "metadatas"])
        except Exception as e:
            logger.error(f"Recent retrieval failed: {e}")
            return []

        if not all_data["ids"]:
            return []

        entries = []
        now = datetime.now()
        for i, mid in enumerate(all_data["ids"]):
            meta = all_data["metadatas"][i]
            created_at = meta.get("created_at", now.isoformat())
            entries.append((mid, all_data["documents"][i], meta, created_at))

        entries.sort(key=lambda x: x[3], reverse=True)

        results: list[MemorySearchResult] = []
        for idx, (mid, doc, meta, _) in enumerate(entries[:n]):
            memory = MemoryNode.from_metadata(mid, doc, meta)
            recency_rank_boost = 1.0 - (idx / max(n, 1)) * 0.5
            time_decay = memory.time_decay
            priority = 0.6 * recency_rank_boost + 0.4 * time_decay
            results.append(MemorySearchResult(
                memory=memory,
                priority=priority,
                semantic_similarity=0.0,
            ))

        return results

    def retrieve_by_emotion(
        self,
        valence_range: tuple[float, float] = (-1.0, 1.0),
        min_arousal: float = 0.5,
        top_k: int = 5,
    ) -> list[MemorySearchResult]:
        try:
            all_data = self.collection.get(include=["documents", "metadatas"])
        except Exception as e:
            logger.error(f"Emotion-based retrieval failed: {e}")
            return []

        results: list[MemorySearchResult] = []
        for i, mid in enumerate(all_data["ids"]):
            meta = all_data["metadatas"][i]
            v = meta.get("emotional_valence", 0.0)
            a = meta.get("emotional_arousal", 0.0)
            if valence_range[0] <= v <= valence_range[1] and a >= min_arousal:
                memory = MemoryNode.from_metadata(mid, all_data["documents"][i], meta)
                results.append(MemorySearchResult(
                    memory=memory,
                    priority=a,
                    semantic_similarity=0.0,
                ))

        results.sort(key=lambda x: x.priority, reverse=True)
        return results[:top_k]

    def update_access(self, memory_id: str) -> None:
        try:
            result = self.collection.get(ids=[memory_id], include=["metadatas"])
            if not result["metadatas"]:
                return
            meta = result["metadatas"][0]
            meta["access_count"] = meta.get("access_count", 0) + 1
            meta["call_frequency"] = min(1.0, meta.get("call_frequency", 0.0) + 0.05)
            self.collection.update(ids=[memory_id], metadatas=[meta])
        except Exception as e:
            logger.warning(f"Failed to update access for {memory_id}: {e}")

    def update_emotion(
        self,
        memory_id: str,
        valence: float | None = None,
        arousal: float | None = None,
    ) -> None:
        try:
            result = self.collection.get(ids=[memory_id], include=["metadatas"])
            if not result["metadatas"]:
                return
            meta = result["metadatas"][0]
            if valence is not None:
                meta["emotional_valence"] = valence
            if arousal is not None:
                meta["emotional_arousal"] = arousal
            self.collection.update(ids=[memory_id], metadatas=[meta])
        except Exception as e:
            logger.warning(f"Failed to update emotion for {memory_id}: {e}")

    def update_abstraction(self, memory_id: str, abstraction: float) -> None:
        try:
            result = self.collection.get(ids=[memory_id], include=["metadatas"])
            if not result["metadatas"]:
                return
            meta = result["metadatas"][0]
            meta["abstraction"] = abstraction
            self.collection.update(ids=[memory_id], metadatas=[meta])
        except Exception as e:
            logger.warning(f"Failed to update abstraction for {memory_id}: {e}")

    def decay_all(self) -> int:
        try:
            all_data = self.collection.get(include=["metadatas"])
            if not all_data["ids"]:
                return 0
            count = 0
            for i, mid in enumerate(all_data["ids"]):
                meta = all_data["metadatas"][i]
                created = datetime.fromisoformat(meta.get("created_at", datetime.now().isoformat()))
                hours_elapsed = (datetime.now() - created).total_seconds() / 3600
                new_decay = float(np.exp(-self.decay_lambda * hours_elapsed))
                if abs(new_decay - meta.get("time_decay", 1.0)) > 0.01:
                    meta["time_decay"] = new_decay
                    self.collection.update(ids=[mid], metadatas=[meta])
                    count += 1
            logger.info(f"Decay updated for {count} memories")
            return count
        except Exception as e:
            logger.error(f"Decay update failed: {e}")
            return 0

    def get_emotional_anchors(self, min_arousal: float = 0.7) -> list[MemoryNode]:
        return [
            r.memory for r in self.retrieve_by_emotion(min_arousal=min_arousal, top_k=20)
        ]

    def get_stats(self) -> dict[str, Any]:
        count = self.collection.count()
        if count == 0:
            return {"total": 0}
        all_data = self.collection.get(include=["metadatas"])
        arousals = [m.get("emotional_arousal", 0.0) for m in all_data["metadatas"]]
        valences = [m.get("emotional_valence", 0.0) for m in all_data["metadatas"]]
        decays = [m.get("time_decay", 1.0) for m in all_data["metadatas"]]
        return {
            "total": count,
            "avg_arousal": float(np.mean(arousals)) if arousals else 0.0,
            "avg_valence": float(np.mean(valences)) if valences else 0.0,
            "avg_decay": float(np.mean(decays)) if decays else 1.0,
            "high_arousal_count": sum(1 for a in arousals if a >= 0.7),
            "low_decay_count": sum(1 for d in decays if d < 0.3),
        }

    def delete(self, memory_id: str) -> None:
        try:
            self.collection.delete(ids=[memory_id])
            logger.debug(f"Memory deleted: {memory_id}")
        except Exception as e:
            logger.warning(f"Failed to delete memory {memory_id}: {e}")

    def clear(self) -> None:
        try:
            all_ids = self.collection.get()["ids"]
            if all_ids:
                self.collection.delete(ids=all_ids)
            logger.info("All memories cleared")
        except Exception as e:
            logger.error(f"Failed to clear memories: {e}")
