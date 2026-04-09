from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import networkx as nx
from loguru import logger


class EdgeType(str, Enum):
    CAUSAL = "causal"
    TEMPORAL = "temporal"
    SEMANTIC = "semantic"
    EMOTIONAL = "emotional"


@dataclass
class MemoryEdge:
    source_id: str
    target_id: str
    strength: float = 0.5
    edge_type: EdgeType = EdgeType.SEMANTIC
    co_activation_count: int = 0
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "strength": self.strength,
            "edge_type": self.edge_type.value,
            "co_activation_count": self.co_activation_count,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryEdge:
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            strength=data.get("strength", 0.5),
            edge_type=EdgeType(data.get("edge_type", "semantic")),
            co_activation_count=data.get("co_activation_count", 0),
            created_at=data.get("created_at", ""),
        )


class HebbianNetwork:
    def __init__(
        self,
        persist_path: str = "./memory_db/hebbian_graph.json",
        learning_rate: float = 0.1,
        decay_rate: float = 0.01,
        activation_threshold: float = 0.3,
        max_strength: float = 1.0,
        min_strength: float = 0.05,
    ):
        self.persist_path = persist_path
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.activation_threshold = activation_threshold
        self.max_strength = max_strength
        self.min_strength = min_strength
        self.graph = nx.DiGraph()
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.persist_path):
            try:
                with open(self.persist_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for edge_data in data.get("edges", []):
                    edge = MemoryEdge.from_dict(edge_data)
                    self.graph.add_edge(
                        edge.source_id,
                        edge.target_id,
                        strength=edge.strength,
                        edge_type=edge.edge_type.value,
                        co_activation_count=edge.co_activation_count,
                        created_at=edge.created_at,
                    )
                logger.info(f"HebbianNetwork loaded: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
            except Exception as e:
                logger.warning(f"Failed to load HebbianNetwork: {e}")

    def save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
            edges = []
            for u, v, data in self.graph.edges(data=True):
                edges.append({
                    "source_id": u,
                    "target_id": v,
                    "strength": data.get("strength", 0.5),
                    "edge_type": data.get("edge_type", "semantic"),
                    "co_activation_count": data.get("co_activation_count", 0),
                    "created_at": data.get("created_at", ""),
                })
            with open(self.persist_path, "w", encoding="utf-8") as f:
                json.dump({"edges": edges}, f, ensure_ascii=False, indent=2)
            logger.debug(f"HebbianNetwork saved: {len(edges)} edges")
        except Exception as e:
            logger.error(f"Failed to save HebbianNetwork: {e}")

    def co_activate(self, memory_ids: list[str], edge_type: EdgeType = EdgeType.SEMANTIC) -> None:
        if len(memory_ids) < 2:
            return
        for i in range(len(memory_ids)):
            for j in range(i + 1, len(memory_ids)):
                self._strengthen_edge(memory_ids[i], memory_ids[j], edge_type)
                self._strengthen_edge(memory_ids[j], memory_ids[i], edge_type)
        self._decay_non_activated(memory_ids)

    def _strengthen_edge(self, source: str, target: str, edge_type: EdgeType) -> None:
        if self.graph.has_edge(source, target):
            current = self.graph[source][target].get("strength", 0.5)
            count = self.graph[source][target].get("co_activation_count", 0) + 1
            new_strength = min(current + self.learning_rate * (1 - current), self.max_strength)
            self.graph[source][target]["strength"] = new_strength
            self.graph[source][target]["co_activation_count"] = count
        else:
            from datetime import datetime
            self.graph.add_edge(
                source, target,
                strength=0.5,
                edge_type=edge_type.value,
                co_activation_count=1,
                created_at=datetime.now().isoformat(),
            )

    def _decay_non_activated(self, activated_ids: list[str]) -> None:
        activated_set = set(activated_ids)
        edges_to_remove = []
        for u, v, data in self.graph.edges(data=True):
            if u in activated_set and v in activated_set:
                continue
            current = data.get("strength", 0.5)
            new_strength = current - self.decay_rate
            if new_strength < self.min_strength:
                edges_to_remove.append((u, v))
            else:
                data["strength"] = new_strength
        for u, v in edges_to_remove:
            self.graph.remove_edge(u, v)

    def get_related(self, memory_id: str, max_depth: int = 2, min_strength: float = 0.3) -> list[tuple[str, float, int]]:
        if memory_id not in self.graph:
            return []
        related = []
        visited = {memory_id}
        queue = [(memory_id, 0)]
        while queue:
            current, depth = queue.pop(0)
            if depth >= max_depth:
                continue
            for neighbor in self.graph.successors(current):
                if neighbor in visited:
                    continue
                edge_data = self.graph[current][neighbor]
                strength = edge_data.get("strength", 0.0)
                if strength >= min_strength:
                    related.append((neighbor, strength, depth + 1))
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
            for neighbor in self.graph.predecessors(current):
                if neighbor in visited:
                    continue
                edge_data = self.graph[neighbor][current]
                strength = edge_data.get("strength", 0.0)
                if strength >= min_strength:
                    related.append((neighbor, strength, depth + 1))
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
        related.sort(key=lambda x: (-x[1], x[2]))
        return related

    def get_spreading_activation(
        self,
        seed_ids: list[str],
        activation_energy: float = 1.0,
        decay_factor: float = 0.5,
        max_steps: int = 3,
        min_activation: float = 0.1,
    ) -> dict[str, float]:
        activation: dict[str, float] = {}
        for sid in seed_ids:
            activation[sid] = activation.get(sid, 0.0) + activation_energy
        for step in range(max_steps):
            new_activation: dict[str, float] = {}
            for node, energy in activation.items():
                if energy < min_activation:
                    continue
                if node not in self.graph:
                    continue
                for neighbor in self.graph.successors(node):
                    edge_strength = self.graph[node][neighbor].get("strength", 0.5)
                    spread = energy * decay_factor * edge_strength
                    if spread >= min_activation:
                        new_activation[neighbor] = new_activation.get(neighbor, 0.0) + spread
            for node, energy in new_activation.items():
                activation[node] = activation.get(node, 0.0) + energy
            if not new_activation:
                break
        return {k: v for k, v in activation.items() if v >= min_activation and k not in seed_ids}

    def get_stats(self) -> dict[str, Any]:
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "avg_strength": sum(d.get("strength", 0) for _, _, d in self.graph.edges(data=True)) / max(self.graph.number_of_edges(), 1),
            "edge_types": dict(nx.get_edge_attributes(self.graph, "edge_type")),
        }

    def prune(self, min_strength: float = 0.1) -> int:
        edges_to_remove = [
            (u, v) for u, v, d in self.graph.edges(data=True)
            if d.get("strength", 0) < min_strength
        ]
        self.graph.remove_edges_from(edges_to_remove)
        isolated = list(nx.isolates(self.graph))
        self.graph.remove_nodes_from(isolated)
        logger.info(f"Pruned {len(edges_to_remove)} edges, {len(isolated)} isolated nodes")
        return len(edges_to_remove)
