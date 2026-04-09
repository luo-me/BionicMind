import pytest
import tempfile
import os
import shutil

from bionic_mind.core.memory import MemoryVectorField, MemoryNode


@pytest.fixture
def memory_field():
    tmpdir = os.path.join(tempfile.gettempdir(), f"bionic_test_{os.getpid()}_{id(object())}")
    os.makedirs(tmpdir, exist_ok=True)
    field = MemoryVectorField(persist_dir=tmpdir)
    yield field
    try:
        field.client.delete_collection("bionic_memories")
    except Exception:
        pass
    try:
        shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception:
        pass


class TestMemoryNode:
    def test_create_default(self):
        node = MemoryNode()
        assert node.time_decay == 1.0
        assert node.abstraction == 0.0
        assert node.emotional_valence == 0.0
        assert node.emotional_arousal == 0.0
        assert node.call_frequency == 0.0

    def test_create_with_content(self):
        node = MemoryNode(content="test memory", emotional_arousal=0.8)
        assert node.content == "test memory"
        assert node.emotional_arousal == 0.8

    def test_to_metadata(self):
        node = MemoryNode(content="test", emotional_valence=0.5, emotional_arousal=0.7)
        meta = node.to_metadata()
        assert meta["emotional_valence"] == 0.5
        assert meta["emotional_arousal"] == 0.7
        assert "created_at" in meta

    def test_from_metadata(self):
        meta = {
            "time_decay": 0.5,
            "abstraction": 0.3,
            "emotional_valence": -0.5,
            "emotional_arousal": 0.9,
            "call_frequency": 0.4,
            "connection_density": 0.2,
            "created_at": "2024-01-01T00:00:00",
            "access_count": 5,
            "memory_type": "episodic",
            "perception": "hello",
            "response": "hi there",
        }
        node = MemoryNode.from_metadata("test_id", "test content", meta)
        assert node.id == "test_id"
        assert node.content == "test content"
        assert node.time_decay == 0.5
        assert node.emotional_arousal == 0.9
        assert node.perception == "hello"
        assert node.response == "hi there"


class TestMemoryVectorField:
    def test_write_and_count(self, memory_field):
        node = MemoryNode(content="hello world")
        mid = memory_field.write(node)
        assert memory_field.collection.count() == 1

    def test_write_batch(self, memory_field):
        nodes = [MemoryNode(content=f"memory {i}") for i in range(5)]
        ids = memory_field.write_batch(nodes)
        assert len(ids) == 5
        assert memory_field.collection.count() == 5

    def test_retrieve_empty(self, memory_field):
        results = memory_field.retrieve("test query")
        assert results == []

    def test_write_and_retrieve(self, memory_field):
        memory_field.write(MemoryNode(content="Python is a programming language"))
        memory_field.write(MemoryNode(content="The weather is sunny today"))
        memory_field.write(MemoryNode(content="Python has great libraries"))

        results = memory_field.retrieve("Python programming", top_k=2)
        assert len(results) <= 2
        assert len(results) > 0
        assert "Python" in results[0].memory.content

    def test_retrieve_with_emotion_context(self, memory_field):
        memory_field.write(MemoryNode(
            content="A very scary experience",
            emotional_valence=-0.8,
            emotional_arousal=0.9,
        ))
        memory_field.write(MemoryNode(
            content="A happy celebration",
            emotional_valence=0.8,
            emotional_arousal=0.7,
        ))

        results = memory_field.retrieve(
            "fear",
            context_emotion={"valence": -0.5, "arousal": 0.8},
            top_k=2,
        )
        assert len(results) > 0

    def test_retrieve_by_emotion(self, memory_field):
        memory_field.write(MemoryNode(
            content="Traumatic event",
            emotional_valence=-0.9,
            emotional_arousal=0.95,
        ))
        memory_field.write(MemoryNode(
            content="Mild conversation",
            emotional_valence=0.1,
            emotional_arousal=0.2,
        ))

        results = memory_field.retrieve_by_emotion(min_arousal=0.7, top_k=5)
        assert len(results) >= 1
        assert results[0].memory.emotional_arousal >= 0.7

    def test_update_access(self, memory_field):
        node = MemoryNode(content="test access update")
        mid = memory_field.write(node)
        memory_field.update_access(mid)

        result = memory_field.collection.get(ids=[mid], include=["metadatas"])
        assert result["metadatas"][0]["access_count"] == 1
        assert result["metadatas"][0]["call_frequency"] > 0

    def test_update_emotion(self, memory_field):
        node = MemoryNode(content="emotion update test")
        mid = memory_field.write(node)
        memory_field.update_emotion(mid, valence=-0.5, arousal=0.8)

        result = memory_field.collection.get(ids=[mid], include=["metadatas"])
        assert result["metadatas"][0]["emotional_valence"] == -0.5
        assert result["metadatas"][0]["emotional_arousal"] == 0.8

    def test_get_stats(self, memory_field):
        memory_field.write(MemoryNode(content="stat 1", emotional_arousal=0.8))
        memory_field.write(MemoryNode(content="stat 2", emotional_arousal=0.2))

        stats = memory_field.get_stats()
        assert stats["total"] == 2
        assert "avg_arousal" in stats

    def test_delete(self, memory_field):
        node = MemoryNode(content="to be deleted")
        mid = memory_field.write(node)
        assert memory_field.collection.count() == 1
        memory_field.delete(mid)
        assert memory_field.collection.count() == 0

    def test_emotional_anchors(self, memory_field):
        memory_field.write(MemoryNode(
            content="deep anchor",
            emotional_arousal=0.95,
            emotional_valence=-0.9,
        ))
        memory_field.write(MemoryNode(
            content="shallow memory",
            emotional_arousal=0.1,
        ))

        anchors = memory_field.get_emotional_anchors(min_arousal=0.7)
        assert len(anchors) >= 1
        assert anchors[0].emotional_arousal >= 0.7

    def test_retrieve_includes_recent(self, memory_field):
        import time
        memory_field.write(MemoryNode(
            content="old memory about cats",
            emotional_arousal=0.9,
            emotional_valence=0.8,
            call_frequency=0.8,
        ))
        time.sleep(0.1)
        memory_field.write(MemoryNode(
            content="recent memory about dogs",
            emotional_arousal=0.1,
        ))

        results = memory_field.retrieve("dogs", top_k=5, include_recent=2)
        contents = [r.memory.content for r in results]
        assert any("dogs" in c or "recent" in c for c in contents)

    def test_perception_response_fields(self, memory_field):
        node = MemoryNode(
            content="what is your name",
            perception="what is your name",
            response="I am BionicMind",
            emotional_arousal=0.3,
        )
        mid = memory_field.write(node)

        result = memory_field.collection.get(ids=[mid], include=["metadatas"])
        meta = result["metadatas"][0]
        assert meta["perception"] == "what is your name"
        assert meta["response"] == "I am BionicMind"
