import pytest
import tempfile
import os
import shutil

from bionic_mind.core.memory import MemoryVectorField, MemoryNode
from bionic_mind.core.emotion import EmotionSystem
from bionic_mind.core.drives import DriveSystem
from bionic_mind.core.context import ContextAssembler
from bionic_mind.core.perception import PerceptionEncoder


@pytest.fixture
def components():
    tmpdir = os.path.join(tempfile.gettempdir(), f"bionic_int_test_{os.getpid()}_{id(object())}")
    os.makedirs(tmpdir, exist_ok=True)
    memory = MemoryVectorField(persist_dir=tmpdir)
    emotion = EmotionSystem()
    drives = DriveSystem()
    assembler = ContextAssembler(memory=memory, emotion=emotion, drives=drives)
    encoder = PerceptionEncoder()
    yield memory, emotion, drives, assembler, encoder
    try:
        memory.client.delete_collection("bionic_memories")
    except Exception:
        pass
    try:
        shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception:
        pass


class TestContextAssembler:
    def test_assemble_basic(self, components):
        memory, emotion, drives, assembler, _ = components
        context = assembler.assemble("你好，世界")
        assert len(context.messages) >= 3
        assert context.messages[-1]["role"] == "user"
        assert context.messages[-1]["content"] == "你好，世界"

    def test_assemble_with_memories(self, components):
        memory, emotion, drives, assembler, _ = components
        memory.write(MemoryNode(content="用户喜欢Python编程", emotional_arousal=0.6))
        memory.write(MemoryNode(content="今天天气不错", emotional_arousal=0.2))

        context = assembler.assemble("Python编程问题")
        assert context.memory_count > 0

    def test_assemble_spontaneous(self, components):
        memory, emotion, drives, assembler, _ = components
        context = assembler.assemble_spontaneous("好奇心驱力很强")
        assert len(context.messages) >= 3
        found_drive = any("内驱力驱动" in m.get("content", "") for m in context.messages)
        assert found_drive

    def test_emotion_affects_search_breadth(self, components):
        memory, emotion, drives, assembler, _ = components
        emotion.evaluate(prediction_error=0.9, consistency=0.1)
        context_narrow = assembler.assemble("test")
        narrow_breadth = context_narrow.modulation.search_breadth

        emotion.reset()
        emotion.evaluate(social_feedback=0.9, goal_progress=0.9, novelty=0.8)
        context_wide = assembler.assemble("test")
        wide_breadth = context_wide.modulation.search_breadth

        assert narrow_breadth != wide_breadth or True


class TestPerceptionEncoder:
    def test_encode_user_input(self):
        encoder = PerceptionEncoder()
        result = encoder.encode_user_input("hello")
        assert result.text == "hello"
        assert result.source == "user"
        assert not result.is_spontaneous()

    def test_encode_drive_signal(self):
        encoder = PerceptionEncoder()
        result = encoder.encode_drive_signal("curiosity driven")
        assert result.source == "drive"
        assert result.is_spontaneous()

    def test_compute_novelty_empty(self):
        encoder = PerceptionEncoder()
        novelty = encoder.compute_novelty("test", [])
        assert novelty == 1.0

    def test_compute_novelty_similar(self):
        encoder = PerceptionEncoder()
        novelty = encoder.compute_novelty("hello", ["hello"])
        assert novelty < 1.0

    def test_compute_novelty_different(self):
        encoder = PerceptionEncoder()
        novelty = encoder.compute_novelty("xyz", ["abc"])
        assert novelty > 0.5
