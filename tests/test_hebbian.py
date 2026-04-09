import pytest
import tempfile
import os
import shutil

from bionic_mind.core.hebbian import HebbianNetwork, MemoryEdge, EdgeType


@pytest.fixture
def hebbian():
    tmpdir = os.path.join(tempfile.gettempdir(), f"hebbian_test_{os.getpid()}_{id(object())}")
    os.makedirs(tmpdir, exist_ok=True)
    path = os.path.join(tmpdir, "hebbian_graph.json")
    net = HebbianNetwork(persist_path=path)
    yield net
    try:
        shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception:
        pass


class TestHebbianNetwork:
    def test_co_activate_creates_edges(self, hebbian):
        hebbian.co_activate(["mem1", "mem2", "mem3"], EdgeType.SEMANTIC)
        assert hebbian.graph.number_of_edges() == 6  # 3*2 bidirectional

    def test_co_activate_strengthens_existing(self, hebbian):
        hebbian.co_activate(["mem1", "mem2"], EdgeType.SEMANTIC)
        s1 = hebbian.graph["mem1"]["mem2"]["strength"]
        hebbian.co_activate(["mem1", "mem2"], EdgeType.SEMANTIC)
        s2 = hebbian.graph["mem1"]["mem2"]["strength"]
        assert s2 > s1

    def test_get_related(self, hebbian):
        hebbian.co_activate(["a", "b"], EdgeType.SEMANTIC)
        hebbian.co_activate(["b", "c"], EdgeType.SEMANTIC)
        related = hebbian.get_related("a", max_depth=2)
        assert len(related) > 0
        related_ids = [r[0] for r in related]
        assert "b" in related_ids

    def test_spreading_activation(self, hebbian):
        hebbian.co_activate(["a", "b"], EdgeType.SEMANTIC)
        hebbian.co_activate(["b", "c"], EdgeType.SEMANTIC)
        activation = hebbian.get_spreading_activation(["a"], activation_energy=1.0)
        assert "b" in activation
        assert activation["b"] > 0

    def test_decay_non_activated(self, hebbian):
        hebbian.co_activate(["a", "b"], EdgeType.SEMANTIC)
        hebbian.co_activate(["c", "d"], EdgeType.SEMANTIC)
        s_before = hebbian.graph["a"]["b"]["strength"]
        hebbian.co_activate(["c", "d"], EdgeType.SEMANTIC)
        s_after = hebbian.graph["a"]["b"]["strength"]
        assert s_after <= s_before

    def test_save_and_load(self, hebbian):
        hebbian.co_activate(["x", "y"], EdgeType.CAUSAL)
        hebbian.save()
        loaded = HebbianNetwork(persist_path=hebbian.persist_path)
        assert loaded.graph.number_of_edges() == 2

    def test_prune(self, hebbian):
        hebbian.co_activate(["a", "b"], EdgeType.SEMANTIC)
        hebbian.graph["a"]["b"]["strength"] = 0.01
        pruned = hebbian.prune(min_strength=0.05)
        assert pruned >= 1

    def test_get_stats(self, hebbian):
        hebbian.co_activate(["a", "b"], EdgeType.SEMANTIC)
        stats = hebbian.get_stats()
        assert stats["nodes"] == 2
        assert stats["edges"] == 2
