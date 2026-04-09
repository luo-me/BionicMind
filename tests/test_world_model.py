from bionic_mind.core.world_model import WorldModel, Prediction, PredictionError


class TestWorldModel:
    def test_predict_no_pattern(self):
        wm = WorldModel()
        pred = wm.predict("hello world", "ctx1")
        assert pred.confidence <= 0.5

    def test_update_and_learn(self):
        wm = WorldModel()
        wm.predict("test context", "ctx1")
        error = wm.update("ctx1", "test response")
        assert 0 <= error.value <= 1

    def test_accuracy_tracking(self):
        wm = WorldModel()
        wm.predict("ctx", "c1")
        wm.update("c1", "response")
        assert wm.get_accuracy() >= 0.0

    def test_avg_error(self):
        wm = WorldModel()
        for i in range(5):
            wm.predict(f"ctx_{i}", f"c{i}")
            wm.update(f"c{i}", f"resp_{i}")
        avg = wm.get_avg_error()
        assert 0 <= avg <= 1

    def test_simulate(self):
        wm = WorldModel()
        results = wm.simulate("test context", ["action_a", "action_b"])
        assert len(results) == 2
        assert results[0]["action"] in ["action_a", "action_b"]

    def test_stats(self):
        wm = WorldModel()
        wm.predict("test", "c1")
        wm.update("c1", "response")
        stats = wm.get_stats()
        assert stats["total_predictions"] == 1
        assert "accuracy" in stats
        assert "patterns_learned" in stats


class TestPredictionError:
    def test_is_surprising(self):
        pe = PredictionError(value=0.8, surprise_level=0.7)
        assert pe.is_surprising

    def test_not_surprising(self):
        pe = PredictionError(value=0.2, surprise_level=0.1)
        assert not pe.is_surprising
